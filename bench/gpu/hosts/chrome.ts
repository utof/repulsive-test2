/**
 * Chrome-over-CDP host. This is the original `drive.ts` transport, moved behind
 * the `BrowserHost` seam unchanged — the flag sets, the `/json` target poll, the
 * raw-WebSocket CDP client and the `window.__runSpike` readiness poll are all
 * the code that passed the Phase 0/Phase 1 gates, and its behaviour is
 * deliberately bit-for-bit what it was.
 *
 * Why it is kept at all when the dev box has no Chrome: Dawn and naga are
 * *measured* to differ on compensated arithmetic (gpuweb/gpuweb#2076), so a
 * numerics gate taken on Firefox is not evidence about Chrome. Chrome must stay
 * selectable for the day it is installed, or the Phase 2H precision argument
 * has no second host to check against.
 *
 * @issue utof/repulsive-test2#54
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §3, §4 G0a
 * @see docs/superpowers/plans/2026-08-13-webgpu-solver-phase0.md Task 1
 */
import { type ChildProcessWithoutNullStreams, spawn } from 'node:child_process';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { isNoise } from '../consoleNoise';
import type { BrowserHost, HostEvidence, HostSession, LaunchVariant } from './types';

const CHROME = process.env.CHROME_BIN ?? 'google-chrome';

/**
 * Candidate flag sets, tried in order until `adapterInfo` reports NVIDIA
 * hardware. The third set is a headed fallback — spec G0a explicitly allows
 * a headed pass after an honest headless attempt.
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §4 G0a
 */
const CHROME_VARIANTS: LaunchVariant[] = [
    {
        label: 'headless-angle-vulkan',
        argv: [
            '--headless=new',
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan',
            '--use-angle=vulkan',
            '--enable-gpu',
            '--no-sandbox',
        ],
        prefs: {},
        env: {},
    },
    {
        label: 'headless-ignore-blocklist',
        argv: [
            '--headless=new',
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan',
            '--ignore-gpu-blocklist',
            '--no-sandbox',
        ],
        prefs: {},
        env: {},
    },
    // headed fallback (spec G0a: allowed after honest headless effort)
    {
        label: 'headed',
        argv: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--no-sandbox'],
        prefs: {},
        env: {},
    },
];

/** Poll the CDP HTTP endpoint until a target for `harness.html` appears. */
async function waitForTarget(
    port: number,
    timeoutMs = 10_000,
): Promise<{ webSocketDebuggerUrl: string }> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        try {
            const res = await fetch(`http://localhost:${port}/json`);
            if (res.ok) {
                const targets = (await res.json()) as Array<{
                    url: string;
                    webSocketDebuggerUrl: string;
                }>;
                const target = targets.find((t) => t.url.endsWith('harness.html'));
                if (target) return target;
            }
        } catch {
            // Chrome's debug HTTP endpoint isn't up yet; keep polling.
        }
        await new Promise((r) => setTimeout(r, 200));
    }
    throw new Error(`No harness.html target on CDP port ${port} within ${timeoutMs}ms`);
}

/**
 * Minimal raw CDP client over Bun's native WebSocket — id-keyed pending-promise
 * map, no library, per plan Task 1 Step 2 transport spec.
 */
class CDPClient {
    private ws: WebSocket;
    private nextId = 1;
    private pending = new Map<
        number,
        { resolve: (v: unknown) => void; reject: (e: unknown) => void }
    >();
    private closed = false;
    consoleLines: string[] = [];

    constructor(url: string) {
        this.ws = new WebSocket(url);
    }

    /**
     * Reject every outstanding `send()` promise and mark the client closed so
     * later `send()` calls fail fast. Without this, a Chrome crash or GPU
     * device-loss mid-spike (realistic for the long G0t/G1/G3 compute
     * workloads) leaves an awaited `Runtime.evaluate` pending forever and the
     * driver hangs. @see https://github.com/utof/repulsive-test2/issues/18 pt.1
     */
    private failPending(err: Error): void {
        this.closed = true;
        for (const { reject } of this.pending.values()) reject(err);
        this.pending.clear();
    }

    async connect(): Promise<void> {
        await new Promise<void>((resolve, reject) => {
            this.ws.addEventListener('open', () => resolve());
            this.ws.addEventListener('error', (e) => reject(e));
        });
        this.ws.addEventListener('message', (ev) => {
            const msg = JSON.parse(ev.data as string);
            const entry = msg.id !== undefined ? this.pending.get(msg.id) : undefined;
            if (entry !== undefined) {
                const { resolve, reject } = entry;
                this.pending.delete(msg.id);
                if (msg.error) reject(new Error(JSON.stringify(msg.error)));
                else resolve(msg.result);
            } else if (msg.method === 'Runtime.consoleAPICalled') {
                const line = (msg.params.args ?? [])
                    .map(
                        (a: { value?: unknown; description?: string }) =>
                            a.value ?? a.description ?? '',
                    )
                    .join(' ');
                if (!isNoise(line)) this.consoleLines.push(line);
            }
        });
        // Persistent (post-handshake) close/error listeners, distinct from
        // the one-shot pair above that only guards the initial handshake.
        this.ws.addEventListener('close', () =>
            this.failPending(new Error('CDP WebSocket closed unexpectedly')),
        );
        this.ws.addEventListener('error', () => this.failPending(new Error('CDP WebSocket error')));
    }

    send(method: string, params: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
        if (this.closed) {
            return Promise.reject(new Error(`CDP WebSocket already closed, cannot send ${method}`));
        }
        const id = this.nextId++;
        return new Promise((resolve, reject) => {
            this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
            this.ws.send(JSON.stringify({ id, method, params }));
        });
    }

    close(): void {
        this.ws.close();
    }
}

/**
 * Poll `typeof window.__runSpike` until the harness page's module script has
 * executed and registered it (Bun bundles bench/gpu/spikes.ts on every
 * request, which is not instant, and the CDP target exists before the page
 * has finished loading it).
 */
async function waitForRunSpikeReady(client: CDPClient, timeoutMs = 15_000): Promise<void> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        const res = (await client.send('Runtime.evaluate', {
            expression: 'typeof window.__runSpike',
            returnByValue: true,
        })) as { result: { value: string } };
        if (res.result.value === 'function') return;
        await new Promise((r) => setTimeout(r, 200));
    }
    throw new Error(`window.__runSpike not ready within ${timeoutMs}ms`);
}

/**
 * Chrome/Dawn host. `shaderBackend: 'dawn'` is what makes a numerics result
 * taken here comparable only with other Dawn results.
 * @issue utof/repulsive-test2#51 trap 3
 */
export const chromeHost: BrowserHost = {
    id: 'chrome',
    shaderBackend: 'dawn',
    binary: CHROME,
    version(): string {
        try {
            const proc = Bun.spawnSync([CHROME, '--version']);
            return proc.stdout.toString().trim() || 'unknown';
        } catch {
            return 'unknown';
        }
    },
    launchVariants: CHROME_VARIANTS,

    /** Chrome has no environment override; a recipe resolves to itself. */
    resolveVariant(variant: LaunchVariant): LaunchVariant {
        return variant;
    },

    async launch(variant: LaunchVariant, url: string, port: number): Promise<HostSession> {
        const profileDir = mkdtempSync(join(tmpdir(), 'gpu-gate-chrome-'));
        const args = [
            ...variant.argv,
            `--remote-debugging-port=${port}`,
            `--user-data-dir=${profileDir}`,
            url,
        ];
        const proc: ChildProcessWithoutNullStreams = spawn(CHROME, args, {
            stdio: 'pipe',
            env: { ...process.env, ...variant.env },
        });
        let stderr = '';
        proc.stderr.on('data', (d) => {
            stderr += d.toString();
        });

        let client: CDPClient | undefined;
        try {
            const target = await waitForTarget(port);
            client = new CDPClient(target.webSocketDebuggerUrl);
            await client.connect();
            await client.send('Runtime.enable');
            // The CDP target for harness.html exists as soon as the tab is created,
            // well before the page has navigated/loaded and spikes.ts has run (which
            // is what defines window.__runSpike) — poll for readiness instead of
            // racing straight into Runtime.evaluate.
            await waitForRunSpikeReady(client);
        } catch (e) {
            client?.close();
            proc.kill();
            throw new Error(`${e instanceof Error ? e.message : String(e)}\nstderr: ${stderr}`);
        }

        const live = client;
        return {
            consoleLines: live.consoleLines,
            async evalSpike(name: string): Promise<Record<string, unknown>> {
                const evalResult = (await live.send('Runtime.evaluate', {
                    expression: `window.__runSpike('${name}')`,
                    awaitPromise: true,
                    returnByValue: true,
                })) as { exceptionDetails?: unknown; result: { value: string } };
                if (evalResult.exceptionDetails) {
                    throw new Error(
                        `Spike '${name}' threw: ${JSON.stringify(evalResult.exceptionDetails)}`,
                    );
                }
                return JSON.parse(evalResult.result.value) as Record<string, unknown>;
            },
            async hostEvidence(): Promise<HostEvidence> {
                // Chrome identifies its adapter through GPUAdapterInfo, so the
                // certification never needs the pin route; reading the flag
                // anyway keeps the two hosts' records the same shape.
                const res = (await live.send('Runtime.evaluate', {
                    expression:
                        'navigator.gpu.requestAdapter().then(a => a ? !!a.info?.isFallbackAdapter : undefined)',
                    awaitPromise: true,
                    returnByValue: true,
                })) as { result: { value?: boolean } };
                return { isFallbackAdapter: res.result.value, devicePin: null };
            },
            close(): void {
                live.close();
                proc.kill();
            },
        };
    },
};
