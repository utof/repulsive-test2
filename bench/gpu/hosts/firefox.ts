/**
 * Firefox host, driven over **WebDriver BiDi**.
 *
 * Why BiDi and not CDP: measured on this box 2026-09-11, Firefox 155.0.1
 * launched with `--remote-debugging-port=P` serves *only* a BiDi WebSocket at
 * `ws://127.0.0.1:P/session`. Every CDP HTTP path the Chrome driver depends on
 * 404s:
 *
 *     $ curl http://localhost:9224/json/version   ->  404 Not Found
 *     $ curl http://localhost:9224/json/list      ->  404 Not Found
 *     $ curl http://localhost:9224/json           ->  404 Not Found
 *     stderr: "WebDriver BiDi listening on ws://127.0.0.1:9224"
 *
 * So `waitForTarget`'s `/json` poll and `Runtime.evaluate` both have no Firefox
 * equivalent, and the CDP shim is not an option. The alternatives considered:
 *
 * - **geckodriver** — not installed, and it is a second process that only
 *   proxies to the same BiDi/Marionette endpoints this speaks directly.
 * - **Playwright/Puppeteer** — a new dependency plus a browser download, for a
 *   transport that is ~120 lines here; Puppeteer's own Firefox path is BiDi
 *   too, so it would add a layer, not remove one.
 * - **Marionette** (`--marionette`, port 2828) — chrome-privileged and useful
 *   for diagnostics, but its length-prefixed framing is a second wire format
 *   to maintain and it grants privileges the harness does not need.
 * - **WebDriver BiDi** — already listening on the flag the driver was passing
 *   anyway, plain JSON over a WebSocket, and `script.evaluate` with
 *   `awaitPromise` is a direct swap for `Runtime.evaluate`.
 *
 * @see https://developer.mozilla.org/en-US/docs/Web/WebDriver/How_to/Create_BiDi_connection
 * @see https://w3c.github.io/webdriver-bidi/
 * @issue utof/repulsive-test2#54
 */
import { type ChildProcessWithoutNullStreams, spawn } from 'node:child_process';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { isNoise } from '../consoleNoise';
import { describeIcdPin, HARDWARE_ICD_CANDIDATES, icdPath, icdPinEnv } from '../vulkanPin';
import type { BrowserHost, HostEvidence, HostSession, LaunchVariant } from './types';

const FIREFOX = process.env.FIREFOX_BIN ?? '/usr/bin/firefox';

/**
 * Prefs every Firefox gate profile gets.
 *
 * `dom.webgpu.enabled` is the one that matters and it is **required**: measured
 * on Firefox 155.0.1/Linux, WebGPU is not on by default — without this pref
 * `navigator.gpu` is `undefined`. (WebGPU shipped on Windows in FF 141 and on
 * macOS in FF 145; Linux is still pref-gated.) The rest just stop a fresh
 * profile from doing first-run network chatter mid-measurement.
 *
 * Note `dom.webgpu.enabled` is a WebIDL pref, not a static pref — it does not
 * appear in `greprefs.js`, only as a string inside libxul. Do not "clean it up"
 * because `about:config` shows it as absent on a default profile.
 * @issue utof/repulsive-test2#54
 */
const BASE_PREFS: Record<string, boolean | number | string> = {
    'dom.webgpu.enabled': true,
    // A gate profile must never be a real browsing profile.
    'browser.shell.checkDefaultBrowser': false,
    'datareporting.policy.dataSubmissionEnabled': false,
    'toolkit.telemetry.enabled': false,
    'browser.aboutwelcome.enabled': false,
};

/**
 * `GPU_VK_ICD` forces one specific Vulkan ICD by name (`nvidia`, `intel`,
 * `lvp`, …) instead of trying `HARDWARE_ICD_CANDIDATES`.
 *
 * Why an escape hatch exists: it is how the guard is *falsified*. Setting
 * `GPU_VK_ICD=lvp` pins the browser to llvmpipe, which flips
 * `isFallbackAdapter` to `true` and makes `vulkaninfo` report
 * `PHYSICAL_DEVICE_TYPE_CPU`, so `certifyAdapter` rules the run `'software'`
 * and the gate comes back INVALID. A guard never observed failing is not a
 * verified guard.
 * @see bench/gpu/adapterCert.ts
 */
const FORCED_ICD = process.env.GPU_VK_ICD;

/**
 * Launch recipes, in order. Each pins the Vulkan loader to one ICD so the
 * physical GPU is decided by us and is identifiable out-of-band, rather than by
 * the loader's default ordering.
 *
 * Why pinning is not optional: measured 2026-09-11, unpinned Firefox 155 on
 * this box runs WebGPU on the **Intel UHD 630 iGPU** and never touches the
 * NVIDIA Quadro RTX 3000 — `nvidia-smi --query-compute-apps` lists no firefox
 * process during a compute spike without the pin and lists one with it. Since
 * `GPUAdapterInfo` is blank, an unpinned run records no trace of which GPU
 * produced the numbers.
 *
 * The final `no-pin` recipe is kept so the harness still *runs* on a box with
 * no `vulkaninfo` or no ICD directory — it will simply fail certification and
 * report INVALID, which is the honest outcome, not a crash.
 */
function firefoxVariants(): LaunchVariant[] {
    const names = FORCED_ICD ? [FORCED_ICD] : [...HARDWARE_ICD_CANDIDATES];
    const pinned = names.map((name) => ({
        label: `headless-vk-${name}`,
        argv: ['--headless'],
        prefs: { ...BASE_PREFS },
        env: icdPinEnv(icdPath(name)),
    }));
    if (FORCED_ICD) return pinned;
    return [
        ...pinned,
        { label: 'headless-no-pin', argv: ['--headless'], prefs: { ...BASE_PREFS }, env: {} },
    ];
}

/** Serialise prefs into a profile `user.js`. */
function writeUserJs(profileDir: string, prefs: Record<string, boolean | number | string>): void {
    const lines = Object.entries(prefs).map(
        ([k, v]) => `user_pref(${JSON.stringify(k)}, ${JSON.stringify(v)});`,
    );
    writeFileSync(join(profileDir, 'user.js'), `${lines.join('\n')}\n`);
}

/**
 * Minimal WebDriver BiDi client — the same id-keyed pending-promise shape the
 * CDP client uses, over BiDi's JSON framing.
 *
 * BiDi differs from CDP in three ways that matter here: a command response is
 * `{id, type: 'success', result}` and an error is `{id, type: 'error', error}`
 * (CDP signals errors with an `error` key), events carry `method`+`params` with
 * no `id`, and a session must be created explicitly with `session.new` before
 * any other command is accepted.
 */
class BiDiClient {
    private ws: WebSocket;
    private nextId = 1;
    private pending = new Map<
        number,
        { resolve: (v: Record<string, unknown>) => void; reject: (e: unknown) => void }
    >();
    private closed = false;
    consoleLines: string[] = [];

    private constructor(ws: WebSocket) {
        this.ws = ws;
    }

    /**
     * Connect, retrying until the remote agent is listening. Firefox binds the
     * BiDi socket some hundreds of ms after process start and prints
     * "WebDriver BiDi listening on …" when it does; polling the handshake is
     * simpler than scraping stderr and does not depend on that log line.
     */
    static async connect(port: number, timeoutMs = 40_000): Promise<BiDiClient> {
        // MDN is explicit that Firefox binds 127.0.0.1, not `localhost` — on a
        // dual-stack box `localhost` can resolve to ::1 and never connect.
        const url = `ws://127.0.0.1:${port}/session`;
        const deadline = Date.now() + timeoutMs;
        let lastErr: unknown;
        while (Date.now() < deadline) {
            try {
                const ws = await new Promise<WebSocket>((resolve, reject) => {
                    const sock = new WebSocket(url);
                    const timer = setTimeout(() => reject(new Error('handshake timeout')), 2_000);
                    sock.addEventListener('open', () => {
                        clearTimeout(timer);
                        resolve(sock);
                    });
                    sock.addEventListener('error', (e) => {
                        clearTimeout(timer);
                        reject(e);
                    });
                });
                return new BiDiClient(ws);
            } catch (e) {
                lastErr = e;
                await new Promise((r) => setTimeout(r, 300));
            }
        }
        throw new Error(
            `No BiDi endpoint on ${url} within ${timeoutMs}ms (last: ${String(lastErr)})`,
        );
    }

    /** @see hosts/chrome.ts `CDPClient.failPending` — same device-loss hang. */
    private failPending(err: Error): void {
        this.closed = true;
        for (const { reject } of this.pending.values()) reject(err);
        this.pending.clear();
    }

    listen(): void {
        this.ws.addEventListener('message', (ev) => {
            const msg = JSON.parse(ev.data as string);
            const entry = msg.id !== undefined ? this.pending.get(msg.id) : undefined;
            if (entry !== undefined) {
                this.pending.delete(msg.id);
                if (msg.type === 'error') entry.reject(new Error(JSON.stringify(msg)));
                else entry.resolve(msg.result as Record<string, unknown>);
            } else if (msg.method === 'log.entryAdded') {
                const line = String(msg.params?.text ?? '');
                if (line && !isNoise(line)) this.consoleLines.push(line);
            }
        });
        this.ws.addEventListener('close', () =>
            this.failPending(new Error('BiDi WebSocket closed unexpectedly')),
        );
        this.ws.addEventListener('error', () =>
            this.failPending(new Error('BiDi WebSocket error')),
        );
    }

    send(method: string, params: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
        if (this.closed) {
            return Promise.reject(
                new Error(`BiDi WebSocket already closed, cannot send ${method}`),
            );
        }
        const id = this.nextId++;
        return new Promise((resolve, reject) => {
            this.pending.set(id, { resolve, reject });
            this.ws.send(JSON.stringify({ id, method, params }));
        });
    }

    close(): void {
        this.ws.close();
    }
}

/** Result shape of BiDi `script.evaluate` for a string-returning expression. */
type ScriptResult = {
    type: 'success' | 'exception';
    result?: { type: string; value?: string };
    exceptionDetails?: unknown;
};

/**
 * Evaluate `expression` in `context`, awaiting any promise it returns.
 *
 * `awaitPromise: true` is BiDi's equivalent of CDP's flag of the same name. The
 * harness's `window.__runSpike` already `JSON.stringify`s its result, so the
 * value comes back as a BiDi primitive string and needs no deep serialisation
 * (`resultOwnership` stays at its default `none`).
 */
async function evaluate(
    client: BiDiClient,
    context: string,
    expression: string,
): Promise<ScriptResult> {
    return (await client.send('script.evaluate', {
        expression,
        target: { context },
        awaitPromise: true,
    })) as ScriptResult;
}

/** @see hosts/chrome.ts `waitForRunSpikeReady` — same rationale, BiDi wire. */
async function waitForRunSpikeReady(
    client: BiDiClient,
    context: string,
    timeoutMs = 20_000,
): Promise<void> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        const res = await evaluate(client, context, 'typeof window.__runSpike');
        if (res.result?.value === 'function') return;
        await new Promise((r) => setTimeout(r, 200));
    }
    throw new Error(`window.__runSpike not ready within ${timeoutMs}ms`);
}

/**
 * Firefox/naga host.
 *
 * `shaderBackend: 'naga'` is the field that stops a compensated-arithmetic
 * result taken here from being read as evidence about Chrome. naga is measured
 * to delete Dekker compensation terms Dawn preserves, and WGSL offers no
 * portable opt-out.
 * @see https://github.com/gpuweb/gpuweb/issues/2076
 * @issue utof/repulsive-test2#51 trap 3
 */
export const firefoxHost: BrowserHost = {
    id: 'firefox',
    shaderBackend: 'naga',
    binary: FIREFOX,
    version(): string {
        try {
            const proc = Bun.spawnSync([FIREFOX, '--version']);
            return proc.stdout.toString().trim() || 'unknown';
        } catch {
            return 'unknown';
        }
    },
    get launchVariants(): LaunchVariant[] {
        return firefoxVariants();
    },

    /**
     * `GPU_VK_ICD` overrides the pin of any recipe, including one replayed from
     * a stored G0a record — otherwise the knock-out would only ever reach the
     * G0a gate and every later gate would quietly run on the recipe G0a passed
     * with. @see the `FORCED_ICD` docstring
     */
    resolveVariant(variant: LaunchVariant): LaunchVariant {
        if (!FORCED_ICD) return variant;
        const path = icdPath(FORCED_ICD);
        return {
            ...variant,
            label: `${variant.label}+forced-vk-${FORCED_ICD}`,
            env: { ...variant.env, ...icdPinEnv(path) },
        };
    },

    async launch(variant: LaunchVariant, url: string, port: number): Promise<HostSession> {
        // A dedicated throwaway profile per run — never the user's daily one.
        const profileDir = mkdtempSync(join(tmpdir(), 'gpu-gate-firefox-'));
        mkdirSync(profileDir, { recursive: true });
        writeUserJs(profileDir, variant.prefs);

        const args = [
            ...variant.argv,
            '--no-remote',
            '--profile',
            profileDir,
            `--remote-debugging-port=${port}`,
            url,
        ];
        const proc: ChildProcessWithoutNullStreams = spawn(FIREFOX, args, {
            stdio: 'pipe',
            env: { ...process.env, ...variant.env },
        });
        let stderr = '';
        proc.stderr.on('data', (d) => {
            stderr += d.toString();
        });

        let client: BiDiClient | undefined;
        let context: string;
        try {
            client = await BiDiClient.connect(port);
            client.listen();
            await client.send('session.new', {
                capabilities: { alwaysMatch: { browserName: 'firefox' } },
            });
            await client.send('session.subscribe', { events: ['log.entryAdded'] });
            const tree = (await client.send('browsingContext.getTree', {})) as {
                contexts: Array<{ context: string }>;
            };
            context = tree.contexts[0].context;
            // Firefox is launched with the harness URL on argv, but the initial
            // context can still be about:blank when BiDi attaches — and WebGPU
            // is gated on a secure context, so `navigator.gpu` is undefined on
            // about:blank (measured). Navigating explicitly and waiting for
            // 'complete' removes that race.
            await client.send('browsingContext.navigate', { context, url, wait: 'complete' });
            await waitForRunSpikeReady(client, context);
        } catch (e) {
            client?.close();
            proc.kill();
            throw new Error(`${e instanceof Error ? e.message : String(e)}\nstderr: ${stderr}`);
        }

        const live = client;
        const ctx = context;
        return {
            consoleLines: live.consoleLines,
            async evalSpike(name: string): Promise<Record<string, unknown>> {
                const res = await evaluate(live, ctx, `window.__runSpike('${name}')`);
                if (res.type !== 'success' || res.result?.value === undefined) {
                    throw new Error(`Spike '${name}' threw: ${JSON.stringify(res)}`);
                }
                return JSON.parse(res.result.value) as Record<string, unknown>;
            },
            async hostEvidence(): Promise<HostEvidence> {
                // `isFallbackAdapter` lives on GPUAdapterInfo in Firefox 155
                // (confirmed: it is in the GPUAdapterInfo prototype's own keys,
                // alongside vendor/architecture/device/description). Measured
                // false under the NVIDIA and Intel ICDs, true under llvmpipe.
                const res = await evaluate(
                    live,
                    ctx,
                    `navigator.gpu.requestAdapter().then(a => JSON.stringify({ f: a ? !!a.info.isFallbackAdapter : null }))`,
                );
                let isFallbackAdapter: boolean | undefined;
                if (res.type === 'success' && res.result?.value !== undefined) {
                    const parsed = JSON.parse(res.result.value) as { f: boolean | null };
                    isFallbackAdapter = parsed.f ?? undefined;
                }
                const pinPath = variant.env.VK_DRIVER_FILES;
                return {
                    isFallbackAdapter,
                    devicePin: pinPath ? describeIcdPin(pinPath) : null,
                };
            },
            close(): void {
                live.close();
                proc.kill();
            },
        };
    },
};
