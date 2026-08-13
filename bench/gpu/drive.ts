// bench/gpu/drive.ts — bun bench/gpu/drive.ts <spike> [--out label]
// Launches Chrome headless (or headed, as a fallback) against the dev server,
// asserts a hardware WebGPU adapter (G0a), runs the named spike, and writes
// bench/results/<date>-gpu-<label>.json.
// INVALID (software adapter) is a distinct outcome from FAIL — see README.md.
// @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §3, §4 G0a
// @see docs/superpowers/plans/2026-08-13-webgpu-solver-phase0.md Task 1
import { type ChildProcessWithoutNullStreams, spawn } from 'node:child_process';
import {
    existsSync,
    mkdirSync,
    mkdtempSync,
    readdirSync,
    readFileSync,
    writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const CHROME = process.env.CHROME_BIN ?? 'google-chrome';
const PORT = 9223;
const DEV_SERVER = process.env.DEV_SERVER_URL ?? 'http://localhost:3000';
const RESULTS_DIR = join(import.meta.dir, '..', 'results');

/**
 * Candidate flag sets, tried in order until `adapterInfo` reports NVIDIA
 * hardware. The third set is a headed fallback — spec G0a explicitly allows
 * a headed pass after an honest headless attempt.
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §4 G0a
 */
const FLAG_SETS: string[][] = [
    [
        '--headless=new',
        '--enable-unsafe-webgpu',
        '--enable-features=Vulkan',
        '--use-angle=vulkan',
        '--enable-gpu',
        '--no-sandbox',
    ],
    [
        '--headless=new',
        '--enable-unsafe-webgpu',
        '--enable-features=Vulkan',
        '--ignore-gpu-blocklist',
        '--no-sandbox',
    ],
    // headed fallback (spec G0a: allowed after honest headless effort)
    ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--no-sandbox'],
];

/**
 * Known SwiftShader/software-fallback console noise lines to filter out of
 * captured console output (they fire even on a hardware adapter and are not
 * evidence of anything). @see bench/gpu/README.md
 */
const CONSOLE_NOISE = [
    /RangeError: createBuffer failed, size \(144\) is too large/,
    /Instance dropped in popErrorScope/,
    /Warning: Encountered two children with the same key, `0`/,
];

type AdapterInfo = {
    vendor: string;
    architecture: string;
    device: string;
    description: string;
};

/**
 * PASS/FAIL classification of an adapter, per spec §3: any result obtained
 * against a software rasterizer is INVALID, never a passing (or failing) gate
 * measurement. @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §3
 */
function classifyAdapter(info: { vendor: string; description: string }): 'hardware' | 'software' {
    const d = `${info.vendor} ${info.description}`.toLowerCase();
    return d.includes('swiftshader') || d.includes('llvmpipe') || d.includes('software')
        ? 'software'
        : 'hardware';
}

function isNoise(line: string): boolean {
    return CONSOLE_NOISE.some((re) => re.test(line));
}

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

type RunOutcome = {
    classification: 'hardware' | 'software';
    adapter: AdapterInfo;
    data: Record<string, unknown> | null;
    consoleLines: string[];
    error?: string;
};

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

/** Evaluate `window.__runSpike(name)` in the page and parse its JSON result. */
async function evalSpike(client: CDPClient, name: string): Promise<Record<string, unknown>> {
    const evalResult = (await client.send('Runtime.evaluate', {
        expression: `window.__runSpike('${name}')`,
        awaitPromise: true,
        returnByValue: true,
    })) as { exceptionDetails?: unknown; result: { value: string } };
    if (evalResult.exceptionDetails) {
        throw new Error(`Spike '${name}' threw: ${JSON.stringify(evalResult.exceptionDetails)}`);
    }
    return JSON.parse(evalResult.result.value) as Record<string, unknown>;
}

/** Launch Chrome with `flags`, run `spikeName` via the harness, tear down. */
async function runOnce(flags: string[], spikeName: string): Promise<RunOutcome> {
    const profileDir = mkdtempSync(join(tmpdir(), 'gpu-gate-chrome-'));
    const args = [
        ...flags,
        `--remote-debugging-port=${PORT}`,
        `--user-data-dir=${profileDir}`,
        `${DEV_SERVER}/bench/gpu/harness.html`,
    ];
    const proc: ChildProcessWithoutNullStreams = spawn(CHROME, args, { stdio: 'pipe' });
    let stderr = '';
    proc.stderr.on('data', (d) => {
        stderr += d.toString();
    });

    let client: CDPClient | undefined;
    try {
        const target = await waitForTarget(PORT);
        client = new CDPClient(target.webSocketDebuggerUrl);
        await client.connect();
        await client.send('Runtime.enable');
        // The CDP target for harness.html exists as soon as the tab is created,
        // well before the page has navigated/loaded and spikes.ts has run (which
        // is what defines window.__runSpike) — poll for readiness instead of
        // racing straight into Runtime.evaluate.
        await waitForRunSpikeReady(client);

        // Explicit adapter classification, run for every spike regardless of
        // its own return shape. Previously `(data.adapter ?? data)` silently
        // disarmed the software-adapter INVALID guard for any spike that
        // doesn't embed adapter fields (every spike but adapterInfo itself).
        // @see https://github.com/utof/repulsive-test2/issues/18 pt.2
        const adapterData = (await evalSpike(client, 'adapterInfo')) as unknown as AdapterInfo;
        const classification = classifyAdapter(adapterData);

        if (spikeName === 'adapterInfo') {
            return {
                classification,
                adapter: adapterData,
                data: adapterData,
                consoleLines: client.consoleLines,
            };
        }

        try {
            const data = await evalSpike(client, spikeName);
            return {
                classification,
                adapter: adapterData,
                data,
                consoleLines: client.consoleLines,
            };
        } catch (e) {
            // The adapter classification above is already established and
            // real; a failure evaluating the requested spike is a genuine
            // FAIL against known-hardware (or known-software), never a
            // reason to relabel the adapter itself as INVALID/software.
            // @see https://github.com/utof/repulsive-test2/issues/18 pt.2
            return {
                classification,
                adapter: adapterData,
                data: null,
                consoleLines: client.consoleLines,
                error: `${e instanceof Error ? e.message : String(e)}\nstderr: ${stderr}`,
            };
        }
    } catch (e) {
        // Failure before we could even determine the adapter (target/connect/
        // ready-poll timeout) — genuinely unknown, treated as 'software' so
        // it never masquerades as a passing hardware run.
        return {
            classification: 'software',
            adapter: { vendor: '', architecture: '', device: '', description: '' },
            data: null,
            consoleLines: client?.consoleLines ?? [],
            error: `${e instanceof Error ? e.message : String(e)}\nstderr: ${stderr}`,
        };
    } finally {
        client?.close();
        proc.kill();
    }
}

function gitShaShort(): string {
    const proc = Bun.spawnSync(['git', 'rev-parse', '--short', 'HEAD']);
    return proc.stdout.toString().trim();
}

function latestG0aFlags(): string[] | null {
    if (!existsSync(RESULTS_DIR)) return null;
    const g0aFiles = readdirSync(RESULTS_DIR)
        .filter((f) => f.includes('gpu-g0a'))
        .sort()
        .reverse();
    for (const f of g0aFiles) {
        const parsed = JSON.parse(readFileSync(join(RESULTS_DIR, f), 'utf-8'));
        if (parsed.status === 'PASS') return parsed.flags;
    }
    return null;
}

async function main() {
    const [, , spikeName, ...rest] = process.argv;
    if (!spikeName) {
        console.error('usage: bun bench/gpu/drive.ts <spike> [--out label]');
        process.exit(1);
    }
    const outIdx = rest.indexOf('--out');
    const label = outIdx >= 0 ? rest[outIdx + 1] : spikeName;

    const isAdapterGate = spikeName === 'adapterInfo';
    const attempts: Array<{
        flags: string[];
        classification: string;
        adapter: AdapterInfo;
        error?: string;
    }> = [];

    let winning: { flags: string[]; outcome: RunOutcome } | undefined;

    if (isAdapterGate) {
        for (const flags of FLAG_SETS) {
            console.error(`[drive] trying flags: ${flags.join(' ')}`);
            const outcome = await runOnce(flags, spikeName);
            attempts.push({
                flags,
                classification: outcome.classification,
                adapter: outcome.adapter,
                error: outcome.error,
            });
            console.error(
                `[drive]   -> ${outcome.classification}${outcome.error ? ` (error: ${outcome.error})` : ''} ${JSON.stringify(outcome.adapter)}`,
            );
            if (outcome.classification === 'hardware') {
                winning = { flags, outcome };
                break;
            }
        }
    } else {
        const flags = latestG0aFlags();
        if (!flags) {
            console.error(
                '[drive] no passing G0a result found — run `adapterInfo --out g0a` first.',
            );
            process.exit(1);
        }
        const outcome = await runOnce(flags, spikeName);
        attempts.push({
            flags,
            classification: outcome.classification,
            adapter: outcome.adapter,
            error: outcome.error,
        });
        winning = { flags, outcome };
    }

    mkdirSync(RESULTS_DIR, { recursive: true });
    const date = new Date().toISOString().slice(0, 10);
    const gate = isAdapterGate ? 'G0a' : spikeName;
    const isHardware = winning?.outcome.classification === 'hardware';
    const hasError = winning?.outcome.error !== undefined;
    // Distinguish adapter classification from spike-eval outcome (#18 pt.2):
    // software adapter is always INVALID; a hardware adapter with a spike
    // error is a real FAIL, not an INVALID relabeling of the adapter.
    const status = isAdapterGate
        ? isHardware && !hasError
            ? 'PASS'
            : 'FAIL' // STOP-BRANCH: no flag set yielded a hardware adapter
        : !winning || !isHardware
          ? 'INVALID' // software (or undetermined) adapter is INVALID, not FAIL
          : hasError
            ? 'FAIL' // hardware adapter, but the spike itself failed
            : 'PASS';

    const result = {
        gate,
        status,
        adapter: winning?.outcome.adapter ?? null,
        flags: winning?.flags ?? null,
        attempts,
        gitShaShort: gitShaShort(),
        date,
        data: winning?.outcome.data ?? null,
        consoleLines: winning?.outcome.consoleLines ?? [],
    };

    const outPath = join(RESULTS_DIR, `${date}-gpu-${label}.json`);
    writeFileSync(outPath, `${JSON.stringify(result, null, 2)}\n`);
    console.error(`[drive] wrote ${outPath}`);

    if (status !== 'PASS') {
        console.error(
            `[drive] STOP-BRANCH: gate ${gate} did not PASS (status=${status}). See ${outPath} for diagnostics.`,
        );
        process.exit(1);
    }
    console.log(JSON.stringify(result, null, 2));
}

main();
