// bench/gpu/drive.ts — bun bench/gpu/drive.ts <spike> [--out label]
// Launches a browser host against the dev server, certifies the WebGPU adapter
// (G0a), runs the named spike, and writes bench/results/<date>-gpu-<label>.json.
// INVALID (software / uncertifiable adapter) is a distinct outcome from FAIL —
// see README.md.
//
// The browser is a variant, not a constant: `GPU_HOST=firefox|chrome` picks.
// Firefox is the default because it is the browser installed on the dev box;
// hardcoding `google-chrome` here is what made every GPU kill gate unrunnable.
// @issue utof/repulsive-test2#54
// @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §3, §4 G0a
// @see docs/superpowers/plans/2026-08-13-webgpu-solver-phase0.md Task 1
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import { type AdapterCertification, certifyAdapter } from './adapterCert';
import { selectHost } from './hosts';
import type { BrowserHost, HostSession, LaunchVariant } from './hosts/types';

const PORT = Number(process.env.GPU_HOST_PORT ?? 9223);
const DEV_SERVER = process.env.DEV_SERVER_URL ?? 'http://localhost:3000';
const RESULTS_DIR = join(import.meta.dir, '..', 'results');

type AdapterInfo = {
    vendor: string;
    architecture: string;
    device: string;
    description: string;
};

const BLANK_ADAPTER: AdapterInfo = { vendor: '', architecture: '', device: '', description: '' };

type RunOutcome = {
    certification: AdapterCertification;
    adapter: AdapterInfo;
    data: Record<string, unknown> | null;
    consoleLines: string[];
    error?: string;
};

/** `'hardware'` iff the run's adapter was certified. @see bench/gpu/adapterCert.ts */
function isHardware(outcome: RunOutcome | undefined): boolean {
    return outcome?.certification.classification === 'hardware';
}

/**
 * Launch `host` with `variant`, run `spikeName` through the harness, tear down.
 *
 * The adapter is certified on every run regardless of the spike's own return
 * shape — previously `(data.adapter ?? data)` silently disarmed the
 * software-adapter INVALID guard for any spike that doesn't embed adapter
 * fields (every spike but adapterInfo itself).
 * @see https://github.com/utof/repulsive-test2/issues/18 pt.2
 */
async function runOnce(
    host: BrowserHost,
    variant: LaunchVariant,
    spikeName: string,
): Promise<RunOutcome> {
    let session: HostSession | undefined;
    try {
        session = await host.launch(variant, `${DEV_SERVER}/bench/gpu/harness.html`, PORT);
    } catch (e) {
        // Failure before we could even determine the adapter (launch/connect/
        // ready-poll timeout) — genuinely unknown, and `certifyAdapter` rules a
        // blank record with no evidence 'software', so it never masquerades as
        // a passing hardware run.
        return {
            certification: certifyAdapter(BLANK_ADAPTER, { devicePin: null }),
            adapter: BLANK_ADAPTER,
            data: null,
            consoleLines: [],
            error: e instanceof Error ? e.message : String(e),
        };
    }

    const live = session;
    try {
        const adapterData = (await live.evalSpike('adapterInfo')) as unknown as AdapterInfo;
        const certification = certifyAdapter(adapterData, await live.hostEvidence());

        if (spikeName === 'adapterInfo') {
            return {
                certification,
                adapter: adapterData,
                data: adapterData,
                consoleLines: live.consoleLines,
            };
        }

        try {
            const data = await live.evalSpike(spikeName);
            return { certification, adapter: adapterData, data, consoleLines: live.consoleLines };
        } catch (e) {
            // The adapter certification above is already established and real;
            // a failure evaluating the requested spike is a genuine FAIL
            // against known-hardware (or known-software), never a reason to
            // relabel the adapter itself as INVALID/software.
            // @see https://github.com/utof/repulsive-test2/issues/18 pt.2
            return {
                certification,
                adapter: adapterData,
                data: null,
                consoleLines: live.consoleLines,
                error: e instanceof Error ? e.message : String(e),
            };
        }
    } catch (e) {
        return {
            certification: certifyAdapter(BLANK_ADAPTER, { devicePin: null }),
            adapter: BLANK_ADAPTER,
            data: null,
            consoleLines: live.consoleLines,
            error: e instanceof Error ? e.message : String(e),
        };
    } finally {
        live.close();
    }
}

function gitShaShort(): string {
    const proc = Bun.spawnSync(['git', 'rev-parse', '--short', 'HEAD']);
    return proc.stdout.toString().trim();
}

/**
 * The launch recipe from the most recent PASSing G0a run **on this host**.
 *
 * Why the host filter: a Chrome flag set replayed under Firefox (or a Firefox
 * ICD pin replayed under Chrome) launches nothing usable, and — worse — a
 * recipe carrying no `VK_DRIVER_FILES` would silently drop the pin that the
 * Firefox certification depends on. Legacy records (pre-#54) carry a bare
 * `flags` array and no `host`; they are read only when the requested host is
 * `chrome`, which is what wrote them.
 * @issue utof/repulsive-test2#54
 */
function latestG0aVariant(hostId: BrowserHost['id']): LaunchVariant | null {
    if (!existsSync(RESULTS_DIR)) return null;
    const g0aFiles = readdirSync(RESULTS_DIR)
        .filter((f) => f.includes('gpu-g0a'))
        .sort()
        .reverse();
    for (const f of g0aFiles) {
        const parsed = JSON.parse(readFileSync(join(RESULTS_DIR, f), 'utf-8'));
        if (parsed.status !== 'PASS') continue;
        if (parsed.launch && parsed.host?.id === hostId) return parsed.launch as LaunchVariant;
        if (!parsed.host && parsed.flags && hostId === 'chrome') {
            return { label: 'legacy-g0a-flags', argv: parsed.flags, prefs: {}, env: {} };
        }
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

    const host = selectHost(process.env.GPU_HOST);
    const hostRecord = {
        id: host.id,
        version: host.version(),
        shaderBackend: host.shaderBackend,
    };
    console.error(
        `[drive] host=${hostRecord.id} (${hostRecord.version}) backend=${hostRecord.shaderBackend}`,
    );

    const isAdapterGate = spikeName === 'adapterInfo';
    const attempts: Array<{
        launch: LaunchVariant;
        certification: AdapterCertification;
        error?: string;
    }> = [];

    let winning: { variant: LaunchVariant; outcome: RunOutcome } | undefined;

    if (isAdapterGate) {
        for (const candidate of host.launchVariants) {
            const variant = host.resolveVariant(candidate);
            console.error(`[drive] trying ${variant.label}: ${variant.argv.join(' ')}`);
            const outcome = await runOnce(host, variant, spikeName);
            attempts.push({
                launch: variant,
                certification: outcome.certification,
                error: outcome.error,
            });
            console.error(
                `[drive]   -> ${outcome.certification.classification} via ${outcome.certification.method}: ${outcome.certification.rationale}${outcome.error ? ` (error: ${outcome.error})` : ''}`,
            );
            if (isHardware(outcome)) {
                winning = { variant, outcome };
                break;
            }
        }
    } else {
        const stored = latestG0aVariant(host.id);
        if (!stored) {
            console.error(
                `[drive] no passing G0a result for host '${host.id}' — run \`adapterInfo --out g0a\` first.`,
            );
            process.exit(1);
        }
        const variant = host.resolveVariant(stored);
        const outcome = await runOnce(host, variant, spikeName);
        attempts.push({
            launch: variant,
            certification: outcome.certification,
            error: outcome.error,
        });
        winning = { variant, outcome };
    }

    mkdirSync(RESULTS_DIR, { recursive: true });
    const date = new Date().toISOString().slice(0, 10);
    const gate = isAdapterGate ? 'G0a' : spikeName;
    const hardware = isHardware(winning?.outcome);
    const hasError = winning?.outcome.error !== undefined;
    // Why: a spike that RETURNS but reports its own red verdict (`data.pass === false`)
    // is a FAIL, not a PASS — without this only a thrown error was ever red, so a gate
    // that measured a failing number still wrote status: PASS and never printed the
    // STOP-BRANCH line below. Strict `=== false` so Phase 0 spikes with no `pass` field
    // are unaffected.
    // @see bench/gpu/README.md — "INVALID vs FAIL semantics"
    // @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md — "Task 4: Perf gate — GPU dE ≥5× CPU dE at N=480 and N=960 (readback counted)"
    const selfReportedFail = (winning?.outcome.data as { pass?: unknown } | null)?.pass === false;
    // Distinguish adapter certification from spike-eval outcome (#18 pt.2):
    // an uncertified adapter is always INVALID; a certified-hardware adapter
    // with a spike error is a real FAIL, not an INVALID relabeling.
    const status = isAdapterGate
        ? hardware && !hasError
            ? 'PASS'
            : 'FAIL' // STOP-BRANCH: no launch recipe yielded a certified hardware adapter
        : !winning || !hardware
          ? 'INVALID' // software (or uncertifiable) adapter is INVALID, not FAIL
          : hasError || selfReportedFail
            ? 'FAIL' // hardware adapter, but the spike threw or reported pass: false
            : 'PASS';

    const result = {
        gate,
        status,
        // The host that produced this measurement. A compensated-arithmetic
        // metric measured under `shaderBackend: 'naga'` is NOT evidence about
        // Dawn and vice versa — WGSL permits reassociation and fusion with no
        // portable opt-out, and naga is measured to delete Dekker compensation
        // terms Dawn preserves.
        // @see https://github.com/gpuweb/gpuweb/issues/2076
        // @issue utof/repulsive-test2#51 trap 3
        host: hostRecord,
        adapter: winning?.outcome.adapter ?? null,
        // How that adapter was certified — read this before believing `status`.
        // @see bench/gpu/adapterCert.ts
        certification: winning?.outcome.certification ?? null,
        launch: winning?.variant ?? null,
        // Retained under its historical name so existing readers of
        // `bench/results/*.json` keep working.
        flags: winning?.variant.argv ?? null,
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

// A configuration error (unknown `GPU_HOST`, missing browser binary) is an
// operator mistake, not a gate result — report it as one line and exit 1
// rather than dumping a module-load stack trace that reads like a crash.
main().catch((e) => {
    console.error(`[drive] ${e instanceof Error ? e.message : String(e)}`);
    process.exit(1);
});
