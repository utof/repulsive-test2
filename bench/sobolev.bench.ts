/**
 * Per-phase benchmark harness for the Sobolev descent step (Repulsive Curves).
 *
 * Drives {@link sobolevStepSet} on parametric closed trefoils at two sizes and
 * two constraint modes, and reports median per-phase wall-clock via the opt-in
 * phase-timing collector (`collectTimings: true`), plus isolated micro-medians
 * for the five hot primitives. Results print as a markdown table and, with
 * `--save`, land in `bench/results/` as a git-SHA-keyed JSON ledger — the
 * "what improved with each iteration" record the milestone tracks. VM noise is
 * ±5% (briefing §1); treat |Δ| < 10% as noise unless reproduced.
 *
 * Usage:
 *   bun bench/sobolev.bench.ts                          # print table only
 *   bun bench/sobolev.bench.ts --save baseline          # + write results JSON
 *   bun bench/sobolev.bench.ts --baseline <path> --save e0-reuse   # + Δ% column
 *   bun bench/sobolev.bench.ts --big                    # add N=240
 *
 * @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 2)
 * @see local_files/2026-07-03-next-steps-briefing.md §1 (measured profile)
 */
import { execSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { DEFAULTS, type ProjectionMode, sobolevStepSet } from '../src/core/optimizer';
import {
    barycenterBlock,
    type ConstraintSet,
    edgeLengths,
    edgeLengthsBlock,
    evaluateConstraintSet,
    totalLength,
    totalLengthBlock,
} from '../src/core/sobolev/constraintSet';
import { barycenterTarget } from '../src/core/sobolev/constraints';
import { assembleA } from '../src/core/sobolev/innerProduct';
import { expandBlockDiag, flatten } from '../src/core/sobolev/layout';
import type { FactorMode } from '../src/core/sobolev/linsolve';
import { solveSaddle } from '../src/core/sobolev/linsolve';
import type { SobolevStepTimings } from '../src/core/sobolev/phaseTimings';
import {
    calculateDisjointPairs,
    calculateEnergy,
    gradientAnalytical,
} from '../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../src/core/testConfigs';

type ConstraintMode = 'total' | 'perEdge';

interface CaseResult {
    name: string;
    nV: number;
    nE: number;
    constraintMode: ConstraintMode;
    projectionMode: ProjectionMode;
    factorMode: FactorMode;
    phases: Record<string, { ms: number; calls: number }>;
    isolated: Record<string, number>;
    fullStepMsMedian: number;
}

interface ResultsFile {
    label: string;
    date: string;
    gitShaShort: string;
    bunVersion: string;
    cases: CaseResult[];
}

const { alpha, beta, epsilon } = DEFAULTS;

// Parametric closed trefoil: p(t) = (sin t + 2 sin 2t, cos t − 2 cos 2t, −sin 3t),
// t = 2πi/N, edges [i, (i+1) mod N]. Deterministic — no Math.random — so every
// repeated step does identical work (the step is pure). @see plan Task 2.
function trefoil(n: number): { vertices: Vec3[]; edges: Edge[] } {
    const vertices: Vec3[] = [];
    const edges: Edge[] = [];
    for (let i = 0; i < n; i++) {
        const t = (2 * Math.PI * i) / n;
        vertices.push([
            Math.sin(t) + 2 * Math.sin(2 * t),
            Math.cos(t) - 2 * Math.cos(2 * t),
            -Math.sin(3 * t),
        ]);
        edges.push([i, (i + 1) % n]);
    }
    return { vertices, edges };
}

function median(xs: number[]): number {
    const s = [...xs].sort((a, b) => a - b);
    const mid = Math.floor(s.length / 2);
    return s.length % 2 === 1 ? s[mid] : 0.5 * (s[mid - 1] + s[mid]);
}

const round3 = (x: number): number => Math.round(x * 1000) / 1000;

// Median of `reps` timed calls after 2 discarded warmups (JIT-stabilizing);
// mirrors the full-step methodology. @see plan Task 2.
function microMedian(fn: () => unknown, reps = 7): number {
    for (let w = 0; w < 2; w++) fn();
    const times: number[] = [];
    for (let i = 0; i < reps; i++) {
        const t0 = performance.now();
        fn();
        times.push(performance.now() - t0);
    }
    return median(times);
}

function buildSet(mode: ConstraintMode, vertices: Vec3[], edges: Edge[]): ConstraintSet {
    // Frozen targets from the INITIAL geometry, mirroring the store lifecycle
    // (spec §3.5). Barycenter FIRST (spec §3.2 row order).
    const x0 = barycenterTarget(vertices, edges);
    return mode === 'total'
        ? [barycenterBlock(x0), totalLengthBlock(totalLength(vertices, edges))]
        : [barycenterBlock(x0), edgeLengthsBlock(edgeLengths(vertices, edges))];
}

// Stable display order for the phase ledger (schema order of SobolevPhaseKey).
const PHASE_ORDER = [
    'dE',
    'energy',
    'bHigh',
    'bLow',
    'assembleA',
    'expand',
    'saddle',
    'factor',
    'projection',
    'lineSearch',
    'step',
];

// Case names: reassemble keeps the historical `N60-total` shape so Δ% joins
// against pre-Task-6 baselines still work; frozen cases get a `-frozen`
// suffix (projectionMode is a case dimension — plan Task 6 step 6.6) and
// LDLᵀ cases a `-ldlt` suffix (factorMode is the A/B dimension of the
// 2026-07-06 milestone — its kill gate reads the 'factor' phase p50 here).
// @see docs/superpowers/plans/2026-07-06-ldlt-factor.md (verification ladder c)
function runCase(
    nV: number,
    constraintMode: ConstraintMode,
    projectionMode: ProjectionMode,
    factorMode: FactorMode,
): CaseResult {
    const { vertices, edges } = trefoil(nV);
    const disjointPairs = calculateDisjointPairs(edges);
    const set = buildSet(constraintMode, vertices, edges);
    // E₀ reuse (Task 4): every measured repeat starts from the SAME `vertices`,
    // so calculateEnergy(vertices) is exactly the E₀ a continuous run would carry
    // over from its previous accepted step (bit-identical by the option's
    // invariant). Passing it mirrors the frame loop's per-step cost — one fewer
    // energy eval — instead of re-measuring the pre-reuse path.
    // @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 4)
    const energyBefore = calculateEnergy(vertices, edges, disjointPairs, alpha, beta, epsilon);
    const opts = {
        mode: 'analytical' as const,
        collectTimings: true,
        energyBefore,
        projectionMode,
        factorMode,
    };

    // 2 warmup full steps (discarded), then K=5 measured — each from the SAME
    // initial vertices (the step is pure/deterministic → identical work).
    for (let w = 0; w < 2; w++) sobolevStepSet(vertices, edges, disjointPairs, set, opts);

    const K = 5;
    const fullMs: number[] = [];
    const perStep: SobolevStepTimings[] = [];
    for (let k = 0; k < K; k++) {
        const t0 = performance.now();
        const r = sobolevStepSet(vertices, edges, disjointPairs, set, opts);
        fullMs.push(performance.now() - t0);
        if (r.timings) perStep.push(r.timings);
    }

    // Per-phase median ms; calls are deterministic across identical steps.
    const phases: Record<string, { ms: number; calls: number }> = {};
    const presentKeys = PHASE_ORDER.filter((k) =>
        perStep.some((t) => t[k as keyof SobolevStepTimings] !== undefined),
    );
    for (const key of presentKeys) {
        const k = key as keyof SobolevStepTimings;
        const mss = perStep.map((t) => t[k]?.ms).filter((x): x is number => x !== undefined);
        const calls = perStep[0]?.[k]?.calls ?? 0;
        phases[key] = { ms: round3(median(mss)), calls };
    }

    // Isolated micro-medians (of 7) — acc is null here (each sobolevStepSet
    // already disarmed the collector), so the timed() wraps are plain calls.
    const A = assembleA(vertices, edges, disjointPairs, alpha, beta, epsilon);
    const A3 = expandBlockDiag(A);
    const dEFlat = flatten(
        gradientAnalytical(vertices, edges, disjointPairs, alpha, beta, epsilon),
    );
    const { C } = evaluateConstraintSet(set, vertices, edges);
    const isolated: Record<string, number> = {
        calculateEnergy: round3(
            microMedian(() =>
                calculateEnergy(vertices, edges, disjointPairs, alpha, beta, epsilon),
            ),
        ),
        gradientAnalytical: round3(
            microMedian(() =>
                gradientAnalytical(vertices, edges, disjointPairs, alpha, beta, epsilon),
            ),
        ),
        assembleA: round3(
            microMedian(() => assembleA(vertices, edges, disjointPairs, alpha, beta, epsilon)),
        ),
        expandBlockDiag: round3(microMedian(() => expandBlockDiag(A))),
        solveSaddle: round3(microMedian(() => solveSaddle(A3, C, dEFlat))),
    };

    return {
        name: `N${nV}-${constraintMode}${projectionMode === 'frozen' ? '-frozen' : ''}${factorMode === 'ldlt' ? '-ldlt' : ''}`,
        nV,
        nE: edges.length,
        constraintMode,
        projectionMode,
        factorMode,
        phases,
        isolated,
        fullStepMsMedian: round3(median(fullMs)),
    };
}

// ── arg parsing ───────────────────────────────────────────────────────────
const argv = process.argv.slice(2);
const big = argv.includes('--big');
const saveIdx = argv.indexOf('--save');
const saveLabel = saveIdx >= 0 ? argv[saveIdx + 1] : undefined;
const baselineIdx = argv.indexOf('--baseline');
const baselinePath = baselineIdx >= 0 ? argv[baselineIdx + 1] : undefined;

const sizes = big ? [60, 120, 240] : [60, 120];
const modes: ConstraintMode[] = ['total', 'perEdge'];

let baseline: ResultsFile | undefined;
if (baselinePath) {
    baseline = JSON.parse(readFileSync(baselinePath, 'utf8')) as ResultsFile;
}

const projModes: ProjectionMode[] = ['reassemble', 'frozen'];
const factorModes: FactorMode[] = ['lu', 'ldlt'];

const cases: CaseResult[] = [];
for (const n of sizes) {
    for (const mode of modes) {
        for (const proj of projModes) {
            for (const fm of factorModes) {
                process.stderr.write(`running N${n}-${mode}-${proj}-${fm}…\n`);
                cases.push(runCase(n, mode, proj, fm));
            }
        }
    }
}

// ── markdown report ─────────────────────────────────────────────────────────
function pct(now: number, base: number | undefined): string {
    if (base === undefined || base === 0) return '';
    const d = ((now - base) / base) * 100;
    return ` (${d >= 0 ? '+' : ''}${d.toFixed(1)}%)`;
}

function baseCase(name: string): CaseResult | undefined {
    return baseline?.cases.find((c) => c.name === name);
}

const gitShaShort = (() => {
    try {
        return execSync('git rev-parse --short HEAD').toString().trim();
    } catch {
        return 'unknown';
    }
})();
const date = new Date().toISOString().slice(0, 10);

const lines: string[] = [];
lines.push(`# Sobolev step bench — ${date} · bun ${Bun.version} · ${gitShaShort}`);
if (baseline) lines.push(`Δ% vs baseline: **${baseline.label}** (${baseline.gitShaShort})`);
lines.push('');
lines.push('## Full step (median of 5)');
lines.push('| case | |V| | |E| | constraint | projection | factor | full step ms |');
lines.push('|---|---|---|---|---|---|---|');
for (const c of cases) {
    const b = baseCase(c.name);
    lines.push(
        `| ${c.name} | ${c.nV} | ${c.nE} | ${c.constraintMode} | ${c.projectionMode} | ${c.factorMode} | ${c.fullStepMsMedian}${pct(c.fullStepMsMedian, b?.fullStepMsMedian)} |`,
    );
}
lines.push('');
for (const c of cases) {
    const b = baseCase(c.name);
    lines.push(`## ${c.name} — per-phase (median ms × calls)`);
    lines.push('| phase | ms | calls |');
    lines.push('|---|---|---|');
    for (const key of PHASE_ORDER) {
        const p = c.phases[key];
        if (!p) continue;
        lines.push(`| ${key} | ${p.ms}${pct(p.ms, b?.phases[key]?.ms)} | ${p.calls} |`);
    }
    lines.push('');
    lines.push(`### ${c.name} — isolated primitives (median of 7)`);
    lines.push('| primitive | ms |');
    lines.push('|---|---|');
    for (const [key, ms] of Object.entries(c.isolated)) {
        lines.push(`| ${key} | ${ms}${pct(ms, b?.isolated[key])} |`);
    }
    lines.push('');
}
console.log(lines.join('\n'));

// ── save ────────────────────────────────────────────────────────────────────
if (saveLabel) {
    const out: ResultsFile = {
        label: saveLabel,
        date,
        gitShaShort,
        bunVersion: Bun.version,
        cases,
    };
    const dir = new URL('./results/', import.meta.url);
    mkdirSync(dir, { recursive: true });
    const path = new URL(`./results/${date}-${saveLabel}.json`, import.meta.url);
    writeFileSync(path, `${JSON.stringify(out, null, 2)}\n`);
    process.stderr.write(`saved ${path.pathname}\n`);
}
