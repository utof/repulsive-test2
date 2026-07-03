/**
 * Stable top-level phase keys. Sub-phases (bHigh/bLow inside assembleA;
 * later 'factor' inside 'saddle') OVERLAP their parents — sums across keys
 * double-count by design; compare like-to-like across bench runs.
 * Why: keys are the bench ledger's schema — renaming one breaks baseline
 * comparability in bench/results/. Add keys; never rename.
 * @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 1)
 */
export type SobolevPhaseKey =
    | 'dE' // differential (analytical or FD), timed at call site
    | 'energy' // calculateEnergy calls (E₀ + Armijo trials), timed at call sites
    | 'bHigh' // assembleBHigh body
    | 'bLow' // assembleBLow body
    | 'assembleA' // assembleA total (parent of bHigh/bLow + the sum loop)
    | 'expand' // expandBlockDiag (drops to 0 calls after Task 4)
    | 'saddle' // one whole saddle solve (build K + factor + backsolve + residual)
    | 'factor' // LU factorization only (appears after Task 4's split)
    | 'projection' // projectOntoConstraintSet total
    | 'lineSearch' // lineSearchStepSet total
    | 'step'; // sobolevStepSet total

/**
 * One phase's accumulated cost across a single step: total wall-clock `ms` and
 * `calls` (a phase can fire multiple times per step — e.g. assembleA once per
 * gradient/projection solve). @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 1)
 */
export interface PhaseSample {
    ms: number;
    calls: number;
}
/**
 * The per-step ledger: a partial map of phase key → {@link PhaseSample}. Only
 * phases that actually fired are present (hence Partial). Attached top-level to
 * `sobolevStepSet`'s result when `collectTimings` is set (NOT inside `stats`,
 * which is compared with toEqual across calls).
 * @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 1)
 */
export type SobolevStepTimings = Partial<Record<SobolevPhaseKey, PhaseSample>>;

let acc: SobolevStepTimings | null = null; // null = collection off (default)

/**
 * Arms the module-scoped collector for one step. Paired with
 * {@link timingsEnd}; single-threaded frame loop / bench only (see reentrancy
 * note in the plan).
 * @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 1)
 */
export function timingsBegin(): void {
    acc = {};
}

/**
 * Disarms the collector and returns the accumulated ledger (null if never
 * armed).
 * @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 1)
 */
export function timingsEnd(): SobolevStepTimings | null {
    const r = acc;
    acc = null;
    return r;
}

/**
 * Wrap a phase. When collection is off this is a plain call — zero overhead
 * beyond one null check; numerics are NEVER affected (pure timing): the wrapped
 * value is returned untouched, so `timed(k, fn)` is observably `fn()`.
 * @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 1)
 */
export function timed<T>(key: SobolevPhaseKey, fn: () => T): T {
    if (acc === null) return fn();
    const t0 = performance.now();
    const r = fn();
    const dt = performance.now() - t0;
    const s = acc[key] ?? (acc[key] = { ms: 0, calls: 0 });
    s.ms += dt;
    s.calls += 1;
    return r;
}
