import {
    type ProjectionMode,
    type SobolevStepStats,
    sobolevStep,
    sobolevStepSet,
    step,
} from './optimizer';
import {
    assertValidConstraintSet,
    barycenterBlock,
    type ConstraintSet,
    edgeLengthsBlock,
    pointBlock,
    totalLengthBlock,
} from './sobolev/constraintSet';
import type { PenaltyConfig } from './sobolev/penalties';
import type { SobolevStepTimings } from './sobolev/phaseTimings';
import type { Edge, Vec3 } from './testConfigs';

// Worker-prep extraction (§D2): dispatchDescentStep + its arg/result types +
// the store-local types it needs (Mode/DescentMode/LengthMode/PinConstraint/
// DescentStepOutcome) moved here VERBATIM from src/store.ts so the same pure
// function can run in a Web Worker; src/store.ts RE-EXPORTS all of these so the
// existing test imports and Viewer.tsx keep working unchanged. Direction is
// store→core ONLY: this module (and all of src/core/**) must never import from
// src/store.ts or any React/zustand module (worker-bundle purity).
// @see docs/superpowers/plans/2026-07-04-worker-solver.md §D2

export type Mode = 'analytical' | 'finiteDiff';

/**
 * Which descent drives the frame loop. 'raw' is the original fixed-step L²
 * gradient descent (τ ≈ 1e-5 scale) and must stay byte-identical — the whole
 * point of the toggle is an A/B comparison against 'sobolev', the constrained
 * fractional Sobolev descent (τ ≈ 1 scale).
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §C
 */
export type DescentMode = 'raw' | 'sobolev';

/**
 * 3-way length-constraint mode for the sobolev ConstraintSet (spec §5.3):
 * 'none' | 'total' (M1 total-length row) | 'perEdge' (M2, |E| rows). The
 * §3.4 totalLength/edgeLengths mutual exclusion is enforced BY CONSTRUCTION —
 * one select, one value. 'total' is the default (preserves the M1 default
 * lengthConstraint = true).
 * @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §5.3, §3.4
 */
export type LengthMode = 'none' | 'total' | 'perEdge';

/**
 * One interactive point-pin constraint: hold vertex `vertexIndex` at the
 * FROZEN world-space `target` (fed to `pointBlock`, Φ = γ_i − target). `target`
 * is in the same coordinates as the `live` buffer. `enabled` is the per-pin UI
 * toggle. `target` is a FROZEN constraint target with the EXACT sobolevEll0
 * lifecycle (frozen-targets anchor below): the frame loop READS pins and NEVER
 * writes them — only user actions (add = snapshot live; drag = ray∩plane; the
 * play/commit/rebuild re-anchor) mutate a target, because a pin whose target
 * tracked the current iterate would be vacuous.
 * @see src/core/sobolev/constraintSet.ts (pointBlock — Φ, identity C, out-of-range NaN backstop)
 * @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §2, §3.5, §5.3
 * @see docs/superpowers/plans/2026-07-03-pin-drag-ui.md (Decisions D2, D5)
 */
export interface PinConstraint {
    vertexIndex: number;
    target: Vec3;
    enabled: boolean;
}

/**
 * Result of {@link dispatchDescentStep}: the union shape of the two steppers.
 * The raw path always reports `accepted: true, converged: false, stats: null`
 * (it has no line search, no termination test, no saddle solve).
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §C (steps 5, 10)
 */
export interface DescentStepOutcome {
    vertices: Vec3[];
    energy: number;
    accepted: boolean;
    converged: boolean;
    stats: SobolevStepStats | null;
    // Per-phase step timings when `collectTimings` was requested in sobolev
    // mode; null in raw mode and when timings were not collected. Surfaced to
    // the UI (Stats.tsx second line) via the store's `sobolevTimings`.
    // @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 3)
    timings: SobolevStepTimings | null;
}

/**
 * The full argument object of {@link dispatchDescentStep}, named so the worker
 * protocol can derive {@link DispatchStepArgs} (this MINUS the topology fields)
 * and so {@link buildStepArgs} has a single typed return contract that both
 * drivers share (the drift guard of §D7).
 * @see docs/superpowers/plans/2026-07-04-worker-solver.md §D7, §D12
 */
export interface DispatchDescentStepArgs {
    descentMode: DescentMode;
    vertices: Vec3[];
    edges: Edge[];
    disjointPairs: number[][];
    mode: Mode;
    stepSize: number;
    x0: Vec3;
    barycenterConstraint?: boolean;
    lengthConstraint?: boolean;
    lengthMode?: LengthMode;
    sobolevL0?: number;
    sobolevEll0?: number[];
    // Opt into per-phase step timings (sobolev mode only); the raw path ignores
    // it and reports `timings: null`.
    // @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 3)
    collectTimings?: boolean;
    // Precomputed E₀ = E(γ₀) at `vertices`, reused as the sobolev step's Armijo
    // baseline instead of recomputing calculateEnergy. MUST be exactly
    // calculateEnergy(vertices, …); the frame loop supplies the previous accepted
    // step's returned energy and nulls it at every !running boundary so staleness
    // is structurally impossible. Ignored on the raw path.
    // @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 4)
    energyBefore?: number;
    // Projection solve strategy passthrough (solver-perf Task 6). Absent →
    // sobolevStepSet's default ('reassemble'), so pre-existing call sites and
    // tests are bit-identical; the app passes the store's projectionMode
    // (store default 'frozen' — the reference-implementation scheme).
    // @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 6)
    projectionMode?: ProjectionMode;
    // Interactive point pins (pin-drag milestone, briefing §5B). Each ENABLED,
    // in-range pin appends one `pointBlock(vertexIndex, target)` AFTER the length
    // block (row order: barycenter, length, pins). Absent/empty → no pointBlocks,
    // bit-identical to the pre-pin dispatch. Disabled or out-of-range pins are
    // dropped so a stale pin can never break the frame loop's descent.
    // @see docs/superpowers/plans/2026-07-03-pin-drag-ui.md (Decision D6)
    // @see src/core/sobolev/constraintSet.ts (pointBlock)
    pins?: PinConstraint[];
    // Soft-constraint penalties (5C): threaded VERBATIM into the sobolev step's
    // opts.penalties. Absent or all-zero ⇒ the core gates on `penaltiesActive`
    // and every code path stays bit-identical to the penalty-free build
    // (plan §2.4); no dispatch-side gating needed. Penalties enter the OBJECTIVE
    // only (energy + dE), never the constraint set — so `set` is untouched.
    // @see docs/superpowers/plans/2026-07-03-sobolev-penalties.md §2.4
    penalties?: PenaltyConfig;
}

/**
 * The descent-mode dispatch, pure and store-independent so it is testable the
 * same way as {@link buildGraphState}: 'sobolev' → constrained Sobolev flow
 * over the ConstraintSet built from the toggle args (barycenter block FIRST
 * when present — spec §3.2 row order; both-off = the empty set, spec §9a),
 * 'raw' → the pre-existing `step()` with the exact arguments the frame loop
 * always passed — the raw path must remain byte-identical when the toggle is
 * 'raw'.
 *
 * The toggle args are OPTIONAL: when BOTH `barycenterConstraint` and
 * `lengthConstraint` are absent this is a pre-M1 call shape and delegates to
 * the legacy barycenter-only `sobolevStep(x0)` bit-identically, so
 * pre-existing call sites and tests are unaffected. An absent individual
 * toggle defaults to true (the store defaults). `x0`, `sobolevL0` and
 * `sobolevEll0` are FROZEN targets (store lifecycle anchor, spec §3.5) — never
 * recomputed here.
 *
 * M2 length precedence (spec §5.3): the 3-way `lengthMode` supersedes the M1
 * `lengthConstraint` boolean; when `lengthMode` is absent it degrades to the M1
 * semantics (`lengthConstraint ?? true` → 'total') so M1 call sites stay
 * bit-identical. The §3.4 totalLength/edgeLengths mutual exclusion is
 * structural — one mode selects at most one length block.
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §C
 * @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §4.2, §5.3, §9a
 */
export function dispatchDescentStep(args: DispatchDescentStepArgs): DescentStepOutcome {
    if (args.descentMode === 'sobolev') {
        if (
            args.barycenterConstraint === undefined &&
            args.lengthConstraint === undefined &&
            args.lengthMode === undefined
        ) {
            // Pre-M1 call shape: legacy barycenter-only path, bit-identical to
            // sobolevStep(x0) (spec §4.2 back-compat).
            const r = sobolevStep(args.vertices, args.edges, args.disjointPairs, args.x0, {
                mode: args.mode,
            });
            return {
                vertices: r.vertices,
                energy: r.energy,
                accepted: r.accepted,
                converged: r.converged,
                stats: r.stats,
                // Pre-M1 back-compat shape: never collects timings (the app never
                // takes this branch — it always passes the M-set toggles).
                timings: null,
            };
        }
        const set: ConstraintSet = [];
        if (args.barycenterConstraint ?? true) set.push(barycenterBlock(args.x0));
        // M2 (spec §5.3): the 3-way lengthMode supersedes the M1 boolean; when
        // absent it degrades to the M1 semantics (lengthConstraint ?? true →
        // 'total') so M1 call sites stay bit-identical.
        const lengthMode: LengthMode =
            args.lengthMode ?? ((args.lengthConstraint ?? true) ? 'total' : 'none');
        if (lengthMode === 'total') {
            // An enabled length constraint requires its frozen L⁰ (spec §3.5).
            // NaN backstop if a caller omits it: Φ becomes NaN, projection can
            // never converge, and the step is REJECTED ('projection_failed')
            // instead of silently drifting or throwing in the frame loop.
            set.push(totalLengthBlock(args.sobolevL0 ?? Number.NaN));
        } else if (lengthMode === 'perEdge') {
            // Same NaN backstop for a missing frozen ℓ⁰ vector: NaN Φ rows →
            // projection can't converge → 'projection_failed', never a throw.
            set.push(edgeLengthsBlock(args.sobolevEll0 ?? args.edges.map(() => Number.NaN)));
        }
        // Interactive pins (briefing §5B): append one pointBlock per ENABLED,
        // in-range pin AFTER the length block (row order barycenter, length,
        // pins). Out-of-range/disabled pins are skipped — a stale pin must never
        // break the frame loop (pointBlock's own NaN backstop would only surface
        // as projection_failed, but dropping it here keeps the rest of the
        // descent alive).
        // @see docs/superpowers/plans/2026-07-03-pin-drag-ui.md (Decision D6)
        for (const pin of args.pins ?? []) {
            if (pin.enabled && pin.vertexIndex >= 0 && pin.vertexIndex < args.vertices.length) {
                set.push(pointBlock(pin.vertexIndex, pin.target));
            }
        }
        // Construction-time rank-rule check (spec §3.4) — validate the set once
        // here, not per-iterate inside sobolevStepSet.
        assertValidConstraintSet(set);
        const r = sobolevStepSet(args.vertices, args.edges, args.disjointPairs, set, {
            mode: args.mode,
            collectTimings: args.collectTimings,
            // E₀ reuse (Task 4): passthrough to the step; undefined → recompute.
            // @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 4)
            energyBefore: args.energyBefore,
            // Projection strategy (Task 6): passthrough; undefined → 'reassemble'.
            projectionMode: args.projectionMode,
            // Penalties (5C): passthrough; undefined/all-zero → core no-ops via
            // penaltiesActive, bit-identical (plan §2.4).
            // @see docs/superpowers/plans/2026-07-03-sobolev-penalties.md §2.4
            penalties: args.penalties,
        });
        return {
            vertices: r.vertices,
            energy: r.energy,
            accepted: r.accepted,
            converged: r.converged,
            stats: r.stats,
            timings: r.timings ?? null,
        };
    }
    const r = step(args.vertices, args.edges, args.disjointPairs, {
        mode: args.mode,
        stepSize: args.stepSize,
    });
    return {
        vertices: r.vertices,
        energy: r.energy,
        accepted: true,
        converged: false,
        stats: null,
        timings: null,
    };
}

/**
 * Narrow structural view of the store state that {@link buildStepArgs} reads —
 * ONLY the fields it maps into the dispatch args (NOT the full SimStore). Both
 * drivers pass the live store through this shape so the worker and main paths
 * can never drift in how they assemble a step (§D7).
 * @see docs/superpowers/plans/2026-07-04-worker-solver.md §D7
 */
export interface StepArgsSource {
    descentMode: DescentMode;
    live: Vec3[];
    graph: { edges: Edge[] };
    disjointPairs: number[][];
    mode: Mode;
    stepSize: number;
    sobolevX0: Vec3;
    sobolevL0: number;
    barycenterConstraint: boolean;
    lengthMode: LengthMode;
    sobolevEll0: number[];
    pins: PinConstraint[];
    projectionMode: ProjectionMode;
    penalties: PenaltyConfig;
}

/**
 * Assemble the {@link dispatchDescentStep} argument object from the store state
 * and a reused E₀. This is the SINGLE param-assembly both drivers call (§D7):
 * the main driver spreads `collectTimings: true` and dispatches inline; the
 * worker driver strips the topology fields (edges/disjointPairs — the worker
 * restores them from its cache, §D4) and posts the rest. Extracting it here is
 * what prevents worker/main parameter drift. `collectTimings` is intentionally
 * NOT set here (hardcoded at each call site, as in the pre-worker frame loop).
 * @see docs/superpowers/plans/2026-07-04-worker-solver.md §D7
 */
export function buildStepArgs(
    state: StepArgsSource,
    energyBefore: number | undefined,
): Omit<DispatchDescentStepArgs, 'collectTimings'> {
    return {
        descentMode: state.descentMode,
        vertices: state.live,
        edges: state.graph.edges,
        disjointPairs: state.disjointPairs,
        mode: state.mode,
        stepSize: state.stepSize,
        // Frozen targets + per-block toggles: dispatch builds the ConstraintSet
        // (barycenter first) from these; the frame loop only READS them.
        // @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §4.2, §5.3, §9a
        x0: state.sobolevX0,
        sobolevL0: state.sobolevL0,
        barycenterConstraint: state.barycenterConstraint,
        lengthMode: state.lengthMode,
        sobolevEll0: state.sobolevEll0,
        pins: state.pins,
        projectionMode: state.projectionMode,
        penalties: state.penalties,
        // Reuse the previous accepted step's E₀ (undefined on a run's first step
        // → fresh recompute). @see plan §2 / solver-perf Task 4.
        energyBefore,
    };
}

/**
 * Per-step worker payload: {@link DispatchDescentStepArgs} MINUS the topology
 * fields (edges, disjointPairs) which the worker restores from its topology
 * cache so per-step messages never carry the O(E²) disjointPairs (§D4).
 * @see docs/superpowers/plans/2026-07-04-worker-solver.md §D4, §D12
 */
export type DispatchStepArgs = Omit<DispatchDescentStepArgs, 'edges' | 'disjointPairs'>;

/**
 * Main→worker protocol (§D4/§D12). `topology` is sent once on init and on every
 * graphVersion change (the worker recomputes disjointPairs via the same
 * deterministic calculateDisjointPairs); `step` carries only the dynamic
 * per-step args tagged with the graphVersion the main thread will match on §D5.
 * @see docs/superpowers/plans/2026-07-04-worker-solver.md §D4, §D12
 */
export type SolverWorkerRequest =
    | { type: 'topology'; graphVersion: number; edges: Edge[] }
    | { type: 'step'; graphVersion: number; args: DispatchStepArgs };

/**
 * Worker→main protocol (§D5/§D12). `result` echoes the request's graphVersion
 * UNTOUCHED so the main thread can DROP a result whose graphVersion no longer
 * matches the store (a preset rebuild / pause landed mid-flight); `error` is
 * posted on any worker-side throw and triggers the §D6 auto-fallback to 'main'.
 * @see docs/superpowers/plans/2026-07-04-worker-solver.md §D5, §D6, §D12
 */
export type SolverWorkerResponse =
    | { type: 'result'; graphVersion: number; result: DescentStepOutcome }
    | { type: 'error'; message: string };
