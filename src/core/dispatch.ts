import {
    type BarnesHutContext,
    barnesHutEnergy,
    barnesHutGradient,
    prepareBarnesHut,
} from './barnesHut';
import {
    DEFAULTS,
    type ProjectionMode,
    SOBOLEV_CONVERGENCE_TOL,
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
import { makeHierarchicalSaddle } from './sobolev/hierarchical/hierarchicalSaddle';
import type { PreparedSaddle } from './sobolev/linsolve';
import type { PenaltyConfig } from './sobolev/penalties';
import type { SobolevStepTimings } from './sobolev/phaseTimings';
import {
    HIERARCHICAL_ACCURACY,
    type HierarchicalAccuracy,
    needsRefinement,
    pairListFits,
    resolveSobolevRoute,
    type SobolevSolver,
} from './sobolevRoute';
import { calculateEnergy } from './tangentPointEnergy';
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
    // The descent field the step ACTUALLY used, at the step's INPUT vertices,
    // when `collectField` was requested (§D14 / issue #9): raw mode → the L²
    // gradient dE; sobolev mode → the g̃ from the FULL ConstraintSet saddle solve
    // (NOT the legacy barycenter-only field). `null` when NOT collected, and
    // `null` when the sobolev saddle was singular (no g̃ exists). Populated on
    // accepted AND rejected/converged outcomes whenever the field was computed —
    // a rejected line search still solved g̃. Named `descentField` (not `field`)
    // to avoid collision with the penalties' `field` concept. GradientArrows
    // renders it directly while running instead of a redundant second-worker
    // recompute; published store-side as `arrowField`.
    // @see docs/superpowers/plans/2026-07-04-worker-solver.md §D14
    descentField: Vec3[] | null;
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
    // Opt into {@link DescentStepOutcome.descentField} — the field the step used
    // at its INPUT vertices (raw dE / sobolev g̃), for the GradientArrows
    // diagnostic (§D14 / issue #9). Sibling of `collectTimings`; threaded to
    // `step` / `sobolevStepSet` the same way. Absent ⇒ `descentField: null`, every
    // code path bit-identical to today. @see plan §D14.
    collectField?: boolean;
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
    // Precomputed dE at `vertices` (WebGPU Phase 1 seam): sobolev+ConstraintSet path
    // only; raw/legacy branches ignore it — true THROUGH dispatchDescentStep only, as
    // the legacy `sobolevStep` forwards opts verbatim (optimizer.ts), so a DIRECT
    // caller passing dE does get it honoured. Rides the worker `step` payload if set;
    // `buildStepArgs` never sets it (the 'gpu' driver dispatches inline).
    // @see SobolevStepOptions.dE (optimizer.ts)
    dE?: Vec3[];
    /**
     * A saddle solve prepared for `vertices` (WebGPU Phase 2a seam). Forwarded to
     * `sobolevStepSet` exactly like `dE`. Carries closures ⇒ never cloneable ⇒ it
     * must not ride the worker payload.
     *
     * TWO ARMS, and the second one has a REQUIREMENT THIS FIELD ALONE CANNOT EXPRESS.
     * `PreparedSaddle` is a `kind`-tagged union (Phase 2H, spec §5.1):
     *  - `kind: 'dense'` — the Phase 2a path. Producer: `makePreparedSaddle` (`src/gpu/saddle.ts`).
     *  - `kind: 'hierarchical'` — the Phase 2H matrix-free path. Producer:
     *    `makeHierarchicalSaddle` (`src/core/sobolev/hierarchical/hierarchicalSaddle.ts`).
     *    **Passing it REQUIRES `projectionMode: 'frozen'` on the same args, or
     *    `sobolevStepSet` THROWS** ("a hierarchical saddle requires projectionMode 'frozen'").
     *    Not a defensive check: the other projection branch calls `assembleAFlat` per projection
     *    iterate (O(N²) f64, 7.2 GB at N = 30 000), so without the refusal a step would solve its
     *    gradient hierarchically and then silently reassemble a DENSE `Ā` for every projection
     *    iterate — a measurement that reads like a hierarchical step and is not. `dispatch.ts`
     *    forwards `projectionMode` already, so the route works; the pairing is the caller's.
     *  - The hierarchical arm is MAIN-THREAD / INLINE ONLY. `solverWorker.ts` throws whenever
     *    `'saddle' in msg.args`, and the object holds closures, so it is not structured-cloneable
     *    by construction. Under the default `solverDriver: 'worker'` (`store.ts`) passing it
     *    through here therefore throws, by design and pinned by a test.
     *
     * Why this lives on the consumer side too, when the producer already documents it: this field
     * is the only caller-facing surface, and it silently WIDENED when `PreparedSaddle` became a
     * union — a public field that accepts a value which throws unless a SECOND field is set is
     * exactly the non-obvious runtime behaviour `CLAUDE.md` requires anchored in place.
     * @see SobolevStepOptions.saddle (optimizer.ts)
     * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.1, §11.2 rows 8, 8b, A33 item 2
     * @see src/core/sobolev/hierarchical/hierarchicalSaddle.ts (the producer, and its REACHABILITY paragraph)
     * @issue utof/repulsive-test2#62
     */
    saddle?: PreparedSaddle;
    /**
     * Which Sobolev machinery to use; resolved against the graph by `resolveSobolevRoute`.
     * Plain data (unlike `saddle`), so it crosses the worker boundary: the worker builds the
     * hierarchical solver itself, inside `dispatchDescentStep`. Absent ⇒ 'dense'.
     * @see src/core/sobolevRoute.ts
     */
    sobolevSolver?: SobolevSolver;
}

/**
 * The tangent-point energy of a graph, by whichever evaluation its size allows: the exact
 * all-pairs sum while the pair list exists ({@link pairListFits}), Barnes–Hut above that.
 * Why one helper: the store shows this number on load and on commit, and the step returns it;
 * all three must switch at the same size or the display jumps by the Barnes–Hut offset (~0.4 %).
 * @see src/core/barnesHut.ts
 */
export function graphEnergy(vertices: Vec3[], edges: Edge[], disjointPairs: number[][]): number {
    const { alpha, beta, epsilon } = DEFAULTS;
    if (pairListFits(vertices.length))
        return calculateEnergy(vertices, edges, disjointPairs, alpha, beta, epsilon);
    return barnesHutEnergy(prepareBarnesHut(vertices, edges, epsilon), alpha, beta);
}

/**
 * Barnes–Hut energy and differential for one step at opening parameter `theta`: the tree at the
 * step's INPUT vertices is built once and shared by E₀ and dE; every other geometry (a
 * line-search trial) gets its own tree.
 * Why keyed on array identity: the step passes its input array through unchanged, and every
 * trial is a fresh array.
 */
function barnesHutFor(input: Vec3[], edges: Edge[], theta?: number) {
    const { alpha, beta, epsilon } = DEFAULTS;
    const opts = theta === undefined ? {} : { theta };
    let inputCtx: BarnesHutContext | null = null;
    const ctxOf = (V: Vec3[]): BarnesHutContext => {
        if (V !== input) return prepareBarnesHut(V, edges, epsilon, opts);
        inputCtx ??= prepareBarnesHut(V, edges, epsilon, opts);
        return inputCtx;
    };
    return {
        energy: (V: Vec3[]): number => barnesHutEnergy(ctxOf(V), alpha, beta),
        gradient: (V: Vec3[]): Vec3[] => barnesHutGradient(ctxOf(V), alpha, beta),
    };
}

/**
 * "Nothing can step this graph": echo the input and report a rejected step, so the frame loop
 * auto-pauses instead of crashing. Why not throw: a thrown step sends the worker down the
 * worker→main fallback chain, and the main path then throws again inside the frame loop.
 */
function refusedOutcome(args: DispatchDescentStepArgs, why: string): DescentStepOutcome {
    console.warn(`dispatchDescentStep: ${why}`);
    return {
        vertices: args.vertices,
        energy: args.energyBefore ?? graphEnergy(args.vertices, args.edges, args.disjointPairs),
        accepted: false,
        converged: false,
        stats: null,
        timings: null,
        descentField: null,
    };
}

/**
 * One Sobolev step on the hierarchical route: matrix-free BCT solve, Barnes–Hut energy and
 * differential, frozen projection. Never reads `disjointPairs`. Returns `null` when the
 * hierarchical solver refuses this constraint set or fails, so the caller can fall back.
 *
 * TWO ATTEMPTS AT MOST (`HIERARCHICAL_ACCURACY`, which records the measurements): the coarse
 * setting first, and the refined one only if `needsRefinement` rejects the coarse result. The
 * coarse attempt is then discarded, not built upon.
 *
 * THE RETURNED `energy` IS ALWAYS THE COARSE-θ ENERGY of the returned vertices, whichever attempt
 * produced them. The frame loop feeds that number back as the next step's `energyBefore`, and the
 * next step starts coarse; a refined-θ energy there would be off by the difference between the
 * two approximations (measured 6e-2 against per-step decreases of 1e-5) and would corrupt the
 * Armijo test. For the same reason the refined attempt ignores `energyBefore` and evaluates its
 * own E₀.
 *
 * Solver configuration: the backward-error certificate is off (`etaAccept: 1`), because the line
 * search is what guards a descent step, and `‖A_σ‖₂` is the O(1) trace bound, which only that
 * certificate would have used. `projectionMode` is forced to `'frozen'`: the other mode
 * reassembles a dense Ā per projection iterate, and `sobolevStepSet` throws on that combination.
 * @see src/core/sobolevRoute.ts (HIERARCHICAL_ACCURACY, needsRefinement)
 */
function hierarchicalSobolevStep(
    args: DispatchDescentStepArgs,
    set: ConstraintSet,
): DescentStepOutcome | null {
    try {
        const [coarse, refined] = HIERARCHICAL_ACCURACY;
        const bhCoarse = barnesHutFor(args.vertices, args.edges, coarse.theta);
        let msBuild = 0;
        let msGradient = 0;
        let msStep = 0;
        const attempt = (acc: HierarchicalAccuracy, isCoarse: boolean) => {
            const bh = isCoarse ? bhCoarse : barnesHutFor(args.vertices, args.edges, acc.theta);
            const t0 = performance.now();
            const saddle = makeHierarchicalSaddle(args.vertices, args.edges, null, {
                schur: { tol: acc.tol, etaAccept: 1 },
                normEstimate: 'trace',
            });
            const t1 = performance.now();
            const dE = bh.gradient(args.vertices);
            const t2 = performance.now();
            const r = sobolevStepSet(args.vertices, args.edges, [], set, {
                mode: 'analytical',
                collectTimings: args.collectTimings,
                collectField: args.collectField,
                energyBefore: isCoarse ? args.energyBefore : undefined,
                projectionMode: 'frozen',
                penalties: args.penalties,
                dE,
                saddle,
                tpeEnergy: bh.energy,
            });
            msBuild += t1 - t0;
            msGradient += t2 - t1;
            msStep += (r.timings?.step?.ms ?? 0) + (t2 - t0);
            return r;
        };

        let r = attempt(coarse, true);
        let usedRefined = false;
        if (
            needsRefinement(
                {
                    accepted: r.accepted,
                    converged: r.converged,
                    gradientL2Norm: r.stats.gradientL2Norm,
                    tau: r.stats.tau,
                },
                SOBOLEV_CONVERGENCE_TOL,
            )
        ) {
            r = attempt(refined, false);
            usedRefined = true;
        }

        // A refined attempt that found no step, this close to the tolerance, is the route's
        // accuracy limit and is reported as such (see SobolevStepFailureReason).
        const stats =
            usedRefined &&
            !r.accepted &&
            !r.converged &&
            r.stats.gradientL2Norm < 10 * SOBOLEV_CONVERGENCE_TOL
                ? { ...r.stats, reason: 'accuracy_floor' as const }
                : r.stats;

        // Coarse-θ energy of whatever is returned — see the TSDoc. A coarse result already
        // carries it; a refined one is re-measured (accepted: the new vertices; otherwise the
        // input, whose coarse energy is `energyBefore` when the caller supplied it).
        const energy = !usedRefined
            ? r.energy
            : r.accepted
              ? bhCoarse.energy(r.vertices)
              : (args.energyBefore ?? bhCoarse.energy(args.vertices));

        // The step's own collector only times what runs inside it; the tree-based differential
        // and the operator build ran out here, so they are added under the keys the dense route
        // uses for the same two jobs (`dE`, `assembleA`). With a refined attempt the three
        // totals cover BOTH attempts; the per-phase entries are the last attempt's.
        const timings = r.timings
            ? {
                  ...r.timings,
                  dE: { ms: msGradient, calls: usedRefined ? 2 : 1 },
                  assembleA: { ms: msBuild, calls: usedRefined ? 2 : 1 },
                  step: { ms: msStep, calls: 1 },
              }
            : null;
        return {
            vertices: r.vertices,
            energy,
            accepted: r.accepted,
            converged: r.converged,
            stats,
            timings,
            descentField: r.descentField ?? null,
        };
    } catch (err) {
        console.warn(
            'dispatchDescentStep: hierarchical route failed for this step:',
            err instanceof Error ? err.message : err,
        );
        return null;
    }
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
                // §D14: the app never reaches this legacy barycenter-only path (it
                // always passes the M-set toggles), and §D14 deliberately surfaces
                // the FULL-ConstraintSet g̃, not the legacy field — so descentField
                // is null here regardless of collectField. @see plan §D14 / issue #9.
                descentField: null,
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
        // Route (sobolevRoute.ts). `sobolevSolver` absent means 'dense', so every caller that
        // predates the option — and every graph small enough to have a pair list — runs exactly
        // the code below, unchanged.
        const route = resolveSobolevRoute(
            args.sobolevSolver ?? 'dense',
            args.vertices.length,
            args.edges,
            args.mode === 'analytical',
        );
        if (route === 'hierarchical') {
            const h = hierarchicalSobolevStep(args, set);
            if (h) return h;
        }
        if (route === null || !pairListFits(args.vertices.length)) {
            return refusedOutcome(
                args,
                route === null
                    ? `no Sobolev route for ${args.vertices.length} vertices that are not one closed loop: the dense route needs the pair list, which stops at PAIR_LIST_MAX_N`
                    : 'the hierarchical route failed and the graph is too large for the dense one',
            );
        }
        const r = sobolevStepSet(args.vertices, args.edges, args.disjointPairs, set, {
            mode: args.mode,
            collectTimings: args.collectTimings,
            // §D14: opt into the step's g̃ (full-set) for the arrows diagnostic;
            // undefined → no field collected. @see plan §D14 / issue #9.
            collectField: args.collectField,
            // E₀ reuse (Task 4): passthrough to the step; undefined → recompute.
            // @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 4)
            energyBefore: args.energyBefore,
            // Projection strategy (Task 6): passthrough; undefined → 'reassemble'.
            projectionMode: args.projectionMode,
            // Penalties (5C): passthrough; undefined/all-zero → core no-ops via
            // penaltiesActive, bit-identical (plan §2.4).
            // @see docs/superpowers/plans/2026-07-03-sobolev-penalties.md §2.4
            penalties: args.penalties,
            // Phase 1 GPU seam passthrough; undefined → core computes dE itself.
            // @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md — "D5 — The seam"
            dE: args.dE,
            // Phase 2a GPU saddle seam passthrough; undefined → core assembles A and
            // factors it itself.
            // @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4
            saddle: args.saddle,
        });
        return {
            vertices: r.vertices,
            energy: r.energy,
            accepted: r.accepted,
            converged: r.converged,
            stats: r.stats,
            timings: r.timings ?? null,
            // §D14: the full-set g̃ (null when not collected or singular saddle).
            // @see plan §D14 / issue #9.
            descentField: r.descentField ?? null,
        };
    }
    if (!pairListFits(args.vertices.length)) {
        // Raw descent above the pair-list limit: the same update, v − stepSize·dE, with the
        // Barnes–Hut differential and energy. 'finiteDiff' has no large-n form (it would need
        // one energy evaluation per coordinate), so both modes use the analytical differential.
        const bh = barnesHutFor(args.vertices, args.edges);
        const grad = bh.gradient(args.vertices);
        const next: Vec3[] = args.vertices.map((v, i) => [
            v[0] - args.stepSize * grad[i][0],
            v[1] - args.stepSize * grad[i][1],
            v[2] - args.stepSize * grad[i][2],
        ]);
        return {
            vertices: next,
            energy: bh.energy(next),
            accepted: true,
            converged: false,
            stats: null,
            timings: null,
            descentField: args.collectField ? grad : null,
        };
    }
    const r = step(args.vertices, args.edges, args.disjointPairs, {
        mode: args.mode,
        stepSize: args.stepSize,
        // §D14: opt into the raw dE for the arrows diagnostic. @see plan §D14.
        collectField: args.collectField,
    });
    return {
        vertices: r.vertices,
        energy: r.energy,
        accepted: true,
        converged: false,
        stats: null,
        timings: null,
        // §D14: the raw L² gradient dE (null when not collected). @see plan §D14.
        descentField: r.field ?? null,
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
    /** Optional so sources that predate the option still type-check; absent ⇒ 'dense'. */
    sobolevSolver?: SobolevSolver;
}

/**
 * Assemble the {@link dispatchDescentStep} argument object from the store state
 * and a reused E₀. This is the SINGLE param-assembly both drivers call (§D7):
 * the main driver spreads `collectTimings: true` and dispatches inline; the
 * worker driver strips the topology fields (edges/disjointPairs — the worker
 * restores them from its cache, §D4) and posts the rest. Extracting it here is
 * what prevents worker/main parameter drift. `collectTimings` and `collectField`
 * are intentionally NOT set here (hardcoded/derived at each call site, as in the
 * pre-worker frame loop — §D14 spreads `collectField: st.showArrows` there).
 * @see docs/superpowers/plans/2026-07-04-worker-solver.md §D7, §D14
 */
export function buildStepArgs(
    state: StepArgsSource,
    energyBefore: number | undefined,
): Omit<DispatchDescentStepArgs, 'collectTimings' | 'collectField'> {
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
        // Spread only when set: the args of a source without the field stay identical to before.
        ...(state.sobolevSolver !== undefined ? { sobolevSolver: state.sobolevSolver } : {}),
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
 * Per-`field` worker payload (§D13-b): the dynamic inputs `computeArrowField`
 * needs beyond the cached topology (edges + disjointPairs come from the worker's
 * topology cache, §D4, exactly like `step`). `x0` is the frozen sobolev
 * barycenter target (ignored in raw mode); `descentMode` selects raw dE vs the
 * sobolev g̃ solve, `mode` selects analytical vs finiteDiff dE.
 * @see docs/superpowers/plans/2026-07-04-worker-solver.md §D13
 */
export interface FieldArgs {
    vertices: Vec3[];
    mode: Mode;
    descentMode: DescentMode;
    x0: Vec3;
}

/**
 * Main→worker protocol (§D4/§D12/§D13). `topology` is sent once on init and on
 * every graphVersion change (the worker recomputes disjointPairs via the same
 * deterministic calculateDisjointPairs); `step` carries only the dynamic
 * per-step args tagged with the graphVersion the main thread will match on §D5;
 * `field` (§D13-b) requests one GradientArrows descent-field compute, tagged
 * with graphVersion the same way so a stale field result can be dropped.
 * @see docs/superpowers/plans/2026-07-04-worker-solver.md §D4, §D12, §D13
 */
export type SolverWorkerRequest =
    | { type: 'topology'; graphVersion: number; edges: Edge[] }
    | { type: 'step'; graphVersion: number; args: DispatchStepArgs }
    | { type: 'field'; graphVersion: number; args: FieldArgs };

/**
 * Worker→main protocol (§D5/§D12/§D13). `result` echoes the request's
 * graphVersion UNTOUCHED so the main thread can DROP a result whose graphVersion
 * no longer matches the store (a preset rebuild / pause landed mid-flight);
 * `fieldResult` echoes it the same way and carries the raw descent field (or
 * `null` = singular saddle → arrows hide, §D13-b); `error` is posted on any
 * worker-side throw and triggers the §D6 auto-fallback (step: to 'main' driver;
 * field: to the synchronous inline compute, §D13-c).
 * @see docs/superpowers/plans/2026-07-04-worker-solver.md §D5, §D6, §D12, §D13
 */
export type SolverWorkerResponse =
    | { type: 'result'; graphVersion: number; result: DescentStepOutcome }
    | { type: 'fieldResult'; graphVersion: number; field: Vec3[] | null }
    | { type: 'error'; message: string };
