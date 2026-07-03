import { create } from 'zustand';
import {
    DEFAULTS,
    type ProjectionMode,
    type SobolevStepStats,
    sobolevStep,
    sobolevStepSet,
    step,
} from './core/optimizer';
import {
    assertValidConstraintSet,
    barycenterBlock,
    type ConstraintSet,
    edgeLengths,
    edgeLengthsBlock,
    pointBlock,
    totalLength,
    totalLengthBlock,
} from './core/sobolev/constraintSet';
import { barycenterTarget } from './core/sobolev/constraints';
import type { SobolevStepTimings } from './core/sobolev/phaseTimings';
import { calculateDisjointPairs, calculateEnergy } from './core/tangentPointEnergy';
import {
    type Edge,
    type GraphState,
    type TestConfig,
    testConfigs,
    type Vec3,
} from './core/testConfigs';

// Re-exported so UI components can import all sim-facing types from the store
// (same pattern as Mode/DescentMode/LengthMode).
export type { ProjectionMode } from './core/optimizer';

const STORAGE_KEY = 'repulsive-test-config';

export interface SavedConfig {
    testId: string;
    params: Record<string, number>;
}

// Persistence. Guarded so importing this module in a non-browser test env is safe.
export function loadSavedConfig(): SavedConfig {
    try {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) return JSON.parse(saved);
    } catch {}
    return { testId: 'crossing', params: {} };
}

export function saveConfig(testId: string, params: Record<string, number>): void {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify({ testId, params }));
    } catch {}
}

export interface GraphBuild {
    graph: GraphState;
    disjointPairs: number[][];
    energy: number;
}

// Pure: build a graph from a preset + params, its disjoint pairs, and its energy.
export function buildGraphState(test: TestConfig, params: Record<string, number>): GraphBuild {
    const graph = test.generate(params);
    const disjointPairs = calculateDisjointPairs(graph.edges);
    const energy = calculateEnergy(
        graph.vertices,
        graph.edges,
        disjointPairs,
        DEFAULTS.alpha,
        DEFAULTS.beta,
        DEFAULTS.epsilon,
    );
    return { graph, disjointPairs, energy };
}

function defaultParams(test: TestConfig): Record<string, number> {
    const p: Record<string, number> = {};
    if (test.params) for (const param of test.params) p[param.name] = param.default;
    return p;
}

function cloneVerts(v: Vec3[]): Vec3[] {
    return v.map((p) => [p[0], p[1], p[2]] as Vec3);
}

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
export function dispatchDescentStep(args: {
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
}): DescentStepOutcome {
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

export interface SimStore {
    // config (React-subscribed, infrequent)
    selectedTestId: string;
    testParams: Record<string, number>;
    mode: Mode;
    stepSize: number;
    descentMode: DescentMode;
    running: boolean;
    // FROZEN constraint targets for the sobolev descent: x₀ (barycenter),
    // L⁰ (total length) and ℓ⁰ (per-edge lengths, spec §5.3). Lifecycle (anchor —
    // do not "fix" by recomputing per frame): recomputed ONCE each time a descent
    // run (re)starts — on play after pause (setRunning(true)), on preset/config
    // change (rebuild), on vertex commit (setRunning(false)) — and NEVER during a
    // run; a target that tracks the current iterate makes its constraint vacuous.
    // Accepted consequence (spec §3.5): L⁰/ℓ⁰ re-anchor to the CURRENT geometry at
    // every pause/play boundary, so length is preserved within a run, but
    // sub-tolerance drift can accumulate across pause cycles.
    // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("set x₀ once at initialization")
    // @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §3.5 (frozen-targets lifecycle)
    // @see src/core/sobolev/constraints.ts (barycenterTarget TSDoc)
    sobolevX0: Vec3;
    sobolevL0: number;
    sobolevEll0: number[];
    // 3-way length mode (spec §5.3) — the SOURCE OF TRUTH for the length
    // constraint. lengthConstraint below is a WRITE-THROUGH MIRROR
    // (true ⟺ lengthMode !== 'none'; its setter delegates to setLengthMode),
    // kept ONLY so the M1 store contract and test/store-constraints.test.ts
    // stay valid unmodified (§4.5 acceptance gate, applied to M2 via §5.5).
    // New code reads/writes lengthMode.
    // @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §5.3, §4.5
    lengthMode: LengthMode;
    // Per-block constraint toggles for the sobolev ConstraintSet, both default
    // true (barycenter preserves pre-M1 behavior; length gives the flow an
    // equilibrium so the ‖g̃‖ termination can fire). Setters clear
    // sobolevConverged ONLY: a converged verdict is per-constraint-set, while
    // sobolevStats describe the last step actually taken and remain valid.
    // @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §4.2, §9a
    barycenterConstraint: boolean;
    lengthConstraint: boolean;
    // Projection solve strategy for the sobolev step (solver-perf Task 6).
    // Default 'frozen' — the reference implementation's per-step
    // factorization-reuse scheme (paper line 734): cheapest, paper-shipped
    // behavior; 'reassemble' is the stricter-step-quality A/B alternative
    // (see the measured trade-off in oracle/README.md "Frozen-projection
    // mode", incl. the junction-fixture τ-backtracking caveat). A/B-selectable
    // in the ControlPanel; does NOT touch the frozen targets or the
    // constraint set, so switching it never invalidates sobolevConverged.
    // @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 6)
    projectionMode: ProjectionMode;
    // Interactive point pins (pin-drag milestone, briefing §5B). Each enabled pin
    // becomes a `pointBlock` in the sobolev ConstraintSet (dispatch, spec §5.3).
    // Default `[]`. FROZEN targets with the sobolevEll0 lifecycle (frozen-targets
    // anchor above): cleared on rebuild (indices invalid in a new graph),
    // re-anchored to live on play/commit, and NEVER mutated by the frame loop.
    // @see docs/superpowers/plans/2026-07-03-pin-drag-ui.md (Decisions D2, D5)
    pins: PinConstraint[];
    // Last sobolev step's diagnostics (null before any sobolev step and in raw
    // mode); `sobolevConverged` mirrors spec §C step 5's termination outcome.
    sobolevStats: SobolevStepStats | null;
    sobolevConverged: boolean;
    // Last sobolev step's per-phase timings (null before any timed step and in
    // raw mode); cleared alongside sobolevStats. Surfaced as the Stats.tsx second
    // line. @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 3)
    sobolevTimings: SobolevStepTimings | null;
    // Why: descent-direction arrows are a user toggle, visible in BOTH paused and
    // running states (GradientArrows recomputes from the live buffer); this flag
    // only gates rendering, never the descent itself.
    showArrows: boolean;
    // graph
    graph: GraphState;
    disjointPairs: number[][];
    // live positions — mutated in place in useFrame; NEVER selected by a component.
    live: Vec3[];
    // stats (React-subscribed, throttled by the writer)
    step: number;
    energy: number;
    zoom: number;
    // remount / view-reset signals
    graphVersion: number;
    viewResetNonce: number;
    // actions
    setPreset(id: string): void;
    setParam(name: string, value: number): void;
    regenerate(): void;
    reset(): void;
    setMode(m: Mode): void;
    setStepSize(s: number): void;
    setDescentMode(m: DescentMode): void;
    setBarycenterConstraint(b: boolean): void;
    setLengthMode(m: LengthMode): void;
    setLengthConstraint(b: boolean): void;
    setProjectionMode(m: ProjectionMode): void;
    // Pin actions (briefing §5B). addPin snapshots the vertex's CURRENT live
    // position as the frozen target (idempotent by vertexIndex); setPinTarget is
    // the drag re-target; removePin/setPinEnabled edit the list.
    // @see docs/superpowers/plans/2026-07-03-pin-drag-ui.md (Decisions D2, D3, D5)
    addPin(vertexIndex: number): void;
    removePin(vertexIndex: number): void;
    setPinEnabled(vertexIndex: number, enabled: boolean): void;
    setPinTarget(vertexIndex: number, target: Vec3): void;
    setShowArrows(b: boolean): void;
    setRunning(b: boolean): void;
    setZoom(z: number): void;
}

function initialConfig() {
    const saved = loadSavedConfig();
    const test = testConfigs.find((t) => t.id === saved.testId) ?? testConfigs[0];
    const params: Record<string, number> = {};
    if (test.params)
        for (const p of test.params) params[p.name] = saved.params[p.name] ?? p.default;
    return { test, params };
}

// The single source of truth shared across React's two reconcilers (DOM UI ↔ the R3F
// <Canvas> scene). The non-reactive `live` buffer is mutated in place by the useFrame
// descent loop and is deliberately NOT React-subscribed; config/stats are. Commit-on-pause
// folds `live` back into `graph.vertices` so paused consumers read current positions.
// Why: curried create<T>()(...) — zustand's documented form that keeps TS generic inference
// sound if a middleware wrapper (persist/devtools) is ever added. @see spec §3, §4.1, §5.
export const useSimStore = create<SimStore>()((set, get) => {
    const { test, params } = initialConfig();
    const built = buildGraphState(test, params);

    // Rebuild the graph from (id, params): resets live buffer, stats, running; bumps graphVersion.
    const rebuild = (id: string, nextParams: Record<string, number>) => {
        const t = testConfigs.find((x) => x.id === id) ?? testConfigs[0];
        const b = buildGraphState(t, nextParams);
        set((s) => ({
            selectedTestId: id,
            testParams: nextParams,
            graph: b.graph,
            disjointPairs: b.disjointPairs,
            live: cloneVerts(b.graph.vertices),
            step: 0,
            energy: b.energy,
            running: false,
            graphVersion: s.graphVersion + 1,
            // Preset/config change = a run (re)start boundary → re-anchor the
            // frozen targets x₀, L⁰, ℓ⁰ (see the frozen-targets lifecycle anchor above).
            sobolevX0: barycenterTarget(b.graph.vertices, b.graph.edges),
            sobolevL0: totalLength(b.graph.vertices, b.graph.edges),
            sobolevEll0: edgeLengths(b.graph.vertices, b.graph.edges),
            // Rebuild changes topology → old pin vertex indices are meaningless,
            // so pins CLEAR (the faithful mirror of the sobolevEll0 "recompute
            // from the new graph" re-anchor when the old data cannot survive).
            // @see docs/superpowers/plans/2026-07-03-pin-drag-ui.md (Decision D5)
            pins: [],
            sobolevStats: null,
            sobolevConverged: false,
            sobolevTimings: null,
        }));
        saveConfig(id, nextParams);
    };

    // Re-anchor every in-range pin's FROZEN target to the current live position
    // (drops any out-of-range pin). Mirrors the sobolevEll0 = edgeLengths(live)
    // re-anchor at the play/commit boundaries; because drag keeps target == live
    // this is a no-op except for absorbing ≤ reference-tolerance projection drift
    // (spec §3.5 "accepted drift across pause cycles"). Fresh Vec3 copies so the
    // frozen targets never alias the mutable live buffer.
    // @see docs/superpowers/plans/2026-07-03-pin-drag-ui.md (Decision D5)
    const reanchorPins = (pins: PinConstraint[], live: Vec3[]): PinConstraint[] =>
        pins
            .filter((p) => p.vertexIndex >= 0 && p.vertexIndex < live.length)
            .map((p) => ({
                ...p,
                target: [
                    live[p.vertexIndex][0],
                    live[p.vertexIndex][1],
                    live[p.vertexIndex][2],
                ] as Vec3,
            }));

    return {
        selectedTestId: test.id,
        testParams: params,
        mode: 'analytical',
        stepSize: 0.001,
        descentMode: 'raw',
        running: false,
        graph: built.graph,
        disjointPairs: built.disjointPairs,
        live: cloneVerts(built.graph.vertices),
        step: 0,
        energy: built.energy,
        sobolevX0: barycenterTarget(built.graph.vertices, built.graph.edges),
        sobolevL0: totalLength(built.graph.vertices, built.graph.edges),
        sobolevEll0: edgeLengths(built.graph.vertices, built.graph.edges),
        lengthMode: 'total',
        barycenterConstraint: true,
        lengthConstraint: true,
        projectionMode: 'frozen',
        pins: [],
        sobolevStats: null,
        sobolevConverged: false,
        sobolevTimings: null,
        showArrows: true,
        zoom: 1,
        graphVersion: 0,
        viewResetNonce: 0,

        setPreset: (id) => {
            const t = testConfigs.find((x) => x.id === id) ?? testConfigs[0];
            rebuild(id, defaultParams(t));
        },
        setParam: (name, value) => {
            const p = { ...get().testParams, [name]: value };
            set({ testParams: p });
            saveConfig(get().selectedTestId, p);
        },
        regenerate: () => rebuild(get().selectedTestId, get().testParams),
        reset: () => {
            rebuild(get().selectedTestId, get().testParams);
            set((s) => ({ viewResetNonce: s.viewResetNonce + 1 }));
        },
        setMode: (m) => set({ mode: m }),
        setStepSize: (s) => set({ stepSize: s }),
        // Mode switch clears the other mode's stale diagnostics; x₀ needs no
        // recompute here — it re-anchors at the next run start (see lifecycle anchor).
        setDescentMode: (m) =>
            set({
                descentMode: m,
                sobolevStats: null,
                sobolevConverged: false,
                sobolevTimings: null,
            }),
        // Toggling a constraint invalidates ONLY the converged verdict (it is
        // per-constraint-set); targets re-anchor at the next run start, and
        // sobolevStats stay — they describe the last step actually taken.
        // @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §4.2, §5.3, §9a
        setBarycenterConstraint: (b) => set({ barycenterConstraint: b, sobolevConverged: false }),
        setLengthMode: (m) =>
            set({ lengthMode: m, lengthConstraint: m !== 'none', sobolevConverged: false }),
        // Legacy M1 setter: writes THROUGH the 3-way mode so the boolean mirror
        // and lengthMode can never diverge (see the lengthMode field anchor).
        setLengthConstraint: (b) => get().setLengthMode(b ? 'total' : 'none'),
        // Solver strategy, not a constraint toggle: leaves targets, stats, and
        // the converged verdict alone (see the projectionMode field anchor).
        setProjectionMode: (m) => set({ projectionMode: m }),
        // Pin actions (briefing §5B). A pin is a per-vertex point constraint; like
        // the other constraint toggles, mutating the pin set invalidates ONLY the
        // converged verdict (sobolevStats describe the last step taken and remain).
        // The frozen target snapshots the CURRENT live position at add time and is
        // otherwise mutated only here or by the play/commit/rebuild re-anchor —
        // never by the frame loop (frozen-targets lifecycle, spec §3.5).
        // @see docs/superpowers/plans/2026-07-03-pin-drag-ui.md (Decisions D2, D3, D5)
        addPin: (vertexIndex) =>
            set((s) => {
                // Idempotent by vertexIndex: a second grab of an already-pinned
                // vertex keeps its existing target (drag re-targets via setPinTarget).
                if (s.pins.some((p) => p.vertexIndex === vertexIndex)) return {};
                const v = s.live[vertexIndex];
                if (v === undefined) return {}; // out-of-range guard — never pin a stale index
                return {
                    pins: [
                        ...s.pins,
                        { vertexIndex, target: [v[0], v[1], v[2]] as Vec3, enabled: true },
                    ],
                    sobolevConverged: false,
                };
            }),
        removePin: (vertexIndex) =>
            set((s) => ({
                pins: s.pins.filter((p) => p.vertexIndex !== vertexIndex),
                sobolevConverged: false,
            })),
        setPinEnabled: (vertexIndex, enabled) =>
            set((s) => ({
                pins: s.pins.map((p) => (p.vertexIndex === vertexIndex ? { ...p, enabled } : p)),
                sobolevConverged: false,
            })),
        // Drag re-target: update only the named pin's frozen target (a fresh Vec3
        // so it never aliases the caller's array). No-op if the pin was removed.
        setPinTarget: (vertexIndex, target) =>
            set((s) => ({
                pins: s.pins.map((p) =>
                    p.vertexIndex === vertexIndex
                        ? { ...p, target: [target[0], target[1], target[2]] as Vec3 }
                        : p,
                ),
                sobolevConverged: false,
            })),
        setShowArrows: (b) => set({ showArrows: b }),
        setRunning: (b) => {
            if (b) {
                // Play = a run (re)starts → re-anchor the frozen targets x₀, L⁰, ℓ⁰
                // from the CURRENT live positions and clear last run's diagnostics (see
                // the frozen-targets lifecycle anchor). The frame loop must never touch them.
                const s = get();
                set({
                    running: true,
                    sobolevX0: barycenterTarget(s.live, s.graph.edges),
                    sobolevL0: totalLength(s.live, s.graph.edges),
                    sobolevEll0: edgeLengths(s.live, s.graph.edges),
                    // Pins are frozen targets on the same lifecycle: re-anchor to
                    // the current live positions at run start (Decision D5).
                    pins: reanchorPins(s.pins, s.live),
                    sobolevStats: null,
                    sobolevConverged: false,
                    sobolevTimings: null,
                });
                return;
            }
            // Commit-on-pause: fold the live buffer back into graph.vertices so paused
            // consumers (GradientArrows) read current positions, and recompute energy.
            // @see spec §4.1 (store.commit) / §6 (stale-arrow fix)
            const s = get();
            set({
                running: false,
                graph: { ...s.graph, vertices: cloneVerts(s.live) },
                energy: calculateEnergy(
                    s.live,
                    s.graph.edges,
                    s.disjointPairs,
                    DEFAULTS.alpha,
                    DEFAULTS.beta,
                    DEFAULTS.epsilon,
                ),
                // Vertex commit = the next run will start from these positions →
                // re-anchor x₀, L⁰, ℓ⁰ now (frozen-targets lifecycle anchor). Diagnostics
                // are deliberately KEPT: on auto-pause they tell the user why it stopped.
                sobolevX0: barycenterTarget(s.live, s.graph.edges),
                sobolevL0: totalLength(s.live, s.graph.edges),
                sobolevEll0: edgeLengths(s.live, s.graph.edges),
                // Vertex commit re-anchors the frozen pin targets from the
                // committed positions, same lifecycle as x₀/L⁰/ℓ⁰ (Decision D5).
                pins: reanchorPins(s.pins, s.live),
            });
        },
        setZoom: (z) => set({ zoom: z }),
    };
});
