import { create } from 'zustand';
import type { DescentMode, LengthMode, Mode, PinConstraint } from './core/dispatch';
import { DEFAULTS, type ProjectionMode, type SobolevStepStats } from './core/optimizer';
import { edgeLengths, totalLength } from './core/sobolev/constraintSet';
import { barycenterTarget } from './core/sobolev/constraints';
import type { PenaltyConfig } from './core/sobolev/penalties';
import type { SobolevStepTimings } from './core/sobolev/phaseTimings';
import { calculateDisjointPairs, calculateEnergy } from './core/tangentPointEnergy';
import { type GraphState, type TestConfig, testConfigs, type Vec3 } from './core/testConfigs';

export type {
    DescentMode,
    DescentStepOutcome,
    DispatchDescentStepArgs,
    DispatchStepArgs,
    LengthMode,
    Mode,
    PinConstraint,
    SolverWorkerRequest,
    SolverWorkerResponse,
    StepArgsSource,
} from './core/dispatch';
// §D2 worker-prep: dispatchDescentStep + step-arg assembly + its arg/result
// types + the store-local types (Mode/DescentMode/LengthMode/PinConstraint/
// DescentStepOutcome) moved to core/dispatch so the same pure function can run
// in a Web Worker; re-exported here so the existing test imports (5 test files)
// and Viewer.tsx keep importing them from the store UNCHANGED.
// @see docs/superpowers/plans/2026-07-04-worker-solver.md §D2
export { buildStepArgs, dispatchDescentStep } from './core/dispatch';
// Re-exported so UI components can import all sim-facing types from the store
// (same pattern as Mode/DescentMode/LengthMode).
export type { ProjectionMode } from './core/optimizer';
// Penalty catalog config (5C) re-exported for the ControlPanel — same pattern
// as ProjectionMode. @see docs/superpowers/plans/2026-07-03-sobolev-penalties.md
export type { PenaltyConfig } from './core/sobolev/penalties';

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

/**
 * Which driver executes {@link dispatchDescentStep} each frame. 'worker'
 * (default) posts the step to an off-main-thread Web Worker so orbit / pin-drag
 * / UI stay at display refresh even when a step is expensive; 'main' is today's
 * synchronous in-frame path — the fallback, the A/B baseline, and the only path
 * the store tests exercise. The frame loop AUTO-falls back to 'main' (via
 * {@link SimStore.setSolverDriver}) if the Worker fails to construct or posts an
 * error (§D6). This is a main-thread concern, so it lives in the store, NOT in
 * worker-bundle-pure src/core/**.
 * @see docs/superpowers/plans/2026-07-04-worker-solver.md §D6
 */
export type SolverDriver = 'worker' | 'main';

export interface SimStore {
    // config (React-subscribed, infrequent)
    selectedTestId: string;
    testParams: Record<string, number>;
    mode: Mode;
    stepSize: number;
    descentMode: DescentMode;
    // Off-main-thread solver driver (default 'worker'); see the SolverDriver type
    // anchor. @see docs/superpowers/plans/2026-07-04-worker-solver.md §D6
    solverDriver: SolverDriver;
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
    // Soft-constraint penalty catalog (5C) for the sobolev OBJECTIVE: totalLength
    // + lengthDiff weights and a field {weight, X}. Default ALL OFF (every weight
    // 0) ⇒ `penaltiesActive` is false and the threaded step is bit-identical to
    // the penalty-free build (plan §2.4). `field.X` keeps a default unit axis so
    // the X input always binds a value even while the field weight is 0
    // (inactive). Changing ANY weight or X bumps `penaltyEpoch` below.
    // @see docs/superpowers/plans/2026-07-03-sobolev-penalties.md §2.4, §4 Task 5
    penalties: PenaltyConfig;
    // Target-length animation rate (paper SelfAvoiding.tex line 760): a per-
    // ACCEPTED-step multiplicative factor for the FROZEN length targets. 1.0 =
    // OFF. The frame loop calls `advanceLengthSchedule` after each accepted step
    // when this ≠ 1; the targets evolve FROM THE SCHEDULE (stored target × rate),
    // never re-read from the live geometry — the deliberate, documented exception
    // to the frozen-targets anchor above. Clamped to [0.9, 1.1] by its setter.
    // @see docs/superpowers/plans/2026-07-03-sobolev-penalties.md §4 Task 5
    lengthGrowthRate: number;
    // E₀-reuse invalidation nonce (plan §2.4). The frame loop chains the previous
    // accepted step's returned `energy` as the next step's energyBefore, valid
    // ONLY under the SAME penalty config. Every penalty-config setter bumps this;
    // the frame loop (Viewer) drops its cached E₀ whenever it changes, forcing a
    // fresh E₀ recompute — without it a mid-run slider move corrupts the Armijo
    // gate (silent wrong-step bug). Constraint-target animation does NOT bump it
    // (targets don't enter the objective — plan §2.4). Same trigger-nonce pattern
    // as viewResetNonce. @see docs/superpowers/plans/2026-07-03-sobolev-penalties.md §2.4
    penaltyEpoch: number;
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
    // Select the solver driver; also the §D6 auto-fallback entry point (the frame
    // loop calls this with 'main' on Worker failure). @see the SolverDriver type.
    setSolverDriver(d: SolverDriver): void;
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
    // Penalty-config setters (5C). Each replaces one knob and bumps penaltyEpoch
    // (E₀ invalidation) + clears sobolevConverged (per-config verdict), keeping
    // sobolevStats (the last step actually taken). @see plan §2.4, §4 Task 5
    setPenaltyTotalLength(weight: number): void;
    setPenaltyLengthDiff(weight: number): void;
    setPenaltyFieldWeight(weight: number): void;
    setPenaltyFieldX(X: Vec3): void;
    // Target-length animation (paper tex line 760). setLengthGrowthRate clamps to
    // [0.9, 1.1]; advanceLengthSchedule scales the frozen length targets by the
    // rate FROM THE SCHEDULE and is called by the frame loop on ACCEPTED steps.
    // @see plan §4 Task 5
    setLengthGrowthRate(rate: number): void;
    advanceLengthSchedule(): void;
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
        // Default off-main-thread (§D6): smooth interaction is the milestone goal.
        solverDriver: 'worker',
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
        // Penalties default ALL OFF (every weight 0 ⇒ penaltiesActive false ⇒
        // bit-identical threading, plan §2.4). field.X is a placeholder unit axis
        // for the X input while the field weight is 0 (inactive).
        penalties: { totalLength: 0, lengthDiff: 0, field: { weight: 0, X: [1, 0, 0] } },
        // Target-length animation OFF by default (rate 1.0 = no schedule advance).
        lengthGrowthRate: 1,
        penaltyEpoch: 0,
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
        // Driver select / §D6 auto-fallback. Plain field write: switching drivers
        // needs no diagnostic/target invalidation (the SAME pure step runs either
        // way — bit-identical, §2); the frame loop reacts by (re)creating or
        // tearing down the worker. @see …worker-solver.md §D6
        setSolverDriver: (d) => set({ solverDriver: d }),
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
        // Penalty-config setters (5C). Each REPLACES one knob of the config, then
        // invalidates the E₀ cache (bump penaltyEpoch — the frame loop chains the
        // previous step's energy as energyBefore, valid ONLY under the SAME
        // penalty config, plan §2.4) and the per-config converged verdict
        // (sobolevStats stay, like the constraint toggles). field.X falls back to
        // the default axis defensively — the store keeps field defined but
        // PenaltyConfig types it optional.
        // @see docs/superpowers/plans/2026-07-03-sobolev-penalties.md §2.4, §4 Task 5
        setPenaltyTotalLength: (weight) =>
            set((s) => ({
                penalties: { ...s.penalties, totalLength: weight },
                penaltyEpoch: s.penaltyEpoch + 1,
                sobolevConverged: false,
            })),
        setPenaltyLengthDiff: (weight) =>
            set((s) => ({
                penalties: { ...s.penalties, lengthDiff: weight },
                penaltyEpoch: s.penaltyEpoch + 1,
                sobolevConverged: false,
            })),
        setPenaltyFieldWeight: (weight) =>
            set((s) => ({
                penalties: {
                    ...s.penalties,
                    field: { weight, X: s.penalties.field?.X ?? [1, 0, 0] },
                },
                penaltyEpoch: s.penaltyEpoch + 1,
                sobolevConverged: false,
            })),
        setPenaltyFieldX: (X) =>
            set((s) => ({
                penalties: {
                    ...s.penalties,
                    field: { weight: s.penalties.field?.weight ?? 0, X: [X[0], X[1], X[2]] },
                },
                penaltyEpoch: s.penaltyEpoch + 1,
                sobolevConverged: false,
            })),
        // Target-length animation rate (paper SelfAvoiding.tex line 760). Clamp to
        // [0.9, 1.1] (my "sensible clamps" choice, plan §4 Task 5): a per-accepted-
        // step factor compounds every frame, so a ±10% band keeps the schedule
        // gentle and, critically, strictly positive — a zero/negative rate would
        // collapse or sign-flip the frozen length targets. NOT a penalty-config
        // change: the rate scales constraint targets, not the objective, so it does
        // NOT bump penaltyEpoch nor clear sobolevConverged (plan §2.4).
        // @see docs/superpowers/plans/2026-07-03-sobolev-penalties.md §4 Task 5
        setLengthGrowthRate: (rate) =>
            set({ lengthGrowthRate: Math.min(1.1, Math.max(0.9, rate)) }),
        // Advance the frozen length schedule by one step (target-length animation,
        // paper tex line 760; plan §4 Task 5). Called by the frame loop AFTER each
        // ACCEPTED sobolev step: scale L⁰ (total mode) / every ℓ⁰_I (per-edge mode)
        // by lengthGrowthRate. The targets evolve FROM THE SCHEDULE — the previous
        // STORED target × rate — NEVER re-read from the live geometry: the
        // deliberate, documented exception to the frozen-targets anchor above,
        // sanctioned by tex line 760 ("we progressively increase or decrease the
        // target length values; the next constraint projection step then enforces
        // the new length"). rate 1.0 / lengthMode 'none' ⇒ no-op. Does NOT touch
        // the E₀ cache: constraint targets don't enter the objective (plan §2.4).
        // @see docs/superpowers/plans/2026-07-03-sobolev-penalties.md §4 Task 5
        // @see local_files/repulsive_orig_paper/SelfAvoiding.tex (line 760)
        advanceLengthSchedule: () =>
            set((s) => {
                const r = s.lengthGrowthRate;
                if (r === 1) return {}; // off — no schedule advance (rate 1.0 no-op)
                if (s.lengthMode === 'total') return { sobolevL0: s.sobolevL0 * r };
                if (s.lengthMode === 'perEdge')
                    return { sobolevEll0: s.sobolevEll0.map((l) => l * r) };
                return {}; // 'none' — no length target to animate
            }),
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
