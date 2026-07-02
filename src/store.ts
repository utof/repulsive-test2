import { create } from 'zustand';
import { DEFAULTS, type SobolevStepStats, sobolevStep, step } from './core/optimizer';
import { barycenterTarget } from './core/sobolev/constraints';
import { calculateDisjointPairs, calculateEnergy } from './core/tangentPointEnergy';
import {
    type Edge,
    type GraphState,
    type TestConfig,
    testConfigs,
    type Vec3,
} from './core/testConfigs';

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
}

/**
 * The descent-mode dispatch, pure and store-independent so it is testable the
 * same way as {@link buildGraphState}: 'sobolev' → `sobolevStep` (frozen-x0
 * constrained Sobolev flow), 'raw' → the pre-existing `step()` with the exact
 * arguments the frame loop always passed — the raw path must remain
 * byte-identical when the toggle is 'raw'.
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §C
 */
export function dispatchDescentStep(args: {
    descentMode: DescentMode;
    vertices: Vec3[];
    edges: Edge[];
    disjointPairs: number[][];
    mode: Mode;
    stepSize: number;
    x0: Vec3;
}): DescentStepOutcome {
    if (args.descentMode === 'sobolev') {
        const r = sobolevStep(args.vertices, args.edges, args.disjointPairs, args.x0, {
            mode: args.mode,
        });
        return {
            vertices: r.vertices,
            energy: r.energy,
            accepted: r.accepted,
            converged: r.converged,
            stats: r.stats,
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
    // FROZEN barycenter constraint target for the sobolev descent. Lifecycle
    // (anchor — do not "fix" by recomputing per frame): recomputed via
    // barycenterTarget ONCE each time a descent run (re)starts — on play after
    // pause (setRunning(true)), on preset/config change (rebuild), on vertex
    // commit (setRunning(false)) — and NEVER during a run; a target that tracks
    // the current iterate makes the constraint vacuous.
    // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("set x₀ once at initialization")
    // @see src/core/sobolev/constraints.ts (barycenterTarget TSDoc)
    sobolevX0: Vec3;
    // Last sobolev step's diagnostics (null before any sobolev step and in raw
    // mode); `sobolevConverged` mirrors spec §C step 5's termination outcome.
    sobolevStats: SobolevStepStats | null;
    sobolevConverged: boolean;
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
            // frozen x₀ (see the sobolevX0 lifecycle anchor above).
            sobolevX0: barycenterTarget(b.graph.vertices, b.graph.edges),
            sobolevStats: null,
            sobolevConverged: false,
        }));
        saveConfig(id, nextParams);
    };

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
        sobolevStats: null,
        sobolevConverged: false,
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
        setDescentMode: (m) => set({ descentMode: m, sobolevStats: null, sobolevConverged: false }),
        setRunning: (b) => {
            if (b) {
                // Play = a run (re)starts → re-anchor the frozen x₀ from the CURRENT
                // live positions and clear last run's diagnostics (see the sobolevX0
                // lifecycle anchor). The frame loop must never touch sobolevX0.
                const s = get();
                set({
                    running: true,
                    sobolevX0: barycenterTarget(s.live, s.graph.edges),
                    sobolevStats: null,
                    sobolevConverged: false,
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
                // re-anchor x₀ now (sobolevX0 lifecycle anchor). Diagnostics are
                // deliberately KEPT: on auto-pause they tell the user why it stopped.
                sobolevX0: barycenterTarget(s.live, s.graph.edges),
            });
        },
        setZoom: (z) => set({ zoom: z }),
    };
});
