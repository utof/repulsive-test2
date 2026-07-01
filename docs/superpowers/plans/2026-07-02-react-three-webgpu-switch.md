# React-Three-Fiber + WebGPU Viewer Switch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Canvas2D + hand-rolled-projection viewer with react-three-fiber v9 rendering through three.js `WebGPURenderer`, splitting `src/index.tsx` into four independently-modifiable areas (core / scene / ui / store) at feature parity.

**Architecture:** The CPU energy/gradient math (`tangentPointEnergy.ts`) moves untouched into `src/core/` and is wrapped by a pure `optimizer.step()`. A zustand `store.ts` bridges React's two reconcilers (DOM UI ↔ `<Canvas>` scene). The descent loop runs inside R3F `useFrame`, mutating a non-reactive live buffer and pushing straight to the GPU — no per-frame React render. DOM controls and stats read/write the store.

**Tech Stack:** Bun (dev server + test runner), React 19, `three@^0.185` (`three/webgpu`), `@react-three/fiber@^9`, `@react-three/drei@^10`, `zustand@^5`, TypeScript strict, Biome.

**Design spec:** `docs/superpowers/specs/2026-07-02-react-three-webgpu-switch-design.md` (read it — this plan implements it).

**Conventions for every task:**
- Formatter is Biome (4-space indent, single quotes, lineWidth 100). The pre-commit hook auto-formats staged files, so exact whitespace in the snippets below is a guide, not a gate.
- `verbatimModuleSyntax` is on: import types with `import type { … }`.
- Append this trailer to every commit message: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Typecheck command (used where unit tests don't apply): `bunx tsc --noEmit`.
- Tests run with `bun test <path>`.

---

## Task 1: Gate-0 — add deps, verify WebGPU boots & iterates under Bun

**This is a decision gate, not TDD.** It answers spec §9: does `three/webgpu` build *and* reload fast enough under `Bun.build`? Per the spec's policy — *verify and flag, do not stop*; if either criterion fails, note it and the fallback is pnpm + Vite (a separate follow-up, out of this plan's scope).

**Files:**
- Modify: `package.json` (deps)
- Create (throwaway, deleted in this task): `src/_spike.tsx`
- Temporarily edit: `index.html`

- [ ] **Step 1: Add the runtime deps and pin React**

Run:
```bash
bun add three@^0.185.1 @react-three/fiber@^9.6.1 @react-three/drei@^10.7.7 zustand@^5.0.14
bun add -d @types/three@^0.185.0
bun add react@~19.2.4 react-dom@~19.2.4
```
Expected: `package.json` `dependencies` now includes `three`, `@react-three/fiber`, `@react-three/drei`, `zustand`, with `react`/`react-dom` at `~19.2.4`; `@types/three` in `devDependencies`. `bun.lock` updated. Watch for peer-warning noise about `react` — a warning is acceptable (spec §3); an install *error* is not.

- [ ] **Step 2: Write the throwaway WebGPU spike**

Create `src/_spike.tsx`:
```tsx
import * as THREE from 'three/webgpu';
import { Canvas, extend, useFrame, type ThreeToJSXElements } from '@react-three/fiber';
import { useRef } from 'react';
import { createRoot } from 'react-dom/client';

declare module '@react-three/fiber' {
    interface ThreeElements extends ThreeToJSXElements<typeof THREE> {}
}
extend(THREE as any);

function Box() {
    const ref = useRef<THREE.Mesh>(null);
    useFrame((_, delta) => {
        if (ref.current) ref.current.rotation.y += delta;
    });
    return (
        <mesh ref={ref}>
            <boxGeometry args={[1, 1, 1]} />
            <meshStandardNodeMaterial color="#4a9eff" />
        </mesh>
    );
}

createRoot(document.getElementById('root')!).render(
    <Canvas
        flat
        camera={{ position: [2, 2, 2] }}
        gl={async (props) => {
            const renderer = new THREE.WebGPURenderer(props as any);
            await renderer.init();
            console.log('backend isWebGPU:', (renderer as any).backend?.isWebGPUBackend);
            return renderer;
        }}
    >
        <ambientLight intensity={0.8} />
        <directionalLight position={[3, 3, 3]} />
        <Box />
    </Canvas>,
);
```

- [ ] **Step 3: Point the dev server at the spike (temporary)**

In `index.html`, change the script tag from `./src/index.tsx` to `./src/_spike.tsx`.

- [ ] **Step 4: Run and verify boot + backend + iteration**

Run: `bun run dev` then open `http://localhost:3000` in a WebGPU-capable browser.
Expected:
1. A blue cube rotates.
2. DevTools console prints `backend isWebGPU: true` (confirms WebGPURenderer, not the WebGL2 fallback).
3. No `Top-level await`/bundling errors in the terminal.
4. **Iteration check:** edit `_spike.tsx` (e.g. change color to `#ff6b6b`), save, hard-reload. Note the terminal rebuild time — a couple seconds is fine, ~10s+ per reload is the DX-failure signal.

**Decision:** If (1)–(3) fail → `three/webgpu` doesn't bundle under Bun. If (4) is unacceptably slow → the uncached per-request `Bun.build` (`server.ts:23-27`) is the problem. Either way, **flag it in the task's completion note and recommend the pnpm+Vite fallback** (spec §9) as a separate effort; do not attempt the swap here. If all pass, proceed.

- [ ] **Step 5: Revert the spike, keep the deps**

Restore `index.html`'s script tag to `./src/index.tsx`. Delete the spike:
```bash
rm src/_spike.tsx
```

- [ ] **Step 6: Commit**

```bash
git add package.json bun.lock
git commit -m "chore: add three/R3F/drei/zustand deps; pin react (Gate-0: WebGPU boots under Bun)"
```

---

## Task 2: Move the research core into `src/core/` and fix all importers

Pure file move + import-path updates. The **math body stays byte-for-byte**; only `tangentPointEnergy.ts`'s header comment (which names its consumers) is edited (spec §7). Golden values must stay bit-identical.

**Files:**
- Move: `src/tangentPointEnergy.ts` → `src/core/tangentPointEnergy.ts`
- Move: `src/testConfigs.ts` → `src/core/testConfigs.ts`
- Modify importers: `src/index.tsx`, `src/viewRotation.ts`, `test/golden.test.ts`, `test/bench.ts`, `test/viewRotation.test.ts`, `test/gradientArrow.test.ts`, `test_gradient.ts`

- [ ] **Step 1: Move the two files with git**

```bash
mkdir -p src/core
git mv src/tangentPointEnergy.ts src/core/tangentPointEnergy.ts
git mv src/testConfigs.ts src/core/testConfigs.ts
```
(The internal `import type { Edge, Vec3 } from './testConfigs'` in `tangentPointEnergy.ts` needs **no change** — both files are now in `core/`.)

- [ ] **Step 2: Update the header comment in `src/core/tangentPointEnergy.ts`**

Replace the consumer line (currently `src/tangentPointEnergy.ts:4-5`):
```
 * SINGLE SOURCE OF TRUTH — imported by src/index.tsx (the app) and test_gradient.ts
 * (which verifies gradientAnalytical against central finite differences).
```
with:
```
 * SINGLE SOURCE OF TRUTH — imported by src/core/optimizer.ts, src/store.ts, and
 * test_gradient.ts (which verifies gradientAnalytical against central finite differences).
```

- [ ] **Step 3: Update every importer's path**

- `src/index.tsx`: `from './tangentPointEnergy'` → `from './core/tangentPointEnergy'`; `from './testConfigs'` → `from './core/testConfigs'`. (Leave `./viewRotation` as-is.)
- `src/viewRotation.ts` line 1: `from './testConfigs'` → `from './core/testConfigs'`.
- `test/golden.test.ts`: `from '../src/tangentPointEnergy'` → `from '../src/core/tangentPointEnergy'`; `from '../src/testConfigs'` → `from '../src/core/testConfigs'`.
- `test/bench.ts`: same two replacements as golden.
- `test/viewRotation.test.ts`: `from '../src/testConfigs'` → `from '../src/core/testConfigs'`. (Leave `../src/viewRotation`.)
- `test/gradientArrow.test.ts`: `from '../src/testConfigs'` → `from '../src/core/testConfigs'`. (Leave `../src/viewRotation`.)
- `test_gradient.ts` (repo root, easy to miss): `from './src/tangentPointEnergy'` → `from './src/core/tangentPointEnergy'`; `from './src/testConfigs'` → `from './src/core/testConfigs'`.

- [ ] **Step 4: Verify tests stay green (golden bit-identical) and knip is clean**

Run: `bun test`
Expected: all tests PASS — crucially `test/golden.test.ts` (bit-identical energy/gradient) is still green. A pure move cannot change IEEE-754 results.
Run: `bunx tsc --noEmit`
Expected: no new errors (the app still imports the old viewer modules, which still exist).
Run: `bunx knip --no-exit-code`
Expected: no *new* unused-file reports for the moved files (they're still imported).

- [ ] **Step 5: Confirm the old app still boots**

Run: `bun run dev`, open `http://localhost:3000`. Expected: the existing Canvas2D viewer works exactly as before (nothing visual changed yet).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: move energy/gradient core + presets into src/core/"
```

---

## Task 3: `core/optimizer.ts` — one pure gradient-descent step (TDD)

Extracts the per-frame math from `index.tsx`'s `animate()` into a pure, testable function.

**Files:**
- Create: `src/core/optimizer.ts`
- Test: `test/optimizer.test.ts`

- [ ] **Step 1: Write the failing test**

Create `test/optimizer.test.ts`:
```ts
import { expect, test } from 'bun:test';
import { step } from '../src/core/optimizer';
import {
    calculateDisjointPairs,
    calculateEnergy,
    gradientAnalytical,
} from '../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../src/core/testConfigs';

const vertices: Vec3[] = [
    [-1, 0, 0.3],
    [1, 0, 0.3],
    [0, -1, -0.3],
    [0, 1, -0.3],
];
const edges: Edge[] = [
    [0, 1],
    [2, 3],
];

test('step applies v - stepSize*grad and reports energy AT the new vertices (analytical)', () => {
    const pairs = calculateDisjointPairs(edges);
    const stepSize = 0.001;
    const grad = gradientAnalytical(vertices, edges, pairs, 3, 6, 1e-10);
    const expected: Vec3[] = vertices.map((v, i) => [
        v[0] - stepSize * grad[i][0],
        v[1] - stepSize * grad[i][1],
        v[2] - stepSize * grad[i][2],
    ]);
    const expectedEnergy = calculateEnergy(expected, edges, pairs, 3, 6, 1e-10);

    const out = step(vertices, edges, pairs, { mode: 'analytical', stepSize });

    expect(out.vertices).toEqual(expected);
    expect(Object.is(out.energy, expectedEnergy)).toBe(true);
});

test('step does not mutate its input vertices', () => {
    const pairs = calculateDisjointPairs(edges);
    step(vertices, edges, pairs, { mode: 'analytical', stepSize: 0.001 });
    expect(vertices[0]).toEqual([-1, 0, 0.3]);
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `bun test test/optimizer.test.ts`
Expected: FAIL — `Cannot find module '../src/core/optimizer'`.

- [ ] **Step 3: Write the implementation**

Create `src/core/optimizer.ts`:
```ts
import {
    calculateEnergy,
    gradientAnalytical,
    gradientFiniteDiff,
} from './tangentPointEnergy';
import type { Edge, Vec3 } from './testConfigs';

// Physical/numeric constants for the tangent-point descent. These were hardcoded
// in the old src/index.tsx; centralised here so the store and scene share one source.
// @see docs/superpowers/specs/2026-07-02-react-three-webgpu-switch-design.md §4.1
export const DEFAULTS = { alpha: 3, beta: 6, epsilon: 1e-10, h: 1e-6 } as const;

export interface StepOptions {
    mode: 'analytical' | 'finiteDiff';
    stepSize: number;
    alpha?: number;
    beta?: number;
    epsilon?: number;
    h?: number;
}

// One gradient-descent step. Pure: returns NEW arrays, never mutates inputs.
// Mirrors the old animate() sequence exactly: grad -> v - stepSize*grad -> energy(new v).
export function step(
    vertices: Vec3[],
    edges: Edge[],
    disjointPairs: number[][],
    opts: StepOptions,
): { vertices: Vec3[]; energy: number } {
    const alpha = opts.alpha ?? DEFAULTS.alpha;
    const beta = opts.beta ?? DEFAULTS.beta;
    const epsilon = opts.epsilon ?? DEFAULTS.epsilon;
    const h = opts.h ?? DEFAULTS.h;

    const grad =
        opts.mode === 'analytical'
            ? gradientAnalytical(vertices, edges, disjointPairs, alpha, beta, epsilon)
            : gradientFiniteDiff(vertices, edges, disjointPairs, alpha, beta, epsilon, h);

    const next: Vec3[] = vertices.map((v, i) => [
        v[0] - opts.stepSize * grad[i][0],
        v[1] - opts.stepSize * grad[i][1],
        v[2] - opts.stepSize * grad[i][2],
    ]);

    const energy = calculateEnergy(next, edges, disjointPairs, alpha, beta, epsilon);
    return { vertices: next, energy };
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `bun test test/optimizer.test.ts`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/core/optimizer.ts test/optimizer.test.ts
git commit -m "feat(core): pure gradient-descent step() extracted from the viewer loop"
```

---

## Task 4: `store.ts` — zustand store + `buildGraphState` helper (TDD the helper)

The store is framework-agnostic (imports only zustand + core). It owns config, the graph, the non-reactive live buffer, throttled stats, persistence, and commit-on-pause.

**Files:**
- Create: `src/store.ts`
- Test: `test/store.test.ts`

- [ ] **Step 1: Write the failing test for the pure helper**

Create `test/store.test.ts`:
```ts
import { expect, test } from 'bun:test';
import { buildGraphState } from '../src/store';
import { DEFAULTS } from '../src/core/optimizer';
import {
    calculateDisjointPairs,
    calculateEnergy,
} from '../src/core/tangentPointEnergy';
import { testConfigs } from '../src/core/testConfigs';

test('buildGraphState builds graph, disjoint pairs, and initial energy for a preset', () => {
    const crossing = testConfigs.find((t) => t.id === 'crossing')!;
    const built = buildGraphState(crossing, {});

    const expectedGraph = crossing.generate({});
    const expectedPairs = calculateDisjointPairs(expectedGraph.edges);
    const expectedEnergy = calculateEnergy(
        expectedGraph.vertices,
        expectedGraph.edges,
        expectedPairs,
        DEFAULTS.alpha,
        DEFAULTS.beta,
        DEFAULTS.epsilon,
    );

    expect(built.graph.vertices.length).toBe(expectedGraph.vertices.length);
    expect(built.graph.edges).toEqual(expectedGraph.edges);
    expect(built.disjointPairs).toEqual(expectedPairs);
    expect(Object.is(built.energy, expectedEnergy)).toBe(true);
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `bun test test/store.test.ts`
Expected: FAIL — `Cannot find module '../src/store'`.

- [ ] **Step 3: Write `src/store.ts`**

```ts
import { create } from 'zustand';
import { DEFAULTS } from './core/optimizer';
import { calculateDisjointPairs, calculateEnergy } from './core/tangentPointEnergy';
import { type GraphState, type TestConfig, testConfigs, type Vec3 } from './core/testConfigs';

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

export interface SimStore {
    // config (React-subscribed, infrequent)
    selectedTestId: string;
    testParams: Record<string, number>;
    mode: Mode;
    stepSize: number;
    running: boolean;
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
    setRunning(b: boolean): void;
    setZoom(z: number): void;
}

function initialConfig() {
    const saved = loadSavedConfig();
    const test = testConfigs.find((t) => t.id === saved.testId) ?? testConfigs[0];
    const params: Record<string, number> = {};
    if (test.params) for (const p of test.params) params[p.name] = saved.params[p.name] ?? p.default;
    return { test, params };
}

export const useSimStore = create<SimStore>((set, get) => {
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
        }));
        saveConfig(id, nextParams);
    };

    return {
        selectedTestId: test.id,
        testParams: params,
        mode: 'analytical',
        stepSize: 0.001,
        running: false,
        graph: built.graph,
        disjointPairs: built.disjointPairs,
        live: cloneVerts(built.graph.vertices),
        step: 0,
        energy: built.energy,
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
        setRunning: (b) => {
            if (b) {
                set({ running: true });
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
            });
        },
        setZoom: (z) => set({ zoom: z }),
    };
});
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `bun test test/store.test.ts`
Expected: PASS (1 test). (Importing `store.ts` executes `create(...)`, which calls `loadSavedConfig()` — the `try/catch` makes the missing `localStorage` a no-op.)

- [ ] **Step 5: Full test + typecheck sweep**

Run: `bun test` → all PASS. Run: `bunx tsc --noEmit` → no new errors.

- [ ] **Step 6: Commit**

```bash
git add src/store.ts test/store.test.ts
git commit -m "feat(store): zustand sim store + buildGraphState + commit-on-pause"
```

---

## Task 5: UI shell — `ui/ControlPanel.tsx`, `ui/Stats.tsx`, `App.tsx`, and swap `index.tsx`

Atomically replaces the old Canvas2D `index.tsx` with the new composition. The 3D `<Viewer/>` is a placeholder until Task 6, but controls + stats become live immediately (driven by the store). **Verification is manual (browser), not TDD** — R3F/DOM widgets aren't unit-tested in this repo.

**Files:**
- Create: `src/ui/ControlPanel.tsx`
- Create: `src/ui/Stats.tsx`
- Create: `src/App.tsx`
- Replace: `src/index.tsx`

- [ ] **Step 1: Write `src/ui/ControlPanel.tsx`**

```tsx
import { testConfigs } from '../core/testConfigs';
import { type Mode, useSimStore } from '../store';

const btn = (bg: string) => ({
    padding: '10px 20px',
    fontSize: 16,
    cursor: 'pointer',
    background: bg,
    color: '#fff',
    border: 'none',
    borderRadius: 5,
});

export function ControlPanel() {
    const selectedTestId = useSimStore((s) => s.selectedTestId);
    const testParams = useSimStore((s) => s.testParams);
    const running = useSimStore((s) => s.running);
    const mode = useSimStore((s) => s.mode);
    const stepSize = useSimStore((s) => s.stepSize);
    const setPreset = useSimStore((s) => s.setPreset);
    const setParam = useSimStore((s) => s.setParam);
    const regenerate = useSimStore((s) => s.regenerate);
    const reset = useSimStore((s) => s.reset);
    const setMode = useSimStore((s) => s.setMode);
    const setStepSize = useSimStore((s) => s.setStepSize);
    const setRunning = useSimStore((s) => s.setRunning);

    const selectedTest = testConfigs.find((t) => t.id === selectedTestId) ?? testConfigs[0];

    return (
        <>
            {/* Test selection + params */}
            <div
                style={{
                    marginBottom: 10,
                    display: 'flex',
                    gap: 15,
                    flexWrap: 'wrap',
                    alignItems: 'center',
                }}
            >
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span>Test:</span>
                    <select
                        value={selectedTestId}
                        onChange={(e) => setPreset(e.target.value)}
                        style={{ padding: 8, fontSize: 14, minWidth: 200 }}
                    >
                        {testConfigs.map((t) => (
                            <option key={t.id} value={t.id}>
                                {t.name}
                            </option>
                        ))}
                    </select>
                </label>

                {selectedTest.params?.map((p) => (
                    <label key={p.name} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span>{p.name}:</span>
                        <input
                            type="range"
                            min={p.min}
                            max={p.max}
                            value={testParams[p.name] ?? p.default}
                            onChange={(e) => setParam(p.name, parseInt(e.target.value))}
                            style={{ width: 80 }}
                        />
                        <span style={{ fontFamily: 'monospace', width: 40 }}>
                            {testParams[p.name] ?? p.default}
                        </span>
                    </label>
                ))}

                {selectedTest.params && (
                    <button type="button" onClick={regenerate} style={btn('#5577cc')}>
                        Regenerate
                    </button>
                )}
            </div>

            {/* Run controls */}
            <div
                style={{
                    marginBottom: 15,
                    display: 'flex',
                    gap: 15,
                    flexWrap: 'wrap',
                    alignItems: 'center',
                }}
            >
                <button
                    type="button"
                    onClick={() => setRunning(!running)}
                    style={btn(running ? '#ff4444' : '#44aa44')}
                >
                    {running ? 'Stop' : 'Start'}
                </button>
                <button type="button" onClick={reset} style={btn('#666')}>
                    Reset
                </button>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span>Mode:</span>
                    <select
                        value={mode}
                        onChange={(e) => setMode(e.target.value as Mode)}
                        style={{ padding: 8, fontSize: 14 }}
                    >
                        <option value="analytical">Analytical</option>
                        <option value="finiteDiff">Finite Diff</option>
                    </select>
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span>Step Size:</span>
                    <input
                        type="range"
                        min="-5"
                        max="-2"
                        step="0.1"
                        value={Math.log10(stepSize)}
                        onChange={(e) => setStepSize(10 ** parseFloat(e.target.value))}
                        style={{ width: 100 }}
                    />
                    <span style={{ fontFamily: 'monospace', width: 60 }}>
                        {stepSize.toExponential(0)}
                    </span>
                </label>
            </div>
        </>
    );
}
```

- [ ] **Step 2: Write `src/ui/Stats.tsx`**

```tsx
import { useSimStore } from '../store';

export function Stats() {
    const step = useSimStore((s) => s.step);
    const energy = useSimStore((s) => s.energy);
    const zoom = useSimStore((s) => s.zoom);
    const mode = useSimStore((s) => s.mode);
    const vertices = useSimStore((s) => s.graph.vertices.length);
    const edges = useSimStore((s) => s.graph.edges.length);

    return (
        <div style={{ marginBottom: 10, fontFamily: 'monospace' }}>
            <span style={{ marginRight: 20 }}>Step: {step}</span>
            <span style={{ marginRight: 20 }}>Energy: {energy.toFixed(6)}</span>
            <span style={{ marginRight: 20 }}>Vertices: {vertices}</span>
            <span style={{ marginRight: 20 }}>Edges: {edges}</span>
            <span style={{ marginRight: 20 }}>Zoom: {zoom.toFixed(2)}x</span>
            <span style={{ color: mode === 'analytical' ? '#00ff88' : '#ffaa00' }}>
                Gradient: {mode === 'analytical' ? 'Analytical (green)' : 'Finite Diff (orange)'}
            </span>
        </div>
    );
}
```

- [ ] **Step 3: Write `src/App.tsx` (Viewer placeholder for now)**

```tsx
import { ControlPanel } from './ui/ControlPanel';
import { Stats } from './ui/Stats';

export function App() {
    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', padding: 20 }}>
            <h1 style={{ fontSize: 24, marginBottom: 10 }}>Repulsive Energy Gradient Descent</h1>
            <ControlPanel />
            <Stats />
            <div style={{ flex: 1, position: 'relative', border: '2px solid #333', borderRadius: 8 }}>
                {/* <Viewer/> mounts here in Task 6 */}
            </div>
            <p style={{ marginTop: 10, color: '#888', fontSize: 14 }}>
                Drag to rotate. Scroll to zoom. Arrows show negative gradient (descent direction).
            </p>
        </div>
    );
}
```

- [ ] **Step 4: Replace `src/index.tsx` with the mount only**

Replace the ENTIRE contents of `src/index.tsx` with:
```tsx
import { createRoot } from 'react-dom/client';
import { App } from './App';

createRoot(document.getElementById('root')!).render(<App />);
```

- [ ] **Step 5: Typecheck + boot + verify controls drive the store**

Run: `bunx tsc --noEmit` → no errors.
Run: `bun run dev`, open `http://localhost:3000`. Expected:
- Title, control row, and stats render.
- The Test dropdown lists all 7 presets; selecting one changes `Vertices`/`Edges`/`Energy` in Stats (energy is non-zero *before* Start — parity with old behavior).
- Param sliders + Regenerate change the stats for parametric presets.
- Start/Stop toggles the button label (no motion yet — no Viewer).
- (The 3D area is an empty bordered box for now.)

Run: `bun test` → still all green.

- [ ] **Step 6: Commit**

```bash
git add src/ui/ControlPanel.tsx src/ui/Stats.tsx src/App.tsx src/index.tsx
git commit -m "feat(ui): control panel + stats + App shell; mount via store (Canvas2D app removed)"
```

---

## Task 6: `scene/Viewer.tsx` + `scene/Curve.tsx` — render the graph under WebGPU

Adds the R3F `<Canvas>` (WebGPURenderer), OrbitControls, lights, and the edge/vertex rendering. Edges use the **confirmed** thin-line path (`<lineSegments>` + `<lineBasicNodeMaterial>`); the fat-line upgrade is Task 10. Manual (browser) verification.

**Files:**
- Create: `src/scene/Viewer.tsx`
- Create: `src/scene/Curve.tsx`
- Modify: `src/App.tsx` (mount `<Viewer/>`)

- [ ] **Step 1: Write `src/scene/Curve.tsx`**

```tsx
import { useFrame } from '@react-three/fiber';
import { useMemo, useRef } from 'react';
import * as THREE from 'three/webgpu';
import { useSimStore } from '../store';

const VERTEX_RADIUS = 0.06;
const tmp = new THREE.Object3D();

// Reads graph topology from the store (re-renders only on rebuild — Viewer keys us on
// graphVersion), and pulls live positions every frame from the non-reactive buffer.
export function Curve() {
    const edges = useSimStore((s) => s.graph.edges);
    const count = useSimStore((s) => s.graph.vertices.length);

    const lineGeom = useMemo(() => {
        const g = new THREE.BufferGeometry();
        g.setAttribute(
            'position',
            new THREE.BufferAttribute(new Float32Array(edges.length * 2 * 3), 3),
        );
        return g;
    }, [edges]);

    const meshRef = useRef<THREE.InstancedMesh>(null);

    useFrame(() => {
        const live = useSimStore.getState().live;

        const pos = lineGeom.getAttribute('position') as THREE.BufferAttribute;
        const arr = pos.array as Float32Array;
        for (let e = 0; e < edges.length; e++) {
            const [a, b] = edges[e];
            const va = live[a];
            const vb = live[b];
            arr[e * 6 + 0] = va[0];
            arr[e * 6 + 1] = va[1];
            arr[e * 6 + 2] = va[2];
            arr[e * 6 + 3] = vb[0];
            arr[e * 6 + 4] = vb[1];
            arr[e * 6 + 5] = vb[2];
        }
        pos.needsUpdate = true;

        const mesh = meshRef.current;
        if (mesh) {
            for (let i = 0; i < count; i++) {
                tmp.position.set(live[i][0], live[i][1], live[i][2]);
                tmp.updateMatrix();
                mesh.setMatrixAt(i, tmp.matrix);
            }
            mesh.instanceMatrix.needsUpdate = true;
        }
    });

    return (
        <group>
            <lineSegments geometry={lineGeom}>
                <lineBasicNodeMaterial color="#4a9eff" />
            </lineSegments>
            <instancedMesh ref={meshRef} args={[undefined, undefined, count]}>
                <sphereGeometry args={[VERTEX_RADIUS, 16, 16]} />
                <meshStandardNodeMaterial color="#ff6b6b" />
            </instancedMesh>
        </group>
    );
}
```

- [ ] **Step 2: Write `src/scene/Viewer.tsx`**

```tsx
import { OrbitControls } from '@react-three/drei';
import { Canvas, extend, type ThreeToJSXElements } from '@react-three/fiber';
import { useEffect, useRef } from 'react';
import * as THREE from 'three/webgpu';
import { useSimStore } from '../store';
import { Curve } from './Curve';

declare module '@react-three/fiber' {
    interface ThreeElements extends ThreeToJSXElements<typeof THREE> {}
}
extend(THREE as any);

const CAMERA_POS: [number, number, number] = [3, 3, 3];

export function Viewer() {
    // Guard: WebGPU unavailable (old browser / non-secure context) — surface, don't blank.
    if (typeof navigator !== 'undefined' && !navigator.gpu) {
        return (
            <div style={{ padding: 20, color: '#ffaa00' }}>
                WebGPU is unavailable in this browser/context. Use a WebGPU-capable browser
                over http://localhost.
            </div>
        );
    }

    const controls = useRef<any>(null);
    const viewResetNonce = useSimStore((s) => s.viewResetNonce);
    const graphVersion = useSimStore((s) => s.graphVersion);

    // Reset re-centres the camera (old Reset zeroed zoom). @see spec §4.1.
    useEffect(() => {
        controls.current?.reset?.();
    }, [viewResetNonce]);

    return (
        <Canvas
            flat
            camera={{ position: CAMERA_POS, fov: 50 }}
            gl={async (props) => {
                const renderer = new THREE.WebGPURenderer(props as any);
                await renderer.init();
                return renderer;
            }}
        >
            <ambientLight intensity={0.8} />
            <directionalLight position={[5, 5, 5]} intensity={0.6} />
            <OrbitControls ref={controls} minPolarAngle={0} maxPolarAngle={Math.PI} />
            <Curve key={graphVersion} />
        </Canvas>
    );
}
```

- [ ] **Step 3: Mount `<Viewer/>` in `src/App.tsx`**

Add the import at the top:
```tsx
import { Viewer } from './scene/Viewer';
```
Replace the placeholder `<div style={{ flex: 1, ... }}>{/* <Viewer/> ... */}</div>` with:
```tsx
            <div style={{ flex: 1, position: 'relative' }}>
                <Viewer />
            </div>
```

- [ ] **Step 4: Typecheck + boot + verify rendering**

Run: `bunx tsc --noEmit`.
Expected: no errors. **If `'three/webgpu'` or a NodeMaterial JSX intrinsic reports missing types**, that's the known bleeding-edge type gap (spec §6) — add a minimal `// @ts-expect-error` on the offending line or a one-line ambient shim `src/scene/three-webgpu.d.ts`; the runtime browser check below is the real gate.
Run: `bun run dev`, open `http://localhost:3000`. Expected:
- Blue edges + red vertex spheres render for the default `crossing` preset.
- Left-drag orbits; scroll dollies (zoom). Orbit cannot flip upside-down (polar clamp).
- Switching presets / Regenerate rebuilds the visible geometry (Curve remounts on `graphVersion`).
- Colors are the exact viz palette (no washed-out tone-mapping — thanks to `<Canvas flat>`).

- [ ] **Step 5: Commit**

```bash
git add src/scene/Viewer.tsx src/scene/Curve.tsx src/App.tsx
git commit -m "feat(scene): WebGPU Canvas + OrbitControls + edge/vertex rendering"
```

---

## Task 7: `scene/Simulation` — the descent loop (useFrame) + throttled stats + zoom

Runs `optimizer.step()` each frame while `running`, mutating the live buffer in place; throttles `{step, energy}` and camera-distance→`zoom` to the store. Stop triggers the store's commit-on-pause. Manual verification.

**Files:**
- Modify: `src/scene/Viewer.tsx` (add the `<Simulation/>` component + mount it)

- [ ] **Step 1: Add `Simulation` to `src/scene/Viewer.tsx`**

Add imports at the top:
```tsx
import { useFrame } from '@react-three/fiber';
import { step as descentStep } from '../core/optimizer';
```
Add this component above `export function Viewer()`:
```tsx
const BASE_DISTANCE = Math.hypot(...CAMERA_POS);

// The optimization loop. Lives inside <Canvas> so it can use useFrame. Mutates the
// live buffer in place (no React render); publishes stats/zoom throttled (~10Hz / 5Hz).
function Simulation() {
    const statAcc = useRef(0);
    const camAcc = useRef(0);
    const iters = useRef(0);

    useFrame((state, delta) => {
        const st = useSimStore.getState();

        if (st.running) {
            const { vertices, energy } = descentStep(st.live, st.graph.edges, st.disjointPairs, {
                mode: st.mode,
                stepSize: st.stepSize,
            });
            for (let i = 0; i < st.live.length; i++) {
                const v = vertices[i];
                const l = st.live[i];
                l[0] = v[0];
                l[1] = v[1];
                l[2] = v[2];
            }
            iters.current += 1;
            statAcc.current += delta;
            if (statAcc.current > 0.1) {
                statAcc.current = 0;
                useSimStore.setState({ step: iters.current, energy });
            }
        } else {
            // keep the iteration counter in sync with a rebuilt/committed state
            iters.current = st.step;
        }

        camAcc.current += delta;
        if (camAcc.current > 0.2) {
            camAcc.current = 0;
            const dist = state.camera.position.length();
            useSimStore.getState().setZoom(dist > 0 ? BASE_DISTANCE / dist : 1);
        }
    });

    return null;
}
```

- [ ] **Step 2: Mount `<Simulation/>` inside the Canvas**

In `Viewer`'s returned JSX, add `<Simulation />` as a child of `<Canvas>` (e.g. right after `<Curve key={graphVersion} />`):
```tsx
            <Curve key={graphVersion} />
            <Simulation />
```

- [ ] **Step 3: Typecheck + boot + verify descent**

Run: `bunx tsc --noEmit`.
Run: `bun run dev`, open `http://localhost:3000`. Expected:
- Press **Start**: the geometry moves (gradient descent), `Step` climbs, `Energy` decreases in Stats.
- Switch **Mode** to Finite Diff and Start again: still descends (slower per frame).
- Adjust **Step Size**: larger steps move faster.
- Press **Stop**: motion halts; `Energy` shows the committed value; the geometry stays where it stopped (commit-on-pause).
- **Zoom** stat updates as you scroll.
- **Reset** returns to the preset's initial geometry and re-centres the camera.

- [ ] **Step 4: Commit**

```bash
git add src/scene/Viewer.tsx
git commit -m "feat(scene): useFrame descent loop with throttled stats + commit-on-pause"
```

---

## Task 8: `scene/GradientArrows.tsx` — real-3D negative-gradient arrows (shown when paused)

Instanced cones oriented in world space toward the descent direction. This replaces the old `projectArrow` hack; three's camera handles foreshortening. Manual verification.

**Files:**
- Create: `src/scene/GradientArrows.tsx`
- Modify: `src/scene/Viewer.tsx` (mount `<GradientArrows/>`)

- [ ] **Step 1: Write `src/scene/GradientArrows.tsx`**

```tsx
import { useLayoutEffect, useMemo, useRef } from 'react';
import * as THREE from 'three/webgpu';
import { DEFAULTS } from '../core/optimizer';
import { gradientAnalytical, gradientFiniteDiff, norm } from '../core/tangentPointEnergy';
import type { Vec3 } from '../core/testConfigs';
import { useSimStore } from '../store';

// World-space analogue of the old MIN/MAX pixel clamps: keep near-convergence tiny
// gradients visible; the perspective-blowup MAX case is now handled by real 3D geometry.
// @see spec §4.1 (GradientArrows) / §6.
const ARROW_SCALE = 0.2;
const MIN_WORLD = 0.08;
const MAX_WORLD = 1.2;
const CONE_RADIUS = 0.03;

const UP = new THREE.Vector3(0, 1, 0);
const tmp = new THREE.Object3D();
const dirVec = new THREE.Vector3();

interface Arrow {
    pos: Vec3;
    dir: Vec3; // unit negative-gradient
    len: number; // world units
}

export function GradientArrows() {
    const running = useSimStore((s) => s.running);
    const mode = useSimStore((s) => s.mode);
    const graph = useSimStore((s) => s.graph);
    const disjointPairs = useSimStore((s) => s.disjointPairs);

    // Recompute only when paused config changes. graph.vertices is current here because
    // <Simulation/>'s stop path committed the live buffer (spec §6 stale-arrow fix).
    const arrows = useMemo<Arrow[]>(() => {
        if (running) return [];
        const grad =
            mode === 'analytical'
                ? gradientAnalytical(
                      graph.vertices,
                      graph.edges,
                      disjointPairs,
                      DEFAULTS.alpha,
                      DEFAULTS.beta,
                      DEFAULTS.epsilon,
                  )
                : gradientFiniteDiff(
                      graph.vertices,
                      graph.edges,
                      disjointPairs,
                      DEFAULTS.alpha,
                      DEFAULTS.beta,
                      DEFAULTS.epsilon,
                      DEFAULTS.h,
                  );
        const out: Arrow[] = [];
        for (let i = 0; i < graph.vertices.length; i++) {
            const g = grad[i];
            const gn = norm(g);
            if (gn <= 1e-6) continue; // parity with old index.tsx skip
            out.push({
                pos: graph.vertices[i],
                dir: [-g[0] / gn, -g[1] / gn, -g[2] / gn],
                len: Math.max(MIN_WORLD, Math.min(MAX_WORLD, Math.log(1 + gn) * ARROW_SCALE)),
            });
        }
        return out;
    }, [running, mode, graph, disjointPairs]);

    const meshRef = useRef<THREE.InstancedMesh>(null);
    useLayoutEffect(() => {
        const mesh = meshRef.current;
        if (!mesh) return;
        for (let i = 0; i < arrows.length; i++) {
            const a = arrows[i];
            dirVec.set(a.dir[0], a.dir[1], a.dir[2]);
            tmp.position.set(a.pos[0], a.pos[1], a.pos[2]);
            tmp.quaternion.setFromUnitVectors(UP, dirVec);
            tmp.scale.set(1, a.len, 1);
            tmp.updateMatrix();
            mesh.setMatrixAt(i, tmp.matrix);
        }
        mesh.count = arrows.length;
        mesh.instanceMatrix.needsUpdate = true;
    }, [arrows]);

    if (running) return null;

    const color = mode === 'analytical' ? '#00ff88' : '#ffaa00';
    const maxCount = graph.vertices.length;
    return (
        <instancedMesh ref={meshRef} args={[undefined, undefined, maxCount]}>
            <coneGeometry args={[CONE_RADIUS, 1, 8]} />
            <meshBasicNodeMaterial color={color} />
        </instancedMesh>
    );
}
```

- [ ] **Step 2: Mount `<GradientArrows/>` in `src/scene/Viewer.tsx`**

Add the import:
```tsx
import { GradientArrows } from './GradientArrows';
```
Add it (keyed on `graphVersion`, like `Curve`) inside `<Canvas>`:
```tsx
            <Curve key={graphVersion} />
            <GradientArrows key={graphVersion} />
            <Simulation />
```

- [ ] **Step 3: Typecheck + boot + verify arrows**

Run: `bunx tsc --noEmit`.
Run: `bun run dev`, open `http://localhost:3000`. Expected:
- While **paused**: green cones point along the descent direction at each vertex; near-flat (tiny-gradient) vertices still show a small arrow (MIN_WORLD floor); no arrows on `‖g‖≤1e-6` vertices.
- Switch **Mode** to Finite Diff (paused): arrows turn **orange** and match the analytical directions closely.
- Press **Start**: arrows disappear; press **Stop**: arrows reappear at the new (committed) positions — *not* the pre-run positions (confirms the stale-arrow fix).
- Orbit around: arrows foreshorten naturally and never whip/flip at the screen center (the old bug class is gone).

- [ ] **Step 4: Commit**

```bash
git add src/scene/GradientArrows.tsx src/scene/Viewer.tsx
git commit -m "feat(scene): real-3D instanced gradient arrows (paused), commit-aware"
```

---

## Task 9: Delete the dead projection module + its tests; final parity verification

`viewRotation.ts` and its two tests are now dead (OrbitControls + the camera own projection). Remove them and run the full gate.

**Files:**
- Delete: `src/viewRotation.ts`
- Delete: `test/viewRotation.test.ts`
- Delete: `test/gradientArrow.test.ts`

- [ ] **Step 1: Confirm nothing still imports the module**

Run: `grep -rn "viewRotation" src test test_gradient.ts`
Expected: matches ONLY in the three files being deleted (`src/viewRotation.ts`, `test/viewRotation.test.ts`, `test/gradientArrow.test.ts`). If anything in `src/` or `App`/`scene` still imports it, stop and fix that first.

- [ ] **Step 2: Delete the files**

```bash
git rm src/viewRotation.ts test/viewRotation.test.ts test/gradientArrow.test.ts
```

- [ ] **Step 3: Full verification sweep**

Run: `bun test`
Expected: PASS — `golden.test.ts` (bit-identical), `bench.ts` unaffected, `optimizer.test.ts`, `store.test.ts`. No references to the deleted tests.
Run: `bunx tsc --noEmit` → no errors.
Run: `bunx knip --no-exit-code`
Expected: no unused-file/dep reports for the new modules; `viewRotation` no longer listed.
Run: `bun run dev` and walk the **full parity checklist (spec §8)**: 7 presets · param sliders + Regenerate · Start/Stop/Reset (Reset re-centres camera) · Analytical↔FiniteDiff · step-size slider · stats (step/energy/vertices/edges/zoom + green/orange label) · energy shown before Start · orbit + zoom · arrows when paused (green/orange, tiny ones visible) · localStorage persistence (reload keeps last preset+params).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: delete hand-rolled viewRotation projection + its tests (OrbitControls owns it)"
```

---

## Task 10 (optional, verify-at-runtime): upgrade edges to fat WebGPU lines

Realizes spec §6's *primary* edge design (fat blue lines, parity with old `lineWidth: 3`). Uses the WebGPU-safe line classes. **If it doesn't render, keep the thin-line fallback from Task 6** — do not block on this.

**Files:**
- Modify: `src/scene/Viewer.tsx` (extend the line classes)
- Modify: `src/scene/Curve.tsx` (swap `<lineSegments>` for `LineSegments2`)

- [ ] **Step 1: Register the WebGPU line classes in `src/scene/Viewer.tsx`**

Add near the top (these live in three addons, not the `THREE` namespace, so `extend(THREE)` doesn't cover them):
```tsx
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineSegments2 } from 'three/addons/lines/webgpu/LineSegments2.js';
// Line2NodeMaterial ships in the three/webgpu namespace (already imported as THREE.*).
extend({ LineSegments2, LineSegmentsGeometry, Line2NodeMaterial: THREE.Line2NodeMaterial });
```
(If TS complains about missing types for the `three/addons/...` paths, add `// @ts-expect-error` on the import lines — the runtime browser check is the gate.)

- [ ] **Step 2: Swap the edge primitive in `src/scene/Curve.tsx`**

Replace the `lineGeom` `useMemo` and the `<lineSegments>` element. New geometry:
```tsx
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';

// ...inside Curve():
const lineGeom = useMemo(() => new LineSegmentsGeometry(), [edges]);
```
Update the per-frame writer to feed flat positions to `setPositions`:
```tsx
// in useFrame, replace the edge-writing block with:
const flat = new Float32Array(edges.length * 6);
for (let e = 0; e < edges.length; e++) {
    const [a, b] = edges[e];
    const va = live[a];
    const vb = live[b];
    flat[e * 6 + 0] = va[0];
    flat[e * 6 + 1] = va[1];
    flat[e * 6 + 2] = va[2];
    flat[e * 6 + 3] = vb[0];
    flat[e * 6 + 4] = vb[1];
    flat[e * 6 + 5] = vb[2];
}
(lineGeom as any).setPositions(flat);
```
Replace the JSX element:
```tsx
<lineSegments2 args={[lineGeom as any]}>
    <line2NodeMaterial color="#4a9eff" linewidth={3} worldUnits={false} />
</lineSegments2>
```

- [ ] **Step 3: Typecheck + boot + verify (or revert)**

Run: `bunx tsc --noEmit` (expect possible addon-type noise; suppress as above).
Run: `bun run dev`. Expected: edges render as **fat** blue lines (width ~3px), still track the descent, orbit/zoom fine.
**If edges disappear, error, or render wrong** (WebGPU line support is the least-certain API in the spec): revert this task with `git checkout -- src/scene/Curve.tsx src/scene/Viewer.tsx` and keep the thin-line version — it's a legitimate, spec-sanctioned fallback.

- [ ] **Step 4: Commit (only if it works)**

```bash
git add src/scene/Curve.tsx src/scene/Viewer.tsx
git commit -m "feat(scene): fat WebGPU lines for edges (LineSegments2 + Line2NodeMaterial)"
```

---

## Follow-up (NOT in this plan): React Compiler

Per spec §11, enabling `babel-plugin-react-compiler` is a separate milestone sequenced *after* the bundler question (Gate-0) settles — ~3 lines under Vite, or a Babel stage under Bun. All components above are written Rules-of-React-clean (hooks unconditional, no ref mutation during render), so that milestone is config + verification only.

---

## Self-Review (completed by plan author)

**Spec coverage:** §2 non-goals honored (math untouched, Task 2; not pixel-parity noted in Task 9). §3 stack/pins → Task 1. §4 architecture (core/scene/ui/store) → Tasks 2–8. §5 data flow (live buffer + throttled stats + commit) → Tasks 4, 7. §6 WebGPU wiring (flat, casts, NodeMaterials, WebGPU-safe lines, OrbitControls polar) → Tasks 6, 10. §7 deletions + move collateral (incl. `test_gradient.ts`, header comment) → Tasks 2, 9. §8 parity checklist → Task 9 Step 3. §9 Gate-0 (build + iteration) + WebGPU-unavailable + dark-window-avoided (atomic swap Task 5) → Tasks 1, 6. §11 React Compiler → Follow-up section.

**Placeholder scan:** no TBD/TODO; every code step shows complete code; commands have expected output.

**Type consistency:** store surface (`useSimStore`, `setPreset/setParam/regenerate/reset/setMode/setStepSize/setRunning/setZoom`, `live`, `graphVersion`, `viewResetNonce`, `graph`, `disjointPairs`, `mode`, `stepSize`, `zoom`, `step`, `energy`) is defined in Task 4 and consumed identically in Tasks 5–8. `optimizer.step(vertices, edges, disjointPairs, {mode, stepSize})` + `DEFAULTS` defined in Task 3, used in Tasks 4, 7, 8. `buildGraphState(test, params)` defined and consumed in Task 4.
