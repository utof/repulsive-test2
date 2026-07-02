# Design: Switch the viewer to react-three-fiber + WebGPU

**Status:** approved (brainstorming), revised after adversarial review — ready for implementation plan
**Branch:** `feat/react-three-switch`
**Date:** 2026-07-02
**Revision:** r2 — fixes from Opus adversarial review: WebGPU-safe line classes (was a blocker), stale-arrow commit, §7 move-collateral, tone-mapping, Bun re-bundle DX, plus React Compiler follow-up milestone.

## 1. Goal

Replace the hand-rolled Canvas2D + manual-projection viewer with **react-three-fiber (R3F) v9** rendering through three.js **`WebGPURenderer`**, and in doing so split today's 513-line `src/index.tsx` monolith into four independently-modifiable areas: the **research/optimization core**, the **scene/rendering**, the **UI controls**, and a thin **store** that glues them. Each area must be understandable and changeable — by a human or an AI — without reading the others.

## 2. Non-goals (explicitly out of scope)

- **No GPU compute.** WebGPU is used as a *renderer only*. The energy/gradient math in `tangentPointEnergy.ts` stays in JS on the CPU, byte-for-byte unchanged. Porting the O(pairs) hot loop to TSL compute shaders is a **separate future milestone**. (Brainstorming decision: "Renderer only (now)".)
- **No math/behavior changes.** No new presets, no new gradient variants, no numerical tuning. Feature-parity migration.
- **No unrelated refactoring** of the core math beyond moving files and updating import paths.
- **Not pixel-parity.** The old `project()` is a bespoke weak-perspective (`fov=3`, `s=fov/(fov+z+3)`, `×150·zoom`; `viewRotation.ts:35-41`) with fixed 8px vertex dots. three's `PerspectiveCamera` + OrbitControls foreshortens differently and vertices become fixed *world*-radius spheres. Parity target is *behavioral* (orbit/zoom/descent/arrows), not pixel-identical framing.

## 3. Decisions locked in

- **Renderer-only WebGPU** (see non-goals).
- **State bridge = zustand.** R3F's `<Canvas>` is a *separate React reconciler*, so outer-app React Context does not cross into the scene (confirmed idiomatic — the mutable-buffer-in-`useFrame` + throttled-`set()` split is the standard R3F transient-update pattern; no race, single rAF thread). zustand is already a transitive dep of R3F, so the cost is one explicit dep. (Approved over prop-drilling.)
- **Verified stack** (researched + review-verified 2026-07-02, all versions current latest & mutually peer-compatible; citations §10):

  | Package | Pin | Notes |
  |---|---|---|
  | `three` | `^0.185.1` | WebGPU entry `three/webgpu`; stable since r171 |
  | `@react-three/fiber` | `^9.6.1` | v9's async `gl` callback is the WebGPU enabler; peer `react/react-dom >=19 <19.3` |
  | `@react-three/drei` | `^10.7.7` | `<OrbitControls>` renderer-agnostic (peer `@react-three/fiber ^9`, `three >=0.159`) |
  | `@types/three` | `^0.185.0` | keep aligned to `three` major.minor |
  | `zustand` | `^5.0.14` | explicit dep (already transitive via R3F 9.6.1) |

- **Pin React *and* React-DOM to `~19.2`.** R3F 9.6.1's peer is `>=19 <19.3` for both; current `^19.2.4` can float to 19.3. A mismatch is a peer *warning*, not a hard break, but narrow both ranges to stay in-spec.
- **React Compiler is a separate follow-up milestone, not this one** — but every new `scene/` + `ui/` component MUST be written Rules-of-React-clean (no ref mutation during render, no conditional hooks) so enabling it later is flip-a-switch. Rationale + sequencing in §11.

## 4. Target architecture

```
src/
  core/                     ── Research / optimization. Pure TS. Zero React, zero Three.
    tangentPointEnergy.ts   ·  (moved, math body UNCHANGED) energy + gradients + disjoint pairs
    testConfigs.ts          ·  (moved) presets + generators + Vec3/Edge/GraphState types
    optimizer.ts            ·  NEW — one pure descent step (extracted from index.tsx animate())
  scene/                    ── Rendering. R3F + three/webgpu. Knows nothing about UI widgets.
    Viewer.tsx              ·  <Canvas flat> + WebGPURenderer gl factory + OrbitControls + camera/lights
                               + inner <Simulation/> (useFrame descent loop, commit-on-pause)
    Curve.tsx               ·  edges (WebGPU-safe LineSegments2) + vertices (instancedMesh spheres)
    GradientArrows.tsx      ·  instancedMesh cones oriented in real 3D for the −gradient
  ui/                       ── React DOM controls. Knows nothing about Three.
    ControlPanel.tsx        ·  preset select, param sliders, start/stop/reset, mode, step-size
    Stats.tsx               ·  step / energy / vertex / edge / zoom + gradient-mode label
  store.ts                  ── zustand: config + live position buffer + throttled stats + persistence
  App.tsx                   ── composition root: <ControlPanel/> + <Viewer/> + <Stats/>
  index.tsx                 ── createRoot mount only
```

### 4.1 Module responsibilities & interfaces

**`core/optimizer.ts`** — pure, no React/Three. Owns the descent-step math and its constants (currently hardcoded in `index.tsx`: `alpha=3, beta=6, epsilon=1e-10, h=1e-6`).
```ts
export const DEFAULTS = { alpha: 3, beta: 6, epsilon: 1e-10, h: 1e-6 } as const;
export interface StepOptions { mode: 'analytical' | 'finiteDiff'; stepSize: number;
  alpha?: number; beta?: number; epsilon?: number; h?: number; }
// One gradient-descent step. Returns the new vertex positions and the energy AT those
// new positions. Pure: identical inputs → identical outputs (CPU, bit-stable).
export function step(vertices: Vec3[], edges: Edge[], disjointPairs: number[][],
  opts: StepOptions): { vertices: Vec3[]; energy: number };
```
Internally: pick `gradientAnalytical | gradientFiniteDiff`, compute `v − stepSize·grad`, then `calculateEnergy` on the result — exactly the sequence in today's `animate()`.

**`store.ts`** — zustand. Single source of truth shared across the reconciler boundary.
- *Config (React-subscribed, changes infrequently):* `selectedTestId`, `testParams`, `mode`, `stepSize`, `running`.
- *Graph:* `graph: GraphState`, `disjointPairs: number[][]` (recomputed via `calculateDisjointPairs` whenever the graph is rebuilt — mirrors today's `disjointPairs.current` ref).
- *Stats (React-subscribed, THROTTLED):* `step`, `energy`. **`energy` is computed, not zeroed, on every graph rebuild** (parity with `index.tsx:286-296`, which shows a meaningful energy *before* Start).
- *Live position buffer (NOT React-subscribed):* a mutable `Vec3[]`/`Float32Array` the descent loop mutates in place every frame; the scene reads it in `useFrame`. This avoids today's per-frame `setGraph` full-tree re-render. Invariant: while `running`, the live buffer is ahead of `graph.vertices`; on pause it is **committed back** (see below) so the two agree.
- *Actions:* `setPreset(id)`, `setParam(name,val)`, `regenerate()`, `reset()`, `setMode(m)`, `setStepSize(s)`, `setRunning(b)`, **`commit()`**. Preset/param/regenerate/reset rebuild the graph, recompute `disjointPairs`, recompute `energy`, and reset `step` + the live buffer. `commit()` copies the live buffer into `graph.vertices` (via `set()`) and recomputes `energy` — called by `<Simulation/>` on the running→false transition so paused consumers (arrows) read current positions.
- *Persistence:* initialise from `loadSavedConfig()`; subscribe to `(selectedTestId, testParams)` → `saveConfig()` (localStorage helpers move here, key `repulsive-test-config` unchanged).
- *Camera reset is NOT store state.* Old `Reset` also re-centres the view (`index.tsx:332`, `setZoom(1)`); under R3F that is `controls.reset()` on the OrbitControls ref, wired scene-side and triggered by the same Reset action.

**`scene/Viewer.tsx`** — the only file that touches WebGPU wiring. Hosts:
- module-scope `extend(THREE)` (cast `as any`) + the `declare module '@react-three/fiber'` augmentation, plus explicit `extend({ LineSegments2, LineSegmentsGeometry, Line2NodeMaterial })` for the line classes (not covered by `extend(THREE)`);
- the `<Canvas flat gl={…}>` async factory (`flat` = `NoToneMapping`, see §6);
- `<OrbitControls minPolarAngle={0} maxPolarAngle={Math.PI} />` (defaults, stated explicitly; reproduces the old `clampPolar`);
- camera + lights; `frameloop` left at its default `"always"` (a `"demand"` loop would stall the `useFrame` descent);
- an inner `<Simulation/>` running the `useFrame` loop: when `running`, call `optimizer.step(...)`, write into the live buffer, throttle `{step, energy}` to the store; on the running→false edge call `store.commit()`; when paused, idle.

**`scene/Curve.tsx`** — reads `graph.edges` + the live position buffer. Renders **one WebGPU-safe `LineSegments2`** (see §6) for all edges (single draw call) and one `instancedMesh` sphere for vertices (`setMatrixAt` per vertex, `instanceMatrix.needsUpdate=true`), both using `*NodeMaterial`. Syncs GPU attributes from the live buffer each frame.

**`scene/GradientArrows.tsx`** — shown only when **paused** (parity). Reads vertex positions from **`graph.vertices`** (current, because `<Simulation/>` committed the live buffer on pause — this is the fix for the stale-arrow hole). Recompute is keyed on `{running, graph.vertices, mode}` so the green↔orange toggle refreshes. Computes the gradient (`gradientAnalytical | gradientFiniteDiff`), skips vertices with `‖g‖ ≤ 1e-6` (parity with `index.tsx:185`), and renders an `instancedMesh` of cones: position at the vertex, orient with `quaternion.setFromUnitVectors([0,1,0], dir)` toward the **negative** gradient, length = `clamp(log(1+‖g‖)·scale, minWorld, maxWorld)` in **world units** (the world-space analogue of the old `MIN_ARROW_LEN`/`MAX_ARROW_LEN` pixel clamps — keeps near-convergence tiny gradients visible; the perspective-blowup MAX case is now handled by real 3D geometry).

**`ui/ControlPanel.tsx`, `ui/Stats.tsx`** — plain React DOM reading/writing the store. No Three imports.

## 5. Data flow

```
ControlPanel ──writes config──▶ store ◀──reads config── Viewer/Simulation
                                  │                          │ useFrame: optimizer.step()
Stats ◀──throttled step/energy───┤                          │ mutates live buffer (no React render)
persistence ◀──preset/params─────┘                          │ on run→pause: store.commit()
                                                             ▼
                                              Curve (live buffer) + GradientArrows (graph.vertices)
```

**Why this is faster than today:** the current `animate()` calls `setGraph(...)` every frame, re-rendering the whole React tree at 60 fps. Here the descent loop mutates a buffer inside `useFrame` and pushes directly to the GPU; React only re-renders on config changes, throttled stats, and the single commit at pause.

## 6. WebGPU wiring (grounded)

- **One import registry.** Everything three comes from `three/webgpu`; never mix with `'three'` (mixing WebGL-only materials/classes yields e.g. `Material 'MeshStandardMaterial' is not compatible`, R3F discussion #3043). Standard materials do **not** render under WebGPU — use `*NodeMaterial` throughout (`meshBasicNodeMaterial`/`meshStandardNodeMaterial`/`lineBasicNodeMaterial`/`Line2NodeMaterial`). TSL helpers, if ever needed, from `three/tsl`.
- **Setup (module scope, once):** `extend(THREE as any)` + `declare module '@react-three/fiber' { interface ThreeElements extends ThreeToJSXElements<typeof THREE> {} }`, plus `extend({ LineSegments2, LineSegmentsGeometry, Line2NodeMaterial })`.
- **Canonical async factory (R3F v9 awaits the returned promise; casts required under `strict`):**
  ```tsx
  import * as THREE from 'three/webgpu'
  <Canvas flat gl={async (props) => { const r = new THREE.WebGPURenderer(props as any); await r.init(); return r }}>
  ```
- **Colour fidelity (`flat`).** R3F `<Canvas>` defaults `flat={false}` → `ACESFilmicToneMapping` + sRGB, and materials default `toneMapped=true`, which desaturates the intentional data-viz palette (`#4a9eff` edges, `#ff6b6b` verts, `#00ff88`/`#ffaa00` arrows). Set `<Canvas flat>` (`NoToneMapping`) — and/or `material.toneMapped=false` — to preserve exact colours. Applies under WebGPURenderer too.
- **OrbitControls replaces `viewRotation.ts` entirely.** Polar mapping verified correct: `viewRotation.ts:132-134` gives `phi = π/2 − rotationX`, so `rotationX∈[−π/2,π/2] ⇔ phi∈[0,π]`; `minPolarAngle=0`/`maxPolarAngle=π` (OrbitControls defaults) reproduce the old clamp faithfully. Cite the old module so the intent survives.
- **Edges — WebGPU-safe fat lines (primary).** `LineSegments2`/`LineMaterial` are **WebGL-only** (three.js docs: *"can only be used with WebGLRenderer… use `Line2NodeMaterial`"*). Use the WebGPU variants:
  - `import { LineSegments2 } from 'three/addons/lines/webgpu/LineSegments2.js'`
  - `import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js'`
  - material `Line2NodeMaterial` from `three/webgpu`; all three passed to `extend(...)`.
  - One batched geometry = one draw call (fat blue lines, parity with today's `lineWidth: 3`). **Verify at runtime — no first-party WebGPU example** (and confirm `three/addons/lines/webgpu/*` resolves under `Bun.build`; fold into Gate-0).
- **Edges — confirmed fallback.** `<lineSegments>` (`THREE.LineSegments`) + `<lineBasicNodeMaterial>` (1px). Note even `LineBasicMaterial` is WebGPU-incompatible (three.js #29526), so the fallback material must be `LineBasicNodeMaterial`, which exists in `three/webgpu`. **Do not use drei `<Line>`** — it wraps `LineMaterial` (WebGL-only), so it is ~certainly broken under WebGPU, not merely uncertain.
- **Arrows become real 3D geometry**, which **dissolves the entire "gradient arrow whips at the vanishing point" bug class**: three's camera does perspective foreshortening for free, so `projectArrow`'s hand-rolled orthographic-direction hack (and the commits spent taming it) are no longer needed.

## 7. Deleted / untouched

**Delete:**
- `src/viewRotation.ts` — three's camera + OrbitControls own projection now (its own migration note says to delete it).
- `test/viewRotation.test.ts`, `test/gradientArrow.test.ts` — they test the hand-rolled projection/arrow math that no longer exists.
- the Canvas2D drawing + manual mouse/wheel/rotation state in `src/index.tsx`.

**Untouched invariant (the hard line):** the **math body** of `core/tangentPointEnergy.ts` is byte-for-byte the current `src/tangentPointEnergy.ts` — no numeric or op-order change. The file's **header comment** (`src/tangentPointEnergy.ts:4-5`, "imported by src/index.tsx and test_gradient.ts") *will* be updated to name the new consumers; that is the one permitted edit and it satisfies CLAUDE.md's "anchor the magic" (an accurate anchor beats a byte-frozen stale one). `test/golden.test.ts` + `test/bench.ts` keep **bit-identical** expected values — a pure file move cannot change IEEE-754 results; only import path strings move. If any golden value shifts, the migration is wrong — stop.

**Move collateral (all importers of the moved files — verified by grep):**
- `test/golden.test.ts`, `test/bench.ts`: `../src/tangentPointEnergy` → `../src/core/tangentPointEnergy`, `../src/testConfigs` → `../src/core/testConfigs`.
- `test_gradient.ts` (repo **root**, a `knip.json` entry): `./src/tangentPointEnergy` → `./src/core/tangentPointEnergy`, `./src/testConfigs` → `./src/core/testConfigs`. Easy to miss — it is outside `test/`.
- `knip.json` needs **no change** (its entry paths — `server.ts`, `src/index.tsx`, `test_gradient.ts`, `test/**` — are all stable); re-run `knip` after the move to confirm green.

**Add:**
- `test/optimizer.test.ts` — deterministic unit test: one `step()` on a known config equals a manual `v − stepSize·grad`, and `energy` is `Object.is`-equal to a hardcoded expected value (not the weaker "energy decreases," which isn't guaranteed for an arbitrary config/step).

## 8. Feature-parity checklist

All must survive: 7 presets (crossing, helix, linked-rings, knot, stress, random, chain) · per-preset param sliders + Regenerate · Start/Stop/Reset (**Reset also re-centres the camera via `controls.reset()`**) · analytical ↔ finiteDiff toggle · log-scale step-size slider · stats (step/energy/vertices/edges/**zoom**) · **gradient-mode text label** ("Analytical (green)"/"Finite Diff (orange)") · **energy shown before Start** (computed on rebuild) · orbit-drag + wheel-zoom · gradient arrows shown when paused, coloured green/orange, **tiny-gradient arrows kept visible via the world-space min clamp**, `‖g‖≤1e-6` skipped · localStorage persistence of `{testId, params}`.

## 9. Risks & sequencing

**Gate-0 spike (first step): does `three/webgpu` build AND iterate fast enough under `Bun.build`?**
`three/webgpu` uses top-level await + `import.meta`; Bun's `target:'browser'` is ESM so TLA *should* work, but this is **unconfirmed end-to-end**, and `server.ts:23-27` runs `Bun.build({ minify:false })` **per request with no cache** — re-bundling all of three/webgpu on every `.tsx` reload is a plausible multi-second DX hit. Gate-0 success criteria are therefore **both**: (a) a minimal `<Canvas flat>` + spinning cube (NodeMaterial) boots in the browser under WebGPU; (b) reload iteration is acceptable — else add a build cache to `server.ts` **or** trip the fallback. Also confirm `three/addons/lines/webgpu/*` resolves under Bun.
- **Policy (per user):** *verify and flag, but do not stop.* If Bun fails either criterion, note it and **fall back to pnpm + Vite** (first-class three/R3F/WebGPU support; Vite dep-pre-bundling caches three, fixing the reload cost) rather than blocking. The four-area architecture is bundler-independent, so a swap does not change §4. (A Vite fallback also makes §11's React Compiler ~3 lines.)

**Second risk — drei `<Line>` under WebGPU:** ~certainly broken (wraps WebGL-only `LineMaterial`). Mitigation is already the primary/fallback line design in §6; do not rely on drei `<Line>`.

**WebGPU-unavailable:** `renderer.init()` rejects on browsers without WebGPU / non-secure contexts. localhost dev is fine; no graceful degradation is in scope, but the failure mode should be surfaced (an error boundary / message), not a blank canvas.

**Mid-migration the app is dark.** After the core move (M2), the still-present `src/index.tsx` imports dead paths until M5 rewrites it, so `bun run dev` 500s from M2→M5 (tests stay green throughout — golden/bench/optimizer). Either accept the dark window or move+rewrite `index.tsx`/`App.tsx` together; note it in the plan.

**Rough milestone order** (details → implementation plan):
1. Gate-0 spike (deps + WebGPU cube boots & iterates under Bun; else pnpm+Vite).
2. `core/` move + `optimizer.ts` + test path fixes (incl. `test_gradient.ts`) + `optimizer.test.ts` (core green, golden bit-identical, knip green).
3. `store.ts` (config + graph + persistence + live buffer + commit).
4. `scene/` (Viewer + Curve + GradientArrows) at parity — **riskiest milestone; warrants its own review checkpoint / possible sub-plan.**
5. `ui/` (ControlPanel + Stats) + `App.tsx` wiring; rewrite `index.tsx` mount (with M2's dark window closed here).
6. Delete `viewRotation.ts` + its two tests; final parity verification.

## 10. Citations (researched + verified 2026-07-02)

- R3F Canvas / async `gl` / `extend` + `ThreeToJSXElements`: https://r3f.docs.pmnd.rs/api/canvas · v9 migration: https://r3f.docs.pmnd.rs/tutorials/v9-migration-guide
- WebGL-only line classes: https://threejs.org/docs/pages/LineSegments2.html · https://threejs.org/docs/pages/LineMaterial.html · `LineBasicMaterial` incompat: three.js issue #29526
- Mixed WebGL-material-under-WebGPU incompatibility: https://github.com/pmndrs/react-three-fiber/discussions/3043
- WebGPU init warning (resolved by awaited `init()`): https://github.com/pmndrs/react-three-fiber/issues/3403
- three TSL / compute (future milestone reference only): https://threejs.org/docs/pages/TSL.html
- Bun build config (ESM/TLA): https://bun.com/reference/bun/BuildConfig
- React Compiler 1.0 (2025-10-07) install + Vite/Babel integration: https://react.dev/learn/react-compiler/installation

## 11. Follow-up milestone — React Compiler (NOT in this migration)

React Compiler 1.0 (released 2025-10-07) ships only as `babel-plugin-react-compiler` and **requires a Babel pass**. Kept separate from this migration for three reasons: (1) it is a build-pipeline change and the pipeline is exactly what Gate-0 may swap (Bun → Vite); deciding it before the bundler is a dependency inversion; (2) its payoff here is small by construction — §5 keeps the hot path *out* of React (the compiler memoizes render output, not `useFrame` mutation), so it only tidies the infrequent control-panel renders; (3) keeping it out preserves a clean "did the switch work?" signal.

**Enablement path (post-Gate-0):**
- **If on Vite (fallback taken):** ~3 lines — `@vitejs/plugin-react` with `babel: { plugins: ['babel-plugin-react-compiler'] }`, or `@rolldown/plugin-babel`.
- **If still on Bun.build:** needs a Babel stage (a `Bun.build` plugin invoking `@babel/core`, or a prebuild pass) — larger, and its own reason to keep this separate.
- Add the React Compiler ESLint rule when enabling. Because §3 mandates compiler-clean components from the start, this milestone is config + verification, not a component rewrite.
