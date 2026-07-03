# Pin-drag UI (point-constraint picking) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the interactive point-constraint milestone deferred by
`docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md` §5.3 ("interactive
vertex pinning/dragging is a separate future milestone") and queued as briefing §5B.
The `pointBlock` machinery + goldens shipped in constraints-M2; this milestone is
**wiring + scene interaction + UI only, zero new math, zero oracle work**:

- Store `pins: PinConstraint[]` (`{vertexIndex, target, enabled}`) with the frozen-target
  lifecycle mirroring the `sobolevEll0` anchor (`src/store.ts`), and per-pin actions.
- `dispatchDescentStep` appends one `pointBlock(vertexIndex, target)` per enabled,
  in-range pin — mirroring the `lengthMode` → length-block pattern already in the dispatch.
- Raycast vertex picking + camera-plane drag in `src/scene/` using R3F pointer events.
- A per-pin toggle UI (enable / remove) in `src/ui/ControlPanel.tsx`, existing style.
- Store/flow tests in the `test/store-constraints-m2.test.ts` style. NO goldens, NO oracle,
  NO tolerance/linsolve changes (`pointBlock` is already golden-verified in M2).

**Architecture:** `pointBlock` (constraintSet.ts) is already generic end-to-end through
`sobolevStepSet`; the only core-adjacent change is store state + a dispatch append. The
scene work is a new self-contained `src/scene/PinControls.tsx` (an invisible raycast overlay
mesh + drag handlers + visible pin markers) plus three additive lines in `Viewer.tsx`
(pass `pins`, `makeDefault` on OrbitControls, render `<PinControls/>`). No existing exported
signature changes; `dispatchDescentStep` gains an OPTIONAL `pins?` arg (same prescribed-extension
pattern as M1/M2 added `lengthMode?`/`sobolevEll0?`).

**Tech Stack:** TypeScript (strict) + bun test; zustand store; React 19 + R3F v9
(`@react-three/fiber@9.6.1`) + drei (`@react-three/drei@10.7.7`) over `three/webgpu@0.185`.
R3F pointer-event + drei controls APIs were verified via context7 this session (see
"API verification" below) — not written from memory.

**Branch/commit discipline (overrides the generic skill's per-task commits):** branch
`feat/pin-drag-ui`. The plan doc gets its own `docs:` commit. ONE `feat` commit for the whole
milestone, message starting `feat(pins): interactive point-constraint pin-drag UI`, iterated
with `--amend` (CLAUDE.md one-commit-per-plan rule). Do NOT push. No task commits before Task 4.

---

## Decisions (record every micro-decision here — CLAUDE.md subagent-briefing rule)

- **D1 — base-branch reconciliation (FLAGGED BLOCKER).** The provisioned worktree was
  checked out at `827e303` (pre-sobolev; `src/` had only `index.tsx`+`testConfigs.ts`), which
  contains NONE of the M2 machinery / store / scene this milestone builds on. All required
  reading (store frozen-target lifecycle, `pointBlock`, ControlPanel, `test/store-constraints-m2.test.ts`)
  exists only on `feat/sobolev-solver-perf` (tip `affa17e`), which the session-start git snapshot
  named as the current branch. Smallest reasonable decision: base `feat/pin-drag-ui` on `affa17e`
  (`git checkout -b feat/pin-drag-ui affa17e`), not the stale worktree HEAD. Baseline after
  switch: `bun test` = 191 pass / 0 fail across 22 files.

- **D2 — pin data shape.** `pins: PinConstraint[]`, `PinConstraint = {vertexIndex: number;
  target: Vec3; enabled: boolean}`. `target` is a world-space point in the same coordinates as
  the `live` buffer (so `pointBlock`'s Φ = γ_i − target is coordinate-consistent). `enabled`
  is the per-pin toggle (§5B "add/remove/enable pins"). Array is small (a handful of pins);
  no map/keying needed — `vertexIndex` is the identity (one pin per vertex, deduped on add).

- **D3 — drag interaction model (the running-descent interaction, §5B explicit ask).** Pointer
  DOWN on a vertex sphere → if that vertex is not yet pinned, create a pin at its CURRENT live
  position (so the pin initially just holds the vertex where it is), then begin dragging that
  pin. Pointer MOVE → set BOTH `pins[i].target` and `live[vertexIndex]` to the current
  ray∩drag-plane point. Pointer UP → end drag (pin persists). Rationale for moving `live` too
  (not target-only): it gives immediate direct-manipulation feedback whether or not descent is
  running, AND it keeps `target == live[vertexIndex]` so the frozen-lifecycle re-anchor (D5) is
  a no-op in normal flow. **Interaction with running descent:** the frame loop reads `pins` and
  builds `pointBlock(i, target)`; each accepted step projects vertex i onto `target` (≤ 1e-4
  reference tol) and relaxes the rest of the curve around the held vertex. So: drag moves the
  target; descent holds the pinned vertex there and untangles everything else. Sub-frame the
  pointer handler's direct `live[i]` write makes the grabbed vertex track the cursor 1:1; the
  next descent step reasserts it (redundant while running, necessary while paused).

- **D4 — drag plane.** A camera-facing plane through the picked vertex, frozen at grab time:
  normal = `e.camera.getWorldDirection()`, point = the vertex's live position
  (`THREE.Plane.setFromNormalAndCoplanarPoint`). Each pointermove intersects the current mouse
  ray (`e.ray.intersectPlane`) with it. Camera-facing ⇒ the ray (from the camera) is never
  parallel to the plane ⇒ intersection always exists. Simplest standard R3F drag; no unproject
  math by hand.

- **D5 — frozen-target lifecycle (mirror `sobolevEll0` exactly, spec §3.5).** The frame loop
  READS `pins` and NEVER writes it (identical invariant to `sobolevEll0`: "a target that tracks
  the current iterate makes its constraint vacuous"). Pin targets change only via user actions
  (add = snapshot live; drag = ray∩plane). At the three `sobolevEll0` re-anchor boundaries:
  (1) **rebuild** (setPreset/regenerate/reset) — CLEAR pins (`pins: []`): topology/indices change,
  so old vertex indices are meaningless in the new graph — the faithful mirror of "recompute
  from the NEW graph" when the old data cannot survive; (2) **play** (setRunning true) and
  (3) **commit** (setRunning false) — re-anchor each IN-RANGE pin's `target := live[vertexIndex]`
  (mirrors `sobolevEll0 = edgeLengths(live, …)`; because drag keeps target == live this is a
  no-op except for absorbing ≤1e-4 projection drift, the spec §3.5 "accepted drift across pause
  cycles"). Out-of-range pins (should not occur — rebuild clears) are dropped by the re-anchor.

- **D6 — dispatch order + robustness.** `dispatchDescentStep` appends pointBlocks AFTER the
  length block: row order `[barycenter?, length?, ...enabled∧in-range pointBlocks in array order]`.
  Only `enabled && 0 ≤ vertexIndex < vertices.length` pins produce a block (a disabled or stale
  pin must never break the descent). `assertValidConstraintSet` is still called (pins add no
  mutual-exclusion; spec §3.4 permits pin + length — infeasible target combos surface as the
  existing `projection_failed` rejection, never a throw).

- **D7 — OrbitControls suppression during drag.** `<OrbitControls makeDefault />` (verified drei
  API) registers the instance in R3F state; PinControls reads it via `useThree(s => s.controls)`
  and sets `controls.enabled = false` on drag start / `true` on end (the documented
  TransformControls pattern), plus `e.stopPropagation()` + `e.target.setPointerCapture(e.pointerId)`
  so the grab beats sibling handlers and the drag continues off the sphere. Accepted minor
  limitation: OrbitControls listens on the DOM element directly, so a single pointerdown frame
  could register before `enabled=false` takes effect — cosmetically negligible for a dev tool
  (headless SwiftShader is our only display here anyway; not app-verifiable interactively).

- **D8 — pin visuals + hover affordance (mine to pick).** Pick target = an InstancedMesh of
  invisible spheres (radius 0.12 > the 0.06 visual vertex, easier grabbing; material
  `transparent opacity=0 depthWrite=false`, so `object.visible` stays true → still raycastable;
  `computeBoundingSphere()` refreshed each frame so moved instances still hit). Visible pin
  markers: one small sphere per pin at its `target`, gold `#ffd700` when enabled, dim
  `#6a6a6a` when disabled, radius 0.09. Hover: set `gl.domElement.style.cursor` to `grab`
  on over / `grabbing` while dragging / `''` on out.

- **D9 — no-op in raw mode.** Pins only feed the sobolev ConstraintSet. In raw descent mode the
  dispatch's raw branch ignores pins entirely (unchanged). Picking/markers still work (they are
  descent-mode-agnostic scene affordances); the ControlPanel pin list notes pins affect sobolev.

- **D10 — abnormal drag termination + hover-accurate end cursor (review findings 1–2).**
  Review found that `endDrag` on `onPointerUp` alone strands `controls.enabled = false` (and a
  `grabbing` cursor) on abnormal termination — mid-drag remount via the `pins-${graphVersion}`
  key, browser `pointercancel`/`lostpointercapture` — and that the unconditional `'grab'` at
  drag end can show with nothing hovered. Fix, grounded in the INSTALLED
  `@react-three/fiber@9.6.1` events source (`dist/events-b389eeca.esm.js`): (a) this R3F version
  NEVER dispatches the object-level `onPointerCancel`/`onLostPointerCapture` props — its canvas
  handlers for both (`handlePointer` cancelation switch, ~lines 817–842) only call
  `cancelPointer([])`, which fires **`onPointerOut`** on every hovered object (~787–810); and
  during a capture the dragged instance is pinned into the hover/intersections set via the
  captured intersection (~657–662), so it can only go "out" through a cancel. Therefore the
  abnormal hook is `onPointerOut` while dragging, filtered to `e.instanceId === dragIndex`
  (outs for OTHER instances the cursor crosses mid-drag are ordinary and must not end the
  drag); it restores `controls.enabled`/cursor WITHOUT touching capture (already gone on these
  paths — releasing again can throw). (b) A `useEffect` unmount cleanup unconditionally
  restores `controls.enabled = true` + default cursor (covers the remount path; nothing else
  in the app disables the default controls). (c) `endDrag`'s `releasePointerCapture` is wrapped
  in try/catch (DOM release throws on a dead pointerId). (d) The end-of-drag cursor comes from
  a fresh raycast of the pick mesh with the release ray (R3F hover bookkeeping is stale-true
  for the captured instance): `'grab'` if hit, default otherwise — in the normal flow the
  dragged vertex sits on the release ray, so this lands on `'grab'`.

## API verification (context7, this session — do not re-derive)

- R3F v9 mesh pointer handlers `onPointerDown/Move/Up/Over/Out`, `ThreeEvent<PointerEvent>`,
  `e.stopPropagation()`, `e.target.setPointerCapture(e.pointerId)` / `releasePointerCapture`.
  Event carries Three intersection data spread onto it: `point`, `ray: THREE.Ray`,
  `camera`, and `instanceId` (InstancedMesh instance index — three.js `Intersection.instanceId`).
  (`/pmndrs/react-three-fiber`, docs/API/events.mdx.)
- drei `<OrbitControls makeDefault />` sets R3F `state.controls`; retrieve via
  `useThree((s) => s.controls)`; toggle `controls.enabled` to gate orbiting during manipulation
  (exactly what TransformControls does). (`/pmndrs/drei`, docs/controls/introduction.mdx.)
- Verified against the INSTALLED `@react-three/fiber@9.6.1` source (post-review, D10): the
  object-level `onPointerCancel`/`onLostPointerCapture` props exist in `EventHandlers` but are
  never dispatched — the canvas-level handlers for those two DOM events only run
  `cancelPointer([])`, which surfaces as `onPointerOut` on hovered objects; captured
  intersections are re-appended to every event's hit list, keeping the dragged instance
  "hovered" for the duration of the capture. (`dist/events-b389eeca.esm.js` — handlePointer
  cancelation switch, cancelPointer, intersect's capturedMap append.)

## Ground-truth references (cite these in @see, never "Task N")

- Pin math: `pointBlock` — `src/core/sobolev/constraintSet.ts` (Φ = γ_i − target, identity C,
  NaN backstop for out-of-range index). Spec §2 / §5.1.
- Frozen lifecycle anchor to mirror: `src/store.ts` `sobolevX0`/`sobolevL0`/`sobolevEll0`
  field TSDoc + the three re-anchor sites (rebuild, `setRunning(true)`, `setRunning(false)`).
  Spec §3.5.
- Dispatch pattern to mirror: `dispatchDescentStep` length-block append in `src/store.ts`.

## File structure

| File | Change |
|---|---|
| `src/store.ts` | +`PinConstraint` type, `pins` field, `addPin/removePin/setPinEnabled/setPinTarget` actions; rebuild clears pins; play/commit re-anchor in-range pin targets from live; `dispatchDescentStep` gains `pins?: PinConstraint[]` + appends `pointBlock`s (D6); import `pointBlock`. |
| `src/scene/PinControls.tsx` | NEW: invisible raycast overlay InstancedMesh + drag handlers (D3/D4/D7) + visible gold pin markers (D8). |
| `src/scene/Viewer.tsx` | `makeDefault` on OrbitControls; pass `pins` into the dispatch; render `<PinControls/>`. |
| `src/ui/ControlPanel.tsx` | Per-pin list: enable checkbox + remove button per pin; "click a vertex to pin" hint. |
| `test/store-pins.test.ts` | NEW: pin add/dedupe/remove/enable/target actions; frozen lifecycle (clear on rebuild, re-anchor on play/commit, frame-loop never mutates); dispatch appends pointBlocks bit-identically to an explicit `sobolevStepSet` set; disabled/out-of-range excluded. |

Impl-file budget: store.ts, PinControls.tsx, Viewer.tsx, ControlPanel.tsx = 4 impl files. ✓
Existing exported signatures unchanged; `dispatchDescentStep` args object gains one OPTIONAL
field (`pins?`). Zero edits to existing test files.

---

### Task 1: Store — pins state + actions + frozen lifecycle + dispatch wiring (TDD)

**Files:** Test `test/store-pins.test.ts` (new); modify `src/store.ts`.

**Contract:** existing exported signatures unchanged; `dispatchDescentStep` args gain OPTIONAL
`pins?: PinConstraint[]` only. Every existing test passes WITHOUT edits.

- [ ] **1.1 Write the failing test** `test/store-pins.test.ts` (style of `store-constraints-m2.test.ts`;
  the store is a shared singleton — restore any default the test mutates, and prefer `setPreset('crossing')`
  at the top of each test to reset topology). Cover:
  - `addPin(i)` appends `{vertexIndex:i, target: <live[i] snapshot>, enabled:true}`; a second
    `addPin(i)` does NOT duplicate (idempotent by vertexIndex). `removePin(i)` removes it.
    `setPinEnabled(i,false)` flips enabled. `setPinTarget(i,[…])` updates target only.
  - Frozen lifecycle: `setPreset`/`regenerate` clears pins; `setRunning(true)` and
    `setRunning(false)` re-anchor an in-range pin's target to `live[i]` (perturb `live[i]`
    first, assert target follows on play); mid-run mutation of `live` does NOT move
    `pins` (same array identity, mirror of the sobolevEll0 test).
  - Dispatch: with `barycenterConstraint:true, lengthMode:'total'` and one enabled pin,
    `dispatchDescentStep({…, pins})` is bit-identical (energy `toBe`, flattened vertices
    `toEqual`) to `sobolevStepSet(…, [barycenterBlock(x0), totalLengthBlock(L0), pointBlock(i,target)])`.
    A disabled pin and an out-of-range pin (vertexIndex 999) both produce NO block (result
    matches the no-pin set). Pins absent ≡ pre-existing dispatch (bit-identical to the M2 path).

- [ ] **1.2 Run RED.** `bun test test/store-pins.test.ts` — fails: `pins`/`addPin`/… and the
  `pins?` dispatch arg do not exist.

- [ ] **1.3 Implement in `src/store.ts`.** Add `PinConstraint` interface (TSDoc: frozen target,
  @see pointBlock + spec §2/§3.5). Import `pointBlock`. Add `pins: PinConstraint[]` (default `[]`)
  and the four actions with the same "clear sobolevConverged, leave stats" convention as the
  constraint toggles. In `dispatchDescentStep`: add `pins?: PinConstraint[]`; in the sobolev
  branch, after the length block, `for (const p of args.pins ?? []) if (p.enabled && p.vertexIndex
  in-range) set.push(pointBlock(p.vertexIndex, p.target));` (anchor: D6). In `rebuild`: add
  `pins: []`. In `setRunning(true)` and the `setRunning(false)` commit: re-anchor via a helper
  that maps in-range pins to `{...p, target: [live[i][0..2]]}` and drops out-of-range (anchor: D5).

- [ ] **1.4 Run GREEN.** `bun test test/store-pins.test.ts` all pass; full `bun test` (191 +
  new) all pass; `bunx tsc --noEmit` clean.

### Task 2: Scene — PinControls (picking + drag + markers) + Viewer wiring

**Files:** NEW `src/scene/PinControls.tsx`; modify `src/scene/Viewer.tsx`. Not unit-tested
(R3F interaction) — gate is `bunx tsc --noEmit` + boot check.

- [ ] **2.1 `PinControls.tsx`.** Component rendered inside `<Canvas>`. Reads `pins` (reactive)
  and vertex `count` from the store; reads `live`/`edges` via `getState()` each frame (Curve.tsx
  pattern). Renders (a) an invisible pick InstancedMesh (radius 0.12, `transparent opacity={0}
  depthWrite={false}`) whose instances track live vertices each frame + `computeBoundingSphere()`
  (D8), with `onPointerOver/Out/Down/Move/Up`; (b) one gold/dim marker sphere per pin at its
  target (D8). Drag state in refs (`dragIndex`, `dragPlane`). Handlers per D3/D4/D7 using
  `useThree(s => s.controls)` and `useThree(s => s.gl)` for the cursor. All magic constants get
  a `Why:`/`@see` anchor citing this plan's decisions + spec §5.3.

- [ ] **2.2 `Viewer.tsx`.** Add `makeDefault` to the existing `<OrbitControls>` (keeps `ref`);
  add `pins: st.pins` to the `dispatchDescentStep({…})` call (anchor: mirrors the sobolevEll0
  passthrough); render `<PinControls key={`pins-${graphVersion}`} />` alongside `<Curve/>`
  (distinct key prefix, same remount contract note as Curve/GradientArrows).

- [ ] **2.3** `bunx tsc --noEmit` clean; full `bun test` still green.

### Task 3: UI — per-pin toggle list in ControlPanel

**Files:** modify `src/ui/ControlPanel.tsx`.

- [ ] **3.1** Subscribe `pins`, `removePin`, `setPinEnabled`. Add a pins row after the run-controls
  block: for each pin, `Pin v{vertexIndex}` + enable checkbox (`setPinEnabled`) + a small remove
  button (`removePin`); when `pins` is empty show a muted hint "click a vertex to pin it". Match
  the existing label/`btn()` style; note pins apply to sobolev descent (same sobolev scoping
  comment as the constraint toggles). No behavior change when there are no pins.

- [ ] **3.2** `bunx tsc --noEmit` clean; full `bun test` green.

### Task 4: Verify + commit + report [orchestrator]

- [ ] **4.1** Full `bun test` (record final count vs 191 baseline), `bunx tsc --noEmit` clean,
  `bun run knip` (pin exports may show as used now; note anything expected). Boot check optional
  (headless WebGPU = SwiftShader; console errors environmental, per project memory).
- [ ] **4.2** ONE `feat(pins): …` commit (all impl + test files), `--amend` to iterate. Do NOT push.
- [ ] **4.3** Report per the task's "Report back" format: plan summary + decisions, files/commits,
  baseline vs final test counts + `bun test` tail, doubts/risky spots, worktree path + branch.

## Explicitly OUT of scope

New goldens / oracle work; any change to `pointBlock`, tolerances, `linsolve.ts`, or the
frozen-projection scheme; surface/tangent constraints; multi-select / box-select of vertices;
undo/redo; persisting pins to localStorage; changing any existing default (projectionMode,
constraint defaults, raw path byte-identity).
