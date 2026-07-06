# Off-main-thread solver (worker driver) — implementation spec

**Milestone:** move `dispatchDescentStep` execution into a Web Worker so the
render thread never blocks on a descent step. **Goal: interaction smoothness**
(orbit / pin-drag / UI at display refresh regardless of step cost). **Non-goal:
solver speed** — steps/second stays ≈ today's (the curve's convergence rate per
wall-second is explicitly NOT expected to improve; do not "optimize" the solver
in this milestone).

**Branch:** implementer works on `feat/worker-solver-impl` created from the tip
of `feat/worker-solver` (this spec's commit). Orchestrator merges back
`--ff-only` after review.

This doc is THE spec that code comments cite (cite as
`docs/superpowers/plans/2026-07-04-worker-solver.md §Dn` — never task numbers).

## §1 Measured motivation (do not re-derive)

The step runs synchronously inside `useFrame` (`src/scene/Viewer.tsx`,
`Simulation()`): frame time = step + render. Measured full-step medians on this
machine (bench ledger `bench/results/2026-07-03-frozen-reuse.json`, frozen
mode): N=60-total ≈ 19.6 ms, N=120-total ≈ 47.4 ms, N=120-perEdge ≈ 71.2 ms →
12–40 fps with the main thread blocked for most of each frame. The scene render
itself is trivial at |V| ≤ ~300. Alternatives (Barnes–Hut, multigrid, LDLᵀ)
were measured/assessed 2026-07-04 and give ≤1.2–1.5× at these sizes — they do
not fix jank; this does. @see local_files/2026-07-04-next-steps-after-worker.md

## §2 Architecture: stateless compute-server worker

The worker holds NO authoritative state. The store (main thread) remains the
single source of truth. Per step, the main thread sends the exact argument set
`dispatchDescentStep` receives today; the worker runs it and posts back the
result. Because the same pure function runs on the same inputs, per-step
results are **bit-identical** to the main-thread path (gate: §T2 test).

All current sequencing semantics survive verbatim on the main thread:

- **E₀ reuse cache** (`lastEnergy` ref) and its `!running`-boundary nulling —
  @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 4).
- **penaltyEpoch invalidation** checked before each SEND (was: before each
  dispatch) — @see docs/superpowers/plans/2026-07-03-sobolev-penalties.md §2.4.
- **advanceLengthSchedule() on accepted steps only** — the round-trip per step
  guarantees the next step sees the advanced targets, same as today.
- **Copy-then-discard in-place mutation of `st.live`** (Curve's non-reactive
  buffer identity) — @see docs/superpowers/specs/2026-07-02-react-three-webgpu-switch-design.md §5.

## §3 Decisions (binding)

- **D1 — at most ONE in-flight step.** Main thread sends a step request only
  when none is outstanding and `running` is true. No queueing, no pipelining.
  Why: preserves today's step-serialized semantics (schedule advance, E₀
  chaining, pin updates between steps) with zero drift.
- **D2 — extraction, not duplication.** `dispatchDescentStep`, its args/result
  types, and the store-local types it needs (`Mode`, `DescentMode`,
  `LengthMode`, `PinConstraint`, `DescentStepOutcome`) move from `src/store.ts`
  to a new `src/core/dispatch.ts`. `src/store.ts` RE-EXPORTS all of them so the
  five existing test imports and `Viewer.tsx` keep working unchanged. Direction
  store→core only; `src/core/**` must never import from `src/store.ts` or any
  React/zustand module (worker bundle purity).
- **D3 — worker URL is a dev-server path string.**
  `new Worker('/src/worker/solverWorker.ts', { type: 'module' })`. The dev
  server bundles ANY requested `.ts` path on the fly (`server.ts` — the
  `path.endsWith('.ts')` branch), so this needs no bundler config. Do NOT use
  `new URL(..., import.meta.url)` — `Bun.build` does not emit worker chunks for
  browser targets; the plain string is load-bearing. Leave a comment citing
  this § and `server.ts`.
- **D4 — topology is sent once, not per step.** On worker init and on every
  `graphVersion` change, main sends a `topology` message with `graphVersion`
  and `edges`; the worker recomputes `disjointPairs` via
  `calculateDisjointPairs(edges)` (same deterministic function ⇒ identical
  arrays; O(E²) once per topology is negligible). Per-step messages then carry
  only vertices + dynamic config (~4 KB at N=120) — never the O(E²)
  disjointPairs.
- **D5 — every step request/result is tagged with `graphVersion`.** On result
  arrival, DROP it (with a `console.warn`) if `graphVersion` mismatches the
  current store value OR `running` is false. Why: a preset rebuild or pause can
  land mid-flight; applying a stale result would mutate committed/foreign
  buffers. Dropping is safe because the E₀ cache is nulled at the same
  boundaries (§2).
- **D6 — `solverDriver: 'worker' | 'main'` store field, default `'worker'`.**
  The `'main'` path is today's exact synchronous code, kept intact (fallback,
  A/B, and the only path store tests exercise). Auto-fallback to `'main'` with
  `console.error` if `Worker` construction fails or the worker posts an
  `error` message. UI: a small select in ControlPanel next to the
  projectionMode control, same styling pattern.
- **D7 — param assembly is a shared pure helper.** Extract the argument-object
  construction currently inlined in `Simulation` (Viewer.tsx `useFrame`) into
  `buildStepArgs(state, energyBefore)` in `src/core/dispatch.ts`, taking a
  narrow structural interface (only the fields it reads — descentMode, live,
  graph.edges, disjointPairs, mode, stepSize, sobolevX0, sobolevL0,
  barycenterConstraint, lengthMode, sobolevEll0, pins, projectionMode,
  penalties), NOT the full `SimStore` type. BOTH drivers call it — this is what
  prevents worker/main param drift. `collectTimings: true` stays hardcoded at
  the call sites as today.
- **D8 — result application is one function used by both drivers.** The
  accepted/rejected handling in `Simulation` (in-place `live` mutation, E₀
  caching, schedule advance, throttled stats publish at ~10 Hz via elapsed-time
  accumulation, auto-pause on `!accepted`) moves to a single
  `applyStepOutcome(...)` helper inside `Viewer.tsx` (it touches the store and
  refs — it is NOT core code). The worker driver calls it from `onmessage`;
  the main driver calls it inline as today. Behavior must match the current
  Simulation branch-for-branch (Viewer.tsx:126–176 at spec time).
- **D9 — pins/penalty changes mid-run need no special handling.** They reach
  the solver on the next SEND (one step of latency) — identical in substance to
  today's next-frame pickup. Do not add extra synchronization.
- **D10 — no SharedArrayBuffer.** postMessage structured clone only. Payloads
  are ~4 KB/step; SAB would require COOP/COEP headers on `server.ts` for zero
  measurable benefit at our sizes. (Revisit only in a future GPU/scale
  milestone.)
- **D11 — worker import surface.** `src/worker/solverWorker.ts` imports ONLY
  from `src/core/**`. It contains: topology cache
  (`{graphVersion, edges, disjointPairs}`), a message switch
  (`topology` | `step`), a try/catch posting `{type:'error', message}` on
  throw. Nothing else. Target ≤ ~80 lines.
- **D12 — message protocol types live in `src/core/dispatch.ts`** (exported,
  TSDoc'd, used by both worker and Viewer):

```ts
/** Main→worker. @see docs/superpowers/plans/2026-07-04-worker-solver.md §D4/§D12 */
export type SolverWorkerRequest =
    | { type: 'topology'; graphVersion: number; edges: Edge[] }
    | { type: 'step'; graphVersion: number; args: DispatchStepArgs };
    // DispatchStepArgs = dispatchDescentStep's args MINUS edges/disjointPairs
    // (supplied worker-side from the topology cache, §D4).

/** Worker→main. @see …worker-solver.md §D5/§D12 */
export type SolverWorkerResponse =
    | { type: 'result'; graphVersion: number; result: DescentStepOutcome }
    | { type: 'error'; message: string };
```

## §4 Tasks (one implementer, one commit per task, iterate via `--amend`)

**T1 — extract dispatch to core (mechanical, bit-identical).**
Move per §D2 (function + listed types + `buildStepArgs` per §D7, protocol
types per §D12); store re-exports; Viewer switched to build args via
`buildStepArgs` but still calling `dispatchDescentStep` synchronously
(behavior unchanged). All TSDoc moves with the code; add `@see` to this spec
on the new file header. Gate: `bun test` 219/0 unchanged, `bunx tsc --noEmit`
clean, no golden touched. Commit:
`refactor(core): extract dispatchDescentStep + step-arg assembly to core/dispatch (worker prep)`.

**T2 — worker + round-trip bit-identity test.**
`src/worker/solverWorker.ts` per §D11. New `test/worker-solver.test.ts`: spawn
the real worker under bun (`new Worker(new URL('../src/worker/solverWorker.ts',
import.meta.url).href)` — Bun runs TS module workers natively; this URL form is
fine in tests, §D3 applies to the BROWSER call site only), send topology + a
step on a small fixture (reuse a config from `src/core/testConfigs.ts`, both
`raw` and `sobolev` descentModes), assert the result deep-equals the
synchronous `dispatchDescentStep` output — exact equality on numbers (bit
identity), not approximate. Also assert the §D5 mismatch-drop contract at the
protocol level (worker echoes the request's graphVersion untouched). Gate:
tests green incl. new file. Commit:
`feat(worker): stateless solver worker + bit-identity round-trip tests`.

**T3 — async driver in Simulation + store toggle + UI.**
Rework `Simulation` per §D1/§D5/§D6/§D8: `useFrame` keeps camera/zoom
publishing and (worker mode) only "send if idle && running"; `onmessage`
applies results via the shared `applyStepOutcome`. Store gets `solverDriver`
(+ setter, default `'worker'`, auto-fallback per §D6); ControlPanel gets the
select. Add store test for the toggle + fallback default in the existing
store-test style. Gate: full suite green, tsc clean; manual smoke (§6).
Commit: `feat(worker): async worker driver in frame loop + solverDriver toggle`.

## §5 Hard gates (never relax)

- `src/core/sobolev/**`, `src/core/tangentPointEnergy.ts`,
  `src/core/optimizer.ts`, `oracle/**`, `test/golden.json`, all goldens:
  **untouched**. (T1 moves code INTO `src/core/dispatch.ts` — a new file — and
  edits only `src/store.ts` + `src/scene/Viewer.tsx` around it.)
- No new dependency, no version bump, no lockfile churn.
- Every exported symbol carries TSDoc with `@see` per CLAUDE.md; non-obvious
  runtime behavior (D3's string URL, D5's drop rule, D1's single-flight)
  gets an inline comment citing this spec's §.
- Existing tests must pass UNMODIFIED except imports if strictly forced (they
  should not be — §D2 re-exports exist precisely for this).

## §6 Acceptance / verification

1. `bun test` green (219 + new), `bunx tsc --noEmit` clean, at every task
   commit.
2. T2's bit-identity assertions are the correctness contract — do not weaken
   to approximate equality; if exact equality fails, the drivers diverged and
   that IS the bug.
3. Manual smoke (orchestrator, after merge): `bun run dev`, largest preset,
   run solver, orbit + drag a pin mid-run → camera stays fluid while steps
   land at their own cadence; toggle `solverDriver` to `'main'` → today's
   jank returns (proves the A/B). Headless note: SwiftShader/WebGPU console
   errors in headless environments are environmental, not app bugs.

## §7 Orchestration protocol (for the overseeing session)

- ONE Opus implementer subagent for T1–T3 (they are coupled through
  store/Viewer/dispatch — do not parallelize), in a git worktree, branch
  `feat/worker-solver-impl` from this spec's commit. **Worktree provisioner
  hands out stale HEADs** — the implementer prompt MUST open with: verify
  `git log --oneline -1` matches the expected sha, else fetch/branch from it.
- ONE Fable reviewer subagent after implementation: spec-compliance +
  code-review in a single pass (checklist: every §D decision, §5 gates, TSDoc
  coverage, test quality incl. that T2 asserts EXACT equality, behavior parity
  of §D8 against Viewer.tsx:126–176). Findings route back to the SAME
  implementer to fix via `git commit --amend` per task; re-review deltas only.
- Paste CLAUDE.md's "Subagent briefing" block verbatim into every agent prompt.
- Environment: `bun run typecheck` does NOT exist — use `bunx tsc --noEmit`.
  lefthook/biome rewrites files on commit (JSON array wrapping, exponent
  case) — diff semantically. knip flags ~14 PRE-EXISTING unused exports — not
  yours, leave them.

## §8 Decision log (implementer appends here)

- `worker.onerror` wired to the §D6 fallback in addition to D6's two listed
  triggers (ctor throw, posted `error`): a worker that fails to LOAD fires
  `onerror` without a ctor throw and can never post a protocol error, so without
  this `inFlight` would stick and the worker driver would stall. Reviewer signed
  off.
- `SolverDriver` type lives in `src/store.ts`, not `src/core/dispatch.ts` — it is
  a main-thread-only concern (worker vs main driver), and keeping it out of
  `src/core/**` preserves worker-bundle purity (§D2). Reviewer signed off.

## §D13 Arrows field off-thread (2026-07-06 fix)

**Measured breach.** T3 shipped the STEP path off-thread, but headless CDP
profiling on the `stress` preset (200 vertices, worker driver, defaults)
measured a 400–733 ms contiguous main-thread longtask once per solver step
(17–18 longtasks totalling ~8 s per 10 s run; rAF gaps up to 733 ms). CPU
profile hottest stack: `kernelDerivs ← gradientAnalytical ← (useFrame) ← update
← loop`. With the Arrows checkbox OFF: 0 longtasks, 0 rAF gaps >60 ms, rAF p95
16.7 ms. Root cause: `src/scene/GradientArrows.tsx` (default-on via
`showArrows: true`, src/store.ts) recomputed, ON the main thread,
`gradientAnalytical` (O(E²)) — plus in sobolev mode a full dense
`solveConstrainedGradient` saddle solve — every time `st.step` changed
(~10 Hz), throttled to 5 Hz. This is a blocker-class breach of the milestone
goal (interaction smoothness regardless of step cost).

**Decisions.**
- **D13-a — shared pure helper.** New `src/core/arrowField.ts` exporting
  `computeArrowField(vertices, edges, disjointPairs, mode, descentMode, x0):
  Vec3[] | null` = the field computation previously inlined in
  GradientArrows.tsx (raw `gradientAnalytical`/`gradientFiniteDiff` under
  `DEFAULTS`; sobolev `solveConstrainedGradient(...).gTilde`; `null` on the
  singular-saddle throw → caller hides arrows). Returns the RAW field, not
  negated. Imports only from `src/core/**` (worker-bundle purity, §D2).
  Bit-identical to today's field (gate: field round-trip tests).
- **D13-b — worker protocol extension.** `SolverWorkerRequest` gains
  `{ type:'field'; graphVersion; args: FieldArgs }` (`FieldArgs =
  { vertices; mode; descentMode; x0 }`); `SolverWorkerResponse` gains
  `{ type:'fieldResult'; graphVersion; field: Vec3[] | null }`. The worker's
  `field` branch restores edges + disjointPairs from the topology cache (same
  §D4 contract as `step`), calls `computeArrowField`, and echoes graphVersion
  untouched; a field before topology throws → the outer catch posts
  `{type:'error'}`.
- **D13-c — GradientArrows goes async.** A DEDICATED arrows worker (separate
  instance from Simulation's step worker so a ~500 ms field compute at N=200
  runs in parallel with descent steps instead of serializing behind them),
  constructed in a `showArrows`-gated effect (§D3 path string, null-onmessage
  teardown). useFrame keeps the key/throttle logic but POSTs a field request
  (single-flight, §D1-style) instead of computing inline; `fieldResult` applies
  the received field against the CURRENT `st.live` (≤1-step direction staleness
  accepted for a ≤5 Hz diagnostic), drops on graphVersion mismatch, and hides
  arrows on `field === null`. On worker construction failure or any worker error
  the component falls back PERMANENTLY (ref flag) to the synchronous
  `computeArrowField` — the same helper, so the paths cannot drift. Independent
  of the store's `solverDriver` toggle (that governs the STEP path only).

**Acceptance gate.** stress preset, arrows ON, worker driver → 0 main-thread
longtasks >60 ms attributable to the field compute (the `gradientAnalytical ←
GradientArrows` stack disappears from the main-thread CPU profile; it now runs
on the arrows worker thread).
