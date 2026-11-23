# Adversarial plan review — round 1
Plan: `docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md` (branch `feat/webgpu-phase1`)
Reviewed against: spec `2026-08-13-webgpu-solver-design.md`, PREC/IMPL research docs, `bench/gpu/README.md`, `bench/gpu/spikes.ts`, `src/core/tangentPointEnergy.ts`, `optimizer.ts`, `dispatch.ts`, `Viewer.tsx`, `store.ts`, `penalties.ts`, `phaseTimings.ts`, `testConfigs.ts`, `fixtures.ts`, `PinControls.tsx`, installed `three@0.185.1` sources and `@types/three`.
Everything marked **[verified]** was executed or read this session (script `/tmp/verify.ts`; outputs quoted). No repo file was modified.

---

## Findings

### 1. [Critical] `wgslFn(gradientKernelWgsl(p))` throws at build — the source is not a single function
**Location:** Task 1 `gradientKernelWgsl`/`energyKernelWgsl` (plan lines 273-276, 382-385); Task 3 `GpuTangentPoint` constructor (`wgslFn(gradientKernelWgsl(params))`, `wgslFn(energyKernelWgsl(params))`).
**Evidence:** `gradientKernelWgsl` returns `${pairKernelWgsl(p)}` (a `struct KD {...}` + six helper `fn`s) *followed by* `fn tpGradient(...)`. three's parser is anchored at the start of the trimmed source:
`node_modules/three/src/renderers/webgpu/nodes/WGSLNodeFunction.js:4`
```
const declarationRegexp = /^[fn]*\s*([a-z_0-9]+)?\s*\(([\s\S]*?)\)\s*[\-\>]*\s*([a-z_0-9]+(?:<[\s\S]+?>)?)/i;
```
and `parse()` (`:78-135`) does `source.match(declarationRegexp)` then `throw new Error('THREE.WGSLNodeFunction: Function is not a WGSL code.')` when the match fails. A source beginning with `struct KD {` cannot match `^[fn]*\s*name?\s*\(`. Even if it matched a helper, `blockCode = source.substring(declaration[0].length)` would swallow every following function into the first one's body. Every Phase 0 `wgslFn` (`spikes.ts:167,205,424,806,901`) is a single bare function — the plan silently departs from the only pattern it verified.
**Fix:** `wgslFn(code, includes)` is the supported mechanism (`FunctionNode.js:168`, `CodeNode.js:99-113` builds each include first so it lands before the function at module scope). Change the generators to return the entry function **only** and expose the shared block separately; in `tangentPoint.ts`:
```ts
import { code, wgslFn } from 'three/tsl';
const shared = code(pairKernelWgsl(params), [], 'wgsl');
const gradFn = wgslFn(gradientKernelWgsl(params), [shared]);
const energyFn = wgslFn(energyKernelWgsl(params), [shared]);
```
Update Task 1's tests accordingly (`gradientKernelWgsl(P)` must *not* contain `struct KD`; `pairKernelWgsl` must not contain `fn tpGradient`). Add a unit test that the entry source's first non-blank token is `fn` — cheap regression guard for exactly this parser.

### 2. [Critical] WGSL pointer access-mode mismatch: `p` declared `read` but bound `read_write`; `slots` declared `read` in the gather kernel but bound `read_write`
**Location:** Task 1 `diff2/diffGuard/tpGradient/tpEnergy` (`p: ptr<storage, array<f32>, read>`), `tpVertexGather` (`slots: ptr<storage, array<f32>, read>`); Task 3 constructor (`this.positions = attributeArray(...)` with no `.toReadOnly()`, `this.slots = instancedArray(...)`).
**Evidence:** In compute stage three emits the buffer's declared access from the *node*: `WGSLNodeBuilder.js:1191-1220` (`getNodeAccess` → `node.access`; `getStorageAccess`), used at `:2155` (`storage, ${ this.getStorageAccess(bufferNode, shaderStage) }`), and the call site passes `'&' + node.build(builder)` (`FunctionCallNode.js:110-114`). `attributeArray`/`instancedArray` default to READ_WRITE — Phase 0 had to call `.toReadOnly()` on every `read` buffer (`spikes.ts:200-201,415-416`). WGSL pointer types include the access mode; `&p` of type `ptr<storage, array<f32>, read_write>` is not convertible to a `ptr<storage, array<f32>, read>` parameter → shader compile error. Same for `slots` (one node, READ_WRITE, bound into two kernels declared with different modes).
**Fix:** `this.positions = attributeArray(new Float32Array(6V), 'float').toReadOnly();` (CPU upload via `needsUpdate` does not need shader write access). For `slots`, declare `read_write` in `tpVertexGather` too (simplest), or keep two separate `storage(attr)` nodes over the same attribute with different access — the first is safer. Add to Task 3's verify-step: "every `ptr<storage,…,MODE>` parameter mode equals the bound node's access".

### 3. [Critical] Degeneracy guards use `>=` against a guard that is exactly 0 on collinear pairs → `1/0` → NaN; the `crossing` preset (in the T2/T3 fixture set) triggers it, and the plan's own D3 test fails
**Location:** D3; Task 1 `kernelDerivs` (`if (rc >= crossGuard(e, d))`, `if (rd >= dGuard)`), `tpGradient` (`if (reI >= diffGuard(...))`); Task 2 `crossDegenerate`/`diffDegenerate` and their test.
**Evidence:**
- `crossGuard` = `32·u·‖(|e_y d_z|+|e_z d_y|, …)‖`. For `e=(1,0,0)`, `d=(-2,0,0)` every product is exactly 0 → guard = 0, `rc` = 0 → `0 >= 0` is **true** → `invRc = 1.0/0.0 = inf` → `dc_dd = cross3(c,e)*inf = 0*inf = NaN` (and WGSL implementations "may assume no NaN/Inf", PREC §Q5 line 132 — behaviour is not even portable). CPU: `rc >= 1e-14` false → zeros (`tangentPointEnergy.ts:210`).
- `testConfigs.ts` `crossing` preset: vertices `[-1.5,0,0.3],[-0.5,0,0.3],[0.5,0,0.3],[1.5,0,0.3]`, edges `[0,1],[1,2],[2,3]`; `(0,1)` and `(2,3)` are disjoint and exactly collinear. T3 on `crossing` → cosine NaN → `t3pass` false → **branch stops by the plan's own rule**. In the app, `allFinite` fails every step on that preset → permanent silent CPU fallback.
- **[verified]** the plan's Task 2 mirror on the plan's own test input: `crossDegenerate([4,0,0],[2,0,0])` → `false` (test expects `true`); `crossDegenerate([1,1,0],[2,2,0])` → `true`. Same hole in `diffGuard` when both vertices are exactly at the origin (`0 >= 0`).
**Fix:** invert the comparison so a zero guard is degenerate: WGSL `if (rc > crossGuard(e, d))`, `if (rd > dGuard)`, `if (reI > diffGuard(...))`; mirrors `return hypot(c) <= GUARD * U * hypot(s)` / `Math.sqrt(n2) <= Math.sqrt(g2)`. Add `crossing` explicitly to the D3 tests ("exactly collinear axis-aligned ⇒ degenerate; NaN-free") and add a "no NaN in `gradient()` on every preset" assertion to `phase1Tolerance`.

### 4. [Critical] Task 1's `powExpr` tests do not match Task 1's `powExpr` implementation
**Location:** Task 1 Step 1 test vs Step 3 implementation.
**Evidence [verified]:** the plan's implementation yields `powExpr('x',3) === "(((x) * x) * x)"` and `powExpr('x',-2) === "(1.0 / (((x) * x)))"`; the tests assert `'((x * x) * x)'` and `'(1.0 / ((x * x)))'`. A TDD implementer following the plan verbatim hits red-after-green with no guidance on which side is authoritative; worse, the downstream `toContain` checks on generated kernels (e.g. `'let invD = 1.0 / d_eps;'`) still pass, so the mismatch could be "fixed" by loosening the tests.
**Fix:** make the implementation produce the tested strings: `let e = x; for (i=1;i<k;i++) e = \`(${e} * ${x})\`; if (k===1) e = \`(${x})\`; return n<0 ? \`(1.0 / (${e}))\` : e;` — or change the expectations. State which is canonical.

### 5. [Major] Device-loss / error hooks call the previous handlers unbound → `TypeError` inside `_onDeviceLost`
**Location:** Task 7 Step 1 (`const prevLost = renderer.onDeviceLost; … prevLost(info);` and the `onError` twin).
**Evidence:** `Renderer.js:595` `this.onDeviceLost = this._onDeviceLost;` (unbound method), and `_onDeviceLost(info)` (`:1225-1240`) ends with `this._isDeviceLost = true;`. The backend invokes `renderer.onDeviceLost(deviceLossInfo)` (`WebGPUBackend.js:273`) — a method call — but the plan's arrow wrapper then calls `prevLost(info)` with `this === undefined` → "Cannot set properties of undefined (setting '_isDeviceLost')". The renderer never records the loss, so `renderer.compute()`'s `_isDeviceLost` short-circuit (`:2720`) never engages. `_onError` happens not to use `this` today, but wrapping it the same way is fragile.
**Fix:** `prevLost.call(renderer, info)` / `prevErr.call(renderer, info)`; and record in the plan that these are **overrides of unbound methods** so the next agent doesn't "simplify" the `.call`.

### 6. [Major] `GpuDriver.step` passes `st.live` by reference across an `await`; pin-drag mutates `live` in place → dE and solve see different positions
**Location:** Task 6 `GpuDriver.step` (`buildStepArgs(st, …)` → `args.vertices === st.live`; `await this.engine.gradient(args.vertices)`; then `dispatchDescentStep({...args, dE})`).
**Evidence:** `dispatch.ts:333` `vertices: state.live` (reference). `PinControls.tsx:169-174` writes `live[i][0..2] = hit.*` on every pointer-move, including while running. The GPU upload snapshots positions at `writeHiLoInterleaved` time; after the readback resolves, `dispatchDescentStep` reads the *mutated* `live` → the saddle solve consumes a dE computed at different positions (the CPU line search's E₀/Armijo is then also internally inconsistent for that step). The worker driver is immune because `postMessage` structured-clones the vertices; the main driver is immune because it is synchronous. This is a new hazard the plan introduces, not pre-existing.
**Fix:** snapshot once in `step()`: `const vertices = st.live.map((v) => [v[0], v[1], v[2]] as Vec3); const args = { ...buildStepArgs(st, energyBefore), vertices };` and pass the same array to both `gradient()` and `dispatchDescentStep`. O(N) copy, negligible. Add a reviewer-lens note for Task 7: "no `live` reference survives an await".

### 7. [Major] D4's T2 error budget understates the error by ~5–10×; the ">10× inside T2" claim is not supported
**Location:** D4 first paragraph.
**Evidence:** The estimate counts only the sequential per-edge sum (≈√E·u ≈ 2e-6) averaged over E independent partials (→ 6e-8). It omits (a) the two-pass tree reduction itself — up to 14 rounding levels on the running total, ≈√14·u ≈ 2e-7 typical, 14u ≈ 8e-7 worst; (b) per-term arithmetic error of `f` — ~10 f32 ops (`c_eps^3`, `invD^6`, product) ≈ several u per term, which averages down but not to zero; (c) the ε placement itself adds a *systematic* 1e-10-relative bias that is invisible in f32 (fine). Realistic total ≈ 2–5e-7 against a 1e-6 gate: a 2–5× margin, not >10×. Phase 0's only production-kernel datapoint is G2 at 4.6e-6 vs a 1e-5 gate (README:191) — also ~2×. The conclusion (T2 passes) is probably right, but the plan presents a wrong derivation as pre-registered evidence.
**Fix:** rewrite the budget to include the tree term (state ≈2–5e-7 expected, 1e-6 gate, margin ~3×) and keep the blocked-sum contingency. Optionally pre-register the expected T2 range so an anomalously *good* number (e.g. 1e-9, which would indicate the comparison is vacuous) is also investigated.

### 8. [Major] Task 3 (and Task 7) are too large for a single implementer/reviewer pass
**Location:** Task 3 creates 4 files + refactors `spikes.ts` + writes 2 tests + runs the T1–T3 gates; Task 7 rewrites the frame loop's single-flight machinery plus boot hooks plus UI.
**Evidence:** CLAUDE.md "simple" test fails on every clause for Task 3; the plan's own protocol says use the orchestrator model for 1/3/7. A reviewer cannot line-check a 64-lane reduction, a wgslFn binding layer, a bench refactor and a gate run in one pass — and findings 1–3 are exactly the kind of thing that hides in an oversized task.
**Fix:** split Task 3 → 3a `compare.ts` refactor + `reduce.ts` (pure/TSL, unit-testable via a reduction-only spike on a known vector, e.g. sum of 1..n); 3b `GpuTangentPoint` + constructor test + a `trefoil(12)` per-vertex gradient dump spike against `gradientAnalytical` (the plan already names this as triage — make it a step); 3c `phase1Tolerance` + gate run + commit JSON. Split Task 7 → 7a token/generation refactor of the *existing* worker path only (`bun test` + manual worker run must be unchanged), 7b GPU branch + hooks + UI.

### 9. [Major] Spec amendment is incomplete: §6 still asserts "the GPU energy is the ε=0 energy"
**Location:** Task 5 spec amendments (only §1 and §2.3).
**Evidence:** spec `:431-433` "**ε-semantics drift:** the GPU energy is the ε=0 energy [PREC Q1]. All fixtures with intentionally-degenerate edges must go through the explicit f32 degeneracy branch tests, not rely on ε." — this contradicts the amended §2.3 after Task 5. The D2 justification itself is sound: **[verified]** `calculateEnergy` ε=0 vs ε=1e-10 on `nearTouchPair`: relErr 5.25e-7 @1e-3, 5.25e-6 @1e-4, 5.25e-5 @1e-5, **5.25e-4 @1e-6** — matches the plan to 3 digits, and PREC Q1's "inert" measurement (`:44-51`) was at N=128 O(1) distances only. So the amendment is justified; it just has to be applied consistently.
**Fix:** amend §6's bullet in the same Task 5 commit: "the GPU kernel keeps `+ε` after every norm (Phase 1 plan D2); degenerate edges are handled by the explicit f32-scale guards (D3) — ε is not relied on for degeneracy protection."

### 10. [Minor] `dE` length check throws inside `timed('step')` → `timingsBegin()` armed with no `timingsEnd()`
**Location:** Task 5 Step 2 (`if (opts.dE && opts.dE.length !== vertices.length) throw …` placed inside the `timed('step', …)` closure).
**Evidence:** `optimizer.ts:262-263` `if (collect) timingsBegin(); const outcome = timed('step', …)`; `phaseTimings.ts:69-77` — with `acc` left non-null every subsequent `timed()` in the process (including tests that expect collection off) accumulates into a stale ledger until the next `timingsBegin`. Numerics unaffected, but the "provably inert when off" invariant (`optimizer.ts:247-255`) is broken after one programmer error.
**Fix:** validate before `timingsBegin()` (first line of `sobolevStepSet`), or in `dispatchDescentStep`.

### 11. [Minor] `GpuTangentPoint.dispose()` does not free GPU buffers
**Location:** Task 3 `dispose()` — `b.value.dispose?.()`.
**Evidence:** `BufferAttribute`/`StorageBufferAttribute` have no `dispose` in r185 (`core/BufferAttribute.js` — `setUsage` at `:167`, no dispose); the optional call is a no-op, so each `graphVersion` rebuild leaks V·6·4 + E·6·4 + … bytes of VRAM (small, but unbounded over a session of preset switches). `ComputeNode.dispose()` exists (`ComputeNode.js:116`) and does clear pipelines/bindings.
**Fix:** check whether r185's backend frees a storage buffer on pipeline/binding deletion (`Bindings.deleteForCompute`); if not, keep one `GpuTangentPoint` per (V,E) shape and re-upload topology into existing buffers, or document the leak with a `@issue`.

### 12. [Minor] Plan/code name drift and leftover instructions
- Task 7 says "Loop component"; the component is `Simulation` (`Viewer.tsx:57`).
- Task 6 `driver.test.ts` carries an implementer instruction comment ("use a REAL weight field…") while already using the real `totalLength` field (`penalties.ts:33-37`); delete the comment.
- Task 7 step 3: place `dispatchGeneration.current++` **before** the `if (solverDriver === 'main') return;` early return, otherwise switching to `'main'` does not bump (harmless today only because the `.then` also checks `now.solverDriver !== 'gpu'`).
- D2 wording: on an FTZ driver `c^α / 0` yields `inf`, not "divide by zero" as a fault; the point (non-finite → fallback) stands.
- `REDUCE_LANES` is exported but unconsumed → `bun run knip` (Task 8) will flag it; either consume it in `tangentPoint.ts` for the workgroup size (`[REDUCE_LANES]` instead of literal `[64]`) or don't export.
- Task 8 negative-constraint line 2 (`git diff … | grep …`) prints but asserts nothing; pipe to a line count and compare, or drop it.
- Task 3 test imports `src/gpu/tangentPoint` which imports `three/webgpu` — **[verified]** loadable under Bun (`typeof WebGPURenderer === 'function'`), so the constructor-domain test is viable.

### 13. [Minor] Worker lifecycle effect churns the worker on every `worker ↔ gpu` toggle
**Location:** Task 7 step 3 (`if (solverDriver === 'main') return;` with deps `[solverDriver, handleWorkerMessage]`).
**Evidence:** `Viewer.tsx:207-246` — deps include `solverDriver`, so a `'gpu'→'worker'` selection tears down and recreates the worker (topology resend, pipeline-cold). Correct, but the plan's rationale ("worker alive under 'gpu'") reads as if the instance survives.
**Fix:** either accept and say so, or key the effect on a derived boolean `needsWorker = solverDriver !== 'main'`.

---

## Things checked that are correct (so the author doesn't re-litigate them)
- **D1 gather derivation, both roles, traced by hand** against `gradientAnalytical` (`tangentPointEnergy.ts:246-378`): role A (`sI·dPowCoeff·∓ê_I` to i1/i2, `+base·df_dd` to i, `∓base·df_de` to i1/i2) and role B for pair (J,I) (`sJ'·∓ê_I` to i1/i2 with `sJ' = 0.25·ℓ_J^{1−α}·ΣF'`, `−base'·df_dd` to b∈I with `base' = 0.25·ℓ_J^{1−α}·ℓ_I`) are exactly the CPU's terms landing on I's endpoints; term order (i1,j1),(i1,j2),(i2,j1),(i2,j2) mirrors `pairs` (`:308-313`); `select(i2,i1,ii==0u)` picks i1 first; `×0.5` in the edge kernel ≡ `:385-389`.
- **D2 algebra:** `coeff_d = −β·f·invD = −β·c^α·invD^{β+1} = −β·c^α/d^{β+1}` ✓; magnitudes at gap 1e-6 (c≈4e-6, c³≈6e-17, invD^6=1e36, f≈6e19, coeff_d≈4e26) inside f32; overflow floor `(1/4e-7)^6 ≈ 2.4e38 < 3.4e38` ✓; `d^{β+1}=1e-42` is subnormal (min normal 1.18e-38) ✓.
- **D3 noise-floor scale** `u²·M` for the two-float residual is right (lo ≤ u·M, its rounding ≤ u²·M; the 32× covers the two further roundings); the CPU-vs-GPU semantic difference (absolute 1e-14 vs relative ~4.5e-13·M) only matters where `f` already overflows.
- **TSL/API names** all exist in r185: `workgroupArray` (`WorkgroupInfoNode.js:232`), `invocationLocalIndex` (`IndexNode.js:157`), `workgroupId` (`ComputeBuiltinNode.js:203`), `workgroupBarrier` (`BarrierNode.js:75`), `compute(node,count,workgroupSize)` (`ComputeNode.js:291`, d.ts `NodeElements.compute(count, workgroupSize?)`), `renderer.compute([...])` accepts an array in one pass (`Renderer.js:2757-2811`, `beginCompute/finishCompute` per group), `getArrayBufferAsync(attribute)` (`:1951`), `ComputeNode.dispose` (`:116`), `resolveTimestampsAsync`/`info.compute.timestamp` (used in Phase 0), `setUsage` (`BufferAttribute.js:167`) and `Attributes.update` honours `DynamicDrawUsage`/version (`Attributes.js:103-107`), `uint(n)`/`instanceIndex` as wgslFn scalar args (Phase 0 `spikes.ts:176`), `@types/three` `onError: (errorMessage: string)` vs runtime object (`Renderer.js:1247`) — the plan's cast is warranted.
- `count`-based `compute()` auto-prepends `if (instanceIndex >= count) return;` (`ComputeNode.js:213-218`); the reduction passes dispatch exactly `groups·64` invocations so no lane early-returns before a barrier ✓.
- **dE seam tests:** zero `dE` ⇒ RHS zero (gradient solve takes only the Jacobian, `gradient.ts:101-103`) ⇒ `gTilde = 0` ⇒ `gradientL2Norm < 1e-4` ⇒ `converged`, input echoed (`optimizer.ts:354-366`) ✓; `toEqual` parity with `collectTimings: false` ✓; `PenaltyConfig.totalLength` is a real field ✓; `withSeed` restore check is sound.
- **Spec conformance:** Phase 1 energy-source policy, one upload + one readback per step, served-config predicate, boot G2 through the production kernel, `'gpu'→'worker'` chain, hardware-adapter INVALID rule, readback-inclusive perf timing — all present.

---

## Verdict: **REJECT** (revise and re-review; the fixes are mechanical but findings 1–4 each block Task 1 or Task 3 outright)

Top 3 for the author:
1. Fix the wgslFn/`includes` structure (#1) and the pointer access modes (#2) together, and add a Task 3 step that compiles both kernels on the hardware adapter *before* any gate — a "kernels compile + `trefoil(12)` per-vertex gradient matches `gradientAnalytical` to 1e-5" spike. That single spike would have caught #1, #2 and #3.
2. Flip the guard comparisons to strict `>` (degenerate iff `≤ guard`) in WGSL **and** the mirrors, and put `crossing` in the D3 tests (#3); fix `powExpr` vs its tests (#4).
3. Bind the previous device-loss handlers with `.call(renderer, …)` (#5) and snapshot `live` inside `GpuDriver.step` (#6); then split Tasks 3 and 7 (#8) and finish the spec amendment in §6 (#9).
