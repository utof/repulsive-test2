# AI research: WebGPU compute for the O(E²) pair kernel + iterative solve

> **Provenance:** deep-research subagent report, delivered 2026-08-13, saved
> verbatim below (only this banner added). API claims were verified by the agent
> against the installed `node_modules/three@0.185.1` source (file:line cited in
> the report) or docs/papers fetched that session. Companion doc:
> `docs/2026-08-13-ai-research-gpu-precision.md` (precision strategy — the two
> must agree on tolerance gates before any GPU milestone plan is written).
> Context: issue utof/repulsive-test2#1 (rescoped perf roadmap); target is
> realtime (>15 fps) untangling at N>1000, per-phase state of the CPU solver in
> `bench/results/2026-07-06-ldlt-ab.json`.

---

# WebGPU compute for the O(E²) tangent-point kernel + iterative solve — research report

All API claims below are verified against the **installed** `node_modules/three@0.185.1` source (file:line cited) or against a doc/paper fetched this session. No files were modified.

---

## (a) Per-question findings

### 1. TSL compute maturity + API surface (three r185.1)

TSL's compute surface is complete enough for this kernel. Verified API:

- `Fn(() => {...})().compute(count, [workgroupSize])` — default workgroup size `[64]`; `count` may be number or array (`docs/pages/TSL.html`).
- Storage buffers: `instancedArray(count, 'vec3'|'float'|'uint'|Struct)` (read-write), `attributeArray(data, ...)` (read-only). Element access `.element(indexNode)` → `StorageArrayElementNode` (`src/nodes/accessors/StorageBufferNode.js`).
- Builtins: `instanceIndex`, `invocationLocalIndex`, `workgroupId.x`, `invocationSubgroupIndex`.
- **Atomics exist** and are exported from `three/tsl`: `atomicAdd/Sub/Min/Max/And/Or/Xor/Load/Store` — `node_modules/three/src/nodes/gpgpu/AtomicFunctionNode.js:186-274`. Make a buffer atomic via `.toAtomic()` (`StorageBufferNode.js:288`) or per-struct-member `{type, atomic:true}` (`src/nodes/core/StructTypeNode.js:19-23`).
- Shared memory + sync: `workgroupArray(type, count)`, `workgroupBarrier()`, `storageBarrier()`.
- Readback: `renderer.getArrayBufferAsync(attribute, target, offset, count)` with `THREE.ReadbackBuffer` as target (`src/renderers/common/ReadbackBuffer.js:9`).
- **`computeAsync()` is deprecated since r181.** Correct pattern: `await renderer.init()` once at startup (already done at `src/scene/Viewer.tsx:380`), then synchronous `renderer.compute()`.

**Recommendation: TSL, not raw WGSL.** three owns bind-group layout, pipeline caching, and buffer lifetime; hand-rolling WGSL against `renderer.backend.device` duplicates all of it and must be re-synced on every three upgrade. TSL has a documented escape hatch for the parts where the arithmetic must be exact: `wgslFn(src, includes)` / `wgsl(src, includes)` inject native WGSL into the node graph (used in `examples/webgpu_materials.html`). That matters here because op order is a load-bearing invariant (`src/core/tangentPointEnergy.ts:26-27, 92-96`) and you don't want TSL's node compiler reassociating it. **Practical split: TSL for buffers/dispatch/plumbing, `wgslFn` for the ~200-flop pair-kernel body.**

Reference implementation to copy: **`examples/webgpu_compute_reduce.html`** in the three.js repo — official multi-pass workgroup tree reduction, exactly the shape you need.

### 2. Zero-readback rendering — works, usage flags verified

`WebGPUBackend.createStorageAttribute()` creates the buffer with `STORAGE | VERTEX | COPY_SRC | COPY_DST` — **`node_modules/three/src/renderers/webgpu/WebGPUBackend.js:2564`**. So one `StorageBufferAttribute` can be written by a compute pass and bound as a vertex attribute in the same frame, no readback, no copy.

Do **not** create a second adapter/device — `renderer.backend.device` is the device (`WebGPUBackend.js:111, 292`), and buffers cannot be shared across devices.

**Repo-specific catch:** `Line2NodeMaterial` hardcodes `attribute('instanceStart')` / `attribute('instanceEnd')` in its vertex node — **`node_modules/three/src/materials/nodes/Line2NodeMaterial.js:117-118`** — with no `positionNode` hook for them. And `src/scene/Curve.tsx:37-49` allocates a fresh `Float32Array` every frame and calls `setPositions()`, a CPU path by construction. Zero-readback means writing compute output directly into the interleaved `instanceStart`/`instanceEnd` storage buffer instead of going through `LineSegmentsGeometry.setPositions()`. Feasible given the usage flags, but it's a rewrite of `Curve.tsx`, not a tweak.

### 3. Float accumulation — no f32 atomicAdd; the workarounds differ sharply in determinism

WGSL atomics are `i32`/`u32` only. f32 atomics remain an open proposal: https://github.com/gpuweb/gpuweb/issues/4894

Three options, **not equally good for this repo**:

1. **Fixed-point `atomicAdd` on i32** (quantize float → integer). Standard workaround (https://toji.dev/webgpu-best-practices/compute-vertex-data.html). Underrated property here: integer addition is associative ⇒ **bit-reproducible regardless of thread order**, which fits `golden.test.ts` culture. Cost: must pick a scale factor and prove no overflow.
2. **CAS loop** (`atomicCompareExchangeWeak` on bitcast f32). Summation order nondeterministic ⇒ run-to-run variance. Given exact-equality gates, I'd treat this as **disqualified**.
3. **Per-workgroup partials + deterministic tree reduction, no atomics.** Deterministic if tree shape is fixed. This is what `webgpu_compute_reduce.html` does, and the literature prefers it — atomics are repeatedly flagged as a perf killer vs. tournament-tree reduction.

**Sharpest finding: you may not need atomics at all.** I traced the gradient's write targets — the inner loop scatters into all four vertices, `i1`,`i2` (edge I) *and* `j1`,`j2` (edge J) (`src/core/tangentPointEnergy.ts`, `addToGrad` calls at rel. lines 86-106 of the 270-392 block). Because `disjointPairs` contains both `(I,J)` and `(J,I)` and the total is halved (`tangentPointEnergy.ts:106-110`), the contributions landing on edge I's vertices come from exactly two sources: the I-role terms of `(I,J)`, and the J-role terms of `(J,I)` — **both loops over J with edge I held fixed**. So a kernel keyed on edge I that evaluates *both roles* accumulates only into `i1`/`i2`: a **pure gather, zero atomics, fully deterministic**. Price is ~2× flops (kernel evaluated twice per ordered pair), which on a GPU is an excellent trade against atomic contention. **Make this the default design.**

### 4. Dispatch overhead — the real budget constraint; numbers are unforgiving

From a 2026 characterization across 4 GPU vendors / 3 backends / 3 browsers — https://arxiv.org/abs/2604.02344

| Backend / browser | True per-dispatch overhead |
|---|---|
| Vulkan (Dawn, NVIDIA RTX 5090) | 23.8 µs |
| Vulkan (wgpu, range incl. AMD iGPU) | 24–36 µs |
| Chrome / D3D12 (RTX 2000 / Intel iGPU) | 58.7 / 66.5 µs |
| Safari / Metal (M2) | 31.7 µs |
| wgpu-native / Metal (M2) | 71.1 µs |
| **Firefox (all platforms)** | **~1,040 µs — rate-limited** |

Breakdown (wgpu/Vulkan, 100 sequential dispatches): **submit 12.9 µs (40% of total)**, encoder create 6.4 µs, encoder finish 6.1 µs, pass begin 3.2 µs, set pipeline 1.4 µs, set bind group 1.0 µs, **actual `dispatchWorkgroups` 0.6 µs**, pass end 0.7 µs.

**Methodology warning for your gate culture:** single-dispatch measurement overestimates by ~20× (496.8 µs measured vs 23.8 µs true) because it conflates GPU-CPU sync. Benchmark *sequential* dispatches with **one sync at the end**, or your pre-registered numbers are garbage.

**Budget math for this repo** (you're on Linux ⇒ Chrome/Vulkan ⇒ ~30 µs): a CG iteration needs ~5 dispatches (matvec, 2 dot-product reductions × 2 passes each, axpys). 50 CG iters ≈ 250 dispatches ≈ **7.5 ms of pure CPU overhead per solve** — half a 60fps frame before any arithmetic. On Chrome/D3D12 that's ~15 ms and you're over budget. **Firefox is effectively disqualified** for a dispatch-heavy per-frame loop.

**Mitigation is verified and cheap.** `Renderer.compute()` accepts an **array**: it calls `backend.beginCompute()` once, loops all dispatches into a single encoder, then `finishCompute()` submits **once** — `node_modules/three/src/renderers/common/Renderer.js:2718` (`const computeList = Array.isArray(computeNodes) ? computeNodes : [computeNodes]`, then `backend.beginCompute()` → per-node loop → `backend.finishCompute()`), and `WebGPUBackend.js:1566, 1705-1711` (`submit(this.device, groupData.cmdEncoderGPU.finish())`).

⇒ `renderer.compute([a, b, c, ...])` amortizes the dominant 12.9 µs submit across the whole batch. `renderer.compute(a); renderer.compute(b);` pays it **per call**. Batch the entire CG iteration — ideally the whole fixed-iteration solve — into one array. (Measured kernel fusion on Vulkan: 1.41–1.67× for a 6→1 dispatch merge, p<0.001; 312→fewer dispatches across a forward pass gave +53% throughput. Metal saw ~0.95×, i.e. no benefit.)

**Readback / line search:** a sync costs ~470 µs on top of dispatch, plus ≥1 frame latency for `mapAsync`. A backtracking line search with 8 trial steps comparing energies on CPU ≈ 4 ms + 8 frames latency — unusable. Standard trick, confirmed: keep backtracking **GPU-resident with a fixed iteration count** and a mask/select — evaluate all trial step sizes, pick the winner in-shader, so loop count is a compile-time constant and the CPU never learns the answer. Read back once per frame at most; ideally never (see §2).

### 5. Web Workers — possible, but don't

WebGPU in workers: Chromium 113+, Firefox 141+; `OffscreenCanvas.getContext('webgpu')` available in worker scope (https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API, https://caniuse.com/mdn-api_offscreencanvas_getcontext_webgpu_context).

**Recommendation: main thread, beside the renderer.** Devices cannot be shared across threads, so a worker-side `GPUDevice` gets its own buffers ⇒ forces readback + `postMessage` of positions every frame ⇒ throws away the §2 zero-copy win that is the main reason to go GPU. The existing worker (`src/worker/solverWorker.ts`) exists to keep **CPU-bound JS** off the main thread (`docs/superpowers/plans/2026-07-04-worker-solver.md` §2, §D11) — but GPU dispatch is not CPU-bound, it's ~30 µs CPU per dispatch. Keep the worker as the CPU fallback it already is; add a third driver alongside `'main'`/`'worker'`. The worker's stateless compute-server contract (§D11) survives untouched.

### 6. Benchmarking — `timestamp-query` is available

Construct with `trackTimestamp: true` (requests the `timestamp-query` feature, injects timestamp writes around every compute/render pass). Then `await renderer.resolveTimestampsAsync(THREE.TimestampQuery.COMPUTE)` returns elapsed GPU ms and stores it at `renderer.info.compute.timestamp` — `src/renderers/common/Backend.js` (`resolveTimestampsAsync(type)` → `queryPool.resolveQueriesAsync()` → `this.renderer.info[type].timestamp = duration`). Constants: `TimestampQuery.COMPUTE === 'compute'`, `.RENDER === 'render'` (`src/constants.js`). `webgpu_compute_reduce.html` calls it after each `renderer.compute()`.

Two caveats for the gate doc:
- `trackTimestamp` / `resolveTimestampsAsync` are **experimental, not on the public `WebGPURenderer` API page** ([PR #30359](https://github.com/mrdoob/three.js/pull/30359) reworked them into `TimestampQueryPool`; see also [PR #30299](https://github.com/mrdoob/three.js/pull/30299)). Pin r185; re-verify on every three upgrade.
- Pre-register that A/B numbers come from **sequential dispatches with a single terminal sync**, and report CPU wall time and GPU timestamp **separately** — else the ~20× sync-conflation artifact makes any GPU path look bad and any batching change look miraculous.

---

## (b) Recommended architecture for THIS repo

- **Device ownership:** the existing `WebGPURenderer` (`src/scene/Viewer.tsx:376-381`) owns the one and only `GPUDevice`. GPU solver reads `renderer.backend.device` for limits (`device.limits.maxComputeWorkgroupSizeX`) and otherwise goes through `renderer.compute()`. No second adapter.
- **Thread:** main thread, beside the renderer. Worker stays CPU fallback (§5).
- **Buffers, all GPU-resident:**
  - vertex positions as a `StorageBufferAttribute` that doubles as the fat-line vertex attribute (§2)
  - per-vertex gradient
  - static topology (edges + CSR-flattened `disjointPairs`), uploaded once per `graphVersion` — mirrors the worker's existing topology-cache contract (`src/worker/solverWorker.ts:32`, plan §D4)
  - CG scratch vectors
  - small scalars buffer (energy, dot products)
- **Kernels:**
  1. gather-formulated pair energy+gradient keyed on edge I, **both roles**, no atomics (§3)
  2. fused CG iteration
  3. two-pass workgroup tree reduction for dot products + energy, modeled on `webgpu_compute_reduce.html`, workgroup size from `device.limits.maxComputeWorkgroupSizeX`
  4. GPU-resident **fixed-count** backtracking line search (§4)
- **Dispatch discipline:** one `renderer.compute([...])` array per descent step, **never** a call per kernel. Single highest-leverage decision in the design.
- **Readback:** none in steady state. `getArrayBufferAsync` on demand for UI/stats only, throttled well below frame rate.

---

## (c) Three sharpest implementation risks

1. **Bit-identity will break, and the repo's whole gate culture rests on it.** `golden.test.ts` asserts exact equality; `CLAUDE.md` + `tangentPointEnergy.ts:26-27, 92-96, 106-109` treat op order as load-bearing. f32-vs-f64 alone breaks it; the §3 gather reformulation changes summation order on top of that. **You need a new gate concept — relative-error tolerance vs. the f64 CPU reference plus a convergence-behavior gate — pre-registered *before* any GPU code lands**, or the milestone has no honest pass/fail criterion. This overlaps `gpu-precision-research`; the two must agree on tolerance before either writes a plan.

2. **Dispatch overhead may sink the CG loop regardless of kernel quality.** 250 dispatches/solve = 7.5 ms (Vulkan) / ~15 ms (D3D12) of pure API cost. If the batched-array submit doesn't amortize as expected, the iterative solve doesn't fit in a frame at *any* kernel quality. **Cheaply falsifiable before writing the physics kernel:** dispatch 250 trivial no-op compute nodes as one array vs. 250 separate `renderer.compute()` calls, measure both, kill the milestone if batching doesn't deliver. I'd make this **gate #1**.

3. **Zero-readback rendering requires rewriting `Curve.tsx`, and `Line2NodeMaterial` doesn't cooperate.** Material hardcodes its position attributes (`Line2NodeMaterial.js:117-118`); `Curve.tsx:37-49` is built around a per-frame CPU array + `setPositions()`. Usage flags make zero-copy possible (`WebGPUBackend.js:2564`), but integration is a genuine render-path rewrite against a fat-line material three exposes no hook for. Fallback is one `getArrayBufferAsync`/frame ≈ 0.5 ms + a frame of latency, eroding much of the GPU win. **Prototype the rendering integration early, not last** — it can invalidate the architecture.

Sources: https://arxiv.org/abs/2604.02344 · https://github.com/gpuweb/gpuweb/issues/4894 · https://toji.dev/webgpu-best-practices/compute-vertex-data.html · https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API · https://caniuse.com/mdn-api_offscreencanvas_getcontext_webgpu_context · https://github.com/mrdoob/three.js/pull/30359 · https://github.com/mrdoob/three.js/blob/dev/examples/webgpu_compute_reduce.html · https://shi-yan.github.io/webgpuunleashed/Compute/prefix_sum.html
