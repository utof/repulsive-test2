# bench/gpu — GPU gate harness

CDP-driven headless-Chrome harness for the WebGPU solver Phase 0 kill gates.
@see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §3–§5
@see docs/superpowers/plans/2026-08-13-webgpu-solver-phase0.md

## Winning launch recipe (G0a)

Verified 2026-08-13 against the real NVIDIA Quadro RTX 3000 in this machine
(`git a1f6022`). **Headless succeeded on the first flag set** — no headed
fallback was needed:

```
google-chrome \
  --headless=new \
  --enable-unsafe-webgpu \
  --enable-features=Vulkan \
  --use-angle=vulkan \
  --enable-gpu \
  --no-sandbox \
  --remote-debugging-port=9223 \
  --user-data-dir=<fresh temp dir> \
  http://localhost:3000/bench/gpu/harness.html
```

Result: `vendor: "nvidia"`, `architecture: "turing"` (Quadro RTX 3000 is a
Turing-class part) — a genuine hardware adapter, not SwiftShader/llvmpipe.
`bench/gpu/drive.ts` tries `FLAG_SETS` in this order automatically and
records whichever one first classifies as `hardware`; the raw result lives at
`bench/results/2026-08-13-gpu-g0a.json`.

A **fresh `--user-data-dir` per Chrome launch is required** — reusing the
default profile can make `google-chrome` hand off to an already-running
instance and silently ignore the new process's flags (including
`--remote-debugging-port`), which looks like a hang, not a flag failure.

## INVALID vs FAIL semantics

- **FAIL** — a gate ran against a confirmed hardware adapter and the
  measurement itself did not meet its threshold (e.g. G1's batching ratio,
  G3's CV). This is real, recordable evidence.
- **INVALID** — a gate ran against a software adapter (SwiftShader, llvmpipe,
  or anything `classifyAdapter()` in `drive.ts` calls `'software'`). Software
  rasterizers do not exercise real GPU dispatch/timing/precision behavior, so
  *any* number from such a run is worthless for a gate decision — per spec §3
  it must be **rerun with the recipe above**, never recorded as a gate
  result. `drive.ts` marks non-G0a runs `"status": "INVALID"` automatically
  when the adapter it detects is software; it never fabricates a PASS/FAIL
  for those.
- G0a itself is special: it *is* the hardware-adapter check. If no flag set
  (headless or headed) yields hardware, that is a **STOP-BRANCH**
  (`"status": "FAIL"` with full per-flag-set `attempts` diagnostics) — the
  rest of the browser-dependent gates cannot run honestly. See plan Task 1
  Step 3.

## Running a spike

Dev server must be up: `bun run dev` (serves `bench/gpu/harness.html` and
bundles `bench/gpu/spikes.ts` on the fly).

```
bun bench/gpu/drive.ts adapterInfo --out g0a   # G0a — must be run first
bun bench/gpu/drive.ts <spike> --out <label>   # any later spike, once G0a has PASSed
```

- `adapterInfo` (gate `G0a`) tries every entry in `FLAG_SETS` until one
  classifies as hardware, recording every attempt.
- Every other spike reads the flags from the most recent **PASSing**
  `bench/results/*-gpu-g0a*.json` and launches Chrome once with those flags —
  it does not re-search `FLAG_SETS`. If no passing G0a result file exists,
  the driver refuses to run and exits non-zero.
- Results are written to `bench/results/<date>-gpu-<label>.json`:
  `{ gate, status, adapter, flags, attempts, gitShaShort, date, data, consoleLines }`.

## Console noise to ignore

These lines can appear even on a genuine hardware adapter and are not
evidence of software fallback; `drive.ts`'s `CONSOLE_NOISE` filters them out
of `consoleLines` automatically:

- `RangeError: createBuffer failed, size (144) is too large` (~60/s from
  three's fat-lines)
- `Instance dropped in popErrorScope`
- Duplicate-key `0` React warning (`Warning: Encountered two children with
  the same key, \`0\`...`)

**Hardware-headless canvas-present noise (verified 2026-08-13, phase0 T4):**
when the *app* (not the harness page) runs under `--headless=new` on the
hardware Vulkan adapter, every presented frame spams a 4-line Dawn
validation chain starting with:

- `GPUValidationError: Requested allocation size (…) is smaller than the
  image requires (…) at ImportMemory (…MemoryServiceImplementationOpaqueFD.cpp…)`
  followed by `[Invalid Texture]` / `[Invalid TextureView]` /
  `[Invalid CommandBuffer]` cascade errors on `renderContext_*`.

This is the headless compositor frame-export path (canvas presentation),
NOT an app bug and NOT adapter fallback — A/B verified identical with and
without the `trackTimestamp` change, and absent in headed mode. It only
affects pages that *present a canvas* headlessly; pure compute spikes are
untouched. G4 (which renders frames) verifies via a buffer readback, not
the presented image, so these lines don't invalidate it — but grep them
out of any console-based assertion.

## G3 — timestamp benchmarking viability

Verified 2026-08-13 against the Quadro RTX 3000, same recipe as G0a (no
`--enable-webgpu-developer-features` needed): 5 runs of 100 batched dispatches
of a 65536-thread FMA kernel (`Loop(4096)` inner iterations, tuned so the
100-dispatch total lands at ~16 ms, comfortably above Dawn's 100 µs
quantization floor) gave `totalsMs ≈ [16.04, 16.01, 16.03, 15.99, 15.99]`,
`cv ≈ 0.0013` (0.13%) — **PASS** against the `cv < 0.1` gate by two orders of
magnitude. `renderer.info.compute.timestamp` is overwritten (not
accumulated) by each `resolveTimestampsAsync()` call
(`WebGPUTimestampQueryPool.js:94-121` resets `currentQueryIndex`/
`queryOffsets` inside `_resolveQueries()`), so no `renderer.info.reset()`
between runs was needed. GPU-timestamp benchmarking (`trackTimestamp` +
`resolveTimestampsAsync`) is viable for the remaining perf gates on this
machine; no CPU-wall-clock fallback required.

## G0t — throughput probe (G5 estimator input)

Verified 2026-08-13 against the Quadro RTX 3000, G3 methodology (5 runs,
medians, GPU timestamps — G3 PASSed so no wall-clock fallback needed):

- **FMA rate:** 2²² threads × 64-iteration raw-WGSL multiply-add chain
  (`wgslFn`, data-dependent per-thread accumulator so neither TSL nor the
  driver compiler can fold it), batched 20× per run. `fmaGflops ≈ 1505`
  (≈28% of the ≈5.3 TFLOPs f32 peak assumption — inside the 20–70% sanity
  band).
- **Dense matvec:** `y = M·x`, one thread per row, 3072×3072 f32 `M` seeded
  deterministically (xorshift32, no `Math.random`) and uploaded, `wgslFn`
  with `ptr<storage, ..., read>`/`read_write` params, batched 200× per run.
  `matvecMs ≈ 0.463` (inside the expected 0.1–0.5 ms band, ≈3.2–4.1× the
  ≈0.11–0.15 ms bandwidth floor — consistent with a naive, non-tiled
  per-row kernel, not a sign the kernel was optimized away), `matvecGflops
  ≈ 40.75`.
- Both `fmaTotalsMs`/`matvecTotalsMs` arrays show the median (not mean) is
  the right statistic: the matvec's first run is a ~2× outlier (`177 ms` vs.
  a steady `~92-93 ms`), a one-time pipeline/shader-compile warm-up cost,
  not throughput signal.
- `wgslFn` storage-buffer-parameter syntax (`ptr<storage, array<f32>,
  read|read_write>` + a named-object call site, e.g.
  `matvecFn({ M: matBuf, x: xBuf, y: yBuf, row: instanceIndex })`) has no
  local three.js example in this repo's `node_modules`; verified against
  https://discourse.threejs.org/t/how-to-use-storagebufferattribute-as-a-input-to-wgslfn/73006
  and https://blog.maximeheckel.com/posts/field-guide-to-tsl-and-webgpu/
  (fetched 2026-08-13).
- Full numbers: `bench/results/2026-08-13-gpu-g0t.json`.

## G1 — dispatch-batching kill gate

Verified 2026-08-13 against the Quadro RTX 3000, same recipe as G0a: 250
DISTINCT no-op compute nodes (each a separate `Fn(() => {...})()` closure
over its own index `i`, `.compute(1)` — 1 workgroup each), 5 runs/medians,
sequential-dispatches-with-a-single-terminal-sync methodology (CPU wall
time of the submission loop only; `device.queue.onSubmittedWorkDone()` is a
between-run barrier, not part of the timed region)
[research doc §4, `docs/2026-08-13-ai-research-webgpu-compute.md`]:

- Arm A (batched, one `renderer.compute([...250 nodes])`):
  `batchedMs ≈ 0.3` (runs `[0.8, 0.3, 0.3, 0.5, 0.3]`).
- Arm B (unbatched, 250 separate `renderer.compute(node)` calls):
  `unbatchedMs ≈ 2.3` (runs `[3.2, 2.3, 2.2, 4.0, 2.2]`).
- `ratio ≈ 0.130`.
- **PASS**: `batchedMs (0.3) < 2` AND `batchedMs (0.3) < 0.25 × unbatchedMs
  (0.575)`. Both numbers land well below the plan's rough expected band
  (≈0.8–1.7 ms batched / ≈7.5 ms unbatched) — this machine's Vulkan/Dawn
  dispatch overhead is lower than the cross-machine estimate in the
  implementation research doc, but the batching *ratio* (the load-bearing
  half of the gate) holds by a wide margin either way.
- Per-frame iterative GPU solve loop is NOT dead — dispatch batching
  amortizes as required; the milestone is not reduced to kernels-only on
  this gate.
- Full result: `bench/results/2026-08-13-gpu-g1.json`.

## Phase 0 gate report

Filled by Task 12 once every gate below has a committed result.

| Gate | Result | Number | Consequence |
|---|---|---|---|
| G0a | PASS | nvidia/turing, hardware Vulkan adapter | browser gates unblocked |
| G0t | recorded | fmaGflops≈1505, matvecMs≈0.463, matvecGflops≈40.75 | G5 estimator input |
| G1 | PASS | batchedMs≈0.3, unbatchedMs≈2.3, ratio≈0.130 | per-frame GPU solve loop not dead; batching amortizes |
| G2 (spike) | | | |
| G3 | PASS | cv = 0.0013 (< 0.1 gate) | GPU-timestamp benchmarking viable, no wall-clock fallback needed |
| G4 | | | |
| G6 | | | |
| Baselines | | | |
