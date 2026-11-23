# WebGPU Solver — Phase 2a (GPU Cholesky of the Sobolev Block) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the O(n³) saddle factorization off the CPU — factor the shifted N×N Sobolev block `A_σ` once per step with an f32 WebGPU Cholesky, read the factor back, and serve the gradient solve and every frozen projection solve from it in CPU f64 via the exact Schur+Woodbury reformulation with iterative refinement — behind a new source-selection seam that is bit-identical when absent.

**Architecture:** Three layers, each independently gated. (1) `src/core/` gains ONE public-type delta — an optional `saddle?: PreparedSaddle` that rides `dispatchDescentStep → sobolevStepSet → solveConstrainedGradientSetFrozen` exactly as Phase 1's `dE?` does, selecting the *source* of the factorization and never its arithmetic; absent ⇒ every existing path is byte-for-byte unchanged. (2) `src/gpu/saddle.ts` is pure CPU f64 — the shift, the Schur complement, the Woodbury correction and the IR loop — and is fully testable with a CPU f64 Cholesky before a line of WGSL exists (gate CR). (3) `src/gpu/cholesky.ts` is the only new renderer-bound file: three kernels per column driven by a GPU-side column counter in one `renderer.compute([...])` batch, a status flag in the same u32 meta buffer, and one readback through a reused `THREE.ReadbackBuffer`. **The plan splits at a `G5-confirm` gate after Task 5**: the (a)-vs-(b) solve-path decision is re-decided on the *measured* spike before any driver, store or self-test integration begins.

**Tech Stack:** Bun (tests, CDP gate driver), TypeScript strict, three `0.185.1` (`three/webgpu` + `three/tsl`: `wgslFn`, `attributeArray`, `instancedArray`, `instanceIndex`, `Fn().compute(count,[wg])`, `renderer.compute([...])`, `renderer.getArrayBufferAsync(attr, target)`, `THREE.ReadbackBuffer`, `renderer.backend.device.limits`), Zustand 5, React 19 / R3F 9, numpy/scipy via `uv` for the oracle twin.

**Spec:** `docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md` — §0 (state detection), §1 (scope, slices, non-goals), §2 (the partition and `R₂`), §3 (G5 and its AMENDS), §4.1 (the algebra), §4.2 (precision model), §4.3 (the kernel), §4.4 (the seam — public-type delta (iv)), §4.5 (driver flow, routing, three fallback triggers), §5 (the six gates + boot, all bars), §6 (kill gates), §7 (risks), §8 (open questions this plan answers). Parent spec `docs/superpowers/specs/2026-08-13-webgpu-solver-design.md` cited as **[P §n]**.

**Phase entry:** Phase 1 merged (PR #26, `bf0abcf`). Spec §0's state detector reads: `src/gpu/cholesky.ts` absent ⇒ Phase 2 not started; `src/core/sobolev/linsolve.ts` has no `'external'` factorization kind ⇒ the core seam has not landed. Both hold at branch point.

## Global Constraints

- **Slice 2a only** (spec §1). The GPU computes **dE (Phase 1, unchanged) and the Cholesky factor of `A_σ`**. `assembleAFlat`, every backsolve, the Schur/Woodbury algebra, iterative refinement, the projection, the line search and every energy stay **CPU f64**. 2b (GPU assembly) and 2c (GPU line search) are out of scope and must not be anticipated in code.
- **No `src/core/**` numeric changes** (spec §1, [P §1]). `assembleAFlat`, `ldltFactor`, `luFactor`, `buildSaddleMatrix`, `lineSearchStepSet`, `projectOntoConstraintSet`, `FactorMode`, `factorMode` are untouched. The seam is **source-selection only**; absent ⇒ bit-identical, and the full `bun test` suite is the backstop.
- **Sanctioned public-type delta (iv)** (spec §4.4 — AMENDS [P §1], which names only (i)/(ii)/(iii)): `ExternalFactorization`, `ExternalSolveError`, `PreparedSaddle` in `linsolve.ts`; `SaddleFactorization` widened; `solveConstrainedGradientSetFrozen` gains a TENTH optional positional parameter; `SobolevStepOptions.saddle?`; `DispatchDescentStepArgs.saddle?`; `SimStore.gpuSolveAvailable` + `setGpuSolveAvailable`. **No other export.** `DescentStepOutcome` is NOT touched — `usedGpuSolve` is a field of `GpuDriver.step`'s own return, beside `usedGpu`.
- **The GPU solve serves `barycenter + totalLength (+ pins)` only** (spec §1). `edgeLengths` (perEdge) never reaches it — [G6] already routes perEdge N ≥ 480 to CPU f64. `'reassemble'` projection mode stays CPU permanently.
- **`solve` must self-certify or throw** (spec §4.1 step 4). Nothing in `src/core/` thresholds `residual`, so an unconverged `z` returned from `PreparedSaddle.solve` is a silently wrong descent direction. Exhausting 4 IR iterations without reaching `1e-10` ⇒ `throw new ExternalSolveError(...)`.
- **One device** — the Viewer's `WebGPURenderer`. Never a second adapter (`Viewer.tsx:110`). Main thread only; `PreparedSaddle` carries closures ⇒ never structured-cloneable ⇒ never travels to the worker.
- **One `renderer.compute([...])` array per GPU stage; two batches per step** (spec §4.5): the dE batch first (so the CPU `assembleAFlat` overlaps it), then the Cholesky batch.
- **Gates run on the hardware adapter** (`bench/gpu/drive.ts` classifies; software adapter ⇒ **INVALID**, never PASS/FAIL). Results committed as `bench/results/<date>-gpu-phase2a-*.json` keyed by `gitShaShort` + adapter. **Every gate records its measured VALUE, not only its verdict** (spec §5).
- **Pre-registered bars are not negotiable and live in the spec, not here** (CLAUDE.md). If this plan appears to need a gate the spec does not carry, that is a spec fix, not a plan section. The bars, verbatim from spec §5:
  - **CR** ‖Δz‖/‖z‖ ≤ `max(1e-12, 10·κ₂(K,N)·u_f64)` = **1.6e-12 / 1.6e-11 / 1.6e-10 / 1.6e-9** at N=120/240/480/960.
  - **K1** green ≤ **2.72e-6 / 5.71e-7 / 2.23e-7 / 7.68e-8** at N=64/240/480/960; **red** ≥ `e_max(N)` = **1.72e-4 / 7.66e-6 / 1.10e-6 / 1.45e-7**; amber in between (not a kill; T4 at that N decides the served N).
  - **T4** rel residual ≤ **1e-10** within ≤ **4** IR iterations, per N.
  - **T5** energy after K=50 steps within **±1 %** of the f64 trajectory; no accepted step increases energy; median of 3; trefoil N=**120 and 480**.
  - **G5-confirm** at N=960: **green** whole step ≤ **586 ms**; **red** whole step ≥ **879 ms**; **amber** between ⇒ green requires solve path ≤ **283 ms**, else red. N=480 recorded, not gated.
  - **P2a** gated at **N=960 only**: speedup ≥ **8×**; GPU factor wall ≤ **100 ms**. **N=480 RECORDED, NOT GATED.** Precondition: **no P2a number is read as pass/fail until `R₂` is ATTRIBUTED** (Task 7 D4).
  - **Boot** `max|L_gpu − L_cpu| / max|L_cpu|` < **1e-5** on trefoil(64) (a correct f32 kernel measures 9.04e-8).
- **Kill gates** (spec §6): CR red ⇒ no GPU work starts. K1 red at N=480 ⇒ (a) not confirmed ⇒ the milestone routes to **building variant (b)**, not to kernels-only. `G5-confirm` red ⇒ **2a stops at the spike and the milestone re-plans onto (b)**. T4 red at N ≤ 480 is a kill. P2a red at N=960 ⇒ 2a does not merge. Boot red on the dev box ⇒ no merge.
- **Conventions:** Biome (4-space, single quotes, width 100); `import type`; `bunx tsc --noEmit` clean; `bun test <path>`; `bunx biome check --write` on touched files; `bunx knip` no new findings. **TSDoc on every export with `@see <spec/plan path>` or a `Why:` line, and an inline anchor on every guard, every `0.5`, every ε, and every op-order-preserving line** (CLAUDE.md — the reference is what stops a future agent from "cleaning up" an intentional line). **One commit per task.** Trailer on every commit:
  ```
  Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01AbdMPo3QLVT8dq89qjyeAo
  ```
- Reviewer nits follow CLAUDE.md's inline-fix gate (fix inline only if EVERY gate holds, else `gh issue create` with labels). Blockers are fixed on the branch regardless of size.
- **`codebase-memory-mcp` failed to connect while this plan was written**; every `file:line` below was re-derived with `rg`/`sed` against `bf0abcf` and re-verified at authoring time. Re-verify before editing if the branch has moved.

---

## File structure

| Path | Responsibility |
|---|---|
| `src/core/sobolev/linsolve.ts` | **Modify.** `ExternalFactorization`, `ExternalSolveError`, `PreparedSaddle` (new exports); `SaddleFactorization` widened; `solveFactored` gains the `'external'` branch. Nothing else changes. |
| `src/core/sobolev/gradient.ts` | **Modify.** `solveConstrainedGradientSetFrozen` gains the 10th positional param `saddle?: PreparedSaddle`; with it, `A = saddle.A`, `fac = saddle.factor(C)`, and the solve routes through the already-exported `solveSaddleFrozen`. |
| `src/core/optimizer.ts` | **Modify.** `SobolevStepOptions += saddle?`; forward to the frozen solve; rethrow `ExternalSolveError` from the bare catch at `:361`; disarm the timing collector on the throw path. |
| `src/core/dispatch.ts` | **Modify.** `DispatchDescentStepArgs += saddle?`, forwarded exactly like `dE`. |
| `src/worker/solverWorker.ts` | **Modify.** Defensive throw if a `saddle` ever appears in a worker payload (it cannot be cloned). |
| `src/gpu/saddle.ts` | **Create.** Pure CPU f64: `choleskyF64`, `makePreparedSaddle(A, L, sigma, n)` — spec §4.1 steps 1–4 (shift, Schur, Woodbury, IR + the `ExternalSolveError` throw). No `three` import ⇒ unit-testable under `bun test`. |
| `src/gpu/cholesky.ts` | **Create.** The only new renderer-bound file. WGSL sources for `colFactor` / `trailingUpdate` / `finalize`, the GPU-side column counter + status flag, `GpuCholesky` (upload `A_σ`, one batch of 3N nodes, one readback through a reused `ReadbackBuffer`), `GPU_SOLVE_MAX_N`. |
| `src/gpu/driver.ts` | **Modify (Task 6).** `gpuSolveSupported`, the two-batch step flow, the three fallback triggers, `usedGpuSolve` in the return, `timings.factor`. |
| `src/gpu/selfTest.ts` | **Modify (Task 6).** `runGpuSolveSelfTest` — factor trefoil(64)'s `A_σ`, compare with the CPU f64 Cholesky, bar 1e-5. |
| `src/store.ts` | **Modify (Task 6).** `gpuSolveAvailable: boolean` + `setGpuSolveAvailable`. |
| `src/scene/Viewer.tsx` | **Modify (Task 6).** The second capability bit mirrored at ALL FOUR `setGpuAvailable` sites (486, 636, 655, 669), `poisoned` latch included. |
| `oracle/tpe_phase2a_reformulation.py` | **Create (Task 3).** The numpy twin of §4.1 steps 1–3 — CR's independent reference, so a CR red is localisable to TS or to the algebra. |
| `bench/gpu/phase2a.ts` | **Create.** Browser-side spikes: `phase2aCholeskyK1` (K1 + T4 + factor wall), `phase2aG5Confirm`, `phase2aStepGate` (P2a), `phase2aT5`, `phase2aR2Attribution`. Registered into `spikes.ts` beside `phase1Spikes`. |
| `test/gpu/saddleSeam.test.ts` | **Create (Task 1).** Absent-seam bit-identity + present-seam routing + `ExternalSolveError` propagation. |
| `test/gpu/saddle.test.ts` | **Create (Task 2).** `choleskyF64`, Schur, Woodbury, IR, the throw. |
| `test/gpu/cr.test.ts` | **Create (Task 3).** The CR gate as a `bun test`, all fixtures, κ-relative bar, `diag(M)` / `κ(M)` / `min diag(L)²` recorded. |
| `test/gpu/cholesky.test.ts` | **Create (Task 4).** Pure parts only: WGSL source generation, `GPU_SOLVE_MAX_N` arithmetic, meta-buffer encoding. |
| `bench/gpu/README.md` | **Modify (Task 7).** "Phase 2a gate report". |

---

## Decisions (read before Task 1; each is falsifiable)

### D1 — Disarming the timing collector: catch-and-rethrow, NOT a bare `finally`

Spec §4.4 requires that a throw out of `timed('step', …)` must not leave the module-scoped collector armed, and writes the requirement as "`finally { if (collect) timingsEnd(); }`". **Implemented literally that is a bug**, and the plan implements the requirement, not the sketch.

`timingsEnd()` is not idempotent — it returns the ledger *and* nulls `acc` (`src/core/sobolev/phaseTimings.ts:56-63`). The success path reads the ledger at `optimizer.ts:461` (`if (collect) { const timings = timingsEnd(); if (timings) …}`). A `finally` that calls `timingsEnd()` unconditionally runs on the SUCCESS path too, nulls `acc` first, and the later read returns `null` — so `collectTimings: true` would silently stop returning `timings`. That breaks `bench/sobolev.bench.ts`, Phase 1's `if (usedGpu && outcome.timings)` stamp (`driver.ts:175`) and every gate in spec §5 that reads `timings.factor`.

**Decision:** disarm on the throw path only.

```ts
let outcome: ReturnType<typeof stepBody>;
try {
    outcome = timed('step', stepBody);
} catch (e) {
    // The collector is module-scoped and only timingsEnd() disarms it
    // (sobolev/phaseTimings.ts:56-63), so a throw out of timed('step', …) would
    // leak THIS step's partial ledger into the NEXT step. Disarm on the throw
    // path ONLY: an unconditional `finally` would null `acc` before the success
    // path reads it below and silently drop `timings` from every collected step.
    // @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4
    if (collect) timingsEnd();
    throw e;
}
```

Falsifiable: Task 1 Step 3 writes a test that asserts BOTH halves — a throwing step leaves the next step's `timings.step.calls === 1`, and a non-throwing collected step still returns `timings`. The bare-`finally` form fails the second assertion.

### D2 — `PreparedSaddle` and `ExternalSolveError` live in `linsolve.ts`

Spec §8 leaves this open (`linsolve.ts` vs a new `src/core/sobolev/prepared.ts`). **Decision: `linsolve.ts`.** `ExternalFactorization` is a member of the `SaddleFactorization` union declared at `linsolve.ts:399`, and `solveFactored` (`:696`) is the only consumer of that union. A separate file would have to import the union and be imported back by `linsolve.ts` for the `'external'` branch — a cycle for zero gain. `PreparedSaddle.factor` returns a `SaddleFactorization`, so it belongs with the union too.

### D3 — `G5-confirm` is measured WITHOUT the driver

Spec §5 pre-registers `G5-confirm` to run "after the Cholesky + K1 spike and **BEFORE any driver / store / self-test integration**", yet its statistic is a **whole descent step** p50. These are compatible: the spike calls `dispatchDescentStep({ …args, dE, saddle })` directly in the browser harness, using the Task 1 seam and a `PreparedSaddle` built by hand from Task 4's `GpuCholesky` output. No `GpuDriver.step`, no store bit, no boot self-test is required to time a whole step. Task 5 does exactly this; Task 6 does not exist yet when the gate is read.

**Both readings are reported** (spec §3, ruling of record): `wholeStepMs` = p50 of `dispatchDescentStep` wall time including the GPU factor's upload/compute/readback, and `solvePathMs` = p50 of (factor wall + all Schur/Woodbury/IR/backsolve time for the gradient solve and every projection solve), i.e. the phase that replaces `saddle` in [BASE]. Neither is derived from the other; both are recorded in the results JSON whatever the verdict.

### D4 — `R₂` is ATTRIBUTED by a within-step measurement, not by recording it again

Spec §5 makes attribution a precondition on reading ANY P2a number, and spec §2 states the hypothesis to be ruled in or out: `R₂` = 60.4 / 7.9 / 4.4 ms at N=480/960/1000 is **non-monotonic in N**, which is not a cost profile, and [BASE]'s N=1000 row forces `R < 0` by 5.28 ms, which no amount of work can explain.

The confound is structural: in [BASE] **each key is an independent median over K=5 steps**, so `step` and `Σkeys` are medians of *different* step samples and need not be consistent. **Decision: attribute by measuring the same quantity WITHIN one step.** For each of K=5 individual steps at N=480 and N=960, record the whole `SobolevStepTimings` ledger of THAT step and compute `r₂ᵢ = stepᵢ − dEᵢ − assembleAᵢ − saddleᵢ − energyᵢ`. Then:

- If the per-step `r₂ᵢ` are ≈0 (say `|median r₂ᵢ| < 0.1 × R₂`), `R₂` is **measurement error** — an artifact of medianing keys independently. Named, and the P2a numerator/denominator are then quoted without it as a sensitivity row.
- If the per-step `r₂ᵢ` reproduce ≈60 ms at N=480, it is **work**, and one `timed()` probe is added around the largest unwrapped region of `sobolevStepSet` (the `l2CurveNorm` + `lineSearchStepSet` orchestration outside the `energy`/`projection` wraps) to name it. That probe is a NEW phase key — `phaseTimings.ts`'s schema note says "Add keys; never rename", so this is sanctioned.

Either branch names something. "Recording `R₂` again is not attributing it" (spec §5).

### D5 — The column index reaches the shader through a GPU-side counter; the fallback is measured, not assumed

Spec §4.3 fixes this: the array holds the SAME three node objects repeated N times and `finalize` increments a 1-element u32 counter, because a uniform shared across a batched `renderer.compute([...])` presents its LAST written value to every dispatch (`Renderer.js:2718-2817` calls `updateForCompute` per node and submits once). The inter-dispatch visibility this needs is guaranteed — each dispatch is its own synchronization scope ([gpuweb/gpuweb#4434](https://github.com/gpuweb/gpuweb/discussions/4434)) — and Phase 1's shipped two-pass reduction already depends on it (`src/gpu/reduce.ts:66-67,116`).

**Decision:** build the counter form. Task 4 Step 9 measures node-build wall time for the counter form (3 builds) and, only if the counter is demonstrably wrong on this adapter (boot fixture red with a correct CPU reference), for the per-node-uniform fallback (3N builds) before choosing it. The `IndirectStorageBufferAttribute` escape hatch (`node_modules/three/src/renderers/common/IndirectStorageBufferAttribute.js`, honoured at `WebGPUBackend.js:1636-1644`) is **named and not built** — spec §6 licenses it only if P2a's factor-wall bar fails.

### D6 — `A_σ` is built into a REUSED buffer

[JSALG]'s measured spread on the "A_σ build + f32 cast" constant is 10.7× at N=480 (0.474–5.080 ms), and the spec attributes it entirely to allocation pressure: the `churn` regime re-allocates the N×N f32 staging buffer per iteration and its GC lands in the neighbouring backsolve/matvec timings too. Spec §4.3 commits 2a to the `reuse` regime. **Decision:** `GpuCholesky` owns one `Float32Array(N*N)` staging buffer for the life of the instance and `makePreparedSaddle` never allocates an N×N array per solve. A per-step allocation here is a P2a regression, not a style question.

### D7 — The readback target is one reused `THREE.ReadbackBuffer` per engine

Verified in `node_modules/three` at r0.185.1: `Renderer.getArrayBufferAsync(attribute, target = null, offset = 0, count = -1)` (`renderers/common/Renderer.js:1951`) returns the `ReadbackBuffer` itself when `target.isReadbackBuffer` (`:1954`); `ReadbackBuffer` is constructed with a byte length, exposes `.buffer: ArrayBuffer|null`, `.release()` and `.dispose()` (`renderers/common/ReadbackBuffer.js:9-78`), and is exported from `three/webgpu` (`src/Three.WebGPU.js:16`, typed at `@types/three/src/renderers/common/ReadbackBuffer.d.ts:14`). **`release()` is mandatory before reuse** — `WebGPUAttributeUtils.js:401` throws `'ReadbackBuffer must be released before being used again'`. With `target = null` (Phase 1's form, `tangentPoint.ts:195`) three allocates and destroys a staging buffer per call, which at 4 MB per step is exactly the cost spec §4.3 avoids.

### D8 — The `'external'` branch must be tested with `'kind' in fac`, not `fac.kind`

`LuFactorization` (`linsolve.ts:219-223`) has **no** `kind` field, so `fac.kind === 'external'` is a TypeScript error on the union. `solveFactored`'s existing body is `return 'kind' in fac ? ldltSolveFactored(fac, rhs) : luSolveFactored(fac, rhs);` (`:696-698`) and its anchor comment at `:693-695` explains that the `'lu'` branch is the verbatim pre-existing call. **Both survive**: the new branch is prepended as `'kind' in fac && fac.kind === 'external'`, the LDLᵀ test becomes the second arm, and the LU arm is untouched.

---

## The `G5-confirm` split

**Tasks 1–5 run first. Tasks 6–7 do not start until `G5-confirm` is green.**

Task 5 measures variant (a) — the thing spec §3 could only *estimate* — and spec §5's `G5-confirm` row re-decides [P §4 G5] on that measurement, on the parent's own statistic, with **both** readings reported. This is the parent's own escape hatch ([P §4 G5]: the spike "confirms the estimate BEFORE Phase 2 commits to (a)"), and it is why Tasks 4–5 are scheduled before any integration: a red discards a spike rather than an integrated branch.

| `G5-confirm` at N=960 | consequence |
|---|---|
| **green** — whole-step p50 ≤ **586 ms** | §3's provisional decision becomes the decision. **Tasks 6–7 start.** |
| **amber** — whole-step p50 in (586, 879) ms AND solve-path p50 ≤ **283 ms** | green by the solve-path reading. **Tasks 6–7 start**, and the README records that the whole-step reading did not decide. |
| **amber** — whole-step p50 in (586, 879) ms AND solve-path p50 > **283 ms** | **red.** |
| **red** — whole-step p50 ≥ **879 ms** (inside or above variant (b)'s measured band) | **Slice 2a STOPS at the spike.** Tasks 6–7 are NOT started. The milestone re-plans onto variant (b) — spec §3 shows (b) is a viable fallback (6.1–14.6× on the solve path, 4.9–8.2× on the whole step at N=960), not a dead end. Record the measured numbers, commit them, and hand back to the orchestrator. |

The executing agent **must stop and report at this gate**, not read past it. Tasks 6 and 7 are written on the assumption that (a) was confirmed; running them after a red would be building on a decision the gate reversed.

---

## Task 1: The core seam — `PreparedSaddle`, `ExternalSolveError`, and absent-seam bit-identity

**Files:**
- Modify: `src/core/sobolev/linsolve.ts` (new exports after `:399`; `solveFactored` at `:696-698`)
- Modify: `src/core/sobolev/gradient.ts:84-115` (10th positional param)
- Modify: `src/core/optimizer.ts:95` (`SobolevStepOptions`), `:281-286` + `:456-461` (collector disarm), `:338-383` (rethrow), `:344-354` (forward)
- Modify: `src/core/dispatch.ts:164` (`saddle?`), `:268-270` (forward)
- Modify: `src/worker/solverWorker.ts:34` (defensive throw)
- Test: `test/gpu/saddleSeam.test.ts` (create)

**Interfaces:**
- Consumes: nothing (first task).
- Produces — later tasks depend on these EXACT names and shapes:
  ```ts
  export interface ExternalFactorization { kind: 'external'; solve(rhs: number[]): number[] }
  export type SaddleFactorization = LuFactorization | LdltFactorization | ExternalFactorization;
  export class ExternalSolveError extends Error {}
  export interface PreparedSaddle {
      A: Float64Array;
      factor(C: number[][]): SaddleFactorization;
  }
  ```
  `solveConstrainedGradientSetFrozen(vertices, edges, disjointPairs, alpha, beta, epsilon, dE, set, factorMode?, saddle?)` — `saddle` is the TENTH positional parameter. `SobolevStepOptions.saddle?: PreparedSaddle`. `DispatchDescentStepArgs.saddle?: PreparedSaddle`.

- [ ] **Step 1: Write the failing absent-seam bit-identity test**

This is the load-bearing test of the whole task: the two control-flow edits below (the rethrow at `optimizer.ts:361` and the collector disarm) sit on the default path, and "unreachable when `saddle` is absent" must be **tested, not argued**.

Create `test/gpu/saddleSeam.test.ts`:

```ts
import { expect, test } from 'bun:test';
import { type DispatchDescentStepArgs, dispatchDescentStep } from '../../src/core/dispatch';
import { trefoil } from '../../src/core/fixtures';
import { DEFAULTS } from '../../src/core/optimizer';
import {
    ExternalSolveError,
    type PreparedSaddle,
    type SaddleFactorization,
} from '../../src/core/sobolev/linsolve';
import { assembleAFlat } from '../../src/core/sobolev/innerProduct';
import { calculateDisjointPairs } from '../../src/core/tangentPointEnergy';
import type { Vec3 } from '../../src/core/testConfigs';

function args(n = 24): DispatchDescentStepArgs {
    const { vertices, edges } = trefoil(n);
    const x0: Vec3 = [0, 0, 0];
    for (const v of vertices) {
        x0[0] += v[0] / n;
        x0[1] += v[1] / n;
        x0[2] += v[2] / n;
    }
    const L0 = edges.reduce(
        (s, [a, b]) =>
            s +
            Math.hypot(
                vertices[a][0] - vertices[b][0],
                vertices[a][1] - vertices[b][1],
                vertices[a][2] - vertices[b][2],
            ),
        0,
    );
    return {
        descentMode: 'sobolev',
        vertices,
        edges,
        disjointPairs: calculateDisjointPairs(edges),
        mode: 'analytical',
        stepSize: 1,
        x0,
        barycenterConstraint: true,
        lengthMode: 'total',
        sobolevL0: L0,
        projectionMode: 'frozen',
        collectTimings: false,
        collectField: true,
    };
}

// The seam's whole contract: with `saddle` ABSENT the step must be byte-for-byte
// what it was before the seam existed — INCLUDING the two control-flow edits this
// task adds on the default path (the ExternalSolveError rethrow at optimizer.ts:361
// and the timing-collector disarm). Arguing they are unreachable is not enough.
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4
test('saddle seam ABSENT: the step is toEqual-identical to the pre-seam path', () => {
    const a = args();
    // Guard the comparison's meaning: a rejected/converged step would compare two echoes.
    expect(dispatchDescentStep(a).accepted).toBe(true);
    expect(dispatchDescentStep(args())).toEqual(dispatchDescentStep(a));
});

test('saddle seam ABSENT: collectTimings still returns a ledger (the disarm is throw-path only)', () => {
    const r = dispatchDescentStep({ ...args(), collectTimings: true });
    expect(r.timings).toBeDefined();
    expect(r.timings?.step?.calls).toBe(1);
    expect(r.timings?.saddle?.calls).toBe(2);
});
```

- [ ] **Step 2: Run it to confirm the imports fail**

Run: `bun test test/gpu/saddleSeam.test.ts`
Expected: FAIL — `ExternalSolveError` / `PreparedSaddle` are not exported from `linsolve.ts`.

- [ ] **Step 3: Add the three new exports to `linsolve.ts`**

Insert immediately after the `SaddleFactorization` declaration at `linsolve.ts:399`, and widen that union in place:

```ts
/**
 * A factorization supplied from OUTSIDE the core — the GPU Cholesky path.
 * `solve(rhs)` returns z for the FULL (3n+k) saddle system K z = rhs, having
 * itself certified the result (spec §4.1 step 4); it throws
 * {@link ExternalSolveError} rather than returning an uncertified z.
 * Why: nothing in `src/core/` ever compares `residual` against a threshold —
 * `linsolve.ts:800`/`:903` compute it and `optimizer.ts:355,395,449` forward it
 * as a stat — so a solve that converged to 1e-3 is indistinguishable at every
 * call site from one that converged to 1e-12.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1, §4.4
 */
export interface ExternalFactorization {
    kind: 'external';
    solve(rhs: number[]): number[];
}

/**
 * Thrown by an external solve that cannot certify its own result — IR exhausted
 * its 4 iterations without reaching 1e-10 (spec §4.1 step 4). Typed (not a bare
 * Error) because `optimizer.ts`'s singular-saddle catch must let exactly this one
 * through: converting it to 'singular_system' would hide an unconverged descent
 * direction behind a rejected step.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1, §4.4, §4.5
 */
export class ExternalSolveError extends Error {}

/**
 * A saddle solve prepared OUTSIDE the step, for the step's INPUT vertices.
 * `A` MUST be the Sobolev matrix of the same vertices the step receives (the
 * driver copies `live` once and uses that copy for dE, A and the step — the
 * Phase 1 rule at `src/gpu/driver.ts:164`). `factor` is called at most once per
 * step, synchronously, with the C the CORE evaluated; `solve` then runs for the
 * gradient rhs and for each projection rhs. Carries closures ⇒ NOT
 * structured-cloneable ⇒ it can never travel to the worker.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4
 */
export interface PreparedSaddle {
    A: Float64Array;
    factor(C: number[][]): SaddleFactorization;
}
```

Widen the union at `:399` (keep its existing TSDoc, append one line to it):

```ts
export type SaddleFactorization = LuFactorization | LdltFactorization | ExternalFactorization;
```

- [ ] **Step 4: Add the `'external'` branch to `solveFactored`**

`linsolve.ts:696-698` becomes — note the `'kind' in fac` test is REQUIRED (D8: `LuFactorization` has no `kind`, so `fac.kind` alone is a TS error on the union), and the `'lu'` arm stays the verbatim pre-existing call:

```ts
function solveFactored(fac: SaddleFactorization, rhs: number[]): number[] {
    // 'external' first: the GPU path's own solve() (spec §4.1 steps 2-4). The
    // `'kind' in fac` guard is not redundant — LuFactorization (:219-223) has no
    // `kind` field, so `fac.kind === …` alone is a TS error on this union.
    // @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4
    if ('kind' in fac && fac.kind === 'external') return fac.solve(rhs);
    return 'kind' in fac ? ldltSolveFactored(fac, rhs) : luSolveFactored(fac, rhs);
}
```

- [ ] **Step 5: Run the seam tests and the full suite**

Run: `bun test test/gpu/saddleSeam.test.ts && bun test`
Expected: both new tests PASS; the full suite is at its pre-branch count with 0 failures (307 pass at `bf0abcf` — a change in that number here means the seam was not source-selection-only).

- [ ] **Step 6: Commit the type surface**

```bash
git add src/core/sobolev/linsolve.ts test/gpu/saddleSeam.test.ts
git commit -m "$(cat <<'EOF'
feat(core): 'external' saddle factorization kind + PreparedSaddle/ExternalSolveError (phase2a T1)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AbdMPo3QLVT8dq89qjyeAo
EOF
)"
```

- [ ] **Step 7: Write the failing present-seam routing test**

Append to `test/gpu/saddleSeam.test.ts`. This proves the seam actually reaches the solve, using a `PreparedSaddle` whose `solve` is the CPU path scaled — the same "strong probe" shape as Phase 1's `dE` seam test.

```ts
// A PreparedSaddle that delegates to the CPU LU factorization of the SAME K the
// core would have built, so `saddle` present and `saddle` absent must agree to
// round-off. This is the routing probe; the numerics live in Task 2.
function identityPrepared(a: DispatchDescentStepArgs): PreparedSaddle {
    const A = assembleAFlat(
        a.vertices,
        a.edges,
        a.disjointPairs,
        DEFAULTS.alpha,
        DEFAULTS.beta,
        DEFAULTS.epsilon,
    );
    return {
        A,
        factor(C: number[][]): SaddleFactorization {
            const n = a.vertices.length;
            const m = 3 * n;
            const k = C.length;
            const size = m + k;
            const K: number[][] = Array.from({ length: size }, () => new Array(size).fill(0));
            for (let b = 0; b < 3; b++)
                for (let i = 0; i < n; i++)
                    for (let j = 0; j < n; j++) K[b * n + i][b * n + j] = A[i * n + j];
            for (let r = 0; r < k; r++)
                for (let c = 0; c < m; c++) {
                    K[m + r][c] = C[r][c];
                    K[c][m + r] = C[r][c];
                }
            return {
                kind: 'external',
                solve: (rhs: number[]) => luSolve(K, rhs),
            };
        },
    };
}

test('saddle seam PRESENT: routes through the external solve and agrees with the CPU path', () => {
    const a = args();
    const base = dispatchDescentStep(a);
    const viaSeam = dispatchDescentStep({ ...args(), saddle: identityPrepared(a) });
    expect(viaSeam.accepted).toBe(base.accepted);
    const b = base.descentField as Vec3[];
    const t = viaSeam.descentField as Vec3[];
    let checked = 0;
    for (let i = 0; i < b.length; i++)
        for (let c = 0; c < 3; c++) {
            if (Math.abs(b[i][c]) < 1e-12) continue;
            expect(t[i][c] / b[i][c]).toBeCloseTo(1, 8);
            checked++;
        }
    expect(checked).toBeGreaterThan(0);
});

test('saddle seam: a thrown ExternalSolveError propagates out of dispatchDescentStep', () => {
    const a = args();
    const boom: PreparedSaddle = {
        A: assembleAFlat(
            a.vertices,
            a.edges,
            a.disjointPairs,
            DEFAULTS.alpha,
            DEFAULTS.beta,
            DEFAULTS.epsilon,
        ),
        factor: () => ({
            kind: 'external',
            solve: () => {
                throw new ExternalSolveError('IR did not converge');
            },
        }),
    };
    // NOT swallowed into 'singular_system': the driver must be able to see it and
    // re-run the step on the CPU path (spec §4.5 fallback trigger 3).
    expect(() => dispatchDescentStep({ ...args(), saddle: boom })).toThrow(ExternalSolveError);
});

test('saddle seam: a throw does not leave the timing collector armed', () => {
    const a = args();
    const boom: PreparedSaddle = {
        A: assembleAFlat(
            a.vertices,
            a.edges,
            a.disjointPairs,
            DEFAULTS.alpha,
            DEFAULTS.beta,
            DEFAULTS.epsilon,
        ),
        factor: () => ({
            kind: 'external',
            solve: () => {
                throw new ExternalSolveError('IR did not converge');
            },
        }),
    };
    expect(() =>
        dispatchDescentStep({ ...args(), saddle: boom, collectTimings: true }),
    ).toThrow(ExternalSolveError);
    // If the collector stayed armed, THIS step's ledger would carry the previous
    // step's phases and `step.calls` would be 2. @see spec §4.4, D1.
    const next = dispatchDescentStep({ ...args(), collectTimings: true });
    expect(next.timings?.step?.calls).toBe(1);
});
```

Add `luSolve` to the `linsolve` import at the top of the file.

- [ ] **Step 8: Run it to verify it fails**

Run: `bun test test/gpu/saddleSeam.test.ts`
Expected: FAIL — `saddle` is not a known property of `DispatchDescentStepArgs`; nothing forwards it.

- [ ] **Step 9: Thread `saddle` through `gradient.ts`**

`solveConstrainedGradientSetFrozen` (`gradient.ts:84-115`) gains the TENTH positional parameter and the branch. `factorMode` already rides in this way (`optimizer.ts:352`); the function has no options object (nine positional params at `:84-94`).

```ts
export function solveConstrainedGradientSetFrozen(
    vertices: Vec3[],
    edges: Edge[],
    disjointPairs: number[][],
    alpha: number,
    beta: number,
    epsilon: number,
    dE: Vec3[],
    set: ConstraintSet,
    factorMode?: FactorMode,
    saddle?: PreparedSaddle,
): { gTilde: Vec3[]; lambda: number[]; residual: number; frozen: FrozenSaddleOperator } {
    const { C } = evaluateConstraintSet(set, vertices, edges);
    if (saddle) {
        // Source selection, never arithmetic (the Phase 1 `dE?` rule): A comes from
        // the caller instead of assembleAFlat, the factorization comes from the GPU
        // Cholesky, and the solve routes through the ALREADY-EXPORTED
        // solveSaddleFrozen (:882-905) so the residual stays the byte-identical
        // structured matvec the goldens gate — `solveFactored` and
        // `structuredSaddleResidual` are module-private and must stay that way.
        // C is evaluated by the CORE above, never by the driver.
        // @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4
        const frozen: FrozenSaddleOperator = {
            a: saddle.A,
            n: vertices.length,
            C,
            fac: saddle.factor(C),
        };
        const { x, lambda, residual } = timed('saddle', () =>
            solveSaddleFrozen(frozen, flatten(dE)),
        );
        return { gTilde: unflatten(x), lambda, residual, frozen };
    }
    const A = assembleAFlat(vertices, edges, disjointPairs, alpha, beta, epsilon);
    const { x, lambda, residual, fac } = timed('saddle', () =>
        solveSaddleFromA(A, vertices.length, C, flatten(dE), undefined, factorMode),
    );
    return {
        gTilde: unflatten(x),
        lambda,
        residual,
        frozen: { a: A, n: vertices.length, C, fac },
    };
}
```

Note the one structural change on the ABSENT path: `evaluateConstraintSet` moves above the `assembleAFlat` call. Both are pure functions of `(vertices, edges, set)` with no shared state, so the reordering is observationally inert — the Step 1 bit-identity test and the full suite are the proof, and the existing anchor comments on `assembleAFlat` and on the `[dE; 0]` rhs must be carried across unchanged.

Extend the import at `gradient.ts:17`:

```ts
import {
    type FactorMode,
    type FrozenSaddleOperator,
    type PreparedSaddle,
    solveSaddleFromA,
    solveSaddleFrozen,
} from './linsolve';
```

- [ ] **Step 10: Thread `saddle` through `optimizer.ts`**

Add to `SobolevStepOptions` (after `factorMode?` at `:174`):

```ts
    /**
     * A saddle solve prepared OUTSIDE this step for THESE vertices (WebGPU Phase 2a
     * seam). Selects the SOURCE of A and of the factorization, never the arithmetic;
     * absent ⇒ every path below is bit-identical (the golden suites are the backstop),
     * exactly like `dE?`. Its `solve` self-certifies or throws ExternalSolveError,
     * which this function deliberately does NOT convert to 'singular_system'.
     * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4
     */
    saddle?: PreparedSaddle;
```

Forward it at the call site (`:344-354`) as the tenth argument:

```ts
                const solved = solveConstrainedGradientSetFrozen(
                    vertices,
                    edges,
                    disjointPairs,
                    alpha,
                    beta,
                    epsilon,
                    dE,
                    set,
                    opts.factorMode,
                    opts.saddle,
                );
```

Make the bare catch at `:361` rethrow:

```ts
            } catch (e) {
                // The ONE throw this catch must not swallow: an external solve that
                // could not certify its result (spec §4.1 step 4). Converting it to
                // 'singular_system' would present an unconverged descent direction as
                // a rejected step, with usedGpuSolve still true — the driver has to see
                // it to re-run this step on the CPU path (spec §4.5 trigger 3).
                // Unreachable when `saddle` is absent; test/gpu/saddleSeam.test.ts
                // proves the absent path is still bit-identical WITH this branch here.
                // @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4
                if (e instanceof ExternalSolveError) throw e;
                // Exactly singular saddle system (e.g. an isolated vertex → zero Ā rows).
                // …existing body unchanged…
```

Apply D1's collector disarm around the `timed('step', …)` call at `:282`. Keep the existing `const outcome = timed('step', …)` body verbatim; only the wrapping changes.

Import `ExternalSolveError` and `type PreparedSaddle` from `./sobolev/linsolve`.

- [ ] **Step 11: Thread `saddle` through `dispatch.ts` and guard the worker**

`dispatch.ts` — add beside `dE?` at `:164`:

```ts
    /**
     * A saddle solve prepared for `vertices` (WebGPU Phase 2a seam). Forwarded to
     * `sobolevStepSet` exactly like `dE`. Carries closures ⇒ never cloneable ⇒ it
     * must not ride the worker payload.
     * @see SobolevStepOptions.saddle (optimizer.ts)
     */
    saddle?: PreparedSaddle;
```

and forward it beside `dE: args.dE` at `:270`:

```ts
            saddle: args.saddle,
```

`src/worker/solverWorker.ts` — first statement of `self.onmessage` (`:34`):

```ts
    // Defensive: PreparedSaddle carries closures and cannot be structured-cloned, so
    // it can only appear here if a future caller routed a 'gpu' step to the worker.
    // The 'gpu' driver is main-thread only ([P §2.1]); fail loudly rather than solve
    // with a silently missing factorization.
    // @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4
    if ('saddle' in (event.data as Record<string, unknown>))
        throw new Error('solverWorker: `saddle` cannot cross the worker boundary');
```

- [ ] **Step 12: Run the tests**

Run: `bun test test/gpu/saddleSeam.test.ts`
Expected: all five tests PASS.

- [ ] **Step 13: Run the full suite, the typechecker and the linter**

Run: `bunx tsc --noEmit && bun test && bunx biome check --write src/core src/worker test/gpu && bunx knip`
Expected: `tsc` exit 0; `bun test` = pre-branch pass count + 5, 0 failures; biome = pre-branch baseline (6 errors / 8 warnings at `bf0abcf`, none in files this task touched); `knip` exit 0.

- [ ] **Step 14: Commit**

```bash
git add src/core src/worker test/gpu/saddleSeam.test.ts
git commit -m "$(cat <<'EOF'
feat(core): saddle? seam through dispatch/optimizer/gradient, ExternalSolveError rethrow (phase2a T1)

Absent ⇒ bit-identical, tested rather than argued: the rethrow at optimizer.ts:361
and the throw-path timing-collector disarm are both on the default path.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AbdMPo3QLVT8dq89qjyeAo
EOF
)"
```

---

## Task 2: `src/gpu/saddle.ts` — the CPU f64 Schur/Woodbury/IR algebra

**Files:**
- Create: `src/gpu/saddle.ts`
- Test: `test/gpu/saddle.test.ts`

**Interfaces:**
- Consumes: `ExternalSolveError`, `PreparedSaddle`, `SaddleFactorization` (Task 1); `luSolve` (`linsolve.ts:33`, already exported).
- Produces:
  ```ts
  export const SADDLE_IR_TOL = 1e-10;
  export const SADDLE_IR_MAX_REFINEMENTS = 4;
  /** Lower-triangular Cholesky factor of a row-major SPD n×n matrix. Throws on a non-positive pivot. */
  export function choleskyF64(Asigma: Float64Array, n: number): Float64Array;
  /** σ = trace(A)/n and A_σ = A + (σ/n)·𝟙𝟙ᵀ, row-major, in a caller-owned buffer. */
  export function shiftSobolev(A: Float64Array, n: number, out?: Float64Array): { Asigma: Float64Array; sigma: number };
  export interface IrTrace { refinements: number; relResiduals: number[] }
  export function makePreparedSaddle(
      A: Float64Array, L: Float64Array, sigma: number, n: number,
      onTrace?: (t: IrTrace) => void,
  ): PreparedSaddle;
  ```

This file has **no `three` import** — it is plain TS and runs under `bun test`, which is what lets gate CR run before a line of WGSL exists (spec §5: "2a, CPU-only (`bun test`), BEFORE any WGSL").

- [ ] **Step 1: Write the failing Cholesky test**

Create `test/gpu/saddle.test.ts`:

```ts
import { expect, test } from 'bun:test';
import { trefoil } from '../../src/core/fixtures';
import { DEFAULTS } from '../../src/core/optimizer';
import { assembleAFlat } from '../../src/core/sobolev/innerProduct';
import { calculateDisjointPairs } from '../../src/core/tangentPointEnergy';
import { choleskyF64, shiftSobolev } from '../../src/gpu/saddle';

function sobolevA(n: number): { A: Float64Array; n: number } {
    const { vertices, edges } = trefoil(n);
    const dp = calculateDisjointPairs(edges);
    return {
        A: assembleAFlat(vertices, edges, dp, DEFAULTS.alpha, DEFAULTS.beta, DEFAULTS.epsilon),
        n,
    };
}

test('choleskyF64 reconstructs A_σ to f64 round-off', () => {
    const { A, n } = sobolevA(60);
    const { Asigma } = shiftSobolev(A, n);
    const L = choleskyF64(Asigma, n);
    let maxAbs = 0;
    let maxErr = 0;
    for (let i = 0; i < n; i++)
        for (let j = 0; j <= i; j++) {
            let s = 0;
            for (let k = 0; k <= j; k++) s += L[i * n + k] * L[j * n + k];
            maxErr = Math.max(maxErr, Math.abs(s - Asigma[i * n + j]));
            maxAbs = Math.max(maxAbs, Math.abs(Asigma[i * n + j]));
        }
    expect(maxErr / maxAbs).toBeLessThan(1e-12);
});

test('the shift is positive exactly on null(A): A_σ(𝟙/√n) = σ(𝟙/√n)', () => {
    // Spec §4.1 step 1 and §4.3's pivot-floor note both rest on this. A 𝟙 ≈ 0
    // (null(A) = span(𝟙), one-dimensional even for the disconnected presets, §2),
    // so A_σ 𝟙 = (σ/n)·𝟙𝟙ᵀ𝟙 = σ𝟙.
    const { A, n } = sobolevA(60);
    const { Asigma, sigma } = shiftSobolev(A, n);
    const one = new Float64Array(n).fill(1 / Math.sqrt(n));
    let maxDev = 0;
    for (let i = 0; i < n; i++) {
        let s = 0;
        for (let j = 0; j < n; j++) s += Asigma[i * n + j] * one[j];
        maxDev = Math.max(maxDev, Math.abs(s - sigma * one[i]));
    }
    expect(maxDev / sigma).toBeLessThan(1e-10);
});

test('choleskyF64 throws on a non-positive pivot (unshifted A is only PSD)', () => {
    const { A, n } = sobolevA(60);
    expect(() => choleskyF64(A, n)).toThrow(/non-positive pivot/);
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `bun test test/gpu/saddle.test.ts`
Expected: FAIL — `src/gpu/saddle.ts` does not exist.

- [ ] **Step 3: Write the shift and the factorization**

Create `src/gpu/saddle.ts`:

```ts
// src/gpu/saddle.ts — the CPU f64 half of the Phase 2a solve: shift, Schur, Woodbury,
// iterative refinement, on a factor produced elsewhere (the GPU Cholesky, or the f64
// reference used by gate CR). NO `three` import — this file must run under `bun test`
// so CR can gate the algebra BEFORE any WGSL exists (spec §5).
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1, §4.2, §5 (CR)
import {
    ExternalSolveError,
    luSolve,
    type PreparedSaddle,
    type SaddleFactorization,
} from '../core/sobolev/linsolve';

/**
 * IR stops at this relative residual — the same 1e-10 T4 gates on [P §3], so a
 * solve that returns has already met T4's bar against the f64 K.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1 step 4, §5 (T4)
 */
export const SADDLE_IR_TOL = 1e-10;

/**
 * Refinement budget. Measured ([K1CAL], spec §4.2): a LAPACK-quality f32 factor needs
 * z₀ + 3 at N=960, so N=960 has exactly ONE spare — a GPU factor 3× worse lands
 * exactly at this bound, which is why exceeding it is a typed throw and not a warning.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.2
 */
export const SADDLE_IR_MAX_REFINEMENTS = 4;

/**
 * σ = trace(A)/n and A_σ = A + (σ/n)·𝟙𝟙ᵀ (row-major, n×n).
 * Why this exact shift: null(A) = span(𝟙) and is ONE-dimensional at every N measured,
 * including the disconnected `crossing` and `linked-rings` presets (spec §2), so a
 * rank-1 shift is enough to make A_σ SPD — and it is positive exactly on the null
 * direction, `A_σ(𝟙/√n) = σ(𝟙/√n)`. It is removed EXACTLY by the Woodbury step, so it
 * does not perturb the metric (do not confuse with `buildSaddleMatrix`'s "no
 * regularizing identity" rule, `linsolve.ts:133-154`, which protects the metric).
 * `out` lets the caller reuse one buffer across steps (plan D6 — a per-step N×N
 * allocation is a measured P2a regression, not a style question).
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1 step 1
 */
export function shiftSobolev(
    A: Float64Array,
    n: number,
    out?: Float64Array,
): { Asigma: Float64Array; sigma: number } {
    let tr = 0;
    for (let i = 0; i < n; i++) tr += A[i * n + i];
    const sigma = tr / n;
    const s = sigma / n;
    const Asigma = out ?? new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) Asigma[i] = A[i] + s;
    return { Asigma, sigma };
}

/**
 * Lower-triangular Cholesky factor L of a row-major SPD n×n matrix, L Lᵀ = A_σ.
 * The f64 REFERENCE — gate CR runs the whole §4.1 reformulation on this factor so a
 * CR red localises to the algebra, never to the kernel; the GPU kernel's f32 result is
 * promoted into the same shape. Left-looking (the GPU kernel is right-looking, §4.3):
 * the op orders differ deliberately and CR's bar is κ-relative, so they must not be
 * "unified".
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1, §4.3, §5 (CR)
 */
export function choleskyF64(Asigma: Float64Array, n: number): Float64Array {
    const L = new Float64Array(n * n);
    for (let j = 0; j < n; j++) {
        let d = Asigma[j * n + j];
        for (let k = 0; k < j; k++) d -= L[j * n + k] * L[j * n + k];
        // Same guard the GPU `finalize` kernel applies (§4.3): `!(d > 0)` also
        // rejects NaN, which `d <= 0` would let through.
        if (!(d > 0) || !Number.isFinite(d))
            throw new Error(`choleskyF64: non-positive pivot ${d} at column ${j}`);
        const r = Math.sqrt(d);
        L[j * n + j] = r;
        for (let i = j + 1; i < n; i++) {
            let s = Asigma[i * n + j];
            for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
            L[i * n + j] = s / r;
        }
    }
    return L;
}
```

- [ ] **Step 4: Run the tests**

Run: `bun test test/gpu/saddle.test.ts`
Expected: all three PASS.

- [ ] **Step 5: Write the failing solve test**

Append to `test/gpu/saddle.test.ts`:

```ts
import { barycenterBlock, evaluateConstraintSet, totalLengthBlock } from '../../src/core/sobolev/constraintSet';
import { flatten } from '../../src/core/sobolev/layout';
import { solveSaddleFromA } from '../../src/core/sobolev/linsolve';
import { ExternalSolveError } from '../../src/core/sobolev/linsolve';
import { gradientAnalytical } from '../../src/core/tangentPointEnergy';
import { makePreparedSaddle } from '../../src/gpu/saddle';
import type { Vec3 } from '../../src/core/testConfigs';

function system(n: number) {
    const { vertices, edges } = trefoil(n);
    const dp = calculateDisjointPairs(edges);
    const { alpha, beta, epsilon } = DEFAULTS;
    const A = assembleAFlat(vertices, edges, dp, alpha, beta, epsilon);
    const x0: Vec3 = [0, 0, 0];
    for (const v of vertices) {
        x0[0] += v[0] / n;
        x0[1] += v[1] / n;
        x0[2] += v[2] / n;
    }
    const L0 = edges.reduce(
        (s, [a, b]) =>
            s + Math.hypot(
                vertices[a][0] - vertices[b][0],
                vertices[a][1] - vertices[b][1],
                vertices[a][2] - vertices[b][2],
            ),
        0,
    );
    const set = [barycenterBlock(x0), totalLengthBlock(L0)];
    const { C } = evaluateConstraintSet(set, vertices, edges);
    const dE = gradientAnalytical(vertices, edges, dp, alpha, beta, epsilon);
    return { vertices, edges, A, C, rhsTop: flatten(dE), n };
}

test('makePreparedSaddle with an EXACT f64 factor reproduces solveSaddleFromA', () => {
    const { A, C, rhsTop, n } = system(60);
    const { Asigma, sigma } = shiftSobolev(A, n);
    const L = choleskyF64(Asigma, n);
    const prepared = makePreparedSaddle(A, L, sigma, n);
    const z = prepared.factor(C).solve([...rhsTop, ...new Array(C.length).fill(0)]);
    const ref = solveSaddleFromA(A, n, C, rhsTop);
    const zref = [...ref.x, ...ref.lambda];
    let num = 0;
    let den = 0;
    for (let i = 0; i < z.length; i++) {
        num += (z[i] - zref[i]) ** 2;
        den += zref[i] ** 2;
    }
    // κ(K) ≈ 3.5e3 at N=60; the κ-relative floor of this comparison is ≈ κ·u_f64.
    expect(Math.sqrt(num / den)).toBeLessThan(1e-10);
});

test('makePreparedSaddle throws ExternalSolveError when IR cannot converge', () => {
    const { A, C, rhsTop, n } = system(60);
    const { Asigma, sigma } = shiftSobolev(A, n);
    const L = choleskyF64(Asigma, n);
    // A factor corrupted far past any backward error a real kernel produces: IR's
    // contraction ‖I − M⁻¹K‖ ≤ ε·κ exceeds 1, so it cannot converge at any budget.
    for (let i = 0; i < n; i++) L[i * n + i] *= 1 + 0.3 * ((i % 7) - 3);
    const prepared = makePreparedSaddle(A, L, sigma, n);
    expect(() =>
        prepared.factor(C).solve([...rhsTop, ...new Array(C.length).fill(0)]),
    ).toThrow(ExternalSolveError);
});

test('the IR trace records the convergence curve (spec §5 results JSON)', () => {
    const { A, C, rhsTop, n } = system(60);
    const { Asigma, sigma } = shiftSobolev(A, n);
    const L = choleskyF64(Asigma, n);
    let trace: { refinements: number; relResiduals: number[] } | null = null;
    const prepared = makePreparedSaddle(A, L, sigma, n, (t) => {
        trace = t;
    });
    prepared.factor(C).solve([...rhsTop, ...new Array(C.length).fill(0)]);
    expect(trace).not.toBeNull();
    expect(trace?.refinements).toBeLessThanOrEqual(4);
    expect(trace?.relResiduals.at(-1)).toBeLessThanOrEqual(1e-10);
});
```

- [ ] **Step 6: Run it to verify it fails**

Run: `bun test test/gpu/saddle.test.ts`
Expected: FAIL — `makePreparedSaddle` is not exported.

- [ ] **Step 7: Implement the Schur/Woodbury/IR core**

Append to `src/gpu/saddle.ts`. The layout is **coordinate-major** throughout: `Ā = I₃ ⊗ A_σ`, block `b` of a 3n vector is coordinate `b` over ALL vertices, occupying rows `[b·n, (b+1)·n)` — this is what `solveSaddleFromA` writes (`linsolve.ts:770-779`, `off = b * n`) and what the oracle uses (`oracle/tpe_stage1_oracle.py:245-248`). Writing `A ⊗ I₃` (vertex-major) here would be wrong and every test above would fail.

```ts
/** Forward/back substitution against L Lᵀ, one n-vector, in place into `out`. */
function solveWithL(L: Float64Array, n: number, rhs: ArrayLike<number>, out: Float64Array): void {
    for (let i = 0; i < n; i++) {
        let s = rhs[i];
        for (let k = 0; k < i; k++) s -= L[i * n + k] * out[k];
        out[i] = s / L[i * n + i];
    }
    for (let i = n - 1; i >= 0; i--) {
        let s = out[i];
        for (let k = i + 1; k < n; k++) s -= L[k * n + i] * out[k];
        out[i] = s / L[i * n + i];
    }
}

/** Â⁻¹ applied to a 3n vector — three INDEPENDENT n-backsolves, one per coordinate block. */
function applyAhatInv(L: Float64Array, n: number, v: ArrayLike<number>): Float64Array {
    const out = new Float64Array(3 * n);
    const buf = new Float64Array(n);
    const rhs = new Float64Array(n);
    for (let b = 0; b < 3; b++) {
        for (let i = 0; i < n; i++) rhs[i] = v[b * n + i];
        solveWithL(L, n, rhs, buf);
        for (let i = 0; i < n; i++) out[b * n + i] = buf[i];
    }
    return out;
}

/** IR diagnostics for one solve — the convergence curve spec §5's results JSON records. */
export interface IrTrace {
    refinements: number;
    relResiduals: number[];
}

/**
 * The Phase 2a external solve: spec §4.1 steps 2–4 on a factor produced elsewhere.
 * `A` is the UNSHIFTED f64 Sobolev matrix (the residuals of step 4 are measured against
 * the f64 K, which is what makes T4 hold as [P §3] wrote it); `L` is the lower-triangular
 * Cholesky factor of A_σ, f64-promoted from f32 on the GPU path; `sigma` is the shift the
 * factor was built with.
 *
 * Per-step cost (k = C.length): 3k+1 setup n-backsolves, then 3 per solve plus O(nk)
 * — spec §4.1 step 4. With P pins k = 4+3P and the setup is 13+9P (spec §7).
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1, §4.2
 */
export function makePreparedSaddle(
    A: Float64Array,
    L: Float64Array,
    sigma: number,
    n: number,
    onTrace?: (t: IrTrace) => void,
): PreparedSaddle {
    return {
        A,
        factor(C: number[][]): SaddleFactorization {
            const m = 3 * n;
            const k = C.length;
            const invSqrtN = 1 / Math.sqrt(n);

            // Setup, ONCE per step (the factor is frozen, [P §2.5]).
            // Z = Â⁻¹Cᵀ — 3k n-backsolves.
            const Z: Float64Array[] = C.map((row) => applyAhatInv(L, n, row));
            // S = C Z, k×k, SPD. κ(S) = 5.06…5.28 across N=120…960 (spec §4.1), so a
            // plain LU is ample; luSolve is the core's own, already gated.
            const S: number[][] = Array.from({ length: k }, (_, r) =>
                Array.from({ length: k }, (_, c) => {
                    let s = 0;
                    for (let i = 0; i < m; i++) s += C[r][i] * Z[c][i];
                    return s;
                }),
            );
            // W = K̂⁻¹U, three columns. Each column of P = I₃⊗(𝟙/√n) is supported on ONE
            // contiguous n-block, so Â⁻¹p_b needs exactly ONE n-backsolve of 𝟙/√n —
            // computed here and reused for all three coordinates (spec §4.1 step 3).
            const onesRhs = new Float64Array(n).fill(invSqrtN);
            const u = new Float64Array(n);
            solveWithL(L, n, onesRhs, u);
            const W: Float64Array[] = [];
            for (let b = 0; b < 3; b++) {
                const y = new Float64Array(m);
                for (let i = 0; i < n; i++) y[b * n + i] = u[i];
                // K̂⁻¹[p_b; 0] through the same Schur machinery: λ = S⁻¹(C y), x = y − Zλ.
                const Cy = C.map((row) => {
                    let s = 0;
                    for (let i = 0; i < m; i++) s += row[i] * y[i];
                    return s;
                });
                const lam = luSolve(S, Cy);
                const w = new Float64Array(m + k);
                for (let i = 0; i < m; i++) {
                    let s = y[i];
                    for (let r = 0; r < k; r++) s -= Z[r][i] * lam[r];
                    w[i] = s;
                }
                for (let r = 0; r < k; r++) w[m + r] = lam[r];
                W.push(w);
            }
            // M = σ⁻¹I₃ − UᵀW, 3×3. HEAVILY CANCELLING: measured diag(M) is 9.2e4×
            // smaller than σ⁻¹ at N=960 (spec §7), so its formation destroys ~5 decimal
            // digits and the loss grows ∝ κ. Form and invert it in f64, NEVER f32, and
            // NEVER on the GPU — a hard constraint that also binds slice 2c (§4.7).
            const M: number[][] = Array.from({ length: 3 }, (_, a) =>
                Array.from({ length: 3 }, (_, b) => {
                    let s = 0;
                    for (let i = 0; i < n; i++) s += invSqrtN * W[b][a * n + i];
                    return (a === b ? 1 / sigma : 0) - s;
                }),
            );

            /** ẑ = K̂⁻¹r via Schur, then Woodbury back to K⁻¹r. Spec §4.1 steps 2–3. */
            const applyKinv = (rhs: ArrayLike<number>): Float64Array => {
                const y = applyAhatInv(L, n, rhs);
                const rhsBottom = new Array<number>(k);
                for (let r = 0; r < k; r++) rhsBottom[r] = rhs[m + r];
                const Cy = C.map((row, r) => {
                    let s = 0;
                    for (let i = 0; i < m; i++) s += row[i] * y[i];
                    return s - rhsBottom[r];
                });
                const lam = luSolve(S, Cy);
                const xh = new Float64Array(m);
                for (let i = 0; i < m; i++) {
                    let s = y[i];
                    for (let r = 0; r < k; r++) s -= Z[r][i] * lam[r];
                    xh[i] = s;
                }
                // Woodbury: z = ẑ + W M⁻¹ (Pᵀx̂). UᵀK̂⁻¹r = Uᵀẑ = Pᵀx̂ because U = [P; 0].
                const Ptx = [0, 0, 0];
                for (let a = 0; a < 3; a++) {
                    let s = 0;
                    for (let i = 0; i < n; i++) s += invSqrtN * xh[a * n + i];
                    Ptx[a] = s;
                }
                const mu = luSolve(M, Ptx);
                const z = new Float64Array(m + k);
                for (let i = 0; i < m; i++) {
                    let s = xh[i];
                    for (let a = 0; a < 3; a++) s += W[a][i] * mu[a];
                    z[i] = s;
                }
                for (let r = 0; r < k; r++) {
                    let s = lam[r];
                    for (let a = 0; a < 3; a++) s += W[a][m + r] * mu[a];
                    z[m + r] = s;
                }
                return z;
            };

            /** r = rhs − K z against the f64 A and C — K is never materialized. */
            const residualOf = (rhs: ArrayLike<number>, z: Float64Array): Float64Array => {
                const r = new Float64Array(m + k);
                for (let b = 0; b < 3; b++)
                    for (let i = 0; i < n; i++) {
                        let s = 0;
                        for (let j = 0; j < n; j++) s += A[i * n + j] * z[b * n + j];
                        r[b * n + i] = s;
                    }
                for (let q = 0; q < k; q++) {
                    const lamq = z[m + q];
                    for (let i = 0; i < m; i++) r[i] += C[q][i] * lamq;
                }
                for (let q = 0; q < k; q++) {
                    let s = 0;
                    for (let i = 0; i < m; i++) s += C[q][i] * z[i];
                    r[m + q] = s;
                }
                for (let i = 0; i < m + k; i++) r[i] = rhs[i] - r[i];
                return r;
            };

            return {
                kind: 'external',
                solve(rhs: number[]): number[] {
                    let rhsNorm = 0;
                    for (let i = 0; i < rhs.length; i++) rhsNorm += rhs[i] * rhs[i];
                    rhsNorm = Math.sqrt(rhsNorm);
                    const z = applyKinv(rhs);
                    const relResiduals: number[] = [];
                    // A zero rhs has a zero solution and no relative residual to measure;
                    // returning here avoids 0/0 becoming a spurious ExternalSolveError.
                    if (rhsNorm === 0) {
                        onTrace?.({ refinements: 0, relResiduals: [0] });
                        return Array.from(z);
                    }
                    for (let it = 0; ; it++) {
                        const r = residualOf(rhs, z);
                        let rn = 0;
                        for (let i = 0; i < r.length; i++) rn += r[i] * r[i];
                        const rel = Math.sqrt(rn) / rhsNorm;
                        relResiduals.push(rel);
                        if (rel <= SADDLE_IR_TOL) {
                            onTrace?.({ refinements: it, relResiduals });
                            return Array.from(z);
                        }
                        if (it >= SADDLE_IR_MAX_REFINEMENTS) {
                            onTrace?.({ refinements: it, relResiduals });
                            // The ONLY failure detectable during the step. Returning an
                            // unconverged z is a contract violation, not a degraded mode:
                            // nothing in src/core/ thresholds `residual`, so the line
                            // search would reject the wrong g̃ as `armijo_failed` and the
                            // run would auto-pause with usedGpuSolve still true.
                            // @see spec §4.1 step 4, §4.5 (fallback triggers 2 and 3)
                            throw new ExternalSolveError(
                                `saddle IR: relative residual ${rel.toExponential(3)} after ` +
                                    `${SADDLE_IR_MAX_REFINEMENTS} refinements (bar ${SADDLE_IR_TOL})`,
                            );
                        }
                        const d = applyKinv(r);
                        for (let i = 0; i < z.length; i++) z[i] += d[i];
                    }
                },
            };
        },
    };
}
```

- [ ] **Step 8: Run the tests**

Run: `bun test test/gpu/saddle.test.ts`
Expected: all six PASS.

- [ ] **Step 9: Typecheck, lint, full suite, commit**

Run: `bunx tsc --noEmit && bun test && bunx biome check --write src/gpu/saddle.ts test/gpu/saddle.test.ts && bunx knip`
Expected: `tsc` exit 0; suite green; biome baseline; knip exit 0.

```bash
git add src/gpu/saddle.ts test/gpu/saddle.test.ts
git commit -m "$(cat <<'EOF'
feat(gpu): src/gpu/saddle.ts — shift + Schur + Woodbury + IR on a supplied factor (phase2a T2)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AbdMPo3QLVT8dq89qjyeAo
EOF
)"
```

---

## Task 3: Gate CR — the reformulation is exact, proven before any WGSL

**Files:**
- Create: `test/gpu/cr.test.ts`
- Create: `oracle/tpe_phase2a_reformulation.py`
- Create: `bench/results/<today>-gpu-phase2a-cr.json` (written by the test run)
- Test: `test/gpu/cr.test.ts`

**Interfaces:**
- Consumes: `shiftSobolev`, `choleskyF64`, `makePreparedSaddle` (Task 2); `solveSaddleFromA` (`linsolve.ts:742`).
- Produces: `bench/results/<date>-gpu-phase2a-cr.json` — `{ gitShaShort, fixtures: [{ name, n, k, relErr, bar, verdict, diagM, kappaM, minDiagLSq }] }`, consumed by Task 7's README report.

**Gate CR, verbatim from spec §5:** quantity `‖Δz‖/‖z‖` between z from §4.1 steps 1–3 on a CPU **f64** Cholesky and `solveSaddleFromA`'s z. Bar `max(1e-12, 10·κ₂(K,N)·u_f64)` = **1.6e-12 / 1.6e-11 / 1.6e-10 / 1.6e-9** at N=120/240/480/960, κ₂ from the committed [G6] table, `u_f64 = 2⁻⁵³`. Fixtures: `crossing` N=8; trefoil N=60/120/240/480/**960**; sets: barycenter+totalLength, and +1 pin. CPU-only (`bun test`), **BEFORE any WGSL**. **CR red ⇒ no GPU work starts** (spec §6). CR also records `diag(M)`, `κ(M)` and `min diag(L)²` per fixture so the Woodbury conditioning margin of spec §7 is a number, not a pass/fail.

- [ ] **Step 1: Write the failing CR test**

Create `test/gpu/cr.test.ts`. The bar is κ-relative because the quantity has a floor of ≈κ₂(K)·u_f64 and its own reference `solveSaddleFromA` is only that accurate — a flat 1e-10 is 1160× loose at N=120 and crossed at N≈1200 (spec §5).

```ts
import { expect, test } from 'bun:test';
import { writeFileSync } from 'node:fs';
import { execSync } from 'node:child_process';
import { trefoil } from '../../src/core/fixtures';
import { DEFAULTS } from '../../src/core/optimizer';
import {
    barycenterBlock,
    evaluateConstraintSet,
    pointBlock,
    totalLengthBlock,
} from '../../src/core/sobolev/constraintSet';
import { flatten } from '../../src/core/sobolev/layout';
import { assembleAFlat } from '../../src/core/sobolev/innerProduct';
import { solveSaddleFromA } from '../../src/core/sobolev/linsolve';
import { calculateDisjointPairs, gradientAnalytical } from '../../src/core/tangentPointEnergy';
import { crossing } from '../../src/core/testConfigs';
import { choleskyF64, makePreparedSaddle, shiftSobolev } from '../../src/gpu/saddle';
import type { Vec3 } from '../../src/core/testConfigs';

const U_F64 = 2 ** -53;
/**
 * κ₂(K) from the COMMITTED [G6] sweep (`bench/results/2026-08-13-gpu-phase0-g6.json`),
 * cross-checked to 6 significant figures by [K1CAL]. N=8 and N=60 are below [G6]'s
 * smallest sampled N, so their bar falls back to the 1e-12 floor of
 * `max(1e-12, 10·κ·u_f64)` — which is what the floor is FOR.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §5 (CR)
 */
const KAPPA_K: Record<number, number> = {
    120: 1.425e3,
    240: 1.398e4,
    480: 1.405e5,
    960: 1.418e6,
};

function crBar(n: number): number {
    const kappa = KAPPA_K[n];
    return kappa === undefined ? 1e-12 : Math.max(1e-12, 10 * kappa * U_F64);
}
```

- [ ] **Step 2: Add the fixture builder and the measurement**

Append to `test/gpu/cr.test.ts`:

```ts
interface CrRow {
    name: string;
    n: number;
    k: number;
    relErr: number;
    bar: number;
    verdict: 'PASS' | 'FAIL';
    diagM: number[];
    kappaM: number;
    minDiagLSq: number;
}

const rows: CrRow[] = [];

function measure(
    name: string,
    vertices: Vec3[],
    edges: [number, number][],
    withPin: boolean,
): CrRow {
    const n = vertices.length;
    const dp = calculateDisjointPairs(edges);
    const { alpha, beta, epsilon } = DEFAULTS;
    const A = assembleAFlat(vertices, edges, dp, alpha, beta, epsilon);
    const x0: Vec3 = [0, 0, 0];
    for (const v of vertices) {
        x0[0] += v[0] / n;
        x0[1] += v[1] / n;
        x0[2] += v[2] / n;
    }
    const L0 = edges.reduce(
        (s, [a, b]) =>
            s + Math.hypot(
                vertices[a][0] - vertices[b][0],
                vertices[a][1] - vertices[b][1],
                vertices[a][2] - vertices[b][2],
            ),
        0,
    );
    const set = withPin
        ? [barycenterBlock(x0), totalLengthBlock(L0), pointBlock(0, vertices[0])]
        : [barycenterBlock(x0), totalLengthBlock(L0)];
    const { C } = evaluateConstraintSet(set, vertices, edges);
    const rhsTop = flatten(gradientAnalytical(vertices, edges, dp, alpha, beta, epsilon));

    const { Asigma, sigma } = shiftSobolev(A, n);
    const L = choleskyF64(Asigma, n);
    const prepared = makePreparedSaddle(A, L, sigma, n);
    const z = prepared.factor(C).solve([...rhsTop, ...new Array(C.length).fill(0)]);
    const ref = solveSaddleFromA(A, n, C, rhsTop);
    const zref = [...ref.x, ...ref.lambda];

    let num = 0;
    let den = 0;
    for (let i = 0; i < z.length; i++) {
        num += (z[i] - zref[i]) ** 2;
        den += zref[i] ** 2;
    }
    const relErr = Math.sqrt(num / den);

    // Spec §7: diag(M) is 9.2e4× smaller than σ⁻¹ at N=960 and the loss grows ∝ κ, so
    // the margin is RECORDED as a number rather than asserted — it is what makes f32
    // formation of M impossible from N=480 on, which is a hard constraint on slice 2c.
    const { diagM, kappaM } = probeWoodbury(A, L, sigma, n, C);
    let minDiagLSq = Number.POSITIVE_INFINITY;
    for (let i = 0; i < n; i++) minDiagLSq = Math.min(minDiagLSq, L[i * n + i] ** 2);

    const bar = crBar(n);
    const row: CrRow = {
        name,
        n,
        k: C.length,
        relErr,
        bar,
        verdict: relErr <= bar ? 'PASS' : 'FAIL',
        diagM,
        kappaM,
        minDiagLSq,
    };
    rows.push(row);
    return row;
}
```

`probeWoodbury` re-forms `M` exactly as `makePreparedSaddle` does and returns `diag(M)` plus `κ₂(M)` from its 3×3 singular values; write it beside `measure` in the same file (it is test-only diagnostics, not production code, so it does not belong in `src/gpu/saddle.ts`).

- [ ] **Step 3: Add the gate assertions**

```ts
const FIXTURES: { name: string; make: () => { vertices: Vec3[]; edges: [number, number][] } }[] = [
    { name: 'crossing8', make: () => crossing.generate({ ...crossing.defaults, count: 8 }) },
    { name: 'trefoil60', make: () => trefoil(60) },
    { name: 'trefoil120', make: () => trefoil(120) },
    { name: 'trefoil240', make: () => trefoil(240) },
    { name: 'trefoil480', make: () => trefoil(480) },
    { name: 'trefoil960', make: () => trefoil(960) },
];

for (const f of FIXTURES)
    for (const withPin of [false, true]) {
        const label = `${f.name}${withPin ? '+pin' : ''}`;
        test(`CR: ${label} — §4.1 reformulation matches solveSaddleFromA within the κ-relative bar`, () => {
            const { vertices, edges } = f.make();
            const row = measure(label, vertices, edges, withPin);
            expect(row.relErr).toBeLessThanOrEqual(row.bar);
        });
    }

test('CR: write the committed results JSON', () => {
    const sha = execSync('git rev-parse --short HEAD').toString().trim();
    const date = new Date().toISOString().slice(0, 10);
    writeFileSync(
        `bench/results/${date}-gpu-phase2a-cr.json`,
        `${JSON.stringify({ gate: 'CR', gitShaShort: sha, uF64: U_F64, fixtures: rows }, null, 2)}\n`,
    );
    expect(rows.length).toBe(FIXTURES.length * 2);
});
```

The write test runs last because `bun test` executes `test()` blocks in declaration order within a file.

- [ ] **Step 4: Run it — a red here STOPS the branch**

Run: `bun test test/gpu/cr.test.ts`
Expected: 13 PASS. **If any fixture fails, do not proceed to Task 4.** Spec §6: "CR red → the reformulation is wrong; fix the math — no GPU work starts until CR is green." Localise with the numpy twin (next step) before changing `src/gpu/saddle.ts`.

Note N=960 makes `assembleAFlat` + a dense 960×960 Cholesky + a (2884)² reference LU; expect this file to take minutes, not seconds.

- [ ] **Step 5: Write the numpy twin**

Create `oracle/tpe_phase2a_reformulation.py` — an independent implementation of spec §4.1 steps 1–3 in numpy, over the same trefoil parametrization, importing (not copying) `oracle/tpe_stage1_oracle.py`'s `assemble_inner_product` and `barycenter_phi_and_C`, exactly as `bench/gpu/k1-calibration.py` does. It prints `‖Δz‖/‖z‖` against `numpy.linalg.solve` on the assembled K at N ∈ {60, 120, 240, 480}. Its purpose is localisation: if `test/gpu/cr.test.ts` is red and the twin is green, the defect is in the TS; if both are red, it is in the algebra of spec §4.1 and the spec is wrong.

Run: `uv run --with numpy --with scipy python oracle/tpe_phase2a_reformulation.py`
Expected: every N reports a relative error at the κ·u_f64 floor.

- [ ] **Step 6: Commit**

```bash
git add test/gpu/cr.test.ts oracle/tpe_phase2a_reformulation.py bench/results/*-gpu-phase2a-cr.json
git commit -m "$(cat <<'EOF'
test(gpu): gate CR — Schur/Woodbury reformulation exact vs solveSaddleFromA, κ-relative bar (phase2a T3)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AbdMPo3QLVT8dq89qjyeAo
EOF
)"
```

---

## Task 4: `src/gpu/cholesky.ts` — the f32 GPU Cholesky kernel and its engine

**Files:**
- Create: `src/gpu/cholesky.ts`
- Test: `test/gpu/cholesky.test.ts` (pure parts only — WGSL source, the triangular index, `gpuSolveMaxN`, meta encoding)

**Interfaces:**
- Consumes: `shiftSobolev` (Task 2).
- Produces:
  ```ts
  export const CHOLESKY_FALLBACK_MAX_N = 4096;
  /** floor(sqrt(device.limits.maxStorageBufferBindingSize / 4)); 4096 if the limit is unreadable. */
  export function gpuSolveMaxN(renderer: WebGPURenderer): number;
  /** Pure mirror of the WGSL triangular-index inverse: t ↦ (i, k) with 0 ≤ k ≤ i. */
  export function triIndex(t: number): { i: number; k: number };
  export function colFactorWgsl(): string;
  export function trailingUpdateWgsl(): string;
  export function finalizeWgsl(): string;
  export class GpuCholesky {
      constructor(renderer: WebGPURenderer, n: number);
      /** Uploads A_σ, runs 3n dispatches in ONE batch, reads L + status back. */
      factor(Asigma: Float64Array): Promise<{ L: Float64Array; status: number; wallMs: number }>;
      dispose(): void;
  }
  ```

### D9 — the trailing update's thread↦(i,k) mapping is triangular, and its `sqrt` needs a correction

Spec §4.3 prices the kernel at **N³/2 = 4.4e8** thread launches at N=960, not N³ — i.e. `n(n+1)/2` threads per dispatch, over the LOWER TRIANGLE, with `k > j` as the per-thread guard (the dispatch size cannot shrink per column: `WebGPUBackend.compute` caches the dispatch size keyed on the NODE OBJECT, `:1652-1685`, and the same node object is repeated n times). Inverting the triangular index needs `i = floor((√(8t+1) − 1)/2)`, and WGSL's `sqrt` is **f32**: at t ≈ 4.6e5 the argument is ≈3.7e6 — exactly representable — but the rounded root can land one row off at a boundary. The kernel therefore carries a two-sided correction, and `triIndex` is its pure TS mirror so the correction is unit-tested against brute force without a GPU.

- [ ] **Step 1: Write the failing pure-parts test**

Create `test/gpu/cholesky.test.ts`:

```ts
import { expect, test } from 'bun:test';
import {
    CHOLESKY_FALLBACK_MAX_N,
    colFactorWgsl,
    finalizeWgsl,
    trailingUpdateWgsl,
    triIndex,
} from '../../src/gpu/cholesky';

test('triIndex inverts the lower-triangular packing exactly', () => {
    // Brute force the whole map for n = 256 (32 896 entries): every t must land on the
    // (i, k) the packing t = i(i+1)/2 + k defines, with 0 <= k <= i.
    const n = 256;
    let t = 0;
    for (let i = 0; i < n; i++)
        for (let k = 0; k <= i; k++) {
            const got = triIndex(t);
            expect(got.i).toBe(i);
            expect(got.k).toBe(k);
            t++;
        }
    expect(t).toBe((n * (n + 1)) / 2);
});

test('triIndex survives the f32-sqrt boundary at the N=960 thread count', () => {
    // The kernel's own domain: t up to n(n+1)/2 - 1 at n = 960. Row starts and row ends
    // are where an f32 sqrt rounds across a boundary.
    const n = 960;
    for (let i = 0; i < n; i++) {
        const start = (i * (i + 1)) / 2;
        expect(triIndex(start)).toEqual({ i, k: 0 });
        expect(triIndex(start + i)).toEqual({ i, k: i });
    }
});

test('each WGSL source declares exactly one bare fn (three parses one per wgslFn)', () => {
    // WGSLNodeFunction.js:4 matches `^fn name(...)`; a second bare fn is a parse failure.
    for (const src of [colFactorWgsl(), trailingUpdateWgsl(), finalizeWgsl()])
        expect(src.match(/^fn\s+\w+\s*\(/gm)?.length).toBe(1);
});

test('the fallback bound is the documented 4096', () => {
    expect(CHOLESKY_FALLBACK_MAX_N).toBe(4096);
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `bun test test/gpu/cholesky.test.ts`
Expected: FAIL — `src/gpu/cholesky.ts` does not exist.

- [ ] **Step 3: Write the pure parts**

Create `src/gpu/cholesky.ts` (the WGSL generators, `triIndex`, `gpuSolveMaxN`):

```ts
// src/gpu/cholesky.ts — the f32 GPU Cholesky of the shifted Sobolev block A_σ.
// Unblocked right-looking: three kernels per column in ONE renderer.compute([...])
// batch, the column index carried by a GPU-side counter (spec §4.3 — a uniform shared
// across a batched array would present its LAST value to every dispatch).
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.3, §5 (K1, Boot)
import { attributeArray, instancedArray, instanceIndex, uint, wgslFn } from 'three/tsl';
import type { StorageBufferNode, WebGPURenderer } from 'three/webgpu';
import * as THREE from 'three/webgpu';

/**
 * Bound used when the device limit is unreadable. NOT a claim about the hardware —
 * spec §4.3: `rg 4096 src/` was empty before this line, i.e. the first draft's
 * "N ≤ 4096 by construction" described no construction that existed.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.3
 */
export const CHOLESKY_FALLBACK_MAX_N = 4096;

/**
 * Largest N whose N×N f32 A_σ fits one storage-buffer binding.
 * **The DEVICE limit, not the adapter's, and they are not interchangeable** (spec §4.3):
 * three requests the device with `requiredLimits: parameters.requiredLimits`
 * (`WebGPUBackend.js:243`) defaulted to `{}` (`:93`), and a limit absent from
 * `requiredLimits` takes WebGPU's DEFAULT (128 MiB) rather than the adapter's supported
 * maximum — so this returns 5792 here, against 23170 had it been sized from a 2 GiB
 * adapter: 4× too large in N, 16× in bytes, reinstating the buffer-creation validation
 * error this function exists to avoid. The adapter is also unreachable: it is a LOCAL in
 * `WebGPUBackend.js:217` and never stored, and `Viewer.tsx:110` forbids a second
 * `requestAdapter()`.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.3
 */
export function gpuSolveMaxN(renderer: WebGPURenderer): number {
    const limits = (
        renderer.backend as unknown as {
            device?: { limits?: { maxStorageBufferBindingSize?: number } };
        }
    ).device?.limits;
    const bytes = limits?.maxStorageBufferBindingSize;
    if (typeof bytes !== 'number' || !Number.isFinite(bytes) || bytes <= 0)
        return CHOLESKY_FALLBACK_MAX_N;
    return Math.floor(Math.sqrt(bytes / 4));
}

/**
 * Pure mirror of the WGSL trailing-update index inverse: t ↦ (i, k) over the packing
 * `t = i(i+1)/2 + k`, `0 ≤ k ≤ i`. Kept in TS so the f32-sqrt correction (D9) is
 * unit-tested against brute force without a GPU. Keep the two correction loops
 * identical to the WGSL below — they are the reason the mapping is exact at the
 * row boundaries where an f32 root rounds across.
 * @see docs/superpowers/plans/2026-09-03-webgpu-solver-phase2a.md (D9)
 */
export function triIndex(t: number): { i: number; k: number } {
    let i = Math.floor((Math.sqrt(8 * t + 1) - 1) * 0.5);
    while (i > 0 && (i * (i + 1)) / 2 > t) i -= 1;
    while (((i + 1) * (i + 2)) / 2 <= t) i += 1;
    return { i, k: t - (i * (i + 1)) / 2 };
}

/**
 * Column scale: thread per row i > j does `A_ij /= √A_jj`. `A_jj` is still the RAW
 * pivot d_j here — `finalize` writes √d_j after the trailing update, which never
 * touches row/column j. Bailing out on a non-positive pivot stops NaN from being
 * smeared across the trailing block before `finalize` can record the failing column.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.3
 */
export function colFactorWgsl(): string {
    return `fn cholColFactor(
    a: ptr<storage, array<f32>, read_write>,
    meta: ptr<storage, array<u32>, read_write>,
    n: u32,
    I: u32
) -> void {
    let j = (*meta)[0];
    if ( I <= j || I >= n ) { return; }
    let d = (*a)[ j * n + j ];
    if ( !( d > 0.0 ) ) { return; }
    (*a)[ I * n + j ] = (*a)[ I * n + j ] / sqrt( d );
}`;
}

/**
 * Trailing update: thread per (i, k) with j < k ≤ i, INCLUDING i = k so the trailing
 * DIAGONAL is updated (it is the next column's pivot). `A_ik −= L_ij·L_kj`; row and
 * column j are never touched. The dispatch size is fixed at the j = 0 worst case —
 * `n(n+1)/2` threads for EVERY column — because `WebGPUBackend.compute` caches the
 * dispatch size on the NODE OBJECT (`:1652-1685`) and the same node is repeated n
 * times; `k > j` is the thread's GUARD, not a shrinking dispatch. That 3× waste is
 * inside the spec's factor-wall estimate. Do not "optimize" it away without the
 * IndirectStorageBufferAttribute route (§4.3), which is pre-registered and not built.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.3, §7
 */
export function trailingUpdateWgsl(): string {
    return `fn cholTrailingUpdate(
    a: ptr<storage, array<f32>, read_write>,
    meta: ptr<storage, array<u32>, read_write>,
    n: u32,
    tri: u32,
    I: u32
) -> void {
    if ( I >= tri ) { return; }
    let j = (*meta)[0];
    var i : u32 = u32( floor( ( sqrt( 8.0 * f32( I ) + 1.0 ) - 1.0 ) * 0.5 ) );
    loop { if ( i > 0u && i * ( i + 1u ) / 2u > I ) { i = i - 1u; } else { break; } }
    loop { if ( ( i + 1u ) * ( i + 2u ) / 2u <= I ) { i = i + 1u; } else { break; } }
    let k = I - i * ( i + 1u ) / 2u;
    if ( k <= j || i >= n ) { return; }
    (*a)[ i * n + k ] = (*a)[ i * n + k ] - (*a)[ i * n + j ] * (*a)[ k * n + j ];
}`;
}

/**
 * ONE thread: reads the raw pivot d_j, records the FIRST failing column in the status
 * slot as `1 + j` (so a K1 failure is diagnosable without a second readback), writes
 * `A_jj = √d_j`, and advances the column counter. `!(d > 0.0)` already rejects NaN
 * (every comparison with NaN is false); the upper bound rejects +∞, which `> 0` admits.
 * First failure wins — a later column would otherwise overwrite the diagnosis with
 * garbage produced BY the first failure.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.3
 */
export function finalizeWgsl(): string {
    return `fn cholFinalize(
    a: ptr<storage, array<f32>, read_write>,
    meta: ptr<storage, array<u32>, read_write>,
    n: u32,
    I: u32
) -> void {
    if ( I != 0u ) { return; }
    let j = (*meta)[0];
    let d = (*a)[ j * n + j ];
    let ok = d > 0.0 && d < 3.4028234e38;
    if ( !ok && (*meta)[1] == 0u ) { (*meta)[1] = 1u + j; }
    (*a)[ j * n + j ] = sqrt( d );
    (*meta)[0] = j + 1u;
}`;
}
```

- [ ] **Step 4: Run the pure test**

Run: `bun test test/gpu/cholesky.test.ts`
Expected: 4 PASS.

- [ ] **Step 5: Write the `GpuCholesky` engine**

Append to `src/gpu/cholesky.ts`. Note the readback target: **one reused `THREE.ReadbackBuffer` per engine** (D7) — `getArrayBufferAsync(attr, null)` allocates and destroys a 4 MB staging buffer per step, which is the cost spec §4.3 avoids; `release()` before reuse is mandatory (`WebGPUAttributeUtils.js:401` throws otherwise).

```ts
/** Workgroup size — one shape for every solver dispatch, as `src/gpu/reduce.ts` documents. */
const CHOLESKY_LANES = 64;

/**
 * Owns the A_σ buffer, the 2-slot u32 meta buffer `[counter, status]`, and the 3n
 * compute nodes for one N. New N ⇒ new instance. `factor` is the ONE upload + ONE
 * batch + ONE readback per step (spec §2.5, §4.3); `wallMs` covers all three, which is
 * what the P2a factor-wall bar counts.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.3, §4.5, §5
 */
export class GpuCholesky {
    private readonly n: number;
    private readonly renderer: WebGPURenderer;
    private readonly a: StorageBufferNode<'float'>;
    private readonly meta: StorageBufferNode<'uint'>;
    private readonly nodes: unknown[];
    private readonly readback: THREE.ReadbackBuffer;
    private readonly staging: Float32Array;
    private disposed = false;

    constructor(renderer: WebGPURenderer, n: number) {
        if (n < 1) throw new Error('GpuCholesky: n must be >= 1');
        if (n > gpuSolveMaxN(renderer))
            throw new Error(`GpuCholesky: n=${n} exceeds GPU_SOLVE_MAX_N=${gpuSolveMaxN(renderer)}`);
        this.renderer = renderer;
        this.n = n;
        // A_σ is BOTH uploaded per step and GPU-written, so it stays read_write — no
        // `.toReadOnly()`, unlike Phase 1's uploaded positions buffer (spec §4.3).
        this.a = attributeArray(new Float32Array(n * n), 'float');
        this.a.value.setUsage(THREE.DynamicDrawUsage);
        this.meta = instancedArray(2, 'uint');
        // D6: ONE staging buffer for the life of the instance. A per-step N×N allocation
        // is what [JSALG]'s 10.7× `churn` spread measures, and it lands in the
        // neighbouring backsolve/matvec timings too.
        this.staging = new Float32Array(n * n);
        this.readback = new THREE.ReadbackBuffer(n * n * 4);

        const tri = (n * (n + 1)) / 2;
        const colFn = wgslFn(colFactorWgsl());
        const trailFn = wgslFn(trailingUpdateWgsl());
        const finFn = wgslFn(finalizeWgsl());
        const common = { a: this.a, meta: this.meta, n: uint(n), I: instanceIndex };
        // THREE node objects, repeated n times — 3 node builds total, not 3n shader
        // compiles. `j` reaches the shader from meta[0], which `finalize` increments:
        // a uniform shared across a batched array would present its LAST written value
        // to EVERY dispatch (`Renderer.js:2718-2817` calls updateForCompute per node and
        // submits once). The inter-dispatch visibility this needs is guaranteed — each
        // dispatch is its own synchronization scope (gpuweb/gpuweb#4434) — and Phase 1's
        // shipped two-pass reduction already depends on it (`src/gpu/reduce.ts:66-67,116`).
        // @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.3
        const colNode = colFn(common).compute(n, [CHOLESKY_LANES]);
        const trailNode = trailFn({ ...common, tri: uint(tri) }).compute(tri, [CHOLESKY_LANES]);
        const finNode = finFn(common).compute(1, [1]);
        const nodes: unknown[] = [];
        for (let j = 0; j < n; j++) nodes.push(colNode, trailNode, finNode);
        this.nodes = nodes;
    }

    /**
     * Factors A_σ in place on the GPU and reads the result back. On return the lower
     * triangle of `L` holds the factor and the strict upper triangle is garbage — the
     * caller promotes the lower triangle to f64 and never reads above the diagonal.
     * `status` is 0 on success and `1 + j` for the first non-positive / non-finite
     * pivot at column j; a nonzero status means the CPU path serves this step and the
     * factor must NOT reach `makePreparedSaddle` (never a NaN-laden L into the algebra).
     */
    async factor(Asigma: Float64Array): Promise<{ L: Float64Array; status: number; wallMs: number }> {
        if (this.disposed) throw new Error('GpuCholesky: disposed');
        const n = this.n;
        if (Asigma.length !== n * n) throw new Error('GpuCholesky: A_σ size mismatch');
        const t0 = performance.now();
        for (let i = 0; i < n * n; i++) this.staging[i] = Asigma[i];
        (this.a.value.array as Float32Array).set(this.staging);
        this.a.value.needsUpdate = true;
        (this.meta.value.array as Uint32Array).set([0, 0]);
        this.meta.value.needsUpdate = true;
        this.renderer.compute(this.nodes as never); // 3n dispatches, ONE submit (spec §4.3)
        const meta = new Uint32Array(await this.renderer.getArrayBufferAsync(this.meta.value));
        const status = meta[1];
        await this.renderer.getArrayBufferAsync(this.a.value, this.readback);
        const flat = new Float32Array(this.readback.buffer as ArrayBuffer);
        const L = new Float64Array(n * n);
        for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) L[i * n + j] = flat[i * n + j];
        // Mandatory before the next factor(): WebGPUAttributeUtils.js:401 throws
        // 'ReadbackBuffer must be released before being used again'.
        this.readback.release();
        return { L, status, wallMs: performance.now() - t0 };
    }

    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        this.readback.dispose();
    }
}
```

- [ ] **Step 6: Typecheck and lint**

Run: `bunx tsc --noEmit && bunx biome check --write src/gpu/cholesky.ts test/gpu/cholesky.test.ts && bun test`
Expected: `tsc` exit 0; suite green.

- [ ] **Step 7: Commit**

```bash
git add src/gpu/cholesky.ts test/gpu/cholesky.test.ts
git commit -m "$(cat <<'EOF'
feat(gpu): GpuCholesky — 3 kernels/column, GPU-side column counter, reused ReadbackBuffer (phase2a T4)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AbdMPo3QLVT8dq89qjyeAo
EOF
)"
```

---

## Task 5: The spike — K1, T4, the factor wall, and the `G5-confirm` decision

**Files:**
- Create: `bench/gpu/phase2a.ts` (browser-side spikes)
- Modify: `bench/gpu/spikes.ts` (register `phase2aSpikes` beside `phase1Spikes`)
- Create: `bench/results/<today>-gpu-phase2a-cholesky.json`, `…-t4.json`, `…-g5confirm.json`

**Interfaces:**
- Consumes: `GpuCholesky`, `gpuSolveMaxN` (Task 4); `shiftSobolev`, `choleskyF64`, `makePreparedSaddle` (Task 2); the `saddle?` seam (Task 1).
- Produces: the three results JSONs Task 7's README report reads, and **the `G5-confirm` verdict that decides whether Tasks 6–7 run at all**.

**This task runs entirely through `bench/gpu/drive.ts` on the hardware adapter.** A software adapter makes every number here INVALID, never PASS/FAIL (`bench/gpu/README.md`).

**No driver, no store, no self-test is touched.** Per D3, the spike calls `dispatchDescentStep({ …args, dE, saddle })` directly with a `PreparedSaddle` built by hand from `GpuCholesky`'s output — which is why `G5-confirm` can be read "BEFORE any driver / store / self-test integration" (spec §5) even though its statistic is a whole descent step.

- [ ] **Step 1: Write the K1 + factor-wall spike**

Create `bench/gpu/phase2a.ts` with `phase2aCholeskyK1`, registered into `phase2aSpikes`. For each N ∈ {64, 240, 480, 960} on `trefoil(N)` (default params; record the fixture checksum):

1. `A = assembleAFlat(...)`; `{ Asigma, sigma } = shiftSobolev(A, N)`.
2. `{ L, status, wallMs } = await new GpuCholesky(renderer, N).factor(Asigma)`.
3. **K1 value** — f64-measured from the read-back L: `‖LLᵀ − A_σ‖_F / ‖A_σ‖_F`.
4. **Verdict** against the spec §5 bars (green / amber / red), with `status` required to be 0 in all cases.
5. **Factor wall** = `wallMs` (upload + 3N dispatches + readback), p50 of 5 runs after 1 warm-up ([P1 D4] method).
6. **Boot metric** at N=64 for Task 6's self-test bar: `max|L_gpu − L_cpu| / max|L_cpu|` against `choleskyF64` (spec §5: a correct kernel measures 9.04e-8).

```ts
const K1_GREEN: Record<number, number> = { 64: 2.72e-6, 240: 5.71e-7, 480: 2.23e-7, 960: 7.68e-8 };
const K1_RED: Record<number, number> = { 64: 1.72e-4, 240: 7.66e-6, 480: 1.10e-6, 960: 1.45e-7 };
function k1Verdict(n: number, value: number): 'green' | 'amber' | 'red' {
    if (value >= K1_RED[n]) return 'red';
    return value <= K1_GREEN[n] ? 'green' : 'amber';
}
```

Every row records the **value**, not only the verdict (spec §5).

- [ ] **Step 2: Run the K1 spike and read it against spec §6**

Run: `bun bench/gpu/drive.ts phase2aCholeskyK1`
Expected: `bench/results/<today>-gpu-phase2a-cholesky.json` written, adapter classified `hardware`.

**Stop conditions, from spec §6 — read them before continuing:**
- **K1 red at N=480** (`≥ 1.10e-6`) ⇒ variant (a) is NOT confirmed. This is the parent's confirmation spike. Record the measured error and **stop**: the milestone routes to **building variant (b)**, not to kernels-only (spec §6 corrects the first draft's premise — on the measured CG count, (b) delivers 6.1–14.6× on the solve path).
- **K1 amber at N=960** with T4 green at N=960 is an **expected** outcome, not a surprise (spec §4.2: N=960's IR budget has exactly one spare iteration). Record and continue.
- **`status ≠ 0`** at any N is a kernel bug, not a numerics result: `1 + j` names the failing column. Fix before reading K1.

- [ ] **Step 3: Write the T4 spike**

`phase2aT4` — for each fixture in spec §5's T4 row (`crossing` N=8; trefoil N=60/120/240/480/960, total mode; perEdge N ≤ 240 **informational only**), build the `PreparedSaddle` on the GPU factor and record, per rhs shape (gradient `[dE; 0]` and projection `[0; −Φ]`):

- the relative residual `‖r − Kz‖/‖r‖` after IR (f64, from `solveSaddleFrozen`'s own `residual`), and
- the IR iteration count and the full `relResiduals` curve from Task 2's `IrTrace`.

**Bar:** ≤ **1e-10** within ≤ **4** iterations, per N. **T4 red at N ≤ 480 is a kill** (spec §6); T4 red at a larger N narrows the served N.

Run: `bun bench/gpu/drive.ts phase2aT4`
Expected: `bench/results/<today>-gpu-phase2a-t4.json`.

- [ ] **Step 4: Write the `G5-confirm` spike**

`phase2aG5Confirm` — trefoil N=480/960, total mode, frozen projection. For each N, 1 warm-up + 5 measured steps ([P1 D4]), and record **both** readings (D3):

```ts
// The parent's statistic is the whole descent STEP; the solve-path reading is recorded
// beside it because [P §4 G5]'s words admit both and the spec requires both reported.
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §3, §5 (G5-confirm)
const wholeStepMs = p50(perStep.map((s) => s.stepWallMs));   // includes upload/compute/readback
const solvePathMs = p50(perStep.map((s) => s.factorWallMs + s.algebraMs));
```

where `algebraMs` is the summed wall time of every Schur/Woodbury/IR/backsolve call for the gradient solve and every projection solve of that step — i.e. exactly the work that replaces [BASE]'s `saddle` phase.

- [ ] **Step 5: Run it**

Run: `bun bench/gpu/drive.ts phase2aG5Confirm`
Expected: `bench/results/<today>-gpu-phase2a-g5confirm.json` with `wholeStepMs` and `solvePathMs` at both N, the (b) comparison bands from spec §3 copied in, and the verdict.

- [ ] **Step 6: READ THE GATE AND STOP IF IT IS NOT GREEN**

Apply spec §5's `G5-confirm` row at **N=960** exactly as the split table above states: green ≤ 586 ms whole step; red ≥ 879 ms; amber ⇒ green requires solve path ≤ 283 ms, else red. **N=480 is recorded, not gated** (no (b) whole-step band is measured there).

- **Green (either reading)** ⇒ §3's provisional decision becomes the decision. Continue to Task 6.
- **Red** ⇒ **STOP. Do not start Task 6.** Commit the measurements, write what was measured into `bench/gpu/README.md`, and report to the orchestrator that slice 2a stops at the spike and the milestone re-plans onto variant (b). A red here discards a spike, which is the entire reason the gate sits at this point in the plan rather than after integration.

- [ ] **Step 7: Commit the spike and its results**

```bash
git add bench/gpu/phase2a.ts bench/gpu/spikes.ts bench/results/*-gpu-phase2a-cholesky.json bench/results/*-gpu-phase2a-t4.json bench/results/*-gpu-phase2a-g5confirm.json
git commit -m "$(cat <<'EOF'
bench(gpu): K1 + T4 + factor wall + G5-confirm on the measured Cholesky spike (phase2a T5)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AbdMPo3QLVT8dq89qjyeAo
EOF
)"
```

---

> **Tasks 6 and 7 start only if `G5-confirm` was green.** If it was red, slice 2a is finished at Task 5 and the milestone re-plans onto variant (b) — see the split table above.

## Task 6: Driver, store and boot self-test integration

**Files:**
- Modify: `src/gpu/driver.ts:24-46` (`gpuSolveSupported`), `:116-186` (`step`)
- Modify: `src/gpu/selfTest.ts` (add `runGpuSolveSelfTest`)
- Modify: `src/store.ts:132` / `:252` / `:364` / `:418` (`gpuSolveAvailable`, `setGpuSolveAvailable`)
- Modify: `src/scene/Viewer.tsx:486`, `:636`, `:655`, `:669` (all four reset sites, `poisoned` latch included)
- Test: `test/gpu/driver.test.ts` (extend), `test/gpu/selfTest.test.ts` (extend), `test/store-solver-driver.test.ts` (extend)

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces: `GpuDriver.step` returns `Promise<{ outcome: DescentStepOutcome; usedGpu: boolean; usedGpuSolve: boolean }>`; `gpuSolveSupported(st): boolean`; `runGpuSolveSelfTest(renderer): Promise<{ ok: boolean; relErr: number; error?: string }>`; `SimStore.gpuSolveAvailable` + `setGpuSolveAvailable`.

- [ ] **Step 1: Write the failing routing-predicate test**

Extend `test/gpu/driver.test.ts`. Spec §4.5's predicate is the Phase 1 predicate AND `projectionMode === 'frozen'` AND the constraint set ∈ {barycenter, totalLength, pins} (no `edgeLengths`) AND `N ≤ GPU_SOLVE_MAX_N` AND `gpuSolveAvailable`.

```ts
test('gpuSolveSupported: perEdge is never served by the GPU solve', () => {
    // [G6] already routes perEdge N >= 480 to CPU f64, and k = |E|+3 constraint rows
    // make the Schur block O(N³) — spec §1 non-goal, not an omission.
    expect(gpuSolveSupported({ ...frozenTotalState(), lengthMode: 'perEdge' })).toBe(false);
});

test('gpuSolveSupported: reassemble projection mode is never served', () => {
    expect(gpuSolveSupported({ ...frozenTotalState(), projectionMode: 'reassemble' })).toBe(false);
});

test('gpuSolveSupported: N above GPU_SOLVE_MAX_N falls back rather than throwing', () => {
    // The failure mode this replaces is a buffer-creation VALIDATION error, not the
    // graceful per-step CPU fallback §4.5 promises. @see spec §4.3
    expect(gpuSolveSupported({ ...frozenTotalState(), maxN: 100, vertexCount: 101 })).toBe(false);
});

test('gpuSolveSupported: gpuSolveAvailable is an independent bit from gpuAvailable', () => {
    expect(gpuSolveSupported({ ...frozenTotalState(), gpuSolveAvailable: false })).toBe(false);
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `bun test test/gpu/driver.test.ts`
Expected: FAIL — `gpuSolveSupported` is not exported.

- [ ] **Step 3: Add `gpuSolveSupported` and the store bit**

`src/store.ts` — mirror `gpuAvailable` exactly (declaration at `:132`, setter type at `:252`, initial value `false` at `:364`, implementation at `:418`):

```ts
    /**
     * Boot verdict for the GPU SOLVE (phase 2a), independent of {@link gpuAvailable}
     * (which gates GPU dE). Two capability bits, because a device can factor wrongly
     * while its dE kernel is fine — and both must be reset on device loss (spec §7).
     * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.3, §4.4, §7
     */
    gpuSolveAvailable: boolean;
```

`src/gpu/driver.ts` — beside `gpuStepSupported` at `:24-46`.

- [ ] **Step 4: Extend `GpuDriver.step` with the two-batch flow**

Spec §4.5, in order: copy `live` ONCE (the existing rule at `:164` — do not add a second copy); issue the dE batch; run `assembleAFlat` + `shiftSobolev` on the CPU **while it is in flight**; upload A_σ and issue the Cholesky batch; read dE back; read L + status back; build the `PreparedSaddle`; run the step.

```ts
            // Two batches, not one (spec §4.5): the Cholesky needs A_σ uploaded, and A_σ
            // costs a full assembleAFlat — the largest remaining CPU term (86 / 355 ms at
            // N=480/960, [BASE]). Issuing dE first lets that assembly overlap the GPU dE
            // work (3.2 / 5.1 ms) instead of serialising behind it, and it keeps
            // timings.dE and timings.factor separable so T4/P2a measure what they claim.
            // Cost: one extra submit (~0.5 ms), inside the P2a budget.
            // @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.5
```

Then the three fallback triggers — **one action each, all per-step, none a driver flip** (spec §4.5):

```ts
            let usedGpuSolve = false;
            let saddle: PreparedSaddle | undefined;
            if (solveEligible && status === 0 && allFiniteFlat(L)) {
                saddle = makePreparedSaddle(A, L, sigma, n);
                usedGpuSolve = true;
            }
            let outcome: DescentStepOutcome;
            try {
                outcome = dispatchDescentStep({ ...args, ...(usedGpu ? { dE } : {}), ...(saddle ? { saddle } : {}), collectTimings: true, collectField: st.showArrows });
            } catch (e) {
                // Triggers (2) and (3) of spec §4.5: an ExternalSolveError from a
                // PROJECTION solve propagates straight out (lineSearchStepSet sits outside
                // optimizer.ts's try/catch); one from the GRADIENT solve is rethrown by
                // that catch (§4.4). Both discard the prepared saddle and re-run THIS step
                // with the SAME vertices copy and the SAME dE — safe because the step is
                // pure in its inputs (bench/sobolev.bench.ts already depends on that).
                // Price: one wasted partial step, ≲2× a CPU step. Accepted because a
                // silently wrong g̃ is not acceptable at any price (§4.1 step 4).
                if (!(e instanceof ExternalSolveError)) throw e;
                usedGpuSolve = false;
                outcome = dispatchDescentStep({ ...args, ...(usedGpu ? { dE } : {}), collectTimings: true, collectField: st.showArrows });
            }
```

Trigger (1) — nonzero status or non-finite L — is handled by the `if` above, BEFORE `dispatchDescentStep`, so that step simply runs without `saddle` (never a NaN-laden L into the CPU algebra).

Stamp `timings.factor` with the GPU wall time (upload + compute + readback), the same way Phase 1 stamps `timings.dE` at `:175-181`, and widen the return to `{ outcome, usedGpu, usedGpuSolve }`.

- [ ] **Step 5: Extend the boot self-test**

`src/gpu/selfTest.ts` — `runGpuSolveSelfTest(renderer)`: factor the committed 64×64 fixture (`A_σ` of `trefoil(64)`, deterministic) with `GpuCholesky` and compare against `choleskyF64`:

```ts
/**
 * Boot gate for the GPU SOLVE. Bar `max|L_gpu − L_cpu|/max|L_cpu| < 1e-5` — a FORWARD
 * difference, not K1's backward error. 1e-5 is ≈111× above the 9.04e-8 a correct f32
 * kernel measures at N=64 ([K1CAL]); the slack is deliberate, because this test gates
 * AVAILABILITY on an unknown adapter's summation order, not numerics, and a broken
 * kernel (dropped term, off-by-one column) produces O(1) relative error, not 1e-5.
 * Never throws — a throw is a failed gate, exactly like `runGpuSelfTest`.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.3, §5 (Boot)
 */
```

- [ ] **Step 6: Mirror the capability bit at ALL FOUR Viewer sites**

`src/scene/Viewer.tsx` — spec §7 names them: `486`, `636`, `655`, `669`. `655`/`669` are the device-loss and uncaptured-error hooks; `636` carries the `poisoned` latch (`:629-636`) that stops a late self-test result from re-arming a poisoned device. **The second bit needs all four mirrored, latch included** — a missed site is a device that reports `gpuSolveAvailable: true` after a loss.

Test it: extend `test/gpu/selfTest.test.ts` with a case that asserts a late `runGpuSolveSelfTest` resolution after `poisoned = true` leaves `gpuSolveAvailable` false.

- [ ] **Step 7: Run everything and commit**

Run: `bunx tsc --noEmit && bun test && bunx biome check --write src/gpu src/store.ts src/scene/Viewer.tsx test && bunx knip`

```bash
git add src/gpu src/store.ts src/scene/Viewer.tsx test
git commit -m "$(cat <<'EOF'
feat(gpu): two-batch GpuDriver.step with the GPU factor, gpuSolveAvailable, solve boot gate (phase2a T6)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AbdMPo3QLVT8dq89qjyeAo
EOF
)"
```

---

## Task 7: The gate benches — `R₂` attribution, P2a, T5, and the README report

**Files:**
- Modify: `bench/gpu/phase2a.ts` (add `phase2aR2Attribution`, `phase2aStepGate`, `phase2aT5`)
- Modify: `bench/gpu/README.md` (add "Phase 2a gate report")
- Create: `bench/results/<today>-gpu-phase2a-r2.json`, `…-step.json`, `…-t5.json`

**Interfaces:**
- Consumes: everything above.
- Produces: the committed gate record for the branch.

- [ ] **Step 1: Attribute `R₂` — FIRST, because no P2a number may be read before it**

Spec §5 makes this a **precondition**, not a deferred question: "no P2a number is read as pass/fail until `R₂` is ATTRIBUTED", and "Recording `R₂` again is not attributing it: plan 2a's gate task must **name the work, or name the measurement error**."

Implement D4's within-step measurement as `phase2aR2Attribution`: at N=480 and N=960, run 1 warm-up + 5 CPU-only steps with `collectTimings: true`, and for **each individual step** record the whole ledger and compute

```ts
const r2 = t.step.ms - t.dE.ms - t.assembleA.ms - t.saddle.ms - t.energy.ms;
```

Report `r2` per step, its median, and the [BASE]-style cross-step figure for comparison. Then write the verdict into the results JSON as one of two strings:

- `"measurement-error"` — the per-step `|median r₂ᵢ|` is below 10 % of [BASE]'s `R₂` at that N. `R₂` is then an artifact of medianing each key over an independent sample of steps; §2's independent evidence is that [BASE]'s N=1000 row forces `R < 0` by 5.28 ms, which no amount of work can explain. Record the P2a numbers both with and without it.
- `"work"` — the per-step `r₂ᵢ` reproduce ≈60 ms at N=480. Add ONE `timed()` probe around the largest unwrapped region of `sobolevStepSet` (the `l2CurveNorm` + `lineSearchStepSet` orchestration outside the `energy` / `projection` wraps) and name the phase. `phaseTimings.ts:1-7` sanctions this — "Add keys; never rename".

Run: `bun bench/gpu/drive.ts phase2aR2Attribution`
Expected: `bench/results/<today>-gpu-phase2a-r2.json` with a verdict string that is not `"unknown"`. **If it is `"unknown"`, stop — P2a cannot be read.**

- [ ] **Step 2: Run the P2a step gate**

`phase2aStepGate` — spec §5's P2a row, exactly:

- Fixtures: trefoil N=480/960, total, frozen; **plus trefoil N=960 + 3 pins** (k = 4+3P; the setup is 13+9P n-backsolves, spec §7 — **recorded, not gated**).
- Method: **PAIRED** — both medians from the SAME browser session, on the same fixture, interleaved; 1 warm-up, 5 runs, medians, wall-clock including all uploads and readbacks ([P1 D4]).
- Also re-measure [JSALG]'s three JS f64 constants **in the same session** and record them beside the verdict (spec §5), so a session that is slow for unrelated reasons is visible rather than silently inside the verdict.
- **Gated at N=960 only: speedup ≥ 8×**, and GPU factor wall ≤ **100 ms** at N=960.
- **N=480 is RECORDED, NOT GATED** — its margin (14.8 %) is smaller than the median-of-5's own low-side noise (15.0 %), so a green there would not distinguish a real win from a lucky median. Report it; it cannot stop the branch.

Run: `bun bench/gpu/drive.ts phase2aStepGate`

Spec §6 on a red: if the factor wall > 100 ms at N=960, build the blocked variant or the indirect dispatch of §4.3 (one pre-registered extra task); if the speedup is still < 8× with the factor under the bar, **2a does not merge** and the per-phase table says where the time went. A low N=480 reading with N=960 green is an **Amdahl** result, not a kernel result (the N=480 ceiling is 4.49–5.05×).

- [ ] **Step 3: Run the T5 trajectory gate**

`phase2aT5` — energy after K=50 steps within **±1 %** of the f64 trajectory; no accepted step increases energy; f64 recomputed from positions; median of 3; trefoil N=**120 and 480**, total, frozen. (Spec §5 AMENDS [P §3]: T5 drops its perEdge fixture because §1 does not serve perEdge, and GAINS N=480 because at N=120 alone κ·u_f32 = 8.5e-5 versus 8.4e-2 at N=960 — 1000× away from the regime every κ-dependent defect lives in.)

Run: `bun bench/gpu/drive.ts phase2aT5`

A red ⇒ investigate before merging (the flakiness rule of [P §3]); a **reproducible** red is a kill.

- [ ] **Step 4: Write the README gate report**

`bench/gpu/README.md` gains a "Phase 2a gate report" section in the same shape as the Phase 0 and Phase 1 reports: one row per gate with its **measured value**, its bar, its verdict, the fixture, the adapter, and the `gitShaShort` of the run. Include the `G5-confirm` numbers from Task 5 and both readings, the K1 amber/green/red per N, the IR convergence curves, `diag(M)` / `κ(M)` / `min diag(L)²` from CR, the `R₂` verdict, and the same-session [JSALG] constants. Follow the file's own "Provenance" rule for the sha.

- [ ] **Step 5: Full verification**

Run: `bunx tsc --noEmit && bun test && bunx biome check src && bunx knip`
Expected: `tsc` exit 0; suite green with the new tests; biome at the pre-branch baseline; knip exit 0.

- [ ] **Step 6: Commit**

```bash
git add bench/gpu/phase2a.ts bench/gpu/README.md bench/results/*-gpu-phase2a-*.json
git commit -m "$(cat <<'EOF'
bench(gpu): R2 attribution, P2a paired step gate, T5, phase 2a gate report (phase2a T7)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AbdMPo3QLVT8dq89qjyeAo
EOF
)"
```

---

## Self-review

**1. Spec coverage.**

| Spec section | Task |
|---|---|
| §1 scope table (2a column), non-goals | Global Constraints; Task 6 Step 1 tests perEdge / reassemble exclusion |
| §2 partition, `R₂` | Task 7 Step 1 (D4) |
| §3 G5, AMENDS 1–3 | Task 5 Steps 4–6 (`G5-confirm`) |
| §4.1 shift / Schur / Woodbury / IR + the throw | Task 2 |
| §4.2 precision model, f64 everywhere but the factor | Task 2 (residuals vs the f64 K), Task 5 Step 3 (T4) |
| §4.3 kernel, counter, status flag, `GPU_SOLVE_MAX_N`, boot fixture | Task 4; Task 6 Step 5 |
| §4.4 the seam, both control-flow edits | Task 1 (D1, D2, D8) |
| §4.5 two batches, routing predicate, three fallback triggers | Task 6 Steps 3–4 |
| §4.6 / §4.7 (slices 2b / 2c) | **Out of scope** — no task, by design |
| §5 CR / K1 / T4 / T5 / G5-confirm / P2a / Boot | Tasks 3, 5, 5, 7, 5, 7, 6 |
| §6 kill gates | Task 5 Steps 2 and 6; Task 7 Steps 2–3; the split table |
| §7 risks (`diag(M)`, pins, two capability bits, readback size) | Task 3 Step 2 (recorded), Task 7 Step 2 (pinned fixture), Task 6 Step 6, Task 5 Step 1 |
| §8 open questions | D2 (where the types live), D5 (counter vs uniform vs indirect), Task 2 `IrTrace` (results-JSON schema). The `colFactor`/`trailingUpdate` fusion stays open — spec §8 conditions it on the dispatch floor dominating P2a, which is a Task 7 reading, not a Task 4 choice. |

**Gap accepted deliberately:** spec §5's T4 row lists "perEdge N ≤ 240 informational only". Task 5 Step 3 collects it but nothing gates on it, matching the spec's own word "informational".

**2. Placeholder scan.** No "TBD", no "add error handling", no "similar to Task N". Tasks 5, 6 and 7 give the spike bodies as their measurement specification plus the exact constants, verdict functions and stop conditions rather than 400 lines of browser-harness plumbing — the plumbing is `bench/gpu/phase1.ts`'s shape, which the implementer reads directly and which no plan can usefully transcribe. Every **numeric bar, verdict rule and stop condition** in those tasks is written out.

**3. Type consistency.** `PreparedSaddle` / `ExternalSolveError` / `ExternalFactorization` are declared once (Task 1) and used with the same names in Tasks 2, 5, 6. `makePreparedSaddle(A, L, sigma, n, onTrace?)` has the same signature in Tasks 2, 3, 5, 6. `shiftSobolev` returns `{ Asigma, sigma }` everywhere. `GpuCholesky.factor` returns `{ L, status, wallMs }` in Tasks 4, 5, 6. `gpuSolveMaxN` (Task 4) is what Task 6's routing predicate calls; the spec calls the resulting constant `GPU_SOLVE_MAX_N` and the code makes it a function because it is a **device** query, not a constant (spec §4.3: "queried at boot, not asserted").

**4. One deliberate departure from the spec's letter, recorded:** D1. Spec §4.4 sketches the collector disarm as `finally { if (collect) timingsEnd(); }`; implemented literally that discards the ledger on the SUCCESS path, because `timingsEnd()` also nulls the accumulator (`phaseTimings.ts:56-63`) and `optimizer.ts:461` reads it afterwards. The plan implements the spec's stated REQUIREMENT ("no behaviour change when nothing throws") with a catch-and-rethrow, and Task 1 Step 7 tests both halves.
