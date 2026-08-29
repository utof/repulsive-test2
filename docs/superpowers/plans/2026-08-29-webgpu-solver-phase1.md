# WebGPU Solver — Phase 1 (Gradient Kernel + `'gpu'` Driver) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the production tangent-point dE gather kernel (two-float positions, `wgslFn` body, no atomics) behind an opt-in `'gpu'` solver driver, with the spec's T1/T2/T3 tolerance gates run through the *production* kernel and the Phase-1 perf gate (GPU dE phase ≥5× CPU dE phase at N=480 and N=960, readback counted) recorded — while every energy, the saddle solve and the line search stay CPU f64, byte-for-byte unchanged.

**Architecture:** `src/gpu/` is a new, renderer-bound module. Pure parts (WGSL source generation, topology packing, f32 degeneracy bounds, the driver's eligibility predicate) are plain TS with Bun unit tests. GPU parts (`GpuTangentPoint`: one interleaved hi/lo positions buffer, gradient gather + vertex-gather kernels, energy kernel + two-pass tree reduction, one readback) are verified only in the browser harness (`bench/gpu/drive.ts`) on the hardware adapter, per spec §3. The CPU seam is one optional, non-numeric field — `dE?: Vec3[]` (precomputed differential at the input vertices) threaded `dispatchDescentStep → sobolevStepSet`; absent ⇒ every existing path is bit-identical. The `'gpu'` driver lives in the Viewer frame loop beside the worker driver under a shared single-flight token; unsupported configs go to the worker per step; any GPU failure flips the selection `'gpu' → 'worker'`.

**Tech Stack:** Bun (tests, CDP driver), TypeScript strict, three `0.185.1` (`three/webgpu` + `three/tsl`: `wgslFn`, `attributeArray`, `instancedArray`, `Fn().compute(count,[wg])`, `workgroupArray`, `workgroupBarrier`, `invocationLocalIndex`, `workgroupId`, `renderer.compute([...])`, `renderer.getArrayBufferAsync`, `renderer.onDeviceLost`/`onError`), Zustand 5, React 19 / R3F 9.

**Spec:** `docs/superpowers/specs/2026-08-13-webgpu-solver-design.md` — §1 (scope, sanctioned type deltas), §2.1–2.3, §2.6, §3 (T1–T3), §5 Phase 1, §6 (device loss), §7 (open questions this plan answers). Evidence: `docs/2026-08-13-ai-research-webgpu-compute.md` [IMPL], `docs/2026-08-13-ai-research-gpu-precision.md` [PREC], Phase 0 report `bench/gpu/README.md`.

**Phase entry:** Phase 0 merged at PR #21; `bench/gpu/README.md` ("Go/no-go reading") states Phase 1 is unblocked. G4 (render) is a Phase 3 concern; G6 (perEdge κ) constrains Phase 2's solve. Neither touches this plan. Spec §0's "no `src/gpu/` ⇒ Phase 0 not started" heuristic is stale — Phase 0 deliberately shipped no product code.

## Global Constraints

- **Phase 1 only** (spec §5): the GPU computes **dE only**. E₀, every Armijo trial energy, projection, saddle solve, line search: CPU f64, untouched. The GPU energy kernel is written and T2-gated but consumed only by gates and the boot self-test — never by descent (spec §2.3 never-mix).
- **No `src/core/**` numeric changes** (spec §1). Sanctioned public-type deltas: (i) `SolverDriver` += `'gpu'`; **(iii, added by this plan — Task 5 amends the spec)** `SobolevStepOptions.dE?` / `DispatchDescentStepArgs.dE?` — a precomputed differential at the input vertices. It selects the *source* of dE, not its arithmetic; absent ⇒ bit-identical (golden suite is the backstop). Chosen over an out-of-core re-implementation of `sobolevStepSet`'s orchestration because Phase 2 must hook the same function again and a parallel orchestrator drifts (D5).
- **One device**: the Viewer's `WebGPURenderer` (`src/scene/Viewer.tsx:376-381`, already `trackTimestamp: true`). Never a second adapter. Main thread. No WebGPU in the worker.
- **Precision (spec §2.3)**: positions as hi/lo f32 pairs in ONE interleaved buffer; every coordinate difference is `(hi_i − hi_j) + (lo_i − lo_j)`; integer exponents by repeated multiplication (generated from α/β at kernel-build time); `pow` only for non-integer exponents; **`+ eps` after every norm is KEPT identical to the CPU formula** (D2 — ε is *not* inert at near-touch scale: measured 5.3e-4 at gap 1e-6, 50× the T1 tolerance); degeneracy guards are explicit, scale-relative f32 bounds (D3), never the CPU `< 1e-14` constants.
- **Gather, no atomics (spec §2.2)**: thread per edge I evaluates both ordered roles and writes only I's two endpoint slots; a second kernel gathers slots into vertices via a vertex→slot CSR; the `×0.5` symmetry factor is applied in the edge kernel.
- **One `renderer.compute([...])` per GPU stage; exactly one positions upload + one readback per step** (spec §5 Phase 1, §2.5).
- **`'gpu'` serves sobolev + analytical + penalties-off only** (spec §1); any other config → that step goes to `'worker'` without changing the selected driver.
- **Gates run on the hardware adapter** (`drive.ts` classifies; software = INVALID). Results committed as `bench/results/<date>-gpu-phase1-*.json`, keyed by SHA + adapter. A red T1/T2/T3 or perf gate **stops the branch**; tolerances, fixtures and the readback-inclusive timing are not negotiable.
- **Fixtures**: T1 = `nearTouchPair(gap)` for gap ∈ {1e-3,1e-4,1e-5,1e-6}; T2/T3 = every `testConfigs` preset at default params (with `Math.random` seeded for the run) + `trefoil(240)` + `trefoil(960)`.
- Conventions (as Phase 0): Biome (4-space, single quotes, width 100); `import type`; `bunx tsc --noEmit`; `bun test <path>`; TSDoc with `@see` on every export and inline anchors on every guard/`0.5`/op-order line (CLAUDE.md); one commit per task; trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Reviewer nits follow CLAUDE.md's inline-fix gate (fix inline only if every gate holds, else `gh issue create` with labels). Blockers are fixed on the branch regardless.

---

## File structure

| Path | Responsibility |
|---|---|
| `src/gpu/wgsl.ts` | Pure: WGSL source generators (pair body, gradient gather, vertex gather, energy) parameterised by (α, β, ε); `powExpr`; guard constants. No three import. |
| `src/gpu/topology.ts` | Pure: CSR packing (edges, disjoint pairs, vertex→slot incidence) + in-place interleaved hi/lo writer + f32 degeneracy-bound mirrors for tests. |
| `src/gpu/reduce.ts` | `makeReduceSum` — fixed 64-lane two-pass workgroup tree (spec §2.2), TSL. |
| `src/gpu/tangentPoint.ts` | `GpuTangentPoint` — buffers + compute nodes on the renderer; `gradient()`, `energy()`, `dispose()`. Only file importing `three/tsl`. |
| `src/gpu/selfTest.ts` | Boot gate: G2 through the production kernel (`nearTouchPair(1e-6)`, relErr < 1e-5). |
| `src/gpu/driver.ts` | `gpuStepSupported`, `allFinite`, `GpuDriver` (topology-on-graphVersion, GPU dE, non-finite fallback, `dispatchDescentStep({...args, dE})`, timing stamp). |
| `src/core/optimizer.ts`, `src/core/dispatch.ts` | The `dE?` seam (Task 5). |
| `src/store.ts` | `SolverDriver` += `'gpu'`; `gpuAvailable` + `setGpuAvailable`. |
| `src/scene/Viewer.tsx` | Boot self-test + device-loss/error hooks; `'gpu'` branch in `useFrame`; worker alive under `'gpu'`. |
| `src/ui/ControlPanel.tsx` | `GPU` option, disabled until `gpuAvailable`. |
| `bench/gpu/compare.ts` | `relErrComparator`/`cosineComparator`/`xorshift32` extracted from `spikes.ts` (pure refactor) + `withSeed`. |
| `bench/gpu/phase1.ts` | Spikes `phase1Tolerance`, `phase1DeGate`, `phase1DriverSmoke`; registered into `spikes.ts`. |
| `test/gpu/*.test.ts` | Unit tests for the pure modules and the seam. |
| `bench/gpu/README.md` | Phase 1 gate report. |

---

## Decisions (read before Task 1; each is falsifiable)

### D1 — Gather derivation

`gradientAnalytical` (`src/core/tangentPointEnergy.ts:139-392`) adds, for each **ordered** pair (I,J), I=(i1,i2), J=(j1,j2), with `base = 0.25·ℓ_I^{1−α}·ℓ_J`, `sI = 0.25·ℓ_J·ΣF`, `sJ = 0.25·ℓ_I^{1−α}·ΣF`, `dPowCoeff = (1−α)ℓ_I^{−α}`:

- `grad[i1] += sI·dPowCoeff·(−ê_I)`, `grad[i2] += sI·dPowCoeff·ê_I`
- `grad[j1] += sJ·(−ê_J)`, `grad[j2] += sJ·ê_J`
- per term t=(i,j), d = v_i − v_j: `grad[i] += base·df_dd`, `grad[j] −= base·df_dd`, `grad[i1] −= base·df_de`, `grad[i2] += base·df_de`

`disjointPairs` holds both (I,J) and (J,I) [IMPL §3], so everything landing on I's endpoints comes from the pair (I,J) with I first (**role A**) and the pair (J,I) with I second (**role B**). Thread I loops J ∈ disjoint(I) once and evaluates:

- **Role A** (e = e_I, d = v_i − v_j, i∈I, j∈J): `sI` terms; `+base·df_dd` to i; `∓base·df_de` to i1/i2.
- **Role B** (e = e_J, d = v_a − v_b, a∈J, b∈I): `sJ'` terms with our I as that pair's "J" (`grad[i1] += sJ'·(−ê_I)`, `grad[i2] += sJ'·ê_I`); `−base'·df_dd` to b. (Role B's `df_de` and `sI'` terms land on J's vertices — J's own thread computes them.)

8 kernel evaluations per (I,J) instead of 4 — the ~2× flops [IMPL §3] traded for zero atomics + determinism. Term order (i1,j1),(i1,j2),(i2,j1),(i2,j2) mirrors the CPU. Thread I writes `0.5·g1`, `0.5·g2` to `slots[6I..6I+5]` (the `×0.5` = `gradientAnalytical`'s final loop, `tangentPointEnergy.ts:380-390`). The vertex-gather kernel sums each vertex's incident slots in stored order — zero atomics, deterministic.

### D2 — ε placement and f32 range (spec §7 "degeneracy threshold derivation")

**ε stays.** [PREC Q1] measured ε=1e-10 inert on a trefoil at O(1) distances. At near-touch scale it is not: f32 ulp at 1e-6 is ≈6e-14 < 1e-10, so `rd + eps` moves `d_eps` by 1e-4 relative and β=6 amplifies it. Measured in f64 with the CPU kernel (2026-08-30, `calculateEnergy` ε=0 vs ε=1e-10 on `nearTouchPair`): relErr 5.25e-7 @1e-3, 5.25e-6 @1e-4, **5.25e-5 @1e-5, 5.25e-4 @1e-6** — the last two exceed T1's 1e-5. An ε=0 GPU kernel therefore fails T1 by construction against the ε=1e-10 CPU oracle (which is the energy *definition*, `tangentPointEnergy.ts:61-65`). Phase 0's G2 kernel kept `+ eps` (`bench/gpu/spikes.ts:459-466`) and passed at 4.6e-6. The spec's "GPU kernel is mathematically the ε=0 energy" (§2.3) is refined by Task 5's amendment to: "ε is added after every norm exactly as on the CPU; it is inert only where the CPU's own ε is inert; degeneracy protection is the explicit f32 guard of D3."

**Reciprocal-power form.** With β=6 at gap 1e-6: `d_eps^β = 1e-36` is a normal f32 (min normal 1.18e-38) but `d_eps^{β+1} = 1e-42` is a denormal, and WGSL permits flush-to-zero → the CPU's `coeff_d = −β·c^α / d^{β+1}` would divide by zero on a FTZ driver. The kernel uses the algebraically identical:

```
invD    = 1 / d_eps              (≤ 1e6 at gap 1e-6)
invDb   = invD^β                 (≤ 1e36 — normal; overflows only below gap ≈ 4e-7, outside the T1 fixture floor)
f       = c_eps^α · invDb
coeff_c = α · c_eps^{α−1} · invDb
coeff_d = −β · f · invD          (= −β·c^α/d^{β+1} exactly, without forming d^{β+1})
```

Magnitudes at gap 1e-6, extent 4: c ≈ 4e-6 → c³ ≈ 6e-17, f ≈ 6e19, coeff_c ≈ 5e25, coeff_d ≈ 4e26 — inside f32 range. Below gap ≈ 4e-7 `invDb` overflows to `inf`; the driver detects any non-finite readback and runs that step on the CPU dE path (Task 6) — a correct f64 answer, never a silently wrong one.

### D3 — Degeneracy guards (replace `< 1e-14`, `tangentPointEnergy.ts:173,211,253,283`)

The CPU guards test pre-ε lengths before forming a unit vector. In two-float arithmetic the noise floor of a coordinate difference of magnitude M is the rounding of the `lo` residual: `u²·M` with `u = 2⁻²⁴`. Bounds (mirrored in `src/gpu/topology.ts` for tests):

- vector difference `v` (e_I, e_J, d): degenerate iff `|v| < ‖(32·u²·max(|hi_a|,|hi_b|))_{xyz}‖`;
- cross product `c = e×d`: degenerate iff `|c| < 32·u·‖(|e_y d_z|+|e_z d_y|, |e_z d_x|+|e_x d_z|, |e_x d_y|+|e_y d_x|)‖` (two f32 products and one subtraction per component).

Semantics identical to the CPU: a degenerate `d` or `c` zeroes only the corresponding unit-vector-derived quantities (`dHat`, `dc_dd`, `dc_de`); `f` itself is still computed with `+ eps` (as the CPU does). A degenerate edge (`e_I` or `e_J`) zeroes `ê`. **Comparison direction is load-bearing:** "degenerate iff `length ≤ guard`" (WGSL non-degenerate branch is `if (r > guard)`). With `>=` an exactly-collinear pair has guard = 0 AND `rc` = 0, so `0 >= 0` takes the `1/rc` branch → `inf·0 = NaN`. The `crossing` preset (`testConfigs.ts` — edges (0,1) and (2,3) exactly collinear on the x-axis) is in the T2/T3 fixture set and would NaN the whole gradient. Tests (Task 2) prove: coincident vertices ⇒ degenerate; `nearTouchPair(1e-6)` differences and cross products ⇒ NOT degenerate; exactly collinear axis-aligned cross (`[4,0,0]×[2,0,0]`, guard exactly 0) ⇒ degenerate; scaling the fixture by 0.05 preserves every classification. `phase1Tolerance` additionally asserts every gradient is finite on every fixture.

### D4 — T2 error budget and timing method

Error sources for the GPU total energy at E=960, u = 6e-8: (a) per-edge sequential f32 sum of ≈E same-sign terms, ≈√E·u ≈ 2e-6 per partial, averaging over E independent partials to ≈6e-8 on the total; (b) the two-pass tree reduction — up to ~14 rounding levels on the running total, ≈√14·u ≈ 2e-7 typical, 14u ≈ 8e-7 worst case; (c) per-term arithmetic in `f` (~10 f32 ops: `c_eps³`, `invD⁶`, product), a few u per term, averaging down but not to zero. **Expected total ≈ 2–5e-7 against the 1e-6 gate — a ~2–5× margin, not a large one** (Phase 0's only production-arithmetic datapoint, G2 at 4.6e-6 vs 1e-5, is also ~2×). Pre-registered: T2 in [5e-8, 1e-6] is the expected band; a value *below* 1e-9 is investigated as a vacuous comparison. Contingency if T2 fails only at N=960: switch in-thread accumulation to 16-wide blocked sums before touching anything else.

Perf gate: `gpuDeMs` = `performance.now()` around **pack → upload → compute (edge + vertex gather, one submit) → `getArrayBufferAsync` → `Vec3[]`**; `cpuDeMs` = `performance.now()` around `gradientAnalytical` in the same browser. 1 warm-up, 5 runs, medians, N=480 and N=960. GPU timestamps recorded as `gpuComputeOnlyMs` for information only.

### D5 — The seam

`sobolevStepSet` computes dE at `src/core/optimizer.ts:278-282` and nothing downstream cares where it came from. Option A (rejected): re-implement its orchestration (singular echo, convergence test, line-search call, timings ledger, `descentField`) in `src/gpu/` — ~60 duplicated lines that must be mirrored on every change, and Phase 2 *will* change it. Option B (chosen): `opts.dE ?? (analytical | finiteDiff)` — 1 expression + 2 optional fields + spec §1 amendment. Test: supplying the CPU analytical dE is `toEqual`-identical to omitting it (Task 5).

---

### Task 1: WGSL source generators (pure)

**Files:**
- Create: `src/gpu/wgsl.ts`
- Test: `test/gpu/wgsl.test.ts`

**Interfaces:**
- Produces: `export const F32_U = 2 ** -24`, `export const GUARD_FACTOR = 32`; `export function powExpr(x: string, n: number): string`; `export interface KernelParams { alpha: number; beta: number; epsilon: number }`; `pairKernelWgsl(p)`, `gradientKernelWgsl(p)` (entry `tpGradient`), `vertexGatherWgsl()` (entry `tpVertexGather`), `energyKernelWgsl(p)` (entry `tpEnergy`).

- [ ] **Step 1: Failing tests**

```ts
// test/gpu/wgsl.test.ts
import { expect, test } from 'bun:test';
import { energyKernelWgsl, gradientKernelWgsl, pairKernelWgsl, powExpr, vertexGatherWgsl } from '../../src/gpu/wgsl';

const P = { alpha: 3, beta: 6, epsilon: 1e-10 };

test('powExpr: integer exponents expand to repeated multiplication, never pow()', () => {
    expect(powExpr('x', 1)).toBe('(x)');
    expect(powExpr('x', 3)).toBe('((x * x) * x)');
    expect(powExpr('x', 0)).toBe('1.0');
    expect(powExpr('x', -2)).toBe('(1.0 / ((x * x)))');
    expect(powExpr('x', 2.5)).toBe('pow(x, 2.5)');
});

test('pair kernel (shared include): no pow() for integer α/β; two-float differences; eps kept; reciprocal-power form; guards strict', () => {
    const src = pairKernelWgsl(P);
    expect(src).not.toMatch(/\bpow\(/);
    expect(src).toContain('(p[6u * i + 0u] - p[6u * j + 0u]) + (p[6u * i + 3u] - p[6u * j + 3u])');
    expect(src).toContain('let eps = 1e-10;');
    expect(src).toContain('let d_eps = rd + eps;');            // D2: ε after norm, like the CPU
    expect(src).toContain('let invD = 1.0 / d_eps;');
    expect(src).toContain('let coeff_d = -beta * f * invD;');  // D2: never forms d^(β+1)
    expect(src).not.toContain('* rd;');                        // no d6*rd product anywhere
    expect(src).toContain('if (rd > dGuard)');                 // D3: degenerate iff ≤ guard (guard may be exactly 0)
    expect(src).toContain('if (rc > crossGuard(e, d))');
    expect(src).not.toMatch(/>=\s*(dGuard|crossGuard|diffGuard)/);
});

test('pair kernel: non-integer α uses pow() for that exponent only', () => {
    const src = pairKernelWgsl({ alpha: 2.5, beta: 6, epsilon: 1e-10 });
    expect(src).toContain('pow(c_eps, 2.5)');
    expect(src).not.toContain('pow(invD');
});

test('entry sources are ONE bare fn each (three WGSLNodeFunction parser: ^fn name(...)); shared block lives in the include', () => {
    for (const src of [gradientKernelWgsl(P), vertexGatherWgsl(), energyKernelWgsl(P)]) {
        expect(src.trim().startsWith('fn ')).toBe(true);
        expect(src).not.toContain('struct KD');
        expect((src.match(/\bfn\s+\w+\s*\(/g) ?? []).length).toBe(1);
    }
    expect(pairKernelWgsl(P)).not.toContain('fn tpGradient(');
});

test('entry points + symmetry factor location', () => {
    expect(gradientKernelWgsl(P)).toContain('fn tpGradient(');
    expect(gradientKernelWgsl(P)).toContain('0.5 * g1');       // D1: ×0.5 in the edge kernel
    expect(gradientKernelWgsl(P)).toContain('if (reI > diffGuard(p, i2, i1))');
    expect(vertexGatherWgsl()).toContain('fn tpVertexGather(');
    expect(vertexGatherWgsl()).not.toContain('0.5');
    expect(energyKernelWgsl(P)).toContain('fn tpEnergy(');
    expect(energyKernelWgsl(P)).toContain('partials[I] = 0.5 * acc;');
});
```

- [ ] **Step 2: Run** — `bun test test/gpu/wgsl.test.ts` → FAIL (module missing).

- [ ] **Step 3: Implement**

```ts
// src/gpu/wgsl.ts — pure WGSL source generation for the tangent-point kernels.
// No three.js import: unit-testable in Bun; the single home of the
// precision-critical op order. Positions buffer `p` is interleaved
// [hi.x,hi.y,hi.z,lo.x,lo.y,lo.z] per vertex (6 floats).
// @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (Task 1, D1–D3)
// @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.2, §2.3

/** f32 unit roundoff. @see plan D3 */
export const F32_U = 2 ** -24;
/** Guard multiplier over the two-float noise floor. @see plan D3 */
export const GUARD_FACTOR = 32;

export interface KernelParams { alpha: number; beta: number; epsilon: number }

/**
 * `x^n` in WGSL: repeated multiplication for integer n (spec §2.3), `pow()`
 * only for non-integer n. Left-assoc `((x*x)*x)` on purpose — plan D2.
 * @see docs/2026-08-13-ai-research-gpu-precision.md Q1 ("WGSL's weaker pow")
 */
export function powExpr(x: string, n: number): string {
    if (!Number.isInteger(n)) return `pow(${x}, ${n})`;
    if (n === 0) return '1.0';
    const k = Math.abs(n);
    // k=1 → "(x)"; k=3 → "((x * x) * x)" — the Task 1 test strings are canonical.
    let e = x;
    for (let i = 1; i < k; i++) e = `(${e} * ${x})`;
    if (k === 1) e = `(${x})`;
    return n < 0 ? `(1.0 / (${e}))` : e;
}

const f32 = (v: number): string => (/[.e]/i.test(String(v)) ? String(v) : `${v}.0`);

/**
 * Shared WGSL block — module-scope struct + helper fns ONLY, no entry point.
 * Consumed as a `wgsl(src)` INCLUDE of each entry `wgslFn` (three r185
 * `FunctionNode.js:168` wgslFn(code, includes); `CodeNode.js:99-113` builds
 * includes first). It must NOT be concatenated into the entry source:
 * `WGSLNodeFunction.js:4` parses `^fn name(...)` and throws
 * "Function is not a WGSL code." on anything else (plan review r1 #1).
 * Contents: two-float difference, manual cross (CPU component order,
 * tangentPointEnergy.ts:93-95), D3 degeneracy guards, and `kernelDerivs`
 * (CPU closure at tangentPointEnergy.ts:164-248) in the reciprocal-power
 * form of plan D2. `+ eps` after every norm is KEPT (plan D2).
 */
export function pairKernelWgsl(p: KernelParams): string {
    const { alpha, beta, epsilon } = p;
    return /* wgsl */ `
        struct KD { f: f32, df_dd: vec3<f32>, df_de: vec3<f32> }

        // (hi_i - hi_j) + (lo_i - lo_j) BEFORE anything else — the reassociation G2 gates. @see spec §2.3
        fn diff2(p: ptr<storage, array<f32>, read>, i: u32, j: u32) -> vec3<f32> {
            let x = (p[6u * i + 0u] - p[6u * j + 0u]) + (p[6u * i + 3u] - p[6u * j + 3u]);
            let y = (p[6u * i + 1u] - p[6u * j + 1u]) + (p[6u * i + 4u] - p[6u * j + 4u]);
            let z = (p[6u * i + 2u] - p[6u * j + 2u]) + (p[6u * i + 5u] - p[6u * j + 5u]);
            return vec3<f32>(x, y, z);
        }
        // D3: two-float noise floor of a difference between vertices i and j.
        fn diffGuard(p: ptr<storage, array<f32>, read>, i: u32, j: u32) -> f32 {
            let k = ${f32(GUARD_FACTOR)} * ${f32(F32_U * F32_U)};
            let gx = k * max(abs(p[6u * i + 0u]), abs(p[6u * j + 0u]));
            let gy = k * max(abs(p[6u * i + 1u]), abs(p[6u * j + 1u]));
            let gz = k * max(abs(p[6u * i + 2u]), abs(p[6u * j + 2u]));
            return sqrt(gx * gx + gy * gy + gz * gz);
        }
        fn cross3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
            return vec3<f32>(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
        }
        // D3: noise floor of a cross product from its per-component product scales.
        fn crossGuard(e: vec3<f32>, d: vec3<f32>) -> f32 {
            let sx = abs(e.y * d.z) + abs(e.z * d.y);
            let sy = abs(e.z * d.x) + abs(e.x * d.z);
            let sz = abs(e.x * d.y) + abs(e.y * d.x);
            return ${f32(GUARD_FACTOR)} * ${f32(F32_U)} * sqrt(sx * sx + sy * sy + sz * sz);
        }
        fn norm3(v: vec3<f32>) -> f32 { return sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }

        // dGuard: precomputed diffGuard for d (caller passes it; avoids re-reading p).
        fn kernelDerivs(e: vec3<f32>, d: vec3<f32>, dGuard: f32) -> KD {
            let alpha = ${f32(alpha)};
            let beta = ${f32(beta)};
            let eps = ${f32(epsilon)};
            let rd = norm3(d);
            var dHat = vec3<f32>(0.0);
            if (rd > dGuard) { dHat = d * (1.0 / rd); }           // CPU: !(rd < 1e-14) → D3; STRICT: guard may be exactly 0 (collinear/coincident) and 0>=0 would give 1/0
            let d_eps = rd + eps;                                  // ε after norm — tangentPointEnergy.ts:61-65, plan D2
            let c = cross3(e, d);
            let rc = norm3(c);
            let c_eps = rc + eps;
            let invD = 1.0 / d_eps;
            let invDb = ${powExpr('invD', beta)};                  // d_eps^-β — D2
            let cPowA = ${powExpr('c_eps', alpha)};
            let f = cPowA * invDb;
            var dc_dd = vec3<f32>(0.0);
            var dc_de = vec3<f32>(0.0);
            if (rc > crossGuard(e, d)) {                           // CPU: rc >= 1e-14 → D3; STRICT (see rd above; the `crossing` preset has guard == rc == 0)
                let invRc = 1.0 / rc;
                dc_dd = cross3(c, e) * invRc;
                dc_de = cross3(d, c) * invRc;
            }
            let coeff_c = alpha * ${powExpr('c_eps', alpha - 1)} * invDb;
            let coeff_d = -beta * f * invD;                        // = -β c^α / d^(β+1) without forming d^(β+1) — D2
            var out: KD;
            out.f = f;
            out.df_dd = coeff_c * dc_dd + coeff_d * dHat;
            out.df_de = coeff_c * dc_de;
            return out;
        }
    `;
}

/**
 * Gradient gather — one invocation per edge I; both ordered roles (plan D1);
 * writes ONLY I's two endpoint slots, already ×0.5 (the CPU's final loop,
 * tangentPointEnergy.ts:380-390). Buffers: p (6V f32), edges (2E u32),
 * pairOffsets (E+1), pairIndices, slots (6E f32).
 * ENTRY FUNCTION ONLY — `pairKernelWgsl` is attached as a wgslFn include
 * (see its TSDoc). Every `ptr<storage,…,MODE>` here must equal the bound
 * node's access: `p` and topology are `.toReadOnly()` nodes ⇒ `read`;
 * `slots` is a read_write node ⇒ `read_write` (WGSLNodeBuilder.js:2155
 * emits the NODE's access; a mode mismatch is a shader compile error).
 * @see src/core/tangentPointEnergy.ts:250-378 (the loop being gathered)
 */
export function gradientKernelWgsl(p: KernelParams): string {
    const { alpha, epsilon } = p;
    return /* wgsl */ `
        fn tpGradient(
            p: ptr<storage, array<f32>, read>,
            edges: ptr<storage, array<u32>, read>,
            pairOffsets: ptr<storage, array<u32>, read>,
            pairIndices: ptr<storage, array<u32>, read>,
            slots: ptr<storage, array<f32>, read_write>,
            edgeCount: u32,
            I: u32
        ) -> void {
            if (I >= edgeCount) { return; }
            let alpha = ${f32(alpha)};
            let eps = ${f32(epsilon)};
            let i1 = edges[2u * I + 0u];
            let i2 = edges[2u * I + 1u];
            let eI = diff2(p, i2, i1);
            let reI = norm3(eI);
            var eIhat = vec3<f32>(0.0);
            if (reI > diffGuard(p, i2, i1)) { eIhat = eI * (1.0 / reI); }   // STRICT — D3
            let ellI = reI + eps;
            let ellIpow = ${powExpr('ellI', 1 - alpha)};
            let dPowCoeff = (1.0 - alpha) * ${powExpr('ellI', -alpha)};
            var g1 = vec3<f32>(0.0);
            var g2 = vec3<f32>(0.0);
            for (var k = pairOffsets[I]; k < pairOffsets[I + 1u]; k = k + 1u) {
                let J = pairIndices[k];
                let j1 = edges[2u * J + 0u];
                let j2 = edges[2u * J + 1u];
                let eJ = diff2(p, j2, j1);
                let ellJ = norm3(eJ) + eps;
                let ellJpow = ${powExpr('ellJ', 1 - alpha)};

                // ---- Role A: ordered pair (I,J), e = eI, d = v_i - v_j (plan D1)
                var sumF = 0.0;
                let baseA = 0.25 * ellIpow * ellJ;
                for (var ii = 0u; ii < 2u; ii = ii + 1u) {
                    for (var jj = 0u; jj < 2u; jj = jj + 1u) {
                        let i = select(i2, i1, ii == 0u);
                        let j = select(j2, j1, jj == 0u);
                        let kd = kernelDerivs(eI, diff2(p, i, j), diffGuard(p, i, j));
                        sumF = sumF + kd.f;
                        if (ii == 0u) { g1 = g1 + baseA * kd.df_dd; } else { g2 = g2 + baseA * kd.df_dd; }
                        g1 = g1 - baseA * kd.df_de;
                        g2 = g2 + baseA * kd.df_de;
                    }
                }
                let sI = 0.25 * ellJ * sumF;
                g1 = g1 + sI * dPowCoeff * (-eIhat);
                g2 = g2 + sI * dPowCoeff * eIhat;

                // ---- Role B: ordered pair (J,I), e = eJ, d = v_a - v_b, a in J, b in I (plan D1)
                var sumFb = 0.0;
                let baseB = 0.25 * ellJpow * ellI;
                for (var aa = 0u; aa < 2u; aa = aa + 1u) {
                    for (var bb = 0u; bb < 2u; bb = bb + 1u) {
                        let a = select(j2, j1, aa == 0u);
                        let b = select(i2, i1, bb == 0u);
                        let kd = kernelDerivs(eJ, diff2(p, a, b), diffGuard(p, a, b));
                        sumFb = sumFb + kd.f;
                        if (bb == 0u) { g1 = g1 - baseB * kd.df_dd; } else { g2 = g2 - baseB * kd.df_dd; }
                    }
                }
                let sJb = 0.25 * ellJpow * sumFb;   // dE via ell_J of (J,I): that "J" is our I
                g1 = g1 + sJb * (-eIhat);
                g2 = g2 + sJb * eIhat;
            }
            // ×0.5: disjointPairs lists both (I,J) and (J,I) — gradientAnalytical's final loop. @see plan D1
            let h1 = 0.5 * g1;
            let h2 = 0.5 * g2;
            slots[6u * I + 0u] = h1.x; slots[6u * I + 1u] = h1.y; slots[6u * I + 2u] = h1.z;
            slots[6u * I + 3u] = h2.x; slots[6u * I + 4u] = h2.y; slots[6u * I + 5u] = h2.z;
        }
    `;
}

/**
 * Vertex gather — one invocation per vertex v; sums its incident endpoint
 * slots (vertex→slot CSR, stored order) into grad[3v..3v+2]. No atomics.
 * `slots` is declared read_write here although only read: it is the SAME
 * read_write node the gradient kernel writes, and the pointer mode must
 * match the node's access (review r1 #2).
 * @see plan D1
 */
export function vertexGatherWgsl(): string {
    return /* wgsl */ `
        fn tpVertexGather(
            slots: ptr<storage, array<f32>, read_write>,
            incidentOffsets: ptr<storage, array<u32>, read>,
            incidentSlots: ptr<storage, array<u32>, read>,
            grad: ptr<storage, array<f32>, read_write>,
            vertexCount: u32,
            v: u32
        ) -> void {
            if (v >= vertexCount) { return; }
            var g = vec3<f32>(0.0);
            for (var k = incidentOffsets[v]; k < incidentOffsets[v + 1u]; k = k + 1u) {
                let s = incidentSlots[k];
                g = g + vec3<f32>(slots[3u * s + 0u], slots[3u * s + 1u], slots[3u * s + 2u]);
            }
            grad[3u * v + 0u] = g.x; grad[3u * v + 1u] = g.y; grad[3u * v + 2u] = g.z;
        }
    `;
}

/**
 * Per-edge partial energy, role A only (ordered pairs summed over ALL I
 * cover both orders), ×0.5 = calculateEnergy's `totalEnergy / 2`.
 * @see src/core/tangentPointEnergy.ts:48-113
 */
export function energyKernelWgsl(p: KernelParams): string {
    const { alpha, epsilon } = p;
    // ENTRY FUNCTION ONLY — pairKernelWgsl is attached as a wgslFn include (review r1 #1).
    return /* wgsl */ `
        fn tpEnergy(
            p: ptr<storage, array<f32>, read>,
            edges: ptr<storage, array<u32>, read>,
            pairOffsets: ptr<storage, array<u32>, read>,
            pairIndices: ptr<storage, array<u32>, read>,
            partials: ptr<storage, array<f32>, read_write>,
            edgeCount: u32,
            I: u32
        ) -> void {
            if (I >= edgeCount) { return; }
            let alpha = ${f32(alpha)};
            let eps = ${f32(epsilon)};
            let i1 = edges[2u * I + 0u];
            let i2 = edges[2u * I + 1u];
            let eI = diff2(p, i2, i1);
            let ellI = norm3(eI) + eps;
            let ellIpow = ${powExpr('ellI', 1 - alpha)};
            var acc = 0.0;
            for (var k = pairOffsets[I]; k < pairOffsets[I + 1u]; k = k + 1u) {
                let J = pairIndices[k];
                let j1 = edges[2u * J + 0u];
                let j2 = edges[2u * J + 1u];
                let ellJ = norm3(diff2(p, j2, j1)) + eps;
                var sumK = 0.0;
                for (var ii = 0u; ii < 2u; ii = ii + 1u) {
                    for (var jj = 0u; jj < 2u; jj = jj + 1u) {
                        let i = select(i2, i1, ii == 0u);
                        let j = select(j2, j1, jj == 0u);
                        sumK = sumK + kernelDerivs(eI, diff2(p, i, j), diffGuard(p, i, j)).f;
                    }
                }
                acc = acc + 0.25 * ellIpow * ellJ * sumK;
            }
            // /2 — calculateEnergy's `totalEnergy / 2` (both pair orders listed). @see tangentPointEnergy.ts:106-112
            partials[I] = 0.5 * acc;
        }
    `;
}
```

- [ ] **Step 4: Run** — PASS; `bunx tsc --noEmit` clean.
- [ ] **Step 5: Commit** — `feat(gpu): WGSL generators for tangent-point kernels (phase1 T1)`

---

### Task 2: Topology packing, interleaved hi/lo writer, guard mirrors (pure)

**Files:**
- Create: `src/gpu/topology.ts`
- Test: `test/gpu/topology.test.ts`

**Interfaces:**
- Produces:
  ```ts
  export interface PackedTopology {
      vertexCount: number; edgeCount: number;
      edges: Uint32Array;            // [i1,i2] per edge
      pairOffsets: Uint32Array;      // E+1, CSR
      pairIndices: Uint32Array;      // J's in disjointPairs[I] order
      incidentOffsets: Uint32Array;  // V+1, CSR
      incidentSlots: Uint32Array;    // slot = 2*edge + endpoint(0|1)
  }
  export function packTopology(vertexCount: number, edges: Edge[], disjointPairs: number[][]): PackedTopology;
  export function writeHiLoInterleaved(vertices: Vec3[], target: Float32Array): void;  // [hi.xyz, lo.xyz] per vertex
  export function diffDegenerate(a: Vec3, b: Vec3): boolean;   // D3 mirror
  export function crossDegenerate(e: Vec3, d: Vec3): boolean;  // D3 mirror
  ```

- [ ] **Step 1: Failing tests**

```ts
// test/gpu/topology.test.ts
import { expect, test } from 'bun:test';
import { nearTouchPair, splitHiLo, trefoil } from '../../src/core/fixtures';
import { calculateDisjointPairs } from '../../src/core/tangentPointEnergy';
import type { Vec3 } from '../../src/core/testConfigs';
import { crossDegenerate, diffDegenerate, packTopology, writeHiLoInterleaved } from '../../src/gpu/topology';

test('packTopology: exact layout on a 4-vertex chain', () => {
    const edges: [number, number][] = [[0, 1], [1, 2], [2, 3]];
    const dp = calculateDisjointPairs(edges); // [[2],[],[0]]
    const t = packTopology(4, edges, dp);
    expect(Array.from(t.edges)).toEqual([0, 1, 1, 2, 2, 3]);
    expect(Array.from(t.pairOffsets)).toEqual([0, 1, 1, 2]);
    expect(Array.from(t.pairIndices)).toEqual([2, 0]);
    expect(Array.from(t.incidentOffsets)).toEqual([0, 1, 3, 5, 6]);
    expect(Array.from(t.incidentSlots)).toEqual([0, 1, 2, 3, 4, 5]);
});

test('packTopology: CSR mirrors calculateDisjointPairs order on trefoil(12)', () => {
    const { vertices, edges } = trefoil(12);
    const dp = calculateDisjointPairs(edges);
    const t = packTopology(vertices.length, edges, dp);
    for (let I = 0; I < 12; I++)
        expect(Array.from(t.pairIndices.slice(t.pairOffsets[I], t.pairOffsets[I + 1]))).toEqual(dp[I]);
});

test('packTopology: rejects malformed topology with exact messages', () => {
    expect(() => packTopology(2, [[0, 5]], [[]])).toThrow('packTopology: vertex index out of range');
    expect(() => packTopology(2, [[0, 1]], [])).toThrow('packTopology: disjointPairs.length !== edges.length');
    expect(() => packTopology(2, [[0, 1]], [[0]])).toThrow('packTopology: self-pair');
    expect(() => packTopology(3, [[0, 1], [1, 2]], [[1], [0]])).toThrow('packTopology: listed pair shares a vertex');
});

test('writeHiLoInterleaved: [hi.xyz, lo.xyz] per vertex, equals splitHiLo, exact-length only', () => {
    const { vertices } = trefoil(5);
    const ref = splitHiLo(vertices);
    const out = new Float32Array(30);
    writeHiLoInterleaved(vertices, out);
    for (let i = 0; i < 5; i++) for (let d = 0; d < 3; d++) {
        expect(out[6 * i + d]).toBe(ref.hi[3 * i + d]);
        expect(out[6 * i + 3 + d]).toBe(ref.lo[3 * i + d]);
    }
    expect(() => writeHiLoInterleaved(vertices, new Float32Array(29))).toThrow('writeHiLoInterleaved: expected 30 floats, got 29');
});

test('D3 guards: coincident ⇒ degenerate; near-touch(1e-6) ⇒ not; collinear cross ⇒ degenerate (incl. guard exactly 0); scale-invariant', () => {
    const { vertices } = nearTouchPair(1e-6);
    expect(diffDegenerate(vertices[0], vertices[0])).toBe(true);
    expect(diffDegenerate([0, 0, 0], [0, 0, 0])).toBe(true);           // guard exactly 0 AND length 0 ⇒ degenerate (≤)
    expect(diffDegenerate(vertices[0], vertices[2])).toBe(false);
    const e: Vec3 = [4, 0, 0];
    const d: Vec3 = [0, -1e-6, 0];
    expect(crossDegenerate(e, d)).toBe(false);
    expect(crossDegenerate(e, [2, 0, 0])).toBe(true);                 // axis-aligned collinear: every product 0 ⇒ guard 0 ⇒ must still be degenerate
    expect(crossDegenerate([1, 1, 0], [2, 2, 0])).toBe(true);         // collinear, non-zero products
    const s = (v: Vec3): Vec3 => [v[0] * 0.05, v[1] * 0.05, v[2] * 0.05];
    expect(diffDegenerate(s(vertices[0]), s(vertices[2]))).toBe(false);
    expect(crossDegenerate(s(e), s(d))).toBe(false);
});

test('D3 guards on the crossing preset: its collinear disjoint pair (edges 0 and 2) is degenerate, its non-collinear pairs are not', () => {
    const { vertices, edges } = testConfigs.find((c) => c.id === 'crossing')!.generate();
    const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    const e0 = sub(vertices[edges[0][1]], vertices[edges[0][0]]);
    const d02 = sub(vertices[edges[0][0]], vertices[edges[2][0]]);   // v0 − v2, on the same line
    expect(crossDegenerate(e0, d02)).toBe(true);
    const d03 = sub(vertices[edges[0][0]], vertices[edges[3][0]]);   // v0 − v4, off the line
    expect(crossDegenerate(e0, d03)).toBe(false);
});
```
(add `import { testConfigs } from '../../src/core/testConfigs';` to the test file)

- [ ] **Step 2: Run** — FAIL (module missing).
- [ ] **Step 3: Implement**

```ts
// src/gpu/topology.ts — pure CPU-side packing + D3 guard mirrors.
// @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (Task 2, D1, D3)
import type { Edge, Vec3 } from '../core/testConfigs';
import { F32_U, GUARD_FACTOR } from './wgsl';

export interface PackedTopology {
    vertexCount: number; edgeCount: number;
    edges: Uint32Array; pairOffsets: Uint32Array; pairIndices: Uint32Array;
    incidentOffsets: Uint32Array; incidentSlots: Uint32Array;
}

/**
 * CSR-flatten edges, disjoint pairs (calculateDisjointPairs order preserved —
 * the GPU loops the same (I,J) sequence the CPU does) and vertex→endpoint-slot
 * incidence (slot = 2·edge + endpoint). Uploaded once per graphVersion
 * (spec §2.2). Never sorts: stored order is part of reproducibility.
 * Malformed input is a programmer error (throws), not a runtime fallback.
 */
export function packTopology(vertexCount: number, edges: Edge[], disjointPairs: number[][]): PackedTopology {
    const E = edges.length;
    if (disjointPairs.length !== E) throw new Error('packTopology: disjointPairs.length !== edges.length');
    const flat = new Uint32Array(2 * E);
    const deg = new Uint32Array(vertexCount + 1);
    for (let I = 0; I < E; I++) {
        const [a, b] = edges[I];
        if (a < 0 || b < 0 || a >= vertexCount || b >= vertexCount) throw new Error('packTopology: vertex index out of range');
        flat[2 * I] = a; flat[2 * I + 1] = b;
        deg[a + 1]++; deg[b + 1]++;
    }
    const pairOffsets = new Uint32Array(E + 1);
    for (let I = 0; I < E; I++) pairOffsets[I + 1] = pairOffsets[I] + disjointPairs[I].length;
    const pairIndices = new Uint32Array(pairOffsets[E]);
    for (let I = 0; I < E; I++) {
        for (let k = 0; k < disjointPairs[I].length; k++) {
            const J = disjointPairs[I][k];
            if (J === I) throw new Error('packTopology: self-pair');
            if (J < 0 || J >= E) throw new Error('packTopology: pair index out of range');
            const [a, b] = edges[I]; const [c, d] = edges[J];
            if (a === c || a === d || b === c || b === d) throw new Error('packTopology: listed pair shares a vertex');
            pairIndices[pairOffsets[I] + k] = J;
        }
    }
    const incidentOffsets = new Uint32Array(vertexCount + 1);
    for (let v = 0; v < vertexCount; v++) incidentOffsets[v + 1] = incidentOffsets[v] + deg[v + 1];
    const fill = incidentOffsets.slice(0, vertexCount);
    const incidentSlots = new Uint32Array(incidentOffsets[vertexCount]);
    for (let I = 0; I < E; I++) {
        incidentSlots[fill[flat[2 * I]]++] = 2 * I;
        incidentSlots[fill[flat[2 * I + 1]]++] = 2 * I + 1;
    }
    return { vertexCount, edgeCount: E, edges: flat, pairOffsets, pairIndices, incidentOffsets, incidentSlots };
}

/**
 * In-place interleaved [hi.xyz, lo.xyz] write (zero per-step allocation):
 * hi = fround(c), lo = fround(c − hi) — same split as fixtures.ts splitHiLo.
 * @see src/core/fixtures.ts (splitHiLo), spec §2.3
 */
export function writeHiLoInterleaved(vertices: Vec3[], target: Float32Array): void {
    if (target.length !== 6 * vertices.length)
        throw new Error(`writeHiLoInterleaved: expected ${6 * vertices.length} floats, got ${target.length}`);
    for (let i = 0; i < vertices.length; i++) for (let d = 0; d < 3; d++) {
        const c = vertices[i][d]; const h = Math.fround(c);
        target[6 * i + d] = h; target[6 * i + 3 + d] = Math.fround(c - h);
    }
}

/** D3 mirror of WGSL `diffGuard`: |a−b| below the two-float noise floor 32·u²·max|hi|. */
export function diffDegenerate(a: Vec3, b: Vec3): boolean {
    const k = GUARD_FACTOR * F32_U * F32_U;
    let g2 = 0; let n2 = 0;
    for (let d = 0; d < 3; d++) {
        const g = k * Math.max(Math.abs(Math.fround(a[d])), Math.abs(Math.fround(b[d])));
        g2 += g * g;
        const diff = a[d] - b[d]; n2 += diff * diff;
    }
    // `<=`, not `<`: a guard of exactly 0 with a length of exactly 0 IS degenerate (mirrors WGSL `if (r > guard)`). @see plan D3
    return Math.sqrt(n2) <= Math.sqrt(g2);
}

/** D3 mirror of WGSL `crossGuard`: |e×d| at or below 32·u·‖per-component product scale‖. */
export function crossDegenerate(e: Vec3, d: Vec3): boolean {
    const c = [e[1] * d[2] - e[2] * d[1], e[2] * d[0] - e[0] * d[2], e[0] * d[1] - e[1] * d[0]];
    const s = [Math.abs(e[1] * d[2]) + Math.abs(e[2] * d[1]), Math.abs(e[2] * d[0]) + Math.abs(e[0] * d[2]), Math.abs(e[0] * d[1]) + Math.abs(e[1] * d[0])];
    return Math.hypot(...c) <= GUARD_FACTOR * F32_U * Math.hypot(...s); // `<=` — see diffDegenerate
}
```

- [ ] **Step 4: Run** — PASS; tsc clean.
- [ ] **Step 5: Commit** — `feat(gpu): topology CSR + interleaved hi/lo writer + f32 guard mirrors (phase1 T2)`

---

### Task 3: `GpuTangentPoint` + reduction + T1/T2/T3 through the production kernel

**Split into three commits (review r1 #8) — 3a: Steps 1–3 (`compare.ts` refactor + `reduce.ts` + a reduction-only spike); 3b: Step 4 + Step 4½ (`GpuTangentPoint` + the kernel-compile/`trefoil(12)` smoke spike — the step that would have caught r1 #1/#2/#3); 3c: Steps 5–8 (`phase1Tolerance` gate run + JSON).** Each sub-task gets its own implementer + reviewer pass.

**Files:**
- Create: `src/gpu/reduce.ts`, `src/gpu/tangentPoint.ts`, `bench/gpu/compare.ts`, `bench/gpu/phase1.ts`
- Modify: `bench/gpu/spikes.ts` (import comparators/xorshift32 from `compare.ts`; `Object.assign(spikes, phase1Spikes)`)
- Test: `test/gpu/compare.test.ts`, `test/gpu/tangentPoint.test.ts` (constructor-domain only)
- Created by command: `bench/results/<date>-gpu-phase1-reduce.json` (3a), `…-phase1-kernel-smoke.json` (3b), `…-phase1-tolerance.json` (3c)

**Interfaces:**
- Produces:
  ```ts
  export const REDUCE_MAX_EDGES = 16_384;
  export class GpuTangentPoint {
      constructor(renderer: WebGPURenderer, topology: PackedTopology, params?: KernelParams /* DEFAULTS */);
      readonly vertexCount: number;
      gradient(vertices: Vec3[]): Promise<{ dE: Vec3[]; wallMs: number }>;
      energy(vertices: Vec3[]): Promise<{ energy: number; wallMs: number }>;
      dispose(): void;
  }
  export function makeReduceSum(input, n): { nodes: unknown[]; out }   // reduce.ts
  export function withSeed<T>(seed: number, run: () => T): T           // compare.ts
  ```

- [ ] **Step 1: Pure refactor + failing tests**

Move `relErrComparator`/`cosineComparator` (`bench/gpu/spikes.ts:763-786`) and `xorshift32` (`spikes.ts:111-121`) verbatim into `bench/gpu/compare.ts` (exported), import them back in `spikes.ts`; add `withSeed` (monkeypatches `Math.random` with `xorshift32(seed)` for the duration, restores in `finally`). Make `cosineComparator` zero-safe: both norms 0 ⇒ 1; one ⇒ −1.

```ts
// test/gpu/compare.test.ts
import { expect, test } from 'bun:test';
import { cosineComparator, relErrComparator, withSeed } from '../../bench/gpu/compare';
test('comparators + withSeed', () => {
    expect(relErrComparator(1.00001, 1)).toBeCloseTo(1e-5, 10);
    expect(cosineComparator([1, 0, 0], [1, 0, 0])).toBe(1);
    expect(cosineComparator([0, 0], [0, 0])).toBe(1);
    expect(cosineComparator([1, 0], [0, 0])).toBe(-1);
    expect(() => cosineComparator([1], [1, 2])).toThrow();
    const orig = Math.random;
    const a = withSeed(7, () => [Math.random(), Math.random()]);
    const b = withSeed(7, () => [Math.random(), Math.random()]);
    expect(a).toEqual(b);
    expect(Math.random).toBe(orig); // restored
});
```
```ts
// test/gpu/tangentPoint.test.ts — constructor-domain (no renderer access before the check)
import { expect, test } from 'bun:test';
import { GpuTangentPoint, REDUCE_MAX_EDGES } from '../../src/gpu/tangentPoint';
import { packTopology } from '../../src/gpu/topology';
test('GpuTangentPoint rejects edge counts beyond the two-pass reduction domain before touching the renderer', () => {
    const E = REDUCE_MAX_EDGES + 1;
    const edges = Array.from({ length: E }, (_, i) => [i, (i + 1) % E] as [number, number]);
    const dp = edges.map(() => []);
    const renderer = new Proxy({}, { get() { throw new Error('renderer touched'); } });
    expect(() => new GpuTangentPoint(renderer as never, packTopology(E, edges, dp))).toThrow(
        'GpuTangentPoint: edge count exceeds two-pass reduction limit',
    );
});
```

- [ ] **Step 2: Run** — both FAIL (modules missing).

- [ ] **Step 3: Reduction (spec §2.2; modeled on three r185 `examples/webgpu_compute_reduce.html` REDUCE 2, lines 543-579)**

API verification (CLAUDE.md FIRE — first use of these symbols in the repo): `workgroupArray` `node_modules/three/src/nodes/gpgpu/WorkgroupInfoNode.js:232`; `invocationLocalIndex` `core/IndexNode.js:157`; `workgroupId` `gpgpu/ComputeBuiltinNode.js:203`; `workgroupBarrier` `gpgpu/BarrierNode.js:75`; `compute(node, count, workgroupSize)` `gpgpu/ComputeNode.js:291`; `If` `tsl/TSLCore.js:1181`. All present in the installed 0.185.1 (checked 2026-08-30).

```ts
// src/gpu/reduce.ts — fixed 64-lane two-pass workgroup-memory tree sum (spec §2.2).
// No subgroupAdd (vendor-dependent tree), no Kahan/two-sum (WGSL §15.7.5 may reassociate) — PREC Q3/Q5.
// @see https://github.com/mrdoob/three.js/blob/r185/examples/webgpu_compute_reduce.html (REDUCE 2)
import { Fn, If, float, instancedArray, invocationLocalIndex, uint, workgroupArray, workgroupBarrier, workgroupId } from 'three/tsl';

export const REDUCE_LANES = 64;
/** Two 64-lane passes (128 loads each) cover ≤ 128·128 = 16,384 inputs. */
export const REDUCE_MAX_EDGES = 128 * 128;

type Arr = ReturnType<typeof instancedArray>;

/**
 * Pass 1: ceil(n/128) groups; lane l loads `base+l` and `base+l+64` (guarded),
 * tree strides 32..1 with a barrier after each; lane 0 writes the partial.
 * Pass 2: one group over ≤128 partials, same tree, writes out[0].
 * Deterministic tree shape ⇒ reproducible sums. Throws above REDUCE_MAX_EDGES.
 */
export function makeReduceSum(input: Arr, n: number): { nodes: unknown[]; out: Arr } {
    if (n > REDUCE_MAX_EDGES) throw new Error('makeReduceSum: input exceeds two-pass reduction limit');
    const groups = Math.max(1, Math.ceil(n / (2 * REDUCE_LANES)));
    const partials = instancedArray(groups, 'float');
    const out = instancedArray(1, 'float');
    const pass = (src: Arr, count: number, dst: Arr, dispatchGroups: number) =>
        Fn(() => {
            const shared = workgroupArray('float', REDUCE_LANES);
            const lane = invocationLocalIndex;
            const base = workgroupId.x.mul(uint(2 * REDUCE_LANES));
            const a = base.add(lane);
            const b = a.add(uint(REDUCE_LANES));
            const va = float(0).toVar('va');
            const vb = float(0).toVar('vb');
            If(a.lessThan(uint(count)), () => { va.assign(src.element(a)); });
            If(b.lessThan(uint(count)), () => { vb.assign(src.element(b)); });
            shared.element(lane).assign(va.add(vb));
            workgroupBarrier();
            for (const s of [32, 16, 8, 4, 2, 1]) {
                If(lane.lessThan(uint(s)), () => {
                    shared.element(lane).addAssign(shared.element(lane.add(uint(s))));
                });
                workgroupBarrier();
            }
            If(lane.equal(uint(0)), () => { dst.element(workgroupId.x).assign(shared.element(uint(0))); });
        })().compute(dispatchGroups * REDUCE_LANES, [REDUCE_LANES]);
    return { nodes: [pass(input, n, partials, groups), pass(partials, groups, out, 1)], out };
}
```

Reduction-only spike (3a's browser check; add to `bench/gpu/phase1.ts`): `phase1Reduce` fills an `instancedArray(n,'float')` with `1..n` for n ∈ {1, 63, 64, 65, 128, 129, 960, 16384}, runs `makeReduceSum`, reads `out[0]` and asserts it equals `n(n+1)/2` exactly (integers < 2²⁴ are exact in f32 — so any wrong tree/guard shows as a wrong integer, not as noise). Run: `bun bench/gpu/drive.ts phase1Reduce --out phase1-reduce` → all rows exact. **Commit 3a:** `feat(gpu): compare.ts refactor + 64-lane two-pass reduction, exact on 1..n (phase1 T3a)`.

- [ ] **Step 4: `GpuTangentPoint`** (3b)

```ts
// src/gpu/tangentPoint.ts — renderer-bound tangent-point kernels. The ONLY src/gpu file importing three/tsl.
// TSL plumbing + wgslFn body (spec §2.1); one device; one upload + one readback per call (spec §5).
// @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (Task 3, D1–D4)
import * as THREE from 'three/webgpu';
import { attributeArray, instanceIndex, instancedArray, uint, wgsl, wgslFn } from 'three/tsl';
import { DEFAULTS } from '../core/optimizer';
import type { Vec3 } from '../core/testConfigs';
import { REDUCE_LANES, REDUCE_MAX_EDGES, makeReduceSum } from './reduce';
import { type PackedTopology, writeHiLoInterleaved } from './topology';
import { energyKernelWgsl, gradientKernelWgsl, type KernelParams, pairKernelWgsl, vertexGatherWgsl } from './wgsl';

export { REDUCE_MAX_EDGES };

/**
 * Owns the storage buffers + compute nodes for one topology. `wallMs` covers
 * pack→upload→compute→readback→Vec3[] (plan D4; the ≥5× gate counts readback
 * inside the GPU dE phase — spec §5, review-3 F9). Energy is for gates and
 * the boot self-test only — never fed to descent (spec §2.3 never-mix).
 * New topology (graphVersion) ⇒ new instance; new (α,β,ε) ⇒ new instance.
 */
export class GpuTangentPoint {
    readonly vertexCount: number;
    private readonly positions; private readonly slots; private readonly grad; private readonly partials;
    private readonly gradNodes: unknown[]; private readonly energyNodes: unknown[]; private readonly energyOut;
    private disposed = false;

    constructor(private readonly renderer: THREE.WebGPURenderer, topology: PackedTopology, params: KernelParams = DEFAULTS) {
        if (topology.edgeCount > REDUCE_MAX_EDGES) throw new Error('GpuTangentPoint: edge count exceeds two-pass reduction limit');
        this.vertexCount = topology.vertexCount;
        // One dynamic positions buffer: [hi.xyz, lo.xyz] per vertex. setUsage verified: three/src/core/BufferAttribute.js:167.
        // .toReadOnly(): the kernels declare `p: ptr<storage, array<f32>, read>` and the emitted access comes from the NODE
        // (WGSLNodeBuilder.js:2155) — a read_write node bound to a `read` pointer param is a shader compile error (review r1 #2).
        // CPU upload via needsUpdate needs no shader write access.
        this.positions = attributeArray(new Float32Array(6 * topology.vertexCount), 'float').toReadOnly();
        this.positions.value.setUsage(THREE.DynamicDrawUsage);
        const edges = attributeArray(topology.edges, 'uint').toReadOnly();
        const pairOffsets = attributeArray(topology.pairOffsets, 'uint').toReadOnly();
        const pairIndices = attributeArray(topology.pairIndices, 'uint').toReadOnly();
        const incidentOffsets = attributeArray(topology.incidentOffsets, 'uint').toReadOnly();
        const incidentSlots = attributeArray(topology.incidentSlots, 'uint').toReadOnly();
        this.slots = instancedArray(6 * topology.edgeCount, 'float');
        this.grad = instancedArray(3 * topology.vertexCount, 'float');
        this.partials = instancedArray(topology.edgeCount, 'float');
        const common = { p: this.positions, edges, pairOffsets, pairIndices, edgeCount: uint(topology.edgeCount), I: instanceIndex };
        // Shared struct+helpers go in as an INCLUDE (CodeNode.js:170 wgsl(src, includes); FunctionNode.js:168
        // wgslFn(code, includes)) — three's WGSL parser accepts exactly one bare `fn` per wgslFn (review r1 #1).
        const shared = wgsl(pairKernelWgsl(params));
        const gradFn = wgslFn(gradientKernelWgsl(params), [shared]);
        const gatherFn = wgslFn(vertexGatherWgsl());
        const energyFn = wgslFn(energyKernelWgsl(params), [shared]);
        this.gradNodes = [
            gradFn({ ...common, slots: this.slots }).compute(topology.edgeCount, [REDUCE_LANES]),
            gatherFn({ slots: this.slots, incidentOffsets, incidentSlots, grad: this.grad, vertexCount: uint(topology.vertexCount), v: instanceIndex }).compute(topology.vertexCount, [REDUCE_LANES]),
        ];
        const red = makeReduceSum(this.partials, topology.edgeCount);
        this.energyOut = red.out;
        this.energyNodes = [energyFn({ ...common, partials: this.partials }).compute(topology.edgeCount, [REDUCE_LANES]), ...red.nodes];
    }

    private upload(vertices: Vec3[]): void {
        if (this.disposed) throw new Error('GpuTangentPoint: disposed');
        if (vertices.length !== this.vertexCount) throw new Error('GpuTangentPoint: vertex count mismatch');
        writeHiLoInterleaved(vertices, this.positions.value.array as Float32Array);
        this.positions.value.needsUpdate = true; // the one upload (G4-spike pattern, spikes.ts:660-668)
    }

    async gradient(vertices: Vec3[]): Promise<{ dE: Vec3[]; wallMs: number }> {
        const t0 = performance.now();
        this.upload(vertices);
        this.renderer.compute(this.gradNodes as never); // edge gather + vertex gather, ONE submit (spec §2.5)
        const flat = new Float32Array(await this.renderer.getArrayBufferAsync(this.grad.value));
        const dE: Vec3[] = Array.from({ length: this.vertexCount }, (_, i) => [flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]]);
        return { dE, wallMs: performance.now() - t0 };
    }

    async energy(vertices: Vec3[]): Promise<{ energy: number; wallMs: number }> {
        const t0 = performance.now();
        this.upload(vertices);
        this.renderer.compute(this.energyNodes as never); // kernel + 2 reduce passes, ONE submit
        const out = new Float32Array(await this.renderer.getArrayBufferAsync(this.energyOut.value));
        return { energy: out[0], wallMs: performance.now() - t0 };
    }

    dispose(): void {
        this.disposed = true;
        // ComputeNode.dispose (ComputeNode.js:116) clears pipelines/bindings; BufferAttribute.dispose (BufferAttribute.js:683)
        // dispatches 'dispose' so the backend frees the GPU buffer. Both are real methods in r185 — call them unconditionally.
        for (const n of [...this.gradNodes, ...this.energyNodes]) (n as { dispose: () => void }).dispose();
        for (const b of [this.positions, this.slots, this.grad, this.partials, this.energyOut]) b.value.dispose();
    }
}
```

Verify-step (FIRE): `attributeArray(...).value` is the `StorageBufferAttribute` (Phase 0 readback used `buf.value`, `spikes.ts:496`); `uint(n)` and `instanceIndex` as wgslFn scalar args follow the g0t pattern (`spikes.ts:176`). If `uint()` cannot be passed as a wgslFn argument in r185, bake the count into the WGSL source as a literal instead — record which in the commit message. **Access-mode check:** every `ptr<storage, …, MODE>` parameter mode must equal the bound node's access (`read` ⇔ `.toReadOnly()` node; `read_write` ⇔ plain `instancedArray`).

- [ ] **Step 4½: Kernel smoke spike (3b's browser check — compiles + matches CPU per vertex)**

Add `phase1KernelSmoke` to `bench/gpu/phase1.ts`: for `trefoil(12)`, `crossing` (collinear disjoint pair — D3 guard path) and `nearTouchPair(1e-6)`: build `GpuTangentPoint`, call `gradient()` and `energy()`; assert (a) no exception (shader compiled and bound), (b) every gradient component finite, (c) per-vertex `|dE_gpu − dE_cpu| / max(|dE_cpu|, 1e-12) < 1e-4` against `gradientAnalytical`, (d) energy relErr < 1e-5. **Pass criteria = (a), (b), (d); (c) is informational triage output** (per-vertex relative error is not a pre-registered gate — the gates are T1–T3 in Step 5). Return every per-vertex row so a wrong-role/wrong-sign term is visible immediately (D1 triage). Run: `bun bench/gpu/drive.ts phase1KernelSmoke --out phase1-kernel-smoke` → PASS. **Commit 3b:** `feat(gpu): GpuTangentPoint kernels compile and match CPU per-vertex on trefoil(12)/crossing/near-touch (phase1 T3b)`.

- [ ] **Step 5: Phase-1 tolerance spike** (3c)

```ts
// bench/gpu/phase1.ts — Phase 1 gates through the PRODUCTION kernel.
// @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §3 (T1–T3), §4 G2, §5 Phase 1
import * as THREE from 'three/webgpu';
import { nearTouchPair, trefoil } from '../../src/core/fixtures';
import { DEFAULTS } from '../../src/core/optimizer';
import { calculateDisjointPairs, calculateEnergy, gradientAnalytical } from '../../src/core/tangentPointEnergy';
import { type Edge, testConfigs, type Vec3 } from '../../src/core/testConfigs';
import { GpuTangentPoint } from '../../src/gpu/tangentPoint';
import { packTopology } from '../../src/gpu/topology';
import { cosineComparator, relErrComparator, withSeed } from './compare';

const { alpha, beta, epsilon } = DEFAULTS;
const engine = (r: THREE.WebGPURenderer, vertices: Vec3[], edges: Edge[]) => {
    const dp = calculateDisjointPairs(edges);
    return { dp, gpu: new GpuTangentPoint(r, packTopology(vertices.length, edges, dp)) };
};
const defaults = (c: (typeof testConfigs)[number]) => Object.fromEntries((c.params ?? []).map((q) => [q.name, q.default]));

export const phase1Spikes: Record<string, () => Promise<Record<string, unknown>>> = {};

/** T1 = single-pair energy (2-edge graph); T2/T3 over presets + trefoil 240/960; gap=1e-6 row doubles as G2-production. */
phase1Spikes.phase1Tolerance = async () => {
    const renderer = new THREE.WebGPURenderer();
    await renderer.init();
    const t1 = [];
    for (const gap of [1e-3, 1e-4, 1e-5, 1e-6]) {
        const { vertices, edges } = nearTouchPair(gap);
        const { dp, gpu } = engine(renderer, vertices, edges);
        try {
            const cpu64 = calculateEnergy(vertices, edges, dp, alpha, beta, epsilon);
            const { energy } = await gpu.energy(vertices);
            const relErr = relErrComparator(energy, cpu64);
            t1.push({ gap, gpu: energy, cpu64, relErr, pass: relErr < 1e-5 });
        } finally { gpu.dispose(); }
    }
    const fixtures = withSeed(0x5eed, () => [
        ...testConfigs.map((c) => ({ name: c.id, ...c.generate(defaults(c)) })),
        { name: 'trefoil240', ...trefoil(240) },
        { name: 'trefoil960', ...trefoil(960) },
    ]);
    const t23 = [];
    for (const f of fixtures) {
        const { dp, gpu } = engine(renderer, f.vertices, f.edges);
        try {
            const e64 = calculateEnergy(f.vertices, f.edges, dp, alpha, beta, epsilon);
            const g64 = gradientAnalytical(f.vertices, f.edges, dp, alpha, beta, epsilon);
            const { energy } = await gpu.energy(f.vertices);
            const { dE } = await gpu.gradient(f.vertices);
            const t2 = relErrComparator(energy, e64);
            const t3 = cosineComparator(dE.flat(), g64.flat());
            const finite = dE.every((v) => v.every(Number.isFinite)); // D3: a NaN anywhere (e.g. `crossing` collinear pair) fails the fixture
            t23.push({ name: f.name, nV: f.vertices.length, t2, t2pass: t2 < 1e-6, t3, t3pass: t3 > 1 - 1e-6, finite });
        } finally { gpu.dispose(); }
    }
    const pass = t1.every((r) => r.pass) && t23.every((r) => r.t2pass && r.t3pass && r.finite);
    return { gate: 'phase1Tolerance', seed: 0x5eed, t1, t23, g2Production: t1.find((r) => r.gap === 1e-6),
        worst: { t1: Math.max(...t1.map((r) => r.relErr)), t2: Math.max(...t23.map((r) => r.t2)), t3: Math.min(...t23.map((r) => r.t3)) }, pass };
};
```

Register in `bench/gpu/spikes.ts` before `window.__runSpike`: `import { phase1Spikes } from './phase1'; Object.assign(spikes, phase1Spikes);`

- [ ] **Step 6: Run** — `bun run dev` (separate shell); `bun bench/gpu/drive.ts phase1Tolerance --out phase1-tolerance`.
Expected: `data.pass === true`. Triage if red: T1 @1e-6 only → two-float op order in `diff2` (G2 passed in Phase 0 with the same arithmetic); T1 at all gaps → ε or pow form; T2 only @960 → D4 contingency; T3 → role-B (D1) first — compare per-vertex on `trefoil(12)` against `gradientAnalytical`. **Any red stops the branch after triage; never relax a tolerance.**
- [ ] **Step 7: Static** — `bunx tsc --noEmit`; `bun test test/gpu test/golden.test.ts`; `bunx biome check src/gpu bench/gpu`.
- [ ] **Step 8: Commit** — `feat(gpu): GpuTangentPoint kernels + 64-lane reduction; T1-T3 green through production kernel (phase1 T3)` with the results JSON.

---

### Task 4: Perf gate — GPU dE ≥5× CPU dE at N=480 and N=960 (readback counted)

**Files:**
- Modify: `bench/gpu/phase1.ts` (add `phase1DeGate`)
- Created by command: `bench/results/<date>-gpu-phase1-de-gate.json`

- [ ] **Step 1: Red first** — register `phase1DeGate` throwing `'phase1DeGate: not implemented'`; run `bun bench/gpu/drive.ts phase1DeGate --out phase1-de-gate` → hardware-classified FAIL with that message (proves the driver path is live; do not stage the file).
- [ ] **Step 2: Implement (method D4)**

```ts
const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];

phase1Spikes.phase1DeGate = async () => {
    const renderer = new THREE.WebGPURenderer({ trackTimestamp: true });
    await renderer.init();
    const rows = [];
    for (const N of [480, 960]) {
        const { vertices, edges } = trefoil(N);
        const { dp, gpu } = engine(renderer, vertices, edges);
        try {
            gradientAnalytical(vertices, edges, dp, alpha, beta, epsilon); // CPU warm-up
            await gpu.gradient(vertices);                                  // GPU warm-up (pipeline compile)
            const cpuMs: number[] = []; const gpuMs: number[] = []; const gpuOnly: number[] = [];
            for (let r = 0; r < 5; r++) {
                const { wallMs } = await gpu.gradient(vertices);
                gpuMs.push(wallMs);
                await renderer.resolveTimestampsAsync(THREE.TimestampQuery.COMPUTE);
                gpuOnly.push(renderer.info.compute.timestamp);
                const t0 = performance.now();
                gradientAnalytical(vertices, edges, dp, alpha, beta, epsilon);
                cpuMs.push(performance.now() - t0);
            }
            const speedup = median(cpuMs) / median(gpuMs);
            rows.push({ N, cpuMs, gpuMs, cpuP50Ms: median(cpuMs), gpuP50Ms: median(gpuMs), gpuComputeOnlyP50Ms: median(gpuOnly), speedup, pass: speedup >= 5 });
        } finally { gpu.dispose(); }
    }
    return { gate: 'phase1DeGate', methodology: 'wall-clock incl. pack+upload+compute+readback+Vec3[]; 1 warm-up, 5 runs, medians; same browser for CPU', fullStepGate: false, rows, pass: rows.every((r) => r.pass) };
};
```

- [ ] **Step 3: Run** — expected `speedup ≥ 5` at both N (Phase 0 G0t ≈1.5 TFLOP/s; the kernel is ~E²·8·60 flops ≈ 4e8 at N=960 → sub-ms compute; CPU dE at N=960 ≈ 0.5 s in Bun). If a row fails, rerun **once** to classify noise; disagreement ⇒ flaky ⇒ stop and investigate. Never average or cherry-pick runs. **FAIL → stop-branch per spec §5.**
- [ ] **Step 4: Commit** — `feat(bench/gpu): phase1 dE ≥5× gate — measured (phase1 T4)` with results JSON.

---

### Task 5: The `dE` seam in core + spec amendment

**Files:**
- Modify: `src/core/optimizer.ts` (`SobolevStepOptions`; the `dETpe` expression at :278-282), `src/core/dispatch.ts` (`DispatchDescentStepArgs.dE`; passthrough at the `sobolevStepSet` call)
- Modify: `docs/superpowers/specs/2026-08-13-webgpu-solver-design.md` §1 (sanctioned delta (iii)) and §2.3 (ε sentence per D2)
- Test: `test/gpu/dESeam.test.ts`

- [ ] **Step 1: Failing test**

```ts
// test/gpu/dESeam.test.ts
import { expect, test } from 'bun:test';
import { type DispatchDescentStepArgs, dispatchDescentStep } from '../../src/core/dispatch';
import { trefoil } from '../../src/core/fixtures';
import { DEFAULTS } from '../../src/core/optimizer';
import { calculateDisjointPairs, gradientAnalytical } from '../../src/core/tangentPointEnergy';
import type { Vec3 } from '../../src/core/testConfigs';

function args(): DispatchDescentStepArgs {
    const { vertices, edges } = trefoil(24);
    const x0: Vec3 = [0, 0, 0];
    for (const v of vertices) { x0[0] += v[0] / 24; x0[1] += v[1] / 24; x0[2] += v[2] / 24; }
    const L0 = edges.reduce((s, [a, b]) => s + Math.hypot(vertices[a][0] - vertices[b][0], vertices[a][1] - vertices[b][1], vertices[a][2] - vertices[b][2]), 0);
    return { descentMode: 'sobolev', vertices, edges, disjointPairs: calculateDisjointPairs(edges), mode: 'analytical', stepSize: 1,
        x0, barycenterConstraint: true, lengthMode: 'total', sobolevL0: L0, projectionMode: 'frozen', collectTimings: false, collectField: true };
}

test('dE seam: supplying the CPU analytical dE is toEqual-identical to omitting it', () => {
    const a = args();
    const dE = gradientAnalytical(a.vertices, a.edges, a.disjointPairs, DEFAULTS.alpha, DEFAULTS.beta, DEFAULTS.epsilon);
    expect(dispatchDescentStep({ ...args(), dE })).toEqual(dispatchDescentStep(a));
});

test('dE seam: a supplied dE is actually used (zero field ⇒ converged, vertices echoed)', () => {
    const a = args();
    const r = dispatchDescentStep({ ...a, dE: a.vertices.map(() => [0, 0, 0] as Vec3) });
    expect(r.converged).toBe(true);
    expect(r.vertices).toEqual(a.vertices);
});

test('dE seam: length mismatch throws', () => {
    expect(() => dispatchDescentStep({ ...args(), dE: [[0, 0, 0]] })).toThrow('dE length mismatch');
});
```

- [ ] **Step 2: Implement**

`src/core/optimizer.ts` — add to `SobolevStepOptions`:
```ts
    /**
     * Precomputed tangent-point differential at the INPUT vertices — the
     * WebGPU milestone's Phase 1 seam (spec §1 sanctioned delta (iii), §2.6).
     * When present it replaces the analytical/FD computation ONLY; penalties,
     * solve, projection, line search are untouched. MUST be dE_tpe of
     * `vertices` under the same α/β/ε — the caller owns that (the 'gpu'
     * driver computes it from the very vertices it passes). Absent ⇒ every
     * path bit-identical (golden suite). Not a numeric change: selects the
     * SOURCE of dE, not its arithmetic.
     * @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (D5, Task 5)
     */
    dE?: Vec3[];
```
and replace the `dETpe` computation (`optimizer.ts:278-282`) with:
```ts
            const dETpe = timed('dE', () =>
                opts.dE ??
                (opts.mode === 'analytical'
                    ? gradientAnalytical(vertices, edges, disjointPairs, alpha, beta, epsilon)
                    : gradientFiniteDiff(vertices, edges, disjointPairs, alpha, beta, epsilon, h)),
            );
```
The length check goes in `sobolevStepSet` **before** `timingsBegin()` (`optimizer.ts:262`) so a throw cannot leave the timing collector armed: `if (opts.dE && opts.dE.length !== vertices.length) throw new Error('sobolevStepSet: dE length mismatch');` as the first statement of the function body (review r1: previously sat inside the timed closure).

`src/core/dispatch.ts` — add `dE?: Vec3[];` to `DispatchDescentStepArgs` (TSDoc: "sobolev + ConstraintSet path only; the M1-legacy `sobolevStep` branch and the raw branch ignore it — `@see` the optimizer field") and pass `dE: args.dE` in the `sobolevStepSet` opts object.

Spec amendments (same commit):
- §1 after (ii): `(iii) [Phase 1 plan D5] SobolevStepOptions / DispatchDescentStepArgs gain optional dE?: Vec3[] — a precomputed differential at the input vertices; absent ⇒ bit-identical. Non-numeric: it selects the SOURCE of dE, not its arithmetic.`
- §2.3, replace "the GPU kernel is mathematically the ε=0 energy (ε=1e-10 is provably inert in f32), so degeneracy protection is an explicit branch…" with: "ε is added after every norm exactly as on the CPU (it is part of the energy definition and is NOT inert at near-touch scale — 5e-4 relative at gap 1e-6, Phase 1 plan D2); degeneracy protection is an explicit f32-scale branch, derived in the Phase 1 plan (D3)."
- §6 (device-loss section, the bullet "GPU energy is the ε=0 energy…", spec lines ~431-433): replace with "GPU energy/gradient use the same ε=1e-10 as the CPU (Phase 1 plan D2); only the degeneracy guard scale differs (f32, plan D3)." — the same claim must not survive in two places with two answers.

- [ ] **Step 3: Run** — `bun test test/gpu/dESeam.test.ts test/golden.test.ts test/optimizer-sobolev.test.ts test/sobolev` → PASS; full `bun test` green; tsc clean.
- [ ] **Step 4: Commit** — `feat(core): optional precomputed dE seam + spec §1 delta (iii), §2.3 ε refinement (phase1 T5)`

---

### Task 6: Store, boot self-test, driver module

**Files:**
- Modify: `src/store.ts` (`SolverDriver`, `gpuAvailable`, `setGpuAvailable`)
- Create: `src/gpu/selfTest.ts`, `src/gpu/driver.ts`
- Test: `test/store-solver-driver.test.ts` (extend), `test/gpu/selfTest.test.ts`, `test/gpu/driver.test.ts`

**Interfaces:**
- Produces: `export type SolverDriver = 'gpu' | 'worker' | 'main'`; store `gpuAvailable: boolean` (default `false`) + `setGpuAvailable(b: boolean)`; `selfTestVerdict(gpu, cpu64): { ok, relErr }`, `runGpuSelfTest(renderer): Promise<{ ok, relErr, error? }>`; `gpuStepSupported(s)`, `allFinite(v)`, `class GpuDriver { constructor(renderer); step(st, energyBefore): Promise<{ outcome, usedGpu }>; dispose() }`.

- [ ] **Step 1: Failing tests**

```ts
// test/store-solver-driver.test.ts — append
test('store: gpuAvailable defaults false; setSolverDriver accepts gpu; default driver unchanged', () => {
    expect(useSimStore.getState().solverDriver).toBe('worker');
    expect(useSimStore.getState().gpuAvailable).toBe(false);
    useSimStore.getState().setGpuAvailable(true);
    expect(useSimStore.getState().gpuAvailable).toBe(true);
    useSimStore.getState().setSolverDriver('gpu');
    expect(useSimStore.getState().solverDriver).toBe('gpu');
    useSimStore.getState().setSolverDriver('worker');
    useSimStore.getState().setGpuAvailable(false);
});
```
```ts
// test/gpu/selfTest.test.ts
import { expect, test } from 'bun:test';
import { selfTestVerdict } from '../../src/gpu/selfTest';
test('selfTestVerdict: relErr < 1e-5 passes; NaN/inf/≥1e-5 fail', () => {
    expect(selfTestVerdict(1 + 5e-6, 1).ok).toBe(true);
    expect(selfTestVerdict(1.001, 1).ok).toBe(false);
    expect(selfTestVerdict(Number.NaN, 1).ok).toBe(false);
    expect(selfTestVerdict(Number.POSITIVE_INFINITY, 1).ok).toBe(false);
});
```
```ts
// test/gpu/driver.test.ts
import { expect, test } from 'bun:test';
import { allFinite, gpuStepSupported } from '../../src/gpu/driver';
import { useSimStore } from '../../src/store';
const pen0 = useSimStore.getState().penalties; // the store's all-zero default
test('gpuStepSupported: sobolev + analytical + penalties-off only (spec §1)', () => {
    expect(gpuStepSupported({ descentMode: 'sobolev', mode: 'analytical', penalties: pen0 })).toBe(true);
    expect(gpuStepSupported({ descentMode: 'raw', mode: 'analytical', penalties: pen0 })).toBe(false);
    expect(gpuStepSupported({ descentMode: 'sobolev', mode: 'finiteDiff', penalties: pen0 })).toBe(false);
    expect(gpuStepSupported({ descentMode: 'sobolev', mode: 'analytical', penalties: { ...pen0, totalLength: 1 } })).toBe(false);
});
test('allFinite rejects NaN/inf', () => {
    expect(allFinite([[0, 1, 2]])).toBe(true);
    expect(allFinite([[0, Number.NaN, 2]])).toBe(false);
    expect(allFinite([[Number.POSITIVE_INFINITY, 0, 0]])).toBe(false);
});
```

- [ ] **Step 2: Implement**

`src/store.ts`: `export type SolverDriver = 'gpu' | 'worker' | 'main';` — update its TSDoc: `'gpu'` = main-thread GPU dE + CPU f64 everything else (Phase 1), selectable only when `gpuAvailable`; per-step worker fallback for unsupported configs; failure chain `gpu → worker → main`. Add `gpuAvailable: boolean` (default `false`) + `setGpuAvailable`, `@see` spec §2.6. Default `solverDriver` stays `'worker'`.

```ts
// src/gpu/selfTest.ts — boot gate: G2 through the production kernel, once per renderer/adapter.
// @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.6 (boot gates), §4 G2; PREC risk 1 (per-adapter)
import type { WebGPURenderer } from 'three/webgpu';
import { nearTouchPair } from '../core/fixtures';
import { DEFAULTS } from '../core/optimizer';
import { calculateDisjointPairs, calculateEnergy } from '../core/tangentPointEnergy';
import { GpuTangentPoint } from './tangentPoint';
import { packTopology } from './topology';

/** Pure verdict: T1 tolerance (spec §3); non-finite ⇒ fail. */
export function selfTestVerdict(gpu: number, cpu64: number): { ok: boolean; relErr: number } {
    const relErr = Math.abs(gpu - cpu64) / Math.abs(cpu64);
    return { ok: Number.isFinite(relErr) && relErr < 1e-5, relErr };
}

/** Never throws — any exception is a failed gate (⇒ `gpuAvailable=false`). */
export async function runGpuSelfTest(renderer: WebGPURenderer): Promise<{ ok: boolean; relErr: number; error?: string }> {
    let gpu: GpuTangentPoint | null = null;
    try {
        const { vertices, edges } = nearTouchPair(1e-6);
        const dp = calculateDisjointPairs(edges);
        gpu = new GpuTangentPoint(renderer, packTopology(vertices.length, edges, dp));
        const { energy } = await gpu.energy(vertices);
        return selfTestVerdict(energy, calculateEnergy(vertices, edges, dp, DEFAULTS.alpha, DEFAULTS.beta, DEFAULTS.epsilon));
    } catch (e) {
        return { ok: false, relErr: Number.NaN, error: e instanceof Error ? e.message : String(e) };
    } finally { gpu?.dispose(); }
}
```

```ts
// src/gpu/driver.ts — the 'gpu' solver driver (Phase 1: GPU dE, CPU everything else).
// @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §1 (served configs), §2.6, §5 Phase 1
import type { WebGPURenderer } from 'three/webgpu';
import { buildStepArgs, type DescentStepOutcome, dispatchDescentStep, type StepArgsSource } from '../core/dispatch';
import { penaltiesActive } from '../core/sobolev/penalties';
import type { Vec3 } from '../core/testConfigs';
import { GpuTangentPoint } from './tangentPoint';
import { packTopology } from './topology';

/** Spec §1: 'gpu' serves sobolev + analytical + penalties-off ONLY. */
export function gpuStepSupported(s: Pick<StepArgsSource, 'descentMode' | 'mode' | 'penalties'>): boolean {
    return s.descentMode === 'sobolev' && s.mode === 'analytical' && !penaltiesActive(s.penalties);
}

/** Non-finite GPU output (f32 overflow below gap≈4e-7, plan D2) must never reach the solve. */
export function allFinite(v: Vec3[]): boolean {
    for (const p of v) if (!Number.isFinite(p[0]) || !Number.isFinite(p[1]) || !Number.isFinite(p[2])) return false;
    return true;
}

export type GpuStepSource = StepArgsSource & { graphVersion: number; showArrows: boolean };

/**
 * Per-step orchestration. Engine rebuilt on graphVersion change (mirrors the
 * worker's §D4 topology cache). Non-finite dE ⇒ this step runs with CPU dE
 * (`usedGpu=false`, Viewer warns). Thrown errors propagate ⇒ Viewer flips the
 * driver to 'worker' (§2.6). `timings.dE` is overwritten with the GPU wall
 * time (pack→readback→Vec3[]) so Stats shows the honest phase cost (D4);
 * `timings.step` gains the same interval so `step ⊇ dE` stays true.
 */
export class GpuDriver {
    private engine: GpuTangentPoint | null = null;
    private topoVersion = -1;
    constructor(private readonly renderer: WebGPURenderer) {}

    async step(st: GpuStepSource, energyBefore: number | undefined): Promise<{ outcome: DescentStepOutcome; usedGpu: boolean }> {
        if (!this.engine || st.graphVersion !== this.topoVersion) {
            this.engine?.dispose();
            this.engine = new GpuTangentPoint(this.renderer, packTopology(st.live.length, st.graph.edges, st.disjointPairs));
            this.topoVersion = st.graphVersion;
        }
        // `buildStepArgs` passes `st.live` by REFERENCE (dispatch.ts:333) and PinControls.tsx:169-174 mutates
        // `live` in place across our `await` below — so copy ONCE here and hand the SAME copy to the kernel
        // and to the step; otherwise dE would be evaluated at different vertices than the step uses
        // (dE seam contract, Task 5 TSDoc). Do not "simplify" this to `args.vertices`.
        const vertices = st.live.map((v) => [v[0], v[1], v[2]] as Vec3);
        const args = { ...buildStepArgs(st, energyBefore), vertices };
        const { dE, wallMs } = await this.engine.gradient(vertices);
        const usedGpu = allFinite(dE);
        const outcome = dispatchDescentStep({ ...args, ...(usedGpu ? { dE } : {}), collectTimings: true, collectField: st.showArrows });
        if (usedGpu && outcome.timings) {
            outcome.timings.dE = { ms: wallMs, calls: 1 };
            const s = outcome.timings.step;
            if (s) outcome.timings.step = { ms: s.ms + wallMs, calls: s.calls };
        }
        return { outcome, usedGpu };
    }

    dispose(): void { this.engine?.dispose(); this.engine = null; this.topoVersion = -1; }
}
```

- [ ] **Step 3: Run** — the three test files PASS; `bun test` green (the union widening must not break `worker-solver.test.ts`); tsc clean.
- [ ] **Step 4: Commit** — `feat(gpu): 'gpu' SolverDriver, gpuAvailable, boot self-test, GpuDriver (phase1 T6)`

---

### Task 7: Viewer + ControlPanel integration

**Files:**
- Modify: `src/scene/Viewer.tsx` (`Simulation` component `:57`: refs/effects/useFrame; `gl` factory `:376-383`), `src/ui/ControlPanel.tsx`
- Verification: tsc + `bun test` + `phase1DriverSmoke` (Task 8) + headed manual check.

**Split into two commits (review r1 #8) — 7a: Step 2 items 1–3 only (generation-token refactor of the EXISTING worker path; no GPU code; behavior unchanged; `bun test` + headed check that worker driver + preset switch mid-run still work). 7b: Steps 1, 2 items 4–5, 3 (GPU branch + boot/device-loss hooks + UI).** 7a is a mechanical refactor a reviewer can verify as semantics-preserving; 7b is the part that needs the async/stale-result lens.

- [ ] **Step 1: Boot + device-loss hooks in the `gl` factory (`Viewer.tsx:376-383`)**

After `await renderer.init()`:
```ts
                // Phase 1 boot gate (spec §2.6): G2 through the production kernel, per adapter.
                runGpuSelfTest(renderer).then((r) => {
                    useSimStore.getState().setGpuAvailable(r.ok);
                    if (!r.ok) console.warn('gpu: boot self-test failed; GPU driver unavailable', r);
                });
                // Device loss / uncaptured error ⇒ GPU unavailable + fall back (spec §6).
                // r185 Renderer.js:1225/1247 — onError receives an info OBJECT although
                // @types/three Renderer.d.ts:264 says string; bridge without `any`.
                // Both hooks are UNBOUND prototype methods that use `this` (Renderer.js:1225/1247) — invoke with `.call(renderer, …)`.
                const prevLost = renderer.onDeviceLost;
                renderer.onDeviceLost = (info) => {
                    useSimStore.getState().setGpuAvailable(false);
                    if (useSimStore.getState().solverDriver === 'gpu') useSimStore.getState().setSolverDriver('worker');
                    prevLost.call(renderer, info);
                };
                const errHooks = renderer as unknown as { onError: (info: string | { message?: string }) => void };
                const prevErr = errHooks.onError;
                errHooks.onError = (info) => {
                    useSimStore.getState().setGpuAvailable(false);
                    if (useSimStore.getState().solverDriver === 'gpu') useSimStore.getState().setSolverDriver('worker');
                    prevErr.call(renderer, info);
                };
```

- [ ] **Step 2: `Simulation` component (`Viewer.tsx:57`)**

1. `const gl = useThree((s) => s.gl) as unknown as WebGPURenderer;` `const gpuDriverRef = useRef<GpuDriver | null>(null);` `const dispatchGeneration = useRef(0);`
2. Replace `const inFlight = useRef(false)` with `const inFlight = useRef<null | { backend: 'gpu' | 'worker'; generation: number }>(null)`; every existing `inFlight.current = false` → `= null`; existing `!inFlight.current` checks keep working; the worker send sets `{ backend: 'worker', generation: dispatchGeneration.current }`; `handleWorkerMessage` clears/applies only when `inFlight.current?.backend === 'worker' && inFlight.current.generation === dispatchGeneration.current` (else drops the result with the existing warn).
3. Worker lifecycle effect (`Viewer.tsx:207-246`): the FIRST two statements of the effect body, before any early return, are `dispatchGeneration.current++; inFlight.current = null;` (so a 'main' toggle also invalidates in-flight results). Then `const needsWorker = solverDriver !== 'main'; if (!needsWorker) return;` (worker alive under `'gpu'` for per-step fallback). Note: because the effect depends on `solverDriver`, toggling worker↔gpu recreates the worker — accepted for Phase 1 (cost = one worker boot + topology re-send; no correctness issue since the generation bump drops anything stale).
4. GPU lifecycle effect on `[solverDriver, gl, gpuAvailable]`: when `'gpu'`: if `!gpuAvailable` → `setSolverDriver('worker')`; else `gpuDriverRef.current = new GpuDriver(gl)`; cleanup `dispose()`, null ref, `inFlight.current = null`.
5. `useFrame` — replace `if (st.solverDriver === 'worker') {` with:
```ts
            const useWorker = st.solverDriver === 'worker' || (st.solverDriver === 'gpu' && !gpuStepSupported(st));
            if (st.solverDriver === 'gpu' && !useWorker) {
                const drv = gpuDriverRef.current;
                if (drv && !inFlight.current) {
                    const token = { backend: 'gpu' as const, generation: dispatchGeneration.current };
                    inFlight.current = token;
                    const gv = st.graphVersion;
                    drv.step(st, lastEnergy.current ?? undefined).then(({ outcome, usedGpu }) => {
                        if (inFlight.current === token) inFlight.current = null;
                        const now = useSimStore.getState();
                        // Same §D5 drop rule as the worker path + generation + driver still 'gpu'.
                        if (token.generation !== dispatchGeneration.current || gv !== now.graphVersion || !now.running || now.solverDriver !== 'gpu') return;
                        if (!usedGpu) console.warn('solverDriver: gpu dE non-finite; step used CPU dE');
                        const t = performance.now();
                        const elapsed = (t - lastResultTime.current) / 1000;
                        lastResultTime.current = t;
                        applyStepOutcome(outcome, elapsed);
                    }).catch((err) => {
                        console.error('solverDriver: gpu step failed; falling back to worker:', err);
                        if (inFlight.current === token) inFlight.current = null;
                        useSimStore.getState().setGpuAvailable(false);
                        useSimStore.getState().setSolverDriver('worker');
                    });
                }
            } else if (useWorker) {
```
(existing worker block body unchanged; `else` main branch unchanged).

- [ ] **Step 3: ControlPanel** — `const gpuAvailable = useSimStore((s) => s.gpuAvailable);` and `<option value="gpu" disabled={!gpuAvailable}>GPU (experimental){gpuAvailable ? '' : ' — unavailable'}</option>` first in the select.
- [ ] **Step 4: Verify** — `bunx tsc --noEmit`; `bun test`; `bunx biome check src/scene/Viewer.tsx src/ui/ControlPanel.tsx`. Headed manual (`bun run dev`, Chrome with `--enable-unsafe-webgpu`): select GPU → trefoil preset runs, Stats dE shows GPU ms; enable a penalty → keeps stepping (worker path), select still says GPU; disable → back to GPU; switch preset mid-run → no double-apply, no stale result.
- [ ] **Step 5: Commit** — `feat(viewer): 'gpu' driver in the frame loop with generation-token single-flight, worker fallback, UI option (phase1 T7)`

---

### Task 8: Driver smoke gate (T5 preview) + final verification + gate report

**Files:**
- Modify: `bench/gpu/phase1.ts` (add `phase1DriverSmoke`), `bench/gpu/README.md`
- Refresh by command: all three `phase1-*` results JSON at final HEAD

- [ ] **Step 1: Spike** — K=20 sobolev steps on `trefoil(120)` (barycenter + total-length, frozen, penalties off) from the same start: (a) CPU `dispatchDescentStep`; (b) `GpuTangentPoint.gradient` → `dispatchDescentStep({...args, dE})`; both chain `energyBefore` from the previous accepted step exactly as the Viewer does. Record per-step energies, `maxRelEnergyDiff`, `acceptedA/B`, `monotone` (no accepted step increases energy — CPU f64 energies, which is all Phase 1 has). Pass (preview, not the T5 gate): `maxRelEnergyDiff < 1e-2` and monotone.
- [ ] **Step 2: Run** — `bun bench/gpu/drive.ts phase1DriverSmoke --out phase1-driver-smoke` → PASS.
- [ ] **Step 3: Negative-constraint proofs** (with `BASE` = commit before Task 1):
```bash
git diff --exit-code "$BASE" -- src/core/tangentPointEnergy.ts src/core/sobolev   # numerics untouched
# only the dE? seam: exactly the lines listed in Task 5 (TSDoc block + field in each file, the length check, the `opts.dE ??` line, `dE: args.dE`). Assert a count ceiling, then eyeball:
test "$(git diff "$BASE" -- src/core/optimizer.ts src/core/dispatch.ts | grep -c '^[+-][^+-]')" -le 40 || { echo 'seam diff too large'; exit 1; }
git diff "$BASE" -- src/core/optimizer.ts src/core/dispatch.ts | grep '^[+-][^+-]'
git diff --exit-code "$BASE" -- package.json bun.lock
rg -n 'pow\(|subgroupAdd|atomic' src/gpu   # only powExpr's non-integer branch + comments may match
bun test && bunx tsc --noEmit && bun run lint && bun run knip && git diff --check
```
- [ ] **Step 4: Re-run all three gates at final HEAD** — `phase1Tolerance`, `phase1DeGate`, `phase1DriverSmoke`; confirm each JSON's `gitShaShort` is HEAD.
- [ ] **Step 5: README "Phase 1 gate report"** — table `| Gate | Fixture/method | Result | Threshold | Consequence |` with rows T1 (worst relErr, gap), T2 (worst), T3 (min cosine), G2-production, dE gate N=480 / N=960 (cpu p50 / gpu p50 / speedup), driver smoke; each cites its JSON field. State: all energies/solve/line search CPU f64; full-step speedup NOT a gate; default stays `worker`; G4 → Phase 3, G6 → Phase 2 unchanged; the ε refinement of spec §2.3.
- [ ] **Step 6: Commit** — `docs(bench/gpu): phase1 gate report + final gate refresh (phase1 T8)`

---

## Execution & review protocol

`superpowers:subagent-driven-development` in a worktree on `feat/webgpu-phase1`. One fresh implementer per task (Sonnet unless the task fails CLAUDE.md's "simple" test — Tasks 1, 3b, 3c, 7b do: use the orchestrator's model; 3a and 7a are mechanical and may go to Sonnet), one task-scoped reviewer per task, whole-branch review after Task 8. Reviewer lenses: Tasks 1–3 numerical/WGSL; Task 7 async state/stale-result; Tasks 4, 8 benchmark methodology. Nits → CLAUDE.md inline-fix gate; blockers fixed on branch.

## Stop conditions
- Any T1/T2/T3 red, INVALID adapter, or speedup < 5 at either N ⇒ Phase 1 does not merge; record and report.
- Any `src/core` diff beyond the Task 5 seam lines, any dependency/lockfile change, a second device, or a CPU/GPU energy comparison inside one run ⇒ spec violation, remove.
- Flaky perf pass/fail (two runs disagree) ⇒ investigate before merging.

## Self-review (done while writing)
- Spec coverage: §2.1 → T3; §2.2 (gather, tree, topology per graphVersion) → T1/T3/T6; §2.3 → D2/D3 + T1; §2.6 (driver, boot gate, fallback chain) → T6/T7; §3 T1–T3 → T3; §5 gate → T4; §6 device loss → T7; §7 readback strategy (per-step `getArrayBufferAsync`, no pipelining in Phase 1) + threshold derivation (D3) answered.
- Commit granularity: Task 3 → 3a (reduce), 3b (kernels + smoke), 3c (tolerance gate); Task 7 → 7a (token refactor, no GPU), 7b (GPU branch + hooks + UI). Each sub-commit has its own implementer + reviewer pass.
- Type consistency: `GpuTangentPoint(renderer, topology, params?)` in T3/T6; `gradient → {dE, wallMs}` in T3/T4/T6/T8; `dE?: Vec3[]` identical across optimizer/dispatch/driver; `gpuStepSupported` in T6/T7; `inFlight` token shape in T7 only.
- Placeholders: none.

## Deferred (NOT here)
GPU-resident gradient consumption, readback pipelining, GPU energy in descent, solve/line search → Phase 2 (after G5). Curve.tsx / position authority / G7 → Phase 3.
