# Design: Consolidate & optimize the tangent-point energy hot path

- **Date:** 2026-07-01
- **Status:** Approved (design); spec reviewed (Opus subagent) and revised
- **Scope:** Performance quick-wins #1–3 on the interactive energy/gradient compute path, plus the consolidation needed to verify them safely.

## Goal

Speed up the interactive tangent-point-energy gradient/energy computation (the O(E²) disjoint-edge-pair loop) by ~5–10× **without changing any results**, and remove the duplicated math so the optimization is guarded by the existing finite-difference test.

## Background

The same energy + analytical-gradient math currently exists in **two** near-identical copies:

- `src/index.tsx` — the copy the app actually runs (`calculateEnergy`, `gradientAnalytical`, `gradientFiniteDiff`, `calculateDisjointPairs`). 3D-only (`Vec3`).
- `test_gradient.ts` — a separate copy plus an independent central-difference checker (`checkGradients`) that validates the analytical gradient. Has a `dimension` (2D/3D) param.

The hot path is the O(E²) loop; the current helpers (`subtract`/`scale`/`add`/`cross3D`/`norm`/`dot`) allocate a fresh array on every call, dozens of times per edge pair. A benchmark of the equivalent loop measured ~9–20× from removing those allocations, using flat `Float64Array`, and an integer-exponent `Math.pow` fast path, with results unchanged (bit-identical for the allocation/flat-array parts; within tight tolerance once the `pow` fast path is added — see Verification).

## Scope

**In:** consolidate the core math into one module; apply optimizations #1–3 to it; keep results provably unchanged.

**Out (tracked in issue #1):** flat `Float64Array` through UI/state/rendering; React/rAF loop fixes (#4–5); Barnes–Hut/BVH; WASM-SIMD; WebGPU.

## Design

### New module: `src/tangentPointEnergy.ts`

Single source of truth for the core math. Public exports:

- `calculateDisjointPairs(edges): number[][]`
- `calculateEnergy(vertices: Vec3[], edges, disjointPairs, alpha, beta, epsilon): number`
- `gradientAnalytical(vertices: Vec3[], edges, disjointPairs, alpha, beta, epsilon): Vec3[]`
- `gradientFiniteDiff(...)` — the app's existing forward-diff, moved verbatim (used by the viewer's "Finite Diff" mode, for both the gradient overlay and descent).

The viewer also uses `norm` on a gradient vector when drawing arrows; it will import that (or an equivalent) from the module rather than keep a private copy.

The **interface stays `Vec3[]` in / `Vec3[]`|`number` out**. The flat `Float64Array` representation lives **only inside** these functions: each flattens its `Vec3[]` inputs on entry and un-flattens the gradient on return. The O(V) conversion is negligible against the O(E²) loop, and call sites barely change.

### Authoritative source & consolidation

`src/index.tsx`'s current implementation is the **one authoritative copy**. It is moved **verbatim** into the module (Step 1, no logic edits). Then:

- `src/index.tsx` imports the four functions from the module and deletes its inline copies. **App behavior is byte-for-byte unchanged.**
- `test_gradient.ts` imports `calculateEnergy`, `gradientAnalytical`, `calculateDisjointPairs` (the code under test) from the module, and **keeps its own independent central-difference checker + `checkGradients`** as test infrastructure. Its 2D square config is expressed as 3D with `z=0`.

We do **not** hand-merge the two copies into a new version.

**Signature reconciliation (the reorder trap).** The two copies do not share a signature, so `test_gradient.ts`'s call sites must be *rewritten* to the module's API, not blindly re-pointed:

- `calculateEnergy` — module: `(vertices, edges, disjointPairs, alpha, beta, epsilon)`; the test's copy: `(vertices, edges, alpha, beta, disjointPairs, dimension, epsilon)` — `disjointPairs` is swapped past `alpha, beta` and there is an extra `dimension`. The kept central-diff checker's internal call must be reordered accordingly.
- `gradientAnalytical` — the first six args already align; drop the trailing `dimension`.
- Rename `calculateDisjointEdgePairs` → `calculateDisjointPairs` at the test call sites.
- Delete the test's now-unused local vector helpers (else knip flags them).

**Types for the `tsc --noEmit` gate.** The module's params are `Vec3[]`/`Edge[]` (tuples); the test's config literals infer as `number[][]`, which is **not** assignable to tuple types. Annotate the test configs as `Vec3[]`/`Edge[]`, and retype Test 1's square from `[x,y]` to `[x,y,0]` — mandatory for the typecheck gate, not merely "expressed as z=0".

### 3D-only; 2D embeds as z=0

The module is 3D-only. A planar 2D config with `z=0` produces identical results: `cross3D(e,d)` → `[0,0,cross2D(e,d)]`, so `‖cross3D‖` = `|cross2D|` (energy), and the gradient matches too — the x/y contributions equal the old 2D branch and every z-component is identically 0. So `test_gradient.ts`'s 2D square becomes a `z=0` 3D config and Test 1 (which checks the gradient) stays green. This removes dimension branching and lets the kernel unroll to x/y/z scalars.

### Optimizations (#1–3), applied after the verbatim move

1. **No per-iteration allocations** — inline the vector helpers to scalar x/y/z locals in the hot loop.
2. **Flat `Float64Array`** buffers for vertices and gradient (stride 3), internal to the compute functions.
3. **Integer-exponent `Math.pow` fast path** — detect integer-ness **per exponent** (not one global switch): use `x*x*x`-style multiplication when an exponent is a non-negative integer, else fall back to `Math.pow`. The app default α=3/β=6 is all-integer; **Test 3 is α=2 (integer), β=4.5 (fractional)**, so the β-side must take the fallback while the α-side is fast-pathed. Also reuse `c^(α-1)·c = c^α` and `d^β·d = d^(β+1)` to cut `pow` calls. This step changes low-order bits (see Verification).

### Invariants the inlined kernel must preserve

When unrolling helpers to x/y/z scalars, keep these exactly (most likely to drift):

- `epsilon` is added **after** the norm: `c_eps = ‖e×d‖ + ε`, `d_eps = ‖d‖ + ε`.
- The direction-zeroing guards use the **pre-epsilon** norm: `r < 1e-14` (unit of `d`), `rc >= 1e-14` (unit of the cross vector).
- The `/2` symmetry factor is applied to **both** the energy and the gradient (`disjointPairs` contains both `(I,J)` and `(J,I)`).
- `dot`/`cross` summation stays in x→y→z order — this is what keeps opts #1/#2 bit-identical.

No circular import: `tangentPointEnergy.ts` imports only *types* from `testConfigs.ts` (which imports nothing back), so the chain `testConfigs ← tangentPointEnergy ← {index.tsx, test_gradient.ts}` is acyclic.

## Verification (the safety net)

**Golden-output guard.** Capture golden **before any change**, from the *pre-change* `index.tsx` code (via a temporary export / throwaway harness — not from the freshly-moved module, which would validate the move against itself).

- **Freeze the inputs.** Several configs (`createStressTest`, `createRandomGraph`, `createRandomChain`) call `Math.random()` in `generate()`, so re-generating yields different graphs. Golden must store the concrete `{vertices, edges}` arrays it was computed from and replay *those exact arrays* through both old and new code.
- **Cover the fractional exponent.** The app hardcodes α=3/β=6 (all integer), so also include an **α=2, β=4.5** case in the golden set — otherwise opt #3's fractional fallback is guarded only by Test 3's loose `checkGradients` tolerance.
- **Compare with the right metric.** Energy is a positive sum → relative error is fine. The **gradient** has components that are ~0 or nearly cancel, so pure relative error explodes there (the same near-zero trap `checkGradients` already solves). Use the combined tolerance `|a−b| ≤ atol + rtol·max(|a|,|b|)` (atol=1e-6, rtol=1e-5) for the gradient compare.

Step-by-step, each step gated:

1. **Verbatim move**, **opt #1 (inline scalars)**, and **opt #2 (flat arrays)** preserve the floating-point operation order → assert **bit-identical** to golden. Plus: `test_gradient.ts` 3/3, `src/index.tsx` builds, `bunx tsc --noEmit` clean, lefthook green.
2. **Opt #3 (integer-pow)** changes low-order bits → assert the **combined abs+rel** gradient tolerance above (relative on energy), *not* bit-identical. Same 3/3 + build + typecheck gates. Re-run the benchmark after each opt to record the speedup.

Because every step is diffed against the original's actual numbers, the calculation cannot silently change (DRY guardrail).

## Success criteria

- The core energy/gradient math lives in exactly one place (`index.tsx`'s inline copies deleted); knip reports no unused exports/files.
- App behavior unchanged (verbatim move + opts #1/#2 verified bit-identical; opt #3 matches golden within the combined abs+rel tolerance `atol=1e-6, rtol=1e-5`).
- `test_gradient.ts` 3/3 pass; `bunx tsc --noEmit` clean; `src/index.tsx` builds; lefthook green.
- Measured speedup on the O(E²) loop recorded (target ~5–10×).

## Follow-ups

Deferred work is tracked in **issue #1** (full Float64Array data model, React/rAF loop, Barnes–Hut, WASM-SIMD, WebGPU, rendering batching).
