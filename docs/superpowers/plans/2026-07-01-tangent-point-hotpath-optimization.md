# Tangent-Point Hot-Path Consolidation & Optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the tangent-point energy/gradient math from `src/index.tsx` into one shared module, then optimize it (~5–10× on the O(E²) loop) with results provably unchanged.

**Architecture:** Move the *live* `index.tsx` implementation verbatim into `src/tangentPointEnergy.ts`; both `index.tsx` and `test_gradient.ts` import it. A golden-output fixture captured from the *pre-change* code + the existing finite-diff test gate every step: bit-identical for the verbatim move and the allocation/flat-array opts, combined abs+rel tolerance once the integer-pow fast path perturbs low bits.

**Tech Stack:** Bun 1.3.11, TypeScript (strict), React 19 (viewer only), `bun:test`, biome/knip/lefthook.

**Spec:** `docs/superpowers/specs/2026-07-01-tangent-point-hotpath-optimization-design.md`

---

## File Structure

- **Create `src/tangentPointEnergy.ts`** — single source of truth: `calculateDisjointPairs`, `calculateEnergy`, `gradientAnalytical`, `gradientFiniteDiff`, `norm` (exported for the viewer's arrow drawing). Imports only *types* (`Vec3`, `Edge`) from `./testConfigs`. No DOM, no side effects.
- **Create `test/golden.json`** — fixture: for each frozen config, the exact `{vertices, edges, alpha, beta, epsilon}` plus the `energy` and `gradient` computed by the *pre-change* code.
- **Create `test/golden.test.ts`** — replays each fixture's exact arrays through the module and compares (strict `===` first; combined tolerance after opt #3).
- **Modify `src/index.tsx`** — delete the inline math (lines 5–258), import from the module.
- **Modify `test_gradient.ts`** — import `calculateEnergy`/`gradientAnalytical`/`calculateDisjointPairs` from the module (reconciling the different signature), keep its own central-diff checker, retype configs to `Vec3[]`/`Edge[]` with `z=0`.

Current line anchors in `src/index.tsx` (pre-change): helpers `6–25`, `calculateDisjointPairs` `28–42`, `calculateEnergy` `45–81`, `gradientFiniteDiff` `84–106`, `gradientAnalytical` `109–258`, viewer `norm` use `470`, entry `createRoot` `727`.

---

## Task 1: Capture the golden baseline from pre-change code

Golden MUST come from the current `index.tsx` code, before any move. `index.tsx` runs `createRoot(...)` at import (needs DOM), so we temporarily guard it and export the functions, capture, then revert.

**Files:**
- Modify (temporary, reverted this task): `src/index.tsx`
- Create (temporary, deleted this task): `capture-golden.ts`
- Create: `test/golden.json`

- [ ] **Step 1: Temporarily export the functions and guard the DOM entry**

In `src/index.tsx`, add `export` to three declarations:
- `function calculateDisjointPairs(` → `export function calculateDisjointPairs(` (line 28)
- `function calculateEnergy(` → `export function calculateEnergy(` (line 45)
- `function gradientAnalytical(` → `export function gradientAnalytical(` (line 109)

And guard the entry (line 727):
```tsx
if (typeof document !== 'undefined') {
    createRoot(document.getElementById('root')!).render(<App />);
}
```

- [ ] **Step 2: Write the capture script**

Create `capture-golden.ts` at the repo root:
```ts
import { writeFileSync } from 'fs';
import { calculateEnergy, gradientAnalytical, calculateDisjointPairs } from './src/index.tsx';
import type { Vec3, Edge } from './src/testConfigs';

const epsilon = 1e-10;

// Deterministic configs (no Math.random): 2D-as-z=0, 3D, and a fractional-exponent case.
const configs: { name: string; alpha: number; beta: number; vertices: Vec3[]; edges: Edge[] }[] = [
    {
        name: 'square2D_z0', alpha: 3, beta: 6,
        vertices: [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        edges: [[0, 1], [1, 2], [2, 3], [3, 0]],
    },
    {
        name: 'zigzag3D', alpha: 3, beta: 6,
        vertices: [[0, 0, 0], [1, 0, 0], [1.5, 1, 0.5], [0.5, 1.5, 0], [0, 1, -0.5]],
        edges: [[0, 1], [1, 2], [2, 3], [3, 4]],
    },
    {
        name: 'planar_a2_b45', alpha: 2, beta: 4.5,
        vertices: [[0, 0, 0], [1, 0.1, 0], [2, 0, 0], [0, 2, 0], [1, 2.1, 0], [2, 2, 0]],
        edges: [[0, 1], [1, 2], [3, 4], [4, 5]],
    },
];

// Pre-change signatures: calculateEnergy(vertices, edges, disjointPairs, alpha, beta, epsilon);
// gradientAnalytical(vertices, edges, disjointPairs, alpha, beta, epsilon).
const golden = configs.map(c => {
    const disjoint = calculateDisjointPairs(c.edges);
    return {
        name: c.name, alpha: c.alpha, beta: c.beta, epsilon,
        vertices: c.vertices, edges: c.edges,
        energy: calculateEnergy(c.vertices, c.edges, disjoint, c.alpha, c.beta, epsilon),
        gradient: gradientAnalytical(c.vertices, c.edges, disjoint, c.alpha, c.beta, epsilon),
    };
});

writeFileSync('test/golden.json', JSON.stringify(golden, null, 2) + '\n');
console.log(`Wrote test/golden.json (${golden.length} configs)`);
```

- [ ] **Step 3: Create the test dir and run the capture**

Run: `mkdir -p test && bun run capture-golden.ts`
Expected: `Wrote test/golden.json (3 configs)` and a `test/golden.json` containing three entries, each with non-zero `energy` and a `gradient` array of `[x,y,z]` triples.

- [ ] **Step 4: Sanity-check the fixture**

Run: `bun -e "const g=require('./test/golden.json'); console.log(g.map(c=>[c.name, c.energy.toExponential(3), c.gradient.length]))"`
Expected: three rows; energies finite and > 0; gradient lengths 4, 5, 6 respectively.

- [ ] **Step 5: Revert the temporary index.tsx edits and remove the script**

Run: `git checkout src/index.tsx && rm capture-golden.ts`
Expected: `git status --short` shows only `?? test/golden.json` (plus the untracked plan/spec if not yet committed).

- [ ] **Step 6: Commit the fixture**

```bash
git add test/golden.json
git commit -m "test: capture golden energy/gradient baseline (pre-optimization)"
```

---

## Task 2: Create the module (verbatim move) + golden test, rewire callers

**Files:**
- Create: `src/tangentPointEnergy.ts`
- Create: `test/golden.test.ts`
- Modify: `src/index.tsx` (delete lines 5–258, add import)
- Modify: `test_gradient.ts` (rewire imports, reconcile signatures, retype configs)

- [ ] **Step 1: Create the module by moving the code verbatim**

Create `src/tangentPointEnergy.ts`. Move **unchanged** from the current `src/index.tsx`:
- the type import: `import type { Vec3, Edge } from './testConfigs';`
- the six helpers (lines 6–25: `cross3D`, `dot`, `subtract`, `scale`, `norm`, `add`)
- `calculateDisjointPairs` (28–42), `calculateEnergy` (45–81), `gradientFiniteDiff` (84–106), `gradientAnalytical` (109–258)

Add `export` to: `norm`, `calculateDisjointPairs`, `calculateEnergy`, `gradientFiniteDiff`, `gradientAnalytical`. Do **not** edit any logic, whitespace, or operation order — this is a pure relocation (DRY guardrail).

- [ ] **Step 2: Verify the move is byte-for-byte verbatim**

Run (compares `index.tsx` lines 5–258 against the module body with the prepended type-import line removed and `export ` prefixes stripped):
```bash
diff <(git show HEAD:src/index.tsx | sed -n '5,258p') \
     <(grep -v '^import type' src/tangentPointEnergy.ts | sed 's/^export //')
```
Expected: **no output** (byte-identical). If anything differs, the move was not verbatim — fix before continuing. (The bit-identical golden test in Step 4 is the real correctness gate; this check just guards against transcription errors.)

- [ ] **Step 3: Write the golden test (strict/bit-identical mode)**

Create `test/golden.test.ts`:
```ts
import { test, expect } from 'bun:test';
import { readFileSync } from 'fs';
import { calculateEnergy, gradientAnalytical, calculateDisjointPairs } from '../src/tangentPointEnergy';
import type { Vec3, Edge } from '../src/testConfigs';

// Load at runtime (avoids needing resolveJsonModule in tsconfig; test/** is typechecked).
const golden = JSON.parse(readFileSync(new URL('./golden.json', import.meta.url), 'utf8')) as any[];

// STRICT uses toBe (Object.is): valid only while results are bit-identical AND finite
// (JSON has no -0/NaN/Infinity). Flip to false once opt #3 (integer-pow) perturbs low bits.
const STRICT = true;
const ATOL = 1e-6;
const RTOL = 1e-5;

for (const c of golden) {
    test(`golden: ${c.name}`, () => {
        const vertices = c.vertices as Vec3[];
        const edges = c.edges as Edge[];
        const disjoint = calculateDisjointPairs(edges);

        const energy = calculateEnergy(vertices, edges, disjoint, c.alpha, c.beta, c.epsilon);
        const grad = gradientAnalytical(vertices, edges, disjoint, c.alpha, c.beta, c.epsilon);

        if (STRICT) {
            expect(energy).toBe(c.energy);
        } else {
            expect(Math.abs(energy - c.energy)).toBeLessThanOrEqual(1e-9 * Math.max(1, Math.abs(c.energy)));
        }

        for (let v = 0; v < grad.length; v++) {
            for (let d = 0; d < 3; d++) {
                const got = grad[v][d];
                const want = c.gradient[v][d];
                if (STRICT) {
                    expect(got).toBe(want);
                } else {
                    expect(Math.abs(got - want)).toBeLessThanOrEqual(ATOL + RTOL * Math.max(Math.abs(got), Math.abs(want)));
                }
            }
        }
    });
}
```

- [ ] **Step 4: Run the golden test — must pass bit-identical**

Run: `bun test test/golden.test.ts`
Expected: 3 pass. (The module is the same code that produced golden, so equality is exact.) If any fail, the move was not verbatim — revisit Step 1.

- [ ] **Step 5: Rewire `src/index.tsx` to import from the module**

Delete lines 5–258 (the `// Vector math helpers` block through the end of `gradientAnalytical`). Replace the React import region so the top reads:
```tsx
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import { testConfigs, type GraphState, type Vec3 } from './testConfigs';
import {
    calculateDisjointPairs,
    calculateEnergy,
    gradientAnalytical,
    gradientFiniteDiff,
    norm,
} from './tangentPointEnergy';
```
Note `type Edge` is **dropped** from the `./testConfigs` import — after the math is removed, `Edge` is unused in `index.tsx` (its only remaining `edges` reference is `graph.edges.length` at line 694, a value access), so biome `noUnusedImports` would flag it. Keep `Vec3` (used at the `negGrad` annotation ~line 474) and `GraphState` (state type ~line 359). Leave the rest of `index.tsx` (UI helpers `loadSavedConfig`, `rotate3D`, `project`, `projectDirection`, `logScale`, `App`, entry) untouched — all call sites already use these names/signatures, so no call-site edits are needed.

- [ ] **Step 6: Rewire `test_gradient.ts` — imports, signatures, types**

At the top of `test_gradient.ts`, replace the import and delete its local copies of `cross3D`, `cross2D`, `dot`, `subtract`, `scale`, `norm`, `add`, `calculateDisjointEdgePairs`, `calculateEnergy`, `gradientAnalytical`. Keep `calculateGradientFiniteDiff` (the central-diff checker) and `checkGradients`. New import — **no `norm`** (every `norm()` call was inside the deleted local `calculateEnergy`; the survivors don't use it, and biome `noUnusedImports` would flag it):
```ts
import { testConfigs, type GraphState, type Vec3, type Edge } from './src/testConfigs';
import {
    calculateEnergy,
    gradientAnalytical,
    calculateDisjointPairs,
} from './src/tangentPointEnergy';
```

**Retype the kept central-diff checker.** It is currently typed `number[][]` and threads a `dimension`, both of which fail against the module's `Vec3[]`/`Edge[]` signatures (`number[][]` is not assignable to the tuple types). Rewrite its header/body — load-bearing changes: params typed `Vec3[]`/`Edge[]`, `new Array(dimension)`/`d < dimension` → `3`, `[...vtx]` → `[...vtx] as Vec3`, and every `calculateEnergy(...)` reordered to `(vertices, edges, disjointPairs, alpha, beta, epsilon)` with no `dimension`:
```ts
function calculateGradientFiniteDiff(
    vertices: Vec3[], edges: Edge[], alpha: number, beta: number,
    disjointPairs: number[][], epsilon: number, h: number,
): Vec3[] {
    const gradient: Vec3[] = vertices.map(() => [0, 0, 0]);
    for (let v = 0; v < vertices.length; v++) {
        for (let d = 0; d < 3; d++) {
            const plus = vertices.map(vtx => [...vtx] as Vec3);
            const minus = vertices.map(vtx => [...vtx] as Vec3);
            plus[v][d] += h; minus[v][d] -= h;
            const Ep = calculateEnergy(plus, edges, disjointPairs, alpha, beta, epsilon);
            const Em = calculateEnergy(minus, edges, disjointPairs, alpha, beta, epsilon);
            gradient[v][d] = (Ep - Em) / (2 * h);
        }
    }
    return gradient;
}
```

Reconcile the remaining call sites (line numbers approximate — match the actual file):
- Rename `calculateDisjointEdgePairs(...)` → `calculateDisjointPairs(...)` (~lines 390, 431, 471).
- Drop the trailing `dimension` literal from `gradientAnalytical(...)` calls: `..., epsilon, 2)` → `..., epsilon)` (~line 393); `..., epsilon, 3)` → `..., epsilon)` (~lines 434, 474).
- Drop the trailing `dimension` literal from the `calculateGradientFiniteDiff(...)` calls (~lines 392, 433, 473) and match the retyped signature above.

Retype the configs for strict TS and 3D:
- Test 1 square: `const vertices2D: Vec3[] = [[0,0,0],[1,0,0],[1,1,0],[0,1,0]];` and `const edges2D: Edge[] = [[0,1],[1,2],[2,3],[3,0]];` (was 2D `[x,y]`; now `z=0`).
- Tests 2 & 3: annotate `const vertices3D: Vec3[] = [...]` and `const edges3D: Edge[] = [...]` (literals otherwise infer as `number[][]`, not assignable to `Vec3[]`/`Edge[]`).
- Delete any leftover `dimension` locals; the module is 3D-only, so there is no `cross2D`/2D branch anymore (`z=0` covers the old 2D case).

- [ ] **Step 7: Update knip config, then run all gates**

First add `test/**` to `knip.json` `entry` — the test files aren't imported by the app entries, and `test/bench.ts` (added in Task 3) is not a `.test.ts`, so knip would otherwise report them as unused files:
```json
{
  "$schema": "https://unpkg.com/knip@6/schema.json",
  "entry": ["server.ts", "src/index.tsx", "test_gradient.ts", "test/**"],
  "project": ["**/*.{ts,tsx,js}"]
}
```

Run:
```bash
bun test test/golden.test.ts          # expect 3 pass
bun run test_gradient.ts              # expect 3x "✓ PASSED"
bunx tsc --noEmit                     # expect no output (clean)
bun build src/index.tsx --target=browser >/dev/null && echo BUILD_OK
bunx knip                             # expect no unused files/exports
```
Expected: golden 3/3, test_gradient 3/3, tsc clean, `BUILD_OK`, knip clean (deleted `index.tsx`/`test_gradient.ts` copies gone; module exports all used; `test/**` reachable so the fixtures/bench aren't flagged).

- [ ] **Step 8: Commit**

```bash
git add src/tangentPointEnergy.ts test/golden.test.ts knip.json src/index.tsx test_gradient.ts
git commit -m "refactor: extract tangent-point energy/gradient into src/tangentPointEnergy.ts"
```

---

## Task 3: Opt #1 — inline vector helpers to scalar x/y/z (bit-identical)

Inline `subtract`/`scale`/`add`/`cross3D`/`norm`/`dot` inside `calculateEnergy` and `gradientAnalytical` to scalar locals, eliminating the per-call array allocations. **Preserve operation order** so results stay bit-identical.

**Files:** Modify `src/tangentPointEnergy.ts`.

- [ ] **Step 1: Inline the hot loops**

Transform every helper call in `calculateEnergy` and `gradientAnalytical` (and the internal `kernelDerivs`/`safeUnit`) to scalar math. Example — the `calculateEnergy` inner kernel:

Before:
```ts
const d = subtract(vertices[i], vertices[j]);
const d_norm = norm(d) + epsilon;
const c_norm = norm(cross3D(e_I, d)) + epsilon;
sumK += Math.pow(c_norm, alpha) / Math.pow(d_norm, beta);
```
After (same operations, same order, no arrays):
```ts
const dx = vertices[i][0] - vertices[j][0];
const dy = vertices[i][1] - vertices[j][1];
const dz = vertices[i][2] - vertices[j][2];
const d_norm = Math.sqrt(dx * dx + dy * dy + dz * dz) + epsilon;
const cx = eIy * dz - eIz * dy;
const cy = eIz * dx - eIx * dz;
const cz = eIx * dy - eIy * dx;
const c_norm = Math.sqrt(cx * cx + cy * cy + cz * cz) + epsilon;
sumK += Math.pow(c_norm, alpha) / Math.pow(d_norm, beta);
```
Apply the same treatment throughout `gradientAnalytical`/`kernelDerivs`/`safeUnit`, honoring the **invariants** (spec §Invariants): `epsilon` added *after* the norm; zeroing guards on the *pre-epsilon* norm (`r < 1e-14`, `rc >= 1e-14`); `dot`/`cross` summed in x→y→z order; the `/2` factor on both energy and gradient.

- [ ] **Step 2: Run gates — still bit-identical (STRICT stays true)**

Run:
```bash
bun test test/golden.test.ts    # expect 3 pass (bit-identical)
bun run test_gradient.ts        # expect 3x PASSED
bunx tsc --noEmit               # clean
bun build src/index.tsx --target=browser >/dev/null && echo BUILD_OK
```
If any golden test fails, an inlining changed the math or operation order — diff against the invariants and fix before proceeding.

- [ ] **Step 3: Add the benchmark harness**

Create `test/bench.ts`:
```ts
import { calculateEnergy, calculateDisjointPairs } from '../src/tangentPointEnergy';
import type { Vec3, Edge } from '../src/testConfigs';

// Fixed pseudo-trefoil-ish chain, N points, deterministic (index-based, no Math.random).
function makeChain(n: number): { vertices: Vec3[]; edges: Edge[] } {
    const vertices: Vec3[] = [];
    const edges: Edge[] = [];
    for (let i = 0; i < n; i++) {
        const t = (i / n) * Math.PI * 4;
        vertices.push([Math.cos(t), Math.sin(t), Math.sin(2 * t) * 0.5]);
        if (i > 0) edges.push([i - 1, i]);
    }
    return { vertices, edges };
}

for (const n of [50, 128, 256]) {
    const { vertices, edges } = makeChain(n);
    const dis = calculateDisjointPairs(edges);
    const iters = 20;
    const t0 = performance.now();
    for (let k = 0; k < iters; k++) calculateEnergy(vertices, edges, dis, 3, 6, 1e-10);
    const ms = (performance.now() - t0) / iters;
    console.log(`N=${n}: ${ms.toFixed(3)} ms/energy`);
}
```

- [ ] **Step 4: Benchmark**

Run: `bun run test/bench.ts` and record the ms/energy figures for the commit message. (Also commit `test/bench.ts` — it's added to the knip `entry` in Task 2 Step 7, so it won't be flagged unused.)

- [ ] **Step 5: Commit**

```bash
git add src/tangentPointEnergy.ts test/bench.ts
git commit -m "perf: inline vector helpers to scalar math in the energy/gradient hot loop"
```

---

## Task 4: Opt #2 — flat Float64Array internals (bit-identical)

Inside `calculateEnergy` and `gradientAnalytical`, convert the `Vec3[]` input to a flat `Float64Array` (stride 3) once on entry; index it as `V[3*i + d]`. `gradientAnalytical` accumulates into a flat gradient buffer and un-flattens to `Vec3[]` on return. Interface (`Vec3[]` in / out) is unchanged.

**Files:** Modify `src/tangentPointEnergy.ts`.

- [ ] **Step 1: Flatten inputs at the top of each compute function**

Example for `calculateEnergy`:
```ts
const n = vertices.length;
const V = new Float64Array(n * 3);
for (let i = 0; i < n; i++) { V[3*i] = vertices[i][0]; V[3*i+1] = vertices[i][1]; V[3*i+2] = vertices[i][2]; }
```
Replace `vertices[i][0]` reads with `V[3*i]`, etc. For `gradientAnalytical`, accumulate into `const G = new Float64Array(n * 3)` and un-flatten before returning:
```ts
const gradient: Vec3[] = [];
for (let i = 0; i < n; i++) gradient.push([G[3*i], G[3*i+1], G[3*i+2]]);
return gradient;
```
Values are identical IEEE-754 doubles, and operation order is unchanged → results stay bit-identical.

- [ ] **Step 2: Run gates — still bit-identical**

Run:
```bash
bun test test/golden.test.ts    # expect 3 pass (bit-identical)
bun run test_gradient.ts        # expect 3x PASSED
bunx tsc --noEmit               # clean
bun build src/index.tsx --target=browser >/dev/null && echo BUILD_OK
bun run test/bench.ts           # record new figures
```

- [ ] **Step 3: Commit**

```bash
git add src/tangentPointEnergy.ts
git commit -m "perf: use flat Float64Array buffers inside energy/gradient compute"
```

---

## Task 5: Opt #3 — integer-pow fast path (tolerance, not bit-identical)

Replace `Math.pow` with an integer-exponent fast path, per exponent, and cut `pow` count by reuse. This perturbs low bits, so the golden gate switches to combined abs+rel tolerance.

**Files:** Modify `src/tangentPointEnergy.ts`, `test/golden.test.ts`.

- [ ] **Step 1: Add an integer-power helper and apply it per exponent**

In `src/tangentPointEnergy.ts`:
```ts
// Fast path for small non-negative integer exponents; falls back to Math.pow otherwise.
function ipow(base: number, exp: number): number {
    if (Number.isInteger(exp) && exp >= 0 && exp <= 64) {
        let r = 1;
        for (let k = 0; k < exp; k++) r *= base;
        return r;
    }
    return Math.pow(base, exp);
}
```
Use `ipow` for each exponent independently. The app default α=3/β=6 are integers (fast); Test 3's β=4.5 is fractional (falls back), α=2 is integer (fast). Where the code already has `Math.pow(c_eps, alpha)` and separately `Math.pow(c_eps, alpha - 1)`, reuse `cPowAm1 * c_eps === c_eps^alpha` to save a `pow`, and `dPowB * d_eps === d_eps^(beta+1)`. Keep the algebra identical to the current expressions — only the evaluation of the powers changes.

- [ ] **Step 2: Switch the golden test to tolerance mode**

In `test/golden.test.ts`, change `const STRICT = true;` → `const STRICT = false;`.

- [ ] **Step 3: Run gates — tolerance**

Run:
```bash
bun test test/golden.test.ts    # expect 3 pass (within atol=1e-6, rtol=1e-5)
bun run test_gradient.ts        # expect 3x PASSED
bunx tsc --noEmit               # clean
bun build src/index.tsx --target=browser >/dev/null && echo BUILD_OK
bun run test/bench.ts           # record final figures
```
Expected: golden 3/3 within tolerance (including `planar_a2_b45`, which exercises the fractional fallback); test_gradient 3/3; measurable speedup vs Task 2.

- [ ] **Step 4: Commit**

```bash
git add src/tangentPointEnergy.ts test/golden.test.ts
git commit -m "perf: integer-exponent Math.pow fast path in the kernel"
```

---

## Task 6: Add anchoring comments to critical / "magic" sections

This repo is heavily vibe-coded: several lines look removable but are load-bearing. Comment them with a reference to the spec/issue so a future agent (or human) doesn't "clean them up" and regress. Add these comments (adjust to the final variable names; keep the claim, verify it holds).

**Files:** Modify `src/tangentPointEnergy.ts`, `test_gradient.ts`.

- [ ] **Step 1: Module header (`src/tangentPointEnergy.ts`, top of file)**
```ts
/**
 * Tangent-point (repulsive-curves) energy and its analytical gradient.
 *
 * SINGLE SOURCE OF TRUTH — imported by src/index.tsx (the app) and test_gradient.ts
 * (which verifies gradientAnalytical against central finite differences).
 * Design/rationale: docs/superpowers/specs/2026-07-01-tangent-point-hotpath-optimization-design.md
 *
 * 3D ONLY BY DESIGN. 2D configs embed as z=0: ‖cross3D‖ with z=0 equals |cross2D|, and the
 * gradient's z-components are identically 0. Do NOT re-introduce a 2D branch.
 * Deferred optimizations (flat Float64Array through the UI, Barnes–Hut, WASM/WebGPU): issue #1.
 */
```

- [ ] **Step 2: The epsilon regularization (at `c_eps` / `d_eps`)**
```ts
// ε is added AFTER the norm: kernel = (‖e×d‖ + ε)^α / (‖d‖ + ε)^β. This is part of the energy
// DEFINITION (regularization), not a guard — moving ε inside/outside the norm changes the energy.
```

- [ ] **Step 3: The pre-epsilon direction guards (in `safeUnit` and the cross-norm guard)**
```ts
// Guard tests the PRE-ε length (r / rc, not r+ε): a ~0-length vector has no defined unit
// direction, so we zero that derivative. Intentional — using the +ε value here would be wrong.
```

- [ ] **Step 4: The /2 symmetry factor (energy return + gradient's final scaling loop)**
```ts
// /2 because disjointPairs lists BOTH (I,J) and (J,I) — every unordered pair is summed twice.
// The gradient divides by 2 for the SAME reason. Keep energy and gradient in lockstep, or the
// analytical gradient stops matching the energy (test_gradient.ts fails).
```

- [ ] **Step 5: The integer-pow fast path (at `ipow`)**
```ts
// Per-exponent integer fast path (perf — issue #1 / spec opt #3). MUST stay per-exponent: the app
// uses α=3,β=6 (integer) but test_gradient uses α=2,β=4.5, so the β side falls back to Math.pow.
// Results are NOT bit-identical to Math.pow — the golden test uses a combined abs+rel tolerance
// (atol=1e-6, rtol=1e-5), not equality. Do not assume x*x*x === Math.pow(x,3).
```

- [ ] **Step 6: The inlined-scalar op order (near the inlined kernel)**
```ts
// Vector math is inlined to x/y/z scalars to avoid per-iteration array allocation (most of the
// measured speedup). Keep the x→y→z accumulation order — it's what keeps opts #1/#2 bit-identical.
```

- [ ] **Step 7: `test_gradient.ts` — central-difference rationale (at `calculateGradientFiniteDiff`)**
```ts
// CENTRAL difference (E(+h)−E(−h))/2h, NOT forward. Forward diff fabricates an O(h) "gradient"
// where the true derivative is 0 (out-of-plane z of a planar config) and false-failed a correct
// gradient. See spec §Verification and the "test: rewrite gradient verification" commit.
```

- [ ] **Step 8: `test_gradient.ts` — the tolerance in `checkGradients`**
```ts
// Combined abs+rel tolerance |a−b| ≤ atol + rtol·max(|a|,|b|), NOT pure relative error.
// Components whose true value is ≈0 make |Δ|/|value| blow up, so pure relative error false-fails.
// Intentional — do not "simplify" to relative-only.
```

- [ ] **Step 9: Re-verify each comment against the code, then commit**

Read each comment against the line it annotates — every claim must be true. Comments are the only change, so behavior is unchanged:
```bash
bun run test_gradient.ts && bunx tsc --noEmit    # still 3/3, clean
git add src/tangentPointEnergy.ts test_gradient.ts
git commit -m "docs: anchor critical invariants with comments referencing spec/issue #1"
```

---

## Task 7: Final verification

- [ ] **Step 1: Full gate sweep**

Run:
```bash
bun test test/golden.test.ts && bun run test_gradient.ts && bunx tsc --noEmit \
  && bun build src/index.tsx --target=browser >/dev/null && echo BUILD_OK \
  && bunx knip
```
Expected: golden 3/3, test_gradient 3/3, tsc clean, `BUILD_OK`, knip reports no unused files/exports.

- [ ] **Step 2: Confirm the viewer runs**

Run: `PORT=3001 timeout 4 bun run dev` and confirm it logs `Server running at http://localhost:3001` with no build error. (Manual visual check of the canvas is optional.)

- [ ] **Step 3: Record the speedup**

Note the Task 2 (pre-opt, if benchmarked) vs Task 5 ms/energy in the final commit or PR description.

---

## Self-Review

**Spec coverage:**
- Consolidation into one module → Task 2. Verbatim move + verification → Task 2 Steps 1–2. Signature reconciliation & TS retyping → Task 2 Step 6. ✓
- Golden guard from pre-change code, frozen inputs, fractional case, combined tolerance → Task 1 + Task 5 Step 2. ✓
- Opts #1/#2 bit-identical, #3 tolerance → Tasks 3/4 (STRICT true) vs Task 5 (STRICT false). ✓
- Invariants preserved → referenced in Task 3 Step 1. ✓
- `norm` exported for the viewer → Task 2 Steps 1 & 5. ✓
- Anchoring comments on load-bearing invariants (ε placement, pre-ε guards, /2 symmetry, per-exponent pow, op order, central-diff, tolerance) → Task 6 (each claim re-verified in Step 9). ✓
- Success criteria (knip clean, build, tsc, test 3/3, speedup) → Task 7. ✓

**Placeholders:** the `disjoint == null ? [] : disjoint` line in Task 1 Step 2 is explicitly flagged and corrected in the same step. No other TBD/TODO.

**Type consistency:** module signatures `(vertices, edges, disjointPairs, alpha, beta, epsilon)` used identically in the capture script (Task 1, corrected), golden test (Task 2), and test_gradient rewire (Task 2 Step 6). `Vec3`/`Edge` imported from `./testConfigs` everywhere.

---

## Execution Handoff

See the header sub-skill note. Recommended: superpowers:subagent-driven-development (fresh subagent per task, review between tasks).
