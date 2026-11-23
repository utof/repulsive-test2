# Plan: the medial-sphere method as a second untangling core

Date 2026-11-24. Status: plan, before any code. Source: Noma, Jacobson, Singh, "Medial Sphere
Preconditioning for Knot Untangling and Volume-Filling Curves", SIGGRAPH Asia 2025, and its 2-D
predecessor (Noma et al., TOG 2024). Local copies: `local_files/papers/` (see its README).

**Licence.** The authors' code is GPL-3. It is saved locally for reference and is NOT to be
ported: everything here is implemented from the paper's formulas.

## The method in our words

Energy `E = length + α · E_M`. From every node, in each of N = 16 directions `x` of its normal
plane, grow the largest sphere that touches the node and contains no other node, up to radius
`r_max`; `E_M` is the sum of squared distances from the node to those sphere centres. With the
centres frozen, the gradient at node `i` is

    g_i = −κ_i n_i  +  α (2π/N) Σ_l (γ_i − m_il),        m_il = γ_i + R_il x_l

so each direction pulls the node towards the centre of its empty sphere, harder where there
is more room. That equalises the spacing between strands; the length term shortens the curve.
Everything is LOCAL: no node further than `2 r_max` matters.

The sphere radius has a closed form (the paper's `R_il`): over the other nodes `j` with
`x_l · (γ_j − γ_i) > 0`,  `R_il = min( r_max, min_j |γ_j − γ_i|² / (2 x_l · (γ_j − γ_i)) )`.
One neighbour query per node (a uniform grid of cell `2 r_max`), no binary search.

Step: `γ ← γ − τ P⁻¹ g` with `P = L* + λ M`, where `L*` is the curve Laplacian plus weak links
(β = 0.01) between nodes that share a sphere. Fixed large τ, backtracked by a line search whose
only test is continuous collision detection. The curve is re-sampled to equal spacing
`h < r/5` every step. Untangling mode: `r` = current average sphere size, `r_max = 0.9 r`,
λ = 1 — this tightens the curve, so it SHRINKS at equal spacing and the node count falls as it
goes; stop when the gradient per node is below `5e-3 r` or 10 nodes are left. From their code's
parameters: thickness `r ≈ 3.36 / √α`.

Their numbers (M1 MacBook Pro, C++): Knot128 dataset 100% untangled, 8.7 s and 720 iterations
on average; Trefoil100 100%, 9.1 s. Ours on Knot128 #1 (JS, one thread, this laptop): 223 s
for 120 tangent-point steps.

## What we have, and what is missing

| Piece | State |
|---|---|
| Neighbour queries | a uniform grid is ~40 lines; nothing reusable is needed |
| `P⁻¹ g` | no sparse Cholesky in this repo. Conjugate gradients, preconditioned by the cyclic tridiagonal Laplacian solve we already have (`laplacian.ts`); the β-links are a small perturbation of it |
| Collision test | 2-D done (`planarCollision.ts`, PR #112). 3-D edge–edge (a cubic per pair) to write |
| Re-sampling | to write; trivial for a closed curve |
| Selectable in the app | a third descent mode next to raw and Sobolev |

## What the paper does not cover (our risks)

1. **Junctions.** A node of degree ≠ 2 has no normal plane. Untested by anyone.
2. **Pins and hard constraints.** None in the paper. Pins look easy (drop the node from the
   solve); fixed length contradicts the shrinking and would need a different mode.
3. **No shrinking for layouts.** Untangling mode shrinks the curve to nothing. A graph layout
   must instead hold a clearance `r` at fixed size — the volume-filling mode, which needs a
   boundary or a length budget to stop it growing.
4. **No long-range force.** Two far-apart tangles do not know of each other except through
   the solve. Fine for knots (their results), unknown for graphs.

## Steps and decision points

1. **Spike A — closed curves in 3-D, the paper's own setting.** Energy, closed-form radius,
   grid, PCG, re-sampling, 3-D collision test, the untangling loop. Benchmark on the Knot128
   and Trefoil100 curves we already have and on our smooth knots: time and iterations to
   their stopping rule, knot type kept (collision test never bypassed), against their table
   and against our tangent-point runs. **Go on if** it untangles the dataset knots and is at
   least 5× faster than our tangent-point flow on the same curve and machine.
2. **Spike B — flat graphs.** Junction rule, pins, a non-shrinking mode. Benchmark at 1k, 5k
   and 20k segments against the tangent-point flow: time per step, crossings held, minimum
   clearance. **Go on if** it is at least 10× faster per step at 5k+ segments and keeps a
   clean drawing clean.
3. If both pass: make it selectable in the app, then build the drag interaction on it.
   If A passes and B fails: keep it as the knot untangler; graphs get the GPU all-pairs
   gradient for the tangent-point energy instead.
4. Independent of the above, smaller ideas from the same papers, each measured on its own:
   near-contact links in OUR preconditioner (for tight tangles above 4 000 points); a
   collision-bounded first trial step instead of restarting at 1.

## Results (2026-11-24)

### Spike A — closed curves in 3-D: PASSED

`src/core/medialSphere.ts`, benchmark `bench/app/medialSphereKnots.ts`, every dataset knot
(`bench/results/2026-11-24-medial-sphere-knots.jsonl`; JS, one thread, this laptop).

| Dataset | Curves | Knot type kept | Crossings: start → end (tidy prototype) | Time, mean (max) | Steps, mean |
|---|---|---|---|---|---|
| Trefoil100 | 100 | 100 | 366 → 3.0 (3) | 1.10 s (2.38 s) | 207 |
| Knot128 (perturbed) | 128 | 128 | 276 → 11.1 (8.9) | 1.31 s (2.53 s) | 229 |

- "Knot type kept" is the knot determinant modulo two primes, audited every 25 steps
  (`src/core/knotDeterminant.ts`) — an invariant computed from a projection, sharing no code
  with the collision test. It is live: with the collision bound loosened 6× it trips within
  4 steps (3 → 1, 21 → 1). It cannot tell apart two knots with the same determinant.
- "Crossings" is the fewest seen in 12 fixed projections, an upper bound on the crossing number.
- The paper reports 8.7 s and 9.1 s per knot on an M1 MacBook; the two are not comparable
  (different stopping rule — see below). Against OUR tangent-point flow the bar was 5×: that
  flow needed 223 s for 120 steps on one Knot128 curve and was not done.

Deviations from the paper that the numbers depend on:
1. **Stopping rule.** The paper stops at a gradient per node below `5e-3·r`. Ours never gets
   there: once the knot is tight the gradient chatters at 0.1–0.4 (the pull switches between
   "circle capped" and "circle hit" from step to step). The benchmark stops on a plateau of
   the curve length (1% over 100 steps). Open: find what damps this in the reference.
2. **Collision test.** A conservative bound (`maxSafeStep`), not swept-surface CCD. It cut the
   trial step short in 91–95% of steps, so it costs step length; the totals above include that.
3. **Re-sampling** drops a node only when no other edge pierces its corner triangle. The first
   version used a distance heuristic.
4. **Solve.** CG preconditioned by the cyclic tridiagonal part: at most 8 iterations.

What it is not: the result is a TIGHT knot (the curve shrinks to rope-length scale and is
re-sampled to ~60–180 points), not the smooth, evenly spread embedding the tangent-point
energy gives. As an untangler it is two orders of magnitude faster; as a "make this curve
beautiful at fixed length" flow it is a different thing.

### Spike B — flat graphs with junctions

`src/core/medialSphereGraph.ts` (an experiment — the paper has no junctions), benchmark
`bench/app/medialGraphLayout.ts`. What had to change from the closed-curve method, each found
by a failure:

| Failure seen | Cause | Change |
|---|---|---|
| Junctions and leaves drift across empty space | sector-bisector probes do not sum to zero when all are capped | the pull is written as a deficit `Σ (R − rMax)·x` (identical to the paper for ±normal) |
| The whole graph freezes (step 0.4 → 1e-14 in 50 steps) | one global safe step is throttled by the single tightest pair | per-node motion limits (`nodeMotionCaps`) |
| Every edge wiggles, the drawing swells to 1.5–2.4× its size | edges meeting at a junction are always within range, so they repel forever (the space-filling behaviour) | edges that share a junction do not interact |
| A squeezed-then-released edge stays a zigzag | springs do not resist bending | a small constant tension (the paper's length term at 0.5) |
| The drawing opens very slowly | mass weight 1 turns whole-drawing modes into plain gradient descent | mass weight 0.1 |
