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

Benchmark (`bench/results/2026-11-24-medial-graph-layout.jsonl`; a jittered grid whose middle
third is squeezed to a fifth, 8 segments per edge; crossings before → after, clearance =
smallest distance between non-incident edges):

| Segments | Medial-sphere graph flow | Tangent-point flow + crossing guard |
|---|---|---|
| 896 | 17 ms/step; converged in 196 steps (3.4 s); crossings 0 → 0; clearance 0.17 → 0.65 | 759 ms/step; 10 steps (7.6 s); 0 → 0; 0.17 → 0.64 |
| 4,896 | 201 ms/step; NOT converged in 300 steps; 0 → 0; clearance 0.14 → 0.11 | first step rejected (this flow does not run a graph of 4,608 points) |
| 20,160 | 1,173 ms/step; NOT converged; 0 → 0; clearance → under 1e-4 | not attempted (minutes per step) |

**Verdict: PARTIAL.** Against the bar ("≥ 10× faster per step at 5k+ segments and keeps a clean
drawing clean"): it is 44× faster per step where both run, it is the only one of the two that
runs at 5k and 20k, and it never created a crossing. But at 5k and 20k it does not converge and
lets strands be pressed to within rounding of each other, so it is not a usable layout engine
at the target size yet.

What is known about the failure at scale:
- The node movement per step stays near 4e-2·r instead of decaying (chatter, as in Spike A).
- A contact barrier on the push (`barrier` = 1 or 2: the deficit times `(rMax/R)^barrier`) does
  NOT fix it — tried on the 4,896-segment grid for 300 steps; clearance ends at 0.12 and
  under 1e-4. The option is left in, off by default.
- The drawing opens to height 13.8 of a rest height of 17 and stops: whole-drawing motion is
  slow. The mass term screens the Laplacian beyond about one grid cell, and lowering it makes
  the Jacobi-preconditioned CG need 170–200 iterations.
- Time per step grows faster than the size (60 µs per node-step against 16 in Spike A): the
  per-node limits look at every pair within the full range.

Next for the graph flow, in order: (1) find and damp the chatter (it is shared with Spike A, so
it is worth doing once, properly: an energy that is consistent with the frozen-centre gradient,
or a line search on it); (2) a real sparse solver so the mass term can go to ~0; (3) per-node
limits that only look as far as the node wants to move.

## Update (2026-11-24, later): what made both flows settle, and what still fails

The tables above are the FIRST version's. Everything below supersedes them; the result files
were regenerated on a quiet machine.

### The knot flow

| Finding | Evidence | Change |
|---|---|---|
| The paper's step length (10–20) is unstable here | at 10 the collision bound cut 91–95% of steps short and the gradient per node chattered at 0.1–0.4; the stopping rule never fired | step length 1 (`P`'s Laplacian is the length term's Hessian, so a plain step is stable only below 2) |
| The paper's λ and its stopping threshold are not scale-free | the same trefoil stopped on the gradient rule after 107 steps in dataset units (r ≈ 1.2) and never when scaled to extent 2 (r ≈ 0.2) | mass weight `λ/r²`; gradient measure = pointwise gradient × r; identical step counts at both scales (123 / 123, 214 / 214) |
| Too little damping drives strands into contact | with `λ/r²`, λ = 2: trefoil 47 (largest r) stuck with a zero safe step from step ~100 | λ = 4; a run of 5 zero steps stops the pass as `stuck` |
| The audit can be fooled by strands in contact | on that stuck curve one projection of five gave determinant 1, the other four 3 | a projection whose crossing has no clear over/under is refused (as one with a crossing at a vertex is) |

Dataset, final (`bench/results/2026-11-24-medial-sphere-knots.jsonl`):

| Dataset | Knot type kept | Crossings: start → end (prototype) | Time, mean (max) | Steps, mean | Stopped on the gradient rule |
|---|---|---|---|---|---|
| Trefoil100 | 100 / 100 | 366 → 3.0 (3) | 1.21 s (2.60 s) | 164 | 100 |
| Knot128 | 128 / 128 | 276 → 10.7 (8.9) | 1.56 s (3.46 s) | 297 | 86 (41 plateau, 1 collapsed) |

The collision bound now limits 1% of steps and refuses none. Not explained: what makes a step
of 10–20 stable for the authors (their energy carries a factor 2 ours does not, which goes
the wrong way).

### The hand-off to the tangent-point flow (`bench/results/2026-11-24-untangle-handoff.jsonl`)

| Knot | Points | Pass | Energy at hand-off | Step after | Raw: energy start → after 10 steps | Raw step |
|---|---|---|---|---|---|---|
| trefoil0001 | 2,808 | 2.4 s | 17.8 | 1.3 s | 44,222 → 2,094 | 11.9 s |
| knot0001 (unknot) | 1,998 | 1.1 s | 0.34, converged at once | — | 20,765 → 900 | 5.5 s |
| knot0002 | 1,611 | 1.0 s | 16.5 | 1.0 s | 11,377 → 416 | 3.2 s |
| knot0064 | 2,298 | 2.3 s | 65.6 | 1.7 s | 20,993 → 1,211 | 8.0 s |

### The graph flow

| Failure seen | Cause | Change |
|---|---|---|
| Move per step stays at 3.6e-2·r for 300 steps at 4,896 segments | the mass term was 10× too soft exactly at the nodes in contact, which overshoot every step | diagonal = (floor + contact stiffness per node)·α·w |
| Strands pressed together until a rounding error apart | the paper's pull saturates; a spring under 5× compression out-pushes it | contact barrier (deficit × rMax/R), with its stiffness in the diagonal |
| "Crossing" between two edges of one junction, at distance 1e-14 | they had no interaction at all and folded onto each other | minimum-angle rule (default 30°): they push apart only below it |
| "Crossing" of an edge with itself, at 5e-18 | an edge's own points had no interaction | fold rule: a circle tighter than a quarter of the arc between two points of one edge is a fold |
| Tried, worse | contact stiffness as links instead of a diagonal: strands ended at zero clearance | — |

Benchmark, final (`bench/results/2026-11-24-medial-graph-layout.jsonl`):

| Segments | Medial-sphere graph flow | Tangent-point flow + crossing guard |
|---|---|---|
| 896 | 23 ms/step; converged in 260 steps (6.0 s); crossings 0 → 0; clearance 0.17 → 0.67 | 879 ms/step; 10 steps (8.8 s); 0 → 0; 0.17 → 0.64 |
| 4,896 | 173 ms/step; 600 steps, not converged (move 6.7e-4·r); 0 → 0; clearance 0.14 → 0.19; the band opens but stays wavy | first step rejected |
| 20,160 | 844 ms/step; 356 steps in the 300 s budget; **2 crossings reported**, clearance 0.03 | not attempted |

**Verdict: still PARTIAL.** 38× faster per step where both run; stable and crossing-free up
to 4,896 segments; not converged there, and it fails at 20,160.

Why it fails at 20,160, and the next step: the probes test a node against other NODES. Two
strands closer than about one segment length cannot see each other (the nearest node is to
the side, so the circle through it is large), so neither the barrier nor the fold rule acts,
and the motion limits then let the gap shrink by 90% per step down to rounding. The paper
avoids this by keeping the spacing h under r/5 everywhere, which a compressed graph edge
violates. **Next: probe a node against SEGMENTS** (the largest circle that stays clear of a
segment has a closed form). That removes the blind spot, and it removes the need for 8 points
per edge — which is where the cost at 20k has to come from.

## Update 2 (2026-11-24): segment probes and bending springs — the graph flow holds at 20k

Two more causes found on the graph flow, both of the kind "the probes cannot see it":

| Failure seen | Cause | Change |
|---|---|---|
| Strands a rounding error apart at 20,160 segments, unseen by any force | probes tested a node against other NODES; a strand closer than about one segment length has its nearest node off to the side, so the circle through it is large | probes test a node against whole SEGMENTS: `min_s |P(s)|² / (2 x·P(s))` in closed form (two endpoints and the roots of a quadratic) |
| A junction 1e-15 from the second segment of its own edge, while unrelated edges kept 0.33 | a chain under compression folds up along its own line (an accordion); probes look sideways and see nothing approach along the tangent | bending springs: a spring across every two consecutive segments of an edge, rest length "straight", as a link in P |

Benchmark (`bench/results/2026-11-24-medial-graph-layout.jsonl`, quiet machine):

| Segments | Medial-sphere graph flow | Tangent-point flow + crossing guard |
|---|---|---|
| 896 | 28 ms/step; converged in 334 steps (9.5 s); crossings 0 → 0; clearance 0.17 → 0.70 | 788 ms/step; 10 steps (7.9 s); 0 → 0; 0.17 → 0.64 |
| 4,896 | 197 ms/step; 600 steps (118 s), move down to 1e-3·r; 0 → 0; clearance 0.14 → 0.68; nothing at its motion limit after ~300 steps | first step rejected |
| 20,160 | 1,217 ms/step; 247 steps in the 300 s budget; 0 → 0; clearance 0.146 → 0.173, still opening | not attempted |

An all-pairs crossing count every 5 steps (same-edge pairs included) found none in 400 steps
at 4,896 segments and none in 300 steps at 20,160.

**Verdict: the topology holds at every size and the layout is right up to ~5k segments. At 20k
it is correct but slow** — it needs on the order of a thousand steps at 1.2 s each.

What the cost at 20k is made of, and the levers (not yet pulled):
1. **Points per edge.** With segment probes the 8 points per edge are no longer needed for the
   probes to see anything; they are only the edges' ability to bend. 2–3 would cut the size 3×.
2. **The solve.** Jacobi-preconditioned CG takes 140–160 iterations per coordinate; a sparse
   Cholesky or a multilevel preconditioner on this graph Laplacian would take a handful.
3. **Whole-drawing motion.** Opening a band means moving half the drawing; with a better solve
   the mass floor can drop and that mode gets faster.

### Lever 1 measured: points per edge (same day)

With segment probes, fewer points per edge are enough (rows with `pieces` 3 and 2 in the
results file; the rows without that field are 8 per edge):

| Grid | Points per edge | Segments | Time per step | Clearance | Crossings |
|---|---|---|---|---|---|
| 18×18 | 8 | 4,896 | 183 ms | 0.14 → 0.68 | 0 |
| 18×18 | 3 | 1,836 | 45 ms | 0.14 → 0.67 | 0 |
| 18×18 | 2 | 1,224 | 21 ms | 0.14 → 0.65 | 0 |
| 36×36 | 8 | 20,160 | 1,300 ms | 0.146 → 0.157, not opened in 240 s | 0 |
| 36×36 | 3 | 7,560 | 314 ms | 0.146 → 0.137, not opened in 240 s | 0 |
| 36×36 | 2 | 5,040 | 106 ms | 0.146 → 0.65, opened (1,500 steps, 158 s) | 0 |

The cost per step is about 21 µs per segment, and CG needs 40 iterations at 2 per edge
against 157 at 8. A graph is therefore best run at 2 segments per edge; the earlier "20k
segments" case was the 1,296-node grid at 8 per edge. What remains slow is the NUMBER of steps
(whole-drawing motion), which is the solver lever, not this one.

## The graph flow in the app, and the drag (2026-11-24)

- Store switch `graphFlow` ("Graph flow" in the panel) makes the local graph flow the layout
  engine for a FLAT graph: the frame loop ticks it instead of a tangent-point step. Rest
  lengths are the geometry at the first tick of a run; the clearance is a fraction of the mean
  edge length. A graph that is not flat falls back to the ordinary descent and says so.
- New preset `flat-grid` (a jittered planar grid), the only flat preset.
- **The drag is a bounded spring, not a forced move.** A pin whose target is elsewhere pulls
  its node with `pinStiffness·distance/r`, capped at `pinForceMax`; the node otherwise takes
  part in the flow, so the contact barrier acts on it. The first version marched a pinned node
  towards its target at the speed its motion limit allowed: a 2×2 blob dragged into a 12×12
  grid stopped after a third of the way, in contact, for good. With the spring it arrives
  (0.2 from its target after 900 steps at 11 ms each), with 0 crossings at every step.

**What the drag can and cannot do — a limit of the idea, not of the code.** The flow never
lets an edge pass through another, so the drawing's embedding is fixed: which face every node
sits in cannot change. Dragging a blob therefore drags the map along and deforms it; a blob
cannot be carried ACROSS the map into another region, because its own edges would have to
cross the edges in between. The benchmark picture shows exactly that: the corner block pulled
towards the centre shears the whole grid into a diamond.

So the mind-map move ("this blob does not belong here, put it over there and make room")
needs one more piece, and it is a choice of semantics:
1. *Ghost drag:* the dragged group's own edges are exempt from the collision limits while
   it is dragged (they pass over the drawing); on drop, the flow runs with whatever crossings
   the new position has. Cheap; leaves crossings where the graph's connections make them
   unavoidable — which is honest, since moving a connected blob far away DOES stretch its
   links across what lies between.
2. *Cut and re-insert:* detach the group, route its connecting edges around obstacles
   (an edge-routing problem), re-attach. Planar when the graph allows it; much more work.
Recommended: 1 first (it is what "move it there" means in every note-taking tool), then
look at the crossings it leaves and decide whether 2 is worth building.
