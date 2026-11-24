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

## The group move, and real graphs (2026-11-24, evening)

Code: `src/core/medialGraphMove.ts`; step changes in `src/core/medialSphereGraph.ts`;
benchmarks `bench/app/medialGroupMove.ts` and `bench/app/pkmGraphs.ts`.

### The move in four phases

1. **Pick up.** The group's nodes and every edge touching them become *carried*: the step
   neither moves them nor lets anything see them (`MedialGraph.ghost`).
2. **Carry.** The group translates rigidly with the pointer; its links to the rest are drawn
   straight.
3. **Set down.** The group is shrunk about one anchor until it fits inside the face it was
   dropped in (full size if it fits, else halved until it does). The links stay carried.
4. **Grow back.** Before every flow step the group is scaled up rigidly as far as the gap to
   its surroundings allows (`growGroup`: a point may move at most `0.45·gap·min(1, gap/0.5r)`,
   the scale at most 25 % per step), and the flow makes room — or, where that is too slow,
   the rest is moved out of the way directly (see "Making room directly" below). At full
   size the group is released, its links are laid straight and the crossings that exist at
   that moment become *allowed*.

What was tried and dropped, with the reason measured:
- Inflating the group with springs from 1/8 size: the edges buckle.
- Landing the links at set-down: a tight fan of links eats the room the group needs to grow
  into (0.09 → 0.02 of a clearance on the grid) and growth never starts.

### Allowed crossings are counted, not exempted

`MedialGraph.passPairs` maps a pair of original edges to the number of times they may cross.
Such a pair is left out of the motion limits, and the interior points of the two edges do
not repel each other. After every step the crossings of each pair are counted; if a pair has
more than its allowance, both edges' moves of that step are undone. The allowance only ever
goes down, and a pair that no longer crosses and has moved apart is dropped.

Two simpler rules were measured first and both fail:
- *Exempt the pair of edges altogether:* the two edges weave through each other, 3 → 7
  crossings on the tree fixture.
- *Exempt the pair of segments that cross:* the crossing cannot slide past a node, so it
  jams against one and the gap goes to 1e-16.

### Three changes to the step itself

- **Reaction force.** A probe circle cut by a segment used to push only the probing node.
  The segment now gets the opposite push, shared between its two ends — at every contact
  between two UNRELATED edges, where the probing node is held (pinned: it cannot give way),
  and where a long segment is involved. Without it: a long link lying across a group's side
  closes on it geometrically (halving every step, never touching, never leaving); an edge
  pressed against a pinned group's corner stays at the gap floor (clearance 0.008 after a
  move, 0.17 with it); a real 88-node map settles with two strands 0.007 apart (0.34 with
  it). NOT applied to the two rules about an edge's own shape (the fold rule, the minimum
  angle at a junction), which are one-sided by design: reacting to those as well
  (`reaction: 'all'`) cost half the opening speed of a finely divided grid (18×18 at 8
  points per edge, 600 steps: clearance 0.28 against 0.68). That regression was caught by
  re-running the layout benchmark before merging; with the final rule the benchmark is as
  good or better everywhere (8×8 converges in 285 steps instead of 334).
- **Long segments** no longer set the reach of the neighbour search for everything: first
  by testing them against everything directly, in the end by cutting them into pieces for
  the search (see "Where a step's time goes").
- **A floor under every gap.** Two segments already nearer than 2 % of the clearance, and
  nearer than 2 % of the shorter of the two, may not get nearer; the step holds their nodes.
  Found on real data: see below. The second condition was added after the first version
  jammed the 18×18 grid at 8 points per edge (clearance stuck at 0.20 where it reaches 0.68):
  on a finely divided drawing two edges of one junction lie a fraction of a short segment
  apart by construction, and that is not a dangerous gap.

### Real mind maps and vaults

Data: public repositories of FreeMind maps and Obsidian vaults, link structure only, kept in
`local_files/pkm-graphs/` (git-ignored, no licence on any of them; see its README).
`scripts/pkmExtract.ts` turns a vault or a `.mm` file into a node/edge list;
`bench/app/pkmGraphs.ts` runs the flow and a group move on each, drawn twice: as mind-map
programs draw it (a tidy tree) and as a graph view would (a force layout).

What real maps have that the synthetic ones did not:
- **Hubs with 70–203 children.** A fan of 200 nearly parallel edges starts with gaps of a few
  thousandths of a clearance. On the 1,237-node map one such gap went 4e-3 → 6e-15 in 60
  steps and the crossing counter reported 44–208 false crossings. The gap floor fixes it: 0
  crossings, smallest gap unchanged at 3.2e-3.
- **Very long edges.** A hub's links to its far children are a hundred times the typical
  length. They made the neighbour search 10× slower per segment (fixed, see below) and,
  having only 2 points, they cannot bend out of a growing group's way (open, see the end).
- **A vault's link graph is nowhere near planar.** 180 notes with 883 links, force layout:
  14,803 crossings. 6,576 notes with 34,832 links: 54 million. The same 180 notes as a
  spanning tree: 0. A readable vault drawing therefore needs a tree (or other sparse)
  skeleton that the flow keeps planar, with the remaining links drawn on top and ignored by
  the flow. That is a product decision, not a solver one.

(The result tables are at the end, under "Results on the final code".)

## Where a step's time goes, and a faster step (2026-11-24, night)

The plan above named "the solve" as the next lever (a sparse factorisation instead of
Jacobi-CG). Measured before building it, on the 36×36 grid at 2 points per edge (5,040
segments):

- A 100× softer mass term (`lambda` 0.02 → 0.0002) changes the opening by almost nothing
  (clearance after 400 steps 0.62 → 0.65) and triples the CG iterations (43 → 117) at the SAME
  time per step. So neither the number of steps nor the time per step is the solve's.
- A CPU profile of the step: the solve (`applyP` + CG) is 3 %. `Math.hypot` alone was 22 %,
  the neighbour grid 13 %, the pair search of the motion limits 16 %, the probes 19 %.
- A big group's grow-back is not limited by the solve either: the same softer mass term left
  the 134-node branch at 0.44 of its size after 400 steps (0.45 with the default).

**So no sparse solver was built.** Instead the hot loops were made cheaper, with the
trajectory unchanged to the printed digits:

1. the neighbour grid looks only at occupied cells (a flat drawing has one layer in z: 9
   bucket lookups per query instead of 27);
2. the pair search works on segment MIDPOINTS with a per-pair test, and the motion limits use
   it instead of their own copy;
3. the probes use the same midpoint grid, and reject by distance before the per-edge rules;
4. no `Math.hypot` in the hot loops.

Layout benchmark, same fixtures (`bench/results/2026-11-24-medial-graph-layout.jsonl`, the
last four graph-flow rows; "before" are the rows recorded earlier the same day; the "after"
run shared the machine with one other benchmark, so if anything it is pessimistic):

| Grid | Points per edge | Segments | Before | After | Result |
|---|---|---|---|---|---|
| 8×8 | 8 | 896 | 28 ms/step | 19 ms/step | converged in 285 steps (334), clearance 0.70 |
| 18×18 | 8 | 4,896 | 197 ms | 122 ms | 0.687 after 600 steps (0.679) |
| 18×18 | 2 | 1,224 | 21 ms | 12 ms | converged in 1,066 steps at 0.660 (0.650 after 1,500) |
| 36×36 | 2 | 5,040 | 106 ms | 56 ms | 0.655 after 1,500 steps (0.648) |

So 1.5–1.9× per step. (The step counts and clearances differ slightly from the recorded
rows because of the reaction rule described above, not because of the speed-up, which left
the trajectory unchanged to the printed digits.)

5. (found on the real maps) long segments are no longer tested against everything: every
   segment is cut into pieces no longer than twice the typical length for the searches
   (`segmentPieces`). A mind map drawn as a tidy tree has a third of its segments "long" (a
   hub's links to its far children) and cost 10× a grid's time per segment: the 944-node map
   went from 119 ms to 26 ms per step, 187 → 39 ms during a group move.

That is about 11 µs per segment per step. At the 30,000-vertex target (about 60,000 segments
at 2 per edge) a step would still take 0.7 s: constant factors will not get there. What will
is not stepping the whole drawing — only the part that is moving (see "What is next").

## Making room directly (2026-11-24, night)

**The failure.** A group set down shrunk grows back as the flow's contact forces push the rest
away. That grows it only as fast as the TIGHTEST gap around it opens, and a big group has a
long boundary: the 134-node branch of the 600-node tree went from 0.25 to about half its size
and stalled; 3 of the 14 real-map moves did not finish either. A softer mass term changes
nothing (measured above), so it is not the solver.

**The idea.** The room a group needs is known in advance: a disc around its anchor has to
become `k` times bigger. So the rest of the drawing can be moved there in one smooth motion,
instead of being pushed one gap at a time (`makeRoom` in `src/core/medialGraphMove.ts`,
default; `makeRoom: 'flow'` keeps the old behaviour as the comparison):

- inside the disc (the group's radius + 2r) everything is enlarged with the group, so the
  group and whatever lies between its branches keep their shape exactly;
- far outside every ring keeps its AREA and only moves outwards — nothing far away is
  squeezed, and the shift dies off like 1/distance;
- in between, a band where the area magnification falls from k² to 1, so the map has no
  crease (`roomRadius`).

**Why it cannot create a crossing.** The moved positions are accepted only if, moving every
node in a straight line from its old place to its new one, no node meets a segment — an exact
test for that motion (the signed area of node and segment is a quadratic in time), run over
every pair of segments within reach of each other. On top of that no small gap may lose more
than 30 %, and pairs that are allowed to cross may not cross more often. A trial that fails
is halved; if none passes, the group grows by what the gaps allow, as before.
Why an exact test and not the flow's "nobody moves more than 0.45 of the gap": making room
moves neighbours TOGETHER by many times the gap between them.

Two things found while building it:
- The disc must reach beyond the group (the 2r). With the group's farthest point exactly on
  the rim, a segment of the rest lying across the rim lagged behind it at every call; the gap
  between them closed geometrically (5e-7 after 40 calls) and the group stopped at 0.998.
- Clearing a region of the group's own OUTLINE instead of a disc (so that a group that is not
  round moves less of the drawing) was tried and dropped: a little less distortion of the
  rest, but the sheared band along the outline's flanks made the test refuse (the grid block
  stopped at 0.61 of its size) and the tree kept a tenth of the clearance.

## Results on the final code (2026-11-24, night)

"Waiting" = the rest is pushed away by the flow's contact forces only (`makeRoom: 'flow'`);
"making room" = the default. Same fixtures, same code otherwise.

### Group moves on grids and trees (`bench/results/2026-11-24-group-move.jsonl`, 800 steps)

| Fixture | Group | Set down at | Full size: waiting | Full size: making room | Crossings (held) | ms/step |
|---|---|---|---|---|---|---|
| 12×12 grid | 2×2 corner block to the middle | 1/8 | 36 steps | **10 steps** | 40 / 40 | 7 |
| 36×36 grid | same | 1/4 | 36 steps | **7 steps** | 132 / 132 | 64 |
| tree, 150 nodes | 40-node branch a third of a turn | 1/8 | 289 steps | **37 steps** | 4 / 2 | 6 |
| tree, 600 nodes | 134-node branch | 1/4 | not reached (0.64 after 800) | **138 steps** | — / 0 | 16 |

The crossings never grow after release, in any run. The grid numbers are the honest cost of
the request: a block moved into the middle of a grid keeps 4 links to the corner, and each
crosses everything in between. What making room costs the rest: its mean length error
(|length / rest length − 1|) is 0.04–0.14 at the moment of release against 0.05–0.12 when
waiting, and 0.05–0.10 by the end either way.

### Real mind maps and a vault (`bench/results/2026-11-24-pkm-graphs.jsonl`, 200 steps to settle, up to 600 for the move)

| Graph | Layout | Nodes | Crossings start → settled | Clearance start → settled | ms/step | Branch moved | Full size: waiting | Full size: making room | Crossings after |
|---|---|---|---|---|---|---|---|---|---|
| cli map | tidy tree | 944 | 0 → 0 | 0.56 → 0.57 | 44 | 42 | not reached (0.85) | 575 steps | 3 |
| cli map | force | 944 | 24 → 24 | 0.004 → 0.37 | 33 | 42 | fits at once | fits at once | 24 |
| linux map | tidy tree | 693 | 0 → 0 | 0.56 → 0.60 | 14 | 28 | fits at once | fits at once | 9 |
| linux map | force | 693 | 10 → 10 | 0.05 → 0.40 | 18 | 28 | 74 steps | **45 steps** | 13 |
| os map | tidy tree | 3,975 | 347 → 347 | 0.01 → 0.02 | 96 | 135 | fits at once | fits at once | 348 |
| os map | force | 3,975 | 582 → 580 | 0.001 → 0.006 | 94 | 135 | not reached (0.90) | **230 steps** | 579 |
| python map | tidy tree | 1,237 | 0 → 0 | 0.58 → 0.60 | 47 | 124 | fits at once | fits at once | 5 |
| python map | force | 1,237 | 54 → 54 | 0.008 → 0.24 | 62 | 124 | not reached (0.72) | **45 steps** | 53 |
| react-fiber map | tidy tree | 88 | 21 → 21 | 0.22 → 0.50 | 1.2 | 9 | 75 steps | 75 steps | 17 |
| react-fiber map | force | 88 | 7 → 7 | 0.11 → 0.34 | 1.2 | 9 | 74 steps | 74 steps | 10 |
| react map | tidy tree | 123 | 0 → 0 | 0.56 → 0.60 | 1.9 | 11 | fits at once | fits at once | 0 |
| react map | force | 123 | 2 → 2 | 0.007 → 0.42 | 1.8 | 11 | fits at once | fits at once | 2 |
| help vault as a tree | tidy tree | 180 | 0 → 0 | 0.60 → 0.59 | 4.2 | 18 | 401 steps | **28 steps** | 1 |
| help vault as a tree | force | 180 | 0 → 0 | 0.10 → 0.56 | 5.8 | 18 | fits at once | fits at once | 0 |

(Clearance is in units of the target clearance. Crossings at the start come from the layout
that made the drawing, not from the flow; the flow holds whatever it is given. ms/step was
measured with four benchmark processes running side by side.)

Reading it:
- Every move now ends at full size; three did not before.
- The flow never adds a crossing, on any real map, in either mode.
- The four maps of 700–4,000 nodes took 34–819 ms per step in the first run of this
  benchmark and take 14–96 ms now (the os map: 819 → 96 ms), from the search changes above.
- Two rows are identical in both modes (the react-fiber map): there every room-making trial
  is refused and the group grows by what the gaps allow. Not yet looked into.

### In the browser (`bench/results/2026-11-24-app-in-browser.jsonl`, headless Firefox)

| Check | Result |
|---|---|
| 8×8 grid, one node dragged 2.8 away | 0.14 from its target after 8 s, 0.007 after 20 s; 0 crossings; 2.5 ms/step |
| 8×8 grid, 2×2 block moved through | carried, back at full size 43 ms after release, 16 crossings held, on target |
| 12×12 grid, same | full size 130 ms after release, 40 crossings held, on target; 8 ms/step |

(The drag is slower over its last tenth than before the reaction rule: 0.02 from the target
after 8 s then. It still arrives.)

### What is still open

1. **An edge much longer than the group, passing right beside it, cannot get out of the
   way.** The cli map as a tidy tree: one 121-unit straight link (a hub with 200 children in
   a column) lies 0.2 from the group; its ends are far away and hardly move, the edge has 2
   points and cannot bend, so the exact test refuses every larger step and the move takes 575
   steps. Edges need more points where they are long (points in proportion to length, not a
   fixed number per edge).
2. **The crossings a move leaves are mostly avoidable on trees**: a branch of a tree can
   always be re-attached without crossings, the links just have to go around instead of
   straight. That is the case for routing.
3. **A drawing that starts with hundreds of crossings stays cramped** (the os map: clearance
   0.02 after settling): allowed crossings pin the strands that cross. Untangling such a
   drawing is a different job from keeping a clean one clean.

## Only step what moves: sleeping nodes (2026-11-24, morning after)

`medialGraphStep` takes `sleep: { below?, after? }`. A node that has moved less than
`below·r` (default 2e-4) in each of `after` (10) steps in a row goes to sleep: no probes, no
motion limits, held in the solve. It wakes when anything within reach of it moves more than
that, when the caller moves it (found by comparing positions with what the last step left),
or when it is dragged or held. With nothing awake and nothing touched the step returns before
building anything. The app's tick uses it; the stats line says "at rest" or how many nodes
are awake. Off by default in the core, so the benchmarks of the flow measure every node.

`bench/app/medialLocalDrag.ts` (`bench/results/2026-11-24-local-drag.jsonl`): a jittered grid
without tension (so that it comes to rest), settled, then its middle node dragged 2 units.

| Grid | Nodes | At rest, per step: all awake → sleeping | One node dragged: all awake → sleeping | Awake while dragging | Difference at the end |
|---|---|---|---|---|---|
| 60×60 | 10,680 | 162 ms → 0.2 ms | 159 ms → 110 ms | 5,112 (48 %) | 0.06 r |
| 120×120 | 42,960 | 546 ms → 4 ms | 584 ms → 252 ms | 6,889 (16 %) | 0.05 r |

("Difference at the end" = the largest distance between any node's final position with and
without sleeping. 0 crossings in every run.)

Reading it:
- **At rest the cost is gone**, at any size. That was the app's common state costing a full
  step per frame.
- **A drag is not local in an elastic drawing**: with every node awake, dragging one node 2
  units moves 6,900 nodes (60×60) or 10,900 (120×120) by more than a hundredth of a
  clearance. The number awake grows far more slowly than the map (5,100 → 6,900 for four
  times the nodes), which is the property wanted.
- **But the time does not follow yet**: 252 ms with 16 % awake. The lengths, the springs and
  the solve still run over every node whenever anything is awake; that fixed part is now
  most of the step, and is the next thing to make local.
- A larger threshold buys little and costs accuracy: at 5e-3 the 60×60 drag takes 57 ms but
  ends 1.3 r away from the all-awake result.

## A hub with 2,900 children: the pair search again (2026-11-24, midday)

The 6,576-note vault drawn as a tidy tree (one index note has 2,900 children) kept its 0
crossings but took **2.2 s per step**, 15× a grid's cost per segment. A profile put 78 % of
it in the pair search behind the motion limits. Three causes, three changes
(`forNearSegmentPairs`, `segmentPieces`):

| Cause | Change |
|---|---|
| Every pair of the hub's 2,900 edges was found and then thrown away (they share the hub, so they are never a pair): 8 million a step | each piece has a "home", the end of its segment with the most segments at it; a grid cell lists its pieces sorted by home, and a search skips the block with its own home (99 million candidates skipped per search) |
| Nearly EVERY segment of that drawing is long, so "pieces no longer than twice the typical length" meant pieces 156 units long and the whole drawing in 98 grid cells | the piece length follows the SEARCH radius: `sqrt(mean length · radius)`, at least the radius |
| Per candidate, data read from all over memory; per piece and cell, a map lookup and two binary searches | the pieces' data copied in cell order, distance rejected first; neighbour cells and big runs in per-cell tables (typed arrays: with plain arrays the same loop was 5× slower) |

| Drawing | Before | After |
|---|---|---|
| vault as a tidy tree, 6,576 notes, per step | 2,232 ms | about 750 ms |
| the same during a group move | 2,119 ms | about 610 ms |
| cli mind map, tidy tree, 944 nodes | 44 ms | 20 ms |
| python mind map, tidy tree, 1,237 nodes | 47 ms | 29 ms |
| os mind map, 3,975 nodes (tidy / force) | 96 / 94 ms | 71 / 76 ms |
| layout benchmark, 36×36 at 2 points | 56 ms | 52 ms |

Same results everywhere (steps, clearances, crossings, steps to full size). The "before"
column of the real maps was measured with four processes running, the "after" with two, so
part of their gain is the quieter machine; the vault's 3× is not.

**What is left is inherent to a grid search on this drawing:** 23 million candidate checks to
find 230 thousand pairs, because 2,900 edges fanning out of one point stay within reach of
their neighbours along most of their length. A search that knows about fans (edges of one
hub sorted by angle: only angular neighbours can be near) would be output-sized; so would not
drawing 2,900 children as one fan. Not done.

## Turning and re-attaching a moved group (2026-11-24, afternoon)

A carried group kept the direction it had. Moved to the other side of its parent, a branch
then points back INTO the map: its root on the far side, its link through its own body.

**Turning** (`setGroupOffset(…, { turn: 'links' })`, `facingAngle`). The group turns about its
centroid by the angle that brings the nodes carrying its links nearest to those links' other
ends: least sum of squared link lengths, which in the plane is one `atan2` (the 2-D orthogonal
Procrustes fit). Two details matter in use:

- It is applied RELATIVE to its value at pick-up, so picking a group up turns nothing; a branch
  drawn a little askew stays that much askew.
- In the app the dragged node stays under the pointer (`hold`). The turn depends on where the
  centroid is and, with a node held, the centroid on the turn; repeated substitution diverges
  as soon as the held node is farther from the centroid than the centroid is from the parent
  (measured: 3 units off the pointer), so the angle is found as a root on the circle
  (`turnHolding`), the one nearest the current angle.

**Hanging** (`hangGroupOn`, `reattachGroup`, `nearestOutsideNode`). The group's one link leaves
its old parent and goes to another node; the graph itself changes. The group is turned to face
the new parent and backed off to one link length. When the links land, a hung link goes to the
NEAREST NODE ITS END CAN SEE if the chosen one is not visible (`rehangOnVisible`): a branch
dropped "next to" a node is set down in a face of the drawing that need not touch that node
(tree150: 2 crossings hung blindly, 0 with the rule).

**A landed link takes up slack** (`MedialGraph.slack`). Found on the first turned clip: a link
is measured when it lands, while the drawing around is still pushed apart by the room that was
made; then everything settles back and the link has excess length it cannot lose. It landed 6.3
long between ends that settled 2.4 apart and stood as a Z. Now its rest length follows its ends
down, to no less than its length before the move (the floor is what keeps it from reeling the
group in).

### Measured (`bench/app/medialGroupMove.ts … warp <none|links|hang>`, `pkmGraphs.ts` likewise)

| Fixture | keep direction | turn | hang |
|---|---|---|---|
| tree, 150 nodes: steps to full size / crossings | 37 / 2 | 26 / 0 | 28 / 0 |
| tree, 600 nodes | 138 / 0 | 124 / 0 | 92 / 0 |
| 12×12 grid, 2×2 block (4 links) | 10 / 40 | 7 / 43 | as turn |
| 36×36 grid | 7 / 132 | 4 / 131 | as turn |

| 14 moves on real maps (a branch to the far side of the root) | keep direction | turn | hang |
|---|---|---|---|
| steps to full size, summed | 1,629 | 888 | 663 |
| crossings added by the moves, summed | +8 | −11 | −8 |
| moves at full size within 400 steps | 14 | 13 | 14 |

(Negative: a move can also REMOVE crossings the drawing had, by taking a branch out of them.)

**Where it does not help.**
- A block in a grid has links on several sides; the fit tilts it off the grid's axes for a
  small gain in link length: 3 more crossings on the small grid and the block's tightest gap
  at the floor (0.008 against 0.059). A measure of how well the links agree (all pulling one
  way: turn; from all sides: keep the direction) is the missing guard.
- One real move (the os map, force layout, 135 nodes) no longer reaches full size in 400 steps
  (0.71; unturned: 249 steps): turned, it is set down in a tighter place. The angle is chosen
  for the links alone; it does not look at what is in the way.
- (Hanging in the app: done the same day, see below.)

### Set down by score (`settleTurn`, same day)

The angle above is chosen for the links alone. At the moment of setting down, `settleTurn`
counts crossings for 17 candidate angles (the one the group has, the one it was picked up
with, 15 more round the circle) and takes the least; among equals, the nearest to the angle it
has, so in the open nothing changes. The score is crossings of the group's body with the rest
(temporary, but each costs growing time) + 8 × crossings of its links with the rest and with
the group's OWN body (permanent). The app does this on release when turning is on.

| 14 moves on real maps | keep direction | turn | turn, then set down by score |
|---|---|---|---|
| steps to full size, summed | 1,629 | 888 | 191 |
| crossings added by the moves, summed | +8 | −11 | −20 |
| moves at full size within 400 steps | 14 | 13 | 13 |

| Fixture: steps / crossings / tightest gap | keep direction | turn | by score |
|---|---|---|---|
| 12×12 grid, 2×2 block | 10 / 40 / 0.06 | 7 / 43 / 0.01 | 7 / 38 / 0.31 |
| 36×36 grid | 7 / 132 / 0.03 | 4 / 131 / 0.01 | 7 / 135 / 0.08 |
| tree, 150 nodes | 37 / 2 / 0.19 | 26 / 0 / 0.25 | 26 / 0 / 0.25 |
| tree, 600 nodes | 138 / 0 / 0.44 | 124 / 0 / 0.12 | 162 / 1 / 0.45 |

Two things learnt on the way:
- The first score left out a link crossing the group's own body, and then rated the UNTURNED
  angle best on exactly the moves where it is worst (a real map: 10 crossings against 1).
- The weight of link crossings hardly matters (1 and 8 differ on one move; 3, 8 and 1000 not
  at all in the first version).

Still open: the 600-node tree is worse by score (162 steps and 1 crossing against 124 and 0) —
crossings where a group is set down only predict those where its links land, after the rest
has made room; the os map's force-layout move still stops at 0.7 of its size at any angle
tried; the large grid gains nothing.

### Hanging in the app (same day)

"On release, hang a one-link group on the nearest node" (off by default: it changes the
graph). On release the group's link goes to the node nearest to where the dragged node was let
go, the group is turned and set down as above, and the app's edge list follows the flow's
segments (`mirrorEdges` in the store: the positions, pins and the running flow stay; only the
one link's ends and the pair list change). New preset "Flat Tree" (a radial mind map, 8–3,000
nodes): the grid has no branch to try this on.

The research on this (`_local/research/2026-11-24-11-moving-subgraphs-rotation-and-routing.md`)
found no published method for exactly this problem; what systems do is this fit, a damped
torque, or a search over some twenty candidate angles scored on link length, crossings and
overlap. The search is the next step for both open points above; for a tree branch, the wedge
of free directions at the new parent.

## A step costs what is awake (2026-11-24, afternoon)

With sleeping nodes the probes and the motion limits already cost only what is awake; the rest
of a step still ran over every node. A profile of a drag in a 42,960-node grid with 13 % awake
(240 ms a step) put 103 ms in the linear solve, 30 ms in the spring and bending passes and
30 ms in cutting and indexing every segment for the two neighbour searches.

| Part | Before | Change | After |
|---|---|---|---|
| Solve | over all nodes, held ones as identity rows | over the free nodes only (`slot`, `freeNode`, couplings between two free nodes) | 103 → 18 ms |
| Springs, bending | every segment | skipped between two held nodes | 30 → 9 ms |
| Neighbour searches | every segment cut into pieces and put in a grid, twice a step | the cut is made once and shared; with sleepers, only the segments near something that can move, probe or wake get pieces (`relevantSegments`) | about 30 ms less |

**All three give the same numbers as before, bit for bit**, which is the check that was used:
positions after four scenarios (plain, pinned and dragged, sleeping drag with a hand edit, group
move with a turn — also a drawing with a long link and a hub) hash the same on main and here.
Two things make that possible and are easy to lose:

- A held node's unknown and right-hand side are exactly 0, so every term its row and column
  would add to a sum is an exact 0; free nodes and segments keep their order.
- A grid over SOME of the pieces has the same cells as the grid over all of them
  (`SegmentPieces.origin`: the least midpoint over all segments, kept or not). Without it the
  restricted search found the same neighbours in another ORDER, and the result differed in
  the last bits (which then moved sleep decisions: 0.8 % more nodes awake). No needed segment
  was ever missing — checked at every step of the scenario — it was only the order.

`relevantSegments` is a superset by construction (a coarse mask of cells 2·reach wide, the 3×3
cells around every sample of every segment that may be awake) and has a brute-force test; with
the marks reduced to one cell the test fails.

### Measured (alternating main / this, least time of 20–30 steps; the machine was under other load)

| Drawing | Awake | main | now |
|---|---|---|---|
| 10,680 nodes, one node dragged | 12 % | 45–54 ms | 22–32 ms |
| 42,960 nodes, one node dragged | 2 % | 134–183 ms | 58–74 ms |
| 10,680 nodes, everything awake | 100 % | 112–127 ms | 107–118 ms |
| 42,960 nodes, everything awake | 100 % | 512–519 ms | 521–533 ms |

**The mask had to be written without small arrays** (same day, later). A statement-level
profile at 2 % awake showed `relevantSegments` and the origin loop themselves taking 50 of
120 ms: both made a two-element array per node or per segment (`for (const w of [a, b])`).
Written out, 6 ms; same results. Alternating with the first version: 42,960 nodes at 2 %
awake 73–79 → 46–47 ms (least of 20 steps), 10,680 nodes unchanged.

**Seen in the same profile, not yet settled:** the plain loops inside `medialGraphStep` look
several times slower than they should (segment lengths: 9 ms for 57,120 segments; 2 ms as a
function of its own, `segmentLengths`). The suspicion is that the engine's optimising tiers
do not take a function of that size, which would also slow the probes — the largest part of
a step with everything awake. Raising JavaScriptCore's size limit by environment variable
changed nothing, and the statement timers disturb what they measure, so this needs a clean
experiment: move the probes' loop into its own function and time it without timers.
`Math.hypot` is NOT the cause (replaced by a square root everywhere in both files: 2 ms of
79, and the results change in the last bits; reverted).

**Still not flat in the size of the drawing.** At 2 % awake a step of the 42,960-node drawing
costs about 46 ms where 10 would be in proportion: what is left are plain passes over all nodes
and segments (lengths, the pull's per-node set-up, position copies, the mask itself). Next:
profile at 2 % awake and give those passes the awake list; beyond that, neighbour lists kept
from step to step (a skin distance, rebuilt when something has moved half of it).

## What to cut: a crossing-free skeleton and an overlay (2026-11-24, evening)

A note vault cannot be drawn without crossings (180 notes and 883 links: 14,803 crossings in a
force layout; 6,576 notes: 54 million), and by Euler's bound (at most 3n − 6 links in a planar
drawing) at least 40 % of its links have to go whatever is done. So: which links form the
drawing that is kept crossing-free, and which are drawn on top of it?

`src/core/planarSkeleton.ts`, in the order of use:

1. `linkScores`: per link, the number of neighbours its two ends share. A link inside a tight
   group of notes is structure; one between notes that share nothing else is a jump.
2. `spanningTreeBy`: a spanning tree from the most linked note — plain breadth-first (`bfs`),
   breadth-first with the best-scored link to the level above (`bfs-best`), or the tree with
   the highest total score whatever its depth (`max-weight`).
3. `orderSiblings`: children that link to each other are put next to each other (chains,
   best links first): two children of one note can only be joined if nothing stands between.
4. The tree is drawn (any tree drawing without crossings).
5. `readmitLinks`: the other links come back in, best first, each only if drawn straight it
   stays a clearance away from everything already kept. The rest is the overlay.

No planarity test, no new embedding: the drawing is kept, which is what the flow needs.

### Measured (`bench/app/planarSkeleton.ts`; tidy tree drawing, siblings ordered)

| Vault | Links | Bound 3n − 6 | Tree | bfs | bfs-best | max-weight |
|---|---|---|---|---|---|---|
| 180 notes | 883 | 534 | 179 | 269 kept | 280 | **291** (33 % of the links, 54 % of the bound) |
| 6,576 notes | 34,832 | 19,722 | 6,575 | 6,639 | **7,173** (21 %, 36 % of the bound) | 6,789 |

- Every kept drawing has 0 crossings and still has 0 after the flow has run on it.
- Ordering the siblings is worth 35–46 links on the small vault and up to 381 on the large one
  (`bfs-best`: 217 → 598 taken back in).
- The force layout of the tree takes back fewer (small vault: 64 against 112) but looks like a
  graph, not a list.
- `max-weight` keeps most of the SCORE in the tree (66 % against 10–19 %) but its tree is 20
  levels deep on the large vault instead of 5, and a step of the flow on it is 4× slower.

**Why the large vault gains so little.** Of its 28,257 links outside the tree, only 107 join two
children of the same note (82 of them end up next to each other, and all 82 are kept). 25,942
join a note to a note one level up that is not its parent: a note linked from several index
notes. No ordering can draw those in a tree; they are what the overlay is for — or what copies
of a note under each of its parents would turn into tree links (the research's suggestion for
index notes; not built).

**What this settles for "untangle first, then move":** for a vault the crossing-free part IS
essentially a tree plus a few per cent; four fifths of the links are overlay whatever the
method. So the flow keeps the skeleton planar, a move acts on the skeleton, and the overlay is
display. Not built yet: showing the overlay in the app, and choosing the skeleton there.

## What is next

| # | Step | Why |
|---|---|---|
| 1 | The remaining whole-drawing passes of a step | solve, springs and neighbour index are awake-only now; at 2 % awake a 43,000-node step is still 46 ms where 10 would be in proportion; first the experiment on the step function's size (probes in a function of their own) |
| 2 | Points on an edge in proportion to its length | open item 1; also what lets long links curve |
| 3 | Route a moved group's links around what lies between | open item 2 |
| 3b | The skeleton in the app: choose it for a loaded graph, draw the overlay on top (faint, on demand) | "What to cut" is core + benchmark only |
| 4 | The move that never reaches full size (os map, 135 nodes); a score that predicts where links LAND | what set-down by score leaves open |
