# Sobolev flow constraints: total length, per-edge length, point pins

**Status:** approved design, not yet implemented.
**Scope:** two milestones (M1: total-length constraint; M2: per-edge length + point
constraints), each ONE commit. Treat each milestone as its own plan unit under
CLAUDE.md's one-commit-per-plan rule.
**Branch:** continue on `feat/sobolev-stage1` (or a child branch off it).

## 0. How to use this document (fresh-session entry point)

You are implementing the next milestone of the Repulsive Curves (Yu/Schumacher/
Crane 2021) Sobolev descent. Do this, in order:

1. Read `CLAUDE.md` (binding house rules: TSDoc `@see` on every export, anchor
   comments, inline-fix gate, disagreement protocol).
2. Read the files in §1 "Required reading" below.
3. Detect state: `git log --oneline | head -20`. If no commit mentions
   "constraints M1", implement **M1 (§4)**. If M1 is committed but nothing
   mentions "constraints M2", implement **M2 (§5)**. If both exist, stop and ask.
4. Confirm the baseline is green before touching anything:
   `bun test` (all pass) and `bunx tsc --noEmit` (clean).
5. Execute the milestone. Tests first where the spec defines expected values
   (the Jacobian rows and goldens make everything falsifiable — watch new tests
   fail before implementing).
6. Finish with the report format in §9. Do NOT push.

Everything you must respect about the existing code is stated here or in the
required reading; if a symbol you need is missing from both, read the source —
do not guess.

## 1. Required reading

| What | Where | Why |
|---|---|---|
| House rules | `CLAUDE.md` | binding |
| Existing constraint machinery | `src/core/sobolev/constraints.ts` | the pattern every new row follows (raw-ℓ, T=0 guard, `blockIndex` layout) |
| Saddle solve + composition | `src/core/sobolev/linsolve.ts`, `src/core/sobolev/gradient.ts` | the solver interface constraint rows feed into |
| Projection + line search | `src/core/sobolev/lineSearch.ts` | the loop that must become constraint-set-generic |
| App-facing step + store | `src/core/optimizer.ts` (`sobolevStep`), `src/store.ts` (frozen-x0 lifecycle anchor), `src/scene/Viewer.tsx` (Simulation) | where config/targets thread through |
| Verification harness | `oracle/README.md` | how goldens are produced and gated; regen commands |
| Paper ground truth | `local_files/repulsive-curves-excerpts.tex` §"Constraints and Potentials" (lines ~741–802) and §"Constraints" (~597–660) | the constraint catalog quoted in §2 below |
| Conventions + §E property list | `local_files/2026-07-02-sobolev-gradient-rsrch-results.md` §B, §E | raw-ℓ rule, x₀ freezing, FD-Jacobian check pattern (prop 7) |
| What is paper-sourced vs our invention | `local_files/2026-07-02-sobolev-formula-audit.md` items 7–9 | keep that ledger accurate for everything you add |

`local_files/` is local-only (not committed) but present on this machine.

## 2. Ground truth math (paper-verbatim; do not re-derive Φ)

From the paper's constraint catalog (excerpts §"Constraints and Potentials"),
with ℓ_I = ‖γ_{i2} − γ_{i1}‖ the RAW geometric edge length and
T_I = e_I/ℓ_I the unit tangent:

- **Barycenter** (already implemented): Φ_bar(γ) = Σ_{I∈E} ℓ_I·(m_I − x₀) ∈ R³.
- **Total length** (M1): Φ_len(γ) = L⁰ − Σ_{I∈E} ℓ_I ∈ R  (1 row).
- **Edge length** (M2): Φ_{len,I}(γ) = ℓ⁰_I − ℓ_I ∈ R  (1 row per edge, |E| rows).
- **Point** (M2): Φ_{pt,i}(γ) = γ_i − x_i ∈ R³  (3 rows per pinned vertex).

Jacobian rows (C = dΦ, columns in the coordinate-block layout of
`src/core/sobolev/layout.ts` via `blockIndex(coord, vertex, n)`), using
dℓ_I = T_I·(dγ_{i2} − dγ_{i1}):

- **Total length** row: for every edge I=(i1,i2), accumulate
  `∂Φ_len/∂γ_{i1} += +T_I` and `∂Φ_len/∂γ_{i2} += −T_I`
  (signs from Φ = L⁰ − Σℓ: dΦ = −Σ dℓ_I). Junctions/endpoints need no special
  case — every incident edge adds its term to the same vertex columns, exactly
  as in `barycenterPhiAndC`.
- **Edge length** row I: `+T_I` at i1, `−T_I` at i2, that edge only.
  (Note: the total-length row is exactly the SUM of all edge-length rows —
  this is the rank-dependence rule in §3.4.)
- **Point** rows for vertex i: identity block,
  `C[r][blockIndex(r, i, n)] = 1` for r ∈ {0,1,2}. No length terms.

Conventions that are LOAD-BEARING and already anchored in the code — follow
them verbatim and keep the anchor comments:

- RAW lengths, no +ε: constraints are geometric, not part of the regularized
  energy. Same rule as `barycenterPhiAndC` (see its inline anchor citing
  rsrch-results §B "Use raw geometric lengths ... not ℓ^ε").
- Degenerate guard: T_I = [0,0,0] when ℓ_I < 1e-14 (same guard, same constant,
  same comment style as `constraints.ts`). A degenerate edge zeroes its
  edge-length row → singular system → the existing `singular_system` rejection
  path in `sobolevStep` is the backstop (never crash).
- Signs are paper-verbatim (target-minus-current for lengths). The saddle
  solve and the projection RHS (−Φ) only require internal consistency, but
  keep the paper's signs so future audits can diff against the excerpts 1:1.

## 3. Design (both milestones)

### 3.1 ConstraintSet abstraction — new module `src/core/sobolev/constraintSet.ts`

The current code hardcodes `barycenterPhiAndC` inside `solveConstrainedGradient`
and `projectBarycenter`. Generalize to a stacked-blocks model:

```ts
interface ConstraintEval { phi: number[]; C: number[][] }        // k rows, 3|V| cols
interface ConstraintBlock {
    kind: 'barycenter' | 'totalLength' | 'edgeLengths' | 'point';
    evaluate(vertices: Vec3[], edges: Edge[]): ConstraintEval;   // this block's rows
    scale(vertices: Vec3[], edges: Edge[]): number;              // tolerance scale, §3.3
}
type ConstraintSet = ConstraintBlock[];                          // stacked in array order
```

Builders: `barycenterBlock(x0)` (wraps the existing `barycenterPhiAndC` +
`barycenterScale` — do NOT reimplement them), `totalLengthBlock(L0)` (M1),
`edgeLengthsBlock(ell0: number[])`, `pointBlock(vertexIndex, target)` (M2).
Plus `evaluateConstraintSet(set, vertices, edges): ConstraintEval` that stacks
all blocks' rows in set order.

### 3.2 Back-compat is the regression proof (hard requirement)

- The generalized internals take a `ConstraintSet`. The EXISTING exported
  signatures (`solveConstrainedGradient(..., x0)`, `projectBarycenter(..., x0, opts)`,
  `lineSearchStep(..., x0, opts)`, `sobolevStep(..., x0, opts)`) stay intact and
  delegate with `[barycenterBlock(x0)]`.
- Row order: barycenter block FIRST when present. With the barycenter-only
  set this must be numerically bit-identical to today's path — proven by:
  **all existing tests (including golden diffs) pass UNMODIFIED.** If a golden
  test needs edits, the refactor is wrong; stop and fix the refactor.
- New generalized entry points are new exports (suggested:
  `solveConstrainedGradientSet`, `projectOntoConstraintSet`,
  `lineSearchStepSet`, `sobolevStepSet` — naming is implementer's taste, the
  delegation structure is not).

### 3.3 Projection stopping tolerance with mixed rows (OUR invention — flag it)

> **SUPERSEDED VALUE (2026-07-03, solver-perf Task 6):** tolAbs = tolRel =
> **1e-4** — the reference implementation's `backproj_threshold`
> (ythea/repulsive-curves src/tpe_flow_sc.cpp:15). The per-block RULE below is
> unchanged and still ours. The original 1e-10 caused a false kill of the
> paper's factorization-reuse scheme; it survives only in the read-only
> stage-1 oracle and its pinned golden tests. The 1e-8 drift gates quoted in
> §4.4.3/§5.2/§5.4 correspondingly became stopping-rule-tracking bounds
> (≤ 1e-4·scale_b). @see oracle/README.md ("Projection tolerance provenance")

Current: ‖Φ‖₂ ≤ max(tolAbs, tolRel·barycenterScale). With heterogeneous rows,
check per block: converged iff EVERY block satisfies
`‖Φ_block‖₂ ≤ max(tolAbs, tolRel·block.scale(...))`, with tolAbs = tolRel = 1e-10
and maxIter = 8 unchanged. Block scales:

- barycenter → existing `barycenterScale` (unchanged);
- totalLength → `max(1, L)` where L = Σℓ_I (raw);
- edgeLengths → `max(1, L)`;
- point → `max(1, R)` with R = max distance from any vertex to the pin target.

This scaling is our tunable choice, NOT paper-sourced — say so in a TSDoc
anchor exactly the way `lineSearch.ts` flags its constants (cite
`local_files/2026-07-02-sobolev-formula-audit.md` item 9), and add it to the
§8 audit checklist.

### 3.4 Rank rules (construction-time errors, not solve-time mysteries)

- `totalLengthBlock` and `edgeLengthsBlock` are MUTUALLY EXCLUSIVE in one set
  (total row = sum of edge rows ⇒ exact rank deficiency ⇒ singular saddle).
  Composing both must throw at set-construction time with a message citing
  this section. The UI must not offer the combination (§4.3/§5.3).
- Pinning both endpoints of a length-constrained edge is allowed (generically
  independent rows) but can make projection infeasible if targets disagree;
  that surfaces as the existing `projection_failed` rejection — acceptable,
  no special handling.
- Anything that still slips to a singular solve hits `sobolevStep`'s existing
  `singular_system` reject-and-report path. Never throw from the frame loop.

### 3.5 Frozen-targets lifecycle (same anchor as x₀ — extend it, don't fork it)

L⁰, ℓ⁰ (vector), and pin targets are FROZEN constraint targets with exactly the
x₀ lifecycle documented in `src/store.ts` (the `sobolevX0` anchor): recomputed
ONCE per run (re)start — on play, on preset/config change, on vertex commit —
NEVER during a run. Consequence to note in the anchor: L⁰ re-anchors to the
current length at each pause/play boundary, so per-run length is preserved but
sub-tolerance drift can accumulate across pause cycles — accepted.

### 3.6 Oracle extension (goldens for the new rows)

`oracle/tpe_stage1_oracle.py` is a read-only deliverable — do NOT edit it.
Add a NEW script (suggested `oracle/tpe_constraints_oracle.py`) that imports
the stage-1 oracle's functions and adds the new constraint rows, emitting new
golden files (`oracle/golden/<fixture>-length.json` etc. — never overwrite the
existing five goldens). Regen commands follow `oracle/README.md`:

```bash
uv run --with numpy --with scipy python oracle/tpe_constraints_oracle.py \
  oracle/fixtures/crossing.json oracle/golden/crossing-length.json
```

Update `oracle/README.md`'s file list and status when you add it.

## 4. Milestone M1 — total-length constraint

Commit message starts `feat(sobolev): total-length constraint (constraints M1)`.

### 4.1 Core

- `constraintSet.ts` with `barycenterBlock`, `totalLengthBlock`,
  `evaluateConstraintSet` (§3.1), block scales (§3.3), the mutual-exclusion
  throw stubbed for M2's `edgeLengthsBlock` (or added in M2 — implementer's
  call, but the rule text in §3.4 gets cited either way).
- Generalize `gradient.ts` + `lineSearch.ts` internals per §3.2 (wrappers keep
  old signatures; goldens unmodified).
- `sobolevStepSet(vertices, edges, disjointPairs, set, opts)` in
  `optimizer.ts`; `sobolevStep` delegates with the barycenter-only set.

### 4.2 Store

- `lengthConstraint: boolean` (default **true** — without it the flow dilates
  forever and the ‖g̃‖ < 1e-4 termination never fires; with it, sobolev runs
  have an equilibrium), `setLengthConstraint(b)`.
- `sobolevL0: number` frozen alongside `sobolevX0` at the same three lifecycle
  points (§3.5). The Simulation dispatch builds the ConstraintSet from
  (sobolevX0, lengthConstraint ? sobolevL0 : nothing) — raw mode untouched.

### 4.3 UI

- ControlPanel: "Fix length" checkbox next to the Descent select (affects
  sobolev mode only; visually group or disable it in raw mode).
- Stats (sobolev mode): current total length L and drift |L − L⁰|/L⁰ alongside
  the existing τ/residual/‖g̃‖ readouts.

### 4.4 Tests (new file; existing tests UNMODIFIED)

1. **FD Jacobian**: total-length row vs central finite differences of Φ_len on
   the crossing fixture, per §E prop 7's pattern; gate ~1e-6 relative for
   h=1e-6 (see `oracle/README.md` "Known tolerance caveats" for why not
   tighter).
2. **Golden diff**: constrained g̃ + one line-search step vs the new oracle
   golden on ≥2 fixtures (crossing + one loop fixture), same tolerances as
   `test/sobolev/gradient.test.ts` / `lineSearch.test.ts` (1e-10 rel on g̃,
   τ exact, 1e-12 energy rel).
3. **Flow property** (oracle-independent, pattern of the lineSearch flow
   test): ≥5 sobolev steps on crossing with barycenter+length: every step
   accepted, energy strictly decreases, and |L − L⁰|/L⁰ ≤ 1e-8 after EVERY
   step (projection tolerance, not drift).
4. **Back-compat**: `sobolevStep` (x0 signature) output on crossing is
   bit-identical to `sobolevStepSet` with `[barycenterBlock(x0)]`.
5. Store: L⁰ freezing at the three lifecycle points (mirror the existing
   `sobolevX0` test in `test/optimizer-sobolev.test.ts`).

### 4.5 Acceptance gates

`bunx tsc --noEmit` clean; `bun test` all green with zero edits to existing
test files; oracle property/golden scripts pass for the new goldens; boot the
app (`bun run dev`, see the headless recipe in the project memory if no
display) and confirm: sobolev + fixed length on the crossing preset descends
with L drift ≤ 1e-8 and — unlike before — ‖g̃‖ now DECREASES over time. Quote
a few frames in the report.

## 5. Milestone M2 — per-edge length + point constraints

Commit message starts `feat(sobolev): per-edge length + point constraints (constraints M2)`.

### 5.1 Core

- `edgeLengthsBlock(ell0: number[])` (|E| rows) and
  `pointBlock(vertexIndex, target)` (3 rows) per §2; the §3.4 mutual-exclusion
  throw goes live.
- Saddle system size becomes 3|V| + 3 + |E| (+3 per pin) — still dense LU,
  still within the stage-1 |V| ≤ ~300 budget; no solver changes.

### 5.2 Oracle

Extend the M1 constraints oracle script with both block types; new goldens
(`<fixture>-edgelengths.json`, `<fixture>-point.json`). Existing goldens
untouched.

### 5.3 Store/UI

- Replace the M1 checkbox with a 3-way "Length" select: `none | total | per-edge`
  (mutual exclusion enforced by construction). Frozen ℓ⁰ vector follows §3.5.
- Point constraints: MACHINERY + tests only. No picking UI in this milestone —
  interactive vertex pinning/dragging is a separate future milestone (raycast
  + drag interaction is scene work, out of scope here). knip will flag the
  point-constraint exports as unused: expected, note it in the report; knip is
  non-blocking.

### 5.4 Tests

1. FD Jacobian per block type (edge-length rows on a junction fixture too —
   `oracle/fixtures/junction-y.json` — to exercise shared-vertex accumulation;
   point rows trivially).
2. Golden diffs for both block types on ≥2 fixtures.
3. Flow property: per-edge mode on the crossing fixture — every ℓ_I drift
   ≤ 1e-8 per step, energy decreases ("isometric untangling").
4. Rank rule: composing totalLength + edgeLengths throws at construction; a
   pinned vertex + per-edge lengths on its edges still solves (accepted step)
   on the crossing fixture.
5. Point block: pin vertex 0 on crossing, run 3 steps — γ₀ stays within 1e-8
   of the target while energy decreases.

### 5.5 Acceptance gates

Same as §4.5, plus the in-app check runs the `per-edge` mode.

## 6. Explicitly OUT of scope

Interactive vertex picking/dragging; surface and tangent constraints (in the
paper catalog — leave for later); any solver change (multigrid, factorization
reuse — the spec-guarded reassembly-per-iterate stays); any raw-descent-path
change (A/B contract: raw stays byte-identical); React Compiler; flat-array
refactor (issue #1).

## 7. Budget

M1: ≤ ~8 impl files (constraintSet.ts new; gradient.ts, lineSearch.ts,
optimizer.ts, store.ts, ControlPanel.tsx, Stats.tsx, Viewer.tsx touched),
+1 oracle script, +goldens, +1–2 test files. M2: ≤ ~6 impl files. One commit
each, `--amend` while iterating, do not push.

## 8. Optional math-audit checklist (hand this spec to an isolated auditor)

If a verification pass is wanted before/after implementation, the auditor
should check exactly these claims against `repulsive-curves-excerpts.tex`:

1. Jacobian signs in §2 (Φ = target − current ⇒ +T_I at i1, −T_I at i2).
2. The rank-dependence rule (§3.4): total row = Σ edge rows, hence exclusion.
3. The per-block tolerance scaling (§3.3) — our invention; is it sane, and
   are there row combinations where max(1, L)/max(1, R) misscale badly?
4. Feasibility: barycenter + total length rows generically independent
   (4 rows); barycenter + all |E| edge rows on closed loops.
5. That nothing here needs the dℓ terms frozen (paper defines C := dΦ, full
   Jacobian — same as audit item 8 already confirmed for the barycenter).

## 9a. Addendum (2026-07-03, user request): toggles for ALL constraints

User directive during M1 execution: "i want control panel toggles for all
constraints/penalties". Deltas to the sections above (no penalty terms exist
yet — this covers constraint blocks):

- **§3.2/§3.4**: the generalized internals must accept a ConstraintSet WITHOUT
  the barycenter block, including the EMPTY set (k = 0 rows → the saddle system
  degenerates to Ā·g̃ = dE; `buildSaddleMatrix`/`solveSaddle` already handle
  k = 0). Projection onto the empty set converges trivially at iteration 0.
  Row order rule unchanged: barycenter first WHEN present.
- **§4.2 store**: additionally `barycenterConstraint: boolean` (default
  **true** — preserves current behavior) + setter; the Simulation dispatch
  includes `barycenterBlock(sobolevX0)` only when enabled.
- **§4.3 UI**: a checkbox per constraint block (M1: "Barycenter" and
  "Fix length"), not just the length one. Same sobolev-only scoping.
- **§5.3 (M2)**: the per-edge/total/none Length select replaces the length
  checkbox as written, and the Barycenter checkbox stays.

## 9. Report format (end of each milestone)

Files touched with line counts; WHICH existing exports changed signature
(must be: none — wrappers only) and which tests were edited (must be: none
existing); verbatim `bun test` tail; the flow-test numbers (energies, τ, L
drift per step); oracle regen output for the new goldens; boot-check
observations (quote τ/‖g̃‖/L for a few frames); any ambiguity resolved and
how; commit hash.
