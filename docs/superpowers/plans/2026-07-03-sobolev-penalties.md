# 2026-07-03 — Sobolev penalties milestone (briefing §5C): soft constraints + target-length animation

**Goal.** Implement the paper's penalty ("soft constraint") catalog for the
constrained Sobolev descent, plus the target-length animation hook, per
`local_files/2026-07-03-next-steps-briefing.md` §5C. Penalties enter the
OBJECTIVE (energy + differential), never the constraint rows; the fractional
H^s inner product is unchanged — paper-verbatim: "the energies considered here
involve lower-order derivatives … we can continue to use the fractional
Sobolev inner product without modification"
(`local_files/repulsive_orig_paper/SelfAvoiding.tex` line 769).

**Executor.** Fable inline (derivation-bearing milestone, user directive
2026-07-03), no reviewer pass. Verification is the usual oracle→golden→TS
pattern plus FD gates on both sides.

**Sources (verbatim formulas).** SelfAvoiding.tex §Constraints and Potentials,
lines 762–767 (penalty catalog), line 760 (target-length animation), line 734
tail (soft penalties as constraint-enforcement alternative).

## §1 Scope

IN:
1. **Total-length penalty** Ê_len(γ) = Σ_{I∈E} ℓ_I (tex line 763).
2. **Length-difference penalty** Ê_diff(γ) = Σ_{v∈V_int} (ℓ_{I_v} − ℓ_{J_v})²
   (tex line 764), V_int = vertices of degree EXACTLY 2.
3. **Field-alignment penalty** Ê_X(γ) = Σ_{I∈E} ℓ_I·|T_I × X|² with a
   CONSTANT unit field X (tex line 766, discretization verbatim).
4. Oracle penalty mode + new goldens; TS core module + threading through
   `sobolevStepSet`/`lineSearchStepSet`; store/UI (weight sliders, X control,
   target-length animation) — UI task sequenced AFTER `feat/pin-drag-ui`
   merges (both touch `src/store.ts` / `src/ui/ControlPanel.tsx`).

OUT (with reasons):
- **Surface potential** (tex line 765): needs a triangulated surface M + BVH —
  machinery the app does not have; separate milestone if ever.
- **Spatially varying X(c_I)**: adds a ∂X/∂c chain-rule term to the field
  gradient; v1 is a constant field (the `c_I` evaluation point only matters
  for varying fields). Documented limitation, not a silent one.
- Penalties as constraint rows, any change to projection/tolerances/H^s.

## §2 Derivations (spec-of-record — code comments cite this section)

Conventions (same as the constraint machinery, spec
`docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md` §2):
- Edge I = (a, b): e_I = γ_b − γ_a, ℓ_I = |e_I| RAW (no +ε — penalties are
  geometric, same rule as `total_length` in `oracle/tpe_constraints_oracle.py`).
- T_I = e_I/ℓ_I with the safe-unit guard: ℓ_I < 1e-14 ⇒ T_I := 0 (same
  constant as `safe_unit` / the TS constraint rows) ⇒ degenerate edges
  contribute ZERO to every penalty energy and gradient below.
- Gradient orientation: ∂E/∂γ (ascent representative), identical to
  `gradientAnalytical`'s dE — penalties ADD to dE before the saddle solve.
- Summation order (bit-stable goldens): edges in index order; V_int vertices
  in index order.

### 2.1 Total length

E = w·Σ_I ℓ_I.  ∂ℓ_I/∂γ_a = −T_I, ∂ℓ_I/∂γ_b = +T_I, so per edge I=(a,b):

    dE[a] += −w·T_I,   dE[b] += +w·T_I

Cross-check: this is exactly −w × (the totalLength constraint row), since
Φ_len = L⁰ − Σℓ has row +T_I at a, −T_I at b
(`oracle/tpe_constraints_oracle.py` `total_length_phi_and_C`). The briefing's
"totalLength row negated" note is this identity.

### 2.2 Length difference

E = w·Σ_{v∈V_int} (ℓ_{I_v} − ℓ_{J_v})², V_int = {v : deg(v) = 2} (paper
line 764: "interior" vertices; degree-1 endpoints and degree≥3 junctions are
EXCLUDED). I_v, J_v = the two incident edges of v, ordered by ascending edge
index — the choice is arbitrary because both the value and the gradient are
symmetric under I↔J swap: (ℓ_I−ℓ_J)² and 2(ℓ_I−ℓ_J)(dℓ_I−dℓ_J) each flip
sign twice.

Per interior vertex v, with d_v = ℓ_{I_v} − ℓ_{J_v}:

    for edge I_v=(a,b):  dE[a] += −2w·d_v·T_{I_v},  dE[b] += +2w·d_v·T_{I_v}
    for edge J_v=(a,b):  dE[a] += +2w·d_v·T_{J_v},  dE[b] += −2w·d_v·T_{J_v}

(a 3-vertex stencil: v and its two neighbors; each edge endpoint gets the
±T pattern of ∂ℓ/∂γ scaled by ±2w·d_v).

### 2.3 Field alignment (constant unit X)

E = w·Σ_I ℓ_I·|T_I × X|². With |T_I| = 1 and |X| = 1:
|T×X|² = 1 − (T·X)², hence E_I = ℓ − (e·X)²/ℓ with u := e·X. Then

    ∂E_I/∂e = T − (2u/ℓ)·X + (u/ℓ)²·T = (1 + (T·X)²)·T − 2(T·X)·X
    dE[a] += −w·g_I,  dE[b] += +w·g_I,  where g_I := (1+(T·X)²)T − 2(T·X)X

Limit checks (sanity, also unit-tested): e ∥ X ⇒ g = 2T − 2T = 0 (aligned is
stationary); e ⊥ X ⇒ g = T (E_I = ℓ, pure length growth resistance). Note g
is BOUNDED as ℓ→0 (u/ℓ = T·X), so the only degenerate-edge handling needed is
the T:=0 guard above. X is normalized ONCE at config read; |X| < 1e-14 ⇒ the
field penalty is inactive that step (weight treated as 0).

### 2.4 Objective composition

E_total(γ) = E_tpe(γ) + Σ_k w_k·Ê_k(γ), dE_total = dE_tpe + Σ_k w_k·dÊ_k.

- dE_total feeds the saddle solve (H^s inner product UNCHANGED, tex line 769).
- Armijo gates on E_total: E_total(γ_proj) ≤ E_total(γ₀) − c₁·τ·(dE_totalᵀp).
  The line-search TSDoc's "the ENERGY is not pluggable" contract is preserved
  in spirit: the energy is still the app's own objective — the objective now
  includes the active penalties.
- `energyBefore` (E₀ reuse, solver-perf Task 4) invariant becomes "the TOTAL
  objective at γ₀ under the SAME penalty config"; the returned `energy` of an
  accepted step is the total objective so chaining stays free. The store must
  INVALIDATE its cached energy when the penalty config changes (Task 5).
- Convergence test (‖g̃‖_{L²ₕ} < 1e-4) and the saddle residual
  self-certification are untouched.
- All-zero / absent penalty config ⇒ every code path BIT-IDENTICAL to today
  (guarded by `penaltiesActive`; explicit bit-identity test, same pattern as
  `frozenProjection.test.ts`'s default-path guard).

## §3 Decision ledger (à la formula-audit item 9 — OUR choices, not paper's)

| choice | value | why |
|---|---|---|
| penalty weights | free knobs, golden presets recorded in each golden's `conventions` | paper gives no values |
| V_int ordering | vertex index asc; I_v/J_v by edge index asc | determinism; math is swap-symmetric |
| X handling | normalize once; ‖X‖<1e-14 ⇒ inactive | paper says "unit vector field" |
| degenerate edges | T:=0 at ℓ<1e-14 ⇒ zero contribution | same guard as constraint rows |
| returned `energy` | total objective | E₀-reuse chaining (§2.4) |
| oracle dE split | FD for TPE part + ANALYTIC penalty gradient (mirrored formulas both sides) | golden agreement at 1e-9 needs identical penalty formulas; FD-of-total would inject O(h²) mismatch vs TS analytic |
| oracle penalty presets | named presets in oracle (weights + X hardcoded, values chosen at golden generation and recorded) | short regen commands, no JSON-arg parsing |
| new oracle constraint mode `bary` | barycenter-only set | the paper's soft-mode comparison runs penalties WITHOUT hard length constraints (tex Multiresolution fig caption); penalty goldens need that configuration |

## §4 Tasks

1. **Plan doc** (this file), committed alone.
2. **Oracle**: extend `oracle/tpe_constraints_oracle.py` with (a) mode `bary`
   (barycenter-only set); (b) penalty presets (`pen-length`, `pen-diff`,
   `pen-field`, `pen-combo`) adding analytic penalty gradients (formulas §2,
   mirrored op order) to dE and penalty energies to the line-search objective;
   (c) embedded property checks: central-FD check of the analytic penalty
   gradient (rel ≤ 1e-6, `fd_jacobian_check` pattern), descent positivity and
   energy decrease in the TOTAL objective, penalty actually active
   (E_pen > 0, dE_total ≠ dE_tpe); (d) goldens on `crossing` (closed loop) and
   `junction-y` (degree-3 junction + open endpoints — exercises the V_int
   exclusions). Update `oracle/README.md`. Gate: all property checks pass;
   pre-existing goldens byte-identical (penalty code inert unless preset given).
3. **TS core** `src/core/sobolev/penalties.ts` (TDD): `PenaltyConfig`
   { totalLength?, lengthDiff?, field?: {weight, X} }, `penaltiesActive`,
   `penaltyEnergy`, `penaltyGradient` (formulas §2, TSDoc @see this doc + tex
   lines). Unit tests `test/sobolev/penalties.test.ts`: central-FD gradient
   check per penalty on crossing + junction-y geometry; V_int exclusion
   (endpoint + junction vertices contribute nothing); field limit cases (∥→0,
   ⊥→T); degenerate-edge zero-contribution.
4. **Threading**: `SobolevStepOptions.penalties?` / `LineSearchOptions.
   penalties?` → dE_total before the solve, E_total in Armijo (§2.4). Tests:
   golden diffs vs Task-2 goldens at the M2-era tolerances (1e-9 rel on
   dE_total/g̃/vertices, τ exact); zero-config bit-identity vs a no-option run.
   Gate: entire pre-existing suite untouched and green.
5. **Store/UI** (**BLOCKED until `feat/pin-drag-ui` merges** — shared files):
   penalty weights + X in store, threading into `dispatchDescentStep`,
   energyBefore invalidation on config change (§2.4), weight sliders + X
   control in ControlPanel, **target-length animation**: per accepted step,
   scale the length targets L⁰/ℓ⁰_I by a user rate and rebuild the constraint
   blocks from the SCHEDULE (never from the current iterate) — deliberate,
   documented exception to the frozen-targets anchor, sanctioned by tex line
   760 ("we progressively increase or decrease the target length values; the
   next constraint projection step then enforces the new length").
6. **Bench sanity**: one `bench/sobolev.bench.ts` run with penalties active —
   no gate (penalties are O(E), expected noise-level); record in ledger only
   if regression > noise appears.

## §5 Gates (pre-registered)

- FD rel ≤ 1e-6 on every penalty gradient, BOTH sides (oracle property check,
  TS unit test).
- Existing goldens and tests: zero changes, all green ("fix the code, never
  the tolerance").
- New goldens: TS vs oracle at 1e-9 rel (dE_total, g̃, accepted vertices),
  τ and projection-iteration counts exact.
- Bit-identity with penalties disabled (explicit test).
- Saddle residual self-certification (≤1e-10) untouched on every solve.
