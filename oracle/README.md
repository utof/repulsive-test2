# Stage-1 Sobolev-gradient oracle harness

Independent Python reference implementation of the fractional Sobolev gradient
(Repulsive Curves, Yu/Schumacher/Crane 2021) plus the fixtures, goldens, and
property checks that gate the TypeScript implementation.

Why this exists: the TS Sobolev code (Stage 1) is verified by *diffing against
an independent implementation*, not by refereeing derivations. The oracle and
the TS code can only agree if both implement the stated math.
Spec/conventions: `local_files/sobolev-gradient-handoff.md`;
formulas + property list: `local_files/2026-07-02-sobolev-gradient-rsrch-results.md`
(local-only docs, not committed).

## Files

- `tpe_stage1_oracle.py` — the reference implementation (deep-research
  deliverable, treat as read-only; numpy + scipy, deterministic). Computes
  energy, FD dE, B, B⁰, A, Ā (=A3), barycenter Φ/C, constrained Sobolev
  gradient g̃, and one accepted line-search step.
- `gen_fixtures.ts` — serializes the deterministic app presets (+ a hand-built
  degree-3 junction graph) to `fixtures/*.json`.
- `fixtures/*.json` — oracle inputs `{name, vertices, edges, alpha, beta, epsilon}`.
- `golden/*.json` — oracle outputs; the diff targets for the TS milestones.
- `tpe_constraints_oracle.py` — constraints oracle (M1: barycenter + total
  length; M2: per-edge lengths + point pins) built ON TOP of
  `tpe_stage1_oracle.py` (imports it; the stage-1 script stays read-only).
  Modes `length` / `edgelengths` / `point` emit `golden/<fixture>-length.json`,
  `golden/<fixture>-edgelengths.json`, `golden/<fixture>-point.json` and run
  their own embedded property checks (FD Jacobian of the stacked C, saddle
  residual, descent positivity, accepted step, energy decrease, per-block Φ
  tolerance, and the mode drift: total-length drift / max per-edge drift /
  pin distance, each ≤ 1e-8). Spec:
  `docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md` §3.6, §5.2.
- `check_properties.py` — §E property checklist run against the goldens
  (symmetry, PSD, constant nullspace, quadratic-form identity via an
  independent direct-sum transcription, saddle residuals, descent positivity,
  Armijo, barycenter preservation, scaling laws, orientation invariance).
- `compare_energy.ts` — cross-language gate: verified TS energy/dE vs the
  oracle's, on the same fixtures.

## Regenerate / verify

```bash
bun oracle/gen_fixtures.ts
for f in crossing junction-y helix linked-rings knot; do
  uv run --with numpy --with scipy python oracle/tpe_stage1_oracle.py \
    oracle/fixtures/$f.json oracle/golden/$f.json
  uv run --with numpy --with scipy python oracle/check_properties.py \
    oracle/fixtures/$f.json oracle/golden/$f.json
done
bun oracle/compare_energy.ts
```

Constraints goldens (M1, total-length constraint — self-checking, no separate
property script needed):

```bash
for f in crossing linked-rings; do
  uv run --with numpy --with scipy python oracle/tpe_constraints_oracle.py \
    oracle/fixtures/$f.json oracle/golden/$f-length.json
done
```

Constraints goldens (M2, per-edge length + point pin — same self-checking
script, modes `edgelengths` / `point`):

```bash
for f in crossing junction-y; do
  uv run --with numpy --with scipy python oracle/tpe_constraints_oracle.py \
    oracle/fixtures/$f.json oracle/golden/$f-edgelengths.json edgelengths
done
for f in crossing linked-rings; do
  uv run --with numpy --with scipy python oracle/tpe_constraints_oracle.py \
    oracle/fixtures/$f.json oracle/golden/$f-point.json point
done
```

Status (2026-07-02): all property checks and cross-language gates pass on all
five fixtures; every fixture accepts the full τ=1 line-search step with one
projection iteration (vs raw-descent τ ≈ 1e-5…1e-2 — the point of the metric).

Status (2026-07-03, constraints M1): `crossing-length.json` and
`linked-rings-length.json` generated with all embedded property checks green
on both (τ = 1 accepted, energy decreases, length drift 7.5e-11 / 1.6e-16).
The five stage-1 goldens are untouched.

Status (2026-07-03, constraints M2): `crossing-edgelengths.json`,
`junction-y-edgelengths.json`, `crossing-point.json`,
`linked-rings-point.json` generated with all embedded property checks green
(τ = 1 accepted on all four; max per-edge drift 2.2e-16 / 1.1e-15; pin
distance 2.3e-22 / 1.4e-24). The `length`-mode output was verified
semantically identical to the committed M1 goldens after the M2 script
extension. Stage-1 and M1 goldens are untouched.

Status (2026-07-03, solver-perf Task 6 / tolerance provenance): **all six
M1/M2 goldens regenerated** at the reference projection tolerance 1e-4 (see
"Projection tolerance provenance" below) — τ unchanged on every fixture,
projection iteration counts drop (looser stop), measured drift still
1e-12…1e-7. The five STAGE-1 goldens are untouched (frozen 1e-10-era
contract; their TS tests pin the tolerance explicitly).

## Projection tolerance provenance (READ BEFORE TOUCHING ANY TOLERANCE)

The projection stopping tolerance is **1e-4** (`tolAbs = tolRel = 1e-4` in the
per-block rule ‖Φ_b‖₂ ≤ max(tolAbs, tolRel·scale_b)). The VALUE is the
authors' reference implementation's `backproj_threshold = 1e-4`
(github.com/ythea/repulsive-curves — formerly icethrush —
`src/tpe_flow_sc.cpp:15`); the per-block scaled RULE is ours (spec §3.3).
Reference-code facts, verified 2026-07-03 at file:line level:

- The reference dense path assembles + LU-factors ONE saddle matrix
  [[Ā, Cᵀ], [C, 0]] per time step, at the pre-step curve, and reuses that
  factorization for the gradient solve AND every projection Newton iterate
  (`ProjectGradient` + `LSBackproject`, `tpe_flow_sc.cpp`) — the paper's
  line-734 remark describes exactly this.
- Their projection allows at most **3** Newton iterates per attempt
  (`tpe_flow_sc.cpp:306`); the fallback on non-convergence is HALVING τ with
  the same LU, never refactorizing.
- Their loop always performs ≥1 solve before measuring (solve-then-check);
  ours checks before correcting, so our "0 iterations" ≈ their "1 Newton
  step". Don't compare iteration counts across the two naively.

History: the tolerance was originally 1e-10 (our invention, audit item 9).
Gating the frozen mode at ≤3 iterations *at 1e-10* produced a **false kill**
on 2026-07-03 (table below) — a quasi-Newton projection converges linearly,
so demanding 6 extra digits costs ~2× the iterations. The stage-1 oracle
(`tpe_stage1_oracle.py`, read-only deliverable) keeps its baked-in 1e-10; its
five goldens are a frozen 1e-10-era contract and the TS stage-1 golden tests
pin `{tolAbs: 1e-10, tolRel: 1e-10}` explicitly. Everything else (TS defaults,
this oracle, M1/M2 goldens — regenerated 2026-07-03) runs the reference value.
Post-step drift gates now TRACK the stopping rule (≤ 1e-4·scale_b) instead of
the old fixed 1e-8; measured actual drift stays far below the bound
(1e-12…1e-7 on the six goldens) because the last Newton correction overshoots.

## Frozen-projection mode (solver-perf Task 6 — UN-KILLED at reference tolerance, ported to TS)

`tpe_constraints_oracle.py` accepts an optional trailing `projection` argument,
`reassemble` (default) | `frozen`:

```bash
for f in "crossing length" "linked-rings length" "knot length" \
         "crossing edgelengths" "junction-y edgelengths" "crossing point"; do
  set -- $f
  uv run --with numpy --with scipy python oracle/tpe_constraints_oracle.py \
    oracle/fixtures/$1.json oracle/golden/$1-$2-frozen.json $2 frozen
done
```

Frozen semantics (plan-locked): factor K(γ₀) = [[Ā(γ₀), C(γ₀)ᵀ], [C(γ₀), 0]]
ONCE via `scipy.linalg.lu_factor`; the gradient solve consumes it; every
projection iterate of every τ-trial solves `lu_solve(fac, [0; −Φ(γ^q)])` with
Φ evaluated FRESH at the current iterate (quasi-Newton: frozen metric + frozen
Jacobian, live residual) — the reference implementation's scheme (see the
provenance section above). Stopping rule, tolerances, max_iter = 8, and failure
semantics are identical to the reassemble path; frozen mode adds one property
gate, `projection_iterations ≤ 3` on the accepted step (the reference-impl
hard cap).

**Status (2026-07-03, revised same day): the earlier KILL was a tolerance
artifact; at the reference tolerance every gate passes and the TS port is
live** (`projectionMode: 'frozen' | 'reassemble'` through
`sobolevStepSet`/store/UI, default 'reassemble' at the function level, store
default 'frozen'; goldens `golden/*-frozen.json`, tests
`test/sobolev/frozenProjection.test.ts` — τ/iterations exact, vertices ~1e-16).

A/B at the REFERENCE tolerance 1e-4 (both modes, accepted step):

| fixture-mode | reassemble τ / iters | frozen τ / iters | gate ≤ 3 |
|---|---|---|---|
| crossing-length | 1.0 / 1 | 1.0 / 1 | pass |
| linked-rings-length | 1.0 / 2 | 1.0 / 2 | pass |
| knot-length | 0.25 / 1 | 0.25 / 1 | pass |
| crossing-edgelengths | 1.0 / 1 | 1.0 / 1 | pass |
| junction-y-edgelengths | 1.0 / 2 | **0.5** / 3 | pass (**τ regression**) |
| crossing-point | 1.0 / 2 | 1.0 / 2 | pass |

The one standing trade-off: on `junction-y-edgelengths` the frozen τ=1 trial's
projection does not converge (stale junction Jacobian) and the step backtracks
to τ = 0.5 — the reference implementation behaves the same way by
construction (3-iterate cap + τ-halving fallback). Bench
(`bench/results/2026-07-03-frozen-reuse.json`, closed trefoil): frozen ≈
**1.9× full-step** vs reassemble at the same commit (N120-total 89.2→47.4 ms;
N120-perEdge 135.8→71.2 ms); per step, assemblies+factorizations drop 2→1 and
the projection phase collapses 33.6→0.6 ms at N120-total.

Historical (2026-07-03, PRE-provenance, tolerance 1e-10 — the false kill;
kept so nobody re-runs this experiment at the wrong tolerance):

| fixture-mode | reassemble τ / iters | frozen τ / iters | gate ≤ 3 |
|---|---|---|---|
| crossing-length | 1.0 / 1 | 1.0 / 3 | pass |
| linked-rings-length | 1.0 / 3 | 1.0 / **5** | **FAIL (kill)** |
| knot-length | 0.25 / 2 | 0.25 / **4** | **FAIL (kill)** |
| crossing-edgelengths | 1.0 / 2 | 1.0 / 3 | pass |
| junction-y-edgelengths | 1.0 / 3 | **0.25** / **6** | **FAIL** (+ τ regression) |
| crossing-point | 1.0 / 2 | 1.0 / **7** | **FAIL** |

@see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 6)

## Penalties / soft constraints (5C)

`tpe_constraints_oracle.py` accepts an optional trailing penalty-preset
argument (after the projection argument) and a new constraint mode `bary`
(barycenter-only set — the soft-constraint flow configuration). Penalties are
the paper's catalog (SelfAvoiding.tex lines 762–767): total length Σℓ_I,
length difference Σ_{V_int}(ℓ_I−ℓ_J)², field alignment Σℓ_I|T_I×X|² with a
constant unit X. They enter the OBJECTIVE only — analytic gradients added to
dE before the saddle solve, energies added to the Armijo gate — never the
constraint rows; H^s inner product unchanged (tex line 769). Formulas +
decision ledger: `docs/superpowers/plans/2026-07-03-sobolev-penalties.md`.

```bash
for f in "crossing bary pen-length" "helix bary pen-field" \
         "junction-y bary pen-diff" "crossing length pen-combo"; do
  set -- $f
  uv run --with numpy --with scipy python oracle/tpe_constraints_oracle.py \
    oracle/fixtures/$1.json oracle/golden/$1-$2-$3.json $2 reassemble $3
done
```

Embedded property gates (beyond the standard ones): central-FD check of the
analytic penalty gradient (rel ≤ 1e-6; measured 3e-10…5e-10), penalty
actually perturbs dE, energy decrease in the TOTAL objective. Preset weights
are OUR knobs (recorded in each golden's `penalties` field). The field preset
uses X = [1,0,1]/√2 and the helix fixture DELIBERATELY: with X orthogonal to
a planar fixture's tangents the field penalty degenerates to the length
penalty (|T×X|² ≡ 1) and its (T·X) terms would be untested. Without a preset
the penalty code is inert — verified byte-identical regeneration of
`crossing-length.json` (2026-07-03). Status (2026-07-03): all four penalty
goldens generated, every property check passing (pen-combo composes hard
totalLength + all three penalties; accepts at τ=0.25 — soft length pulling
against the hard constraint is expected to backtrack once).

Formula audit (2026-07-02): an isolated auditor diffed the results doc against
verbatim paper excerpts — 11/11 items CONFIRMED, zero mismatches; report at
`local_files/2026-07-02-sobolev-formula-audit.md`. Two standing cautions from it:
(1) the paper's appendix scaling paragraph (SelfAvoiding.tex line 849) has an
internal typo (2s+1 vs 2σ+1) — do NOT "correct" the c^(−7/3) scaling test to
c^(−13/3) on its strength; (2) the line-search/projection constants (c₁=1e-4,
ρ=1/2, τ_min=1e-12, the mass-lumped L²ₕ norm, tol max(1e-10, 1e-10·L·R),
max_iter=8) are our tunable choices, not paper constants.

Known tolerance caveats (measured, not guessed):

- Analytical-vs-FD dE gates at 1e-4 (norm-rel) for h=1e-6 forward differences:
  the gap is O(h) truncation error (verified by h-scaling in
  `compare_energy.ts` header), not implementation error.
- Cross-language FD-vs-FD dE is capped near ~1e-9 rel by (energy roundoff)/h;
  don't tighten that gate below 1e-6 without switching to central differences.
