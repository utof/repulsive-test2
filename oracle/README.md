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
