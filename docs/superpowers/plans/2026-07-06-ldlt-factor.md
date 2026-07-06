# LDLᵀ (Bunch–Kaufman) factor swap for the saddle solve — 2026-07-06

Branch `feat/ldlt-factor` off main @ a316ff9. Pre-registered in
`docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md` step 5.6 and
re-opened by the user per `local_files/2026-07-04-next-steps-after-worker.md`
§3 item 2. This is the milestone's ONE planning doc.

## What and why

The saddle matrix `K = [[Ā, Cᵀ], [C, 0]]` (linsolve.ts `buildSaddleMatrix` /
`solveSaddleFromA`) is symmetric **indefinite** — the k×k lower-right block is
exactly 0, so Cholesky is impossible (Golub & Van Loan, *Matrix Computations*
§4.4; also the standing anchor in `src/core/sobolev/linsolve.ts` header: "Do
NOT use Cholesky on the saddle matrix"). The sanctioned dense factorization for
symmetric indefinite systems is LDLᵀ with Bunch–Kaufman partial pivoting
(1×1/2×2 pivots) — ~n³/6 multiply-adds vs LU's ~n³/3, a theoretical ~2× on the
FACTOR phase. LU with partial pivoting remains the spec's dense fallback
(`local_files/2026-07-02-sobolev-gradient-rsrch-results.md` §B); this milestone
is a measure-first A/B, not a correctness fix.

**Measured prize to beat** (briefing §3 item 2): ~1.35×@N120-total /
~1.55×@N120-perEdge.

## Kill gates (binding, pre-registered — perf plan step 5.6)

- ANY golden-gated test failure ⇒ KILL.
- <1.3× FACTOR-phase win (factor timing p50, not total step) at N=120 ⇒ KILL.
- The self-certifying residual `‖K·z − rhs‖₂ / max(1, ‖rhs‖₂)` (linsolve.ts
  header anchor — "computed on EVERY solve, never skipped") must survive in
  BOTH paths. Removing/weakening it in either path ⇒ automatic kill of the
  change, no measurement needed.
- On kill: stop implementing, commit this doc + a kill-note with the measured
  numbers. A committed negative result is a success.

## Pinned decisions (made — recorded here, not to be reopened)

### 1. Algorithm: LAPACK `dsytf2`-style unblocked Bunch–Kaufman, lower triangle

Verified against the reference source this session (2026-07-06,
<https://raw.githubusercontent.com/Reference-LAPACK/lapack/master/SRC/dsytf2.f>
UPLO='L' branch, and `dsytrs.f` for the solve). The details we implement
verbatim (0-based translation):

- `alpha = (1 + sqrt(17)) / 8` (≈0.6404), the Bunch–Kaufman growth-optimal
  threshold.
- At column `k`: `absakk = |A(k,k)|`; `imax` = argmax over `|A(i,k)|`, i>k;
  `colmax = |A(imax,k)|`.
- `max(absakk, colmax) == 0` ⇒ the column is exactly zero ⇒ singular. LAPACK
  sets INFO and continues; **we throw instead** (see decision 3).
- Pivot choice (three-way, exact LAPACK inequality forms):
  1. `absakk >= alpha*colmax` ⇒ 1×1 pivot, no interchange.
  2. else compute `rowmax` = max |off-diagonal| in row/col `imax` of the
     trailing submatrix (row part `A(imax, k..imax-1)`, column part
     `A(imax+1.., imax)`); if `absakk >= alpha*colmax*(colmax/rowmax)` ⇒ 1×1,
     no interchange.
  3. else if `|A(imax,imax)| >= alpha*rowmax` ⇒ 1×1 pivot, interchange
     `k ↔ imax`.
  4. else ⇒ 2×2 pivot, interchange `k+1 ↔ imax` (`kstep = 2`).
- Interchanges are SYMMETRIC and touch only the trailing submatrix
  (columns < k of packed L are NOT re-permuted — that is why the solve applies
  interchanges interleaved, not as an up-front `P·rhs` like LU).
- 1×1 elimination: `d11 = 1/A(k,k)`; trailing lower-triangle rank-1 update
  `A(i,j) -= A(i,k) * (d11*A(j,k))` for `k < j <= i < n`; then scale column
  `A(i,k) *= d11`.
- 2×2 elimination (verbatim dsytf2 'L', k < n-1):
  `d21 = A(k+1,k); d11 = A(k+1,k+1)/d21; d22 = A(k,k)/d21;
  t = 1/(d11*d22 − 1); d21 = t/d21;` then per trailing column `j >= k+2`:
  `wk = d21*(d11*A(j,k) − A(j,k+1)); wkp1 = d21*(d22*A(j,k+1) − A(j,k))`;
  update `A(i,j) -= A(i,k)*wk + A(i,k+1)*wkp1` for `i >= j`; store
  `A(j,k) = wk; A(j,k+1) = wkp1`.
  (`d21 ≠ 0` is guaranteed: after the interchange `|A(k+1,k)| = colmax > 0`.)
- `ipiv` keeps LAPACK's **1-based** encoding in an `Int32Array` so the sign
  test works at index 0: `ipiv[k] = kp+1` for a 1×1 pivot (row/col `k ↔ kp`
  interchanged), `ipiv[k] = ipiv[k+1] = −(kp+1)` for a 2×2 pivot (row/col
  `k+1 ↔ kp` interchanged).
- Solve = `dsytrs` 'L', single RHS: forward sweep (interchange, apply inv(L(k))
  columns, divide by the 1×1/2×2 D block — 2×2 via the exact dsytrs formula
  `akm1k = A(k+1,k); akm1 = A(k,k)/akm1k; ak = A(k+1,k+1)/akm1k;
  denom = akm1*ak − 1; bkm1 = b(k)/akm1k; bk = b(k+1)/akm1k;
  b(k) = (ak*bkm1 − bk)/denom; b(k+1) = (akm1*bk − bkm1)/denom`), then backward
  inv(Lᵀ) sweep with the interchanges re-applied in reverse.

**Storage/layout**: operates IN PLACE on the same flat row-major
`Float64Array` buffer `solveSaddleFromA` assembles (exactly `luFactor`'s
convention, including buffer consumption). Only the lower triangle is read and
written; the upper triangle becomes dead storage after factorization (the
buffer is consumed either way). LAPACK is column-major so its cache-friendly
inner loops are column-wise; ours is row-major, so the trailing updates are
restructured row-wise with the per-column coefficients (`d11*A(j,k)`, `wk`,
`wkp1`) precomputed into O(n) scratch arrays. This is numerically EXACT w.r.t.
dsytf2: each trailing element receives exactly one fused update expression per
elimination step, and that expression (operand order included) is LAPACK's —
only the order in which distinct elements are visited changes, which cannot
change any element's value.

**Bit-identity note**: the LDLᵀ path needs NO bit-relationship to the LU path
— it is gated by the self-certifying residual (≤1e-10) and by tolerance
cross-checks against LU. Only the DEFAULT path must stay bit-identical while
`factorMode` defaults to `'lu'`.

### 2. Reference path stays LU (cross-check oracle)

The `number[][]` reference path (`luSolve`/`solveSaddle`) is NOT touched. It
remains the golden-tested reference and serves as the A/B cross-check oracle
in the new tests (LDLᵀ solves are compared against `luSolve`/`solveSaddle`
results). Only the typed-array fast path (`solveSaddleFromA` + factored
solves) gains the LDLᵀ option.

### 3. Breakdown contract — mirror the existing singular contract exactly

Verified behavior of the current path: `luFactor`/`luSolveFactored` throw
`Error` messages containing "singular" (zero pivot column) or "treating as
singular" (non-finite input, scanned up front). Callers:

- `sobolevStepSet` catches ANY solve throw and folds it into the
  `reason: 'singular_system'` rejection — the frame loop never sees an
  exception (src/core/optimizer.ts:316–337).
- `projectOntoConstraintSet` catches and returns `ok: false` with the current
  iterate (src/core/sobolev/lineSearch.ts:276–278).

`ldltFactor`/`ldltSolveFactored` therefore: (a) scan for non-finite entries up
front and throw "… treating as singular" exactly like `luFactor`; (b) throw
"… singular (pivot column k …)" when `max(absakk, colmax) == 0` — where LAPACK
would set INFO>0 and emit an exactly-zero D(k,k), we throw, because a zero
D block makes the subsequent solve divide by zero and the existing contract
is throw-and-catch, never NaN propagation. NO new throw type reaches the frame
loop.

### 4. Selection: `factorMode: 'lu' | 'ldlt'`, default `'lu'` until the gates pass

Threaded like `projectionMode` (bench-dimension A/B):

- `SobolevStepOptions.factorMode` (optimizer.ts) → passed to
  `solveConstrainedGradientSetFrozen` (gradient.ts) → `solveSaddleFromA`
  (linsolve.ts, new optional trailing parameter).
- `ProjectConstraintSetOptions.factorMode` (lineSearch.ts) for the
  per-iterate REASSEMBLE projection path; the FROZEN projection path needs no
  option — `FrozenSaddleOperator.fac` widens to
  `LuFactorization | LdltFactorization` and `solveSaddleFrozen` dispatches on
  the factorization kind.
- `bench/sobolev.bench.ts`: factorMode becomes a case dimension; `'lu'` cases
  keep their historical names (baseline Δ% joins keep working), `'ldlt'` cases
  get a `-ldlt` suffix.
- Deviation from the projectionMode precedent, recorded: NO store/UI toggle.
  projectionMode is a user-facing semantic choice (stale-Jacobian quasi-Newton
  vs strict); factorMode is not — both factorizations solve the same system to
  ≤1e-10 certified residual, so once gated there is nothing for a user to
  choose. The app/worker pick up LDLᵀ automatically via the default flip; no
  dispatch/store/UI file changes at any point in this milestone.
- Default REMAINS `'lu'` in every intermediate commit (each bit-identical by
  default); the FLIP-to-default commit is separate and LAST, and only happens
  if the gates pass.

### 5. Verification ladder

(a) **Unit** (`test/sobolev/ldlt.test.ts`): deterministic seeded random
symmetric indefinite matrices (mulberry32-style seeded PRNG in the TEST file
only — runtime paths stay Math.random-free, matching the repo's deterministic
fixture convention, e.g. bench trefoil / linsolveFlat's trig-formula systems):
reconstruct `A = P(1)L(1)…P(s)L(s)·D·(…)ᵀ` from the packed factors + ipiv
(LAPACK dsytrf's documented product form) to rel ~1e-12 against the original
matrix, and cross-check `ldltSolveFactored` against `luSolve` on the same
systems. Include: a matrix forcing 2×2 pivots (zero diagonal block, the saddle
shape), the real crossing-fixture saddle system vs `solveSaddle`, the
zero-column singular throw, the non-finite throws, and dev-time (not
committed) scipy `ldl`/`sytrf` cross-check of ipiv via
`uv run --with numpy --with scipy`.

(b) **Property** (same test file): on ALL committed fixtures
(`oracle/fixtures/*.json` — crossing, helix, junction-y, knot, linked-rings),
`sobolevStepSet` with `factorMode:'ldlt'` vs `'lu'`: both residuals ≤1e-10
(self-certification), step vertices rel ≤1e-9, accepted/converged verdicts
equal. Goldens themselves unchanged; the default path is untouched (existing
232+ tests are the bit-identity backstop).

(c) **Bench**: `bun bench/sobolev.bench.ts --baseline
bench/results/2026-07-03-frozen-reuse.json --save ldlt-ab` with factorMode as
a case dimension; gate on FACTOR-phase p50, LU vs LDLᵀ, at N=60/120-total and
N=120-perEdge minimum. Machine noise up to 5× between isolated runs is
documented (repo session logs; JIT/thermal) — use per-phase medians, repeat
runs, and report the spread.

## Commit plan

1. This doc (own commit, first).
2. TDD: failing tests, then `ldltFactor`/`ldltSolveFactored` (~250 lines) +
   factorMode threading. Default path bit-identical; `bun test` all green;
   `bunx tsc --noEmit` clean.
3. Bench A/B + gate verdict recorded below (amended into this doc's commit or
   a final commit). EITHER the default-flip commit (gates pass) OR a kill-note
   here (gates fail). No push, no PR.

## A/B results & gate verdict

_To be filled at step 3._
