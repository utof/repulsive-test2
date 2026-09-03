# WebGPU Solver — Phase 2 Design Spec (the solve)

**Status:** DRAFT 2026-09-03, awaiting blind review. Child of
`docs/superpowers/specs/2026-08-13-webgpu-solver-design.md` (the parent spec —
cited as **[P §n]**); everything the parent decides stays decided unless a
section here says **AMENDS [P §n]** and gives the reason. Fresh-session entry
point: read §0, then [P §2.4, §2.5, §4 G5, §5].

> **This document owns decisions. Every gate NUMBER lives in
> `docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-gate-calibration.md`**
> — the gate-calibration note, cited below as **[CAL §n]**. The split exists
> because three independent blind review rounds each found a different
> instance of one defect: gate arithmetic hand-transcribed into design prose,
> where nothing re-checks it. The rule that replaces the transcriptions is
> binding on both documents:
>
> **No gate number appears in prose unless a committed script emitted it into
> a committed JSON.** This spec cites `[K1CAL]` / `[JSALG]` / `[BASE]` keys and
> [CAL] sections; it does not restate values. Where the two documents disagree
> about a *number*, [CAL] wins; where they disagree about a *decision*, this
> document wins and [CAL] is wrong.

**Goal of Phase 2 [P §5]:** move the saddle solve — the O(N³) factorization
that is 85 % of a descent step at N=1000 — onto the GPU, keep every number
the CPU path produces verifiable against it, and re-measure the full-step
speedup honestly. Positions stay CPU-authoritative [P §2.8]; the line search
moves last.

**Evidence base (measured on this machine, hardware adapter):**
- `bench/results/2026-08-13-cpu-baseline-largeN.json` — CPU f64 per-phase step
  baselines N=240…1000 (Bun). Cited as **[BASE]**.
- `bench/gpu/k1-calibration.py` → `bench/results/2026-09-04-gpu-phase2a-k1-calibration.json`
  — the gate-calibration run this spec's K1 bars, boot bar and precision model
  are derived from: α, β (five perturbation directions per N), κ₂(A_σ), κ₂(K),
  r₀/ρ per rhs shape, at trefoil N=64/240/480/960. CPU-only numpy/LAPACK;
  re-run with `uv run --with numpy --with scipy python bench/gpu/k1-calibration.py`.
  Cited as **[K1CAL]**.
- `bench/gpu/jsAlgebra.ts` → `bench/results/2026-09-04-js-f64-algebra.json` —
  the JS f64 throughput of the three loops slice 2a leaves on the CPU
  (N-backsolve, N×N matvec, A_σ build + f32 cast) at N=480/960/1000, each a
  median-of-15 over 30 sessions spanning five processes and two allocation
  regimes. The committed run records `loadOk: false` (1-minute loadavg 6.10 →
  5.85, unrelated concurrent work on this box), so its **upper ends are
  over-estimates and its lower ends are the real rate** — see §5's "read the
  upper end as an over-estimate". Re-run alone with `bun bench/gpu/jsAlgebra.ts`.
  Cited as **[JSALG]**.
- `bench/gpu/README.md` "Phase 0 gate report" + "Phase 1 gate report":
  G0t throughput probe (`fmaGflops≈1505`, `matvecMs≈0.463` at 3072²), G1
  batching (250 dispatches batched ≈ 0.3 ms), G6 conditioning sweep
  (κ(K) total mode ∝ N^3.285: 1.405e5 at N=480, 1.418e6 at N=960; perEdge
  κ·u_f32 > 0.1 at N ≥ 480 → CPU only), Phase 1 dE gate (GPU dE p50 3.2 ms /
  5.1 ms at N=480/960, readback included). Cited as **[G0t] [G1] [G6] [P1]**.
- `docs/2026-08-13-ai-research-gpu-precision.md` Q2 (LU-IR converges while
  κ·u_f ≲ 1; 1–3 refinements to 1e-10 at N≤128), Q3 (compensated arithmetic
  is NOT portable in WGSL — §15.7.5 permits reassociation; df64 ≈ 4× cost).
  Cited as **[PREC Qn]**.
- `docs/2026-07-04-ai-research-multigrid.md` §3 (projected saddle forms).

---

## 0. State detection (fresh session)

- `src/gpu/cholesky.ts` absent → Phase 2 not started (this spec + plan 2a only).
- `src/core/sobolev/linsolve.ts` has no `'external'` factorization kind → the
  core seam (§4.4) has not landed.
- `bench/results/*-gpu-phase2a-*.json` present → slice 2a gates ran; read the
  README "Phase 2a gate report" before touching anything.
- Slices are sequential branches: 2a (this plan) → 2b → 2c (§1).

## 1. Scope

Phase 2 is three slices; each is a branch + plan; this spec covers the
architecture and gates of all three, and the accompanying plan covers **2a
only**. **AMENDS [P §2.5, §5 Phase 2]** — the parent decides twice, and
emphatically, that these are not separable: "line search and solve are one
inseparable GPU stage" ([P §2.5]) and "Phase 2 — the solve + frozen-mode GPU
line search (one inseparable stage, §2.4/§2.5)" ([P §5]). This spec splits them
across three sequential merges, and that is the single largest departure from
the parent in this document. Reason: `sobolevStepSet` and everything under it
is synchronous (`src/core/optimizer.ts:246-470`), and WebGPU has no synchronous
readback. A GPU result can enter the CPU step only if it is computed BEFORE the
step starts (the Phase 1 `dE?` seam pattern) — which the factorization can be,
and the line search's per-trial solves cannot, since each trial's rhs depends
on the previous trial's readback. Consequence the parent will care about:
[P §2.4]'s rule that "a variant that wins the gradient solve but loses the line
search's projections has not won" is **not** discharged by 2a; it is discharged
by 2c's G5 full-step re-measure (§4.7). Until 2c lands, the line search's
projection solves are served by the same read-back factor on the CPU (§4.1) and
are measured inside P2a (§5) — won on the CPU, not skipped.

| Slice | Moves to GPU | Stays CPU f64 | Gates |
|---|---|---|---|
| **2a — factor** | Cholesky of the N×N Sobolev block (§4.1, §4.3), consumed through a new core seam (§4.4) | A assembly, all backsolves (on the read-back factor), iterative refinement, projection, line search, all energies | CR, K1, T4, T5, `G5-confirm`, P2a, boot self-test (§5) |
| **2b — assembly** | A assembly (gather kernels, §4.6) and the per-step upload it removes | backsolves, IR (on the promoted f32 A), line search, energies | T4a + T4 re-based on the promoted K (§4.6) |
| **2c — line search** | fixed-count projection loop + Armijo with GPU energies, GPU backsolves against the resident factor (§4.7); async step orchestrator in `src/gpu/` | position authority; f64 energy at pause/commit | τ-agreement, T5, G5 full-step re-measure [P §2.5] |

**In scope (2a):** `src/gpu/cholesky.ts` (kernel + engine), `src/gpu/saddle.ts`
(CPU-side Schur/Woodbury/IR on the read-back factor), the core seam (§4.4),
`GpuDriver.step` extension, boot self-test extension, the seven gates of §5 and
their results JSONs, README gate report. **Ordering constraint (§5's
`G5-confirm` row):** the §4.4 core seam must land BEFORE `G5-confirm`, because
that gate's statistic is a whole-descent-step p50 for variant (a) and the seam
is the only route an external factor has into `sobolevStepSet`. "Before any
driver / store / self-test integration" constrains `GpuDriver.step` routing,
the store bit and the boot self-test — not the seam.

**Out of scope (explicit non-goals):**
- Any change to the CPU numerics: `assembleAFlat`, `ldltFactor`,
  `lineSearchStepSet`, `projectOntoConstraintSet` are byte-for-byte unchanged
  [P §1]. The seam is source-selection only (§4.4); absent ⇒ bit-identical
  (golden suites are the backstop).
- `perEdge` constraints on the GPU solve: k = |E|+3 constraint rows make the
  Schur block O(N³) and [G6] already routes perEdge N ≥ 480 to CPU f64. The
  GPU solve serves **barycenter + totalLength (+ pins)** only; perEdge steps
  run the Phase 1 path (GPU dE, CPU factor).
- `'reassemble'` projection mode [P §2.5]: it re-factors per trial by design;
  stays CPU permanently.
- Multigrid / BCT preconditioning [P §1]; Bunch–Kaufman on the GPU (§3);
  df64 emulation [PREC Q3]; f16.
- Phase 3 items: position-authority flip, zero-readback rendering, G7.

## 2. Where a step's time goes (why the factor first)

**Numbers: [CAL §B].** This section states the structure; the [BASE] phase
table, the `R₂` definition and its measured values, the `s_p` bracket, the
`R < 0` falsification at N=1000 and the Amdahl ceilings are all there.

**The phase keys are NOT a partition.** `src/core/sobolev/phaseTimings.ts:1-7`
says so in its own schema note ("Sub-phases … OVERLAP their parents — sums
across keys double-count by design"). Verified nesting at HEAD:
`step ⊃ {dE, assembleA, saddle(gradient), lineSearch}`;
`lineSearch ⊃ {energy, projection}` (`optimizer.ts:399`, `lineSearch.ts:447,
513`); `projection ⊃ saddle(projection solve)` (`lineSearch.ts:260`);
`saddle ⊃ factor` (`gradient.ts:106` → `linsolve.ts:796`). Any speedup
arithmetic done on the raw keys double-counts `projection` and silently drops
`saddle − factor`; the first draft of this spec did exactly that, which is why
its P2a predictions came out ABOVE the Amdahl ceiling. [CAL §B] carries the
corrected identity, whose only aggregate 2a needs is

> **`R₂ = step − dE − assembleA − saddle − energy`** — all the CPU work of a
> step that is neither dE, nor assembly, nor the solve, nor the energy.

`R₂` is carried in every prediction even though it may be measurement slack
rather than work, because carrying it makes the 2a prediction pessimistic
rather than flattering. **Whether it is work or slack is a pre-registered
precondition on reading P2a, not an open question** (§5, §8).

**What slice 2a actually moves.** Exactly one phase leaves the CPU: `factor`
(dE left in Phase 1). The rest of `saddle` — building the (3N+k)² K (66.5 MB at
N=960), the dense backsolve, the structured residual — is not *kept* either: it
is **replaced** by the §4.1 Schur/Woodbury algebra on the read-back N×N factor,
priced term by term in [CAL §H.2]. Everything else (`assembleA`, `energy`,
`R₂`) stays byte-for-byte (§1). So 2a has an **Amdahl ceiling**
`step / (assembleA + energy + R₂ + algebra)`, evaluated in [CAL §B]. **No bar in
§5 sits above it**, and that invariant — not any particular ceiling value — is
what this spec asserts. Because the new Schur algebra is cheaper than the
in-`saddle` backsolves it replaces, the cruder reading
`step/(step − factor − dE)` now *understates* rather than agrees; [CAL §B]'s
table, not that shortcut, is the arithmetic of record.

Phase 1 took dE to 3–5 ms [P1]. The factor is O(n³) on n = 3N+k because
`solveSaddleFromA` (`linsolve.ts:742-803`) expands the N×N Sobolev matrix
`A` block-diagonally into K = [[I₃⊗A, Cᵀ],[C, 0]] (coordinate-major, §4.1)
and factors ALL of K with Bunch–Kaufman LDLᵀ (`ldltFactor`,
`linsolve.ts:431-589`). Two structural facts drive this design: (i) the
expensive object is `I₃⊗A` — ONE N×N matrix used three times; factoring N×N
instead of (3N+k)×(3N+k) is 27× fewer flops; (ii) A is symmetric positive
semidefinite with `null(A) = span(𝟙)`, **exactly one-dimensional**, so a
*shifted* A is SPD and Cholesky-factorable without pivoting — the only dense
factorization that is straightforward on a GPU.

The reason for (ii) is NOT "every entry is a difference stencil"
(`assembleBHighFlat`, `innerProduct.ts:38-140`; `assembleBLowFlat`, `:177-295`):
a difference stencil alone gives one constant PER CONNECTED COMPONENT, and two
shipped presets are disconnected — `crossing` (2 components,
`createCrossingLines`, `src/core/testConfigs.ts:17-38`), which is CR's and T4's
own first fixture, and `linked-rings` (`createLinkedRings`, `:57-78`). The dimension is 1 anyway, because the disjoint-pair
coupling in `assembleBHighFlat` links the components. Measured: dim null(A) = 1
with λ₂(A) = 3.236 (`crossing`, N=8) and 4.887 (`linked-rings`, N=32), A_σ SPD
in both, f64 and f32 Cholesky both succeeding; ‖A𝟙‖/(‖A‖_F√N) ≈ 6.1e-17 and
λ₂/λ_max = 7.15e-4 … 7.18e-7 at N=120…960 — no second near-null direction
anywhere in range. This is load-bearing: the shift of §4.1 step 1 is rank-1, so
on a genuinely multi-component A it would NOT restore definiteness and every
step would pay a full upload + factor + readback only to hit the status flag.

## 3. G5 — the solve-path decision [P §4 G5], executed here

**Numbers: [CAL §D].** The (a) factor-wall estimate, the measured CG iteration
counts, variant (b)'s pricing and both arms' bands live there. This section
states the decision and what licenses it.

[P §4 G5] pre-registers: variant (b) (f32 GPU CG + f64 IR) is measured,
variant (a) (GPU factorization) is estimated from [G0t] + the dispatch model,
and (b) is chosen unless (a)'s estimate wins by >1.5× at N=960, in which
case a minimal GPU-factorization spike, T4-checked, confirms before commit.

**The statistic is the parent's, and both arms are priced on it.** [P §4 G5]
says "p50 full-**STEP** time (gradient solve + line-search projection solves
per §2.4, readbacks included)", and [P §2.4] restates it ("a variant that wins
the gradient solve but loses the line search's projections has not won"). The
first draft priced (a) on the `factor` phase alone and (b) on a whole step —
two different statistics, inflating the margin ~30×. In [CAL §D] both arms are
priced on the parent's solve-path-per-step cost, and the whole-descent-step
reading is given too, since the parent's words admit both.

**(b) is a measured-model arm, not a built one, and not an estimated one.**
Its dominant unknown is the iteration count, and that count is *measured* on
the real shifted systems the repo's oracle assembles ([K1CAL] `rows[].cgIters`,
five rhs directions per N, plain CG and Jacobi-PCG) rather than bounded by the
textbook `½√κ·ln(2/tol)` figure the first draft used. Two structural facts
survive from that measurement: CG needs the same shift + Schur reformulation as
(a) (§4.1), so (b) is **not** cheaper to build; and Jacobi cannot change the
h-dependence of a fractional-Laplacian-class operator, so no preconditioner in
scope closes the gap (the one that would — MG, issue #5 — is out of scope
[P §1]). **Variant (b) is a viable fallback, not the dead end the first draft
asserted** ("≤ 3×", computed from the textbook bound) — §6's K1-red branch
depended on that assertion and is corrected there.

**Provisional decision: variant (a) — confirmed or overturned by the
`G5-confirm` gate (§5), not by this paragraph.** [CAL §D.3] gives the margin on
both readings. The shape of it is what matters here:

- On the **solve-path** statistic the parent's >1.5× rule fires at **every
  point of both bands**, pessimistic corner included.
- On the strictest reading — the **whole descent step**, where both variants
  carry the same unmoved `assembleA + energy + R₂` (§2) — the rule fires at the
  central estimate but **NOT at the pessimistic corner**.

That surviving corner is ~30× weaker than the first draft's "two orders of
magnitude", and it is named here so the AMENDS below is judged on the real
margin. Two facts **argue** for (a) at that corner — the Amdahl compression
that produces it is removed by 2b/2c (once assembly and the energies are
GPU-side the unmoved term → ≈0 and the ratio returns to the solve-path number),
and (b) re-pays a full inner solve on every line-search projection of every
step forever, where (a) pays the factor once per step and backsolves
thereafter. **Neither settles it.** Both are arguments about a quantity that is
still *estimated* on the (a) side, and [P §4 G5]'s bias policy is explicitly
asymmetric against exactly that move
(`2026-08-13-webgpu-solver-design.md:355-357`: "measured beats estimated — (b)
is chosen unless (a)'s estimate wins by >1.5× at N=960, in which case a minimal
GPU-factorization spike … is budgeted to confirm the estimate BEFORE Phase 2
commits to (a)"). Settling a corner the parent's own rule declines to settle,
by argument, is the one move that policy forbids. So it is not settled here:
**§5's `G5-confirm` gate re-decides G5 on the measurement**, after the Cholesky
spike and before any driver integration, with a pre-registered branch back onto
(b). §6 carries the consequence.

**AMENDS [P §4 G5]** — three departures, each named per §0:
1. **(b) is not built.** The parent made (b) *measured* precisely so the
   decision would not be estimate-vs-estimate (review-2 F1). It stays measured
   in the sense that matters — its dominant unknown, the iteration count, is
   measured on the real systems above — but no CG kernel is written.
   Justification: on the parent's own statistic the rule fires across the whole
   solve-path band, and the single corner where it does not fire — the
   whole-step pessimistic one — is precisely where §5's `G5-confirm` gate
   re-decides on the measurement rather than this justification standing in for
   one. Both readings and their corners: [CAL §D.3].
2. **The confirmation spike is substituted.** [P §4 G5] budgets "a single f32
   **blocked-LDLᵀ** of a committed N=480 system, **T4-checked**". Plan 2a's
   Cholesky task is an **unblocked Cholesky** gated by **K1** (a backward-error
   gate) *and* by T4 at N=480, which §6 keeps as the kill. Reason: §4.1
   justifies Cholesky over LDLᵀ (pivot search is a global reduction per column
   on a GPU) and §4.3 justifies unblocked-first; T4 is not dropped — K1 is
   added in front of it because a backward-error number localises a kernel bug
   that a T4 red only reports.
3. **The statistic is restated, not changed** (see above): (a) was priced on
   the `factor` phase in the first draft.

## 4. Architecture (decided; changing any of these requires re-review)

### 4.1 The solve: shifted-block Cholesky + Schur + Woodbury — exact for the original K

Notation: N vertices, A ∈ ℝ^{N×N} the CPU-assembled Sobolev matrix
(`assembleAFlat`), and — matching the layout the code and the oracle actually
use — **Ā = I₃ ⊗ A**, i.e. **coordinate-major**: block b of a 3N vector is
coordinate b over ALL vertices, occupying rows [bN, (b+1)N).
`solveSaddleFromA` writes exactly this (`linsolve.ts:770-779`: `off = b * n`,
`dst = (off + i) * size + off`), and so does the oracle
(`oracle/tpe_stage1_oracle.py:245-248`, `np.block([[A,0,0],[0,A,0],[0,0,A]])`).
(An earlier draft wrote `A ⊗ I₃`, the interleaved vertex-major convention — the
opposite of the code, and a notation an implementer would have to correct
before writing the first test. Only the notation changes; the algebra below is
unchanged and was reproduced independently to 4.6e-11 at N=960.)
C ∈ ℝ^{k×3N} is the constraint Jacobian (`evaluateConstraintSet`),
K = [[Ā, Cᵀ],[C, 0]], rhs r = [b; d]. Let **P = I₃ ⊗ (𝟙_N/√N) ∈ ℝ^{3N×3}** —
an orthonormal basis of null(Ā); column b is 1/√N on rows [bN, (b+1)N) and 0
elsewhere — and U = [P; 0].

1. **Shift.** σ = trace(A)/N. A_σ = A + (σ/N)·𝟙𝟙ᵀ is SPD (the shift is
   positive exactly on null(A); `A_σ(𝟙/√N) = σ(𝟙/√N)`, and λ_min(A_σ) =
   λ₂(A) > 0 at every N measured, including the disconnected `crossing` and
   `linked-rings` presets — null(A) is one-dimensional there too, §2).
   Â = I₃ ⊗ A_σ = Ā + σPPᵀ. **One** N×N Cholesky A_σ = LLᵀ serves all three
   coordinate blocks. (Do NOT confuse with `buildSaddleMatrix`'s "no
   regularizing identity" rule, `linsolve.ts:133-154`: that rule protects the
   METRIC; the shift here is removed exactly in step 3.)
2. **Schur on K̂ = [[Â, Cᵀ],[C, 0]].** For rhs [b; d]: y = Â⁻¹b (three
   N-backsolves); Z = Â⁻¹Cᵀ (3k N-backsolves, once per step — the factor is
   frozen [P §2.5]); S = C Z (k×k SPD, κ(S) = 5.06…5.28 across N=120…960);
   λ = S⁻¹(C y − d); x̂ = y − Zλ. ẑ = [x̂; λ] = K̂⁻¹r.
3. **Woodbury back to K.** K = K̂ − σUUᵀ ⇒
   K⁻¹ = K̂⁻¹ + K̂⁻¹U (σ⁻¹I₃ − UᵀK̂⁻¹U)⁻¹ UᵀK̂⁻¹.
   W = K̂⁻¹U (3 columns, once per step; because Â is block-diagonal and each
   column of P is supported on ONE contiguous N-block, its top block needs
   exactly ONE extra N-backsolve of 𝟙/√N, reused for all three coordinates),
   M = σ⁻¹I₃ − UᵀW (3×3). Then z = ẑ + W M⁻¹ (Pᵀx̂). This is exact algebra;
   the reference gate CR (§5) proves it against `solveSaddleFromA` to a
   κ-relative bar BEFORE any WGSL. **M's FORMATION is heavily cancelling**
   (§7: `diag(M)` is 9.2e4× smaller than σ⁻¹ at N=960) — form and invert it in
   f64, never in f32, and never on the GPU (a hard constraint on §4.7).
4. **Iterative refinement (IR)** [PREC Q2, P §2.4(a)]: z₀ from step 3;
   repeat r_i = r − K z_i (f64, with the f64 A and C), δ = K⁻¹r_i via steps
   2–3 (same factor), z_{i+1} = z_i + δ, until `‖r_i‖₂/‖r‖₂ ≤ 1e-10` or 4
   iterations. **The criterion is the PURE RELATIVE residual — divide by
   `‖r‖₂`, never by `max(1, ‖r‖₂)`.** The core's existing `residual` stat uses
   the latter (`structuredSaddleResidual`, `linsolve.ts:848`) and is therefore
   absolute for any rhs with `‖r‖₂ ≤ 1`, which the projection rhs `[0; −Φ]`
   always is; reusing it here would accept a solve that is ~1e4× worse than
   this criterion claims (§4.2, verdict I2). Per-step costs at k=4: factor-dependent setup = 3k+1 = 13
   N-backsolves; each solve = 3 N-backsolves + O(Nk); each residual = 3N²
   flops. All CPU f64 in 2a (§4.2). (At k = 4+3P with P pins the setup is
   13+9P backsolves — §7.)
   **Failure action — the one failure that is only detectable DURING the
   step.** Exhausting the 4 iterations without reaching 1e-10 MUST `throw` a
   typed `ExternalSolveError` out of `PreparedSaddle.solve` (§4.4);
   `GpuDriver.step` catches it, discards the prepared saddle and re-runs the
   step on the Phase 1 CPU path with `usedGpuSolve: false` (§4.5), exactly as
   a nonzero Cholesky status flag already does. An unconverged z must never
   reach the line search, and nothing else in the system would stop it: the
   residual IS computed (`linsolve.ts:800`, `:903`) and forwarded as a stat
   (`optimizer.ts:355, 395, 449`), but it is **never compared against any
   threshold anywhere in `src/core/`** — a solve that converged to 1e-3 is
   indistinguishable at every call site from one that converged to 1e-12.
   Without the throw, a factor that passes K1 and reports status 0 but stalls
   IR at ~1e-6 (§5's amber band) yields a wrong g̃, the line search rejects it
   as `not_a_descent_direction` / `armijo_failed`, the run auto-pauses — a
   wrong answer presented as a rejected step, with `usedGpuSolve` still true.

Preconditions and their failure modes: A_σ not PD (isolated vertex, non-finite
entry) → Cholesky reports a non-positive pivot → CPU path for this step; S or M
singular (C rank-deficient) → the same, through the existing
`'singular_system'` contract (the `optimizer.ts:338-383` try/catch). **IR
non-convergence is NOT covered by that contract** — `'singular_system'` fires
only on a throw, and a finite-but-bad L throws nothing; that is exactly why
step 4 introduces one. C is evaluated by the core at the step's vertices, never
by the driver (§4.4).

Why not the alternatives: Bunch–Kaufman LDLᵀ of the full K on the GPU —
pivot search is a global reduction per column and the 3N+k size costs 27×
the flops; augmented-Lagrangian Ā + CᵀWC — destroys the ⊗ block structure
(the barycenter rows couple coordinates, `constraints.ts:67-120`, verified:
all three coordinate blocks of every barycenter row are nonzero); CG —
§3.

### 4.2 Precision model [P §2.3 unchanged] + what T4 measures

**Numbers: [CAL §E.5, §E.6].** The measured r₀/ρ table, the κ ratios and the
IR-budget arithmetic are there; the model and its consequences are here.

- **Factor: f32 on GPU.** The matrix actually factored is A_σ (N×N), not K.
  κ₂(A_σ) is within **2 %** of κ₂(K) at every N measured
  ([K1CAL] `rows[].kappaRatioASigmaOverK`), so borrowing [G6]'s κ(K) as the
  proxy is legitimate; it is stated rather than assumed because they are
  different matrices and nothing in [G6] measured A_σ. [K1CAL] recomputes
  κ₂(K) on the same systems and reproduces [G6] to 6 significant figures, so
  the two tables are one measurement, not two. `kappaKTimesUf32` is well inside
  the LU-IR convergence condition κ·u_f ≲ 1 [PREC Q2] at every N in range.
- **ONE precision model, measured.** (The first draft carried two mutually
  inconsistent ones: a `c·N·u_f32 ≈ 1e-6..1e-4` backward error for r₀ and a
  `κ·u_f32` contraction — those are two different values for the same ε and
  cannot both hold.) For a factor with backward error ε = ‖ΔK‖/‖K‖ the initial
  relative residual is ≈ ε·(‖K‖‖z‖/‖r‖) and the IR contraction is
  ‖I − M⁻¹K‖ ≤ ε·κ — both proportional to the same ε. That proportionality is
  the whole model, and it is what licenses scaling [K1CAL]'s measured `r0` and
  `rho` linearly in ε wherever this spec reasons about a hypothetical kernel.
  It is measured against the f64 K, for both rhs shapes, with an f32 LAPACK
  factor ([K1CAL] `rows[].irModel`), and the table is in [CAL §E.6].
  **The constant relating r₀ to κ·u_f32 is a summary, not a law, and no bar
  rests on it** — it spreads 12.9× across the measured points and is not flat
  in N ([CAL §E.6]). The quantities the gates actually use — K1's `alpha` and
  `betaMin` — are bisected per N ([K1CAL], §5), never taken from a constant.
- **Everything else f64 on CPU** in 2a: A (CPU-assembled), C, Schur/Woodbury
  algebra, backsolves on the f32 factor promoted to f64, residuals, updates.
  IR therefore converges to the solution of the **f64** K — T4 holds exactly
  as the parent wrote it (residual vs f64 K ≤ 1e-10, ≤ 4 iterations,
  [K1CAL] `t4.tol` / `t4.maxIters`).
- **What T4's residual IS, and what it is not.** T4's quantity is the **pure
  relative** residual `‖r − K z‖₂ / ‖r‖₂`, which is also what [K1CAL] measures
  (`bench/gpu/k1-calibration.py`, `np.linalg.norm(r - K @ z) / nr`). It is
  **NOT** the core's existing `residual` stat. `structuredSaddleResidual`
  returns `‖r − K z‖₂ / max(1, ‖r‖₂)` (`linsolve.ts:848`), i.e. an *absolute*
  residual whenever `‖r‖₂ ≤ 1` — and the projection rhs is `[0; −Φ]` with
  `‖Φ‖` driven toward the projection tolerance
  (`lineSearch.ts:207-208, :261-262`, `tolAbs = tolRel = 1e-4`), so `‖r‖₂ ≪ 1`
  always and the core's number is smaller than the relative one by the full
  factor `1/‖r‖₂` — four orders of magnitude in the tolerance neighbourhood,
  in the **permissive** direction. An implementer who certifies §4.1 step 4's
  IR loop, or measures T4, with `solveSaddleFrozen`'s returned `residual`
  (`linsolve.ts:903`, which §4.4 points at) gets a number that passes 1e-10
  while sitting at ~1e-6 relative — **exactly the amber failure the throw
  exists to catch**. Both the step-4 criterion and the §5 T4 row therefore say
  `‖r‖₂`-relative explicitly, and plan 2a's IR code must normalise by `‖r‖₂`
  itself rather than reuse the core stat.
- **Pre-registered consequence: at N=960 a factor sitting exactly on K1's green
  bar exhausts the IR budget exactly.** Scaling [K1CAL]'s `irModel.gradient`
  `r0` / `rho` linearly in ε, a LAPACK-quality factor (ε = `alpha`) needs z₀ + 3
  refinements against a budget of z₀ + 4 — one spare — while a factor at
  `k1Bar` needs all four, with none. The model's exhaustion level sits ABOVE
  `k1Bar` and BELOW `e_max = betaMin`, so **K1-green implies T4-passes under
  the model, and K1-red is where T4 failure is also measured**; the arithmetic
  and the whole ladder are in [CAL §E.5]. Two things follow that this spec
  must state, not merely renumber:
  1. **The expected outcome for N=960 is that it is SERVED, with no iteration
     margin** — not that the served N narrows to 480. The first calibration of
     [K1CAL] put a plausible GPU factor (~3× `alpha`) inside K1's amber band
     and this spec pre-registered the narrowing; the corrected calibration
     ([CAL §E.1]) puts the same factor **green**, and the prediction inverts.
     This is the one place a calibration change altered what the design
     predicts. §6's T4 branch is unchanged **as a branch** — it is a
     conditional, and no GPU kernel has been measured.
  2. **K1 is not what catches a factor between the green bar and the
     exhaustion level.** The §4.1-step-4 `ExternalSolveError` throw is, per
     step, at run time. That is the load-bearing reason the throw exists, and
     it is why §4.5's fallback path is not optional.
  N ≤ 480 passes in ≤ 3 solves with ≥ 1 spare ([K1CAL]
  `irModel.<rhs>.solvesTo1e10`). The measured IR curve is a deliverable at
  every N either way (§5's results JSON).
- **2b changes one thing:** A comes from the GPU in f32; IR then converges to
  the f32-assembled K. **AMENDS [P §3 T4] for 2b onward:** T4's residual is
  measured against K built from the *promoted* GPU A and the f64 C, and a new
  gate T4a bounds the assembly error itself (§4.6). Rationale: the f64 A is
  exactly the O(N²) CPU cost 2b removes; [PREC "Recommended architecture"
  Stage 1] already places assembly in f32; the metric's rounding perturbs the
  descent DIRECTION, which T5 and (in 2c) τ-agreement judge, not the solve.
- **Never-mix [P §2.3]** is untouched by 2a/2b: every energy is CPU f64. 2c
  makes ALL energies of a run GPU f32 (E₀ chained like `energyBefore`, every
  trial GPU) — one source per run, the f32 Armijo margin argument of
  [P §2.5].
- Compensated arithmetic is not used anywhere on the GPU [PREC Q3]; nothing in
  the Cholesky needs it — plain f32 sums, for the reason given in §4.3 (the
  pivot floor is set by λ₂(A), not by the shift).

### 4.3 The GPU Cholesky kernel (`src/gpu/cholesky.ts`)

- **Layout:** A_σ row-major f32 in one storage buffer (N×N, no padding).
  Factor IN PLACE: on completion the lower triangle holds L, the strict upper
  triangle is garbage; the CPU reads back the whole buffer once (4 MB at
  N=1000) and promotes the lower triangle to f64. The buffer is both uploaded
  (per step) and GPU-written, so it stays `read_write` (no `.toReadOnly()`,
  unlike Phase 1's uploaded position buffer); the readback reuses ONE
  `THREE.ReadbackBuffer` target per engine
  (`renderer.getArrayBufferAsync(attr, target)`, released after each use —
  `WebGPUAttributeUtils.js:385-460` in r0.185.1) instead of allocating a 4 MB
  staging buffer per step.
- **Size bound — queried at boot, not asserted.** The first draft said
  "N ≤ 4096 by construction"; **no such construction exists** (`rg 4096 src/`
  is empty). The only size bound on the GPU path is
  `REDUCE_MAX_EDGES = 128*128 = 16384` (`src/gpu/reduce.ts:37`), applied to the
  EDGE count by `gpuTopologySupported` (`src/gpu/driver.ts:44-46`) — and at
  N = 16384 the A_σ buffer is 1.07 GB: 8× over WebGPU's default
  `maxStorageBufferBindingSize` (128 MiB) and 4× over `maxBufferSize`
  (256 MiB), with the crossover at N ≈ 5793. The failure mode would be a
  buffer-creation validation error, NOT the graceful per-step CPU fallback
  §4.5 promises. Therefore
  `GPU_SOLVE_MAX_N = floor(sqrt(renderer.backend.device.limits.maxStorageBufferBindingSize / 4))`,
  queried ONCE at boot beside the self-test (`src/gpu/selfTest.ts:68-101`,
  consumed at `src/scene/Viewer.tsx:636`), with a hard fallback of 4096 if the
  limit is unreadable, and an explicit `N ≤ GPU_SOLVE_MAX_N` term in §4.5's
  routing predicate. **The DEVICE limit, not the adapter's, and this is not
  interchangeable.** Verified in `node_modules/three` at r0.185.1
  (`require('three/package.json').version` → `0.185.1`): (i) the adapter is a
  LOCAL in `WebGPUBackend.js:217` and is never stored — `grep -n "this.adapter"
  WebGPUBackend.js` is empty, only `this.device = device` survives (`:292`) —
  so `renderer.backend.adapter` is `undefined` and reaching the adapter would
  need a second `navigator.gpu.requestAdapter()`, which `Viewer.tsx:110`'s own
  comment forbids ("shares with rendering, **never a second adapter**");
  `renderer.backend` is public (`Renderer.js:113`). (ii) Even reachable it is
  the wrong number: three requests the device with `requiredLimits:
  parameters.requiredLimits` (`WebGPUBackend.js:243`) and defaults that to `{}`
  (`:93`), and a limit absent from `requiredLimits` takes WebGPU's DEFAULT, not
  the adapter's supported maximum — so this device carries 134217728 B (128 MiB)
  regardless of what the Quadro RTX 3000's adapter advertises (commonly 2 GiB),
  and buffer-creation validation is against the device.
  **Resulting bound: `floor(sqrt(134217728/4)) = 5792`** (5792² · 4 B =
  134189056 B ≤ 128 MiB; 5793 overflows it), against `floor(sqrt(2147483644/4))
  = 23170` had it been sized from a 2 GiB adapter — 4× too large in N, 16× in
  bytes, which would reinstate exactly the validation-error failure mode this
  bullet exists to remove. Note also `WebGPUBackend.js:213` requests
  `featureLevel: 'compatibility'` and `:254` sets `compatibilityMode` from
  `!device.features.has('core-features-and-limits')`, under which defaults can
  be lower still — one more reason to read the number at boot rather than
  assert it. Neither bound binds in Phase 2 (N ≤ 1000, §1); the constant exists
  so that the routing predicate degrades gracefully rather than throwing.
- **Algorithm, first implementation: unblocked right-looking Cholesky** —
  three kernels per column j = 0..N−1, in this order: `colFactor` (thread per
  row i > j): A_ij /= √A_jj — the diagonal is still the raw pivot d_j here;
  `trailingUpdate` (thread per (i,k), j < k ≤ i, INCLUDING i = k so the
  trailing diagonal is updated): A_ik −= L_ij·L_kj — never touches row/column
  j; `finalize` (ONE thread): reads d = A_jj, sets the status flag if
  `!(d > 0)` or d is non-finite, writes A_jj = √d, and advances the column
  counter. All 3N nodes go into ONE `renderer.compute([...])` array per
  factorization [P §2.5 "one array per GPU stage"; G1].
- **How column index j reaches the shader — a GPU-side counter, not a
  uniform.** The array holds the SAME three node objects repeated N times
  (3 node builds total, not 3N shader compiles), and j is read at dispatch
  time from a 1-element u32 storage buffer that `finalize` increments. This
  is forced, not a preference: in three r0.185.1 a batched
  `renderer.compute([...])` calls `updateForCompute` per node and submits
  once (`Renderer.js:2718-2817`), so a uniform shared across the array would
  present its LAST written value to every dispatch. The fallback, if the
  counter proves unworkable, is one distinct `uniform` node baked per compute
  node (3N node builds — correct but compile-heavy); the plan measures build
  time before choosing it. The inter-dispatch visibility this depends on
  (dispatch n+1 sees dispatch n's storage writes, same pass, no barrier) is
  guaranteed — each dispatch is its own synchronization scope
  ([gpuweb/gpuweb#4434](https://github.com/gpuweb/gpuweb/discussions/4434)) —
  and Phase 1's shipped two-pass reduction already depends on it
  (`src/gpu/reduce.ts:66-67,116`).
- **The dispatch SIZE is fixed at the j=0 worst case — priced, not fixed.**
  Repeating one node object N times does dispatch N times (`Renderer.compute`
  iterates the array and calls `backend.compute` per entry,
  `Renderer.js:2718`), but `WebGPUBackend.compute` resolves the dispatch size
  from `computeNode.dispatchSize || computeNode.count`
  (`WebGPUBackend.js:1630`) and caches it keyed on the NODE OBJECT
  (`:1652-1685`), while `renderer.compute(nodes, dispatchSize)` takes a single
  override for the whole array. So the shared `trailingUpdate` node launches
  ~N²/2 threads for EVERY column: N³/2 = 4.4e8 launches at N=960 instead of
  N³/6 = 1.5e8. "Thread per (i,k), j < k ≤ i" above is the thread's GUARD, not
  a shrinking dispatch. The 3× waste is inside the factor-wall estimate of [CAL §D.1].
  **Escape hatch, named but not built:** r0.185.1 ships
  `IndirectStorageBufferAttribute`
  (`renderers/common/IndirectStorageBufferAttribute.js:12`) and
  `WebGPUBackend.compute` honours it (`:1636-1644`) via
  `dispatchWorkgroupsIndirect`, so `finalize` can write the next column's
  workgroup count — the same GPU-side-counter trick applied to the dispatch
  size. Build it only if P2a's factor-wall bar fails (§6).
- **Why unblocked first:** it is the smallest kernel that can pass K1; its
  dispatch-floor cost already beats the CPU factor by more than an order of
  magnitude on the solve path ([CAL §D.1]) and Phase 2 has no absolute latency
  gate [P §4 G7 is Phase 3]. A blocked
  (panel/TRSM/SYRK) variant is pre-registered as the optimization if P2a's
  factor-time bar (§5) fails; not before.
- **Status flag:** shares the small u32 meta buffer with the column counter
  (`[counter, status]`); `finalize` writes `status = 1 + j` on a non-positive or
  non-finite pivot at column j — encoding the failing column, so a K1 failure is
  diagnosable without a second readback. Read back with the factor; nonzero ⇒
  CPU path for this step (never a NaN-laden L into the CPU algebra).
- Kernel body is `wgslFn` [P §2.1]; the reduction inside `colFactor` is none
  (a column scale) — no tree reduction is needed in this kernel at all. Plain
  f32 sums suffice. **The reason is λ₂(A), not the shift:** the first draft
  said "the shift keeps pivots ≥ σ/N-ish on the null direction", which is wrong
  twice — `A_σ(𝟙/√N) = σ(𝟙/√N)`, so the eigenvalue on the null direction is
  σ = 1.9e4 at N=960, not σ/N = 20; and the measured pivot floor is
  `min diag(L) = 3.47`, set by λ₂(A) = 3.3e-2, which is itself 6× BELOW σ/N.
  The conclusion survives; the stated reason did not, and it would mislead
  anyone re-deriving the guard.
- **Boot self-test extension** (`runGpuSelfTest`): factor a committed 64×64
  SPD fixture (A_σ of `trefoil(64)`, deterministic) and compare with a CPU
  f64 Cholesky on the forward metric `max|L_gpu − L_cpu| / max|L_cpu|`, against
  the bar of §5 (derivation and the value a correct kernel produces:
  [CAL §G]), else the GPU SOLVE is
  unavailable for the session while GPU dE stays as Phase 1 left it — two
  independent capability bits in the store: `gpuAvailable` (dE,
  `src/store.ts:132`) and a new `gpuSolveAvailable` (§4.4's delta list; §7 for
  its four reset sites).

### 4.4 AMENDS [P §1] — public-type delta (iv): the core seam

[P §1] names exactly THREE sanctioned public-type changes — (i) `SolverDriver`
widens with `'gpu'`; (ii) `DescentStepOutcome`'s stats-only variant in Phase 3;
(iii) `SobolevStepOptions` / `DispatchDescentStepArgs` gain `dE?: Vec3[]` —
and there is no (iv). This section adds one, marked, because that list exists
precisely so a fourth cannot appear silently (parent review-3 F1). Reason: the
factor is computed OUTSIDE the step (§1: `sobolevStepSet` is synchronous) and
must reach it; every alternative — a module global, a re-entrant hook, a second
entry point — is worse. It mirrors the Phase 1 `dE?` seam (plan 2026-08-29 D5:
select the SOURCE, never the arithmetic; absent ⇒ bit-identical).

**Every symbol the delta adds or changes — and nothing else:**

```ts
// src/core/sobolev/linsolve.ts  (new exports)
export interface ExternalFactorization { kind: 'external'; solve(rhs: number[]): number[] }
export type SaddleFactorization = LuFactorization | LdltFactorization | ExternalFactorization; // widened
/** Thrown by an external solve that cannot certify its own result (§4.1 step 4). */
export class ExternalSolveError extends Error {}
/** A saddle solve prepared OUTSIDE the step for the step's input vertices. */
export interface PreparedSaddle {
    A: Float64Array;                            // Sobolev matrix at the input vertices (f64; 2b: promoted f32)
    factor(C: number[][]): SaddleFactorization; // builds Z, W, S, M for the core-evaluated C; solve() = §4.1 steps 2–4
}
// src/core/sobolev/gradient.ts
//   solveConstrainedGradientSetFrozen gains a TENTH optional POSITIONAL parameter
//   `saddle?: PreparedSaddle` — exactly how `factorMode?` rides in today (optimizer.ts:352).
//   (It has no options object: nine positional params at gradient.ts:84-94.)
// src/core/optimizer.ts   SobolevStepOptions      += saddle?: PreparedSaddle
// src/core/dispatch.ts    DispatchDescentStepArgs += saddle?: PreparedSaddle   (forwarded like dE)
// src/store.ts            SimStore                += gpuSolveAvailable: boolean; setGpuSolveAvailable(b): void
```

`usedGpuSolve` is **not** a core type change: it is a field of the DRIVER's own
return — `GpuDriver.step`'s `Promise<{ outcome, usedGpu, usedGpuSolve }>`
(`driver.ts:116-119`, where `usedGpu` is already a sibling of `outcome`, not a
field inside it). `DescentStepOutcome` (`dispatch.ts:75-98`) is untouched, so
the parent's (ii) obligation — `SolverWorkerResponse`'s carried type plus the
worker plan's §T2 deep-equality tests (`test/worker-solver.test.ts:108, :208`)
— does not fire here.

**The seam adds no other exports.** The first draft had `gradient.ts` call
`solveFactored` and `structuredSaddleResidual`; both are module-PRIVATE
(`linsolve.ts:696`, `:810`, and `gradient.ts:17` imports only
`{ FactorMode, FrozenSaddleOperator, solveSaddleFromA }`), so that spelling
needed either two more public symbols or a duplicated residual — and a
duplicate would break the "absent ⇒ bit-identical" backstop the moment it
drifted, invisibly, because the golden suites exercise the seam-ABSENT path.
Instead the external path routes through **`solveSaddleFrozen`, which is
already exported** (`linsolve.ts:882-905`) and already does exactly
`solveFactored(fac, rhs)` + `structuredSaddleResidual(a, n, C, z, rhs)`
(`:902-903`) against the same `{a, n, C, fac}` operator
`solveConstrainedGradientSetFrozen` already builds (`gradient.ts:113`):

- `solveConstrainedGradientSetFrozen` (`gradient.ts:84-115`) with `saddle`:
  `A = saddle.A` (no `assembleAFlat`), `fac = saddle.factor(C)`,
  `frozen = { a: A, n, C, fac }`, and the solve is
  `solveSaddleFrozen(frozen, flatten(dE))` — same `{x, lambda, residual}`
  shape, and the residual stays the byte-identical structured matvec the
  goldens gate. Without `saddle`: the existing `assembleAFlat` +
  `solveSaddleFromA` lines, untouched.
- `solveFactored` (`linsolve.ts:696-698`) gains the `'external'` branch. Plan
  note: `LuFactorization` (`:219-223`) has NO `kind` field, so the test must be
  `'kind' in fac && fac.kind === 'external'` — `fac.kind === …` alone is a TS
  error on the union. The `'lu'` branch stays the verbatim pre-existing call,
  and its anchor comment (`:693-695`) must survive.
- `solveSaddleFrozen` and `projectOntoConstraintSet` then work unchanged — the
  frozen operator simply carries the external kind.
- **TWO control-flow changes in `optimizer.ts`, and they are different.** The
  first draft called this "one"; `sobolevStepSet` is in `optimizer.ts` too, so
  the count was wrong. Both are listed, because each has its own test.
- **(i) The gradient-solve catch must rethrow.** The catch at `:361-383` is a
  bare `catch {}` that converts EVERY throw into `'singular_system'`, which
  would swallow the §4.1-step-4 error before the driver could see it. It must
  rethrow:
  `catch (e) { if (e instanceof ExternalSolveError) throw e; /* …unchanged… */ }`.
  With `saddle` absent no `ExternalSolveError` can exist, so the branch is
  unreachable and the path stays bit-identical.
- **(ii) The phase-timing collector must be disarmed on the throwing path —
  by CATCH-AND-RETHROW, never by an unconditional `finally`.**
  `timingsBegin()` (`optimizer.ts:281`) arms a module-scoped collector that
  only `timingsEnd()` disarms (`phaseTimings.ts:40, :49, :57-61`), so a throw
  out of `timed('step', …)` would leak this step's timings into the next one —
  the same hazard Phase 1 guarded by putting the `dE` length check BEFORE
  `timingsBegin()` (`optimizer.ts:263-266`).
  **`finally { if (collect) timingsEnd(); }` is WRONG and is not a no-op on the
  success path.** `timingsEnd()` is destructive, not idempotent: it returns
  `acc` and sets `acc = null` (`phaseTimings.ts:57-61`). The success path
  already reads the ledger at `optimizer.ts:461` (`const timings =
  timingsEnd();`, inside `if (collect) { … if (timings) … }`). A `finally`
  wrapped around `timed('step', …)` runs FIRST, so `:461` returns `null`,
  `if (timings)` is false, and **every normal step returns with no `timings`
  at all** — silently killing P2a's per-phase table and `GpuDriver.step`'s
  timing patch (`driver.ts:175`, `if (usedGpu && outcome.timings)`), which is
  how `timings.dE` and `timings.factor` reach the gate JSONs.
  The required shape disarms only on the throwing path:
  ```ts
  try { /* timed('step', …) as today */ }
  catch (e) { if (collect) timingsEnd(); throw e; }   // discard + rethrow
  ```
  and `:461` is left exactly as it is.
  **Both halves get a test:** (1) a throwing step must leave the collector
  disarmed — arm, throw, then assert the NEXT step's `timings` contains no key
  from the throwing step; (2) a non-throwing step with `collectTimings: true`
  must still return a populated `timings`. Test (2) is the one that fails on
  the `finally` spelling, and nothing else in the suite would catch it.

Contract (TSDoc, like `dE?`):
- `saddle.A` MUST be the matrix of the SAME vertices the step receives (the
  driver copies `live` once and uses that copy for dE, A and the step — the
  Phase 1 rule at `driver.ts:164`).
- `factor` is called at most once per step, synchronously; `solve` is called
  for the gradient rhs and for each projection rhs.
- `solve` MUST either return a result whose residual it has itself certified
  (§4.1 step 4) or `throw ExternalSolveError`. Returning an unconverged z is a
  contract violation, not a degraded mode: nothing in `src/core/` thresholds
  `residual`. **The certification is against `‖r_i‖₂/‖r‖₂`, computed by
  `src/gpu/saddle.ts` itself.** It may NOT be delegated to the `residual` that
  `solveSaddleFrozen` returns (`linsolve.ts:903`), which normalises by
  `max(1, ‖r‖₂)` and is permissive by the full factor `1/‖r‖₂` on the
  projection rhs (§4.1 step 4, §4.2). The value `solveSaddleFrozen` returns is
  still forwarded as the step's `residual` stat, byte-identically — the seam
  changes what the EXTERNAL solve certifies, never what the core reports.
- `PreparedSaddle` carries closures ⇒ not structured-cloneable ⇒ it can never
  travel to the worker: `solverWorker.ts` throws if `saddle` is present
  (defensive; the `'gpu'` driver is main-thread only [P §2.1]).
- No other core change. `factorMode`, `FactorMode`, `buildSaddleMatrix`,
  `ldltFactor`, `luFactor`, `assembleAFlat`, `lineSearchStepSet`,
  `projectOntoConstraintSet` untouched.

### 4.5 Driver flow per step (2a) — `GpuDriver.step` (`src/gpu/driver.ts:116-186`)

```
vertices = copy(live)                                   (existing rule)
upload positions (hi/lo, existing)
renderer.compute([ dE kernels… ])                      BATCH 1 (async, not awaited yet)
   └─ while it runs: A = assembleAFlat(…)   CPU f64  ~86 / 355 ms at N=480/960  [BASE]
                     Aσ = f32(A + (σ/N)𝟙𝟙ᵀ)  CPU      N² adds, ~1.4 / 3.2 ms
upload Aσ ; renderer.compute([ cholesky nodes… ])       BATCH 2
readback dE (existing) ; readback L + status flag       getArrayBufferAsync
if status ≠ 0 or L non-finite → no saddle (Phase 1 path for this step)
saddle = makePreparedSaddle(A, L, σ)                    (src/gpu/saddle.ts — closures over CPU f64 algebra)
try   outcome = dispatchDescentStep({ …args, dE, saddle, collectTimings: true })
catch ExternalSolveError → outcome = dispatchDescentStep({ …args, dE, collectTimings: true })   // usedGpuSolve: false
timings: dE (GPU wall, existing) ; assembleA (driver-side CPU) ; factor = GPU wall incl. upload+readback ; saddle/projection wraps fire inside the core as today
```

- **Why two batches, not one:** the Cholesky nodes need Aσ uploaded, and Aσ
  costs a full CPU `assembleAFlat` (the largest remaining CPU term, §2). Issuing
  the dE batch first lets that assembly overlap the GPU dE work instead of
  serialising behind it (dE is 3.2 / 5.1 ms against an 86 / 355 ms assembly, so
  the overlap is complete), and it keeps the two wall-clock timings separable
  (`timings.dE` vs `timings.factor`) so T4/P2a measure what they claim. Cost:
  one extra submit (~0.5 ms), well inside the P2a budget.
- **Routing predicate** `gpuSolveSupported(st)`: Phase 1 predicate
  (`sobolev`, `analytical`, penalties off, topology buildable) AND
  `projectionMode === 'frozen'` AND constraint set ∈ {barycenter,
  totalLength, pins} (no `edgeLengths`) AND **`N ≤ GPU_SOLVE_MAX_N`** (§4.3)
  AND `gpuSolveAvailable`. Unsupported ⇒ the step runs exactly as Phase 1
  (GPU dE, CPU factor), driver unchanged [P §2.6].
- **Three fallback triggers, one action.** (1) Status flag nonzero or
  non-finite L — detected BEFORE `dispatchDescentStep`, so the step simply runs
  without `saddle`. (2) `ExternalSolveError` from a PROJECTION solve — thrown
  inside `lineSearchStepSet`, which sits outside `optimizer.ts`'s try/catch, so
  it propagates straight out of `dispatchDescentStep`. (3) `ExternalSolveError`
  from the GRADIENT solve — inside that catch, which rethrows it (§4.4). In (2)
  and (3) `GpuDriver.step` catches, DISCARDS the prepared saddle and re-runs
  `dispatchDescentStep` for this step with the same `vertices` copy, the same
  `dE`, and no `saddle`, reporting `usedGpuSolve: false` — exactly what a
  nonzero status flag does. The re-run is safe because the step is pure in its
  inputs; the bench harness already depends on that (`bench/sobolev.bench.ts`
  runs K=5 measured steps from the SAME `vertices` array — "the step is
  pure/deterministic → identical work"). Price of a fallback step: one wasted
  partial step, ≲2× a CPU step. Acceptable, because §5's K1/T4 bars make it
  rare and a silently wrong g̃ is not acceptable at any price.
- None of the three is a driver flip: all are per-step and counted in the
  driver's return.
- Device loss / uncaptured error hooks: unchanged from Phase 1 (`gpu→worker`),
  except that they must now reset BOTH capability bits (§7).
- Main-thread cost: assembly + Schur algebra + line search still run on the
  main thread (Phase 1 already does the whole CPU step there); 2a makes it
  roughly **4–12× shorter**, bounded by the honest ceilings of [CAL §B] — not
  the "~7–13×" the first draft claimed at N=480, which was above that N's
  ceiling. (An earlier fold quoted the ceilings here and in §6 as 4.49–5.05× /
  12.33–13.20×; those figures reproduce from nothing and are withdrawn —
  [CAL §B] carries the only ones. This is exactly the transcription class the
  split exists to end: no ceiling value is restated in this document.)
  2b/2c remove the remaining O(N²) CPU work.

### 4.6 Slice 2b — A assembly on the GPU (design sketch; its plan details)

- Gather formulation, no atomics [P §2.2]: A_ij for i ≠ j receives
  contributions only from ordered disjoint edge pairs (I ∋ i, J ∋ j) — at
  most 4 for a curve — so a thread per (i, j) gathers over the vertex→edge
  CSR of both endpoints with a membership test against a sorted
  `disjointPairs` CSR; the diagonal-block terms (i, j both in edge I) sum
  over ALL J disjoint from I and get their own thread-per-edge kernel;
  the `0.5·(B_ij + B_ji)` symmetrisation is folded into the gather (both
  orderings computed by the same thread). Low-order B⁰ is a second pass of
  the same shape. Output: A_σ f32 already shifted, GPU-resident — no upload.
- **T4a (assembly gate):** on the T2/T3 fixture set, max_ij |A_gpu − A_f64| /
  max|A_f64| < 1e-5 (scale-relative, T1-style) AND for x ∈ {dE, 3 seeded
  random vectors} ‖A_gpu x − A_f64 x‖/‖A_f64 x‖ < 1e-6 (matvec-level,
  T2-style), f64-measured from the read-back A. **Both levels are INHERITED
  from [P §3], not originated here** — T1's `rel err < 1e-5` and T2's
  `rel err < 1e-6`, `2026-08-13-webgpu-solver-design.md:276-277` — so §5's
  ban on hand-transcribed bars is satisfied by citation, the same way T5's
  `±1 %` is (§5).
- IR in 2b reads A back once per step (4 MB at N=1000) and runs on the
  promoted f32 A — T4 as amended in §4.2.

### 4.7 Slice 2c — GPU line search (design sketch; its plan details)

- A new async orchestrator `src/gpu/step.ts` mirrors `lineSearchStepSet` +
  `projectOntoConstraintSet` [P §2.5 "fixed-count quasi-Newton correction
  loop"]: trial positions, Φ/C of the barycenter/totalLength/pin blocks in
  shader, projection corrections as multi-RHS triangular solves against the
  resident L plus the k×k / 3×3 algebra (on the GPU via tiny kernels or on
  the CPU with one small readback per trial — plan-level measurement),
  trial energies via the Phase 1 energy kernel, Armijo in shader; ONE
  `renderer.compute([...])` per step; one positions + stats readback per
  step [P §2.8 Phases 0–2].
- **Hard constraint inherited from §4.1/§7: the 3×3 M block may NOT be formed
  in f32.** `M = σ⁻¹I₃ − UᵀK̂⁻¹U` is a difference of two O(σ⁻¹) quantities that
  agree to 1 part in 9.2e4 at N=960, so its formation destroys ~5 decimal
  digits and the loss grows ∝ κ — at f32's 24 bits there are ZERO significant
  digits left by N=480. Whichever way 2c splits the small algebra, M's
  formation and inversion stay f64 on the CPU (the readback is 9 floats).
- All energies of a run GPU f32 (never-mix, §4.2); `energyBefore` chaining
  reproduced from the GPU energy.
- Gates: **τ-agreement** exactly as [P §2.5] (shadow CPU line search from the
  same state; k-agreement ≥ 95 % over K=50 steps per T5 fixture); **T5**
  [P §3]; the honest full-step re-measure (G5 clause) at N=480/960 — this is
  where [P §2.4]'s "a variant that wins the gradient solve but loses the line
  search's projections has not won" is discharged (§1). **Not to be confused
  with §5's `G5-confirm`**, which is a 2a gate that decides (a)-vs-(b) on the
  spike before integration; this one is 2c's final full-step number once the
  line search is GPU-side, and by then the variant question is closed.

## 5. Verification gates (pre-registered; run on the hardware adapter, INVALID on software — `bench/gpu/README.md`)

**Every level in this section is a citation, not a transcription.** The
derivations, the measured inputs and the evaluated bars are in **[CAL]**; this
section owns each gate's *quantity*, its *pass semantics*, its *fixtures* and
its *preconditions*. That division is binding (see the header): three blind
rounds each found a different hand-transcribed bar in this section, so a bar
this section ORIGINATES that is not a `[K1CAL]`/`[JSALG]`/`[BASE]` key or a
formula over such keys is not a bar — **with exactly one further category,
written down here so that the rule is honestly amended rather than silently
broken by its own document.** (Levels this section does not originate but
INHERITS are checked against the parent that pre-registered them, not against
[CAL]: T5's `±1 %` over `K=50` steps [P §3] and τ-agreement's `≥ 95 %`
[P §2.5].)

**The one further category: a DECLARED POLICY TARGET.** A level that is
*chosen, not measured* — a success criterion no measurement can derive, and
whose deletion deletes the milestone's point. P2a's `speedup ≥ 8×` and its
`factor wall ≤ 100 ms` are the only two in this document. A declared target is
admissible ONLY if all three of these hold:

1. it is **labelled a declared target** where the table below states it;
2. it **names who declared it, and the decision it serves**; and
3. it is **never presented as derived or measured** — no `[K1CAL]`/`[JSALG]`/
   `[BASE]` key, no expression and no [CAL] section is offered as its source.
   [CAL] may report the *margin* a prediction leaves against it; that is a
   different claim from producing it.

A level failing any of the three is a hand-transcribed bar and is still banned.
The rule always targeted numbers that *pretend* to provenance — which is what
all three Criticals were. A number that claims none cannot go stale silently;
it also cannot borrow authority it has not got.

Three of the bars (CR, K1, `G5-confirm`) are derived from **measured**
quantities rather than from a round number or a textbook bound, because the
first draft's round numbers were shown to be uninformative — a flat 1e-4 K1 bar
sits two orders of magnitude above the level at which T4 failure is measured at
N=960, so a factor well inside it would pass K1 while needing an order of
magnitude more refinements than T4's budget allows. **A bar that cannot fail before T4 does is not a gate.**
[CAL §E.3] carries that arithmetic. **Boot is the exception and is labelled
as one:** its `1e-5` is a *chosen round level*, not a formula over
`rows[].bootMetric` — [CAL §G] says so and reports the 110.6× margin the
measurement leaves against the choice, which is a report, not a derivation. It
is admissible without being a declared policy target because no decision turns
on its value (it gates *availability* on an unknown adapter's summation order),
and it originates in [CAL §G], which this section cites rather than
transcribes.

**K1's construction, in words.** Two inputs, both bisected per N, neither
fitted: (α) what a *correct* f32 Cholesky achieves — [K1CAL] `rows[].alpha`,
**flat in N**, not the `c·N·u_f32` an early draft assumed; and (β) the largest
backward error T4 still survives, by bisection on the §4.1 IR loop. **β is not
a level, it is a distribution**: it is direction-dependent, so it is measured
over five pre-registered directions ([K1CAL] top-level `betaSeeds`, so a re-run
reproduces every bar) and **each bar is derived from the MINIMUM**, never a
median and never a fit. The bar is the **geometric mean of α and β_min**
([K1CAL] `rows[].k1Bar`) — equidistant in log space from "indistinguishable
from a correct kernel" and "the worst T4-fatal level measured", with equal
slack each way (`k1SlackAboveAlpha == k1SlackBelowBetaMin`). Two consequences,
both load-bearing:

- **The `0.177/κ₂(K)` fit an earlier fold used is deleted.** `β·κ₂(K)` is not a
  law — [K1CAL] `rows[].betaTimesKappaKMin` spreads 6.3× across N and is worst
  at exactly the end where the fit was the only source ([CAL §E]). With every
  row bisected, nothing is inferred.
- **κ₂(K) at N=64 is measured too** ([K1CAL] `rows[].kappaK`), not read off
  [G6]'s `N^3.285` growth fit, because N=64 is below [G6]'s smallest sampled N.

**What `e_max(N) = betaMin(N)` does and does not assert.** It is the lowest
backward error at which T4 failure has been *observed*, over five directions. A
factor at that level failed T4 in the worst direction measured; **it is not a
proof that every factor at that level fails**, and §6's red branch is worded
accordingly. Taking the minimum rather than the median is what makes the amber
band do its job — a factor between the two is red here and would be amber under
a median-derived `e_max` while already failing T4 in a majority of directions
([CAL §E.2] works the current example). This is also why K1 has three bands and
not two.

**The N=960 window, stated as an ordering rather than an adjective.** K1's
admissible window at a given N is `betaMin/alpha`. N=960's is the narrowest
measured, by ~6.5× over N=480's — and that ORDERING, not any adjective about
narrowness, is why §6's kill is K1 at **N=480** and not at N=960: N=480 is
where a red is least ambiguous. (An earlier fold called N=960 "a narrow gate by
construction … cannot be widened without becoming uninformative" on a window
that the corrected calibration has since widened 3.1×; the adjective no longer
holds and has been dropped. [CAL §E.4] states what changed and why the argument
survives.)

**CR's bar is κ-relative**, because its quantity (`‖Δz‖/‖z‖`, a *solution*
error) has a floor of ≈ `κ₂(K)·u_f64` and its own reference is only that
accurate: `solveSaddleFromA`, CR's own REFERENCE, reaches 1e-10 error at
N ≈ 1000, the parent's own G7 target [P §4 G7(i)]. A flat 1e-10 bar is
therefore vacuous where the gate can run and meaningless where it would bite,
and the first draft's fixture list also stopped at N=480 — so the only sizes
where a flat bar could have failed were the sizes it never ran. **CR's rhs is
pinned** ([CAL §F]): both shapes, gradient `[b; 0]` and projection `[0; d]`,
the same seeded pair [K1CAL] uses, each reported and each held to the bar. It
had to be pinned because the bar's slack is rhs-dependent and nothing in the
first draft said which rhs CR used.

| Gate | Quantity | Tolerance / bar | Fixtures | Slice |
|---|---|---|---|---|
| **CR** | reference reformulation: z from §4.1 steps 1–3 using a CPU **f64** Cholesky vs `solveSaddleFromA` z, for BOTH pinned rhs shapes ([CAL §F]) | `‖Δz‖/‖z‖ ≤ max(1e-12, 10·κ₂(K,N)·u_f64)`, κ₂ = [K1CAL] `rows[].kappaK`, `u_f64 = 2⁻⁵³` — evaluated in [CAL §F]. The measured slack below the bar is a **deliverable of this gate**, per rhs and per N, not a published input ([CAL §J.3]) | crossing N=8; trefoil N=60/120/240/480/**960**; sets: barycenter+totalLength, +1 pin | 2a, CPU-only (`bun test`), BEFORE any WGSL |
| **K1** | GPU f32 Cholesky backward error `‖LLᵀ − A_σ‖_F/‖A_σ‖_F`, f64-measured from the read-back L; the **value** is recorded, not just pass/fail | **green** ≤ [K1CAL] `rows[].k1Bar`; **amber** in (`k1Bar`, `e_max(N)`) — not a kill by itself, the served N is then decided by T4 at that N and the value goes in the README; **red** ≥ `e_max(N) =` [K1CAL] `rows[].betaMin`. Status flag 0 in all cases. Values: [CAL §E] | A_σ of trefoil N=64/240/480/960 (the parent's "committed N=480 system" = trefoil(480), default params, checksum recorded) | 2a |
| **T4** [P §3] | **pure relative** residual `‖r − K z‖₂/‖r‖₂` (f64) after IR — NOT `structuredSaddleResidual`'s `/max(1, ‖r‖₂)` value (§4.2); IR iterations | ≤ [K1CAL] `t4.tol` within ≤ [K1CAL] `t4.maxIters` iterations, per N | crossing N=8; trefoil N=60/120/240/480/960, total mode (+ perEdge N ≤ 240 informational only) | 2a; 2b re-based (§4.2) |
| **T5** [P §3] | energy after K=50 steps within ±1 % of the f64 trajectory; no accepted step increases energy; f64 recomputed from positions; median of 3 | as stated | trefoil N=120 **and N=480**, total, frozen | 2a, 2c |
| **G5-confirm** (perf, decision) | **a PAIRED same-session ratio**, exactly as P2a is: `r_a = (CPU f64 p50) / (variant-(a) p50)`, both medians from the SAME browser session on the same fixture, interleaved, wall-clock incl. uploads/readbacks; whole-descent-step **and** solve-path readings, BOTH reported | at **N=960**, against variant (b)'s bands expressed in the same currency (a ratio against [BASE]): **green** `r_a_step ≥ 1.5·max(r_b_step)`; **red** `r_a_step ≤ max(r_b_step)` ([P §4 G5]'s rule fires nowhere); **amber** in between — the whole-step reading does not decide, so the solve-path reading must, and green then requires `r_a_solve ≥ 1.5·max(r_b_solve)`, else red. Formulas, evaluation and the residual unit correction: **[CAL §I]**. **Precondition: [CAL §J.2]** — the levels are not readable until a committed emitter writes them. N=480 recorded, not gated (no (b) whole-step band is measured there) | trefoil N=480/960, total, frozen | **2a, after the Cholesky + K1 spike, after the §4.4 core seam has landed, and BEFORE any driver / store / self-test integration** |
| **P2a** (perf) | full-step p50, `'gpu'` driver with solve, vs CPU f64 step p50 — a **PAIRED** measurement: both medians from the SAME browser session, on the same fixture, interleaved; 1 warm-up, 5 runs, medians, wall-clock incl. all uploads/readbacks [P1 D4 method]; plus GPU factor wall (upload+compute+readback) | **gated at N=960 only: speedup ≥ 8×**; factor wall ≤ 100 ms at N=960 (unblocked bar). **Both levels are DECLARED POLICY TARGETS — chosen, not measured** (§5's one further category). **Declared by this spec**, not by [P §4], which pre-registers neither: the `8×` is the milestone's success criterion for the decision *"is shipping variant (a) worth its complexity"*, and the `100 ms` wall is the trigger for the decision *"is the blocked variant / the §4.3 indirect dispatch built"* (§6). Neither is derived from a measurement and neither is a [CAL] output — [CAL §H.4] reports the *margin* the term-by-term prediction leaves against the `8×`, and [CAL §H.5] the headroom against the wall, which is a different claim from producing them. **That headroom is currently NEGATIVE: [CAL §H.5]/[CAL §D.1]'s term-by-term model puts the pessimistic corner of the wall at 101 ms, 1.1 ms ABOVE this declared 100 ms target** (it read as 1.2× headroom only while the launch term used a `2.0e10 thread/s` rate that [G0t] does not support — verdict S1). The target is unchanged, because a declared target is not moved by a prediction; the consequence is that §6's `wall > 100 ms` branch — build the blocked variant or §4.3's indirect dispatch — is a **predicted outcome at the pessimistic corner**, not a remote contingency. The optimistic corner, 54 ms, clears it. **N=480 is RECORDED, NOT GATED** ([CAL §H.4]). **TWO preconditions, both pre-registered, neither deferred:** (1) `R₂` is ATTRIBUTED — named work, or named measurement error, per this row's own session (§2, [CAL §H.4]); (2) **five browser CPU f64 FULL-STEP p50s exist at N=480/960 and the noise bootstrap has been re-run on them** ([CAL §J.1]) — until then the bands behind this bar are PROVISIONAL | trefoil N=480/960, total, frozen; **plus trefoil N=960 + 3 pins** (k = 4+3P, §4.1's setup cost is 13+9P backsolves — recorded, not gated) | 2a |
| **Boot** | self-test §4.3 | `max|L_gpu − L_cpu|/max|L_cpu| <` the bar of [CAL §G] — a **chosen round level** (`1e-5`), *checked* against [K1CAL] `rows[].bootMetric` (110.6× slack) rather than derived from it | trefoil(64) | 2a |

**AMENDS [P §3]** — two fixture-set narrowings, marked per §0: T4's "both
constraint modes" becomes total mode gating + perEdge informational, and T5
drops its perEdge fixture — because §1 does not serve perEdge on the GPU solve
([G6] already routes perEdge N ≥ 480 to CPU f64). T5 *gains* N=480 (it is the
only trajectory-level gate, and its single N=120 fixture sits 1000× away in
`kappaKTimesUf32` from N=960 — the regime every κ-dependent defect lives in);
CR gains N=960 for the reason in its bar derivation.

**Boot's metric is a *forward* difference, not K1's backward error**, and its
bar is deliberately slack: the boot test gates *availability* on an unknown
adapter's summation order, not numerics, and a broken kernel (dropped term,
off-by-one column) produces O(1) relative error, not a marginal one. It is
100× tighter than the first draft's bar, which sat only ~21× below the T4-fatal
level in this metric's currency. Values: [CAL §G].

### What licenses a P2a bar, and why N=480 does not carry one

**Numbers: [CAL §H].** The term-by-term prediction, the [JSALG] constants, the
noise bootstrap and the decision table are there. Three things are decisions,
and they live here:

- **The PAIRING is what licenses a bar, not the size of the swing.** P2a's
  denominator swings ±13 % *between* browser sessions ([P1]'s three de-gate
  runs, `bench/gpu/README.md:456`). P2a is specified as a paired same-session
  measurement precisely so that cancels: a session that is 13 % slow is 13 %
  slow on both sides of the ratio. Without the pairing no bar in this table
  would be honest; with it, a bar is decided by the kernel rather than by
  machine load. **This sentence, not the size of the between-session swing, is
  the justification** — and the same reasoning is why `G5-confirm` is now a
  paired ratio too ([CAL §I.1]).
- **N=480 is RECORDED, NOT GATED; P2a's kill authority rests on N=960 alone.**
  At N=480 a bar survives only if the pairing is assumed to cancel the
  numerator's sampling error perfectly, and then by about one percentage point
  — a green would barely distinguish a real win from a lucky median, and a red
  would barely distinguish a broken kernel from an unlucky one. The N=480 value
  is still measured, reported and put in the README (it is the Amdahl-limited
  size where a regression would show first); it just cannot stop the branch.
  This is a real narrowing of P2a's power and is stated as one; the route back
  is a run count large enough to push the strict band below the margin, which
  would be an amendment to [P1 D4]'s method, not a re-reading of a table.
- **The noise defence is PROVISIONAL, and the reason is a defect this spec
  owns.** The bootstrap behind those bands is built on [P1]'s five committed
  `cpuMs` samples per N — and those time `gradientAnalytical` alone
  (`bench/gpu/phase1.ts:324-326`; the run's own JSON carries
  `data.fullStepGate: false`). They are CPU **dE-phase** p50s. An earlier fold
  of this spec called them CPU **step** times while, thirty lines later,
  dividing the identical five values by [BASE]'s **dE** to derive the
  browser-scale factors ([CAL §C]) — one set of numbers used as two different
  statistics.
  **P2a's actual denominator, a browser CPU full-step p50, has never been
  measured in this repo.** A step's median-of-5 spread is not a dE phase's: a
  step additionally carries a variable Armijo trial count and a variable
  projection iteration count (`lineSearch.ts:209`, `maxIter = 8`). So the 8×
  bar stands as pre-registered — its margin is a property of the prediction and
  does not move — but its noise defence is not readable until [CAL §J.1] runs.
  That measurement is a **precondition of P2a**, alongside the `R₂` one, not an
  open question (§8).
- **`R₂` must be ATTRIBUTED before any P2a number is read as pass/fail.**
  Recording `R₂` again is not attributing it: plan 2a's gate task must **name
  the work, or name the measurement error**. [CAL §B] supplies one concrete
  candidate for the error branch — [BASE]'s own N=1000 row forces `R < 0`,
  which no amount of work can explain — so "measurement error" is a hypothesis
  with evidence behind it, not a way of waving the term away. The two blind
  reviews of this spec disagreed by exactly this term (one derived a ceiling
  top-down from `step`, the other a prediction bottom-up without `R₂`). §8's
  list is explicitly not where this lives.
- **`G5-confirm`'s amber test is the least marginal bar in the milestone**,
  which is why the whole-step reading is allowed to be inconclusive: (a)'s own
  predicted solve path clears the amber-green level at both ends of the
  constants' range (1.20× and 1.99× of margin), so only a kernel materially
  worse than this spec's model can land amber-then-red ([CAL §I.2]). **After
  verdict S1 that allowance is load-bearing, not precautionary: an on-model (a)
  is now predicted amber at BOTH corners**, not only the pessimistic one, so the
  solve-path reading carries the whole decision ([CAL §I.2]).

Results: `bench/results/<date>-gpu-phase2a-{cr,cholesky,t4,t5,step}.json`
keyed by `gitShaShort` + adapter, same provenance convention as Phase 1
(README "Provenance"); every gate records its measured VALUE, not only its
verdict, so a later regression is visible as a number. A red CR/T4/P2a at
N=960, a K1 red at N=480, or a `G5-confirm` red, stops the branch (§6).

**Where these bars come from is in the repository, not in a scratch
directory, and not in this document's prose.** Every derived quantity cited
above — α, β and its five directions, κ₂(A_σ), κ₂(K), the boot metric, r₀/ρ,
and variant (b)'s CG iteration count — is produced by
`bench/gpu/k1-calibration.py` into
`bench/results/2026-09-04-gpu-phase2a-k1-calibration.json` [K1CAL], and the
three JS f64 constants by `bench/gpu/jsAlgebra.ts` into
`bench/results/2026-09-04-js-f64-algebra.json` [JSALG]. Both are committed,
both re-run with one command (§1), and both key `gitShaShort` to the spec
commit's PARENT rather than to run-time HEAD, because this document is a
single amended commit and run-time HEAD would name a commit that is rewritten
on the next amend — the failure `bench/gpu/README.md` "Provenance" records.
Plan 2a re-runs both and, for the JS constants, records the same-session values
alongside the P2a verdict.

**Two bars have no emitter yet, and are therefore formulas, not levels.**
`G5-confirm`'s ratio bars ([CAL §J.2]) and P2a's noise bands ([CAL §J.1]) are
not produced by any committed script today. Under the rule in this document's
header they are consequently NOT quotable as numbers anywhere, and plan 2a's
gate tasks own producing them. That is the whole mechanism of the split: a
level that exists only as prose is a level that goes stale silently — which is
what happened here three review rounds running.

## 6. Kill gates and consequences

- **CR red** → the reformulation is wrong; fix the math (it is 40 lines of
  f64 TS + a numpy twin in `oracle/`) — no GPU work starts until CR is green.
- **K1 red on N=480** (`≥ e_max(480) =` [K1CAL] `rows[n=480].betaMin`, the
  parent's confirmation spike) → variant (a) is not confirmed; the README
  records the measured error. **Corrected consequence:** the first draft said
  the milestone then ships kernels-only "since (b) is predicted non-viable
  (§3)" — that premise is false. On the measured CG iteration count
  ([K1CAL] `rows[].cgIters`, not the textbook bound) variant (b) is a viable
  fallback on both readings ([CAL §D.2]). So a K1 red at N=480 routes to
  **building variant (b)** — the parent's original choice, whose only
  disqualifying evidence was an estimate this spec has now replaced with a
  measurement — and kernels-only is the outcome only if (b) is then measured
  below the P2a bars too.
- **K1 amber** (between `k1Bar` and `e_max(N) = betaMin(N)`, both [K1CAL]) →
  not a kill; T4 at that N decides whether the N is served, and the amber value
  is recorded. The amber band is doing real work: β is direction-dependent
  (§5), so a red is "T4 failure has been measured at this level in the worst of
  five directions", not "T4 cannot survive this" — T4 itself, at that N, is the
  arbiter. **Expected outcome at N=960, per §4.2:** the plausible-GPU-factor
  level is *green*, not amber, and it exhausts the IR budget exactly. So the
  branch to plan for at N=960 is **green K1 with T4 green and zero iteration
  spare**, and the §4.1-step-4 throw is what catches the rest. (An earlier fold
  pre-registered the opposite — amber at N=960 and a narrowing of the served N
  — on a calibration since corrected; [CAL §E.5] records the inversion.)
- **`G5-confirm` red at N=960** (`r_a_step ≤ max(r_b_step)`, or amber with
  `r_a_solve < 1.5·max(r_b_solve)` — §5, [CAL §I.2]) → **slice 2a stops at the
  spike and the milestone re-plans onto variant (b).** This is the parent's own
  escape ([P §4 G5]: the spike "confirms the estimate BEFORE Phase 2 commits to
  (a)"), and it is a gate rather than a judgement because the one corner where
  the parent's >1.5× rule does not fire — the whole-step pessimistic corner
  ([CAL §D.3]) — is a corner this spec must not settle by argument. **The gate
  compares two RATIOS, not two wall-clock levels**, and that is not cosmetic:
  the first draft's ms-level bars would have fired RED on a variant (a) landing
  exactly on this spec's own model, at both ends of its predicted range,
  because (a)'s side is necessarily a Chrome measurement
  (`bench/gpu/drive.ts:20`) and (b)'s band is built from Bun timings — the red
  bar was a threshold on the JIT ratio, not on kernel quality ([CAL §I.1]).
  It costs nothing extra — Tasks 4–5 are the spike, already scheduled, and the
  gate runs after the §4.4 seam and before any driver/store/self-test
  integration, so a red discards a spike rather than an integrated branch.
  §3 shows (b) is a viable fallback, not a dead end. If it is green, §3's
  provisional decision becomes the decision and Tasks 6–7 start.
- **T4 red at some N** → GPU solve is served only for N below the largest
  passing N (narrowing, the [G6] semantics), recorded in the README and the
  routing predicate; not a milestone kill. T4 red at N ≤ 480 IS a kill.
- **P2a red** (N=960 only — N=480 is recorded, not gated, §5) → if the factor
  wall > 100 ms at N=960 the blocked variant (or the indirect dispatch of §4.3)
  is built (one pre-registered extra task); if the step speedup is still < 8×
  with the factor under the bar, 2a does not merge and the per-phase table says
  where the time went. A low N=480 reading with N=960 green is an **Amdahl**
  result, not a kernel result — N=480's ceiling is in [CAL §B], and it is low
  — so it is recorded and may narrow the served N, but it cannot kill 2a on its
  own. **No P2a verdict is read at all until BOTH of §5's preconditions clear**
  (`R₂` attributed, and the browser CPU full-step samples measured).
- **T5 red** → investigate before merging (flakiness rule of [P §3]); a
  reproducible red is a kill.
- **Boot self-test red on the dev box** → the branch does not merge.

## 7. Risks not covered by gates

- **Per-node uniforms inside a batched compute array** (§4.3): if three
  binds one uniform value for the whole batch, 3N nodes with distinct
  `uniform` objects are needed (plan verifies against r0.185.1 source; the
  fallback is a storage-buffer column-index table indexed by a per-node
  constant). Named here because it decides the kernel's plumbing.
- **Fixed worst-case dispatch size** (§4.3): the shared-node scheme cannot
  shrink `trailingUpdate`'s dispatch per column, so it launches ~3× the useful
  threads (N³/2 vs N³/6). Priced into the factor-wall estimate of [CAL §D.1]; the escape
  hatch (`IndirectStorageBufferAttribute`) is named in §4.3, not built.
- **Readback size** (4 MB at N=1000 per step, 8 MB in 2b): `mapAsync`
  throughput on this box is unmeasured; P2a's factor-wall bar catches it.
- **Woodbury 3×3 conditioning — the first draft's "its entries are O(1)" is
  wrong by up to 9 orders.** Measured `diag(M)` = 4.88e-5 / 1.24e-6 / 2.74e-8 /
  **5.65e-10** at N=120/240/480/960, against σ⁻¹ = 5.58e-3 / 1.24e-3 / 2.58e-4 /
  5.22e-5: `M = σ⁻¹I₃ − UᵀK̂⁻¹U` is a difference of two O(σ⁻¹) quantities that
  agree to 1 part in 9.2e4 at N=960, so its FORMATION destroys ~5 decimal
  digits and the loss grows ∝ κ. κ(M) itself is benign (5.06→5.28 across
  N=120…960) and the loss is not today's dominant error, so 2a is unaffected in
  f64 — but at f32's 24 bits this formation has **zero** significant digits left
  by N=480, which is a hard constraint on §4.7 (see there). CR records
  `diag(M)`, `κ(M)` and `min diag(L)²` per fixture so the margin is a number,
  not a pass/fail.
- **Pins inflate every solve cost** (§4.5 routes them): `pointBlock` writes 3 rows
  per pin (`src/core/sobolev/constraintSet.ts:271-286`), so k = 4+3P and §4.1's
  setup is 13+9P N-backsolves — at [JSALG]'s central N=960 backsolve (1.954 ms)
  that is **25.4 ms at P=0, 113.3 at P=5, 201.3 at P=10**. P2a's pinned fixture (§5) puts the k-dependence on the record; if it
  reads worse than the model, the routing predicate gains a P bound before pins
  meet the GPU path in production.
- **JS backsolve cost** — at N=960, 13 setup N-backsolves + 8·3 more (4 IR
  solves each for the gradient and the projection rhs) + 8·3 N×N residual
  matvecs ≈ **106 ms central, 88–135 ms** across [JSALG]'s range ([CAL §H.2]) —
  is the 2a CPU floor by design; 2c moves them to the GPU.
- **Main-thread jank** is inherited from Phase 1 (not new); 2b/2c shrink it.
- **Two capability bits** (`gpuAvailable`, `gpuSolveAvailable`) must both be
  reset on device loss. `setGpuAvailable` has FOUR call sites in
  `src/scene/Viewer.tsx` (486, 636, 655, 669) — 655/669 are the device-loss and
  uncaptured-error hooks, 636 carries the `poisoned` latch (`:629-636`) that
  stops a late self-test result from re-arming a poisoned device. The second bit
  needs all four mirrored, latch included; test it.

## 8. Open questions (deferred to the plan)

These are implementation choices with no bearing on a gate, and none of them
blocks 2a. **Three things that look like open questions are NOT on this list,
because each is a pre-registered precondition on reading a gate — a blocker by
construction, and listing it here as well would contradict that:** `R₂`'s
attribution (§5); the five browser CPU full-step p50s P2a's denominator has
never had ([CAL §J.1]); and the emitter that turns `G5-confirm`'s formulas into
levels ([CAL §J.2]). [CAL §J] is the standing list of calibration debts, and it
is not this list.

- Per-node uniform vs storage-slot column index vs indirect dispatch (§7).
- Whether `colFactor` and `trailingUpdate` fuse into one dispatch per column
  via a workgroup-scoped scheme (only if the dispatch floor dominates P2a).
- Where `PreparedSaddle` and `ExternalSolveError` live (`linsolve.ts` vs a new
  `src/core/sobolev/prepared.ts`).
- Results-JSON schema for the IR convergence curve (per N, per rhs).
