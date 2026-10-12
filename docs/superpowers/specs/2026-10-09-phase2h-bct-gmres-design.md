# Phase 2H Design Spec — the hierarchical solver (BCT + preconditioned GMRES)

**Issue:** utof/repulsive-test2#58. **Epic:** #51. **Supersedes the solve architecture of**
`docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md` (`[DESIGN]`), which is marked
stale against #51 by commit `2a01332`.

**Status:** pre-registered. Kill gates are in §7 of THIS document, not in a plan
(CLAUDE.md: "Pre-registered kill gates go in the spec, not the plan").

**Provenance of every number below.** This spec quotes only values re-derived from the raw JSON in
`.superpowers/sdd/2026-09-11-phase2h-queue/scratch/`. Three measurement reports stand behind it
(#57 cost spike, #59 gate derivation, #64 θ decision), and all three contain numbers that their own
raw data contradicts. Those are listed in §10.3 and are **not** quoted here. Two digests did the
cross-checking: `reports/digest-57.md` (§11: eleven report-vs-raw disagreements) and
`reports/digest-59.md` (§0: six defects). When this spec and a report disagree, the spec is keyed to
the JSON and wins.

---

## 0. State detection (fresh session)

Run these before touching anything; they tell you where the branch is.

```bash
git log --oneline -3                      # feat/phase2h-spec branches from main @ 9dd5551
bun test                                  # 402 pass / 0 fail expected (397 pre-2H + 5 bench/stats)
bunx tsc --noEmit                         # clean
ls src/core/sobolev/hierarchical/ 2>/dev/null   # absent ⇒ nothing in src/ yet
wc -l .superpowers/sdd/2026-09-11-phase2h-queue/scratch/*.ts | tail -1   # 3823 — the PROTOTYPE
ls .superpowers/sdd/2026-09-11-phase2h-queue/scratch/e1b-left-f64*.json   # task 0a output, present
git checkout -- bench/results/gpu-phase2a-cr.json   # known churn, @issue utof/repulsive-test2#47
```

`codebase-memory-mcp` project name is `media-vboxuser-G-samsung-0utoffiles-code-repulsive-test2`.
`deepwiki` is not installed — use `gh` or WebFetch for GitHub.

---

## 1. Scope

### 1.1 What this phase replaces

Phase 2 assembled a dense `3n × 3n`-equivalent Sobolev operator and factored it (Cholesky on the
shifted block, then Schur + Woodbury for the constraints). Phase 2H **never forms the operator**. It
applies it hierarchically and solves iteratively.

This is a replacement, not an acceleration. `[YSC §6.3]` (arXiv:2006.07859) states the consequence
directly:

> "Since the hierarchical matrix-vector multiply does not build an explicit matrix, we use an
> iterative method to solve our linear systems."

Once the operator is hierarchical **there is no `A` to factor**. BCT and dense Cholesky are mutually
exclusive; the two cannot be stacked. `[DESIGN §1:118]` listed "Multigrid / BCT preconditioning" as
an explicit out-of-scope non-goal; `[DESIGN]:119-127` already carries the correction ruling that
non-goal contradicted by the scale target. **This spec executes that correction.**

### 1.2 Why the dense path cannot reach the target

CLAUDE.md records the target: **N=960 is the FLOOR; the goal is 1,000–30,000 vertices.**

| ceiling | value | source |
|---|---|---|
| dense f32 `A_σ` vs WebGPU's default 128 MiB `maxStorageBufferBindingSize` (`4N²` bytes) | **N ≈ 5 793** | `[DESIGN]:461-465`. **NOT enforced in code** — `GPU_SOLVE_MAX_N` exists only in plan/spec prose (§10.3); CLAUDE.md calls it "enforced", which is wrong |
| dense f32 `A` at N=30 000 | **3 433 MiB** — unallocatable on this GPU | `-prenear.json` `memMb.denseF32Equivalent` |
| dense factorization at N=30 000 | ~9e12 flops | `[DESIGN]` |
| numerics ceiling of the current kernel + fan-in + extended accumulator + GMRES-IR | **N ≈ 6 900** | `#51`, `research-precision-r1.md:562-609` |

Numerics bind before memory does. **No amount of tuning takes the dense path to 30 000.**

### 1.3 In scope

1. The hierarchical operator `A` (BVH + block cluster tree), f32 and f64 accumulator variants of one
   traversal.
2. The `Ã⁻¹ = (−Δ)⁻¹ L_{2−s} (−Δ)⁻¹` preconditioner, including the `L_{2−s}` operator that the
   existing 30k projection never budgeted.
3. **Left**-preconditioned GMRES (§5.3 — right preconditioning is measured unusable).
4. A mixed-precision scheme in which the f32 hierarchical apply is the **inner** solver, never the
   solver (§6).
5. The seam change from "matrix + factor" to "matvec + iterate" (§4).
6. The constraint fork for `k = O(n)` (§8, #62).
7. Pre-registered kill gates H1–H5 + R1 (§7).

### 1.4 Explicit non-goals

- **Changing the CPU f64 reference path.** `assembleAFlat`, `ldltFactor`, `lineSearchStepSet`,
  `projectOntoConstraintSet` stay byte-for-byte unchanged. The hierarchical path is selected at the
  seam; absent ⇒ bit-identical, with the golden suites as backstop. This is `[DESIGN §1]`'s rule and
  it survives unchanged.
- **Deleting the dense path.** It is the f64 reference every gate in §7 measures against, and it is
  the fallback below the crossover where the RS preconditioner is *worse than none* (§5.2). Park it,
  do not remove it.
- `df64` emulation, `f16`. Untouched by #51.
- GPU implementation of the hierarchical apply. **This spec specifies and gates the CPU
  implementation.** Every GPU number in the source reports is extrapolated from zero GPU runs
  (§10.1); promoting them to a GPU work item requires its own measurement, and #63 means it must be
  wall-clock. The architecture is chosen so the GPU port is possible (array-of-structs near field,
  CSR-like block lists), and §9.4 records what the port must not break.

---

## 2. The decision, and what is actually settled

Three axes, with their current status stated honestly:

| axis | status | the number that settles it |
|---|---|---|
| **memory** | **SETTLED** | ~33–34 MiB f32 for `A` at n=30 000 (two independent estimates, §3.4). Near pairs scale **n^0.9796**, i.e. linear. The 128 MiB binding limit is never approached on trefoil-class geometry. |
| **accuracy** | **RESOLVED, guarded** | The hierarchical operator is spectrally equivalent to `A`: A-relative defect **2.44–11.81 × δ₂** over all 15 (n, θ) cells, where a random perturbation of identical spectral norm-distance gives **1.83e3–5.22e6 × δ₂** — a ratio of up to **4.4e5×** at n=4 000. End to end the hierarchical direction takes **99.69–99.91 %** of the exact operator's energy decrease. §2.2. |
| **cost** | **PLAUSIBLE, not settled, and ~25 % worse than previously reported** | 194 ms per matvec at n=30 000 on one CPU thread (measured, loaded box). Beyond that, 12 threads and GPU are extrapolated; the near-field precompute (§3.3, ~3.5 s/step) was never counted; and the iteration multiplier is **32, not 26** — the 26 was measured on the preconditioning side this spec forbids (§5.3.1a, newly measured). |

### 2.1 The #57 ⊕ #59 collision, and why it dissolved

Read literally, the two measurement reports killed this phase: #57 measured BCT truncation
`4.039e-2` at θ=0.5 (`out-accuracy.json` row θ=0.5, `relErrRandom`), and #59 measured the tolerable
perturbation falling as `n^{-2.76}` to `1.1146e-5` by n=960 (`e3-bisect-large.json`,
`betaCos90Min`). Four thousand times over bar.

**Both measurements are correct. The inference was not.** #59 perturbed the exact `A` with a
**random symmetric** `E` (`e2_inexact.py:28-50 bct_proxy`), and for a generic `E` the damage to the
descent direction scales as `κ₂(A_σ)·δ`. The BCT does not perturb `A`; it **replaces** it with a
different, spectrally equivalent operator. #59 flagged its own model as "the weakest link" and
predicted the real BCT's tolerance would come out *lower*. **That prediction is falsified by a
factor of 655× to 921 600×** (#64, `out-64-beta-struct.json`).

`δ` is the wrong currency for this gate. That is the finding, and §7 is built on it.

### 2.2 The operator is SPD by construction, not by luck

`A_bct = Mᵀ(diag(K1) − K)M + Eᵀ(diag(K⁰1) − K⁰)E`. Every entry of the hierarchical `K` is
non-negative (near entries are kernel values or an exact 0 for adjacent pairs; far entries are
`ℓ_I ℓ_J · k(x_A,x_B) > 0`), and `K1` is computed by applying **that same hierarchical operator** to
the ones vector. So `diag(K1) − K` is exactly the graph Laplacian of a non-negatively weighted graph
⇒ PSD with nullspace the constants, **regardless of how coarse the admissibility is**. `Mᵀ(·)M` and
`Eᵀ(·)E` preserve PSD.

Measured consequences, every (n, θ) in n=240…4 000 × θ∈{0.5, 0.25, 0.1}:

- **zero negative eigenvalues at every one of the 15 cells** (`negEigCount` = 0 throughout);
- **`numpy.linalg.cholesky(A_bct,σ)` succeeds at every cell** (`choleskyBctSigma` = OK ×15);
- symmetry defect `max|A−Aᵀ|/max|A|` ≤ **1.63e-16** (`symDefect`, max over all cells);
- the hierarchical operator annihilates constants **better than `assembleAFlat` does**:
  `‖A_bct·1‖₂/‖A‖₂` = **2.19e-15 … 2.59e-14** against the exact operator's own
  `‖A·1‖₂/‖A‖₂` = **7.96e-15 … 1.27e-13** (`bctOnesRel` vs `AonesRel`), i.e. roughly 4× smaller at
  every n.

> **Three different "A·1" quantities exist and they must not be conflated.** (i) The **apply**,
> `max_i |(A_bct·1)_i|`, is **exactly 0** — but that is checked only by the `W2` assertion in
> `run-64-verify.ts:117` ("must be EXACTLY 0"), whose output is **console-only with no saved
> artefact**, the same provenance weakness as §5.4's 33.9 %. (ii) `bctOnesRel` in the saved JSON is
> the **densified** operator's normalised row sum — 2.19e-15…2.59e-14, correctly **nonzero**.
> (iii) `bctOnesAbs` is that same quantity unnormalised (3.9e-12…3.3e-8) and is not comparable
> across n at all.
>
> The exact zero in (i) is structural and must be asserted exactly; a zero in (ii) would mean the
> densifier is faking it. §11.3 makes (i) a standing test precisely because it currently is not one.
> @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §2.2

**No θ exists at which it stops being SPD**, and the argument above says none can. This is what
licenses calling it a different-but-valid Sobolev metric rather than a corrupted one.

> **Anchor.** `A_bct·1 = 0` holding *exactly for the apply* but **not** for assembled rows is not a
> bug and must not be "fixed": the apply's zero is structural (`M·1 = 0` because
> `du = (u[i₂]−u[i₁])/ℓ` on a constant is `0−0`, and `diag(K1)·1 − K·1 = 0` bit-exactly because
> `K1` comes from the same apply), while a row-sum computed from `N` separately-accumulated columns
> is roundoff. An exactly-zero row sum would mean the densifier is not doing what it claims.
> @see .superpowers/sdd/2026-09-11-phase2h-queue/reports/64-theta-decision.md §1.1, §4

### 2.3 One consequence for the shift

`tr(A_bct)/tr(A)` = **0.9891 … 1.0000**. Computing the rank-1 shift `σ = tr(A)/n` from the
hierarchical operator instead of the exact one moves it by ≤ 1.1 %.

**Ruling: `σ` is computed from the hierarchical operator's own trace.** Reason: the production path
has no exact trace to read, and a gate that silently depends on one is not reproducible at n=30 000.
The ≤1.1 % difference is recorded, not corrected. `A_σ = A + (σ/n)11ᵀ` with
`σ = tr(A_bct)/n`, matching `bench/gpu/k1-calibration.py:228-230`'s form.

---

## 3. The cost model

### 3.1 Where the time goes

Measured at n=30 000, trefoil (`torusKnot`), θ=0.5, leaf=8, `endpoint4` quadrature, near field
precomputed, **single CPU thread, f64, loaded box (1-min load average 35.56)**, min of 21 reps after
3 warmups (`out-bct-torusKnot-theta0.5-leaf8-endpoint4-prenear.json`, row n=30000):

| | ms | share |
|---|---:|---:|
| `farOnlyMin` | 13.360 | **7.1 %** |
| `nearOnlyMin` | 173.908 | **92.9 %** |
| `fullMin` (separately timed) | 194.193 | — |

**κ is a near-field constant, not a tree constant.** Any optimisation effort aimed at the tree is
aimed at 7 % of the work.

> Note the denominator: 7.1/92.9 is over `farOnly + nearOnly` = 187.27 ms, not over `fullMin`.
> Against `fullMin` it is 6.9 % / 89.6 %; the 6.9 ms gap is because `full` is a separate timed run.
> The split closes to within 3.6 % at n=30 000 only — at n=10 000 the parts sum to 32.4 ms against a
> full of 45.70 ms. Do not quote the split at small n.

### 3.2 Scaling — linear, and NOT `n log n`

| fit (5 sizes: n = 960, 2 000, 5 000, 10 000, 30 000; trefoil, θ=0.5, leaf=8) | exponent | R² |
|---|---:|---:|
| near pairs | **0.9796** | 0.9581 |
| far blocks | 1.1529 | 0.9716 |
| matvec ms (near precomputed) | **1.0464** | 0.9708 |
| memory (near precomputed) | 1.0015 | 0.9845 |

Other geometries agree on the near-pair exponent: tangle **0.9834**, circle **0.9463**.

The `n log₂ n` model mis-predicts by **±36.7 %** (`fits.torusKnot.nearPairs.nLogN.maxRelDev`
= 0.3673, constant 7.1899). **Use the linear model.**

> **Mechanism, and it matters for every gate in §7.** Per-vertex near pairs *oscillate*:
> 93.3 → 104.5 → 63.7 → 60.4 → 108.2 across those five sizes. The median-split BVH forces the leaf
> count to a power of two (128/256/1024/2048/4096), so mean leaf size wanders over **4.88–7.81** as
> `n` crosses powers of two. **A 5-point log-log fit through an oscillating series is a locator, not
> a law.** #64 ignored exactly this warning for a spectral quantity and had to retract a
> recommendation (§10.2).
>
> One correction to the source report: it attributes the oscillation to near pairs going as
> `leaf-size²`. Per *block* that is right, but the number of blocks per leaf falls as `1/leaf`, and
> per **vertex** the data show `κ_near / mean-leaf-size` = 12.4 / 13.4 / 13.0 / 12.4 / 14.8 —
> roughly **constant**, i.e. `κ_near` is linear in mean leaf size. The qualitative mechanism stands;
> the exponent does not.

### 3.3 The cost nobody budgeted: near-field precompute

Assembling the near field once per gradient step and reusing it across GMRES iterations is worth
**12.75×** on the matvec (2 476.30 ms on the fly → 194.19 ms precomputed, n=30 000). It is
mandatory.

**But the assembly itself costs 3.2–3.5 s at n=30 000 on one CPU thread.** Two measurements, from
the two different runs: **3 512.5 ms** (`-prenear.json` `build.precomputeNearMs` — the *same run*
that produced the 194 ms, so this is the internally consistent pair) and **3 182.9 ms**
(`out-64-kappa-n30000.json` `nearPrecomputeMs`, θ=0.25: 8 136 ms, θ=0.1: 5 574 ms — all noisy and
loaded). That is **~18× one matvec**, **it is not inside the 194 ms, and no prior projection counted
it.** Break-even against on-the-fly is **1.54** matvecs, so it pays for itself immediately at 26
iterations — but as a per-gradient-step one-time cost of ~3.5 s it dominates the 5 s budget
conversation on CPU all by itself.

**The spec budgets it as a separate line item** (§3.5) and §9.1 records it as the largest single
unquantified risk in the GPU port.

### 3.4 Memory

At n=30 000, trefoil, θ=0.5, near precomputed (`-prenear.json` `memMb`):

| | f64 | f32 (estimated) |
|---|---:|---:|
| tree | 4.1736 MiB | — |
| block lists | 1.6576 MiB | — |
| **`hierarchicalTotal`** | **60.2785 MiB** | **33.0–34.3 MiB** |

Two independent f32 estimates exist — **33.0 MiB** (reconstructed: indices/tree/blocks 5.83 + far
values 2.22 + two near arrays 24.8) and **34.34 MiB** (`out-64-kappa-n30000.json` `bytesF32Est`,
36 003 304 bytes). **Both are EXTRAPOLATED.** Quote the range, never a single figure. The source
reports give the second as "36 MB" by dividing by 10⁶; in MiB the two estimates agree to 4 %.

**What that estimate omits**, and the spec must carry each separately: `L_{2−s}` (+25 MiB f64 /
+12 MiB f32, DERIVED, never implemented); the GMRES Krylov basis; the 12-column working vector sets
(1.4 MiB per set at n=30 000); geometry — positions, tangents, lengths — (0.8 MiB).

> `memMb` in the raw JSON is genuinely MiB: `treeBytes` 4.1735687255859375 × 2²⁰ = 4 376 320 bytes
> exactly. `bytesF32Est` in #64's file is raw **bytes**. The two keys use different units, which is
> where the MB/MiB confusion in the source tables originates.

**Largest single storage binding** — the number that must stay under
`maxStorageBufferBindingSize` = **128 MiB**:

All figures re-derived from `out-64-kappa-n30000.json` (`nearPairs`, `bytesF32Est` — note that key
is **raw bytes**, which is how the source tables came to be quoted in MB):

| geometry | θ | near pairs | f32 total for `A` | largest single binding |
|---|---:|---:|---:|---:|
| trefoil | 0.5 | 3 245 868 | 34.34 MiB | **12.38 MiB** |
| trefoil | 0.25 | 6 785 670 | 64.89 MiB | 25.89 MiB |
| tangle | 0.5 | 9 681 766 | 89.23 MiB | **36.93 MiB** |
| tangle | 0.25 | 15 946 490 | **143.01 MiB** | 60.83 MiB |
| tangle | 0.1 | 24 886 020 | **218.35 MiB** | 94.93 MiB |
| circle | 0.5 | 659 598 | 11.85 MiB | 2.52 MiB |

> **Unit correction — the binding limit is in MiB and the source figures are not.** Both digests and
> #64's table report "13.0 MB / 38.7 MB" for the largest binding, which is `pairs × 4 / 10⁶`.
> `maxStorageBufferBindingSize` is **128 × 2²⁰ bytes**, so the comparison must be in MiB:
> `3 245 868 × 4 / 2²⁰` = **12.38 MiB** (trefoil) and `9 681 766 × 4 / 2²⁰` = **36.93 MiB**
> (tangle). Re-derived here from `stats.nearPairs` in the raw JSON. The conclusion is unchanged and
> the headroom is slightly *larger* than reported — but a 4.8 % unit error in the one number that is
> compared against a hard device limit is exactly the kind that survives into a kill gate.

> **Ruling: the two per-kernel near-value arrays are bound SEPARATELY, never concatenated.** Each is
> `nearPairs × 4 B`; concatenated they double. The source report's "the near-value array (25 MB f32)
> is the largest single buffer" is the *sum* of the two and is wrong as stated for a per-binding
> limit. Separate bindings also keep tangle-class geometry at 36.93 MiB where the concatenation
> (73.9 MiB) would start to crowd the limit.
> @see .superpowers/sdd/2026-09-11-phase2h-queue/reports/digest-57.md §7, §11 item 6

**Binding size is not the binding constraint; total device memory is.** Every per-binding figure
above sits comfortably under 128 MiB, including adversarial geometry at the tightest θ. But
`tangle` reaches **143 MiB at θ=0.25 and 218 MiB at θ=0.1** for `A` *alone*, ~240/250+ MiB once
`L_{2−s}` is counted. The Quadro RTX 3000's 6 GB absorbs that, so it is not a kill — but it is a
real cost, it is one more reason θ=0.1 is out (§4.3), and it means the θ=0.25 fallback is not free
on adversarial geometry either.

### 3.5 The per-iteration and per-step budget

One `A` matvec is **12 columns** of `K`, not 3 and not 1. `A` acts block-diagonally on the 3
coordinate components (`src/core/sobolev/layout.ts` `expandBlockDiag`); the high-order `B` has `M u`
R³-valued (a tangent direction) so it needs 3 `K`-applies per scalar component ⇒ 3×3 = **9
columns**; the low-order `B⁰` needs 1 per component ⇒ **3 columns**. `K1` and `K⁰1` are
geometry-only, computed once per build.

**The measured 194 ms already is the 12-column matvec.** A cost model treating one `A` apply as a
single scalar traversal is 12× optimistic on column count (the source report says "4× optimistic",
which is relative to a 3-column model). Because the precomputed near field is memory-bound, kernel
evaluation amortises across columns, so the true ratio is *below* 12× — **unmeasured; there is no
`ncols=1` or `ncols=3` timing in any JSON.**

Per preconditioned GMRES iteration at n=30 000, trefoil, θ=0.5, 1 CPU thread:

| term | ms | status |
|---|---:|---|
| `A` matvec (12 columns) | **194.19** | **MEASURED** |
| `L_{2−s}` (3 columns, same tree, same block lists) | ~43 | **DERIVED** as 3/12 of near work; never implemented |
| `2 × (−Δ)⁻¹` | **1.61** | **MEASURED** (0.804 ms each) |
| **per iteration** | **~239–241** | DERIVED |
| × **3** iterations ← **the production number** | **0.72 s** | **MEASURED** — see the ruling below |
| × 32 iterations (gate configuration, `τ_η = 1e-6`) | 7.7 s | calibration only — H3's instrument, never the production stop (§7.4) |
| × 42 iterations (τ_true=1e-4) | 10.1 s | MEASURED; the *backward-error* budget, not the production one |
| ~~× 26 iterations~~ | ~~6.3 s~~ | **superseded.** The 26 every prior projection used is the `rs/**right**/f64` count, which this architecture forbids (§5.3.1) |

> **Three different iteration counts have been conflated across this phase and they differ by ≥14×;
> this spec's own first draft charged production for the gate's stopping rule.** Now separated, all
> MEASURED (`e2-inexact.json` δ=0 left-prec; `e1-closer-0b-n*.json`):
>
> | what it is | count at n=960 | who needs it |
> |---|---:|---|
> | **direction quality** — `min cos∠_A ≥ 0.9` over 5 rhs | **3** (2 at n=480) | **production.** What a descent step actually requires |
> | gate currency — `η ≤ τ_η = 1e-6` | ~32 | H3 only (§7.4) |
> | backward-error floor — `argmin η` | 36–60 | H2's probe schedule (§7.3) |
>
> **Production needs a good direction, not a small residual.** MEASURED at the loosest stop
> (recurrence 1e-2): `min cos∠_A` = **0.9115 at n=480 in 2 iterations** and **0.9486 at n=960 in 3**,
> and the quantity **improves with n** — it crosses 0.9 between n=240 (0.885) and n=480, in the same
> direction as H4a's baseline, so n=30 000 is comfortably clear on that trend.
>
> **Why the gate needs τ ≤ 1e-4 while production does not.** `q > 0 ⇒ cos∠_A > 0` is a theorem (§7.6)
> and holds at **any** τ; what degrades at a loose stop is only its *sharpness* as a 0.9-classifier,
> and only at the small n where the solve has not yet reached 0.9 itself. So H4b′ runs per solve at the
> production stop as a **soundness** check, while the `cos ≥ 0.9` *calibration* happens at τ ≤ 1e-4
> where §7.6's confusion matrix is clean. Conflating the two inflated this budget by **14×**.
>
> **Consequence for what to optimise.** At 3 iterations the GMRES cost is 0.72 s against a MEASURED
> **3 513 ms** near-field precompute (§3.3) — so the per-gradient-step budget is **~83 %
> precompute**, and the iteration count stops being the thing worth tuning. The projection solves are
> *not* a third term: MEASURED `projectionIterations` is median **0**, max 2 at n=240 and 4 at n=480.
| **+ near-field precompute, once per gradient step** | **+3 513** | **MEASURED**, §3.3 |

`(−Δ)⁻¹` is **0.83 % of the matvec**. Drop it from the risk register — it is a Sherman–Morrison
cyclic-tridiagonal solve (Numerical Recipes §2.7 `cyclic`: two Thomas sweeps plus a rank-1
correction, no stored factorization, no pivoting), O(N) confirmed with `nsPerVertex` falling
67.4 → 30.6 over n=960…100 000.

> One caveat the spec carries into §9: the cyclic solve's relative residual **degrades with n** —
> 2.2e-10 at n=960 to 1.8e-7 at n=100 000 — attributed to the absence of pivoting. At n=30 000 it is
> 1.697e-9. Fine for a preconditioner; it is not a solve.

### 3.6 What is extrapolated

**Every GPU number in every source report is extrapolated from zero GPU runs.** The chain is:
`194 ms measured` → "40–200× for a bandwidth-bound kernel on a 336 GB/s Quadro RTX 3000" (assumed)
→ 1–5 ms/matvec on trefoil → ×3.29 (the *tangle* time ratio) → 3–16 ms → ×26 iterations (itself
superseded by **42**, §3.5) ×1.23 for
`L_{2−s}` → **0.10–0.51 s/solve**.

Three things make that chain weaker than it reads:

1. The "40–200×" is bandwidth reasoning with no kernel behind it, and the "20–25 % of peak bandwidth
   for a gather-heavy SpMV" is assumed.
2. The upper end of "3–16 ms" is an **adversarial-geometry** figure (tangle) applied to a
   representative-geometry claim.
3. The source report's own table labels 3–16 ms *per iteration* while its text, the issue comments,
   `cost64.py` and #64 all treat it as *per matvec* — a 1.23× discrepancy sitting inside a range
   quoted to two significant figures.
   **This spec defines it as per `A` matvec.**

It also excludes submission and readback latency entirely. **§7 therefore contains no timing gate**,
and §9.1 carries the GPU cost as an open risk with a named measurement protocol rather than a bar.

---

## 4. θ — the admissibility parameter

### 4.1 The ruling

**θ = 0.5 by default, guarded per-solve by H4b′ and end-to-end by H4c, with θ = 0.25 as the
runtime fallback — and the default is decided by H6 (§7.9), not asserted here.**

This is CLAUDE.md's "prefer swappable strategies over one hardcoded kernel" applied to a parameter
where five measured currencies and one unpredictable residual risk point in opposite directions. θ
is a runtime knob with a measured trip condition, not a compile-time constant.

> **The guard is NOT H4b, and this is a correction to #64's recommendation.** #64 recommends
> *"ship θ=0.5 with H4b as a mandatory per-solve guard"*, which requires H4b to enforce
> `cos∠_A ≥ 0.9` per solve. **That instrument does not exist**: §7.6 proves H4b's surrogate is
> `≥ 1` identically (algebra, plus 122/122 of the samples where it is defined, and 16 measured false
> passes with true `cos`
> down to 0.0281), so the 0.9 bar sits below the quantity's structural minimum and cannot be
> tripped. §7.6 further proves that no reference-free lower bound on `cos∠_A` from a single iterate
> exists at all. What survives — `min q > 0` — catches *catastrophic* failure soundly but **cannot
> distinguish θ=0.5 from θ=0.25**, and cannot see the `cosBound = 0.8234` worst case that is the
> whole θ risk.

**So the θ ruling is re-based on three instruments that do exist**, plus one pre-registered
measurement that can overturn it:

| instrument | what it covers | what it cannot see |
|---|---|---|
| **H4b′** `min q > 0`, per solve, reference-free | catastrophic direction failure; GREEN *certifies* `cos∠_A > 0` | anything in `0 < cos < 0.9`, incl. `cosBound = 0.8234` and the f32 wall |
| **H4c** `R_E ≥ 0.95` **and** accepted fraction `== 1.0`, per calibration | end-to-end energy delivery and line-search refusals; MEASURED separation **0.99691 vs 0.86027** | a θ difference — all 15 BCT arms pass |
| **H6** adversarial rhs at θ=0.5, n=960 and 2 000 | **exactly the gap the other two cannot**: the sub-0.9 generalized modes | — |

**H6 is the decision rule, and it is cheap** (the bad modes are already computed by
`scratch/run64_badmodes.py`): if `min cos∠_A < 0.9` on an rhs built from the sub-0.9 modes, the
default flips to θ=0.25 — the only θ with **zero** poorly-approximated modes at every n tested
(worst-case guarantee 0.9984) — and §4.4 is what that costs. Every `cos∠_A` in the corpus is against
**random** rhs, and a converged flow is precisely where the rhs stops being random, so this is the
measurement that matters.

**θ = 0.1 is off the table**: it buys nothing over 0.25 on any measured currency and costs 22 % more
near-field work, and on tangle-class geometry it reaches 218 MiB f32 (§3.4).

### 4.2 What says θ=0.5 works

All MEASURED, on `trefoil(n)` via the repo's own `assembleAFlat` as reference (cross-checked against
#59's independently assembled cache at **1.24e-16 / 1.44e-16** relative — two assembly paths, TS and
Python, agreeing to machine precision):

| currency | θ=0.5 | degrades with n? |
|---|---|---|
| `cos∠_A`, exact solve, min over 5 pinned rhs, n=240…4 000 | **0.9955 – 0.9984** | **no** |
| `cos∠_A`, stopped left-preconditioned GMRES @1e-4, vs the δ=0 baseline | within **0.0022** at every n | **no** |
| end-to-end ΔE / ΔE(exact) through the repo's own `lineSearchStepSet`, n=240/480/960 | **0.99691 / 0.99867 / 0.99909** | **no — it improves** |
| margin `β(cos≥0.9) ÷ the operator's own δ₂`, n=240/480/960/2 000 † | **5.01 / 4.94 / 4.76 / 5.16×** | **no — flat across a 66 000× change in κ₂(A_σ)** |
| A-relative defect ÷ δ₂ (all θ, n=240…4 000) | **2.44 – 11.81** | mild drift only |

> † **This row is evidence, never a bar.** It is a legitimate measurement — the operator tolerates
> ~4.8× more error along its own structured error direction than it actually has, and that ratio
> could have come out below 1 and did not. But as a *predicate* it reduces to `δ ≤ 4.8·δ`, because
> numerator and denominator scale together; and both inputs are bisected against a **dense**
> `A_σ + δ‖A_σ‖₂E`, so neither is measurable above N ≈ 5 793. §7.2 retires it along with every other
> δ-magnitude bar.

**The line search is discriminating, not rubber-stamping.** Median accepted `τ` is 0.0039–0.00098,
not 1, and the control arm — a random symmetric perturbation rebuilt at every step to the *same*
spectral norm-distance as that step's hierarchical operator — is rejected as
`not_a_descent_direction` **on step 0** at n=240 (`cos∠_A = −0.111530`) and at step 1 at n=480. Same
δ, opposite outcome. That control is what makes this section falsifiable.

### 4.3 What says θ=0.5 is risky

At θ=0.5, **0.4–2.2 % of the generalized spectrum sits below 0.9**, and **which n is worst is
unpredictable**:

| n | c₁ = λ_min(A_bct,σ ; A_σ) | # modes < 0.9 | # < 0.5 |
|---:|---:|---:|---:|
| 240 | 0.8597 | 3 | 0 |
| 480 | 0.8432 | 4 | 0 |
| 960 | **0.3634** | 21 (2.19 %) | **4** |
| 2 000 | **0.2886** | 18 (0.90 %) | 1 |
| 4 000 | 0.6979 | 17 (0.43 %) | 0 |
| 240–4 000 at θ=0.25 | 0.9726 → 0.9240 | **0 at every n** | **0** |
| 240–4 000 at θ=0.1 | 0.9974 → 0.9921 | **0 at every n** | **0** |

`c₁` at θ=0.5 **oscillates over a 2.98× range with R² = 0.216 against any power law** — it is not
decaying, it is bouncing, and it bounces with BVH leaf alignment (§3.2), not with size. The worst
value measured anywhere (0.2886 at n=2 000) gives a worst-case direction guarantee
`2√(c₁c₂)/(c₁+c₂)` = **0.8235**, under the 0.9 bar. At θ≤0.25 the worst guarantee over the whole
sweep is **0.9984**.

**You cannot buy this off with an n threshold, because there is no monotone trend to threshold on.**
Any rule of the form "θ=0.5 below N, θ=0.25 above" would be fitting noise. You *can* buy it off by
measuring the direction per solve — which is free, and which **§7's H4b′ does only for catastrophic
failure.** `min q > 0` certifies `cos∠_A > 0`, not `cos∠_A ≥ 0.9`, so it cannot see the
`cosBound = 0.8234` worst case that is this subsection's entire subject (§7.6 caveat 2). The
instrument that addresses *this* risk is **H6** (§7.9); §4.1 carries the ruling.

Two further caveats, recorded because they bound the ruling:

- **The cause of θ=0.5's bad modes is not established.** They are strongly localized (44–71 % of
  their mass in 5 % of the vertices) but sit where the curve is *farther* from itself than average
  (0.52–0.57 against a curve mean of 0.496) — so the obvious explanation, admissibility
  conservatism at self-approach regions, is **falsified by measurement**. What replaces it is a
  hypothesis with no evidence. Anyone planning to "fix θ=0.5 and keep the 2.09× saving" is opening
  an investigation, not applying a known patch.
- **Adversarial rhs untested.** Every `cos∠_A` above is against five *random* rhs, and a converged
  flow is exactly where the rhs stops being random. §9.2.

### 4.4 What θ costs

`κ_near` MEASURED at n=30 000 (`out-64-kappa-n30000.json`); everything right of it DERIVED from
§3.5's model:

| θ | κ_near (trefoil) | ×θ=0.5 | κ_near (tangle) | κ_near (circle) | largest binding, trefoil |
|---:|---:|---:|---:|---:|---:|
| **0.5** | **108.20** | 1.00× | 322.73 | 21.99 | 12.38 MiB |
| 0.4 | 134.79 | 1.25× | — | — | 15.43 MiB |
| 0.3 | 202.91 | 1.88× | — | — | 23.22 MiB |
| **0.25** | **226.19** | **2.09×** | 531.55 | 21.99 | 25.89 MiB |
| 0.1 | 275.19 | 2.54× | 829.53 | 65.92 | 31.49 MiB |

Geometry spread at θ=0.5 is **14.67×** (circle 21.99 → trefoil 108.20 → tangle 322.73), so the cost
half of this table is geometry-robust in *shape*. The accuracy half is **not** — every direction,
SPD and line-search result is on `trefoil(n)` alone, and `tangle`-class geometry is the obvious gap
(§9.2).

The fallback's price: θ=0.25 doubles the dominant term, moving the per-iteration cost from 241 ms to
501 ms and the extrapolated GPU solve from 0.10–0.51 s to **0.20–1.07 s**, i.e. out of the top of
#51's claimed 0.05–0.5 s band.

### 4.5 θ is not buying step quality

The end-to-end ΔE spread across **all** θ is **≤ 0.0008**, and the ordering is not even monotone (at
n=960 θ=0.5 beats θ=0.25: 0.99909 vs 0.99855; at n=240 θ=0.5 is last). **θ buys a worst-case
guarantee, not a measurably better step.** That is the whole content of the cost decision, and it is
why a cheap runtime guard dominates a conservative constant.

---

## 5. Architecture

### 5.1 The seam: from "matrix + factor" to "matvec + iterate"

The existing Phase-2 seam is nearly right already. `ExternalFactorization`
(`src/core/sobolev/linsolve.ts:415`) is a black-box `solve(rhs) → x`, and a GMRES iteration fits
behind it unchanged. **Two exported interfaces carry the dense matrix, not one** — and the first
draft of this section named only the first, which is the defect this subsection now leads with.

```ts
// src/core/sobolev/linsolve.ts:440-443 — TODAY
export interface PreparedSaddle {
    A: Float64Array;                            // ← 2H never forms this
    factor(C: number[][]): SaddleFactorization;
}
```

`A` is not decoration: `solveSaddleFrozen` (`:932-955`) packs it into `FrozenSaddleOperator.a`
(`:915-920`) and `structuredSaddleResidual` (`:860`, module-private) consumes it as a dense
row-major `n×n` buffer to compute the self-certifying residual. **So the residual needs a matvec
too, not a matrix.** That is the entire architectural delta.

**The second carrier, which the first draft missed.** `FrozenSaddleOperator` (`:915-920`) has a
**required** `a: Float64Array`, and it has *two* production consumers, not one
(`trace_path(solveSaddleFrozen, inbound)`):

| caller | file:line | covered by |
|---|---|---|
| `solveConstrainedGradientSetFrozen` | `src/core/sobolev/gradient.ts:127-133` | task 8 |
| **`projectOntoConstraintSet`** | **`src/core/sobolev/lineSearch.ts:259-267`** | **nothing, in the first draft** |

`projectOntoConstraintSet` has exactly two routes and neither works hierarchically: the frozen branch
needs `frozenOp.a`, and the fallback assembles a dense `assembleAFlat` **per projection iterate** —
3.4 GB at N=30 000, the very thing this phase exists to escape. Worse, §1.4 declares that function
"byte-for-byte unchanged". **So on the hierarchical path the Sobolev step's *projection* half had no
route at all.**

> **And `a` is not the real blocker — `fac` is.** On the dense path "frozen" means *factor once,
> back-substitute many times*, and `a` is retained only so each reuse can recompute
> `structuredSaddleResidual`. The actual solve goes through `fac`, a `SaddleFactorization`. **The
> hierarchical path has no factorization by construction** (§1.1), so `FrozenSaddleOperator` cannot
> be patched by making `a` optional: the whole *meaning* of "frozen" changes from "reuse the
> factorization" to "reuse the tree and the near-field precompute, and run a fresh GMRES per solve".
> That is a semantic change, and it is why the delta below is a discriminated union rather than two
> optional fields.
>
> **The cost of that change is measured and small**, which is why it is survivable: a hierarchical
> reuse solve is a full GMRES run rather than a back-substitution, but MEASURED
> `projectionIterations` is median **0** (max 2 at n=240, 4 at n=480), so the projection adds 0–4
> solves per τ trial, not a new dominant term. §3.5.

**The 2H delta — a discriminated union, not optional fields.** The first draft used `A?` + `op?`,
which **does not compile**: widening a required member is a breaking change for every reader of
`prepared.A`, and there are two —

```
query_graph → src/core/optimizer.ts:291        (the `saddle.A length mismatch` shape guard)
              src/core/sobolev/gradient.ts:127 (`a: saddle.A` into FrozenSaddleOperator)
```

Applying `A?: Float64Array` verbatim to a copy of the tree and running `bunx tsc --noEmit` yields
`optimizer.ts(291,24) TS18048: 'opts.saddle.A' is possibly 'undefined'` and
`gradient.ts(127,13) TS2322`, where the unpatched copy is clean (`"strict": true`). A union forces
both readers to handle both cases explicitly instead of silently accepting `undefined`:

```ts
// src/core/sobolev/linsolve.ts
// CORRECTED 2026-10-11 (A24's census): `SobolevOperator` ALREADY EXISTS, and not here — it is at
// `src/core/sobolev/hierarchical/operator.ts:20-23`, with `HierarchicalOperator` at `:30-33` and
// `makeHierarchicalOperator` at `:70-76`. Task 8 must IMPORT it, not re-declare it in `linsolve.ts`;
// a second declaration would give two structurally identical types that `tsc` accepts and that
// drift. The sketch below is kept for the shape only.
/**
 * Applies Ā to one flat 3n coordinate vector, block-diagonally over the 3
 * components. Present ⇒ the residual and every solve route through this instead
 * of a dense `A`; absent ⇒ today's dense path, bit-identical.
 * Why: the hierarchical operator is never assembled, so there is no `A` to read
 * (arXiv:2006.07859 §6.3). @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.1
 */
export interface SobolevOperator {
    applyA(u: ArrayLike<number>, out: Float64Array): void;   // 3n → 3n
    n: number;
}

/** Dense path — today's behaviour, bit-identical. */
export interface DensePreparedSaddle {
    kind: 'dense';
    A: Float64Array;
    factor(C: number[][]): SaddleFactorization;
}

/** Hierarchical path — no matrix, no factorization; `prepare` returns a reusable solver. */
export interface HierarchicalPreparedSaddle {
    kind: 'hierarchical';
    op: SobolevOperator;
    prepare(C: number[][]): HierarchicalSaddleSolver;
}

/** The 2H seam. Exhaustive `switch (saddle.kind)` at both readers. */
export type PreparedSaddle = DensePreparedSaddle | HierarchicalPreparedSaddle;

/**
 * The hierarchical analogue of a frozen factorization: the tree and near-field
 * precompute are reused across solves, each solve is a fresh preconditioned GMRES.
 * Why: there is no factorization to freeze (§1.1), so "frozen" means reuse of the
 * BUILD, not of a decomposition. @see §5.1
 */
export interface HierarchicalSaddleSolver {
    solve(rhsTop: ArrayLike<number>, rhsBottom?: ArrayLike<number>):
        { x: number[]; lambda: number[]; residual: number };
}
```

**`FrozenSaddleOperator` gets the same treatment** — a `kind`-tagged union whose hierarchical arm
carries `{ op, solver }` instead of `{ a, fac }` — and `solveSaddleFrozen` dispatches on it. Both of
its callers are then exhaustive, which is what gives `projectOntoConstraintSet` a route.

> **Migration is mechanical but must be stated, since `kind` is a new required member on a type an
> existing producer builds.** `src/gpu/saddle.ts makePreparedSaddle` adds `kind: 'dense'`;
> `optimizer.ts:291`'s guard becomes an **`if` on the hierarchical arm with the dense check as its
> `else if`** — *shipped form, `optimizer.ts:297-318`*:
>
> ```ts
> if (opts.saddle?.kind === 'hierarchical') {
>     if (opts.saddle.op.n !== vertices.length) throw new Error('sobolevStepSet: saddle.op.n mismatch');
>     if (opts.projectionMode !== 'frozen') throw new Error(/* …requires projectionMode 'frozen' */);
> } else if (opts.saddle && opts.saddle.A.length !== vertices.length ** 2)
>     throw new Error('sobolevStepSet: saddle.A length mismatch');
> ```
>
> (there is no `A.length` to check hierarchically, and `test/gpu/saddleSeam.test.ts:287-298` pins the
> current throw, so the dense message must not change); `gradient.ts:126-134` switches on `kind`.
>
> > **A25, RULED 2026-10-11 (task 8, R1) — this replaced a ternary, and the ternary was WRONG in a way
> > that no type error would have caught.** This box previously specified
> > `saddle.kind === 'dense' ? saddle.A.length !== n**2 : saddle.op.n !== n`. **An untagged object has
> > `kind === undefined`, which is `!== 'dense'`, so the ternary routes it to the HIERARCHICAL arm** and
> > throws a TypeError naming `opts.saddle.op`. Two things depend on it naming `.A` instead:
> > `saddleSeam.test.ts:289-290`, which pins `/saddle\.A length/`, and — the one that would have gone
> > unnoticed — **`saddleSeam.test.ts:342`'s worker falsifier**, whose comment records that posting a
> > plain `{}` is rejected "by a TypeError naming `opts.saddle.A`, whose text ALSO contains" the string
> > the loose-guard mutant would match. That comment is load-bearing: it is the evidence that a
> > mis-spelled `'saddle' in msg` guard cannot pass for the real one. **The ternary would have silently
> > invalidated a falsifier while every test stayed green**, because the dense arm is reached by the
> > *absence* of a tag, not the presence of one. The `if/else if` above keeps untagged ⇒ dense.
> > **Consequence for the census: fact 6's predicted breakage DOES NOT OCCUR and is withdrawn** —
> > `saddleSeam.test.ts:289-290` needed no edit, because narrowing `identityPrepared`'s declared return
> > to `DensePreparedSaddle` makes the spread carry `kind: 'dense'` on its own. **General rule this
> > generalises to: when a discriminated union's guard decides which arm an UNTAGGED value falls to,
> > write the predicate so the arm with the pinned error message is the default.**
>
> **CORRECTED 2026-10-11 (A24's census), then WITHDRAWN the same day by A25 — the predicted break
> DOES NOT OCCUR under the shipped `if/else if` guard above; it was a property of the retracted
> ternary only. MEASURED by task 8: `saddleSeam.test.ts` needed three literal edits and two return-type
> narrowings, but NOT line 289-290. Kept here because the reasoning is still the reason the guard has
> its shipped shape.** ~~This guard rewrite BREAKS `saddleSeam.test.ts:289-290` and
> the break is not a message change.~~ That test spreads `identityPrepared(a)` and overrides `A`,
> expecting `/saddle\.A length/`. Under the ternary above, a literal **without** `kind` does not
> take the dense branch — it reaches `saddle.op.n` and throws a **TypeError** instead of the pinned
> message. So the test-side literals MUST gain `kind: 'dense'` as part of this task (sites enumerated in
> §11.4; **the count is whatever `tsc` reports** — the census's prose said "five" while its own
> enumeration lists six, so trust neither number over the compiler), and
> §11.4's *"existing goldens, unmodified"* is true of `oracle/golden/*.json` and FALSE of the seam
> test files: `saddleSeam.test.ts:104-125`, `:129-144`, `:179-198`, `gradient.test.ts:173-193`,
> `cr.test.ts:363`, `ldlt.test.ts:337`.
>
> **EXTENDED 2026-10-11 by task 8-pre (`d3672c2`): the bit-level golden adds three more sites**, so the
> list above is six *pre-8-pre* sites and not a closed count. `test/sobolev/denseSaddleBitGolden.test.ts`
> breaks at compile time at its `densePrepared()` literal, its `op` literal in `runSolves()`, and its
> `.frozen.a/.C/.n` reads in T1b. **Allowed edits there are `kind: 'dense'` plus a narrowing, and
> nothing else: `test/sobolev/denseSaddleBitGolden.json` must stay byte-identical** — verify with
> `git show d3672c2:test/sobolev/denseSaddleBitGolden.json | cmp - test/sobolev/denseSaddleBitGolden.json`.
> Regenerating that fixture to turn a red test green is forbidden and its guarded regeneration path
> always ends red by design.

`structuredSaddleResidual` gains an operator-shaped overload. **Its dense branch keeps its current
loop and term order byte-for-byte** — that value is golden-gated, and the comment at `:855-859`
records that it was extracted verbatim from `solveSaddleFromA` for exactly this reason. The new
branch is a sibling, never a rewrite of the old one.

> **Anchor.** Do not "simplify" the two branches into one. The dense branch's operand order is
> load-bearing (bit-identity against committed goldens); the hierarchical branch cannot reproduce it
> because it never has the matrix entries. Two branches is the design.
> @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.1

This is the **fifth** sanctioned public-type change on top of `[DESIGN §4.4]`'s (i)–(iv). Declared
here, marked, for the same reason that list exists: so a fifth cannot appear silently. **It touches
four pre-existing files, not two** — `linsolve.ts`, `gradient.ts`, `optimizer.ts`, `lineSearch.ts`,
plus `src/gpu/saddle.ts` for the `kind: 'dense'` tag. §11.2 task 8's file list says so.

### 5.2 The preconditioner

`Ã⁻¹ = (−Δ)⁻¹ L_{2−s} (−Δ)⁻¹`, from Repulsive Surfaces `[RS §5.2]` (arXiv:2107.01664), which
replaced multigrid precisely to avoid a hierarchy. It transfers to curves: the authors' own
[Repulsor](https://github.com/HenrikSchumacher/Repulsor) is templated on `DOM_DIM`, and `DOM_DIM=1`
is curves.

**Exponent check passes, verified in source not assumed.** `src/core/sobolev/innerProduct.ts:52-54`:

```ts
const s = (beta - 1) / alpha;   // = 5/3 at alpha=3, beta=6
const sigma = s - 1;            // = 2/3
const distExp = 2 * sigma + 1;  // = 7/3
```

So `2 − s = 1/3 ∈ (0,2)` and the fractional operator is well posed. Two corollaries worth recording,
because both have confused a reader already: `2s = 10/3 = 3.3333`, which is why
`κ₂(A_σ) ∝ n^3.3338` measured (#59) matches theory to four digits; and the high-order kernel's
distance exponent is `2σ+1 = **7/3**`, *not* `2s+1 = 13/3` — which is what #57's `d^{-7/3}`
quadrature argument (§5.4) refers to.

**Mesh independence is measured, and it is the economic case for the whole phase:**

| | n=64 → 1 920 |
|---|---|
| `κ₂(A_σ)` unpreconditioned | 1.37e4 → **1.63e8** (`∝ n^3.3338`) |
| `κ₂(Ã⁻¹A_σ)` | **1.09765e4 → 1.12690e4** (`∝ n^0.0010`) |

Five orders of magnitude of conditioning growth removed, flat to within 2.7 % over a 30× size range.

> **SCOPE, A38 item 9 (2026-10-12): the preconditioned row above is the LAB `pinv((−Δ)⁺)` construction,
> not the shipped operator.** `task-6-review-math.md:171-178` measures both on the same `A_bct,σ` and the
> same hierarchical `L_f`: lab **1.09765e4** (the digits above) vs **shipped `massWeight=1`: 59.04** at
> n=64. The shipped operator is ~100× BETTER in absolute terms and **NOT flat** — 9c measures the worst
> four-shape `κ_excl` growing **n^+0.47 at c=30** and **n^+0.15 at c=35** (90.8 → 309.9 and 115.7 → 233.0
> over n = 64 → 4 000). **The "flat" claim does not transfer to the shipped path; the headroom does, and
> it is 48× larger than §6 quotes** (720× at n=4 000, not 14.9×). Do not rebuild anything on flatness.

> **Trap, measured, do not rediscover.** ~~The RS preconditioner is **worse than no preconditioner
> below n ≈ 223.**~~ **A28, RULED 2026-10-11 (task 7b): `n ≈ 223` is REFUTED by 2.8×. The crossover is
> bracketed in `n ∈ [72, 80]`.** The trap's *direction* survives — there is a small-n regime where RS
> does not pay — but the band is a third of the size claimed, and the original figure came from a
> wall-clock ratio contaminated by the residency effect A29 describes.
>
> Settled on **load-independent integers** instead of wall-clock: RS is cheaper ⟺
> `r < iters_none/iters_rs − 1`, both counted to the same `τ_η` on the same currency, where `r` is the
> cost ratio of one RS iteration to one unpreconditioned iteration. **At n=240 the break-even `r` is
> 8.0, and no `r` anyone has measured on this machine exceeds 2.82** — so across the whole
> `80 ≲ n ≤ 223` band that this trap called "worse", **RS is cheaper by 1.6–6.3×**. The wall-clock
> reading survives only at `n ≤ 72`. RS cost per iteration, exact from the counters:
> **1 × `A_σ` apply + 6 × `solveLaplacian` + 1 × `applyLowOrderColumns`**.
>
> The operational half of the trap is UNCHANGED and is the part that matters: **build the BCT before
> the preconditioner and you will measure a regression and blame the wrong component.** That stands on
> its own, independent of where the crossover sits. The dense path also remains the right answer at
> small n for the separate reason in §1 (N ≤ 5 793 fits a dense `A_σ`). Also: unpreconditioned CG
> is measured at **1 217 iterations at n=960** growing superlinearly, and Jacobi buys 9.6 %. **Build
> the BCT before the preconditioner and you will measure a regression and blame the wrong
> component.** The preconditioner lands first or together, never after.
> @issue utof/repulsive-test2#51 (trap 1)

`L_{2−s}` is 3 columns on the same tree and the same block lists as `A`, costing ~+25 % of the near
work. **It has never been implemented and never been measured** — its +43 ms is derived from a 3/12
column ratio, and the original 30k projection did not count it at all.

### 5.3 GMRES — left preconditioning, and never trust the recurrence residual

**Left preconditioning is mandatory.** The reason is the **side × precision interaction**: under an
f32 matvec, right-preconditioned `cos∠_A` degrades **0.9947 → 0.5190 → −0.1362** across n = 240 →
480 → 960 — it crosses zero, so the "descent" direction points uphill — while left holds
**0.9968 / 0.9995 / 0.9991**, flat. Production is f32-inner by construction (§6), so this is the
production regime, not a corner. Separately and independently of precision: **never trust the
recurrence residual** (second ruling below, untouched).

> ⚠️ **READ THE CORRECTION BOX BELOW BEFORE QUOTING THIS SECTION.** The mechanism originally stated
> here — *"right-preconditioned f32 gives `cos∠_A = 0.142264` at n=480 while its recurrence residual
> reads a healthy 1e-4"* — is **RETRACTED** (it is an f32 artefact of the lab harness, not a property
> of the side), and the cost half of the ruling (*"left is 6–20× cheaper"*) is **WITHDRAWN**. The
> ruling itself survives on the figures in the paragraph above. Do not quote `0.142`, `5–7 iterations`,
> `36–39`, or `6–20×` from this spec in any form other than as a retraction.

> **CORRECTED 2026-10-11 — the ruling SURVIVES, the stated mechanism is RETRACTED, and the correct
> mechanism is stronger.** Blind reviewer B could not make the right-preconditioning knock-out trip on
> the shipped operator in 14 configurations (`min cos∠_A` never below 0.978), which was correct and is
> what prompted the re-derivation below.
>
> **The 0.142 is not a property of the preconditioning side. It is a property of f32.** The source run
> is `e2-inexact.json` → n=480 → `side: 'right'`, **`delta: 0.0`** — and `delta = 0.0` does **not**
> mean exact arithmetic. `e2_inexact.py:120-122` reads
> `Ad = bct_proxy(A, delta, sd)` / `Ad32 = Ad.astype(np.float32)` /
> `op = lambda v: (Ad32 @ v.astype(np.float32)).astype(np.float64)`: `bct_proxy` returns `A`
> **unchanged** at `delta = 0`, but the matvec is cast to **f32 unconditionally**. So the figure this
> ruling rested on was measured in f32 and attributed to the side.
>
> **MEASURED, the 2×2 that disentangles them** (`scratch-f5/side_f32_confound.py`, same lab, same 5
> pinned seeds, same `maxit = min(400, 2n)`, same recurrence stop 1e-4; `cos∠_A` median, min over
> seeds in brackets):
>
> | n | side | `f64` iters | `f64` rec | `f64` true | **`f64` cos∠_A** | `f32` iters | `f32` true | **`f32` cos∠_A** |
> |---:|---|---:|---:|---:|---:|---:|---:|---:|
> | 240 | right | 36 | 7.92e-05 | **7.92e-05** | **1.000000** [1.000000] | 37 | 6.68e-02 | 0.994706 [0.977649] |
> | 480 | right | 38 | 8.64e-05 | **8.64e-05** | **1.000000** [1.000000] | 45 | 1.162e+00 | 0.519022 [0.056857] |
> | 960 | right | 39 | 7.91e-05 | **7.91e-05** | **1.000000** [1.000000] | 142 | 3.263e+01 | **−0.136189** [−0.273369] |
> | 240 | left | 6 | 9.13e-05 | 6.158e-01 | 0.996800 | 6 | 6.147e-01 | 0.996824 [0.990811] |
> | 480 | left | 5 | 7.00e-05 | 6.620e-01 | 0.999512 | 6 | 6.605e-01 | 0.999536 [0.998877] |
> | 960 | left | 7 | 7.34e-05 | 6.301e-01 | 0.999664 | 7 | 6.409e-01 | 0.999123 [0.998998] |
>
> **In f64, right preconditioning is not merely adequate — it is exact.** `cos∠_A = 1.000000` for
> **5/5 seeds at every n**, and `rec == true` to five significant figures, precisely as the theory in
> this section's second paragraph requires. Nothing is destroyed. The original sentence
> *"the direction is destroyed"* described an f32 effect.
>
> **What makes left mandatory is the side × precision INTERACTION, and it is monotone in n toward the
> target.** Right-preconditioned `cos∠_A` under an f32 matvec degrades **0.9947 → 0.5190 → −0.1362**
> over n = 240 → 480 → 960 — it **crosses zero**, i.e. the "descent" direction points uphill — while
> left holds **0.9968 / 0.9995 / 0.9991**, flat. Mechanism: right preconditioning recovers
> `x = Ã⁻¹u`, so the f32 matvec perturbation is amplified by the un-preconditioning step, and the
> quantity GMRES minimises (the residual of the `u` system) is not the quantity the application needs.
> Left never forms `u`. **Production is f32-inner by construction (§6), so this is the production
> regime, not a corner.**
>
> > **WITHDRAWN the same day, by the same confound this section is about — the cost half of the
> > argument was invalid.** It read: *"Cost agrees: left needs 5–7 iterations at every n while right
> > needs 36–39 in f64 and 37 → 45 → 142 in f32 — left is 6–20× cheaper and its count is flat where
> > right's grows."* **Those counts are not comparable.** They all stop at a *recurrence* residual of
> > 1e-4, and for left that is not the true residual: the table above has `left/f64` at n=960 stopping
> > with a recurrence residual of 7.34e-05 and a **true relative residual of 6.301e-01** — an
> > understatement of **8 584×** — while `right/f64` stops with the two equal to five digits. So
> > "6–20× cheaper" compares left stopped early against right stopped properly. **MEASURED on an
> > equal-TRUE-residual basis the two sides are within 1.00–1.26×, growing with n** (trefoil, f64,
> > task 6e §5.LR, whose independent figure for left's recurrence understatement at n=1 920 is
> > 26×…2e4×). **There is no iteration-count argument for either side.** I made the identical error
> > the first sentence of this section makes — quoting two different residuals as if they were one
> > quantity — in the act of correcting it.
>
> **So the whole case for left is f32 direction robustness, and that is sufficient.** At n=960 under an
> f32 matvec left reaches `cos∠_A = 0.9991` and right reaches **−0.1362**. **Stated limit, because it
> is NOT measured:** left's 0.9991 is at *its* stopping point (7 iterations, true residual 0.63) and
> right's −0.1362 at *its* (142 iterations), so the equal-work and equal-true-residual comparisons **in
> f32** do not exist yet. What makes the asymmetry decision-grade regardless is the mechanism — right
> amplifies the f32 perturbation through `x = Ã⁻¹u`, left never forms `u` — plus the **sign change** and
> its monotone growth in n. **Task 10 owns closing the equal-accuracy f32 comparison**; 6e could not,
> because `gmres_lab.gmres` does not return the iterates it needs.
>
> Note also what A18 implies about the loose stop: `[YSC]`'s own standard is *"we need only a
> reasonable intermediate descent direction"*. A left solve stopping at a 0.63 true residual with
> `cos∠_A = 0.9991` may be exactly the right trade — but that is §7.9's question, not this one's.
>
> **Why reviewer B could not reproduce it, and this is a measurement lesson not a disagreement.** Its
> f32 configuration was an `F32`-variant `prepareA` — an f32 **operator** with f64 outer algebra —
> whereas the lab casts the **input vector** to f32 on every matvec. The lab's is the harsher and more
> faithful simulation of an f32 GPU kernel. Memory `simulate-bugs-at-target-precision` names exactly
> this: a simulation that is insufficiently f32 reports the defect as absent. **How f32 the production
> BCT path really is therefore decides where `cos∠_A` crosses zero, and that is a task-10
> measurement, not an assumption.** The *direction* of the asymmetry is not in doubt.
>
> **The second ruling in this section is untouched and is now independently confirmed.** `left/f64`
> stops at a recurrence residual of 7.3e-05 with a **true** relative residual of **6.301e-01** at
> n=960 — in exact arithmetic. The recurrence residual is not a convergence criterion, and that has
> nothing to do with precision.

**The recurrence residual is not a convergence criterion.** Left-preconditioned, it measures
`‖Ã⁻¹r‖/‖Ã⁻¹b‖`, which under-reports the true relative residual by **3.9e3–6.6e3×** even in exact
arithmetic. Worked example, n=960, δ=0, stopped at 1e-4: recurrence reads **1.0e-04**, the true
relative residual is **6.378e-01**, and the backward error is 6.733e-06. All three are "the
residual" and the outer two differ by **five** orders of magnitude (6.378e-01 vs 6.733e-06).

**Rulings:**
- Report the recurrence residual as a **labelled non-gate diagnostic**, never as the convergence
  value. §7's gates recompute from `b` and an exact matvec.
- The stopping tolerance is a **parameter**, recorded per solve, never a constant buried in the
  loop. H2 and H3 pin different values (§7) and both must be reachable.
- `maxit` is recorded alongside the iteration count so a censored count is detectable (§7, H3's
  INVALID predicate).

### 5.3.1 The iteration count is NOT settled, and the quoted figure is from the wrong configuration

Every projection to date — #51's, #57's, #64's cost table, and §3.5 above — multiplies the
per-iteration cost by **26**, sourced to "GMRES iterations 21–26, flat n=64 → 960". Re-derived from
the raw JSON, that figure does not support the architecture this spec mandates. **Three problems,
found by reading `e1_gmres.py` and `e5_growth.py` against their own output.**

**(a) The two experiments count iterations to different residuals, ~10× apart.**

| | stopping quantity | source |
|---|---|---|
| **E1** | the **TRUE** relative residual `‖b − Ax_k‖/‖b‖` vs the exact f64 operator | `e1_gmres.py:30-32` — `tr = r["true"]`, then `first_below(tr, t)` |
| **E5** | the **RECURRENCE** residual `\|g[k+1]\|/β₀` | `e5_growth.py:38-41` via `e2_inexact.gmres_stopped` |

RS-preconditioned medians over the 5 pinned seeds:

| | n=64 | 120 | 240 | 480 | 960 |
|---|---:|---:|---:|---:|---:|
| E1 true-residual, to 1e-2 (`rs/right/f64`) | 21 | 23 | 23 | 26 | 25 |
| E1 true-residual, to 1e-4 | 29 | 33 | 36 | 38 | 39 |
| **E5 recurrence, to 1e-2** | **2** | **2** | **2** | **2** | **3** |
| **E5 recurrence, to 1e-4** | **8** | **7** | **6** | **6** | **7** |
| E5 recurrence, to 1e-8 | 29 | 33 | 25 | 20 | 17 |

**(b) The quoted 21–26 is RIGHT-preconditioned, and §5.3 forbids right preconditioning.**
`e1_gmres.py:62-67` enumerates exactly six cases — `none/right/f64`, `none/right/f32`,
`jacobi/right/f32`, `rs/right/f64`, `rs/right/f32`, `rs/left/f32`. **There is no `rs/left/f64` case
anywhere in E1**, so the left-preconditioned f64 count — what the production reference path actually
does — has never been measured. The one left case, `rs/left/f32`, gives to-1e-2 counts of
**25 / 28 / 28 / 33 / 35**: a **×1.4 increase** across n=64→960, *not* flat, and censored at 1e-4
for n ≥ 480.

**(c) The recurrence residual's under-reporting factor grows as `n^3.06`.** `recTrueGapMax`
(true ÷ recurrence at the last probed iteration) for `rs/left/f32`:

| n | 64 | 120 | 240 | 480 | 960 |
|---|---:|---:|---:|---:|---:|
| gap | 4.372e8 | 1.404e9 | 1.142e10 | 2.056e11 | **1.717e12** |

**3 926× growth over n=64→960, fitted exponent +3.056** — tracking `κ₂(A_σ) ∝ n^3.3338`. So the
left-preconditioned recurrence residual is not merely a constant ~6 000× optimistic (§5.3's
headline, which is the *median* gap): **the optimism itself grows as n³.**

> **Ruling, and it is the single most consequential one in this document.** An iteration count
> measured against the recurrence residual gates a quantity whose **stopping criterion gets
> monotonically easier as n grows**. The flat-or-falling RS count (29/33/25/20/17 at τ=1e-8) is
> partly an artefact of that, not evidence of mesh independence.
>
> **This is precisely the error #59 retracted for H1's β** — *"a relative criterion cannot gate a
> quantity whose anchor is degrading"* — committed against a different anchor that degrades ~500×
> faster. §7's H3 therefore **may not count to a recurrence-residual tolerance.** It counts to a
> true-residual or backward-error tolerance formed with the exact χ=0 matvec (one exact matvec per
> probe, gate mode only — H1/H2 already require that entry point), or it gates the gap's growth
> exponent as an explicit companion property. §7 states which.
>
> @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.3.1

### 5.3.1a The missing measurement, now made

`rs/left/f64` was run while authoring this spec — `scratch/e1b_left_f64.py` →
`scratch/e1b-left-f64.json`, same `gmres` from `gmres_lab.py`, same five pinned seeds, same
tolerances, counts against the true residual exactly as `e1_gmres.py:30-32` does.

**`rs/right/f64` was re-run in the same script as a reproduction control, and it reproduced E1's
published row exactly at all five sizes (21 / 23 / 23 / 26 / 25).** Without that the new numbers
would be worthless; it is asserted in the script and recorded as `controlReproducesE1: true`.

Median iterations to a **true** relative residual, over the 5 pinned seeds:

| case | n=64 | 120 | 240 | 480 | 960 | growth |
|---|---:|---:|---:|---:|---:|---:|
| to 1e-2, `rs/right/f64` *(control — E1's published row)* | 21 | 23 | 23 | 26 | 25 | ×1.19 |
| **to 1e-2, `rs/left/f64`** ← **the mandated architecture** | **25** | **27** | **26** | **32** | **32** | **×1.28** |
| to 1e-2, `rs/left/f32` | 25 | 28 | 28 | 33 | 34 | ×1.36 |
| to 1e-4, `rs/right/f64` | 29 | 33 | 36 | 38 | 39 | ×1.34 |
| **to 1e-4, `rs/left/f64`** | **29** | **35** | **38** | **41** | **42** | **×1.45** |
| to 1e-4, `rs/left/f32` | 29 | 35 | 40 | **—** | **—** | *cannot reach 1e-4 at n≥480* |

**Three results, all MEASURED:**

1. **The mandated configuration costs ~28 % more iterations than the quoted one.** At n=960, left
   f64 needs **32** to 1e-2 where right f64 needs 25. **§3.5's budget multiplier is 32, not 26** —
   and **42** if the solve must reach 1e-4.
2. **It is mildly growing, not flat** (×1.28 to 1e-2, ×1.45 to 1e-4 over a 15× size range). The
   "flat in n" claim was a property of the right-preconditioned row.
3. **In the honest currency, mesh independence holds anyway.** `G(n) = iters(2n)/iters(n)` on the
   three **true** doublings, left/f64, to 1e-4: **1.0857 / 1.0789 / 1.0244** — under 1.09, and
   *decreasing*. Against baselines of 2.0211 (none) and 2.0518 (Jacobi). This is a better-founded
   H3 than either candidate derived from the recurrence residual, and §7 uses it.

**And the recurrence residual is unusable regardless of precision.** `recTrueGapMax` for `left/f64`
is 2.89e4 / 8.37e4 / 7.52e5 / 8.94e6 / 3.22e7 — **exponent +2.59**. The n³ degradation in §5.3.1(c)
is a property of *left preconditioning*, not of f32. Note also that for `right/f64` the recurrence
and true counts are **identical** (21/23/23/26/25 both ways), which is the documented exact-arithmetic
property of right preconditioning — and precisely why the right-preconditioned row looked so clean.

> **The trade this spec is making, stated plainly.** Right preconditioning gives a cheaper iteration
> count *and* a recurrence residual that means what it says. It is nonetheless **forbidden**,
> because at f32 it destroys the descent direction. Left preconditioning costs ~28 % more iterations
> and a residual that must be recomputed externally. **That is the correct trade**: a 28 % cost is
> recoverable, a silently wrong direction is not.
>
> > **CITATION CORRECTED 2026-10-11.** This paragraph cited `cos∠_A = 0.142` at n=480; §10.2 forbids
> > quoting that figure, which §5.3's correction box shows is an **f32 artefact of
> > `e2_inexact.py:120-122`, not a property of the side** — in f64 right preconditioning gives
> > `cos∠_A = 1.000000` at every n. The live figure is **`cos∠_A = −0.1362` (median, −0.2734 worst)
> > for `side: 'right'` under an f32 matvec at n=960**, against left's 0.9991: a sign change, monotone
> > in n. **The conclusion of this paragraph is untouched and its "~28 %" was RIGHT ALL ALONG** — §5.3.1a
> > measured the equal-accuracy cost, and task 6e's independent left-vs-right redo puts it at
> > **1.00–1.26×**, i.e. the same ≤28 %. That is also what falsifies the "left is 6–20× cheaper" claim
> > briefly made in §5.3 and withdrawn there: **the contradiction was detectable from this paragraph
> > alone, with no new measurement.**
> @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.3.1a

### 5.3.2 The f32 wall, measured directly rather than extrapolated

`e1-gmres.json` `trueFloorMin` — the smallest true relative residual the solve ever reaches:

| n | 64 | 120 | 240 | 480 | 960 |
|---|---:|---:|---:|---:|---:|
| `rs/left/f32` | 6.68e-7 | 3.56e-6 | 2.43e-5 | 5.26e-4 | **5.04e-3** |
| `rs/right/f32` | 6.07e-4 | 4.05e-3 | 4.32e-2 | 2.88e-1 | **6.30e-1** |
| `rs/right/f64` | 1.77e-12 | 1.40e-11 | 1.61e-10 | 2.26e-9 | 2.94e-8 |

**At n=960 the f32 solve's true residual never gets below 5.0e-3**, and right-preconditioned f32
never below 0.63. This is a direct measurement, it needs no extrapolation, and it bites **much
earlier than #59's superseded n≈3 310 crossover**. It is independent corroboration of §6's ruling that
f32 is an inner solver only — and it suggests the mixed-precision switch belongs well below n=2 000,
not at the wall.

### 5.4 The near field must use the repo's quadrature

The hierarchical operator's near field uses the repo's **4-point endpoint** rule, matching
`assembleAFlat`. It must not use `[YSC]` Eq. 32's 1-point midpoint rule.

**The two disagree by 33.9 % at n=960 with no BCT involved anywhere** — it is a direct comparison of
the O(m²) double sum under the two rules on `torusKnot(960)`, pinned to `assembleAFlat` at
~1.4e-15. (32.6 % at n=400; two points, slowly growing rather than converging.) The mechanism: for a
nearly-adjacent disjoint edge pair the closest endpoint pair is about half the midpoint distance,
and the kernel goes as `d^{-7/3}` (= `d^{-(2σ+1)}`, §5.2).

**Mandated:**
1. Near-field kernel = the repo's 4-point endpoint rule (4 `sqrt` + 4 `pow` per pair vs 1).
2. The far field has no choice — a rank-1 lump *is* a midpoint rule. Accepted.
3. Every golden/oracle comparison is against `assembleAFlat`, **never** against an Eq.-32-verbatim
   reference.
4. A V0-style bit-level pin (`exactA(endpoint4)·u` vs `assembleAFlat·u` ≲ 1e-15) is a **unit gate**,
   not a spike check.

> **Caveat the spec carries, because no saved artefact backs the headline number.** The 33.9 % is
> console output of a spike script; no JSON or log in `scratch/` contains it, and it was reproduced
> in prose by #64 but never re-measured. **Item 4 above is therefore load-bearing**: it converts an
> unreproducible spike number into a standing test. Re-measure once while implementing it and record
> the value.
>
> Separately: the shipped BCT is a **hybrid** — `endpoint4` near, midpoint rank-1 far — and its
> error is measured against pure `endpoint4`. So the 4.039e-2 truncation contains a far-field
> midpoint-vs-endpoint component that was never separated out. Far blocks are θ-separated so it
> should be small; it is not measured.

### 5.5 The tree

- **BVH over R⁶ tangent-points** `(x_I, T_I)`. Split axis cycles the six coordinates, choosing the
  axis that minimises the sum of squared child-box diameters; the splitting plane is the **per-axis
  median** (quickselect). The median was chosen because the paper does not say which candidate plane
  it scores, and median guarantees balance — **at the price of the power-of-two leaf quantisation
  that makes κ_near and c₁ oscillate** (§3.2, §4.3). A SAH or midpoint split would change that
  oscillation and is **untested**.
- **Leaf size 8** (max; mean 4.88–7.81). Swept once only, at n=960 θ=0.5: leaf 4/8/16/32 gives
  κ_near 36.2/93.3/220.8/474.4 and `relErrRandom` 5.699e-2/4.039e-2/3.960e-2/3.869e-3. There is **no
  leaf sweep at n=30 000, no leaf×θ grid, and no timing per leaf** — so **leaf=8 is a convention,
  not an optimum.** Recorded as such; §9.3 lists the sweep as cheap future work.
- **Admissibility** = `[YSC]` Eq. 28 barycentre form `max(r_x^A, r_x^B) ≤ θ·|x_A − x_B|`, with the
  tangent-space parameter `θ_T` **off**. For an inadmissible pair where exactly one node is a leaf,
  split the other (a deviation from the paper, recorded).
- **Barycentre separation is more permissive than Repulsor's.** `[RS §5.1.4]` separates on
  `dist(conv(I), conv(J))` via GJK; barycentre distance ≥ hull distance, so this tree **admits more
  blocks than Repulsor's would** and every near count here is an optimistic lower bound. By how
  much is **not measured.**

### 5.6 Module layout

```
src/core/sobolev/hierarchical/
  bvh.ts           # R⁶ tangent-point BVH, median split, leaf=8
  bct.ts           # block cluster tree: admissible/inadmissible block lists, adjacency veto
  nearField.ts     # endpoint4 near assembly (once per gradient step) + the precomputed arrays
  apply.ts         # the 12-column traversal; f32 and f64 accumulator variants of ONE code path
  operator.ts      # A_bct as a SobolevOperator (§5.1): applyA, trace/sigma, K1 and K⁰1
  laplacian.ts     # (−Δ)⁻¹, Sherman–Morrison cyclic tridiagonal (NR §2.7)
  lowOrder.ts      # L_{2−s}, 3 columns on the same tree
  precond.ts       # Ã⁻¹ = (−Δ)⁻¹ L_{2−s} (−Δ)⁻¹, with the constant-mode correction
  gmres.ts         # left-preconditioned GMRES; exposes iters, maxit, tol, per-iter recurrence
  nullspace.ts     # the k=O(n) constraint route (§8)
```

**Non-negotiable structural requirement: f32 and f64 are accumulator variants of ONE traversal, not
two implementations.** Two reasons, both measured: H1's `δ_precision` is defined as f32-vs-f64 *at
the same θ* and is meaningless if the traversals differ; and this repo has already been burned by an
f64 simulation of an f32 kernel reporting zero error and hiding the bug. χ (or θ) is a **runtime**
parameter whose minimum value is the exact all-inadmissible path, through the same traversal code —
§7's H1 INVALID predicate exists to catch a χ that was never plumbed.

---

## 6. Precision: f32 is an inner solver, never the solver

**Plain f32 walls at n ≈ 1 200–2 000.** Four agreeing derivations (tabulated in §7.11), and the
mechanism is not the BCT:

> **SCOPE CORRECTION, A38 item 8 (2026-10-12): this range is the TREFOIL's.** Measured on four shapes,
> the exponent below is confirmed (+3.30…+3.34) but the PREFACTOR spans **4 506×**, so the
> unpreconditioned f32 wall spans **n ≈ 394 (arc) to n ≈ 4 907 (helix)** — the arc's is BELOW the n = 960
> floor. Any quotation of a scalar f32 wall must name its fixture.

`κ₂(A_σ) ∝ n^3.3338` measured against `2s = 10/3` theory, so `κ₂·u_f32` reaches 0.838 by n=1 920.
**The BCT fixes storage and cost. It does nothing whatsoever for conditioning.**

Directly measured on the real hierarchical operator at n=2 000, min over 5 rhs, stopped
left-preconditioned GMRES @1e-4:

| θ | f64 `cos∠_A` | f32 `cos∠_A` |
|---:|---:|---:|
| 0.5 | 0.998280 | **0.924352** |
| 0.25 | 0.999595 | **0.904223** |
| 0.1 | 0.999628 | **0.773241** |

**The wall is θ-independent and θ cannot buy it back** — θ=0.1, the *most accurate* operator, is the
**worst** in f32.

> **#59's n ≈ 3 310 is superseded; do not quote it.** That crossover reproduces exactly and is wrong
> for one reason: it substitutes the **dense `sgemv`** f32 floor for the **hierarchical apply** noise,
> which §7.2 measures at 5–13× larger and growing 2–3× faster. §7.11 gives the four corrected
> derivations (1 228 / 1 773 / 1 225 / 1 900) and the one statement that needs no fit at all: at
> n=1 920/2 000 the MEASURED hierarchical f32 low-kernel apply noise (8.123e-06) is **5.9× ABOVE**
> the MEASURED random-direction tolerance (1.383e-06), while at n=960 it is **0.30×** of it. **The
> crossing is strictly inside (960, 2 000], measured.** (The upper end is 2 000, not 1 920: the apply
> noise is measured at n=2 000 and the tolerance at n=1 920, and no n above 960 has both.)

**Ruling: the f32 hierarchical apply is the inner solver of a mixed-precision scheme.** The outer
iteration and all orthogonalisation stay f64. Above the wall, an f32-only configuration is
**declared out of range by the gates' own output** (§7.11) — the gates' result, not their failure.

**What gates the scheme is H5 property 3: `κ₂(Ã⁻¹A_σ)·u_inner ≤ 1e-2`** (MEASURED 6.716e-04, 14.9×
margin, flat in n). This is the operative number, and it is why the preconditioner is not optional:
`κ₂(A_σ)·u_f32` reaches 1 at **n ≈ 2 025** (DERIVED from the MEASURED 0.8369 at n=1 920 and the
MEASURED `n^3.3338` law: `1920 × (1/0.8369)^(1/3.3338)`), while the *preconditioned* number does not move
at all over n=64…30 000. Carson & Higham (2018) `[CH18]` is the precondition
(`φ = (c₁κ∞(A)+c₂)u_s` sufficiently less than 1); property 3 quantifies "sufficiently" at 100×.

> **Range caveat that is load-bearing:** `κ₂(Ã⁻¹A_σ)` is MEASURED flat only to n=480 in JSON (and
> *reported* to n=1 920 with no surviving artefact — §7.8). **Above n=1 920 its flatness is ASSUMED**,
> and that assumption carries the whole mixed-precision scheme. §7.11 item 2 is the run that closes
> it, and it does not need a dense matrix.

**Above n ≈ 960 no reference-free gate can detect the f32 wall** — only H1-b (operator-level,
extrapolated at the target) and H4a (needs a dense f64 reference, so it stops at n≈960). §7.11 states
this as a pre-registered limitation rather than leaving it implicit.

> **Both f32 measurements share one caveat, and its sign is now known — it is UNSAFE, not safe.**
> #59's and #64's f32 arms are **dense f32 matvecs of the hierarchical matrix**
> (`scratch/gmres_lab.py:67-75`), not genuine f32 hierarchical applies. #59 argued this is
> conservative, on the reasoning that a real BCT sums hierarchically so its f32 floor must be
> *lower*. **MEASURED, that reasoning is falsified by 5–13× in the unsafe direction:** the
> hierarchical apply noise at n=960 is 1.064e-06 (high kernel) / 3.002e-06 (low kernel) against the
> dense `sgemv`'s `alphaMatvecF32 = 2.229e-07`, and it grows `n^0.829`/`n^1.176` against the dense
> floor's `n^0.370`. This single correction is what moves the wall from n≈3 310 to n≈1 228–1 773.
>
> ⚠ **SCOPED 2026-10-12 by `[A40]`/`[A42]`, MEASURED: the noise statement above is TRUE and its WALL
> INFERENCE IS FALSE.** The genuine apply really is 5.5× noisier than the dense `sgemv` **on random
> directions** (1.05e-6 vs 1.9e-7, independently reproduced). But that does **not** transfer to the
> solve: at the `K_s` operating point the genuine arm's A-angle is **88–1370× SMALLER** than the dense-f32
> recipe's, because error entering through `Mᵀ` is derivative-shaped and damped by `A_σ^{-1}` while the
> dense recipe's error is white. **"UNSAFE" is STRUCK for the A-angle at `K_s`; it STANDS for
> random-direction apply noise**, and `[§7.11]` rows 1/1b are relabelled as random-direction apply noise,
> not wall evidence. The sentence "this single correction is what moves the wall to n≈1 228–1 773" is
> **withdrawn**: the dense-f32 recipe's wall is ≈1 900 (reproduced), the genuine arm's is ≥ 9 300.
>
> ~~**A genuine f32 hierarchical apply driven through the whole solve has still never been run**~~ —
> **FALSE as of 2026-10-12.** Task 7b ran one (`[A39 item 2]`) and row 10 has now run it as a controlled
> two-arm differential (`[A40]`, `[A42]`). §7.11's range ruling IS invalidated for the shipped operator,
> in the safe direction.
>
> **MECHANISM (`[A43]`, row 10 S4.2–S4.5; adopted as proposed).** The genuine apply's rounding error is
> suppressed across the whole low-wavenumber band, not only at the constant mode:
> `R = a_k(genuine)/a_k(dense cast)` = **0.0009–0.05 at `k = 2…16` on 10/10 (shape, n) cells**, so the
> "constant mode only" account is **refuted (0/10)**, and the registered "suppression across low `k`"
> threshold was met on only **6/10 and is NOT confirmed by its own rule**. The property that survives the
> controls is that **the error operator has exactly zero row sums, so it acts on differences of `x`**: a
> dense f32 cast with each row repaired to sum to zero (diagonal repaired in f64) matches the genuine
> arm's `A_σ⁻¹` amplification (**0.79–1.03×, 10/10**), whereas the same cast with the diagonal exact but
> row sums non-zero stays at the dense level (**0.62–1.38×, 10/10**). The `Mᵀ…M` factorisation is
> therefore **not necessary** for the effect, and `A·1 = 0` is the `k = 0` shadow of a statement about
> every row. In the genuine apply the high-order zero (`M·1 = 0`) is reassociation-proof and the low-order
> zero is conditional on `diag(K⁰1)` and `K⁰·1` agreeing (invisible at 10 ulps, 7–10× at 10³ ulps; the
> low-order term carries ~3 % of the `K_s` angle). **It does NOT protect the stagnation reading**, which
> is set by the f32 storage of `A·x` (`κ₂(A_σ)·u_f32`). *Zero-row-sums is a POST-HOC finding (its two
> controls were each registered before running); the registered hypothesis it replaced is not confirmed.*

> **Trap, measured, not transferable.** WGSL permits reassociation and fusion with **no portable
> opt-out** ([gpuweb#2076](https://github.com/gpuweb/gpuweb/issues/2076), open since 2021), and
> naga/wgpu is *measured* to delete Dekker compensation terms where Dawn preserves them. Firefox
> uses naga, Chrome uses Dawn. **Compensated-arithmetic results are not transferable between
> browsers**, so any gate that depends on them records its adapter and is INVALID on an uncertified
> one.

---

## 7. Pre-registered kill gates

<!-- GATES:BEGIN -->
### 7.0 How to read this section

Every gate is **pre-registered here, before implementation**, per CLAUDE.md ("pre-registered kill
gates go in the spec, not the plan"). Each carries a quantity, a bar, an INVALID predicate set, at
least one knock-out that must trip it, and at least one case that must NOT trip it.

**Number tags, used on every value below.** `MEASURED` (JSON artefact + key exists), `DERIVED`
(arithmetic from measured inputs, shown), `CHOSEN` (a criterion, not a measurement), `EXTRAPOLATED`
(a fit). **A bar is never EXTRAPOLATED.** Fits appear only in non-vacuity arguments.

**Three verdicts, not two.** `GREEN` / `RED` / `INVALID`. `INVALID` means the gate did not measure
what it claims to measure, and it is **not** a pass — a run with any INVALID gate cannot be
published as a result. This distinction is load-bearing: memory `gates-name-their-impossible-values`
— a gate that cannot be tripped reads exactly like one that passed, and §7.4's H4b history is this
repo's own instance of that failure.

**Where a bar's basis changed from #59, §7 says so and §10.2 carries the retraction.** The full
re-derivation from raw JSON, with every reproduction and every confusion matrix, is
`.superpowers/sdd/2026-09-11-phase2h-queue/reports/gate-recalibration.md`. That report is the
evidence; this section is the ruling.

### 7.1 The set

| gate | gates what | bar | mode | binds at n=30 000? |
|---|---|---|---|---|
| **H1-a** | θ-monotonicity of the compression error | `δ_compress(θ_{k+1}) ≤ 1.001·δ_compress(θ_k)` descending | calibration | yes |
| **H1-b** | the f32 accumulation band | `δ_precision ≤ 10·(κ_near+κ_far)·u_f32` | calibration | yes — and it is one of only two gates that can see the f32 wall |
| **H2** | solve backward error, run to the floor | `η ≤ 50·u_f32 = 2.98e-6` | calibration | yes (8.2× margin, EXTRAPOLATED) |
| **H3** | iteration count, and its growth | `iters(30 000) ≤ 64`; companion `G ≥ 2.0 ⇒ RED` | calibration | yes |
| **H4a** | direction quality against an f64 reference | `min cos∠_A ≥ 0.9` | calibration | **no — vacuous above n≈960, and that is its output** |
| **H4b′** | direction quality, reference-free | `min q > 0` | **per-solve, production** | yes |
| **H4c** | end-to-end energy decrease + acceptance | `R_E ≥ 0.95` **and** accepted fraction `== 1.0` | calibration | yes |
| **H5** | preconditioner integrity (3 properties) | nullity `== 1`; **property 2 RE-REGISTERED TWICE (#71, §7.8): the live bar is the per-fixture level `κ_excl ≤ 40` at n = 64…960 — BOTH `\|p\| ≤ 0.05` (A22) AND saturation `R ≤ 0.75` (A32) are WITHDRAWN**; `κ_excl(Ã⁻¹A_σ)·u_inner ≤ 1e-2` | calibration | properties 1 and 3 yes; property 2 limited to n ≤ 1 920 |
| **H6** | the θ default, under an adversarial rhs | `cos∠_A ≥ 0.9` at θ=0.5 on the sub-0.9 modes | calibration, **once** | decides §4.1, does not kill the phase |
| **R1** | reporting obligation, not a kill gate | three served-N ceilings, each with its binding gate's VALUE | — | — |

**No gate in H1–H6 reads a clock.** Every gated quantity is a norm ratio, an integer count, a
nullity, or a boolean. Issue #63 (`timestamp-query` returns all zeros on this host) therefore blocks
nothing in this section. H3's *bar* imports an extrapolated GPU per-iteration time as a **budget
constant written into this spec** — the gate still only counts integers, and if that constant is
later measured the bar changes while the gate does not.

**Coupling, which #59 left unstated.** H1-a, H2 and H4b′ all consume the χ=0 exact matvec. An
INVALID H1 at a given n makes **H2 and H4b′ INVALID at that n as well.** This is the difference
between one bad gate and three silently-green ones.

**The χ=0 reference must stream.** Not a materialised near-block list, not precomputed near values.
At N=30 000, L=8 a materialised block list is 1.4e7 entries (~112 MB) and precomputed near values
are `2 × N(N−1)/2 × 4 B = ` **3.60 GB** — which is the dense storage this whole phase exists to
escape, and it exceeds the Quadro RTX 3000's 6 GB. The measured 12.75× speed-up from precomputing
(#57) is a real and measured incentive to violate this, which is why it is a requirement and not a
preference. Cost of compliance: **63.58 ns per pair·column** (MEASURED, 1-thread CPU f64) ⇒ `N(N−1)/2
= 4.4998e8` pairs × 12 columns = **343 s ≈ 5.7 min per exact matvec at N=30 000.** This is why
H1/H2/H4a/H4c are calibration-mode gates and H4b′ is the only one that runs per solve.

### 7.2 H1 — operator fidelity: record the triple, gate monotonicity and the precision band

**There is no bar on δ's magnitude.** `δ` is RECORDED, not gated. Three candidate anchors for a
magnitude bar all require a dense matrix and so are unmeasurable above N ≈ 5 793 — the very cap this
phase exists to escape (§1.2): `α = alphaMatvecF32` is a dense f32 `sgemv`; `β_struct` is bisected by
repeatedly solving with a dense `A_σ + δ‖A_σ‖₂E`; and the **margin `β_struct ÷ δ₂` = 5.01/4.94/4.76/
5.16×** (MEASURED, n=240/480/960/2 000) is flat in n because numerator and denominator scale
together, so as a *predicate* it reduces to `δ ≤ 4.8·δ`, true for any operator. **That margin is a
margin report, not a bar** — §4.2 lists it as evidence, and it must never be quoted as a gate.

Independently: #59's `δ_bar = √(α·β) = 4.41e-04` against a **correct** operator's MEASURED
`δ_total = 2.236e-02` at n=960 would put a working implementation permanently 50× over its own bar.

**Recorded triple, per gated n, on the 5 pinned rhs directions:** `δ_compress` (θ-matvec vs χ=0, both
f64), `δ_precision` (f32 vs f64, same θ), `δ_total` (f32 θ-matvec vs f64 χ=0).

**H1-a — bar: `δ_compress(θ_{k+1}) ≤ 1.001 · δ_compress(θ_k)`** on the descending ladder
θ ∈ {0.5, 0.25, 0.1}, per fixture, per n. CHOSEN form (the 1.001 absorbs f64 reduction-order noise);
**MEASURED 15/15 cells** on the existing accuracy sweep, tightest margin 12 % (n=4 000, 0.5→0.25).
This is a *structural* bar: it contains no n and no absolute level, so it is non-vacuous at every
scale forever. What it checks is that refining θ actually refines the operator — i.e. that θ is
genuinely plumbed through the traversal and that admissibility is not inverted.

*Knock-outs.* (1) θ hardcoded to the production value ⇒ all three rungs return the same number ⇒
passes `≤ 1.001` but trips the INVALID tie predicate. This is exactly why ties must be flagged rather
than silently accepted. (2) Admissibility test inverted (`≥ θ·|x_A−x_B|`) ⇒ `δ_compress` rises as θ
falls ⇒ **RED**; NOT MEASURED, one sign flip in `bct.ts` to produce. (3) *Must NOT trip:* the
`circle` fixture's 0.5→0.25 tie (`κ_near` identical) ⇒ **GREEN with tie flag**, MEASURED.

**H1-b — bar: `δ_precision ≤ 10 · (κ_near + κ_far) · u_f32`**, `u_f32 = 5.96e-08`. DERIVED from the
worst-case relative error `≈ m·u` of a sequential f32 sum of `m = κ_near + κ_far` terms; the **×10 is
CHOSEN** headroom for the per-kernel `pow`/`sqrt` chain. At n=960, θ=0.5: `κ_near = 93.3`,
`κ_far = 3.68` (MEASURED) ⇒ bar **5.78e-05**, against a MEASURED hierarchical apply noise of
1.064e-06 (high kernel) / 3.002e-06 (low kernel) and #64's independent combined-apply check at
1.533e-06 ⇒ **19–54× margin.**

> **Provenance caveat on 1.533e-06.** That value is a console-only check
> (`scratch/run-64-verify.ts:129`) that writes no JSON, so it is REPORTED, not archived — the same
> defect class as H5 property 2's missing support (§7.8). The 19–54× margin does **not** depend on it:
> the archived `out-accuracy.json` `f32NoiseHi = 1.0639e-06` and `f32NoiseLow = 3.0024e-06` give
> 19–54× on their own. The gate harness must emit `h1bValue` to JSON so this stops being
> report-only.

This bar exists because **#59's conservatism assumption is falsified by #57's own JSON, in the unsafe
direction, by 5–13×.** #59 justified using the dense `sgemv` α as H1's floor by asserting that a BCT
sums hierarchically so its f32 floor must be *lower*. MEASURED at n=960: hierarchical apply noise
1.06e-6 / 3.00e-6 versus dense `sgemv` `alphaMatvecF32 = 2.229e-07`, and the hierarchical noise grows
`n^0.829` / `n^1.176` against the dense floor's `n^0.370` (4-point fits, R² 0.996 / 0.985,
EXTRAPOLATED). §7.8 is where that correction lands.

*Knock-outs.* (1) f64 accumulator ⇒ `δ_precision → O(1e-16)` and `δ_total → δ_compress`; a *must-NOT-
trip* consistency check whose second half is the INVALID triangle clause. MEASURED anchor: 1.533e-6
f32-vs-f64, identically 0 for f64-vs-f64. (2) Accumulator truncated to f16, or a lost
pairwise/Kahan sum ⇒ the same model predicts `97 × 4.88e-4 = 4.7e-2`, **820× over the bar.**
**NOT MEASURED — this knock-out must be produced during implementation** (§7.11 item 8).

*INVALID (any ⇒ H1, H2 and H4b′ all INVALID at that n).*

| predicate | what it means |
|---|---|
| `δ_compress(θ=0.5) == δ_compress(θ=0.1)` exactly, at a fixture where `κ_near` differs between the two | θ/χ was never plumbed; one code path is being compared with itself |
| `δ_total > 1.01·(δ_compress + δ_precision)` | violates the triangle inequality ⇒ the three were measured against *different* references |
| `δ_total < δ_precision / 2` | `δ_total` cannot be materially smaller than its own precision component |
| `AdapterCertification.classification ≠ 'hardware'` (`bench/gpu/adapterCert.ts:48`) | ran on a software adapter |

### 7.3 H2 — solve backward error, run to the floor, independent of H1

```
η(x̃) = ‖b − A_σ x̃‖₂ / ( ‖A_σ‖₂·‖x̃‖₂ + ‖b‖₂ )
```
Rigal–Gaches (1967) / Higham, *ASNA* Thm 7.1. **Bar: `η ≤ 50·u_f32 = 2.98e-06`** — CHOSEN multiplier
over a MEASURED floor.

> **AMENDED 2026-10-11 (A21) — the gated quantity is `η_bct`, and every number below was already an
> `η_bct` number.** `x̃` is produced by the BCT operator, so the residual that measures *this solve's*
> backward error is formed with the **same** operator:
> `η_bct = ‖b − A_bct,σ x̃‖₂ / (‖A_σ‖₂·‖x̃‖₂ + ‖b‖₂)`. Backward error is legitimately self-referential
> — that is what Rigal–Gaches means.
>
> **`η_exact`** (residual from the χ=0 streaming matvec) is **RETAINED and RECORDED, but it is not the
> bar.** PROVEN by blind review A, not observed: its floor is exactly `‖E x̃‖₂/(‖A_σ‖₂‖x̃‖₂+‖b‖₂)` with
> `E = A_χ0 − A_bct` — predicted/measured ratio **1.0000** (n=64: 1.8728e-03; n=240: 9.3471e-05) and
> linear in `δ` over **12 decades** in a controlled all-f64 sweep. So `η_exact` measures
> **`δ_compress`**: it is an H1-family instrument wearing H2's name. At θ=0.5 a **correct** operator
> reads `min η_exact` = 1.68e-03 / 7.28e-05 / 3.01e-05 / 2.40e-05 at n=64/240/480/960 — **563× down to
> 8× OVER the bar**, i.e. a bar no correct implementation can pass, the mirror image of A16's defect.
> `η_bct` meanwhile reaches **3.7e-16**: the solver is backward stable and the old instrument could
> not see it.
>
> **The `τ_η` ladder survives unchanged, which is why this is an amendment and not a re-measurement
> campaign.** `e1_gmres.py:32-33` passes `exact_A=A` while `matvec` is `mv_f64(A)` / `mv_f32(A)`.
> **In the f64 arm those are the same matrix**, so operator mismatch is **exactly 0** and
> `η_exact ≡ η_bct` bit-for-bit. H3's ladder (36 / 42 / 43 / 41 / 39 / 37) and its
> `iters(30 000) ≈ 40.6` projection are measured on that arm, so they are already `η_bct` numbers and
> stand verbatim.
>
> **`etaFloorMax` is the f32 arm, where the mismatch is NOT zero — and that turns out not to matter.
> MEASURED, because the bookkeeping alone could not settle it.** `mv_f32(A)` applies `A` in f32 while
> `exact_A` is the f64 `A`, so the two residuals are formed against genuinely different operators:
> `‖A_f64 − A_applied‖₂/‖A‖₂` = **3.039e-08** at n=240 and **3.285e-08** at n=960, i.e. 0.51–0.55
> `u_f32`, against a floor of the same order. The structural worry is therefore real, and it predicts
> that `etaFloorMax` could be dominated by operator mismatch rather than by solver backward error —
> the confound this box is about, five decades smaller.
>
> **It is not.** Computing both residuals on the same iterates
> (`scratch-f5/eta_bct_vs_exact_f32.py`, left-preconditioned f32, run to the floor, 5 pinned seeds):
>
> | n | `η_exact` floor, worst of 5 | `η_bct` floor, worst of 5 | **ratio, per seed** | argmin iterate agrees |
> |---:|---:|---:|---|---|
> | 240 | 4.2699e-08 (0.72 `u_f32`) | 4.0064e-08 (0.67 `u_f32`) | 1.035 – 1.177 | 4 / 5 (one off by 1) |
> | 960 | 7.0847e-08 (1.19 `u_f32`) | 6.9565e-08 (1.17 `u_f32`) | **1.018 – 1.059** | 5 / 5 |
>
> **The two η agree to 2–18 %, and the agreement TIGHTENS with n** (worst ratio 1.066 → 1.018). So
> the f32 column is already an `η_bct` measurement for every practical purpose; the `n^0.34` growth
> is not the mismatch's `n`-dependence, since the mismatch ratio is flat-to-shrinking; and **7b does
> not need to re-record the floor column.**
>
> **The reason is worth keeping, because it is the general rule this whole box turns on.** In f32 the
> operator mismatch (0.5 `u_f32`) and the solver's *attainable* backward error are the same size,
> because the Krylov vectors themselves pass through the f32 matvec — the iteration cannot solve the
> f32 system more accurately than f32 roundoff either, so neither floor can get under the other. At
> θ=0.5 the mismatch is `δ_compress ≈ 1e-2` while the f64 solver reaches 3.7e-16 — **fourteen decades
> apart**, and that is why `η_exact` and `η_bct` diverge there and not here. **`η_exact ≈ η_bct`
> exactly when the operator mismatch is at roundoff scale, and they separate when it is a model
> error.** H1 owns model error; H2 owns roundoff.
>
> The bar `50·u_f32`, the floors, the margins and the `n^0.34` fit all stand as written. The knock-out
> *set* changes (see *Knock-outs* below).

`β_η = 0.43·β_H1` is **RETRACTED** on two independent grounds: it inherits H1's dead random-E anchor,
and the 0.43 transfer was measured for random E only — the η-per-structured-δ ratio is measured
nowhere. Worse, at a 1e-4 recurrence stop a **perfect** operator (δ=0) is RED against `β_η` at every
n by 1.58–3.68× (MEASURED: η = 1.705e-02 / 3.470e-03 / 6.231e-04 / 6.215e-05 / 6.733e-06 at
n=64…960 vs `β_η` = 8.93e-3 / 9.85e-4 / 1.69e-4 / 3.08e-5 / 4.26e-6), and still RED at a 1e-6 stop
for n ≥ 240. **So the stopping rule is not a detail: H2 is only meaningful run to the floor.**

**Pinned decisions.**
- **Run to the floor.** `maxit = min(600, 2·dim)` — exactly E1's budget, so the measured floors
  transfer literally — and the gated value is `min over probed iterates of η`.
- **AMENDED (A21).** `η_bct`'s residual is formed with the **BCT matvec the iteration applied**,
  recomputed from `b` — not accumulated, and never the recurrence (anti-trap 2). `η_exact`'s residual
  is formed with the **χ=0 exact, streaming** matvec, also recomputed from `b`. **Both are recorded;
  `η_bct` is the gated one.** This bullet originally read *"`A_σ x̃` from the χ=0 exact, streaming
  matvec… **Never** the BCT used inside the iteration"*, which forbade the only reachable form of the
  gate.
- **`‖A_σ‖₂` from 30 power iterations on the HIERARCHICAL (θ) matvec**, not the χ=0 one. This is a
  change from #59 and it is what makes H2 affordable: 30 χ=0 matvecs is 2.86 h of 1-thread CPU at
  N=30 000. Justified MEASURED: the generalized spectrum of `(A_bct,σ, A_σ)` lies in `[0.9240,
  1.0333]` at **θ=0.25** — but production is **θ=0.5**, where the same generalized spectrum reaches
  `c₁ = 0.2886`, i.e. a ≤ 71 % deviation, not 7.6 %. **The first draft of this section quoted the
  θ=0.25 figure against a θ=0.5 production path; that was the wrong number.** The ruling survives on a
  different and tighter measurement: what H2 needs is the ratio of the *largest* eigenvalue (the
  2-norm), not the spread of the whole spectrum, and `lamMaxBct/lamMaxA` deviates by **≤ 0.53 %** at
  every measured n and θ (`out-64-direction.json`). That is sound for the physical reason that the top
  eigenvector is smooth and therefore well approximated by the far-field expansion — **the
  poorly-approximated modes sit at the bottom of the spectrum, which a norm estimate never sees.**
  A ≤ 0.53 % error in a **denominator**, against a bar with ≥ 25× margin. At n ≤ 960 both values are
  recorded, so the transfer is pinned rather than assumed.
- **30 power iterations suffices, for a reason #59 did not state.** `A_σ`'s top eigenvalues are nearly
  degenerate: for order `2s = 10/3` the relative gap is `(1+2/n)^{2s} − 1 ≈ 6.67/n` = 0.70 % at n=960
  and 0.022 % at n=30 000 (DERIVED; MEASURED anchor `lamMaxA == normAsigma2` at every n). Convergence
  is therefore slow **and the answer is correspondingly insensitive** — the value error is bounded by
  the relative gap, which *shrinks* with n. The residual risk is a start vector nearly orthogonal to
  the top block, which INVALID clause 4 catches for free.

**Two anti-traps, both verified in source.**
1. **Not `structuredSaddleResidual`** (`src/core/sobolev/linsolve.ts:860`; call sites `:850`, `:953`).
   Three reasons: its `Math.max(1, Math.sqrt(rhsNormSq))` denominator makes it not a relative residual
   when `‖rhs‖ < 1`; it sums the **constraint rows** too, so it is a saddle-system residual, not
   `A_σ`'s backward error; and it takes `a: Float64Array` and runs a triple loop over `n`, i.e. **it
   requires the dense matrix and cannot exist on the hierarchical path at all.**
2. **Not the recurrence residual.** Left-preconditioned, that is `‖Ã⁻¹r‖/‖Ã⁻¹b‖`, and its MEASURED
   under-report is **4.372e8 / 1.404e9 / 1.142e10 / 2.056e11 / 1.717e12** at n=64…960 — 3 926× growth,
   fitted exponent **+3.056**, tracking `κ₂(A_σ) ∝ n^3.334`. §5.3 is the rule; this is the number.

| | n=64 | 120 | 240 | 480 | 960 |
|---|---:|---:|---:|---:|---:|
| `etaFloorMax`, worst of 5 directions, MEASURED | 4.503e-08 | 5.762e-08 | 6.567e-08 | 8.475e-08 | **1.182e-07** |
| in units of `u_f32`, DERIVED | 0.76 | 0.97 | 1.10 | 1.42 | **1.98** |
| margin to the 50 u bar, DERIVED | 66× | 52× | 45× | 35× | **25×** |

`etaFloorMax ≈ 1.081e-08·n^0.3406` (R² 0.981, EXTRAPOLATED) ⇒ 3.62e-07 = 6.08 u at n=30 000, **margin
8.2×**. The fit is deliberately **not** in the bar; the bar is a constant and the fit is only the
non-vacuity argument.

> **A26, RULED 2026-10-11 (task 7b) — EVERY NUMBER IN THE TABLE AND THE FIT ABOVE IS A LAB NUMBER AND
> DOES NOT TRANSFER. THE ×50 BAR SURVIVES; ITS STATED JUSTIFICATION IS ARITHMETICALLY WRONG AND IS
> REPLACED HERE. THE SIGN OF THE EXPONENT FLIPS.** Re-measured on the SHIPPED operator (θ=0.5,
> `{kind:'lists'}`, leaf 8, shipped `makePreconditioner` default `30/ext²`, left, trefoil at native
> extent ≈5.4716, seeds 0–4, `tol:0` run-to-floor, min over 106 probes):
>
> | | n=64 | 120 | 240 | 480 | 960 | 1920 | 3840 |
> |---|---:|---:|---:|---:|---:|---:|---:|
> | `etaBctFloorMax` f32, MEASURED | 2.4547e-08 | 2.3229e-08 | 2.0522e-08 | 2.0114e-08 | 1.8552e-08 | 1.9453e-08 | 1.8245e-08 † |
> | in `u_f32`, DERIVED | **0.412** | 0.390 | 0.344 | 0.337 | **0.311** | 0.326 | 0.306 † |
> | margin to the 50 u bar | **121×** | 128× | 145× | 148× | **161×** | 153× | 163× † |
>
> † n=3840 f32 is **one seed**, not worst-of-5 (487.8 s/solve ⇒ 41 min for the full cell; the budget
> went to the f64 arm and an extra H3 octave). Flagged, not hidden.
>
> **The shipped floor is 1.8–6.4× LOWER than the lab's and it FALLS with n**: fit
> `3.294e-08·n^(−0.0774)`, R² 0.843, against the lab's `n^(+0.3406)`. EXTRAPOLATED to n=30 000:
> **0.249 `u_f32`, margin 201×** — 24× more margin than the 8.2× above, from the opposite sign of the
> exponent. The f64 arm's floor is flat at 3.3–5.3 `u_f64` (R² 0.026, no resolvable trend) and is the
> floor H3's rung must be judged against (A27).
>
> **Three consequences, all of which are corrections to the prose above, not to the bar:**
> 1. *"3.62e-07 = 6.08 u at n=30 000, margin 8.2×"* does not reproduce. **Do not quote it.**
> 2. **The `n^0.34` consistency remark with `[CH18]` Cor. 4.2 loses its observation entirely** — there
>    is no growth to explain. The shipped floor is flat-to-mildly-decaying and below 0.5 `u_f32` at
>    every n. `[CH18]` remains the precondition reference at `:1116`; it is no longer evidence here.
> 3. **The ×50's non-vacuity arithmetic is restated.** The gap between the worst measured floor
>    (**0.412 u**, not 1.98 u) and the nearest real defect (knock-out 1 at 1 678 u) is **4 073×**, not
>    850×, and its log-scale centre is **≈26 u**, not 50 u. **×50 STAYS** — it sits comfortably inside
>    a wider gap than advertised (121–163× above the floor, 33× below the defect), and a bar with more
>    margin than claimed is not a reason to move it. Tightening to 26 u would buy sensitivity against
>    no known defect: §3.4's calibration shows nothing lives between the floor and knock-out 1.
>
> **Both extrapolations are EXTRAPOLATED** from n ≤ 3840 — a 7.8× range extension. A20's two-point
> mass-weight extrapolation missed by five rungs; treat these the same way.

*Knock-outs.* (1) Report the **recurrence** residual as η at n=960, δ=0, stop 1e-4: recurrence reads
1.0e-04 = 1 678 u while the correct η is 6.733e-06 and the **true** relative residual is 6.378e-01 ⇒
**RED**, 34× over bar, MEASURED. **This one survives A21 unchanged** and is now H2's only MEASURED
knock-out. (2) **RE-SCOPED by A21, and it changes verdict.** *"A real operator defect, δ=1e-2 at
n=960 ⇒ η = 3.672e-03 = 61 602 u ⇒ RED, 1 232× over bar, MEASURED"* is a knock-out of **`η_exact`**,
not of `η_bct`. `η_bct` measures backward error against the operator the iteration actually applied,
so it is **blind to operator defects by construction**: a backward-stable solve of the *wrong*
operator reads 3.7e-16 and green. That is the correct division of labour — H1 and H1-b own operator
fidelity, and `η_exact` re-detects it here as the `δ_compress` monitor — but it must be said out
loud, because `η_exact` carries **no bar** under A21, so this knock-out's RED is now **informational,
not gating**. (3) **VOID under A21.** *"Measure η with the BCT instead of χ=0"* is no longer a
knock-out of anything: the BCT form **is** the gate. The reverse substitution is what INVALID clause
3 now guards (`η_exact` bit-equal to `η_bct` ⇒ the monitor was computed with the iteration's own
operator and is dead). (4) *Must NOT trip:* δ=0 run to floor at every n=64…960 ⇒ **GREEN**, ≥25×
margin, MEASURED.

~~**One knock-out must be ADDED, and task 7b owns it.** With (2) demoted and (3) void, H2 is left with
a single MEASURED knock-out, and that one (substituting the recurrence residual) tests the
*instrument's plumbing* rather than the *solver*. Register and MEASURE a genuine solver defect that
`η_bct` can see — the natural candidate is **skipping the final Givens update** (or otherwise
returning an `x̃` that is not the Arnoldi least-squares minimiser over `K_k`), which leaves `η_bct`
at the pre-update level while every recurrence quantity still looks converged.~~

> **A27, RULED 2026-10-11 (task 7b MEASURED it) — THE REGISTERED KNOCK-OUT DOES NOT TRIP, AND NO
> MUTANT OF ITS SHAPE CAN. H2's KNOCK-OUT SET IS CLOSED AT ONE, WITH THE REASON STATED RATHER THAN
> LEFT AS A VACANCY.**
>
> Measured, three faithful readings of the phrase above, f32 and f64, seeds 0–4:
>
> | mutant | effect on the gated `η_bct` floor, **per-seed paired** | verdict |
> |---|---|---|
> | undo rotation `k−1` on `R`'s last column and on `g` (recurrence, `g`, `iters`, basis all bit-identical to the shipped run) | f32 n=240 **1.016–1.221×**, f64 n=240 2.80–5.69×; f32 n=960 1.0001–1.0263×, f64 n=960 7.10–13.52× | **does not trip — the bar is 134× above the worst f32 mutant, which reaches 7.48e-03 of it** |
> | return the minimiser over `K_{k−1}` instead of `K_k` | f32 **0.9768–1.0591×** (seed 0 is **BELOW 1**), f64 1.0000–1.0341× | **does not trip at all** |
> | `y[k−1] = 0` | f32 **1.0000× at all five seeds**, f64 1.0000–1.0313× | **does not trip at all** |
>
> **The f32 arm has NO resolving power for these mutants, and one seed proves it the hard way:**
> `prev` at f32 seed 0 floors **lower than the correct run** (0.9768). A defect cannot genuinely improve
> the backward error; this is the f32 plateau of §6 showing up as noise that swamps the signal. **Only
> the f64 arm carries signal here, and even there the largest effect is ~9 decades under the bar.** So
> the conclusion is stronger than "the knock-out does not trip": on the arm H2 is gated on, these
> mutants are not measurable at all.
>
> **These are PER-SEED PAIRED ratios (each seed's mutant against its own control), corrected
> 2026-10-11 from 7b's own first pass**, which divided range endpoint by range endpoint and so reported
> f32 1.07–1.09× and "exactly 1.00× at every seed" for the latter two. **No verdict changes** — the
> worst mutant is still two orders of magnitude under the bar — but the earlier spread was not a
> statistic, and one `prev` seed lands *below* 1, which an endpoint ratio cannot show. The
> n=240 cells were re-measured from scratch into `scratch-7b/ko-givens-240.json` after 7b found that
> `ko.ts` had keyed its output file on mode alone, letting the n=960 run silently overwrite n=240; the
> re-run reproduces the delivered ranges exactly (f64 1.021e-15–1.838e-15, f32 1.920e-08–2.230e-08).
> The figures were right and briefly **unfalsifiable**, which is the real defect — an absent artefact
> reads exactly like a present one, the same shape as `gates-name-their-impossible-values`.
>
> **Why no mutant of this shape can trip, derived and then confirmed to 2 %.** H2 is defined
> *run-to-floor* with the gated value the **minimum over 106 probes**. At the floor the Arnoldi
> least-squares problem is solved to roundoff, so a defect worth one rotation is a defect worth
> roundoff (undoing rotation `k−1` scales `y[k−1]` by `1/cs[k−1]²`, and `cs → 1` at the floor).
> Calibrating with `x̃ = (1+ε)x*` gives `η_bct = 1.208e-3·ε` at n=240, so the **detection threshold**
> is
>
> ```
> ε_crit  ≈  bar · (‖A_σ‖·‖x̃‖ + ‖b‖) / ‖b‖
> ```
>
> = **2.47e-3 at n=240** (predicted 2.420e-3, 2 % agreement) and **0.156 — a 16 % relative solution
> error — at n=960**: a **63× degradation over one factor of 4 in n**. With `κ₂(A_σ) ∝ n^3.334`,
> **`ε_crit` at n=30 000 is O(1e4)**.
>
> **THE LIMITATION THIS STATES, WHICH IS THE POINT OF THE AMENDMENT:** *"`x̃` is not the least-squares
> minimiser"* is a **FORWARD**-error defect and H2 gates a **BACKWARD** error. A small backward error
> bounds the forward error only through `κ·η` (Rigal–Gaches / Higham *ASNA* Thm 7.1, which §7.3 cites
> correctly). **So at target scale the 50·`u_f32` bar on `η_bct` permits an unbounded relative
> solution error, and H2 must not be read as certifying solution or direction quality. H4a
> (`cos∠_A`) is the gate that owns that, and it carries the load alone.** This is not a defect in H2;
> it is what backward error means. The knock-out §7.3 wanted **does exist** — at ε = 2.42e-3 the
> `η_bct` reaches 0.979× the bar while the recurrence residual still reads 1.140e-07, exactly the
> "every recurrence quantity looks converged" condition — it simply needs ~2.5e-3 of solution error,
> not the ~1e-6 that one skipped rotation produces.
>
> **Two alternatives considered and REJECTED, in writing.** (a) Register the `(1+ε)` mutant with an
> **n-dependent** `ε_crit` from the identity above: rejected because a gate whose knock-out magnitude
> grows with `n` is self-describing weakness — it would document the gate's blindness as if it were a
> gate. (b) A knock-out that moves the **backward** error (residual formed against the wrong `σ`, or
> against `A_bct` at the wrong θ): rejected for now because it is **unmeasured**, and registering an
> unmeasured knock-out is precisely the defect A24 was raised for. Either may be revisited by task 10,
> which owns the f32 direction work and will have `cos∠_A` instrumentation in hand.
>
> **Memory `gates-name-their-impossible-values` is therefore answered, not satisfied:** H2's single
> live knock-out tests the instrument's plumbing, and that is now a *stated and justified* scope
> rather than an unexamined gap. The gate measures what it measures.

*INVALID.* **`η_bct == 0` or `η_exact == 0` exactly while `‖b‖ > 0`** — the residual was not
computed at all. **A21 RE-SCOPES this clause's original rationale.** It read *"`η == 0` exactly (the
residual was formed with the operator that produced `x̃`)"*; that rationale applies to **`η_exact`
ONLY**. `η_bct` is *required* to use the operator that produced `x̃` — that is what backward error
means — so as originally written this clause ruled the only reachable form of H2 INVALID. Or
`η < u_op/10` **where `u_op` is the unit roundoff of the operator's own precision** — #59's unscoped
`η < u_f32/10` rules every correct f64 run INVALID, since the f64 arms reach `etaFloorMin` of
2.047e-16…6.001e-16 (MEASURED), and §7.9's mixed-precision scheme has an f64 residual path by
construction; or `η_exact` bit-equal to `η_bct` (both must be computed and recorded); or
`‖A_σ‖₂_est < ‖A_σx̃‖₂/‖x̃‖₂`, i.e. the power estimate is below a free exact lower bound, so it never
entered the top eigenspace.

> **A21 FOLLOW-UP RULING, 2026-10-11 — three clause-scoping gaps A21 left, two of them ambiguities I
> introduced.** A21 re-scoped clause 1's *rationale* to `η_exact` but widened its *predicate* to both
> η, and said nothing about which η clauses 2 and 3 govern. Found by the scoped re-review of
> `f8239a9`, which verified against the exported `etaViolations` that the code implements only the
> `η_exact` half of each — *"post-A21 the INVALID set guards `η_exact`, which carries no bar, and
> leaves `η_bct`, which IS the bar, unguarded by clauses 1 and 2."*
>
> **Clause 1 — applies to BOTH η, independently.** `η_bct == 0` with `‖b‖ > 0` means the gated residual
> was never computed. Verified non-breaking: extending the predicate is **61 pass / 0 fail**, so no
> existing test encodes the narrow behaviour — which also means the corrected clause ships untested
> and **must carry its own case**. (The `‖b‖ > 0` half is already structural: `gmres.ts` early-returns
> on `bNorm === 0` before any probe.)
>
> **Clause 2 — applies to BOTH η, independently, and the reason differs per η.** For `η_bct`, a
> backward error below the precision's own roundoff is impossible, so it means the residual was not
> formed in that precision. For `η_exact`, a value below `u_op/10` means the monitor is not reading
> the χ=0 operator at all — at θ=0.5 `δ_compress ≈ 1e-2`, five decades above any roundoff. **Same
> threshold, two different impossibilities.** A21 wrote `η < u_op/10` without saying which η; that
> ambiguity is resolved here rather than left to an implementer.
>
> **Clause 3 — SCOPED to `nFar > 0`, and it becomes two-sided. This one the re-review did not reach
> and it is the most interesting of the three.** Clause 3 rules bit-equality of `η_exact` and `η_bct`
> INVALID, to catch "one η computed and copied to both". **But §7.1 requires the χ=0
> all-inadmissible path to exist as a runtime configuration, and on it `A_bct` IS the exact operator**
> — so the two η legitimately agree, and clause 3 would rule a perfectly valid χ=0 run INVALID. That
> is the mirror of the defect A21 fixed. **Ruling: clause 3 fires only when the iteration's operator
> has at least one far-field block (`nFar > 0`, `bct.ts:170`). At `nFar == 0` the clause INVERTS —
> the two η must agree to roundoff, and a DISAGREEMENT is then the defect**, because it means the
> χ=0 apply and the streaming exact matvec are not the same operator. Do not assert bit-equality
> there: the two paths may legitimately differ in summation order, so the `nFar == 0` form is an
> agreement-to-roundoff check, not a bit check.
>
> **Clause 5 — ADOPTED into the spec, which is where it should have been all along. The code found
> it first.** `etaViolations` has carried a fifth clause since `7f67939`, added unprompted by task
> 7a's implementer: **`non-finite` — `η_exact` or `η_bct` not finite ⇒ INVALID**, with the reason in
> its TSDoc: *"NOT in the spec. NaN fails every `<` and `===` above, so a NaN `η` would read as valid
> on all four (memory `gates-name-their-impossible-values`)."* It is right, and `grep -c "non-finite"`
> over this document returned **0** — a live INVALID clause that no spec text registered, which means
> `probe.invalid` could emit a value task 7b's harness has no definition for. **Registered here as
> clause 5.** Widen it when convenient: it tests `η_exact` and `η_bct` only, so an infinite or NaN
> `‖A_σ‖₂_est` leaves clause 4's `normA2 < lowerBound` comparison `false` and is caught only
> transitively, via the NaN it induces in η (see #76).
>
> **Scope of the CODE change this ruling implies, kept deliberately small.** `etaViolations` already
> receives `etaBct` (`gmres.ts:292`), so clauses 1 and 2 widen with **no signature change**.
> **Clause 3's `nFar` scoping is a HARNESS obligation, not a `gmres.ts` change**: `etaViolations` has
> no access to `nFar` and giving it one is an API change to an exported function, out of proportion to
> a configuration nothing currently runs — H2 is defined at θ=0.5, and χ=0 is a baseline experiment.
> Whoever runs H2 at `nFar == 0` owns applying the inverted form. Recorded so it is not discovered as
> a spurious INVALID.
>
> **Why this is a new task and NOT a third round of task 7a** (adjudicated in writing per
> `CLAUDE.md`'s 3-round cap): the gap is in an amendment **I** landed *after* 7a was implemented, and
> `task-7a-fix-brief.md` never asked for clause 1. The implementer built what its brief specified and
> the re-reviewer verified exactly that. Treating this as round 3 would score my own spec defect
> against the task and trip the "the task is mis-sized" rule for the wrong reason. It ships as its own
> small commit with the queued citation fixes.

**Cost, and the one mitigation this spec picks.** One χ=0 matvec per probed iterate. At `maxit = 600`,
N=30 000 that is 600 × 5.7 min = **57 h per direction** — not runnable. **Ruling: probe η on a
two-stage schedule with the schedule RECORDED — every iterate to k=80, then every 20th.** This is
MEASURED, not chosen: `e1-closer-0b-n*.json` `etaArgminIter` puts the floor at iterations
**36–60** (per-seed, n=64…480), so a uniform `p = 20` would take only 2–3 probes before the floor and
could miss it entirely. Cost at N=30 000: 80 + 26 = **106 probes ≈ 10.1 h per direction**, against
the uniform-`p=20` schedule's 2.9 h — the extra 7 h is what buys a floor that is actually located.

The schedule is safe in one direction only, and that is why a coarse tail is acceptable: the gated
value is a *minimum* over probes, so a sparser probe can only report a **higher** η — conservative,
no false pass. The dense head is what guarantees the minimum is not missed entirely.
Secondarily, H2 is capped at the largest n where it is runnable, which R1 (§7.10) already puts at
n ≈ 960 for the f32-only configuration.

### 7.4 H3 — iteration count: the absolute count is primary, `G` is a companion

**`α_G` is WITHDRAWN. Both of #59's candidate values are measured at a stopping rule this spec's §5.3
forbids, and one of them is not an instance of the quantity it claims to measure.**

Three independent reasons, all re-derived from raw JSON:

1. **`G` is defined as `iters(2n)/iters(n)`, and 64→120 is ×1.875.** `α_G = 1.1034` (and the
   D1-corrected 1.1379) is computed on that step. It is not an instance of `G` at all — a type error,
   not a conservatism choice. The three *true* doublings in `e5-growth.json` are 0.7576 / 0.8000 /
   0.8500.
2. **Both are counted to a fixed *recurrence*-residual τ, whose optimism grows `n^3.056`** (§7.3
   anti-trap 2). A count to a fixed recurrence τ is a count to a **monotonically easier** target as n
   grows. The flat-or-falling series 29/33/25/20/17 is therefore partly a measurement artefact — and
   this is precisely the error #59 itself retracted for H1's β ("a relative criterion cannot gate a
   quantity whose anchor is degrading"), committed here against an anchor that degrades ~500× faster
   than the one that triggered the retraction.
3. #59's printed third value **1.5163 reproduces from no JSON** and must not be quoted (§10.3).

**Quantity, re-specified:**
```
iters(n) = median over the 5 pinned directions of the iteration count to reach
           η(x̃_k) ≤ τ_η,  τ_η pinned ONCE per gate run, identical at n and 2n,
           η = eta_bct, formed with the BCT matvec the iteration applied
                      (H2's instrument AS AMENDED BY A21 — NOT the chi=0 exact matvec)
left-preconditioned GMRES, f32 operator, maxit RECORDED
G(n) = iters(2n) / iters(n)
```

> **AMENDED 2026-10-11 (A21) — H3 counts on `η_bct`, and the ladder below is unchanged.** §7.4's
> justification for choosing η at all — *"the only residual-family quantity whose floor does not
> degrade with n"* — was derived from mismatch-0 data (`e1_gmres.py:32-33`, `exact_A=A` with
> `matvec = mv_f64(A)`: *the same matrix*), i.e. it was an argument about `η_bct` all along, and it
> survives verbatim under this reading. The arithmetic matters: `min η_exact` over 120 probes at
> n=64 is **1.87e-03**, so on the production path `τ_η = 1e-8` is **unreachable at every n** —
> `iters == maxit`, the censoring predicate fires, and **H3 would be INVALID at every n**, taking
> `iters(30 000) ≤ 64` and the economic case for the phase with it. Counting on `η_bct` is what makes
> the gate reachable, and it costs nothing: the ladder **36 / 42 / 43 / 41 / 39 / 37** and the
> projection `iters(30 000) ≈ 40.6` are already `η_bct` numbers.

**H3 is measured on the f64 operator at `τ_η = 1e-12`** (was `1e-8`; see A31 immediately below, which
supersedes the ladder and the ruling in this subsection).

> **A31, RULED 2026-10-11 (task 7b MEASURED the deciding octave) — THE RUNG MOVES FROM `1e-8` TO
> `1e-12`. THIS IS A GATE-DEFINITION CHANGE, AND IT IS THE THIRD TIME THIS GATE HAS BEEN RE-PINNED.**
>
> Shipped-operator ladder, f64, θ=0.5, left, shipped preconditioner default, trefoil native extent,
> seeds 0–4, median [per-seed min]. **Every number in the table below this box is a LAB number and is
> 22–31 % high; do not quote it for the shipped path** (A30).
>
> | `τ_η` | 64 | 120 | 240 | 480 | 960 | 1920 | 3840 | **7680** | n=7680 as % of peak | verdict |
> |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
> | 1e-6 | 21 | 23 | 23 | 23 | 12 | 6 | 3 | **1** | 2 % | collapsed by n=960 |
> | 1e-8 | 26 | 29 | 30 | 29 | **32** | 27 | 19 | **15** | **47 %** | **COLLAPSING — fails the predicate** |
> | 1e-10 | 30 | 36 | 38 | 38 | **39** | 35 | 37 | **29** | **74 %** | **DECLINING — rejected, see below** |
> | **1e-12** | 35 | 43 | 46 | 47 | **48** | 45 | 43 | **47** | **98 %** | **FLAT — SELECTED** |
> | 1e-14 | 40 | 49 | 54 | 56 | 58 | 54 | 51 | not measured | 88 % @ 3840 | rejected on the bar, below |
>
> **Why `1e-10` is rejected even though the pre-registered predicate passes it.** The predicate
> ("median at the largest n ≥ 15 AND ≥ 0.5 × peak") passes 1e-10 at 29 ≥ 19.5. **The predicate is a
> threshold on a LEVEL and the disqualifying property is a TREND**: 1e-10 falls **39 → 35 → 37 → 29**
> over the last three doublings, monotone apart from one blip. **That blip is the whole lesson** — at
> n=3840 1e-10 read 37 against a peak of 39 and looked flat; it was a non-monotonicity in a falling
> series, and had the ladder stopped there 1e-10 would have been ruled usable. §10.4's *"extend one
> point past where the fit looks clean"* is what caught it, for the third time in this document.
>
> **A falsifiable prediction was pre-registered and it decided this.** The controller's collapse-walk
> (onset moves 4× in n per two decades of τ, from 1e-6 at n≈960 and 1e-8 at n≈3840) predicted 1e-10 at
> **≈30** at n=7680 and put its onset at **n ≈ 15 360 — inside the 1 000–30 000 target**. 7b's
> competing mechanism (collapse set by `η`'s normaliser inflating as `‖A_σ‖ ∝ n^2.3`, so two decades of
> τ should buy `100^(1/2.3) ≈ 7.4×` in n) predicted **37–39** and onset at ≈28 000. **MEASURED: 29,
> per-seed 29/27/28/29/31.** The walk's prediction holds; **the normaliser mechanism is REFUTED by 8
> iterations, far outside the per-seed spread.** Recorded as a refutation rather than reinterpreted:
> whatever sets collapse onset, it is not normaliser inflation alone.
>
> **Why `1e-14` is not selected. MEASURED 2026-10-11 (A34) — this is no longer a projection from the
> short ladder: `iters(7680) = 56`, median of 5 (56/56/56/55/55), all `stop='eta-target'`, 0 censored,
> 0 degenerate, 17.5 s.** The conservative projection to n=30 000 is **67.8** (worst `G` = 1.1020) or
> **67.3** (last `G` = 1.0980) — **over the bar of 64 on both**, and the only route that clears is the
> power fit at 61.4 (R² 0.397), which is **laxer than the worst-`G` route used to clear 1e-12** and is
> ruled out for exactly that reason. The earlier figure of 68.0 came from the n ≤ 3840 ladder (7.8×
> extension); **the measurement moved it 0.3 %, to 67.8 at a 3.91× extension.** Why it barely moved:
> the measured growth `56/51 = 1.0980` almost exactly equals the in-hand `G_worst = 1.1020`, so
> base-up (+9.8 %) and exponent-down (one doubling, −9.3 %) all but cancel — the near-degenerate case
> of the arithmetic in the correction box below.
>
> **The cell is cross-checked, not taken on trust:** the looser rungs read off this run's own probe
> array are **bit-identical** to the earlier n=7680 cell — 1e-8 `[15,16,15,14,17]`, 1e-10
> `[29,27,28,29,31]`, 1e-12 `[47,49,47,47,48]` — so this run, the χ=0 run and the `A9_NO_EXACT` stub
> run agree on every shared quantity.
>
> **1e-14 fails on COST, not on stability, and a re-derivation must be argued on cost.** It is in fact
> **the FLATTEST rung measured — 97 % of peak**, against 1e-12's 96 %, 1e-10's 59 % and 1e-8's 25 %;
> reachable at every n, 0 censored, 0 degenerate at every cell. Tightening the rung made the count
> *more* stable, not less. Its only failing is the bar: ≈10 more iterations than 1e-12 at every n.
>
> **Clause 4's fallback is therefore VIABLE AT A KNOWN PRICE, which is the point of having measured
> it:** *"the rung moves to 1e-14 and the 64 bar is re-derived, not relaxed to fit"* is executable, and
> **the re-derived bar must be ≥ 68 — i.e. +6 % on 64.** It is no longer a fallback resting on an
> unmeasured number. **It does NOT put 1e-14 on 1e-12's footing:** 1e-14 stops at n=7680 (a 3.91×
> extension) where 1e-12 reaches n=15 360 (1.95×), so if clause 4 is ever taken, 1e-14 wants its own
> n=15 360 cell — ≈20 s, no longer a budget question.
>
> **1e-12 holds both kill clauses and is flat at 96 % of peak, so there is no need for 1e-14 and no
> reason to prefer it — and that ruling is unchanged by the measurement.** The original ladder
> (n ≤ 3840) is still not comparable with the rows that reach n=15 360. **1e-12 holds both kill clauses and is flat at 96 % of peak, so there is no
> need for 1e-14 and no reason to prefer it.** Selecting it would also mean adopting a less conservative
> projection route than the one used to clear 1e-12, or relaxing a kill-gate bar while tightening its
> target — both are how a gate stops being falsifiable.
>
> > **CORRECTED 2026-10-11 — MY STATED REASON WAS WRONG, and it was a capability claim I committed to a
> > spec without checking.** This paragraph read *"no further measurement can rescue it … worst-`G` is a
> > maximum over doublings, so adding n=7680 can only hold or raise it; 1e-14 therefore cannot pass the
> > conservative route no matter what is measured next."* **The premise is correct and the conclusion
> > does not follow.** The projection is not `G_worst`; it is
> >
> > ```
> > iters(30 000) = iters(n_last) · G_worst ^ log2(30 000 / n_last)
> > ```
> >
> > and **adding an octave also doubles `n_last`, which cuts the exponent by exactly 1.** A rising base
> > against a shrinking exponent can move either way. **This is falsified by the very measurement that
> > confirmed 1e-12:** adding n=15 360 left 1e-12's `G_worst` at exactly **1.0930** and its projection
> > fell **56.0 → 50.1** (margin 1.14× → **1.28×**), purely because the exponent went 1.966 → 0.966.
> >
> > **~~1e-14's outcome is therefore genuinely unmeasured, and roughly even.~~ SUPERSEDED THE SAME DAY
> > BY THE MEASUREMENT (A34): it came in at 56 and FAILED.** The boundary analysis below was correct and
> > is kept because it is what made the cell worth running — read it as the reason for the measurement,
> > not as an open question. The decision boundary is a
> > single number: **1e-14 clears the 64 bar iff `iters(7680) ≤ 52.9`** — i.e. iff it is flat or
> > declining. Even *exactly flat* at 51 clears, projecting 61.7. Its neighbours bracket the boundary:
> > 1e-12 grew 9 % over that doubling (which would put 1e-14 at ≈55.7 and fail), 1e-10 fell 22 % (≈40,
> > clearing easily).
> >
> > **DISCHARGED SAME DAY (A34): the measurement was taken and it cost 17.5 s, not 10 min.**
> > `iters(7680) = 56 > 52.9`, so **1e-14 FAILS**, and the paragraph above is now measurement rather
> > than projection. **The distinction worth keeping on the record: this refuted my REASONING and
> > confirmed my CONCLUSION.** 1e-14 tracked 1e-12's +9.8 % rather than 1e-10's −22 %, which is exactly
> > the mechanism I argued from when I pre-registered **52–58, flat-to-mildly-growing, therefore FAILS**
> > (failure condition: ≤ 51) before the cell ran. *"No measurement can rescue it"* was still a false
> > capability claim — the conclusion being right does not make the argument sound, and a claim that
> > cannot be checked is the defect regardless of which way the number lands.
> >
> > **Consequence for clause 4 below — the part that matters.** The fallback *"the rung moves to 1e-14
> > and the 64 bar is re-derived"* no longer **rests on behaviour nobody has measured**: it is viable at
> > a re-derived bar of **≥ 68**. The instruction that stood here — *"if that fallback is ever taken,
> > measuring 1e-14 at n ≥ 7680 is the FIRST step"* — is **DISCHARGED**; the first step now is the
> > n=15 360 cell, so 1e-14's ladder reaches the same extension as 1e-12's.
> > Memory `capability-claims-need-falsifying` names this exact failure — *"X cannot be computed here"
> > is as checkable as a number.* I did invite the attack when dispatching the measurement, and the
> > measuring agent took it; the error was writing the claim into the spec before it was tested.
>
> **What `1e-12` is selected ON, stated so the trade is auditable.** Two independent arguments agree,
> which is the strongest position this ladder supports: (i) its data are flattest — 98 % of peak at the
> largest n measured; (ii) the collapse-walk puts its onset at ≈61 000, outside the target range.
> Projections to n=30 000: **56.0 via worst `G`, 56.0 via last `G`, 50.2 via the power fit — all under
> 64.**
>
> **Three honest costs of this choice, none of which changes it:**
> 1. **The margin is thin and got THINNER when the octave was added.** 1e-12 is the only surviving rung
>    still *growing*: its worst `G` is now its **last** doubling (3840→7680, 43→47, `G = 1.0930`),
>    displacing the old worst of 1.0698, so the conservative projection moved 52.5 → **56.0** and the
>    margin to the bar fell from 1.22× to **1.14×** — 14 % headroom over a 3.9× extrapolation. A20's
>    two-point precedent applies here as much as anywhere.
> 2. **This does NOT satisfy §7.4's own floor-proximity criterion.** Applied to the f64 floor
>    (2.8e-16…5.9e-16, the arm H3 actually runs on — §7.4 used the **f32** floor, A30) that criterion
>    selects ≈1e-14. 1e-12 sits **1.9e3×** above the f64 floor. **The trade is explicit: non-degeneracy
>    and bar margin were preferred over floor proximity**, because a rung that collapses stops
>    measuring anything while a rung that is merely loose still measures mesh-independence.
> 3. **The power fits are not evidence** (R² = 0.479 / 0.008 / 0.327; the 1e-10 fit explains ~none of
>    the variance because the ladder rises then falls and a monomial cannot represent that). The `p`
>    column survives only for continuity with §7.4's method. **The `G`-based routes are the defensible
>    ones**, and `last G` is actively misleading for the collapsing rungs (0.79 reads as "shrinking
>    nicely" when the gate is dying).
>
> **PRE-REGISTERED KILL CONDITION for `1e-12`** (registered because it is the only rung still growing
> and 1.14× is thin): at **n = 15 360**, 1e-12 is INVALID as a gate rung if either its median falls
> **below 0.5 × 48 = 24**, or its worst-`G` projection to n=30 000 **exceeds 64**. If either fires the
> rung moves to 1e-14 and **the 64 bar must be re-derived rather than relaxed to fit.**
>
> > **DISCHARGED 2026-10-11 — BOTH CLAUSES HOLD, verdicted mechanically with no interpretation
> > required.** n = 15 360, same regime, 5 seeds:
> >
> > | clause | threshold | MEASURED | verdict |
> > |---|---|---|---|
> > | (a) median collapse | median < 24 | **46** (per-seed 46/46/48/46/46) | **HOLDS — 1.92× clear** |
> > | (b) projection | worst-`G` proj > 64 | `46 × 1.0930^0.966` = **50.1** | **HOLDS — 1.28× clear** |
> >
> > **The margin IMPROVED rather than eroding: 1.14× → 1.28×**, and 1e-12's three projection routes now
> > agree to within 11 % (45.1 / 48.8 / **50.1**). The extrapolation to 30 000 is down to **1.95×** —
> > the lowest it will ever be, since 30 000 is under one doubling from 15 360. **`G_worst` did not
> > move** (the new 7680→15 360 doubling is `G = 0.9787`, which does not displace 1.0930); the
> > projection fell because the exponent did. 1e-12 reads **46 at n=15 360, 96 % of its peak**.
> >
> > **MY SECOND PREDICTION WAS NOT MET, and it matters for how 1e-10 is rejected.** I predicted 1e-10
> > *"at or below ~0.5 × 39 ≈ 20, i.e. clearly collapsed"* at n=15 360. **Measured 23** — above my own
> > threshold. The walk's qualitative claim held (1e-10 declines monotonically through the target range:
> > 39 → 35 → 37 → 29 → **23**, now 59 % of peak) but its quantitative second prediction did not.
> > **Consequently 1e-10 at 23 STILL PASSES the pre-registered predicate (23 ≥ 0.5 × 39 = 19.5), even at
> > n=15 360. Its rejection rests ENTIRELY on the trend, never on the predicate** — which is the
> > limitation of the predicate noted above, restated here so nobody later reads "the predicate rejected
> > 1e-10". It did not, at any n.
> >
> > **One cell-specific caveat, stated because it voids figures.** The n=15 360 cell replaced the χ=0
> > reference operator with the BCT matvec (`A9_NO_EXACT=1`), because χ=0 makes every pair near and
> > `precomputeNear` would need **3.77 GB** at that size against ~6 GB available. Sound for H3 and
> > proved rather than asserted: `η_bct` is computed from `op` alone, and re-running n=7680 seed 0 with
> > the stub reproduced the χ=0 run's `itersToTau` **and `etaBctMin` bit-for-bit** (7.416124851836749e-13)
> > in 3.3 s against 111.6 s. **But with the stub `η_exact === η_bct` bit-for-bit, so INVALID clause 3
> > (`exact-equals-bct`) FIRES on all five seeds — correctly, since the stub literally computes one η and
> > uses it for both. In this cell every `η_exact` figure and clause 3 are VOID; `η_bct`, `itersToTau`,
> > `stop`, `censored` and `degenerate` are valid.** This is clause 3's **second** structural false
> > positive after `x̃ = 0` (§3.5) — see `@issue utof/repulsive-test2#81`.

~~MEASURED over the full ladder to n=1 920~~ **(SUPERSEDED BY A31 — lab numbers, retained for
provenance only)** (`e1-closer-0b-n*.json`, `rs/left`, median of 5, per-seed minimum in brackets):

| `τ_η` | f32 operator, n=64…1 920 | f64 operator, n=64…1 920 | usable? |
|---|---|---|---|
| 1e-2 | 10, 2, 1, 1, 1, 1 | 10, 2, 1, 1, 1, 1 | no — met before the iteration does anything |
| 1e-4 | 26, 27, 16, 3, **1, 1** | 26, 26, 15, 2, 1, 1 | **no — collapses** |
| 1e-6 | 31, 35, 36, 32, **19, 12** [min 9] | 31, 33, 35, 31, 18, 11 [min 9] | **no — collapses one octave later** |
| **1e-8** | **censored at every n** | **36, 42, 43, 41, 39, 37** [min 29] | **YES, f64 only** |
| 1e-10 | censored | 39, 49, 52, 51, 47, 45 [min 38] | yes, but needlessly tight |

**There is no fixed `τ_η` that is both reachable and non-degenerate on the f32 operator.** The f32 η
floor is ≈3e-8, so 1e-8 censors 5/5 at every n, while 1e-6 — the tightest reachable f32 target —
still decays to 12 by n=1 920. That is structural, not a tuning problem.

**So H3 is defined on the f64 operator, and the scope limit is stated rather than hidden:** H3
measures the **preconditioner's spectral mesh-independence**, which is a property of `Ã⁻¹A_σ` and
therefore precision-independent. It does **not** test the production precision — that is H1-b's job
(the apply perturbation) and H4a's (direction quality in f32). The f32 configuration's own iteration
behaviour is RECORDED as a companion via run-to-floor: `etaArgminIter`, MEASURED at **41–60 and flat
at ≈53–60 for n ≥ 240**.

> **This is the second correction to this section, and the first one was mine.** The amended draft
> pinned `τ_η = 1e-6` on the strength of the n ≤ 480 rungs alone (31/35/36/32, which looked flat).
> The n=960 and n=1 920 rungs — measured 20 minutes later — are **19 and 12**. The pattern is
> identical to the 1e-4 failure it replaced, one octave out. **§10.4's own process note is the
> lesson: "extend every sweep one point past where the fit looks clean."** It was written in this
> document and then violated in it. Any future change to `τ_η` must re-run the whole ladder to
> n=1 920, not the first four rungs.

> **This is a correction to this spec's own first draft, and it is the failure mode the whole section
> is about.** The draft pinned `τ_η = 1e-4`. At that value the count falls `26 → 3` over n=64…480, so
> the primary bar `iters(30 000) ≤ 64` would have been satisfied by **3**, with no INVALID predicate
> able to catch it — a gate that cannot be tripped, exactly what memory
> `gates-name-their-impossible-values` warns about, in the gate carrying the entire cost case. The
> error in the reasoning was precise and worth naming: **a non-degrading residual *floor* does not
> imply a non-degrading *level*.** η's floor grows only `n^0.34`, but η's level at a fixed iteration
> count **falls as `n^-2.90`**, so a fixed loose target becomes monotonically easier with n — the
> identical pathology this spec rejected the recurrence residual for (§5.3). Choosing a currency is
> not enough; the target must also sit close enough to that currency's floor to stay hard. `1e-6` is
> ~25× above the measured floor (≈4e-8): reachable at every n, trivial at none.

**Why `η` and not the true relative residual.** Both are n-invariant *targets*; their attainable
*floors* are not. MEASURED, `rs/left/f32`: `trueFloorMin` grows 6.68e-07 → **5.04e-03** over
n=64…960 (**n^3.36**) while `etaFloorMax` grows 4.50e-08 → 1.18e-07 (**n^0.34**). A true-residual
target of 1e-4 is therefore **unreachable in f32 at n ≥ 480** — `e1-gmres.json` returns `None` for
all 5 seeds, which is exactly the censoring the INVALID set must catch — and 1e-2 is reachable only
to n ≈ 1 225. **`η` is the only residual-family quantity whose floor does not degrade with n**, so it
is the only one that can anchor an n-invariant count, and it unifies H2 and H3 on one instrument.

**Bar: `iters(30 000) ≤ 64`.** CHOSEN, and it has two independent derivations that agree to 1.3 %:

- **From the budget:** #51's 0.5 s per solve ÷ the optimistic 7.8 ms per iteration = 64. The 7.8 ms is
  **EXTRAPOLATED** from #64's 6.3 ms GPU matvec × the measured `501/406 = 1.234` per-iteration factor
  for `L_{2−s}` + 2×`(−Δ)⁻¹`, and rests on **zero GPU runs** (§3.6, §9.1).
- **From the only left-preconditioned measurement that exists**, which this spec commissioned
  (§5.3.1a, `scratch/e1b-left-f64.json`, control reproduces E1 exactly). `rs/left/f64`, median of 5,
  counting to a fixed **true**-residual τ=1e-4 — a currency whose floor is ~1e-8 in f64 and therefore
  does not degrade over this range:

  | n | 64 | 120 | 240 | 480 | 960 |
  |---|---:|---:|---:|---:|---:|
  | `iters` to τ_true=1e-4 | 29 | 35 | 38 | 41 | **42** |
  | true doublings `G` | — | — | **1.0857** | **1.0789** | **1.0244** |

  `G` **decays toward 1** with n, i.e. asymptotic mesh independence, and the worst doubling is the
  least asymptotic one. Projecting the worst sustained `G` from n=960 over `log₂(30000/960) = 4.9658`
  doublings: `42 × 1.0857^4.9658 = ` **63.2**. With the last measured `G = 1.0244`: **47.4**. Both
  under 64, from a MEASURED count on the correct preconditioning side.

  **Superseded by the gate's own currency, which is better.** The above counts to a fixed *true*
  residual; H3 counts to **`τ_η = 1e-12`** (**A31** — the rung moved from `1e-8`, which collapses:
  shipped ladder 32 → 27 → 19 → **15** at n = 960 → 7 680). **The ladder and projection in the rest of
  this bullet are LAB numbers at the SUPERSEDED rung and are retained for provenance only
  (A30, A31); the shipped figures are 1e-12's 35/43/46/47/48/45/43/47 projecting 50.2–56.0.** Do not
  quote `40.6` or the `1.6× margin` — the coincidence that 1e-10's shipped projection also lands near
  40.6 is noted in 7b's §2.5 and is **not** corroboration: different rung, different operator, longer
  ladder. ~~The MEASURED ladder is
  **36 / 42 / 43 / 41 / 39 / 37** over n=64…1 920 with true doublings **1.0238 / 0.9535 / 0.9512 /
  0.9487** — flat, then mildly decaying. Projecting the worst (1.0238) from n=1 920 over
  `log₂(30000/1920) = 3.9658` doublings gives **`iters(30 000) ≈ 40.6`**, a **1.6× margin** under the
  bar of 64 rather than the near-zero margin the retracted pairing implied. This is the number to
  quote.~~ **(End of the superseded lab passage. The number to quote is now 1e-12's conservative
  projection, 56.0, a 1.14× margin — A31.)**

> **RETRACTED: these two numbers are not independent corroboration, and reading them that way was an
> error.** 64 is a **budget ceiling** ("the most iterations 0.5 s can buy at 7.8 ms") and 63.2 is a
> **predicted need** ("the count the measured growth implies"). They are not two estimates of one
> quantity. Their agreement therefore means **zero margin**, not confirmation: need ≈ afford is the
> worst case to be in, not the best. And the prediction is not robust to integer noise — the n=240
> per-seed counts are `[38, 38, 40, 40, 38]`, a 3–2 median split, and taking 40 instead of 38 moves
> the projection to **71.9**, over the bar. The honest statement is that the architecture's projected
> iteration count and its affordable iteration count are the same size to within integer noise, which
> is a **risk finding**, and §9.1 carries it as one.

**Why the absolute count is primary and the ratio is not.** At the production stopping rule the
counts are 6–8 (`e5-growth.json rs tau0.0001`), so a ±1 integer wobble moves `G` by 14–17 % — half
the distance from 1.0 to any plausible bar. **The ratio cannot be resolved there.** This argument is
independent of currency and it is what demotes `G`.

**Companion: `G(n)` RECORDED with its quantisation interval `G·(1/iters(n) + 1/iters(2n))`; RED iff
`G − interval ≥ 2.0`.** `β_G = 2.0` is DERIVED — `G = 2` ⟺ iteration count linear in N ⟺ no
asymptotic advantage over a direct method — and is confirmed by two MEASURED baselines at 2.0211
(unpreconditioned) and 2.0518 (Jacobi) at 480→960. **`β_G` is immune to the τ objection above**: the
unpreconditioned count at n=960 is exactly 960 = the Krylov dimension, with `maxit = 2000`, i.e.
genuine exact termination, which is τ-independent.

**One measured fact that settles the currency question on its own, with no fit:** at n=960 the
recurrence-residual count to τ=1e-8 is **17**, while the true-residual count to τ=1e-4 is **42**. The
recurrence target is nominally four orders of magnitude *tighter* and needs **2.5× fewer**
iterations. Every cost projection in this phase that multiplies by 17–26 is understating the
iteration count by that factor; §3.5 uses 42.

*Knock-outs.* (1) Replace `Ã⁻¹` with Jacobi ⇒ `G = 2.0518` at 480→960, band 0.0072 ⇒ 2.0446 ≥ 2.0 ⇒
**RED**, MEASURED. (2) Replace `Ã⁻¹` with the identity ⇒ `G = 2.0211`, band 0.0064 ⇒ 2.0146 ⇒
**RED**, MEASURED. Both are at recurrence τ=1e-8 and must be re-measured at `τ_η`; their verdict is
robust because both terminate at the Krylov dimension, which no change of τ can move below 2.
(3) *Must NOT trip:* H5's defective `L^{2−s}` ⇒ `gmresIters1e-4` identical within 1 at every n ⇒
**GREEN**. H3 is blind to that defect **by design** — which is why H5 exists.

*INVALID.*

| predicate | what it means |
|---|---|
| `iters ∈ {0, 1}` | the stopping test fired before any real iteration (τ above the initial η) |
| `iters == maxit` | the count is **censored, not measured**. Compare against `maxit`, **never** against the system dimension — `e5-growth.json none/n=960` is exactly `[960]×5` with `maxit = 2000`, which is valid exact Krylov termination |
| **any** of the 5 directions failed to reach `τ_η` within `maxit` | a median over survivors is censored. MEASURED failure mode: `rs/left/f32` returns `None` for **5/5** seeds at τ_true=1e-4, n ≥ 480. A harness returning `None`/`nan` is not covered by `iters == maxit` |
| `τ_η`, the preconditioning side, or `maxit` differ between n and 2n | `G` compares two different quantities |
| `G·(1/iters(n) + 1/iters(2n)) > (G_bar − 1)/2` | the ratio is quantisation-dominated. **Pre-registered escape:** tighten `τ_η` until `iters(n) ≥ 15`, re-run the **whole** n-ladder at the new `τ_η`, and record it. Changing τ per-n would make `G` meaningless |

**At the pinned `τ_η = 1e-6` the last predicate does NOT fire, and that is the whole reason `τ_η` is
pinned there.** MEASURED counts are 31/35/36/32, so the band is `≈ 1.0 × (1/32 + 1/32) = 0.0625`
against a threshold of `(β_G − 1)/2 = 0.5` — an 8× margin.

> **The first draft got this backwards twice over, and both errors came from the withdrawn `α_G`.** It
> claimed the predicate "fires at n=30 000 in the good case" using `iters(960) = 17` and a threshold
> of `0.152`. The 17 is a *recurrence*-currency count this section rejects (§7.4's opening), and the
> `0.152` is `(1.3038 − 1)/2` from the withdrawn `α_G`-derived bar — with the live companion bar
> `β_G = 2.0` the threshold is **0.5**, so even the draft's own 0.184 band would not have fired. The
> escape clause stays, because it is what a future τ choice would need; but it is a contingency, not
> a prediction.

### 7.5 H4a — the only gate that can certify direction quality

**Bar: `min over the 5 pinned directions of cos∠_A(x̃, x*) ≥ 0.9`**, with `x*` from a CPU f64 dense
solve. CHOSEN number; MEASURED δ=0 baseline, min over 5, τ=1e-4, left-preconditioned: **0.970381 /
0.986126 / 0.990811 / 0.998846 / 0.999314** at n=64…960 — it **improves** with n, worst-case margin
0.0704. #64 independently: 0.990811 / 0.998874 / 0.999239 / 0.999622 at n=240…2 000.

**`ε_A` is demoted to a RECORDED value.** #59's `ε_A ≤ 2·ε_A(δ=0)` is the self-anchoring form #59
itself retracted elsewhere, and its anchor is measurably bad: MEASURED `ε_A(δ=0)` medians are
1.472e-01 / 1.292e-01 / 8.047e-02 / 3.086e-02 / **3.143e-02** / **2.464e-01** at n=64…1 920 — already
**non-monotone at n=960** (+1.9 %) and then jumping **7.8×** at n=1 920. Separately, at τ=1e-4 `ε_A`
is dominated by the *stopping tolerance* rather than by the operator, so no absolute `ε_A` bar can
separate the two. `cos∠_A` has neither pathology.

**H4a's second, irreplaceable job: it is H4b′'s calibration.** `cosHat/cos_true = 1/√(1−ε_A²)`
exactly (DERIVED, §7.6). H4a is the only place that ratio can be measured, i.e. the only place
H4b′'s blindness can be quantified instead of assumed. **The gate must report it.**

*INVALID.* `x*` not from an f64 dense solve; or H4a's `cos` bit-equal to H4b′'s surrogate (the
surrogate was used as the reference). #59 named no INVALID predicate for H4a at all.

**Non-vacuity: H4a is vacuous above n ≈ 960, and that is its OUTPUT, not a defect** — a
pre-registered statement that above n=960 no gate in this set certifies direction quality (§7.8).
Hard ceiling N ≈ 5 793 from the `4N²` binding cap (§1.2); practical ceiling 960 from the f64
reference's own measured degradation at n=1 920. Restricting the *gate* to n ≤ 960 would be a
post-hoc fence at the edge of the data rather than a property, and it would contribute zero coverage
in the target range — so the restriction is published as a range limit with a named consequence,
not as a repair.

### 7.6 H4b′ — the bar is retracted because the quantity could not fail

**This is the most consequential correction in §7, it is algebraic, and it is confirmed on 175 raw
samples.** It is also this repo's own instance of memory `gates-name-their-impossible-values`.

#59 specifies `‖x*‖²_A = bᵀA⁻¹b` replaced by "its monotone computable lower bound `2x̃ᵀb − x̃ᵀA_σx̃`
… so the reported cos is a **lower** bound — the safe direction for a gate." Write `a = x̃ᵀA_σx̃`,
`c = x̃ᵀb`, `q = 2c − a`, `P = ‖x*‖²_A`:

- `q ≤ P`, with gap `‖x̃−x*‖²_A`. **#59's bound is correct** — `q` is a lower bound on `P`.
- The surrogate is `cosHat = c/√(a·q)`. Substituting a lower bound for `P` in the **denominator**
  makes the ratio **larger**: `cosHat` is an **UPPER** bound on `cos∠_A`, not a lower one.
- `c² − a·q = c² − 2ac + a² = (c−a)² ≥ 0`, so **`cosHat² ≥ 1` always**, with equality iff `a = c`.

**`cosHat ≥ 1 ≥ 0.9` identically: the bar sits below the quantity's structural minimum, so H4b as
specified cannot be tripped.** Its only two reachable outcomes are `q ≤ 0` (which #59 classifies as
INVALID) and GREEN. Confirmed on raw data, not only algebra. Of the **175** left-preconditioned
samples at τ=1e-4, **122 have `q > 0`** — and **all 122 have `cosHat ≥ 1`, zero below** — while the
remaining **53 have `q ≤ 0`**, where `cosHat` is not even defined (`a·q ≤ 0`). So the surrogate is
`≥ 1` on 100 % of the samples where it exists, and the other 53 are RED under the replacement bar; **1050/1050** samples satisfy the derived theorem
`ε_A < 1 ⇒ cos_true > 0`; and there are **16 measured FALSE PASSES** at the pinned stop, worst cases
true `cos∠_A` = **0.0281** / 0.0607 / 0.0201 (n=960, δ=1e-1/1e-1/1e-2) all reading GREEN at
`cosHat` = 2.53 / 2.06 / 1.00. **#59's own knock-out does not fire either**: at δ=1e-3, n=960 all 5
seeds have `ε_A > 1`, so `q ≤ 0` and #59's INVALID predicate fires in place of RED. H4b never
returned RED on its own failure ladder.

**There is no repair that keeps a 0.9 bar reference-free.** All computable quantities are functions of
`(x̃, b, A_σx̃)`, and `cos² = c²/(aP)`, so `cos ≥ 0.9` is exactly an **upper**-bound test on
`P = bᵀA⁻¹b`; from `x̃` alone only *lower* bounds on `P` exist (`q`, and the tighter `c²/a ≥ q`,
DERIVED: `c²/a − q = (c−a)²/a ≥ 0`). An upper bound needs global spectral information —
`P ≤ ‖b‖²/λ_min(A_σ)`, which is κ-loose. **A reference-free lower bound on `cos∠_A` from a single
iterate does not exist.**

**What survives is better than it sounds. Quantity: `q = 2x̃ᵀb − x̃ᵀA_σx̃`, reduced as min over the 5
pinned directions. Bar: `min q > 0`** — structural, with **no free constant to mis-calibrate.** Two
sound implications, both proved and both measured:

- `q ≤ 0` ⟺ the quadratic model's value at `x̃` is `½a − c ≥ 0` = its value at `0` ⟺ **the step does
  not decrease the model energy at all.** A hard, computable, reference-free defect statement.
- `q > 0` ⟺ `ε_A < 1` ⟹ `⟨x̃,x*⟩_A = (‖x̃‖²_A + ‖x*‖²_A − ‖x̃−x*‖²_A)/2 > 0` ⟹ **`cos∠_A > 0`.**
  GREEN therefore *certifies* a descent direction. MEASURED on 1050/1050 samples. This is #59's own
  weaker criterion, implemented soundly for the first time.

At the pinned production stop τ=1e-4, `min q > 0` is a **perfect classifier** of `min cos_true ≥ 0.9`
on all 70 measured (n, δ, side) cells — 0 false kills, 0 misses. **The gate therefore requires
τ ≤ 1e-4**: at τ=1e-2 it misses 18 of 35 left-preconditioned cells, because at that stop even a
*perfect* operator has true `cos` of 0.823–0.885 — a stopping-tolerance artefact, not an operator
defect. That is also why no H4 bar may be read at τ=1e-2.

**Three caveats travel with the perfect score, because it does not transfer.**
1. The implication is one-directional in theory (`q>0 ⇒ cos>0`, not `cos ≥ 0.9`). The perfect
   agreement holds because the random-E failure mode is **catastrophic** — nothing in the sweep lands
   in the `0 < cos < 0.9` band.
2. **The real BCT's worst case does land in that band.** #64's worst MEASURED `c₁ = 0.2886` (n=2 000,
   θ=0.5) gives a direction guarantee `2√(c₁c₂)/(c₁+c₂) = 0.8234`; for `cos = 0.8234`,
   `ε_A ≈ 0.567 < 1` ⇒ `q > 0` ⇒ GREEN. **H4b′ cannot see 0.8234.**
3. It cannot see the f32 wall. At n=2 000 the f32 `cos∠_A` is 0.9244 / 0.9042 / **0.7732** at
   θ=0.5/0.25/0.1; the 0.7732 case has `ε_A ≈ 0.634 < 1` ⇒ GREEN.

**Consequence for §4.1, and it is why H6 exists.** #64's recommendation — *"ship θ=0.5 with H4b as a
mandatory per-solve guard"* — rests on H4b enforcing `cos ≥ 0.9` per solve. **That instrument does
not exist, and no reference-free instrument can replace it.** H4b′ cannot distinguish θ=0.5 from
θ=0.25. §4.1's ruling is therefore re-based on H4b′ (catastrophic failures, per solve) + H4c
(end-to-end, per calibration) + **H6** (the one measurement that can discriminate).

*Which matvec?* #59's cost line says "one exact matvec"; #64 says "no exact `A`". **Ruling: the
production (θ) matvec.** `q` is then the model-energy decrease *in the hierarchical metric* — which
is the metric the step is actually taken in — and it stays a free per-solve guard instead of a
5.7-min-per-solve calibration measurement at N=30 000. Both are recorded at n ≤ 960 to pin the
difference.

*INVALID.* `x̃ᵀb ≤ 0` — then `cos ≤ 0` already, `q ≤ 0` adds nothing, no margin can be reported ⇒
fall back to H4a where a reference exists. **`q ≤ 0` is RED, not INVALID** (#59 had it as INVALID,
which is exactly why its own knock-out never fired).

**Non-vacuity at n=30 000: yes.** `q > 0` is scale-free, costs one `A_σx̃` matvec and two inner
products, and the δ=0 baselines *improve* with n (`ε_A` max over 5 at τ=1e-4: 0.2442 / 0.1687 /
0.1373 / 0.0483 / 0.0394 at n=64…960).

### 7.7 H4c — energy decrease and acceptance, not the step length

#59 had **no measurement** behind H4c's bar. There is one, and **it false-positives on 8 of 15
measured BCT arms.** Under #59's bar (median accepted τ ≥ half the f64 reference's), MEASURED at
n=240/480 (K=20) and n=960 (K=10): 8 of 15 BCT-family arms are **RED** and 4 more pass only by
**exact equality** — while **every one of the 15 delivers ≥ 99.691 % of the exact operator's energy
decrease.** The ratio is also non-monotone in θ (at n=960, θ=0.1 scores 1.000 while θ=0.25 scores
0.125). That is the tell: **τ is a step *length* in a metric that differs between arms**, so Armijo
backtracking lands on a different power of ½. τ measures scale, not quality.

**Bar, two conjuncts, both must hold:**
1. **`R_E = ΔE_total(candidate) / ΔE_total(f64 exact) ≥ 0.95`** over K=20 steps. CHOSEN 0.95;
   **MEASURED separation: 15/15 BCT arms ≥ 0.99691, 6/6 random arms ≤ 0.86027** ⇒ margins 1.049×
   (green side) and 1.104× (red side). **Both sides measured** — which #59's τ bar never had.
2. **Accepted-step fraction `== 1.0`** (no refusals in K steps). MEASURED: 15/15 BCT arms at 1.0;
   0/6 random arms (`0/1`, `1/2`, `5/6` at n=240; `1/2`, `0/1`, `9/10` at n=480), every failure
   `not_a_descent_direction`.

> **Conjunct 1's RED side is not calibrated, and the first draft's "6/6 random arms ≤ 0.86027" must
> not be read as one.** Those denominators *are* the arms' `K_run` — 1, 2, 6, 2, 1, 10 — because a
> refused step ends the trajectory. Every random arm's `ΔE_total` is therefore accumulated over 1–10
> steps and compared against a 20-step reference, which is exactly what this gate's **own INVALID
> clause (`K_run < K`) forbids folding into a ratio.** The green side stands (15/15 BCT arms ran the
> full K=20 and score ≥ 0.99691, giving the `≥ 0.95` bar a measured 1.049× margin there); the red side
> is **NOT ESTABLISHED.**
>
> **This does not weaken the gate, because conjunct 2 is what separates the populations** — 15/15
> versus 0/6, on a quantity that is *defined* for a truncated run and is in fact how truncation
> manifests. Conjunct 1's job is the complementary one: catching a direction that is accepted but
> weak. **The run that would calibrate its red side** forces a random arm through all K=20 steps by
> accepting every trial regardless of Armijo, measuring energy delivery decoupled from acceptance;
> §7.11 item 9 carries it.

Both are needed. #59 correctly rejected energy decrease *alone* as primary (δ=1e-4 at n=960 has
`cos = 0.4288` and is still a descent direction) — conjunct 2 covers that, because a direction that
bad gets refused. Conversely conjunct 1 covers a direction that is accepted but weak.

**K is pinned at 20, not 50.** K=50 has never been run anywhere. K=20 is MEASURED at n=240 and n=480
with full 12-arm JSON; n=960 ran K=10 with **only 4 of its 12 arms** (`rand` and `bctdir` controls did
not run) and its numbers exist only in `out-64-linesearch-960.log`, not in a JSON. Measured cost of
K=20 at n=960: 801–969 s per arm.

**The `reason` set has FIVE strings, not three**, verified by reading `lineSearchStepSet`
(`src/core/sobolev/lineSearch.ts:444-583`; `LineSearchStepResult` at `:410-421`; wrapper
`lineSearchStep` at `:593-617`): `zero_or_nonfinite_gradient_norm`, `not_a_descent_direction`,
`tau_below_min`, `armijo_failed`, `projection_failed`, across **four** `accepted:false` exits, all of
which set `tau: 0`. Defaults read from the same source: `c1 = 1e-4`, `shrink = 0.5`, `tau0 = 1`,
**`tauMin = 1e-12`** ⇒ maximum backtrack count `log₂(1/1e-12) = 39.9`. All five counts are recorded.

*INVALID.* `K_run < K` (the trajectory terminated early — report it as a refusal count, never fold a
truncated run into a ratio); or `ΔE_total(reference) ≤ 0` (no reference decrease to normalise by). τ
is RECORDED alongside `accepted` and `reason`; `records[].ms` stays a labelled non-gate value.

### 7.8 H5 — preconditioner integrity, and the mixed-precision precondition

**Property 1: `dim null(L^{2−s}) == 1`** (constants only). MEASURED: **1** for the vertex-collocated
variant and **2** for the literal `[RS §5.1.1]` identity variant, at every n ∈ {64,120,240,480}.
Mechanism: on a closed polyline `|E| = |V|`, so `U` is square and for even n annihilates the
checkerboard `(−1)^i`; every fixture has even n.

**Property 2: `κ_eig(Ã⁻¹A_σ)` flat, `|p| ≤ 0.05`** over the fixture set. **MEASURED to n=1 920 with
an artefact** (`e0-closer-0b.json`, produced after the first draft): 10 976.50 / 11 180.93 / 11 244.96
/ 11 262.28 / **11 267.3 / 11 269.0** at n=64…1 920 ⇒ log-log LSQ exponent **p = 0.006450** over all
six points (**0.012152** over the first four), against the defect variant's 1.014e+15 / 4.891e+14 /
1.875e+14 / 1.947e+13 — **11 orders high.** Either exponent clears the bar by ≥ 4×.

> **RE-REGISTERED 2026-10-11 — issue #71 RULED. `|p| ≤ 0.05` is WITHDRAWN as the property-2 bar, and
> `κ_eig` is replaced by `κ_excl`.** Everything above is kept as the record of what was measured; none
> of it is the live gate.
>
> **Why the slope bar had to go — two independent defects, both MEASURED (#71).** (i) Every `κ_eig`
> in the corpus is `max|w|/min|w|` over the **full** spectrum (`e0_sanity.py:34-37`) while this
> section's prose said *"excluding the constant mode"*. Since `Ã⁻¹A_σ·1 = 1` **exactly** (derived:
> `A_σ1 = σ1`, `Π1 = 0`, `1ᵀ1 = n`) and the smallest 1⊥ eigenvalue is MEASURED at 3.76–7.17, the
> minimum is pinned at 1 and **`κ_eig ≡ λ_max`**. The headline `p = 0.006450` measures how slowly
> `λ_max` grows, not how slowly conditioning grows. Genuinely excluding the constant mode gives the
> lab **p = 0.1823** over all six points — 28× the quoted figure and **RED against its own bar**.
> (ii) `κ_excl` is a **saturating** quantity: production's local exponent **halves every doubling**
> (0.557 / 0.365 / 0.183 / 0.093 / 0.049), so a log-log slope fitted over any window starting early
> enough exceeds 0.05 **for a perfectly correct operator**. The only passing fit is the two-point
> `{960, 1920}` one (p = 0.0493, passing by 0.0007), which is not evidence. A slope bar is the wrong
> instrument for a bounded quantity in pre-asymptotic rise.
> (iii) **And the decisive one, because it needs no appeal to pre-asymptotics: on `κ_excl` the slope is
> ANTI-CORRELATED with operator quality.** MEASURED, the *recommended* operator's log-log exponent is
> **HIGHER than the `m̂ → 0` reference's on every named point set** — **0.233 vs 0.182** over all six
> points and **0.049 vs 0.027** over the last two — while being **148× LOWER in absolute `κ_excl`**.
> So a slope bar tight enough to fail the worse operator **fails the better one first.** That is not a
> threshold to be re-tuned; it is the instrument ranking two operators in the wrong order, which
> disqualifies the quantity outright. It also sharpens why the derivation instruction discussed below
> was wrong: on this quantity the `m̂ → 0` reference is not merely a looser standard, it is a
> **flattered** one.
>
> ### ⚠️ EVERYTHING FROM HERE TO THE A32 BOX BELOW IS SUPERSEDED — READ A32 FIRST
>
> **The saturation bar described in the next three paragraphs is WITHDRAWN (A32, 2026-10-11).** It is
> kept verbatim as the record of what was ruled and on what evidence. **`R = 0.556`, the 1.35× margin,
> the `c = 20/25/40` row, "trips m̂ = 70 (41.8)", and "lab 0.491 and production 0.556 both pass the
> same bar" are LAB-OPERATOR figures and must not be quoted as production.** On the shipped TS
> operator at θ = 0.5 (§4.1's ruled target), `κ_excl` is non-monotone and `R` is −0.4354 on this very
> triple.
>
> **~~PRIMARY BAR~~ WITHDRAWN BAR — saturation, not slope.** Over three consecutive doublings of the same fixture,
> ```
> R = ( κ_excl(4m) − κ_excl(2m) ) / ( κ_excl(2m) − κ_excl(m) )   with (m, 2m, 4m) = (480, 960, 1920)
> Bar: R ≤ 0.75.      RED iff R > 0.75.
> ```
> DERIVED, not fitted: `R < 1` *is* convergence of the increments, and `R = 0.75` admits a geometric
> tail summing to 4× the last increment — i.e. a bounded `κ_excl`, which is the property "mesh
> independence" names. Three points, no regression, no point-set ambiguity, and **dimensionless**, so
> it transfers across fixtures and across mass weights. MEASURED at `massWeight = 30/ext²` (§5.2, #72):
> **R = 0.556**, a **1.35× margin**; `c = 20` 0.516, `c = 25` 0.589, `c = 40` 0.570 — the whole live
> band passes with ≥ 1.27×. It **trips** the degenerate regimes it has to: m̂ = 1 000 ⇒ 0.791 ⇒ RED,
> m̂ = 1e4 ⇒ 0.935 ⇒ RED. It is also monotone-safe at the optimum, which the slope bar was not. **And
> the `m̂ → 0` reference passes the same bar at 0.491** — the shared-reference property no tight level
> bar could deliver, and the one the withdrawn derivation instruction was reaching for.
>
> **COMPANION BAR — level, and it is a PER-FIXTURE bar.** `κ_excl(n) ≤ 40` on the **unnormalised
> trefoil** at every gated n. CHOSEN: 1.97× over the MEASURED maximum 20.3 (n=1 920) and 1.89× over
> the Aitken-extrapolated asymptote 21.2. It trips m̂ = 70 (41.8), 100 (77), 300 (559), 1 000 (5 151)
> and does not trip c ∈ [20, 40]. **#71's own proposal of `≤ 25` is REJECTED as too thin** — it
> survives at 1.23× and *fails* at c = 20 (0.93×), so it would red-flag a live candidate.
> **`K = 40` must NOT be applied to another fixture.** MEASURED at c = 30, `κ_excl` max is **138**
> (arc, n=480) and **159** (helix, n=960, still rising) against the trefoil's 20.3 — a 7.8× spread.
> Shape-universality is not available for a level bar at any K that is also tight on the trefoil; the
> dimensionless `R` is what carries across shapes.
>
> **My own instruction that the bar be one the paper-faithful (`m̂ → 0`) reference also passes was
> REFUTED, by a number, and the refutation is why `R` is primary.** A shared *level* bound the lab
> passes needs `K ≥ 3 000.3`, which would leave the shipped operator **≥ 100× slack** — unable to see
> a 10× degradation. The dimensionless `R` achieves what the instruction was reaching for without
> that cost: lab **0.491** and production **0.556** both pass the same bar.
>
> **The `m̂ → 0` column goes through the corpus's only pseudo-inverse, and that was falsified rather
> than assumed. IT IS NOT A TOLERANCE ARTEFACT.** Every `m̂ → 0` number here comes through
> `np.linalg.pinv(Π L Π, rcond=1e-12)` (`gmres_lab.py:174-196`); every other column uses a genuine
> `inv(L + mw·M)`. MEASURED (`scratch/rcond_check.json`, 5 cells: trefoil/helix at n=480 and 960,
> arc at n=480), **five independent corroborations**:
> 1. **`rcond` invariance over seven decades.** `rcond` ∈ {1e-8, 1e-10, 1e-12, 1e-13, 1e-14, 1e-15}
>    gives `κ_excl`, `λ_min`, `λ_max` identical **to every printed digit** in all 5 cells, with
>    exactly **1 mode discarded** every time.
> 2. **The mechanism, which makes it over-determined rather than lucky.** The null-mode gap is
>    **11–12 decades wide**: trefoil n=480 has `σ/σ_max` = 3.139e-17 then 3.664e-05; helix n=960 has
>    1.665e-17 then 2.677e-06. Any `rcond` in that window discards exactly the one true null mode.
> 3. **A tolerance-FREE reference agrees.** `D⁺ = Q_⊥(Q_⊥ᵀ L Q_⊥)⁻¹Q_⊥ᵀ` with `Q_⊥` an orthonormal
>    basis of `1⊥` — exact for a connected graph, no tolerance anywhere — matches `pinv(rcond=1e-12)`
>    to **1.6e-11 … 6.4e-11** relative, worst over all cells.
> 4. **The genuine-inverse path reproduces the limit, and this is the one that matters** because it
>    contains no pseudo-inverse at all: `m̂ = 1e-6` gives `κ_excl` = 5 145 498 on the helix at n=960
>    against `pinv`'s 5 145 665 — agreement to **3.2e-5** relative.
> 5. **An independent eigensolver agrees.** The symmetric-definite pencil `eigh(Ã⁻¹, A⁻¹)` — avoiding
>    the non-symmetric `eigvals` used everywhere else — reproduces `λ_max` to **11 digits** and
>    returns `λ_min` over the FULL spectrum as **1.0000000 ± 6.4e-6**, which independently confirms
>    #71's derivation that the constant eigenvalue is exactly 1.
>
> So the lab's cross-check value 0.491 stands, and **option B's rejection on H5 property 3 is
> confirmed, not conceded**: the helix's `λ_max` is **1.597e7** at n=480 and **1.598e7** at n=960 —
> essentially flat in `n`, so `κ_incl·u_f32` = **0.952** against the 1e-2 bar is a real spectral
> property of the mass-free operator on that shape, **95× RED**, and not a discretisation or
> tolerance effect. Also recorded: the **arc**'s `m̂ → 0` conditioning is benign (`κ_excl` = 65.1 at
> n=480), so "the lab is badly conditioned" is shape-specific and must not be stated generally.
>
> **Unaffected by this re-registration:** property 1 (nullity); property 3, which passes under either
> definition with ≥ 2 200× margin (`κ_excl·u_f32` worst **2.9e-5** at c = 30, helix n=960 — 341×
> margin); and A15's knock-out detector (`nullDim == 0` ∧ `|λ₁ − 1|`, `precond.test.ts:441-443`),
> since `κ_excl` is structurally invariant to the knock-out that adds the checkerboard mode.
> **Reporting requirement (#71 item 1, adopted):** both columns are emitted, labelled `incl_const`
> and `excl_const`, and every fit names its point set.

> ---
>
> ### ⚠️ A32 — **`R ≤ 0.75` IS WITHDRAWN TOO. The level bar is now the primary and only numeric bar.**
>
> **Ruled 2026-10-11, on the SHIPPED TS operator, measured at BOTH θ values** — the measurement A22 did
> not have. A22's `R` was measured on the lab/python operator; it does not survive the operator this
> repo ships, and the two figures above (`R = 0.556`, "trips m̂ = 70 at 41.8") are **lab figures that
> must not be quoted as production.** Probes were throwaway test files, reproducible from
> `src/core/sobolev/hierarchical/{bvh,bct,apply,lowOrder,operator,precond}.ts` +
> `test/sobolev/hierarchical/denseSpectrum.ts` with no scratch JSON.
>
> **θ NOTATION, stated once because it is easy to get wrong:** `DEFAULT_THETA = 0.25` is the value
> the code ships today (`bct.ts:95`, deliberate and anchored — the flip to 0.5 costs 2.09× near-field
> work and waits on H6's pre-registration, #51); **θ = 0.5 is §4.1's RULED TARGET, not the current
> default.** Below, θ = 0.5 is labelled the *target* and θ = 0.25 the *shipped* value. **The level bar
> passes at both, which is the whole reason it is the one that ships.**
>
> **(1) `κ_excl` is NOT MONOTONE in `n` on the shipped operator.** Trefoil, θ = 0.5 (the ruled
> target), default m̂ = 30:
>
> | n | 120 | 240 | 480 | 960 | 1 920 | 3 840 |
> |---|---|---|---|---|---|---|
> | `κ_excl` | 11.8922 | 15.4790 | 17.5102 | **20.8632** | **19.4034** | 20.2410 |
> | `λ₂` | 5.256820 | 4.303840 | 3.923570 | **3.325457** | **3.643102** | 3.544197 |
> | `λ_max` | 62.515310 | 66.619120 | 68.702477 | 69.379731 | 70.688587 | 71.738180 |
>
> `λ_max` is monotone and saturating; **`λ₂` wobbles ~15 % and is not monotone past n = 480**, and it
> drags `κ_excl` with it. NOT an instrument failure: `nullDim = 0` and `λ₁ = 1.000000000` in every
> cell, `rankTol ≈ 7e-7` against `λ₂ ≈ 3.5` — four decades of headroom. Reproduced bit-for-bit twice.
>
> **(2) `R` therefore returns a DIFFERENT VERDICT on every doubling triple.**
>
> | triple | Δ₁ | Δ₂ | **R** | vs bar 0.75 |
> |---|---|---|---|---|
> | 120 / 240 / 480 | 3.5868 | 2.0312 | **0.5663** | pass |
> | 240 / 480 / 960 | 2.0312 | 3.3530 | **1.6509** | **FAILS — on the CORRECT operator** |
> | **480 / 960 / 1 920 — the triple A22 REGISTERED** | 3.3530 | −1.4598 | **−0.4354** | passes **VACUOUSLY** |
> | 960 / 1 920 / 3 840 | −1.4598 | 0.8376 | **−0.5740** | passes vacuously |
>
> On the registered triple `R` is **negative**, and `R ≤ 0.75` admits every negative number — so there
> the bar **could not be tripped by any amount of non-monotonicity** (CLAUDE.md
> `gates-name-their-impossible-values`). On the adjacent triple it **red-flags a correct operator**,
> which is the exact defect that killed `|p| ≤ 0.05`. A22's derivation — *"`R < 1` is convergence of
> the increments and `R = 0.75` admits a geometric tail summing to 4× the last increment"* — presumes
> `R ∈ (0,1)`; with mixed-sign increments the tail bound is **void, not merely loose.**
>
> **(3) Mechanism, and why NO point set fixes it.** Successive increments past n = 480 are **0.8–3.4
> on a value of ≈ 20, i.e. 4–17 %**, while `λ₂`'s own wobble is **≈ 15 %**. `R` is a ratio of two
> differences **each at the noise floor of the quantity being differenced.** That is a property of the
> statistic, not of the window: re-registering the triple cannot repair it.
>
> **(4) `θ` barely moves the magnitude but it changes the SIGN.** At θ = 0.25 the same ladder is
> monotone — 18.3644 / 19.6713 / 20.4533 at n = 480/960/1 920 — giving **R = 0.5982** and an Aitken
> asymptote of **21.617**, which reproduces A22's 0.556 and 21.2. The κ magnitudes differ by only
> 3–5 % between θ, which is why this was easy to miss; the gate flips from *passes by derivation* to
> *passes by accident*. (Adjacent to #69, not the same defect — #69 is spatial scale / `κ_near`.)
>
> **(5) THE LEVEL BAR SURVIVES, and its sensitivity floor is NOT where A22 put it.** `κ_excl ≤ 40`,
> max over the whole θ = 0.5 (target) ladder **20.8632** ⇒ **1.92× margin**; at θ = 0.25 (shipped) max
> **19.6713** ⇒ **2.03×**. **It passes at the shipped θ AND at the one §4.1 rules we move to, so the
> gate does not have to be recalibrated when H6 flips `DEFAULT_THETA`.** No differencing, no point-set ambiguity, monotonicity not required, robust to θ.
> Trip checks at **n = 960, θ = 0.5 (target)**: m̂ = 100 ⇒ **65.9672 TRIPS**, m̂ = 300 and 1 000 trip by far.
> **But m̂ = 70 ⇒ 36.7279, which does NOT trip 40.** A22's 41.8 for m̂ = 70 is the lab operator.
> **The demonstrated sensitivity floor on the shipped operator at production θ is m̂ ∈ (70, 100].**
>
> **(6) A SECOND instrument defect, the same shape as #71's original one.** `κ_excl := λ_max/λ₂` is
> "excluding the constant mode" **only while the constant mode is the smallest eigenvalue.** MEASURED:
> `Ã⁻¹A_σ·1 = 1` to **1.8652e-14 at every m̂ ∈ {30, 70, 100, 300}** — bit-identical across all four,
> as expected since the `+11ᵀ/(nσ)` correction is mass-independent. But at m̂ = 100, **`λ₁ = 0.7486`
> and `λ₂ = 0.7652` are BOTH below 1**, so two non-constant modes sit under the constant one and the
> gated quantity **silently changes meaning in exactly the regime used to demonstrate that the gate
> trips** (and understates: the true 1⊥ minimum gives 50.479/0.7486 = **67.4**, not 65.97). The
> shipped path *is* guarded — `precond.test.ts` asserts `|λ₁ − 1| < 1e-11·λ_max` and `λ₂ > 1.5`, which
> is precisely this precondition; the trip demonstrations were not.
>
> **(7) Cost — why the ladder stops at n = 960** (upper bounds; a subagent was running):
> n = 480 **1.7 s**, n = 960 **7.7 s**, n = 1 920 **46 s** (229 MB), n = 3 840 **442 s** (705 MB).
> `eig` is the O(n³) term and dominates past n = 960. n ≤ 960 is affordable in the default suite
> (+≈ 8 s); n ≥ 1 920 is not. **A32 deliberately registers no bar that needs those points.**
>
> **THE LIVE GATE for H5 property 2, superseding A22's primary/companion split:**
> **`κ_excl ≤ 40` on the unnormalised trefoil, asserted at n = 64 / 120 / 240 / 480 / 960.** Bounded
> `κ_excl` across a 15× range of `n` **is** mesh independence — stated as a bound rather than a slope,
> which is what A22's own derivation said the property names. `R`, the Aitken asymptote and both
> log-log exponents stay **LOGGED with their point set named, and are never asserted.** `K = 40`
> remains **PER-FIXTURE** (arc 138, helix 159) and the `λ₁`/`λ₂` pair is promoted to a **stated
> precondition** of the `excl_const` definition.
>
> **Still true from A22 and NOT re-opened:** `|p| ≤ 0.05` is withdrawn; `κ_eig → κ_excl`; property 1
> and property 3 unaffected; the rank-ordering argument against the slope bar (the recommended
> operator's exponent is *higher* than the lab's while 148× better in absolute `κ_excl`).

> **Provenance, now closed — and #59's numbers were right all along.** #59 reported "measured flat to
> n=1 920, `p = 0.0010`", but its two largest values came from `e0_sanity.py`, which wrote no JSON and
> whose log was not saved. That artefact now exists and **reproduces #59's values exactly, including
> the two that had no archive** — 11 267.3 and 11 269.0. What does *not* reproduce is `p = 0.0010`:
> the real exponents are **0.006450** (all six points) and **0.012152** (four *smallest*) — but
> `p = 0.0010` does reproduce, as **0.00099 over the four LARGEST points**. So #59 was right a second
> time: it fitted the large-n end, which is the defensible choice for an asymptotic-flatness claim and
> which this spec's earlier "reproduces from neither set" got wrong. The lesson is therefore much
> narrower than "the numbers were wrong": **every number was sound and only the archive was missing**
> — which is indistinguishable from fabrication until someone re-runs it. Every reported number needs
> a file, and every fit needs its point set named.

**Property 3 (NEW): `κ_eig(Ã⁻¹A_σ) · u_inner ≤ 1e-2`, RECORDED — a diagnostic, not a license.**
CHOSEN threshold; MEASURED **6.716e-04** at n=1 920 (6.712e-04 at n=480, 6.715e-04 at n=960 — the
series is flat to 4 significant figures) ⇒ **14.9× margin**, n-independent by property 2.

> **This was drafted as "the gate that licenses f32-inner" on a citation that does not support it.**
> The draft attributed the threshold to `[CH18]`'s `φ = (c₁κ∞(A)+c₂)u_s`. Checked against the source
> (Carson & Higham, *Accelerating the Solution of Linear Systems by Iterative Refinement in Three
> Precisions*, SISC 40(2) 2018, https://eprints.maths.manchester.ac.uk/2629/): **(i)** CH18's
> conditions are on `κ∞(A)` — the **original** matrix, **infinity** norm — not the preconditioned
> operator in the 2-norm; **(ii)** its `κ∞(A) ≤ 1e8` relaxation is derived specifically for **GMRES
> preconditioned by low-precision LU factors of A**, and Phase 2H has **no factorization at all**
> (§1.1), so the framework does not apply in either direction — this spec can borrow neither its
> ceiling as a limit nor its `φ < 1` as a license; **(iii)** the `φ` form could not be located in
> CH18 at all — the nearest match is `u_s‖E_i‖_∞ < 1` at eq. (3.8) of the **five**-precision successor
> (https://www.lip6.fr/Theo.Mary/doc/GMRESIR5.pdf), so `[CH18]` is a probable misattribution of a
> later paper and "Cor. 4.2" remains unlocated. **Do not cite `[CH18]` for this threshold.**
>
> **The applicable theory is inexact Krylov**, not iterative refinement: GMRES whose matvec is
> computed approximately, analysed by Simoncini & Szyld and by van den Eshof & Sleijpen, and the
> standard framework for FMM/BEM-accelerated Krylov solves — structurally the same problem as a
> hierarchical approximate matvec inside GMRES. It bounds the **tolerable matvec perturbation per
> iteration**, which is **H1-b's `δ_precision`** (§7.2), not a condition-number ceiling. **So this
> spec already measures the right quantity and cited the wrong reason for it.** What actually licenses
> f32-inner is (a) H1-b's measured bound on the apply perturbation and (b) the f64 outer residual
> formed against the exact operator (§7.3 anti-trap 2) — stated as an argument in §7.9, not deferred
> to a citation.
>
> **Norm caveat, MEASURED rather than conceded.** `κ∞/κ_eig` on the preconditioned operator is
> **1.07 → 1.21** over n=64…960, fitting `n^0.045` ⇒ ≈ **1.41 at n=30 000**. Breaching a 1e-2 bar
> would need ≥ 15×, so the conversion carries ~10× headroom; the worst-case `κ∞ ≤ n·κ₂` bound does not
> bind here. On the *unpreconditioned* operator the ratio is flat at ≈ 3.0. **Any bar that multiplies
> a condition number by a unit roundoff must state its norm.**
>
> **Notation, and it decides reproducibility.** `κ_eig` means `λ_max/λ_min` of `Ã⁻¹A_σ` (equivalently
> `κ(Ã^{-1/2}A_σÃ^{-1/2})`) — what `e4_precond_integrity.py:37-38` computes and what governs Krylov
> convergence. `Ã⁻¹A_σ` is a product of two SPD matrices and is **not symmetric**, so the
> singular-value condition number `cond(Ã⁻¹A_σ, 2)` is a *different* quantity: MEASURED 1.4544e4 vs
> 1.0977e4 at n=64, a **1.33× gap** persisting across the range. The draft wrote `κ₂`, which denotes
> the latter. An implementer who reads `κ₂`, calls `cond(M @ A, 2)` and lands 1.33× off the ledger
> cannot distinguish a notation mismatch from a bug. **Record both.**

The table below is why property 3 is worth recording even demoted — **it is the only quantity in the
set that is non-vacuous at n=30 000 and bears directly on precision:**

| n | `κ₂(A_σ)` | `κ₂(A_σ)·u_f32` | `κ₂(Ã⁻¹A_σ)·u_f32` |
|---:|---:|---:|---:|
| 960 | 1.393e+06 | 8.300e-02 | **6.715e-04** (MEASURED) |
| 1 920 | 1.404e+07 | **8.369e-01** | **6.716e-04** (MEASURED) |
| 4 000 | 1.622e+08 | 9.668e+00 | 6.716e-04 (ASSUMED flat) |
| 30 000 | 1.341e+11 (EXTRAPOLATED) | **7.991e+03** | 6.716e-04 (ASSUMED flat) |

`κ·u_f32 ≈ 1` at **n ≈ 2 025** on the unpreconditioned operator (DERIVED from this table's own
MEASURED 0.8369 at n=1 920; the first draft said 1 900, which the same table contradicts) — one of the four f32-wall estimates
in §6 — **while the preconditioned number does not move at all.**

> **FIXTURE IDENTIFIED + INDEPENDENTLY REPRODUCED, A38 item 8 (2026-10-12).** This table is the
> **trefoil**. A four-shape probe of `cond₂(D^−1/2 A_σ D^−1/2)` gives 1.233e6 / 1.237e7 / **1.425e8** at
> n = 960 / 1 920 / 4 000 — **11–13 % of this table at every rung, including the n = 4 000 entry that was
> EXTRAPOLATED here** (mine is lower by the amount Jacobi scaling predicts). So the table is confirmed
> AND its `n ≈ 2 025` is shape-scoped: the same probe gives **n ≈ 394 (arc), 948 (circle), 2 104
> (trefoil), 4 907 (helix)**. The exponent is geometry-independent; the prefactor varies **4 506×**.
> The preconditioned column's flatness is correspondingly a one-shape measurement — see A38 item 8 (iii).


*INVALID.* `dim null == 0` — structurally impossible, since the constants are in the nullspace by
construction for every variant, so a zero nullity means the rank tolerance is mis-set. Or
`dim null ≥ 3` ⇒ **INVALID, not RED**: the known defect gives exactly 2 (one extra mode, the
checkerboard), so 3+ is not a known failure shape and most likely means a mis-set tolerance or a
different operator, and must not be scored. Or `λ₂(L^{2−s})/λ_max(L^{2−s}) < 10 ×` the relative rank
tolerance (`e4_precond_integrity.py:33` uses `|λ| < 1e-8·λ_max`) ⇒ **INVALID — scoped to the
production (vertex-collocated) variant ONLY.** That scoping is not cosmetic: MEASURED, the
`rs-identity` knock-out variant has `λ₂/λ_max ≈ 1e-15…1e-16` and so **trips this predicate at every
n**, which would report the knock-out as INVALID instead of RED and defeat it — the identical error
class as #59 classifying H4b's `q ≤ 0` as INVALID (§7.6). The knock-out's tiny ratio is *correct*
behaviour: its nullity genuinely is 2, so `λ₂` genuinely is a null mode. **Now MEASURED
(`e4-closer-0b.json`), and the margin is ~10⁶×**, so this predicate is a free always-pass guard rather
than a live hazard: `λ₁ ≈ 3.55e-15` (the genuine constant mode) while `λ₂/λ_max` runs **0.204 →
0.070** over n=64…480 against a 1e-7 threshold, decaying only as `n^-0.53` — still ≈ 8e-3, i.e. 8e4×
above threshold, at n=30 000. **The nullity-1 result is structural, not tolerance-determined** —
which the first draft could not assert, because `λ₂` existed in no artefact. Keep the predicate (it is
free) and keep reporting all four values.

*Knock-outs.* (1) The literal `[RS §5.1.1]` identity `L^σ v = 2Uᵀ[diag(diag(a)⁻¹Ha) − H]Uv` ⇒ nullity
**2**, `κ₂ = 1.014e+15` ⇒ **RED on all three properties** (property 3: 1.014e15 × 5.96e-8 = 6.0e7),
MEASURED. (2) Drop the constant-mode correction `+11ᵀ/(N·σ)` ⇒ `Ã` singular on constants ⇒ nullity
RED; NOT MEASURED, cheap. (3) *Must NOT trip:* the correct vertex-collocated variant at every n ⇒
nullity 1, `p = 0.006450`, `κ·u = 6.716e-04` ⇒ **GREEN** on all three, MEASURED.

**Why H5 is the only gate that catches this defect.** With the defective preconditioner,
`gmresIters1e-4` is identical within 1 at every n (H3 green), `cosAMin ≥ 0.9412` (H4a and H4b′ green),
and `ε_A` degrades only 14–75 % — inside any plausible direction bar. MEASURED. The damage is mild at
τ=1e-4 and bites as τ tightens or accumulates over descent steps.

**Range.** Property 1 is an integer at any n (inverse iteration, matrix-free, O(N) storage).
Property 2 needs the explicit dense preconditioned operator, so it is **limited to n ≤ 1 920**; its
large-n observable is H3's ratio. **Property 3 inherits that range, so `κ₂(Ã⁻¹A_σ)` above n = 1 920 is
ASSUMED flat, not measured — and that assumption is load-bearing for the entire mixed-precision
scheme.** §7.11 item 2 names the run that closes it.

### 7.9 H6 — the θ default, under an adversarial rhs

**New gate. It exists because §7.6 removed the instrument §4.1's ruling was resting on**, and because
it is the only cheap measurement that can discriminate θ=0.5 from θ=0.25.

```
Quantity: min over the adversarial rhs set of cos∠_A(x̃, x*) at θ = 0.5,
          f64 operator, left-preconditioned GMRES, τ = 1e-4, x* from a dense f64 solve.
Adversarial rhs: built from the generalized eigenvectors of (A_bct,σ, A_σ) whose
          generalized eigenvalue lies below 0.9 at that n — the "bad modes".
Gated at:  n = 960 and n = 2 000.
Bar:       min cos∠_A ≥ 0.9.
```

**Why it can actually fail, which is what makes it a gate.** Every `cos∠_A` in the entire corpus is
measured against **5 random rhs**, and a converged flow is exactly where the rhs stops being random.
At θ=0.5 the MEASURED worst generalized eigenvalue is `c₁ = 0.2886` (n=2 000), giving a worst-case
direction guarantee `2√(c₁c₂)/(c₁+c₂) = ` **0.8234** — *below the bar* — while every measured random
rhs gives ≥ 0.9955. The gap between 0.8234 and 0.9955 is the entire θ risk, and nothing else in the
set can see into it: H4b′ cannot (§7.6 caveat 2), and H4a has only ever been pointed at random rhs.

**RED ⇒ the θ default flips to 0.25; it does not kill the phase.** θ=0.25 is MEASURED to have **zero**
modes below 0.9 at every n from 240 to 4 000 and a worst-case guarantee of 0.9984. The cost of the
flip is MEASURED: 2.09× near-field work (`κ_near` 108.2 → 226.2 at n=30 000) and the GPU projection
moving from 0.10–0.51 s to 0.20–1.07 s per solve (EXTRAPOLATED, §3.6).

*INVALID.* No generalized eigenvalue below 0.9 at that n (there is then no adversarial direction to
build — report **NOT APPLICABLE** at that n, which is itself a publishable result); or `x*` not from
an f64 dense solve.

**Cost: low, and that is the point.** The bad modes are already computed by `scratch/run64_badmodes.py`,
and the gate needs one dense f64 solve per n ≤ 2 000. #64 calls this *"cheaper and far more decisive
than another `c₁` point"*; this spec agrees and promotes it from a deferred question to a
pre-registered gate.

**Clock: none.**

### 7.10 R1 — three ceilings, not one AND-ed number

R1 is a **mandatory reporting obligation, not a kill gate.** #59's form — "the largest n where H1, H2
and H4b are simultaneously green" — now reports a falsely **HIGH** ceiling: H4b′ cannot fail on the
real operator (§7.6), H2 is non-vacuous to n=30 000 (§7.3), and H1 has no magnitude bar (§7.2). **An
AND of green verdicts over gates that cannot fail is not a ceiling.**

```
Publish THREE ceilings, each with (a) the gate that bound it, (b) that gate's VALUE at the
first n where it stopped being green, (c) the precision configuration it applies to:

  servedN_f32only     = 960   MEASURED. Bound by H4a (f32 cos∠_A = 0.7732 at n=2 000, θ=0.1;
                              0.9042 at θ=0.25), corroborated by the f32 true-residual floor
                              5.04e-3 at n=960. Four derivations put the crossing at 1 200-2 000;
                              960 is the last MEASURED pass.

  servedN_referenced  = 960   MEASURED. Bound by the dense f64 reference's own degradation
                              (ε_A(δ=0) jumps 7.8x at n=1 920). Hard cap N ~ 5 793 (§1.2).
                              Above this n, NO gate in the set certifies direction quality.

  servedN_mixed       = NOT ESTABLISHED.  Blocked by two named measurements (§7.11 items 1 and 2),
                              not by a red gate. Publish the words, never a number.

Rules:
  - If any contributing gate returns INVALID at or below a reported ceiling, that ceiling is
    INVALID and is published as "NOT ESTABLISHED" together with the INVALID predicate that fired.
  - "NOT ESTABLISHED" is a required publishable value. A ceiling that no gate binds must never be
    reported as a number - memory `gates-name-their-impossible-values`, applied to a report.
  - Each ceiling carries its binding gate's VALUE, not only its verdict.
```

### 7.11 Where the f32 wall sits, and what the gates can and cannot see

> **SCOPE, A38 item 10 (2026-10-12): every derivation in the table below is the TREFOIL, and three of
> them share one instrument family.** The agreement is therefore common-mode on geometry — and
> derivation 3, re-measured on four shapes (A38 item 8), spans **n ≈ 394 (arc) to 4 907 (helix)**, with
> the arc's wall BELOW the n = 960 floor. The geometry-independence of derivations 1, 1b, 2 and 4 is
> **UNTESTED, not established.** Per this section's own rule, an unbound ceiling is "NOT ESTABLISHED".

**#59's n ≈ 3 310 is superseded.** That crossover reproduces exactly (3 310.4 from
`α_denseF32 = 1.555e-08·n^0.3697` against `β_rand = 1621·n^-2.7605`) and is wrong for one reason: it
substitutes the **dense `sgemv`** f32 floor for the **hierarchical apply** noise, which §7.2 shows is
5–13× larger and grows 2–3× faster. Four independent derivations now agree on **1 200–2 000**:

| # | derivation | crossover | tag |
|---|---|---:|---|
| 1 | MEASURED hierarchical f32 apply noise (low kernel, `n^1.176`) vs `β_rand` | **n ≈ 1 228** | EXTRAPOLATED |
| 1b | same, high-order kernel (`n^0.829`) | **n ≈ 1 773** | EXTRAPOLATED |
| 2 | MEASURED f32 left-prec true-residual floor `4.09e-13·n^3.364` reaching 1e-2 | **n ≈ 1 225** | EXTRAPOLATED |
| 3 | `κ₂(A_σ)·u_f32 ≈ 1` on the unpreconditioned operator (§7.8) | **n ≈ 2 025** | DERIVED from a MEASURED law |
| 4 | #64's direct observation: f32 `cos∠_A` = 0.9244/0.9042/**0.7732** vs f64 ≥ 0.9983 | breached **at n = 2 000** | **MEASURED** |

**The one statement that needs no fit at all:** at n=1 920/2 000 the MEASURED hierarchical f32
low-kernel apply noise (**8.123e-06**) is **5.9× ABOVE** the MEASURED random-direction tolerance
`betaCos90Min` (**1.383e-06**); at n=960 it is **0.30×** of it (3.002e-06 vs 9.912e-06). **The
crossing therefore happens strictly inside (960, **2 000**], measured, with no extrapolation —
the upper end is 2 000 because `f32NoiseLow` is measured at n=2 000 while `betaCos90Min` is measured at
n=1 920, and **no n above 960 carries both quantities.** The fit-free claim is the bracket, not its
upper endpoint.** The
random-direction tolerance is the right comparison *for f32 rounding specifically*, because rounding
noise **is** generic — #64's structured-vs-random distinction applies to the compression error, not to
the precision error.

**Pre-registered statement: the f32-only configuration is OUT OF RANGE for n > 960**, 960 being the
largest n at which an f32 left-preconditioned solve is measured to reach a fixed accuracy target at
all. **This is the gates' output, not their failure.** Which gate says it:

| gate | at the f32 wall |
|---|---|
| **H1-b** | **RED.** EXTRAPOLATED `δ_precision` (low kernel) = 2.06e-04 at n=30 000 against a bar of 6.76e-05. The gate that detects it *in the operator* |
| **H2** | **does NOT detect it.** Backward error is insensitive by construction — the solve is backward-stable with respect to the operator that was actually applied |
| **H3** | **INVALID, not RED** — a fixed `τ_η` becomes unreachable, so the censoring predicate fires. That is the correct output: the configuration cannot be costed because it cannot converge |
| **H4a** | **RED** where a reference exists (n ≤ 960 measured; 0.9983 vs 0.7732 at n=2 000) |
| **H4b′** | **does NOT detect it** (§7.6 caveat 3) |
| **H5 property 3** | **GREEN**, because it measures the *preconditioned* number — which is exactly the point |

> **The uncomfortable summary, stated rather than buried: above n ≈ 960 no reference-free gate in this
> set can detect the f32 wall.** The only two detectors are H1-b (operator-level, and its n=30 000
> value is EXTRAPOLATED) and H4a (needs a dense f64 reference, so it stops at n≈960). This is the
> strongest single argument for keeping H4a despite its range, and for item 1 below.

**What the mixed-precision scheme is gated on instead.** CLAUDE.md's constraint — the f32 BCT is the
INNER solver, never the solver — becomes gateable as:

1. **H5 property 3** (`κ₂(Ã⁻¹A_σ)·u_inner ≤ 1e-2`, MEASURED 6.716e-04, flat in n, 14.9× margin).
   **RECORDED, not the licensing gate** — see §7.8. `κ₂(A_σ)·u_f32` would fail at n ≈ 2 025; the
   preconditioned number does not move. **What licenses f32-inner is (a) H1-b's measured bound on the
   apply perturbation and (b) this clause's f64 outer residual** — an inexact-Krylov argument
   (Simoncini & Szyld; van den Eshof & Sleijpen), not an iterative-refinement one. Attainable accuracy
   is governed by the **f64** outer residual against the exact operator, which is why
   `κ₂(A_σ)·u_f32 = 7 991` at n=30 000 rules out **f32-only** — already out of range per §7.11 —
   without ruling out f32-inner.
2. **H2's anti-trap 3 promoted to a structural requirement:** the residual driving the outer
   refinement is formed in **f64**, from `b` and the exact operator, never from the f32 BCT used
   inside. H2's `η_exact` bit-equal to `η_bct` INVALID clause is what enforces it.
3. **H1-b's `δ_precision` is the inner solver's accuracy budget**, recorded per θ, per n, per
   precision.
4. **`u_inner` and `u_outer` are recorded per run, and a bar derived at one precision pair is NOT
   transferable to another.** This applies to H2's floor with force: **E1's outer algebra ran in f64
   with only the matvec in f32**, so every `etaFloor` quoted in this spec is for an f64-outer /
   f32-matvec solve. If orthogonalisation moves to f32, H2's bar must be re-derived. NOT MEASURED.

**The caveat that cannot be removed, and whose sign is now known.** Every f32 arm in this entire
corpus is a **dense f32 matvec of the hierarchical matrix** (`scratch/gmres_lab.py:67-75`), not a
genuine f32 hierarchical apply. #59 argued that this is conservative; §7.2 shows the opposite by
5–13× for the apply noise alone. Whether the same reversal survives once the f32 apply is driven
through the whole solve is the single highest-value unknown in this spec.

**Un-run measurements that §7 depends on**, ordered by what they could change. Each is a task in §11.

| # | measurement | what it changes | cost |
|---|---|---|---|
| 1 | **A genuine f32 hierarchical apply driven through the whole solve** (`applyKF32` → left-prec GMRES, n=480…1 920, 5 seeds, `cos∠_A` vs dense f64 `x*`, η vs χ=0) | `servedN_mixed`; H1-b's n=30 000 verdict; whether the 1 200–2 000 bracket moves. **The one run that could invalidate §7.11's range ruling** | low |
| 2 | ~~E0 with a JSON emitter at n=64…1 920~~ **DONE** (`e0-closer-0b.json`): flat to n=1 920, `p = 0.006450`, `κ·u = 6.716e-04`, 14.9× margin, and it reproduces #59's two unarchived values exactly. **Still open above n=1 920** — matrix-free Lanczos at n=4 000, 8 000 | H5 p2/p3's range, hence `servedN_mixed`. Flatness above 1 920 remains **ASSUMED** | low–medium; the preconditioned operator is matrix-free, so no dense matrix is needed |
| 3 | ~~`iters(n)` to a fixed `τ_η`, left-preconditioned~~ **DONE** (`e1-closer-0b-n*.json`), and it **changed the gate**: `τ_η = 1e-4` is vacuous (26→3), `1e-6` is the usable rung (31/35/36/32). §7.4 is rewritten around it. **Still open: n=960 and 1 920 rungs** | H3's entire calibration | done for n ≤ 480 |
| 4 | ~~The iteration index at which η reaches its floor~~ **DONE** (`etaArgminIter` = **36–60**), which moved H2's ruling from a uniform `p = 20` to a two-stage dense-then-sparse schedule (§7.3) | H2's probe schedule | done |
| 5 | **H6's adversarial rhs** (§7.9) | the θ default | low |
| 6 | **One wall-clock GPU matvec at n=30 000** on the pinned Quadro RTX 3000 (amortised-K protocol, ICD pinned, `nvidia-smi --query-compute-apps` confirmed inside the window — #63 makes `timestamp-query` useless here) | **H3's primary bar is only as good as the 7.8 ms it divides into**, and that number rests on zero GPU runs | low |
| 7 | **H1-b's missing knock-out** — truncate the near-field accumulator to f16 and confirm `δ_precision` exceeds the bar (model predicts 4.7e-2, 820× over) | H1-b currently has a measured value and a measured must-NOT-trip, but no measured defect that exceeds its bar | low |
| 8 | ~~`λ₁`, `λ₂`, `λ_max` and the rank tolerance~~ **DONE** (`e4-closer-0b.json`): `λ₂/λ_max` = 0.204→0.070 against a 1e-7 threshold, a **~10⁶× margin** decaying only as `n^-0.53`. H5 property 1's nullity is **structural, not tolerance-determined** | closed | done |
| 9 | **H4c at n=960 with its 6 missing control arms** | H4c's bar is calibrated at K=20 on n=240/480 only | ~3 h |
| 10a | **H3's n=960 and n=1 920 rungs at `τ_η = 1e-6`** — the ladder is measured only to n=480, so `G` currently has one true doubling (240→480) in the correct currency | H3's companion `G`, and whether `iters` stays ~32 or grows | low — same script, 2 more n |
| 10b | **One GPU wall-clock matvec**, whenever a kernel exists | **H3's budget-derived bar is unanchored without it** (§11.2 records it as out of scope for this phase) | blocked on §1.4 |
| 10c | **H4c conjunct 1's RED side** — a random arm forced through all K=20 steps by accepting every trial regardless of Armijo | conjunct 1 currently has a measured green side and **no** valid red side (§7.7) | low |
| 10 | **`tangle`-class geometry, and any n > 4 000** | H1-a's monotonicity, H4a's baselines and §7.11's crossing are all single-geometry (`trefoil`), while `κ_near` spans **14.67×** across geometries | medium / high |

### 7.12 Result emitter

`bench/results/<date>-gpu-phase2h-gate-calibration.json`. Changes from #59 §7: `beta2x*` and
`deltaBar` **removed** (retracted criterion, §10.2); `betaCos90*` **not added** (no δ bar survives,
§7.2); `etaBar` replaced by `etaFloor*` plus the absolute bar; `h1a*`, `h1b*`, `h4b*`, `h4c*`,
`h5p3*`, `h6*`, `r1` added. **Every labelled non-gate value gets an explicit key so it cannot be
silently promoted to a gate.**

```jsonc
{
  "issue": "utof/repulsive-test2#58",
  "generatedAt": "<ISO>",
  "gitShaShort": "<spec commit's parent>",       // bench/gpu/README.md "Provenance"
  "adapterCert": { /* AdapterCertification verbatim; adapterCert.ts:47 */ },
  "precision": { "uOuter": 1.11e-16, "uInner": 5.96e-08, "operator": "f32", "outerAlgebra": "f64" },
  "clock": null,                                  // no gate in H1-H6 reads one (#63)
  "fixtures": { "family": "trefoil", "leafSize": 8, "quadrature": "endpoint4" },
  "seeds": [0, 1, 2, 3, 4],
  "thetaLadder": [0.5, 0.25, 0.1],
  "chiZeroStreaming": true,                       // §7.1 requirement, not a preference
  "rows": [{
    "n": 960, "theta": 0.5, "kappaNear": 93.3, "kappaFar": 3.68,
    "sigma": 19168.9, "normAsigmaEst": 4.56646e4, "normAsigmaSource": "power30@theta",
    "normAsigmaLowerBoundCheck": true,            // H2 INVALID #4
    "kappaASigma": 1.39265e6, "kappaPrecond": 1.12673e4,

    // H1 - recorded triple, per direction then min/median/max
    "deltaCompressPerSeed": [], "deltaCompressMedian": 0, "deltaCompressMax": 0,
    "deltaPrecisionPerSeed": [], "deltaPrecisionMax": 0,
    "deltaTotalPerSeed": [], "deltaTotalMax": 0,
    "h1aLadder": [{ "theta": 0.5, "deltaCompressMedian": 0 }],  // one entry per rung, SAME n
    "h1aMonotone": true, "h1aTieFlagged": false,
    "h1bBar": 5.78e-5, "h1bValue": 1.533e-6,

    // H2 - run to floor
    "etaFloorPerSeed": [], "etaFloorMin": 0, "etaFloorMax": 0,
    "etaProbeEvery": 20, "etaMaxit": 600, "etaFloorIterIndexPerSeed": [],
    "etaBct": 0,                                   // the mis-measured twin; INVALID if bit-equal
    "recurrenceResidualAtStop": 1.0e-4,            // LABELLED NON-GATE
    "trueRelResidualAtStop": 6.378e-1,             // LABELLED NON-GATE

    // H3
    "tauEta": 1e-12, "precondSide": "left", "h3Operator": "f64", // A31: 1e-8 and 1e-10 DEGENERATE on
                                                 // the shipped operator (15 and 29 at n=7680, 47 %
                                                 // and 74 % of peak); 1e-14 MEASURED at n=7680 (56)
                                                 // projects 67.8 > the 64 bar -- A34, no longer an
                                                 // extrapolation. f32 cannot reach 1e-8 or tighter
                                                 // at any n.
    "etaArgminIterF32": [],                      // the f32 companion, RECORDED (run-to-floor)
    "itersPerSeed": [], "itersMedian": 0, "maxit": 400, "itersCensoredCount": 0,
    "growthRatio": 0, "growthQuantisationBand": 0, "growthValidAtThisN": true,
    "perIterationMsWallClock": null,               // LABELLED NON-GATE; re-derives H3's bar (item 6)

    // H4
    "h4aCosPerSeed": [], "h4aCosMin": 0, "h4aEpsAPerSeed": [], "h4aEpsAMedian": 0,
    "h4bQPerSeed": [], "h4bQMin": 0, "h4bCosHatPerSeed": [],  // cosHat RECORDED, never gated
    "h4bOverstatementRatio": [],                   // cosHat/cos_true = 1/sqrt(1-epsA^2); H4a calibrates H4b
    "h4bMatvec": "theta", "h4bQChiZero": 0,        // both recorded at n <= 960 (§7.6)
    "h4cK": 20, "h4cAcceptedFraction": 1.0, "h4cDeRatio": 0.99909,
    "h4cTauMedian": 0.03515625, "h4cTauMedianReference": 0.09375,  // RECORDED, not gated
    "h4cReasonCounts": { "zero_or_nonfinite_gradient_norm": 0, "not_a_descent_direction": 0,
                         "tau_below_min": 0, "armijo_failed": 0, "projection_failed": 0 },

    // H5
    "nullDimLfrac": 1, "lfracLam1": 0, "lfracLam2": 0, "lfracLamMax": 0, "rankTol": 1e-8,
    "kappaPrecondExponentP": 0.01215, "h5p3Value": 6.716e-4, "h5p3Bar": 1e-2,

    // H6 - gated once, at n = 960 and 2000
    "h6BadModeCount": 0, "h6CosMin": null, "h6Applicable": true,

    "verdicts": { "H1a": "GREEN", "H1b": "GREEN", "H2": "GREEN", "H3": "GREEN", "H4a": "GREEN",
                  "H4b": "GREEN", "H4c": "GREEN", "H5": "GREEN", "H6": "GREEN" },
    "seconds": 0
  }],
  "r1": {
    "servedNF32Only": 960, "servedNF32OnlyBoundBy": "H4a", "servedNF32OnlyValue": 0.7732,
    "servedNReferenced": 960, "servedNReferencedBoundBy": "dense f64 reference range",
    "servedNMixed": null, "servedNMixedStatus": "NOT ESTABLISHED",
    "servedNMixedBlockedBy": ["genuine f32 hierarchical apply unmeasured",
                              "kappaPrecond unmeasured above n=1920"]
  }
}
```
<!-- GATES:END -->

---

## 8. The constraint fork (#62)

### 8.1 The problem, verified in source

`src/core/sobolev/constraintSet.ts:206-250` (`edgeLengthsBlock`), lines `:211-213`:

```ts
const n = vertices.length;
const m = edges.length;
const phi = new Array<number>(m).fill(0);
const C: number[][] = Array.from({ length: m }, () => new Array<number>(3 * n).fill(0));
```

So `C` is `|E| × 3n`. **On a closed polyline `|E| = n`, so `k = O(n)` exactly** — not a loose bound.
With `barycenterBlock` stacked, `k = |E| + 3`. The set is assembled at `src/core/dispatch.ts:244`.

With a dense factorization, `k+1` backsolves against an already-computed factor are cheap — the
factor is the expensive part and it amortises. **With an iterative solve there is no factor to
amortise**: each of the `k+1` systems is a fresh GMRES run with its own matvecs and preconditioner
applications. At `k = O(n)` the per-step cost goes from `O(N·iters)` to `O(N²·iters)` — **worse than
the dense path it replaced.**

**And the escape hatch is gone.** `[DESIGN §1:112-115]` declared `perEdge` out of scope on the
grounds that *"perEdge steps run the Phase 1 path (GPU dE, CPU factor)"*. That is only available
while a CPU factor exists. #51 retires the factor, so the workaround retires with it.

### 8.2 The ruling: a hybrid, Schur for `O(1)` blocks and nullspace for `edgeLengths`

```
barycenterBlock   (3 rows)        → Schur complement. k = O(1), Schur is cheap here.
pointBlock        (3 rows/pin)    → Schur complement. Same reason.
totalLengthBlock  (1 row)         → Schur complement.
edgeLengthsBlock  (|E| rows)      → NULLSPACE route.
```

Two facts make the hybrid the natural answer rather than a compromise:

- `k = O(1)` blocks are exactly where the Schur complement is cheap, so there is nothing to gain by
  moving them.
- `docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md:153` records that
  `totalLengthBlock` and `edgeLengthsBlock` are **MUTUALLY EXCLUSIVE in one set**. So the
  combinations that must actually work are: `{barycenter, totalLength, pins}` (all-Schur, the
  existing path) and `{barycenter, edgeLengths, pins}` (hybrid). **There is no set in which both
  routes must handle `O(n)` rows simultaneously.**

### 8.3 What the implementation must settle

1. **Represent the nullspace implicitly.** Never form a `3n × (3n−k)` basis. For `edgeLengths` on a
   closed polyline the Jacobian has a one-row-per-edge structure whose nullspace projection is an
   O(n) operator — specify it as a matvec, like everything else in §5.
2. **Order of projection and preconditioning.** Both are linear operators applied per GMRES
   iteration, and **the order decides whether the projected operator stays symmetric positive
   definite**. Get this wrong and GMRES converges on a non-symmetric operator while every residual
   looks fine — the §5.3 failure mode again. ~~The projected-preconditioned operator must be verified
   SPD on the constrained subspace before any convergence number is quoted.~~
   **GATE AMENDED — A37 item 2. The sentence above is not implementable as written and, taken
   literally, REJECTS THE CORRECT ORDER:** a product of two SPD matrices is not symmetric, so the
   shipped order's Euclidean `symDefect(Ã⁻¹A)` on the constrained subspace is **1.40e-1 and must
   be**. Worse, a restriction-based symmetry/spectrum test is **BLIND to the one order this warning
   is about**: `Zᵀ P = Zᵀ`, so `P Ã⁻¹ P · P A P` and `Ã⁻¹ P A P` have the same restriction, symDefect
   and spectrum **to every digit**, and only V-leakage separates them (2.59e-13 vs 4.10e-1).
   **Three quantities are required, all three:** (i) **leakage** `max_z ‖(I−P) K z‖ / ‖K z‖`;
   (ii) symmetry AND positivity of `Zᵀ P A_σ P Z` — *the operator being inverted, NOT the
   preconditioned product* (measured 6.05e-16, `λ_min = +0.651`); (iii) self-adjointness in the
   preconditioner's own inner product. Still binding: **no convergence number before all three.**
3. **`[G6]`'s separate exclusion still applies.** `[DESIGN]:49-51` excluded `perEdge` for a
   **conditioning** reason — `κ·u_f32 > 0.1 at N ≥ 480` — which is independent of the solve
   architecture. **A nullspace route does not inherit an exemption just because it avoids the Schur
   complement.** It must be checked against that bound separately, and §6's mixed-precision ruling
   is what it has to lean on.
   **FRAMING INVERTED — A37 item 3: the projection is NOT the conditioning risk, the OPERATOR is.**
   Measured: `cond(P A_σ P|V)/cond(A_σ) = 0.960 ± 0.003`, i.e. projecting *improves* conditioning by
   4 % (Cauchy interlacing). Crossovers for `κ·u_f32 = 0.1`: the **operator** at **n ≈ 1 027**, the
   projection solve at **n ≈ 8 956** — the operator binds **8.8× earlier**, which independently
   reproduces §6's f32 wall at n ≈ 1 228–1 773 from a new direction. The conclusion of the paragraph
   above is right (§6's ruling is what the route leans on); the quantity it points at is not the
   binding one. **Separately TRUE and newly measured:** the projection solve's own
   `cond(C_e C_eᵀ)` grows **exactly `n²`** (160× range, `cond/n² = 0.0209`) to **1.883e7 at
   N = 30 000**, so the projection must stay **f64** above n ≈ 9 000 — cheap, but deliberate.

Reference: `[YSC]` (arXiv:2006.07859) constraint handling; Repulsor at `DOM_DIM=1`.

---

## 9. Risks not covered by gates

### 9.1 GPU cost (highest)

Every GPU figure is extrapolated through an unmeasured 40–200× bandwidth factor (§3.6), and the
near-field precompute (~3.5 s on one CPU thread, §3.3) has **no GPU estimate at all**. §7 contains
no timing gate because there is nothing honest to set one against.

**The measurement protocol, when a kernel exists.** #63 is the reason it is not the obvious one:
Firefox **advertises `timestamp-query` and returns all zeros**, and zeros are indistinguishable from
a very fast kernel — it fails silently in the direction that *confirms* an optimistic prediction.

1. Wall clock only. Amortise: K ≈ 50–200 matvecs, **one** `mapAsync` at the end, minus a K=0
   baseline, ÷K.
2. Separately measure single-matvec-plus-readback. **The gap is the per-iteration readback tax GMRES
   pays for its dot product** — which is the main GPU design risk for this phase, and the reason #51
   expected Phase 3's zero-readback work to land with or before 2H.
3. **Pin the Vulkan ICD.** Unpinned, Firefox runs WebGPU on the **Intel UHD 630**, not the Quadro
   RTX 3000 — about an order of magnitude less bandwidth, and with blank `adapterInfo` **nothing in
   the result JSON reveals it**. Confirm with `nvidia-smi --query-compute-apps` *inside* the
   measurement window. `bench/gpu/adapterCert.ts` is the sanctioned route; `classifyAdapter` rules a
   blank record `'software'` **by design — do not relax it.**

### 9.2 Measurement gaps, ranked by what they could change

| gap | why it matters | cost to close |
|---|---|---|
| **Genuine f32 hierarchical apply** driven through the whole solve | both f32 walls rest on a *dense* f32 stand-in (§6); this decides where the mixed-precision switch goes, and it is the one run that could invalidate §7.11's whole range ruling | low — §11 **task 10** |
| **Adversarial rhs** that excites θ=0.5's 0.4–2.2 % bad modes | every `cos∠_A` is against 5 random rhs; a converged flow is exactly where the rhs stops being random. #64 calls this "cheaper and far more decisive than another `c₁` point" | low — **promoted to gate H6** (§7.9) and §11 **task 0c**, because §7.6 removed the instrument §4.1 was resting on |
| **`κ₂(Ã⁻¹A_σ)` above n = 1 920** | ~~MEASURED flat only to n=480 in JSON; **ASSUMED** flat above 1 920, and that assumption carries the entire mixed-precision scheme via H5 property 3~~ **CLOSED by 9c — A38 item 9. 9c's matrix-free Lanczos route — the exact instrument this row costed — swept four shapes to n=4 000. The bar is met with 720× headroom at n=4 000 (vs §6's quoted 14.9×, which is the LAB operator), but the quantity GROWS (n^+0.15…+0.47), so THE ASSUMED FLATNESS IS FALSE WHILE THE RULING SURVIVES. Build on the measured margin, never on flatness.** | ~~low–medium~~ **spent — no new task needed** |
| **One wall-clock GPU matvec at n=30 000** | **H3's primary bar is 0.5 s ÷ 7.8 ms, and the 7.8 ms rests on zero GPU runs** (§3.6) | low — §11 **task 11** |
| **Accuracy on tangle-class geometry** | all direction/SPD/line-search results are on `trefoil` alone; κ_near spans 14.67× across geometries | medium |
| **Accuracy at n > 4 000** | every BCT error number is n≤4 000 (θ=0.5) or n=960 (θ sweep); nothing at the 30 000 target | high — the exact reference is O(N²) |
| `c₁(θ=0.5)` at n ≥ 8 000 | θ=0.5's residual risk is an oscillation of unknown period | high |
| A real optimiser trajectory | all fixtures are synthetic; a curve mid-flow may be worse than tangle | medium |

### 9.3 Cheap unexplored wins

- **Leaf size.** Swept once, at one n, one θ, with no timing. leaf=8 is a convention (§5.5).
- **The adjacency certificate** is conservative and costs +15.4 % (trefoil) / +199 % (circle) — but
  measured **only at θ=0.5**, and it vetoes nothing at θ≤0.4. A sharp O(1) alternative (per-cluster
  min/max edge index + wrap flag) is proposed, unimplemented, unmeasured.
- **Split plane.** Median split is what creates the power-of-two oscillation that weakens every fit
  in this document (§3.2, §4.3). SAH is untested.

### 9.4 What the GPU port must not break

The near field must stay an array-of-structs / CSR-like layout with the two per-kernel value arrays
**separately bound** (§3.4), because that is what keeps the largest single binding at 12.38 MiB
(trefoil) / 36.93 MiB (tangle) instead of 24.8 / 73.9 MiB. And `(−Δ)⁻¹` is a *sequential* two-sweep Thomas
solve — it is 0.83 % of the matvec on CPU, but it does not parallelise, so on GPU it may stop being
negligible. Measure it before assuming the CPU split transfers.

---

## 10. Provenance and corrections

### 10.1 Everything in this spec that is not measured

| claim | status |
|---|---|
| all GPU timings (1–5 ms, 3–16 ms, 0.10–0.51 s) | **EXTRAPOLATED**, zero GPU runs |
| 12-thread CPU (~1.0 s) | **ASSUMED** — no multithreaded run exists |
| `~33 MiB` / `34.3 MiB` f32 memory | **EXTRAPOLATED** (two estimators) |
| `L_{2−s}` cost (+43 ms) and memory (+12 MiB f32) | **DERIVED** from a 3/12 column ratio; never implemented |
| GMRES iteration count at n=30 000 | **EXTRAPOLATED** from n≤960. The n≤960 counts are now MEASURED in the correct configuration (§5.3.1a) and grow ×1.28–1.45 over that range, so "flat" is withdrawn. This is what H3 gates |
| `c_far = 7.2 ns` as a constant | **not constant**: 4.87 → 7.20 ns over n=5 000…30 000. Far is 7 %, so immaterial, but do not call it a constant |
| 33.9 % quadrature disagreement | **report-only**, no saved artefact (§5.4) |
| "near pairs go as leaf²" | **unsupported**; per-vertex it is linear in leaf size (§3.2) |
| "bad modes track BVH leaf alignment" | **hypothesis** — the mechanism is measured for `κ_near`, inferred for `c₁` |
| `[RS §5.1.5]`'s O(N²)-time/O(N)-storage exact reference | **ESTABLISHED by derivation** (§7.1), not cited — RS says only "the action of H is evaluated exactly". The O(N) storage holds **only if the χ=0 path streams**; precomputing near values at χ=0 costs **3.60 GB** at N=30 000, which is the dense storage this phase exists to escape. Measured constant 63.58 ns per pair·column ⇒ 343 s per exact matvec at N=30 000 |
| **`κ₂(Ã⁻¹A_σ)` flat above n = 1 920** | **ASSUMED.** MEASURED flat to n=480 (JSON) and *reported* to 1 920 with no surviving artefact. **This assumption carries the entire mixed-precision scheme** via H5 property 3 (§7.8). §11 task 0b(iv) |
| **H3's per-iteration 7.8 ms**, which its primary bar divides the 0.5 s budget into | **EXTRAPOLATED** from a bandwidth argument with **zero GPU runs**. The bar is a pre-registered constant, so the gate reads no clock — but the bar is only as good as this number. §11 task 11 |
| **Every f32 arm in the entire corpus** | a **dense f32 matvec of the hierarchical matrix** (`scratch/gmres_lab.py:67-75`), not a genuine f32 hierarchical apply. #59 called this conservative; §7.2 falsifies that by 5–13× in the **unsafe** direction. §11 task 10 |
| H1-b's knock-out, H5's rank-tolerance margin, H4c at n=960 with controls, any `tangle` or n>4 000 accuracy | **NOT MEASURED.** Each is a numbered item in §7.11 with the run that closes it |

### 10.2 Retracted claims — do not quote these

| retracted | by what | replaced with |
|---|---|---|
| "GMRES iteration count is unmeasured at any n" (#57) | reading #59's JSON, in the same directory | 21–26 to 1e-2, 28→39 to 1e-4, flat n=64…960 |
| "f32 is free here, so trap 3 is not a 2H blocker" (#57) | #59's bisection; #64 §7.1 | f32 walls at **n≈1 200–2 000**, θ-independent (§6, §7.11). **Wrong in the dangerous direction** |
| "θ=0.5 is probably not a usable operating point" (#57 §0, §7.9, §9.2) | #64 | VIABLE at θ=0.5 guarded by H4b′ + H4c + H6 (§4.1). **#57's body was never amended — do not quote it on θ** |
| `β` self-anchoring "≤ 2× the δ=0 baseline" criterion (#59) | its own n=1 920 point, off by **28×** | absolute `cos∠_A ≥ 0.9`. Cause: the δ=0 anchor itself jumps 7.8× at n=1 920, so the rule silently loosened exactly where the solve was failing. **A relative criterion cannot gate a quantity whose anchor is degrading** |
| `c₁(θ=0.5) ∝ n^{-0.583}` ⇒ recommend θ=0.25 (#64) | its own fifth data point | no trend; R²=0.216 (§4.3) |
| "a real BCT's β comes out lower than the random model" (#59's flagged prediction) | #64 | **655× to 921 600× higher** |
| "bad modes sit at self-approach regions" (#64's own hypothesis) | #64's own measurement | falsified; they sit where the curve is *farther* from itself (§4.3) |

**A30, 2026-10-11 — every LAB figure task 7b could not reproduce on the SHIPPED operator.** These are
not errors of arithmetic; each is a number measured on the lab preconditioner (`gmres_lab.py` /
`e1_gmres.py`, dense, no compression, `κ_excl ≈ 3000`) and quoted as if it described the shipped path
(hierarchical at θ=0.5, shipped `makePreconditioner`, `κ_excl ≈ 20.3`, i.e. **150× better
conditioned**). **The verdict survives in every row below; only the number dies.** Where a row says
the *reason* dies too, the reason was load-bearing and its replacement is given.

| retracted | measured on the shipped operator | what it changes |
|---|---|---|
| §7.3's `etaFloorMax` 0.76 → 1.98 `u_f32`, fit `n^(+0.3406)`, 6.08 u / 8.2× at n=30 000 | **0.412 → 0.311 `u_f32`**, fit `n^(−0.077)`, **0.249 u / 201×** | A26. Bar ×50 unchanged; the `[CH18]` `n^0.34` remark loses its observation; the 850× gap is **4 073×** |
| §7.3's A21 box: `η_exact/η_bct` = 1.035–1.177 at n=240, 1.018–1.059 at n=960, and *"7b does not need to re-record the floor column"* | **3.7e3–6.9e3 at n=240, 9.5e2–2.0e3 at n=960** (f32 arm) — four to five decades | The lab has **no compression at all** (`e1_gmres.py:32-33` passes the same dense matrix as both `exact_A` and the matvec source), so its only mismatch is f32 roundoff. The box's own rule predicts this. **Its instruction to skip the floor column was wrong — the floor moved (A26)** |
| §7.4's `τ_η = 1e-8` ladder 36/42/43/41/39/37 | **26/29/30/29/32/27, and 19 at n=3840** | 22–31 % lower at every n, and the n=3840 point is the collapse pattern — see §10.4's third failure and A31 |
| §7.4's *"1e-6 is ~25× above the measured floor (≈4e-8)"* | `4e-8` is the **f32** floor; H3 runs on the **f64** arm, whose floor is **2.8e-16…5.9e-16** | The rung was selected for one arm using the other arm's floor. The pinned rung sits **1.9e7×** above the floor it is measured against; applying §7.4's own criterion to the correct floor selects ≈1.2e-14 |
| §7.4's *"the unpreconditioned count at n=960 is exactly 960 = the Krylov dimension … genuine exact termination, which is τ-independent"* | **1 670** (per seed 1700/1670/1670/1800/1640) — **1.74× the Krylov dimension**, `stop='eta-target'`, not censored | **`β_G`'s verdict survives and is stronger** (knock-outs RED at 2.5313 / 2.2877 vs the spec's 2.0518 / 2.0211; production RS GREEN at 1.0309). **The stated REASON does not survive:** there is no exact termination here — Arnoldi loses orthogonality long before step `n` at `κ₂ ≈ 1.4e6`. Do not re-derive anything from "exact termination" |
| §7.3 anti-trap 2's recurrence under-report 4.372e8 → 1.717e12, exponent +3.056 | **85.6 → 2.28e4**, exponent **≈+2.07**; 1.00e6 at n=3840 | Direction and growth reproduce; magnitudes are **6.7 decades** smaller, because the shipped `Ã⁻¹A_σ` is ~150× better conditioned. §5.3's rule stands; its number does not |
| §7.4's `trueFloorMin` 6.68e-07 → 5.04e-03, `n^3.36` | **8.3e-02 → 8.7e-01**, `n^0.88`, **> 1 at n ≥ 1920** | Compression-dominated, not roundoff-dominated. **This REINFORCES §7.4's conclusion** that `η` is the only viable currency — a "floor" above 1 is no floor at all |
| §7.3 / §7.4's `etaArgminIter` = 36–60 / "41–60, flat at ≈53–60 for n ≥ 240" | f64 **63–100** for n ≥ 240 (one seed 180); f32 **27–460 and a PLATEAU, not a located minimum** | *"Every iterate to k=80"* no longer brackets the f64 floor — the probe head of 80 is being pressed. **`etaArgminIterF32` is not a meaningful companion statistic on this operator** and should not be reported as one |
| §7.3's follow-up ruling: `nFar == 0` is *"a configuration nothing currently runs"* | **`nFar = 0` at n=64, extent 1.0, θ=0.5** — reachable from production θ by rescaling the curve | The inverted clause-3 obligation can be triggered without anyone selecting χ=0. **Discharged in place:** the two η agree (3.8614e-16 both) without being bit-equal, `invalid: []` |
| The brief's *"the lab casts only the input vector to f32"* | `gmres_lab.py:67-75` does **all three** — f32 operator, f32 input vector, f32 (sgemv) accumulation — and its own docstring says so | Reviewer B's failure to reproduce the knock-out is still explained by using the f32 *operator* alone, but the lab arm is harsher than described. Any "f32" figure must name **which** f32 |

**CONFIRMED by the same campaign, recorded because confirmations matter too:** §7.3's `η_exact`
minima at θ=0.5 (1.68e-3 / 7.28e-5 / 3.01e-5 / 2.40e-5) reproduce at **0.59–1.02×** and are identical
on both arms to 4–5 digits, so the `δ_compress` monitor is sound; #69's `κ_near` 38.05 / `nFar` 1186 →
64.51 / 2092 reproduce **exactly**; §7.4's *"f32 censored at every n for `τ_η` ≤ 1e-8"* reproduces
exactly; and the INVALID-clause union over all 14 (n, arm) cells is **EMPTY**.

**Gate bars retracted in this spec's recalibration** (evidence:
`.superpowers/sdd/2026-09-11-phase2h-queue/reports/gate-recalibration.md`; rulings: §7):

| retracted | by what | replaced with |
|---|---|---|
| **H4b: `cos ≥ 0.9` on "a monotone computable lower bound"** (#59) | algebra — `c²−a·q = (c−a)² ≥ 0` ⇒ the surrogate is `≥ 1` **identically**, so it is an UPPER bound and the bar sits below its structural minimum. Confirmed on 122/122 of the samples where the surrogate is defined (53 more have `q ≤ 0`, where it is undefined); **16 measured false passes** with true `cos` down to **0.0281**; #59's own knock-outs return INVALID, never RED | `min q > 0` — structural, no free constant. `q ≤ 0` is **RED, not INVALID** (§7.6). This is the repo's own instance of memory `gates-name-their-impossible-values` |
| **#64: "ship θ=0.5 with H4b as a mandatory per-solve guard"** | the above | θ=0.5 guarded by H4b′ (catastrophic only) + H4c (end-to-end) + **H6** (the sub-0.9 modes). **H4b′ cannot distinguish θ=0.5 from θ=0.25** (§4.1, §7.9) |
| **H1: `δ_total ≤ δ_bar = √(α·β)`** (#59) | all three candidate anchors need a **dense** matrix ⇒ unmeasurable above N≈5 793; and `√(αβ) = 4.41e-04` vs a *correct* operator's measured `δ_total = 2.236e-02` at n=960 ⇒ permanent AMBER | **no bar on δ's magnitude.** H1-a gates θ-monotonicity; H1-b gates the precision band (§7.2) |
| **H2: `β_η = 0.43·β_H1`** (#59) | a **perfect** operator (δ=0) is RED at every n by 1.58–3.68× at the 1e-4 stop, and still RED for n≥240 at 1e-6; the 0.43 transfer was measured for random E only | `η ≤ 50·u_f32 = 2.98e-6`, **run to the floor**, independent of H1 (§7.3) |
| **H2's INVALID `η < u_f32/10`** (#59) | as written it rules **every correct f64 run** INVALID — the f64 arms reach `etaFloorMin` of 2.047e-16 | `η < u_op/10`, scoped to the operator's own precision (§7.3) |
| **H3's `α_G`** — both 1.1034 (#59) and the D1-corrected 1.1379 | 64→120 is ×1.875, so neither is an instance of `G = iters(2n)/iters(n)`; **and both count to a fixed recurrence τ whose optimism grows `n^3.056`** — the exact error #59 retracted for H1's β | `α_G` **WITHDRAWN**. Primary bar `iters(30 000) ≤ 64`; companion `G ≥ 2.0 ⇒ RED` (§7.4) |
| **H4a: `ε_A ≤ 2·ε_A(δ=0)`** (#59) | the self-anchoring form #59 itself retracted; its anchor is non-monotone **at n=960** and jumps **7.8×** at n=1 920 | `cos∠_A ≥ 0.9` absolute; `ε_A` RECORDED. `cos∠_A(δ=0)` *improves* with n, 0.9704 → 0.9993 (§7.5) |
| **H4c: median accepted τ ≥ ½ the reference's** (#59) | **8 of 15** measured BCT arms RED, 4 more passing only by exact equality, while **every one delivers ≥ 99.691 %** of the exact energy decrease; the ratio is non-monotone in θ. τ measures step *scale*, not quality | `R_E ≥ 0.95` **and** accepted fraction `== 1.0`; both sides MEASURED (15/15 ≥ 0.99691 vs 6/6 ≤ 0.86027). τ RECORDED (§7.7) |
| **H4c at K=50** (#59) | K=50 has never been run anywhere | **K=20**, the measured configuration (§7.7) |
| **"the hierarchical f32 floor is lower than the dense `sgemv` — conservative in the safe direction"** (#59) | #57's own JSON: hierarchical apply noise 1.06e-6 / 3.00e-6 vs dense 2.23e-07 at n=960, growing `n^0.829`/`n^1.176` vs `n^0.370` | **falsified by 5–13× in the UNSAFE direction.** This is what moves the f32 wall from n≈3 310 to n≈1 228–1 773 (§6, §7.11) |
| **H5 property 2 "measured flat to n=1 920, `p = 0.0010`"** (#59) | `e0_sanity.py` writes no JSON and its log is not saved; the value appears in no artefact, and `p = 0.0010` reproduces from neither point set | **"measured flat to n=480 (JSON); reported to n=1 920 (no artefact)"**, `p = 0.01215` re-derived. ~~Bar `\|p\| ≤ 0.05` survives by ≥4×~~ — **that resolution is itself SUPERSEDED 2026-10-11 (#71, A22): the bar is WITHDRAWN.** `κ_eig` was `λ_max` in disguise (the constant eigenvalue is exactly 1, so the full-spectrum minimum is pinned there); on the constant-mode-EXCLUDED quantity the reference's own exponent is 0.1823, RED against its own bar; and the slope is **anti-correlated with operator quality** (recommended 0.233 vs reference 0.182 while 148× better in absolute `κ_excl`). Replaced by ~~saturation `R ≤ 0.75` +~~ a per-fixture level `κ_excl ≤ 40` (§7.8) — **the saturation half is itself WITHDRAWN by A32**, leaving the level bar as the only numeric bar for property 2 |
| **H3's `τ_η = 1e-4`** (first draft) | MEASURED after the draft: the count collapses `26 → 3` over n=64…480, so `iters(30 000) ≤ 64` is satisfied by **3** with no INVALID predicate to catch it. The reasoning error: a non-degrading residual *floor* does not imply a non-degrading *level* — η's level at fixed iteration count falls as `n^-2.90` | superseded twice; see the next row |
| **H3's `τ_η = 1e-6` on the f32 operator** (second draft — **my own error, 20 min after the first**) | picked from the n ≤ 480 rungs alone (31/35/36/32, apparently flat). The n=960 and n=1 920 rungs are **19 and 12** — the same collapse, one octave out. And it is structural: the f32 η floor is ≈3e-8, so no fixed `τ_η` is both reachable and non-degenerate in f32 | **`τ_η = 1e-8` on the f64 operator**: 36/42/43/41/39/37 over n=64…1 920, doublings 1.0238/0.9535/0.9512/0.9487, `iters(30 000) ≈ 40.6` vs the bar 64. H3's scope narrows to the preconditioner's **spectral** mesh-independence; precision is H1-b's and H4a's job. f32 run-to-floor RECORDED as a companion (§7.4) |
| **"`p = 0.0010` reproduces from neither point set"** (first draft, inherited from the recalibration) | it reproduces as **0.00099 over the four LARGEST points** — the defensible choice for an asymptotic-flatness claim. #59 was right a second time | `p` always quoted with its point set (§7.8) |
| **H5 p1's rank-tolerance INVALID predicate, unscoped** (first draft) | the `rs-identity` **knock-out** has `λ₂/λ_max ≈ 1e-15` and trips it at every n, so the knock-out would report INVALID instead of RED — the same error class as #59 classifying H4b's `q ≤ 0` as INVALID | scoped to the production vertex-collocated variant only (§7.8) |
| **"the two derivations landing on 63.2 and 64 is the strongest single fact in this section"** (first draft) | 64 is a **budget ceiling**, 63.2 a **predicted need** — not two estimates of one quantity. Agreement means **zero margin**, not corroboration. And the n=240 per-seed counts `[38,38,40,40,38]` are a 3–2 median split; taking 40 moves the projection to **71.9**, over the bar | a risk finding in §9.1, not a corroboration claim (§7.4) |
| **H5 property 3 as "the gate that licenses f32-inner"**, on `[CH18]`'s `φ = (c₁κ∞(A)+c₂)u_s` (first draft) | CH18's conditions are on `κ∞(A)` — original matrix, infinity norm — and its `κ∞(A) ≤ 1e8` relaxation is derived for **GMRES preconditioned by low-precision LU factors**, which this architecture does not have. The `φ` form is not in CH18; nearest match is eq. (3.8) of the **five**-precision successor | property 3 **RECORDED as a diagnostic**; the f32-inner license re-anchored on H1-b + the f64 outer residual, citing **inexact Krylov** (§7.8, §7.9) |
| **`κ₂(Ã⁻¹A_σ)` as the notation for H5 p2/p3** (first draft) | `Ã⁻¹A_σ` is a product of two SPD matrices, hence not symmetric, so `cond(·,2)` (singular values) ≠ `λ_max/λ_min` (eigenvalues) — MEASURED 1.4544e4 vs 1.0977e4 at n=64, a **1.33× gap**. The artefact computes the eigenvalue ratio; the notation denoted the other one | **`κ_eig`**, with both recorded (§7.8) |
| **"the crossing is strictly inside (960, 1 920], measured"** (first draft) | the apply noise is measured at n=**2 000** and the tolerance at n=**1 920**; no n above 960 carries both | **(960, 2 000]** (§7.11) |
| **`δ_precision` margin anchored on 1.533e-06** (first draft) | that value is console-only (`run-64-verify.ts:129`), written to no JSON | the archived `f32NoiseHi`/`f32NoiseLow` give the same 19–54× (§7.2) |
| **H2's "≤ 7.6 % error in a denominator"** (first draft) | measured at **θ=0.25**; production is θ=0.5, where the generalized spectrum reaches `c₁ = 0.2886`, i.e. ≤ 71 % | re-based on `lamMaxBct/lamMaxA ≤ **0.53 %**` — the norm ratio, not the spectral spread (§7.3) |
| **H4c conjunct 1's red side "6/6 random arms ≤ 0.86027"** (first draft) | those arms have `K_run` = 1,2,6,2,1,10 against K=20, which the gate's **own** INVALID clause forbids folding into a ratio | red side **NOT ESTABLISHED**; conjunct 2 carries the separation (§7.7) |
| **"42 iterations" as the production budget** (first draft) | conflates three counts differing ≥14×. `min cos∠_A ≥ 0.9` is MEASURED at **2 iterations (n=480) / 3 (n=960)** | production **3**; 32 is H3's gate currency; 42 is the backward-error budget (§3.5) |
| **Repulsor's `H1Metric.hpp` / `SetH1MassWeight` as the source of the 1e-3 mass default** (first draft) | **neither symbol occurs anywhere in this repo outside this spec** — no URL, no commit, no path | attribution withdrawn; mass weight **decided = 1** by measurement, since 1e-3 breaches a 1e-9 residual on 7 of 9 cases at n ≥ 10 000 by up to 182× (§11.2, §12 q1). **SUPERSEDED 2026-10-11 — issue #72, amendment A20: `massWeight` has units length⁻², so a bare constant makes the operator SCALE-DEPENDENT.** RULED **`massWeight = 30 / ext²`**, `ext` = max bounding-box side, i.e. the dimensionless `m̂ = massWeight·ext² = 30`. Measured over **8 shapes × 4–6 sizes × 19–27 mass values** (1089 cells). `ℓ = total length` REFUTED (per-shape optimum spans 96×; worst-shape regret 78×; an L-unit constant at the trefoil's optimum costs the arc 239×). `ℓ = mean edge length` KILLED (every `c` crosses the optimum, after which `κ_excl` grows `n^{+3.7…4.0}`; the +4 is DERIVED as `low ~ m̂⁻²` ∘ `m̂ ∝ n²`). Extent wins mechanistically: the two-branch slope is **4.000 / 4.15 / 3.97–4.00** (circle/trefoil/helix) in extent units across a 9× range of total length. `c = 30` from a PRE-REGISTERED minimax-absolute criterion at n=960, subset-confound controlled. **`c = 30` is 29.94 to within 0.2 % = today's `massWeight = 1` at the trefoil's extent**, so this is behaviour-preserving on the production fixture — luck, not vindication. **The 1e-3 evidence above is NOT overturned**: 1e-3 is far below the live band, and A20's measurements agree it is bad. **OPEN:** the minimax point drifts up with n (25@480, 30@960); `c` is re-checked at n ≥ 4 000 in task 9, NOT frozen on n ≤ 960 data |
| **"`PreparedSaddle` is the one member that breaks" / "task 8 is the only task touching production code"** (first draft) | `FrozenSaddleOperator.a` is a second required dense carrier, and `projectOntoConstraintSet` (`lineSearch.ts:259-267`) is a second consumer with no hierarchical route. The `A?` delta does not compile: TS18048 at `optimizer.ts:291`, TS2322 at `gradient.ts:127` | a `kind`-tagged **discriminated union** on both interfaces, task 8 across **five** files, plus task 8b for the producer (§5.1, §11.2) |
| **R1 as "the largest n where H1, H2, H4b are simultaneously green"** (#59) | with the re-specified gates this reports a falsely **HIGH** ceiling — H4b′ cannot fail, H2 is non-vacuous to 30 000, H1 has no magnitude bar | **three** ceilings, each with its binding gate's VALUE; `servedN_mixed` = **NOT ESTABLISHED** (§7.10) |

### 10.3 Numbers in the source reports that their own raw data contradicts

Carried here so they are not re-imported. Full lists in `digest-57.md` §11 and `digest-59.md` §0.

- matvec fit exponent: report **0.907**, JSON **1.0464**. `0.907` appears in no scratch file.
- H3's `α_G`: report takes iters=32 at n=120; `e5-growth.json` median is **33**. The report also
  prints three different values for one bar (1.4855 / 1.5086 / 1.5163), and **1.5163 reproduces from
  no JSON** — the nearest reconstruction, `√(1.1034 × 2.0833) = 1.5162`, would be a
  cross-contamination of the RS and *Jacobi* tables. Unresolved; do not quote it. All three are moot
  (§10.2: `α_G` withdrawn).
- **H5 property 2's n=960 and n=1 920 values exist in no artefact.** `e0_sanity.py` writes no JSON
  and its log is not saved; a grep for its values across all of `scratch/*.log` returns nothing. The
  four JSON-verifiable values match #59 exactly; the two largest do not exist outside report prose.
- **`lineSearchStepSet` has FIVE failure `reason` strings, not three** (#59 names two): source read
  at `lineSearch.ts:444-583` gives `zero_or_nonfinite_gradient_norm`, `not_a_descent_direction`,
  `tau_below_min`, `armijo_failed`, `projection_failed`, across three `accepted:false` exits, all
  setting `tau: 0`. Also undocumented anywhere: **`tauMin = 1e-12`**, so the maximum backtrack count
  is `log₂(1/1e-12) = 39.9`.
- `AdapterCertification.classification` is at `bench/gpu/adapterCert.ts:48` (digest-59 cites `:29`,
  which is inside the module docstring).
- **`GPU_SOLVE_MAX_N` does not exist in `src/**` or `bench/**`** — only in plan and spec docs. The
  N ≈ 5 793 cap that CLAUDE.md describes as "enforced" is currently **unenforced in code**.
- "3–16 ms": labelled per-iteration in one table, per-matvec everywhere else. **This spec says per
  matvec.**
- "0.10–0.51 s at n=30 000": not in #57 at all (#57 says 0.08–0.4 s); it is #64's `cost64.py`.
- largest single f32 buffer: report **25 MB**, actual **12.38 MiB** per array (25 is the sum of the
  two) — **and the reported per-array figure is in MB where the device limit is in MiB** (§3.4's
  unit correction, re-derived here from `stats.nearPairs`).
- near-field precompute at n=30 000 θ=0.5: **3 512.5 ms** in the same run as the 194 ms matvec
  (`-prenear.json` `build.precomputeNearMs`), vs 3 182.9 ms in #64's separate run. Neither report
  put either number in a budget.
- "4× optimistic": that is 12 columns vs 3; vs one scalar apply it is 12×.
- tangle f32 total: report ~97 MB, JSON 93 562 104 B = **89.23 MiB** (quoted in the source as 93.6 MB).
- the handoff's "6.55e-6 at n=480" is `beta2xMin` from the **retracted** criterion; the gate's own
  `betaCos90Min` at n=480 is **7.156e-5**.
- `structuredSaddleResidual` is at `linsolve.ts:860` (reports say `:855` and `:848`).
- **"GMRES iterations 21–26, flat n=64→960"** (#51, the handoff, every cost table) is E1's
  `rs/**right**/f64` row. The architecture mandates **left** preconditioning, which this spec
  measured for the first time (§5.3.1a): **25/27/26/32/32** to τ_true=1e-2, i.e. ~28 % more at n=960
  and mildly growing. And since H4b′ is only a valid classifier at **τ ≤ 1e-4** (§7.6), the
  *production* multiplier is the 1e-4 column: **29/35/38/41/42**. **The correct budget multiplier is
  42, not 26** — a 1.6× cost understatement in every prior projection.
- **The recurrence/true currency gap, in one measured pair, no fit:** at n=960 the recurrence count
  to τ=1e-8 is **17** while the true-residual count to τ=1e-4 is **42**. The recurrence target is
  nominally four orders *tighter* and needs **2.5× fewer** iterations (§7.4).
- **E1 and E5 report iteration counts against different residuals** (true vs recurrence) and
  disagree by ~10× at τ=1e-2. Neither report says which it used. §5.3.1.

### 10.4 Process note that earned its place

**Every agent in this phase's measurement round self-corrected after reading a sibling's output, not
after being challenged.** #57 retracted two claims; #59 retracted a criterion its own n=1 920 point
falsified 28×; #64 retracted the fit its own fifth point killed. And #64 records that had its sweep
stopped at n=2 000 — where two earlier attempts died — it would have recommended the wrong θ for a
falsified reason.

**Extend every sweep one point past where the fit looks clean.** That is the single most valuable
procedural lesson available to whoever implements this spec.

> **It has now failed twice inside this document, both times on H3.** #59 pinned `α_G` on a 64→120
> step that is not a doubling. This spec then pinned `τ_η = 1e-6` on four rungs that looked flat and
> collapsed at the fifth and sixth (19, 12). Writing the lesson down demonstrably does not prevent
> it — **the only thing that did was running the next two rungs.** Treat any flat series of four
> points as unverified until there are six, and treat a gate calibrated on a partial ladder as
> provisional no matter how clean it looks.

> **It has now failed a THIRD time, on the same gate. A27/A29's companion, RULED 2026-10-11 (task
> 7b).** The `τ_η = 1e-8` rung that replaced the collapsed 1e-6 was itself pinned on six rungs to
> n=1 920 and **declines 32 → 27 → 19 at n = 960 → 1 920 → 3 840** — the identical collapse pattern,
> two octaves out, exposed by extending exactly one point further than this spec did. The lesson's
> own correction box says *"any future change to `τ_η` must re-run the whole ladder to n=1 920"*; that
> was not far enough. **For a gate whose target is n = 30 000, "one point past clean" means one point
> past the LARGEST n you can afford, not one point past the previous ladder.**

### 10.5 A29 — the instrument finding that invalidated three earlier wall-clock ratios

**RULED 2026-10-11 (task 7b). The 6–9× one-sided `applyA` degradation recorded in this phase's ledger
as caused by *subagent load* is caused by **CO-RESIDENCY**, and it reproduces on an idle machine.**
MEASURED at n=960, nothing else running:

| resident operators | `applyA` | vs one resident | `Ã⁻¹` |
|---|---:|---:|---:|
| one `HierarchicalA` | 1.717 ms | — | ×0.84 |
| three co-resident | **10.212 ms** | **×5.95** | ×3.1 |
| in-suite | 38.172 ms | ×22.2 | — |

The working-set mechanism originally proposed is right; the **trigger** named was wrong, and the
difference matters operationally: *"run it when no other agent is running"* does not isolate the
instrument, because the contention is between operators inside one process. **"Isolated" has to mean
ONE OPERATOR PER PROCESS.** 7b's own first `cost.ts` figures were wrong by **8× in `r`** for this
reason before it re-ran them one-per-process, and the `n ≈ 223` crossover (A28) is the oldest casualty
of the same confound. **Any wall-clock ratio in this document whose two sides were measured with
different numbers of resident operators is void.** This is why A28 is settled on iteration-count
integers rather than on timings, and why §7.12 emits `clock: null` with wall-clock figures quarantined
in a block named `costWallClock_NONGATE`.

---

## 11. Implementation

No separate plan document (ruled 2026-10-09). The task sequence, its ordering constraints and its
verification are here.

### 11.1 Ordering constraints, each with a reason

1. **`L_{2−s}` and `Ã⁻¹` before any ITERATION COUNT is measured.** §5.2's trap: measure iterations
   against an unpreconditioned operator and the result looks like a regression in the wrong
   component. Restated from the first draft's "before the BCT", which the task numbering contradicts
   (task 3 is `bct.ts`, task 6 is `lowOrder.ts`/`precond.ts`); the numbering is correct and satisfies
   the real constraint, because the first iteration count is measured in task 7 (`gmres.ts`).
2. **The `endpoint4` bit-level pin (§5.4 item 4) before any accuracy number is quoted.** It is the
   only standing check that the near field agrees with `assembleAFlat`, and the 33.9 % figure it
   replaces has no saved artefact.
3. **The linearity check on the apply before anything built from basis-vector applies.** #64 gated
   its own harness this way and it is why its densification results are trustworthy: applying the
   operator to `N` unit vectors is only legitimate if the apply is linear, and nothing else would
   catch a data-dependent branch or a per-vector renormalisation.
4. **f32 and f64 as accumulator variants of one traversal from the start** (§5.6). Retrofitting this
   is how an f64 simulation of an f32 kernel ends up reporting zero error.

### 11.2 Tasks

| # | task | files | verification |
|---|---|---|---|
| ~~0a~~ | ~~Close the iteration-count gap~~ — **DONE while authoring this spec.** `scratch/e1b_left_f64.py` → `e1b-left-f64.json`, control reproduces E1 at all five n | `scratch/` only | ✅ §5.3.1a. Budget multiplier is **32** (1e-2) / **42** (1e-4), not 26 |
| 0b | **Calibration closers** — four edits totalling ~8 lines, each unblocking a gate that §7 ships WITHOUT a measured anchor. (i) `e1_gmres.py`: `first_below(et, t)` alongside the existing `first_below(tr, t)` ⇒ `iters(n)` at fixed `τ_η`, H3's companion band (§7.11 item 3); (ii) same file, `argmin` of η ⇒ H2's probe schedule (item 4); (iii) `e4_precond_integrity.py`: record `λ₁`, `λ₂`, `λ_max`, `rankTol` ⇒ makes H5 property 1 falsifiable instead of tolerance-dependent (item 8); (iv) `e0_sanity.py`: add a JSON emitter and re-run at n=64…1 920 (caches exist) ⇒ restores H5 property 2's missing support (item 2a) | `scratch/` only | H3's `G` resolvable (`iters ≥ 15` at the chosen `τ_η`); H2's probe `p` chosen from data rather than assumed; `λ₂/λ_max ≥ 10·rankTol`; `p` re-fitted on 6 real points |
| 0c | **H6 — the adversarial rhs** (§7.9). Build `b` from the sub-0.9 generalized modes at n=960 and 2 000, θ=0.5; measure `min cos∠_A` and `q`. **This decides §4.1's θ default**, so it runs before task 3 fixes a default. Eigenvectors already exist in `scratch/run64_badmodes.py` | `scratch/` only | `min cos∠_A ≥ 0.9` ⇒ default stays 0.5; `< 0.9` ⇒ default flips to 0.25 and §4.1/§4.4 are amended. Also records `q` on the same rhs, which is the **first** test of H4b′ outside the random-rhs regime |
| 1 | `laplacian.ts` — Sherman–Morrison cyclic tridiagonal `(−Δ)⁻¹`, **raw parameter default mass weight 1 (decided, §12 q1); the PRECONDITIONER's default is `30/ext²` — A20 / #72, see the ruling box below** | new + test | O(N) scaling. **Per-case bars from `out-tridiag.json`, mass=1**: `closed-cyclic` ≤5e-11 at n=30 000, ≤5e-10 at n=100 000; `open-tridiag` and `closed-circle-cyclic` on their own measured rows. **The flat `≤1e-9` bar is withdrawn** — see below |
| 2 | `bvh.ts` — R⁶ tangent-point BVH, median split, leaf=8. **PORT from `scratch/bvh.ts`** (see the prototype ruling below) | new + test | leaf count = power of two; mean leaf size in 4.88–7.81 at the five reference sizes |
| 3 | `bct.ts` — block lists, Eq. 28 admissibility, adjacency veto, **χ/θ as a runtime parameter whose minimum is the exact all-inadmissible path**, and that χ=0 path must **stream** (§7.1) | new + test | `κ_near` reproduces 108.20 at n=30 000 θ=0.5 and 93.3 at n=960; `adjacencyVetoed` = 0 at θ≤0.4; the `circle` 0.5→0.25 `κ_near` tie is *flagged*, not swallowed. Memory-assert that χ=0 allocates no `O(N²)` near array. **H1-a is NOT verified here** — `δ_compress` is a norm ratio of two *applies*, so it cannot be computed from block lists alone; it moves to task 4. The inverted-admissibility knock-out (`≥ θ·\|x_A−x_B\|`, one sign flip) is authored here and asserted there |
| 4 | `nearField.ts` + `apply.ts` — `endpoint4` near assembly, the 12-column traversal, **f32/f64 accumulator variants of one path** | new + test | **§5.4's bit-level pin**; the §11.1 linearity check; `A_bct·1 = 0` exactly for the apply; f32-vs-f64 apply ≈1.5e-6. **H1-b**: `δ_precision ≤ 10(κ_near+κ_far)u_f32` (bar 5.78e-5 at n=960, measured 1.5e-6), and **produce the missing knock-out** — truncate the near accumulator to f16 and confirm it exceeds the bar (model predicts 4.7e-2, 820×; §7.11 item 7) |
| 5 | `operator.ts` — `SobolevOperator`, `σ = tr(A_bct)/n`, `K1`/`K⁰1` | new + test | SPD: Cholesky of `A_bct,σ` succeeds; symmetry defect ≤1.6e-16; `tr(A_bct)/tr(A)` in 0.989–1.000 |
| 6 | `lowOrder.ts` + `precond.ts` — `L_{2−s}`, `Ã⁻¹`, constant-mode correction | new + test | **H5, all three properties** (§7.8): `dim null(L^{2−s}) == 1` (defect variant gives 2); **property 2's live bar is the per-fixture level `κ_excl ≤ 40` on the unnormalised trefoil at n = 64/120/240/480/960, SHIPPED. `\|p\| ≤ 0.05` is WITHDRAWN (A22) and so is saturation `R ≤ 0.75` (A32): on the shipped operator at θ=0.5 (§4.1's ruled target; `DEFAULT_THETA` is still 0.25) `κ_excl` is non-monotone, so `R` = 1.6509 on 240/480/960 (fails on a CORRECT operator) and −0.4354 on the registered 480/960/1920 (passes vacuously). `R` is logged, never asserted**; **`κ_excl(Ã⁻¹A_σ)·u_inner ≤ 1e-2`** — the precondition of the whole mixed-precision scheme. Both knock-outs trip; `λ₁`/`λ₂`/`λ_max`/`rankTol` reported so nullity is falsifiable; nullity ≥ 3 ⇒ INVALID, not RED |
| 7 | `gmres.ts` — left-preconditioned GMRES exposing `x̃`, `b`, `iters`, `maxit`, `tol`, the per-iteration recurrence residual **and an `η` probe hook on the χ=0 matvec with a recorded probe stride** (§7.3) | new + test | **H2 + H3** on one instrument. ~~The right-preconditioning knock-out must trip (`cos∠_A = 0.142` at n=480 while recurrence reads 1e-4)~~ **RE-ASSIGNED 2026-10-11 to task 10, with a new observable — see §5.3's correction box. This task CANNOT host it:** the knock-out needs a genuine f32 matvec, and `cos∠_A = 0.142` was never the right number (it is an f32 artefact of `e2_inexact.py:120-122`, not a side effect; in f64 right-preconditioning gives `cos∠_A = 1.000000` at every n). The replacement observable is **`cos∠_A ≤ 0` for `side: 'right'` under an f32 matvec at n=960** (MEASURED −0.136 median, −0.273 min) against left's 0.9991 — falsifiable, and it has a sign change rather than a magnitude to argue about. **What task 7 DOES own instead:** the `side` plumbing differential (reviewer B: each of 3 side-plumbing mutants fails 9–17 tests) and the guard refusing `side: 'right'` without `knockOut: true`. The recurrence-residual-as-η knock-out must trip RED at 34× the bar; **`η_exact` must never be bit-equal to `η_bct`** (both computed); `iters == maxit` and any-direction-censored ⇒ INVALID, and compare against `maxit`, **never** against `n`; **the gated η is `η_bct` (A21, §7.3) and `etaTarget` must read `η_bct`** |
| 8 | Seam: the `kind`-tagged `PreparedSaddle` and `FrozenSaddleOperator` unions, `HierarchicalSaddleSolver`, operator-shaped `structuredSaddleResidual`, **and the `projectOntoConstraintSet` hierarchical route** (§5.1) | `linsolve.ts`, `gradient.ts`, **`optimizer.ts`**, **`lineSearch.ts`**, **`src/gpu/saddle.ts`** | **the dense branch stays byte-identical** — **the existing goldens CANNOT measure this (A24, 2026-10-11); a bit-level dense-path golden was a PRE-TASK-8 deliverable and task 8 was BLOCKED until it existed** (box below). **DISCHARGED 2026-10-11 at `d3672c2`** — `test/sobolev/denseSaddleBitGolden.{test.ts,json}`, 44 tests, 26 728 numbers pinned with `Object.is`, captured with all five seam files untouched vs `main` (SHA-256 recorded in `provenance.seamFileSha256`). **Non-vacuity proven by mutation and independently reproduced by the controller:** a one-ULP perturbation of `z[0]` in `solveSaddleFrozen` turns the new gate **41/44 RED while all 54 pre-existing files stay 708/0 green**; same clean contrast for a one-ULP `x[0]` in `lineSearch.ts`'s reassemble projection branch, `Math.fround` on `structuredSaddleResidual`, and a one-ULP `a[0]` in `gradient.ts`'s Prepared arm. **Known sensitivity limit:** a 1-ULP change to an O(1e-3) projection correction is partly absorbed when added to an O(1) coordinate, so T1 (which pins the solve before absorption) is the sharp instrument, not T2/T3. **WARNING for this task, from the same mutation sweep: `test/gpu/dESeam.test.ts:91` (`expect(scaled.energy).not.toBe(baseline.energy)`) rides on ~3 ULP of rounding by its own comment's admission and flips in EITHER direction on any change to the arithmetic realisation (`@issue utof/repulsive-test2#80`). If it goes red during this task, check whether the two energies became bit-equal BEFORE concluding you caused a regression**; `kind: 'dense'` ⇒ bit-identical, and `optimizer.ts:291`'s dense throw message is pinned by `test/gpu/saddleSeam.test.ts:287-298`. `bunx tsc --noEmit` clean is a hard gate here: the first-draft `A?` delta produced TS18048 + TS2322  **SHIPPED 2026-10-11 at `a8569d8`** — 11 files, 650+/17−; `tsc` 0, suite **762/0/56**, golden fixture byte-identical to `d3672c2` (`cmp` empty), golden 44/44, all four verified by the controller. `lineSearch.ts` needed **comment only** (`+10/−0`): because `solveSaddleFrozen` dispatches on `kind` internally, `projectOntoConstraintSet`'s body is byte-for-byte unchanged, which is what §1.4 demands. Non-vacuity run not asserted: two `src` mutants each turn exactly one of the 10 new tests red (controller independently reproduced the residual-reordering mutant: 1 fail, the named test, 9 pass). **Guard shape corrected — see A25 in §5.1.** |
| 8b ✅ **SHIPPED (`4df665b`) — see A33, A35 AND A36. A36: this row's `cos∠_A ≥ 0.9` bar CANNOT FAIL for the reason it is written about (no θ trips it — 0.9987 is the worst any θ produces, because the adjacency certificate has no θ term) and it is BLIND to a saddle built for the wrong geometry (0.999839 vs 0.999840 while `g̃` is 4.04 relative away — issue #83). Two companion cells ship with it: mod-translation relErr of `g̃` ≤ 1e-2, and an EXACTNESS differential at θ=0 where `nFar`=0, ≤ 1e-10 at n=240 (measured 2.6e-12, the only cell that can see a plumbing defect)** | **`hierarchicalSaddle.ts` — the producer.** A `makeHierarchicalSaddle(vertices, …): HierarchicalPreparedSaddle` factory plus its wiring through `dispatch.ts` → `optimizer.ts` `opts.saddle` | new + `dispatch.ts` | **without this the phase ships an unreachable feature**: tasks 1–8 widen a public type that nothing in the repo ever writes, and the dense path stays bit-identical only because it is still the only path. End-to-end: one `sobolevStepSet` on the hierarchical route at n=960, `cos∠_A ≥ 0.9` against the dense f64 reference — **BAR AMENDED, A36: met at 0.999840, but it is NOT a compression gate and NOT a wrong-geometry gate; the two companion cells above are what discriminate, and the θ=0 exactness cell is now REQUIRED**  **CROSS-TASK CONTRACT FROM TASK 8 (`a8569d8`, binding): this task MUST pass `projectionMode: 'frozen'` when it wires the hierarchical saddle through `dispatch.ts`.** `optimizer.ts:308-327` (post-8b; was `:297-318`) REFUSES a hierarchical saddle without it, pre-arm, by design: `projectOntoConstraintSet`'s other branch calls `assembleAFlat` **per projection iterate** (O(N²) f64, **7.2 GB at N=30 000**), so the combination cannot run at target scale, and below that scale it would solve the GRADIENT hierarchically while silently reassembling a DENSE Ā for every projection iterate — a step that reads as hierarchical and is not (CLAUDE.md: gates name their impossible values). An auto-upgrade was rejected because `opts` and behaviour would then disagree. **If 8b needs to pass either mode, the throw may be downgraded ONLY with the dense-projection fallback surfaced in `stats`** — nothing downstream can currently distinguish the two. **Also carried to 8b: (R2) `factorMode` is silently inert on the hierarchical arm — add the guard or the one-line TSDoc; (R4) the seam tests cover one fixture (trefoil n=24, k=4) and one step, so 8b owns the k=0 and `edgeLengths` cells against a REAL operator.** |
| 9 ✅ **SHIPPED (`01079a1`) — see A37. It routes the MIXED `{barycenter, edgeLengths, pins}` set by COMPOSING the projector with an UNMODIFIED `prepareSchur` (the controller's hypothesis that this breaks 9b's reduction was REFUTED: `C_edge N = 0` bit-exactly, so `N ⊂ V` and the reduction transfers verbatim). §8.3's two gates were BOTH mis-specified — see A37 items 2 and 3.** | `nullspace.ts` — the `k=O(n)` constraint route for `edgeLengths` (§8.2) | new + test | projected-preconditioned operator verified SPD on the constrained subspace **before** any convergence number; `[G6]`'s conditioning bound checked separately. **BAR AMENDED, A37 item 6: `λ` is asserted at n ≤ 240 only and LOGGED at n=960. The edge multipliers are not solved for, they are RECOVERED post-solve through `(C_e C_eᵀ)⁻¹`, so their error carries that matrix's `cond ≈ 1.9e4` on top of `x`'s: over n = 240 → 960 `x` grows 1.9× while `λ` grows 16× on `grad/−φ` and 85× on `gen/−φ`, reaching 4.24e-10 — OVER a 1e-10 bar. At n=960 assert `x` plus `edgeConstraintResidual` instead. A36: this row's `k=0` / gauge cells also inherit A36 item 3 — on a translation-singular set the dense path is not a reference for the STEP either (the dense `g̃` is 82 % translation, so `lineSearch.ts` normalises it to a 1.77× shorter step and a 4.0 % energy gap, with BOTH arms `accepted: true`); a green `k=0` cell must not be read as the two arms taking the same step. A35: this row's bar inherits 9b's amendment — compare against the DD-refined truth, not the dense path, or it is unmeasurable at n=960 for the same reason. And it MUST include a `gen/−φ` cell (generic top with non-zero mean): 9b measured that a flipped multiplier sign is INVISIBLE on every `grad/0` and `zero/−φ` cell of every k=3 set, so a suite without `gen/−φ` ships a sign error.** |
| 9b ⬆️ **PROMOTED AHEAD OF 8b (A33)** | `schur.ts` — the `k=O(1)` half of §8.2's hybrid (barycentre, 3 rows; per-pin, 3 rows; tangent, 1 row) | new + test | the hybrid is a two-branch ruling and the first draft gave a task to only one branch. Dispatch on `k`: Schur for `k=O(1)`, nullspace for `edgeLengths`' `\|E\|` rows. ~~Verify each branch reproduces the dense path's `x` to 1e-10 at n=960 on its own constraint set~~ **BAR AMENDED, A35: `1e-10` against the DOUBLE-DOUBLE-REFINED TRUTH of the dense matrix of the same operator.** The literal bar is unmeetable BY ANY SOLVER at n=960 — dense LDLᵀ's own error against truth grows `~n^3.3` (4e-14…1.6e-13 at n=120 → **1.4e-10…6.4e-10 at n=960**), so it crosses 1e-10 between n=480 and n=960 and the "disagreement" equals the oracle's own error to 3 digits. Shipped Schur is **3.2e-12…9.1e-12** from truth, **25–200× closer than the oracle it was told to match**. Literal LDLᵀ agreement is asserted at **n ≤ 240** (5.6e-13…5.4e-12) and LOGGED at n=960. **SHIPPED** |
| **9c** ✅ **DONE — see A38. All three outputs answered: `c* = 35` at BOTH new rungs (grid-only, NOT converged); the `λ₂ = 1` crossing mapped (arc and circle are ALREADY below 1 in the production regime); and THIS ROW'S OWN `rs-identity` SENTENCE IS FALSE (the bar does NOT fail inside 40–45 — 148.5 and 118.0, first failure at m̂ ≈ 48.9). → #88 proposes `c`: 30 → 35; #89, #90 carry the two `λ₂` conflicts.** | **The `c` re-check at n ≥ 4 000 — A20's OPEN item, which had NO OWNER until 2026-10-11. A20 points at *"task 9's calibration"*; THIS ROW is what it means.** Row 9 is `nullspace.ts` and does no calibration, and **`9b` was already taken by `schur.ts`** — the item was assigned to a task that cannot do it. Scope: `κ_excl` over a `c` grid at n ≥ 4 000 on the four shapes present at every n (arc / circle / helix / trefoil — A20 proves these carry every binding constraint), **plus `λ₂(n, m̂)` jointly, which issue #78 needs and which is the same eigensolve** | `bench` / `scratch` only — measurement, no `src` change | **Three pre-registered outputs.** (1) The minimax-absolute `c` at n ≥ 4 000, against A20's five-rung trend **15 / 20 / 20 / 25 / 30** at n = 64…960 and its extrapolated **40–45** at N = 30 000 — report whether the trend continues, flattens, or breaks; do NOT bank the extrapolation either way. (2) **Where `λ₂ = 1` falls in `(n, m̂)`** — `κ_excl = λ_max/λ₂` silently measures the wrong mode once `λ₂ < 1` (#78; measured `λ₂` at n=64: 6.62 @ m̂=30 → 0.878 @ m̂=90, and `precond.test.ts`'s module comment records `λ₂ ≈ 1.3` at n=30 000 for the OLD m̂), so if `λ₂ < 1` anywhere in the production regime **every `κ_excl` figure there is not κ_excl** and the N=30 000 conditioning claims rest on a broken instrument. (3) The `rs-identity` knock-out bar `> 100`, re-derived at whatever `c` lands: MEASURED **273.3 at m̂=30, 118.0 at 45, 66.4 at 60** (#78), ~~so it FAILS inside A20's own projected band~~ **FALSE — A38 item 2: it does NOT. At the binding cell (n=64) the ratio is 148.5 at 40 and 118.0 at 45, both above 100; the first failure is at m̂ ≈ 48.9. The struck claim rested on the 66.4 at m̂=60, which is OUTSIDE the band it names. Replicated value at m̂=30 is 271.78, not 273.3 (0.55%, and #78's own rounded κs give 271.6).** — re-derive it as part of changing `c`, measuring first, never after a red. |
| 10 ⚠️ **PLUS TWO CARRIES FROM A38 item 4 (`A_σ`'s conditioning, measured): (d) `cond₂` of the unit-diagonal `A_σ` grows **n^+3.34** (arc: 3.27e8 → 3.85e10 over n = 960 → 4 000), so an f32 **factor-or-solve** of `A_σ` is out of reach ALREADY AT THE n=960 FLOOR (19.5× past `1/eps_f32` on the arc) while f64 stays safe to N ≈ 130 000 — and the pre-registered `chol_pivot_ratio` vs `sqrt(eps)` flag is WITHDRAWN as not scale-free; (e) **`cond₂` bounds SOLVING, not APPLYING** — this row's deliverable is an f32 **matvec**, whose error goes as `eps32 × growth`, so (d) must NOT be quoted against it. THREE CARRIES FROM A37 before it measures anything: (a) the PROJECTION SOLVE must stay f64 above n ≈ 9 000 (`κ·u_f32 = 1.12` at N=30 000 — an f32 projector stops being a projector), while the OPERATOR is the earlier-binding quantity at n ≈ 1 027; (b) the `rhsTop = 0` projection iterate is NOT free on the nullspace route — it costs a FULL GMRES because `d ≠ 0 ⇒ x_p ≠ 0 ⇒ b̂ ≠ 0`, so 9b's zero-apply shortcut does not exist there and a line search leaning on it will read as a regression; (c) issue #85 — `edgeLengthsBlock` allocates `C` densely, `3n²` f64 = 21.6 GB at N=30 000, so the `C: number[][]` seam is itself an O(N²) wall for every consumer except `prepareNullspace`** | Mixed precision: f32 inner, f64 outer (§6) | `gmres.ts`, `apply.ts` | **a genuine f32 hierarchical apply through the whole solve** (§9.2's top gap) — locate the wall, do not inherit it |
| — | ~~GPU wall-clock probe~~ | — | **NOT a task in this phase.** §7.11 item 6 wants one wall-clock matvec at n=30 000 to anchor H3's budget half, but §1.4 puts the GPU apply out of scope and §9.1 concedes the protocol needs "a kernel, when one exists" — so there is nothing to time. **Consequence, pre-registered: H3's budget-derived bar stays unanchored for this phase**, and its only live support is the CPU-f64 projection (§7.4), which §7.4 itself marks as integer-noise-sensitive. Do not quote `0.5 s ÷ 7.8 ms` as measured |
| 11 | Gate harness, emitting §7.12's schema to `bench/results/<date>-gpu-phase2h-gate-calibration.json`. **TypeScript, not Python** (`bench/gpu/hCalibration.ts`) for H1-a, H1-b, H2, H3, H4a, H4b′, H4c — those gates are defined against the TS instruments built in tasks 3–8 (`applyA`, `lineSearchStepSet`, the GMRES `η` hook) and a `.py` harness cannot reach them. H5 and H6 stay in `scratch/*.py`, where their dense-eigenvalue instruments already live | new | every gate in §7 reports its **VALUE, not only its verdict**; every INVALID predicate reachable in a test; **R1 publishes three ceilings** and `NOT ESTABLISHED` where no gate binds; `clock: null` |

### ⚠️ A33 — **task 8b is RE-SEQUENCED behind 9b, and its deliverable is re-scoped. Three defects in my own task table, all found by reading source.**

Ruled 2026-10-11, from the task-8b seam census
(`.superpowers/sdd/2026-10-09-phase2h-bct-gmres-design/task-8b-census.md`). I was about to dispatch 8b
from the row above; the row describes work that **cannot be completed in the order given.**

**(1) `prepare(C)` HAS NOTHING TO BUILD ON. 8b depends on 9b.** `HierarchicalPreparedSaddle.prepare(C)`
must return a solver for the **full saddle** `[[Ā, Cᵀ], [C, 0]]` — `gradient.ts:134-135` calls
`saddle.prepare(C).solve(flatten(dE))` and the projection calls the same solver with
`rhsBottom = −Φ(γ^q)` (`lineSearch.ts:276-277`). **VERIFIED: `src/core/sobolev/hierarchical/` contains
ZERO constraint machinery** — a grep for `schur|Schur|lambda|constraint` over all ten files returns
nothing outside comments, **none of its 38 exported functions takes a `C: number[][]`**, and
`schur.ts` does not exist. **The negative claim was then re-verified across spellings rather than left
on one grep** (CLAUDE.md `capability-claims-need-falsifying`), and it is stronger than first stated:
`grep -rn 'number\[\]\[\]' src/core/sobolev/hierarchical/*.ts` returns **zero hits for ANY parameter
name**, and no exported symbol in the directory — of 38 functions plus 9 consts, interfaces and types
included — has a name matching `constr|schur|saddle|lagrang|lambda|project|null`. Both checks are one
command each, so a future reader can falsify this rather than trust it. `gmres` solves `A_σ x = b` only. **So the producer cannot produce a working
solver before `schur.ts` exists**, and `schur.ts` is row 9b. **Row 9b is promoted ahead of 8b;** the
new order is **9b → 8b → 9 → 9c → 10 → 11.** Row 9 (`nullspace.ts`) stays after 8b: it serves the
`k = O(n)` route, which 8b's `k = O(1)` end-to-end cell does not need.

**(2) "Wiring through `dispatch.ts`" CANNOT REACH THE DEFAULT PATH, and the row implies it can.**
`src/worker/solverWorker.ts:51-52` **throws** `'solverWorker: `saddle` cannot cross the worker
boundary'`, `store.ts:362` defaults `solverDriver: 'worker'`, and a `HierarchicalPreparedSaddle` holds
closures so it is **not structured-cloneable by construction** (`linsolve.ts:431-450` says so for the
dense arm already). `dispatch.ts:282` merely forwards `saddle: args.saddle`; **no site in `src/` sets
it.** **RE-SCOPED: 8b delivers reachability on the MAIN-THREAD / inline driver only**, and must say so
in `stats` or TSDoc rather than leaving a UI path that throws. Making the hierarchical route work under
the worker driver is **a separate, unscheduled task** — it needs the saddle BUILT INSIDE the worker from
cloneable inputs, which is a different seam from the one tasks 1–8 widened. **Not smuggled into 8b.**

**(3) `edgeLengths` is assigned to TWO rows at once — my error, and 8b loses it.** Row 8b's carry R4
says *"8b owns the k=0 and `edgeLengths` cells against a REAL operator"*, while row 9 and §8.2 route
`edgeLengths`' `|E|` rows through `nullspace.ts`, and §8.1 measures that an iterative solve with
`k = O(n)` costs O(N²·iters). **R4 is CORRECTED: 8b owns the `k = 0` cell only.** The `edgeLengths`
cell moves to **row 9**, where the route it exercises actually lives.

**Four more facts the census established that 8b's brief must carry (each would otherwise be an
implementer's wrong turn):**
- **TOPOLOGY LIMIT — the hierarchical stack only builds on a natural-order closed ring.**
  `assembleCurveLaplacian` accepts only edges `(i, i+1)` plus the wrap `(n−1, 0)`, and
  `prepareLowOrder('vertex')` throws `CurveTopologyError` unless `|E| = |V|` with each vertex starting
  exactly one edge (`laplacian.ts:13-14` says dispatch must route everything else to dense). **The
  dense golden's fixtures (`crossing`, `junction-y`, `linked-rings`, `knot`) are not all rings**, so the
  end-to-end cell must use `trefoil` and the dispatch must have a documented dense fallback.
- **NO PINNED PRODUCTION `tol`, and no stop rule.** `gmres` requires `tol` (`gmres.ts:444`) and
  `gmres.ts:51-52` states the recurrence residual is not a convergence criterion. The spec gives
  iteration counts but never a production stop rule, and **nothing in `src/core/` thresholds the
  returned `residual`** (`linsolve.ts:410-413`). 8b must choose one and register it; `censored` /
  `stop` have no defined handling at the seam.
- **Non-convergence has no typed error.** The only typed one is `ExternalSolveError`, documented for
  external factorizations. A plain `Error` silently becomes `'singular_system'` in the gradient solve
  (`optimizer.ts:420-450`) and `ok:false` in projection (`lineSearch.ts:322`) — i.e. **GMRES failing to
  converge would be reported as a singular system.**
- **`alpha`/`beta`/`epsilon` are NOT forwarded by `dispatch.ts`;** the core falls back to `DEFAULTS`, so
  a saddle built outside `sobolevStepSet` must use the identical values or the operator and the step
  disagree silently (`src/gpu/driver.ts:147-155` documents the same trap for the GPU engine).
- **SCALE — added 2026-10-11, an omission in this box's first draft.** `buildBvh` **does not
  normalise**, and its header states tree quality depends on the caller scaling to extent ≈ 1
  (issue #69, where `κ_near` moves **1.70×** across the two scales in the corpus). **And the
  provenance is split, which is worse than a bare precondition and is what #69 actually names:**
  `bvh.ts:32-38` records that the spike **normalised every fixture to bbox max extent 1**
  (`scratch/curves.ts:13-17`) and that **every `out-bct-*.json` oracle — e.g. `κ_near = 108.20` —
  was measured that way**, while every shipped test builds from the **unnormalised** `trefoil` at
  extent **5.4716**. The mechanism is stated there too: the split score sums squared diameters of
  positions (curve scale) against unit tangents (scale 1), so at large scale positions swamp
  tangents, the chosen axis changes, and tree quality — hence `κ_near` and far-field accuracy —
  changes with it. **8b is the first code to take ARBITRARY production vertices**, whose extent is
  whatever the user's curve happens to be. 9b is unaffected (it builds
  from `trefoil` like its peers). **8b must either normalise before `buildBvh` and state the
  transform, or record in `stats` the extent it built at** — silently building a tree at extent
  5.47 and then at extent 0.01 on the next step is a quality change nothing downstream can see.
  This is the same class as #72's mass weight, which was closed by making the quantity
  scale-equivariant rather than by picking a better constant.

**Process note.** All three defects are in the SEAM, not the numbers — which is exactly what memory
`plan-the-seam-not-the-numbers` says to expect, and I still wrote a row whose dependency order was
wrong. The census cost ~30 min and was recovered from a transcript after the agent could not write its
own file; it still paid for itself three times over before a single line of 8b was written. **The rule
to keep: census the seam before dispatching the task, not after the implementer reports it blocked.**

### ⚠️ A35 — task 9b SHIPPED, and it refuted row 9b's bar, my Schur algebra, and two of its own knock-outs

Ruled 2026-10-11 from task 9b's report (`.superpowers/sdd/2026-10-09-phase2h-bct-gmres-design/task-9b-report.md`).
Delivered `src/core/sobolev/hierarchical/schur.ts` + `test/sobolev/hierarchical/schur.test.ts` (45 tests),
no existing file touched. **Four of its findings change the spec and three of them are corrections to things I
wrote.**

**(1) ROW 9b's BAR WAS UNMEETABLE, AND NOT BECAUSE OF THE SOLVER — ACCEPTED, AMENDED ABOVE.** The implementer
built an independent third instrument (double-double iterative refinement, converging in ONE step to
`|δz|/|z| = 4.1e-17…5.1e-17`) instead of accepting the oracle I handed it, and measured that **dense LDLᵀ — the
oracle I named — is 1.4e-10…6.4e-10 from truth at n=960 while the Schur solver is 3.2e-12…9.1e-12.** My bar
therefore measured the *oracle's* error: a Schur implementation 100× worse would have passed it identically.
Dense LU is different again (6.4e-11…2.4e-10), which is itself evidence the dense path is not a fixed point.
**This is the defect class this branch keeps producing — a gate that cannot discriminate what it claims to —
and this time it was in a bar I wrote in a task row.**

**(2) MY SCHUR ALGEBRA IN THE 9b BRIEF WAS WRONG, AND I HAD ALREADY FLAGGED THE REASON.** I wrote
`W = A⁻¹Cᵀ`, `S = C W`, `S λ = C y − d`, `x = y − W λ`. **`W` does not exist: `Ā` is SINGULAR**, nullity exactly
1 per coordinate block (`Ā·1 = 0`; `λ₁ = 3e-15` at n=48, `3e-13` at n=240), and `gmres` can only invert the
SHIFTED `A_σ`. Substituting `A_σ` for `A` in the textbook reduction **solves the shifted saddle and is wrong by
1.5e-3…1.7e0 on every pinning set.** The exact reduction needs a `(k+3)`-system carrying the solvability
condition `Gᵀλ = Nᵀb` with `G = C N` (`N` the per-block constant basis), derived in the module TSDoc and
**verified in dense arithmetic to 1e-14 BEFORE any GMRES was involved.** I named this interaction in the brief
— *"the σ shift and the barycentre rows both act on the constant mode, settle it in source, do not guess"* —
and then wrote the incomplete formula anyway. **Flagging a trap is not the same as not falling into it;** what
saved it was telling the implementer to let the oracle decide rather than my reasoning.

**(3) THE DENSE PATH IS NOT A REFERENCE FOR `x` WHERE THE SADDLE IS SINGULAR — AND THAT IS 8b's ONLY CELL.**
On `k = 0` and totalLength-only sets the dense saddle is translation-singular, and **dense LDLᵀ and dense LU
return DIFFERENT translations for the same system** (`mean(x)` = 6.0e-2 vs 8.0e-2 at n=48, k=0). `schur.ts`
returns the `A_σ` gauge (`Nᵀx = Nᵀb/σ`). **Consequence for A33, which made `k = 0` the one constraint cell 8b
owns end-to-end: its dense reference is meaningful only MOD TRANSLATION.** `cos∠_A` is blind to a translation,
so 8b's stated gate survives untouched — but any literal `x` comparison 8b adds **must remove the per-block
mean** or it will fail against a correct solver.

**(4) PRODUCTION `tol` RULED `1e-12`, from the measured knee** (A33 recorded that none existed). Error vs truth
against `tol`: 1e-10 → 2.2e-10…5.7e-10 (**over the bar**), 1e-11 → 2.1e-11…7.3e-11, **1e-12 → 3.7e-12…7.6e-12
(13× under the bar)**, 1e-13 → 2.2e-12…3.2e-12 (**the floor**), 1e-14 identical to 1e-13. Nothing is bought
below 1e-13, and 1e-12 costs 46 iterations against 51. `DEFAULT_ETA_ACCEPT = 1e-10` is a **certificate, not the
accuracy target** (40× above the measured η at the default tol). **Calibration worth keeping: at the stopping
iterate `η/tol = 1.0…3.7 across n = 48…960`, i.e. the recurrence residual is within 4× of the true BACKWARD
error at the stop.** That does NOT contradict `gmres.ts`'s 1e8…1e12 under-reporting figure — that one is of the
relative residual `‖r‖/‖b‖`, a different quantity, and conflating the two would be a provenance error of the
kind A17 catalogued.

**(5) TWO KNOCK-OUTS ARE BLIND TO THE BAR. Both are recorded rather than papered over, and one of them is a
near-miss worth naming.**
- **`lambda-sign` is INVISIBLE on the `grad/0` and `zero/−φ` cells of every k=3 set** (bary, pin) — **NARROWED by A37 item 12: the blindness is a property of `k=3` SETS, not of gradient right-hand sides. It DISAPPEARS at `kOther=6` (`|λ|=36.6`, caught at 2.04) because `G` stops being square. Do not restate this as "sign errors are invisible on `grad` rhs."** — because
  `Gᵀλ = Nᵀb` with `G` square and invertible forces `λ ≈ 0` for a mean-free top. **A suite of `grad/0` +
  `zero/−φ` cells alone would have shipped a flipped multiplier sign.** The `gen/−φ` cell catches it and is
  now a REQUIRED cell, propagated to row 9 above.
- **`no-symmetrise` is NOT caught at the shipped `tol` on ANY cell.** `S`'s asymmetry is just the GMRES error
  in `W` (`sDefect` 5e-14…9e-13), so factoring the lower triangle instead of the average moves `x` by < 1e-10.
  **Ruling: the symmetrisation stays, and it is explicitly NOT protected by the 1e-10 bar** — it is hygiene
  with a bounded effect. The live gate is the loose-`tol` differential the implementer added (at `tol = 1e-6`
  on barypin the symmetrised solver is closer in all three cells, asserted as *"unsymmetrised is > 1.1× worse"*).
  **This is exactly the case CLAUDE.md's anchoring rule exists for: a line a future agent can delete with every
  test still green.** Its comment must say so.
- **CORPUS TRAP, and it would have made the whole table read green:** the trefoil is Z₃-symmetric, so a
  barycentre block AT its own barycentre is a DEGENERATE cell — the naive formula agrees to 7e-15 at n=48 with
  `λ ≈ 0`. **Constraint targets must be offset so `Φ ≠ 0` in every test.** Carried to rows 9, 9c and 8b.

**(6) SEAM DECISION, ASSIGNED TO 8b, WITH ITS RISK NAMED.** `SchurConvergenceError` / `SchurRouteError` extend
`ExternalSolveError` because it is the only class the seam lets through uncaught — which does fix A33's
*"GMRES non-convergence is reported as `'singular_system'`"*, the lesser evil, and the implementer flagged it as
mine to rule rather than deciding it silently. **But `ExternalSolveError` is documented as the GPU path's
"could not certify" signal and `driver.ts` treats it as "re-run on the CPU path" — which is the WRONG recovery
for a CPU GMRES that failed to converge, since re-running it will fail the same way.** RULED: **8b introduces a
neutral base `SolveNotCertifiedError` in `linsolve.ts` with `ExternalSolveError` extending it, and widens the
two catches (`optimizer.ts:429`, `lineSearch.ts:321`).** Added to 8b's carry set. Until then the mis-recovery
is unreachable (nothing builds a hierarchical saddle yet), which is the only reason this is not a blocker.

**(7) THE ACCURACY CEILING ON ANY CROSS-PATH COMPARISON AT n=960 IS ≈2e-10, AND THAT BOUNDS EVERY LATER GATE.**
Two evaluations of the **same** operator that differ by only **2e-15 relative** — the θ=0 streaming source versus
`assembleAFlat` — have **true solutions 1.8e-10…2.2e-10 apart at n=960**, an amplification of ~1e5 consistent
with `κ·u_f64`. So the independent-operator comparison is asserted only at **n=240**. **Generalise this before
writing any further cross-path bar: at n=960 no comparison between two independently-evaluated operators can
assert better than ~2e-10, no matter how exact either solver is.** A17's provenance discipline and this are the
same lesson at two scales.

**(8) THE COMPRESSION ERROR DOMINATES THE SOLVER ERROR BY ~8 ORDERS, WHICH REFRAMES 8b's GATE.** θ=0.25
compression alone moves `x` by **1.6e-3…4.5e-3** against the true `Ā` solve, against the Schur solver's
3.2e-12…9.1e-12. **So 8b's `cos∠_A ≥ 0.9` gate measures the OPERATOR's compression, not the solve** — which is
the right thing for it to measure, but the row must say so, or a future reader will take a passing `cos∠_A` as
evidence about the linear solver. It is evidence about θ.

**(9) σ-INDEPENDENCE IS THE FALSIFIER FOR THE CORRECTED REDUCTION, and it is clean.** With σ scaled **×4 and
×0.25**, the shipped `x` moves by **≤ 9.0e-15** (σ cancels exactly, as the derivation requires) while the naive
route moves by **1.4e-4…1.8e-3**. That is a 1e11 separation between a correct and an incorrect reduction on a
quantity neither the 1e-10 bar nor the knock-out table looks at — **the strongest single check in this task, and
it should be the first thing row 9 reproduces for the nullspace route.** `cond(S) ≤ 59.6` across every cell.

**Also confirmed, not assumed:** the `prepare`-once/`solve`-many cost split is structurally pinned by a
counting preconditioner, not inferred — prepare = 316 applies = Σ(iters+1) over columns, one `solve` = 41, a
SECOND solve with a different rhs = 48 (**one GMRES, not k**), and a `rhsTop = 0` projection solve = **0
applies**, short-circuited before `gmres`. `SCHUR_MAX_K = 16`; `edgeLengthsBlock` is refused by name with
`SchurRouteError`.

### ⚠️ A36 — task 8b SHIPPED, and ROW 8b's GATE CANNOT FAIL FOR THE REASON IT IS WRITTEN ABOUT

Ruled 2026-10-11 from task 8b's report
(`.superpowers/sdd/2026-10-09-phase2h-bct-gmres-design/task-8b-report.md`), commit `4df665b`.
Delivered `src/core/sobolev/hierarchical/hierarchicalSaddle.ts` (365 lines) +
`test/sobolev/hierarchicalSaddleReal.test.ts` (24 tests), the `SolveNotCertifiedError` seam fix, and
the R2 TSDoc. Suite **832 pass / 0 fail / 58 files**. **The phase is now reachable:** tasks 1–9b
widened a public seam that nothing in the repo wrote.

**(1) `cos∠_Ā ≥ 0.9` IS NOT A COMPRESSION GATE — NO θ CAN TRIP IT. The bar is kept, and two cells
that CAN discriminate are added.** Measured at n=240, bary, one shared dense reference:

| θ | 0.25 | 0.5 | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|---|
| `cos∠_Ā` | 0.999963 | 0.999307 | 0.998749 | 0.998749 | 0.998749 | 0.998749 | 0.998749 | 0.998749 |
| `nFar / nNear` | 408/418 | 274/162 | 274/126 | 274/126 | 274/126 | 274/126 | 274/126 | 274/126 |

θ ≥ 1 is **the same tree**, so the worst `cos∠_Ā` any θ can produce on a gradient rhs is **0.9987 —
11 % above the bar.** **Mechanism, verified in source rather than inferred from the table:** the
adjacency certificate (`bct.ts:24-38`, `d > r_x^A + r_x^B + (maxEll_A + maxEll_B)/2`) contains **no θ
term**, so blocks that pass Eq. 28 but fail the certificate are split further regardless of θ and the
near field has a θ-independent floor. **This is the same defect as row 9b's bar (A35 item 1) in a
different costume: a bar that cannot discriminate what it claims to.** Row 8b's gate measures θ's
compression (A35 item 8 said so) and θ's compression cannot reach it.

**Two companion cells, both shipped and both asserted:**
- **mod-translation relative error of `g̃`, bar `1e-2`.** Clean 2.226e-3 at n=960 / 2.859e-3 at n=240
  (4.5× margin); θ=0.5 gives 2.86e-2 so the bar is loose enough for a legitimate θ. **It trips where
  the cos bar does not:** `alpha=2` → cos **0.790**, `beta=8` → **0.878**, `maxit=1` → **0.778**, and
  `alpha=4` / `beta=4` / `epsilon=1e-2` / `leafSize=1` trip modTrans while passing cos. Every one of
  those is reachable through **shipped options with no knock-out flag** — they are A33's
  *"α/β/ε are NOT forwarded by `dispatch.ts`"* trap, so the falsifiers are the real failure mode.
- **An EXACTNESS differential at `theta: 0` (`nFar = 0`), bar `1e-10` at n=240 — ADD THIS TO ROW 8b.**
  Measured **2.6108e-12**, i.e. **1e8 tighter than the compressed cell, and the only cell in the file
  that can see a plumbing defect** (both other cells are dominated by θ's 1.6e-3…4.5e-3, so neither
  can tell a correct composition from one defective by up to ~1e-3). Two independent routes to
  `nFar = 0` — `theta: 0` and `leafSize: 64` — agree to **4 digits** (2.610814e-12 vs 2.610823e-12).
  At n=960 the same cell reads **2.533e-10, which independently reproduces A35 item 7's ≈2e-10
  cross-path floor from a direction A35 did not use.** Cost: 2 s. **Generalise: every compressed-path
  gate on this branch should be paired with its `nFar = 0` exact-limit cell, which is nearly free
  because the traversal code is shared (§7.1).** **QUALIFIED by A37 item 14: `fixture(48)` at the
  default `leafSize = 8` ALREADY has `nFar = 0`, so this pairing SILENTLY MEASURES NOTHING at n = 48 —
  the compressed and exact cells are the same cell. Take it at n ≥ 240 and assert `nFar > 0` at the
  size you take it.**

**(2) `cos∠_Ā` IS BLIND TO A SADDLE BUILT FOR THE WRONG GEOMETRY — issue #83.** A saddle built from
`2 × vertices` and used on the originals reads **0.999839 against the clean 0.999840 — one part in
4e-6 — while its `g̃` is 4.036 RELATIVE away.** It passes `op.n`, which is the only shape the seam
checks. The *"`A` MUST be the Sobolev matrix of the same vertices the step receives"* rule
(`linsolve.ts:462`, and `optimizer.ts:125` for `dE`) is **TSDoc prose with no enforcement anywhere** —
confirmed by grep: every actual check on the seam path is a length check. This affects the **dense/GPU
Phase-2a seam identically**, not just 2H. `cos∠_A` is a direction in a seminorm, scale-invariant in
each argument; it cannot supply the missing enforcement and must not be read as doing so.

**(3) AT `k = 0` THE GAUGE CHANGES THE REALISED STEP, NOT JUST `x` — A35 ITEM 3 WAS INCOMPLETE.**
A35 ruled that the dense path is not a reference for `x` where the saddle is translation-singular, and
that `cos∠_A` is blind to a translation so *"8b's stated gate survives untouched"*. Both hold. **What
it did not say is that the STEP differs too.** n=960, k=0:

| quantity | dense | hierarchical |
|---|---|---|
| `cos∠_Ā(g̃_h, g̃_d)` | **0.999839539** — identical to the k=3 bary cell to 9 digits | — |
| per-block means of `g̃` | −1.019e-1, 1.750e-1, −5.773e-2 | −1.3e-17, 3.7e-17, −2.1e-17 (`A_σ` gauge, exactly mean-free) |
| LITERAL relErr of `g̃` | **8.203e-1** | — |
| MOD-TRANSLATION relErr | **2.2259e-3** — the k=3 cell's value | — |
| `l2CurveNorm(g̃)` | 1.3716 | 0.7756 → **ratio 1.7686** |
| energy after the step | 15.5321 | 14.9168 → **gap 0.6153 (4.0 %)** |
| τ / projectionIterations / accepted | 1 / 0 / **true** | 1 / 0 / **true** |

**Mechanism, traced to source:** `lineSearch.ts:499-517` normalises the direction by
`l2CurveNorm(g̃)` before stepping. The dense k=0 `g̃` is **82 % translation**, which inflates that norm,
so the dense arm takes a **1.77× SHORTER** step along an identical direction. Verified by
decomposition at n=240: `g̃_h − g̃_d` is a pure translation to **1.24e-3 relative**, the energy is
invariant under translating the result (14.991871589080532 both ways, 14 digits), and the mean-free
parts agree to 2.859e-3 — exactly the θ error of the k=3 cell. **Neither arm is wrong — the
translation really is free in `x`.** **RULED, and it is a reading rule, not a bug:** on a
translation-singular set the dense path is not a reference for the **step** either, only for `x` mod
translation; **a green `k = 0` cell must never be read as "the two arms produce the same step".**
Both steps report `accepted: true` with finite stats, so nothing downstream flags the divergence.
**Carried to rows 9 and 9c**, whose `k = 0` and gauge cells inherit it.

**(4) A35 ITEM 6's STATED HARM DOES NOT EXIST IN CODE. The ruling stands; my justification was
false.** I wrote that `ExternalSolveError` is *"documented as the GPU path's 'could not certify' signal
and `driver.ts` treats it as re-run on the CPU path — the WRONG recovery for a CPU GMRES"*. Measured:
**`src/gpu/driver.ts` has no `catch` at all** (195 lines), the only `ExternalSolveError` producer is
`src/gpu/saddle.ts:446` whose only callers are `test/gpu/`, and no `src/` site catches either class.
*"Re-run on the CPU path"* is `[DESIGN §4.5]` trigger 3 — **spec text, not shipped code.** The
implementer made the change anyway on the narrower reason it could check and wrote THAT into the
TSDoc: `SchurConvergenceError extends ExternalSolveError` would make the shipped type contradict two
shipped TSDoc contracts (`linsolve.ts`'s *"an external solve"* and `optimizer.ts:207`) and destroy
`instanceof ExternalSolveError` as the discriminant a future GPU fallback would be written against.
So the harm is **latent and documentary**, and the TSDoc says so. **The lesson is A17's and A35 item
1's again: I cited a spec's description of behaviour as though it were the behaviour.** A spec claim
about code is as checkable as a number (memory `capability-claims-need-falsifying`), and this one cost
one grep.

**(5) A33's "not structured-cloneable by construction" claim is now MEASURED, and it held.** I
asserted it from the presence of closures. `structuredClone({saddle})` on a real saddle throws
`DataCloneError: The object can not be cloned.` — the first actual check. Main-thread/inline-only
reachability is pinned **two** ways: that, plus a cloneable `{kind:'hierarchical', op:{n}}` stand-in
posted to the real worker returning the documented refusal.

**(6) Row 8b's remaining carries, as DELIVERED.** `projectionMode: 'frozen'` asserted to fire for
`'reassemble'` AND `undefined`, with the clean config asserted to pass so neither side reads
vacuously; extent **recorded, not normalised** (#69) and cross-checked against an **independent** bbox
loop rather than an import of `curveExtent` (memory `common-mode-blindness`); `tol` pinned
`=== 1e-12`; targets offset so `Φ = −1.4413 ≠ 0`; topology refused up front by
`HierarchicalSaddleUnsupportedError` naming the dense path, asserted for open chain, junction,
duplicated edge and `n < 3`, with the ring asserted to pass. `normA2` is cached once per factory and
asserted **bit-identical** with `Object.is` (not a tolerance) — the one numerical liberty taken, and
it fixes the waste 9b's report flagged.

**(7) THREE ISSUES FILED, none of them inline-fixable.** **#82** — `DEFAULTS` lives in `optimizer.ts`,
so seven modules import the step driver for four numbers and `hierarchicalSaddle.ts` is the first leaf
numerics module to do it; the cycle goes live if the factory is ever wired in from `gradient.ts`.
**#83** — (2) above. **#84** — `file:line` citation rot: **8 of 97** citations in `src/` are already
detectably wrong (one overshoots `bct.ts` by 203 lines), 8b had to hand-update **13** anchors its own
edits moved, and **2 were already stale before it started**. That last one matters structurally: this
repo's anchoring rule tells a future agent to *"check its reference before changing it"*, and the rule
degrades in proportion to how many references are wrong.

**Process note — two failures worth keeping.** (a) **The implementer's full-suite run was killed when
its shell was torn down at idle**, ~25 min in, and **a dead run is indistinguishable from a slow one
in the log**: the log simply stops with no totals while `ps -C bun` is already empty. The brief's
foreground-only instruction did not prevent it, because the agent went idle *while the foreground job
ran*. The suite was re-run by the controller under `nohup`. **Rule: the orchestrator runs the gate
suite, or the worker's run is detached from its shell — never a bare foreground job the agent then
idles on.** (b) **A33's body was split by A35 in this file** — its item (3) and five carried facts sat
*below* all of A35, where they read as A35's, and A35 has its own item (3) about a different thing.
Fixed here by moving A33's tail back under its header. **This is the sixth-instance defect again
(A32/A22): amendment boxes are append-only prose and an insertion silently orphans whatever follows.**

### ⚠️ A37 — task 9 SHIPPED, my STAGE-0a hypothesis was REFUTED, and §8.3's OWN TWO GATES are both mis-specified

Ruled 2026-10-11 from task 9's report
(`.superpowers/sdd/2026-10-09-phase2h-bct-gmres-design/task-9-report.md`). Delivered
`src/core/sobolev/hierarchical/nullspace.ts` + `test/sobolev/hierarchical/nullspace.test.ts`
(35 tests), **no pre-existing file touched**, so nothing existing can regress. This box is unusually
long because §8 is the only section of this spec whose two normative gates were *both* wrong, and
both were found by measurement before shipping.

**(1) THE MIXED SET ROUTES BY COMPOSITION. MY HYPOTHESIS WAS REFUTED, AND IT FAILED ON ONE LINE OF
ARITHMETIC I HAD ALREADY READ.** I wrote in 9's brief that composing the nullspace projection with
`prepareSchur` would break 9b's exact reduction, because `P Ā P` has a strictly larger nullspace so
`A_σ⁻¹ N = N/σ` and `G = C N` would stop holding. **Every clause of that premise is true and the
conclusion does not follow.** The enlargement is *exactly* `V^⊥ = range(I−P)`, orthogonal to the
subspace the iteration lives in and never entered. The fact that settles it:

> **`C_edge N = 0` EXACTLY — bit-exact, not 1e-16.** Row `r` writes `+T_r[c]` at `(c,i1)` and
> `−T_r[c]` at `(c,i2)` (`constraintSet.ts:206-250`), so each coordinate block's sum is
> `T_r[c] + (−T_r[c])`, which is 0 in IEEE754 for any finite `T`. **Independently re-verified by the
> controller: 0 of 144 entries at n=48 and 0 of 720 at n=240 are non-zero.**

So `N ⊂ V`, `P N = N` (1.87e-15), `Nᵀ P = Nᵀ`, `nullDim(P Ā P)` is `n + 3` — `n` zeros from `V^⊥`
plus **exactly 3** inside `V`, basis still `N` — and `P A_σ P N = σ N` to 3.55e-14 (1.7e-16
relative). `Ĉ_o N = C_o P N = C_o N = G` and `Nᵀ b̂ = Nᵀ b` exactly, so the solvability condition
`Gᵀλ = Nᵀb` is *the same equation* it is in 9b. **9b's `prepareSchur` is reused UNMODIFIED.**
**RULED: option (B) ships, and row 9 owns the MIXED set `{barycenter, edgeLengths, pins}`, not only
the pure one.** Option (A) (nullspace of the whole stacked `C`) is rejected on three measured
grounds: `cond(CCᵀ)` 4.7–9× worse, needs a bordered non-`TriSystem` solver, and — decisively — it has
no gauge branch so it **fails outright on the pure edge set** (1.26e3 / 5.53e2). Option (C)
(defer the mixed set) is not needed.
**The lesson is not "I was wrong about numerics".** I had quoted that exact row structure earlier in
the same session, deriving the cyclic-tridiagonal Gram from it, and never asked the adjacent
question about the constant modes. **Same source line, two questions, one asked.** Non-vacuity is
proven, not assumed: dropping *only* the constant-mode correction moves `x` by **5.6e-2 … 3.8e+0**,
12–14 orders above clean, so the piece my hypothesis said could not survive projection is exactly
the piece whose removal is catastrophic.

**(2) §8.3 ITEM 2's GATE, READ LITERALLY, WOULD REJECT THE CORRECT ORDER — AND IS BLIND TO THE ONE
ORDER ITS OWN WARNING IS ABOUT.** §8.3 item 2 requires that *"the projected-preconditioned operator
must be verified SPD on the constrained subspace before any convergence number is quoted"*. Two
defects, both measured at n = 48/96/120:

- **A product of two SPD matrices is not symmetric.** The Euclidean symmetric defect of `M⁻¹A` on `V`
  is **1.40e-1 … 2.05e-1 for the SHIPPED order** and must be. Taken literally the gate disqualifies
  every candidate.
- **The restriction-based test cannot see the failure mode §8.3 names.** `Zᵀ P = Zᵀ`, so
  `Zᵀ (M P A P) Z ≡ Zᵀ (P M P A P) Z` **identically**: the correct order O1 (`P M P A P`) and the
  incorrect O2 (`M P A P`, preconditioner not projected) have the same restriction, the same
  symDefect and the same spectrum **to every digit.** The only quantity that separates them is
  **leakage out of `V`** — `2.59e-13` vs `4.10e-1`, **twelve orders**. A reviewer who checked
  symmetry and positivity on `V`, exactly as §8.3 says, **would pass O2** — and O2 is precisely the
  §5.3 failure mode the spec is warning about: the iterate drifts out of the constraint subspace
  while the restricted operator looks perfect.

**§8.3 ITEM 2 IS AMENDED. The three quantities that actually discriminate, all three required:**
(i) **leakage** `max_{z∈Z} ‖(I−P) K z‖ / ‖K z‖`; (ii) symmetry **and** positivity of
`Zᵀ P A_σ P Z` — *the operator being inverted, NOT the preconditioned product*; (iii)
self-adjointness in the preconditioner's own inner product,
`symDefect((Zᵀ P M P Z)⁻¹ · Zᵀ K Z)`. Shipped order **O1 `P M P A P`**: leakage 2.59e-13,
M-selfAdjDefect 2.60e-13, and on `V` by similarity transform `λ_min = 1.0000 exactly`
(the constant mode the `+11ᵀ/(nσ)` correction pins there — `N ⊂ V` showing up spectrally),
λ_max 59.8/69.5/72.2, `nullDim = 0`, **all eigenvalues positive**. Disqualified with their numbers,
recorded so a future agent cannot "simplify" the order back: **O2** leakage 4.10e-1; **O4**
(`P A P M P`, right-preconditioned) M-selfAdjDefect 1.59e-1, *and* already gated behind `knockOut`
by `gmres.ts` (§5.3); **O5** (`P M A P`) M-selfAdjDefect 1.81e-3.
**O3 (`P M P A`) is not a distinct order** — `P(I−P) = 0` makes it identical to O1 to 13 digits. The
implementation keeps both projections anyway, because `P² = P` holds only to 1.1e-15 and the
re-projection is what bounds drift out of `V` at a cost of one `solveLaplacian` against a GMRES
iteration's two hierarchical applies. **That is an anchored line: removing the inner projection as
dead code is correct in exact arithmetic and wrong in floating point.**

**This is the NINTH instance of the non-discriminating-gate defect on this branch, and the first
found in this spec's own normative text rather than in a task row.** A22's saturation statistic,
A35's row-9b bar, A36's row-8b bar, and now §8.3 item 2.

**(3) §8.3 ITEM 3's FRAMING IS INVERTED: THE PROJECTION IS NOT THE CONDITIONING RISK, THE OPERATOR
IS.** Item 3 warns that *"a nullspace route does not inherit an exemption just because it avoids the
Schur complement"* and sends the implementation to check `[G6]`'s `κ·u_f32 > 0.1` bound. Correct as
far as it goes, and **the quantity it implicitly points at is the wrong one.** Measured:
`cond(P A_σ P|V) / cond(A_σ) = 0.960 ± 0.003` — the projection *improves* conditioning by 4 %, which
is Cauchy interlacing, not luck. Crossovers for `κ·u_f32 = 0.1`: the **operator** at **n ≈ 1 027**,
the **projection solve** at **n ≈ 8 956** — the operator binds **8.8× earlier**. So the route needs
no conditioning exemption; it needs §6's mixed-precision ruling, which is already the spec's answer.
**And this independently reproduces §6's f32 wall at n ≈ 1 228–1 773 from a new direction** — a third
arrival at the same number.

**(4) MY STAGE-0b MEASUREMENTS CONFIRMED, EXTENDED TO n = 7 680, WITH TWO CORRECTIONS TO MY OWN
NUMBERS.** `cond(C_edge C_edgeᵀ)` is **exactly `n²`** over a **160× range** — 4.771e+1 at n=48 to
**1.2340e+6 at n=7 680**, successive doubling ratios 4.03 / (×2.5)²=6.27 / 4.00 / 4.00 / 4.00 / 4.00,
with `cond/n²` stable at **0.0209**. Mechanism: refining a fixed smooth closed curve aligns
consecutive tangents, so the matrix tends to the ring Laplacian (λ_max → 4, λ_min ~ 1/n²).
**Extrapolation confirmed to the digit: κ(30 000) = 1.883e+7, `κ·u_f32 = 1.12`, breaching `[G6]`'s
0.1 bound by 11.2×, crossover n = 8 956.** In f64 the same κ gives 2.1e-9, 4.8e7× under the bound.
Corrections to what I put in the brief:
- **The diagonal is NOT exactly 2.** It is `2|T_r|²` with `|T_r|² = 1 + O(u)`, max deviation **4 ulps
  (8.9e-16)**. My "exactly 2.000000" was a `toFixed(6)` artefact of my own probe — **a gate written
  `=== 2` would fail.** (That probe had already cost me one bug: `spectrum()` MUTATES its input, so
  my first table's `diag`/`off` columns were post-eigensolve garbage. Two defects in one 30-line
  probe, one caught by a contradiction between my own numbers and one caught by the implementer.)
- **The off-band entries are *identically* 0, not "below 1e-15"** — the dot product of two
  non-adjacent edge rows has no overlapping terms at all, so there is nothing to round.
- **A hazard I failed to name:** `solveLaplacian`'s `thomas` is **unpivoted**, justified by *strict*
  diagonal dominance, while `C_edge C_edgeᵀ` is only *weakly* dominant as the tangents align
  (`2|T_r|²` vs `→ 2`). Measured safe rather than assumed: min Thomas pivot **≥ 1.001 at every n**
  (NR's `gamma = −b[0]` doubles `b[0]`), worst `triResidual` 6.72e-11 at n = 7 680 — fine in f64,
  which is item (3)'s point. **`laplacian.ts`'s generic `TriSystem` + `solveLaplacian` are reused; no
  new solver was written.**

**(5) THE ROUTE AND THE DENSE PATH CROSS, AND A35 ITEM 1 IS NOW PINNED BY AN EXECUTABLE CHECK.**
Shipped route vs DD-refined truth, `baryedge`, θ=0.25, `tol=1e-12`: n=48 **7.53e-13** (dense LDLᵀ
1.95e-15, i.e. dense is **400× better**), n=240 1.97e-12 (LDLᵀ 4.28e-13), n=960 **3.787e-12** (LDLᵀ
**1.3196e-10**, i.e. dense is **35× worse**). **The crossing is the amendment:** the dense path is a
good oracle at n ≤ 240 and is not one at n=960. The test asserts **both halves** —
`route < 1e-10` AND `denseLDLᵀ > 1e-10` — so A35 item 1 is now enforced by a failing-if-wrong check
instead of by prose. **Third independent arrival at that conclusion** (9b from refinement divergence,
8b at 2.533e-10, this from the crossing). Dense **LU** at n=960 is 2.918e-11 — inside the bar, 7.7×
worse than the route, and disagreeing with dense LDLᵀ by more than either is from truth.
**Keep the provenance separate:** the dense-ALGEBRA validation of the composition is 8.0e-15 (n=48) /
5.0e-14 (n=96); the END-TO-END iterative route is 7.53e-13 … 3.787e-12. The ~2-decade gap is the
`tol = 1e-12` stop, not a discrepancy, and conflating the two would be exactly the A17 provenance
error.

**(6) `λ` DEGRADES ~n² FASTER THAN `x`, AND IT IS STRUCTURAL — ROW 9's BAR IS AMENDED
ACCORDINGLY.** Over n = 240 → 960, `x` grows **1.9×** while `λ` grows **16×** on `grad/−φ`
(4.06e-12 → 6.62e-11) and **85×** on `gen/−φ` (5.01e-12 → **4.24e-10**). Mechanism: the edge
multipliers are **not solved for, they are recovered** post-solve by
`λ_e = (C_e C_eᵀ)⁻¹ C_e (b − Ā x − C_oᵀ λ_o)`, so their error carries `cond(C_e C_eᵀ) ≈ 1.9e4` *on
top of* `x`'s error and on top of a cancellation in `b − Ā x`. **A 1e-10 `λ` bar at n=960 passes
`grad/−φ` with 1.5× margin and FAILS `gen/−φ` at 4.24e-10.** RULED: assert `λ` at **n ≤ 240**
(4e-12…9e-12 across all 12 cells); at n=960 assert `x` + `edgeConstraintResidual` (3.978e-13) and
**log** `λ`. This is A35 item 7 / carry 3's rule applied to a quantity the rule had not been applied
to. **Also: the saddle residual is NOT a proxy for either** — 5.87e-12 at n=48 but **1.45e-8** at
n=240 on `baryedgepin`, 2 500× worse, while `x` moves 4×; it is a difference of `O(‖Ā‖·‖x‖)` terms
and `‖Ā‖` grows with n, so it is scale-carrying and must be reported, never barred tightly.

**(7) A SECOND O(N²) WALL, IN THE CONSTRAINT PATH — issue #85, and it is a scale-target blocker.**
`edgeLengthsBlock` allocates `C` **densely**: `m × 3n` f64 = `3n²` doubles = **22 MB at n=960, 805 MB
at the dense-`A` ceiling N≈5 793, and 21.6 GB at N=30 000** — of which, at 6 nonzeros per row,
**1.4 MB is data.** **§8.1 prices the `k = O(n)` route in flops and never prices the allocation.**
So a route whose solver is O(n) per iteration still cannot reach the target if its entry point is
`C: number[][]`. `prepareNullspace` therefore takes the edge block **sparsely**
(`{vertices, edges, rowStart}`, re-deriving `±T` in O(n)) and only the `O(1)` rows densely where
`k ≤ 16` makes `k × 3n` fine. **That fixes one consumer only:** `constraintSet.ts` still materialises
the dense `C` for `evaluateConstraintSet`, which `dispatch.ts:244` calls every step. **This is the
same shape of wall as the `4N²` dense-`A` cap that motivated the whole phase, one layer over, and it
was unremarked until now.** Carried to task 10.

**(8) ACCEPTED DISAGREEMENT: the route's refusal is a `SolveNotCertifiedError`, not a plain
`Error`.** My carry said *"reuse 8b's refusal, do not add a second convention."* The implementer
reused the **convention** and not the **type**, correctly: 8b's
`HierarchicalSaddleUnsupportedError` is a plain `Error`, which is right where 8b throws it (factory
time, nothing downstream to fold into), but `prepareNullspace` is called on the **solve** path where
`optimizer.ts` folds a bare `Error` into `'singular_system'` — so an open chain would be reported as
a *singular matrix* rather than *wrong topology for this route*, which is A33 / A35 item 6's exact
complaint. `NullspaceRouteError extends SolveNotCertifiedError`; tests assert both `instanceof`s and
that the message names the DENSE path. **One deliberate exception:** a zero-length edge stays a plain
`Error`, because that genuinely *is* a singular system — and it makes `laplacian.ts`'s zero-pivot
guard **reachable and tested for the first time**, having been documented as unreachable for
`assembleCurveLaplacian` systems and kept only to guard a hand-built `TriSystem`. This route is that
hand-built `TriSystem`.

**(9) THE IMPLEMENTER REJECTED ITS OWN VACUOUS GATE.** Its first draft threw on
`projector.idempotenceDefect > 1e-6`; the real value in f64 is **3e-16 … 1e-15** and no admissible
`TriSystem` on a closed ring can produce 1e-6, so the gate was structurally untrippable — it would
read exactly like a passing gate while testing nothing. Replaced by reporting all five projector
defects and **tripping them with an injected wrong-geometry projector (0.279)**. Recorded because
this is the first time on this branch the defect was caught *by the implementer, before shipping*,
rather than by a later reader — and the brief's instruction that produced it was one sentence:
*"assume your first draft of any gate is vacuous until you have tripped it."*

**(10) CARRY CHECKS THAT CHANGED SOMETHING.** The `totalLength` + `edgeLengths` exclusivity throw
**exists and is reachable** (`assertValidConstraintSet`, citing
`2026-07-03-sobolev-constraints-design.md` §3.4) — no issue needed — **but my worry was confirmed
real for this route specifically:** `C_edge C_edgeᵀ` is built from the edge rows alone and stays SPD,
so the nullspace route **would not notice** the exact rank deficiency that stacking them creates. The
construction-time guard is the only thing protecting it, which is now stated in the module.
And the `ell0` perturbation carry earned its place outright: without it `edgeLengths(v,e)` returns
the current lengths, `Φ ≡ 0`, `d = 0`, `x_p = 0`, and **the `skip-particular` knock-out would have
been a no-op** — one of the two §5.1 knock-outs would have been vacuous.

**(11) SCOPE LIMITS, STATED SO THEY ARE NOT LATER READ AS GENERAL.** (a) **One fixture** — the
trefoil at n ∈ {48, 96, 240, 960}. The *algebra* (`C_edge N = 0`, the cyclic-tridiagonal Gram,
`P N = N`) is topology-level and holds for any natural-order closed ring; the **conditioning
extrapolations are curve-specific**, and a near-degenerate curve with one very short edge would be
worse than `n²` suggests. Unprobed. (b) **`θ`-compression dominates every accuracy number here**: the
compressed cell is 3.1e-3 from the true `Ā`, so §3.1's figures mean *"the route solves the `A_bct` it
was handed"*, never *"the route solves `Ā`"* — the n=960 cell's 3.787e-12 must never be quoted as a
claim about the physics. (c) **`λ_e` at scale is unmeasured past n=960**: extrapolating, it passes
~1e-8 near n ≈ 3 000 and ~1e-6 near n ≈ 10 000. Nothing in-tree consumes `λ_e` yet; if anything ever
does quantitatively (a dual-feasibility test, a KKT stopping rule) it will need a refinement step,
which one extra Thomas solve would likely buy back — **unmeasured, not added.** (d) `SCHUR_MAX_K = 16`
is inherited for the `O(1)` rows, so 20 pins plus `edgeLengths` is refused even though the nullspace
half scales fine; whether 16 is still the right ceiling once the `|E|` rows are free is a task-10
question.

**(12) A35 ITEM 5 IS NARROWER THAN I WROTE IT: the `lambda-sign` blindness is a property of `k=3`
SETS, not of `grad/−φ` rhs.** A35 said a flipped multiplier sign *"is INVISIBLE on the `grad/0` and
`zero/−φ` cells of every k=3 set"* and I propagated that to row 9 as a required-cell rule. Measured
here: at `kOther = 3` it is indeed invisible on `grad/−φ` (`|λ_O(1)| = 1.11e-16`, separation
1.06e-16) and caught on `gen/−φ` (3.18e-2) — A35 holds. **But the blindness DISAPPEARS at
`kOther = 6`:** `|λ| = 36.6`, caught at 2.04. Mechanism: with `G` square and invertible,
`Gᵀλ = Nᵀb` forces `λ ≈ 0` for a mean-free top; at `kOther = 6` **`G` stops being square**, the
forcing goes away, and the sign shows up on every rhs. **RULED: the `gen/−φ` requirement stands, but
its JUSTIFICATION is `k = 3`-specific** — do not restate it as "sign errors are invisible on
gradient right-hand sides," which is what A35's wording invites.

**(13) TWO KNOCK-OUTS PASS THE η CERTIFICATE WHILE VIOLATING THE CONSTRAINTS BY 5–7 %. No η-based
gate can see either, and this is §5.3's failure mode with numbers on it.**

| knock-out | η at stop | iterations | `edgeConstraintResidual` |
|---|---|---|---|
| `precond-unprojected` | **8.52e-13** | **identical to clean** | **4.57e-2** |
| `skip-particular` | **1.46e-15** | — | **7.21e-2** |
| clean | 8.52e-13 | 29 | 3.978e-13 |

`precond-unprojected` is the O2 order of item (2) reached from the other side: it converges to the
stated backward error, in the same iteration count, on a direction that leaves the constraint
subspace by 4.6 %. **This is why `edgeConstraintResidual` is a RETURNED FIELD on the solve result and
not a test-only quantity** — the certificate the whole phase leans on (`η`, A35 item 4's
`DEFAULT_ETA_ACCEPT`) is *structurally blind* to constraint violation, because η measures the
residual of the system the solver thinks it is solving. **Generalise, and carry to every later task:
on a constrained route, η is necessary and NOT sufficient; a constraint residual must be reported
alongside it.** Nothing in `src/core/` thresholds either one (`linsolve.ts:410-413`), so this is a
reporting obligation, not yet an enforced gate.

**(14) THREE OPERATIONAL TRAPS, each of which cost a measurement.**
- **`fixture(48)` at the default `leafSize = 8` ALREADY has `nFar = 0`.** So A36 item 1's
  exact-limit pairing **silently measures nothing at n = 48** — the "compressed" and "exact" cells are
  the same cell. It must be done at **n ≥ 240**, and there is now a test asserting that `nFar > 0` at
  the size the pairing is taken. **This qualifies A36 item 1**, which gave the bar (1e-10 at n=240)
  without saying why n=48 is unusable.
- **Two `nohup` runs pointed at the SAME log path interleave through the file offset and NUL-pad each
  other**, producing a file that parses as a plausible log with **two `EXIT=` lines and a STALE
  pass/fail summary.** Use a fresh path per run; **if a log contains two `EXIT=` lines, discard it.**
  This is a worse failure than the killed-run of A36's process note, because the killed run looked
  *unfinished* while this one looks *finished and wrong*.
- **Provenance to keep distinct** (A17 again): `λ_min = +0.651` is for `Zᵀ P A_σ P Z`, *the operator
  being inverted*; `λ_min = 1.0000` is for the *preconditioned product* on `V`. Both appear in task
  9's report, they are different quantities, and quoting either for the other inverts the meaning of
  the SPD check.

> **A24, 2026-10-11 — THE DENSE BYTE-IDENTITY GATE HAS NO INSTRUMENT, AND TASK 8 IS BLOCKED UNTIL IT
> DOES.** Row 8 and §11.4 both designate *"existing goldens"* as the gate for
> *"the dense branch stays byte-identical"*. They cannot measure it. Falsified by command, not argued:
>
> 1. The repo's only self-declared bit gate is `test/golden.test.ts:26-38` (*"must match to the bit
>    (Object.is)"*), and it covers `calculateEnergy` and `gradientAnalytical` — **not the saddle solve.**
> 2. `grep -c 'toBe('` over the saddle tests: `linsolve.test.ts` **0** (8 tolerance assertions),
>    `linsolveFlat.test.ts` **0** (10), `saddleSeam.test.ts` 3, `frozenProjection.test.ts` 8. Reading
>    all eleven: every one is on `accepted` (bool), `timings.step.calls` (int),
>    `conventions.projection` (string), `tau`, `projection_iterations`, or `energy`. **None covers a
>    solution vector**; all vertex/solution comparisons are relative ≤1e-9.
> 3. The two numeric ones are **structurally discrete**: `tau` is **1.0, 0.5 or 0.25 in all 22
>    goldens** — a backtracking-halving count — and `projection_iterations` is an integer 0-3. A
>    dense-solve perturbation must change the NUMBER OF HALVINGS before `toBe` can see it.
> 4. The one exact continuous assertion, `frozenProjection.test.ts:264`
>    `expect(explicit.energy).toBe(omitted.energy)`, is **same-build A/B** — `saddleSeam.test.ts:55-65`
>    says so itself and names the golden suites as its backstop. Both paths move together.
>
> **So the gate reads green whatever task 8 does to the dense path.** That is #70's defect — *a check
> that cannot fail is indistinguishable from one that passed* — on the only hard gate this branch has,
> protecting the only pre-existing production code the phase touches.
>
> **Why a prerequisite and not an emergency:** `git diff --numstat main...HEAD -- src` is additive on
> every file (`N+/0-`) bar `selfTest.ts` (3+/2-), and all five seam files are **UNTOUCHED** on the
> branch. So (a) **task 8 is the first task here that modifies pre-existing production code** — the
> "highest integration risk" label is structural, and it is why tasks 1-7's reviews were clean; and
> (b) the reference captures identically at `main` or at the branch tip, so there is no race — but
> **after task 8's diff lands there is no uncontaminated path left to capture.**
>
> **RULING: author a bit-level dense-path golden BEFORE task 8**, pinning the solution vectors
> (`solveSaddleFromA` / `solveSaddleFrozen` outputs and `sobolevStepSet`'s resulting vertices) with
> `Object.is`, in the shape `test/golden.test.ts` already uses. This is the ordering constraint §11.1
> was missing. Found by the task-8 seam census
> (`.superpowers/sdd/2026-10-09-phase2h-bct-gmres-design/task-8-seam-census.md`) and falsified by the
> controller.

> **Row 7 was SPLIT 2026-10-10 into 7a / 7b / 7c, and §7.3 / §7.4 reference `task 7b` — recorded here
> because those references otherwise resolve to nothing.** `gmres.ts` takes its preconditioner as an
> injected `Preconditioner`, so the ALGORITHM has no mass-weight dependence while every CALIBRATED
> NUMBER does. Hence: **7a** = the algorithm and its instruments, no calibrated bars (landed
> `7f67939` + `f8239a9`); **7b** = the H2/H3 calibration ladder re-measured on the SHIPPED operator,
> LEFT-preconditioned, plus the new H2 knock-out A21 left to measure (§7.3) — it is 7b that §7.3's
> *"task 7b owns"* and §7.4's *"task 7b's harness"* mean; **7c** = the A21 follow-up clause widening
> and four anchoring comments (landed `c24a80e`). **8-pre** (A24's bit-level dense golden) is likewise
> a task with no row of its own; it blocks row 8.
>
> **Why this note exists at all:** a forward reference to a task ID that is not in this table reads as
> assigned and is not — the defect that left A20's `c` re-check ownerless until it was noticed by
> accident. **An OPEN or deferred item must name a row that EXISTS here and whose scope column
> contains the verb the item needs.**

**The prototype ruling: PORT tasks 2–4, do not re-derive.** `scratch/` already contains **3 823 lines
of TypeScript** — `bct.ts` (811), `operators.ts` (455), `bvh.ts` (330), `curves.ts` (224),
`tridiag.ts` (171) and nine `run-*.ts` drivers — and **this is the code that produced every measured
constant this spec quotes**: `κ_near = 108.20`, the 194 ms matvec, every `out-bct-*.json` and
`out-64-*.json`. §0's `ls src/core/sobolev/hierarchical/` check is true but misleading, and the first
draft's "new + test" on tasks 2–4 asked an implementer to reproduce `108.20` from §5.5's prose —
which only works if they guess the same split tie-breaks, leaf stopping rule and adjacency veto. That
is reverse-engineering against a hidden answer key, not TDD.

> **Port, with eyes open about what a port inherits.** The measurements are only valid for the code
> that produced them, so re-deriving would invalidate every constant in §7 as a test oracle. But a
> port inherits the prototype's bugs, and **one is already known**: `tridiag.ts` is where task 1's
> `closed-circle-cyclic` residual comes out **159× over** the first draft's bar (see the mass-weight
> ruling). So: port module-by-module, and for each one state whether a measured constant is being
> used as an **oracle** (the port must reproduce it bit-for-bit) or as a **target** (the port may
> improve on it). Tasks 2–4 use them as oracles; task 1 does not, because its number is wrong.

**The mass-weight ruling: mass = 1, and §12 q1 is CLOSED.**
> **SUPERSEDED 2026-10-11 for the PRECONDITIONER's default (A20 / #72, shipped `ae92267`), and this
> prose was not swept when A20 landed — found by task #72's implementer, which is the FOURTH time a
> re-ruled bar on this branch survived in a second location.** `massWeight` has units `length⁻²`, so
> a bare constant is scale-dependent: under `massWeight = 1` the shipped `κ_excl` moved **62× / 57×**
> (n=64 / 240) between bbox extent 5.4716 and 1.0. `makePreconditioner`'s default is now
> `DEFAULT_MASS_WEIGHT_HAT / ext² = 30/ext²`, after which `κ_excl` moves **≤2 %** over extent
> 1.0 / 5.4716 / 16.4 (the residual is `buildBvh`'s own scale dependence, #69).
>
> **What survives unchanged:** everything below about **1e-3 vs 1**. `laplacian.ts`'s OWN raw
> parameter default is still 1, and the 1e-3 evidence is not overturned — re-confirmed on the solve
> it was about (`triResidual` of the `(−Δ)` solve, unit-extent trefoil): at n=10 000 / 30 000,
> mass 1e-3 gives **2.2e-9 / 1.8e-9**, mass 1 gives 3.6e-12 / 2.0e-11, and **mass 30 (the new
> default at ext 1) gives 5.1e-13 / 3.8e-12** — 7× / 5× better than mass 1. So the new default
> improves the very residual this ruling was decided on.
>
> **Read `mass = 1` below as `laplacian.ts`'s raw default, NOT as the preconditioner's.**

The first draft left it open and set
task 1's bar at a flat `≤1e-9`. Those are inconsistent, because the parameter decides the bar.
MEASURED (`out-tridiag.json`, `relResidual`):

| n | case | mass=1e-3 | mass=1 |
|---:|---|---:|---:|
| 10 000 | `closed-cyclic` | 2.1408e-09 ✗ | 3.2063e-12 ✓ |
| 10 000 | `closed-circle-cyclic` | 2.4350e-08 ✗ | — |
| 30 000 | `closed-cyclic` | 1.6968e-09 ✗ | **2.1214e-11 ✓** |
| 30 000 | `closed-circle-cyclic` | **1.5881e-07 ✗ (159×)** | — |
| 100 000 | `closed-cyclic` | 1.8245e-07 ✗ (182×) | 1.9401e-10 ✓ |

**At mass=1e-3, 7 of the 9 measured cases at n ≥ 10 000 breach `≤1e-9`, by up to 182×. At mass=1
every measured case passes with ≥ 5× margin.** `circle` is one of the three fixture geometries used
throughout §3–§4, so this is not an exotic corner. And there is no verified external reason to prefer
1e-3: the first draft attributed the 1e-3 default to Repulsor's `SetH1MassWeight` / `H1Metric.hpp`,
and **neither symbol occurs anywhere in this repo outside this spec** — no URL, no commit, no path.
That attribution is withdrawn (§10.2). Per CLAUDE.md's swappable-strategies rule the weight stays a
parameter, **default 1**, with the 1e-3 arm's failures recorded above so nobody re-defaults it
silently.

**Task 0a was executed during spec authoring** (§5.3.1a) because §5.3.1 found the cost model's
central multiplier to be sourced from a configuration this architecture forbids, and one CPU run
settled it. It touched only `scratch/`.

**Tasks 0b and 0c are scratch-only and run first.** Between them they are about 8 lines of edits plus
one new short script, and they close four of the five gaps §7 currently ships with: H3's companion
green band has no measured anchor without 0b(i), H2's probe schedule is assumed without 0b(ii), H5
property 1 is tolerance-dependent without 0b(iii), H5 property 2 has no artefact without 0b(iv), and
**§4.1's θ default is undecided without 0c.** None of them touch `src/**`, so they carry no
regression risk and do not wait on tasks 1–11.

> **§7.11 items 1, 9 and 10 are deliberately NOT tasks here.** Item 1 (a genuine f32 hierarchical
> apply through the whole solve) *is* task 10 — it cannot be done earlier because it needs the real
> apply. Item 9 (H4c's 6 missing control arms at n=960, ~3 h) and item 10 (`tangle` geometry and
> n > 4 000) are calibration-breadth work that belongs after task 11 exists to run them, and both are
> recorded in §9.2 as standing measurement gaps rather than silently dropped.

**Task 8 is the only one that touches pre-existing production code.** Everything else is new files. That is
deliberate: it keeps the blast radius of the architecture change to one seam, and it means tasks
1–7 can be verified against measured constants without any risk to the dense path.

### 11.3 TDD, and what "red" means here

Standard order per task: a failing test that encodes a **measured** constant, then the
implementation. Three rules specific to this spec:

- **A test that pins a measured constant must cite it** — file, JSON key, and the conditions
  (`n`, θ, leaf, quadrature, precision). A bare `expect(x).toBeCloseTo(108.2)` is unmaintainable;
  the next reader must be able to find where 108.2 came from.
- **Never assert an exactly-zero value without saying why it is structural.** `A_bct·1 = 0` for the
  apply is structural (§2.2) and must be asserted exactly; a *row-sum* zero would mean the test
  harness is faking it. CLAUDE.md's "gates name their impossible values" applies to tests too.
- **Tolerances come from §7's measured α floors**, not from taste. If a test needs a tolerance that
  no measurement supports, that is a finding, not a number to pick.

### 11.4 Done

- `bun test` green, `bunx tsc --noEmit` clean.
- Every §7 gate runs and **records its value**; every INVALID predicate has a test that reaches it.
- The dense path is bit-identical with `op` absent — **against the bit-level dense-path golden A24 requires, NOT against the pre-existing goldens, which cannot measure it.** The
  `oracle/golden/*.json` files are unmodified; **the seam TEST files are not** — adding a `kind`
  discriminant necessarily edits the object literals at `saddleSeam.test.ts:104-125`, `:129-144`,
  `:179-198`, `gradient.test.ts:173-193`, `cr.test.ts:363` and `ldlt.test.ts:337` — six sites across
  four files, and **`tsc` is the oracle for the count, not this list**.
- `git diff` on `src/core/tangentPointEnergy.ts` is **empty** (@issue utof/repulsive-test2#52 —
  `bun run check` autofixes repo-wide and has silently rewritten `Math.pow` → `**` in that hot path;
  commit `52c3569` is the guard).
- R1's served-N ceiling published (§7).

---

### ⚠️ A38 — task 9c is DONE and ACCEPTED; **row 9c's own sentence about the `rs-identity` bar is FALSE**; A20's open item is answered at the grid's resolution and NOT converged; and `A_σ`'s conditioning is measured for the first time

Report: `.superpowers/sdd/2026-10-09-phase2h-bct-gmres-design/task-9c-report.md` (918 lines), data
`scratch/sweep_shapes_9c.json` (105 rows). Controller ruling and the four dispatch conditions:
`progress.md`, "TASK 9c ACCEPTED". Measurement-only task: no `src`/`test`/`bench` change, nothing
committed. Every figure below is the **Python dense exact all-pairs `A_σ`**, not the TS BVH/BCT `A_σ`.

**1. Output (1) — the minimax-absolute `c` at n ≥ 4 000.** Grid argmin **35 at n = 1 920 (worst
`κ_excl` 207.42, arc binding) and 35 at n = 4 000 (232.98, arc binding)**, continuing A20's
15 / 20 / 20 / 25 / 30. **The trend FLATTENS at the 5-wide grid's resolution; it does not BREAK.** The
supplementary continuous arc/helix switch point is 30.32 → 32.91 → 34.84 over n = 960 → 4 000, with
decelerating per-octave increments (+3.96, +2.59, +1.82) — so at n = 1 920 the grid argmin overstates
the continuous point by about one step. **`35 → 35` IS NOT CONVERGENCE:** the continuous point still
moves +1.93 per octave and the worst `κ_excl` at the argmin still rises (159.15 → 207.42 → 232.98;
exponents +0.382 then +0.158, LS +0.266 over 960…4 000). **A20's extrapolated 40–45 at N = 30 000 is
neither confirmed nor refuted and is NOT banked.** Argmin interior to its grid at both new rungs.

The mechanism, which is the part worth carrying: **at `c = 30` the binding shape is the HELIX and its
`κ_excl` grows n^+0.47 over the last three rungs; at `c = 35` and above the binding shape is the ARC
and it grows n^+0.15 (+0.11 at 40, +0.09 at 45).** The minimax `c` rises with n because the helix side
degrades faster than the arc side, and the shipped value sits on the steep side of the switch. Per-shape
argmins span 0 (arc, which wants no mass term at all from n = 480 up) to 70 (helix at n = 4 000).
**→ issue #88** proposes `c`: 30 → 35, gated on a TS/BCT re-measurement first; it is an exported
constant, so the inline-fix gate sends it to an issue rather than an edit.

**2. ROW 9c'S OWN SENTENCE IS FALSE and is struck here.** The row says the `rs-identity` knock-out bar
`> 100` "**so it FAILS inside A20's own projected band**". ~~It fails inside 40–45.~~ **It does not.**
Measured at the binding cell (trefoil, n = 64): **148.5 at m̂ = 40 and 118.0 at 45 — both above 100.**
The bar first fails between m̂ = 48 (103.8) and 49 (99.7), i.e. at **m̂ ≈ 48.9**, four above the band's
top. The claim rested on the **66.4 at m̂ = 60**, which is OUTSIDE the band the sentence names. At the
landed `c = 35` the ratio is **196.4, margin ×1.96**, and it survives at every measured n (1 103 at
n = 120, 5 048 at 240 — the ratio grows ×4–6 per rung, so n = 64 binds at every `c`). The correct
statement: **the bar needs no change for `c ≤ 48`; the margin at 45 is thin (×1.18) and shrinks as `c`
rises.** `precond.test.ts:571` is sound as written.

**3. Output (2) — where `λ₂ = 1` falls in `(n, m̂)`.** `lam_min` crosses DOWN through 1 at
**m̂ ≈ 85.4 (trefoil, n = 1 920 and 4 000; 84.0 → 85.4 over n = 64 → 4 000, essentially
n-independent), ≈ 93.1 (helix), ≈ 27.7 (circle, stable)**; the **arc has `lam_min < 1` at EVERY m̂
including 0 for n ≥ 240** (0.2986 at n = 4 000, m̂ = 0). So at the shipped `c = 30` and at 40/45 the
arc and circle are below 1 while helix and trefoil are above — **`κ_excl` as `λ_max/λ₂` is already not
`κ_excl` on the arc and circle in the production regime**, which is #78 item 2 confirmed. Every figure
in this phase that needs the 1⊥ minimum must use the explicit-`Q` projection, never
deflation-by-value: in this regime the constant mode and the smallest 1⊥ mode cannot be told apart by
value. **The shipped guard `precond.test.ts:1092` (`λ₂ > 1.5`, on the trefoil) is NOT near firing** —
margin **×2.48 at n = 4 000, m̂ = 30** (identical at every m̂ from 20 to 40, because the trefoil's
`lam_min` is a mass-weight-independent mode there), ×2.21 at 45, ×1.82 at 50; it would fire at
**m̂ ≈ 69** at every n. **Two conflicts go to issues:** `precond.test.ts:547`'s "`λ₂` extrapolates to
≈1.3 at n = 30 000" is contradicted by a flattening trefoil `lam_min` (3.719 at n = 4 000, local slope
−0.013; continuation gives ≈3.6) → **#89**; and the **helix `lam_min` is the one quantity still
falling** (3.093 → 2.131 → **1.588** at n = 960/1 920/4 000, slopes −0.80, −0.54, −0.40, in a
mass-independent band so `c` cannot fix it) → **#90**.

**4. `A_σ`'s OWN CONDITIONING, measured for the first time on this branch** (controller probes
`scratch-ctl/cholpivot.py` + `cholpivot_scaled.py`, JSON beside them). The task-10 carry above asked
for `chol_pivot_ratio` at every (shape, n) with the arc factored first; 9c did not report it, so the
controller measured it. **All 12 cells factor (`dpotrf info = 0`): `A_σ` is SPD-factorable at n = 4 000
on all four shapes.** The raw ratio `(min diag L / max diag L)²` falls as **n^−2.28** on the arc
(3.10e-6 → 6.38e-7 → **1.19e-7** at n = 960/1 920/4 000).

**The ratio is NOT scale-free, and the Jacobi-scaled follow-up proves the decay is REAL rather than a
dynamic-range artefact** — `cond₂(D^−1/2 A_σ D^−1/2)` on the arc is **3.27e8 → 3.32e9 → 3.85e10**, i.e.
**n^+3.34**, and the scaled pivot ratio decays at −2.33 against the raw −2.28 (scaling moves the level
2.1×, the trend not at all). Consequences, all of which replace the carry's `sqrt(eps)` flag:

- **The `sqrt(eps_f64) ≈ 1.5e-8` flag I pre-registered measures the wrong quantity.** Worst measured is
  1.19e-7 (7.9× above it, not crossed), but the raw trend would cross it at N ≈ 9 950 for geometric
  reasons. **Drop it. Use `cond₂` of the unit-diagonal matrix.**
- **f64 is safe across the whole target band.** `cond₂ ~ 1/eps_f64 = 4.5e15` is not reached until
  **N ≈ 130 000** on the n^3.34 trend. At N = 30 000, `cond₂ ≈ 3.2e13`.
- **But a dense DIRECT f64 solve at the scale target carries `cond₂·u_f64 ≈ 7.2e-3`** — within one
  order of magnitude of f64's useful limit, against `κ_excl ≈ 233` for the PRECONDITIONED operator at
  n = 4 000. **This is a quantified reason the preconditioner is load-bearing rather than an
  optimisation**, and it is independent of the memory argument (`[DESIGN §1]`, CLAUDE.md: 3.6 GB and
  ~9e12 flops at N = 30 000).
- **An f32 FACTORISATION or SOLVE of `A_σ` is already out of reach at the n = 960 FLOOR**, not at
  target scale: `cond₂` exceeds `1/eps_f32 = 8.4e6` by **39× on the arc at n = 960** and 21× on the
  circle at n = 1 920. There is no f32 factorisation in `src` today (the only Cholesky is
  `src/gpu/saddle.ts:92 choleskyF64`, f64 with an explicit `!(d > 0) || !Number.isFinite(d)` throw), so
  nothing is broken — but **no f32 factor path may be added without scaling first and re-measuring.**
- **CROSS-VALIDATED BY AN INDEPENDENT INSTRUMENT (added 2026-10-12, task 9c round 2 §5.5).** LAPACK
  `dpocon`'s 1-norm estimate on the already-computed Cholesky factor gives `cond₁(arc)` = 8.05e9 ->
  **9.30e10** over n = 1 920 -> 4 000, exponent **+3.335**, against my Jacobi-scaled `cond₂` exponent
  **+3.34**. Different quantities, different codebases, agreeing to **three digits** — and the raw
  pivot ratio ρ disagrees at **−2.29** on the same cells. **That settles which instrument is the
  n-extrapolation one: `rcond₁`/`cond₂`, never ρ.** On the right quantity the arc's factor carries
  `cond₁·u_f64` = 2.1e-5 at n = 4 000 (~4.7 correct digits), 2.6e-5 at n = 4 300, 2.1e-4 at n = 8 000,
  **1.7e-2 at N = 30 000** — so the dense-`A_σ` arc route is **accuracy-limited near the SCALE TARGET,
  not near n = 4 300.** A 9c conclusion to the contrary is struck (controller ruling R11 in
  `progress.md`): its ρ bar of 1e-7 was set as "6.7× √ε" and therefore inherited **this item's own
  withdrawn flag**. 9c's trip table also shows ρ and `rcond₁` trip in the **same decade** of a
  definiteness perturbation (both pass at δ = 1e-7, both trip at δ = 1e-8), so preferring `rcond₁`
  costs no sensitivity. **CONFIRMED ON 7 RUNGS × 4 SHAPES (9c round 3, `checks_9c.py ladder`):** the
  arc's LOCAL `cond₁` exponents converge **+3.39, +3.36, +3.35, +3.34, +3.34, +3.34** over
  n = 64…4 000 while its ρ exponents sit at −2.26…−2.29 — the split is not an artefact of a two-point
  fit, and it reproduces on `dpocon` alone without my instrument. Circle and trefoil show the same
  split (~−2.3 vs +3.33). **The two instruments agree on the EXPONENT only and were never
  cross-checked on the LEVEL** — `cond₁` and `cond₂` differ by the 1-norm/2-norm constant and are not
  interchangeable in absolute value. The harness now gates `rcond₁ ≥ 1e-12` (the value pre-registered
  in its docstring before any factor was measured; **ruled NOT to be loosened** — it is conservative,
  not wrong, and the model that would justify loosening has a 40× spread across its own probe points). **Any guard on this quantity gates `rcond₁`; ρ is a diagnostic only** — a ρ bar
  at 6.7×√ε refuses a HEALTHY matrix from n ≈ 4 300 up, which is as costly as a gate that cannot fire.
- **THE DISTINCTION ROW 10 MUST NOT LOSE.** `cond₂` bounds the error of **SOLVING** with a matrix, not
  of **APPLYING** it. Row 10's deliverable is an f32 **matvec** inside an f64 outer iteration, whose
  error goes as `eps32 × growth`, **not** `eps32 × cond`. **These numbers must not be quoted as
  evidence against the f32 apply.** They bound exactly one thing: an f32 factor-or-solve of `A_σ`.
  (This is the `tolerance-transfer-across-provenance` / `figures-carry-their-regime` failure mode; it
  has already cost this branch four repeats.)

**5. A GATE-METHODOLOGY DEFECT IN THIS SPEC'S OWN DISPATCH PRACTICE — mine, not the worker's.** 9c's
matrix-free Lanczos route **failed its pre-registered gate, 99/101**, and the route was used anyway
under a rule the report labels POST-HOC. The two failures are helix at m̂ = 0 and 0.03, on the
**residual bar only** (3.6e-8, 1.4e-8 vs 1e-8), while their `κ` agreement with the dense reference is
2.9e-7 and 8.1e-8 — **35× and 120× INSIDE the 1e-5 bar that gates the quantity actually used**. Those
cells are inputs to no output (helix `κ` at m̂ = 0 is 5.1e6 against contenders near 200). **The defect
is that I wrote the gate as a CONJUNCTION over a cell set that was a superset of the decision region,
without saying what a failure outside the decision region means** — which promotes an irrelevant cell
into a blocking failure and leaves the worker choosing between a task-destroying stop and an
unpre-registerable rule. **Binding rule for every remaining gate in this phase: declare the DECISION
REGION and gate it as a conjunction; report the COVERAGE REGION as a distribution, never as a
conjunct.** Accepted with the failing family quarantined; the single sweep row that violates even the
post-hoc rule (helix n = 1 920, m̂ = 0) is excluded from every claim.

**6. "EQUIVALENT MUTANT" IS NOT "VACUOUS GATE" — the first correctly-diagnosed non-trip on this
branch.** All four `Π` (mean-removal) drops in `minv_core` leave `κ` within 3e-10 of undropped, so they
do not trip 9c's gate. That is **not** a tenth vacuous gate: 9c hypothesised only one `Π` was
redundant, **falsified its own hypothesis** with a supplementary run, and then showed algebraically and
by measurement why all four are (`|Lf·1|_∞/|Lf|_max` = 1.0e-15; `|mean(L v)|/rms(L v)` = 5.6e-15 for
`v ⊥ z0`). A vacuous gate **cannot fail for the reason it names**; an equivalent mutant **has nothing
to detect**. The `Π`s stay in as defensive code. The gate's live bars are proven by real trips:
`θ = 0` in the SA run, the `rs-identity` `Lf` swap, a wrong factor order, mass-weight perturbations at
1e-6 (trefoil) / 1e-5 (arc), and a slack `eigsh` tolerance — the last of which trips **only** the
residual bar, which is why that bar exists and is not droppable.

**7. Residual limits of 9c, pre-registered here so nothing quotes past them.** (a) **No independent
eigensolve exists at n = 4 000 at all** — every figure at that rung is the matrix-free route
self-validated by residual plus the n ≤ 1 920 cross-checks; route-vs-dense agreement does not degrade
with n (3.9e-6 at 960, 3.7e-6 at 1 920 on the arc), and a 17–33 % decision cannot be moved by 1e-5, but
**any future claim needing better than 1e-2 on an arc `κ` at n ≥ 1 920 must first establish an
independent route** — three dense routes already spread 9.5e-6 there, so the arc `κ` is known to ≈1e-5
by ANY route. (b) `lam_next` from the matrix-free route is informational only (41/105 unconverged;
Krylov skips degenerate copies) — no conclusion may use it. (c) The vectorised assembler is now
validated against the pure-Python loop **at n = 4 000 too** (trefoil, max entrywise rel **5.80e-14**
against a 1e-12 bar, 99.963 % bit-identical — `scratch/validate_vec4000_9c.json`, run by the
controller after the 12 733 s loop reference finished); at n ≤ 1 920 all four shapes pass at 5.5e-14.
n = 4 000 covers the trefoil only. (d) The replicated `rs-identity` ratio at m̂ = 30 is **271.78
against #78's published 273.3** (0.55 %); #78's own rounded `κ`s give 271.6, so **273.3 is not
recoverable from #78's own table**. Unexplained, affects nothing, not a regression.

**8. THE f32 WALL IS A ONE-SHAPE NUMBER, and the exponent behind it is now confirmed on four**
(controller probe `scratch-ctl/cholpivot_scaled.py` / `.json`, 12 cells; added 2026-10-12 after item 4).
`cond₂` of the unit-diagonal `D^−1/2 A_σ D^−1/2`:

| shape | n=960 | n=1 920 | n=4 000 | exponent | `cond₂·u_f32` at 4 000 | f32 wall (`·u_f32 = 1`) |
|---|---:|---:|---:|---:|---:|---:|
| **arc** | 3.270e8 | 3.322e9 | **3.853e10** | **+3.34** | **2 297** | **n ≈ 394** |
| circle | 1.749e7 | 1.759e8 | 2.029e9 | +3.33 | 121 | n ≈ 948 |
| trefoil | 1.233e6 | 1.237e7 | 1.425e8 | +3.33 | 8.49 | n ≈ 2 104 |
| helix | 8.003e4 | 7.592e5 | 8.550e6 | +3.30 ⚠ | 0.510 | n ≈ 4 907 ⚠ |

⚠ **HELIX CAVEAT (added 2026-10-12, 9c round 3):** on a 7-rung `cond₁` ladder the helix is the one
shape **not yet asymptotic** (local exponents −2.23 / +3.28 against the other three shapes' ~−2.3 /
+3.33). My +3.30 above is a 3-rung fit, so **the helix row — and the n ≈ 4 907 wall derived from it —
is the least settled of the four.** It is also the row that sets the UPPER end of item 10's 12.5×
spread, so treat that endpoint as softer than the arc's.

**(a) The GROWTH LAW is geometry-independent and reproduces this spec's own theory.** All four shapes
give +3.30…+3.34 against §5's `2s = 10/3 = 3.3333` and #59's measured `n^3.3338` — **the first
multi-shape confirmation of that exponent on this branch**, on a different instrument (dense exact
`A_σ`, Jacobi-scaled) than #59 used.

**(b) The PREFACTOR varies 4 506× with geometry, and §6 / §7.8 quote one shape's.** §7.8's table
(`κ₂(A_σ)` = 1.393e6 / 1.404e7 / 1.622e8 at n = 960 / 1 920 / 4 000) and the `n ≈ 2 025` wall derived
from it are **the TREFOIL's**. My trefoil column agrees to **11–13 % at every rung, including n = 4 000
where §7.8's value was EXTRAPOLATED** — mutual validation of two independent instruments, with mine
lower by the amount Jacobi scaling predicts. But the wall is therefore **not one number: n ≈ 394 (arc)
to n ≈ 4 907 (helix), a 12.5× spread, and the ARC'S WALL IS BELOW THE n = 960 FLOOR.**

**Binding consequences.** (i) **§6's "Plain f32 walls at n ≈ 1 200–2 000" and §7.11 item 3's
"n ≈ 2 025" must name their fixture.** Row 11's R1 "publishes three ceilings" — a ceiling that moves
12.5× with the fixture is not publishable as a scalar; publish per shape or name the shape.
(ii) **The preconditioner's SHAPE-flattening is now measurable and large:** unpreconditioned spread at
n = 4 000 is **4 506×** (arc/helix), while 9c's *preconditioned* `κ_excl` at the same rung on the same
four shapes spans **233.0 (arc) → 19.65 (trefoil), 11.9×**. `Ã⁻¹` removes roughly **379× of geometry
dependence** on top of the five orders of n-dependence §7.8 already records — **a stronger form of this
spec's central thesis, free from data already on disk.** (iii) **§6's own most load-bearing assumption
is single-shape:** it states that `κ₂(Ã⁻¹A_σ)` flatness above n = 1 920 is **ASSUMED** and "carries the
whole mixed-precision scheme" — measured on one shape, while the quantity it must flatten varies
4 506×. (ii) is evidence the flattening generalises, but `κ_excl` and `κ₂(Ã⁻¹A_σ)` are different
normalisations (207 vs 1.13e4 at n = 1 920) and are **not** interchangeable. Closing this is a
row-10/11 input, not a claim. **And none of (i)–(iii) licenses quoting these numbers against row 10's
f32 APPLY — see item 4's last bullet.**

**9. §9.2'S THIRD-RANKED GAP — "`κ₂(Ã⁻¹A_σ)` above n = 1 920, ASSUMED flat, and that assumption
carries the entire mixed-precision scheme" — IS ANSWERED BY 9c's DATA. The bar is met with 48× MORE
room than §6 claims, by an operator that does NOT satisfy §6's stated flatness mechanism.**

§9.2 costed this gap as "low–medium — task 0b(iv) + **matrix-free Lanczos; needs no dense matrix**".
9c built and gated exactly that route and swept it over four shapes at n = 64…4 000, so the gap closes
without a new task. Two corrections fall out of tracing the numbers to their source:

**(a) §7.8's `κ₂(Ã⁻¹A_σ)` row and §6's `6.716e-04` are the LAB `pinv((−Δ)⁺)` construction, not the
shipped operator.** Provenance: `task-6-review-math.md:171-178` measures both on the SAME `A_bct,σ`
(θ=0.25) and the SAME hierarchical `L_f`, unnormalised trefoil — **lab 1.09765e4 / 1.10955e4 /
1.12258e4 / 1.13050e4** at n = 64/120/240/480 (the digits §7.8 quotes, matching its oracle to 5
figures) against **shipped `massWeight=1`: 59.04 / 64.60 / 69.70 / 71.94**. And `1.1269e4 × u_f32
(5.96e-8) = 6.716e-04` exactly, so §6's property-3 figure and its "14.9× margin" are the lab number.
(The shipped column's own identity as `λ_max` in disguise is #71, closed.)

**(b) The shipped quantity is ~100× BETTER in absolute terms and NOT flat.** 9c's `κ_excl` at fixed `c`,
worst over arc/circle/helix/trefoil, from §8.2's grid-independent view:

| | n=64 | n=960 | n=1 920 | n=4 000 | exponent (last 3 rungs) |
|---|---:|---:|---:|---:|---:|
| `c = 30` (shipped) | 90.8 | 159.2 | 231.1 | **309.9** | **+0.466** |
| `c = 35` | 115.7 | 189.4 | 207.4 | **233.0** | **+0.145** |

Against §7.8's claimed `∝ n^0.0010`. **So §6's mechanism — "the preconditioned number does not move at
all" — is FALSE for the shipped operator: it moves as n^+0.15 to n^+0.47 depending on `c`.**

**(c) The ruling survives anyway, with far more margin than the spec claims.** Property 3 is
`κ₂(Ã⁻¹A_σ)·u_inner ≤ 1e-2`, i.e. `κ ≤ 1.68e5` at `u_f32 = 5.96e-8`:

| n | worst `κ_excl` at the grid argmin | `·u_inner` | headroom to 1e-2 |
|---:|---:|---:|---:|
| 960 | 159.15 | 9.49e-06 | **1 054×** |
| 1 920 | 207.42 | 1.24e-05 | **809×** |
| 4 000 | 232.98 | 1.39e-05 | **720×** |

**720× at n = 4 000 against §6's quoted 14.9× — the spec is pessimistic by 48× because it quotes the
lab operator.** Continuing the WORST exponent (+0.466, at `c = 30`) to N = 30 000 gives `κ ≈ 800` and
`·u_inner ≈ 4.8e-5`, still **~200× inside the bar**. That continuation is an EXTRAPOLATION and per
§7.0 cannot be a bar — it is stated as headroom, not as a gate value.

**Net effect on row 10: the mixed-precision precondition is in far better shape than §6 states, and
for a different reason than §6 gives.** `[CH18]`'s `φ = (c₁κ∞(A)+c₂)u_s ≪ 1` is satisfied by ~3
orders at every measured rung on four shapes, not by 1.2 orders on one. What is NOT established is
flatness, and nothing should be built on flatness again — build on the measured margin.

**10. "FOUR INDEPENDENT DERIVATIONS AGREE ON 1 200–2 000" IS A COMMON-MODE AGREEMENT: all four are
the TREFOIL, and the one of them I have now measured on four shapes spans 12.5×.** This sets row 10's
scope and row 11's R1 ceilings, so it is recorded as a binding scope item, not an observation.

§7.11's table lists derivations 1 (`n ≈ 1 228`), 1b (1 773), 2 (1 225), 3 (2 025) and 4 (breached at
n = 2 000). **Every one is measured on `trefoil`/`torusKnot`** — and §9.2 row 5 already says so in
general terms: *"all direction/SPD/line-search results are on `trefoil` alone; `κ_near` spans 14.67×
across geometries."* Derivations 1, 1b and 2 additionally share one instrument family (the measured
apply-noise / residual-floor stream), so the count of genuinely independent axes is **lower than
four** on both fixture and instrument.

**Agreement across derivations is evidence of robustness only where the derivations VARY what could be
wrong.** These vary the *estimator* and hold the *geometry* fixed — memory `common-mode-blindness`: a
differential cannot pin anything its paths share. And the shared factor is now measured to matter:
**A38 item 8 gives derivation 3 per shape as n ≈ 394 (arc) / 948 (circle) / 2 104 (trefoil) / 4 907
(helix)**, a 12.5× spread, with the trefoil sitting mid-range and the arc's wall **below the n = 960
floor.** So at least one of the four is strongly geometry-dependent, and the geometry-independence of
the other three is **UNTESTED**, not established.

**Consequences, binding:**

- **Row 10 cannot answer "locate the wall" with one fixture** — the answer is a range, and a single
  number from a single shape would reproduce exactly the error A38 item 8 corrects. **It must locate
  the wall on at least the ARC and the HELIX** (the measured extremes of the 4 506× conditioning
  prefactor) in addition to whatever fixture it uses for continuity with §7.11. This is not scope
  creep: it is the minimum that answers the question the row asks.
- **Row 11's R1 must publish the f32 ceiling per shape or name its shape**, and where only one shape
  is measured the others are **"NOT ESTABLISHED"** — §7.11's own rule, applied to its own band.
- **§6's "Plain f32 walls at n ≈ 1 200–2 000" keeps its number only as a trefoil-class statement.**
- **What would falsify this item:** derivations 1, 1b or 2 re-measured on arc and helix coming out
  within ~1.5× of their trefoil values. That is a cheap check and it is row 10's first stage.

**11. §7.3's INVALID CLAUSE 2 IS ASYMMETRIC, and row 10 is the first task that can trip its blind
side.** Found pre-dispatch while censusing the f32 seam (`task-10-f32-census.md` Q6); recorded here
because it is a defect in a pre-registered INVALID predicate, not in code.

`EtaProbeOptions.uOp` (`gmres.ts:355-358`) is documented *"REQUIRED, with no default, so it cannot
silently be `u_f32` for an f64 run (INVALID clause 2)"*, and validation (`:564-566`) checks only
`uOp > 0 && Number.isFinite(uOp)`. `HierarchicalOperator` (`operator.ts:30-33`) carries no variant, so
**an f32 operator pairs with `U_F64` with no error.** Clause 2 fires on `0 < η < u_op/10`:

| case | η | `uOp/10` | clause 2 fires? |
|---|---:|---:|---|
| f64 operator mislabelled `U_F32` | ~1e-16 | 6e-9 | **YES** — the case the TSDoc names |
| **f32 operator mislabelled `U_F64`** | ~1e-5 | 1.1e-17 | **NO — structurally blind** |

The guard was built against the only misuse that existed when it was written (every operator was f64),
and **row 10 introduces the mirror-image misuse it does not cover.** The self-check compounds it:
`gmres.ts:305-306` states that an η below `u_op/10` means the residual was not formed in that
precision, but `:687` forms the residual in f64 **regardless**, so that check also rests entirely on
the caller's `uOp`. A green η with a mismatched `uOp` is a vacuous result — the defect class this
branch has shipped ten times, pre-armed on the one seam row 10 must cross.

**Binding on row 10** (brief §4.1): every f32 run passes `uOp: U_F32` **and the report carries a
RECORDED cross-check that the pairing is correct**, not an assertion. A minimal `src` guard (the
operator carrying its own variant so `uOp` can be derived or validated) is **in scope for row 10**,
which already names `gmres.ts` and `apply.ts`; if it grows beyond minimal it becomes an issue.

**Not yet fixed, and deliberately so:** narrowing clause 2 to catch both directions requires the
operator to expose its precision, which widens an exported interface — a decision that belongs with
row 10's measurement rather than ahead of it.

## 12. Open questions deferred to implementation

1. ~~**The mass weight in `(−Δ)`.**~~ **CLOSED: `laplacian.ts`'s raw default is 1; the
   PRECONDITIONER's default is `30/ext²` (A20 / #72, shipped `ae92267`) because `massWeight` has
   units `length⁻²` and a bare constant made `κ_excl` move 62× with curve extent. The 1e-3-vs-1
   measurement below stands and the new default beats both on it.** The operator is stiffness
   `(w/ℓ)[[1,−1],[−1,1]]` plus mass `(w·ℓ/6)[[2,1],[1,2]]`; pure stiffness is **singular on a closed
   curve**, so the mass term is load-bearing. The weight is decided by measurement in §11.2's
   mass-weight ruling: at 1e-3, 7 of 9 measured cases at n ≥ 10 000 breach a 1e-9 relative residual by
   up to 182×; at 1 every case passes with ≥5× margin. The `H1Metric.hpp` / `SetH1MassWeight`
   attribution for the 1e-3 default is **withdrawn** — neither symbol exists anywhere in this repo
   outside this spec (§10.2). It remains a runtime parameter; only the default is decided.
2. **Where exactly the mixed-precision switch goes**, pending task 10's measurement.
3. **Whether θ=0.5's bad modes are fixable** (§4.3) — an open investigation, not a known patch.
   Note this is now *downstream* of H6 (§7.9, task 0c): H6 decides whether the bad modes matter in
   practice before anyone spends effort on fixing them.
4. **Whether the per-iteration GPU readback tax forces Phase 3's zero-readback work** into this
   phase (§9.1 item 2). #61 ruled Phase 3 stays after 2H; that ruling was about ordering, and the
   readback tax could reopen it.
5. **Leaf size, split plane, adjacency certificate** (§9.3) — all cheap, all unexplored.

---

## A39 — task 10's Stage 0 falsified three controller claims before measuring anything (2026-10-12)

Row 10's Stage 0 was dispatched as "verify the brief against source and tell me where it is wrong."
It found three material defects, all three reproduced by me against source, all three mine.
**This is the fourth, fifth and sixth controller claim killed on this branch by a worker's
pre-measurement check** (after the √ε flag, the R4 mechanism, and the §7.11 fixture) — the pattern is
now the branch's strongest process result and is carried in `[A38 item 10]`'s spirit: a stage whose
only job is to attack the brief pays for itself every time.

### 1. The `u_f32` convention was wrong by a factor of 2, so EVERY wall in A38 item 8 was 1.23× too small — CORRECTED IN PLACE

`A38 item 8`'s walls were derived from `cond₂ · u = 1` with **`u = 2^-23 = 1.19e-7`, the machine
EPSILON.** This repo's own constant is the **unit roundoff**: `gmres.ts:74` is
`export const U_F32 = 2 ** -24`, with TSDoc that says so explicitly ("NOT the machine epsilon"), and
five further sites agree numerically (`apply.test.ts:55`, `lowOrder.test.ts:57`, `precond.test.ts:92`,
`oracle/check_kappa_peredge.py:61`, `bench/gpu/k1-calibration.py:89`, all `5.96e-8`).
**The spec was internally inconsistent about its own constant:** `[A37]`'s `κ·u_f32 = 1.12` at
N = 30 000 (`:3494`) is correct under `2^-24` (`1.883e7 × 5.9605e-8 = 1.1222`), so A38 contradicted A37.

Re-derived at `cond₂ = 2^24 = 1.678e7` from the n = 4 000 anchor and each shape's own exponent:

| shape | A38 as published (`2^-23`) | **CORRECTED (`U_F32 = 2^-24`)** |
|---|---:|---:|
| arc | n ≈ 319 | **n ≈ 394** |
| circle | n ≈ 770 | **n ≈ 948** |
| trefoil | n ≈ 1 710 | **n ≈ 2 104** |
| helix | n ≈ 3 978 ⚠ | **n ≈ 4 907 ⚠** |

The factor is `2^(1/3.33) = 1.232` in `n`, uniform, so **every RATIO and the 12.5× spread survive
unchanged** and item 8's and item 10's conclusions all stand. Two readings do shift: the arc's wall
(394) is still below the n = 960 floor but by 2.4× not 3.0×, and **the trefoil's own wall (2 104) now
sits just ABOVE §7.11's 1 200–2 000 band** — so the band is at the edge of even its own fixture, which
*strengthens* item 10. Corrected at all 7 sites plus the `cond₂·u` column (`2 297 / 121 / 8.49 / 0.510`)
and row 10's "19.5× past `1/u_f32`" (was "39×"). **Pre-edit copy: `/tmp/spec-pre-c3.md`.**
**Why it survived:** the number was never cross-checked against the repo's named constant, only against
itself. Any `u_f32` quoted with a wall must now name `2^-24` or `2^-23` in the same sentence.

### 2. An f32 apply HAS already been run through the whole solve — row 10's novelty claim was false

`[§9.2]`'s top gap and row 10's deliverable both read "a genuine f32 hierarchical apply through the
whole solve", and my f32 census confirmed it unmeasured. **Both are wrong.** Task 7b did exactly that:
`scratch-7b/h23.ts:194-205` builds `prepareA(..., { variant: F32 })`, wraps it as a
`HierarchicalOperator` reusing the f64 `σ`, and `ko.ts:70-71,110-111` drives it through `gmres` with
`probe.uOp = U_F32` over n = 64…3 840 × 5 seeds, measuring an `η_bct` floor of **0.306–0.412 `u_f32`**
(`task-7b-report.md:84-96`).

**Why the census missed it, and the general lesson:** the entire
`.superpowers/sdd/2026-10-09-phase2h-bct-gmres-design/` tree — which holds every campaign script of
this phase — **is excluded from the codebase-memory graph index.** A census driven off the graph, as
briefed, is structurally unable to see prior scratch measurements. **Any "has this been measured
before?" question must grep the SDD tree in addition to querying the graph**; the graph answers "what
does the shipped code do", never "what has this campaign already run". Added to the subagent briefing.

What is genuinely unmeasured, and so still row 10's: the **forward-direction** differential (7b read
residuals, never `x̃32` vs `x̃64`), shapes other than the Lissajous trefoil, the bbox-normalised
fixtures, and θ ≠ 0.5.

### 3. "Trefoil" is TWO different curves in this corpus

`src/core/fixtures.ts:10 trefoil` is the Lissajous form `(sin t + 2 sin 2t, cos t − 2 cos 2t, −sin 3t)`
at native extent 5.4716 — the curve every Python probe in A38 item 8 and §7.11 derivation 3 uses.
`test/sobolev/hierarchical/fixtures.ts:79 torusKnot` is a `(p,q) = (2,3)` torus knot on `R = 2,
r = 0.8`, bbox-normalised — the curve §7.11 derivations 1/1b use. Same knot type, **different geometry**
(`L/extent` 5.268 vs 5.568, checked numerically by row 10's `scratch-10/shape_check.ts`).

**Effect on A38 item 10, both ways.** The four derivations are NOT all on one fixture, so the
common-mode critique is weaker than item 10 states at the exact-curve level. But it holds at the level
that matters: the two curves were never cross-measured, the 12.5× per-shape spread dwarfs any plausible
trefoil-to-trefoil difference, and **the corrected trefoil wall now falls outside the band anyway.**
Item 10's conclusion is unchanged; its "all four are trefoil" wording is now "all four are
trefoil-TOPOLOGY, across two distinct geometries, neither cross-measured."
**THE HELIX HAS THE SAME DEFECT, and it bites row 10 harder.** `test/.../fixtures.ts openHelix` is
**not** the Python helix either: bbox dims `[0.667, 0.667, 1.0]` against the Python curve's `[1, 1, 0.6]`
(MEASURED by row 10's `shape_check.ts`). Since item 10 names the arc and the helix precisely because they
are the two **measured extremes** of the 4 506× prefactor, a probe run on `openHelix` cannot connect to
that measurement and does not satisfy the row. **Ruling (2026-10-12): row 10 builds scratch-local
`arc(n)` and `helix(n)` matching the Python geometry**, descriptor-checked before either carries a
predicted wall; `openHelix` is kept as a separate coverage shape with no predicted wall. The TS corpus
has no arc at all and no matching helix, so the alternative was to leave item 10 unanswerable.

**Binding:** no number may be quoted for "the trefoil" or "the helix" without naming which curve. Of the
four shapes in item 8's table, **only the circle has a TS fixture that matches its Python counterpart** —
the arc has none, and the trefoil and helix each have a same-named TS curve with different geometry. Any
future task transferring a per-shape figure from the Python probes to the TS harness must run a geometry
descriptor first; three of four will not transfer without a new generator.

---

## A40 — row 10 Stage 1: the GENUINE f32 hierarchical arm is 88–1370× quieter than the dense-f32 lab recipe, which REVERSES §6's sign (2026-10-12)

**MEASURED, one cell (`torusKnot`, n = 960, θ = 0.5, 5 seeds), f32 inner / f64 outer through the whole
solve.** The A-angle between the f32 and f64 solution iterates on the **identical** block lists:

| arm | 1 − cos∠_A at `K_s` | 1 − cos∠_A at stagnation | angle |
|---|---:|---:|---|
| **genuine hierarchical F32** | **1.0e-10** | **9.5e-8** | 8e-4° / 2.5e-2° |
| dense-f32 stand-ins (C1/C2/Q1/Q2, the lab recipe) | 88–1370× larger | 2.9–61× larger | — |

**§6's figure is a DENSE-f32 number and does not transfer to the shipped operator.** Reaching the
τ = 0.9 bar by n = 2 000 would require an exponent of **28** (at `K_s`) or **19** (at stagnation) in
`1 − cos`; the measured local exponents are **3.2 and 5.1**. §6's "f32 walls at n ≈ 1 200–2 000" is
hereby scoped to the dense-f32 recipe it was measured on, and **`[A38 item 8]`'s per-shape walls are
likewise dense-f32 numbers** — they bound a *factorisation*, never this matvec (which `[A38 item 4(e)]`
already warned about; this is the measurement that confirms it).

**THE LOWER BOUND, and it cuts both ways.** At the measured stagnation exponent 5.1, six orders of
growth puts the genuine wall at `n ≈ 960 × 10^(6/5.1) ≈ 14 400`. **CORRECTED 2026-10-12 by `[A42]`: the
stagnation statistic has a 0.44–1.91× NULL-REALISATION spread, so a two-point exponent over a 4× n-range
carries ≈ ±1.0, and the bound is a RANGE — `n* ≈ 9 300 … 28 300`, not 14 400.** The conclusions survive
the widening and are stated against the range: the genuine f32 arm **clears the dense path's hard ceiling
`N ≈ 5 793` (`GPU_SOLVE_MAX_N`) at every point of the range** — f32 is safe across the entire
dense-reachable regime — while the whole range lies **at or below the project's 30 000 target**, so f32 is
**NOT established as safe at 30k** (at N = 30 000 the central extrapolation gives `1 − cos ≈ 4`). Still a
two-point extrapolation, which on this branch is the class of claim that has failed most often
(`[A39 item 1]`, 9c's helix ladder).
>
> ⚠ **CORRECTED AGAIN 2026-10-12 by `[A44]`, and the clause that failed is "at every point of the
> range".** That range (9 300 … 28 300) is **torusKnot's**, and row 10's 3-rung ladder found the
> **circle's** stagnation bound reaches **5 356 at `p_worst+1` — BELOW the 5 793 ceiling.** So "clears the
> ceiling at every point of the range" holds for torusKnot and **fails for the circle at that width.**
> It is not repaired by more seeds and must not be: the circle's own stagnation spread at n = 1 920 is
> **33×** on a single seed, so **the stagnation reading cannot support a ceiling claim for any shape** —
> which is `[A42] §3`'s own ruling (bound, never an instrument) applied to a number I then published
> anyway. **The ceiling claim is withdrawn at the stagnation reading entirely** and survives only at
> `K_s`, where `[A44]` puts the bar at ≥ 5.5e4 on every shape.

### The mechanism, and the half of it that is reassociation-proof

First order: `e = A_σ^{-1}(A_arm − A_bct) x`, which reproduces all three arms' angles within 1.3×. Apply
noise on the solution vector is comparable across arms; `A_σ^{-1}` amplifies the **dense** arms' error
~80× more, because the genuine apply keeps `A·1 = 0` **exactly** and so puts no weight on the near-null
constant mode that `A_σ^{-1}` amplifies. On random vectors the genuine apply is 5.5× *noisier*
(1.05e-6 vs 1.9e-7) — **the advantage is specific to the solution vector**, which is why §6's
random-vector "UNSAFE" caveat read the sign backwards for this quantity.

**The two terms of Eq. 31 do NOT have equally safe zeros** (controller-verified against `apply.ts:6-8`):

- **High-order `Mᵀ(diag(K1) − K) M`: `M·1 = 0` is EXACT in any precision and REASSOCIATION-PROOF.**
  `(M u)_I = ((u_i2 − u_i1)/l_I) T_I`, and `u_i1 == u_i2` ⇒ the subtraction is exactly 0 in IEEE754
  regardless of width, rounding mode or op order. No WGSL reassociation can break this half.
- **Low-order `Eᵀ(diag(K⁰1) − K⁰) E`: the zero is CONDITIONAL.** `(E·1)_I = 1`, so it needs
  `diag(K⁰1)·1` and `K⁰·1` to agree, i.e. identical summation order in the diagonal and the matvec.
  **This is the only half WGSL reassociation (gpuweb#2076) threatens**, which bounds the GPU risk to the
  low-order term's share of the operator rather than leaving it open-ended.

> ⚠ **THE MECHANISM IS UNFALSIFIED AS OF THIS AMENDMENT.** The first-order model measures
> `(A_arm − A_bct)x` and numerically applies `A_σ^{-1}`, so it is a *decomposition of the observation,
> not a prediction* — it cannot discriminate between mechanisms. The falsifier has been commissioned and
> has not run: **a mutant that breaks `A·1 = 0` inside the genuine apply** (perturb the low-order
> diagonal so `diag(K⁰1) ≠ K⁰·1`, which is also the realistic GPU failure), with the prediction stated
> first that the genuine arm's angle must rise toward the dense arms'. **If it does not rise, this entire
> amendment's mechanism is withdrawn, not weakened** — the measured 88–1370× would stand as an
> unexplained observation. Pending alongside it: the per-term decomposition, and exclusion of the
> competing hypothesis that the kernel chain is not genuinely f32 throughout — `apply.test.ts:664-692`
> documents itself as **BLIND to kernel-chain rounding** (one `fround` on an otherwise-f64 chain moved
> `delta_precision` by −1.2 % and the test still passed), so the matching δ of 1.0515067e-6 is consistent
> with both. The rounding-placement PINs (`nearField.test.ts:262-303`, `apply.test.ts:736-781`, run green
> at Stage 1), **not** the δ band, are what bear on that hypothesis.
> **§6's "UNSAFE" caveat is NOT struck until the mutant runs.**

### Rulings carried (controller, 2026-10-12)

1. **The genuine arm is the decision arm; Q2 is the pessimistic bracket, NOT the decision arm.** Row 10's
   charge is "locate the wall, **do not inherit it**", and the shipped path *is* the hierarchical apply —
   adopting a dense-f32 stand-in as the decision arm would re-inherit §6's number through a side door.
   Q2's role is to **price the structural zero**. Named trigger for revisiting: mechanism confirmed AND a
   GPU-side `A·1 = 0` check fails — both outside row 10, the second is Phase 3.
2. **The wall is OPERATING-POINT DEPENDENT** — 100× in level and 3.2 vs 5.1 in exponent between `K_s` and
   stagnation. The pre-registered both-reported branch fires; **stagnation is primary** (conservative, and
   it is the reading a solver user experiences). No single wall number for the genuine arm either.
3. **Stage 2 bounds the wall rather than locating it** — locating costs hours for a number no reachable
   configuration can use. Shape dependence (`[A38 item 10]`'s question) is answered by `1 − W` **levels
   and local exponents at fixed n**, one cell per shape, with no bracket and no extrapolation.
4. **No wall, bound or exponent from two points.** Local exponents per adjacent rung pair in 9c's ladder
   format, never a single fit; if adjacent locals disagree by >15 % the quantity is not asymptotic and the
   bound is quoted from the **worst** local exponent.
5. **G4 (`adjacentSkipped` equality) is VACUOUS, not PASS** — `3m` at every θ is structurally determined,
   so the gate cannot fail and a passing read is not evidence (`[A38 item 6]`; memory
   `gates-name-their-impossible-values`). No mutant is to be built for it.
6. **The `uOp` mislabel asymmetry of `[A38 item 11]` is now MEASURED**, not argued: an f32 operator
   labelled `U_F64` reads `η_bct_min/uOp = 1.5e8` and **INVALID clause 2 stays silent**, while the
   two-sided band catches it. Recorded on #92.

---

## A41 — A38 item 10's binding requirement is UNEXECUTABLE: the arc and the helix cannot be solved in this repo at all (2026-10-12)

`[A38 item 10]` requires row 10 to **"locate the wall on at least the ARC and the HELIX"**, chosen because
they are the two measured extremes of the 4 506× conditioning prefactor. **Both are open chains, and no
open chain can go through left-preconditioned GMRES here.** Adjudicated in writing rather than dropped.

**Five independent barriers (controller-verified at every site):**
1. `prepareLowOrder` throws `CurveTopologyError` — `lowOrder.ts:426-431`: *"vertex N starts no edge, so it
   has no slot to collocate on; the 'vertex' variant needs a closed curve (|E| = |V|)"*.
2. It is **deliberate**, not an oversight — `lowOrder.ts:61-65`: the vertex variant collocates
   `sE[2·slot]`, a bijection *exactly* on a closed curve; an open chain has `m = n − 1` and no slot for the
   last vertex, so it throws "rather than silently dropping a vertex." Pinned by `lowOrder.test.ts:719-733`.
3. **`laplacian.ts` restricts the preconditioner's topology anyway** (`lowOrder.ts:65`), so lifting
   barrier 1 alone would not suffice.
4. The only open-chain-capable variant, the **rs-identity knock-out**, is refused by `makePreconditioner`
   (`precond.ts:306-313`) for a sound reason: *nullity 2 for even n ⇒ `Ã` singular on the checkerboard ⇒
   `Ã⁻¹` stops being a preconditioner*.
5. `makeHierarchicalSaddle` asserts a closed natural ring (`hierarchicalSaddle.ts:284`).

Unpreconditioned runs do not converge at the n of interest. **Filed as utof/repulsive-test2#93.**

### Ruling

- **Open chains are OUT OF THE PRODUCTION DOMAIN at solve level.** Phase 2H's domain is the closed natural
  ring; nothing shipped needs an open chain. This is a research-capability gap, not a defect.
- **NO scratch-local open-chain preconditioner.** Row 10 proposed it and it is DECLINED: it puts
  uncharacterised new numerics in front of the question being measured, so the f32 result would stop being
  interpretable. If #93 is ever built it needs its own characterisation (spectrum, nullity, κ ladder)
  before any f32 figure is read through it.
- **No closed substitute reproduces the extremes, so none is adopted.** Among closed shapes the reachable
  `cond₂` spread is ~14× (circle 1.75e7 vs trefoil 1.23e6 at n=960) against arc/helix's 4 506×. A fourth
  closed shape (e.g. the (2,5) `cinquefoil`) adds geometry but not range; optional coverage at most.
- **The arc's and helix's cond-proxy walls (394, 4 907) now carry a NARROWER status than `[A38 item 8]`
  gave them: they are unpreconditioned dense-`A_σ` FACTORISATION numbers, for shapes that cannot be solved
  in this repo at all.** Per `[A40]` they were already not the genuine arm's walls; per #93 they are also
  unreachable. They remain valid as conditioning data and as the input to item 10's spread, and must be
  quoted with that scope attached.
- **`[A38 item 10]`'s "must locate the wall on the arc and the helix" is SUPERSEDED**, in two parts: by
  `[A40]` (the per-shape walls are a dense-f32 factorisation concept that does not transfer to the shipped
  matvec) and by this item (the shapes are unsolvable). **Item 10's underlying question — is the f32
  ceiling geometry-dependent? — survives and is answered on closed shapes at solve level and on ALL FIVE
  shapes at apply level** (next item).

### The open shapes ARE available at apply level, and that is where the mechanism question belongs

Row 10 measured, on arc, helix and `openHelix`: `A·1 = 0` **exactly in both F64 and F32**, `A_σ·1 = σ`
exactly, symmetry ~1e-16. No preconditioner is needed for an apply-level differential, so all five shapes
are measurable there — and `[A40]`'s mechanism, now the branch's most load-bearing claim, needs no solve.

> ⚠ **`[A40]`'s MECHANISM MAY NAME THE WRONG MODE — including in my own wording, and this is now the first
> thing to measure.** A40 says the genuine apply "puts no weight on the near-null constant mode that
> `A_σ^{-1}` amplifies." But `A_σ` **pins** the constant mode at `σ` (that is what `A_σ·1 = σ` means; σ ≈
> 1.9e4 at trefoil n=960), while `A_σ`'s smallest eigenvalue belongs to the lowest *non-constant* smooth
> mode. So `A_σ^{-1}` **damps** the constant mode by `1/σ` and amplifies smooth low-frequency modes by
> `1/λ₂` — and excluding the constant mode would buy approximately NOTHING, which cannot explain 88–1370×.
>
> **Competing hypothesis, pre-registered (controller, not yet measured):** the genuine error is `d = Mᵀ r`,
> a discrete *divergence* on the outside. `Mᵀ` differentiates, suppressing low-wavenumber content —
> precisely what `A_σ^{-1}` amplifies — while the dense arm's error is broadband. `M·1 = 0` is then the
> `k = 0` special case of a suppression acting across all low `k`, which *can* explain a factor of hundreds.
>
> **Discriminating measurement, commissioned:** resolve `d_arm = (A_arm − A_bct64)x` by wavenumber (DFT on
> closed curves, smooth-basis projections on open ones), both arms, five shapes. **If the genuine spectrum
> is suppressed at low `k` by ~one power of `k`, the divergence account holds; if both spectra are flat at
> low `k` and differ only at `k = 0`, the constant-mode account holds; if neither pattern appears, A40's
> mechanism is WITHDRAWN.** `⟨1,d⟩/(‖1‖‖d‖)` is reported per arm as the direct constant-mode test.
> **Until this returns, A40's measured 88–1370× stands and its EXPLANATION does not.**

---

## A42 — the mechanism SURVIVED its falsifier, the constant-mode account is REFUTED, and both of my explanations were wrong in different ways (2026-10-12)

Row 10 ran `[A40]`'s commissioned mutant and the per-term decomposition. All figures `torusKnot` n=960,
seeds 0–4, A-angle to the f64 solution, worst-of-5.

### 1. The mutant: PASSED, with a dose-response and a prediction fixed before the run

A ulps-per-slot perturbation of `k1Low` inside the real `applyA(H32)`, with the predicted angle
`hypot(GEN32, C2 · null-defect ratio)` — **coefficient taken from the dense-cast arm C2, not fitted.**

| dose | angle at `K_s` | measured / predicted |
|---|---|---|
| 10³ ulps | 4.4e-3 … 7.9e-3° | 0.60–0.92, **5/5 seeds** (band ⅓–3) |
| 10⁴ ulps | 4.3e-2 … 7.9e-2° | same |
| 10⁵ ulps | 0.44 … 0.80° | same |

Log-log slope **0.996–1.006** (band 0.7–1.3). A realistic 10-ulp mismatch gives 1.00–1.02× the genuine
value — **invisible**; matching the dense cast would take ≈ 6 450 ulps.

### 2. It is SPECTRAL LOCATION, not error size — and that REFUTES the constant-mode account

At a = 10³ the apply error **on the solution vector is unchanged** (`dpX` 5.66e-4 vs 5.60e-4) while the
angle rises **7–10×**. Angle per unit `dpX` moves 0.015–0.028 rad (genuine) → 0.14–0.24 → 0.75–1.4 →
0.9–1.8. So the same amount of error, placed differently in the spectrum, costs 10× more.

**`[A41]`'s pre-registered three-way verdict resolved as NEITHER — see `[A43]`.** The constant-mode
account is REFUTED (0/10) and my own divergence-prefilter account was **NOT CONFIRMED** by its own
registered threshold (6/10 against 7/10). ⚠ **The sentence this paragraph originally carried — "errors
entering through `Mᵀ` stay derivative-shaped and are damped; errors entering through `Eᵀ` are white and
amplified ≈ 80×" — is SUPERSEDED by `[A43]`**: that contrast names *one way of having zero row sums*, not
the property itself, and row 10 withdrew it. Read it as "zero row sums of the error operator preserved /
broken". What survives here unchanged is the **refutation**, which is **direct, not inferential**:
perturbing `diag(K1)` leaves
`A·1 = 0` **exactly** zero (because `M·1 = 0` happens first, so the perturbed diagonal never sees the
constant vector) yet it is the **more** harmful mutant. A protection that is fully intact while the damage
grows is not the protection doing the work.

> ⚠ **`[A41]`'s "the GPU risk is bounded to the low-order term's share" is WRONG and is withdrawn.**
> (Its replacement framing below is itself re-scoped by `[A43]`: the operative property is zero row sums,
> and the high/low asymmetry is about *which chain preserves them under reassociation*, not about `Mᵀ`
> versus `Eᵀ` as such.)
> Measured: a `diag(K1)` mismatch is **≈ 4× more harmful per ulp** than a `diag(K⁰1)` mismatch, even
> though the high-order constant-mode zero survives it. The two chains fail differently —
> **low-order: reassociation destroys a structural zero and buys the ≈ 80× amplification; high-order: no
> zero to lose, but ≈ 4× the per-ulp magnitude sensitivity** (the high-order term dominates the operator;
> `LO32` is only 2.8–4.5 % of `HI32`). **WGSL consistency is therefore required on BOTH diagonals**,
> not only on `diag(K⁰1)`. Qualitative risk is low-order; quantitative sensitivity is high-order; A41
> conflated them.

Additivity in squares 0.98–1.04 (band 2×) and `HI32` = 0.99–1.03× `GEN32` at `K_s`: the high-order chain
carries the angle. **`[§6]`'s "UNSAFE" is struck for the A-angle at `K_s` and stands for random-direction
apply noise** (annotated in place at `:1157`).

### 3. THE STAGNATION READING IS A DIFFERENT MECHANISM, AND IT IS NOT AN INSTRUMENT

Two of row 10's own pre-registered predictions **failed at stagnation**, and it reported both:

- `BOTH32_OUT64` (both chains f32, output accumulation f64) is **0.04–0.28× `GEN32`** — band was 2×,
  **fails 5/5 seeds**.
- Additivity in squares reaches **2.43** on one seed — band was 2×.
- **Cause, identified:** rounding the stored apply result to f32 **once** (`BOTH32_OUT1`) reproduces the
  `GEN32` range (0.26–1.58×). **The stagnation excess is the f32 STORAGE of `A·x` — the generic `κ·u`
  noise path — which the structural zero never protected.** That is why the genuine/dense gap collapses
  from 88–1370× at `K_s` to 2.9–61× at stagnation.

> ⚠ **Binding: the stagnation reading is a BOUND, never an instrument.** A null control (a = 1, nothing
> physical changed) moves it **0.44–1.91×**, so it cannot support any band tighter than ~2× and **P-M2
> cannot be tested there at all.** Consequences: (a) every stagnation figure is published **with its seed
> distribution**, never as a point; (b) **all mechanism tests and all tight-band ratios go to `K_s`**;
> (c) `[A40]`'s wall bound widens to the range in §1 above. Row 10's own correction, recorded: the
> first-order model's "≤1.3× on all three arms" holds **at `K_s` only** — at stagnation it under-predicts
> the noise-driven arms by ≈ 12×.

**ACTIONABLE DESIGN FINDING, not just a caveat:** since the stagnation excess is output *storage*,
**f32 chains with f64 output accumulation cut the stagnation error 3.5–25×** (`BOTH32_OUT64` =
0.04–0.28× `GEN32`). That is a cheap mitigation available to any consumer that can afford an f64 output
buffer, and it is the configuration a GPU path should prefer.

### 4. The kernel-chain alternative hypothesis is EXCLUDED, by a pin shown to have power

`[A40]`'s competing hypothesis (the chain not genuinely f32, exploiting
`apply.test.ts:664-692`'s documented blindness) is excluded: an independent per-op `Math.fround`
transcription over **every** near pair (86 658/kernel) and **every** far block (3 530/kernel) of the actual
`H32` gives **0 mismatches in all four sets**, and the round-once mutant disagrees on **2 488/3 530 and
3 068/3 530** far blocks — **so the pin is not vacuous.** This is the A38-item-6 standard met.

**Remaining honest gap (row 10 flagged it itself):** the accumulation chain inside `applyK` has no
**independent** per-op pin. `terms.ts` is bit-identical to the real `applyA` at n=240 and n=960, but it is
a *transcription*, so it shares the original's op order by construction — it proves faithfulness, not
correctness of the order (memory `common-mode-blindness`). The order is covered end-to-end by the
`out-accuracy.json` oracles (`apply.ts:10-16`), not by a per-op pin. **Filed as utof/repulsive-test2#94.**

### 5. Gate bookkeeping executed (`[A40]` ruling 6)

- **G3** renamed `G3.inBand_atKs_f64NotAtFloor`, with `G3.inBand_floorRuns_VALID` true 10/10. The
  `[A38 item 11]` asymmetry is now measured at two n: clause-2 ratio **1.49e8** (n=960) and **1.35e8**
  (n=240) with clause 2 **silent**, the band flagging it, and the opposite mislabel firing at 1.2e-5 (#92).
- **G4** relabelled **VACUOUS**, reason recorded, no mutant built (as ruled).
- Operating point declared dependent; stagnation primary **as a bound**, `K_s` beside it — now qualified
  by §3 above.

---

## A43 — the spectrum test returned NEITHER: MY hypothesis failed its own threshold, and the real property is ZERO ROW SUMS (2026-10-12)

Row 10 ran `[A41]`'s commissioned wavenumber decomposition on **all five shapes × n ∈ {480, 960}** (10
cells, flat test field, rules registered before running). **The registered outcome is NEITHER, and the
hypothesis that failed is mine.**

| registered rule | result |
|---|---|
| **INSTRUMENT** (`p(HIGH) − p(LOW) ≥ 0.7` on ≥7/10) | **10/10**, differences 0.80–1.24 — it genuinely separates a derivative-shaped error from a white one |
| **H_C — constant mode only** | **0/10. REFUTED.** `R = a_k(GEN32)/a_k(C2)` is **0.0009–0.05 at `k = 2…16`** and 0.0002–0.031 at `k = 1` in **every** cell: the genuine error is 20–1 000× below the dense cast at **every** low wavenumber, not only at `k = 0`. Independently, GEN32's `k = 0` amplitude sits at the same level as its `k = 1…8` amplitude in every cell — **`k = 0` is not special.** |
| **H_L — suppression across low `k` (the CONTROLLER's hypothesis)** | **6/10 against a registered 7/10. NOT CONFIRMED.** All four misses fail on the **slope criterion alone** (torusKnot 960 at 0.36, trefoil 480 at 0.22, helix 480 at **0.48 — 0.02 under the window**, helix 960 at −0.23); `R(1) ≤ 0.3` and `|p(C2)| ≤ 0.5` hold **10/10**. |
| **P-S3** (a gap exists on every shape, `amp(C2)/amp(GEN32) ≥ 10` on ≥8/10) | **10/10**, min 11.4 (helix 480) |

> ⚠ **RULING ON MY OWN HYPOTHESIS: H_L IS NOT CONFIRMED, AND I AM NOT RESCUING IT.** I wrote that
> threshold in `[A41]`, before the data. Three rescues were available and all three are refused:
> **(i)** moving the 0.5 slope window that helix 480 missed by 0.02 — forbidden, and maximally
> self-serving on my own hypothesis; **(ii)** using the post-hoc F1 result (removing output rounding
> steepens the slope to 1.06–1.45 on four shapes) to explain away the misses — **a post-hoc finding
> cannot un-fail a pre-registered test**; **(iii)** calling it "confirmed in substance" because
> `R ≪ 1` held 10/10 — that is rescue (i) in prose. **Row 10 declined to move the window and explicitly
> left the call to me; the call is NEITHER.** H_L was **directionally right and wrong about functional
> form**: the measured genuine spectrum is a *low flat floor* plus a branch rising ∝ `k` above `k ≈ 10–16`,
> not a graded power law, so a slope criterion was the right instrument and it caught the error. Anyone
> wanting H_L must register the no-output-rounding version and test it fresh.

### What the property actually is — post-hoc, but with its own confound control

**The error operator has exactly zero row sums, so it acts on DIFFERENCES of `x`.** `A·1 = 0` is the
`k = 0` shadow of a statement about every row, and **the `Mᵀ…M` factorisation is NOT necessary**:

- **F2 (registered before running): a dense f32 cast with each row repaired in f64 to sum to zero (C2R)
  reproduces the GENUINE arm's amplification — 0.79–1.03×, 10/10** — and is 0.001–0.089× C2's.
- **F3 (a control row 10 added after spotting a confound in F2, registered before running): the same cast
  with the diagonal kept exact but row sums NOT zero (C2D) stays at the dense level, 0.62–1.38× C2,
  10/10**, slope −0.16…0.04 (white). **So it is the zero row sums, not the unrounded diagonal.**

For the genuine arm the property holds **by construction** in the high-order chain (`M·1 = 0`,
reassociation-proof) and **conditionally** in the low-order chain (while `diag(K⁰1)` and `K⁰·1` agree).
`[A42]`'s "`Mᵀ` versus `Eᵀ`" contrast is **superseded**: it names one way of having zero row sums, not the
property. Row 10 withdrew two of its own earlier sentences on the same ground.

**Not shown (row 10 states it):** that the genuine arm's *own rounding* errors are suppressed for this
reason rather than for something specific to the hierarchical kernels — (3) is demonstrated on a dense
probe and on the mutants. **The helix's spectrum is flat even without output rounding and is unexplained.**

### A38 item 10's SHAPE-DEPENDENCE QUESTION IS ANSWERED, and the answer inverts it

`amp = relA/dpX` at n = 960, across all five shapes:

| | spread | ordering |
|---|---|---|
| **genuine F32** | **0.034–0.045 (1.3×) — SHAPE-INDEPENDENT** | circle .0445, torusKnot .0444, trefoil .0419, arc .0424, helix .0341 |
| dense f32 cast | **0.65–49 (75×) — tracks CONDITIONING** | helix .65, trefoil 1.4, torusKnot 1.9, circle 7.1, **arc 49** |

**The genuine error's cost does not track conditioning at all.** So the genuine/dense gap is **largest
exactly where the dense cond-proxy wall is lowest** (arc: 430–1 163×, wall 394) and **smallest where it is
highest** (helix: 11–19×, wall 4 907). Three statements are all true and are recorded together:

1. **`[A38 item 10]`'s criticism was CORRECT about the dense route** — per-shape walls really do span
   4 506× in prefactor there, and quoting one fixture's was wrong.
2. **`[§7.11]`'s geometry-independence instinct is right for the SHIPPED operator** — but it was
   unjustified, because it was measured on the route where amplification *is* geometry-dependent.
3. **Item 10's deliverable is met in a different currency than it asked for**: not a per-shape wall
   (unexecutable, `[A41]`/#93) but a per-shape *error-structure* measurement, on all five shapes
   including the two that cannot be solved. Item 10 is **CLOSED**.

Arc and helix figures keep their standing scope: unpreconditioned dense-`A_σ` factorisation numbers for
shapes that cannot be solved in this repo at all (#93), the helix additionally resting on exponents still
rising at the last rung.

### The θ = 0 σ check (controller-required) CONFIRMS compression and proves something stronger

The σ gap collapses to **≤ 1.5e-14 on all 8 (shape, n) cells** at θ = 0, against **−0.8 % … −4.1 %** at
θ = 0.5. So the 1–4 % gaps are BCT compression of the diagonal, as asserted — **and, beyond the asserted
claim, the TS operator on the TS generators IS the Python dense `A` to round-off.** That licenses
transferring figures between the Python probes and the TS harness, which `[A39 item 3]`'s
fixture-provenance mess had put in doubt. The generators are also bit-for-bit against `mhat_lib`
(max vertex difference 0.0).

### Process: row 10 self-reported an ordering erratum, which is reference behaviour

F1's "prediction" was written **after** its arms had already run. Row 10 caught this, labelled F1 "a
post-hoc description in prediction clothing" carrying **no falsification weight**, and — where F1's own
criterion was ambiguous between "at `k = 1`" and "at every `k ≤ 4`" — **reported both readings rather than
choosing the favourable one.** F2 and F3 were each registered before running. This is the standard the
branch should be held to, and it is why F1 is not permitted to rescue H_L above.

---

## A44 — row 10's closed-shape ladder: f32 ARITHMETIC clears 30k by 2–20×, and the binding constraint is OUTPUT STORAGE (2026-10-12)

3 shapes × n ∈ {480, 960, 1 920}, genuine F32 arm, Q2/C2 dense stand-ins on `torusKnot`, all
pre-registered rows met. **Row 10 is answered. The two operating points give OPPOSITE answers about the
30k target, and the reason is now known, so the disagreement is a result rather than an ambiguity.**

### 1. `K_s` (isolates the ARITHMETIC): f32 clears 30 000 with margin

| shape | local exponents of `1 − cos` | agreement |
|---|---|---|
| torusKnot | **3.30, 3.44** (Stage 1's two-point value was 3.2) | 4 % |
| trefoil | 2.95, 3.16 | 7 % |
| circle | 3.22, 3.49 | 8 % |

All three pass the <15 % rule of `[A42]`'s ruling 5, so for the **first time on this branch a wall-class
exponent is ASYMPTOTIC rather than a two-point fit.** Per-seed locals scatter 2.75–3.85, so a single local
carries ≈ ±0.5 — quoted, not hidden.

**The 0.1 bar sits at or above 5.5e4 on every shape even at `p_worst+1`**; at `p_worst` it is 3.9e5
(torusKnot), 6.4e5 (trefoil), 1.4e5 (circle). **At N = 30 000, `1 − cos ≤ 6.6e-3`** — a factor ~15 inside
the bar. The f32 error also stays **below the compression error** up to n ≈ 3.4e4 (circle) … 1.6e5
(trefoil), i.e. across the whole target range f32 is not the dominant error. `E_c/E_p` at n=960 is
**500–4 150**.

### 2. Stagnation (bound-only): every shape crosses below 30 000, and it is a DIFFERENT constraint

Bound **6.4e3 … 2.4e4** across shapes (torusKnot 9.8k/19.5k, trefoil 10.4k/20.1k, circle 6.4k/23.7k at
`p_worst`/`p_min`). **Stagnation exponents are uninformative — per-seed locals run −1.7 … 9.4**, which is
`[A42] §3`'s non-instrument ruling reconfirmed on a ladder. `n*(K_s)/n*(∞)` is **22–62** at `p_worst` and
≥ 8.6 in every pairing.

> **THE SYNTHESIS, and it is the actionable conclusion of row 10.** `K_s` isolates the f32 **arithmetic**;
> stagnation is dominated by the f32 **storage** of `A·x` (`[A42] §3`, identified and reproduced there).
> The two readings therefore do not disagree about the operator — they measure different things, and
> **the binding constraint on f32 at 30k is output-storage precision, NOT hierarchical f32 arithmetic.**
> `[A42]` already measured the mitigation: **f32 chains with f64 output accumulation cut the stagnation
> error 3.5–25×** (one cell). **So the supported design statement is: run the chains in f32, accumulate the
> output in f64.** ⚠ The mitigation is measured at ONE cell and has NOT been laddered, so this is
> conditional until the `OUT64` ratio is measured across rungs — commissioned as the only remaining
> measurement row 10 is authorised to take.

### 3. The genuine/dense gap GROWS with n at `K_s`, and is NOT a power law

Median dense/genuine angle ratio across the three rungs: **Q2 37 → 108 → 245** (locals 1.54, 1.19),
**C2 18.1 → 76.6 → 121** (locals 2.09, 0.66). **The locals decelerate and disagree by 29 % and 216 %, so
the gap is NOT extrapolated** — correctly refused rather than fitted. The dense arms' own exponents
decelerate too (C2 8.06 → 4.21, Q2 6.63 → 5.18). Dense walls at `K_s`: **5.6k–15k against ≥ 1.2e5 for the
genuine arm**; at stagnation Q2 is 2.4k–3.2k against torusKnot's 7.9k–19.5k. The `DEN64` densifier control
is within **4.7e-7** of the genuine angle, so the comparison is not a densification artefact.

### 4. P1 / P2: formally NEITHER, with the descriptive reading on P1's side

No bracket exists and the absolute reading is uncalibrated, so `[§2.4]`'s verdict is **NEITHER** — the
pre-registered outcome, and the expected one once `[A40]` scoped the band to the dense recipe.
**Descriptively, at `K_s`, geometry dependence is real and in P1's direction**: circle/trefoil angle ratio
4.2, 4.6, 5.2 (P1's own number was 3.5) and extrapolated `n*_c/n*_t` **0.21–0.22** against P1's `< 0.6`
threshold — valid to quote because the exponents agree within 10 %. At stagnation it is uninformative
(0.61–1.18). **Reported as descriptive, not as a verdict.**

**This closes a loop with `[A43]` cleanly.** A43 measured `A_σ⁻¹` amplification as **shape-independent**;
the ladder finds the cross-shape angle differences live in **`dpX`** instead — the circle's `dpX` is 10–19×
the other shapes' while its amplification is not. Together: **geometry affects the f32 error through the
APPLY-ERROR MAGNITUDE, not through conditioning-driven amplification.** Mechanically, the `K_s` angle is
the f32 apply error on the smooth solution growing ≈ `n^2.6`, partly offset by amplification falling
≈ `n^-0.9`.

### 5. Gates

`L-PL` and `L-KS128` met; `K_s` is 41–52 in every cell with none coarse or censored; **G1, G2a, G3 (34/34
seeds), G6 (0 prefix mismatches)** and the bitwise f32 re-run (**8/8 cells**) all pass. `G4` remains
**VACUOUS** per `[A42]`'s ruling.

---

## A45 — row 10 CLOSED: the f64-output mitigation STRENGTHENS with n, and the same WGSL clause that creates the hazard blocks its GPU form (2026-10-12)

`[A44] §2`'s commissioned measurement, run as a **ratio at fixed n** (never a wall or an exponent, since
stagnation per-seed locals run −1.7…9.4). `ρ = OUT64/GEN32` stagnation angle, torusKnot, median seeds 0–2:

| n | ρ | improvement |
|---:|---:|---:|
| 480 | 0.154 | 6.5× |
| 960 | 0.094 | 10.7× |
| 1 920 | **0.044** | **23×** |

**Pre-registered outcome: STRENGTHENS** (registered rule ρ(1920) ≤ 0.5·ρ(480); measured **0.28**). It does
**not** decay toward 1, so `[A44] §2` needs no rescoping. **And the direction does not rest on a median of
three**: the per-seed range at 1 920 (0.023–0.067) sits **entirely below** the 480 range's minimum (0.093).

**The control that matters most: with an f64 output accumulator the stagnation reading COLLAPSES ONTO
`K_s`** — OUT64's stagnation angle sits at its own `K_s` level (ratio 1.07–1.24, max 1.33) while leaving
the `K_s` angle itself unchanged (0.97–1.01×). Since `[A44] §1` puts the `K_s` bar at ≥5.5e4 on every
shape, **the two operating points stop disagreeing once the output is accumulated in f64.** OUT1 reproduces
the GEN32 excess to within 1.6–2.0×, so the store is *most* of the excess but not all — stated as measured,
not rounded to "all".

> **ESTABLISHED:** *if an f64-grade output accumulator exists, stagnation collapses onto `K_s` and f32
> clears 30 000.* The arithmetic-model conditionality in `[A44] §2` is discharged.

### ⚠ The GPU form of this mitigation is blocked by the SAME clause that created the hazard

**Portable WGSL has no native f64**, so a GPU accumulator must be emulated (df64 / compensated summation).
The repo's own prior research already settled the portability, and more sharply than row 10 stated it —
`docs/2026-08-13-ai-research-gpu-precision.md` **Q3**, cited in the Phase-2 design spec's source list as
`[PREC Q3]`: *"compensated arithmetic is NOT portable in WGSL — §15.7.5 permits reassociation; df64 ≈ 4×
cost."*

**The coupling is the point, and it is not a coincidence.** Compensated summation (Dekker/Knuth two-sum)
is correct only because specific intermediate roundings happen at specific points. **WGSL §15.7.5's
reassociation permission is the very same clause `[A42]` identified as the threat to the low-order
structural zero.** So one WGSL licence simultaneously (a) creates the precision hazard and (b) invalidates
the standard software fix for it. A GPU port cannot assume either the structural zero or the emulated
accumulator survives; both need measuring on the real kernel, and at ≈4× cost the accumulator is not free
even where it works.

**Scope of the established statement, both limits recorded:**
1. **Measured on torusKnot only**, as instructed. The **circle** is the shape where the stagnation
   precision angle comes within **7–17×** of the compression angle (`[A44]`), and it is **unmeasured** —
   the first place to look if the mitigation is ever load-bearing.
2. **Phase 3 must re-test this on the real WGSL kernel** with the same OUT64-vs-shipped ratio. A
   transcription cannot answer it — that is exactly **#94**'s gap (a transcription shares the op order by
   construction, so it cannot detect a reassociation the compiler is permitted to make).

### Stage 3 applied, and verified by the controller

`src/` carries **two comment-only edits**: `bct.ts` +13/−1 (a `Why:` on `traverseBct`, a pointer in
`buildBct`) and `gmres.ts` +8 (the `exactApplyA` TSDoc). **Controller-verified: `git diff -U0 -- src/`
contains ZERO non-comment added or removed lines**, 20 insertions / 1 deletion across the two files.
Biome format and lint clean; `bct.test.ts` + `gmres.test.ts` **109 pass / 0 fail**. Both cite spec
amendments (`[A44] §5`, `[A42] §5`, `[A44] §1`) rather than task numbers, per the repo's anchoring rule.

### ROW 10 IS CLOSED

Delivered: the genuine f32 hierarchical apply through the whole solve (`[A40]`), its mechanism with a
surviving falsifier and a refuted alternative (`[A42]`, `[A43]`), the per-shape question closed in a
different currency than it was asked in (`[A43]`, `[A41]`/#93), an asymptotic `K_s` exponent ladder
(`[A44]`), and a discharged design statement with its GPU limit named (this item). P1/P2: **formally
NEITHER**, descriptive reading on P1's side. Issues opened: **#92, #93, #94**.
