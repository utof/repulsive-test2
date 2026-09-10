# WebGPU Solver — Phase 2 Gate Calibration Note

**What this document owns.** Every *measured number and every band* that a
Phase 2 gate is priced against: the phase partition of a CPU descent step, the
browser-scale factors, the G5 (a)-vs-(b) arithmetic, K1's α/β calibration,
CR's bar derivation and Boot's chosen bar, P2a's term-by-term prediction
and its noise analysis, and `G5-confirm`'s pass/fail levels.

**What it does NOT own.** Any structural or algorithmic decision. The algebra,
the kernel design, the core seam, the driver flow, the gate *definitions* and
their pass semantics, the kill-gate branches, the risks and the open questions
all live in the design this calibrates:
**`docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md`** (cited
below as **[DESIGN §n]**). Where the two disagree about a *number*, this note
wins; where they disagree about a *decision*, the design wins and this note is
wrong and must be corrected.

**Why the split exists.** Three independent blind review rounds each found a
different instance of one defect: gate arithmetic hand-transcribed into design
prose, where nothing re-checks it. Patching the instances leaves the generator
intact. This note is the generator's replacement.

> **Binding rule.** *No gate number appears in prose unless a committed script
> emitted it into a committed JSON.* [DESIGN] cites keys; this note quotes
> values and always names the file and the key. A level that no committed
> emitter produces is not a bar — it is a **formula plus a pre-registered
> precondition** (§J).

---

## A. Provenance, and the three tiers a number can be in

| tag | emitter → committed artifact |
|---|---|
| **[BASE]** | `bench/sobolev.bench.ts` → `bench/results/2026-08-13-cpu-baseline-largeN.json` (`bunVersion: "1.3.11"`) |
| **[K1CAL]** | `bench/gpu/k1-calibration.py` → `bench/results/2026-09-04-gpu-phase2a-k1-calibration.json` |
| **[JSALG]** | `bench/gpu/jsAlgebra.ts` → `bench/results/2026-09-04-js-f64-algebra.json` (`"tool": "bun 1.3.11"`) |
| **[P1]** | `bench/gpu/phase1.ts` via `bench/gpu/drive.ts` → `bench/results/2026-09-03-gpu-phase1-de-gate.json` |
| **[G6]** | `bench/results/2026-08-13-gpu-phase0-g6.json` |
| **[G0t]** | `spikes.g0t` (`bench/gpu/spikes.ts:131`) via `bench/gpu/drive.ts:423` → `bench/results/2026-08-13-gpu-g0t.json` (`gitShaShort: "4fec523"`; narrative `bench/gpu/README.md:121-149`) |
| **[G1]** | `spikes.g1` (`bench/gpu/spikes.ts:253`) via `bench/gpu/drive.ts:423` → `bench/results/2026-08-13-gpu-g1.json` (`gitShaShort: "5b35746"`; narrative `bench/gpu/README.md:151-175`) |

**Tier 1 — emitted.** A value that exists as a key in one of those files. Every
Tier-1 quote below carries its key. A Tier-1 value may be a bar.

**Tier 2 — derived.** Arithmetic over Tier-1 values, performed here, with the
expression written out so a reader can re-check it without leaving the page.
Tier-2 values are quoted **in this note only** and never in [DESIGN].

**Tier 3 — bars.** A gate's pass/fail level. It must be Tier 1, or a **formula
over Tier-1 keys** whose evaluation is owed to a committed emitter (§J).
`CR`, `K1`, `T4`, `Boot` are Tier 1 today. `P2a` and `G5-confirm` are not
(§H, §I), which is why both carry a precondition.

A note on units, because it is the defect that produced `G5-confirm`'s false
kill: **[BASE] and [JSALG] are Bun timings; every GPU-inclusive measurement is
necessarily a Chrome timing**, because `bench/gpu/drive.ts:1-6,20` launches
`google-chrome` against the dev server for every GPU spike and there is no
WebGPU outside the browser in this repo. A ms-vs-ms comparison across that
boundary is a comparison of JIT ratios. Every gate here is therefore stated as
a **ratio measured inside one session**, never as a wall-clock level.

---

## B. Where a step's time goes

**The phase keys are NOT a partition.** `src/core/sobolev/phaseTimings.ts:1-7`
says so in its own schema note ("Sub-phases … OVERLAP their parents — sums
across keys double-count by design"). Verified nesting at HEAD:
`step ⊃ {dE, assembleA, saddle(gradient), lineSearch}`;
`lineSearch ⊃ {energy, projection}` (`optimizer.ts:399`, `lineSearch.ts:447,
513`); `projection ⊃ saddle(projection solve)` (`lineSearch.ts:260`);
`saddle ⊃ factor` (`gradient.ts:106` → `linsolve.ts:796`). Speedup arithmetic
done on the raw keys double-counts `projection` and drops `saddle − factor`;
the first draft of the design did exactly that, which is why its P2a
predictions came out ABOVE the Amdahl ceiling.

**Tier 1 — [BASE] raw keys**, frozen mode, total-length, LDLᵀ, Bun (ms, one
step; `cases[].phases.<key>.ms` for `constraintMode: "total"`; each key is an
independent median over K=5 steps, so the keys need not sum to `step`):

| N | step | dE | assembleA | saddle (2 calls) | factor | lineSearch | energy | projection |
|---|---|---|---|---|---|---|---|---|
| 480 | 1064.317 | 142.777 | 85.941 | 728.993 | 700.162 | 59.040 | 46.163 | 13.175 |
| 960 | 7194.500 | 553.503 | 355.442 | 6186.043 | 6039.673 | 143.800 | 91.572 | 53.174 |
| 1000 | 7816.380 | 570.655 | 372.971 | 6771.142 | 6640.457 | 160.970 | 97.205 | 54.077 |

`phases.saddle.calls = 2` and `phases.energy.calls = 1` at every N ⇒ **one
gradient solve, one projection correction, one Armijo trial** on these
fixtures (the harness supplies `energyBefore`, `bench/sobolev.bench.ts:152-157`,
so the E₀ eval never fires). That is the per-step solve count the whole cost
model uses; the 8×8 worst case (`maxIter = 8` in `projectOntoConstraintSet`,
`lineSearch.ts:209`, × up to 8 Armijo trials) does not occur on the gate
fixtures.

**The partition (Tier 2).** Write `s_p` for the projection solve's share of
`saddle` (one key, two call sites — not separately measurable). The disjoint
terms of a step are
`step = dE + assembleA + (saddle − s_p) + energy + s_p + (projection − s_p) + LS_other + R`
— `s_p` appears on its own because it is the projection SOLVE, which `saddle`
and `projection` both contain; subtracting it from each and not adding it back
once would drop it from the step entirely. It cancels against `(saddle − s_p)`,
so the identity collapses to
`step = dE + assembleA + saddle + energy + (projection − s_p) + LS_other + R`,
with `LS_other = lineSearch − energy − projection` (−0.3 / −0.9 / +9.7 ms at
N=480/960/1000) and `R` the unattributed remainder.

> **R₂ = step − dE − assembleA − saddle − energy** = `(projection − s_p) + LS_other + R`
> = **60.443 ms (N=480) · 7.940 ms (N=960) · 4.410 ms (N=1000)**

— all the CPU work of a step that is neither dE, nor assembly, nor the solve,
nor the energy. It does not scale with N, so it is probably median-of-medians
slack rather than work; it is carried anyway, because carrying it makes the 2a
prediction pessimistic rather than flattering.

`s_p` itself is bracketed, not measured: `s_p ≤ projection` and `R ≥ 0 ⇒
s_p ≥ projection + LS_other − R₂`, giving **s_p ∈ [44.3, 53.2] ms at N=960**
and **[0, 13.2] ms at N=480** — consistent with a 2N²-flop backsolve of the
(3N+k)-sized factor plus its structured residual.

**The `R ≥ 0` premise is itself falsified at N=1000**, and that is evidence,
not a nuisance: the same bracket there is **[59.36, 54.08] — empty by 5.28 ms**,
so `R < 0`. A negative remainder cannot be work. Each key in [BASE] is an
independent median over K=5 steps, so the keys are medians of different step
samples and need not add up; 5.28 ms of inconsistency at N=1000 is the same
order as `R₂` itself there (4.41 ms) and at N=960 (7.94 ms). This is the
concrete form of the measurement-error branch that [DESIGN §5] requires plan 2a
to rule in or out before any P2a number is read: `R₂` at N=480 (60.4 ms) is
either work or this artifact scaled up, and no published number distinguishes
them.

**Amdahl ceilings (Tier 2).** Exactly one phase leaves the CPU in 2a: `factor`.
`assembleA`, `energy` and `R₂` stay byte-for-byte, so with the GPU factor, the
GPU dE and the new CPU algebra all free the ceiling is
`step / (assembleA + energy + R₂)` = `1064.317/192.547` = **5.53×** at N=480 and
`7194.5/454.954` = **15.81×** at N=960. Costing the new algebra honestly at
[JSALG]'s measured range (§H's term table: 17.5–41.3 ms at N=480,
89.5–138.7 ms at N=960):

```
N=480  1064.317/(85.941+46.163+60.443+41.3) = 4.551      1064.317/(…+17.5) = 5.067
N=960   7194.500/(355.442+91.572+7.940+138.7) = 12.119    7194.500/(…+89.5) = 13.214
```

> **Honest ceilings: 4.55–5.07× at N=480 and 12.12–13.21× at N=960.**
> These four numbers are the ONLY ceilings. Two earlier prose sites carried
> 4.49–5.05× and 12.33–13.20× and cross-referenced sections that never
> contained them; they reproduce from nothing and are withdrawn (verdict I4).

The cruder reading `step/(step − factor − dE)` gives 4.81× / 11.96×, which is
3.3 % / 6.4 % below the honest central figures (4.97× / 12.78×) — the new Schur
algebra is **cheaper** than the in-`saddle` backsolves it replaces (108 vs
146.3 ms at N=960, 22 vs 28.8 at N=480, i.e. `saddle − factor`), so the crude
reading understates rather than agrees. This section, not that shortcut, is the
arithmetic of record.

---

## C. The browser-scale factors — what they are, and what they are not

[P1]'s `data.rows[].cpuMs` are the only browser CPU timings that exist for this
milestone. **They time `gradientAnalytical` and nothing else**:
`bench/gpu/phase1.ts:324-326` is
`const t0 = performance.now(); gradientAnalytical(vertices, edges, dp, alpha, beta, epsilon); cpuMs.push(performance.now() - t0);`
and the run's own JSON carries `data.fullStepGate: false`, whose meaning is
documented at `bench/gpu/phase1.ts:~295` — "records that this measures dE
alone, not a whole descent step".

**Tier 1 — the five committed browser CPU *dE* samples per N**
([P1] `data.rows[].cpuMs`, `data.rows[].cpuP50Ms`):

| N | five samples (ms) | p50 | within-session spread |
|---|---|---|---|
| 480 | 226.1 / 215.4 / 261.0 / 253.4 / 341.1 | **253.4** | 1.58× |
| 960 | 974.6 / 1158.7 / 977.6 / 961.9 / 911.1 | **974.6** | 1.27× |

**Tier 2 — the factors.** Against [BASE]'s Bun `dE` (142.777 / 553.503):

```
253.4 / 142.777 = 1.7748     974.6 / 553.503 = 1.7608
```

> **Browser-scale factors ×1.775 (N=480) and ×1.761 (N=960).**

**The named modelling assumption.** These are the Chrome-vs-Bun JIT ratio of
**one** phase — `gradientAnalytical`, an O(E²) pairwise loop — extrapolated to
a dense-matrix assembly, a triangular backsolve, an O(N²) matvec and an
unattributed remainder. That extrapolation may well be fine; it is *not*
measured, and nothing in this milestone measures it. Any statement that
browser-scales a CPU term is carrying this assumption (verdict I6). It is the
only thing standing between the Bun predictions of §B/§H and the browser
numbers every gate is actually read in — which is why §I and §H put both gates
on **paired same-session ratios**, where the factor cancels instead of being
applied.

**What these numbers are NOT.** They are not CPU *step* times. As step times
they would say the browser runs a whole descent step in 253.4 ms against
[BASE]'s Bun 1064.317 ms — the browser 4.2× faster than Bun — which is not
credible and contradicts the ×1.775 factor derived from the same five values
one paragraph up. A browser CPU **full-step** p50 has **never been measured**
in this repo. §H and §J carry the consequence.

---

## D. G5 — the (a)-vs-(b) arithmetic

The decision this arithmetic feeds is [DESIGN §3]'s (build variant (a),
confirmed or overturned by `G5-confirm`). [P §4 G5]'s statistic is the p50
**full-step** time with the line-search projection solves included; both arms
are priced on it, and the solve-path reading is given too because the parent's
words admit both.

### D.1 Variant (a) — estimate, N=960, total mode

- Cholesky of the shifted N×N block: `n³/3 = 2.95e8` flops.
- Unblocked right-looking ([DESIGN §4.3]): **three** kernels per column ⇒
  **3N = 2880** dispatches in one `renderer.compute([...])` batch. [G1]
  `data.batchedMs = 0.300` over `N = 250` batched dispatches
  (`bench/gpu/spikes.ts:258`) is a CPU-side **submission** wall only
  (`bench/gpu/README.md:156-158`); at 5–15 µs GPU-side per small dispatch, 2880
  of them are **14–43 ms**.
  **`5–15 µs` is an UNSOURCED ESTIMATE.** It is in none of §A's three tiers: no
  committed artifact in this repo measures GPU-side execution of a small
  dispatch. [G1] times CPU submission of *no-op* nodes and so neither is nor
  bounds it; [G0t] has no per-dispatch key
  (`fmaGflops, fmaTotalsMs, matvecMs, matvecGflops, matvecTotalsMs, method`);
  and `docs/2026-08-13-ai-research-webgpu-compute.md:63-76`'s per-dispatch table
  is CPU-side overhead on other hardware. It is labelled rather than deleted
  because deleting it deletes the dispatch term altogether — but it is **not a
  bar, and no gate may be read off it.** **What it propagates into:** this
  14–43 ms dispatch term, hence the 54–101 ms factor wall below, hence (a)'s
  bands in §D.1/§D.3, [DESIGN §5]'s declared `≤ 100 ms` factor-wall target, and
  §I.2's sanity check. Plan 2a's Cholesky spike measures the wall directly and
  is what retires this estimate; until it runs, every number downstream inherits
  its uncertainty.
- The trailing update is **bandwidth**-bound, not flop-bound — 0.125–0.17
  flop/byte against the matvec's 0.5 — so [G0t]'s 40 GFLOP/s does not transfer:
  traffic `2·(N³/6)·4 B = 1.18 GB` at N=960 against the 81.5 GB/s that [G0t]'s
  own matvec implies (3072²·4 B in 0.463 ms) → **14.5 ms**. Because the
  shared-node scheme fixes the dispatch size at the j=0 worst case
  ([DESIGN §4.3]), `N³/2 = 4.4e8` threads launch instead of `1.5e8`, **37.6 ms**
  at [G0t]'s **1.176e10 thread/s**, overlapping the traffic.
  **That rate is [G0t]'s own, and it is a LOWER BOUND, not a launch measurement.**
  `bench/gpu/spikes.ts:145-170` launches `FMA_N = 1<<22` threads ×
  `FMA_REPEATS = 20` per run, and the committed median of
  `bench/results/2026-08-13-gpu-g0t.json` `data.fmaTotalsMs` is 7.1336 ms, so
  `(2²²·20)/7.1336e-3 = 1.176e10 thread/s`. That kernel runs 64 FMA iterations
  per thread and is deliberately compute-bound (`spikes.ts:152-159`), so nearly
  all of the 7.1336 ms is arithmetic: 1.176e10 bounds the true no-work launch
  rate from **below**, and the 37.6 ms it yields therefore bounds the launch
  term from **above**. It is used in that direction deliberately — [P §4 G5]'s
  bias policy is asymmetric against (a), and this is the corner that policy
  requires. **An earlier draft wrote `2.0e10 thread/s` and attributed it to
  [G0t]; that is 1.70× the only artifact cited and appears nowhere in it**
  (its `data` keys are `fmaGflops, fmaTotalsMs, matvecMs, matvecGflops,
  matvecTotalsMs, method` — there is no launch-rate key). It is withdrawn:
  verdict S1.
- Upload 3.7 MB + readback 3.7 MB (`mapAsync` throughput on this box
  unmeasured) ≈ 2–6 ms.

> **Factor wall: 54–101 ms.** Optimistic `14 + max(14.5, 37.6) + 2 = 53.6`
> (launch overlaps traffic); pessimistic `43 + 14.5 + 37.6 + 6 = 101.1` (it does
> not). §H.2's N=480 column is the same three terms at N=480:
> `7.2 + max(1.8, 4.7) + 2 = 13.9` and `21.6 + 1.8 + 4.7 + 2 = 30.1` ⇒
> **14–30 ms** (1440 dispatches; traffic `2·(480³/6)·4 B` ÷ 81.5 GB/s; launch
> `480³/2 = 5.5e7` threads ÷ 1.176e10; upload+readback 1.8 MB ≈ 2 ms).
> Blocked variant, or the indirect dispatch of [DESIGN §4.3]: **~20–45 ms** —
> a rough figure whose composition is **not** itemised here; only its launch
> sub-term is, `N³/6 = 1.5e8` threads = 12.5 ms at 1.176e10 against 7.4 ms at
> the withdrawn 2.0e10, so both ends carry +5.2 ms and nothing else about that
> reading is claimed.
>
> **The pessimistic corner, 101 ms, is ABOVE [DESIGN §5]'s declared
> `≤ 100 ms` factor-wall target.** §H.5 states the consequence.

CPU-side Schur/Woodbury/IR on the read-back factor, at [JSALG]'s committed
throughput (§H's term table): **106 ms central, 88–135 ms** across that
benchmark's measured range — 13 setup backsolves 25.4, gradient 4 K-solves +
4 IR residuals 40.5, projection the same 40.5.

> **(a) solve path per step: 142–236 ms** (88 + 54 optimistic, 135 + 101
> pessimistic) against the 6186.043 ms `saddle` phase it replaces [BASE]:
> `6186.043/236 = 26.2×` … `6186.043/142 = 43.6×`.
> **(a) whole step: 598–695 ms** (both variants carry the same unmoved
> `assembleA + energy + R₂` = `355.442+91.572+7.940` = **454.954 ms**; the
> totals are §H.2's term sums, 598.5 and 694.7):
> `7194.5/695 = 10.35×` … `7194.5/598 = 12.03×`.

### D.2 Variant (b) — measured-model, N=960, total mode

CG needs an SPD system, i.e. the same shift + Schur reformulation as (a)
([DESIGN §4.1]), so (b) is not cheaper to build than the parent assumed. Its
cost driver is the iteration count, and that count is **measured**, not bounded.

**Tier 1** — [K1CAL] `rows[].cgIters` (numpy over
`oracle/tpe_stage1_oracle.assemble_inner_product` on trefoil(N), α=3, β=6,
ε=1e-10, `tol: 0.001`, five rhs directions, `seeds: [0,1,2,3,4]`):

| n | `cgMin` / `cgMedian` / `cgMax` | `jacobiPcgMin` / `jacobiPcgMedian` | `textbookBound` |
|---|---|---|---|
| 480 | 387 / 388 / 411 | 371 / 376 | 1411.6 |
| 960 | 1138 / 1217 / 1229 | 1023 / 1100 | 4484.9 |

Jacobi buys ~3.1 % at N=480 and ~9.6 % at N=960 (`(388−376)/388 = 0.03093`,
`(1217−1100)/1217 = 0.09614`, median vs median): not the null result "no help"
would suggest, but nowhere near changing the h-dependence of a
fractional-Laplacian-class operator. The
measured count is ~4× below the textbook `½√κ·ln(2/tol)` bound because the
spectrum is clustered. The count is committed rather than deferred because
`G5-confirm`'s thresholds are priced off it: it is a gate input.

**Tier 2.** At [DESIGN §3]'s 5 dispatches + one 3×(N×N) matvec ≈ 0.1 ms per
iteration, one inner solve is **102.3 ms** at (b)'s measured best
(`jacobiPcgMin = 1023`) and **122.9 ms** at its worst (`cgMax = 1229`). A step
needs 4–8 inner solves (gradient + IR corrections + the projection correction)
plus the same f64 IR residuals (12–24 matvecs at [JSALG]'s range,
14.8–42.6 ms):

> **(b) solve path per step: 424–1026 ms** ⇒ `6186.043/1026 = 6.03×` …
> `6186.043/424 = 14.59×` vs the CPU `saddle` phase.
> **(b) whole step: 879–1481 ms** (same unmoved 454.954 ms) ⇒
> `7194.5/1481 = 4.86×` … `7194.5/879 = 8.18×` vs the CPU step.

**(b) is deliberately priced at its measured BEST at the fast end** — fewest
iterations, Jacobi included even though the design does not otherwise build it
— because that end is what `G5-confirm` tests (a) against, and a faster (b)
makes the gate harder for (a) to pass. [P §4 G5]'s bias policy is asymmetric
against (a), so every remaining choice here is resolved the same way. The bands
are quoted unrounded for the same reason.

### D.3 The margin, stated honestly

| statistic | (a) | (b) | (a)/(b) |
|---|---|---|---|
| solve path per step | 142–236 ms | 424–1026 ms | **1.80–7.23×** |
| whole descent step | 598–695 ms | 879–1481 ms | **1.26–2.48×** |

`424/236 = 1.797`, so on the **solve-path** statistic the parent's >1.5× rule
fires at **every point of the band**, pessimistic corner included. The one
corner where it does **not** fire is the **whole-step** reading:
`879/695 = 1.265`. (An earlier draft asserted the rule "does not quite fire" at
the then-current 1.93 on the solve path, and a second site quoted the
whole-step corner as 1.31×; both were transcription errors — verdict I3.)
That whole-step corner is the reason the decision is a gate and not a
judgement.

---

## E. K1 calibration

K1's currency is the backward error `‖LLᵀ − A_σ‖_F/‖A_σ‖_F`, f64-measured from
the read-back L. Its bar is the geometric mean of two **bisected, per-N**
quantities — equidistant in log space from "indistinguishable from a correct
kernel" and "the worst T4-fatal level measured".

**Tier 1 — [K1CAL] `rows[]`, keyed by `n`.** `betaSeeds: [0,1,2,3,4]` at the
file's top level pre-registers the five perturbation directions, so a re-run
reproduces every bar.

| n | `kappaK` | `alpha` | `betaMin` / `betaMedian` / `betaMax` | `betaSpread` | `k1Bar` | `k1SlackAboveAlpha` = `k1SlackBelowBetaMin` |
|---|---|---|---|---|---|---|
| 64 | 1.9484e2 | 4.2986e-08 | **5.2504e-04** / 5.6126e-04 / 6.0539e-04 | 1.153 | **4.7507e-06** | 110.52 |
| 120 | 1.4251e3 | 4.7409e-08 | **1.0788e-04** / 1.4166e-04 / 1.9277e-04 | 1.787 | **2.2616e-06** | 47.70 |
| 240 | 1.3978e4 | 4.2551e-08 | **2.3330e-05** / 3.6916e-05 / 4.0965e-05 | 1.756 | **9.9634e-07** | 23.42 |
| 480 | 1.4046e5 | 4.5401e-08 | **3.2893e-06** / 3.7355e-06 / 4.6498e-06 | 1.414 | **3.8644e-07** | 8.51 |
| 960 | 1.4177e6 | 4.0748e-08 | **4.5284e-07** / 5.0572e-07 / 6.7855e-07 | 1.498 | **1.3584e-07** | 3.33 |

> **K1 green** ≤ `k1Bar`. **K1 red** ≥ `e_max(n) = betaMin(n)`. **Amber**
> between. All three are Tier-1 keys; nothing is transcribed and nothing is
> fitted. `κ₂(K)` at n=64 is measured here (1.9484e2), not read off [G6]'s
> `N^3.285` growth fit, because n=64 is below [G6]'s smallest sampled N.

**α is flat in N** (4.07e-8 … 4.75e-8 across a 15× range of n), NOT the
`c·N·u_f32 ≈ 6e-5` an early draft used.

**β is a distribution, not a level**, and each bar takes the **minimum** over
the five directions (`betaMin`), never the median and never a fit.
`betaSpread` = `betaMax/betaMin` runs 1.15–1.79 across n.

**The `0.177/κ₂(K)` fit an earlier fold used is deleted, and its
anti-conservatism is not restated** — it was quantified against a calibration
that has since been superseded, so the old figures would themselves be a
transcription of a dead run. The standing evidence that `β·κ₂(K)` is not a law
is Tier 1: `betaTimesKappaKMin` measures **0.1023 / 0.1537 / 0.3261 / 0.4620 /
0.6420** at n=64/120/240/480/960 — a **6.3×** spread, monotone in n, worst at
exactly the end where a fit would have been the only source. With every row
bisected, nothing is inferred.

### E.1 Where the calibration stood before 2026-09-04, and why it moved

`bench/gpu/k1-calibration.py`'s `ir_iters` predicate seeded its refinement loop
at `z = 0` rather than at `z₀`, spending the first pass on the residual of the
zero vector (identically 1) and leaving `maxit − 1` real refinements inside the
budget. Every β it bisected was therefore understated, every bar too tight, and
the gate biased toward **false-killing a correct kernel**. The fix and the
single-variable A/B are recorded in
`.superpowers/sdd/2026-09-03-webgpu-phase2-spec/reports/spec-fix-c2-calibration.md`.
`alpha`, `bootMetric`, `kappaK`, `kappaASigma`, `cgIters` and the whole
`irModel` block are **bit-identical** across that re-run — none of them route
through `ir_iters` — so the table above is a one-variable correction, not a
fresh measurement campaign. β rose ~3.0× uniformly and every bar rose ~√3.

### E.2 What `e_max` does and does not assert (worked, from the current JSON)

`e_max(n) = betaMin(n)` is the lowest backward error at which T4 failure has
been *observed*, over five directions. A factor at that level failed T4 in the
worst direction measured; it is **not** a proof that every factor at that level
fails, and [DESIGN §6]'s red branch is worded accordingly.

Taking the minimum rather than the median is what makes the amber band do its
job. At n=960 the five directions are
`betaPerSeed = [6.7735e-07, 5.0026e-07, 4.5284e-07, 5.0572e-07, 6.7855e-07]`.
A factor at **4.8e-07** is above three of them: under a median-derived `e_max`
(`betaMedian = 5.0572e-07`) it would be **amber** while already failing T4 in
3 of the 5 directions; under `e_max = betaMin = 4.5284e-07` it is **red**.

### E.3 Why a round bar would not have been a gate

The first draft's flat 1e-4 K1 bar sits `1e-4/4.5284e-7` = **220.8×** above the
level at which T4 failure is measured at n=960. A factor 10× *inside* that bar
— 1e-5 — would pass K1 while sitting `1e-5/4.5284e-7` = **22.1×** above the
fatal level. Scaling [K1CAL]'s `irModel.gradient.r0` and `.rho` linearly in ε
([DESIGN §4.2]'s one precision model) puts that factor at
`f = 1e-5/alpha = 245.4`, i.e. `r0 ≈ 0.267`, `ρ ≈ 0.425`, which needs
**26 refinements** to reach T4's 1e-10 against a budget of 4. A bar that cannot
fail before T4 does is not a gate.

### E.4 The N=960 window, and whether "narrow by construction" is still honest

Slack is equal on both sides by construction (`k1SlackAboveAlpha ==
k1SlackBelowBetaMin`, both `= √(betaMin/alpha)`). The entire admissible window
`betaMin/alpha` measures **12214 / 2276 / 548 / 72.5 / 11.11** at
n=64/120/240/480/960.

**Judgement, stated because it was asked for.** Before the `ir_iters` fix the
n=960 window was 3.55× and the design called K1 there "a narrow gate by
construction … cannot be widened without becoming uninformative". At **11.11×**
that sentence is no longer honest as written: an 11× window is not narrow in
any absolute sense, and a bar with 3.33× of slack on each side is a real gate
with real room. What survives, and is all the argument ever needed, is the
**ordering**: n=960's window is still the narrowest measured, by **6.5×** over
n=480's 72.5×. So the design now says the ordering, not the adjective — the
kill in [DESIGN §6] is K1 at n=480 because that is where the window is widest
among the gated sizes and a red there is therefore least ambiguous, not because
n=960's gate is too tight to be trusted.

### E.5 The IR budget at n=960 versus K1's own green bar

[DESIGN §4.2] pre-registers a consequence about how much iterative-refinement
budget a plausible GPU factor leaves. The arithmetic is Tier 2 over [K1CAL]'s
`irModel.gradient` at n=960 (`r0 = 1.087e-03`, `rho = 1.7309e-03`, both
bit-identical across the recalibration) under the linear-in-ε model, against
`t4: {maxIters: 4, tol: 1e-10}`:

| ε, as a multiple `f` of `alpha` | ε | refinements past z₀ needed |
|---|---|---|
| 1.000 (a LAPACK-quality factor) | 4.075e-08 | 3 — one spare |
| 3.000 | 1.222e-07 | 4 — exactly at the bound |
| **3.334 (`k1Bar/alpha`, i.e. exactly at K1 green)** | **1.358e-07** | **4 — exactly at the bound, zero spare** |
| 6.341 (model exhaustion level) | 2.584e-07 | 5 — over budget |
| 11.11 (`betaMin/alpha`, i.e. K1 red) | 4.528e-07 | 5 — over budget, and T4 failure is *measured* here |

Three things follow, and they are the substance [DESIGN §4.2] carries:

1. **K1-green implies T4-passes-under-the-model at n=960, but with zero
   iteration spare.** The model's exhaustion level (`f ≈ 6.34`) sits above the
   green bar (`f = 3.334`), so no K1-green factor is predicted to blow the
   budget — but a factor sitting *at* the bar needs all four refinements.
2. **The prediction inverted at the recalibration, and this is the one place a
   number change altered what the design predicts.** Under the pre-fix bars
   (`k1Bar = 7.68e-08`, `e_max = 1.45e-07`) a 3α factor at 1.22e-07 was *inside
   the amber band*, and the design's pre-registered expected outcome was that
   "the served N narrows to 480". Under the corrected bars that same factor is
   **green**, and the expected outcome is that **n=960 is served, with no
   iteration margin**.
3. **K1 is therefore not what catches a factor between the green bar and the
   exhaustion level** — the [DESIGN §4.1 step 4] `ExternalSolveError` throw is,
   per step, at run time. That is a design consequence, not a calibration one,
   and it is why the throw exists.

`irModel.<rhs>.solvesTo1e10` (2 / 2 / 3 / 3 / 4 at n=64/120/240/480/960) is
**unchanged** by the recalibration and remains the source for
[DESIGN §4.2]'s "N=960 needs z₀ + 3 refinements against a budget of z₀ + 4":
`residualCurve[0] = 1.0` is the `z=0` entry the consumer compensates for, so a
`solvesTo1e10` of 4 is z₀ plus **three** refinements. That sentence looks like
the recalibrated quantity and is not one.

### E.6 The one precision model, measured

**Tier 1** — [K1CAL] `rows[].irModel.<gradient|projection>`, against the f64 K,
with an f32 LAPACK factor. Ranges below span the two rhs shapes.

| n | `kappaK` | `r0` | `rho` (contraction) | `solvesTo1e10` |
|---|---|---|---|---|
| 64 | 1.9484e2 | 8.91e-07 … 1.03e-06 | 4.90e-07 … 5.08e-07 | 2 |
| 120 | 1.4251e3 | 3.11e-06 … 3.50e-06 | 2.49e-06 … 9.36e-06 | 2 |
| 240 | 1.3978e4 | 2.23e-05 … 2.83e-05 | 1.52e-05 … 5.99e-05 | 3 |
| 480 | 1.4046e5 | 2.30e-04 … 2.41e-04 | 1.19e-04 … 2.48e-04 | 3 |
| 960 | 1.4177e6 | 1.09e-03 … 1.49e-03 | 7.23e-04 … 1.73e-03 | 4 |

All of it is **bit-identical** across the 2026-09-04 recalibration (§E.1).

`r0` and `rho` track each other and both track `ε·κ`, which is the model. But
**the constant is a summary, not a law.** Across the 20 measured
(n, rhs, quantity) points, `r0/kappaKTimesUf32` and `rho/kappaKTimesUf32` run
**0.0086 … 0.110** — a 12.9× spread that an earlier "≈ 0.03" concealed (0.03 is
~3.5× from each end) — and it is not flat in n: the `r0` ratio falls
monotonically **0.077 → 0.041 → 0.034 → 0.027 → 0.013** at n=64…960, so there
is residual n-dependence beyond κ, mildly in the favourable direction. No bar
reads this constant; K1's `alpha` and `betaMin` are bisected per n.

---

## F. CR's bar, and the rhs it must be read against

CR's quantity is `‖Δz‖/‖z‖`, a **solution** error, so it has a floor of
≈ `κ₂(K)·u_f64` and its own reference (`solveSaddleFromA`) is only that
accurate. A flat 1e-10 bar is vacuous where the gate can run and meaningless
where it would bite: `solveSaddleFromA` itself reaches 1e-10 error at N ≈ 1000,
the parent's own G7 target.

> **Bar: `‖Δz‖/‖z‖ ≤ max(1e-12, 10·κ₂(K,N)·u_f64)`**, with `κ₂` read from
> [K1CAL] `rows[].kappaK` (which reproduces [G6] to 6 significant figures on
> the same systems) and `u_f64 = 2⁻⁵³`. Evaluated: **1.6e-12 / 1.6e-11 /
> 1.6e-10 / 1.6e-9** at n=120/240/480/960. Tier 1 input, Tier 2 evaluation,
> one multiplication apart.

**The rhs is now pinned, because the bar is rhs-dependent and nothing pinned
it** (verdict I5). CR runs **both** rhs shapes, the same two [K1CAL] already
uses (`bench/gpu/k1-calibration.py:252-254`, `np.random.default_rng(0)`):

- **gradient shape** `r = [b; 0]`, `b ∈ ℝ^{3N}`;
- **projection shape** `r = [0; d]`, `d ∈ ℝ^{k}`.

Both are reported per fixture, and the bar applies to each. This is the
convention `irModel.gradient` / `irModel.projection` already reports under, so
CR's numbers and [K1CAL]'s are readable against each other.

**Withdrawn:** the claim that the bar sits "≈20–33× above the measured value",
and the four reformulation errors 8.64e-14 / 7.07e-13 / 5.15e-12 / 4.81e-11
that it rested on. **No committed artifact contains them.** [K1CAL]'s per-row
keys are exactly `alpha, assembleSeconds, betaMax, betaMedian, betaMin,
betaPerSeed, betaSeeds, betaSpread, betaTimesKappaKMedian, betaTimesKappaKMin,
bootMetric, cgIters, irModel, k, k1Bar, k1SlackAboveAlpha, k1SlackBelowBetaMin,
kappaASigma, kappaK, kappaKTimesUf32, kappaRatioASigmaOverK, minDiagL, n,
sigma` — there is no reformulation-error key. The slack is a **deliverable of
plan 2a's CR task**, per rhs shape and per N (§J), not a published input. The
bar itself is unaffected: it never depended on the slack.

---

## G. Boot self-test bar

**Tier 1** — [K1CAL] `rows[].bootMetric`, a *forward* difference
`max|L_f32 − L_f64| / max|L_f64|` for a correct f32 LAPACK kernel:
**9.0399e-08 / 3.7638e-07 / 1.3978e-06 / 2.3740e-06 / 1.3463e-05** at
n=64/120/240/480/960. It grows with κ, which is why the fixture is the
smallest one.

> **Bar: `< 1e-5` on trefoil(64) — a CHOSEN ROUND LEVEL, not a formula over
> `bootMetric`.** Nothing in [K1CAL] produces `1e-5`; what `bootMetric` supplies
> is the *check* that the choice is safe — `1e-5 / 9.0399e-08` = **110.6×** above
> the correct value. Reporting that margin is a different claim from producing
> the level, exactly the distinction [DESIGN §5]'s amended rule draws. It is a
> **slack choice, not a declared policy target**: it is not a success criterion
> and no decision turns on its value, it is the width of the window a
> correct-but-differently-summed adapter is allowed, so it does not enter §5's
> "one further category" and does not disturb the claim that P2a's two levels
> are the only members. The slack is deliberate, because the boot test gates
> *availability* on an unknown adapter's summation order, not numerics, and a
> broken kernel (dropped term, off-by-one column) produces O(1) relative error,
> not 1e-5. It is 100× tighter than the first draft's 1e-4, which sat only
> ~21× below the T4-fatal level in this metric's currency.

Unaffected by the 2026-09-04 recalibration (`bootMetric` is bit-identical).

---

## H. P2a — the prediction, term by term, and its noise

### H.1 The three JS f64 constants

**Tier 1** — [JSALG], flat `Float64Array`, Bun 1.3.11; each constant a
median-of-15 over 30 sessions across five processes and two allocation regimes
(median [min–max], ms):

| constant | N=480 | N=960 | spread |
|---|---|---|---|
| N-backsolve fwd+back on a dense lower-triangular factor | 0.432 [0.370–0.768] | 1.954 [1.584–2.502] | 2.1× / 1.6× |
| N×N matvec | 0.376 [0.323–0.687] | 1.425 [1.237–1.777] | 2.1× / 1.4× |
| A_σ build + f32 cast into a REUSED buffer | 1.466 [0.342–5.144] | 1.603 [1.239–3.521] | 15.1× / 2.8× |

**Read the upper end as an over-estimate and the lower end as the real rate.**
Two effects widen them:

1. *Allocation regime.* [JSALG] measures a `reuse` regime (every buffer
   driver-owned — what [DESIGN §4.3] commits 2a to) and a `churn` regime
   (the N×N f32 staging buffer re-allocated per iteration, whose GC then also
   lands in the neighbouring timings). At the median the two are barely
   distinguishable (`churn`/`reuse` = 0.90–1.27× across the six constants), so
   the regime is not what produces the wide tails.
2. *Machine load, the dominant term.* The committed run records
   `loadavg 6.10 → 5.85` and `loadOk: false` against a threshold of 1.5.
   Across eight invocations spanning loadavg 6–19 the per-run **minimum** is
   stable (N=960 backsolve 1.58 / 1.59 / 1.61 / 1.71 ms on the four
   least-loaded) while the per-run maximum moves by 3×. Contention only ever
   *adds* time, so the minimum is the uncontended rate and the upper tail is
   contamination.

Consequence, stated rather than hidden: **every pessimistic figure below
over-estimates the CPU algebra**, so every bar checked against it has more real
margin than the arithmetic shows — the safe direction. Plan 2a re-measures
these in the same session as P2a on a quiet box and records `loadavg` with them.

### H.2 The term table (Tier 2)

Solve counts from [K1CAL] `irModel.<rhs>.solvesTo1e10`: 3 solves at N=480,
4 at N=960 ⇒ `3·3` / `4·3` N-backsolves and `3·3` / `4·3` matvecs per solved
rhs, for TWO rhs (the gradient and the one projection correction, §B).

| term (Bun ms; median [min–max]) | N=480 | N=960 |
|---|---|---|
| `assembleA` (stays CPU in 2a) | 85.9 | 355.4 |
| A_σ build + f32 cast | 1.5 [0.3–5.1] | 1.6 [1.2–3.5] |
| GPU dE wall (3.2 / 5.1 ms [P1]), overlapped with `assembleA` | 0 net | 0 net |
| **GPU factor wall** (upload + 3N dispatches + readback, §D.1) | **14–30** | **54–101** |
| 13 setup N-backsolves (3k+1, k=4) | 5.6 [4.8–10.0] | 25.4 [20.6–32.5] |
| gradient: 3 / 4 K-solves × 3 N-backsolves | 3.9 [3.3–6.9] | 23.4 [19.0–30.0] |
| gradient: 3 / 4 IR residuals × 3 N×N matvecs | 3.4 [2.9–6.2] | 17.1 [14.8–21.3] |
| projection: same again | 7.3 [6.2–13.1] | 40.5 [33.9–51.4] |
| `energy` (CPU f64) | 46.2 | 91.6 |
| `R₂` (§B unattributed remainder) | 60.4 | 7.9 |
| **total** (optimistic = all constants at min + low wall; pessimistic = all at max + high wall) | **224–264** | **598–695** |
| **speedup vs [BASE] `step`** | **4.03–4.75×** | **10.35–12.03×** |

The pessimistic column moves all three constants to their maximum together.
That is deliberate and conservative: they share one machine, so they co-move
under load — it is not an assumption of independence.

Every prediction sits strictly below the matching ceiling corner of §B
(4.75 < 5.07, 4.03 < 4.55; 12.03 < 13.21, 10.35 < 12.12).

**A browser-scale row is deliberately NOT published here.** Applying §C's
×1.775/×1.761 to the CPU terms while leaving the GPU term unscaled produces a
prediction in browser milliseconds, and a bar read against it would be a bar on
the JIT ratio. P2a and `G5-confirm` are ratio gates precisely so that the
factor cancels instead of being applied (§I). The residual effect of the
unscaled GPU term is quantified in §I.2 and is carried by the bar, not by the
prose.

### H.3 The pairing, and the noise — **PROVISIONAL**

**The PAIRING is what licenses a bar, not the size of the swing.** P2a's
denominator swings ±13 % *between* browser sessions ([P1] dE p50 223→290→253 ms
at N=480 across the three de-gate runs, `bench/gpu/README.md:456`). P2a is
specified as a paired same-session measurement precisely so that cancels: a
session that is 13 % slow is 13 % slow on both sides of the ratio.

**Pairing does not remove each median's own sampling error.** [P1 D4]'s method
is 1 warm-up + 5 runs, medians, so the ratio carries the sampling error of a
median-of-5. Bootstrapping the five committed samples bounds it two ways, and
the two bracket however much of the numerator's error the pairing cancels:

| | 90 % band | low-side, generous (numerator cancels) | low-side, strict (both carry error) |
|---|---|---|---|
| N=480 | 0.850×–1.346× | **15.0 %** | **23.5 %** |
| N=960 | — | **6.5 %** | **15.6 %** |

> ### ⚠ These bands are PROVISIONAL, and here is exactly why.
> They are bootstrapped from [P1]'s five `cpuMs` samples per N — and those are
> **dE-phase times**, not step times (§C: `bench/gpu/phase1.ts:324-326` times
> `gradientAnalytical` alone; `data.fullStepGate: false`). P2a's denominator is
> a browser CPU **full descent step** p50, and **no committed samples of that
> quantity exist.** A step's median-of-5 spread is not a dE phase's: a step
> additionally carries a variable Armijo trial count and a variable projection
> iteration count (`lineSearch.ts:209`, `maxIter = 8`), neither of which the dE
> phase has. The bootstrap arithmetic is correct; its input is the wrong
> statistic. The bands are retained as the best available placeholder and are
> **not readable as pass/fail** until §J.1 discharges them.

(The bootstrap resamples five discrete values, so it bounds the noise at n=5
but cannot say what a larger n would buy; that is a reason not to gate on it,
not a reason to assume it shrinks.)

### H.4 The arithmetic of the decision — **PROVISIONAL, same reason**

**The `candidate bar` column is not derived here.** `8×` — and the `3.5×` this
table goes on to reject — are *chosen* levels; the `8×` is [DESIGN §5]'s
**declared policy target**, and nothing in this note produces it. What this
table does is **check** a declared target against the prediction: a bar must sit
strictly below the PESSIMISTIC end of the prediction and clear the noise of the
statistic it is read from.

| | pessimistic prediction | candidate bar | margin | noise, generous | noise, strict | verdict |
|---|---|---|---|---|---|---|
| N=480 | 4.03× | 3.5× | **1.152× (15.2 %)** | 15.0 % | **23.5 %** | clears only the generous bound, **by 0.2 of a point** ⇒ **not gated** |
| N=960 | 10.35× | 8× | **1.294× (29.4 %)** | 6.5 % | 15.6 % | clears **both** bounds ⇒ **gated** |

**N=960's verdict does not depend on how the pairing is accounted for; N=480's
depends on it entirely.** At N=480 the bar survives only if the pairing is
assumed to cancel the numerator's sampling error perfectly, and then by two
tenths of a percentage point (15.2 % against 15.0 %) — under the S1-corrected
launch rate that clearance is no longer meaningfully distinguishable from zero,
which strengthens rather than changes the recorded-not-gated call.

> **N=480 is RECORDED, NOT GATED; P2a's kill authority rests on N=960 alone.**
> This conclusion is *robust to the C1 defect* — it was already the
> conservative call — and so it stands as the design's decision. The **8× bar
> at N=960 is retained as pre-registered**, but its noise defence is
> PROVISIONAL until §J.1 runs, because the "clears both bounds" row above is
> read from a dE-derived bootstrap. Re-running the bootstrap on real step
> samples can only move the noise columns; the margin column is a property of
> the prediction and does not move.

**`R₂` must be ATTRIBUTED before any P2a number is read as pass/fail.** `R₂` is
60.4 ms at N=480 and does not scale with N (7.9 at N=960, 4.4 at N=1000), which
is not a cost profile; it is either real work or an artifact of the
phase-timing overlap, and nothing else distinguishes them. It is worth
**1.30–1.37× of the N=480 result** (work: 4.03–4.75×; slack: 5.23–6.51×) and
only 1.01× at N=960 (10.35–12.03× vs 10.47–12.19×). Recording `R₂` again is not
attributing it: plan 2a's gate task must **name the work, or name the
measurement error**. §B supplies one concrete candidate for the error branch —
[BASE]'s own N=1000 row forces `R < 0` by 5.28 ms, which no amount of work can
explain.

### H.5 Two remaining reads

- **N=960's factor wall has NO headroom against its bar: the pessimistic end
  of the model, 101 ms, is ABOVE [DESIGN §5]'s declared `≤ 100 ms` target**
  (§D.1's `43 + 14.5 + 37.6 + 6`). It was read as `1.2×` headroom (85 vs 100)
  only while the launch term used the withdrawn `2.0e10 thread/s`; at [G0t]'s
  own 1.176e10 it is 101 vs 100 — this note's own pessimistic corner exceeds the
  target by 1.1 ms. **This is a declared policy target that this note's own
  pessimistic model does not meet, and it is recorded as such rather than
  rounded away.** Three things bound how far it goes, and none of them retire
  it: (i) the corner sums the dispatch term at its MAXIMUM `15 µs` — itself an
  unsourced estimate (§D.1) — with zero launch/traffic overlap, so it is the
  worst of two worsts; (ii) 1.176e10 is a LOWER bound on the launch rate
  (§D.1), so 37.6 ms is an UPPER bound on the launch term; (iii) the optimistic
  end, 54 ms, is well under the bar. **What it means operationally:** the
  `> 100 ms` branch of [DESIGN §6] — build the blocked variant or §4.3's
  indirect dispatch — is now a predicted outcome at the pessimistic corner, not
  a remote contingency, and its own estimate (~20–45 ms) clears the target.
  The wall is still the least-modelled term here (`mapAsync` throughput on this
  box is unmeasured) and plan 2a's Cholesky spike measures it directly.
- A measured 695 ms at N=960 is **on model**, not over: an early draft
  predicted 560 ms because it omitted the per-solve backsolves, the IR
  residuals and the projection path — the CPU algebra alone is 106 ms central
  (88–135 ms) at N=960, **4.3×** its entire 25 ms allowance for CPU-side solve
  work.

---

## I. `G5-confirm` — a paired same-session ratio, not a wall-clock level

### I.1 What was wrong, and the remedy

The gate's pass/fail levels were **Bun milliseconds** (879 / 586 / 424 / 283),
derived from (b)'s band, while the gate's own measurement is necessarily a
**Chrome** wall-clock (`bench/gpu/drive.ts:20`). On §C's own ×1.761 factor a
variant (a) landing exactly on the model of §D.1 measures
`7194.5 × 1.761 / 12.52 = 1011.9 ms` at its optimistic corner and
`7194.5 × 1.761 / 11.05 = 1146.6 ms` at its pessimistic one, **both above the
879 ms red bar**: a correct kernel was pre-registered to fail, and the red bar
was a threshold on the Bun-vs-Chrome JIT ratio rather than on kernel quality.

> **Remedy: `G5-confirm` becomes a paired same-session ratio, exactly as P2a
> already is.** Its measured statistic is
> `r_a = (CPU f64 p50) / (variant-(a) p50)`, both medians drawn from the SAME
> browser session, on the same fixture, interleaved, wall-clock including all
> uploads and readbacks — reported for the whole descent step and for the solve
> path. The JIT factor is then in both the numerator and the denominator and
> cancels.

### I.2 The bars, as formulas over Tier-1 keys

(b)'s side is converted to the same currency: a **ratio against [BASE]**, whose
numerator and denominator are both Bun, so it too is JIT-free to first order.

```
r_b_step   = [BASE].N960-total-frozen-ldlt.phases.step.ms   / (b)_whole_step_ms   (§D.2)
r_b_solve  = [BASE].N960-total-frozen-ldlt.phases.saddle.ms / (b)_solve_path_ms   (§D.2)

green : r_a_step  ≥ 1.5 · max(r_b_step)
red   : r_a_step  ≤       max(r_b_step)                  ([P §4 G5]'s rule fires nowhere)
amber : otherwise ⇒ the whole-step reading does not decide, and the solve path must:
        green : r_a_solve ≥ 1.5 · max(r_b_solve)   else red
```

`max(r_b_·)` is (b) **at its measured best**, which is the [P §4 G5]-mandated
direction: a faster (b) makes the gate harder for (a) to pass.

Evaluating the formulas at §D.2's bands (Tier 2 — shown so the shape is
legible, **not** as the operative levels; §J.2):

```
max(r_b_step)  = 7194.500/879 =  8.185×   ⇒ green ≥ 12.277× , red ≤ 8.185×
max(r_b_solve) = 6186.043/424 = 14.590×   ⇒ amber-green ≥ 21.885×
```

**Sanity check that the false kill is gone.** An (a) landing exactly on §D.1's
model measures 10.35–12.03× (Bun-equivalent). Red requires ≤ 8.185× — an
on-model (a) is now **never red** — that was the defect, and it is gone.
**But BOTH corners are now amber, not just the pessimistic one.** Green needs
≥ 12.277× and the optimistic corner is 12.03×; it cleared green by 0.5 % only
while the launch term used the withdrawn `2.0e10 thread/s` (§D.1, verdict S1).
The conclusion survives the unit correction below rather than being an
artifact of the currency: in browser terms the corners are 12.52× and 11.05×
against a worst-corner-corrected green bar of `12.277 × 1.0670 = 13.10×`, so
both are amber there too. **The whole-step reading therefore decides nothing
for an on-model (a), and the solve-path test carries the entire sanity check**
— it passes across the whole band, with `26.212/21.885 = 1.20×` of margin at
the pessimistic corner and `43.564/21.885 = 1.99×` at the optimistic one. That
amber test is the least marginal bar in the milestone, which is exactly why the
whole-step reading is allowed to be inconclusive; S1 makes that allowance
load-bearing rather than precautionary.

**The one residual unit effect, named rather than hidden.** `r_a` is measured
in the browser and `r_b` is computed from Bun timings. The JIT factor cancels
inside each ratio, but (a)'s GPU term does not scale with it, so (a)'s *browser*
ratio is slightly higher than its *Bun* ratio:
`J·T / (J·T_cpu + T_gpu)` with `J = 1.761` gives **+4.06 %** at (a)'s optimistic
corner (T=598, T_gpu=54) and **+6.70 %** at its pessimistic one (T=695,
T_gpu=101). That is a bias *in (a)'s favour*, and [P §4 G5]'s policy is
asymmetric *against* (a), so the emitter of §J.2 applies the worst-corner
correction to the green bar rather than leaving it unstated. The magnitude to
beat: the pre-fix defect biased the same comparison against (a) by **76 %**;
this residual is **≤ 6.70 %** and signed, bounded and corrected.

---

## J. Pre-registered calibration debts

These are **preconditions on reading a gate**, not deferred questions. Each
names what must exist, and until it does the corresponding bar is not readable
as pass/fail.

### J.1 P2a's denominator has never been measured — measure it

**Owed:** five browser CPU f64 **full-descent-step** p50 samples at N=480 and
N=960, taken by [P1 D4]'s method (1 warm-up + 5 runs, medians) in the same
harness, committed as a results JSON, and the §H.3 bootstrap re-run on them.
**Until then:** §H.3's bands and §H.4's noise columns are PROVISIONAL.
**Does not block:** the 8× bar itself, the N=480 recorded-not-gated decision,
or the margin column — none of them are read from the bootstrap.
**Rank:** this is the precondition that sits alongside the `R₂` attribution
already on P2a's gate row; both must clear before a P2a verdict is read.

### J.2 `G5-confirm`'s levels need an emitter

**Owed:** a committed script that reads [BASE] and [JSALG], evaluates §I.2's
formulas including the worst-corner unit correction, and writes the resulting
`r_a` bars into a committed results JSON alongside the (b) bands they came
from. **Until then:** §I.2's evaluated numbers are illustrative Tier 2 and the
operative definition is the formula. **Why an emitter and not a transcription:**
this is the whole point of the split — a level that only exists as prose is a
level that goes stale silently, three times in a row.

### J.3 CR's slack is a deliverable, not an input

**Owed:** plan 2a's CR task reports `‖Δz‖/‖z‖` per fixture, per N, **per rhs
shape** (§F), so the slack below the bar becomes a committed number for the
first time. **Until then:** no slack figure is quotable. **Does not block:** CR
itself — its bar is `10·κ₂(K)·u_f64` and never depended on the slack.

### J.4 Things measured on a loaded box

[JSALG] carries `loadOk: false`. Plan 2a re-measures its three constants in the
same session as P2a and records `loadavg` beside them, so a later reader can
separate contention from algebra (§H.1).

---

## K. Cross-reference index — which [DESIGN] section reads which section here

| [DESIGN] | reads |
|---|---|
| §2 (what leaves the CPU, the ceiling, `R₂`) | §B |
| §3 (the (a)-vs-(b) decision) | §D |
| §4.2 (precision model, IR budget) | §E.5 |
| §4.5 (main-thread cost) | §B (ceilings) |
| §5 CR row | §F |
| §5 K1 row | §E |
| §5 Boot row | §G |
| §5 P2a row | §H, §J.1 |
| §5 `G5-confirm` row | §I, §J.2 |
| §6 kill branches | §E (levels), §D (the (b) fallback), §I |
