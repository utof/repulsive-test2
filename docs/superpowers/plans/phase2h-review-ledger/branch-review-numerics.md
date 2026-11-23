# Branch review — `feat/phase2h-impl`, scope NUMERICS AND GATE POWER

Blind round-1 review. No prior findings were supplied; severities below are mine alone.
Contract: `docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md` (body + amendments A1–A47,
later amendments overriding the body). Merge base `9dd5551`, 47 commits.

Every number quoted as MEASURED in this report was produced by a probe script I wrote and ran during
this review; each finding names its script. The scripts are in `/tmp` (throwaway) and are reproduced
inline in §5 so any reader can re-run them.

---

## 1. Verdicts

### (a) Numerical correctness — **PASS WITH FINDINGS** (all findings Minor)

I re-derived the implemented operator against the production dense assemblers
(`src/core/sobolev/innerProduct.ts:38-145`, `:177-300`) rather than against spec prose, and against
arXiv:2006.07859 App. B.2/B.3.4. **I found no sign error, no wrong factor, no transposed index and no
wrong operand order.** Specifically re-derived and confirmed correct:

| Item | Where | Verdict |
| --- | --- | --- |
| `endpoint4Kernel` symmetry factors `0.5` (hi) / `0.25` (low) | `nearField.ts:~250` | correct: `K = 2w` and `K⁰ = w⁰_IJ + w⁰_JI` against `innerProduct.ts:103-106`, :242-246 |
| low-order `T_I ⊕ T_J` symmetrisation | `nearField.ts` | correct, and matches the documented `w⁰_IJ ≠ w⁰_JI` asymmetry of the dense assembler |
| `Mᵀ`/`Eᵀ` adjoints (`out[i2] += r`, `out[i1] -= r`; `h` added to both ends with the `/2`) | `apply.ts:600-627` | correct adjoints of the `d = (u_{i2}-u_{i1})/ℓ` and `(u_{i1}+u_{i2})/2` operators |
| `A_bct = Mᵀ(diag(K1) − K)M + Eᵀ(diag(K⁰1) − K⁰)E` | `apply.ts` `applyColumns` | structure matches Eq. 31 / App. B.3.4 |
| ε placement — `+epsilon` after EACH norm separately | `nearField.ts:34-39`, `apply.ts` `farKernel` | correct, mirrors `innerProduct.ts:96`, `:242-246` |
| per-exponent `pow` with `pLow = pHi + 4` | `nearField.ts:34-39` | algebraically identical to the repo's `k24/d**distExp` form; the comment's warning that the two differ in f32 by more than the H1-b band is correct and the pin exists |
| `traceA` = `Σ (k1Hi·2|T|²)/ℓ² + 0.5·k1Low` | `operator.ts` | re-derived from `applyColumns`, exact |
| `σ = tr(A_bct)/n`, `A_σ = A + (σ/n)11ᵀ`, `shift = σ·(s/n)` in that order | `gmres.ts:137-141`, `applyAsigma`/`shiftedApply` | **the "do not reassociate" anchor is TRUE and I measured it**: `σ·(Σu/n)` is exact for every σ on the ones vector, while the alternative `(σ/n)·Σu` round-trips inexactly for **≈4.0 % of random σ** (4.0468 % and 4.0164 % on two runs) (3.6e6 draws over n = 64…30 000; e.g. n=120, σ=31.302402703363683 lands 2 u off). So `A_σ1 = σ1` really is exact by construction, not by luck, and `precond.ts`'s "constant-mode eigenvalue exactly σ" premise holds |
| adjacency certificate `d > r_x[A]+r_x[B]+½(maxEll[A]+maxEll[B])` | `bct.ts` | correct and conservative |
| `θ === 0 ⇒ exact` evaluated BEFORE the admissibility inequality | `bct.ts` | correct; the χ=0 guard cannot be bypassed by a 0-valued θ |
| curve-Laplacian strict diagonal dominance (pivot-free Thomas) | `laplacian.ts` | `diag − Σ|off| = 3mm_{i−1} + 3mm_i > 0` (both incident edges contribute); Thomas without pivoting is justified |
| Schur exact constant-mode reduction `[[S,G],[Gᵀ,0]]`, `T = GᵀS⁻¹G` | `schur.ts` | fully re-derived including every sign |
| `applyCe` sign convention (`+T` at `i1`, `−T` at `i2`) | `nullspace.ts` | correct: `constraintSet.ts` `edgeLengthsBlock` defines `Φ = ℓ⁰ − ℓ`, so `dΦ = −dℓ`. I initially suspected a flip here; the suspicion was wrong |
| `L = 2(diag(W1) − W)`, `W_ij = a_i a_j K_ij`, `a_i = (ℓ_{i−1}+ℓ_i)/2` | `lowOrder.ts` | correct |
| `project()` Neumaier two-pass compensated sum | `precond.ts` | correct; and it is what makes the `toBe(0)` structural assertions true rather than lucky |

**Bit-identity claims are honest.** `A_bct·1 = 0` and `L·1 = 0` really are bit-exact for the reason the
comments give (the same operation sequence applied to the ones vector), and per-column independence
makes `nc=3` bit-identical to `nc=1`, so the `expect(...).toBe(0)` assertions are structural verdicts
and not tolerances. The one place a comment could have over-claimed — `apply.ts:353-363`'s ACCUMULATION
ORDER note — explicitly says it reproduces the archived f32 oracle to 0.31 % / 0.036 % and **NOT**
bit-for-bit. `lowOrder.ts`'s module TSDoc correctly scopes `L·1 = 0` to the ones vector only and
publishes `max|L·(7.5·1)| = 5.3e-14…9.9e-14` for the general case. I found **no** bit-identity comment
sitting above code that does not preserve it.

### (b) Gate and test power — **PASS WITH FINDINGS** (2 Important, 7 Minor)

This is a strong gate set by this repo's standards, and the two Important findings are both about one
thing: **a bar whose RED side cannot be reached, and a bar whose RED side can be reached by something
that is not a defect.** Neither is a wrong computation; both change what a published verdict means.

What the harness gets right, and which I verified rather than assumed:
- Every bar is computed from the run's own counts, never copied from a spec literal
  (`hCalibration.ts` `h1bBar`, `H2_BAR = 50 * U_F32`, `[A46 item 5]`). I re-derived the H1-b bar from
  raw counts and got `5.7784e-5` against the harness's `5.778e-5`.
- A six-value verdict vocabulary including `VACUOUS` and `NOT ESTABLISHED`, and it is used:
  `finaliseH3` publishes the quantisation predicate as `VACUOUS` **with its margin value**
  (`[A42]` ruling 5), `finaliseH4b` returns `VACUOUS` by construction, `primaryBar.status` is
  `'NOT MEASURABLE AT THESE RUNGS'` rather than silently projected, and `gateH1a`'s clause-4 adapter
  certification is published as `NOT APPLICABLE` with the reason that on a CPU harness it *cannot not*
  fire (`[A46 item 2]`).
- `gateH4c` publishes `spreadFinding: 'CONJUNCT 1 IS A WEAK GATE'` with the measured 0.9987–1.0034
  spread, i.e. it files the finding against itself.
- `gateH6` publishes `runHere: false`, `valuesAreTranscribed: true`, the withdrawal ruling, **and** a
  control row (the exact δ=0 operator reads 0.7644 against θ=0.5's 0.7856, which is what withdraws
  the RED as a θ signal).
- `etaViolations` adds a clause-5 `non-finite` that is explicitly not in the spec, because NaN fails
  every `<` and `===` in clauses 1–4 — exactly the `gates-name-their-impossible-values` failure.
- The differential tests mostly avoid shared-path blindness on purpose: `bct.ts` is variant-free by
  design so f32 and f64 share ONE block structure and the differential isolates arithmetic;
  `precond.test.ts`'s `extentOf` is a deliberate second copy of `curveExtent` with the reason written
  down; `apply.test.ts`'s H1-b consistency check builds a SECOND prepared instance rather than
  comparing an instance with itself. Several tests carry impossible-value guards
  (`expect(e).toBeGreaterThan(0)`, `expect(differFromRoundOnce).toBeGreaterThan(bct.stats.nFar)`).
- `gateH1a` and `gateH1b` each carry an executed knock-out with `demonstratesRedSideReachable`.

### Spot-check of MEASURED numbers — **13 of 14 reproduce, most to the last quoted digit**

The one that does not is M7 (a magnitude comparison in a `nearField.ts` anchor comment, off by ~240x
in the safe direction). Everything else below came back at the quoted digits.

| Quoted value | Source | I measured | Script |
| --- | --- | --- | --- |
| `κ_near = 93.26875`, `nearPairs = 89538`, `nFar = 3530`, `κ_far = 3.677` (torusKnot n=960 θ=0.5) | `apply.test.ts:645-650` | identical | `h1bf16.ts` |
| `nearPairs = 178320`, `κ_near = 185.75` (θ=0.25) | `apply.test.ts:647-649`, `hCalibration.ts:551` | identical | `h1bf16.ts` |
| H1-b bar `5.778e-5` / `1.1426e-4` | `apply.test.ts:645-651` | `5.7784e-5` / `1.1426e-4` | `h1bprobe.ts` |
| f16 knock-out overshoot `45.6x` | `apply.test.ts:~696`, `gateH1b` | `45.61x` (θ=0.5); `31.89x` at θ=0.25 | `h1bf16.ts` |
| the f16 model's "`m` vs `κ`" notation defect: literal-`m` form is `9.9x` higher | `gateH1b` `modelNotationDefect` | `8112.1/819.2 = 9.90x`, and the κ-form is `819.2x` the bar against the spec's "~820x" | `h1bf16.ts` |
| `30/ext² = 1.0021` at trefoil extent `5.4716` | `precond.test.ts` module TSDoc | `ext = 5.4716`, `massWeight = 1.002070` | `h5probe.ts` |
| `κ_excl = 19.6713` (n=960 θ=0.25) and `20.8632` (θ=0.5) | `precond.test.ts:433-435` | identical to 4 dp | `h5probe960.ts` |
| `κ_excl` trips `m̂=100` at `65.9672`; does not trip `m̂=70` at `36.7279` | `precond.test.ts:436-437` | identical to 4 dp | `h5probe960.ts` |
| `λ₁ = 0.7486`, `λ₂ = 0.7652` at `m̂=100` | `precond.test.ts:421-423` | `0.748643`, `0.765218` | `h5probe960.ts` |
| `max\|L·(c·1)\|` over n ∈ {64,240,480,960} θ=0.25: `0` exactly for c=1 | `lowOrder.ts:40-42` | `0.00e+0` at every rung | `ldot1.ts` |
| same, `5.3e-14 .. 9.9e-14` for c=7.5 | `lowOrder.ts:41` | `5.33e-14 .. 9.95e-14` | `ldot1.ts` |
| same, `2.1e-14 .. 3.2e-14` for c=−3 | `lowOrder.ts:41` | `2.13e-14 .. 3.20e-14` | `ldot1.ts` |
| same, `6.7e-16 .. 9.4e-16` for c=0.1 | `lowOrder.ts:41-42` | `6.66e-16 .. 9.44e-16` | `ldot1.ts` |
| the `σ·(Σu/n)` order is load-bearing (`(σ/n)·Σu` "would round differently") | `gmres.ts:139-141` | the alternative is inexact for ≈4 % of random σ; the shipped order for 0 % | `ldot1.ts` | the four `max|L·(c·1)|` ranges in `lowOrder.ts`'s bit-exactness scoping | spot-check |
| `shiftorder.ts` |

---

## 2. Findings

### Critical — none.

I considered elevating I1 below. I did not, for the reason stated in it: the harness reports `INVALID`
rather than `GREEN` in the counter-regime, so no wrong verdict is published; what is lost is the
*information content* of H5's GREEN. A Critical here would overstate it, and this repo has a recorded
history of unanimous-but-wrong Criticals.

---

### Important

#### I1 — H5 property 2 (`κ_excl < 40`) has no reachable RED side in any configuration the shipped gate can execute

**Where:** bar at `bench/gpu/hCalibration.ts:170` (`H5_KAPPA_EXCL_BAR = 40`), verdict logic
`finaliseH5` (`hCalibration.ts:1895-1965`), measurement `gateH5` (`hCalibration.ts:1967-2018`),
shipped assertion `test/sobolev/hierarchical/precond.test.ts:442`, trip claim
`precond.test.ts:436-438`.

`finaliseH5:1911-1916` correctly makes `λ₁ == 1` and `λ₂ > 1.5` **preconditions** of the quantity:
`κ_excl := λ_max/λ₂` is "excluding the constant mode" only while the constant mode is the smallest
eigenvalue. If either fails, the verdict is `INVALID`, not RED. That is the right design. The
consequence, which is not recorded anywhere, is that **every axis that reaches `κ_excl > 40` also
breaks a precondition**, so the gate cannot publish RED:

MEASURED, trefoil, `θ=0.5`, n=960, shipped operator (`probes/h5probe960.ts`):

```
mhat= 30  l1=1.000000 l2=3.325457 lmax=69.3797 kExcl=20.8632 green   -> GREEN
mhat= 70  l1=1.000000 l2=1.479087 lmax=54.3237 kExcl=36.7279 green   -> INVALID (l2 <= 1.5)
mhat= 85  l1=1.000000 l2=1.022432 lmax=52.1404 kExcl=50.9964 **>40** -> INVALID (l2 <= 1.5)
mhat=100  l1=0.748643 l2=0.765218 lmax=50.4793 kExcl=65.9672 **>40** -> INVALID (l1 != 1)
```

and at n=240 (`probes/h5probe.ts`) the same picture with the band boundary between `m̂=60`
(`λ₂=1.961`, `κ_excl=26.2`) and `m̂=70` (`λ₂=1.467`, `κ_excl=33.3`).

Three consequences, each checkable:

1. **The mass-weight axis cannot produce RED.** Bisecting the `λ₂ = 1.5` boundary at n=960
   (`probes/h5band.ts`) puts it at `m̂ ≈ 69.5`: `m̂=69` gives `λ₂ = 1.519888` with `κ_excl = 35.8543`
   (in band, GREEN) and `m̂=70` gives `λ₂ = 1.479087` with `κ_excl = 36.7279` (out of band, INVALID).
   So the supremum of `κ_excl` over the valid band is **≈ 36.2 against a bar of 40 — a 1.10x
   shortfall.** Every `m̂` that clears 40 is already `INVALID`.
2. **`precond.test.ts:436-438`'s "DEMONSTRATED sensitivity floor is `m̂ ∈ (70, 100]`" is not a
   demonstration of this bar.** Both endpoints sit outside the validity band (`m̂=70` has
   `λ₂=1.479 ≤ 1.5`). The same comment, sixteen lines earlier at `:418-427`, states the rule this
   violates: *"Any future trip demo outside the live mass band must assert this pair or declare the
   quantity invalid."*
3. **Every executed knock-out of H5 lands on the nullity/λ₁ clause, never on the κ_excl bar.**
   `precond.test.ts:571-573` measures `rs-identity` driving `κ_excl` to >100x — but that variant has
   `λ₁ ≈ 0` and `nullDim = 1`, so `finaliseH5` returns `INVALID` there too. The test says this itself
   (`:527-535`: "the KNOCK-OUT gate is `nullDim`/`λ₁`; the GROWTH EXPONENT belongs on `κ_excl`") — and
   then `[A32]` withdrew the growth exponent as a bar, which left the level bound with no trip at all.

The bar is **not** unreachable in principle, which is what makes the fix cheap. On the n axis
(`probes/h5ladder.ts`, trefoil θ=0.5, shipped m̂):

```
n=  64 kExcl= 8.2110      p(kExcl) full ladder = 0.3297  -> kExcl = 40 at n ~ 6 900
n= 120 kExcl=11.8922      p(kExcl) dropping n=64 = 0.2611 -> kExcl = 40 at n ~ 11 600
n= 240 kExcl=15.4790
n= 480 kExcl=17.5102
n= 960 kExcl=20.8632
```

So the bar bites inside the 1 000–30 000 scale target — but `gateH5`'s own
`hCalibration.ts:2015` (`limitations[3]`) records the instrument as O(n³) (~8 s at n=960, 46 s at 1920, 442 s at 3840), so the
gate can never be executed where its bar discriminates. And the fixture axis is closed by
`hCalibration.ts:2012` (`limitations[0]`), which forbids applying the bound off-trefoil (arc 138, helix 159) while `gateH5:1971`
builds `'trefoil'` and nothing else.

**Fix (concrete, smallest unblocking form):** keep the bar, and make the gate state what it is. Either
(a) publish `κ_excl` as `BOUNDED-AT-MEASURED-RUNGS` plus the measured exponent and the extrapolated
crossing n, so the GREEN is read as "flat to n=960, projected to cross 40 at n ≈ 7e3–1.2e4" rather
than as a passed discriminator; or (b) add one executed in-band RED demo. The cheapest in-band defect
I found is an `m̂` sweep **with the band enforced** — i.e. assert that `sup{κ_excl : λ₂ > 1.5}` is
below 40 and publish the value (36.2 at n=960), which converts a 1.10x unreachability into a
*published margin*. Either way, delete or re-scope `precond.test.ts:436-438`: as written it presents
an out-of-band reading as the gate's sensitivity evidence.

**The pattern to copy is already on this branch.** `test/sobolev/hierarchical/schur.test.ts:547-549`
does for the Schur truth bar exactly what H5 property 2 lacks: it sweeps `tol` and then **asserts the
RED side** —

```ts
// Non-vacuity: a loose tol DOES violate the bar, so the truth bar above is a gate that can trip.
expect(at(1e-8).err).toBeGreaterThan(BAR * 100);
expect(at(1e-10).err).toBeGreaterThan(BAR);
```

— and follows it with a knee analysis proving the shipped `tol` sits within 4x of the error floor
(`:553-557`). An `m̂` sweep in `precond.test.ts` with the `λ₂ > 1.5` band enforced, plus
`expect(supInBand).toBeLessThan(40)` on the published value, is the same construction and would close
this finding.

---

#### I2 — H1-b's lower bound is an n-independent literal guarding an n-growing quantity: 1.61x headroom at the corpus's smallest rung, false RED below n ≈ 30, and `gateH1b` silently gates at whatever rung it is handed

**Where:** `bench/gpu/hCalibration.ts:135` (`const H1B_LOWER = 1e-7`), used at `:720`
(`const inBand = d32 >= H1B_LOWER && d32 <= bar`); rung selection at `gateH1b`'s first line
(`const n = ns.includes(960) ? 960 : ns[ns.length - 1]`); same floor hardcoded in the shipped test at
`test/sobolev/hierarchical/apply.test.ts:678` and `:690`
(`expect(d).toBeGreaterThanOrEqual(1e-7)`).

The **upper** side of the band is computed from the run's own counts and therefore scales
(`10·(κ_near+κ_far)·u_f32`). The **lower** side is a bare `1e-7`, calibrated at n=960 — "a decade under
the MEASURED f32 apply noise". But `δ_precision` grows with n, so the floor's margin collapses as n
falls. MEASURED (`probes/h1bprobe.ts`, torusKnot, scalar apply, seed 12345):

```
theta=0.5  n= 32 delta=1.0661e-7  margin above floor = 1.07x     <-- 6.6% from a spurious RED
theta=0.5  n= 64 delta=1.6064e-7  margin above floor = 1.61x
theta=0.5  n=120 delta=2.4159e-7  margin above floor = 2.42x
theta=0.5  n=240 delta=3.0231e-7  margin above floor = 3.02x
theta=0.5  n=480 delta=4.7544e-7  margin above floor = 4.75x
theta=0.5  n=960 delta=1.0515e-6  margin above floor = 10.52x
```

(θ=0.25 is within 1 % of these at every rung.) Scope honestly: with the driver's default ladder
(`hCalibration.ts:2375`, `ns ?? [64,120,240,480,960]`) n=960 is present and the fallback never fires,
so **no default run is at risk.** What is at risk:

- **n=64, the smallest rung in the corpus, has 1.61x headroom on the floor** against 55x on the bar.
  A floor with 61 % headroom is not a comfortable guard, and the row publishes `marginBelowBar` for
  the upper side and **no margin at all for the lower side**, so the shrinkage is invisible in the
  artefact.
- Extrapolating the measured growth (`δ_precision ≈ n^0.71` over n=120…960), the floor crosses the
  quantity at **n ≈ 30**: below that, `verdictOf(allInBand ? 'GREEN' : 'RED')` publishes **RED on a
  correct operator**. n ≈ 30 is off-corpus, so this is reachable only by a deliberately small shard —
  but `gateH1b` takes `ns[ns.length - 1]` **silently**, with no rung-validity check, so nothing stops
  it or flags it.

That is the mirror image of I1: a bar that can be tripped by something that is not a defect is as
uninformative as one that cannot be tripped at all, and worse, because a false RED stops a branch.

The floor is doing real work and should not be deleted — it is the `simulate-bugs-at-target-precision`
guard, and `apply.test.ts:664-670` states its purpose correctly (an "f32" path reporting ~1e-16 is an
unrounded f64 simulation). The problem is only that it is a constant where its neighbour is a formula.

**Fix:** make `gateH1b` refuse to gate below the rung the floor was calibrated at — publish
`NOT ESTABLISHED` with the reason, instead of silently falling back to `ns[ns.length-1]` — and record
the measured margin (`d32 / H1B_LOWER`) in the row so the shrinkage is visible. If a scaling floor is
wanted instead, it must be fitted and published with its point set; `δ_precision` grows as ≈ n^0.71
over n=120…960 on this fixture, which the row data would make checkable.

---

### Minor

#### M1 — H3's response to the only shipped-code preconditioner defect is `INVALID` (censored), not RED; and the `§7.8` rationale quoted in `precond.test.ts` does not hold at H3's live τ_η

**Where:** `hCalibration.ts` `gateH3`/`finaliseH3` (`:1038-1183`), `TAU_ETA = 1e-12`
(`[A31]`'s third re-pin); claim quoted at `test/sobolev/hierarchical/precond.test.ts:~530`
("spec §7.8: H3 within 1 iteration, H4a/H4b′ green, ε_A degraded only 14-75 %"), spec source
`docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md:2488-2491`.

**First, the good news, because it answers the brief's central question for this gate: H3's RED side
IS reachable and I tripped it.** MEASURED (`probes/h3probe.ts`, trefoil θ=0.5, 5 seeds, τ_η=1e-12,
`maxit = min(600, 2·dim)`, the harness's own `b` generator and probe schedule):

```
shipped           n=120 iters=43   n=240 iters=46   n=480 iters=47
identity          n=120 iters=120  n=240 iters=300  n=480 iters=600 (censored)
dropConstantMode  n=120 iters=368  n=240 iters=600  n=480 iters=600 (censored)

shipped    120->240: G=1.0698 band=0.0481 G-band=1.0216 green
identity   120->240: G=2.5000 band=0.0292 G-band=2.4708 **RED**, neither rung censored
```

So `G − band < 2.0` is a live bar, not a formality. Two things are worth recording that are not:

1. **The RED window is narrow by construction.** A defect must slow convergence enough to more than
   double the count per doubling, yet not enough to reach `maxit` at either rung — otherwise
   `finaliseH3` reports `INVALID` (censored), not RED. The identity preconditioner clears that window
   at 120→240 and falls out of it at 240→480; `dropConstantMode` is outside it at every pair
   (`INVALID` from n=240 up). Per this repo's own doctrine that is the recognised defect class, not a
   nitpick: spec §7.8:2472-2474 calls reporting a knock-out as INVALID instead of RED "the identical
   error class as #59 classifying H4b's `q ≤ 0` as INVALID".
2. **The "H3 within 1 iteration" rationale is stale.** Spec §7.8:2489 scopes it to `gmresIters1e-4`
   and predicts the opposite at tighter τ ("The damage is mild at τ=1e-4 and bites as τ tightens").
   H3's live currency is τ_η = 1e-12, where knock-out (2) moves the count 43 → 368 at n=120 (8.6x) and
   censors from n=240 up. The claim that "H5 is the only gate in the set that catches it" is therefore
   true only at the withdrawn τ.

**Fix:** record the measured RED demonstration (identity preconditioner, 120→240, `G−band = 2.4708`,
uncensored) in `gateH3` as a `knockOut` block with `demonstratesRedSideReachable: true` — `gateH1a` and
`gateH1b` already have exactly this shape and H3 is the only gate carrying the 30k-scale claim without
one. Re-scope the `precond.test.ts` parenthetical to "(§7.8, at τ=1e-4; at H3's live τ_η=1e-12 the
count moves 43 → 368 at n=120 and censors above)".

#### M2 — `schur.test.ts`: at n=960 the UNPINNED constraint sets carry no accuracy bar at all

**Where:** `test/sobolev/hierarchical/schur.test.ts:452-471` (`checkCells`), callers at `:488-489`
(`{ literalBar: false, truthBar: true }`), `UNPINNED = new Set(['k0','len'])` at `:273`.

At n=960 the literal bar is deliberately off ("it plateaus at the oracle's own error") on the
assumption that the truth bar takes over. For the two unpinned sets there is no refined truth, so
`if (opts.truthBar && !unpinned)` skips silently and the surviving assertions on those cells are
`r.residual < 1e-6`, `r.gmres.eta < 1e-10` and two length checks. The accuracy numbers are computed
and logged (`literal=9.79e-11`, `7.97e-11` in this run) and asserted against nothing — 2 of the 6
n=960 cases. Residual and η are real self-certifications, so this is not a hole in the verification as
a whole (n=240 asserts the literal bar for the same sets); it is a hole nothing declares.

**Fix:** one line in `checkCells` — `expect(opts.literalBar || (opts.truthBar && !unpinned)).toBe(true)`
— so a cell with no accuracy bar fails loudly instead of passing quietly; or keep `literalBar` on for
unpinned cells at n=960 with its own measured bar.

#### M3 — the test's unit roundoff is a 3-significant-figure decimal where `[A39 item 1]` pins `2**-24`

**Where:** `test/sobolev/hierarchical/apply.test.ts:55` (`const U_F32 = 5.96e-8;`), used by `h1bBar`
at `:635-637`. The harness does it correctly (`hCalibration.ts` `U_F32 = 2**-24`, published as
`uF32Definition: 'U_F32 = 2**-24 (UNIT ROUNDOFF, not machine epsilon 2**-23) — [A39 item 1]'`), and
`[A46 item 5]` requires bars computed and never copied.

Numerically this is a 7.6e-5 relative difference (test bar `5.77798e-5` vs harness bar `5.77842e-5`)
against a 55x margin, so it changes no verdict today. It matters because the test file is the one
place on the branch that writes the unit roundoff as a literal, and `5.96e-8` is also the value
`[A39 item 1]` exists to disambiguate from `2**-23`.

**Fix:** `const U_F32 = 2 ** -24;` in `apply.test.ts:55`.

#### M4 — `apply.test.ts` still asserts a θ ruling that `[A46 item 7]` withdrew, and `gateH6` says so

**Where:** `test/sobolev/hierarchical/apply.test.ts:641` ("theta=0.25 is PRODUCTION after gate H6 read
RED (spec §7.9, 2026-10-09)"), against `hCalibration.ts` `gateH6`'s own `consequence` field:
*"apply.test.ts:644's comment … rests on a withdrawn inference. The CHOICE of production theta stays
open on #51's evidence."*

No numeric effect — the test asserts the band at **both** θ, which is the right behaviour for exactly
the reason the comment gives. The harness identified the stale line and the line was not updated.

**Fix:** replace the clause with "θ=0.25 is `bct.ts` `DEFAULT_THETA`; the production θ is open
(`[A46 item 7]` withdrew H6's RED as a θ signal, #51)". Keeping the two-θ assertion is correct either
way.

#### M5 — `etaViolations` clause 4 compares a lower bound from the EXACT operator against a norm estimated on the HIERARCHICAL operator

**Where:** `src/core/sobolev/hierarchical/gmres.ts:711` (`lowerBound: xNorm > 0 ? axNorm / xNorm : 0`)
with `axNorm = norm(ax)` taken at `:694-696` straight after
`shiftedApply(probeOpts.exactApplyA, …)` — i.e. the χ=0 operator — while `normA2` comes from
`estimateNorm2((u,out) => applyAsigma(op,u,out), dim)` at `:649`, the hierarchical operator. Clause 4
(`:336`, documented at `:310-311` as "the estimate is below a free exact lower bound") therefore reads
`‖A_bct,σ‖₂_est < ‖A_exact,σ x̃‖/‖x̃‖`.

The hazard is bounded, which is why this is Minor: the same block's comment at `:646-648` cites
`lamMaxBct/lamMaxA` deviating ≤ 0.53 %, and a power-iteration shortfall is normally larger than that,
so the clause is dominated by the estimator and the cross-operator term adds ≤ ~0.5 % bias. But the
comparison is not between the quantity and its own lower bound, and a reader of `:310` would conclude
it is.

**Fix:** either compute `lowerBound` from the `ax` produced by `applyAsigma(op, x, ax)` three lines
later (same cost, strictly a lower bound for the operator `normA2` estimates), or record both
Rayleigh quotients and name the operator each belongs to in the TSDoc.

#### M6 — `adjacentSkipped` conflates "structurally adjacent" with "kernel value happened to be exactly 0"

**Where:** `src/core/sobolev/hierarchical/nearField.ts:424-427` (`if (kv === 0) { skipped++; continue; }`)
whose return is folded into the apply-level counter at `src/core/sobolev/hierarchical/apply.ts:333-334`,
which is the counter pinned at `3m` by `test/sobolev/hierarchical/apply.test.ts:1042`
(`expect(H.adjacentSkipped).toBe(3 * n)`).

The precompute-time counter (`nearField.ts:331-342`) is structural — it counts `isAdjacent` — and is
exactly right; the module TSDoc at `:27-32` explains why `3m` must be structural ("a counter pinned at
0 would read like a passing gate"). The apply-time counter is not: a kernel value that reaches exactly
0 is counted as an adjacency. In f32 the documented underflow route gives `Infinity`, not 0
(`fr(d**pLow)` flushing), but the low-order numerator `fr(fr(n1*n1)+fr(n2*n2))` can flush to 0 on
near-collinear edges, which would inflate the counter. The `3m` pin would then catch it — if it ran on
the f32 variant, which I could not find (`:1042` is the f64 path; `nearField.test.ts:206` uses
`pre32.adjacentSkipped`, the structural precompute counter).

**Fix:** return the `kv === 0` count as a separate `zeroKernelSkipped` field instead of adding it to
`adjacentSkipped`, so the `3m` identity stays structural at the apply level too; and add the `3m` pin
on the F32 variant.

#### M7 — the per-exponent-`pow` anchor's magnitude claim does not reproduce (its conclusion does, for a different reason)

**Where:** `src/core/sobolev/hierarchical/nearField.ts:34-39`: *"keep whichever form is here, do not
'simplify' to the other without re-running that pin, **because the two differ in f32 by more than the
H1-b band is wide**."* The two forms are the shipped `cross² / d**pLow` and the repo's low-order
`k24 = cross²/dNorm⁴; k24 / dNorm**pHi`.

MEASURED in f32 (`probes/powform.ts`, 12 004 samples over `d ∈ [1e-3, 1]` × four numerator
magnitudes): **mean relative difference 4.088e-8, max 2.387e-7.** The H1-b band at torusKnot n=960
θ=0.5 is `[1e-7, 5.7784e-5]`, i.e. 5.768e-5 wide — so the per-evaluation difference is **~240x
narrower** than the band, not wider. Even the aggregate, estimating √κ_near ≈ 9.7 random-sign terms
per row, is ≈ 4e-7: about 0.38x the measured `δ_precision` of 1.0515e-6 and ~144x inside the band
width. Swapping the form would **not** move H1-b out of band on this evidence.

The instruction itself is correct and should stay — it is just anchored on the wrong quantity. The
right anchor is `test/sobolev/hierarchical/nearField.test.ts:139-167`, which **is** the cross-form
pin: the oracle there is built with the *other* form (`:144-145`, `k24 / dNorm ** pHi`) and the shipped
`endpoint4Kernel`'s `d ** pLow` is pinned against it at `4e-15` in f64 (`:165-167`). That is a hard,
executed, 4e-15 bar, which is a far stronger reason not to touch the form than a band it clears by
two orders of magnitude.

**Fix:** replace the clause with "…without re-running `nearField.test.ts:139-167`, which pins the two
forms against each other at 4e-15 in f64; in f32 they differ by 4.1e-8 mean / 2.4e-7 max per
evaluation (MEASURED)".

---

## 3. Gates I examined and did NOT file against

Recorded so a re-review does not re-derive them.

| Gate | Can its RED side be reached? | Note |
| --- | --- | --- |
| **H1-a** (θ-monotonicity, `≤ 1.001`) | **Yes, executed.** `gateH1a`'s `invertAdmissibility` knock-out sets `demonstratesRedSideReachable: koRose` from the run itself | the circle tie is pre-registered and explicitly never folded into the conjunction (`[A38 item 5]`) |
| **H1-b** upper side | **Yes, executed.** f16 knock-out exceeds the bar by 45.61x (I reproduced it) | lower side is I2 |
| **H2** (`η_bct ≤ 50 u_f32`) | the 5 INVALID clauses are applied to BOTH η independently, and clause 5 exists because NaN defeats clauses 1–4 | the honest `limitations` entry — "H2 certifies BACKWARD error, not solution or direction quality" — is the correct scoping |
| **H3** | **Yes, executed by me** (M1). `primaryBar` is published as `NOT MEASURABLE AT THESE RUNGS` with the discharge pointer, not projected | quantisation predicate correctly published `VACUOUS` with its margin |
| **H4a** | `H4A_REF_ETA_BAR` and the τ are published; INVALID clause 2 is near-vacuous by the harness's own admission | disclosed, not hidden |
| **H4b′** | **No — and the harness says so**: `finaliseH4b` returns `VACUOUS` by construction unless `q ≤ 0` | correctly labelled; §7.6 is the amendment that made `q ≤ 0` RED rather than INVALID |
| **H4c** | conjunct 1's RED is `NOT ESTABLISHED` over a 0.9987–1.0034 spread — **and `gateH4c` files that against itself** (`spreadFinding: 'CONJUNCT 1 IS A WEAK GATE'`), plus a synthetic alpha-drift knock-out and a `thetaSaturation` finding | self-disclosed; nothing for me to add |
| **H5** | property 1 (`nullDim`/`λ₁`) **yes, executed** (3 knock-out rows, nullity 1/1/2); property 2 is I1; property 3 is published as RECORDED, not a licence | |
| **H6** | transcribed with `runHere: false`, RED withdrawn **with a control row** | the only consequence left dangling is M4 |
| **R1** | publishes `NOT ESTABLISHED` as a *value* wherever nothing binds, per §7.10:2558, and marks the per-shape wall `NOT MEASURABLE` with the A41 barrier | this is the pattern the rest of the set should copy |

**The best gate-power construction on the branch** is `schur.test.ts:519-562`: the `tol` sweep asserts
the RED side of the truth bar (`at(1e-8).err > BAR*100`, `at(1e-10).err > BAR`), then proves the
shipped `tol = 1e-12` sits within 4x of the measured error floor and that 1e-13/1e-14 are the same
floor. It also sets `etaAccept: Number.POSITIVE_INFINITY` so a loose tol is *observable* instead of
refused by the guard — the subtlety most knock-out harnesses miss. I quote it in I1 as the template
H5 property 2 should follow.

**Assertion census** across `test/sobolev/hierarchical/*.test.ts`: 910 `expect(` calls, of which 26
(2.9 %) match the weak-assertion shapes (`toBeGreaterThan(0)`, `toBeDefined()`, `not.toBeNaN()`). I
inspected all 26 and **none is a tautology standing in for a bar** — they are admissibility conditions
(`lowOrder.test.ts:284`, `2−s ∈ (0,2)` per `[RS §5.2]`), non-vacuity guards that prove a path was
exercised at all (`:462`, `nFar > 0`), matvec-only SPD stand-ins (`:607`, `precond.test.ts`), the
deliberate impossible-value guards the brief asks about (`apply.test.ts`'s
`expect(e).toBeGreaterThan(0)` on the bit-level pin and
`expect(differFromRoundOnce).toBeGreaterThan(bct.stats.nFar)` on the rounding-placement pin), and
three timing non-no-op guards whose weakening is documented with its reason
(`lowOrder.test.ts:975-981`: `msInline > msPre` was downgraded to logging because a GC pause inside
an 85 ms window can flip it).

The branch also defends its same-module differentials with measured mutation sensitivity rather than
with argument — e.g. `lowOrder.test.ts:456-460` compares percolation against row-major summation
(which shares the kernel) and records that "a uniform far-kernel ×1.01 moves this to ~1e-2" against
the 1e-13 bar, so the comparison is shown to be sensitive to the defect class it is aimed at.

Two differential tests I checked specifically for shared-path blindness (memory
`common-mode-blindness`) and found sound: `precond.test.ts`'s `extentOf` (deliberate second copy, with
the reason and the issue number), and `apply.test.ts`'s H1-b f64-vs-f64 `toBe(0)` check (second
prepared instance, not the same object). `precond.test.ts:~600-660`'s symmetry test is the best example
on the branch: it records that on mean-free vectors the test is **blind** to the outer `Π` (44/0 passing
with `precond.ts:195` deleted) and adds a general-vector draw that reads 3.3e-3 without it.

---

## 4. Commands run

All three required commands, in the order the brief lists them.

- **`bunx tsc --noEmit`** — exit 0, no output.
- **`bunx biome check`** — 200 files checked: **7 errors / 8 warnings / 7 infos, none of them in code
  this branch wrote.** The diagnostics are in `server.ts` (3), `test_gradient.ts` (6),
  `src/core/sobolev/phaseTimings.ts` (1), `test/store.test.ts` (1),
  `test/testConfigs-random.test.ts` (2), `src/index.tsx` (1), `src/scene/Curve.tsx` (1),
  `src/scene/Viewer.tsx` (3), `src/ui/ControlPanel.tsx`, and one **format** error in
  `bench/results/gpu-phase2a-cr.json`. `git log 9dd5551..HEAD -- <file>` is empty for each, and the
  JSON's format error is present in the **committed** HEAD blob too, so it predates the branch.
  **`bunx biome check src/core/sobolev/hierarchical test/sobolev/hierarchical` → clean, 27 files.**

  On the pre-commit hook: **`REVIEW-CONTEXT.md`'s original claim that the hook never lints was stale,
  and this branch is what made it stale.** Verified from the configuration, not from a hook exit code:
  `lefthook.yml:18` is `bunx biome check --write --no-errors-on-unmatched {staged_files}` with no
  `--linter-enabled=false`, and commit `f037cf1` ("fix(hooks): enable the biome linter in pre-commit")
  is the commit on this branch that removed the flag. I ran lint myself regardless, which is the
  durable practice either way.

  The residual caveat is narrower than the original one and still worth knowing: the hook is scoped to
  `{staged_files}` and runs `--write`, so its green tick certifies *the staged files, after autofix* —
  not the repo, and not a file whose diagnostics appear only because a different file changed. That is
  why the 7 pre-existing errors listed above survive a clean commit history.

  Worth recording as a positive: `lefthook.yml:7-11` justifies enabling the linter with
  *"A check that cannot fail is indistinguishable from one that passed"* (`@issue
  utof/repulsive-test2#70`). That is the same standard I1 and I2 are measured against — the branch
  wrote the rule down and then applied it to its own hook; those two findings ask only that H5
  property 2 and H1-b's floor be held to it as well.

  **One thing for whoever lands this branch, not a review finding:** the working tree carries an
  uncommitted 1 263-insertion / 314-deletion change to `bench/results/gpu-phase2a-cr.json` that
  belongs to no commit on the branch. It needs a decision (commit it with provenance, or revert it)
  before the branch merges — a bench artefact silently differing from the run that produced it is the
  `figures-carry-their-regime` failure.
- **`bun test test/sobolev/hierarchical`** (the branch's core, 14 files) —

  ```
   421 pass
   0 fail
   146330 expect() calls
  Ran 421 tests across 14 files. [2331.33s]
  ```

  **Zero failures.** 146 330 assertions over 421 tests — the density is the branch's own answer to the
  test-strength question. Logged instrument values sit where the comments say (`structuralTraceRel`
  3.0e-16…1.3e-15; schur `vsTruth` 1.1e-13…3.7e-12 against the 1e-10 bar; `sDefect` ≤ 8.9e-14;
  `min legitimate 1/cond(S) = 2.28e-2`).

- **`bun test` on the branch's three out-of-directory files**
  (`hierarchicalSeam`, `hierarchicalSaddleReal`, `denseSaddleBitGolden`) —
  `78 pass / 0 fail / 670 expect() calls across 3 files [89.72s]`. Note that
  `denseSaddleBitGolden`'s n=48 row re-derives `build.normA2` and `build.curveExtent` against
  independent recomputations and they agree exactly — another deliberate second-copy check.

  **Honesty note on the brief's "run it in the FOREGROUND":** the session's Bash tool has a hard 600 s
  foreground cap and auto-backgrounds past it, and the hierarchical suite needs 2331 s (the O(n³)
  dense references in `precond.test.ts` and `schur.test.ts` dominate, and the box was at load average
  7 from concurrent agents — memory `isolate-load-dependent-instruments`). The run above therefore
  completed in the background; I polled it to completion and am quoting its real summary line rather
  than a partial read. The whole-repo `bun test` (which includes pre-existing non-Phase-2H suites) was
  launched the same way; it is not a dependency of anything in this report, because every quantitative
  claim here comes from a probe I ran to completion myself.

---

## 5. Reproduction

Ten throwaway probes, written for this review and archived beside this report at
`docs/superpowers/plans/phase2h-review-ledger/probes/`. Run them with
`bun docs/superpowers/plans/phase2h-review-ledger/probes/<name>.ts` from the repo root (they resolve
the repo by absolute path, so cwd does not matter). Each imports only shipped modules plus
`test/sobolev/hierarchical/denseSpectrum.ts` (the instrument H5's shipped gate itself uses, per
`[A46 item 4]`), so none of them re-implements anything under review.

| Script | What it answers | Finding |
| --- | --- | --- |
| `h5probe.ts <n> <theta>` | `κ_excl`, `λ₁`, `λ₂` and the two precondition flags over `m̂ ∈ {30,45,60,70,85,100,300,1000}` | I1 |
| `h5probe960.ts` | the same at n=960 for `m̂ ∈ {30,70,85,100}`, θ ∈ {0.25,0.5}; reproduces four quoted MEASURED values | I1 + spot-check |
| `h5band.ts` | bisects the `λ₂ = 1.5` boundary at n=960 (m̂ = 66, 68, 69) | I1 |
| `h5ladder.ts` | the `κ_excl` ladder n=64…960 and the growth exponent, with the extrapolated crossing of 40 | I1 |
| `h3probe.ts` | H3's `G` and `band` for shipped / identity / `dropConstantMode` preconditioners at n=120,240,480, 5 seeds, τ_η=1e-12 | M1 |
| `h1bprobe.ts` | `δ_precision` and the `1e-7` floor margin at n=32…960, θ ∈ {0.25,0.5} | I2 |
| `h1bf16.ts` | the f16 knock-out overshoot and both forms of §7.11 item 7’s model at n=960 | spot-check |
| `ldot1.ts` | the four `max|L·(c·1)|` ranges in `lowOrder.ts`'s bit-exactness scoping | spot-check |
| `shiftorder.ts` | the f64 failure rate of reassociating the `σ·(Σu/n)` shift | verdict (a) |
| `powform.ts` | the f32 difference between the two algebraically-identical `pow` forms, against the H1-b band width | M7 |

They are measure-only, and several deliberately construct knock-out configurations — which is why
they live in the ledger rather than in `src/` or `bench/`. Note that `tsconfig.json` has no `include`
and `biome.json` includes `**`, so these files ARE in both gates' scope: I verified
`bunx tsc --noEmit` exits 0 and `bunx biome check docs/.../probes` is clean after adding them, so the
review leaves no new lint or typecheck debt. If a future cleanup wants them out of the gates' scope,
exclude the directory rather than deleting the files (`CLAUDE.md`: review-ledger workspaces are the
process record). `h5probe.ts` uses `(σ/n)·s` for the shift
where the shipped path uses `σ·(s/n)` (`gmres.ts:141-143`); this is immaterial at the 4 decimal places
quoted, and `h5probe960.ts` — the script every reproduced MEASURED value comes from — uses the shipped
order.
