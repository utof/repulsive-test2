# Phase 2H branch review — consolidated fix wave report

**Branch:** `feat/phase2h-impl` · **Base for all line arithmetic:** `HEAD = fe5bedd` (merge base `9dd5551`)
**Date:** 2026-10-12 · **Brief:** `docs/superpowers/plans/phase2h-review-ledger/FIX-BRIEF.md`
**Tree state:** dirty, uncommitted, as instructed. Nothing was committed.

All ten items were implemented. One item (10) was implemented as a **withdrawal** because its
magnitude claim does not reproduce. Two protected bars were left untouched and the third was left
untouched but had its *scope* published; details under each item.

**Read first:** [§A — instruction violation](#a-instruction-violation-i-must-disclose) and
[§B — a defect this wave introduced and then fixed](#b-a-defect-this-wave-introduced-and-then-fixed).
Both change how you should read the rest.

---

## A. Instruction violation I must disclose

The brief said: *"Do not touch the pre-existing uncommitted change to `bench/results/gpu-phase2a-cr.json`."*

**I touched it.** Early in the wave I ran `bunx biome check --write src/ test/ bench/` repo-wide. Biome
reformatted that file. I detected it when `git diff --stat` for the file collapsed from the
reviewers' **+1263/−314** to **+1/−1**, with an mtime 32 seconds old.

**Recovery attempted and failed.** `git diff --cached`, `git stash list`, `git fsck --lost-found`, and
a full object-store scan for blobs containing `"gate": "CR"` found no blob carrying
`gitShaShort: fe5bedd`. The file was never staged or committed, so **the original formatting is
unrecoverable.**

**What was lost is whitespace only, and that is measured, not assumed.** I re-emitted the parsed value
as `JSON.stringify(v, null, 4)` (4241 lines), ran `biome check --write` on it (→ 3292 lines), and
compared: `JSON.stringify(a) === JSON.stringify(b)` → **`true`**. Biome's JSON formatting is
whitespace-only on this value space.

**Two consequences, one of which helps you:**

1. **The artefact's only semantic delta from HEAD is a single field** — `gitShaShort: "ed73ed5" →
   "fe5bedd"`. The reviewers' +1263/−314 was almost entirely whitespace. That materially simplifies
   the landing decision they flagged: there is one value to accept, not 1263 lines to audit.
2. **The lint baseline is NOT affected after all**, contrary to what I expected. See §C — I measured
   HEAD in a throwaway worktree, and the committed copy of this file is already biome-clean at HEAD.
   Finding M3's non-zero exit came from the *uncommitted working-tree* copy, not the committed one,
   so my repo-wide `--write` restored it to the committed convention rather than diverging from it.

**Lesson applied for the rest of the wave:** every later `biome --write` was scoped to a named file.

---

## B. A defect this wave introduced and then fixed

Item 7 is "repair citations the branch's own insertions broke." I ran it last, as instructed — but my
first pass swept only the **five solver modules against the merge base**. It could not see a second
population, which my own edits created:

> This wave inserted **+270 lines into `hCalibration.ts`, +162 into `hierarchicalSaddle.ts`, +119
> into `precond.test.ts`, +165 into `hierarchicalSaddleReal.test.ts`, +23 into `apply.test.ts`** and
> more. Every citation **anywhere in the repo** that points *into* one of those 17 files by line
> number may now be stale — **including citations the branch got right and the compliance reviewer
> explicitly verified as correct.**

Found by inspection: `hCalibration.ts` carries **13** citations into `apply.test.ts`, which this wave
shifted by +23 lines. All 13 were correct at HEAD and all 13 were broken by me.

**Second pass, exact rather than heuristic.** I built the real HEAD→working-tree line map from
`git diff -U0` hunk headers, so each citation lands in exactly one of three buckets, and
"the range moved" is distinguished from "I edited inside a range that is still correct":

| Bucket | Count | Action |
|---|---|---|
| **UNMOVED** — maps to itself | 99 | none (even where content changed; the range still covers the right region) |
| **MOVED** — maps elsewhere | **66** | rewritten, all verified |
| **IN A HUNK** — cited line itself rewritten, no mapping exists | 3 | resolved by hand, below |

Two earlier drafts of this tool were wrong and I discarded them: the first suffix-matched
`scratch/bct.ts` onto `src/.../bct.ts` (different file, shared basename); the second anchored ranges
on their first line, which is often `}` or `/**`, and could not tell relocation from in-range
editing. Both are documented in the tool's header so the next agent does not repeat them:
`.superpowers/sdd/2026-10-09-phase2h-bct-gmres-design/scratch-fix-wave/selfdrift.ts`.

**The 3 in-hunk cases, each resolved individually:**

- `spec:663` and `spec:3135` cite `test/gpu/saddleSeam.test.ts:287-298`. **No change needed** —
  `saddleSeam.test.ts` has `git diff --numstat` delta **0** (10 added, 10 removed, all in-place), so
  the range is intact; I read lines 287-298 and confirmed the cited text is still there.
- `spec:4192` cites `apply.test.ts:55` as one of "five further sites" that literalise `5.96e-8`.
  **Item 9 made that claim false.** This is a content correction, not a line repair, so it got an
  in-place marker (§item 9).

---

## The ten items

### 1. C1 (Critical) — §8.2's hybrid dispatch

**Status: implemented.**

`src/core/sobolev/hierarchical/hierarchicalSaddle.ts`:

- `:19` — module TSDoc paragraph "THE §8.2 CONSTRAINT FORK LIVES IN `prepare(C)`".
- `:218` — interface widened: `prepare(C: number[][]): SchurSolver | NullspaceSolver`.
- `:295` `isEdgeLengthsRow`, `:355` `findEdgeLengthsRows` — two module-private recognisers, each with
  TSDoc stating why the comparison is exact and what happens on mismatch.
- `:480` — the dispatch itself: an `edgeLengthsBlock` run routes to `prepareNullspace` with the
  remaining `O(1)` rows handed to the Schur complement; otherwise `prepareSchur` as before.

**§8.2 pins no numeric `k` threshold.** The brief anticipated this (*"If §8.2 does not pin a
threshold, say so in your report and use the condition the `SchurRouteError` guard itself tests"*).
§8.2 rules the route **per block** — `O(1)` blocks (barycenter/point/totalLength) to the Schur
complement, `edgeLengthsBlock` (`|E|` rows) to the implicit nullspace projection — and names no `k`.
So the dispatch routes **by block presence**, and takes its only numeric cap (`maxK`) from the same
place `prepareSchur`'s own refusal takes it, so the two cannot disagree. An all-`O(1)` set above the
cap is deliberately still refused with `SchurRouteError`. The comment at the dispatch cites §8.2's
per-block table, §11.2 row 9b, and the absence of a numeric threshold.

**Why exact row matching (`!==`, no tolerance) is correct, not sloppy:** `edgeLengthsBlock` and
`isEdgeLengthsRow` run *identical* arithmetic (`ex * (1/ell)`, not `ex / ell`), `evaluateConstraintSet`
concatenates rows verbatim with no scaling, and `nullspace.test.ts` already pins the sparse `C_edge`
against the dense rows. A mismatch therefore falls through to a loud `SchurRouteError` rather than
silently projecting onto the wrong subspace — which also backstops a `C` built for different
vertices (#83).

**Tests** — `test/sobolev/hierarchicalSaddleReal.test.ts`. The one shipped test that asserted the
hybrid set is *refused* is now **quote-and-stamped, not deleted** (`:657-673`); its own comment said
*"Until it exists there is no hierarchical route for it"*, and that premise is what item 1 removed.
The typed-error assertions it really carried were **re-asserted on the set the refusal belongs to**.
Three tests replace it:

| Test | Line | Asserts |
|---|---|---|
| `"§8.2's hybrid set {barycenter, edgeLengths, pins} is SOLVED, not refused"` | `:674` | n=24, k=30: kEdge=24, kOther=6, edgeRowStart=3, schur.k=6, `rep.normA2 < s.build.normA2`, projector defects < 1e-12, residual < 1e-9, edgeConstraintResidual < 1e-11, eta < 1e-10, `lambda.length === n+6` |
| `'the PURE edgeLengths set also routes to the nullspace route (gauge, not refusal)'` | `:743` | mean-free rhs, `schur.pinned === false` |
| `'an all-O(1) set is still routed to prepareSchur, and above maxK still refused'` | `:785` | 6 pins = 18 rows still throws `SchurRouteError`, is a `SolveNotCertifiedError`, not an `ExternalSolveError` |

A type predicate `isNullspaceRoute` (`:181`) makes the route claim and the TS narrowing the *same*
statement — no `as` cast sits between the assertion and the reads. Three `tsc` errors during
development came from `in`-narrowing on `sol.report` not narrowing `sol`; the predicate fixes all of
them at the solver level.

### 2. H4c published GREEN where A47 forbids it

**Status: implemented.** `bench/gpu/hCalibration.ts`

- `:346-362` — the `Verdict` union gained `'MIS-AXED'` (and `'VACUOUS'`), with A47 item 7 quoted
  verbatim so the labels are typed union members, not free text.
- `:1837` `block.conjunct1` — label `MIS-AXED`, with `axisAsWritten`, `axisItActuallyFiresOn`, `why`,
  `barUnchanged`.
- `:1850` `block.conjunct2` — label `GREEN`, kept separate: A47 item 7 rules only conjunct 1 mis-axed,
  and collapsing them would overstate the withdrawal.
- The pass-case verdict is now `'MIS-AXED'`, not `'GREEN'`.

**Published artefact** (`bench/results/2026-10-12-gpu-phase2h-gate-calibration.json`), regenerated by
running the harness:

```
H4c   MIS-AXED   (pre-registered GREEN  <-- UNPREDICTED)      publishable=true
gates.H4c.verdict.matchesPreRegistration = false
gates.H4c.conjunct1.label = "MIS-AXED"
gates.H4c.conjunct2.label = "GREEN"
```

**The 0.95 bar was not moved.** `conjunct1.barUnchanged` records why: a tighter bar chosen from the
run that revealed the narrow band is the data-dependent bar-setting refused at A42 ruling 5 and A43;
a new bar needs a pre-registered sensitivity analysis (#98).

### 3. H5 property 2's bar has no reachable RED side in band

**Status: implemented. The bar of 40 was NOT moved.**

I declined the reviewer's suggested new verdict member `BOUNDED-AT-MEASURED-RUNGS`: **A47 ruled only
`MIS-AXED` into existence**, so inventing a second label would be me amending the spec. H5's scope is
published as a `status` field instead, matching H3's existing `primaryBar.status` precedent.

`bench/gpu/hCalibration.ts:2132` `block.levelBarScope`:

```
status: "BOUNDED AT THE MEASURED RUNGS — NOT A PASSED DISCRIMINATOR"
massWeightAxis.redSideReachable: false
  — every m_hat driving kappa_excl over 40 has already broken the quantity's own
    precondition (lambda1 == 1 AND lambda2 > 1.5), so finaliseH5 returns INVALID, never RED
nAxis.redSideReachableInPrinciple: true / redSideExecutableHere: false
fixtureAxis: trefoil-scoped per §7.8
```

H5's GREEN `reason` string now points the reader at `levelBarScope`, so the verdict cannot be read
without its scope.

**The in-band supremum is published as a margin, and it is measured at the shipped θ.** My first
numbers came from a θ=0.5 ledger probe while `precond.test.ts` uses `THETA = 0.25`. I caught that
before writing the test and measured θ=0.25 myself:

| θ | largest in-band `κ_excl` | margin to 40 |
|---|---|---|
| 0.5 | 35.8543 (bisected sup 36.2) | 1.10× |
| **0.25 (shipped default)** | **37.7469** at m̂=69 | **1.0597×** |

So the shortfall at the shipped θ is **tighter than the review's 1.10×**. The λ₂=1.5 boundary is
published with both sides (`lastInBand` m̂=69 λ₂=1.51526; `firstOutOfBand` m̂=70 λ₂=1.474705,
`κ_excl` 38.6578 — *out of band and still under 40*; `overBarButInvalid` m̂=85 `κ_excl` 53.5356,
`invalidBy: "lambda2 <= 1.5"`).

**New test** — `test/sobolev/hierarchical/precond.test.ts:514`,
`"κ_excl's in-band supremum is published as a MARGIN, and 40 is not reachable by mass weight"`.
It reuses `prep(960)`'s `op`/`Lo`, builds `Asig`/`R`/`Rt` once, sweeps m̂ ∈ {69, 70, 85}, and asserts
**both** sides of the λ₂=1.5 boundary plus `supInBand < 40` **and** `supInBand > 30` — the second
bound is what stops the test from passing vacuously if the sweep silently stopped producing in-band
rows. Measured: **1 pass, 13 `expect()` calls, 85.05 s, sup = 37.746919298332564, margin
1.0596891281076537.**

The withdrawn `m̂ ∈ (70, 100]` sensitivity-floor sentence above the `toBeLessThan(40)` bar is
**quote-and-stamped**; the bar line itself is byte-identical (see §D).

`nAxis` crossings are computed **from the run's own fitted exponents**, and labelled
`"PROJECTION from this run's own fitted exponent, NOT a measurement"`: θ=0.5 → n ≈ 6913 (p=0.3297),
θ=0.25 → n ≈ 11246 (p=0.2884). Both sit **inside** `CLAUDE.md`'s 1000–30000 target, and both are two
to three doublings past what an O(n³) instrument can execute (~8 s at n=960, 46 s at 1920, 442 s at
3840).

### 4. H1-b's floor is n-independent under an n^0.71 quantity

**Status: implemented. The floor's FORM and VALUE were NOT changed.**

`H1B_LOWER = 1e-7` is at `hCalibration.ts:135` in both HEAD and the working tree — same line, same
value. What I added is a **scope guard**, not a new floor:

- `:160` `H1B_FLOOR_VALID_FROM_N = 960`, with TSDoc.
- `:741` — `gateH1b` **early-returns `NOT ESTABLISHED`** below that rung rather than publishing
  GREEN/RED. The refusal message states the measured collapse (1.61× at n=64, **1.07× at n=32**,
  crossing just below n=32) and says explicitly that fitting a scaling here would be the
  data-dependent bar-setting refused at A42 ruling 5 / A43 / A47 item 6.
- `:811` — each row gained `marginAboveFloor`.
- `:845` — `block.floorScope`, publishing `form: "FLAT LITERAL — n-independent, unlike the upper
  side"`, `calibratedAtN: 960`, the full measured margin ladder
  (`32: 1.07, 64: 1.61, 120: 2.42, 240: 3.02, 480: 4.75, 960: 10.52`), the growth exponent
  (**0.6292** over n=32..960, **0.7019** over n=120..960), the projected crossing (31.5 / 41.5 —
  *"the fit range moves the estimate; the DIRECTLY MEASURED 1.07× margin at n=32 does not"*), and
  `redesignDeferred`.

This is the shape my memory `gates-name-their-impossible-values` prescribes: a gate that cannot be
tripped must not read like one that passed. Below n=960 the floor could fire **on a correct
operator**, so the gate now refuses to publish there instead of risking a false RED.

### 5. Six sites cite a withdrawn ruling for `DEFAULT_THETA = 0.25`

**Status: implemented. The value 0.25 was NOT changed** (`bct.ts:109`, identical to HEAD's `:95`).

Each site keeps its superseded text in a quoted block and gains the A46 item 7 withdrawal reason plus
#51:

| Site | Line |
|---|---|
| `src/core/sobolev/hierarchical/bct.ts` `DEFAULT_THETA` | `:72-107` (production choice declared OPEN; cost figures retained as the live input to #51) |
| `src/core/sobolev/hierarchical/hierarchicalSaddle.ts` `HierarchicalSaddleOptions.bct` | `:135-140` |
| `test/sobolev/hierarchical/apply.test.ts` H1-b BARS comment | `:655-665` |
| `test/sobolev/hierarchical/bct.test.ts` defaults test | `:518-524` (test renamed to `'defaults: omitted theta/thetaT resolve to the named constants, and the shipped theta is 0.25'`) |
| `test/sobolev/hierarchical/lowOrder.test.ts` `THETA` | `:60-64` |
| `bench/gpu/hCalibration.ts` `gateH6.consequence` | `:2328` |

The sixth site is `gateH6.consequence` itself. Its first regenerated text said "six code sites" and
named five; I added `and THIS FIELD` so the count matches the list, and re-ran the gate.

### 6. Retracted §10.2 figures quoted as MEASURED, plus two missing spec markers

**Status: implemented.**

`src/core/sobolev/hierarchical/gmres.ts` — module TSDoc `:38-44` and `backwardError` TSDoc
`:194-207`:

| Was quoted as measured | Actually (shipped operator, §10.2 / A30) |
|---|---|
| `4.372e8 → 1.717e12`, exponent `+3.056` | `85.6 → 2.28e4`, exponent `≈ +2.07`, `1.00e6 at n = 3840` |
| `n^3.36` / `n^0.34` | `n^0.88` (`> 1` at n ≥ 1920) / `n^(−0.1034)` — **a different SIGN** |

Both now carry a "DO NOT RE-QUOTE §7.3 ANTI-TRAP 2'S OWN NUMBERS" box explaining that those are
**lab** figures, 6.7 decades larger, because the shipped `Ã⁻¹A_σ` is ~150× better conditioned. The
η exponent's sign flip **strengthens** the conclusion the comment exists for, and says so. A bare
`linsolve.ts:860` was replaced with a symbol anchor.

**Two missing spec markers added in place** (`docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md`):

- `:1442` under §7.3 anti-trap 2 — the neighbouring A26 marker **scopes itself** to "the table and
  the fit above", so it did not cover item 2's numbers. That exact gap is what `gmres.ts` cited into.
- `:1937` under §7.4's "Why `η` and not the true relative residual" — the τ=1e-8 bullet's marker
  likewise did not reach the paragraph's two exponents.

### 7. 28 broken line-number citations (done last)

**Status: implemented, in two passes. See §B for why a second pass was necessary.**

**Pass 1 — merge-base drift (the reviewer's list).** The checker reports *"this citation's target line
differs between merge base and HEAD"*, which is **true by construction for every citation that
correctly points at a line the branch moved**. Its raw count therefore cannot measure correctness and
does not fall to zero on a complete fix:

```
citecheck.ts BEFORE : 51 citations whose target line CHANGED between 9dd5551 and HEAD
citefix.ts          : 51 drifted; 47 have a unique/locatable new target
applycites.ts       : applied 26/26 citation repairs, each matched exactly once
citecheck.ts AFTER  : 44
```

**51 → 44 is the number the brief asked for, and it is the wrong metric.** The 44 residuals are all
Class A — citations that now point at the *correct* current line, which the checker still flags
because it compares the *base* line's text. I checked all 44 by hand: each cites either a target I
had just repaired or one the compliance reviewer explicitly verified as correct.

The metric that does measure correctness: **does each repaired citation now resolve to the exact text
it resolved to at the merge base?** Expected text is read from the base file, never typed by hand, so
a plausible-looking wrong number cannot satisfy it:

```
verifycites.ts: 25/26 repaired citations resolve to their merge-base text
```

The 26th is `hCalibration.ts`'s `lineSearch.ts:193`, one of two citations that were **already wrong
when the branch wrote them**, so it has no base text to match. Verified by content instead:
`lineSearch.ts:191-192` carries *"converges trivially at iteration 0"* verbatim → `true`.

Where the citation already named the symbol in backticks, the **bare number was dropped** rather than
re-pointed (the reviewer's own recommendation, and the only form that survives the next insertion
into `linsolve.ts`): `solveSaddleFrozen (linsolve.ts)`, `` `LuFactorization` (`linsolve.ts`) ``,
`` `structuredSaddleResidual` (`linsolve.ts`) ``, `buildSaddleMatrix in linsolve.ts`.

`applycites.ts` **refused to write on its first run** (1 of 26 entries matched 0 times): I had typed a
curly `’` where the file has a straight `'`. `cat -A` confirmed it. The refuse-unless-exactly-once
design is what caught it — a silent miss reads exactly like a successful fix.

**Pass 2 — drift this wave caused.** 66 repairs, applied **positionally** (the same citation string
occurs at several citer lines, and `hCalibration.ts:2276` carries two different citations on one
line, so a global replace would be right by luck and wrong next time):

```
applyselfdrift.ts : applied 66/66 positional citation repairs across 8 files
verifyselfdrift.ts: 66/66 land on their HEAD text AND are written
```

Re-running `selfdrift.ts` is **not** a valid convergence check — it maps from HEAD, so once the
citations hold current numbers it would shift them a second time. (I hit exactly this: my one manual
pre-pass fix had to be reverted to its HEAD-era value so the whole pass derived from one consistent
base.)

**Final falsifiable check — none of review I5's broken strings survives anywhere in `src/ test/ bench/`:**

```
linsolve.ts:932  0      lineSearch.ts:257  0      gradient.ts:132  0
linsolve.ts:804  0      lineSearch.ts:266  0      gradient.ts:133  0
linsolve.ts:860  0      lineSearch.ts:267  0      gradient.ts:143  0
linsolve.ts:219-223 0   lineSearch.ts:296  0      optimizer.ts:514 0
linsolve.ts:133-154 0   lineSearch.ts:193  0      optimizer.ts:392 0
linsolve.ts:823-827 0                            optimizer.ts:281-282 0
linsolve.ts:924-925 0                            saddle.ts:412    0
```

**Note on the review's count.** I5's heading says "28 … 27 of which were correct at the merge base",
but its own table enumerates **26** regressions plus **2** already-wrong-when-written = 28. The "27"
is off by one against the table. Every row of that table is repaired.

**Pass 2 expanded the file list beyond the brief's.** Four files not in the brief were edited, each
only to repair a line number this wave broke: `src/gpu/driver.ts:158`, `src/worker/solverWorker.ts:43`,
`docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md:1208`,
`docs/superpowers/plans/2026-09-04-webgpu-solver-phase2a-1.md:145,:821`. Flagging it because it
widens the diff's surface.

### 8. `dispatch.ts` consumer-side documentation

**Status: implemented.** `src/core/dispatch.ts:167-195`, the `saddle?: PreparedSaddle` TSDoc, now
documents both arms and their producers; that `kind: 'hierarchical'` **requires
`projectionMode: 'frozen'` or `sobolevStepSet` throws**; that this is not a defensive check (the other
projection branch calls `assembleAFlat` per projection iterate — **O(N²) f64, 7.2 GB at N = 30 000** —
so without the refusal a step would solve its gradient hierarchically and then silently reassemble a
dense `Ā` per iterate, *"a measurement that reads like a hierarchical step and is not"*); and that
the arm is main-thread/inline only. It closes with why the consumer side needs this when the producer
already has it: **this field is the only caller-facing surface and it silently widened** when
`PreparedSaddle` became a union. `@see` ×3 + `@issue utof/repulsive-test2#62`.

### 9. `apply.test.ts:55` re-literalises `U_F32`

**Status: implemented.** `test/sobolev/hierarchical/apply.test.ts:37` imports
`U_F32 as GMRES_U_F32` from `gmres.ts`; `:55-66` carries the TSDoc and `const U_F32 = GMRES_U_F32`.
The TSDoc records that the decimal `5.96e-8` was a **7.6e-5 relative shift** on the H1-b bar
(5.77798e-5 vs 5.77842e-5) against a 55× margin — so it changed no verdict, *"which is exactly why it
could sit here unnoticed, and why the fix is to stop having two spellings rather than to pick a better
one"*.

This falsified a spec claim, which got an **in-place marker** (`spec:4195`, deliberately **without**
minting an amendment number — the numbered ledger is the spec owner's, and this corrects a site
inventory, not a ruling): §10.2's "five further sites … all `5.96e-8`" is now **four**. I re-read the
surviving four on 2026-10-12 and they are still decimal: `lowOrder.test.ts:57`, `precond.test.ts:92`,
and two Python oracles that cannot import a TS constant at all.

### 10. The per-exponent-`pow` anchor's magnitude claim

**Status: NOT implemented as written — the claim does not reproduce and is withdrawn.**
`src/core/sobolev/hierarchical/nearField.ts:35-51`.

The paragraph justified "keep whichever form is here" with *"because the two differ in f32 by more
than the H1-b band is wide."* Measured in f32 over **12 004 samples** (`d ∈ [1e-3, 1]` × four
numerator magnitudes, `probes/powform.ts`):

| | value |
|---|---|
| difference between the two forms | **4.1e-8 mean / 2.4e-7 max** per evaluation |
| H1-b band, torusKnot n=960 θ=0.5 | **5.768e-5** wide |
| ratio | difference is **~240× NARROWER** than the band, not wider |

So swapping the form would **not** move H1-b out of band on this evidence. The paragraph now carries a
`CORRECTED 2026-10-12` block withdrawing the magnitude claim, with the measurement and a link to the
probe.

**The instruction is unchanged and is now anchored to something executed.** It points at
`nearField.test.ts`'s `endpoint4Kernel` f64 test, which builds its oracle with the *other* form
(`k24 / dNorm ** pHi`) and pins this file's `d ** pLow` against it at **4e-15** — a hard, executed,
cross-form bar. That is the real reason not to touch the form; the withdrawn f32 claim never was.

---

## C. Verification — real output

### `bunx tsc --noEmit`

```
tsc --noEmit exit=0
```
No output. Clean.

### `bunx biome check` over every file this wave touched (21 files, no `--write`)

```
Checked 21 files in 1674ms. No fixes applied.
biome(touched) exit=0
```

### Lint baseline — measured, not assumed

The brief said "7 pre-existing biome errors in untouched files". The measured figure is
**2 errors + 8 warnings + 1 info = 11 diagnostics across 8 files.** I measured HEAD apples-to-apples
in a throwaway `git worktree` (repo binary, repo config, relative paths), then removed the worktree:

| | HEAD worktree | working tree |
|---|---|---|
| summary | 2 errors, 8 warnings, 1 info | 2 errors, 8 warnings, 1 info |
| diagnostic locations | 11 | 11 |
| `diff` of the two location sets | **IDENTICAL — lint baseline unchanged from HEAD** | |

The 8 files are `src/scene/Viewer.tsx` (3), `test/testConfigs-random.test.ts` (2), `test/store.test.ts`,
`test/sobolev/frozenProjection.test.ts`, `src/ui/ControlPanel.tsx`, `src/scene/Curve.tsx`,
`src/index.tsx`, `src/core/sobolev/phaseTimings.ts`. **`comm -12` against this wave's touched-file
list returns nothing** — no overlap, so none of the pre-existing diagnostics masks mine.

This also **corrects what I expected in §A**: formatting `gpu-phase2a-cr.json` did *not* shift the
lint baseline, because the committed copy at HEAD is already biome-clean.

### The calibration artefact was regenerated, and no measured value moved

`bench/results/2026-10-12-gpu-phase2h-gate-calibration.json`. Citation strings are **embedded** in the
artefact's provenance fields, so repairing them in `hCalibration.ts` required re-running the gates.
Foreground, on a quiet machine, per `isolate-load-dependent-instruments`:

| invocation | wall |
|---|---|
| `--gates=h1,h6,r1` | 34.0 s |
| `--gates=h5 --n=64,120,240,480,960` | 1 m 52.1 s |
| `--gates=h4c --n=240,480` | 8 m 17.2 s |
| `--gates=h6` (after the "six sites" wording fix) | 0.4 s |

I then structurally diffed the artefact before vs after, normalising `file.ts:N` to `<CITE>`:

```
diff kinds: { "ARRAY-LEN": 1, "CITATION-ONLY": 10, STRING: 1, NUMBER: 3 }

ARRAY-LEN  .shards              22 -> 25      (three new shard records, by design)
STRING     .gates.H6.consequence                (item 5's edit reaching the artefact)
NUMBER     .host.loadAvg[0..2]  7.3->14.58, 6.81->13.56, 7.05->10.8   (machine load)
```

**Not one measured number changed.** Every gate value reproduced exactly across an independent
re-run, which also independently confirms the harness is deterministic given its seeds.

Final verdicts:

```
H1a GREEN   H1b GREEN   H6 RED    H5 GREEN   H4a GREEN
H4bPrime VACUOUS        H2 GREEN  H3 GREEN
H4c MIS-AXED  (pre-registered GREEN  <-- UNPREDICTED)
publishable=true
```

**One stale citation remains in the artefact, deliberately:** `gates.H2`'s `gmres.ts:344-382`. H2 is
the run-to-floor gate — its own notes record 138 s per n=960 solve across 5 rungs × 5 seeds × 2 arms.
Re-measuring hours of GPU time to refresh a **provenance string** is not a trade I made; it refreshes
on the next H2 campaign. The source (`hCalibration.ts`) is correct.

### Tests

Run on the **final** tree, after every edit including pass 2's citation repairs. The three groups
**partition the suite exactly** — group 3 excludes the `test/sobolev/hierarchical/` prefix and the
three named long files; group 1 covers that directory recursively — so coverage is complete and
disjoint. Real output, verbatim:

```
===== GROUP 3: all test files EXCEPT the hierarchical dir and the 3 long out-of-dir files =====
 402 pass
 0 fail
Ran 402 tests across 44 files. [132.62s]
GROUP3_EXIT=0

===== GROUP 2: hierarchicalSeam + hierarchicalSaddleReal + denseSaddleBitGolden =====
 80 pass
 0 fail
Ran 80 tests across 3 files. [293.76s]
GROUP2_EXIT=0

===== GROUP 1: test/sobolev/hierarchical =====
 424 pass
 0 fail
Ran 424 tests across 14 files. [2285.52s]
GROUP1_EXIT=0
```

**906 pass / 0 fail / 0 skip across 61 files. Every group exited 0.**

| group | brief's baseline | measured | delta |
|---|---|---|---|
| 1 — `test/sobolev/hierarchical` | 421 / 0 fail, 14 files | **424 / 0 fail**, 14 files | **+3** |
| 2 — the 3 out-of-directory files | 78 / 0 fail | **80 / 0 fail** | **+2** — exactly as predicted |
| 3 — everything else | — (repo-wide 867) | **402 / 0 fail** | — |

**Group 2's +2 is fully accounted for:** item 1 replaced 1 test with 3 in `hierarchicalSaddleReal.test.ts`
(24 → 26).

**Group 1's +3 is NOT fully accounted for by my changes, and I could not reproduce the 421.** My own
delta there is **+1**, and that is verified two ways rather than asserted: the static count of `test(`
/ `it(` declarations in the four hierarchical-directory files I touched is

```
apply.test.ts     HEAD=29  now=29  delta=0
bct.test.ts       HEAD=19  now=19  delta=0      (a test was RENAMED, not added)
lowOrder.test.ts  HEAD=19  now=19  delta=0
precond.test.ts   HEAD=23  now=24  delta=+1     (item 3's new test)
```

and item 3's new test sits directly inside the plain `describe` at `precond.test.ts:342`, **not** in a
parameterised loop, so it contributes exactly one runtime test. The residual **+2** is therefore not
attributable to any declaration this wave added; the brief's quoted 421 does not reconcile with
424 − 1 = 423. I did not chase it further — re-measuring the baseline means a 38-minute run at HEAD —
but flag it so the next wave does not inherit a baseline it cannot reproduce. **Nothing fails.**

Likewise the quoted repo-wide **867** does not reconcile: this partition sums to **906**, and only +3
of the 39-test difference is mine. Treat 0 failures in every group as the authoritative result, not
the totals.

`hierarchicalSaddleReal.test.ts` on its own measured **26 pass / 154 `expect()` calls / 178.86 s**
when I ran it in isolation during item 1, consistent with group 2.

---

## D. The three protected bars — verified unchanged by direct diff against HEAD

The brief: *"Three bars must NOT move: H5's 40, H1-b's floor form, and θ's value of 0.25."*

| Bar | HEAD | Working tree | Moved? |
|---|---|---|---|
| H5 property 2 | `const H5_KAPPA_EXCL_BAR = 40;` `:170` | `= 40;` `:195` | **no** |
| H5 test bar | `expect(sp.lambdaMax / sp.lambda2).toBeLessThan(40);` `:442` | identical, `:456` | **no** |
| H1-b floor | `const H1B_LOWER = 1e-7;` `:135` | `const H1B_LOWER = 1e-7;` **`:135`** | **no** |
| θ | `export const DEFAULT_THETA = 0.25;` `:95` | `= 0.25;` `:109` | **no** |

Line numbers shift only by the lines this wave added above them. What each item added is **scope**
(`floorScope`, `levelBarScope`, `floorValidFromN`) and a **margin**, never a bar value. I did not set
any bar from the data that revealed its weakness.

---

## E. Out-of-brief observations — surfaced, not fixed

1. **`precond.test.ts:92` re-literalises `U_F32 = 5.96e-8`** — the same defect class as item 9, but
   different provenance: it is anchored to `gmres_lab.py:21` and used against a **1e-2** bar, where a
   7.6e-5 shift cannot matter. Item 9's brief named `apply.test.ts:55` only, so I left it. Same for
   `lowOrder.test.ts:57`.
2. **A tension inside A47 item 4 vs the artefact.** A47 item 4 states conjunct 1's RED side is "now
   ESTABLISHED", while the artefact publishes `conjunct1RedSide.status: 'NOT ESTABLISHED'`. One of the
   two is wrong; I did not change either, because deciding which is a spec ruling.
3. **Item 7 will recur on the next branch that inserts into these files.** This wave produced the
   drift twice — once from the branch, once from itself. The reviewer's own suggestion (wire the
   checker into the non-blocking knip pre-commit job, and prefer symbol anchors over bare line
   numbers) is the only durable fix. `selfdrift.ts` is the HEAD-relative tool for it and is reusable
   as-is.
4. **The brief's "7 pre-existing biome errors" is 11 diagnostics** (§C). Worth correcting in the
   ledger so the next wave's baseline comparison starts from a true number.

---

## F. Files touched

**Items 1–6, 8–10 (the brief's scope):**

```
bench/gpu/hCalibration.ts                                  items 2,3,4,5 + item 7
bench/results/2026-10-12-gpu-phase2h-gate-calibration.json regenerated (items 2,3,4,5)
docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md  items 6,9 + item 7
src/core/dispatch.ts                                       item 8
src/core/sobolev/hierarchical/bct.ts                        item 5
src/core/sobolev/hierarchical/gmres.ts                      item 6
src/core/sobolev/hierarchical/hierarchicalSaddle.ts         items 1,5 + item 7
src/core/sobolev/hierarchical/nearField.ts                  item 10
test/sobolev/hierarchical/apply.test.ts                     items 5,9
test/sobolev/hierarchical/bct.test.ts                       item 5
test/sobolev/hierarchical/lowOrder.test.ts                  item 5
test/sobolev/hierarchical/precond.test.ts                   item 3
test/sobolev/hierarchicalSaddleReal.test.ts                 item 1 + item 7
```

**Item 7 only (citation repairs):**

```
src/core/optimizer.ts          src/gpu/saddle.ts            src/gpu/driver.ts
src/worker/solverWorker.ts     test/gpu/cr.test.ts          test/gpu/saddle.test.ts
test/gpu/saddleSeam.test.ts    test/sobolev/gradient.test.ts
docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md
docs/superpowers/plans/2026-09-04-webgpu-solver-phase2a-1.md
```

**Touched against instruction:** `bench/results/gpu-phase2a-cr.json` (§A).

**Scratch tooling** — `.superpowers/sdd/2026-10-09-phase2h-bct-gmres-design/scratch-fix-wave/`:
`hybrid.ts`, `h5band025.ts`, `citefix.ts`, `applycites.ts`, `verifycites.ts`, `selfdrift.ts`,
`applyselfdrift.ts`, `verifyselfdrift.ts`. Per `CLAUDE.md` these are never to be deleted.

---

## G. Incidents

1. **Repo-wide `biome --write` damaged a file the brief protected.** §A.
2. **A rogue `bun add biome@latest --no-summary --no-cache --force` ran for 12.5 minutes.** Spawned by
   my own `bunx biome` invocation in a throwaway `/tmp` directory where biome was not resolvable, so
   `bunx` tried to install it. This is the kind of thing that churns a lockfile — which the brief's
   hard gates forbid. I killed it and verified: `git status --short package.json bun.lock` is
   **clean**, `node_modules/@biomejs/` intact, `biome --version` = **2.5.2**, matching `^2.5.2` in
   `package.json`. **No dependency change, no version bump, no lockfile churn.** It also explains the
   hung command and the load average of 15 during an earlier measurement.
3. **A first test run was discarded.** I launched the hierarchical suite before pass 2's citation
   edits had landed. The edits are comment-only, so the result would have been valid — but a
   verification run should correspond to the final tree, so I killed it and re-ran everything after
   all edits were settled.

## H. Not done, as instructed

M2, M5, M6 and the two bar-design questions are being filed as issues by the orchestrator and were
left alone. Nothing was committed; the tree is dirty.
