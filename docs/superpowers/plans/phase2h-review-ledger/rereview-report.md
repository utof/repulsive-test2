# Scoped re-review — round 2, branch `feat/phase2h-impl`

**Reviewer:** informed + scoped (round 2 of a capped loop). **Inputs:** `OPEN-FINDINGS.md`,
`fix-wave.diff`, `fix-wave-report.md`, and the repo at the uncommitted working tree vs `HEAD = fe5bedd`.
**Date:** 2026-10-12. Nothing was committed; no subagents were spawned.

Every verdict below was taken against the repo and against `git show fe5bedd:…`, never against the
implementer's report. Where the report's own claim and my measurement disagree, the measurement is
stated and the report's claim is marked corrected.

---

## 1. Per-finding verdicts

| # | Severity | Verdict | Evidence (file:line) |
|---|---|---|---|
| **C1** | CRITICAL | **ADDRESSED** | Dispatch exists at `src/core/sobolev/hierarchical/hierarchicalSaddle.ts:501-524`; recognisers at `:295` (`isEdgeLengthsRow`) and `:355` (`findEdgeLengthsRows`); interface widened at `:218`. `prepareNullspace` now has a `src/` caller (graph: `hierarchicalSaddle.prepare` → `prepareNullspace`, 3 callers total vs 2 test-only before). Three tests at `test/sobolev/hierarchicalSaddleReal.test.ts:674`, `:743`, `:785`. **I ran it: 26 pass / 0 fail / 115.05 s.** |
| **I1** | IMPORTANT | **ADDRESSED** | The self-contradiction is withdrawn in place at `test/sobolev/hierarchical/precond.test.ts:439-450` (quote-and-stamp; the m̂=100/70 demo is relabelled as the precondition firing, not the bar tripping). New in-band-supremum test at `:514-600` asserting **both** sides of the λ₂=1.5 boundary plus `supInBand < 40` **and** `> 30` (the anti-vacuity bound). Scope published at `bench/gpu/hCalibration.ts:2132` and in the artefact as `gates.H5.levelBarScope` — `massWeightAxis.redSideReachable: false` with the band boundary at both θ, `nAxis.redSideReachableInPrinciple: true` / `redSideExecutableHere: false` with the fitted-exponent crossings labelled *"PROJECTION … NOT a measurement"*, and `fixtureAxis` covering the finding's "`gateH5` only ever builds trefoil" half. `gates.H5.verdict.reason` now forward-references `levelBarScope`, so the GREEN cannot be read without its scope. |
| **I2-numerics** | IMPORTANT | **ADDRESSED** | `H1B_FLOOR_VALID_FROM_N = 960` at `bench/gpu/hCalibration.ts:160` with TSDoc at `:136-159`; `gateH1b` early-returns `NOT ESTABLISHED` at `:741-760`; `marginAboveFloor` at `:811` and present on both artefact rows (10.515, 10.496); `floorScope` at `:845` and in the artefact with the full margin ladder, growth exponent and projected crossing. |
| **I2-compliance** | IMPORTANT | **ADDRESSED** | `'MIS-AXED'` is a `Verdict` member at `bench/gpu/hCalibration.ts:362`; 15 occurrences in the harness, 8 in the artefact (was 0 in both). Artefact `gates.H4c.verdict.verdict = "MIS-AXED"`, `preRegistered = "GREEN"`, `matchesPreRegistration = false`, `conjunct1.label = "MIS-AXED"`, `conjunct2.label = "GREEN"`. `H4bPrime` remains `VACUOUS`, so the two labels are now handled alike. |
| **I3-compliance + M4-numerics** | IMPORTANT | **ADDRESSED** | All six sites re-anchored to `[A46 item 7]` + `#51`: `bct.ts:72-107`, `hierarchicalSaddle.ts:135-140`, `apply.test.ts:655-665`, `bct.test.ts:518-524`, `lowOrder.test.ts:60-64`, `hCalibration.ts:2326-2328` (incl. the `gateH6.consequence` field itself). The three surviving "H6 read RED" strings in `src/`/`test/` are all inside `>` quote-blocks of superseded text. |
| **I4-compliance** | IMPORTANT | **ADDRESSED** | `src/core/sobolev/hierarchical/gmres.ts:38-41` now labels `85.6 → 2.28e4 / ≈+2.07 / 1.00e6 at n=3840` as the MEASURED shipped values, and `:42-48` carries a DO-NOT-RE-QUOTE box naming `4.372e8…1.717e12` / `+3.056` as retracted lab figures; same treatment for `n^3.36` / `n^0.34` at `:197`. Spec markers added in place at `docs/…/2026-10-09-phase2h-bct-gmres-design.md:1442-1450` (§7.3 anti-trap 2) and `:1937-1944` (§7.4), both quote-and-stamp, neither deleting the superseded text. |
| **I5-compliance** | IMPORTANT | **ADDRESSED** (within the finding's own `src/`/`test/`/`bench/` scope) | All 19 distinct broken citation strings return **0** occurrences in `src/ test/ bench/` (my own 19-pattern sweep with digit boundaries). 10/10 self-drift repairs into `apply.test.ts` and 3/3 sampled merge-base repairs resolve to byte-identical text — detail in §3. My independent stale-citation sweep over every wave-modified file found **0** real residuals. |
| **I6-compliance** | IMPORTANT | **ADDRESSED** | `src/core/dispatch.ts:166-196` now documents both arms and their producers, the mandatory `projectionMode: 'frozen'` with the O(N²)-per-iterate reason it is not a defensive check, the main-thread-only constraint, and why the consumer side needs it; `@see` ×3 + `@issue utof/repulsive-test2#62`. |
| **M3-numerics** | MINOR | **ADDRESSED** | `test/sobolev/hierarchical/apply.test.ts:37` imports `U_F32 as GMRES_U_F32`; `:55-66` is TSDoc + `const U_F32 = GMRES_U_F32`. The decimal `5.96e-8` is gone from this file. Spec site inventory corrected in place at `spec:4195-4207`, and its `apply.test.ts:66` citation resolves exactly. |
| **M7-numerics** | MINOR | **ADDRESSED as a withdrawal** | `src/core/sobolev/hierarchical/nearField.ts:44-52` carries a `CORRECTED 2026-10-12` block withdrawing the magnitude claim with the 12 004-sample f32 measurement (4.1e-8 mean / 2.4e-7 max vs a 5.768e-5 band, ~240× narrower), keeps the instruction, and re-anchors it to `nearField.test.ts`'s executed 4e-15 cross-form pin. Probe present at `docs/…/phase2h-review-ledger/probes/powform.ts`. |

**10 of 10 ADDRESSED. 0 NOT ADDRESSED.**

---

## 2. Check 1 — the three frozen bars did not move

Verified against `git show fe5bedd:…`, not against the report.

| Bar | HEAD `fe5bedd` | Working tree | Moved? |
|---|---|---|---|
| `H5_KAPPA_EXCL_BAR` | `= 40;` at `hCalibration.ts:170` | `= 40;` at `:195` | **no** |
| H5's test-side bar | `expect(sp.lambdaMax / sp.lambda2).toBeLessThan(40);` at `precond.test.ts:442` | same at `:456` — **byte-identical under `cat -A`** | **no** |
| `H1B_LOWER` | `= 1e-7;` at `hCalibration.ts:135` | `= 1e-7;` at **`:135`** (same line) | **no** |
| `DEFAULT_THETA` | `= 0.25;` at `bct.ts:95` | `= 0.25;` at `:109` | **no** |
| (also checked) `H4C_RE_BAR` | `= 0.95;` at `hCalibration.ts:166` | `= 0.95;` at `:191` | **no** |

What each item added is scope and margin: `floorScope`, `levelBarScope`, `H1B_FLOOR_VALID_FROM_N`,
`marginAboveFloor`, `barUnchanged`. No bar value anywhere in the diff.

**Code-surface control.** I isolated the non-comment lines of the whole diff. Outside the four files
that implement items 1–4 and `apply.test.ts`'s one-line import, **every changed line is a comment**:

```
non-comment changed lines in bct.ts / gmres.ts / nearField.ts / dispatch.ts        -> none
non-comment changed lines in optimizer.ts / driver.ts / saddle.ts / solverWorker.ts
  + cr.test.ts / saddle.test.ts / saddleSeam.test.ts / gradient.test.ts            -> none
non-comment changed lines in apply.test.ts / bct.test.ts / lowOrder.test.ts        -> 4
  (+ import U_F32 as GMRES_U_F32;  -const U_F32 = 5.96e-8;  +const U_F32 = GMRES_U_F32;  1 test rename)
```

So the eight files item 7 touched beyond the brief's list carry **zero** behavioural risk, and the
wave adds **no new exported symbol** (`git diff | grep '^+export'` is empty). The one public-type
change is `HierarchicalSaddle.prepare`'s widened return type, which is C1's fix itself.

`bunx tsc --noEmit` → **exit 0, 0 lines of output.**

---

## 3. Check 2 — the citation repairs

### 3a. None of the 19 broken strings survives in `src/ test/ bench/` — CONFIRMED

My own sweep, digit-bounded so `gradient.ts:132` cannot be satisfied by `gradient.ts:1320`:

```
linsolve.ts:932  0   linsolve.ts:219-223  0   lineSearch.ts:257  0   gradient.ts:132  0   saddle.ts:412        0
linsolve.ts:804  0   linsolve.ts:133-154  0   lineSearch.ts:266  0   gradient.ts:133  0   optimizer.ts:514     0
linsolve.ts:860  0   linsolve.ts:823-827  0   lineSearch.ts:267  0   gradient.ts:143  0   optimizer.ts:392     0
                     linsolve.ts:924-925  0   lineSearch.ts:296  0                        optimizer.ts:281-282 0
                                              lineSearch.ts:193  0
```

### 3b. Self-drift repairs resolve to their pre-edit text — CONFIRMED, 10/10

The population the implementer says its own edits moved. `apply.test.ts` has three hunks
(`@@ -36,0 +37 @@`, `@@ -54,2 +55,12 @@`, `@@ -641,3 +652,15 @@`), giving a HEAD→WT map of
`+0 / +1 / +11 / +23`. Every one of the 13 `apply.test.ts:N` citations in `hCalibration.ts` was
rewritten by the wave. I verified **text identity**, not just the arithmetic — `md5sum` of
`git show fe5bedd:apply.test.ts | sed -n 'M,M′p'` against `sed -n 'N,N′p' apply.test.ts`:

```
181      -> 192            IDENTICAL          632-634  -> 643-645      IDENTICAL
655      -> 678            IDENTICAL          216-230  -> 227-241      IDENTICAL
604      -> 615            IDENTICAL          160-163  -> 171-174      IDENTICAL
89-97    -> 100-108        IDENTICAL          680-689  -> 703-712      IDENTICAL
660-664  -> 683-687        IDENTICAL          644      -> 667          IDENTICAL
```

All ten land on the mapped line exactly (`+11` below HEAD:641, `+23` above it), which is the
independent confirmation that the map, not a guess, produced them. The count goes 13 → 15 because
the wave deleted one `:644` citer line (item 5 rewrote `gateH6.consequence`) and added three new
citations in item 4's `floorScope` and `barSource`; new citations have no pre-image by construction
and I checked those by content instead (`apply.test.ts:687-693` does carry the f32-floor rationale,
`:643-645` the H1-b bar expression).

### 3c. Merge-base repairs — 3/3 sampled, including the worst case

```
gradient.test.ts:240  cites linsolve.ts:1101-1102
  WT  1101-1102 : "forward/back substitution only, no assembly, no factorization ('factor' / never fires here; that is the point of the reuse). The relative residual is"
  BASE 924-925  : IDENTICAL   <- the citation that used to resolve to `ldltFactor(K,size) : luFactor(K,size)`
optimizer.ts:301-303  now "solveSaddleFrozen (linsolve.ts)" (bare number dropped) + "solveSaddleFromA, linsolve.ts:881-883"
  WT  881-883 : the `a.length !== n*n` throw     BASE 804-806 : IDENTICAL
saddleSeam.test.ts:77 cites optimizer.ts:559
  WT  559 : "const timings = timingsEnd();"      BASE 514 : IDENTICAL
```

### 3d. My own independent sweeps found no residual defect

Two checks I wrote rather than reusing the implementer's tooling:

1. **Degenerate/out-of-range anchor audit** over all 228 `<basename>.ts:N` citations in
   `src/ test/ bench/` (193 uniquely resolvable): 2 flagged, **both false positives** —
   `nearField.ts:203` cites `scratch/bct.ts:659-664`, a *different* file my basename resolver
   collapsed onto `src/…/bct.ts` (the exact bug the implementer documents having hit and discarded),
   and `bench/gpu/adapterClass.ts:5` cites a `drive.ts` outside the wave's diff.
2. **Stale-citation sweep**: every citation whose *citer line survives unchanged from HEAD* but whose
   *target text changed*. 2 flagged, **both false positives** — `saddle.test.ts:427 → saddle.ts:35-40`
   and `saddleSeam.test.ts:58 → optimizer.ts:298-329`; in each case the only delta inside the cited
   range is a *nested* citation the wave itself repaired, so the cited content is intact.

**Verdict on the metric dispute.** The implementer is right that the checker's raw count measures
relocation, not incorrectness, and right to replace it with base-text identity. Both halves of their
claim hold under my independent measurement.

---

## 4. Check 3 — the test totals reconcile, and the gap has one cause

**Cause: `bun test test/sobolev/hierarchical` is a path-PREFIX filter, not a directory filter.** It
also loads `test/sobolev/hierarchicalSaddleReal.test.ts` and `test/sobolev/hierarchicalSeam.test.ts`.
Proof, from my own run's log:

```
$ grep -oE "^test/.*\.test\.ts" /tmp/rereview-hier-suite.log | sort -u
test/sobolev/hierarchical/apply.test.ts
test/sobolev/hierarchicalSaddleReal.test.ts        <-- outside the directory
```

`test/sobolev/hierarchical/` holds **12** `.test.ts` files, not 14. The 14 is 12 + those two.

**So the report's "the three groups partition the suite exactly … complete and disjoint" is FALSE.**
Groups 1 and 2 overlap on exactly two files, and **906 double-counts them.** Measured by me, each
file run on its own:

| | files | pass | fail |
|---|---|---|---|
| group 3 (explicit 44-file list, excludes the prefix + the 3 long files) | 44 | **402** | 0 |
| `test/sobolev/hierarchicalSeam.test.ts` | 1 | **10** | 0 |
| `test/sobolev/denseSaddleBitGolden.test.ts` | 1 | **44** | 0 |
| `test/sobolev/hierarchicalSaddleReal.test.ts` | 1 | **26** | 0 |
| group 1 (`test/sobolev/hierarchical` prefix = 14 files) | 14 | see §4b | 0 |

Unique test files repo-wide: `find test -name '*.test.ts' | wc -l` = **59** — exactly the first
review's 59 files. 44 + 12 + 3 = 59. Group 1 (14) + group 2 (3) + group 3 (44) = 61, i.e. **61 − 2
overlapping = 59**, which is the whole arithmetic of the discrepancy.

(That the two 59s coincide is not luck: the 43 other `*.test.ts` files in the repo all live under
`.claude/worktrees/` or `.superpowers/sdd/*/scratch*/`, and `bun test` skips dot-directories, so a
bare `bun test` and `find test` see the same 59.)

**The reconciliation:**

```
true repo-wide = group3 + group1 + denseSaddleBitGolden
               = 402    + 424    + 44                   = 870
report's 906   = 870 + hierarchicalSaddleReal(26) + hierarchicalSeam(10)   [double-counted]
first review   = 867
870 - 867 = +3  = precond.test.ts +1 (item 3's new test)
                + hierarchicalSaddleReal.test.ts +2 (item 1 replaced 1 test with 3)
```

**The +1 static delta the report could not reconcile is explained by the same prefix bug.** The
report computed its static delta over "the four hierarchical-**directory** files I touched" and got
`precond +1`, then concluded group 1's +3 was unaccounted. But the 14-file glob also contains
`hierarchicalSaddleReal.test.ts`, which the same wave took from 24 to 26 tests. **421 → 424 is
exactly +1 and +2, fully attributable. Nothing is unexplained, and no baseline is unreproducible.**

### 4b. Group 1, real output

See the appended summary line below — run in the background per the brief's waiver, polled to its
real summary, on a machine also running my foreground group-2/group-3 checks (which inflates wall
time and nothing else).

---

## 5. Check 4 — C1's dispatch

**Routing is by block presence, and §8.2 pins no numeric `k`.** I read §8.2 myself. The ruling is a
per-block table at `spec:2799-2804`, verbatim:

```
barycenterBlock   (3 rows)        → Schur complement. k = O(1), Schur is cheap here.
pointBlock        (3 rows/pin)    → Schur complement. Same reason.
totalLengthBlock  (1 row)         → Schur complement.
edgeLengthsBlock  (|E| rows)      → NULLSPACE route.
```

and §11.2 row 9b's verification column reads *"Dispatch on `k`: Schur for `k=O(1)`, nullspace for
`edgeLengths`' `|E|` rows"* (`spec:3138`). Neither names a threshold; both route per block. The
implementer's disclosure is accurate, and routing on block presence is what the spec actually says.

**The two cannot disagree about the cap — confirmed in code:**

```
schur.ts:459      const maxK = opts.maxK ?? SCHUR_MAX_K;     // then :466  if (k > maxK) throw SchurRouteError
nullspace.ts:441  const maxK = opts.maxK ?? SCHUR_MAX_K;     // then :442  if (kOther > maxK) throw NullspaceRouteError
hierarchicalSaddle.ts:511   maxK: opts.schur?.maxK,          // the single field, forwarded to BOTH
```

One constant (`SCHUR_MAX_K = 16`, `schur.ts:99`), one caller-settable field, forwarded unchanged to
both routes. Their *predicates* differ by design — `prepareSchur` tests total `k`, `prepareNullspace`
tests `kOther` with the `|E|` rows excluded — which is §8.2's per-block ruling and is documented at
`nullspace.ts:195` (*"the `|E|` rows do not count"*). So they cannot disagree about the cap's value,
and the one place they differ is the ruling itself.

**End to end, and the control still routes to `prepareSchur`.** `test/sobolev/hierarchicalSaddleReal.test.ts`,
**26 pass / 0 fail / 115.05 s, measured by me**:

- `:674` hybrid `{barycenter, edgeLengths, pins}` at n=24, k=30 (above `maxK`, so the all-Schur route
  threw here before): asserts `kEdge=24`, `kOther=6`, `edgeRowStart=3`, `schur.k=6`, the three
  projector defects < 1e-12, `residual < 1e-9`, `edgeConstraintResidual < 1e-11`, `eta < 1e-10`,
  `gmres.censored === false`, `iters > 0`, `lambda.length === n+6`, and a non-zero edge multiplier.
  `rep.normA2 < s.build.normA2` pins that the projected norm is re-estimated rather than inherited.
- `:743` the pure `edgeLengths` set routes to the nullspace route as a gauge (`kOther=0`,
  `schur.pinned === false`), not a refusal.
- `:785` an all-`O(1)` set (`k=4`) still reaches `prepareSchur` — asserted by the *absence* of the
  `projector` field via the `isNullspaceRoute` predicate (`:181`), so no `as` cast stands between the
  claim and the read — and 6 pins (18 rows, above `maxK=16`) still throws `SchurRouteError`, still a
  `SolveNotCertifiedError`, still not an `ExternalSolveError`.

**The dead-code consequence is resolved.** Graph, inbound on `prepareNullspace`: 3 callers, one of
which is `src.core.sobolev.hierarchical.hierarchicalSaddle.prepare`. Before the wave it had test
callers only.

**Recognition exactness.** `isEdgeLengthsRow` (`:295`) reproduces `edgeLengthsBlock`'s spelling
(`ex * (1/ℓ)`, raw ℓ with no `+ε`, the same `ℓ < 1e-14` zeroing) and compares with `!==`, with the
reason anchored at `:334-346`: a tolerant match that stripped the wrong `|E|` rows would hand them
to the projection and nothing downstream could see it, because both `residual` and
`edgeConstraintResidual` are self-consistent against whichever `C_edge` the route built. A mismatch
returns `null` and falls through to a loud `SchurRouteError`. That is also the documented backstop
for a `C` evaluated at different vertices (#83). This is the right call and it is anchored per
`CLAUDE.md`.

---

## 6. New findings

**None at Critical or Important.** Nothing inside the fix diff rises to the round-2 gate.

---

## 7. Ledger notes — not findings, no loop extension

Per the round-2 rules these are recorded here and go to the ledger/issues, not into the loop.

1. **`precond.test.ts:449` quotes `1.10x` where the test it points at publishes `1.06x`.** The
   superseded-sentence stamp says the replacement test *"publishes the 1.10x margin"*; that is the
   θ=0.5 ledger figure, while this file runs at `THETA = 0.25` (`:89`) and the new test's own comment,
   its log line and `gates.H5.levelBarScope.inBandSupremum.marginToBar` all read **1.0597**. Same
   direction, same conclusion, no bar and no verdict affected — Minor, and the kind of cross-reference
   figure the wave otherwise fixed everywhere else.
2. **10 of the 19 broken strings survive in `docs/`**, outside the finding's scope. I5's checker was
   declared over `src/`, `test/`, `bench/` and the brief inherited that scope, so I5 is addressed as
   filed. For the record: `plans/2026-09-04-webgpu-solver-phase2a-1.md` (5),
   `plans/2026-09-03-webgpu-solver-phase2a.md` (2, untracked),
   `specs/2026-10-09-phase2h-bct-gmres-design.md` (2), `specs/2026-09-03-webgpu-solver-phase2-design.md` (1).
   Two of those four are files this wave already edited for item 7, so the decision to stop at the
   code boundary was a scope call, not an oversight — worth folding into the durable fix the report
   proposes in its §E3 (checker in the knip pre-commit job + symbol anchors).
3. **The report's §A claim that its repo-wide `biome --write` "restored `gpu-phase2a-cr.json` to the
   committed convention" is FALSE.** Measured inside the repo so the config applies: HEAD's committed
   copy is format-clean; the working-tree copy is **not** (`× Formatter would have printed the
   following content`, `Found 1 error`). The *semantic* half of §A is **true and I verified it
   independently** — flattening both to leaves gives 3297 vs 3297, 0 added, 0 removed, and exactly one
   differing leaf, `.gitShaShort: 'ed73ed5' → 'fe5bedd'`. The first review had already recorded this
   file as a formatting failure, so the lint baseline did not change; only the report's claim about it
   is wrong. Flagged for honesty, not as a finding — the file is the human's call per the brief and I
   did not touch it.
4. **Lint, measured per file over all 21 modified `.ts`/`.json` files:** 20 clean at exit 0; the single
   error is `bench/results/gpu-phase2a-cr.json` (note 3). The brief's "7 pre-existing biome errors" is
   indeed not the live number — the report's correction to it stands.
5. The report's §E2 (A47 item 4 says conjunct 1's RED side is "now ESTABLISHED" while the artefact
   publishes `NOT ESTABLISHED`) is a real open contradiction and correctly left to the spec owner.
6. **The report's artefact-diff figures are from an intermediate state.** It records
   `.shards 22 -> 25 (three new shard records)`; the final tree is **19 → 26 (+7)**, consistent with
   the four separate harness invocations §C itself lists each appending a record. Harmless — shard
   records are run provenance — but a future reader comparing against "25" will not match. The
   substantive half of that claim is **independently confirmed**: see §9.

### Independent confirmation that no measured value moved

I flattened the artefact at HEAD and at the working tree to leaves and diffed, normalising
`file.ts:N` to `<C>` so citation repairs do not register:

```
leaves HEAD=4563  WT=4756   added=193  removed=0
NUMERIC leaves that changed: 3
  .host.loadAvg[0]  16.63 -> 10.68
  .host.loadAvg[1]  19.42 -> 14.68
  .host.loadAvg[2]  15.82 -> 13.37
NON-CITATION string/bool leaves that changed: 10
  gates.H4c.{verdict.verdict, verdict.reason, verdict.matchesPreRegistration, knockOut.spreadFinding}
  gates.H5.verdict.reason   gates.H6.consequence   verdicts.H4c.{...}   verdicts.H5.reason
```

**Every changed number is machine load; every changed string is one of items 2/3/5's intended
relabels; nothing was removed.** `knockOut.spreadFinding` keeps its measurement verbatim and replaces
only the conclusion label ("CONJUNCT 1 IS A WEAK GATE" → "MIS-AXED, NOT WEAK — [A47 item 6]", with the
rejected wording quoted rather than deleted). So the harness reproduced bit-for-bit across an
independent re-run, which is a stronger result than the report claimed for itself.

---

## 8. One closing note on a frozen bar — explicitly out of scope, stated once

The brief froze `H5_KAPPA_EXCL_BAR = 40` and deferred the redesign to an issue, and the wave honoured
that exactly. For the record only: the fix makes the bar's situation *worse* than I1 described, and it
is the wave's own measurement that does so. At the shipped θ=0.25 the in-band supremum is **37.7469**
against 40 — a **1.0597×** shortfall, tighter than the review's 1.10× at θ=0.5 — while the first
out-of-band point (m̂=70, λ₂=1.4747) reads **38.6578**, still under 40. So on the mass-weight axis
there is no reachable RED side at all, and the margin is 6 %. That is now published rather than
implied, which is what the brief asked for and is the right outcome for this wave. The redesign
belongs to #71/#100-101 under a pre-registered analysis; I am not reporting it as a finding.

---

## 9. Verification commands I ran, with real output

```
bunx tsc --noEmit                                   exit 0, 0 lines
bunx biome check <each of 21 modified ts/json>      20 × exit 0 clean; gpu-phase2a-cr.json exit 1 (out of scope)
bun test <44-file group-3 list>                     402 pass / 0 fail / 53155 expect / 44 files / 51.46s
bun test test/sobolev/hierarchicalSeam.test.ts       10 pass / 0 fail / 328 expect / 1 file / 0.28s
bun test test/sobolev/denseSaddleBitGolden.test.ts   44 pass / 0 fail / 216 expect / 1 file / 2.15s
bun test test/sobolev/hierarchicalSaddleReal.test.ts 26 pass / 0 fail / 1 file / 115.05s
bun test test/sobolev/hierarchical                   (background, see below)
```
