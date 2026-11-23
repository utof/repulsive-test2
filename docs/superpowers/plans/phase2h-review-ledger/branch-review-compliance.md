# Branch `feat/phase2h-impl` — blind whole-branch review: SPEC COMPLIANCE AND CODE QUALITY

Reviewer: blind, no session history, no prior findings. Merge base `9dd5551`, 47 commits, HEAD `fe5bedd`.
Contract: `docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md` (5 004 lines, amendments A1–A47).
Kernel mathematics was explicitly out of my scope.

Every command below was run by me, in the foreground, on this working tree.

---

## Verdicts

| | verdict |
|---|---|
| **(a) Spec compliance** | **FAIL** |
| **(b) Quality and maintainability** | **PASS WITH FINDINGS** |

**The compliance FAIL rests on exactly one finding (C1)** and is not a judgement on the rest. Thirteen
of the fourteen shipped `§11.2` rows have acceptance conditions that are met *and* independently
re-verifiable; I re-derived several of them from scratch (below). One ruled acceptance condition —
row 9b's "**Dispatch on `k`: Schur for `k=O(1)`, nullspace for `edgeLengths`' `|E|` rows**" — is
absent from `src/`, no amendment parks or defers it, and the consequence is that 2 089 priced lines
have no path to them and one of the two constraint combinations `§8.2` says "**must actually
work**" throws. Either ~15 lines in `makeHierarchicalSaddle.prepare` or one written park-with-ruling
closes it.

The quality verdict is a genuine PASS. This is unusually well-anchored code: of ~170 branch-new
exported declarations, **zero exported functions or classes** lack an `@see`/`@issue`/`Why:` anchor;
all seven knock-out routes are guarded and throw loudly; the topology guard is provably correct; and
the bit-level golden's provenance claim verifies against the merge base to the byte. The findings
below are concentrated in citation *maintenance*, not in the code.

### Repo hygiene — all four commands, real output

| command | result |
|---|---|
| `bun test` | **867 pass / 0 fail / 59 files / 199 671 expect() calls**, exit 0, 2 091 s |
| `bunx tsc --noEmit` | **clean**, no output |
| `bunx biome check` | **7 errors / 8 warnings / 7 infos** — **every one pre-existing**, none in a branch-touched file (`test_gradient.ts` ×7, `server.ts` ×3, `phaseTimings.ts`, `Curve.tsx`, `Viewer.tsx` ×3, `ControlPanel.tsx`, `index.tsx`, `frozenProjection.test.ts`, `store.test.ts`, `testConfigs-random.test.ts` ×2). Plus one **formatting** failure on the uncommitted `bench/results/gpu-phase2a-cr.json` — see M3 |
| `bunx knip` | 181 unused files (pre-existing, all under `.superpowers/`, `oracle/gen_fixtures.ts`), 7 unused exports, 18 unused exported types. Branch-new entries and their status in M1/M6 |

`bun test` ran concurrently with two other reviewers' suites on this machine, so 2 091 s is
load-inflated; nothing in `§11.4` gates on wall-clock.

**Correction to the brief's own premise:** `REVIEW-CONTEXT.md:10-11` says `lefthook.yml:11` runs biome
with `--linter-enabled=false`. That is **no longer true on this branch** — commit `f037cf1` removed
the flag (`lefthook.yml:18` is now `bunx biome check --write --no-errors-on-unmatched {staged_files}`),
with the reason anchored in place at `:7-11` and `@issue utof/repulsive-test2#70`. A green pre-commit
hook on this branch *does* mean lint ran. I verified lint independently anyway.

### `§11.4` "Done" checklist, item by item

| item | status | evidence |
|---|---|---|
| `bun test` green, `tsc` clean | ✅ | above |
| Every `§7` gate runs and records its VALUE | ✅ | `bench/results/2026-10-12-gpu-phase2h-gate-calibration.json`: 9 gate blocks each with `verdict` + `reason` + measured values, 87 `invalid*` keys carrying per-row values, `NOT APPLICABLE` / `REPLACED` / `NEAR-VACUOUS` statuses with the amendment that ruled them |
| Every INVALID predicate reachable in a test | ✅ (as amended) | A46 item 4 rules H1-a/H1-b/H2/H3's predicates live in the instruments' tests; H4a's INVALID-1 is non-computable and replaced by ruling 4; the artefact records each one's status |
| Dense path bit-identical vs the A24 golden | ✅ | `test/sobolev/denseSaddleBitGolden.json` **byte-identical since `d3672c2`** (`git diff d3672c2..HEAD` empty), and its `provenance.seamFileSha256` for all five seam files matches `git show 9dd5551:<file> \| sha256sum` **exactly, 5/5** — the claim "captured with all five seam files untouched vs main" is verified, not asserted |
| `oracle/golden/*.json` unmodified | ✅ | `git diff --name-only 9dd5551..HEAD -- oracle` is empty |
| `git diff src/core/tangentPointEnergy.ts` empty | ✅ | not in the branch diff |
| R1's served-N ceiling published | ✅ | three ceilings, `servedNF32Only` / `servedNF32Only{Status,BoundBy,Reason}` + per-shape block, honest `NOT ESTABLISHED` with `boundBy: null` |
| θ default anchored on a live justification | ❌ | **I2** |
| H4c reported with its ruled label | ❌ | **I1** |
| `§8.2` hybrid reachable | ❌ | **C1** |

### Acceptance conditions I re-derived rather than took on trust

- **Row 2** (`bvh.ts`): `bvh.test.ts:54-56` asserts mean leaf size in `[4.88, 7.8125]` and *states why*
  7.81 was widened (`2000/256 = 7.8125` truncates). Power-of-two leaf counts pinned at `:45`.
- **Row 3** (`bct.ts`): `bct.test.ts:139-146` pins `κ_near = 93.26875` at n=960 and **`108.1956` at
  n=30 000** with exact `toBe`, matching `§11.2`'s "93.3" and "108.20". `adjacencyVetoed = 0 at θ≤0.4`
  is at `:264-276` and is **non-vacuous**: it pins 248 at θ=0.5 *and* 0 at θ=0.4, *and* runs a
  certificate-off control proving 0 is structural there, with the comment stating that asserting zero
  at 0.5 would be wrong. Exactly what `§11.3` asks for.
- **Row 5** (`operator.ts`): trace ratio `≥ 0.989` at `operator.test.ts:160`; symmetry defect gate at
  `:201-220` including the "rounded columns cannot give a bit-exact symmetric matrix ⇒ INVALID"
  clause.
- **Row 8b carries A33's SCALE fact**: the factory does not normalise and records
  `build.curveExtent` (`hierarchicalSaddle.ts:344`), reachable on the returned object — I confirmed
  `5.4142` on `trefoil(24)` by running it.
- **Row 10** was ruled measurement-only by A45 ("`git diff -U0 -- src/` contains ZERO non-comment
  added or removed lines"), so its `files: gmres.ts, apply.ts` column is discharged by comment-only
  edits. Spec-compliant — a later amendment overrides the body.
- **`optimizer.ts:174-186`'s factorMode claim** ("0 of 144 coordinates changed, against 118 of 144 on
  the dense arm") is backed by `hierarchicalSaddleReal.test.ts:786` and `:804`.
- **Row 7's `side: 'right'` guard** (`gmres.ts:566-570`) and all six sibling knock-out guards
  (`apply.ts:489`, `lowOrder.ts:374`, `precond.ts:310`, `schur.ts:461`, `nullspace.ts:426`) throw with
  messages naming the spec section. No knock-out is production-reachable.

### Integration seams — per modified pre-existing file

Seven pre-existing `src/` files changed. My conclusion: **no existing caller can observe a behaviour
change, and no default is silently altered.**

- **`optimizer.ts:308-329`** — the dense guard became the `else if` of a new `kind === 'hierarchical'`
  branch. A caller passing a dense saddle takes the identical path; a caller passing `{}` (no `kind`)
  still falls through to the `opts.saddle.A` TypeError that `saddleSeam.test.ts:344-352` relies on.
- **`optimizer.ts:448` / `lineSearch.ts:329`** — the rethrow widened from `ExternalSolveError` to
  `SolveNotCertifiedError`. `ExternalSolveError extends SolveNotCertifiedError`
  (`linsolve.ts:456`), and the only other subclasses (`SchurRouteError`, `SchurConvergenceError`,
  `NullspaceRouteError`) are unreachable without the new arm. **No new error path is reachable on the
  dense/GPU arm.** `SchurRankError` deliberately stays a plain `Error` so it *does* fold into
  `'singular_system'`, and `schur.ts:199-203` states why ("this one really is a singular system").
- **`gradient.ts:159,185`** — adds `kind: 'dense'` to the two returned `frozen` objects. `frozen`
  never crosses the worker boundary and nothing serialises or key-counts it.
- **`linsolve.ts:1128-1140`** — `solveSaddleFrozen`'s guards still run *before* the `kind` dispatch
  (deliberate, with the reason stated at `:1128-1131`: a solver's frozen `C` cannot validate a
  caller's rhs length). Dense arithmetic, operand order and residual operands are unchanged.
- **`src/gpu/saddle.ts:344-350`** — return type narrowed to `DensePreparedSaddle` and `kind: 'dense'`
  added. The narrowing is justified in place; `bun test` is green on it.
- **`src/gpu/selfTest.ts`** — `1e-5` → `T1_REL_ERR`, identical value. `gateTolerances.ts:30` keeps
  `1 - 1e-6` as an expression rather than a decimal literal *specifically* so the float is
  bit-identical to the inline spelling. Careful work.
- **`bench/stats.ts`** (the one non-2H commit, `48b4af0`) — I diffed all four call sites. Each keeps
  its original variant (`jsAlgebra.ts`/`phase1.ts` → `upperMedian`, `spikes.ts`/`sobolev.bench.ts` →
  `median`), `0.5 *` → `/ 2` is bit-identical, and empty-input behaviour is preserved per variant.
  The "no number moved" claim holds.

---

# Findings

## CRITICAL

### C1 — `§8.2`'s hybrid dispatch does not exist; `nullspace.ts` has no production caller and the `{barycenter, edgeLengths, pins}` set throws

**Where.** `src/core/sobolev/hierarchical/hierarchicalSaddle.ts:357-361` — `prepare` calls
`prepareSchur` unconditionally, with no test on `k`:

```ts
prepare: (C: number[][]): SchurSolver =>
    prepareSchur(op, precond, C, {
        ...opts.schur,
        normA2: opts.schur?.normA2 ?? normA2,
    }),
```

**The acceptance condition.** `§11.2` row 9b, verification column, verbatim: *"the hybrid is a
two-branch ruling and the first draft gave a task to only one branch. **Dispatch on `k`: Schur for
`k=O(1)`, nullspace for `edgeLengths`' `|E|` rows.**"* `§8.2` (`:2779-2793`) rules
`edgeLengthsBlock (|E| rows) → NULLSPACE route` and states: *"the combinations that must actually
work are: `{barycenter, totalLength, pins}` (all-Schur, the existing path) and `{barycenter,
edgeLengths, pins}` (hybrid)."*

**No amendment parks it.** I searched A33/A35/A36/A37/A38–A47 for a deferral: A33 item 3 *moves* the
`edgeLengths` cell from row 8b to row 9 ("where the route it exercises actually lives"), A36 claims
reachability only for "tasks 1–9b", and A37 (row 9's amendment) addresses the composition *inside*
`nullspace.ts`, never the wiring. `grep -niE 'dispatch on|dispatches on k|route per block'` over the
whole spec returns row 9b's line and nothing else.

**Falsifiable — executable check, written and run.**
`.superpowers/sdd/2026-10-09-phase2h-bct-gmres-design/scratch-review-compliance/hybridcheck.ts`,
`bun run` output verbatim:

```
build OK, kind = hierarchical n = 24 extent = 5.4142
k = C.length = 30 (barycenter 3 + edgeLengths 24 + pin 3)
RESULT: prepare() THREW SchurRouteError
prepareSchur: k=30 constraint rows exceed maxK=16 (n=24). This is the k = O(n) regime of
edgeLengthsBlock (|E| rows): an iterative solve costs O(N²·iters) there, worse than the dense
path (spec §8.1). Route it through the nullspace route (spec §8.2, §11.2 row 9), or raise maxK
only if k really is O(1).
CONTROL k=6: prepare() succeeded (all-Schur path works).
```

The guard's own message instructs the caller to *"Route it through the nullspace route"* — a route no
code in the repository takes.

**Dead-code consequence, measured.** I ran a reference census over every export of
`src/core/sobolev/hierarchical/*.ts` against all of `src/`, `bench/` and `test/`
(`mcp__codebase-memory-mcp__trace_path` on `prepareNullspace` agrees: `callers_total: 2`, both in
`nullspace.test.ts`). `prepareNullspace` is the **only exported function in the directory with zero
`src/` and zero `bench/` consumers** — every other test-only export (`precomputeFar`,
`applyLowOrder`, `endpoint4Kernel`, `lowOrderExponent`) is a gate instrument the spec requires to be
measured, and the rest are consumed by `bench/gpu/hCalibration.ts`. So `nullspace.ts` (753 src lines)
plus `nullspace.test.ts` (1 336 lines) — **2 089 lines, ~7.4 % of the branch** — are reachable only
from their own test. knip independently flags six of its exported types as unused.

**Why Critical and not Important.** This is the *final* whole-branch review; after it the branch
merges. `§8.2` names the hybrid as one of exactly two combinations that must work and it does not.
The failure is at least loud (a typed `SolveNotCertifiedError` that propagates rather than folding
into `'singular_system'`), which is the only reason this is not worse.

**Fix I would make.** In `makeHierarchicalSaddle.prepare`, dispatch on the shape of `C` and route the
`edgeLengths` rows to `prepareNullspace` — `nullspace.ts`'s `NullspaceEdgeBlock` already takes
`{vertices, edges, rowStart}` and composes an unmodified `prepareSchur` for the `O(1)` rows beside
them (A37), so the composition it needs exists. **Or**, if the dispatch is genuinely out of scope for
this phase, park it in writing: a `gh issue create` plus one amendment sentence re-scoping row 9b's
verification column and a `Why: no production caller yet, see #NN` on `nullspace.ts`'s module TSDoc,
so the next reader does not read 753 lines as live. A silent discard is what `CLAUDE.md`'s review
loop forbids.

---

## IMPORTANT

### I2 — H4c is published `GREEN`; A47 item 7 rules it MIS-AXED and forbids reporting it as GREEN

**Where.** `bench/results/2026-10-12-gpu-phase2h-gate-calibration.json`, `verdicts.H4c`:

```json
{ "verdict": "GREEN",
  "reason": "worst R_E 0.99867 >= 0.95 with accepted fraction 1.0 on every rung (240,480)",
  "preRegistered": "GREEN", "matchesPreRegistration": true }
```

**The ruling.** A47 item 7, verbatim: *"**New verdict label: WEAK/MIS-AXED**, beside A42 ruling 5's
VACUOUS. VACUOUS = the predicate cannot fire. **MIS-AXED** = it fires, but not on the variable it is
written against. Both are distinct from GREEN and **neither may be reported as GREEN**, because a
gate that passed and a gate that could not have failed on the tested axis read identically in an
artefact. H4b′ is VACUOUS; H4c conjunct 1 is MIS-AXED."* A47 item 6 adds: *"`§7.7`'s framing of it as
an end-to-end compression check is withdrawn."*

**Falsifiable, three greps.**
1. `bench/gpu/hCalibration.ts:324` — `type Verdict = 'GREEN' | 'RED' | 'INVALID' | 'VACUOUS' | 'NOT APPLICABLE' | 'NOT ESTABLISHED';`
   — **`'MIS-AXED'` is not a member**, so the harness cannot express the ruled label.
2. `grep -c 'MIS-AXED\|MIS_AXED\|misAxed'` over the artefact → **0**. Over `hCalibration.ts` → **0**.
3. `hCalibration.ts:1608` still reads `// H4c — spec §7.7. The end-to-end descent gate:` and
   `grep -n 'mis-ax\|only instrument\|uniquely\|drift between'` finds no trace of the α/β/ε-drift
   purpose A47 item 6 ruled it into.

H4b′ **is** correctly `VACUOUS` with A42 ruling 5 cited in its `reason`, and `H4bPrime`'s own
`verdict.reason` even records that its VACUOUS label is itself falsifiable — so the harness clearly
understands the distinction and applied it to one of the two gates A47 names.

**Ordering, stated fairly.** A47 is commit `00d21b9` at 10:29:00 and the harness is `fe5bedd` at
10:29:28 — the ruling is 28 seconds older than the code that violates it, and A47 was itself written
*from* this run. So this is a bookkeeping miss at the very end of the row, not a disregarded ruling.
It still ships an artefact that the governing spec says must not read this way, and it is the exact
failure mode `CLAUDE.md`'s `gates-name-their-impossible-values` memory exists for.

**Fix.** Add `'MIS-AXED'` to the `Verdict` union, emit it for H4c conjunct 1 with A47 item 6's reason
string, keep the conjunct-2 GREEN separately (A47 item 4 proved the conjuncts independent: acceptance
stays 1.0 while α drift drives conjunct 1 to 0.868108), and replace `hCalibration.ts:1608`'s
"end-to-end descent gate" with the α/β/ε-drift purpose.

### I3 — Six sites justify `DEFAULT_THETA = 0.25` by "gate H6 read RED", a justification A46 item 7 withdrew — and A46 named one of them by file:line

**The ruling.** A46 item 7, verbatim: *"θ=0.25 reads 0.7649, θ=0.5 reads 0.7856, and the exact δ=0
operator reads 0.7644 — so **θ is not the variable and switching it changes nothing**. … **The RED is
WITHDRAWN as a θ signal**, and `apply.test.ts:644`'s "0.25 is production because H6 read RED" rests on
a withdrawn inference. **The CHOICE of production θ stays open on #51's own evidence.**"*

**The six sites that still state the withdrawn justification:**

| site | text |
|---|---|
| `src/core/sobolev/hierarchical/bct.ts:75` | "**H6 read RED (measured 2026-10-09), so this is 0.25.**" |
| `src/core/sobolev/hierarchical/bct.ts:86` | "the decision rests on H6's pre-registration and not on a post-hoc reading" |
| `src/core/sobolev/hierarchical/hierarchicalSaddle.ts:123` | "`theta` defaults to `bct.ts`'s `DEFAULT_THETA` (0.25, **gate H6**)" |
| `test/sobolev/hierarchical/apply.test.ts:641` | "theta=0.25 is PRODUCTION after gate **H6 read RED** (spec §7.9, 2026-10-09)" — the site A46 named |
| `test/sobolev/hierarchical/bct.test.ts:500` | test **name**: `'…and theta is 0.25 (gate H6 RED)'` |
| `test/sobolev/hierarchical/lowOrder.test.ts:58` | "The production default (`bct.ts` `DEFAULT_THETA`, **H6 RED**, commit 4c7af4d)" |

**This is not pedantry about a number.** The harness itself already complies —
`constants.thetaProductionReason` in the artefact reads: *"OPEN. [A46 item 7] withdrew H6's RED as a
theta signal, so apply.test.ts:644's "0.25 is PRODUCTION after H6 read RED" rests on a withdrawn
inference; the choice stays open on utof/repulsive-test2#51's evidence and is NOT row 11's to make"*,
and emits `thetaProduction: null`. So the artefact and the code now **disagree with each other** about
whether θ is settled. Per `CLAUDE.md`'s stated rationale for anchors ("the reference stops a future
agent from 'cleaning up' an intentional line and regressing"), an anchor that cites a withdrawn gate
is the failure mode itself: a future agent reading `bct.ts:75` concludes the default is gate-decided
and settled.

**Note the value 0.25 is not challenged.** A46 item 7 withdrew the *justification*, not the number;
`apply.test.ts:644`'s own surrounding reasoning ("a precision gate anchored only at a theta the
product no longer uses is a gate on nothing, so the band is asserted at both") stays correct, and the
tests gate at both θ.

**Fix.** Re-anchor all six on `@issue utof/repulsive-test2#51` + A46 item 7 ("the choice is open; 0.25
is the shipped default pending #51"), and rename `bct.test.ts:500`'s test so its name does not assert
a withdrawn cause.

### I4 — `gmres.ts` quotes two figures from `§10.2`'s "Retracted claims — do not quote these", one labelled MEASURED and wrong by 6.7 decades

**Where.** `src/core/sobolev/hierarchical/gmres.ts:37-41`:

> "THE RECURRENCE RESIDUAL IS NOT A CONVERGENCE CRITERION (spec §5.3). Left-preconditioned it is
> `‖Ã⁻¹r‖/‖Ã⁻¹b‖`, and it under-reports the true relative residual by **4.372e8 / 1.404e9 / 1.142e10
> / 2.056e11 / 1.717e12** at n = 64…960 (spec §7.3 anti-trap 2, **MEASURED**; fitted exponent
> **+3.056**, tracking `κ₂(A_σ) ∝ n^3.334`)."

**`§10.2`:2935 retracts exactly that string:**

> | `§7.3` anti-trap 2's recurrence under-report **4.372e8 → 1.717e12**, exponent **+3.056** |
> **85.6 → 2.28e4**, exponent **≈+2.07**; 1.00e6 at n=3840 | … magnitudes are **6.7 decades**
> smaller, because the shipped `Ã⁻¹A_σ` is ~150× better conditioned. **`§5.3`'s rule stands; its
> number does not** |

Second instance: **`gmres.ts:183-184`** — "`‖b‖` alone is the relative residual, whose floor degrades
as `n^3.36` (spec §7.4) where `η`'s grows as `n^0.34`". `§10.2`:2936 replaces `n^3.36` with **`n^0.88`,
> 1 at n ≥ 1920**, and A47 item 11 re-measures the `η` exponent as **−0.1034** (not +0.34 — the sign
is different). The lab figures are not a rounding error away; they are a different operator.

**The second half of this finding is a spec defect, and it is why the code got it wrong.** The two
`§7.3`/`§7.4` passages `gmres.ts` cites carry **no in-place retraction marker**, while their immediate
neighbours do:

- `§7.3`:1453 carries `> **A26, RULED 2026-10-11 (task 7b) — EVERY NUMBER IN THE TABLE AND THE FIT
  ABOVE IS A LAB NUMBER…**` — but anti-trap 2 sits at `:1439-1441`, *above* the table that box scopes
  itself to, so it is **not** covered.
- `§7.4`:1951-1952 carries a "LAB numbers … retained for provenance only (A30, A31)" marker on a
  different bullet, while `:1920-1922`'s `trueFloorMin` / `n^3.36` comparison has none.

So an implementer following `gmres.ts`'s own citation to `§7.3 anti-trap 2` reads the retracted
numbers presented as MEASURED, with nothing on the page to warn them. **This is an unamended
contradiction between the spec body and `§10.2`/A30**, inside the sections the branch's core module
is anchored to.

**Bounded harm, stated fairly.** The *rule* both comments support survives — `§10.2` says so
explicitly ("`§5.3`'s rule stands; its number does not", and the `n^3.36` revision "REINFORCES
`§7.4`'s conclusion"). The hazard is a future agent sizing a tolerance, a probe schedule or a
knock-out bar off a number that is 6.7 decades wrong on the shipped operator. For contrast,
`test/sobolev/hierarchical/lowOrder.test.ts:1018-1032` handles the analogous `p = 0.0010` retraction
*correctly* — it cites `§7.8`'s provenance note and asserts the re-derived `0.00099`. The tests got
this right; this one src comment did not.

**Fix.** Replace both figures in `gmres.ts` with the shipped-operator values (`85.6 → 2.28e4`,
exponent ≈+2.07; `n^0.88`) citing A30 / `§10.2` rather than the uncorrected body, **and** add the A30
pointer in place at `§7.3`:1441 and `§7.4`:1922 in the style already used at `:1453` and `:1951`.

### I5 — 28 line-number citations are broken, 27 of which were correct at the merge base; one now points at the exact code it says never runs

The branch inserts +200 lines into `linsolve.ts`, +61 into `optimizer.ts`, +38 each into
`lineSearch.ts`/`gradient.ts` and +10 into `src/gpu/saddle.ts`. It updated the citations *inside the
lines it rewrote* (`linsolve.ts:411` → `:927`/`:1139`; `lineSearch.ts:310` → `optimizer.ts:448`;
`saddleSeam.test.ts:60` → `:544-552`) and missed every other citation its own insertions invalidated —
including several in the very files it edited.

**Mechanical checker** (resolves every `<module>.ts:<N>` citation in `src/`, `test/`, `bench/` and
prints the target line at `9dd5551` vs at HEAD) — reproducible, saved at
`.superpowers/sdd/2026-10-09-phase2h-bct-gmres-design/scratch-review-compliance/citecheck.ts`; 51 raw
drifts, of which the following are *regressions* (right before, wrong now):

| citer | citation | said (correct at base) | points at now |
|---|---|---|---|
| **`src/core/optimizer.ts:301`** | `linsolve.ts:932-955` | `solveSaddleFrozen` (now at `:1109`) | `structuredSaddleResidual`'s comment |
| **`src/core/optimizer.ts:302`** | `:804-806` | the `a.length !== n*n` throw (now `:881`) | `ldltSolveFactored`'s pivot swap |
| `src/gpu/saddle.ts:23` | `linsolve.ts:219-223` | `LuFactorization` | `*/` |
| `src/gpu/saddle.ts:36` | `linsolve.ts:860` | `structuredSaddleResidual` (now `:937`) | the `factorMode` TSDoc |
| `src/gpu/saddle.ts:63` | `linsolve.ts:133-154` | `buildSaddleMatrix` | `*/` |
| `src/gpu/saddle.ts:130` | `linsolve.ts:823-827` | the coordinate-major `off = b*n` write loop | `solveFactored`'s external branch |
| `src/gpu/saddle.ts:413` | `linsolve.ts:860` | `structuredSaddleResidual` | the `factorMode` TSDoc |
| `test/gpu/cr.test.ts:125` | `lineSearch.ts:257` | `const negPhi = …` | a `blockNorm` check |
| `test/gpu/cr.test.ts:237` | `linsolve.ts:932-955` | `solveSaddleFrozen` | a comment |
| `test/gpu/cr.test.ts:292` | `gradient.ts:133` | `solveSaddleFrozen(frozen, flatten(dE))` | a comment |
| `test/gpu/cr.test.ts:295` | `lineSearch.ts:267` | the `solveSaddleFrozen` call (now `:279`) | `}` |
| `test/gpu/saddle.test.ts:375` | `saddle.ts:412` | `const rel = Math.sqrt(rn)/rhsNorm` | a comment |
| `test/gpu/saddleSeam.test.ts:77` | `optimizer.ts:514` | `const timings = timingsEnd()` (now `:559`) | `? {` |
| `test/gpu/saddleSeam.test.ts:84` | `gradient.ts:143` | the CPU-path `timed('saddle', …)` | `lambda,` |
| `test/gpu/saddleSeam.test.ts:86`, `:173` | `lineSearch.ts:266` | the projection `timed('saddle', …)` (now `:277`) | `break;` |
| `test/gpu/saddleSeam.test.ts:172` | `gradient.ts:132` | the seam-path `timed('saddle', …)` | a comment |
| `test/gpu/saddleSeam.test.ts:212` | `optimizer.ts:392` | the gradient solve's `catch` (now `:429`) | `dE = dETpe;` |
| `test/gpu/saddleSeam.test.ts:222` | `lineSearch.ts:296` | `} catch (e) {` (now `:309`) | `negPhi,` |
| `test/gpu/saddleSeam.test.ts:283`, `:285` | `linsolve.ts:932-955`, `:804-806` | as rows 1–2 | as rows 1–2 |
| `test/gpu/saddleSeam.test.ts:298` | `optimizer.ts:281-282` | the `dE` length guard (now `:298-299`) | the function's parameters |
| `test/sobolev/gradient.test.ts:145` | `gradient.ts:132` **and** `:143` | both `timed('saddle', …)` calls | a comment / `lambda,` |
| **`test/sobolev/gradient.test.ts:240`** | `linsolve.ts:924-925` | the TSDoc *"no assembly, no factorization ('factor' never fires here; that is the point of the reuse)"* | **`factorMode === 'ldlt' ? ldltFactor(K, size) : luFactor(K, size)`** |
| `test/sobolev/gradient.test.ts:241` | `gradient.ts:132` | the seam `timed('saddle', …)` | a comment |

Plus **two citations that were already wrong when this branch wrote them**:

- `src/core/sobolev/hierarchical/gmres.ts:186` — new code citing `structuredSaddleResidual` as
  `linsolve.ts:860`; it is at `:937`, and `:860` was 77 lines off by the time the line was authored.
- `bench/gpu/hCalibration.ts:1735` — cites `lineSearch.ts:193` for *"the empty set converges trivially
  at iteration 0"*; that sentence is at `lineSearch.ts:192` now and was at ~`:182` at the base, so
  `:193` never pointed at it.

**The `gradient.test.ts:240` row is the one that matters most.** The comment reads *"solveSaddleFrozen
deliberately fires no 'factor' sub-phase (linsolve.ts:924-925 — 'no factorization; that is the point
of the reuse')"*, and `linsolve.ts:924-925` is now a `timed('factor', …)` call invoking
`ldltFactor`/`luFactor`. A reader following that citation sees the precise opposite of the claim. In a
repo whose stated reason for anchors is that a future agent must not "clean up" an intentional line,
an anchor that resolves to its own contradiction is worse than no anchor.

**Fix.** Re-point all 28 (the checker prints each one's correct target), and — because this will
recur on the next branch that inserts into `linsolve.ts` — prefer symbol-name anchors
(`` `solveSaddleFrozen` (linsolve.ts) ``) over bare line numbers for the ~15 of these that name a
function, or wire the checker into the knip pre-commit job, which is already non-blocking and
informational.

### I6 — `dispatch.ts:166-172`, the only `src/` entry point a caller uses, never mentions the hierarchical arm or its mandatory `projectionMode`

`src/core/dispatch.ts` is **unchanged by this branch** (`git diff --name-only 9dd5551..HEAD --
src/core/dispatch.ts` is empty), yet its `saddle?: PreparedSaddle` field silently widened to accept
the new arm when `PreparedSaddle` became a union. Its TSDoc still reads, in full:

```
A saddle solve prepared for `vertices` (WebGPU Phase 2a seam). Forwarded to
`sobolevStepSet` exactly like `dE`. Carries closures ⇒ never cloneable ⇒ it
must not ride the worker payload.
@see SobolevStepOptions.saddle (optimizer.ts)
```

A caller reading only this cannot learn that (a) a `kind: 'hierarchical'` value **throws** unless
`projectionMode: 'frozen'` is also passed (`optimizer.ts:324-327`), (b) `makeHierarchicalSaddle` is
the producer, or (c) the arm is main-thread/inline only. `dispatch.ts:271` *does* forward
`projectionMode`, so the route works — I confirmed it — but the contract is undocumented at the only
surface that matters.

A33 item 2 required 8b to "say so in `stats` or TSDoc rather than leaving a UI path that throws", and
`hierarchicalSaddle.ts:25-32` discharges that **excellently** (a dedicated REACHABILITY paragraph
naming `solverWorker.ts:51-52`, `store.ts:362` and the structured-clone argument). The gap is only
that the producer documents the constraint and the consumer-facing entry point does not. Under
`CLAUDE.md`'s "non-obvious runtime behavior gets an inline comment linking to an ADR or GH issue",
a public field that accepts a value which throws unless a *second* field is set is non-obvious
runtime behaviour.

**Fix.** ~6 lines on `dispatch.ts:166-172`: name the two arms, state the `projectionMode: 'frozen'`
requirement and the throw, point at `makeHierarchicalSaddle`, and `@see` the 2H spec `§5.1`.

---

## MINOR

### M1 — six branch-new exported types lack an `@see`/`@issue`/`Why:` anchor

Audited mechanically over every `export` in the 20 changed `src/` files
(`scratch-review-compliance/tsdoc.ts`): **11 of 185 lack an anchor, and 5 of those 11 are
pre-existing** — `optimizer.ts:23` `DEFAULTS`, `:25` `StepOptions`, `:45` `step` (commit `a58f423`),
`lineSearch.ts:413` `LineSearchFailureReason` (`5c526d7`), `saddle.ts:322` `IrTrace` (`b9499a4`).

The six branch-new ones, all plain shape declarations with a descriptive one-liner:
`bct.ts:135 ResolvedBctParams`, `:191 BlockVisitor`, `:197 BctTraversal`, `:423 ThetaLadderRung`,
`:431 ThetaLadder`, `nearField.ts:181 NearGeometry`.

**Zero branch-new exported functions or classes lack an anchor.** Fix: add an `@see` to the spec
section each shape belongs to (one line each).

### M2 — `bench/gpu/hCalibration.ts` (2 418 lines) has no test and is not executed by `bun test`

It produces the branch's published gate artefact and nothing regression-guards it; `grep -rl
hCalibration test/` is empty. A46 item 4 deliberately rules that row 11 "cites" the instruments'
existing coverage rather than re-implementing it, so this is not a spec violation — but a harness this
size with no test is a maintainability risk, and `tsc` is its only oracle. Fix: a small smoke test
asserting the schema keys A46 items 2/5/6 and A47 require (`constants.clock === null`,
`constants.adapterCert === null`, per-gate `fixtures` provenance, `r1.*Status`), which would also have
caught I2.

### M3 — the working tree is dirty with a +1 263/−314 bench result that fails `biome check`

`bench/results/gpu-phase2a-cr.json` is modified and uncommitted; `bunx biome check --reporter=summary`
reports it under "The following files need to be formatted". So `bunx biome check` exits non-zero on
this tree **for that reason alone**, independently of the pre-existing lint errors. Either commit it
(formatted) or revert it before the merge, so the next reviewer's lint run is interpretable.

### M4 — orphaned comment blocks where the `median` declarations were removed

`bench/gpu/phase1.ts:271-284` and `bench/gpu/jsAlgebra.ts:105-110` keep the replaced helper's prose —
including an `@see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (D4 timing method; Task
4)` — but the declaration it documented is gone, so a 14-line doc comment with an `@see` now sits
between two unrelated functions attached to nothing. The content is worth keeping; move it to the
file header or onto the first `upperMedian` call site.

### M5 — the build extent is not visible through `SobolevStepStats`

A33's SCALE fact asks 8b to "record **in `stats`** the extent it built at". It is recorded on
`HierarchicalSaddleBuild.curveExtent` (`hierarchicalSaddle.ts:344`), i.e. visible to whoever *builds*
the saddle but not to whoever consumes `dispatchDescentStep`'s `stats`. On today's inline-only route
the same code does both, so the intent ("a quality change nothing downstream can see" → now it can) is
served; it stops being served the moment a second consumer appears. Worth one sentence in
`HierarchicalSaddleBuild`'s TSDoc noting that `stats` does not carry it.

### M6 — knip: branch-new entries, separated from the pre-existing list, and two false positives

Pre-existing: all 181 unused files (`.superpowers/**`, `oracle/gen_fixtures.ts`), `xorshift32`,
`SOBOLEV_CONVERGENCE_TOL`, `projectBarycenter`, `loadSavedConfig`, `saveConfig`,
`CertificationMethod`, `FieldArgs`, `SobolevStepFailureReason`, `ProjectBarycenter{Options,Result}`,
`FieldPenalty`, `PhaseSample`, `SavedConfig`.

Branch-new, and **false positives** — I verified both are consumed: `T2_REL_ERR` and `T3_COSINE_MIN`
are imported and used in `bench/gpu/spikes.ts:9,992,1004` and `bench/gpu/phase1.ts:25,246,248`. knip
misses them because those files are outside its `project` config. `gateTolerances.ts`'s own claim that
it is "shared by the runtime boot gate and the bench kill gates" is therefore **true**.

Branch-new and genuine, all union members or report shapes that structural typing means no consumer
needs to name — `DenseFrozenSaddleOperator`, `HierarchicalFrozenSaddleOperator` (members of the
exported `FrozenSaddleOperator` union), `ThetaLadderRung`, `HierarchicalSaddleBuild`,
`SmallConditioning`, `SchurSolveResult`, and the six `nullspace.ts` types. The six `nullspace.ts`
entries are a symptom of **C1**, not an independent finding; the rest are normal.

---

## Things I checked that are clean, recorded so the next reviewer does not redo them

- **Error-class hierarchy is deliberate and correct.** `SchurRouteError`, `SchurConvergenceError`,
  `NullspaceRouteError` extend `SolveNotCertifiedError` (so they escape the two `'singular_system'`
  folds); `SchurRankError` extends plain `Error` (so it folds, because "this one really is a singular
  system", `schur.ts:199-203`). Each class's base choice carries a `Why:` naming the alternative and
  the harm it would cause. `linsolve.ts:421-440` even records that the mis-recovery is currently
  LATENT (`src/gpu/driver.ts` has no `catch`) and says "do not delete it as unused".
- **The topology guard is provably correct and loud.** `hierarchicalSaddle.ts:215-251`: `n ≥ 3`,
  `|E| = n`, every edge maps to a unique slot in `{(i,i+1)} ∪ {wrap}`, with the pigeonhole step stated
  at `:250` ("|E| = n and no slot twice ⇒ every slot once, so the wrap is present"). Four distinct
  `HierarchicalSaddleUnsupportedError` messages, each naming the dense fallback.
- **`publishable` can fire both ways.** `hCalibration.ts:2393` is `every(v => v.verdict !== 'INVALID')`
  and `:653` shows a gate can return `'INVALID'` — so it is not a predicate that cannot fail, which is
  the inverse hazard A46 item 2 raised.
- **`operatorSaddleResidual`'s missing `C`-row-length guard is closed upstream** at
  `schur.ts:476-479` (`C[r].length !== m` throws), and `nullspace.test.ts:1207` exercises a short row.
  Only a direct external caller of the exported residual could reach it.
- **All citations the branch *did* update are correct**: `optimizer.ts:448`, `:544-552`, `:314`,
  `:308-327`, `:328-329`; `lineSearch.ts:279`, `:329`; `gradient.ts:139`; `linsolve.ts:410-413`,
  `:927`, `:1139`, `:1133`; `saddle.ts:35-40`. I resolved each one by hand.
- **`§7.3` anti-trap 1's claim about `structuredSaddleResidual`** ("requires the dense matrix and
  cannot exist on the hierarchical path at all") is what `operatorSaddleResidual` exists to answer,
  and its TSDoc's bit-comparability argument (same row visit order, same per-row accumulation order,
  operator output as the starting accumulator) matches the dense sibling at `linsolve.ts:937-970`
  line for line.
