# Open findings, verbatim as the two blind reviewers filed them

Scope rule for the re-review: verdict EACH item `ADDRESSED` or `NOT ADDRESSED` with file:line.
A new finding counts ONLY if it is Critical or Important AND lies inside the fix diff. Everything
else goes to an issue and does not extend the loop. **No new Minor findings** — this is round 2.

## C1 — CRITICAL (compliance reviewer)
§8.2's hybrid dispatch does not exist. `hierarchicalSaddle.ts` `prepare` calls `prepareSchur`
unconditionally with no test on `k`, so `nullspace.ts` (753 src + 1336 test lines) has zero `src/`
and zero `bench/` consumers, and the `{barycenter, edgeLengths, pins}` set that §8.2:2787 says
"must actually work" throws `SchurRouteError`. Falsifier: hybrid k=30 throws, control k=6 succeeds.
Row 9b's verification column requires the dispatch. No amendment parks it.

## I1 — IMPORTANT (numerics reviewer)
H5 property 2's bar `κ_excl < 40` has no reachable RED side in any configuration the shipped gate
can execute. At n=960 the λ₂=1.5 boundary is m̂≈69.5, so the in-band supremum is ≈36.2 (reviewer, at
θ=0.5) against a bar of 40. `precond.test.ts:436-438`'s "DEMONSTRATED sensitivity floor m̂ ∈
(70,100]" is entirely outside the band the same comment defines sixteen lines earlier, and every
executed knock-out lands on nullDim/λ₁ instead. Reachable on the n axis at n≈6.9e3–1.2e4 (inside the
scale target) but the instrument is O(n³). `gateH5:1971` only ever builds trefoil.

## I2-numerics — IMPORTANT
H1-b's lower bound is an n-independent `1e-7` guarding a quantity growing ~n^0.71. Margin above the
floor: 10.52× at n=960, 2.42× at n=120, 1.61× at n=64, 1.07× at n=32 — crossing below n≈30, where
`gateH1b` publishes RED on a correct operator. `gateH1b` takes `ns[ns.length-1]` silently with no
rung-validity check, and the row publishes `marginBelowBar` for the upper side and no margin at all
for the floor.

## I2-compliance — IMPORTANT
H4c published `GREEN`; spec A47 item 7 rules it MIS-AXED and states neither VACUOUS nor MIS-AXED may
be reported as GREEN. `hCalibration.ts`'s `Verdict` union had no such member; 0 occurrences in
harness or artefact. H4b′ *was* correctly labelled VACUOUS, so the distinction was understood and
applied to only one of the two gates.

## I3-compliance + M4-numerics — IMPORTANT
Six sites still justify `DEFAULT_THETA = 0.25` by "H6 read RED", which A46 item 7 withdrew —
including `apply.test.ts:641`, the site A46 named by file:line and which `gateH6`'s own
`consequence` field names. The artefact already complies (`thetaProduction: null`), so code and
artefact disagree about whether θ is settled. The value 0.25 is not challenged, only the anchor.

## I4-compliance — IMPORTANT
`gmres.ts:37-41` quotes §10.2's retracted `4.372e8…1.717e12` / exponent `+3.056` and labels it
MEASURED; shipped values are ~6.7 decades smaller. Second half is a spec defect: the two cited
passages (`§7.3:1439-1441`, `§7.4:1920-1922`) lack the in-place A30 marker their immediate
neighbours at `:1453`/`:1951` carry — an unamended body-vs-amendment contradiction.

## I5-compliance — IMPORTANT
28 broken line-number citations, 27 previously correct, from this branch's own +200 lines in
`linsolve.ts`. Worst: `gradient.test.ts:240` cites `linsolve.ts:924-925` for "no factorization; that
is the point of the reuse", which now resolves to `ldltFactor(K, size) : luFactor(K, size)`.

## I6-compliance — IMPORTANT
`dispatch.ts:166-172` — unchanged by the branch, but its `saddle` type silently widened; the only
caller-facing surface documents neither the arm, the mandatory `projectionMode: 'frozen'`, nor the
producer. The producer's own TSDoc discharges A33 item 2 well; the consumer end is the gap.

## M3-numerics — MINOR
`apply.test.ts:55` writes `U_F32 = 5.96e-8` where A39 item 1 pins `2**-24`.

## M7-numerics — MINOR
The per-exponent-`pow` anchor's magnitude claim does not reproduce: f32 difference is 4.1e-8 mean /
2.4e-7 max, ~240× NARROWER than the H1-b band, not wider. Its conclusion stands on
`nearField.test.ts:139-167`, the cross-form pin at 4e-15.
