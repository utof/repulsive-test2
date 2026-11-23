# Plan review — round 2 (informed, scoped)
Plan: `docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md` (branch `feat/webgpu-phase1`, 1351 lines)
Inputs: `/tmp/plan-review-r1.md` (findings #1–#13), `/tmp/plan-r1-fix.diff`, current plan, spec `2026-08-13-webgpu-solver-design.md`.
Scope for new findings: the fix diff only. Everything marked **[verified]** was read/executed this session (three r185 sources under `node_modules/three/src`, repo sources, `bun /tmp/pe.ts`). No repo file modified.

---

## A. Verdict per r1 finding

| # | Sev | Verdict | Evidence (plan line numbers, current file) |
|---|-----|---------|--------------------------------------------|
| 1 | Critical | **ADDRESSED** | `pairKernelWgsl` no longer concatenated into entries (removed at diff L116/L152; plan :292 "ENTRY FUNCTION ONLY", :412). Constructor uses `const shared = wgsl(pairKernelWgsl(params))` + `wgslFn(src, [shared])` (:821–824). **[verified]** `CodeNode.js:170` `wgsl(src, includes)`, `CodeNode.js:99–113` builds includes before emitting; `FunctionNode.js:168` `wgslFn(code, includes)`; `FunctionNode.generate` calls `super.generate(builder)` first (includes emitted before the fn). New guard test "entry sources are ONE bare fn each" (:152–159) asserts `trim().startsWith('fn ')`, no `struct KD`, exactly one `fn` per entry, and the include has no `fn tpGradient(`. |
| 2 | Critical | **ADDRESSED** | `positions` is `.toReadOnly()` (:808, comment :805–807); `tpVertexGather` declares `slots: ptr<storage, array<f32>, read_write>` (:387) with rationale (:379–381); Verify-step gained the access-mode rule (:866). **[verified]** `StorageBufferNode.js:267` `toReadOnly()` = `setAccess(READ_ONLY)` returns `this`, so `.value.setUsage(...)` on the next line still works. All other pointer modes checked against the bound nodes: `edges/pairOffsets/pairIndices/incidentOffsets/incidentSlots` `.toReadOnly()` (:810–814) ⇔ `read`; `slots/grad/partials` plain `instancedArray` ⇔ `read_write` (:307, :387, :390, :419). Consistent. |
| 3 | Critical | **ADDRESSED** | WGSL: `if (rd > dGuard)` :260, `if (rc > crossGuard(e, d))` :271, `if (reI > diffGuard(...))` :319. Mirrors: `<=` at :638 and :645. D3 prose explains why the direction is load-bearing (:92). Tests: `[4,0,0]×[2,0,0]` (guard exactly 0) :539, `[1,1,0]×[2,2,0]` :540, `diffDegenerate([0,0,0],[0,0,0])` :534, regex `not.toMatch(/>=\s*(dGuard|crossGuard|diffGuard)/)` :143, new `crossing` preset test :545–553, `finite` per fixture in `phase1Tolerance` :927–931. **[verified]** `crossing` has 8 vertices / 6 edges (`testConfigs.ts:19–36`), so `edges[3]` = `[4,5]` and `vertices[edges[3][0]]` = v4 = (0,−1.5,−0.3) is off the x-axis as the test comment says; e0×(v0−v2) = (1,0,0)×(−2,0,0) = 0 with guard 0 ⇒ `<=` ⇒ degenerate. (r1's own description of `crossing` as 4 vertices was wrong; the plan's test is right.) |
| 4 | Critical | **ADDRESSED** | Implementation :198–204 now matches the tests :125–129. **[verified]** by execution: `["(x)","((x * x) * x)","1.0","(1.0 / ((x * x)))","pow(x, 2.5)"]` — all five expectations hold. Comment :200 states the test strings are canonical. |
| 5 | Major | **ADDRESSED** | `prevLost.call(renderer, info)` :1258, `prevErr.call(renderer, info)` :1265, with the "UNBOUND prototype methods that use `this`" note :1254. **[verified]** `Renderer.js:595` `this.onDeviceLost = this._onDeviceLost;`, `:606` `this.onError = this._onError;`, `_onDeviceLost(info)` at `:1225` sets `this._isDeviceLost = true` at `:1237`; `_onError(info)` at `:1247`. `.call(renderer, …)` is correct and necessary for `_isDeviceLost` (short-circuits at `:886/:1521/:2720`). |
| 6 | Major | **ADDRESSED** | `GpuDriver.step` snapshots once: `const vertices = st.live.map(...)`, `const args = { ...buildStepArgs(st, energyBefore), vertices }`, `this.engine.gradient(vertices)` (:1211–1213), and `dispatchDescentStep({ ...args, ...})` reuses `args.vertices`. **[verified]** `dispatch.ts:327–334` `buildStepArgs` returns `Omit<DispatchDescentStepArgs,'collectTimings'|'collectField'>` with `vertices: state.live` (a reference, not a copy — the plan's comment "`buildStepArgs` copies `st.live` internally" at :1208 is **wrong**, see N1 below, but the code is right regardless: the spread-override replaces the reference with the copy). `Vec3` is imported in `driver.ts` (:1172). Type-correct: the override has the same `Vec3[]` type. |
| 7 | Major | **ADDRESSED** | D4 rewritten :96 with the three error sources (per-edge sum, tree reduction, per-term arithmetic), expected 2–5e-7, ~2–5× margin, the pre-registered band [5e-8, 1e-6], the <1e-9 "vacuous" trigger, and the blocked-sum contingency. Matches the r1 derivation. |
| 8 | Major | **ADDRESSED** | Task 3 split 3a/3b/3c (:656, :662, :770 commit 3a, :870–872 Step 4½ + commit 3b, :874 3c); Task 7 split 7a/7b (:1239); protocol updated (:1337: 1, 3b, 3c, 7b on the orchestrator model; 3a/7a Sonnet); self-review line (:1346). The 3a reduction-only spike asserts exact integer sums for n ∈ {1,63,64,65,128,129,960,16384} — a proper guard/tree oracle. Step 4½ smoke covers compile + per-vertex parity on `trefoil(12)`, `crossing`, `nearTouchPair(1e-6)`. |
| 9 | Major | **ADDRESSED** | Task 5 spec amendments now include §6 (:1069) with replacement wording. **[verified]** spec `:431–433` still reads "the GPU energy is the ε=0 energy [PREC Q1]" today (the amendment is applied in Task 5's commit, as planned). |
| 10 | Minor | **ADDRESSED** | Length check removed from inside the `timed('step')` closure (diff L332) and placed "as the first statement of the function body", before `timingsBegin()` (:1062). **[verified]** `optimizer.ts:262` `if (collect) timingsBegin();` and `vertices`/`opts` are parameters of `sobolevStepSet` (`optimizer.ts:231–236`), so the check is in scope as the first statement. |
| 11 | Minor | **ADDRESSED (r1 was factually wrong)** | **[verified]** `BufferAttribute.js:683` `dispose() { this.dispatchEvent({ type: 'dispose' }); }` ("Available only in WebGPURenderer") — the method exists in r185; `ComputeNode.js:116` `dispose()` likewise. The plan now calls both unconditionally (:862–863) and cites both lines (:860–861). Ruling: the r1 claim "no dispose in r185" is refuted; the fix is correct. |
| 12 | Minor | **ADDRESSED** | "Loop component" → `Simulation` component `:57` (:1236, :1269) **[verified]** `Viewer.tsx:57`; implementer comment removed (diff L355); generation bump placed before the early return (:1273); `REDUCE_LANES` consumed (:782, :826–827, :831) so knip will not flag it; negative-constraint line now asserts a ceiling (:1323). D2 "divide by zero" wording: not touched in the diff — inert wording nit, ledger. |
| 13 | Minor | **ADDRESSED** | :1273 introduces `needsWorker = solverDriver !== 'main'` and explicitly accepts the worker↔gpu recreation with its cost and why it is safe. |

**Item C(ii) — effect-order claim, checked against `Viewer.tsx:207–246` [verified]:** the effect's deps are `[solverDriver, handleWorkerMessage]`. Putting `dispatchGeneration.current++; inFlight.current = null;` as the first two statements, then `if (!needsWorker) return;`: (a) toggle → `'main'`: previous run's cleanup fires (terminate worker, `inFlight = null`), the new run bumps the generation and returns — so any GPU/worker promise that resolves later fails the `token.generation !== dispatchGeneration.current` check at :1291; (b) toggle `'worker'` ↔ `'gpu'`: cleanup + bump + new worker; (c) mount: generation becomes 1 — harmless. Under 7a (before Task 6 adds `'gpu'` to `SolverDriver`, `store.ts:108` is `'worker' | 'main'` today), `!== 'main'` ≡ `=== 'worker'`, so 7a is semantics-preserving as the plan claims. The `worker.onerror` handler's `inFlight.current = false` (`Viewer.tsx:229`) is covered by item 2's "every `= false` → `= null`". Behaves as claimed.

**Item C(iii):** confirmed above (#10). **Item C(iv):** confirmed above (#6); see N1 for the wrong comment.

---

## B. New findings inside the fix diff

### Important+ : none.

### Nits (ledger / inline-fix gate; do not extend the loop)

- **N1 — wrong load-bearing comment in `GpuDriver.step` (:1208–1210).** The comment says "`buildStepArgs` copies `st.live` internally". It does not: `dispatch.ts:333` is `vertices: state.live` (reference). The *code* is correct precisely because it does not rely on that. Fix the comment to: "`buildStepArgs` passes `st.live` by REFERENCE (`dispatch.ts:333`); PinControls mutates `live` in place across our `await`, so copy once here and hand the SAME copy to the kernel and the step." Per CLAUDE.md "anchor the magic", a wrong anchor invites the next agent to delete the copy. One-line docs fix — inline.
- **N2 — Task 7 split header cites the wrong finding:** "(review r1 #9)" at :1239 should be "#8" (r1 #9 is the spec §6 amendment). Cosmetic.
- **N3 — Step 4½ smoke thresholds are stricter than the pre-registered T3 gate** (:872: per-vertex relative error < 1e-4 with a 1e-12 floor, vs T3's global cosine > 1−1e-6). On `nearTouchPair(1e-6)` the D2 magnitudes (coeff_d ≈ 4e26) make a per-vertex 1e-4 relative bound plausible but unproven; a red here on 3b would be a *smoke* failure, not a gate failure. Suggest stating in the step that (c) is triage output and (b)+(d) plus "no exception" are the pass criteria, or pre-register the per-vertex threshold as informational. (`crossing` passes (c) vacuously: its CPU gradient is ~1e-21 from the ε-only term and the floor is 1e-12.)
- **N4 — seam-diff ceiling regex** `'^[+-][^+-]'` (:1323) skips blank added/removed lines (`+` alone) — fine as a ceiling; note it counts comment/TSDoc lines, so 30 is tight if the TSDoc block for `dE?` is verbose in both files. Bump to 40 or say "TSDoc ≤ 6 lines per field". Judgment call.
- **N5 — D2 "divide by zero" wording** (r1 #12 bullet 4) was not changed; inert.

---

## C. Source verification summary (all [verified] this session)
- `Renderer.js:595/606/1225/1237/1247` — unbound prototype methods using `this`; `.call(renderer, info)` correct.
- `BufferAttribute.js:683` `dispose()` exists; `ComputeNode.js:116` `dispose()` exists.
- `CodeNode.js:99–113, 170`; `FunctionNode.js:112–140, 168` — `wgsl(src, includes)` / `wgslFn(code, includes)`; includes built before the entry fn.
- `StorageBufferNode.js:267` `toReadOnly()` returns `this`.
- `Viewer.tsx:207–246` effect structure and deps; `optimizer.ts:231–236, 262`; `dispatch.ts:327–334`; `testConfigs.ts:16–36, 271`; `store.ts:108`.
- `powExpr` executed: matches all five test strings.

---

## D. Verdict: **APPROVE**

All thirteen r1 findings are ADDRESSED (r1 #11 is refuted by source: `BufferAttribute.dispose` exists at `BufferAttribute.js:683`). No Important-or-higher finding was introduced by the fix diff. Recommend the author apply N1 (one-line comment correction — the only nit that could cause a future regression) and N2 during the amend; N3–N5 go to the ledger.
