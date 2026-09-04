# WebGPU Solver — Phase 2a-1 (core seam + CPU f64 reformulation + gate CR) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the source-selection seam that lets a factorization computed OUTSIDE a descent step serve that step's gradient and projection solves, implement the exact Schur+Woodbury+IR reformulation in CPU f64, and prove the reformulation exact against the shipped `solveSaddleFromA` (gate CR) — all before a line of WGSL exists.

**Architecture:** Two layers and one gate. (1) `src/core/` gains ONE public-type delta — an optional `saddle?: PreparedSaddle` riding `dispatchDescentStep → sobolevStepSet → solveConstrainedGradientSetFrozen` exactly as Phase 1's `dE?` does, selecting the *source* of A and of the factorization and never the arithmetic; absent ⇒ every existing path is byte-for-byte unchanged. (2) `src/gpu/saddle.ts` is pure CPU f64 — shift, Schur complement, Woodbury correction, iterative refinement — with **no `three` import**, so it runs under `bun test`. (3) Gate CR runs the whole reformulation on an f64 Cholesky and compares against `solveSaddleFromA` to a κ-relative bar. **CR red ⇒ no GPU work starts** [DESIGN §6].

**Tech Stack:** Bun (tests), TypeScript strict, Biome; numpy/scipy via `uv` for the oracle twin. **No `three`, no WebGPU, no browser, no renderer in this plan** — that is what makes 2a-1 independently gated.

**Spec:**
- **[DESIGN]** `docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md` — §1 (scope), §4.1 (the algebra), §4.2 (precision model), §4.4 (the seam — public-type delta (iv)), §5 (gate quantities, fixtures, pass semantics), §6 (kill gates), §7 (risks).
- **[CAL]** `docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-gate-calibration.md` — §A (provenance tiers), §E.5/§E.6 (the measured IR budget), §F (CR's bar and its rhs), §J.3 (CR's slack is a deliverable).
- Parent spec `docs/superpowers/specs/2026-08-13-webgpu-solver-design.md` cited as **[P §n]**.

**Why this plan is 2a-1 and not 2a:** the Phase 2 spec review reached its 3-round blind cap with a silent-failure-class finding still open, and the adjudicated remedy was to SPLIT rather than run a 4th round. 2a-1 is the piece that depends on **no contested speed number**: it is CPU-only, numerics-only, and CR is a correctness bar derived from measured κ. `GpuCholesky` + K1 (**2a-2**) and driver/store integration + P2a + `G5-confirm` (**2a-3**) are separate plans and MUST NOT be anticipated in code here.

**Phase entry:** Phase 1 merged (PR #26, `bf0abcf`). [DESIGN §0]'s state detector reads: `src/gpu/cholesky.ts` absent ⇒ Phase 2 not started; `src/core/sobolev/linsolve.ts` has no `'external'` factorization kind ⇒ the core seam has not landed. Both hold at branch point.

## Global Constraints

- **Slice 2a-1 only.** Deliverables are exactly: the core seam, `src/gpu/saddle.ts`, `test/gpu/saddleSeam.test.ts`, `test/gpu/saddle.test.ts`, `test/gpu/cr.test.ts`, `oracle/tpe_phase2a_reformulation.py`, and one committed CR results JSON. **Nothing renderer-bound.** If a step here appears to need `three`, `GpuCholesky`, the driver, the store or a spike, it belongs to 2a-2/2a-3 — stop and report.
- **No `src/core/**` numeric changes** ([DESIGN §1], [P §1]). `assembleAFlat`, `ldltFactor`, `luFactor`, `buildSaddleMatrix`, `lineSearchStepSet`, `projectOntoConstraintSet`, `FactorMode`, `factorMode` are untouched. The seam is **source-selection only**; absent ⇒ bit-identical.
- **What actually backstops "absent ⇒ bit-identical", precisely.** It is the repo's **committed golden suites** — `test/golden.test.ts` and `test/sobolev/constraintSetFlow.test.ts` against `oracle/golden/*.json`, whose expected values were captured BEFORE this branch existed. Those are the only assertions in the tree that compare post-seam output against a *pre-seam* reference. Task 1's own seam-absent test cannot do it and does not claim to: both of its arms run the NEW code, so a regression the seam introduces is present on both sides and cancels (it proves determinism, which is a precondition for the golden comparison, not identity). Say this plainly rather than letting a test name imply a guarantee it does not carry — the wrong version of this sentence is what makes a future agent delete the golden run and keep the cheap one.
- **Sanctioned public-type delta (iv)** ([DESIGN §4.4] — AMENDS [P §1], which names only (i)/(ii)/(iii)): `ExternalFactorization`, `ExternalSolveError`, `PreparedSaddle` in `linsolve.ts`; `SaddleFactorization` widened; `solveConstrainedGradientSetFrozen` gains a TENTH optional positional parameter; `SobolevStepOptions.saddle?`; `DispatchDescentStepArgs.saddle?`. **No other export.** `DescentStepOutcome` is NOT touched. `SimStore.gpuSolveAvailable` is delta (iv) too but belongs to **2a-3** — do not add it here.
- **`solve` must self-certify or throw** ([DESIGN §4.1] step 4). Nothing in `src/core/` thresholds `residual`, so an unconverged `z` returned from `PreparedSaddle.solve` is a silently wrong descent direction. Exhausting the refinement budget without reaching `1e-10` ⇒ `throw new ExternalSolveError(...)`.
- **The certification is against the PURE relative residual `‖r_i‖₂/‖r‖₂`**, computed by `src/gpu/saddle.ts` itself. It may NOT be delegated to the `residual` that `solveSaddleFrozen` returns (`linsolve.ts:903`), which normalises by `max(1, ‖r‖₂)` and is permissive by the full factor `1/‖r‖₂` on the projection rhs ([DESIGN §4.1] step 4). The value `solveSaddleFrozen` returns is still forwarded as the step's `residual` stat, byte-identically — the seam changes what the EXTERNAL solve certifies, never what the core reports.
- **Coordinate-major throughout.** `Ā = I₃ ⊗ A`: block `b` of a 3N vector is coordinate `b` over ALL vertices, occupying rows `[b·N, (b+1)·N)`. This is what `solveSaddleFromA` writes (`linsolve.ts:770-779`, `off = b * n`) and what the oracle uses (`oracle/tpe_stage1_oracle.py:245-248`). Writing `A ⊗ I₃` (interleaved, vertex-major) is the opposite of the code and every test in Task 2 would fail. [DESIGN §4.1] flags this explicitly because an earlier draft had it backwards.
- **The one live bar — CR** ([DESIGN §5] quantity/fixtures, [CAL §F] level):
  - Quantity `‖Δz‖/‖z‖` between z from [DESIGN §4.1] steps 1–3 on a CPU **f64** Cholesky and `solveSaddleFromA`'s z. **Steps 1–3, NOT 1–4** — the gated vector is the one `schurWoodburySolve` returns, BEFORE step 4's iterative refinement. `PreparedSaddle.solve()` runs steps 2–**4**, so gating on its output would let any defect in steps 1–3 that leaves the relative residual below `SADDLE_IR_TOL` be silently repaired by IR before CR looks — and CR exists precisely to gate the REFORMULATION before any WGSL, while step 4 is the thing that will later be running on a wrong f32 factor. The two quantities are not equal: measured at trefoil960/`gradient:production`, steps 1–4 gives 1.196e-10 and steps 1–3 gives 1.452e-10 (`+pin`: 1.305e-10 vs 1.501e-10), and at trefoil960/`gradient:nullspace1` step 4 moves the answer the WRONG way — 4.711e-10 → 4.959e-10 (`+pin`: 4.369e-10 → 4.685e-10) — which is exactly the point: IR minimises the residual, not the error. The two values differ ONLY on the nine rows where IR fires at all (all at n=960 — Task 3 Step 8 lists them); on the other 147 of the 156 they are identical to the digit (`scratch/irrand0.ts`, `scratch/rowsall.ts`). Both clear the bar, so this is a structural fix, not a red gate. The steps 1–4 value is recorded too, as a separate field.
  - Bar `max(1e-12, 10·κ₂(K,N)·u_f64)`, κ₂ from **[K1CAL] `rows[].kappaK`**, `u_f64 = 2⁻⁵³`. Evaluates to **1.6e-12 / 1.6e-11 / 1.6e-10 / 1.6e-9** at N=120/240/480/960.
  - **Both rhs shapes** ([CAL §F]), across **13 rhs per fixture** (D5): **three GATED** — gradient `r = [b; 0]` with the production `b`, projection `r = [0; d]` with the production `d`, and gradient `r = [𝟙; 0]` (the adversarial null-space probe) — plus **ten RECORDED-not-gated** seeded draws (5 per shape). The bar applies to each gated row; all 13 are reported per fixture with a `gated` flag.
  - **CR red ⇒ no GPU work starts** ([DESIGN §6]) — 2a-2 does not begin.
  - **MEASURED CALIBRATION DEBT, owed to [CAL §F] — this plan records it and does NOT fix it.** The 10× headroom in `max(1e-12, 10·κ₂(K,N)·u_f64)` is **insufficient for a GENERIC rhs at n=960**: measured effective constant ~15·κ·u, and one of the ten seeded draws exceeds the bar **with a correct f64 factor** at each of the two n=960 constraint sets (two rows in the 156-row artifact), which iterative refinement cannot repair — it drives the residual down, not the error, once the error is at the conditioning floor, and here it does not even fire (`gradient:rand0`: 2.583e-9 at both steps 1–3 and steps 1–4, `irRefinements = 0`). The three gated rhs are unaffected (worst gated slack 3.34× at n=960). **Any future change to CR's rhs set must revisit the headroom first.** Evidence: `.superpowers/sdd/2026-09-04-webgpu-phase2a-1/scratch/portable.ts`, `scratch/irrand0.ts`. This is a **spec fix, not a plan section** — the plan must not amend a spec (see the "Pre-registered bars live in the spec" constraint below); Task 3 Step 8 carries it to the orchestrator as an owed item.
- **Bars this plan does NOT gate**, so an executor does not think they are missing: **K1**, **T4**, **Boot** (2a-2 — they need the GPU kernel); **T5**, **P2a**, **`G5-confirm`** (2a-3 — they need the driver, and P2a/`G5-confirm` additionally carry the calibration preconditions of [CAL §J.1]/[CAL §J.2]). `SADDLE_IR_TOL` and `SADDLE_IR_MAX_REFINEMENTS` are *defined* here because [DESIGN §4.1] step 4 puts them in the solve; T4 *measures* them in 2a-2.
- **CR's slack is a deliverable, not an input** ([CAL §J.3]). No slack figure is quotable today — the four reformulation errors an earlier draft quoted were withdrawn because no committed artifact contains them. Task 3's JSON is the first committed source. Do not write a slack number into any doc that Task 3 did not emit.
- **Pre-registered bars live in the spec, not here** (CLAUDE.md). If this plan appears to need a gate the spec does not carry, that is a spec fix, not a plan section.
- **`bunx knip` baseline — it exits 1 TODAY, and every Run step below is written against this, not against exit 0.** Measured on this branch (`e204b04`), captured verbatim into `.superpowers/sdd/2026-09-04-webgpu-phase2a-1/scratch/knip-baseline.txt`:
  - `exit 1`. **Unused files (6):** `bench/gpu/drive.ts`, `bench/gpu/jsAlgebra.ts`, `bench/gpu/phase1.ts`, `bench/gpu/spikes.ts`, `oracle/compare_energy.ts`, `oracle/gen_fixtures.ts`.
  - **Unused exports (8):** `xorshift32`, `SOBOLEV_CONVERGENCE_TOL`, `assembleBHighFlat`, `assembleBLowFlat`, `projectOntoConstraintSet`, `projectBarycenter`, `loadSavedConfig`, `saveConfig`.
  - **Unused exported types (9):** `FieldArgs`, `SobolevStepFailureReason`, `ProjectConstraintSetOptions`, `ProjectBarycenterOptions`, `ProjectConstraintSetResult`, `ProjectBarycenterResult`, `FieldPenalty`, `PhaseSample`, `SavedConfig`.
  - **Compare the NAMED SETS, never the raw counts.** `knip.json`'s `project` glob is `**/*.{ts,tsx,js}`, and knip counts matches under `.superpowers/sdd/**` even though git ignores that tree (`.superpowers/sdd/.gitignore` is a bare `*`) — so every `.ts` file a review or a probe leaves there is reported as an unused file. Measured: the same branch reports **6** unused files with the workspace clean and **10** with the round-1 review probes present, for a reason that has nothing to do with this plan's code. An executor who diffs counts will chase a phantom; an executor who diffs the named sets will not.
  - **Expected new findings after Tasks 1–3: NONE.** `knip.json`'s `entry` includes `test/**`, so a test-only importer counts as a consumer, and every new export has one: `PreparedSaddle` / `ExternalSolveError` / `shiftSobolev` / `choleskyF64` / `makePreparedSaddle` / `saddleSetup` / `woodburyM` / `kappa1of3` / `schurWoodburySolve` from the three new test files; **`ExternalFactorization`** via `asExternal`'s return type; **`SADDLE_IR_TOL`**, **`SADDLE_IR_MAX_REFINEMENTS`** and **`IrTrace`** from Task 2 Step 5's tests, which import them instead of hard-coding `1e-10` and `4` (an earlier draft hard-coded both, which would have added three findings AND left the tests pinning a budget the constants define — the two problems have one fix). If a Run step reports a new name anyway, that is a regression to investigate, not a line to add here.
- **Conventions:** Biome (4-space, single quotes, width 100); `import type`; `bunx tsc --noEmit` clean; `bun test <path>`; `bunx biome check --write` on touched files; `bunx knip` no new findings against the named baseline above. **TSDoc on every export with `@see <spec/plan path>` or a `Why:` line, and an inline anchor on every guard, every ε, every `/2`, and every op-order-preserving line** (CLAUDE.md — the reference is what stops a future agent from "cleaning up" an intentional line). **One commit per task.** Trailer on every commit:
  ```
  Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01JdMoLJiC1mByydnaHhuGbr
  ```
- **Test baseline: `292 pass / 0 fail across 37 files`**, measured on this branch (`e204b04`) immediately before this plan was written. Every "full suite" step compares against 292 + the tests that step adds. (An earlier draft said 307; that is the count on `feat/webgpu-phase2` *with the Phase 1 leftovers*, which is a different branch. Do not use it.)
- Reviewer nits follow CLAUDE.md's inline-fix gate (fix inline only if EVERY gate holds, else `gh issue create` with labels). Blockers are fixed on the branch regardless of size.
- **Locator provenance:** every `file:line` below was re-verified through `codebase-memory-mcp` against `e204b04` while this plan was written (the index was live; `check_index_coverage` reports `no_recorded_issue` + `metadata_match` for all nine seam files). Re-verify before editing if the branch has moved.

---

## File structure

| Path | Responsibility |
|---|---|
| `src/core/sobolev/linsolve.ts` | **Modify.** `ExternalFactorization`, `ExternalSolveError`, `PreparedSaddle` (new exports); `SaddleFactorization` widened at `:399`; `solveFactored` (`:696-698`) gains the `'external'` branch. Nothing else changes. |
| `src/core/sobolev/gradient.ts` | **Modify.** `solveConstrainedGradientSetFrozen` (`:84-115`) gains the 10th positional param `saddle?: PreparedSaddle`; with it, `A = saddle.A`, `fac = saddle.factor(C)`, and the solve routes through the already-exported `solveSaddleFrozen`. |
| `src/core/optimizer.ts` | **Modify.** `saddle.A` shape guard beside the `dE` one at `:265-266`; `SobolevStepOptions += saddle?` (after `factorMode?` at `:174`); forward as the 10th arg at `:343-353`; rethrow `ExternalSolveError` from the bare catch at `:361`; disarm the timing collector on the throw path around `:282`. |
| `src/core/dispatch.ts` | **Modify.** `DispatchDescentStepArgs += saddle?` (beside `dE?` at `:164`), forwarded beside `dE: args.dE` at `:270`, plus a NEW `import type { PreparedSaddle }` line (`dispatch.ts` imports nothing from `./sobolev/linsolve` today, `:1-18`). |
| `src/worker/solverWorker.ts` | **Modify.** Defensive throw if a `saddle` ever appears on a `step` payload's `args` (it cannot be structured-cloned). |
| `src/gpu/saddle.ts` | **Create.** Pure CPU f64: `shiftSobolev`, `choleskyF64`, `makePreparedSaddle`, `asExternal` — [DESIGN §4.1] steps 1–4 (shift, Schur, Woodbury, IR + the `ExternalSolveError` throw). Task 3 Step 1 adds `saddleSetup`, `woodburyM`, `schurWoodburySolve`, `kappa1of3`. No `three` import ⇒ unit-testable under `bun test`. |
| `test/gpu/saddleSeam.test.ts` | **Create (Task 1).** Absent-seam determinism + present-seam routing + `ExternalSolveError` propagation + collector-disarm (both halves, both asserted on the collector's own state) + the `saddle.A` shape guard + the worker guard. |
| `test/gpu/saddle.test.ts` | **Create (Task 2).** `shiftSobolev`, `choleskyF64`, Schur/Woodbury agreement, IR trace, the throw. |
| `test/gpu/cr.test.ts` | **Create (Task 3).** Gate CR as a `bun test`: all fixtures × both constraint sets × both rhs shapes, κ-relative bar read from the committed [K1CAL] JSON, and the results-JSON emitter. |
| `oracle/tpe_phase2a_reformulation.py` | **Create (Task 3).** The numpy twin of [DESIGN §4.1] steps 1–3 — CR's independent reference, so a CR red is localisable to the TS or to the algebra. |
| `bench/results/gpu-phase2a-cr.json` | **Create (Task 3, emitted by the test run).** The committed artifact that retires [CAL §J.3]. FIXED filename — a date-derived one writes a second artifact on a later day instead of refreshing this one; provenance rides `gitShaShort` inside the file. Contains NO timestamp, so the output is byte-stable and `bun test` re-runs produce a diff only when the numbers change. |

---

## Decisions (read before Task 1; each is falsifiable)

### D1 — Disarming the timing collector: catch-and-rethrow, NOT a bare `finally`

[DESIGN §4.4] requires that a throw out of `timed('step', …)` must not leave the module-scoped collector armed. **The `finally { if (collect) timingsEnd(); }` spelling is wrong**, and [DESIGN §4.4] says so explicitly; this plan implements the requirement, not the sketch.

`timingsEnd()` is destructive, not idempotent — it returns the ledger *and* nulls `acc`:

```ts
// src/core/sobolev/phaseTimings.ts:57-61  (verified via codebase-memory-mcp)
export function timingsEnd(): SobolevStepTimings | null {
    const r = acc;
    acc = null;
    return r;
}
```

The success path reads that ledger at `optimizer.ts:460-461` (`if (collect) { const timings = timingsEnd(); if (timings) …}`). A `finally` wrapped around `timed('step', …)` runs FIRST, nulls `acc`, and `:461` then returns `null` — so `collectTimings: true` silently stops returning `timings` on **every normal step**. That breaks `bench/sobolev.bench.ts`, Phase 1's `if (usedGpu && outcome.timings)` stamp (`driver.ts:175`), and every later gate that reads `timings.factor`.

**Decision: disarm on the throw path only.** The existing code is `const outcome = timed('step', (): {…inline return type…} => { …body… });` at `optimizer.ts:282`. A `try`/`catch` cannot assign to a `const`, so hoist the *existing* inline return-type annotation to a local type alias and keep the body verbatim:

```ts
// Declared just above the call site; these are EXACTLY the fields the existing
// inline annotation at optimizer.ts:284-290 already lists — no field is added,
// removed or renamed.
type StepBodyOutcome = {
    vertices: Vec3[];
    energy: number;
    accepted: boolean;
    converged: boolean;
    stats: SobolevStepStats;
};

let outcome: StepBodyOutcome;
try {
    outcome = timed('step', (): StepBodyOutcome => {
        /* …existing body, verbatim… */
    });
} catch (e) {
    // The collector is module-scoped and only timingsEnd() disarms it
    // (sobolev/phaseTimings.ts:57-61), so a throw out of timed('step', …) would
    // leak THIS step's partial ledger into the NEXT step. Disarm on the throw
    // path ONLY: an unconditional `finally` would null `acc` before the success
    // path reads it at :461 and would silently drop `timings` from every
    // collected step.
    // @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4
    if (collect) timingsEnd();
    throw e;
}
```

`optimizer.ts:460-461` is left exactly as it is.

**Falsifiable:** Task 1 writes tests for BOTH halves, and both assert the collector's OWN state rather than a later step's ledger.

- *Half A — the disarm exists.* After a throwing collected step, `expect(timingsEnd()).toBeNull()`; and separately, after a throwing collected step followed by a `collectTimings: false` step, `expect(timingsEnd()).toBeNull()` again. Without the disarm both return the leaked partial ledger.
- *Half B — the disarm is throw-path ONLY.* A non-throwing `collectTimings: true` step must still return a populated `timings`. The bare-`finally` form passes half A and fails this one.

**Do NOT assert half A as "the next collected step's `timings.step.calls === 1`."** That assertion **cannot fail**: `timingsBegin()` is `acc = {}` (`phaseTimings.ts:48-50`) — an UNCONDITIONAL reset, not an initialise-if-null — so the very next collected step destroys any leaked ledger before recording a single phase, and `step.calls` is 1 whether or not the disarm exists. ([DESIGN §4.4]'s own sketch — "assert the NEXT step's `timings` contains no key from the throwing step" — is vacuous for exactly this reason; this plan implements the requirement, not the sketch, here too.) The real hazard the vacuous form also misses is the SECOND test above: a step with `collectTimings: false` never calls `timingsBegin()`, so a leaked `acc` is never reset — `timed()` keeps accumulating into the throwing step's ledger for the process lifetime, paying `performance.now()` on every phase of every later step and handing a cross-step ledger to whoever collects next. `timingsEnd()` is exported, so both halves are one assertion each.

### D2 — `PreparedSaddle` and `ExternalSolveError` live in `linsolve.ts`

[DESIGN §8] leaves this open (`linsolve.ts` vs a new `src/core/sobolev/prepared.ts`). **Decision: `linsolve.ts`.** `ExternalFactorization` is a member of the `SaddleFactorization` union declared at `linsolve.ts:399`, and `solveFactored` (`:696`) is the only consumer of that union. A separate file would have to import the union and be imported back by `linsolve.ts` for the `'external'` branch — a cycle for zero gain. `PreparedSaddle.factor` returns a `SaddleFactorization`, so it belongs with the union too.

### D3 — The `'external'` branch must be tested with `'kind' in fac`, not `fac.kind`

`LuFactorization` (`linsolve.ts:219-223`) has **no** `kind` field, so `fac.kind === 'external'` is a TypeScript error on the union. `solveFactored`'s existing body is `return 'kind' in fac ? ldltSolveFactored(fac, rhs) : luSolveFactored(fac, rhs);` (`:696-698`) and its anchor comment at `:693-695` explains that the `'lu'` branch is the verbatim pre-existing call. **Both survive**: the new branch is prepended as `'kind' in fac && fac.kind === 'external'`, the LDLᵀ test becomes the second arm, and the LU arm is untouched.

### D4 — The worker guard goes INSIDE the existing `try`, and it reads `msg.args`

`solverWorker.ts:34-36` is `self.onmessage = (event) => { const msg = event.data; try { …`, and the matching `catch` posts `{ type: 'error', message }` back to the main thread (`:89-99` — the file is 100 lines). **Decision: the guard is the first statement inside the `try`, not before it.** Placed before the `try` the throw becomes an unhandled worker error with no message channel; placed inside, the existing error path reports it to the caller that made the mistake. Either spelling "fails loudly"; only one fails *legibly*.

**And it must inspect `msg.args`, not `msg`.** `SolverWorkerRequest`'s `step` arm is `{ type: 'step'; graphVersion: number; args: DispatchStepArgs }` (`dispatch.ts:397-400`) and `DispatchStepArgs = Omit<DispatchDescentStepArgs, 'edges' | 'disjointPairs'>` (`:371`), so adding `saddle?` to `DispatchDescentStepArgs` admits it at **`msg.args.saddle`** and nowhere else. `'saddle' in msg` is `false` for every message the protocol can carry — dead code that reads as a live safety property, while `dispatchDescentStep({ ...msg.args, … })` at `:75-79` spreads `saddle` straight through. `msg.type === 'step' && 'saddle' in msg.args` needs no cast: `msg.type === 'step'` has already narrowed `msg.args` to `DispatchStepArgs`, and `in` accepts any object-typed right operand (verified under this repo's `compilerOptions`). Task 1 Step 7 writes the falsifier — no test in an earlier draft touched the worker at all, so the dead guard would have shipped green.

### D5 — CR gates three physically-reachable rhs and RECORDS a ten-vector ensemble

[CAL §F] pins CR to **both** rhs shapes — gradient `[b; 0]` and projection `[0; d]` — citing `k1-calibration.py:252-254`, which draws them as `np.random.default_rng(0).standard_normal(...)`. **That generator is PCG64 + numpy's ziggurat and is not reproducible in TypeScript**, so a JS test cannot use the same vectors.

An earlier draft substituted the production vectors and argued that the choice was inconsequential because "the rhs is not load-bearing". **That claim was measured and is FALSE at n=960.** It was asserted independently by the plan author, the plan reviewer and the research-verdict agent; the experiment refuted all three. All figures below are the SPEC quantity (steps 1–3, per the Gate CR block) under the SHIPPED generator, measured across all twelve fixture configurations — `.superpowers/sdd/2026-09-04-webgpu-phase2a-1/scratch/portable.ts` (spreads: `scratch/portable-spread.ts`; the IR column: `scratch/irrand0.ts`), all re-runnable with `bun run`. At trefoil960, no pin, against the `1.574e-9` bar:

| rhs | relErr (steps 1–3) | slack | verdict |
|---|---|---|---|
| `gradient:production` | 1.452e-10 | 10.8× | PASS |
| `projection:production` | 2.236e-10 | 7.0× | PASS |
| `gradient:nullspace1` | 4.711e-10 | **3.34×** | PASS |
| `gradient:rand0` | **2.583e-9** | **0.61×** | **FAIL** |
| (9 further rows) | — | 1×–6.7× | PASS |

`trefoil960+pin` (k=7) behaves the same way: gated 10.5× / 7.4× / **3.6×**, and the same single ungated row fails, at 0.60× (relErr 2.615e-9). Spread across the thirteen is 17.8× (no pin) / 17.4× (`+pin`). **At N ≤ 480 no ungated row fails at all** — the minimum slack over all thirteen rows there is 7.55× (trefoil480, no pin; 7.61× `+pin`), and spreads run 2.3×–8.8× — so the failures are isolated to the largest fixture rather than sitting on the edge of spreading down the fixture ladder. **Worst GATED slack anywhere is 3.34×** (`gradient:nullspace1` at trefoil960, no pin).

*(The earlier Box–Müller draft of the ensemble produced TWO failures here, at `rand1` and `rand3`, rather than one at `rand0`; under uniforms those two rows pass, at 1.62× and 3.38×. That the failure COUNT moves with an implementation-approximated `Math.log`/`Math.cos` is exactly why the generator is now uniforms — see the PRNG paragraph below. The gated rows were unaffected by the swap.)*

**This is not a defect, and the decisive check is that iterative refinement does not repair it.** Measured on the failing row (`scratch/irrand0.ts`): steps 1–4 returns 2.5830e-9 against steps 1–3's 2.5830e-9 — identical to five significant figures, with `irRefinements = 0`. IR does not merely fail to fix the row; it never fires, because z₀'s relative *residual* is already under `SADDLE_IR_TOL` while its *error* is over the bar. IR drives the *residual* down, not the *error*, once the error is at the conditioning floor: both the reformulation and the LDLᵀ reference are sitting at their own κ·u floor, and a random top-block rhs excites `A`'s worst-conditioned eigendirections while a physical gradient is smooth and carries little weight there. The measured effective constant for a generic rhs at n=960 is ~15·κ·u, above [CAL §F]'s 10× headroom.

**Decision — gate what the solver can physically encounter, plus one adversarial probe; record the rest.**

**GATED (`gated: true`, three rows per fixture):**

- **gradient shape** — `b = flatten(gradientAnalytical(...))`, `d = 0`. Exactly what `solveConstrainedGradientSetFrozen` passes (`gradient.ts:106`).
- **projection shape** — `b = 0`, `d = −Φ(γ^q)`. Exactly what the frozen projection passes (`lineSearch.ts:260-262`, `negPhi`); non-vacuity is D6's job.
- **gradient shape, `b = 𝟙`** (all-ones over 3n) — **NEW, and the point of this decision.** `𝟙` over each coordinate block lies exactly IN `null(Ā) = span{P₀, P₁, P₂}`, which is the direction the shift/Woodbury machinery of [DESIGN §4.1] steps 1 and 3 exists to handle. It is therefore the *adversarial* rhs for the specific algebra CR was created to check — a bug in the `P` / `U` / `M` null-space path shows here and may not show on a physical gradient, whose `𝟙`-component is small. It passes at 3.34× at n=960, so it is a real bar and not a formality.

**RECORDED, NOT GATED (`gated: false`, ten rows per fixture):** five gradient-shape and five projection-shape deterministic pseudo-random draws. **Gating these would turn CR RED at n=960 against a CORRECT f64 factor** — a false kill, which is the exact failure class three rounds of spec review existed to eliminate. They are emitted so the sensitivity is in the artifact rather than in someone's memory.

**This is a superset of [CAL §F], not a substitution, which is why it needs no amendment.** Whatever the spec's "the same two [K1CAL] already uses" was reaching for, vectors of that kind are inside the recorded ensemble; the *gated* subset is defined by what the solver can physically encounter. The governance question is removed rather than answered.

**The PRNG is written out in full in Task 3 Step 2 — splitmix32 driving UNIFORMS on [−1,1). Not numpy, not `Math.random`, and deliberately NOT Box–Muller.** Seeds are `1000 + s` for the gradient shape and `2000 + s` for the projection shape, `s = 0…4`.

`splitmix32` uses only `Math.imul`, XOR, shifts and a division by 2³² — every one of which ECMA-262 specifies exactly, so it is bit-identical across engines and versions and transcribes to Python in about five lines. **Box–Muller would have destroyed that property**, because it calls `Math.log` and `Math.cos`, and ECMA-262 §21.3.2 leaves `Math.log` / `Math.sin` / `Math.cos` **implementation-approximated** — not required to agree bit-for-bit between engines or between versions of one engine. The consequence is not abstract: the pre-registered ungated-failure count below would then be engine-dependent, and an executor on a different Bun build could see a different count and escalate a non-issue. That is a false-alarm generator — the mirror image of the false kill this decision removed in the first place. **Measured:** under Box–Muller, n=960 no-pin had TWO ungated failures (`rand1`, `rand3`); on the identical seeds under uniforms it has ONE, and a different row. The gated rows were unaffected in both schemes, so the gate itself was never at risk — only its pre-registration was.

A uniform draw probes exactly what is wanted here. The ensemble's job is to excite `A`'s badly-conditioned eigendirections, and a uniform vector does that as well as a normal one; nothing in D5's argument depends on the marginal distribution.

**`displaced()` (D6) uses the same generator, at seed 7777, for the same reason** — its three `Math.sin` calls had the identical portability defect, and it feeds a GATED row.

**Cost is not a reason to trim the ensemble.** All thirteen rhs share the ONE reference factorization through the already-exported `solveSaddleFrozen`, so the extra ten rows cost 1.17 s total at n=960 against a 10.1 s reference LDLᵀ — measured (`scratch/cost960b.ts`), i.e. free.

**Record `rhsShape`, `rhsSource` and `gated` in the results JSON** so a later reader knows precisely which vector produced which number, and which of them the verdict rests on.

### D6 — CR's projection rhs must be provably NON-VACUOUS

Φ is evaluated at the fixture's own `x0` and `L0` — which Task 3 computes *from the fixture's own vertices* — so **Φ(γ₀) ≈ 0 by construction**. Feeding `d = −Φ(γ₀)` would give a rhs of norm ~1e-16, `makePreparedSaddle`'s `rhsNorm === 0` early-return would fire or the relative error would be a ratio of two roundoff vectors, and the projection half of CR would pass while testing nothing. That is a vacuous gate.

**Decision:** the projection rhs is evaluated at a DISPLACED iterate, exactly as the real frozen projection is — `A` and `C` at γ₀, `Φ` at `γ^q = γ₀ + δ`, with δ a deterministic seeded perturbation drawn from the SAME exactly-specified generator the ensemble uses (D5) — not `Math.sin`, whose ECMA-262 §21.3.2 approximation licence would make this gated row's value engine-dependent:

This is the ONE spelling of the displacement in this plan — it is the code Task 3 Step 2 implements, quoted here verbatim rather than paraphrased, because two spellings of a "deterministic, no-PRNG" constant invites an implementer to pick the wrong one:

```ts
// γ^q — a deterministic displacement off the constraint manifold, so Φ(γ^q) ≠ 0.
// The frozen projection solves exactly this shape: K(γ₀) frozen, Φ evaluated at the
// moved iterate (lineSearch.ts:250-262). A perturbation of 1e-3 · (bounding scale)
// is large enough that ‖Φ‖ is far above roundoff and small enough to stay in the
// linearisation the projection assumes. A different phase per coordinate keeps the
// displacement from being a pure translation, which the barycenter block would
// absorb into one constraint row.
function displaced(vertices: Vec3[]): Vec3[] {
    // Seed 7777, 3n values, consumed coordinate-major per vertex. NOT Math.sin: ECMA-262
    // §21.3.2 leaves Math.sin implementation-approximated, and this feeds a GATED row.
    const u = uniforms(vertices.length * 3, 7777);
    return vertices.map((v, i) => [
        v[0] + 1e-3 * u[3 * i],
        v[1] + 1e-3 * u[3 * i + 1],
        v[2] + 1e-3 * u[3 * i + 2],
    ]) as Vec3[];
}
```

**And the non-vacuity is asserted, not assumed:** `expect(dNorm).toBeGreaterThan(1e-8)` in every projection-shape fixture. A future fixture change that quietly re-centres the constraints turns CR's projection half red rather than silently green.

### D7 — CR reads κ₂(K) from the committed [K1CAL] JSON; it is never transcribed

[CAL]'s binding rule is "no gate number in prose unless a committed script emitted it into a committed JSON" — the rule that exists because three blind review rounds each found a different hand-transcribed bar. A `const KAPPA_K = { 120: 1.425e3, … }` table in the test file is exactly the failure this forbids, one level down: it goes stale silently if the calibration is ever re-run.

**Decision:** `test/gpu/cr.test.ts` reads `bench/results/2026-09-04-gpu-phase2a-k1-calibration.json` at run time and takes `rows[].kappaK` keyed by `n`. Rows present: n = 64 / 120 / 240 / 480 / 960 (`kappaK` = 194.83671766497642 / 1425.0573344894356 / 13978.240714745769 / 140458.31870989568 / 1417729.0929916357, `k = 4` for all). **N=8 and N=60 have no [K1CAL] row**, so their bar falls back to the `1e-12` floor of `max(1e-12, 10·κ·u_f64)` — which is what the floor is for.

**The read is LAZY and file-relative, and both halves matter.** A moved or renamed artifact must fail the gate loudly rather than default every bar to the floor — but a module-scope `readFileSync` does not achieve that: it throws during IMPORT, aborting the file before any `test()` is registered, so the artifact-presence test never runs and the failure surfaces as an unrelated import error. Deferring the read into a memoised accessor called from inside the tests is what makes the assertion real. And the path is resolved with `new URL('../../…', import.meta.url)`, not the bare repo-relative string, because `readFileSync` resolves that against the CWD — so the bare form works only when `bun test` is invoked from the repo root. Both are the idiom the repo already uses for oracle fixtures (`test/sobolev/constraintSetFlow.test.ts:60-66`).

**The `+pin` fixtures reuse the same-N `kappaK`**, which [K1CAL] measured at `k = 4` (barycenter + totalLength) while the pinned system is `k = 7`. κ₂(K) differs slightly between them. This is a stated approximation, not a silent one: [CAL §F] licenses "κ₂ read from [K1CAL] `rows[].kappaK`" per N, and computing a true κ₂ of a 2884×2884 matrix in the test would cost more than the gate. Record `kappaSource: 'K1CAL:rows[].kappaK (k=4)'` in every row of the JSON.

---

## Task 1: The core seam — `PreparedSaddle`, `ExternalSolveError`, and absent-seam bit-identity

**Files:**
- Modify: `src/core/sobolev/linsolve.ts` (new exports after `:399`; `solveFactored` at `:696-698`)
- Modify: `src/core/sobolev/gradient.ts:84-115` (10th positional param)
- Modify: `src/core/optimizer.ts:265-266` (the `saddle.A` shape guard, beside the `dE` one), `:174` (`SobolevStepOptions`), `:282` (collector disarm), `:343-353` (forward), `:361` (rethrow)
- Modify: `src/core/dispatch.ts:164` (`saddle?`), `:270` (forward), `:1-18` (new `import type` line)
- Modify: `src/worker/solverWorker.ts:36` (defensive throw on `msg.args`, inside the `try` — D4)
- Test: `test/gpu/saddleSeam.test.ts` (create)

**Interfaces:**
- Consumes: nothing (first task).
- Produces — Tasks 2 and 3 depend on these EXACT names and shapes:
  ```ts
  export interface ExternalFactorization { kind: 'external'; solve(rhs: number[]): number[] }
  export type SaddleFactorization = LuFactorization | LdltFactorization | ExternalFactorization;
  export class ExternalSolveError extends Error {}
  export interface PreparedSaddle {
      A: Float64Array;
      factor(C: number[][]): SaddleFactorization;
  }
  ```
  `solveConstrainedGradientSetFrozen(vertices, edges, disjointPairs, alpha, beta, epsilon, dE, set, factorMode?, saddle?)` — `saddle` is the TENTH positional parameter. `SobolevStepOptions.saddle?: PreparedSaddle`. `DispatchDescentStepArgs.saddle?: PreparedSaddle`.

- [ ] **Step 1: Write the failing absent-seam determinism + ledger tests**

The three control-flow edits this task adds (the `saddle.A` shape guard at `optimizer.ts:265-266`, the rethrow at `:361`, and the collector disarm at `:282`) all sit on the DEFAULT path, so "harmless when `saddle` is absent" must be **tested, not argued**. **The test below is one half of that and is deliberately NOT named for the other half.** Both of its arms run the post-seam code, so it establishes determinism — a genuine precondition — and cannot detect a regression that is present on both sides. The pre-seam/post-seam comparison is done by the **committed golden suites** (`test/golden.test.ts`, `test/sobolev/constraintSetFlow.test.ts` against `oracle/golden/*.json`, expected values captured before this branch), which is why Steps 5 and 13 run the FULL suite against the 292 baseline rather than only the new file. See Global Constraints. Do not rename this test back to "bit-identity"; the name is what would make a reader stop running the thing that actually checks it.

Create `test/gpu/saddleSeam.test.ts` (the `args()` helper mirrors the Phase 1 seam test, `test/gpu/dESeam.test.ts:8-40`):

```ts
import { expect, test } from 'bun:test';
import {
    type DispatchDescentStepArgs,
    dispatchDescentStep,
    type SolverWorkerResponse,
} from '../../src/core/dispatch';
import { trefoil } from '../../src/core/fixtures';
import { DEFAULTS } from '../../src/core/optimizer';
import { assembleAFlat } from '../../src/core/sobolev/innerProduct';
import {
    ExternalSolveError,
    luSolve,
    type PreparedSaddle,
    type SaddleFactorization,
} from '../../src/core/sobolev/linsolve';
import { timingsEnd } from '../../src/core/sobolev/phaseTimings';
import { calculateDisjointPairs } from '../../src/core/tangentPointEnergy';
import type { Vec3 } from '../../src/core/testConfigs';

function args(n = 24): DispatchDescentStepArgs {
    const { vertices, edges } = trefoil(n);
    const x0: Vec3 = [0, 0, 0];
    for (const v of vertices) {
        x0[0] += v[0] / n;
        x0[1] += v[1] / n;
        x0[2] += v[2] / n;
    }
    const L0 = edges.reduce(
        (s, [a, b]) =>
            s +
            Math.hypot(
                vertices[a][0] - vertices[b][0],
                vertices[a][1] - vertices[b][1],
                vertices[a][2] - vertices[b][2],
            ),
        0,
    );
    return {
        descentMode: 'sobolev',
        vertices,
        edges,
        disjointPairs: calculateDisjointPairs(edges),
        mode: 'analytical',
        stepSize: 1,
        x0,
        barycenterConstraint: true,
        lengthMode: 'total',
        sobolevL0: L0,
        projectionMode: 'frozen',
        collectTimings: false,
        collectField: true,
    };
}

// What this proves, exactly: with `saddle` ABSENT the step is DETERMINISTIC and leaks
// nothing across calls, with all three of this task's default-path control-flow edits
// in place (the saddle.A shape guard at optimizer.ts:265-266, the ExternalSolveError
// rethrow at :361, the timing-collector disarm at :282).
// What it does NOT prove: identity with the PRE-seam code. Both arms are the post-seam
// build, so a regression the seam introduced is present on both sides and cancels. The
// pre/post backstop is the committed golden suites (test/golden.test.ts,
// test/sobolev/constraintSetFlow.test.ts vs oracle/golden/*.json), which is why Steps 5
// and 13 run the FULL suite against the 292 baseline. Determinism is a PRECONDITION for
// that comparison, which is why this test is still worth having.
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4
test('saddle seam ABSENT: the step is deterministic and leaks nothing between calls', () => {
    const a = args();
    // Guard the comparison's meaning: a rejected/converged step would compare two echoes.
    expect(dispatchDescentStep(a).accepted).toBe(true);
    expect(dispatchDescentStep(args())).toEqual(dispatchDescentStep(a));
});

// D1's SECOND half — the one the `finally` spelling fails. A collected step that does
// NOT throw must still return a populated ledger; an unconditional finally would have
// nulled `acc` before optimizer.ts:461 reads it.
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4 (D1)
test('saddle seam ABSENT: collectTimings still returns a ledger (the disarm is throw-path only)', () => {
    const r = dispatchDescentStep({ ...args(), collectTimings: true });
    expect(r.timings).toBeDefined();
    expect(r.timings?.step?.calls).toBe(1);
    // NOT an exact count: `saddle` fires once for the gradient solve (gradient.ts:106)
    // plus once per frozen projection Newton iteration (lineSearch.ts:260), and that
    // iteration count is fixture- and step-size-dependent. Asserting a literal here
    // would be pinning an incidental number, not the ledger's presence.
    expect(r.timings?.saddle?.calls).toBeGreaterThanOrEqual(1);
});
```

- [ ] **Step 2: Run it to confirm the imports fail**

Run: `bun test test/gpu/saddleSeam.test.ts`
Expected: FAIL — `ExternalSolveError` / `PreparedSaddle` are not exported from `linsolve.ts`.

- [ ] **Step 3: Add the three new exports to `linsolve.ts`**

Insert immediately after the `SaddleFactorization` declaration at `linsolve.ts:399`, and widen that union in place:

```ts
/**
 * A factorization supplied from OUTSIDE the core — the GPU Cholesky path.
 * `solve(rhs)` returns z for the FULL (3n+k) saddle system K z = rhs, having
 * itself certified the result ([DESIGN §4.1] step 4); it throws
 * {@link ExternalSolveError} rather than returning an uncertified z.
 * Why: nothing in `src/core/` ever compares `residual` against a threshold —
 * `linsolve.ts:800`/`:903` compute it and `optimizer.ts:355,395,449` forward it
 * as a stat — so a solve that converged to 1e-3 is indistinguishable at every
 * call site from one that converged to 1e-12.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1, §4.4
 */
export interface ExternalFactorization {
    kind: 'external';
    solve(rhs: number[]): number[];
}

/**
 * Thrown by an external solve that cannot certify its own result — iterative
 * refinement exhausted its budget without reaching 1e-10 ([DESIGN §4.1] step 4).
 * Typed (not a bare Error) because `optimizer.ts`'s singular-saddle catch must let
 * exactly this one through: converting it to 'singular_system' would hide an
 * unconverged descent direction behind a rejected step.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1, §4.4, §4.5
 */
export class ExternalSolveError extends Error {}

/**
 * A saddle solve prepared OUTSIDE the step, for the step's INPUT vertices.
 * `A` MUST be the Sobolev matrix of the same vertices the step receives (the
 * driver copies `live` once and uses that copy for dE, A and the step — the
 * Phase 1 rule at `src/gpu/driver.ts:164`). `factor` is called at most once per
 * step, synchronously, with the C the CORE evaluated; `solve` then runs for the
 * gradient rhs and for each projection rhs. Carries closures ⇒ NOT
 * structured-cloneable ⇒ it can never travel to the worker.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4
 */
export interface PreparedSaddle {
    A: Float64Array;
    factor(C: number[][]): SaddleFactorization;
}
```

Widen the union at `:399` (keep its existing TSDoc, append one line noting the third member):

```ts
export type SaddleFactorization = LuFactorization | LdltFactorization | ExternalFactorization;
```

- [ ] **Step 4: Add the `'external'` branch to `solveFactored`**

`linsolve.ts:696-698` becomes — the `'kind' in fac` test is REQUIRED (D3: `LuFactorization` has no `kind`, so `fac.kind` alone is a TS error on the union), and the `'lu'` arm stays the verbatim pre-existing call with its anchor comment at `:693-695` intact:

```ts
function solveFactored(fac: SaddleFactorization, rhs: number[]): number[] {
    // 'external' first: the GPU path's own solve() ([DESIGN §4.1] steps 2-4). The
    // `'kind' in fac` guard is not redundant — LuFactorization (:219-223) has no
    // `kind` field, so `fac.kind === …` alone is a TS error on this union.
    // @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4
    if ('kind' in fac && fac.kind === 'external') return fac.solve(rhs);
    return 'kind' in fac ? ldltSolveFactored(fac, rhs) : luSolveFactored(fac, rhs);
}
```

- [ ] **Step 5: Run the seam tests and the full suite**

Run: `bun test test/gpu/saddleSeam.test.ts && bun test`
Expected: both new tests PASS; the full suite is `294 pass / 0 fail` (292 baseline + 2). A change in the baseline here means the seam was not source-selection-only.

- [ ] **Step 6: Commit the type surface**

```bash
git add src/core/sobolev/linsolve.ts test/gpu/saddleSeam.test.ts
git commit -m "$(cat <<'MSG'
feat(core): 'external' saddle factorization kind + PreparedSaddle/ExternalSolveError (phase2a-1 T1)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JdMoLJiC1mByydnaHhuGbr
MSG
)"
```

- [ ] **Step 7: Write the failing present-seam routing test**

Append to `test/gpu/saddleSeam.test.ts`. This proves the seam actually reaches the solve, using a `PreparedSaddle` whose `solve` is the CPU path — the same "strong probe" shape as Phase 1's `dE` seam test.

```ts
// A PreparedSaddle that delegates to a plain LU of the SAME K the core would have
// built, so `saddle` present and `saddle` absent must agree to round-off. This is the
// routing probe; the real numerics live in Task 2. Note the coordinate-major block
// write `K[b*n+i][b*n+j]` — Ā = I₃ ⊗ A (Global Constraints).
function identityPrepared(a: DispatchDescentStepArgs): PreparedSaddle {
    const A = assembleAFlat(
        a.vertices,
        a.edges,
        a.disjointPairs,
        DEFAULTS.alpha,
        DEFAULTS.beta,
        DEFAULTS.epsilon,
    );
    return {
        A,
        factor(C: number[][]): SaddleFactorization {
            const n = a.vertices.length;
            const m = 3 * n;
            const k = C.length;
            const size = m + k;
            const K: number[][] = Array.from({ length: size }, () => new Array(size).fill(0));
            for (let b = 0; b < 3; b++)
                for (let i = 0; i < n; i++)
                    for (let j = 0; j < n; j++) K[b * n + i][b * n + j] = A[i * n + j];
            for (let r = 0; r < k; r++)
                for (let c = 0; c < m; c++) {
                    K[m + r][c] = C[r][c];
                    K[c][m + r] = C[r][c];
                }
            return {
                kind: 'external',
                solve: (rhs: number[]) => luSolve(K, rhs),
            };
        },
    };
}

function boomPrepared(a: DispatchDescentStepArgs): PreparedSaddle {
    return {
        A: assembleAFlat(
            a.vertices,
            a.edges,
            a.disjointPairs,
            DEFAULTS.alpha,
            DEFAULTS.beta,
            DEFAULTS.epsilon,
        ),
        factor: () => ({
            kind: 'external',
            solve: () => {
                throw new ExternalSolveError('IR did not converge');
            },
        }),
    };
}

test('saddle seam PRESENT: routes through the external solve and agrees with the CPU path', () => {
    const a = args();
    const base = dispatchDescentStep(a);
    const viaSeam = dispatchDescentStep({ ...args(), saddle: identityPrepared(a) });
    expect(viaSeam.accepted).toBe(base.accepted);
    const b = base.descentField as Vec3[];
    const t = viaSeam.descentField as Vec3[];
    let checked = 0;
    for (let i = 0; i < b.length; i++)
        for (let c = 0; c < 3; c++) {
            if (Math.abs(b[i][c]) < 1e-12) continue;
            expect(t[i][c] / b[i][c]).toBeCloseTo(1, 8);
            checked++;
        }
    // Non-vacuity: if every component were filtered out the loop would assert nothing.
    expect(checked).toBeGreaterThan(0);
});

test('saddle seam: a thrown ExternalSolveError propagates out of dispatchDescentStep', () => {
    // NOT swallowed into 'singular_system': the driver must be able to see it and
    // re-run the step on the CPU path ([DESIGN §4.5] fallback trigger 3).
    expect(() => dispatchDescentStep({ ...args(), saddle: boomPrepared(args()) })).toThrow(
        ExternalSolveError,
    );
});

// D1's FIRST half. Asserts the collector's OWN state, NOT a later step's ledger: the
// obvious spelling — run another collected step and check `timings.step.calls === 1` —
// CANNOT FAIL, because timingsBegin() is `acc = {}` (phaseTimings.ts:48-50), an
// UNCONDITIONAL reset, so the next collected step destroys any leaked ledger before
// recording a phase and `step.calls` is 1 with or without the disarm. timingsEnd()
// returns the leaked partial ledger when the disarm is missing and null when it fired,
// so this discriminates. Do not "simplify" it back — see D1.
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4 (D1)
test('saddle seam: a throw disarms the timing collector', () => {
    expect(() =>
        dispatchDescentStep({
            ...args(),
            saddle: boomPrepared(args()),
            collectTimings: true,
        }),
    ).toThrow(ExternalSolveError);
    expect(timingsEnd()).toBeNull();
});

// D1's first half again, at the hazard the previous test cannot reach: the step AFTER
// the throw has `collectTimings: false`, so it never calls timingsBegin() and a leaked
// `acc` is never reset. Without the disarm, timed() accumulates that step's phases into
// the THROWING step's ledger — for the process lifetime, paying performance.now() on
// every phase of every later step and handing a cross-step ledger to whoever collects
// next. This is the failure the vacuous spelling above would have shipped.
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4 (D1)
test('saddle seam: a leaked collector cannot accumulate into an UNCOLLECTED step', () => {
    expect(() =>
        dispatchDescentStep({
            ...args(),
            saddle: boomPrepared(args()),
            collectTimings: true,
        }),
    ).toThrow(ExternalSolveError);
    dispatchDescentStep({ ...args(), collectTimings: false });
    expect(timingsEnd()).toBeNull();
});

// The `saddle.A` contract ([DESIGN §4.4]: "the matrix of the SAME vertices the step
// receives") enforced, not merely documented. solveSaddleFrozen (linsolve.ts:882-905)
// validates rhsTop/rhsBottom lengths ONLY, while the path it replaces does validate A
// (solveSaddleFromA, :755-757: `if (a.length !== n * n) throw`) — so without this guard
// the seam path is strictly LESS checked than the CPU path. A short A reads out of
// range → undefined → NaN → structuredSaddleResidual returns NaN → nothing in src/core/
// thresholds `residual` (Global Constraints), so the line search rejects a NaN direction
// as `armijo_failed` and the run auto-pauses with usedGpuSolve still true: the same
// silent-failure class [DESIGN §4.1] step 4's throw exists to prevent.
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4
test('saddle seam: a wrong-sized saddle.A throws before anything is solved', () => {
    const a = args();
    const bad: PreparedSaddle = { ...identityPrepared(a), A: new Float64Array(4) };
    expect(() => dispatchDescentStep({ ...a, saddle: bad })).toThrow(/saddle\.A length/);
    // The guard is on the PRE-ARM path (before timingsBegin(), like the dE guard at
    // optimizer.ts:265-266), so it needs no disarm — and this proves it, since a guard
    // moved below timingsBegin() would leave the collector armed here (D1).
    expect(() => dispatchDescentStep({ ...a, saddle: bad, collectTimings: true })).toThrow(
        /saddle\.A length/,
    );
    expect(timingsEnd()).toBeNull();
});

// The worker guard's falsifier. Without it the guard is dead code that READS as a live
// safety property: `saddle` rides `msg.args`, never `msg` (D4). `topology` must be posted
// first, or the worker throws 'step received before topology' and this test would pass
// for the wrong reason — hence the message assertion, not just `type === 'error'`.
// Round-trip shape mirrors test/worker-solver.test.ts:71-92.
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4
test('solverWorker: a saddle on the step payload is rejected', async () => {
    const a = args();
    const response = await new Promise<SolverWorkerResponse>((resolve, reject) => {
        const worker = new Worker(
            new URL('../../src/worker/solverWorker.ts', import.meta.url).href,
        );
        const timer = setTimeout(() => {
            worker.terminate();
            reject(new Error('worker timed out'));
        }, 5000);
        worker.onmessage = (event: MessageEvent<SolverWorkerResponse>) => {
            clearTimeout(timer);
            worker.terminate();
            resolve(event.data);
        };
        worker.onerror = (event: ErrorEvent) => {
            clearTimeout(timer);
            worker.terminate();
            reject(new Error(`worker error: ${event.message}`));
        };
        worker.postMessage({ type: 'topology', graphVersion: 0, edges: a.edges });
        // A PLAIN {}, deliberately: a real PreparedSaddle carries closures, so
        // postMessage would fail with a DataCloneError before the guard ever ran. The
        // cast is what makes the guard reachable from a test at all. `edges` /
        // `disjointPairs` ride along harmlessly — solverWorker.ts:75-79 overwrites both
        // from the topology cache.
        worker.postMessage({
            type: 'step',
            graphVersion: 0,
            args: { ...a, saddle: {} as unknown as PreparedSaddle },
        });
    });
    if (response.type !== 'error') throw new Error(`expected an error, got ${response.type}`);
    expect(response.message).toMatch(/saddle/);
}, 10_000);
```

- [ ] **Step 8: Run it to verify it fails**

Run: `bun test test/gpu/saddleSeam.test.ts`
Expected: FAIL — `saddle` is not a known property of `DispatchDescentStepArgs`; nothing forwards it.

- [ ] **Step 9: Thread `saddle` through `gradient.ts`**

`solveConstrainedGradientSetFrozen` (`gradient.ts:84-115`) gains the TENTH positional parameter and the branch. `factorMode` already rides in this way; the function has no options object (nine positional params at `:84-94`).

```ts
export function solveConstrainedGradientSetFrozen(
    vertices: Vec3[],
    edges: Edge[],
    disjointPairs: number[][],
    alpha: number,
    beta: number,
    epsilon: number,
    dE: Vec3[],
    set: ConstraintSet,
    factorMode?: FactorMode,
    saddle?: PreparedSaddle,
): { gTilde: Vec3[]; lambda: number[]; residual: number; frozen: FrozenSaddleOperator } {
    // Only the Jacobian C enters the gradient solve. Φ itself does NOT: the
    // saddle RHS bottom block is 0 (solveSaddleFromA's default), unlike the
    // constraint-projection solve which passes −Φ there.
    // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("Gradient saddle system" — RHS [dE; 0])
    const { C } = evaluateConstraintSet(set, vertices, edges);
    if (saddle) {
        // Source selection, never arithmetic (the Phase 1 `dE?` rule): A comes from
        // the caller instead of assembleAFlat, the factorization comes from the GPU
        // Cholesky, and the solve routes through the ALREADY-EXPORTED
        // solveSaddleFrozen (:882-905) so the residual stays the byte-identical
        // structured matvec the goldens gate — `solveFactored` and
        // `structuredSaddleResidual` are module-private and must stay that way.
        // C is evaluated by the CORE above, never by the driver.
        // @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4
        const frozen: FrozenSaddleOperator = {
            a: saddle.A,
            n: vertices.length,
            C,
            fac: saddle.factor(C),
        };
        const { x, lambda, residual } = timed('saddle', () =>
            solveSaddleFrozen(frozen, flatten(dE)),
        );
        return { gTilde: unflatten(x), lambda, residual, frozen };
    }
    // Typed-array fast path (solver-perf Task 5): flat scalar A straight into
    // solveSaddleFromA, which writes Ā's diagonal blocks itself — the 'expand'
    // phase (expandBlockDiag) intentionally no longer fires here. 'saddle'
    // wraps the whole solve, same key as before; 'factor' fires inside it.
    // @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Tasks 1, 5)
    const A = assembleAFlat(vertices, edges, disjointPairs, alpha, beta, epsilon);
    const { x, lambda, residual, fac } = timed('saddle', () =>
        solveSaddleFromA(A, vertices.length, C, flatten(dE), undefined, factorMode),
    );
    return {
        gTilde: unflatten(x),
        lambda,
        residual,
        frozen: { a: A, n: vertices.length, C, fac },
    };
}
```

**The one structural change on the ABSENT path:** `evaluateConstraintSet` moves above the `assembleAFlat` call. Both are pure functions of `(vertices, edges, set)` with no shared state, so the reordering is observationally inert — Step 1's bit-identity test and the full suite are the proof. **Both existing anchor comments must be carried across verbatim** (the `[dE; 0]` rhs note now sits above `evaluateConstraintSet`; the solver-perf fast-path note stays on the `assembleAFlat` line).

Extend the import at `gradient.ts:17`:

```ts
import {
    type FactorMode,
    type FrozenSaddleOperator,
    type PreparedSaddle,
    solveSaddleFromA,
    solveSaddleFrozen,
} from './linsolve';
```

- [ ] **Step 10: Thread `saddle` through `optimizer.ts`**

Add to `SobolevStepOptions` (after `factorMode?` at `:174`):

```ts
    /**
     * A saddle solve prepared OUTSIDE this step for THESE vertices (WebGPU Phase 2a
     * seam). Selects the SOURCE of A and of the factorization, never the arithmetic;
     * absent ⇒ every path below is bit-identical (the golden suites are the backstop),
     * exactly like `dE?`. Its `solve` self-certifies or throws ExternalSolveError,
     * which this function deliberately does NOT convert to 'singular_system'.
     * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4
     */
    saddle?: PreparedSaddle;
```

Add the `saddle.A` shape guard beside the existing `dE` one at `:265-266` — as a FIRST statement, before `timingsBegin()` at `:281`, for the same reason Phase 1 put the `dE` guard there:

```ts
    // FIRST statements, before timingsBegin(): a throw must not leave the timing
    // collector armed (this is the PRE-ARM path, so it needs no disarm — D1).
    if (opts.dE && opts.dE.length !== vertices.length)
        throw new Error('sobolevStepSet: dE length mismatch');
    // The saddle mirror of the line above. `saddle.A` goes straight into
    // FrozenSaddleOperator.a, and solveSaddleFrozen (linsolve.ts:882-905) validates
    // rhsTop/rhsBottom lengths ONLY — while the path it replaces DOES validate A
    // (solveSaddleFromA, :755-757). Without this the seam path is strictly less checked
    // than the CPU path, and a short A reads out of range → undefined → NaN, which
    // nothing downstream thresholds (Global Constraints): the run would auto-pause on
    // `armijo_failed` with usedGpuSolve still true.
    // @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4
    if (opts.saddle && opts.saddle.A.length !== vertices.length ** 2)
        throw new Error('sobolevStepSet: saddle.A length mismatch');
```

Forward it at the call site (`:343-353`) as the tenth argument:

```ts
                const solved = solveConstrainedGradientSetFrozen(
                    vertices,
                    edges,
                    disjointPairs,
                    alpha,
                    beta,
                    epsilon,
                    dE,
                    set,
                    opts.factorMode,
                    opts.saddle,
                );
```

Make the bare catch at `:361` rethrow:

```ts
            } catch (e) {
                // The ONE throw this catch must not swallow: an external solve that
                // could not certify its result ([DESIGN §4.1] step 4). Converting it to
                // 'singular_system' would present an unconverged descent direction as
                // a rejected step, with usedGpuSolve still true — the driver has to see
                // it to re-run this step on the CPU path ([DESIGN §4.5] trigger 3).
                // Unreachable when `saddle` is absent; test/gpu/saddleSeam.test.ts
                // proves the absent path is still bit-identical WITH this branch here.
                // @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4
                if (e instanceof ExternalSolveError) throw e;
                // Exactly singular saddle system (e.g. an isolated vertex → zero Ā rows).
                // …existing body unchanged, including its anchor comment…
```

Apply **D1**'s collector disarm around the `timed('step', …)` call at `:282` — hoist the inline return-type annotation to `type StepBodyOutcome`, keep the body verbatim, wrap in `try` / `catch (e) { if (collect) timingsEnd(); throw e; }`. Leave `:460-461` exactly as it is.

Import `ExternalSolveError` and `type PreparedSaddle` from `./sobolev/linsolve`.

- [ ] **Step 11: Thread `saddle` through `dispatch.ts` and guard the worker**

`dispatch.ts` — add beside `dE?` at `:164`:

```ts
    /**
     * A saddle solve prepared for `vertices` (WebGPU Phase 2a seam). Forwarded to
     * `sobolevStepSet` exactly like `dE`. Carries closures ⇒ never cloneable ⇒ it
     * must not ride the worker payload.
     * @see SobolevStepOptions.saddle (optimizer.ts)
     */
    saddle?: PreparedSaddle;
```

and forward it beside `dE: args.dE` at `:270`:

```ts
            saddle: args.saddle,
```

`dispatch.ts` imports nothing from `./sobolev/linsolve` today (`:1-18`), so this also needs a NEW import line — Step 10 spells the equivalent out for `optimizer.ts`; match it:

```ts
import type { PreparedSaddle } from './sobolev/linsolve';
```

`src/worker/solverWorker.ts` — **first statement inside the existing `try`** (`:36`, per D4):

```ts
        // Defensive: PreparedSaddle carries closures and cannot be structured-cloned,
        // so it can only appear here if a future caller routed a 'gpu' step to the
        // worker. The 'gpu' driver is main-thread only ([P §2.1]); fail loudly rather
        // than solve with a silently missing factorization. Inside the try so the
        // existing catch (:89-99) reports it as { type: 'error' } to the caller that
        // erred. It must read `msg.args`, NOT `msg`: SolverWorkerRequest's 'step' arm is
        // { type, graphVersion, args } (dispatch.ts:397-400) and DispatchStepArgs =
        // Omit<DispatchDescentStepArgs, 'edges' | 'disjointPairs'> (:371), so `saddle`
        // can only ever appear at msg.args.saddle — `'saddle' in msg` is false for every
        // message the protocol can carry, i.e. a guard that can never fire while
        // :75-79 spreads `saddle` straight through. No cast is needed: the type test
        // has already narrowed msg.args to DispatchStepArgs.
        // @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4
        if (msg.type === 'step' && 'saddle' in msg.args)
            throw new Error('solverWorker: `saddle` cannot cross the worker boundary');
```

- [ ] **Step 12: Run the tests**

Run: `bun test test/gpu/saddleSeam.test.ts`
Expected: all EIGHT tests PASS — 2 from Step 1 (determinism, collected ledger) + 6 from Step 7 (routing, `ExternalSolveError` propagation, collector disarmed, no accumulation into an uncollected step, `saddle.A` shape guard, worker guard).

- [ ] **Step 13: Run the full suite, the typechecker and the linter**

Run: `bunx tsc --noEmit && bun test && bunx biome check --write src/core src/worker test/gpu; bunx knip`
Expected: `tsc` exit 0; `bun test` = `300 pass / 0 fail` (292 baseline + 8); biome introduces no new findings in the touched files. **`knip` exits 1 — that is the pre-existing baseline, not a failure**, which is why it is chained with `;` and not `&&` (an `&&` chain would report 1 for the whole command whatever the real state is, and an executor could not tell PASS from FAIL). Diff its output against the NAMED baseline sets in Global Constraints; the expected new findings there are **none**. Any new name is a regression.

- [ ] **Step 14: Commit**

```bash
git add src/core src/worker test/gpu/saddleSeam.test.ts
git commit -m "$(cat <<'MSG'
feat(core): saddle? seam through dispatch/optimizer/gradient, ExternalSolveError rethrow (phase2a-1 T1)

Absent ⇒ bit-identical, tested rather than argued: the rethrow at optimizer.ts:361
and the throw-path timing-collector disarm are both on the default path.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JdMoLJiC1mByydnaHhuGbr
MSG
)"
```

---

## Task 2: `src/gpu/saddle.ts` — the CPU f64 Schur/Woodbury/IR algebra

**Files:**
- Create: `src/gpu/saddle.ts`
- Test: `test/gpu/saddle.test.ts`

**Interfaces:**
- Consumes: `ExternalSolveError`, `PreparedSaddle`, `SaddleFactorization` (Task 1); `luSolve` (`linsolve.ts:33`, already exported).
- Produces — Task 3 depends on these EXACT names and shapes:
  ```ts
  export const SADDLE_IR_TOL = 1e-10;
  export const SADDLE_IR_MAX_REFINEMENTS = 4;
  export function shiftSobolev(A: Float64Array, n: number, out?: Float64Array): { Asigma: Float64Array; sigma: number };
  export function choleskyF64(Asigma: Float64Array, n: number): Float64Array;
  export interface IrTrace { refinements: number; relResiduals: number[] }
  export function makePreparedSaddle(
      A: Float64Array, L: Float64Array, sigma: number, n: number,
      onTrace?: (t: IrTrace) => void,
  ): PreparedSaddle;
  export function asExternal(fac: SaddleFactorization): ExternalFactorization;
  ```

**`asExternal` is not optional plumbing — without it none of this task's tests compile.** `PreparedSaddle.factor` returns the WIDENED `SaddleFactorization` union, and [DESIGN §4.4] pins that signature verbatim, so it cannot be narrowed to `ExternalFactorization` at the source. `solve` exists on only one member of the union — `LuFactorization` (`linsolve.ts:219-223`) has neither `solve` nor `kind` — so `prepared.factor(C).solve(rhs)` is `TS2339: Property 'solve' does not exist on type 'SaddleFactorization'`, which stops Step 9's `bunx tsc --noEmit &&` chain before `bun test`. Every consumer that needs `solve` (Task 2 Step 5's three tests, Task 3 Step 3's `measure`) goes through `asExternal`. It uses the SAME `'kind' in fac && fac.kind === 'external'` spelling D3 pins for `solveFactored`, so there is one narrowing idiom in the branch, not two.

This file has **no `three` import** — it is plain TS and runs under `bun test`, which is what lets gate CR run before a line of WGSL exists ([DESIGN §5]: "2a, CPU-only (`bun test`), BEFORE any WGSL").

- [ ] **Step 1: Write the failing Cholesky test**

Create `test/gpu/saddle.test.ts`:

```ts
import { expect, test } from 'bun:test';
import { trefoil } from '../../src/core/fixtures';
import { DEFAULTS } from '../../src/core/optimizer';
import { assembleAFlat } from '../../src/core/sobolev/innerProduct';
import { calculateDisjointPairs } from '../../src/core/tangentPointEnergy';
import { choleskyF64, shiftSobolev } from '../../src/gpu/saddle';

function sobolevA(n: number): { A: Float64Array; n: number } {
    const { vertices, edges } = trefoil(n);
    const dp = calculateDisjointPairs(edges);
    return {
        A: assembleAFlat(vertices, edges, dp, DEFAULTS.alpha, DEFAULTS.beta, DEFAULTS.epsilon),
        n,
    };
}

test('choleskyF64 reconstructs A_σ to f64 round-off', () => {
    const { A, n } = sobolevA(60);
    const { Asigma } = shiftSobolev(A, n);
    const L = choleskyF64(Asigma, n);
    let maxAbs = 0;
    let maxErr = 0;
    for (let i = 0; i < n; i++)
        for (let j = 0; j <= i; j++) {
            let s = 0;
            for (let k = 0; k <= j; k++) s += L[i * n + k] * L[j * n + k];
            maxErr = Math.max(maxErr, Math.abs(s - Asigma[i * n + j]));
            maxAbs = Math.max(maxAbs, Math.abs(Asigma[i * n + j]));
        }
    expect(maxErr / maxAbs).toBeLessThan(1e-12);
});

test('the shift is positive exactly on null(A): A_σ(𝟙/√n) = σ(𝟙/√n)', () => {
    // [DESIGN §4.1] step 1 rests on this. A 𝟙 ≈ 0 (null(A) = span(𝟙), one-dimensional
    // even for the disconnected presets, [DESIGN §2]), so A_σ 𝟙 = (σ/n)·𝟙𝟙ᵀ𝟙 = σ𝟙.
    const { A, n } = sobolevA(60);
    const { Asigma, sigma } = shiftSobolev(A, n);
    const one = new Float64Array(n).fill(1 / Math.sqrt(n));
    let maxDev = 0;
    for (let i = 0; i < n; i++) {
        let s = 0;
        for (let j = 0; j < n; j++) s += Asigma[i * n + j] * one[j];
        maxDev = Math.max(maxDev, Math.abs(s - sigma * one[i]));
    }
    expect(maxDev / sigma).toBeLessThan(1e-10);
});

// The pivot guard is tested on SYNTHETIC 2×2 matrices with exactly-representable
// entries — NEVER on the real unshifted A.
//
// Why the obvious test is impossible: A is PSD with a ONE-dimensional null space
// ([DESIGN §2]), so its terminal Schur complement is 0 ± roundoff and the SIGN of that
// roundoff is an accident, not a property. Measured on this branch: n = 8 / 24 / 60 / 61
// / 62 complete with NO throw — at n=60 choleskyF64 returns having written
// L[59·60+59] = sqrt(2.49e-14) = 1.58e-7, i.e. it silently produces exactly the garbage
// factor the guard exists to reject; n=120 throws d = −3.062e-12 at column 119 and n=240
// throws d = −6.537e-12 at column 239. An assertion pinned to any of those is pinned to a
// rounding accident: a change to assembleAFlat, to trefoil, or to the toolchain flips it,
// and the test then either goes red against a CORRECT guard (whose two natural "repairs"
// both corrupt production code — widening `!(d > 0)` to a tolerance rejects legitimately
// tiny-but-positive pivots on A_σ, and changing the shift breaks [DESIGN §4.1] step 1) or
// silently stops testing anything. The four cases below are exact in f64 and depend on
// nothing.
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.3 (the GPU
//      `finalize` kernel applies the same `!(d > 0)` spelling)
test('choleskyF64 throws on a negative pivot', () => {
    // [[1,2],[2,1]] — eigenvalues 3 and −1. Column 0 passes (d = 1); column 1 has
    // d = 1 − 2² = −3 exactly, no rounding involved. Pinning the value and the column
    // means a guard that fired for the wrong reason would not pass this.
    expect(() => choleskyF64(new Float64Array([1, 2, 2, 1]), 2)).toThrow(
        /non-positive pivot -3 at column 1/,
    );
});

test('choleskyF64 throws on an exactly-zero pivot', () => {
    // [[0,0],[0,1]] — the PSD boundary, reached exactly rather than by roundoff. This is
    // the case the real A only ever approaches, which is why it is synthesised here.
    // Without the guard, d = 0 divides and writes Infinity/NaN into L.
    expect(() => choleskyF64(new Float64Array([0, 0, 0, 1]), 2)).toThrow(
        /non-positive pivot 0 at column 0/,
    );
});

test('choleskyF64 throws on a NaN pivot — the case `d <= 0` would let through', () => {
    // THE reason the guard is spelt `!(d > 0)` and not `d <= 0`: `NaN <= 0` is FALSE, so
    // the `d <= 0` spelling admits NaN, takes sqrt(NaN) = NaN, and silently poisons every
    // later backsolve; `!(NaN > 0)` is TRUE. This test is the only thing in the suite that
    // can catch that "simplification" — no fixture produces a NaN pivot by accident, and
    // the n=120 fixture test an earlier draft used could not have detected it either.
    // Do NOT merge it into the negative-pivot case and do NOT delete it.
    // (The guard's second half, `!Number.isFinite(d)`, covers +Infinity, which `!(d > 0)`
    // alone would pass.)
    // @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.3
    expect(() => choleskyF64(new Float64Array([Number.NaN, 0, 0, 1]), 2)).toThrow(
        /non-positive pivot NaN at column 0/,
    );
});

test('choleskyF64 does NOT throw on an SPD matrix (the positive control)', () => {
    // Without this, the three tests above all pass against a guard that throws
    // unconditionally — i.e. against a completely broken factorization.
    // [[4,1],[1,3]]: L = [[2,0],[0.5,sqrt(2.75)]].
    const L = choleskyF64(new Float64Array([4, 1, 1, 3]), 2);
    expect(L[0]).toBeCloseTo(2, 12);
    expect(L[2]).toBeCloseTo(0.5, 12);
    expect(L[3]).toBeCloseTo(Math.sqrt(2.75), 12);
});
```

All four are 2×2 and cost nothing — the guard is now tested independently of any fixture, so there is no "written recourse" branch to follow and no `n` to keep in sync with `assembleAFlat`.

- [ ] **Step 2: Run it to verify it fails**

Run: `bun test test/gpu/saddle.test.ts`
Expected: FAIL — `src/gpu/saddle.ts` does not exist.

- [ ] **Step 3: Write the shift and the factorization**

Create `src/gpu/saddle.ts`:

```ts
// src/gpu/saddle.ts — the CPU f64 half of the Phase 2a solve: shift, Schur, Woodbury,
// iterative refinement, on a factor produced elsewhere (the GPU Cholesky in slice 2a-2,
// or the f64 reference used by gate CR). NO `three` import — this file must run under
// `bun test` so CR can gate the algebra BEFORE any WGSL exists ([DESIGN §5]).
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1, §4.2, §5 (CR)
import {
    type ExternalFactorization,
    ExternalSolveError,
    luSolve,
    type PreparedSaddle,
    type SaddleFactorization,
} from '../core/sobolev/linsolve';

/**
 * Narrow a {@link SaddleFactorization} to its `'external'` member, throwing if it is
 * anything else.
 * Why this exists at all: `PreparedSaddle.factor` returns the WIDENED union and
 * [DESIGN §4.4] pins that signature verbatim, so it cannot be narrowed at the source —
 * but `solve` is a member of only ONE of the three arms, so `factor(C).solve(rhs)` is a
 * `TS2339` and every `bunx tsc --noEmit` in this plan would fail on it. The test suites
 * and gate CR all need `solve`, so the narrowing lives here once rather than being
 * re-spelt in three files. Uses the SAME `'kind' in fac && fac.kind === 'external'` test
 * D3 pins for `solveFactored`: `LuFactorization` (`linsolve.ts:219-223`) has no `kind`
 * field, so `fac.kind === …` alone is a TS error on this union.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4
 */
export function asExternal(fac: SaddleFactorization): ExternalFactorization {
    if (!('kind' in fac && fac.kind === 'external'))
        throw new Error('asExternal: expected an external factorization');
    return fac;
}

/**
 * IR stops at this PURE relative residual `‖r_i‖₂/‖r‖₂`. Deliberately NOT the core's
 * own `residual` stat: `structuredSaddleResidual` (`linsolve.ts:848`) normalises by
 * `max(1, ‖r‖₂)` and is therefore ABSOLUTE for any rhs with `‖r‖₂ ≤ 1` — which the
 * projection rhs `[0; −Φ]` always is — so reusing it would accept a solve ~1e4× worse
 * than this criterion claims.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1 step 4, §4.2
 */
export const SADDLE_IR_TOL = 1e-10;

/**
 * Refinement budget: z₀ plus this many corrections.
 * Why 4, and why exceeding it throws rather than warns: [CAL §E.5] measures that a
 * factor sitting exactly AT K1's green bar needs all four refinements with ZERO spare,
 * and that the model's exhaustion level sits above the green bar — so K1 is NOT what
 * catches a factor between the two. The [DESIGN §4.1] step-4 throw is, per step, at run
 * time. ([CAL §E.6]'s `solvesTo1e10` of 4 at n=960 is z₀ + THREE refinements: the
 * `residualCurve[0] = 1.0` entry is the z=0 artefact the consumer compensates for.)
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-gate-calibration.md §E.5, §E.6
 */
export const SADDLE_IR_MAX_REFINEMENTS = 4;

/**
 * σ = trace(A)/n and A_σ = A + (σ/n)·𝟙𝟙ᵀ (row-major, n×n).
 * Why this exact shift: null(A) = span(𝟙) and is ONE-dimensional at every N measured,
 * including the disconnected `crossing` and `linked-rings` presets ([DESIGN §2]), so a
 * rank-1 shift is enough to make A_σ SPD — and it is positive exactly on the null
 * direction, `A_σ(𝟙/√n) = σ(𝟙/√n)`. It is removed EXACTLY by the Woodbury step, so it
 * does not perturb the metric (do not confuse with `buildSaddleMatrix`'s "no
 * regularizing identity" rule, `linsolve.ts:133-154`, which protects the METRIC).
 * `out` lets a caller reuse one buffer across steps — a per-step N×N allocation is a
 * measured throughput regression, not a style question ([CAL §H.1]: the "A_σ build +
 * f32 cast" constant spreads 10.7× at N=480 between the churn and reuse regimes).
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1 step 1
 */
export function shiftSobolev(
    A: Float64Array,
    n: number,
    out?: Float64Array,
): { Asigma: Float64Array; sigma: number } {
    let tr = 0;
    for (let i = 0; i < n; i++) tr += A[i * n + i];
    const sigma = tr / n;
    const s = sigma / n;
    const Asigma = out ?? new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) Asigma[i] = A[i] + s;
    return { Asigma, sigma };
}

/**
 * Lower-triangular Cholesky factor L of a row-major SPD n×n matrix, L Lᵀ = A_σ.
 * The f64 REFERENCE — gate CR runs the whole [DESIGN §4.1] reformulation on this factor
 * so a CR red localises to the algebra, never to a kernel; slice 2a-2's f32 GPU result
 * is promoted into the same shape. Left-looking (the GPU kernel is right-looking,
 * [DESIGN §4.3]): the op orders differ deliberately and CR's bar is κ-relative, so they
 * must NOT be "unified".
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1, §4.3, §5 (CR)
 */
export function choleskyF64(Asigma: Float64Array, n: number): Float64Array {
    const L = new Float64Array(n * n);
    for (let j = 0; j < n; j++) {
        let d = Asigma[j * n + j];
        for (let k = 0; k < j; k++) d -= L[j * n + k] * L[j * n + k];
        // Same guard the GPU `finalize` kernel applies ([DESIGN §4.3]): `!(d > 0)` also
        // rejects NaN, which `d <= 0` would let through.
        if (!(d > 0) || !Number.isFinite(d))
            throw new Error(`choleskyF64: non-positive pivot ${d} at column ${j}`);
        const r = Math.sqrt(d);
        L[j * n + j] = r;
        for (let i = j + 1; i < n; i++) {
            let s = Asigma[i * n + j];
            for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
            L[i * n + j] = s / r;
        }
    }
    return L;
}
```

- [ ] **Step 4: Run the tests**

Run: `bun test test/gpu/saddle.test.ts`
Expected: all SIX PASS — the two shift/reconstruction tests at n=60 plus the four synthetic pivot-guard cases (negative, exactly-zero, NaN, and the SPD positive control).

- [ ] **Step 5: Write the failing solve test**

Append to `test/gpu/saddle.test.ts` (add the new imports at the top of the file):

```ts
import {
    barycenterBlock,
    evaluateConstraintSet,
    totalLengthBlock,
} from '../../src/core/sobolev/constraintSet';
import { flatten } from '../../src/core/sobolev/layout';
import { ExternalSolveError, solveSaddleFromA } from '../../src/core/sobolev/linsolve';
import { gradientAnalytical } from '../../src/core/tangentPointEnergy';
import {
    asExternal,
    type IrTrace,
    makePreparedSaddle,
    SADDLE_IR_MAX_REFINEMENTS,
    SADDLE_IR_TOL,
} from '../../src/gpu/saddle';
import type { Vec3 } from '../../src/core/testConfigs';

function system(n: number) {
    const { vertices, edges } = trefoil(n);
    const dp = calculateDisjointPairs(edges);
    const { alpha, beta, epsilon } = DEFAULTS;
    const A = assembleAFlat(vertices, edges, dp, alpha, beta, epsilon);
    const x0: Vec3 = [0, 0, 0];
    for (const v of vertices) {
        x0[0] += v[0] / n;
        x0[1] += v[1] / n;
        x0[2] += v[2] / n;
    }
    const L0 = edges.reduce(
        (s, [a, b]) =>
            s +
            Math.hypot(
                vertices[a][0] - vertices[b][0],
                vertices[a][1] - vertices[b][1],
                vertices[a][2] - vertices[b][2],
            ),
        0,
    );
    const set = [barycenterBlock(x0), totalLengthBlock(L0)];
    const { C } = evaluateConstraintSet(set, vertices, edges);
    const dE = gradientAnalytical(vertices, edges, dp, alpha, beta, epsilon);
    return { vertices, edges, A, C, rhsTop: flatten(dE), n };
}

test('makePreparedSaddle with an EXACT f64 factor reproduces solveSaddleFromA', () => {
    const { A, C, rhsTop, n } = system(60);
    const { Asigma, sigma } = shiftSobolev(A, n);
    const L = choleskyF64(Asigma, n);
    const prepared = makePreparedSaddle(A, L, sigma, n);
    // asExternal, not `.factor(C).solve(...)`: factor() returns the WIDENED
    // SaddleFactorization union ([DESIGN §4.4] pins that signature) and `solve` exists
    // on only one of its three members, so the direct spelling is a TS2339 that stops
    // Step 9's `bunx tsc --noEmit &&` chain before `bun test`.
    const z = asExternal(prepared.factor(C)).solve([...rhsTop, ...new Array(C.length).fill(0)]);
    const ref = solveSaddleFromA(A, n, C, rhsTop);
    const zref = [...ref.x, ...ref.lambda];
    let num = 0;
    let den = 0;
    for (let i = 0; i < z.length; i++) {
        num += (z[i] - zref[i]) ** 2;
        den += zref[i] ** 2;
    }
    // A smoke bar, not the gate: CR (Task 3) is the κ-relative pre-registered one.
    expect(Math.sqrt(num / den)).toBeLessThan(1e-10);
});

test('makePreparedSaddle throws ExternalSolveError when IR cannot converge', () => {
    const { A, C, rhsTop, n } = system(60);
    const { Asigma, sigma } = shiftSobolev(A, n);
    const L = choleskyF64(Asigma, n);
    // A factor corrupted far past any backward error a real kernel produces: IR's
    // contraction ‖I − M⁻¹K‖ exceeds 1, so it cannot converge at any budget.
    for (let i = 0; i < n; i++) L[i * n + i] *= 1 + 0.3 * ((i % 7) - 3);
    const prepared = makePreparedSaddle(A, L, sigma, n);
    const fac = asExternal(prepared.factor(C));
    let err: unknown;
    try {
        fac.solve([...rhsTop, ...new Array(C.length).fill(0)]);
    } catch (e) {
        err = e;
    }
    expect(err).toBeInstanceOf(ExternalSolveError);
    // Pin the budget the throw reports, from the constant rather than a literal: a
    // silent off-by-one in the loop's counting understates every bar derived from it
    // ([CAL §E.5]/[CAL §E.6]), and a hard-coded `4` here would not catch that.
    expect((err as Error).message).toContain(`${SADDLE_IR_MAX_REFINEMENTS} refinements`);
});

test('the IR trace records the convergence curve', () => {
    const { A, C, rhsTop, n } = system(60);
    const { Asigma, sigma } = shiftSobolev(A, n);
    const L = choleskyF64(Asigma, n);
    // An ARRAY, NOT `let trace: IrTrace | null = null`. TypeScript does not account for
    // assignments made inside a callback when narrowing a `let` in the enclosing scope
    // (microsoft/TypeScript#9998), so `trace` stays narrowed to `null` and every
    // `trace?.x` becomes a property access on `never` — two TS2339s under this repo's
    // tsconfig, which stops Step 9's `bunx tsc --noEmit &&` chain before `bun test`.
    // `?.` does not rescue it; `?.` is what produces the `never`. Array mutation is not
    // narrowed away, and `traces` being `const` means no CFA reset applies.
    const traces: IrTrace[] = [];
    const prepared = makePreparedSaddle(A, L, sigma, n, (t) => {
        traces.push(t);
    });
    asExternal(prepared.factor(C)).solve([...rhsTop, ...new Array(C.length).fill(0)]);
    expect(traces).toHaveLength(1);
    // The EXACT count, not `<= SADDLE_IR_MAX_REFINEMENTS`. `solve()` returns only when
    // `rel <= SADDLE_IR_TOL` and otherwise throws, so `refinements <= 4` and
    // `relResiduals.at(-1) <= 1e-10` are both true BY CONSTRUCTION whenever the call
    // returned — they assert nothing. With an exact f64 factor at n=60, z₀ already
    // clears the tolerance, so IR applies ZERO corrections and the curve has one entry;
    // that is the falsifiable claim. (At n=960 the same measurement is 1 — the number
    // this test would notice changing.)
    expect(traces[0].refinements).toBe(0);
    expect(traces[0].relResiduals).toHaveLength(1);
    expect(traces[0].relResiduals[0]).toBeLessThanOrEqual(SADDLE_IR_TOL);
});
```

- [ ] **Step 6: Run it to verify it fails**

Run: `bun test test/gpu/saddle.test.ts`
Expected: FAIL — `makePreparedSaddle` is not exported.

- [ ] **Step 7: Implement the Schur/Woodbury/IR core**

Append to `src/gpu/saddle.ts`. The layout is **coordinate-major** throughout: `Ā = I₃ ⊗ A_σ`, block `b` of a 3n vector is coordinate `b` over ALL vertices, occupying rows `[b·n, (b+1)·n)` — this is what `solveSaddleFromA` writes (`linsolve.ts:770-779`, `off = b * n`) and what the oracle uses (`oracle/tpe_stage1_oracle.py:245-248`). Writing `A ⊗ I₃` (vertex-major) here would be wrong and every test above would fail.

```ts
/** Forward/back substitution against L Lᵀ, one n-vector, in place into `out`. */
function solveWithL(L: Float64Array, n: number, rhs: ArrayLike<number>, out: Float64Array): void {
    for (let i = 0; i < n; i++) {
        let s = rhs[i];
        for (let k = 0; k < i; k++) s -= L[i * n + k] * out[k];
        out[i] = s / L[i * n + i];
    }
    for (let i = n - 1; i >= 0; i--) {
        let s = out[i];
        for (let k = i + 1; k < n; k++) s -= L[k * n + i] * out[k];
        out[i] = s / L[i * n + i];
    }
}

/** Â⁻¹ applied to a 3n vector — three INDEPENDENT n-backsolves, one per coordinate block. */
function applyAhatInv(L: Float64Array, n: number, v: ArrayLike<number>): Float64Array {
    const out = new Float64Array(3 * n);
    const buf = new Float64Array(n);
    const rhs = new Float64Array(n);
    for (let b = 0; b < 3; b++) {
        for (let i = 0; i < n; i++) rhs[i] = v[b * n + i];
        solveWithL(L, n, rhs, buf);
        for (let i = 0; i < n; i++) out[b * n + i] = buf[i];
    }
    return out;
}

/** IR diagnostics for one solve — the convergence curve CR and (in 2a-2) T4 record. */
export interface IrTrace {
    refinements: number;
    relResiduals: number[];
}

/**
 * The Phase 2a external solve: [DESIGN §4.1] steps 2–4 on a factor produced elsewhere.
 * `A` is the UNSHIFTED f64 Sobolev matrix (step 4's residuals are measured against the
 * f64 K, which is what makes T4 hold as [P §3] wrote it); `L` is the lower-triangular
 * Cholesky factor of A_σ (f64-promoted from f32 on the GPU path); `sigma` is the shift
 * the factor was built with.
 *
 * Per-step cost (k = C.length): 3k+1 setup n-backsolves, then 3 per solve plus O(nk)
 * ([DESIGN §4.1] step 4). With P pins k = 4+3P and the setup is 13+9P ([DESIGN §7]).
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1, §4.2
 */
export function makePreparedSaddle(
    A: Float64Array,
    L: Float64Array,
    sigma: number,
    n: number,
    onTrace?: (t: IrTrace) => void,
): PreparedSaddle {
    return {
        A,
        factor(C: number[][]): SaddleFactorization {
            const m = 3 * n;
            const k = C.length;
            const invSqrtN = 1 / Math.sqrt(n);

            // Setup, ONCE per step (the factor is frozen, [P §2.5]).
            // Z = Â⁻¹Cᵀ — 3k n-backsolves.
            const Z: Float64Array[] = C.map((row) => applyAhatInv(L, n, row));
            // S = C Z, k×k, SPD. κ(S) = 5.06…5.28 across N=120…960 ([DESIGN §4.1]), so a
            // plain LU is ample; luSolve is the core's own, already gated.
            const S: number[][] = Array.from({ length: k }, (_, r) =>
                Array.from({ length: k }, (_, c) => {
                    let s = 0;
                    for (let i = 0; i < m; i++) s += C[r][i] * Z[c][i];
                    return s;
                }),
            );
            // W = K̂⁻¹U, three columns. Each column of P = I₃⊗(𝟙/√n) is supported on ONE
            // contiguous n-block, so Â⁻¹p_b needs exactly ONE n-backsolve of 𝟙/√n —
            // computed here and reused for all three coordinates ([DESIGN §4.1] step 3).
            const onesRhs = new Float64Array(n).fill(invSqrtN);
            const u = new Float64Array(n);
            solveWithL(L, n, onesRhs, u);
            const W: Float64Array[] = [];
            for (let b = 0; b < 3; b++) {
                const y = new Float64Array(m);
                for (let i = 0; i < n; i++) y[b * n + i] = u[i];
                // K̂⁻¹[p_b; 0] through the same Schur machinery: λ = S⁻¹(C y), x = y − Zλ.
                const Cy = C.map((row) => {
                    let s = 0;
                    for (let i = 0; i < m; i++) s += row[i] * y[i];
                    return s;
                });
                const lam = luSolve(S, Cy);
                const w = new Float64Array(m + k);
                for (let i = 0; i < m; i++) {
                    let s = y[i];
                    for (let r = 0; r < k; r++) s -= Z[r][i] * lam[r];
                    w[i] = s;
                }
                for (let r = 0; r < k; r++) w[m + r] = lam[r];
                W.push(w);
            }
            // M = σ⁻¹I₃ − UᵀW, 3×3. HEAVILY CANCELLING: measured diag(M) is 9.2e4×
            // smaller than σ⁻¹ at N=960 ([DESIGN §7]), so its formation destroys ~5
            // decimal digits and the loss grows ∝ κ. Form and invert it in f64, NEVER
            // f32 and NEVER on the GPU — a hard constraint that also binds slice 2c
            // ([DESIGN §4.7]). CR records diag(M) and κ(M) so this margin is a number.
            const M: number[][] = Array.from({ length: 3 }, (_, a) =>
                Array.from({ length: 3 }, (_, b) => {
                    let s = 0;
                    for (let i = 0; i < n; i++) s += invSqrtN * W[b][a * n + i];
                    return (a === b ? 1 / sigma : 0) - s;
                }),
            );

            /** ẑ = K̂⁻¹r via Schur, then Woodbury back to K⁻¹r. [DESIGN §4.1] steps 2–3. */
            const applyKinv = (rhs: ArrayLike<number>): Float64Array => {
                const y = applyAhatInv(L, n, rhs);
                const rhsBottom = new Array<number>(k);
                for (let r = 0; r < k; r++) rhsBottom[r] = rhs[m + r];
                const Cy = C.map((row, r) => {
                    let s = 0;
                    for (let i = 0; i < m; i++) s += row[i] * y[i];
                    return s - rhsBottom[r];
                });
                const lam = luSolve(S, Cy);
                const xh = new Float64Array(m);
                for (let i = 0; i < m; i++) {
                    let s = y[i];
                    for (let r = 0; r < k; r++) s -= Z[r][i] * lam[r];
                    xh[i] = s;
                }
                // Woodbury: z = ẑ + W M⁻¹ (Pᵀx̂). UᵀK̂⁻¹r = Uᵀẑ = Pᵀx̂ because U = [P; 0].
                const Ptx = [0, 0, 0];
                for (let a = 0; a < 3; a++) {
                    let s = 0;
                    for (let i = 0; i < n; i++) s += invSqrtN * xh[a * n + i];
                    Ptx[a] = s;
                }
                const mu = luSolve(M, Ptx);
                const z = new Float64Array(m + k);
                for (let i = 0; i < m; i++) {
                    let s = xh[i];
                    for (let a = 0; a < 3; a++) s += W[a][i] * mu[a];
                    z[i] = s;
                }
                for (let r = 0; r < k; r++) {
                    let s = lam[r];
                    for (let a = 0; a < 3; a++) s += W[a][m + r] * mu[a];
                    z[m + r] = s;
                }
                return z;
            };

            /** r = rhs − K z against the f64 A and C — K is never materialized. */
            const residualOf = (rhs: ArrayLike<number>, z: Float64Array): Float64Array => {
                const r = new Float64Array(m + k);
                for (let b = 0; b < 3; b++)
                    for (let i = 0; i < n; i++) {
                        let s = 0;
                        for (let j = 0; j < n; j++) s += A[i * n + j] * z[b * n + j];
                        r[b * n + i] = s;
                    }
                for (let q = 0; q < k; q++) {
                    const lamq = z[m + q];
                    for (let i = 0; i < m; i++) r[i] += C[q][i] * lamq;
                }
                for (let q = 0; q < k; q++) {
                    let s = 0;
                    for (let i = 0; i < m; i++) s += C[q][i] * z[i];
                    r[m + q] = s;
                }
                for (let i = 0; i < m + k; i++) r[i] = rhs[i] - r[i];
                return r;
            };

            return {
                kind: 'external',
                solve(rhs: number[]): number[] {
                    let rhsNorm = 0;
                    for (let i = 0; i < rhs.length; i++) rhsNorm += rhs[i] * rhs[i];
                    rhsNorm = Math.sqrt(rhsNorm);
                    const z = applyKinv(rhs);
                    const relResiduals: number[] = [];
                    // A zero rhs has the exact solution z = 0 and no relative residual to
                    // measure; returning here avoids 0/0 becoming a spurious
                    // ExternalSolveError. (CR's projection fixtures assert a NON-zero rhs
                    // so this branch can never make that gate vacuous — plan D6.)
                    if (rhsNorm === 0) {
                        onTrace?.({ refinements: 0, relResiduals: [0] });
                        return Array.from(z);
                    }
                    for (let it = 0; ; it++) {
                        const r = residualOf(rhs, z);
                        let rn = 0;
                        for (let i = 0; i < r.length; i++) rn += r[i] * r[i];
                        const rel = Math.sqrt(rn) / rhsNorm;
                        relResiduals.push(rel);
                        if (rel <= SADDLE_IR_TOL) {
                            onTrace?.({ refinements: it, relResiduals });
                            return Array.from(z);
                        }
                        // `it` counts refinements ALREADY applied to z₀, so the budget is
                        // z₀ + SADDLE_IR_MAX_REFINEMENTS corrections — the same counting
                        // `k1-calibration.py:266`'s `ir_iters` pins with its self-check.
                        // Off-by-one here understates every derived bar.
                        if (it >= SADDLE_IR_MAX_REFINEMENTS) {
                            onTrace?.({ refinements: it, relResiduals });
                            // The ONLY failure detectable during the step. Returning an
                            // unconverged z is a contract violation, not a degraded mode:
                            // nothing in src/core/ thresholds `residual`, so the line
                            // search would reject the wrong g̃ as `armijo_failed` and the
                            // run would auto-pause with usedGpuSolve still true.
                            // @see [DESIGN] §4.1 step 4, §4.5 (fallback triggers 2 and 3)
                            throw new ExternalSolveError(
                                `saddle IR: relative residual ${rel.toExponential(3)} after ` +
                                    `${SADDLE_IR_MAX_REFINEMENTS} refinements (bar ${SADDLE_IR_TOL})`,
                            );
                        }
                        const d = applyKinv(r);
                        for (let i = 0; i < z.length; i++) z[i] += d[i];
                    }
                },
            };
        },
    };
}
```

- [ ] **Step 8: Run the tests**

Run: `bun test test/gpu/saddle.test.ts`
Expected: all NINE PASS — Step 1's six plus Step 5's three.

- [ ] **Step 9: Typecheck, lint, full suite, commit**

Run: `bunx tsc --noEmit && bun test && bunx biome check --write src/gpu/saddle.ts test/gpu/saddle.test.ts; bunx knip`
Expected: `tsc` exit 0; `bun test` = `309 pass / 0 fail` (300 + 9); biome clean on the touched files. **`knip` exits 1 — the pre-existing baseline**, so it is chained with `;`; diff the NAMED sets against Global Constraints. Expected new findings: none — `SADDLE_IR_TOL`, `SADDLE_IR_MAX_REFINEMENTS`, `IrTrace` and (via `asExternal`) `ExternalFactorization` are all imported by `test/gpu/saddle.test.ts`, and `test/**` is a knip entry. If any of them appears as unused, a test regressed to hard-coding the literal.

```bash
git add src/gpu/saddle.ts test/gpu/saddle.test.ts
git commit -m "$(cat <<'MSG'
feat(gpu): src/gpu/saddle.ts — shift + Schur + Woodbury + IR on a supplied factor (phase2a-1 T2)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JdMoLJiC1mByydnaHhuGbr
MSG
)"
```

---

## Task 3: Gate CR — the reformulation is exact, proven before any WGSL

**Files:**
- Create: `test/gpu/cr.test.ts`
- Create: `oracle/tpe_phase2a_reformulation.py`
- Create: `bench/results/gpu-phase2a-cr.json` (written by the test run, then committed — a FIXED filename, not date-derived; see Step 4)

**Interfaces:**
- Consumes: `shiftSobolev`, `choleskyF64`, `makePreparedSaddle`, `asExternal` (Task 2); `saddleSetup`, `woodburyM`, `schurWoodburySolve`, `kappa1of3` (extracted/added in Step 1 below); `solveSaddleFromA` (`linsolve.ts:742`), `solveSaddleFrozen` (`:882`); `IrTrace` (Task 2).
- Produces: `bench/results/gpu-phase2a-cr.json` — the artifact that retires [CAL §J.3].

**Gate CR** — quantity, fixtures and pass semantics from [DESIGN §5]; level from [CAL §F]:
- **Quantity:** `‖Δz‖/‖z‖` between z from [DESIGN §4.1] steps 1–**3** on a CPU **f64** Cholesky and `solveSaddleFromA`'s z. **This is `schurWoodburySolve`'s output, NOT `PreparedSaddle.solve`'s** — `solve()` is steps 2–**4**, and gating on it would let step 4's iterative refinement silently repair any steps-1–3 defect that leaves the relative residual below `SADDLE_IR_TOL` before CR ever looks. CR's job is to gate the REFORMULATION before any WGSL exists; step 4 is the part that will later be running on a wrong f32 factor, so it must not be inside the measurement. The steps-1–4 value is recorded as `relErrWithIr` — informative, not the bar's quantity. See Global Constraints for the measured gap.
- **Bar:** `max(1e-12, 10·κ₂(K,N)·u_f64)`, κ₂ from [K1CAL] `rows[].kappaK` (read from the committed JSON — D7), `u_f64 = 2⁻⁵³`. Evaluates to 1.6e-12 / 1.6e-11 / 1.6e-10 / 1.6e-9 at N=120/240/480/960.
- **13 rhs per fixture, of which 3 are GATED** (D5, adjudicated on measurement):
  - `gradient` / `production` — `[b; 0]`, `b = flatten(gradientAnalytical(γ₀))`. **GATED.**
  - `projection` / `production` — `[0; d]`, `d = −Φ(γ₀ + δ)`. **GATED.**
  - `gradient` / `nullspace𝟙` — `[𝟙; 0]`, `𝟙` over all 3n. **GATED.** The adversarial direction for the shift/Woodbury null-space machinery specifically: `𝟙` per coordinate block is exactly `null(Ā) = span{P₀,P₁,P₂}`, so a bug in the `P`/`U`/`M` path shows here and may not show on a smooth physical gradient. Measured 3.34× slack at n=960 — a real bar, not a formality.
  - 5 × `gradient` + 5 × `projection` seeded draws (splitmix32 uniforms on [−1,1), seeds `1000+s` / `2000+s`, `s = 0…4`). **RECORDED, `gated: false`.** Gating them would make CR red at n=960 against a CORRECT f64 factor — see the calibration debt in Global Constraints.
  - The bar applies to the 3 gated rows. All 13 are emitted.
- **Fixtures:** `crossing` N=8; trefoil N=60/120/240/480/960; constraint sets `barycenter+totalLength` and the same `+1 pin`. **6 × 2 × 13 = 156 rows.**
- **CPU-only (`bun test`), BEFORE any WGSL. CR red ⇒ no GPU work starts** ([DESIGN §6]) — 2a-2 does not begin; localise with the numpy twin first.
- CR also records `diag(M)`, `κ₁(M)` and `min diag(L)²` per fixture so [DESIGN §7]'s Woodbury conditioning margin is a number rather than a pass/fail.
- CR records the **slack** `bar / relErr` as a per-row FIELD, not as prose. [CAL §J.3] makes the slack a committed deliverable, and Step 8's "do not write a slack number into any doc that this JSON did not emit" is unenforceable if the JSON does not emit it.
- CR records the **IR convergence curve** per row — `irRefinements` and `irRelResiduals` from `makePreparedSaddle`'s `onTrace`. [DESIGN §8] hands "results-JSON schema for the IR convergence curve (per N, per rhs)" to the plan; without it a reader cannot tell from the artifact whether IR fired at all, which is exactly the fact that separates `relErr` from `relErrWithIr`.

- [ ] **Step 1: Extract the factor-dependent setup, the steps-2–3 solve, and the Woodbury diagnostics into `src/gpu/saddle.ts`**

**Three extractions, one argument.** CR must measure and record the SAME quantities production computes; a re-implementation in the test would drift silently and the recorded numbers would then describe nothing. Refactor Task 2 Step 7's `factor()` body into three exported helpers and have `makePreparedSaddle` call them — one implementation, two consumers, each time.

- `saddleSetup` is the `const Z … W.push(w)` block lifted **verbatim** (Z, S and W are computed together because W's construction consumes Z and S).
- `schurWoodburySolve` is the `applyKinv` closure lifted **verbatim**, with its four closed-over values (`L`, `n`, `C`, and the `{Z,S,W}` setup plus `M`) promoted to parameters. **This one is not a convenience: it is CR's gated quantity.** [DESIGN §5] pins CR to steps 1–**3**, and `solve()` is steps 2–**4**; without a callable steps-1–3 entry point, CR can only measure the post-refinement vector.
- `woodburyM` is the `const M: number[][] = Array.from(...)` block lifted verbatim.

`factor()` then becomes:

```ts
        factor(C: number[][]): SaddleFactorization {
            const m = 3 * n;
            const k = C.length;
            const setup = saddleSetup(L, n, C);
            const M = woodburyM(setup.W, sigma, n);
            /** ẑ = K̂⁻¹r via Schur, then Woodbury back to K⁻¹r. [DESIGN §4.1] steps 2–3. */
            const applyKinv = (rhs: ArrayLike<number>): Float64Array =>
                schurWoodburySolve(L, n, C, setup, M, rhs);
            /* …residualOf and the returned { kind: 'external', solve } are UNCHANGED… */
        },
```

and everything below `applyKinv` in Task 2 Step 7 stays exactly as written. The comment blocks that sat above the extracted code move onto the helpers' TSDoc, which is where they now belong. `applyAhatInv`, `solveWithL` and `luSolve` stay module-private and are called by `saddleSetup` and `schurWoodburySolve`.

```ts
/**
 * The factor-dependent setup of [DESIGN §4.1] steps 2–3, computed ONCE per step because
 * the factor is frozen ([P §2.5]): Z = Â⁻¹Cᵀ (3k n-backsolves), S = C Z (k×k SPD), and
 * W = K̂⁻¹U (3 columns, ONE extra n-backsolve of 𝟙/√n reused across all three
 * coordinates, because each column of P = I₃⊗(𝟙/√n) is supported on one contiguous
 * n-block). Exported so gate CR forms W the one way production does.
 * Cost: 3k+1 n-backsolves; with P pins k = 4+3P ⇒ 13+9P ([DESIGN §7]).
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1 steps 2–3
 */
export function saddleSetup(
    L: Float64Array,
    n: number,
    C: number[][],
): { Z: Float64Array[]; S: number[][]; W: Float64Array[] } {
    const m = 3 * n;
    const k = C.length;
    const invSqrtN = 1 / Math.sqrt(n);
    // Z = Â⁻¹Cᵀ — 3k n-backsolves.
    const Z: Float64Array[] = C.map((row) => applyAhatInv(L, n, row));
    // S = C Z, k×k, SPD. κ(S) = 5.06…5.28 across N=120…960 ([DESIGN §4.1]), so a
    // plain LU is ample; luSolve is the core's own, already gated.
    const S: number[][] = Array.from({ length: k }, (_, r) =>
        Array.from({ length: k }, (_, c) => {
            let s = 0;
            for (let i = 0; i < m; i++) s += C[r][i] * Z[c][i];
            return s;
        }),
    );
    // W = K̂⁻¹U, three columns. Each column of P = I₃⊗(𝟙/√n) is supported on ONE
    // contiguous n-block, so Â⁻¹p_b needs exactly ONE n-backsolve of 𝟙/√n —
    // computed here and reused for all three coordinates ([DESIGN §4.1] step 3).
    const onesRhs = new Float64Array(n).fill(invSqrtN);
    const u = new Float64Array(n);
    solveWithL(L, n, onesRhs, u);
    const W: Float64Array[] = [];
    for (let b = 0; b < 3; b++) {
        const y = new Float64Array(m);
        for (let i = 0; i < n; i++) y[b * n + i] = u[i];
        // K̂⁻¹[p_b; 0] through the same Schur machinery: λ = S⁻¹(C y), x = y − Zλ.
        const Cy = C.map((row) => {
            let s = 0;
            for (let i = 0; i < m; i++) s += row[i] * y[i];
            return s;
        });
        const lam = luSolve(S, Cy);
        const w = new Float64Array(m + k);
        for (let i = 0; i < m; i++) {
            let s = y[i];
            for (let r = 0; r < k; r++) s -= Z[r][i] * lam[r];
            w[i] = s;
        }
        for (let r = 0; r < k; r++) w[m + r] = lam[r];
        W.push(w);
    }
    return { Z, S, W };
}

/**
 * [DESIGN §4.1] steps 2–3 on a supplied factor and setup: ẑ = K̂⁻¹r through the Schur
 * complement (λ = S⁻¹(Cy − d), x̂ = y − Zλ), then Woodbury back to K⁻¹r
 * (z = ẑ + W M⁻¹ Pᵀx̂). **This — NOT `PreparedSaddle.solve` — is gate CR's quantity.**
 * `solve()` is steps 2–**4**: it applies step 4's iterative refinement on top, so a
 * defect in steps 1–3 that leaves the relative residual anywhere below
 * {@link SADDLE_IR_TOL} is repaired before it can be measured. CR exists to gate the
 * REFORMULATION before any WGSL exists, and step 4 is the part that will later run on a
 * wrong f32 factor, so it must sit OUTSIDE the measurement. Exported for that reason,
 * and called by `makePreparedSaddle`'s `factor()` as its z₀ and as its per-iteration
 * correction solve — one implementation, two consumers.
 * `UᵀK̂⁻¹r = Uᵀẑ = Pᵀx̂` because U = [P; 0].
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1 steps 2–3, §5 (CR)
 */
export function schurWoodburySolve(
    L: Float64Array,
    n: number,
    C: number[][],
    setup: { Z: Float64Array[]; S: number[][]; W: Float64Array[] },
    M: number[][],
    rhs: ArrayLike<number>,
): Float64Array {
    const { Z, S, W } = setup;
    const m = 3 * n;
    const k = C.length;
    const invSqrtN = 1 / Math.sqrt(n);
    const y = applyAhatInv(L, n, rhs);
    const rhsBottom = new Array<number>(k);
    for (let r = 0; r < k; r++) rhsBottom[r] = rhs[m + r];
    const Cy = C.map((row, r) => {
        let s = 0;
        for (let i = 0; i < m; i++) s += row[i] * y[i];
        return s - rhsBottom[r];
    });
    const lam = luSolve(S, Cy);
    const xh = new Float64Array(m);
    for (let i = 0; i < m; i++) {
        let s = y[i];
        for (let r = 0; r < k; r++) s -= Z[r][i] * lam[r];
        xh[i] = s;
    }
    // Woodbury: z = ẑ + W M⁻¹ (Pᵀx̂). UᵀK̂⁻¹r = Uᵀẑ = Pᵀx̂ because U = [P; 0].
    const Ptx = [0, 0, 0];
    for (let a = 0; a < 3; a++) {
        let s = 0;
        for (let i = 0; i < n; i++) s += invSqrtN * xh[a * n + i];
        Ptx[a] = s;
    }
    const mu = luSolve(M, Ptx);
    const z = new Float64Array(m + k);
    for (let i = 0; i < m; i++) {
        let s = xh[i];
        for (let a = 0; a < 3; a++) s += W[a][i] * mu[a];
        z[i] = s;
    }
    for (let r = 0; r < k; r++) {
        let s = lam[r];
        for (let a = 0; a < 3; a++) s += W[a][m + r] * mu[a];
        z[m + r] = s;
    }
    return z;
}

/**
 * M = σ⁻¹I₃ − UᵀW, the 3×3 Woodbury correction matrix of [DESIGN §4.1] step 3.
 * Exported so gate CR records the SAME matrix production inverts — a second
 * implementation in the test would drift and its recorded margin would describe
 * nothing. HEAVILY CANCELLING: measured diag(M) is 9.2e4× smaller than σ⁻¹ at N=960
 * ([DESIGN §7]), which is why M is formed and inverted in f64 only, never f32 and
 * never on the GPU (a hard constraint that also binds slice 2c, [DESIGN §4.7]).
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1 step 3, §7
 */
export function woodburyM(W: Float64Array[], sigma: number, n: number): number[][] {
    const invSqrtN = 1 / Math.sqrt(n);
    return Array.from({ length: 3 }, (_, a) =>
        Array.from({ length: 3 }, (_, b) => {
            let s = 0;
            for (let i = 0; i < n; i++) s += invSqrtN * W[b][a * n + i];
            return (a === b ? 1 / sigma : 0) - s;
        }),
    );
}

/**
 * Exact 1-norm condition number of a 3×3, via the adjugate inverse.
 * Why κ₁ and not κ₂: κ₂ needs an SVD, and for a 3×3 the two differ by at most 3×
 * — far inside the 9.2e4× margin [DESIGN §7] reports — so κ₁ answers the question
 * ("how much of M's formation survived the cancellation?") without a hand-rolled
 * eigensolver in a gate whose whole point is that it is checkable by inspection.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §7
 */
export function kappa1of3(M: number[][]): number {
    const [[a, b, c], [d, e, f], [g, h, i]] = M;
    const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (det === 0 || !Number.isFinite(det)) return Number.POSITIVE_INFINITY;
    const inv = [
        [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
        [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
        [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
    ];
    const norm1 = (X: number[][]): number =>
        Math.max(...[0, 1, 2].map((col) => Math.abs(X[0][col]) + Math.abs(X[1][col]) + Math.abs(X[2][col])));
    return norm1(M) * norm1(inv);
}
```

`makePreparedSaddle`'s `factor()` is rewritten to the four-line form given at the top of this step: the inline `const Z` / `const S` / `const onesRhs … W.push(w)` block becomes `const setup = saddleSetup(L, n, C);`, the inline `const M: number[][] = Array.from(...)` block becomes `const M = woodburyM(setup.W, sigma, n);`, and the `applyKinv` closure body becomes a single call to `schurWoodburySolve(L, n, C, setup, M, rhs)`. `residualOf` and `solve` are untouched.

Run: `bun test test/gpu/saddle.test.ts && bunx tsc --noEmit`
Expected: Task 2's NINE tests still PASS, **byte-identically** (this is a pure extraction — the numbers are the same op order on the same values; if any test moves, the extraction was not semantics-preserving and the `relErr` of Task 2 Step 5's smoke bar is the thing to look at first).

- [ ] **Step 2: Write the CR harness — bar, fixtures, and the two rhs shapes**

Create `test/gpu/cr.test.ts`:

```ts
import { expect, test } from 'bun:test';
import { execSync } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { trefoil } from '../../src/core/fixtures';
import { DEFAULTS } from '../../src/core/optimizer';
import {
    barycenterBlock,
    evaluateConstraintSet,
    pointBlock,
    totalLengthBlock,
} from '../../src/core/sobolev/constraintSet';
import { assembleAFlat } from '../../src/core/sobolev/innerProduct';
import { flatten } from '../../src/core/sobolev/layout';
import { solveSaddleFromA, solveSaddleFrozen } from '../../src/core/sobolev/linsolve';
import { calculateDisjointPairs, gradientAnalytical } from '../../src/core/tangentPointEnergy';
import { type Edge, testConfigs, type Vec3 } from '../../src/core/testConfigs';
// EVERY symbol `measure` uses, including saddleSetup and schurWoodburySolve. A missing
// name here is not a nit: `bun test` transpiles without typechecking, so it surfaces at
// Step 5 as a ReferenceError inside measure() that fails ALL TWELVE fixture tests at
// once — which Step 5 instructs the executor to read as "CR red ⇒ stop Phase 2".
import {
    asExternal,
    choleskyF64,
    type IrTrace,
    kappa1of3,
    makePreparedSaddle,
    saddleSetup,
    schurWoodburySolve,
    shiftSobolev,
    woodburyM,
} from '../../src/gpu/saddle';

const U_F64 = 2 ** -53;
/** Repo-relative, for the `kappaArtifact` provenance key in the emitted JSON. */
const K1CAL = 'bench/results/2026-09-04-gpu-phase2a-k1-calibration.json';
/**
 * The same artifact resolved relative to THIS FILE, for reading. Not the bare
 * repo-relative string: `readFileSync` resolves that against the CWD, so it would only
 * work when `bun test` is invoked from the repo root. Same idiom the repo already uses
 * for oracle fixtures (`test/sobolev/constraintSetFlow.test.ts:60-66`).
 */
const K1CAL_URL = new URL(`../../${K1CAL}`, import.meta.url);

let kappaCache: Record<number, number> | null = null;

/**
 * κ₂(K) per N, READ from the committed [K1CAL] artifact — never transcribed (plan D7).
 * [CAL]'s binding rule is that no gate number lives in prose unless a committed script
 * emitted it into a committed JSON; a literal table here would be that same failure one
 * level down, going stale silently if the calibration is ever re-run.
 *
 * LAZY, and that is load-bearing: read at module scope, a missing or moved artifact
 * throws during IMPORT, which aborts the whole file before any `test()` registers — so
 * the artifact-presence test below would never run and the failure would surface as an
 * unrelated import error. Deferring the read into the tests makes D7's "the JSON path is
 * asserted to exist" actually true.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-gate-calibration.md §F, §A
 */
function kappaByN(): Record<number, number> {
    if (kappaCache) return kappaCache;
    if (!existsSync(K1CAL_URL))
        throw new Error(
            `CR: the [K1CAL] artifact is missing at ${K1CAL_URL.pathname} — without it every ` +
                'bar would silently fall back to the 1e-12 floor and CR would gate at the wrong level',
        );
    const parsed = JSON.parse(readFileSync(K1CAL_URL, 'utf8')) as {
        rows: { n: number; kappaK: number }[];
    };
    const out: Record<number, number> = {};
    for (const r of parsed.rows) out[r.n] = r.kappaK;
    kappaCache = out;
    return out;
}

// Fail loudly if the artifact moved or its schema changed: without this, every bar
// would silently fall back to the 1e-12 floor and CR would gate on the wrong level.
// This test is the FIRST in the file, so a missing artifact reports as this named test
// failing with the message above rather than as twelve unexplained fixture failures.
test('CR: the [K1CAL] κ₂(K) artifact is present and populated', () => {
    const kappa = kappaByN();
    expect(Object.keys(kappa).length).toBeGreaterThanOrEqual(5);
    expect(kappa[960]).toBeGreaterThan(1e6);
});

/**
 * Bar = max(1e-12, 10·κ₂(K,N)·u_f64) ([CAL §F]). N=8 and N=60 have no [K1CAL] row, so
 * they take the 1e-12 floor — which is what the floor is for.
 */
function crBar(n: number): number {
    const kappa = kappaByN()[n];
    return kappa === undefined ? 1e-12 : Math.max(1e-12, 10 * kappa * U_F64);
}

/**
 * γ^q — a deterministic displacement off the constraint manifold (plan D6). Φ is
 * evaluated at the fixture's own x0/L0, so Φ(γ₀) ≈ 0 by construction and a projection
 * rhs taken there would be pure roundoff — a vacuous gate. The frozen projection solves
 * exactly this shape: K(γ₀) frozen, Φ evaluated at the MOVED iterate
 * (`lineSearch.ts:250-262`). Drawn from {@link uniforms}, not numpy's default_rng(0)
 * (PCG64 + a ziggurat, not reproducible in JS — plan D5) and not `Math.sin` (ECMA-262
 * §21.3.2 implementation-approximated). CR must be re-runnable to the bit, and this
 * value feeds a GATED row.
 */
function displaced(vertices: Vec3[]): Vec3[] {
    // Seed 7777, 3n values, consumed coordinate-major per vertex. NOT Math.sin: ECMA-262
    // §21.3.2 leaves Math.sin implementation-approximated, and this feeds a GATED row.
    const u = uniforms(vertices.length * 3, 7777);
    return vertices.map((v, i) => [
        v[0] + 1e-3 * u[3 * i],
        v[1] + 1e-3 * u[3 * i + 1],
        v[2] + 1e-3 * u[3 * i + 2],
    ]) as Vec3[];
}

/**
 * splitmix32 — the ensemble PRNG (plan D5). NOT numpy (`np.random.default_rng(0)` is
 * PCG64 + a ziggurat and is not reproducible in TypeScript) and NOT `Math.random` (not
 * seedable, so CR would stop being re-runnable to the bit). Ten lines, fully specified,
 * and portable to Python in about five if anyone ever wants to cross-check these rows.
 * @see docs/superpowers/plans/2026-09-04-webgpu-solver-phase2a-1.md (D5)
 */
function splitmix32(seed: number): () => number {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x9e3779b9) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 16), 0x21f0aaad) >>> 0;
        t = Math.imul(t ^ (t >>> 15), 0x735a2d97) >>> 0;
        return ((t ^ (t >>> 15)) >>> 0) / 4294967296;
    };
}

/**
 * Uniform draws on [−1,1) from splitmix32 — the ensemble's vectors, and (at seed 7777)
 * D6's displacement.
 * NOT Box–Muller normals: that needs `Math.log` and `Math.cos`, and ECMA-262 §21.3.2
 * leaves `Math.log` / `Math.sin` / `Math.cos` IMPLEMENTATION-APPROXIMATED — not required
 * to agree bit-for-bit across engines or across versions of one engine. The ensemble's
 * pre-registered failure count (Step 5) would then be engine-dependent, and an executor
 * on a different Bun build would escalate a non-issue. Everything here — `Math.imul`,
 * XOR, shifts, `/ 2**32`, `2*u - 1` — is exactly specified, so these rows are
 * reproducible to the bit anywhere, including in Python.
 * The marginal distribution is not load-bearing: the ensemble exists to excite `A`'s
 * badly-conditioned eigendirections, which a uniform vector does as well as a normal one.
 * @see docs/superpowers/plans/2026-09-04-webgpu-solver-phase2a-1.md (D5, D6)
 */
function uniforms(count: number, seed: number): number[] {
    const u = splitmix32(seed);
    const out: number[] = [];
    while (out.length < count) out.push(2 * u() - 1);
    return out;
}

interface CrRow {
    name: string;
    n: number;
    k: number;
    rhsShape: 'gradient' | 'projection';
    /** `production` | `nullspace1` | `rand0`…`rand4` — the row's identity within a fixture. */
    rhsTag: string;
    /**
     * Whether the bar is ASSERTED on this row (plan D5). The three physically-reachable
     * rhs are gated; the ten seeded draws are recorded only, because two of them exceed
     * the bar at n=960 with a CORRECT f64 factor and gating them would be a false kill.
     */
    gated: boolean;
    rhsSource: string;
    rhsNorm: number;
    /** [DESIGN §5]'s gated quantity: steps 1–3 ONLY (schurWoodburySolve, pre-refinement). */
    relErr: number;
    /** Steps 1–4 (PreparedSaddle.solve, post-refinement). Recorded, NOT gated — see C4. */
    relErrWithIr: number;
    bar: number;
    /** bar / relErr — [CAL §J.3]'s committed deliverable, emitted rather than hand-derived. */
    slack: number;
    /** [DESIGN §8]'s IR convergence curve, per N and per rhs: how many corrections step 4 */
    /** actually applied, and the relative-residual curve it walked. 0 ⇒ z₀ already cleared */
    /** SADDLE_IR_TOL, i.e. relErr and relErrWithIr are the same vector. */
    irRefinements: number;
    irRelResiduals: number[];
    kappaSource: string;
    verdict: 'PASS' | 'FAIL';
    diagM: number[];
    kappaM1: number;
    minDiagLSq: number;
}

const rows: CrRow[] = [];
```

- [ ] **Step 3: Add the measurement**

Append to `test/gpu/cr.test.ts`. **One f64 Cholesky, one `prepared.factor(C)` and one reference LDLᵀ serve BOTH rhs shapes** — the reference factor is reused through the already-exported `solveSaddleFrozen`, which is what makes the N=960 fixture affordable (its LDLᵀ of the 2884×2884 K dominates the runtime).

```ts
function measure(name: string, vertices: Vec3[], edges: Edge[], withPin: boolean): CrRow[] {
    const n = vertices.length;
    const dp = calculateDisjointPairs(edges);
    const { alpha, beta, epsilon } = DEFAULTS;
    const A = assembleAFlat(vertices, edges, dp, alpha, beta, epsilon);
    const x0: Vec3 = [0, 0, 0];
    for (const v of vertices) {
        x0[0] += v[0] / n;
        x0[1] += v[1] / n;
        x0[2] += v[2] / n;
    }
    const L0 = edges.reduce(
        (s, [a, b]) =>
            s +
            Math.hypot(
                vertices[a][0] - vertices[b][0],
                vertices[a][1] - vertices[b][1],
                vertices[a][2] - vertices[b][2],
            ),
        0,
    );
    const set = withPin
        ? [barycenterBlock(x0), totalLengthBlock(L0), pointBlock(0, vertices[0])]
        : [barycenterBlock(x0), totalLengthBlock(L0)];
    const { C } = evaluateConstraintSet(set, vertices, edges);
    const m = 3 * n;
    const k = C.length;

    // The reformulation side: shift, f64 Cholesky, ONE setup and ONE factor for both rhs
    // shapes. `setup` and `M` are the SAME values production computes inside factor()
    // (Step 1's extraction), used here for two things: the steps-1–3 solve that IS the
    // gate's quantity, and the diag(M) / κ₁(M) margin the JSON records.
    const { Asigma, sigma } = shiftSobolev(A, n);
    const L = choleskyF64(Asigma, n);
    const setup = saddleSetup(L, n, C);
    const M = woodburyM(setup.W, sigma, n);
    // onTrace is what makes [DESIGN §8]'s IR curve readable from the artifact. Collected
    // into an ARRAY, not a `let … | null`: TS does not narrow a `let` assigned inside a
    // callback (microsoft/TypeScript#9998) and every `trace?.x` would be a TS2339 on
    // `never`. One entry is pushed per ext.solve() call, in call order.
    const traces: IrTrace[] = [];
    const ext = asExternal(
        makePreparedSaddle(A, L, sigma, n, (t) => {
            traces.push(t);
        }).factor(C),
    );

    const zeroM = new Array<number>(m).fill(0);
    const zeroK = new Array<number>(k).fill(0);
    // rhs (a): the gradient shape [b; 0] — exactly what gradient.ts:106 passes.
    const b = flatten(gradientAnalytical(vertices, edges, dp, alpha, beta, epsilon));
    // rhs (b): the projection shape [0; d], d = −Φ(γ^q) — exactly what lineSearch.ts:260
    // passes, with C and A frozen at γ₀ and Φ at the displaced iterate (plan D6).
    const { phi } = evaluateConstraintSet(set, displaced(vertices), edges);
    const d = phi.map((v) => -v);

    // The 13 rhs of plan D5: 3 gated, 10 recorded. Order is fixed and matters only for
    // readability of the emitted JSON — every row carries its own `gated` flag.
    const shapes: {
        shape: 'gradient' | 'projection';
        tag: string;
        gated: boolean;
        source: string;
        rhs: number[];
    }[] = [
        {
            shape: 'gradient',
            tag: 'production',
            gated: true,
            source: 'flatten(gradientAnalytical(γ₀))',
            rhs: [...b, ...zeroK],
        },
        {
            shape: 'projection',
            tag: 'production',
            gated: true,
            source: '−Φ(γ₀ + δ), δ = 1e-3·uniforms(3n, seed 7777)',
            rhs: [...zeroM, ...d],
        },
        {
            // The adversarial probe (D5). 𝟙 over each coordinate block is exactly
            // null(Ā) = span{P₀,P₁,P₂} — the direction the shift ([DESIGN §4.1] step 1)
            // and the Woodbury correction (step 3) exist to handle, so a bug in the
            // P/U/M path shows HERE and may not show on a physical gradient, whose
            // 𝟙-component is small. NOT normalised: the relative error is scale-free.
            shape: 'gradient',
            tag: 'nullspace1',
            gated: true,
            source: '𝟙 over 3n — in null(Ā) = span{P₀,P₁,P₂}',
            rhs: [...new Array<number>(m).fill(1), ...zeroK],
        },
    ];
    for (let s = 0; s < 5; s++) {
        // Recorded, NOT gated (D5): two of these exceed the bar at n=960 with a CORRECT
        // f64 factor, because a generic rhs excites A's worst-conditioned eigendirections
        // and both sides then sit at their own κ·u floor. Asserting on them would be a
        // false kill. They are emitted so the sensitivity lives in the artifact.
        shapes.push({
            shape: 'gradient',
            tag: `rand${s}`,
            gated: false,
            source: `splitmix32 uniforms on [-1,1), seed ${1000 + s}`,
            rhs: [...uniforms(m, 1000 + s), ...zeroK],
        });
        shapes.push({
            shape: 'projection',
            tag: `rand${s}`,
            gated: false,
            source: `splitmix32 uniforms on [-1,1), seed ${2000 + s}`,
            rhs: [...zeroM, ...uniforms(k, 2000 + s)],
        });
    }

    // The reference side: ONE LDLᵀ of K, reused for the other TWELVE rhs via the already
    // exported frozen operator (linsolve.ts:882-905) — same `solveFactored` path. This is
    // what makes the ensemble free: measured at n=960, the reference LDLᵀ is 10.1 s and
    // all 13 rhs together are 1.17 s. Re-factorizing per rhs would cost ~130 s per
    // fixture and the ensemble would not be affordable.
    const first = solveSaddleFromA(A, n, C, shapes[0].rhs.slice(0, m), shapes[0].rhs.slice(m));
    const frozen = { a: A, n, C, fac: first.fac };

    let minDiagLSq = Number.POSITIVE_INFINITY;
    for (let i = 0; i < n; i++) minDiagLSq = Math.min(minDiagLSq, L[i * n + i] ** 2);

    /** ‖z − zref‖₂ / ‖zref‖₂ — CR's quantity, spelt once for both vectors. */
    const relErrOf = (z: ArrayLike<number>, zref: number[]): number => {
        let num = 0;
        let den = 0;
        for (let i = 0; i < zref.length; i++) {
            num += (z[i] - zref[i]) ** 2;
            den += zref[i] ** 2;
        }
        return Math.sqrt(num / den);
    };

    const out: CrRow[] = [];
    for (let si = 0; si < shapes.length; si++) {
        const s = shapes[si];
        // Row 0's reference is the factorizing solve; every later row reuses its factor.
        const ref =
            si === 0
                ? first
                : solveSaddleFrozen(frozen, s.rhs.slice(0, m), s.rhs.slice(m));
        const zref = [...ref.x, ...ref.lambda];
        let rhsNorm = 0;
        for (const v of s.rhs) rhsNorm += v * v;
        rhsNorm = Math.sqrt(rhsNorm);
        // THE GATED VECTOR — [DESIGN §4.1] steps 1–3, before step 4's refinement, which
        // is what [DESIGN §5] pins. Do not replace this with ext.solve(): step 4 repairs
        // any steps-1–3 defect that leaves the relative residual below SADDLE_IR_TOL, so
        // CR would stop distinguishing "the reformulation is exact" from "the
        // reformulation is approximate and IR rescued it" — and step 4 is precisely the
        // part that will later be running on a wrong f32 factor.
        // @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §5 (CR)
        const z13 = schurWoodburySolve(L, n, C, setup, M, s.rhs);
        // The steps 1–4 vector — recorded, never gated. `traces` grows by exactly one
        // entry per solve, so index it at the length captured before the call.
        // `ext.solve` does NOT throw on the ungated rows that exceed the bar: measured,
        // all 13 rhs at n=960 return normally, because IR's stopping criterion is on the
        // RESIDUAL and the residual converges fine — it is the ERROR that is stuck at the
        // conditioning floor. That is the whole reason those rows are recorded rather
        // than gated (plan D5), and it is why no try/catch is needed here.
        const traceIndex = traces.length;
        const z14 = ext.solve(s.rhs);
        const trace = traces[traceIndex];
        const relErr = relErrOf(z13, zref);
        const bar = crBar(n);
        const row: CrRow = {
            name: `${name}/${s.shape}:${s.tag}`,
            n,
            k,
            rhsShape: s.shape,
            rhsTag: s.tag,
            gated: s.gated,
            rhsSource: s.source,
            rhsNorm,
            relErr,
            relErrWithIr: relErrOf(z14, zref),
            bar,
            // [CAL §J.3]'s deliverable, emitted rather than left to be derived by hand.
            slack: bar / relErr,
            irRefinements: trace.refinements,
            irRelResiduals: trace.relResiduals,
            kappaSource: 'K1CAL:rows[].kappaK (k=4)',
            // Computed for EVERY row, but only the `gated: true` rows' verdicts bind —
            // Step 4 asserts on those three alone. A FAIL on an ungated row at n=960 is
            // the EXPECTED, measured behaviour of a generic rhs against a correct f64
            // factor, not a defect; see D5 and the calibration debt in Global Constraints.
            verdict: relErr <= bar ? 'PASS' : 'FAIL',
            diagM: [M[0][0], M[1][1], M[2][2]],
            kappaM1: kappa1of3(M),
            minDiagLSq,
        };
        rows.push(row);
        out.push(row);
    }
    return out;
}
```

`measure` calls `saddleSetup` once and `factor()` calls it again internally, so each fixture pays two `3k+1` backsolve sets. That is deliberate: CR is a correctness gate with no timing budget, and paying it buys the guarantee that the gated vector, the recorded `diag(M)` and `κ₁(M)` all come from the exported helpers rather than a copy that could drift. **Do not "optimise" this by widening `PreparedSaddle` to expose `W` or the setup** — that interface is the sanctioned public-type delta (iv) and it is closed ([DESIGN §4.4]: "The seam adds no other exports"). The helpers on `src/gpu/saddle.ts` are the sanctioned surface; `PreparedSaddle` is not.

- [ ] **Step 4: Add the gate assertions**

```ts
const FIXTURES: { name: string; make: () => { vertices: Vec3[]; edges: Edge[] }; timeoutMs: number }[] = [
    // `crossing` is a FIXED 8-vertex graph (two disconnected 4-vertex polylines).
    // TestConfig.generate's params bag is OPTIONAL, not absent —
    // `generate: (params?: Record<string, number>) => GraphState`
    // (src/core/testConfigs.ts:9-14, :12) — and `crossing` ignores it, so `c.generate()`
    // is the right call. The cast below is a no-op kept for readability: GraphState is
    // structurally `{ vertices: Vec3[]; edges: Edge[] }` (:4-7). The registry lookup is
    // at :269-273.
    {
        name: 'crossing8',
        make: () => {
            const c = testConfigs.find((t) => t.id === 'crossing');
            if (!c) throw new Error('CR: the `crossing` test config is missing');
            return c.generate() as { vertices: Vec3[]; edges: Edge[] };
        },
        timeoutMs: 30_000,
    },
    { name: 'trefoil60', make: () => trefoil(60), timeoutMs: 30_000 },
    { name: 'trefoil120', make: () => trefoil(120), timeoutMs: 60_000 },
    { name: 'trefoil240', make: () => trefoil(240), timeoutMs: 120_000 },
    { name: 'trefoil480', make: () => trefoil(480), timeoutMs: 300_000 },
    { name: 'trefoil960', make: () => trefoil(960), timeoutMs: 900_000 },
];

for (const f of FIXTURES)
    for (const withPin of [false, true]) {
        const label = `${f.name}${withPin ? '+pin' : ''}`;
        test(
            `CR: ${label} — §4.1 reformulation matches solveSaddleFromA within the κ-relative bar`,
            () => {
                const { vertices, edges } = f.make();
                const measured = measure(label, vertices, edges, withPin);
                expect(measured.length).toBe(13);
                for (const row of measured) {
                    // Non-vacuity (plan D6): a projection rhs at roundoff would make this
                    // fixture pass while testing nothing. Holds for all 13.
                    expect(row.rhsNorm).toBeGreaterThan(1e-8);
                }
                // THE GATE — the three physically-reachable rhs ONLY (plan D5). The ten
                // seeded rows are recorded, not asserted: two of them exceed the bar at
                // n=960 against a CORRECT f64 factor, and IR cannot repair it, so gating
                // them would be a false kill. Pin the count so a future edit cannot
                // quietly drop a gated row and leave this loop asserting over fewer.
                const gated = measured.filter((r) => r.gated);
                expect(gated.length).toBe(3);
                for (const row of gated) {
                    // `relErr` is the steps-1–3 quantity [DESIGN §5] pins
                    // (schurWoodburySolve), never `relErrWithIr` — asserting the
                    // post-refinement value would let step 4 repair the thing CR exists
                    // to measure. See C4 / the Gate CR block above.
                    expect(row.relErr).toBeLessThanOrEqual(row.bar);
                }
            },
            f.timeoutMs,
        );
    }

test('CR: write the committed results JSON', () => {
    const sha = execSync('git rev-parse --short HEAD').toString().trim();
    // FIXED filename, deliberately not date-derived. A `${date}-…` name writes a SECOND
    // artifact on any later calendar day instead of refreshing the committed one, and
    // Step 7's `git add` would then stage both — leaving [CAL §J.3]'s deliverable with
    // two candidate sources and no rule for which is current. Provenance is `gitShaShort`
    // INSIDE the file, where it can be read without parsing a filename.
    //
    // NO TIMESTAMP, deliberately. Every `bun test` rewrites this file, so a timestamp
    // would make it dirty the worktree on every run and force a diff that carries no
    // information. With none, the output is BYTE-STABLE: re-running changes the file only
    // when the NUMBERS change, which is exactly what a committed gate artifact should do.
    // Do not add `emittedAt`, and do not gate the write behind an env var either — an
    // env var is a flag an executor can forget to set, and then the artifact silently
    // goes stale instead of being regenerated.
    writeFileSync(
        'bench/results/gpu-phase2a-cr.json',
        `${JSON.stringify(
            {
                gate: 'CR',
                gitShaShort: sha,
                uF64: U_F64,
                barFormula: 'max(1e-12, 10*kappaK*uF64)',
                quantity: '[DESIGN §4.1] steps 1-3 (schurWoodburySolve) vs solveSaddleFromA',
                // The verdict rests on `gated: true` rows only (plan D5). Stated in the
                // artifact so a reader who has never seen the plan cannot mistake an
                // expected ungated FAIL at n=960 for a red gate.
                gatedRhs: ['gradient:production', 'projection:production', 'gradient:nullspace1'],
                ensemblePrng:
                    'splitmix32 uniforms on [-1,1); seeds 1000+s (gradient), 2000+s (projection), s=0..4; displacement seed 7777',
                kappaArtifact: K1CAL,
                fixtures: rows,
            },
            null,
            2,
        )}\n`,
    );
    expect(rows.length).toBe(FIXTURES.length * 2 * 13); // 6 fixtures × 2 sets × 13 rhs = 156
    expect(rows.filter((r) => r.gated).length).toBe(FIXTURES.length * 2 * 3); // 36
});
```

The emitter runs last because `bun test` executes `test()` blocks in declaration order within a file.

- [ ] **Step 5: Run it — a red here STOPS the branch**

Run: `bun test test/gpu/cr.test.ts`
Expected: **14 PASS** — 1 artifact check + 12 fixture tests (6 fixtures × 2 constraint sets) + 1 emitter. Each fixture test measures **13 rhs**, so 14 tests record **156 rows**, of which **36 are gated**; do not confuse tests with rows (the emitter asserts both counts). **Runtime is ≈40 s for the whole file, not minutes** — measured with this plan's own code: n=960 is 16.2 s per fixture (`assembleAFlat` 4.6 s, f64 Cholesky 0.28 s, `saddleSetup` 0.03 s, the reference LDLᵀ 10.1 s, and **all 13 rhs together 1.17 s** — the ensemble is free because it reuses the one factorization), N=480 ≈1.7 s, N=240 ≈0.4 s, N=8/60/120 together under a second. The per-fixture timeouts above are 30–200× the measured cost and stay that way on purpose (they cost nothing and absorb a slow machine), but do not wait for a hang that will not come: if `trefoil960` has not finished in a couple of minutes, something is wrong, not slow.

**Expected `verdict: 'FAIL'` on UNGATED rows at n=960 — that is not a red gate.** Measured at both n=960 constraint sets, `gradient:rand0` exceeds the `1.574e-9` bar with a correct f64 factor: 2.583e-9 (no pin, slack 0.61×) and 2.615e-9 (`+pin`, slack 0.60×). **Exactly two FAIL verdicts are expected in the whole 156-row artifact**, both of them `gated: false`, both at n=960, both `gradient:rand0`. Every other ungated row passes, and all 36 gated rows pass — worst gated slack **3.34×** (`trefoil960/gradient:nullspace1`), then 3.6× (`trefoil960+pin`), 7.0× and 7.4× (the two production projections at n=960). At N ≤ 480 nothing fails at all: the minimum slack over all thirteen rows there is 7.55×. Only the gated rows are asserted (Step 4), so the file is green. Seeing those two FAIL verdicts in the JSON means the gate is behaving exactly as adjudicated — see D5. **A THIRD ungated FAIL, a DIFFERENT ungated row failing, ANY FAIL at N ≤ 480, or ANY gated failure is new information and must be reported (Step 8) rather than absorbed — the first three do not turn the gate red, the fourth does.** For calibration when reading a near miss: under this generator the next-closest ungated rows at trefoil960 are `gradient:rand4` at 1.37×, `gradient:rand1` at 1.62× and `gradient:rand2` at 1.83× — a row moving from 1.6× to below 1× is a real change in the reformulation's conditioning, not noise.

**If any fixture fails, do NOT proceed to plan 2a-2.** [DESIGN §6]: "CR red → the reformulation is wrong; fix the math — no GPU work starts until CR is green." Localise with the numpy twin (next step) before changing `src/gpu/saddle.ts`.

- [ ] **Step 6: Write the numpy twin**

Create `oracle/tpe_phase2a_reformulation.py` — an INDEPENDENT implementation of [DESIGN §4.1] steps 1–3 in numpy over the same trefoil parametrization, importing (not copying) `oracle/tpe_stage1_oracle.py`'s assembly and barycenter helpers, exactly as `bench/gpu/k1-calibration.py` already does. It prints `‖Δz‖/‖z‖` against `numpy.linalg.solve` on the explicitly assembled K, for **both rhs shapes**, at N ∈ {60, 120, 240, 480}.

Its purpose is localisation, and it is the reason CR can be trusted at all: if `test/gpu/cr.test.ts` is red and the twin is green, the defect is in the TypeScript; if both are red, the defect is in the algebra of [DESIGN §4.1] and the SPEC is wrong. A single implementation cannot distinguish those two, and they have opposite remedies.

Run: `uv run --with numpy --with scipy python oracle/tpe_phase2a_reformulation.py`
Expected: every N and both rhs shapes report a relative error at the κ·u_f64 floor.

- [ ] **Step 7: Full verification and commit**

Run: `bunx tsc --noEmit && bun test && bunx biome check --write src/gpu test/gpu; bunx knip`
Expected: `tsc` exit 0; `bun test` = `323 pass / 0 fail` (309 + 14 from `cr.test.ts`); biome clean on touched files. **`knip` exits 1 — the pre-existing baseline**, hence the `;`; diff the NAMED sets against Global Constraints. Expected new findings: none.

```bash
git add test/gpu/cr.test.ts oracle/tpe_phase2a_reformulation.py src/gpu/saddle.ts bench/results/gpu-phase2a-cr.json
git commit -m "$(cat <<'MSG'
test(gpu): gate CR — Schur/Woodbury reformulation exact vs solveSaddleFromA, both rhs shapes (phase2a-1 T3)

Bar is max(1e-12, 10·κ₂(K,N)·u_f64) with κ₂ READ from the committed K1CAL artifact,
never transcribed. Records the per-fixture slack that [CAL §J.3] owed.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JdMoLJiC1mByydnaHhuGbr
MSG
)"
```

- [ ] **Step 8: Report the gate**

CR's verdict decides whether plan 2a-2 may start. Report to the orchestrator, **reading every number out of the emitted `bench/results/gpu-phase2a-cr.json` — none is computed by hand:**
- per fixture, per rhs shape: `relErr` (the gated steps-1–3 quantity), `bar`, and `slack` — the last is the figure [CAL §J.3] declared "a deliverable, not an input", and it is quotable for the first time once this JSON is committed;
- `relErrWithIr` beside `relErr`, and `irRefinements` — together they say whether step 4 fired at all and by how much it moved the answer. Expect `irRefinements = 0` on 147 of the 156 rows and exactly 1 on the other NINE, all at n=960 (measured over all twelve configs, `scratch/rowsall.ts`): at `trefoil960` the two gradient-shape GATED rows `gradient:production` and `gradient:nullspace1`; at `trefoil960+pin` those same two plus all five ungated `projection:rand0…4` draws. Everything at N ≤ 480 is 0, and `projection:production` and the failing `gradient:rand0` are 0 even at n=960. A row with a nonzero `irRefinements` where none was expected means z₀ did not clear `SADDLE_IR_TOL`, i.e. the reformulation is weaker than CR's green verdict alone suggests — say so explicitly, it is the signal C4's fix exists to surface. Note that on `gradient:nullspace1` the one correction makes `relErrWithIr` slightly WORSE than `relErr` (4.711e-10 → 4.959e-10 at trefoil960); that is expected, not a defect, and it is the concrete reason the bar is on the steps-1–3 quantity;
- `diag(M)`, `κ₁(M)` and `min diag(L)²` at N=480 and N=960, against [DESIGN §7]'s 9.2e4× cancellation claim;
- the twin's agreement;
- **the ungated ensemble's spread and worst slack per N** — the ten `gated: false` rows. Expect exactly TWO FAIL verdicts in the whole artifact, both ungated and both at n=960: `gradient:rand0` in each of the two constraint sets (0.61× no pin, 0.60× `+pin`; spread over the TEN ungated rows 11.0× and 13.9× — over all THIRTEEN rows including the gated ones it is 17.8× and 17.4×, which is the figure D5 quotes). At N ≤ 480 expect all ten to pass, minimum ensemble slack 7.55×. A *different* set of ungated failures, a THIRD one, or any failure at N ≤ 480 is new information about the reformulation's conditioning and must be reported rather than absorbed, even though it does not turn the gate red;
- **the worst GATED slack**, which is the number that says how much room CR actually has: measured 3.34× at `trefoil960/gradient:nullspace1`. If it has dropped materially, say so — the null-space probe is the tightest of the three gated rhs and is the one that would go red first;
- **the calibration debt owed to [CAL §F], verbatim, as an item the orchestrator must carry to the spec** (it is in Global Constraints for the same reason): *the 10× headroom in `max(1e-12, 10·κ₂(K,N)·u_f64)` is insufficient for a GENERIC rhs at n=960 — measured effective constant ~15·κ·u; one of ten seeded draws exceeds the bar with a correct f64 factor at each n=960 constraint set, and iterative refinement cannot repair it — it does not even fire. The gated rhs are unaffected. A future change to CR's rhs set must revisit the headroom.* **This plan does not fix it and must not** — amending a pre-registered bar is a spec change, not a plan section.

**Do not write a slack number into any spec or doc that this JSON did not emit.**

---

## Self-review

**Spec coverage.** [DESIGN §4.4]'s delta (iv) — every symbol it lists is in Task 1 except `SimStore.gpuSolveAvailable`, which is explicitly deferred to 2a-3 in Global Constraints. [DESIGN §4.4]'s two `optimizer.ts` control-flow changes each have their own test (Task 1 Steps 1 and 7), and D1 is tested on the collector's own state by BOTH halves — the half the `finally` spelling fails, and the half the no-disarm spelling fails (which the obvious `next.timings.step.calls === 1` spelling could not do at all, because `timingsBegin()` resets unconditionally). [DESIGN §4.4]'s "`solverWorker.ts` throws if `saddle` is present" is Task 1 Step 11, reading `msg.args` — the only level the protocol can carry it at — with a round-trip falsifier in Step 7. [DESIGN §4.1] steps 1–4 are Task 2 Steps 3 and 7 plus Task 3 Step 1's extraction; the coordinate-major convention is stated in Global Constraints, in Task 2 Step 7, and in Task 1's `identityPrepared`. [DESIGN §5]/[CAL §F]'s CR is Task 3, measuring the steps-1–**3** vector via `schurWoodburySolve` (steps 1–4 recorded beside it, not gated), both rhs shapes, and the [K1CAL]-read bar. [DESIGN §8]'s two open questions are both answered here: `PreparedSaddle`'s home is D2, and the IR-convergence-curve JSON schema is `CrRow.irRefinements` / `irRelResiduals`. [CAL §J.3]'s owed deliverable is `CrRow.slack`, emitted into the artifact rather than left to Step 8's prose. Out of scope by construction: [DESIGN §4.3] (kernel), §4.5 (driver), §4.6/§4.7 (2b/2c), and the K1/T4/T5/Boot/P2a/`G5-confirm` bars — all named in Global Constraints so their absence is deliberate rather than a gap.

**[CAL §F]'s rhs pin is discharged by SUPERSET, not by substitution** (D5, adjudicated on measurement). CR records thirteen rhs per fixture — the two production shapes, the `𝟙` null-space probe, and ten seeded draws of exactly the kind [CAL §F]'s sentence reaches for — and gates the three the solver can physically encounter. Nothing the spec asked to be measured is missing from the artifact; what changed is which rows the verdict rests on, and that is defined by reachability rather than by a reading of the spec. No `[CAL §F]` amendment is needed to execute this plan.

**One spec obligation is RECORDED here and deliberately NOT fixed:** the measured calibration debt on [CAL §F]'s 10× headroom (Global Constraints; Task 3 Step 8). It is a bar, bars live in the spec, and a plan that amended one would be doing the thing this plan's own Global Constraints forbid. It is carried to the orchestrator as an owed item, with its evidence, rather than silently absorbed.

**Placeholder scan.** No "TBD", no "add error handling", no "similar to Task N". Every code step carries the actual code. The one narrative step is Task 3 Step 6 (the numpy twin), which specifies its inputs, its two rhs shapes, its N values, its run command and its pass condition, but not its source — deliberately: a twin transcribed from this plan would inherit this plan's errors and could not localise anything, which is the only reason it exists.

**Type consistency.** `PreparedSaddle` / `ExternalFactorization` / `ExternalSolveError` / `SaddleFactorization` are spelled identically in Task 1's Produces block, Task 2's imports, and Task 3's consumers. `makePreparedSaddle(A, L, sigma, n, onTrace?)` matches between Task 2's Produces block, its implementation and every call site. `shiftSobolev` returns `{ Asigma, sigma }` and is destructured that way everywhere. `saddleSetup` / `schurWoodburySolve` / `woodburyM` / `kappa1of3` are introduced in Task 3 Step 1 and consumed only after it; `saddleSetup` returns `{ Z, S, W }`, and both `factor()` and `measure` bind it as a whole (`const setup = …`) because `schurWoodburySolve` takes it as one parameter. **Every symbol `measure` uses appears in Task 3 Step 2's import STATEMENT, not only in prose** — the earlier draft's Self-review asserted this while `saddleSetup` was missing from the import block, so the check that should have caught it returned the wrong answer; it has been re-run against the literal import list.

`.factor(C)` returns the widened union everywhere, so **no call site writes `.factor(C).solve(...)`**; `solve` reaches its callers through `asExternal` (Task 2 Step 7). That is not style: `LuFactorization` (`linsolve.ts:219-223`) has no `solve`, so the direct spelling is a `TS2339` that would stop every `bunx tsc --noEmit &&` chain in this plan before `bun test`, and [DESIGN §4.4] pins `factor`'s return type verbatim so it cannot be narrowed at the source.

**Known residuals, stated rather than hidden.**
1. Task 3 Step 1's extraction means Task 2 Step 7's `factor()` body is refactored one step after it is written. Deliberate — Task 2 stands alone and is committed green first, and the extraction is semantics-preserving, proven by Task 2's nine tests still passing unchanged (Step 1's run command). Folding it into Task 2 would have made Task 2's code unjustifiable at the point it is introduced, since nothing in Task 2 consumes `W`, `M` or the steps-1–3 vector from outside. The cost of the split is that a symbol crosses a task boundary, which is exactly how the missing `saddleSetup` import happened; Task 3 Step 2's import block is now the guard against a repeat.
2. Task 2 Step 1 tests the `choleskyF64` pivot guard on four SYNTHETIC 2×2 matrices rather than on the real `A`, which is why that one part of the file is not at n=60 like its neighbours. This is not a gap: the real `A` is PSD with a one-dimensional null space, so its terminal pivot's sign is a rounding accident and cannot carry an assertion at any `n` (the measurement is preserved as an anchor comment in the step). The synthetic cases are exact in f64, cover both halves of `!(d > 0) || !Number.isFinite(d)` — including the NaN case that a `d <= 0` spelling would admit and that no fixture produces by accident — and include an SPD positive control so they cannot pass against a guard that throws unconditionally.
3. `shiftSobolev`'s `out?: Float64Array` parameter has no caller in this plan — its buffer-reuse justification ([CAL §H.1]) is a 2a-2/2a-3 throughput concern. It is a deliberate forward-compatible parameter, not dead code; noted so a later reviewer does not file it as one.
