# WebGPU Solver — Phase 2a-2 (the f32 GPU Cholesky kernel + gates K1, T4, Boot) Implementation Plan

> ## CORRECTION (2026-09-11, utof/repulsive-test2#51): SUPERSEDED — do not implement this plan
>
> **This plan specifies a dense f32 GPU Cholesky of `A_σ`, and #51 retires that
> architecture: in Yu–Schumacher–Crane the block cluster tree is how the Sobolev inner
> product `A` is *applied*, never assembled (arXiv:2006.07859 §6.2, App. B.2/B.3.4; §6.3:
> *"Since the hierarchical matrix-vector multiply does not build an explicit matrix, we use
> an iterative method to solve our linear systems."*), so under Phase 2H there is no `A` to
> factor and no `L` for K1 to measure.**
>
> **Nothing below is withdrawn, edited or deleted**, and the file must not be removed —
> `CLAUDE.md` keeps SDD / review-loop records as the auditable account of *why* the dense
> path was chosen and what it cost to find out it could not scale. Read it as a process
> record, not as an instruction.
>
> The specs this plan cites carry their own CORRECTION markers against #51 (commit
> `2a01332`): [DESIGN] §1 (`:121`), §4.2 (`:452`) and §4.3 (`:514`, `:572`); [CAL] §E
> (`:351`).
>
> **Implementation status at supersession: ZERO tasks executed.** None of the twelve
> deliverable files exist on `feat/webgpu-phase2a-2`; the branch's only code commit
> (`5233624`) is issue #53's `classifyAdapter` fix, which this plan does not own. Working
> drafts of Tasks 1, 2 (pure parts) and 4 do exist, untracked, in
> `.superpowers/sdd/2026-09-11-webgpu-phase2a-2/scratch/fold-verify/` (11 files, 1 353
> lines) — written by the round-1 fold agent to verify the plan was executable.
>
> ### Salvage verdict per task (full census with line cites in utof/repulsive-test2#55)
>
> Line numbers below are this file's CURRENT numbering. #55's own task table was
> written before this banner and its numbers are 43 lower throughout.
>
> | Task | Line | Verdict | One-line salvage note |
> |---|---|---|---|
> | 1 | `:312` | **MIXED** | `checksum.ts` and `phase2aFixtures.ts` survive; `phase2aMetrics.ts` and `choleskyEmulate.ts` die with the factor — but the **method** (pre-gate a GPU kernel's numerics in CPU f32 before any WGSL exists) is the single highest-value item in the plan and transfers verbatim to the BCT matvec + GMRES. |
> | 2 | `:1113` | **DENSE-ONLY** | Three WGSL Cholesky sources + `triIndex`'s lower-triangle packing die. Salvage only `gpuSolveMaxN`'s pattern: read the device limit at boot, never assert it. |
> | 3 | `:1514` | **DENSE-ONLY** | `GpuCholesky` dies, its **plumbing does not**: D6 (two readbacks, timed separately), D8 (the whole constructor is `bun test`-able because `three/tsl` + `three/webgpu` import with no device), the reused `ReadbackBuffer`, the GPU-side counter in one `renderer.compute([…])` batch, and the status-flag meta buffer are what 2H's matvec engine needs too. |
> | 4 | `:1767` | **ARCHITECTURE-INDEPENDENT** | Highest-value survivor. The *pattern* — every bar read from a committed calibration JSON at run time (D2), one exported verdict function per gate, the bar travelling with the verdict, `readFileSync` so the path is a test parameter — is exactly what 2H's replacement gates (#59) need. The signatures (`k1Verdict`, `t4Verdict`, `K1CalibrationRow`) are Cholesky-shaped and get replaced. |
> | 5 | `:2067` | **DENSE-ONLY** | The K1 / factor-wall / Boot spike dies with `GpuCholesky`. The spike **contract** survives: records values and never classifies, `pass` means only "ran cleanly", registration by one `Object.assign` in `bench/gpu/spikes.ts`, hardware-adapter-or-INVALID. |
> | 6 | `:2243` | **DENSE-ONLY** | T4-after-IR-on-a-factor dies. **D7 survives and gets more important, not less**: `onTrace` fires before the throw, so a budget-blowing N still yields its convergence curve — wrap every solve in `try`/`catch` and continue the sweep. Under 2H every solve is iterative and that curve is the primary artifact. |
> | 7 | `:2402` | **ARCHITECTURE-INDEPENDENT** | The emitter + CLI + gate-report format, on [CAL §J.2]'s rationale: *"a level that only exists as prose is a level that goes stale silently, three times in a row."* Its Step 8 handoff also carries live debts that outlive the architecture — the unpaid CR headroom finding, [CAL §J.4], and `SimStore.gpuSolveAvailable`'s four sites including the `poisoned` latch. |
>
> Also re-read before authoring the 2H plan: the `## Decisions` block at `:241` — D2, D4,
> D5, D6, D7 and D8 are about how to *run* a GPU numerics milestone, not about Cholesky.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land `src/gpu/cholesky.ts` — the unblocked right-looking f32 GPU Cholesky of the shifted Sobolev block `A_σ` — and measure the three gates that need it and nothing else: **K1** (backward error of the read-back factor), **T4** (pure relative residual after iterative refinement on that factor), and **Boot** (the forward self-test metric on `trefoil(64)`).

**Architecture:** Three layers, and a hard rule about where a bar may come from. (1) `src/gpu/cholesky.ts` — pure parts (`triIndex`, `gpuSolveMaxN`, three WGSL sources) that `bun test` can check with no GPU, plus the `GpuCholesky` engine: one upload, 3N dispatches in ONE `renderer.compute([...])` batch driven by a GPU-side column counter, one readback through a reused `THREE.ReadbackBuffer`. (2) `bench/gpu/phase2aVerdicts.ts` — a Bun-side module that **reads every bar out of the committed K1 calibration JSON** and classifies; unit-tested with no GPU, which is where a reviewer can check the gate logic line by line. (3) `bench/gpu/phase2a.ts` — the browser spike, which records **values only** and never a transcribed level. **No driver, no store, no `runGpuSelfTest` wiring** — that is 2a-3.

**Tech Stack:** TypeScript strict, Bun (tests + the CDP bench driver), Biome; three.js **r0.185.1** (`three/tsl` `wgslFn` / `attributeArray` / `instancedArray`, `three/webgpu` `ReadbackBuffer`), headless Chrome on the NVIDIA Quadro RTX 3000 via `bench/gpu/drive.ts`.

**Spec:**
- **[DESIGN]** `docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md` — §1 (scope, slices), §4.2 (precision model + what T4 measures), §4.3 (the kernel), §5 (gate quantities, fixtures, pass semantics), §6 (kill gates), §7 (risks).
- **[CAL]** `docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-gate-calibration.md` — §A (provenance tiers), §E (K1 calibration), §E.5 (the IR budget at n=960), §E.6 (the one precision model), §G (the Boot bar), §J.2 (the unpaid `G5-confirm` emitter debt).
- **[K1CAL]** `bench/results/2026-09-04-gpu-phase2a-k1-calibration.json` — **the operative source of every K1 and T4 level in this plan.**
- **[2A1]** `docs/superpowers/plans/2026-09-04-webgpu-solver-phase2a-1.md` — the merged slice whose `src/gpu/saddle.ts` and core seam this plan consumes.
- Parent spec `docs/superpowers/specs/2026-08-13-webgpu-solver-design.md` cited as **[P §n]**.

**Why this plan is 2a-2 and not 2a:** the Phase 2 spec review hit its 3-round blind cap with a silent-failure-class finding open, and the adjudicated remedy was to SPLIT ([2A1] line 16). 2a-1 took the piece that depends on no contested speed number (CPU-only numerics + gate CR, now **green** — `bench/results/gpu-phase2a-cr.json`, every row `"verdict": "PASS"`, which is [DESIGN §6]'s precondition: "CR red ⇒ no GPU work starts"). 2a-2 is the piece that needs the GPU kernel and no driver. **Driver/store integration + T5 + P2a + `G5-confirm` are 2a-3** ([2A1] Global Constraints, line 35) and MUST NOT be anticipated in code here.

**Phase entry, verified at branch point (`fed4042`, branch `feat/webgpu-phase2a-2`):** [DESIGN §0]'s state detector reads `src/gpu/cholesky.ts` **absent** ⇒ Phase 2 kernel not started; `src/core/sobolev/linsolve.ts` **has** the `'external'` factorization kind ⇒ the core seam landed (2a-1); `bench/results/gpu-phase2a-cr.json` **present and PASS** ⇒ slice 2a-1's gate ran green.

> ## BLOCKING — the kernel choice is UNRESOLVED pending a spec amendment. Read before starting any task.
>
> Round-1 review **measured** the unblocked right-looking f32 kernel this plan specifies at
> **3–21× the `alpha` its K1 bars were calibrated on**. [K1CAL] `rows[].alpha` is
> `np.linalg.cholesky` on an f32 matrix, i.e. LAPACK `spotrf` — blocked, `sgemm`-based,
> register-accumulated (`bench/gpu/k1-calibration.py:242-243`); [DESIGN §4.3]'s kernel rewrites
> each trailing entry into **f32 storage once per column**. On that measurement K1 is **amber at
> N=480 and red at N=960**, and T4 **throws at N=480** — [DESIGN §6]'s pre-registered kill. Two
> independently written implementations agree to five significant figures; the loop variant, a
> single-register accumulator and an FMA-contracted variant all measure the same, so it is the
> f32 accumulation *width*, not the ordering.
>
> **Evidence, options and costs:**
> `.superpowers/sdd/2026-09-11-webgpu-phase2a-2/reports/C1-kernel-precision-KILL.md`
> (re-runnable: `bun test ./.superpowers/sdd/2026-09-11-webgpu-phase2a-2/scratch/orch-f32-check.test.ts ./.superpowers/sdd/2026-09-11-webgpu-phase2a-2/scratch/orch-f32-960.test.ts`, ~5 s, no GPU).
>
> Changing the kernel amends **[DESIGN §4.2]** (its "nothing in the Cholesky needs compensated
> arithmetic" bullet) and **[DESIGN §4.3]** (its "smallest kernel that can pass K1" rationale).
> A plan may not carry an architecture or a bar the spec does not — CLAUDE.md, and this plan's
> own Global Constraint. **So this plan still specifies exactly the kernel [DESIGN §4.3]
> specifies, unchanged, and does not decide the amendment.**
>
> **Execution consequence: Tasks 2, 3, 5 and 6 — everything that writes or runs the WGSL kernel
> — MUST NOT START until the amendment is resolved.** Tasks 1, 4 and 7 are kernel-independent
> and may proceed. Task 1 is in fact the measurement that resolves it: whatever kernel the
> amendment lands on, Task 1 emulates it in CPU f32 and reads K1 and T4 off it in ~5 s with no
> GPU, and Tasks 2–3 transliterate whatever Task 1 measured.

---

## Global Constraints

- **Slice 2a-2 only.** Deliverables are exactly: `bench/gpu/checksum.ts`, `bench/gpu/phase2aFixtures.ts`, `bench/gpu/phase2aMetrics.ts`, `bench/gpu/choleskyEmulate.ts`, `test/gpu/phase2aFixtures.test.ts`, `test/gpu/phase2aMetrics.test.ts`, `test/gpu/choleskyEmulate.test.ts`, `src/gpu/cholesky.ts`, `test/gpu/cholesky.test.ts`, `bench/gpu/phase2aVerdicts.ts`, `test/gpu/phase2aVerdicts.test.ts`, `bench/gpu/phase2a.ts`, a one-line registration in `bench/gpu/spikes.ts`, three committed results JSONs, and a `bench/gpu/README.md` gate report. **If a step here appears to need `GpuDriver`, `SimStore`, `runGpuSelfTest`, `Viewer.tsx`, T5, P2a or `G5-confirm`, it belongs to 2a-3 — stop and report.**
- **No `src/core/**` changes at all.** The seam landed in 2a-1 and is finished; this slice is additive under `src/gpu/` and `bench/`. The committed golden suites (`test/golden.test.ts`, `test/sobolev/constraintSetFlow.test.ts` against `oracle/golden/*.json`) remain the backstop and must stay green.
- **THE BAR RULE — every K1 and T4 level is READ from [K1CAL] at run time, never transcribed.** [DESIGN §5]: "a bar this section ORIGINATES that is not a `[K1CAL]`/`[JSALG]`/`[BASE]` key or a formula over such keys is not a bar". **This is not a style preference — the unsplit 2a plan violated it and the violation is a measured false kill.** Its `K1_RED = {…, 960: 1.45e-7}` and `K1_GREEN = {…, 960: 7.68e-8}` are the *pre-2026-09-04-recalibration* levels — **[CAL §E.5] point 2 is where both superseded numbers are written down**; [CAL §E.1] explains why they moved but does not carry them. The committed `rows[n=960].betaMin` is **4.5284e-07**, 3.12× higher. A kernel measuring `2.0e-7` at n=960 is **amber** against the artifact and **red** against the transcription — i.e. the transcription kills a kernel the calibration does not. Measured, both directions, in `.superpowers/sdd/2026-09-11-webgpu-phase2a-2/scratch/verdict-proto.ts`.
  - **`2.0e-7` is a demonstration of the RULE, not a prediction for this kernel.** The emulated kernel reads **8.21e-07** at n=960 (Task 1), which is red against the artifact *and* against the transcription — the two agree there. The rule is what matters and it is unchanged; do not read the illustration as an expectation.
  - **The one exception, and it is labelled as one:** Boot's `1e-5` is a **CHOSEN ROUND LEVEL**, not a formula over `bootMetric` — [CAL §G] says so explicitly and reports the 110.6× margin as a *check*, not a derivation. It is the only literal numeric bar this plan may contain.
- **T4's residual is the PURE relative residual `‖r − Kz‖₂/‖r‖₂`, computed by `src/gpu/saddle.ts`.** It may **NOT** be read from `solveSaddleFrozen`'s returned `residual`, which normalises by `max(1, ‖r‖₂)` ([DESIGN §4.2]; `linsolve.ts` `structuredSaddleResidual`). On the projection rhs `‖r‖₂ ≪ 1` always, so the core's number is smaller by the full factor `1/‖r‖₂` — in the **permissive** direction — and "passes 1e-10 while sitting at ~1e-6 relative … is exactly the amber failure the throw exists to catch". The unsplit 2a plan's Task 5 Step 3 says to read exactly that number. **Do not inherit it.**
  - **"~4 orders of magnitude" is [DESIGN §4.2]'s PRODUCTION figure; on this gate's own fixtures the factor is ~333×.** The committed CR artifact's projection `rhsNorm` is 2.9958e-3 (`crossing8/projection:production`) and 2.5798e-3 (`trefoil120/…`), so `1/‖r‖₂ ≈ 3.3e2` here. Quoted so a reader who checks the arithmetic against these fixtures does not conclude the claim is wrong. It changes nothing: the pure relative residual is used either way.
- **Gates run on the HARDWARE adapter or they are INVALID, never FAIL** (`bench/gpu/README.md` "INVALID vs FAIL semantics"). `drive.ts` marks a run INVALID automatically when it detects a software adapter; a number from SwiftShader/llvmpipe is not evidence. G0a must have a passing result file or the driver refuses to run.
- **`status ≠ 0` from the kernel is a bug, not a numerics result.** `finalize` encodes the first failing column as `1 + j`. Fix it before reading K1 at that N.
- **Never a NaN-laden `L` into the CPU algebra.** A nonzero status means the factor is discarded; the CPU path serves that case. This is the invariant [DESIGN §4.3] states and [DESIGN §4.5] (2a-3) will enforce in routing.
- **Conventions:** Biome (4-space, single quotes, width 100); `import type`; `bunx tsc --noEmit` clean; `bun test <path>`; `bunx biome check --write` on touched files. **TSDoc on every export with `@see <spec/plan path>` or a `Why:` line, and an inline anchor on every guard, every ε, every `/2`, and every op-order-preserving line** (CLAUDE.md — the reference is what stops a future agent from "cleaning up" an intentional line). Comments cite the **spec/plan path**, never a task number.
- **One commit per task.** Trailer on every commit:
  ```
  Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01HCkLyorgDoBAwcj4q3uBdN
  ```
- **Test baseline: `359 pass / 0 fail across 40 files`**, measured on this branch at `fed4042` immediately before this plan was written and **re-confirmed by round-1 review** (the reviewer's independent run: 359 pass / 0 fail / 40 files, 53 075 `expect()` calls, 40.6 s — the expect-count and wall drift run to run and gate nothing; the pass/fail/file numbers are the ones every Run step compares against). Every "full suite" step compares against 359 + the tests that step adds; the end state of this plan is **391 pass / 0 fail across 45 files**. (2a-1 baselined 292/37; the StrykerJS mutation campaign merged after it added 67 tests and 3 files with **no source change**. Do not use 292.)
- **`bench/results/gpu-phase2a-cr.json` is MODIFIED in the working tree at the branch point** — `gitShaShort` `ed73ed5` → `fed4042` plus Biome JSON reformatting from `lefthook.yml`'s pre-commit `organizeImports`/format job. Round-1 review verified the `fixtures` array is byte-identical after a JSON round-trip (156 rows, every gated row still `PASS`), so D5's pin and the Phase-entry check are unaffected — but "the **committed** CR artifact" describes a file that is currently dirty. Commit or restore it before Task 1 so the equivalence test pins a committed object.
- **`bunx tsc --noEmit` exits 0 today.** Any nonzero is this plan's regression.
- **`bunx knip` exits NONZERO today, and every Run step is written against the NAMED SETS, not the counts.** Raw output: `.superpowers/sdd/2026-09-11-webgpu-phase2a-2/scratch/knip-baseline.txt`. `knip.json`'s `project` glob is `**/*.{ts,tsx,js}`, and knip counts files under `.superpowers/**` even though git ignores that tree — **41 of today's 47 "unused files" are review scratch and have nothing to do with this plan.** An executor who diffs counts will chase a phantom.
  - **Unused files excluding `.superpowers/**` (6):** `bench/gpu/drive.ts`, `bench/gpu/jsAlgebra.ts`, `bench/gpu/phase1.ts`, `bench/gpu/spikes.ts`, `oracle/compare_energy.ts`, `oracle/gen_fixtures.ts`.
  - **Unused exports (6):** `xorshift32` (`bench/gpu/compare.ts:12`), `SOBOLEV_CONVERGENCE_TOL`, `projectOntoConstraintSet`, `projectBarycenter`, `loadSavedConfig`, `saveConfig`. *(Changed since 2a-1's baseline of 8: `assembleBHighFlat`/`assembleBLowFlat` dropped off — 2a-1's tests consume them now.)*
  - **Unused exported types (9, unchanged):** `FieldArgs`, `SobolevStepFailureReason`, `ProjectConstraintSetOptions`, `ProjectBarycenterOptions`, `ProjectConstraintSetResult`, `ProjectBarycenterResult`, `FieldPenalty`, `PhaseSample`, `SavedConfig`.
  - **Expected new findings outside `.superpowers/**`: exactly FIVE names, all of them types or constants used only inside their own module.** The earlier "NONE" was wrong about the mechanism: knip flags an export that **no other file imports**, regardless of how heavily the declaring file uses it. Two proofs in today's tree, no new code needed — `SOBOLEV_CONVERGENCE_TOL` is used at `src/core/optimizer.ts:428` in its own file and is on the unused-exports list above; `PhaseSample` is consumed by `SobolevStepTimings` at `src/core/sobolev/phaseTimings.ts:38` in its own file and is on the unused-exported-types list. The five, with the task that introduces each:

    | New name | Kind | File | Introduced | Why no importer |
    |---|---|---|---|---|
    | `T4Case` | type | `bench/gpu/phase2aFixtures.ts` | Task 1 | return type of `buildT4Case` only |
    | `GpuFactorResult` | type | `src/gpu/cholesky.ts` | Task 3 | return type of `factor()` only |
    | `K1_CALIBRATION_PATH` | const | `bench/gpu/phase2aVerdicts.ts` | Task 4 | used only in that module's own bodies and provenance strings |
    | `K1Calibration` | type | same | Task 4 | parameter type of `k1Row` / `k1Verdict` / `t4Verdict` / `deriveVerdicts` only |
    | `K1CalibrationRow` | type | same | Task 4 | return type of `k1Row` only |

    Every **function and value** export this slice adds does have a test importer (`knip.json`'s `entry` includes `test/**`), and `bench/gpu/phase2a.ts` is reachable from `bench/gpu/spikes.ts`, which is already on the unused-files list and therefore adds no new name. **These five are the whole expected delta. A SIXTH name is the signal to investigate** — do not add anything to the baseline.
  - **The unused-FILE count moves every session and is not a signal.** It was 47 when this plan was drafted, 52 at review, **58** at the last check — the growth is this slice's own review scratch under `.superpowers/**`, which git ignores and knip counts. Compare the named sets above, never the counts.
- Reviewer nits follow CLAUDE.md's inline-fix gate (fix inline only if EVERY gate holds, else `gh issue create` with labels). Blockers are fixed on the branch regardless of size.
- **Pre-registered bars live in the spec, not here** (CLAUDE.md). If this plan appears to need a gate the spec does not carry, that is a spec fix, not a plan section.

---

## External knowledge dependencies

`research-gate` is not invokable in this session, so this table was produced by hand and every row was verified **this session** against the installed tree. Full evidence: `.superpowers/sdd/2026-09-11-webgpu-phase2a-2/scratch/three-api-verification.md`.

| Dependency | Why it matters | Status | Evidence |
|---|---|---|---|
| three.js version | every API below is version-pinned | **RESOLVED** — `0.185.1`, matching what [DESIGN §4.3] verified against | `node_modules/three/package.json` |
| `ReadbackBuffer(maxByteLength)`, `.buffer`, `.release()`, `.dispose()`; exported from `three/webgpu` | the reused readback target is how the 4 MB per-step staging allocation is avoided | **RESOLVED** | `renderers/common/ReadbackBuffer.js:16,32,61,70`; `Three.WebGPU.js:16` |
| `release()` invalidates the data (`buffer = null`, `unmap()`) | dictates copy-then-release ordering in `factor()` | **RESOLVED** | `WebGPUAttributeUtils.js` `releaseCallback` |
| "ReadbackBuffer must be released before being used again" throw | a missing `release()` breaks the SECOND factorization, not the first — a silent per-step failure | **RESOLVED**, literal text confirmed | `WebGPUAttributeUtils.js:401` |
| `getArrayBufferAsync(attribute, target = null, offset = 0, count = -1)` | signature the engine calls twice per factor | **RESOLVED** | `renderers/common/Renderer.js:1951` |
| dispatch size cached on the compute node's backend data | the reason `trailingUpdate` cannot shrink its dispatch per column (the 3× thread waste) | **RESOLVED** | `WebGPUBackend.js:1629-1631`, `:1651-1685` |
| `renderer.compute(nodes, dispatchSize)` takes ONE override for the whole array | same | **RESOLVED** | `Renderer.js:2718` |
| adapter is a local, never stored; device requested with `requiredLimits: {}` | `gpuSolveMaxN` must read the **device** limit; the adapter's is both unreachable and wrong | **RESOLVED** | `WebGPUBackend.js:217`, `:292`, `:243`, `:93` |
| `IndirectStorageBufferAttribute` honoured via `dispatchWorkgroupsIndirect` | the named-not-built escape hatch if P2a's factor wall fails (2a-3) | **RESOLVED** | `WebGPUBackend.js:1636-1644` |
| inter-dispatch storage visibility in one batched pass | the GPU-side column counter depends on it | **RESOLVED, and already shipped** — Phase 1's two-pass reduction depends on the same guarantee | `gpuweb/gpuweb#4434`; `src/gpu/reduce.ts:66-67,116` |
| `tsconfig.json` has no `resolveJsonModule` | decides how the verdict module loads [K1CAL] | **RESOLVED** — use `readFileSync`, not a static JSON import; no tsconfig change | `grep tsconfig*.json` |
| **WGSL f32 `sqrt` accuracy / ULP bound** | the `triIndex` seed is `sqrt(8I+1)`; a non-correctly-rounded root is the whole reason D3's correction exists | **UNRESOLVED — and designed around, not resolved.** Two WebFetch attempts against the normative §15.7.4 table failed this session, so **no ULP figure may be written anywhere** (D3). Nothing load-bearing rests on it, for two independent reasons below | WGSL spec §15.7.4 (not retrievable this session); `.superpowers/sdd/2026-09-11-webgpu-phase2a-2/reports/research-verdict-r1.md` finding 6 |

**No row is `NEEDS RESEARCH`.** One row is `UNRESOLVED`, listed rather than left in prose because a dependency table is only honest if it carries the assumed item. It is neutralised twice over, and **both legs are checkable**: (i) the WGSL correction is two **unbounded `loop`s**, not a ±1 clamp, so a seed off by an arbitrary amount still converges to the exact `(i, k)` — the ±1 in D3's test is the perturbation the test injects, not a bound the kernel assumes; (ii) over the whole served domain the argument is exactly representable in f32 — at n=960, `8·I + 1 ≤ 8·461279 + 1 = 3 690 233 < 2²⁴ = 16 777 216` — so the only error available is the root's own, and a root wrong by more than a few ulp would still be absorbed by (i).

---

## What this slice does NOT do, and who owns it

Named so an executor does not think these are gaps, and so 2a-3's author inherits the constraints rather than rediscovering them.

| Deferred | Owner | Why |
|---|---|---|
| `GpuDriver.step` routing, `gpuSolveSupported`, the `N ≤ GPU_SOLVE_MAX_N` term in the routing predicate | **2a-3** | [DESIGN §4.5]; needs the driver |
| `SimStore.gpuSolveAvailable` and its **four** reset sites (`Viewer.tsx` 486/636/655/669, `poisoned` latch included) | **2a-3** | [2A1] Global Constraints; [DESIGN §7] last bullet |
| Wiring the Cholesky self-test into `runGpuSelfTest` | **2a-3** | 2a-2 ships the *function* and its gate reading; the availability bit it feeds is 2a-3's |
| **T5**, **P2a** | **2a-3** | need the driver; P2a additionally blocked on [CAL §J.1] (five browser CPU f64 full-step p50s) and on `R₂` attribution |
| **`G5-confirm`**, and the [CAL §J.2] emitter that makes its levels readable | **2a-3, as its FIRST tasks** | see the ruling below |
| Blocked (panel/TRSM/SYRK) Cholesky; `IndirectStorageBufferAttribute` dispatch | **pre-registered, not built** | [DESIGN §4.3], [DESIGN §6] — built only if P2a's 100 ms factor wall fails, which is a 2a-3 reading |
| 2b (GPU assembly, gate T4a) and 2c (GPU line search, τ-agreement) | **later slices** | [DESIGN §4.6], §4.7 |

### Ruling: `G5-confirm` stays in 2a-3, and 2a-3 must run it before its integration tasks

**The tension, stated rather than hidden.** [DESIGN §5]'s `G5-confirm` slice column says it runs "after the Cholesky + K1 spike, after the §4.4 core seam has landed, and **BEFORE any driver / store / self-test integration**", and the unsplit plan's D3 says it is measured *without* the driver. Read alone, that puts `G5-confirm` at exactly the 2a-2/2a-3 boundary — i.e. arguably here.

**Ruling: it stays in 2a-3, per [2A1]'s partition (line 35).** **Basis:** [CAL §J.2] is an unpaid pre-registered calibration debt — "a committed script that reads [BASE] and [JSALG], evaluates §I.2's formulas including the worst-corner unit correction, and writes the resulting `r_a` bars into a committed results JSON … **Until then:** §I.2's evaluated numbers are illustrative Tier 2 and the operative definition is the formula." A gate whose levels are not yet readable cannot be the gate that decides whether this slice merges, and writing the emitter here would separate a bar from the gate it serves — the exact coupling [CAL §J.2] exists to prevent. **Cost if wrong, stated accurately:** [DESIGN §6] prices a red `G5-confirm` as "a red discards a spike rather than an integrated branch" — and that pricing assumes the gate fires **before the spike merges**. It does not here: 2a-2 merges `src/gpu/cholesky.ts`, `bench/gpu/phase2a*.ts`, `bench/gpu/phase2aVerdicts.ts` and three results JSONs first, so a red `G5-confirm` in 2a-3 leaves variant (a)'s kernel merged on main for a variant the milestone has abandoned. **That is discarded code, not plan-writing cost.** The ruling still stands — its basis is [CAL §J.2]'s unpaid emitter debt, and a gate whose levels are not yet readable cannot decide this merge — but the bill is a merged kernel plus 2a-3's integration tasks, not a plan section. The mitigation is the ordering: 2a-3 runs emitter → `G5-confirm` → STOP-or-integrate. **That ordering is this plan's handoff obligation to 2a-3 and is repeated in the final task's report.**

---

## The three gates this slice reads

Quantities, fixtures and pass semantics are [DESIGN §5]'s; **every level is a key in [K1CAL]** except Boot's, which [CAL §G] declares a chosen round level.

| Gate | Quantity (exact) | Level, as a [K1CAL] key | Fixtures |
|---|---|---|---|
| **K1** | `‖LLᵀ − A_σ‖_F / ‖A_σ‖_F`, **f64-measured from the read-back L**; the *value* is recorded, not just the verdict | **green** ≤ `rows[n].k1Bar`; **red** ≥ `e_max(n) = rows[n].betaMin`; **amber** between. `status` must be 0 in all cases | `A_σ` of trefoil **N=64/240/480/960** |
| **T4** | **pure relative** `‖r − Kz‖₂/‖r‖₂` (f64) after IR, **and** the IR iteration count, per rhs shape (gradient `[b; 0]`, projection `[0; d]`) | ≤ `t4.tol` within ≤ `t4.maxIters` iterations (top-level keys, not per-N) | crossing **N=8**; trefoil **N=60/120/240/480/960**, total mode. [DESIGN §5]'s T4 row also carries "+ perEdge N ≤ 240 informational only": **NOT RUN in 2a-2, and NOT OWED to 2a-3** — [DESIGN §5] marks it informational, [G6] already routes perEdge N ≥ 480 to CPU f64, and no gate reads it. Recorded here so nobody re-derives it as a gap |
| **Boot** | forward difference `max\|L_gpu − L_cpu\| / max\|L_cpu\|` | **`< 1e-5` — a CHOSEN ROUND LEVEL** ([CAL §G]), *checked* against `rows[n=64].bootMetric = 9.0399e-08` (110.6× margin), never derived from it | trefoil(64) |

### Pre-registered expectations — what a correct kernel should read, so a surprise is legible

These are predictions, not bars. They exist so an executor can tell "the kernel is wrong" from "the calibration predicted this".

> **These expectations are what round-1 review falsified for the kernel as specified — see the BLOCKING notice at the top.** They are [CAL §E.5]'s and [DESIGN §6]'s, they are reproduced faithfully below, and they remain the pre-registration this plan inherits. They are also **conditioned on a factor whose ε is ~`alpha`**, and the emulated kernel's ε is 20× that. **Task 1 measures the actual ε before any WGSL exists; read Task 1's table, not this list, as the prediction for the kernel you are about to build.**

- **K1 at n=960 is expected GREEN, not amber** ([CAL §E.5], [DESIGN §6]). A plausible GPU factor sits at ~3× `alpha` = 1.22e-07, below `k1Bar` = 1.3584e-07. **This inverted at the 2026-09-04 recalibration** — the superseded bars put the same factor in the amber band and pre-registered a narrowing of the served N. If you are reading an older note that predicts amber at 960, it is stale.
- **A green K1 at n=960 leaves ZERO iteration spare in T4** ([CAL §E.5] table): a factor exactly at `k1Bar` (`f = k1Bar/alpha = 3.334`) needs all four refinements. So **T4 green at n=960 with `irRefinements = 4` is the expected outcome, not a warning sign.**
  - **The two numbers below are different rows of the same [CAL §E.5] table and only look contradictory.** `rows[].irModel.<rhs>.solvesTo1e10` is 2/2/3/3/4 at n=64/120/240/480/960, where `residualCurve[0] = 1.0` is the `z=0` entry — so a `solvesTo1e10` of 4 means z₀ plus **three** refinements. That is §E.5's **`f = 1.000` row, a LAPACK-quality factor: "3 — one spare"**, and it is what [DESIGN §4.2]'s "N=960 needs z₀ + 3 refinements against a budget of z₀ + 4" quotes. The "all four refinements, zero spare" claim is §E.5's **`f = 3.000` / `f = 3.334` rows, the plausible-GPU-factor and at-the-green-bar levels: "4 — exactly at the bound"**. Same table, different ε; both correct.
- **K1 is NOT what catches a factor between the green bar and the exhaustion level** ([CAL §E.5] point 3). 2a-1's per-step `ExternalSolveError` throw is. That is why the throw exists; it is not defensive clutter.
- **Boot on a correct kernel reads ~9.04e-08** ([K1CAL] `rows[n=64].bootMetric`), 110.6× inside the bar. A broken kernel (dropped term, off-by-one column) produces **O(1)** relative error, not a marginal one — the bar is slack on purpose because it gates *availability* against an unknown adapter's summation order, not numerics ([CAL §G]).

### Stop conditions ([DESIGN §6]) — read them before continuing past any gate

- **K1 red at N=480** (`≥ rows[n=480].betaMin`) — this is the parent's confirmation spike. **STOP.** Variant (a) is not confirmed; record the measured error and report. The milestone routes to **building variant (b)** — *not* to kernels-only. [DESIGN §6] corrects the first draft on exactly this: on the measured CG count ([K1CAL] `rows[].cgIters`) (b) is a viable fallback, and kernels-only is the outcome only if (b) is then measured below the P2a bars too.
- **K1 amber at any N** — not a kill. T4 at that N decides whether the N is served; record the value.
- **K1 red at an N OTHER than 480 — [DESIGN §6] carries no branch for this, and neither may this plan.** §6 has exactly two K1 branches: "K1 red on N=480" and "K1 amber". A red at N=960 (or 240, or 64) falls between them. **This is a spec gap, not an executor's judgement call: STOP and escalate for a [DESIGN §6] amendment.** Do not adjudicate it in the moment — pre-registration exists to stop exactly that, and originating the rule here would be the bar-rule violation this plan's Global Constraints forbid. *(The amendment worth proposing, for the human to rule on, is that a red at an N ≠ 480 reads like §6's own T4-red branch — it narrows the served N to below that N — since `e_max` is "T4 failure measured in the worst of five directions" and T4 at that N is the arbiter. Proposed, not applied.)* **This is the predicted reading at N=960** — see the BLOCKING notice — so treat it as a branch you are likely to take, not a corner.
- **T4 red at N ≤ 480** — **STOP.** A kill.
- **T4 red at a larger N** — narrows the served N ([G6] semantics); record it in the README, and note that the routing predicate (2a-3) must carry the narrowing.
- **Boot red on the dev box** — the branch does not merge.

---

## File structure

| Path | Responsibility |
|---|---|
| `bench/gpu/checksum.ts` | **Create (Task 1).** One function, `fnv1a32Hex(Float64Array)`. No deps. There is no existing checksum helper in the repo. |
| `bench/gpu/phase2aFixtures.ts` | **Create (Task 1).** The fixture + constraint-set + production-rhs construction shared by the emulated pre-gate, the T4 spike and the CR-equivalence test. **No `three` import** — must run under `bun test`. |
| `bench/gpu/phase2aMetrics.ts` | **Create (Task 1).** `backwardErrorF` (K1's quantity) and `forwardMetric` (Boot's), GPU-free and `three`-free so the two numbers the gates are read from are themselves unit-tested against in-repo oracles. |
| `bench/gpu/choleskyEmulate.ts` | **Create (Task 1).** A statement-for-statement CPU transliteration of the three kernels, in f64 or f32. The f64 mode checks index algebra and write coverage against `choleskyF64`; the f32 mode IS the emulated K1/T4 pre-gate. |
| `test/gpu/phase2aFixtures.test.ts` | **Create (Task 1).** The CR-artifact equivalence check, the boot-fixture checksum + assembly budget, and the σ / `min diag(L)` cross-check against the numpy calibration. |
| `test/gpu/phase2aMetrics.test.ts` | **Create (Task 1).** `backwardErrorF` and `forwardMetric` against `choleskyF64` and against a seeded single-entry perturbation. |
| `test/gpu/choleskyEmulate.test.ts` | **Create (Task 1).** Transliteration exactness, write coverage, the status encoding, and the emulated K1 / T4 readings. |
| `src/gpu/cholesky.ts` | **Create.** The only new production file. Pure parts (`CHOLESKY_FALLBACK_MAX_N`, `gpuSolveMaxN`, `triIndex`, three WGSL sources) + the `GpuCholesky` engine. Value-imports `three/tsl` and `three/webgpu`, which is fine under `bun test` (D8). |
| `test/gpu/cholesky.test.ts` | **Create.** Pure parts, the constructor, its guards and the 3n node-array shape. No GPU. |
| `bench/gpu/phase2aVerdicts.ts` | **Create (Task 4).** Loads [K1CAL] with `readFileSync` and derives every K1/T4/Boot verdict. Bun-side only, **no `three`, no GPU**. This is where a reviewer checks the gate logic. |
| `test/gpu/phase2aVerdicts.test.ts` | **Create (Task 4).** The verdict boundaries, the stale-bar falsifiers and the constants-match assertions. |
| `bench/gpu/phase2a.ts` | **Create.** `phase2aSpikes` registry: `phase2aCholeskyK1` and `phase2aT4`. Browser-side, records **values only**. |
| `bench/gpu/spikes.ts` | **Modify.** One import + one `Object.assign(spikes, phase2aSpikes)`, exactly mirroring the existing `phase1Spikes` merge at `:1031`. |
| `bench/results/<date>-gpu-phase2a-cholesky.json` | **Create (measured).** K1 values, factor wall, boot metric, checksums. |
| `bench/results/<date>-gpu-phase2a-t4.json` | **Create (measured).** T4 residuals + IR curves. |
| `bench/results/<date>-gpu-phase2a-verdicts.json` | **Create (derived).** Every verdict, each beside the [K1CAL] key it was read from. |
| `bench/gpu/README.md` | **Modify.** Add a "Phase 2a-2 gate report" section in the format of the existing "Phase 1 gate report" (`:413`). |

**Untouched, and a step that needs them is out of scope:** all of `src/core/**`, `src/gpu/driver.ts`, `src/gpu/selfTest.ts`, `src/store.ts`, `src/scene/Viewer.tsx`, `test/gpu/cr.test.ts`.

---

## Decisions (read before Task 1; each is falsifiable)

### D1 — Use `REDUCE_LANES`, do not introduce a second lanes constant

The unsplit plan declared `const CHOLESKY_LANES = 64`. `src/gpu/reduce.ts:28` already exports `REDUCE_LANES = 64`, and its TSDoc (`reduce.ts:21-26`) says *every* compute node in the GPU solver is intended to dispatch with that workgroup size. Two constants with the same value and the same meaning is exactly the drift that makes a later "why are these different?" unanswerable. Import it.

### D2 — Every K1/T4 bar is READ from [K1CAL] at run time; the verdict module is Bun-side and GPU-free

See Global Constraints for the rule and the measured false kill. Mechanics:

- **`readFileSync`, not a static JSON import.** `tsconfig.json` sets `moduleResolution: "bundler"` and `strict: true` but has **no `resolveJsonModule`**, so `import cal from '….json'` would force a tsconfig change this slice does not need. `readFileSync` also makes the path a *parameter*, so a test can point the loader at a fixture and prove the classification boundaries.
- **The browser spike computes values; it never classifies.** The spike cannot read `bench/results/` anyway, and keeping the classification out of it means the gate logic lives in a file `bun test` can check line by line — which is where [DESIGN §5]'s three blind rounds kept finding transcription bugs.
- **Boot's `1e-5` is the one literal bar**, and it is labelled `BOOT_BAR` with the [CAL §G] citation saying it is a chosen round level.

### D3 — The `triIndex` correction loops: the real reason, and a test that is not vacuous

**The unsplit plan's D9 reason is measured FALSE, and its test is vacuous.** D9 says the f32 root "can land one row off at a boundary". Emulating the shader's root as `Math.fround(Math.sqrt(Math.fround(8t+1)))` — a *correctly rounded* root narrowed to f32 — the seed equals the true `i` for **every** `t` in the n=960 domain: worst `|seed − i| = 0`, over all 461 280 entries (`.superpowers/sdd/2026-09-11-webgpu-phase2a-2/scratch/engine-probe.test.ts`). **The correction loops never execute on that path, so a brute-force test over row starts and row ends passes with the loops deleted.**

- **The real justification:** WGSL does **not** guarantee a correctly-rounded f32 `sqrt` (accuracy is specified in WGSL §15.7.4), so a real adapter may return a root the JS emulation does not. The loops absorb that. **The exact ULP bound could not be confirmed this session** (two WebFetch attempts against the W3C spec failed to surface the accuracy table), so **do not write a ULP figure into a comment** — state the qualitative claim and cite the measurement. It is in the dependency table as the one `UNRESOLVED` row.
- **The loops are UNBOUNDED `loop`s, not a ±1 clamp, and that is what makes the unresolved bound harmless.** Each loop walks until the invariant holds, so a seed wrong by any amount still converges to the exact `(i, k)`; the `±1` in the test below is the *perturbation injected*, not a bound the kernel relies on. Independently, the argument is exactly representable over the served domain — at n=960 the largest is `8·461279 + 1 = 3 690 233 < 2²⁴` — so the only error available is the root's own. **Do not "simplify" the loops into a clamp**: that would convert the unresolved accuracy bound from harmless into load-bearing.
- **The test must perturb the seed, not brute-force the happy path.** Task 2 ships `triIndexFromSeed(t, seed)` — the correction loops with the seed injected — and tests that `seed ± 1` still recovers the exact `(i, k)`. `triIndex(t)` is then `triIndexFromSeed(t, trunc((√(8t+1) − 1)/2))`. That tests the guard; the brute force tests the mapping. Both are kept; only the brute force existed before.

### D4 — The boot fixture is GENERATED from `trefoil(64)` and pinned by a committed checksum

[DESIGN §4.3] asks for "a committed 64×64 SPD fixture (A_σ of `trefoil(64)`, deterministic)". Two readings; this plan takes the generator one.

**Basis:** (i) a committed 4096-entry f64 blob needs either `resolveJsonModule` or a ~43 KB base64 literal; (ii) the property that actually needs pinning is that `trefoil(64)` → `assembleAFlat` → `shiftSobolev` still produce the *same* `A_σ` — a checksum tests that and a blob does **not** (a blob silently diverges from the live generator, and then the boot metric is measuring the kernel against a stale matrix); (iii) `trefoil` is deterministic with no `Math.random` (`src/core/fixtures.ts:10-23`). **`assembleSeconds: 0.698` in [K1CAL] is the numpy oracle's cost at n=64, not production TS** — Task 1 measures the TS cost and records it. **If TS assembly at n=64 exceeds 100 ms, stop and report**: the boot path then needs the committed blob after all, and that is a plan change, not an executor's call.

**Measured, so there is no placeholder to fill in:** `fnv1a32Hex(A_σ of trefoil(64))` = **`4318905e`**, assembly **12.8–21.0 ms** across runs on the dev box (`.superpowers/sdd/2026-09-11-webgpu-phase2a-2/scratch/fixtures-proto.test.ts`, re-runnable). The 100 ms assertion stays as a regression guard — it has ≥4.8× margin, so it is a guard and not a flaky timing test.

**The pin is ENGINE-SCOPED, and "deterministic" above means "deterministic within one engine".** `trefoil` uses `Math.sin`/`Math.cos` and `src/core/tangentPointEnergy.ts` uses `Math.pow` nine times; ECMA-262 §21.3.2 leaves all three **implementation-approximated** — the same reason `test/gpu/cr.test.ts:162-175` gives for refusing Box–Muller. Measured on this machine, same source, two runtimes:

| digest | Bun 1.3.11 / JSC | Node 22.22.3 / V8 |
|---|---|---|
| `trefoil(64)` vertices | `4761d55d` | `901b1de0` |
| a `Math.sin`/`Math.cos` sample | `9dfd03cb` | `c7a9c73c` |
| a `Math.pow` sample | `39155f4e` | `14228644` |

Consequences, and both must be written down or the pin turns into a ritual re-paste:
- **`4318905e` is the Bun/JSC digest and the test records the engine beside it.** If it ever goes red after a Bun upgrade, that is a *legitimate* possible cause and the executor must say which before touching the value — a bare re-paste destroys the pin.
- **The spike's `asigmaChecksum` (Task 5) is a Chrome/V8 digest and is DELIBERATELY NOT compared with it.** It is provenance for that run — "which matrix did this row measure" — not a cross-engine equality claim. Nothing in this plan asserts the two agree, and nothing may.
- **Boot itself is unaffected:** Task 5 derives `L_gpu` and `L_cpu` from the same in-browser `A_σ`, so the forward metric never crosses an engine boundary.

### D5 — T4 reuses gate CR's fixtures and production rhs through an extracted builder; `cr.test.ts` is NOT modified

T4 and CR differ in exactly one substitution: CR factors with `choleskyF64`, T4 factors on the GPU. Everything else — the six fixtures, the `barycenter + totalLength (+ pin)` set, the production `b = flatten(gradientAnalytical(…))` and `d = −Φ(displaced)` — is the same, and it is currently inline in `test/gpu/cr.test.ts:241-300`.

Duplicating it into the spike would let the two drift, and drift between CR's rhs and T4's rhs makes the pair of gates incomparable. Extracting it *out of* `cr.test.ts` would edit a merged gate test, which this slice is not licensed to do.

**Therefore:** Task 1 writes `bench/gpu/phase2aFixtures.ts` as a fresh implementation, and pins it with an **equivalence test against the committed CR artifact** — build the same fixture, factor with `choleskyF64`, run `schurWoodburySolve`, and assert the resulting `relErr` matches `bench/results/gpu-phase2a-cr.json`'s committed row for that fixture. That is a falsifiable check against a number nobody in this slice produced, and it fails loudly if the extraction drifts.

### D6 — `factor()` performs TWO readbacks, not one; both are timed separately

[DESIGN §4.3] says "the CPU reads back the whole buffer once", which is true of the *factor*. The status flag lives in a separate 2-slot u32 buffer and needs its own `getArrayBufferAsync`, so a factorization costs **two** readbacks: a 8-byte meta read and an N²·4-byte factor read.

This is named rather than hidden because P2a's factor-wall bar (2a-3) counts wall time including readbacks, and a second round-trip is a real per-step cost. `factor()` therefore returns `wallMs` (the whole thing) **and** `metaReadbackMs` and `factorReadbackMs` separately, so 2a-3 has the number instead of re-deriving it. Only the factor read uses the reused `ReadbackBuffer`; the meta read passes `target = null` (8 bytes — a reused target would buy nothing and would collide with the factor's).

### D7 — A T4-failing N still yields its IR curve, because `onTrace` fires before the throw

`src/gpu/saddle.ts:424` calls `onTrace?.({ refinements: it, relResiduals })` **immediately before** `throw new ExternalSolveError(...)` at `:440`. So the spike gets the full convergence curve for the N that blew the budget — which is the single most diagnostic artifact T4 can produce. **The spike must wrap each `solve()` in `try/catch`, record the row as `converged: false` with the captured trace, and CONTINUE the sweep.** An uncaught throw would abort the remaining fixtures and lose every later N.

Note the counting: `relResiduals[0]` is the residual of z₀ (before any correction), so `relResiduals.length === refinements + 1`. T4's "iterations" is `refinements`, bar `≤ t4.maxIters`.

### D8 — `bun test` can import `three/tsl` and `three/webgpu`, so the whole constructor is unit-testable

Measured (`scratch/three-import-probe.test.ts`, `scratch/engine-probe.test.ts`): both modules import under `bun test` in ~170 ms; `new THREE.ReadbackBuffer(64)` constructs with no device; and `wgslFn(src)({…}).compute(count, [lanes])` **builds a compute node with no device or renderer**. Repeating one node object gives object identity, so "3 node builds, not 3n shader compiles" is directly assertable.

**Consequence:** `GpuCholesky`'s constructor, its guards and its node-array shape are `bun test` territory; only `factor()` needs hardware. **A plain `bun run` of the same imports HANGS** (three keeps the event loop alive) — probe with `bun test`, never `bun run`.

---

## Task 1: the emulated K1 / T4 pre-gate — the fixtures, the two gate metrics, and a CPU transliteration of the kernel (no GPU, no WGSL, no `three`)

**Why this task exists and why it is first.** Every number K1 and T4 report is a function of three things: the matrix (`A_σ`), the metric (`backwardErrorF` / the IR residual), and the factor. This task builds the first two and a **CPU f32 emulation of the third**, so the gate can be read *before* a line of WGSL exists, in ~5 s, with no GPU. Round-1 review demonstrated the payoff: the emulation is what measured the kernel at 3–21× its calibrated `alpha` and produced the BLOCKING notice at the top of this plan. It is to the kernel what gate CR is to the algebra.

**It is also option-independent.** Whichever kernel the [DESIGN §4.2]/[§4.3] amendment lands on, this task emulates *that* kernel and reads K1 and T4 off it. Tasks 2–3 then transliterate what Task 1 measured, in WGSL.

**Files:**
- Create: `bench/gpu/checksum.ts`
- Create: `bench/gpu/phase2aFixtures.ts`
- Create: `bench/gpu/phase2aMetrics.ts`
- Create: `bench/gpu/choleskyEmulate.ts`
- Create: `test/gpu/phase2aFixtures.test.ts`
- Create: `test/gpu/phase2aMetrics.test.ts`
- Create: `test/gpu/choleskyEmulate.test.ts`

**Interfaces:**
- Consumes: `trefoil`, `testConfigs`, `DEFAULTS`, `assembleAFlat`, `flatten`, `calculateDisjointPairs`, `gradientAnalytical`, `barycenterBlock`/`totalLengthBlock`/`pointBlock`/`evaluateConstraintSet`; from `src/gpu/saddle.ts`: `shiftSobolev`, `choleskyF64`, `makePreparedSaddle`, `asExternal`, `IrTrace`; from `src/core/sobolev/linsolve.ts`: `ExternalSolveError` (**it is NOT re-exported by `saddle.ts`** — see Task 6 Step 1).
- Produces:
  ```ts
  // bench/gpu/checksum.ts
  export function fnv1a32Hex(x: Float64Array): string;

  // bench/gpu/phase2aFixtures.ts
  export interface T4Case { name: string; n: number; k: number; A: Float64Array; Asigma: Float64Array; sigma: number; C: number[][]; gradientRhs: number[]; projectionRhs: number[]; }
  export const PHASE2A_FIXTURES: { name: string; make: () => { vertices: Vec3[]; edges: Edge[] } }[];
  export function buildT4Case(name: string, vertices: Vec3[], edges: Edge[], withPin: boolean): T4Case;

  // bench/gpu/phase2aMetrics.ts
  export function backwardErrorF(L: Float64Array, Asigma: Float64Array, n: number): number;
  export function forwardMetric(Lgpu: Float64Array, Lcpu: Float64Array, n: number): number;

  // bench/gpu/choleskyEmulate.ts
  export function emulateCholesky(
      A0: Float64Array,
      n: number,
      opts: { precision: 'f32' | 'f64'; mode?: 'faithful' | 'fast' },
  ): { a: Float64Array; status: number; writes: Int32Array | null };
  ```

- [ ] **Step 1: Write the failing tests**

Create `test/gpu/phase2aFixtures.test.ts`:

```ts
// test/gpu/phase2aFixtures.test.ts — the fixtures every 2a-2 gate is measured on, pinned
// against two artifacts NOBODY in this slice produced: gate CR's committed rhs norms, and
// the numpy calibration's own σ and min diag(L).
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §5
// @see docs/superpowers/plans/2026-09-11-webgpu-solver-phase2a-2.md (D4, D5)
import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { choleskyF64 } from '../../src/gpu/saddle';
import { fnv1a32Hex } from '../../bench/gpu/checksum';
import { buildT4Case, PHASE2A_FIXTURES } from '../../bench/gpu/phase2aFixtures';

/**
 * Repo-relative for provenance; resolved against THIS FILE for reading. `readFileSync` on
 * the bare string resolves against the CWD, so it would only work when `bun test` runs from
 * the repo root — the idiom `test/gpu/cr.test.ts:50-59` already states and rejects, and the
 * same one `test/sobolev/constraintSetFlow.test.ts` uses for oracle fixtures.
 */
const CR = 'bench/results/gpu-phase2a-cr.json';
const K1CAL = 'bench/results/2026-09-04-gpu-phase2a-k1-calibration.json';
const read = (p: string) => readFileSync(new URL(`../../${p}`, import.meta.url), 'utf8');

const make = (name: string) => {
    const f = PHASE2A_FIXTURES.find((x) => x.name === name);
    if (!f) throw new Error(`missing fixture ${name}`);
    return f.make();
};

test('buildT4Case reproduces gate CR’s fixtures and production rhs, to the bit', () => {
    // D5: T4 and CR differ in exactly ONE substitution (the factor's source). If the rhs
    // construction drifts, the two gates stop being comparable and nothing else notices.
    // `rhsNorm` depends only on the fixture, the constraint set and the rhs — not on the
    // factorization — so the committed CR artifact pins it exactly. CR's `rhsNorm` is the
    // norm of the FULL 3n+k vector (cr.test.ts:385-387), so for [b; 0] it equals ‖b‖ and
    // for [0; d] it equals ‖d‖ — the zero padding contributes exactly 0.
    const crRows = (
        JSON.parse(read(CR)) as { fixtures: { name: string; n: number; k: number; rhsNorm: number }[] }
    ).fixtures;
    const norm = (v: number[]) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    // crossing8 and trefoil120 only: enough to pin both shapes and both k values without
    // paying trefoil960's assembly in the unit suite.
    for (const [name, withPin] of [
        ['crossing8', false],
        ['crossing8', true],
        ['trefoil120', false],
    ] as const) {
        const { vertices, edges } = make(name);
        const c = buildT4Case(name, vertices, edges, withPin);
        const label = `${name}${withPin ? '+pin' : ''}`;
        const g = crRows.find((r) => r.name === `${label}/gradient:production`);
        const p = crRows.find((r) => r.name === `${label}/projection:production`);
        if (!g || !p) throw new Error(`missing CR rows for ${label}`);
        expect(c.n).toBe(g.n);
        expect(c.k).toBe(g.k);
        expect(c.k).toBe(withPin ? 7 : 4); // pointBlock writes 3 rows per pin ([DESIGN §7])
        expect(norm(c.gradientRhs.slice(0, 3 * c.n))).toBeCloseTo(g.rhsNorm, 12);
        expect(norm(c.projectionRhs.slice(3 * c.n))).toBeCloseTo(p.rhsNorm, 12);
    }
});

test('the boot fixture is deterministic UNDER THIS ENGINE, and generating it at boot is cheap enough', () => {
    // D4: the fixture is GENERATED, not a committed blob — so what needs pinning is that
    // trefoil(64) → assembleAFlat → shiftSobolev still produce the same A_σ. A blob could
    // not test that; it would silently diverge from the live generator.
    //
    // ENGINE SCOPE — read D4 before touching the digest. `trefoil` uses Math.sin/Math.cos
    // and tangentPointEnergy uses Math.pow, all three IMPLEMENTATION-APPROXIMATED by
    // ECMA-262 §21.3.2 (the same reason cr.test.ts:162-175 refuses Box–Muller). Measured on
    // the dev box, same source: trefoil(64) vertices digest to 4761d55d under Bun 1.3.11/JSC
    // and 901b1de0 under Node 22.22.3/V8. This digest is therefore a BUN/JSC value. A red
    // here after a Bun upgrade is a legitimate cause — say which cause before editing it; a
    // bare re-paste turns the pin into a no-op.
    const t0 = performance.now();
    const { vertices, edges } = make('trefoil64');
    const c = buildT4Case('trefoil64', vertices, edges, false);
    const ms = performance.now() - t0;
    console.log(`[boot] trefoil(64) A_σ assembly: ${ms.toFixed(1)} ms, checksum ${fnv1a32Hex(c.Asigma)}`);
    expect(c.n).toBe(64);
    expect(fnv1a32Hex(c.Asigma)).toBe('4318905e'); // Bun 1.3.11 / JSC; see above
    // Regression guard, not a timing gate: measured 12.8–21.0 ms on the dev box, so this has
    // ≥4.8× margin. If it ever exceeds 100 ms the boot self-test (2a-3) cannot generate its
    // fixture inline and D4's ruling must be revisited — stop and report, do not raise it.
    expect(ms).toBeLessThan(100);
});

test('the checksum is deterministic and sensitive to a single ulp', () => {
    const a = new Float64Array([1, 2, 3, 4]);
    const b = new Float64Array([1, 2, 3, 4]);
    expect(fnv1a32Hex(a)).toBe(fnv1a32Hex(b));
    // NOT `Math.nextUp(4)` — that is a Stage-1 Math-extensions proposal and does NOT exist
    // in Bun (`bun -e "typeof Math.nextUp"` → undefined; `Math.f16round` is the one that
    // shipped), so the call would throw a TypeError and the test would fail for a reason
    // that has nothing to do with the checksum. 2⁻⁵⁰ is below 4's ulp spacing edge case
    // only in the sense that it is exactly representable next to it: 4 + 2⁻⁵⁰ ≠ 4 in f64.
    b[3] = 4 + 2 ** -50;
    expect(b[3]).not.toBe(4);
    expect(fnv1a32Hex(a)).not.toBe(fnv1a32Hex(b));
});

test('the TypeScript A_σ IS the numpy A_σ the K1 bars were bisected on', () => {
    // LOAD-BEARING AND OTHERWISE UNCHECKED. Every K1 bar (`alpha`, `k1Bar`, `betaMin`) was
    // bisected in numpy (bench/gpu/k1-calibration.py) on numpy's A_σ; the GPU is fed the
    // TypeScript A_σ. If the two assemblies differ, every bar is a bar for a DIFFERENT
    // matrix — a silent failure with no symptom anywhere else in the slice. `sigma` and
    // `minDiagL` are already committed per row, so the check costs one f64 Cholesky per N.
    // n=960 is omitted for cost; the K1 spike records `sigma` per row, so it is covered
    // there against the same key.
    const cal = JSON.parse(read(K1CAL)) as { rows: { n: number; sigma: number; minDiagL: number }[] };
    for (const n of [64, 120, 240, 480]) {
        const row = cal.rows.find((r) => r.n === n);
        if (!row) throw new Error(`no K1 calibration row for n=${n}`);
        const { vertices, edges } = make(`trefoil${n}`);
        const c = buildT4Case(`trefoil${n}`, vertices, edges, false);
        const L = choleskyF64(c.Asigma, n);
        let minDiag = Number.POSITIVE_INFINITY;
        for (let i = 0; i < n; i++) minDiag = Math.min(minDiag, L[i * n + i]);
        // Measured agreement is machine precision: σ rel diff ≤ 1.6e-16, min diag(L) rel
        // diff ≤ 7.0e-13 (the latter grows with n through the factor's own conditioning).
        expect(Math.abs(c.sigma - row.sigma) / row.sigma).toBeLessThan(1e-12);
        expect(Math.abs(minDiag - row.minDiagL) / row.minDiagL).toBeLessThan(1e-10);
    }
});
```

Create `test/gpu/phase2aMetrics.test.ts`:

```ts
// test/gpu/phase2aMetrics.test.ts — the two functions that PRODUCE K1's and Boot's numbers.
// The verdict module makes the classifier falsifiable; without this file the classifier is
// fed a value from an unverified metric, and a wrong `lim`, a `den` over the wrong triangle
// or an off-by-one that reads the strict upper triangle would yield a plausible K1 that
// gets classified with full provenance ceremony. Both are pure and GPU-free, and both have
// oracles already in the repo. Precedent: bench/gpu/compare.ts ↔ test/gpu/compare.test.ts.
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §5
import { expect, test } from 'bun:test';
import { choleskyF64 } from '../../src/gpu/saddle';
import { backwardErrorF, forwardMetric } from '../../bench/gpu/phase2aMetrics';
import { buildT4Case, PHASE2A_FIXTURES } from '../../bench/gpu/phase2aFixtures';

const boot = () => {
    const f = PHASE2A_FIXTURES.find((x) => x.name === 'trefoil64');
    if (!f) throw new Error('missing trefoil64');
    const { vertices, edges } = f.make();
    return buildT4Case('trefoil64', vertices, edges, false);
};

test('backwardErrorF reads roundoff on an EXACT f64 factor of the real A_σ', () => {
    // The oracle: choleskyF64 of A_σ IS a correct factor, so ‖LLᵀ − A_σ‖_F/‖A_σ‖_F must sit
    // at f64 roundoff. A metric with the wrong `lim`, the wrong triangle in `den`, or an
    // index transposed would not land here.
    const c = boot();
    expect(backwardErrorF(choleskyF64(c.Asigma, 64), c.Asigma, 64)).toBeLessThan(1e-14);
});

test('backwardErrorF NEVER reads the strict upper triangle of L', () => {
    // The read-back buffer's strict upper triangle is garbage by construction ([DESIGN §4.3])
    // — the kernels never write it and `factor()` never copies it. Poisoning it must not move
    // the metric by one ulp. This is the off-by-one this test exists to catch.
    const c = boot();
    const L = choleskyF64(c.Asigma, 64);
    const clean = backwardErrorF(L, c.Asigma, 64);
    for (let i = 0; i < 64; i++) for (let j = i + 1; j < 64; j++) L[i * 64 + j] = 1e6;
    expect(backwardErrorF(L, c.Asigma, 64)).toBe(clean);
});

test('forwardMetric is 0 on identical factors and reads the known ratio on one perturbed entry', () => {
    // n=3, explicit: max|L_cpu| = 7, and moving one entry by 0.7 must read exactly 0.1.
    const n = 3;
    const cpu = new Float64Array([2, 0, 0, 3, 4, 0, 5, 6, 7]);
    expect(forwardMetric(Float64Array.from(cpu), cpu, n)).toBe(0);
    const gpu = Float64Array.from(cpu);
    gpu[3] = 3.7;
    expect(forwardMetric(gpu, cpu, n)).toBeCloseTo(0.1, 15);
    // Strict upper is never read here either — Boot compares LOWER triangles only.
    gpu[1] = 1e6;
    expect(forwardMetric(gpu, cpu, n)).toBeCloseTo(0.1, 15);
});
```

Create `test/gpu/choleskyEmulate.test.ts`:

```ts
// test/gpu/choleskyEmulate.test.ts — the emulated kernel: first its ALGEBRA (f64, against
// choleskyF64 and a write-coverage census), then its PRECISION (f32, which is the emulated
// K1 and T4 pre-gate).
//
// HONEST LIMITATION, do not let this file be read as "the kernel is verified": the
// emulation is a HAND TRANSLITERATION of the WGSL strings in src/gpu/cholesky.ts and can
// drift from them. It is bounded by the one-bare-`fn` test, the parameter-name test, the
// triIndex agreement test (all Task 2) and ultimately by K1 on hardware. It narrows the
// untested surface from "the whole algorithm" to "the transliteration"; it does not
// eliminate it.
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.3, §5, §6
// @see docs/superpowers/plans/2026-09-11-webgpu-solver-phase2a-2.md (Task 1)
import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { ExternalSolveError } from '../../src/core/sobolev/linsolve';
import { asExternal, choleskyF64, type IrTrace, makePreparedSaddle } from '../../src/gpu/saddle';
import { emulateCholesky } from '../../bench/gpu/choleskyEmulate';
import { buildT4Case, PHASE2A_FIXTURES } from '../../bench/gpu/phase2aFixtures';
import { backwardErrorF } from '../../bench/gpu/phase2aMetrics';

const K1CAL = 'bench/results/2026-09-04-gpu-phase2a-k1-calibration.json';
const cal = JSON.parse(
    readFileSync(new URL(`../../${K1CAL}`, import.meta.url), 'utf8'),
) as { t4: { tol: number; maxIters: number }; rows: { n: number; alpha: number; k1Bar: number; betaMin: number }[] };
const calRow = (n: number) => {
    const r = cal.rows.find((x) => x.n === n);
    if (!r) throw new Error(`no K1 calibration row for n=${n}`);
    return r;
};
const fixture = (n: number) => {
    const f = PHASE2A_FIXTURES.find((x) => x.name === `trefoil${n}`);
    if (!f) throw new Error(`missing fixture trefoil${n}`);
    const { vertices, edges } = f.make();
    return buildT4Case(`trefoil${n}`, vertices, edges, false);
};

test('in f64 the three kernels reproduce choleskyF64 and write every entry exactly once per column', () => {
    const n = 6;
    // A well-conditioned SPD matrix: M Mᵀ + n I. LCG, not Math.random — reproducible.
    const M = new Float64Array(n * n);
    let s = 12345;
    const rnd = () => ((s = (1103515245 * s + 12345) % 2147483648) / 2147483648) * 2 - 1;
    for (let i = 0; i < n * n; i++) M[i] = rnd();
    const A = new Float64Array(n * n);
    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) {
            let v = i === j ? n : 0;
            for (let t = 0; t < n; t++) v += M[i * n + t] * M[j * n + t];
            A[i * n + j] = v;
        }
    const ref = choleskyF64(A, n);
    const { a, status, writes } = emulateCholesky(A, n, { precision: 'f64', mode: 'faithful' });
    expect(status).toBe(0);
    let worst = 0;
    for (let i = 0; i < n; i++)
        for (let j = 0; j <= i; j++) worst = Math.max(worst, Math.abs(a[i * n + j] - ref[i * n + j]));
    expect(worst).toBeLessThan(1e-12);
    // THE WRITE CENSUS — this is the payload. It is what proves no entry is updated twice
    // or zero times, which is the failure mode a RIGHT-looking factorization has and a
    // left-looking one does not.
    if (!writes) throw new Error('faithful mode must return a write census');
    for (let i = 0; i < n; i++) {
        expect(writes[i * n + i]).toBe(1 + i); // i trailing updates, then one finalize
        for (let j = 0; j < i; j++) expect(writes[i * n + j]).toBe(1 + j); // j updates + one colFactor
        for (let j = i + 1; j < n; j++) expect(writes[i * n + j]).toBe(0); // strict upper NEVER written
    }
});

test('in f64 the kernels reproduce choleskyF64 on the real A_σ of trefoil(64)', () => {
    const c = fixture(64);
    const ref = choleskyF64(c.Asigma, 64);
    const { a, status } = emulateCholesky(c.Asigma, 64, { precision: 'f64', mode: 'faithful' });
    expect(status).toBe(0);
    let worst = 0;
    let scale = 0;
    for (let i = 0; i < 64; i++)
        for (let j = 0; j <= i; j++) {
            worst = Math.max(worst, Math.abs(a[i * 64 + j] - ref[i * 64 + j]));
            scale = Math.max(scale, Math.abs(ref[i * 64 + j]));
        }
    expect(worst / scale).toBeLessThan(1e-12);
});

test('a non-SPD matrix sets status to 1 + the FIRST failing column', () => {
    // finalize encodes the first failing column as 1 + j and never lets a later column
    // overwrite the diagnosis with garbage produced BY the first failure.
    const A = new Float64Array([1, 0, 0, 0, -1, 0, 0, 0, -1]);
    expect(emulateCholesky(A, 3, { precision: 'f64', mode: 'faithful' }).status).toBe(2);
});

test('THE EMULATED K1 PRE-GATE — f32, every gated N, bars read from the artifact', () => {
    // This is the cheapest possible K1: the same quantity, the same A_σ, the same bars, on a
    // CPU f32 emulation of the kernel. A reading here that is not green is the signal to stop
    // BEFORE the WGSL tasks — see the plan's stop-condition block. The assertions below are
    // deliberately only the ones that hold for ANY kernel choice (the kernel ran, the number
    // exists); the BAND is a reading Step 8 acts on, not an assertion, because pre-judging
    // the band here would hard-code an expectation the spec has not settled.
    for (const n of [64, 240, 480, 960]) {
        const row = calRow(n);
        const c = fixture(n);
        // `fast` mode above n=64: it iterates (i,k) directly instead of inverting the thread
        // index tri(960) = 461 280 times per column. The map is a bijection onto {k <= i} and
        // the guard is unchanged — proven by the write census in the first test, which runs
        // in `faithful` mode.
        const { a, status } = emulateCholesky(c.Asigma, n, {
            precision: 'f32',
            mode: n <= 64 ? 'faithful' : 'fast',
        });
        expect(status).toBe(0); // a nonzero status is a transliteration BUG, not a reading
        const k1 = backwardErrorF(a, c.Asigma, n);
        expect(Number.isFinite(k1)).toBe(true);
        const band = k1 >= row.betaMin ? 'RED' : k1 <= row.k1Bar ? 'green' : 'AMBER';
        console.log(
            `[K1 emulated] n=${String(n).padEnd(4)} K1=${k1.toExponential(4)} ` +
                `(${(k1 / row.alpha).toFixed(1)}x alpha) k1Bar=${row.k1Bar.toExponential(4)} ` +
                `eMax=${row.betaMin.toExponential(4)} => ${band}`,
        );
    }
    // MEASURED 5208 ms on this box — bun's DEFAULT per-test timeout is 5000 ms, so without
    // this argument the pre-gate fails on its first run for a reason that has nothing to do
    // with the kernel. `test/gpu/cr.test.ts` sets per-fixture timeouts for the same reason.
}, 300_000);

test('THE EMULATED T4 PRE-GATE — the same f32 factor through the SHIPPED saddle solve, with an f64 control', () => {
    // T4 is the gate that kills. Running it on the emulated factor costs ~3 s and needs no
    // GPU. The f64 CONTROL in the same harness is what makes a red attributable: if the
    // control converges and the f32 factor does not, the factor is the cause and not the
    // scaffolding. The control is the only ASSERTION here, for the same reason as above.
    for (const n of [480, 960]) {
        const c = fixture(n);
        const { a, status } = emulateCholesky(c.Asigma, n, { precision: 'f32', mode: 'fast' });
        expect(status).toBe(0);
        const run = (L: Float64Array) => {
            const traces: IrTrace[] = [];
            // A is the UNSHIFTED matrix: IR converges to the solution of the f64 K, which is
            // what T4 measures against ([DESIGN §4.2]).
            const ext = asExternal(
                makePreparedSaddle(c.A, L, c.sigma, n, (t) => {
                    traces.push(t);
                }).factor(c.C),
            );
            let threw = false;
            try {
                ext.solve(c.gradientRhs);
            } catch (e) {
                // D7: onTrace fires immediately BEFORE the throw, so the curve of the rhs
                // that blew the budget is still captured — the most diagnostic artifact T4
                // can produce.
                if (!(e instanceof ExternalSolveError)) throw e;
                threw = true;
            }
            return { threw, trace: traces[0] };
        };
        const lower = new Float64Array(n * n);
        for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) lower[i * n + j] = a[i * n + j];
        const f32 = run(lower);
        const f64 = run(choleskyF64(c.Asigma, n));
        console.log(
            `[T4 emulated] n=${n} gradient: f32 threw=${f32.threw} refinements=${f32.trace?.refinements} ` +
                `curve=[${f32.trace?.relResiduals.map((x) => x.toExponential(2)).join(', ')}] | ` +
                `f64 control threw=${f64.threw} refinements=${f64.trace?.refinements}`,
        );
        // The control must converge, or nothing above is attributable to the factor.
        expect(f64.threw).toBe(false);
        expect(f64.trace?.refinements).toBeLessThanOrEqual(cal.t4.maxIters);
    }
}, 300_000);
```

- [ ] **Step 2: Run to verify they fail**

Run: `bun test test/gpu/phase2aFixtures.test.ts test/gpu/phase2aMetrics.test.ts test/gpu/choleskyEmulate.test.ts`
Expected: FAIL — none of the four modules exist.

- [ ] **Step 3: Write `bench/gpu/checksum.ts`**

```ts
// bench/gpu/checksum.ts — a fixture checksum. There was no such helper in the repo before
// this file; [DESIGN §5]'s K1 row asks for a recorded fixture checksum and nothing
// implemented one.
// @see docs/superpowers/plans/2026-09-11-webgpu-solver-phase2a-2.md (D4)

/**
 * FNV-1a over the raw IEEE-754 bytes of `x`. Not cryptographic and not meant to be — its
 * job is to notice that a fixture generator changed, and for that a 32-bit digest over the
 * exact bit pattern is enough and is reproducible in any language.
 *
 * Hashing BYTES rather than decimal text is deliberate: a text digest would be blind to a
 * one-ulp change that `toString` rounds away, which is exactly the drift worth catching.
 *
 * **A digest of a `Math.sin`/`Math.cos`/`Math.pow`-derived array is ENGINE-SCOPED** — those
 * are implementation-approximated (ECMA-262 §21.3.2), so the same fixture digests
 * differently under JSC and V8 (measured: D4's table). Record the engine beside any pinned
 * digest, and never cross-compare two engines' digests.
 * @see docs/superpowers/plans/2026-09-11-webgpu-solver-phase2a-2.md (D4)
 */
export function fnv1a32Hex(x: Float64Array): string {
    const bytes = new Uint8Array(x.buffer, x.byteOffset, x.byteLength);
    let h = 0x811c9dc5;
    for (let i = 0; i < bytes.length; i++) {
        h ^= bytes[i];
        // The FNV prime 16777619, via Math.imul so the product stays exact in 32 bits.
        h = Math.imul(h, 0x01000193) >>> 0;
    }
    return h.toString(16).padStart(8, '0');
}
```

- [ ] **Step 4: Write `bench/gpu/phase2aFixtures.ts`**

The fixture list is [DESIGN §5]'s T4 row plus K1's `trefoil(64)`. The rhs construction mirrors `test/gpu/cr.test.ts:241-300` and is pinned against the committed CR artifact by Step 1's test (D5) — **`test/gpu/cr.test.ts` is not modified.**

```ts
// bench/gpu/phase2aFixtures.ts — the fixtures, constraint sets and PRODUCTION right-hand
// sides shared by the emulated pre-gate, the T4 spike and the CR-equivalence test. NO
// `three` import: this must run under `bun test` so the equivalence check against gate CR's
// committed artifact can run with no GPU (plan D5).
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §5 (K1, T4 fixtures)
// @see docs/superpowers/plans/2026-09-11-webgpu-solver-phase2a-2.md (D5)
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
import { calculateDisjointPairs, gradientAnalytical } from '../../src/core/tangentPointEnergy';
import { type Edge, testConfigs, type Vec3 } from '../../src/core/testConfigs';
import { shiftSobolev } from '../../src/gpu/saddle';

/**
 * splitmix32 — the same PRNG gate CR uses, transcribed so the projection rhs is bit-identical
 * between the two gates. NOT `Math.random` (not seedable) and NOT Box–Muller: ECMA-262
 * §21.3.2 leaves `Math.log`/`Math.sin`/`Math.cos` implementation-approximated, so a normal
 * draw would not be reproducible across engines. Everything here — `Math.imul`, XOR, shifts,
 * `/ 2**32` — is exactly specified.
 * @see docs/superpowers/plans/2026-09-04-webgpu-solver-phase2a-1.md (## Decisions, D5, D6)
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

/** Uniform draws on [−1,1). */
function uniforms(count: number, seed: number): number[] {
    const u = splitmix32(seed);
    const out: number[] = [];
    while (out.length < count) out.push(2 * u() - 1);
    return out;
}

/**
 * The displaced iterate the projection rhs is evaluated at: seed 7777, 3n values, consumed
 * coordinate-major per vertex — identical to gate CR's, which is what makes the two gates'
 * projection rows comparable.
 */
function displaced(vertices: Vec3[]): Vec3[] {
    const u = uniforms(vertices.length * 3, 7777);
    return vertices.map((v, i) => [
        v[0] + 1e-3 * u[3 * i],
        v[1] + 1e-3 * u[3 * i + 1],
        v[2] + 1e-3 * u[3 * i + 2],
    ]) as Vec3[];
}

/**
 * K1's fixtures (trefoil 64/240/480/960) unioned with T4's (crossing N=8, trefoil
 * 60/120/240/480/960) — [DESIGN §5]. `crossing` is a FIXED 8-vertex config from the
 * registry, not a generated one.
 */
export const PHASE2A_FIXTURES: { name: string; make: () => { vertices: Vec3[]; edges: Edge[] } }[] =
    [
        {
            name: 'crossing8',
            make: () => {
                const c = testConfigs.find((t) => t.id === 'crossing');
                if (!c) throw new Error('phase2a: the `crossing` test config is missing');
                return c.generate() as { vertices: Vec3[]; edges: Edge[] };
            },
        },
        { name: 'trefoil60', make: () => trefoil(60) },
        { name: 'trefoil64', make: () => trefoil(64) },
        { name: 'trefoil120', make: () => trefoil(120) },
        { name: 'trefoil240', make: () => trefoil(240) },
        { name: 'trefoil480', make: () => trefoil(480) },
        { name: 'trefoil960', make: () => trefoil(960) },
    ];

/** One fixture, assembled: A, its shift, the constraint set, and both production rhs. */
export interface T4Case {
    name: string;
    n: number;
    k: number;
    A: Float64Array;
    Asigma: Float64Array;
    sigma: number;
    C: number[][];
    /** `[b; 0]`, length 3n + k. */
    gradientRhs: number[];
    /** `[0; d]`, d = −Φ(displaced), length 3n + k. */
    projectionRhs: number[];
}

/**
 * Builds one T4 case. `A` is the UNSHIFTED Sobolev matrix — `makePreparedSaddle` measures
 * its residuals against the f64 K, so handing it `Asigma` would certify the solve against
 * the wrong matrix.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1, §4.2
 */
export function buildT4Case(
    name: string,
    vertices: Vec3[],
    edges: Edge[],
    withPin: boolean,
): T4Case {
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
    const k = C.length;
    const { Asigma, sigma } = shiftSobolev(A, n);
    // rhs (a): the gradient shape [b; 0] — what the frozen constrained-gradient solve
    // passes (`src/core/sobolev/gradient.ts`, `flatten(dE)` as rhsTop).
    const b = flatten(gradientAnalytical(vertices, edges, dp, alpha, beta, epsilon));
    // rhs (b): the projection shape [0; d], d = −Φ(γ^q) — what the frozen projection
    // passes (`src/core/sobolev/lineSearch.ts`), with C and A frozen at γ₀ and Φ at the
    // displaced iterate.
    const { phi } = evaluateConstraintSet(set, displaced(vertices), edges);
    return {
        name,
        n,
        k,
        A,
        Asigma,
        sigma,
        C,
        gradientRhs: [...Array.from(b), ...new Array<number>(k).fill(0)],
        projectionRhs: [...new Array<number>(3 * n).fill(0), ...phi.map((v) => -v)],
    };
}
```

- [ ] **Step 5: Write `bench/gpu/phase2aMetrics.ts`**

```ts
// bench/gpu/phase2aMetrics.ts — the two quantities [DESIGN §5] gates on, extracted from the
// browser spike so they can be unit-tested with no GPU. The verdict module makes the
// CLASSIFIER falsifiable; this file is what makes the VALUE it classifies falsifiable.
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §5 (K1, Boot)
// @see docs/superpowers/plans/2026-09-11-webgpu-solver-phase2a-2.md (Task 1)

/**
 * K1's quantity: `‖LLᵀ − A_σ‖_F / ‖A_σ‖_F`, accumulated in f64, reading **only the lower
 * triangle of L**.
 *
 * `(LLᵀ)_ij = Σ_{t ≤ min(i,j)} L_it L_jt` — the upper limit is `min(i, j)` and not `n`
 * because L is zero above the diagonal by construction, so the truncated sum IS the full
 * sum. That also makes the metric blind to the read-back buffer's strict upper triangle,
 * which is garbage by construction ([DESIGN §4.3]) — pinned by a poisoning test.
 * Both numerator and denominator run over the FULL n×n (a Frobenius norm of A_σ, which is
 * symmetric and dense), not over a triangle.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §5
 */
export function backwardErrorF(L: Float64Array, Asigma: Float64Array, n: number): number {
    let num = 0;
    let den = 0;
    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) {
            let s = 0;
            const lim = Math.min(i, j);
            for (let t = 0; t <= lim; t++) s += L[i * n + t] * L[j * n + t];
            const a = Asigma[i * n + j];
            const d = s - a;
            num += d * d;
            den += a * a;
        }
    return Math.sqrt(num) / Math.sqrt(den);
}

/**
 * Boot's FORWARD metric: `max|L_gpu − L_cpu| / max|L_cpu|`, **lower triangle only** — the
 * strict upper triangle of a read-back factor is not defined, so comparing it would measure
 * garbage against zero.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-gate-calibration.md §G
 */
export function forwardMetric(Lgpu: Float64Array, Lcpu: Float64Array, n: number): number {
    let worst = 0;
    let scale = 0;
    for (let i = 0; i < n; i++)
        for (let j = 0; j <= i; j++) {
            worst = Math.max(worst, Math.abs(Lgpu[i * n + j] - Lcpu[i * n + j]));
            scale = Math.max(scale, Math.abs(Lcpu[i * n + j]));
        }
    return worst / scale;
}
```

- [ ] **Step 6: Write `bench/gpu/choleskyEmulate.ts`**

```ts
// bench/gpu/choleskyEmulate.ts — a statement-for-statement CPU transliteration of the three
// WGSL kernels, in f64 (to check the ALGEBRA) or f32 (to measure the PRECISION before any
// GPU exists). This is the emulated K1/T4 pre-gate; it is to the kernel what gate CR is to
// the algebra.
//
// IT IS A HAND TRANSLITERATION AND CAN DRIFT from the WGSL strings it mirrors. Whenever a
// kernel in src/gpu/cholesky.ts changes, change this in the same commit. What bounds the
// drift: the one-bare-`fn` test, the parameter-name test and the triIndex agreement test
// (all test/gpu/cholesky.test.ts), and ultimately K1 on hardware.
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.3
// @see docs/superpowers/plans/2026-09-11-webgpu-solver-phase2a-2.md (Task 1)

/** The WGSL trailing-update index inverse, in f64. See {@link emulateCholesky} on why the
 * seed's precision does not matter here. */
function triIndexEmulated(t: number): { i: number; k: number } {
    let i = Math.floor((Math.sqrt(8 * t + 1) - 1) * 0.5);
    while (i > 0 && (i * (i + 1)) / 2 > t) i -= 1;
    while (((i + 1) * (i + 2)) / 2 <= t) i += 1;
    return { i, k: t - (i * (i + 1)) / 2 };
}

/**
 * Runs the three kernels over a dense buffer, in the plan's dispatch order.
 *
 * **`precision: 'f32'` rounds after EVERY arithmetic operation**, not once at the end:
 * `r(a - r(x*y))` is two roundings, matching the WGSL's two f32 ops, and the upload is
 * rounded before the first kernel because `GpuCholesky.factor` narrows f64 → f32 in its
 * staging loop. A single rounding at the end would understate the error by orders of
 * magnitude — which is exactly the mistake that let an over-optimistic precision model
 * stand. Values are held in a `Float64Array`: an `Math.fround`ed value is exactly
 * representable there, so this is bit-identical to `Float32Array` storage.
 *
 * **Two approximations, both in the CONSERVATIVE direction** (they can only make the
 * emulation look better than hardware, never worse): `Math.sqrt`/`/` here are correctly
 * rounded, while WGSL requires neither; and the trailing-update index seed is computed in
 * f64. The seed's precision is irrelevant by construction — the two correction loops are
 * unbounded and converge to the exact `(i, k)` from any seed (D3) — so this is a
 * simulation convenience, not a semantic change.
 *
 * **`mode`:** `'faithful'` iterates one thread per `I` in `[0, tri)` and inverts the index,
 * exactly as the shader does, and returns a per-entry WRITE CENSUS. `'fast'` iterates
 * `(i, k)` directly — a simulation speedup only, since the thread → `(i, k)` map is a
 * bijection onto `{k ≤ i}` and the guard is unchanged; it returns `writes: null`. Use
 * `'faithful'` at small n to PROVE the coverage, `'fast'` at n ≥ 240 where the faithful
 * loop is `n · tri(n)` ≈ 4.4e8 index inversions.
 */
export function emulateCholesky(
    A0: Float64Array,
    n: number,
    opts: { precision: 'f32' | 'f64'; mode?: 'faithful' | 'fast' },
): { a: Float64Array; status: number; writes: Int32Array | null } {
    const mode = opts.mode ?? 'faithful';
    const r = opts.precision === 'f32' ? Math.fround : (x: number) => x;
    const a = new Float64Array(n * n);
    for (let t = 0; t < n * n; t++) a[t] = r(A0[t]);
    const tri = (n * (n + 1)) / 2;
    const writes = mode === 'faithful' ? new Int32Array(n * n) : null;
    // [counter, status] — the GPU-side meta buffer. `finalize` advances the counter, which
    // is how j reaches every dispatch of a batched node array ([DESIGN §4.3]).
    const meta = [0, 0];
    for (let col = 0; col < n; col++) {
        const j = meta[0];
        // --- cholColFactor: one thread per I in [0, n); A_jj is still the RAW pivot.
        const d0 = a[j * n + j];
        if (d0 > 0) {
            const s = r(Math.sqrt(d0));
            for (let I = 0; I < n; I++) {
                if (I <= j || I >= n) continue;
                a[I * n + j] = r(a[I * n + j] / s);
                if (writes) writes[I * n + j]++;
            }
        }
        // --- cholTrailingUpdate: one thread per (i, k) with j < k <= i < n, diagonal INCLUDED.
        if (mode === 'faithful') {
            for (let I = 0; I < tri; I++) {
                const { i, k } = triIndexEmulated(I);
                if (k <= j || i >= n) continue;
                a[i * n + k] = r(a[i * n + k] - r(a[i * n + j] * a[k * n + j]));
                if (writes) writes[i * n + k]++;
            }
        } else {
            for (let i = j + 1; i < n; i++)
                for (let k = j + 1; k <= i; k++)
                    a[i * n + k] = r(a[i * n + k] - r(a[i * n + j] * a[k * n + j]));
        }
        // --- cholFinalize: ONE thread. First failure wins; `!(d > 0)` also rejects NaN and
        // the upper bound rejects +Inf, which `> 0` admits.
        const d = a[j * n + j];
        const ok = d > 0 && d < 3.4028234e38;
        if (!ok && meta[1] === 0) meta[1] = 1 + j;
        a[j * n + j] = r(Math.sqrt(d));
        if (writes) writes[j * n + j]++;
        meta[0] = j + 1;
    }
    return { a, status: meta[1], writes };
}
```

- [ ] **Step 7: Run the tests**

Run: `bun test test/gpu/phase2aFixtures.test.ts test/gpu/phase2aMetrics.test.ts test/gpu/choleskyEmulate.test.ts`
Expected: **12 PASS** (4 + 3 + 5), ~6 s. The two pre-gate tests print their tables.

**Stop condition (D4):** if the boot-fixture assembly time printed by `phase2aFixtures.test.ts` is ≥ 100 ms, **stop and report** — D4's ruling (generate, don't commit a blob) does not survive it, and that is a plan change.

- [ ] **Step 8: READ THE PRE-GATE — this is the gate, not the assertions**

Copy the two printed tables into the task report verbatim, then apply the plan's stop-condition block to the **emulated** readings:

1. **K1 band at each N.** `green` everywhere ⇒ proceed to Task 2. **`AMBER` or `RED` at any N ⇒ STOP and report** — do not start Task 2. The BLOCKING notice at the top of this plan is exactly this outcome, already measured once; a second confirmation on the implementer's own tree is the evidence the spec amendment needs.
2. **T4 on the emulated factor.** A throw at N ≤ 480 with the f64 control converging is [DESIGN §6]'s kill, measured. **STOP and report.**
3. **A nonzero `status`, or an `f64` control that does NOT converge** ⇒ your transliteration is wrong, not the kernel. Fix it before reading anything.

**Do not adjudicate the amendment here.** Report the numbers; the kernel choice is the human's per the BLOCKING notice.

- [ ] **Step 9: Typecheck, lint, full suite**

Run: `bunx tsc --noEmit && bunx biome check --write bench/gpu/checksum.ts bench/gpu/phase2aFixtures.ts bench/gpu/phase2aMetrics.ts bench/gpu/choleskyEmulate.ts test/gpu/phase2aFixtures.test.ts test/gpu/phase2aMetrics.test.ts test/gpu/choleskyEmulate.test.ts && bun test`
Expected: `tsc` exit 0; `371 pass / 0 fail` (359 + 12).

- [ ] **Step 10: Check knip against the NAMED baseline**

Run: `bunx knip`
Expected: **exactly one new name outside `.superpowers/**`** — `T4Case`, under *Unused exported types*, for the reason tabulated in Global Constraints (it is `buildT4Case`'s return type and no other file imports it). Every other export in this task has a test importer. A second new name is the signal to investigate; do not add anything to the baseline, and do not delete `T4Case` to silence it — the spike and the pre-gate both annotate with it.

- [ ] **Step 11: Commit**

```bash
git add bench/gpu/checksum.ts bench/gpu/phase2aFixtures.ts bench/gpu/phase2aMetrics.ts bench/gpu/choleskyEmulate.ts test/gpu/phase2aFixtures.test.ts test/gpu/phase2aMetrics.test.ts test/gpu/choleskyEmulate.test.ts
git commit -m "$(cat <<'EOF'
feat(bench): phase2a fixtures, gate metrics and the emulated K1/T4 pre-gate

A CPU transliteration of the three Cholesky kernels, in f64 (algebra + write census against
choleskyF64) and f32 (the emulated K1 and T4 readings). Reads the gate in ~6 s with no GPU,
so a precision problem surfaces before any WGSL is written. Fixtures are pinned against gate
CR's committed rhs norms and against the numpy calibration's own sigma / min diag(L).

@see docs/superpowers/plans/2026-09-11-webgpu-solver-phase2a-2.md

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HCkLyorgDoBAwcj4q3uBdN
EOF
)"
```

---

## Task 2: `src/gpu/cholesky.ts` — the pure parts (no GPU, no renderer)

**Files:**
- Create: `src/gpu/cholesky.ts`
- Create: `test/gpu/cholesky.test.ts`

> **BLOCKED until the [DESIGN §4.2]/[§4.3] amendment is resolved** — see the BLOCKING notice at the top. **And bound to Task 1:** `bench/gpu/choleskyEmulate.ts` is a statement-for-statement transliteration of the three WGSL sources below. Whatever the amendment lands on, **the WGSL you write here and the emulation Task 1 measured must be the same algorithm, changed in the same commit.** Step 1 ships the index-agreement test that catches the one drift a machine can catch; the rest is on the author.

**Interfaces:**
- Consumes: `REDUCE_LANES` from `src/gpu/reduce.ts:28` (D1); `emulateCholesky` from `bench/gpu/choleskyEmulate.ts` (Task 1) in the test only.
- Produces:
  ```ts
  export const CHOLESKY_FALLBACK_MAX_N = 4096;
  export function gpuSolveMaxN(renderer: WebGPURenderer): number;
  export function triIndexFromSeed(t: number, seed: number): { i: number; k: number };
  export function triIndex(t: number): { i: number; k: number };
  export function colFactorWgsl(): string;
  export function trailingUpdateWgsl(): string;
  export function finalizeWgsl(): string;
  ```

- [ ] **Step 1: Write the failing test**

Create `test/gpu/cholesky.test.ts`:

```ts
// test/gpu/cholesky.test.ts — the GPU-free half of the f32 Cholesky: the triangular index
// and its ±1 correction, the WGSL contracts three enforces by NAME, and the device-limit
// bound. `factor()` needs hardware and is measured by the spikes, not here.
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.3
// @see docs/superpowers/plans/2026-09-11-webgpu-solver-phase2a-2.md (D1, D3)
import { expect, test } from 'bun:test';
import { emulateCholesky } from '../../bench/gpu/choleskyEmulate';
import {
    CHOLESKY_FALLBACK_MAX_N,
    colFactorWgsl,
    finalizeWgsl,
    gpuSolveMaxN,
    trailingUpdateWgsl,
    triIndex,
    triIndexFromSeed,
} from '../../src/gpu/cholesky';

test('triIndex inverts the lower-triangular packing exactly', () => {
    // Brute force the whole map for n = 256 (32 896 entries): every t must land on the
    // (i, k) the packing t = i(i+1)/2 + k defines, with 0 <= k <= i.
    const n = 256;
    let t = 0;
    for (let i = 0; i < n; i++)
        for (let k = 0; k <= i; k++) {
            const got = triIndex(t);
            expect(got.i).toBe(i);
            expect(got.k).toBe(k);
            t++;
        }
    expect(t).toBe((n * (n + 1)) / 2);
});

test('the ±1 correction recovers the exact row from a WRONG seed — the guard, not the happy path', () => {
    // D3: with a CORRECTLY-ROUNDED root narrowed to f32, the seed is already exact over
    // the whole n=960 domain (measured: worst |seed − i| = 0), so brute-forcing triIndex
    // exercises the correction loops ZERO times and passes with them deleted. WGSL does
    // not guarantee a correctly-rounded f32 sqrt, so the loops must be tested by INJECTING
    // the error they exist to absorb.
    const n = 960;
    for (let i = 0; i < n; i++) {
        const start = (i * (i + 1)) / 2;
        for (const t of [start, start + Math.floor(i / 2), start + i]) {
            const truth = triIndex(t);
            for (const delta of [-1, 0, 1]) {
                const got = triIndexFromSeed(t, truth.i + delta);
                expect(got.i).toBe(truth.i);
                expect(got.k).toBe(truth.k);
            }
        }
    }
});

test('each WGSL source declares exactly one bare fn', () => {
    // WGSLNodeFunction.js matches `^fn name(...)`; a second bare fn is a parse failure,
    // and shared helpers must go in as a `wgsl(src)` include instead
    // (src/gpu/tangentPoint.ts:135-137).
    for (const src of [colFactorWgsl(), trailingUpdateWgsl(), finalizeWgsl()])
        expect(src.match(/^fn\s+\w+\s*\(/gm)?.length).toBe(1);
});

test('the WGSL parameter names are exactly the keys the engine binds', () => {
    // FunctionCallNode.js looks every input up BY NAME and silently substitutes float(0)
    // for a missing one — a console error, not a throw (src/gpu/tangentPoint.ts:124-126).
    // A renamed parameter would therefore factor garbage instead of failing, which is the
    // silent-failure mode this test exists to make loud.
    //
    // SCAN FOR `name:`, DO NOT SPLIT ON `,`. Every parameter here is a
    // `ptr<storage, array<f32>, read_write>`, which CONTAINS two commas — a comma split
    // yields ["I","a","array<f32>","array<u32>","meta","n","read_write>","read_write>","tri"]
    // and fails the very test that guards the silent float(0) substitution. This mirrors
    // three's own parser: WGSLNodeFunction.js:5
    // `propertiesRegexp = /([a-z_0-9]+)\s*:\s*([a-z_0-9]+(?:<[\s\S]+?>)?)/ig`, verified to
    // yield a/meta/n/tri/I on this exact source.
    const params = (src: string) =>
        [
            ...(src.match(/fn\s+\w+\s*\(([\s\S]*?)\)\s*->/)?.[1] ?? '').matchAll(
                /([a-z_0-9]+)\s*:/gi,
            ),
        ]
            .map((m) => m[1])
            .sort();
    expect(params(colFactorWgsl())).toEqual(['I', 'a', 'meta', 'n'].sort());
    expect(params(trailingUpdateWgsl())).toEqual(['I', 'a', 'meta', 'n', 'tri'].sort());
    expect(params(finalizeWgsl())).toEqual(['I', 'a', 'meta', 'n'].sort());
});

test('every kernel carries its OWN out-of-range guard', () => {
    // Thread counts are rounded up to whole workgroups (WebGPUBackend.js:1666) and three's
    // injected early return is conditional on allowEarlyReturns (ComputeNode.js:213), so
    // relying on it is relying on a flag this code does not set.
    expect(colFactorWgsl()).toContain('I >= n');
    expect(trailingUpdateWgsl()).toContain('I >= tri');
    expect(finalizeWgsl()).toContain('I != 0u');
});

test('gpuSolveMaxN reads the DEVICE limit, and falls back when it is unreadable', () => {
    const withLimit = (maxStorageBufferBindingSize: unknown) =>
        gpuSolveMaxN({ backend: { device: { limits: { maxStorageBufferBindingSize } } } } as never);
    // WebGPU's DEFAULT maxStorageBufferBindingSize is 128 MiB, because three requests the
    // device with requiredLimits defaulted to {} (WebGPUBackend.js:243, :93) — so this is
    // the number the dev box actually reports, NOT the adapter's 2 GiB.
    expect(withLimit(134217728)).toBe(5792);
    expect(5792 * 5792 * 4).toBeLessThanOrEqual(134217728);
    expect(5793 * 5793 * 4).toBeGreaterThan(134217728);
    for (const bad of [undefined, null, 0, -1, Number.NaN, Number.POSITIVE_INFINITY, '128'])
        expect(withLimit(bad)).toBe(CHOLESKY_FALLBACK_MAX_N);
    // The adapter is never stored on the backend (WebGPUBackend.js:217 is a local), so a
    // backend with no device at all must fall back rather than throw.
    expect(gpuSolveMaxN({ backend: {} } as never)).toBe(CHOLESKY_FALLBACK_MAX_N);
});

test('the fallback bound is the documented 4096', () => {
    expect(CHOLESKY_FALLBACK_MAX_N).toBe(4096);
});

test('triIndex agrees with the emulated kernel’s index inverse over the whole n=256 domain', () => {
    // ANTI-DRIFT. bench/gpu/choleskyEmulate.ts (Task 1) carries its own copy of this inverse
    // — it has to, because it runs BEFORE this file exists and a bench module may not be
    // imported by src/. Task 1's K1 and T4 readings are only about THIS kernel while the two
    // agree, and this is the one piece of that agreement a machine can check. The rest of the
    // transliteration is bounded only by review and by K1 on hardware.
    const n = 256;
    const A = new Float64Array(n * n);
    // A diagonal SPD matrix whose trailing updates are all no-ops: the emulation's write
    // CENSUS is then the index map itself, entry by entry, with no arithmetic in the way.
    for (let i = 0; i < n; i++) A[i * n + i] = 1;
    const { writes } = emulateCholesky(A, n, { precision: 'f64', mode: 'faithful' });
    if (!writes) throw new Error('faithful mode must return a write census');
    // writes[i][k] counts one trailing update per column j < k, so the census encodes
    // exactly the set triIndex enumerates; walk t and cross-check the (i, k) it names.
    let t = 0;
    for (let i = 0; i < n; i++)
        for (let k = 0; k <= i; k++) {
            const got = triIndex(t);
            expect(got.i).toBe(i);
            expect(got.k).toBe(k);
            expect(writes[i * n + k]).toBe(k < i ? 1 + k : 1 + i);
            t++;
        }
    expect(t).toBe((n * (n + 1)) / 2);
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `bun test test/gpu/cholesky.test.ts`
Expected: FAIL — `src/gpu/cholesky.ts` does not exist.

- [ ] **Step 3: Write the pure parts**

Create `src/gpu/cholesky.ts`:

```ts
// src/gpu/cholesky.ts — the f32 GPU Cholesky of the shifted Sobolev block A_σ.
// Unblocked right-looking: three kernels per column in ONE renderer.compute([...]) batch,
// the column index carried by a GPU-side counter ([DESIGN §4.3] — a uniform shared across
// a batched array would present its LAST written value to every dispatch).
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.3, §5 (K1, Boot)
// @see docs/superpowers/plans/2026-09-11-webgpu-solver-phase2a-2.md
import { REDUCE_LANES } from './reduce';
import type { WebGPURenderer } from 'three/webgpu';

/**
 * Bound used when the device limit is unreadable. NOT a claim about the hardware:
 * [DESIGN §4.3] records that `rg 4096 src/` was empty before this line, i.e. the first
 * draft's "N ≤ 4096 by construction" described a construction that did not exist.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.3
 */
export const CHOLESKY_FALLBACK_MAX_N = 4096;

/**
 * Largest N whose N×N f32 A_σ fits ONE storage-buffer binding.
 *
 * **The DEVICE limit, not the adapter's, and they are not interchangeable** ([DESIGN §4.3]):
 * three requests the device with `requiredLimits: parameters.requiredLimits`
 * (`WebGPUBackend.js:243`) defaulted to `{}` (`:93`), and a limit absent from
 * `requiredLimits` takes WebGPU's DEFAULT (128 MiB) rather than the adapter's supported
 * maximum — so this returns **5792** here, against 23170 had it been sized from a 2 GiB
 * adapter: 4× too large in N, 16× in bytes, which would reinstate exactly the
 * buffer-creation validation error this function exists to avoid. The adapter is also
 * unreachable: it is a LOCAL in `WebGPUBackend.js:217`, never stored, and
 * `src/scene/Viewer.tsx:110` forbids a second `requestAdapter()`.
 *
 * Queried, never asserted — `WebGPUBackend.js:213` requests `featureLevel: 'compatibility'`
 * and `:254` derives `compatibilityMode`, under which defaults can be lower still.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.3
 */
export function gpuSolveMaxN(renderer: WebGPURenderer): number {
    const limits = (
        renderer.backend as unknown as {
            device?: { limits?: { maxStorageBufferBindingSize?: unknown } };
        }
    )?.device?.limits;
    const bytes = limits?.maxStorageBufferBindingSize;
    // `typeof` first: a string '128' would survive Number.isFinite after coercion and
    // silently size the buffer from a unit-less value.
    if (typeof bytes !== 'number' || !Number.isFinite(bytes) || bytes <= 0)
        return CHOLESKY_FALLBACK_MAX_N;
    return Math.floor(Math.sqrt(bytes / 4));
}

/**
 * The ±1 correction of the triangular-index inverse, with the seed row injected.
 *
 * **Why this is separate from {@link triIndex}, and why deleting the loops passes the
 * obvious test:** with a correctly-rounded f64 root narrowed to f32, the seed is ALREADY
 * exact over the whole n=960 domain — measured, worst `|seed − i| = 0` across all 461 280
 * entries — so a brute-force test over `triIndex` executes these loops zero times and
 * stays green with them removed. WGSL does **not** guarantee a correctly-rounded f32
 * `sqrt` (accuracy: WGSL §15.7.4), so a real adapter may hand the shader a root the JS
 * emulation never produces, and the loops are the ±1 guard for that. Taking the seed as a
 * parameter is what lets the test inject the error instead of hoping for it.
 * @see docs/superpowers/plans/2026-09-11-webgpu-solver-phase2a-2.md (D3)
 */
export function triIndexFromSeed(t: number, seed: number): { i: number; k: number } {
    let i = seed;
    // Keep these two loops equivalent to the WGSL below: they are the same correction, and
    // a divergence would make the TS mirror stop testing the shader.
    //
    // EQUIVALENT IN RESULT, NOT IN PATH, at a NEGATIVE seed — do not "fix" either side to
    // match the other. If an adapter's f32 `sqrt(1.0)` returns just under 1.0, the WGSL
    // seed expression evaluates `floor(-ε) = -1.0` and then `u32(-1.0)`. WGSL's float→int
    // VALUE CONVERSION SATURATES (only NaN is indeterminate — gpuweb/gpuweb#3908,
    // https://www.w3.org/TR/WGSL/), so the shader gets `0u` and loop 1 no-ops. This mirror
    // instead gets `i = -1`, skips loop 1 (`i > 0` is false) and reaches the same `(0, 0)`
    // through loop 2. Same answer, different route; both are correct and neither is the
    // other's transliteration at that one input.
    while (i > 0 && (i * (i + 1)) / 2 > t) i -= 1;
    while (((i + 1) * (i + 2)) / 2 <= t) i += 1;
    return { i, k: t - (i * (i + 1)) / 2 };
}

/**
 * Pure mirror of the WGSL trailing-update index inverse: `t ↦ (i, k)` over the packing
 * `t = i(i+1)/2 + k`, `0 ≤ k ≤ i`. Kept in TS so the correction is unit-tested against
 * brute force with no GPU.
 * @see docs/superpowers/plans/2026-09-11-webgpu-solver-phase2a-2.md (D3)
 */
export function triIndex(t: number): { i: number; k: number } {
    return triIndexFromSeed(t, Math.floor((Math.sqrt(8 * t + 1) - 1) * 0.5));
}

/**
 * Column scale: thread per row `i > j` does `A_ij /= √A_jj`. `A_jj` is still the RAW pivot
 * `d_j` here — `finalize` writes `√d_j` AFTER the trailing update, which never touches row
 * or column j. Bailing out on a non-positive pivot stops NaN from being smeared across the
 * trailing block before `finalize` can record the failing column.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.3
 */
export function colFactorWgsl(): string {
    return `fn cholColFactor(
    a: ptr<storage, array<f32>, read_write>,
    meta: ptr<storage, array<u32>, read_write>,
    n: u32,
    I: u32
) -> void {
    let j = (*meta)[0];
    if ( I <= j || I >= n ) { return; }
    let d = (*a)[ j * n + j ];
    if ( !( d > 0.0 ) ) { return; }
    (*a)[ I * n + j ] = (*a)[ I * n + j ] / sqrt( d );
}`;
}

/**
 * Trailing update: thread per `(i, k)` with `j < k ≤ i`, **INCLUDING `i = k`** so the
 * trailing DIAGONAL is updated (it is the next column's pivot). `A_ik −= L_ij·L_kj`; row
 * and column j are never touched.
 *
 * The dispatch size is fixed at the `j = 0` worst case — `n(n+1)/2` threads for EVERY
 * column — because `WebGPUBackend.compute` caches the dispatch size on the NODE OBJECT
 * (`WebGPUBackend.js:1651-1685`) and the same node is repeated n times. `k > j` is the
 * thread's GUARD, not a shrinking dispatch. The resulting 3× waste (N³/2 launches instead
 * of N³/6) is priced into the factor-wall estimate of [CAL §D.1]. **Do not "optimize" it
 * away without the `IndirectStorageBufferAttribute` route ([DESIGN §4.3]), which is
 * pre-registered and deliberately not built in this slice.**
 *
 * **Index range, computed rather than assumed** — three facts, each of which looks like a
 * coincidence and is therefore the kind of line a future agent "cleans up":
 *
 * 1. **No u32 overflow anywhere, including the PRE-DIVISION products.** WGSL evaluates
 *    `i * (i + 1u) / 2u` left to right, so the product exists before the division. Worst
 *    case at n=960 is 920 640; at the `gpuSolveMaxN` ceiling n=5792 it is 33 541 472 —
 *    against u32's 4 294 967 295, a 128× margin. Nothing here needs widening.
 * 2. **`f32(I)` stays exact right up to the device-limit bound, and only just.** `tri` at
 *    n=5792 is 16 776 528 against 2²⁴ = 16 777 216 — **688 apart**; at n=5793 it would not
 *    be exactly representable. The 128 MiB storage-binding limit
 *    (`floor(sqrt(134217728/4)) = 5792`, {@link gpuSolveMaxN}) and the f32 exact-integer
 *    limit therefore coincide to within one n. **That is luck, not design: raising the
 *    binding limit would break this index before it broke the buffer.** Over the served
 *    domain the margin is huge — at n=960, `8·I + 1 ≤ 3 690 233` — which is the second,
 *    independent reason the correction loops are not load-bearing for representation
 *    error (D3 gives the first).
 * 3. **At n=5792 this dispatch needs 262 134 workgroups, over the usual 65 535
 *    `maxComputeWorkgroupsPerDimension`.** `WebGPUBackend.js:1672-1679` splits to 2D
 *    automatically, so it does not fault — but Phase 2 is N ≤ 1000 (7 208 workgroups at
 *    n=960), so that split is **untested ground**, not a known-good path.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.3, §7
 */
export function trailingUpdateWgsl(): string {
    return `fn cholTrailingUpdate(
    a: ptr<storage, array<f32>, read_write>,
    meta: ptr<storage, array<u32>, read_write>,
    n: u32,
    tri: u32,
    I: u32
) -> void {
    if ( I >= tri ) { return; }
    let j = (*meta)[0];
    var i : u32 = u32( floor( ( sqrt( 8.0 * f32( I ) + 1.0 ) - 1.0 ) * 0.5 ) );
    loop { if ( i > 0u && i * ( i + 1u ) / 2u > I ) { i = i - 1u; } else { break; } }
    loop { if ( ( i + 1u ) * ( i + 2u ) / 2u <= I ) { i = i + 1u; } else { break; } }
    let k = I - i * ( i + 1u ) / 2u;
    if ( k <= j || i >= n ) { return; }
    (*a)[ i * n + k ] = (*a)[ i * n + k ] - (*a)[ i * n + j ] * (*a)[ k * n + j ];
}`;
}

/**
 * ONE thread: reads the raw pivot `d_j`, records the FIRST failing column in the status
 * slot as `1 + j` (so a K1 failure is diagnosable without a second readback), writes
 * `A_jj = √d_j`, and advances the column counter.
 *
 * `!(d > 0.0)` already rejects NaN — every comparison with NaN is false — and the upper
 * bound rejects `+∞`, which `> 0` admits. **First failure wins:** a later column would
 * otherwise overwrite the diagnosis with garbage produced BY the first failure.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.3
 */
export function finalizeWgsl(): string {
    return `fn cholFinalize(
    a: ptr<storage, array<f32>, read_write>,
    meta: ptr<storage, array<u32>, read_write>,
    n: u32,
    I: u32
) -> void {
    if ( I != 0u ) { return; }
    let j = (*meta)[0];
    let d = (*a)[ j * n + j ];
    let ok = d > 0.0 && d < 3.4028234e38;
    if ( !ok && (*meta)[1] == 0u ) { (*meta)[1] = 1u + j; }
    (*a)[ j * n + j ] = sqrt( d );
    (*meta)[0] = j + 1u;
}`;
}
```

> **Note on `REDUCE_LANES`:** it is imported here and consumed by the engine in Task 3. If Task 2 is committed alone, Biome/knip will not complain (the import is used by the file's own Task-3 half in the same commit sequence), but if your Task 2 tree has no consumer yet, **move the import to Task 3 rather than adding a lint suppression.**

- [ ] **Step 4: Run the test**

Run: `bun test test/gpu/cholesky.test.ts`
Expected: **8 PASS**.

- [ ] **Step 5: Typecheck, lint, full suite**

Run: `bunx tsc --noEmit && bunx biome check --write src/gpu/cholesky.ts test/gpu/cholesky.test.ts && bun test`
Expected: `tsc` exit 0; `379 pass / 0 fail` (371 + 8).

- [ ] **Step 6: Commit**

```bash
git add src/gpu/cholesky.ts test/gpu/cholesky.test.ts
git commit -m "$(cat <<'EOF'
feat(gpu): cholesky.ts pure parts — triangular index + ±1 correction, WGSL kernels, device-limit bound

@see docs/superpowers/plans/2026-09-11-webgpu-solver-phase2a-2.md

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HCkLyorgDoBAwcj4q3uBdN
EOF
)"
```

---

## Task 3: `GpuCholesky` — the engine (constructor fully unit-tested; `factor()` needs hardware)

**Files:**
- Modify: `src/gpu/cholesky.ts` (append the engine)
- Modify: `test/gpu/cholesky.test.ts` (append the constructor tests)

**Interfaces:**
- Consumes: `colFactorWgsl`, `trailingUpdateWgsl`, `finalizeWgsl`, `gpuSolveMaxN`, `REDUCE_LANES` (Task 2, D1).
- Produces:
  ```ts
  export interface GpuFactorResult {
      L: Float64Array;          // n*n row-major; ONLY the lower triangle is meaningful
      status: number;           // 0 = ok; 1 + j = first non-positive/non-finite pivot at column j
      wallMs: number;           // upload + batch + BOTH readbacks + promotion
      metaReadbackMs: number;   // D6
      factorReadbackMs: number; // D6
  }
  export class GpuCholesky {
      constructor(renderer: WebGPURenderer, n: number);
      readonly nodeCount: number;   // 3n — asserted by the constructor test
      factor(Asigma: Float64Array): Promise<GpuFactorResult>;
      dispose(): void;
  }
  ```

- [ ] **Step 1: Write the failing constructor tests**

Append to `test/gpu/cholesky.test.ts`:

```ts
import { GpuCholesky } from '../../src/gpu/cholesky';

/**
 * A renderer stub. The constructor never touches the device — node building is pure
 * (measured: `wgslFn(src)({…}).compute(…)` builds with no device, plan D8) — so the only
 * field it reads is the storage-binding limit.
 */
const stubRenderer = (maxStorageBufferBindingSize = 134217728) =>
    ({ backend: { device: { limits: { maxStorageBufferBindingSize } } } }) as never;

test('GpuCholesky builds 3n compute nodes from exactly 3 node objects', () => {
    const n = 8;
    const chol = new GpuCholesky(stubRenderer(), n);
    // 3 builds, not 3n shader compiles ([DESIGN §4.3]): the SAME three node objects are
    // repeated, which is why the column index has to come from a GPU-side counter rather
    // than a per-node uniform.
    expect(chol.nodeCount).toBe(3 * n);
    chol.dispose();
});

test('GpuCholesky rejects n outside the device-limit domain before allocating', () => {
    expect(() => new GpuCholesky(stubRenderer(), 0)).toThrow(/n must be >= 1/);
    expect(() => new GpuCholesky(stubRenderer(), -1)).toThrow(/n must be >= 1/);
    // 5793² · 4 B overflows the 128 MiB default binding — the failure mode this guard
    // replaces is a buffer-creation validation error, NOT the graceful per-step CPU
    // fallback [DESIGN §4.5] promises.
    expect(() => new GpuCholesky(stubRenderer(), 5793)).toThrow(/exceeds/);
    const ok = new GpuCholesky(stubRenderer(), 4);
    ok.dispose();
});

test('factor() rejects a mismatched A_σ and a disposed engine without touching the GPU', async () => {
    const chol = new GpuCholesky(stubRenderer(), 4);
    await expect(chol.factor(new Float64Array(15))).rejects.toThrow(/size mismatch/);
    await expect(chol.factor(new Float64Array(17))).rejects.toThrow(/size mismatch/);
    chol.dispose();
    await expect(chol.factor(new Float64Array(16))).rejects.toThrow(/disposed/);
    chol.dispose(); // idempotent
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `bun test test/gpu/cholesky.test.ts`
Expected: FAIL — `GpuCholesky` is not exported.

- [ ] **Step 3: Write the engine**

Append to `src/gpu/cholesky.ts`. Add the value imports at the top of the file:

```ts
import { attributeArray, instancedArray, instanceIndex, uint, wgslFn } from 'three/tsl';
import * as THREE from 'three/webgpu';
import type { StorageBufferNode, WebGPURenderer } from 'three/webgpu';
```

```ts
/** One factorization's outputs. `wallMs` is what P2a's factor-wall bar counts (2a-3). */
export interface GpuFactorResult {
    L: Float64Array;
    status: number;
    wallMs: number;
    metaReadbackMs: number;
    factorReadbackMs: number;
}

/**
 * Owns the A_σ buffer, the 2-slot u32 meta buffer `[counter, status]`, and the 3n compute
 * nodes for ONE n. A new n means a new instance.
 *
 * `factor` is one upload + one batched submit + two readbacks per call (D6 — the factor
 * read is the big one [DESIGN §4.3] describes; the 8-byte meta read is a second
 * round-trip, named here rather than hidden because P2a counts it).
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.3, §4.5, §5
 * @see docs/superpowers/plans/2026-09-11-webgpu-solver-phase2a-2.md (D6)
 */
export class GpuCholesky {
    readonly nodeCount: number;
    private readonly n: number;
    private readonly renderer: WebGPURenderer;
    private readonly a: StorageBufferNode<'float'>;
    private readonly meta: StorageBufferNode<'uint'>;
    private readonly nodes: unknown[];
    private readonly readback: THREE.ReadbackBuffer;
    private readonly staging: Float32Array;
    private disposed = false;

    constructor(renderer: WebGPURenderer, n: number) {
        if (!Number.isInteger(n) || n < 1) throw new Error('GpuCholesky: n must be >= 1');
        const maxN = gpuSolveMaxN(renderer);
        if (n > maxN) throw new Error(`GpuCholesky: n=${n} exceeds GPU_SOLVE_MAX_N=${maxN}`);
        this.renderer = renderer;
        this.n = n;
        // A_σ is BOTH uploaded per step AND GPU-written, so it stays read_write — NO
        // `.toReadOnly()`, unlike Phase 1's uploaded positions buffer. `.toReadOnly()`
        // would bind it to a `ptr<…, read>` and the kernels' `read_write` params would
        // fail to compile (src/gpu/tangentPoint.ts:102-105).
        // @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.3
        this.a = attributeArray(new Float32Array(n * n), 'float');
        this.a.value.setUsage(THREE.DynamicDrawUsage);
        this.meta = instancedArray(2, 'uint');
        // ONE staging buffer for the life of the instance: a per-step n×n allocation is
        // what [JSALG]'s 10.7× `churn` spread measures, and it lands in the neighbouring
        // backsolve/matvec timings too.
        this.staging = new Float32Array(n * n);
        this.readback = new THREE.ReadbackBuffer(n * n * 4);

        const tri = (n * (n + 1)) / 2;
        const colFn = wgslFn(colFactorWgsl());
        const trailFn = wgslFn(trailingUpdateWgsl());
        const finFn = wgslFn(finalizeWgsl());
        // These KEYS must equal the WGSL parameter names: FunctionCallNode.js looks each
        // input up by name and silently substitutes float(0) for a missing one — a console
        // error, not a throw (src/gpu/tangentPoint.ts:124-126). Pinned by the WGSL
        // parameter-name test in test/gpu/cholesky.test.ts.
        const common = { a: this.a, meta: this.meta, n: uint(n), I: instanceIndex };
        const colNode = colFn(common).compute(n, [REDUCE_LANES]);
        // The dispatch is the j=0 worst case for EVERY column — see trailingUpdateWgsl's
        // TSDoc. Shrinking it needs the indirect-dispatch route, which is not built here.
        const trailNode = trailFn({ ...common, tri: uint(tri) }).compute(tri, [REDUCE_LANES]);
        const finNode = finFn(common).compute(1, [1]);
        // The SAME three node objects repeated n times — 3 node builds, not 3n shader
        // compiles. j reaches the shader from meta[0], which `finalize` increments: a
        // uniform shared across a batched array would present its LAST written value to
        // EVERY dispatch (`Renderer.js` calls updateForCompute per node then submits once).
        // The inter-dispatch visibility this needs is guaranteed — each dispatch is its own
        // synchronization scope (gpuweb/gpuweb#4434) — and Phase 1's shipped two-pass
        // reduction already depends on it (src/gpu/reduce.ts:66-67,116).
        // @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.3
        const nodes: unknown[] = [];
        for (let j = 0; j < n; j++) nodes.push(colNode, trailNode, finNode);
        this.nodes = nodes;
        this.nodeCount = nodes.length;
    }

    /**
     * Factors A_σ in place on the GPU and reads the result back. On return the LOWER
     * triangle of `L` holds the factor and the strict upper triangle is zero — the caller
     * promotes the lower triangle to f64 and never reads above the diagonal.
     *
     * `status` is 0 on success and `1 + j` for the first non-positive / non-finite pivot at
     * column j. **A nonzero status means the CPU path serves this step and the factor must
     * NOT reach `makePreparedSaddle`** — never a NaN-laden L into the CPU algebra
     * ([DESIGN §4.3]).
     */
    async factor(Asigma: Float64Array): Promise<GpuFactorResult> {
        if (this.disposed) throw new Error('GpuCholesky: disposed');
        const n = this.n;
        if (Asigma.length !== n * n) throw new Error('GpuCholesky: A_σ size mismatch');
        const t0 = performance.now();
        // Narrow f64 → f32 through the reused staging buffer. `.set` on the attribute array
        // would do the same conversion, but the explicit loop keeps the one allocation-free
        // path visible.
        for (let i = 0; i < n * n; i++) this.staging[i] = Asigma[i];
        (this.a.value.array as Float32Array).set(this.staging);
        this.a.value.needsUpdate = true;
        (this.meta.value.array as Uint32Array).set([0, 0]);
        this.meta.value.needsUpdate = true;
        this.renderer.compute(this.nodes as never); // 3n dispatches, ONE submit ([DESIGN §4.3])

        const tMeta = performance.now();
        // target = null: 8 bytes. A reused ReadbackBuffer would buy nothing here and would
        // collide with the factor read below, which owns the instance's one target (D6).
        const meta = new Uint32Array(await this.renderer.getArrayBufferAsync(this.meta.value));
        const status = meta[1];
        const metaReadbackMs = performance.now() - tMeta;

        const tFactor = performance.now();
        await this.renderer.getArrayBufferAsync(this.a.value, this.readback);
        const flat = new Float32Array(this.readback.buffer as ArrayBuffer);
        const L = new Float64Array(n * n);
        // Copy BEFORE release(): `release()` sets `target.buffer = null` and unmaps the GPU
        // buffer (WebGPUAttributeUtils.js releaseCallback), so `flat` is a view over memory
        // that stops existing the moment we release. Reordering these two lines yields an
        // all-zero / detached factor, silently.
        for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) L[i * n + j] = flat[i * n + j];
        // Mandatory before the NEXT factor(): WebGPUAttributeUtils.js:401 throws
        // 'THREE.WebGPUAttributeUtils: ReadbackBuffer must be released before being used
        // again.' The first call would succeed and the second throw — a per-step failure
        // that a single-shot test misses.
        this.readback.release();
        const factorReadbackMs = performance.now() - tFactor;

        return { L, status, wallMs: performance.now() - t0, metaReadbackMs, factorReadbackMs };
    }

    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        this.readback.dispose();
    }
}
```

- [ ] **Step 4: Run the tests**

Run: `bun test test/gpu/cholesky.test.ts`
Expected: **11 PASS** (8 from Task 2 + 3).

- [ ] **Step 5: Typecheck, lint, full suite**

Run: `bunx tsc --noEmit && bunx biome check --write src/gpu/cholesky.ts test/gpu/cholesky.test.ts && bun test`
Expected: `tsc` exit 0; `382 pass / 0 fail` (379 + 3).

**knip:** this task adds `GpuFactorResult`, which will appear under *Unused exported types* for the reason tabulated in Global Constraints (it is `factor()`'s return type and no other file imports it). That is the second of the five expected names; it is checked at Task 4 Step 6 together with the rest.

- [ ] **Step 6: Commit**

```bash
git add src/gpu/cholesky.ts test/gpu/cholesky.test.ts
git commit -m "$(cat <<'EOF'
feat(gpu): GpuCholesky — 3 kernels/column in one batch, GPU-side column counter, reused ReadbackBuffer

@see docs/superpowers/plans/2026-09-11-webgpu-solver-phase2a-2.md

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HCkLyorgDoBAwcj4q3uBdN
EOF
)"
```

---

## Task 4: the gate logic — every K1/T4/Boot bar read from [K1CAL]

**This is the task a reviewer can actually check line by line: no GPU, no browser, no timing.** Everything that decides a verdict lives here.

**Files:**
- Create: `bench/gpu/phase2aVerdicts.ts`
- Create: `test/gpu/phase2aVerdicts.test.ts`

**Interfaces:**
- Consumes: `SADDLE_IR_TOL`, `SADDLE_IR_MAX_REFINEMENTS` from `src/gpu/saddle.ts`. Nothing else — this module is deliberately GPU-free, `three`-free and fixture-free.
- Produces:
  ```ts
  // bench/gpu/phase2aVerdicts.ts
  export const K1_CALIBRATION_PATH: string;
  export const BOOT_BAR: number;                       // 1e-5 — the ONE literal bar (CAL §G)
  export interface K1CalibrationRow { n: number; alpha: number; betaMin: number; k1Bar: number; bootMetric: number; kappaK: number; }
  export interface K1Calibration { t4: { tol: number; maxIters: number }; rows: K1CalibrationRow[]; }
  export function loadK1Calibration(path?: string | URL): K1Calibration;
  export function k1Row(cal: K1Calibration, n: number): K1CalibrationRow;
  export function k1Verdict(cal: K1Calibration, n: number, value: number): 'green' | 'amber' | 'red';
  export function t4Verdict(cal: K1Calibration, relResidual: number, refinements: number, converged: boolean): 'green' | 'red';
  export function bootVerdict(value: number): 'green' | 'red';
  ```

- [ ] **Step 1: Write the failing tests**

Create `test/gpu/phase2aVerdicts.test.ts`:

```ts
// test/gpu/phase2aVerdicts.test.ts — the gate logic of slice 2a-2, with no GPU in sight.
// Every K1/T4 level is READ from the committed calibration artifact; the only literal bar
// in the slice is Boot's, which [CAL §G] declares a chosen round level.
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §5
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-gate-calibration.md §E, §G
// @see docs/superpowers/plans/2026-09-11-webgpu-solver-phase2a-2.md (D2)
import { expect, test } from 'bun:test';
import { SADDLE_IR_MAX_REFINEMENTS, SADDLE_IR_TOL } from '../../src/gpu/saddle';
import {
    BOOT_BAR,
    bootVerdict,
    K1_CALIBRATION_PATH,
    k1Row,
    k1Verdict,
    loadK1Calibration,
    t4Verdict,
} from '../../bench/gpu/phase2aVerdicts';

// Resolve the artifact against THIS FILE, not the CWD. `loadK1Calibration()`'s default is
// the repo-relative provenance string, which `readFileSync` resolves against the CWD — so
// the bare default only works when `bun test` is invoked from the repo root. The rule and
// the remedy are stated verbatim at `test/gpu/cr.test.ts:50-59`, and `bench/gpu/drive.ts:23`
// does the equivalent with `import.meta.dir`. Passing the URL is also what proves the loader
// is path-parameterised at all (D2).
const cal = loadK1Calibration(new URL(`../../${K1_CALIBRATION_PATH}`, import.meta.url));

test('the calibration artifact loads and carries the five measured rows', () => {
    expect(cal.rows.map((r) => r.n)).toEqual([64, 120, 240, 480, 960]);
    expect(cal.t4.tol).toBe(1e-10);
    expect(cal.t4.maxIters).toBe(4);
    for (const r of cal.rows) {
        // Bisected, per-N, never fitted ([CAL §E]) — so the ordering must hold at every N
        // or a bar has been transcribed rather than read.
        expect(r.alpha).toBeLessThan(r.k1Bar);
        expect(r.k1Bar).toBeLessThan(r.betaMin);
    }
});

test('K1 classifies at the artifact boundaries, inclusive-green and inclusive-red', () => {
    for (const n of [64, 240, 480, 960]) {
        const r = k1Row(cal, n);
        expect(k1Verdict(cal, n, r.alpha)).toBe('green');
        expect(k1Verdict(cal, n, r.k1Bar)).toBe('green'); // green is `<= k1Bar`
        expect(k1Verdict(cal, n, r.betaMin)).toBe('red'); // red is `>= betaMin`
        expect(k1Verdict(cal, n, Math.sqrt(r.k1Bar * r.betaMin))).toBe('amber');
    }
});

test('K1 refuses an N the calibration does not cover, instead of guessing', () => {
    // A silent fallback here would invent a bar for an unmeasured N — the exact class of
    // defect [DESIGN §5]'s "not a [K1CAL] key ⇒ not a bar" rule exists to prevent.
    expect(() => k1Verdict(cal, 500, 1e-7)).toThrow(/no K1 calibration row/);
});

test('K1 reproduces the CORRECTED calibration, not the superseded transcription', () => {
    // The unsplit 2a plan hard-coded K1_RED[960] = 1.45e-7 and K1_GREEN[960] = 7.68e-8 —
    // the pre-2026-09-04 levels ([CAL §E.1]). These two values are where the corrected and
    // the transcribed bars DISAGREE, and the second one is a false kill of a correct
    // kernel. If this test ever goes red, someone has re-transcribed a bar.
    expect(k1Verdict(cal, 960, 1.0e-7)).toBe('green'); // transcription said amber
    expect(k1Verdict(cal, 960, 2.0e-7)).toBe('amber'); // transcription said RED
    expect(k1Row(cal, 960).betaMin).toBeCloseTo(4.5284e-7, 11);
    expect(k1Row(cal, 960).k1Bar).toBeCloseTo(1.3584e-7, 11);
});

test('T4 gates the PURE relative residual against the artifact, and a throw is red', () => {
    expect(t4Verdict(cal, 1e-11, 3, true)).toBe('green');
    expect(t4Verdict(cal, cal.t4.tol, cal.t4.maxIters, true)).toBe('green'); // both inclusive
    expect(t4Verdict(cal, 1e-9, 4, true)).toBe('red'); // residual over bar
    // `refinements = 5` is UNREACHABLE from the shipped solve — `src/gpu/saddle.ts:423`
    // throws at `it >= SADDLE_IR_MAX_REFINEMENTS`, so a converged trace has
    // `refinements ∈ [0, 4]`. Kept anyway: t4Verdict is a bench emitter fed JSON off disk,
    // and a row hand-edited or produced by a future harness must not classify green.
    expect(t4Verdict(cal, 1e-11, 5, true)).toBe('red'); // iterations over budget
    // D7: an exhausted budget surfaces as ExternalSolveError, and the spike records the row
    // with converged=false. A non-converged row is red regardless of its last residual.
    expect(t4Verdict(cal, 1e-11, 3, false)).toBe('red');
    expect(t4Verdict(cal, Number.NaN, 1, true)).toBe('red');
});

test('the SHIPPED IR constants ARE the gate bar — the pin is what keeps them one number', () => {
    // src/gpu/saddle.ts's solve() ENFORCES both before returning (`:414`, `:423`), so while
    // these equal [K1CAL]'s t4 block, `converged === true` already implies
    // `relResidual <= tol` and `refinements <= maxIters` for any row the spike can produce:
    // t4Verdict's two comparisons are unreachable and T4 reduces to "did it throw". That is
    // the SAFE state, and it is what this assertion maintains.
    //
    // DRIFT is the only condition under which those comparisons could ever bite, and it is
    // what makes the gate meaningless rather than redundant: the budget and the bar would
    // become two independent numbers that happen to agree today, and T4 would be measuring
    // the code against itself. Keep the comparisons in t4Verdict (a bench emitter must not
    // trust its input) and keep this pin.
    expect(SADDLE_IR_TOL).toBe(cal.t4.tol);
    expect(SADDLE_IR_MAX_REFINEMENTS).toBe(cal.t4.maxIters);
});

test('Boot is a chosen round level, and the artifact confirms the margin it leaves', () => {
    expect(BOOT_BAR).toBe(1e-5); // [CAL §G] — chosen, NOT a formula over bootMetric
    expect(bootVerdict(9.04e-8)).toBe('green'); // a correct kernel
    expect(bootVerdict(BOOT_BAR)).toBe('red'); // the bar is strict `<`
    expect(bootVerdict(1)).toBe('red'); // a dropped term is O(1), not marginal
    expect(bootVerdict(Number.NaN)).toBe('red');
    // [CAL §G] reports 110.6×. Reproducing it here is a CHECK on the choice, not its source.
    expect(BOOT_BAR / k1Row(cal, 64).bootMetric).toBeCloseTo(110.6, 1);
});

```

- [ ] **Step 2: Run to verify it fails**

Run: `bun test test/gpu/phase2aVerdicts.test.ts`
Expected: FAIL — `bench/gpu/phase2aVerdicts.ts` does not exist.

- [ ] **Step 3: Write `bench/gpu/phase2aVerdicts.ts`**

```ts
// bench/gpu/phase2aVerdicts.ts — every K1/T4/Boot verdict of slice 2a-2, derived from the
// COMMITTED calibration artifact rather than from a level written into this file.
// [DESIGN §5]: "a bar this section ORIGINATES that is not a [K1CAL]/[JSALG]/[BASE] key or
// a formula over such keys is not a bar". Three blind review rounds each found a
// hand-transcribed bar; reading the artifact is what makes that impossible rather than
// merely discouraged.
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §5, §6
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-gate-calibration.md §E, §G
// @see docs/superpowers/plans/2026-09-11-webgpu-solver-phase2a-2.md (D2)
import { readFileSync } from 'node:fs';

/**
 * The operative source of every K1 and T4 level. Committed; [CAL §E] Tier 1.
 *
 * **Repo-relative, for PROVENANCE — it is the string every emitted verdict row names as its
 * `barSource`.** It is also `loadK1Calibration`'s default, and `readFileSync` resolves a
 * bare relative path against the **CWD**, so the default only works when the process runs
 * from the repo root. Callers that can be invoked from elsewhere — every test — must pass
 * `new URL(\`../../${K1_CALIBRATION_PATH}\`, import.meta.url)`. The rule and the remedy are
 * stated verbatim at `test/gpu/cr.test.ts:50-59`; `bench/gpu/drive.ts:23` does the
 * equivalent with `import.meta.dir`.
 */
export const K1_CALIBRATION_PATH = 'bench/results/2026-09-04-gpu-phase2a-k1-calibration.json';

/**
 * Boot's bar — **a CHOSEN ROUND LEVEL, not a formula over `bootMetric`** ([CAL §G]).
 * The artifact supplies the CHECK that the choice is safe (1e-5 / 9.0399e-08 = 110.6×
 * above what a correct kernel measures), which is a different claim from producing it.
 * Deliberately slack: the boot test gates AVAILABILITY against an unknown adapter's
 * summation order, not numerics, and a broken kernel (dropped term, off-by-one column)
 * produces O(1) relative error, not a marginal one.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-gate-calibration.md §G
 */
export const BOOT_BAR = 1e-5;

export interface K1CalibrationRow {
    n: number;
    alpha: number;
    betaMin: number;
    k1Bar: number;
    bootMetric: number;
    kappaK: number;
}

export interface K1Calibration {
    t4: { tol: number; maxIters: number };
    rows: K1CalibrationRow[];
}

/**
 * Reads the calibration at `path`. Path-parameterised so a test can point it at a fixture
 * and prove the classification boundaries; `readFileSync` rather than a static JSON import
 * because `tsconfig.json` sets no `resolveJsonModule` and this slice does not need a
 * tsconfig change (plan D2).
 *
 * Accepts a `URL` so callers can resolve against `import.meta.url` instead of the CWD —
 * see {@link K1_CALIBRATION_PATH}. The default is the repo-relative string and is correct
 * only from the repo root.
 */
export function loadK1Calibration(path: string | URL = K1_CALIBRATION_PATH): K1Calibration {
    return JSON.parse(readFileSync(path, 'utf8')) as K1Calibration;
}

/**
 * The row for `n`, or a throw. **Never a fallback or a nearest-N guess:** inventing a bar
 * for an unmeasured N is precisely the transcription failure this module exists to make
 * impossible, and a silent guess would look like a passing gate.
 */
export function k1Row(cal: K1Calibration, n: number): K1CalibrationRow {
    const row = cal.rows.find((r) => r.n === n);
    if (!row) throw new Error(`no K1 calibration row for n=${n} in ${K1_CALIBRATION_PATH}`);
    return row;
}

/**
 * K1: `‖LLᵀ − A_σ‖_F/‖A_σ‖_F` classified against the artifact's bisected levels.
 *
 * **red is tested FIRST and is `>=`.** `e_max(n) = betaMin(n)` is the lowest backward error
 * at which T4 failure has been OBSERVED over five directions ([CAL §E.2]); taking the
 * minimum rather than the median is what makes the amber band do its job, and testing red
 * first keeps a degenerate artifact (were `k1Bar >= betaMin`) from reporting green.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §5, §6
 */
export function k1Verdict(
    cal: K1Calibration,
    n: number,
    value: number,
): 'green' | 'amber' | 'red' {
    const row = k1Row(cal, n);
    // NaN is not a passing measurement; `!(value < row.betaMin)` would also catch it, but
    // the explicit test says so out loud.
    if (!Number.isFinite(value) || value >= row.betaMin) return 'red';
    return value <= row.k1Bar ? 'green' : 'amber';
}

/**
 * T4: the **pure relative** residual `‖r − Kz‖₂/‖r‖₂` after IR, and the refinement count.
 *
 * `converged === false` means `src/gpu/saddle.ts`'s solve threw `ExternalSolveError` — the
 * budget was exhausted — and that is red whatever the last residual was (plan D7).
 * The residual passed in must NEVER be `solveSaddleFrozen`'s `residual`, which normalises
 * by `max(1, ‖r‖₂)` and is permissive by ~1e4 on the projection rhs ([DESIGN §4.2]).
 */
export function t4Verdict(
    cal: K1Calibration,
    relResidual: number,
    refinements: number,
    converged: boolean,
): 'green' | 'red' {
    if (!converged || !Number.isFinite(relResidual)) return 'red';
    return relResidual <= cal.t4.tol && refinements <= cal.t4.maxIters ? 'green' : 'red';
}

/** Boot: the forward metric against {@link BOOT_BAR}, strict `<` ([CAL §G]). */
export function bootVerdict(value: number): 'green' | 'red' {
    return Number.isFinite(value) && value < BOOT_BAR ? 'green' : 'red';
}
```

- [ ] **Step 4: Run the tests**

Run: `bun test test/gpu/phase2aVerdicts.test.ts`
Expected: **7 PASS.**

- [ ] **Step 5: Typecheck, lint, full suite**

Run: `bunx tsc --noEmit && bunx biome check --write bench/gpu/phase2aVerdicts.ts test/gpu/phase2aVerdicts.test.ts && bun test`
Expected: `tsc` exit 0; `389 pass / 0 fail` (382 + 7).

- [ ] **Step 6: Check knip against the NAMED baseline**

Run: `bunx knip`
Expected: **exactly the FIVE names tabulated in Global Constraints and no sixth** — `T4Case` (Task 1), `GpuFactorResult` (Task 3), and this task's `K1_CALIBRATION_PATH`, `K1Calibration`, `K1CalibrationRow`. All five are types or constants that no *other* file imports; knip flags those regardless of how heavily the declaring file uses them (`SOBOLEV_CONVERGENCE_TOL` and `PhaseSample` are the in-tree proofs). **Do not add them to the baseline and do not delete them to silence knip** — `K1_CALIBRATION_PATH` is the provenance string every emitted verdict row carries, and the two interfaces are the verdict module's public shape. A SIXTH name is the signal to investigate.

- [ ] **Step 7: Commit**

```bash
git add bench/gpu/phase2aVerdicts.ts test/gpu/phase2aVerdicts.test.ts
git commit -m "$(cat <<'EOF'
feat(bench): phase2a gate logic — K1/T4/Boot verdicts read from the committed calibration

Bars are derived from bench/results/2026-09-04-gpu-phase2a-k1-calibration.json at run
time, never transcribed. The only literal bar in the slice is Boot's 1e-5, which CAL §G
declares a chosen round level.

@see docs/superpowers/plans/2026-09-11-webgpu-solver-phase2a-2.md

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HCkLyorgDoBAwcj4q3uBdN
EOF
)"
```

---

## Task 5: the K1 / factor-wall / Boot spike — values only, on the hardware adapter

**Files:**
- Create: `bench/gpu/phase2a.ts`
- Modify: `bench/gpu/spikes.ts` (one import + one `Object.assign`, mirroring `phase1Spikes` at `:1031`)
- Create: `bench/results/<date>-gpu-phase2a-cholesky.json` (measured)

**Interfaces:**
- Consumes: `GpuCholesky` (Task 3); `PHASE2A_FIXTURES`, `buildT4Case`, `fnv1a32Hex`, `backwardErrorF`, `forwardMetric` (Task 1); `choleskyF64` (2a-1).
- Produces: `export const phase2aSpikes: Record<string, () => Promise<Record<string, unknown>>>` with `phase2aCholeskyK1` registered.

**This task runs through `bench/gpu/drive.ts` on the hardware adapter. A software adapter makes every number here INVALID, never PASS/FAIL** (`bench/gpu/README.md`). **No driver, no store, no self-test is touched** — the spike drives `GpuCholesky` directly.

**The spike records VALUES and never classifies** (D2). Its `pass` field means only "the kernel ran cleanly at every N" (status 0, finite L) — `drive.ts` reads `data.pass === false` strictly and would mark the run FAIL. The green/amber/red verdicts are Task 7's, derived from [K1CAL].

- [ ] **Step 1: Write the spike**

Create `bench/gpu/phase2a.ts`:

```ts
// bench/gpu/phase2a.ts — slice 2a-2's browser spikes: K1, the factor wall, the Boot metric
// (this file) and T4. They record VALUES ONLY; every verdict is derived afterwards from the
// committed calibration by bench/gpu/phase2aVerdicts.ts, because a level written into a
// spike is a level that goes stale silently ([DESIGN §5]).
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.3, §5 (K1, T4, Boot)
// @see docs/superpowers/plans/2026-09-11-webgpu-solver-phase2a-2.md
import * as THREE from 'three/webgpu';
import { choleskyF64 } from '../../src/gpu/saddle';
import { GpuCholesky, gpuSolveMaxN } from '../../src/gpu/cholesky';
import { fnv1a32Hex } from './checksum';
import { buildT4Case, PHASE2A_FIXTURES } from './phase2aFixtures';
// The two gate quantities live in their own GPU-free module so they can be unit-tested
// against choleskyF64 and a known perturbation (test/gpu/phase2aMetrics.test.ts). Defining
// them inline here would ship the numbers K1 and Boot are read from entirely untested.
import { backwardErrorF, forwardMetric } from './phase2aMetrics';

export const phase2aSpikes: Record<string, () => Promise<Record<string, unknown>>> = {};

/**
 * Median (upper middle for an even count) — the same helper Phase 1's perf gate uses, for
 * the same reason: one scheduler hiccup in a 5-run sample moves a mean but not a median,
 * and the method is pre-registered as "medians" ([P1 D4]).
 * Sorts a COPY so the raw sample stays in run order in the committed JSON.
 */
const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];

/**
 * K1 + the factor wall + the Boot metric. K1's fixtures are [DESIGN §5]'s: trefoil
 * N=64/240/480/960, `barycenter + totalLength` (the constraint set does not enter A_σ, but
 * the case builder is shared with T4 so the fixture and its checksum are the same object).
 */
phase2aSpikes.phase2aCholeskyK1 = async () => {
    const renderer = new THREE.WebGPURenderer();
    await renderer.init();
    const rows: Record<string, unknown>[] = [];
    let allClean = true;
    for (const n of [64, 240, 480, 960]) {
        const f = PHASE2A_FIXTURES.find((x) => x.name === `trefoil${n}`);
        if (!f) throw new Error(`phase2aCholeskyK1: missing fixture trefoil${n}`);
        const { vertices, edges } = f.make();
        const c = buildT4Case(`trefoil${n}`, vertices, edges, false);
        const chol = new GpuCholesky(renderer, n);
        try {
            // 1 warm-up (pipeline compile) + 5 measured, [P1 D4]. The warm-up is NOT timed.
            const warm = await chol.factor(c.Asigma);
            const walls: number[] = [];
            const metaMs: number[] = [];
            const factorMs: number[] = [];
            let last = warm;
            for (let r = 0; r < 5; r++) {
                last = await chol.factor(c.Asigma);
                walls.push(last.wallMs);
                metaMs.push(last.metaReadbackMs);
                factorMs.push(last.factorReadbackMs);
            }
            const finite = last.L.every((v) => Number.isFinite(v));
            const k1 = backwardErrorF(last.L, c.Asigma, n);
            // Boot's fixture is trefoil(64) only ([DESIGN §5]); at the other N the forward
            // metric is recorded as context, never as a gate reading.
            //
            // COST, so it is not a surprise inside a CDP-evaluated spike: this pays a FULL
            // CPU f64 Cholesky at every N (~1 s at n=960 in Bun) on top of backwardErrorF's
            // ~680 ms, for a number only n=64 is gated on. Kept deliberately — the same
            // forward metric at 240/480/960 is what distinguishes "the kernel is broken"
            // (O(1) at every N) from "the kernel is imprecise" (grows with N), and that
            // distinction is the first thing anyone asks of a red K1. If the spike's wall
            // time becomes the binding constraint, drop the other N — never n=64.
            const boot = forwardMetric(last.L, choleskyF64(c.Asigma, n), n);
            if (last.status !== 0 || !finite || !Number.isFinite(k1)) allClean = false;
            rows.push({
                n,
                fixture: `trefoil${n}`,
                asigmaChecksum: fnv1a32Hex(c.Asigma),
                sigma: c.sigma,
                // THE K1 VALUE. Classified in Task 7 against [K1CAL], not here.
                k1BackwardError: k1,
                bootForwardMetric: boot,
                status: last.status,
                finiteL: finite,
                factorWallMs: walls,
                factorWallP50Ms: median(walls),
                metaReadbackP50Ms: median(metaMs),
                factorReadbackP50Ms: median(factorMs),
            });
        } finally {
            // Per-N dispose: each N owns an n²·4-byte buffer and a ReadbackBuffer, and
            // leaking N=480's would distort N=960's timings (the phase1 gate does the same).
            chol.dispose();
        }
    }
    return {
        gate: 'phase2aCholeskyK1',
        methodology:
            'A_σ from the shipped TS assembly; f32 GPU factor; K1 = ‖LLᵀ − A_σ‖_F/‖A_σ‖_F measured in f64 from the read-back lower triangle; wall = upload + 3n dispatches + both readbacks; 1 warm-up, 5 runs, medians',
        gpuSolveMaxN: gpuSolveMaxN(renderer),
        verdictsAreDerived:
            'green/amber/red are NOT computed here — see bench/gpu/phase2aVerdicts.ts against bench/results/2026-09-04-gpu-phase2a-k1-calibration.json',
        rows,
        // `pass` means the KERNEL RAN CLEANLY, not that the gate is green. drive.ts reads
        // `data.pass === false` strictly and would otherwise mark a legitimately amber run
        // as a harness FAIL.
        pass: allClean,
    };
};
```

- [ ] **Step 2: Register it**

Modify `bench/gpu/spikes.ts` beside the existing `phase1Spikes` merge:

```ts
import { phase2aSpikes } from './phase2a';
// …
// Phase 2a-2 gates (K1, T4, Boot) live in phase2a.ts and share this registry/harness.
// @see docs/superpowers/plans/2026-09-11-webgpu-solver-phase2a-2.md
Object.assign(spikes, phase2aSpikes);
```

- [ ] **Step 3: Typecheck, lint, full suite**

Run: `bunx tsc --noEmit && bunx biome check --write bench/gpu/phase2a.ts bench/gpu/spikes.ts && bun test`
Expected: `tsc` exit 0; `389 pass / 0 fail`, unchanged (this task adds no unit tests — it adds a measurement; the quantities it records are already tested by Task 1).

- [ ] **Step 4: Run the spike on hardware**

Dev server must be up: `bun run dev`. G0a must have a passing result file (`bench/results/*-gpu-g0a*.json`); if not, run `bun bench/gpu/drive.ts adapterInfo --out g0a` first.

Run: `bun bench/gpu/drive.ts phase2aCholeskyK1 --out phase2a-cholesky`
Expected: `bench/results/<date>-gpu-phase2a-cholesky.json`, `"status": "PASS"`, `adapter.vendor = "nvidia"`, `attempts[0].classification = "hardware"`.

**Read these BEFORE continuing:**
- **`status ≠ 0` at any N** ⇒ a kernel bug, not a numerics result. `1 + j` names the failing column. Fix it and re-run; do not record K1 from a run with a nonzero status.
- **`"status": "INVALID"`** ⇒ the adapter was software. The numbers are worthless; re-run with the G0a recipe.
- **`finiteL: false`** ⇒ same class of bug as a nonzero status.
- Do **not** hand-classify K1 here. Task 7 derives every verdict.

- [ ] **Step 5: Commit the spike and its result**

```bash
git add bench/gpu/phase2a.ts bench/gpu/spikes.ts bench/results/*-gpu-phase2a-cholesky.json
git commit -m "$(cat <<'EOF'
bench(gpu): K1 backward error, factor wall and Boot forward metric on the measured f32 Cholesky

Values only — verdicts are derived in bench/gpu/phase2aVerdicts.ts from the committed
calibration artifact.

@see docs/superpowers/plans/2026-09-11-webgpu-solver-phase2a-2.md

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HCkLyorgDoBAwcj4q3uBdN
EOF
)"
```

---

## Task 6: the T4 spike — pure relative residual after IR on the GPU factor

**Files:**
- Modify: `bench/gpu/phase2a.ts` (add `phase2aT4`)
- Create: `bench/results/<date>-gpu-phase2a-t4.json` (measured)

**Interfaces:**
- Consumes: `GpuCholesky` (Task 3); `buildT4Case`, `PHASE2A_FIXTURES` (Task 1); `makePreparedSaddle`, `asExternal`, `IrTrace` from `src/gpu/saddle.ts` and `ExternalSolveError` from `src/core/sobolev/linsolve.ts` (2a-1 — **they are NOT co-located; see Step 1's import block**).
- Produces: the T4 results JSON Task 7 classifies.

- [ ] **Step 1: Write the T4 spike**

**First extend `bench/gpu/phase2a.ts`'s import block.** Task 5's block imports none of what this step uses, and the four symbols are **not co-located** — attributing them all to `src/gpu/saddle.ts` is the mistake that fails the typecheck in Step 2:

```ts
// ExternalSolveError is declared at src/core/sobolev/linsolve.ts:428 and src/gpu/saddle.ts
// does NOT re-export it (saddle.ts:8 imports it for its own throw). The obvious
// `import { …, ExternalSolveError } from '../../src/gpu/saddle'` does not resolve.
import { ExternalSolveError } from '../../src/core/sobolev/linsolve';
// makePreparedSaddle (saddle.ts:338), asExternal (:28), IrTrace (:322).
import { asExternal, makePreparedSaddle } from '../../src/gpu/saddle';
// `import type`, not a value import: tsconfig.json sets `verbatimModuleSyntax: true`.
import type { IrTrace } from '../../src/gpu/saddle';
```

Then append the spike:

```ts
/**
 * T4 — [DESIGN §5]: the **pure relative** residual `‖r − Kz‖₂/‖r‖₂` (f64) after iterative
 * refinement on the GPU factor, plus the refinement count, for both rhs shapes.
 *
 * **The residual is the one `src/gpu/saddle.ts` computes, NOT `solveSaddleFrozen`'s
 * `residual`.** The core's stat divides by `max(1, ‖r‖₂)`, and the projection rhs has
 * `‖r‖₂ ≪ 1` always, so it reads ~1e4× smaller in the PERMISSIVE direction — it "passes
 * 1e-10 while sitting at ~1e-6 relative … exactly the amber failure the throw exists to
 * catch" ([DESIGN §4.2]). The IR loop already normalises by `‖r‖₂` and reports the curve
 * through `onTrace`; this spike reads that and nothing else.
 *
 * Fixtures: crossing N=8, trefoil N=60/120/240/480/960, total mode. perEdge is
 * informational only and is NOT run here ([DESIGN §5]'s T4 row; [G6] already routes
 * perEdge N ≥ 480 to CPU f64).
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1 step 4, §4.2, §5
 * @see docs/superpowers/plans/2026-09-11-webgpu-solver-phase2a-2.md (D7)
 */
phase2aSpikes.phase2aT4 = async () => {
    const renderer = new THREE.WebGPURenderer();
    await renderer.init();
    const rows: Record<string, unknown>[] = [];
    let allClean = true;
    for (const name of ['crossing8', 'trefoil60', 'trefoil120', 'trefoil240', 'trefoil480', 'trefoil960']) {
        const f = PHASE2A_FIXTURES.find((x) => x.name === name);
        if (!f) throw new Error(`phase2aT4: missing fixture ${name}`);
        const { vertices, edges } = f.make();
        const c = buildT4Case(name, vertices, edges, false);
        const chol = new GpuCholesky(renderer, c.n);
        try {
            const { L, status, wallMs } = await chol.factor(c.Asigma);
            if (status !== 0) {
                // A nonzero status means the factor is discarded and the CPU path serves
                // the step ([DESIGN §4.3]) — NEVER a NaN-laden L into the CPU algebra. Record
                // it and move on; this is a kernel bug to fix, not a T4 reading.
                allClean = false;
                rows.push({ name, n: c.n, k: c.k, status, factorWallMs: wallMs, skipped: 'nonzero status' });
                continue;
            }
            const traces: IrTrace[] = [];
            // A is the UNSHIFTED matrix: IR converges to the solution of the f64 K, which
            // is what T4 measures against ([DESIGN §4.2]).
            const ext = asExternal(
                makePreparedSaddle(c.A, L, c.sigma, c.n, (t) => {
                    traces.push(t);
                }).factor(c.C),
            );
            for (const [shape, rhs] of [
                ['gradient', c.gradientRhs],
                ['projection', c.projectionRhs],
            ] as const) {
                const before = traces.length;
                let converged = true;
                let message: string | undefined;
                try {
                    ext.solve(rhs);
                } catch (e) {
                    // D7: `onTrace` fires IMMEDIATELY BEFORE the throw (src/gpu/saddle.ts),
                    // so the curve of the rhs that blew the budget is still captured — and
                    // it is the most diagnostic artifact T4 can produce. Catch and CONTINUE:
                    // letting it propagate would abort every later fixture.
                    if (!(e instanceof ExternalSolveError)) throw e;
                    converged = false;
                    message = e.message;
                    allClean = false;
                }
                const trace = traces[before];
                rows.push({
                    name,
                    n: c.n,
                    k: c.k,
                    rhsShape: shape,
                    converged,
                    error: message,
                    // relResiduals[0] is the residual of z₀, BEFORE any correction, so
                    // relResiduals.length === refinements + 1. T4's "iterations" is
                    // `refinements`. Off-by-one here understates the budget consumed.
                    refinements: trace?.refinements ?? null,
                    relResidual: trace ? trace.relResiduals[trace.relResiduals.length - 1] : null,
                    relResiduals: trace?.relResiduals ?? null,
                    factorWallMs: wallMs,
                });
            }
        } finally {
            chol.dispose();
        }
    }
    return {
        gate: 'phase2aT4',
        methodology:
            'f32 GPU factor promoted to f64; Schur/Woodbury/IR in CPU f64; residual is the PURE relative ‖r − Kz‖₂/‖r‖₂ reported by src/gpu/saddle.ts, never solveSaddleFrozen’s max(1,‖r‖₂)-normalised stat; total mode, no pin',
        verdictsAreDerived:
            'green/red are NOT computed here — see bench/gpu/phase2aVerdicts.ts against the committed calibration t4 block',
        rows,
        pass: allClean,
    };
};
```

- [ ] **Step 2: Typecheck, lint, full suite**

Run: `bunx tsc --noEmit && bunx biome check --write bench/gpu/phase2a.ts && bun test`
Expected: `tsc` exit 0; `389 pass / 0 fail`, unchanged.

- [ ] **Step 3: Run it on hardware**

Run: `bun bench/gpu/drive.ts phase2aT4 --out phase2a-t4`
Expected: `bench/results/<date>-gpu-phase2a-t4.json`, hardware adapter.

**This run is slow at N=960** — the CPU f64 setup is 13 N-backsolves plus 3 per IR step, ~106 ms each at N=960 ([DESIGN §7]). Budget minutes, not seconds.

**There is no CDP evaluation timeout to trip.** `CDPClient.send` (`bench/gpu/drive.ts:172-181`) has none and `evalSpike` (`:214-226`) issues an unbounded `Runtime.evaluate`; only `waitForTarget` and `waitForRunSpikeReady` are bounded. So a long T4 run **HANGS rather than timing out** — if the command stops producing output, that is the failure mode, and you kill it and report the wall time. The *recoverable* failure the harness models is a WebSocket close, which `failPending` (`:132`) turns into a rejection. Either way it is a harness limit to report, not a gate failure.

**Expected shape of a correct result, so a surprise is legible** ([CAL §E.5]): `refinements` should be ≤ 3 at N ≤ 480 with at least one spare, and **4 with zero spare at N=960** — a green K1 at 960 is predicted to consume the entire budget. `converged: false` anywhere is a red row.

- [ ] **Step 4: Commit**

```bash
git add bench/gpu/phase2a.ts bench/results/*-gpu-phase2a-t4.json
git commit -m "$(cat <<'EOF'
bench(gpu): T4 — pure relative residual and IR curve after refinement on the f32 GPU factor

@see docs/superpowers/plans/2026-09-11-webgpu-solver-phase2a-2.md

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HCkLyorgDoBAwcj4q3uBdN
EOF
)"
```

---

## Task 7: derive the verdicts, read the gates, write the report, hand off

**Files:**
- Modify: `bench/gpu/phase2aVerdicts.ts` (add the CLI entry point)
- Create: `test/gpu/phase2aVerdicts.test.ts` additions (the emitter's shape)
- Create: `bench/results/<date>-gpu-phase2a-verdicts.json` (derived)
- Modify: `bench/gpu/README.md` (add "Phase 2a-2 gate report")

**Why an emitter and not a hand-written table:** the same reason [CAL §J.2] gives for `G5-confirm` — "a level that only exists as prose is a level that goes stale silently, three times in a row." Every verdict in the README must be readable out of a committed JSON that names the [K1CAL] key it came from.

- [ ] **Step 1: Write the failing emitter test**

Append to `test/gpu/phase2aVerdicts.test.ts`:

```ts
import { deriveVerdicts } from '../../bench/gpu/phase2aVerdicts';

test('deriveVerdicts reports every verdict beside the artifact key it was read from', () => {
    const out = deriveVerdicts(
        cal,
        {
            rows: [
                // The n=64 row is REQUIRED, not decorative: Boot reads trefoil(64) only
                // ([DESIGN §5]) and deriveVerdicts throws when no n=64 row was measured.
                // Omitting it makes this test throw before it reaches a single assertion.
                { n: 64, k1BackwardError: 1e-8, bootForwardMetric: 9.04e-8, status: 0, factorWallP50Ms: 5 },
                { n: 480, k1BackwardError: 1e-8, bootForwardMetric: 1e-7, status: 0, factorWallP50Ms: 40 },
                { n: 960, k1BackwardError: 2e-7, bootForwardMetric: 1e-5, status: 0, factorWallP50Ms: 90 },
            ],
        },
        { rows: [{ name: 'trefoil480', n: 480, rhsShape: 'gradient', converged: true, refinements: 3, relResidual: 1e-11 }] },
    );
    const k960 = out.k1.find((r) => r.n === 960);
    expect(k960?.verdict).toBe('amber');
    // The bar TRAVELS WITH THE VERDICT: a reader must never have to trust a number that
    // is only in prose.
    expect(k960?.k1Bar).toBe(k1Row(cal, 960).k1Bar);
    expect(k960?.eMax).toBe(k1Row(cal, 960).betaMin);
    expect(k960?.barSource).toContain('k1Bar');
    expect(out.k1.find((r) => r.n === 480)?.verdict).toBe('green');
    expect(out.t4[0].verdict).toBe('green');
    // Boot reads trefoil(64) ONLY ([DESIGN §5]); the metric at other N is context — the
    // n=960 row above carries a bootForwardMetric AT the bar and must not be the one read.
    expect(out.boot.n).toBe(64);
    expect(out.boot.value).toBe(9.04e-8);
    expect(out.boot.verdict).toBe('green');
});

test('deriveVerdicts refuses to classify a run whose kernel did not run cleanly', () => {
    // A nonzero status is a kernel BUG, not a numerics result ([DESIGN §6]). Classifying it
    // would publish a K1 value measured from a factor the kernel itself reported broken.
    expect(() =>
        deriveVerdicts(
            cal,
            { rows: [{ n: 480, k1BackwardError: 1e-8, bootForwardMetric: 1e-7, status: 481, factorWallP50Ms: 40 }] },
            { rows: [] },
        ),
    ).toThrow(/status/);
});
```

- [ ] **Step 2: Add the emitter and its CLI**

Append to `bench/gpu/phase2aVerdicts.ts`:

```ts
/**
 * Derives every 2a-2 verdict from the measured spike outputs and the committed calibration.
 * Each row carries the LEVEL it was judged against and the artifact key that level came
 * from, so the README can be read without trusting prose.
 *
 * Throws on a nonzero kernel status: that is a bug, not a measurement ([DESIGN §6]), and
 * classifying it would publish a K1 value taken from a factor the kernel reported broken.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §5, §6
 */
export function deriveVerdicts(
    cal: K1Calibration,
    cholesky: { rows: Record<string, unknown>[] },
    t4: { rows: Record<string, unknown>[] },
) {
    for (const r of cholesky.rows)
        if (r.status !== 0)
            throw new Error(`phase2a: kernel status ${String(r.status)} at n=${String(r.n)} — fix the kernel before reading K1 ([DESIGN §6]); 1 + j names the failing column`);
    const k1 = cholesky.rows.map((r) => {
        const n = r.n as number;
        const value = r.k1BackwardError as number;
        const row = k1Row(cal, n);
        return {
            n,
            value,
            verdict: k1Verdict(cal, n, value),
            k1Bar: row.k1Bar,
            eMax: row.betaMin,
            alpha: row.alpha,
            factorWallP50Ms: r.factorWallP50Ms,
            barSource: `${K1_CALIBRATION_PATH} rows[n=${n}].k1Bar / .betaMin`,
        };
    });
    const bootRow = cholesky.rows.find((r) => r.n === 64);
    if (!bootRow) throw new Error('phase2a: Boot reads trefoil(64) and no n=64 row was measured');
    const boot = {
        n: 64,
        value: bootRow.bootForwardMetric as number,
        verdict: bootVerdict(bootRow.bootForwardMetric as number),
        bar: BOOT_BAR,
        barSource: 'CAL §G — a CHOSEN ROUND LEVEL, checked against rows[n=64].bootMetric, not derived from it',
        marginVsCorrectKernel: BOOT_BAR / k1Row(cal, 64).bootMetric,
    };
    const t4rows = t4.rows
        .filter((r) => r.skipped === undefined)
        .map((r) => ({
            name: r.name,
            n: r.n,
            rhsShape: r.rhsShape,
            relResidual: r.relResidual,
            refinements: r.refinements,
            converged: r.converged,
            verdict: t4Verdict(
                cal,
                r.relResidual as number,
                r.refinements as number,
                r.converged as boolean,
            ),
            tol: cal.t4.tol,
            maxIters: cal.t4.maxIters,
            barSource: `${K1_CALIBRATION_PATH} t4.tol / t4.maxIters`,
        }));
    return { k1, t4: t4rows, boot };
}
```

Add a CLI tail guarded so importing the module in a test does not run it:

```ts
// CLI: bun bench/gpu/phase2aVerdicts.ts <cholesky.json> <t4.json> [--out <path>]
if (import.meta.main) {
    const [choleskyPath, t4Path] = process.argv.slice(2);
    if (!choleskyPath || !t4Path)
        throw new Error('usage: bun bench/gpu/phase2aVerdicts.ts <cholesky.json> <t4.json>');
    const read = (p: string) =>
        (JSON.parse(readFileSync(p, 'utf8')) as { data: { rows: Record<string, unknown>[] } }).data;
    const cal = loadK1Calibration();
    const out = {
        gate: 'phase2a-2-verdicts',
        derivedFrom: { calibration: K1_CALIBRATION_PATH, cholesky: choleskyPath, t4: t4Path },
        ...deriveVerdicts(cal, read(choleskyPath), read(t4Path)),
    };
    const date = new Date().toISOString().slice(0, 10);
    // Resolved against THIS MODULE, not the CWD — `bench/gpu/drive.ts:23` does the same with
    // `join(import.meta.dir, '..', 'results')`. A bare relative path would silently write the
    // gate artifact into whatever directory the emitter happened to be invoked from.
    writeFileSync(
        new URL(`../results/${date}-gpu-phase2a-verdicts.json`, import.meta.url),
        `${JSON.stringify(out, null, 2)}\n`,
    );
    console.log(JSON.stringify(out, null, 2));
}
```

(Add `writeFileSync` to the `node:fs` import.)

- [ ] **Step 2a: Run to verify it fails**

Between Steps 1 and 2, run: `bun test test/gpu/phase2aVerdicts.test.ts`
Expected: FAIL — `deriveVerdicts` is not exported. Tasks 1–4 all do this; do not skip it here because the file already exists.

- [ ] **Step 3: Run the tests, typecheck, lint, full suite**

Run: `bun test test/gpu/phase2aVerdicts.test.ts && bunx tsc --noEmit && bunx biome check --write bench/gpu/phase2aVerdicts.ts test/gpu/phase2aVerdicts.test.ts && bun test`
Expected: **9 PASS** in that file (7 from Task 4 + 2); `391 pass / 0 fail` overall (389 + 2) across **45 files** — the plan's end state.

- [ ] **Step 4: Derive the verdicts**

Run: `bun bench/gpu/phase2aVerdicts.ts bench/results/<date>-gpu-phase2a-cholesky.json bench/results/<date>-gpu-phase2a-t4.json`
Expected: `bench/results/<date>-gpu-phase2a-verdicts.json`.

- [ ] **Step 5: READ THE GATES — apply [DESIGN §6] literally**

Work through these in order and **write the outcome into the task report**:

1. **K1 at N=480.** `red` (`≥ rows[n=480].betaMin = 3.2893e-06`) ⇒ **STOP.** Variant (a) is not confirmed. Record the measured error, write it into the README, and report that the milestone routes to **building variant (b)** — not to kernels-only ([DESIGN §6] corrects that premise explicitly; on the measured `cgIters` (b) is viable).
2. **K1 at the other N.** `amber` is not a kill — record the value and note that T4 at that N decides whether the N is served. **`green` at N=960 is the PREDICTED outcome** ([CAL §E.5]); amber there is not a surprise either, but it inverts a prediction and must be called out.
3. **T4.** `red` at N ≤ 480 ⇒ **STOP**, a kill. `red` at a larger N narrows the served N ([G6] semantics) — record it, and note that 2a-3's routing predicate must carry the narrowing.
4. **Boot.** `red` ⇒ the branch does not merge.
5. **Factor wall.** Record `factorWallP50Ms` at N=960 beside the **100 ms declared policy target** ([DESIGN §5] P2a row). **This is NOT a 2a-2 gate** — P2a is 2a-3's — but [CAL §H.5] predicts 54 ms at the optimistic corner and **101 ms at the pessimistic one, 1.1 ms ABOVE the target**, so the reading is genuinely informative now and 2a-3 should not be surprised by it. Report it as informational.

- [ ] **Step 6: Write the gate report**

Add a `## Phase 2a-2 gate report` section to `bench/gpu/README.md`, in the format of the existing "Phase 1 gate report" (`:413`): a provenance paragraph naming every committed result file and its `gitShaShort`, then one table row per gate with **fixture / method | measured value + the JSON field it is read from | threshold + the [K1CAL] key it is read from | consequence**.

Rules that make the report honest, all of them already precedents in the Phase 1 report:
- Every number names the JSON file and field it comes from. No number appears only in prose.
- Every threshold names its artifact key, except Boot's, which is labelled **"chosen round level ([CAL §G])"**.
- If the result files were re-run, say which sha they carry and why (the Phase 1 report's provenance paragraph is the model — `drive.ts` records the PARENT sha for a JSON committed in the same commit as its run).
- State the served N explicitly if T4 narrowed it.

- [ ] **Step 7: Commit**

```bash
git add bench/gpu/phase2aVerdicts.ts test/gpu/phase2aVerdicts.test.ts bench/results/*-gpu-phase2a-verdicts.json bench/gpu/README.md
git commit -m "$(cat <<'EOF'
bench(gpu): derive the 2a-2 gate verdicts from the committed calibration + write the report

@see docs/superpowers/plans/2026-09-11-webgpu-solver-phase2a-2.md

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HCkLyorgDoBAwcj4q3uBdN
EOF
)"
```

- [ ] **Step 8: Report to the orchestrator — including the handoff 2a-3 depends on**

Read every number out of the emitted JSONs; compute none by hand.

1. **K1** per N: value, verdict, and the slack to `k1Bar` / `eMax`.
2. **T4** per fixture and rhs shape: residual, refinements, verdict; and whether N=960 consumed the whole budget as [CAL §E.5] predicts.
3. **Boot**: metric, verdict, margin.
4. **Factor wall** p50 at each N, with the meta/factor readback split (D6), against the 100 ms declared target — informational, 2a-3 gates it.
5. **The boot-fixture assembly time and checksum** printed by Task 1 (D4), with the engine and version they were taken under.
6. **Owed to 2a-3, carried forward from this plan's rulings — state all four:**
   - `G5-confirm` and the [CAL §J.2] emitter are 2a-3's **FIRST** tasks, before any driver / store / self-test integration ([DESIGN §5]'s ordering).
   - [CAL §J.1]'s five browser CPU f64 full-step p50s at N=480/960 are still unmeasured; **no P2a verdict may be read until they are**, along with the `R₂` attribution.
   - **The CR headroom debt inherited from 2a-1 is still unpaid** ([2A1] Global Constraints): `max(1e-12, 10·κ₂·u_f64)` is insufficient for a GENERIC rhs at n=960 (measured effective constant ~15·κ·u; two of 156 rows exceed the bar with a CORRECT f64 factor, and IR does not fire). The three gated rhs are unaffected (worst gated slack 3.34×). **Any future change to CR's rhs set must revisit the headroom first.** This is a [CAL §F] spec fix, not a plan section.
   - `SimStore.gpuSolveAvailable` needs **all four** `setGpuAvailable` sites mirrored — `src/scene/Viewer.tsx` 486, 636, 655, 669 — **including the `poisoned` latch** at 636 that stops a late self-test result from re-arming a poisoned device ([DESIGN §7]).
   - **[CAL §J.4] is the one §J debt with no owner anywhere, and it is 2a-3's** — re-measure [JSALG]'s three constants with `loadavg` recorded, **in the same session as P2a**, which is exactly why it cannot be paid earlier or later. `grep -rn 'J\.4' docs/superpowers/` matches only its own heading today; this line is the assignment.

7. **The emulated pre-gate's readings from Task 1**, beside the hardware readings, with the ratio. A hardware K1 far from the emulated one at the same N means either the transliteration drifted or the adapter's `sqrt`/`/` are materially worse than correctly-rounded — both are findings 2a-3 needs.

---

## Self-review

**1. Spec coverage.** [DESIGN §4.3] → Tasks 2–3 (every bullet: layout, in-place factor, the `read_write` A_σ, the reused `ReadbackBuffer`, the device-limit bound, the three kernels and their order, the GPU-side counter, the fixed worst-case dispatch, the status flag, one bare `fn` per `wgslFn`) plus Task 1's CPU transliteration of the same. [DESIGN §4.2] → the T4 residual definition (Global Constraints + Task 6) and the pre-registered IR-budget expectation (gate section + Task 6 Step 3) — **and the BLOCKING notice, because round-1 review falsified §4.2's compensated-arithmetic bullet for this kernel; that is a spec amendment this plan does not make.** [DESIGN §5] K1/T4/Boot rows → Tasks 1 and 4–7, each level read from [K1CAL]. [DESIGN §6] → the stop-condition block, Task 1 Step 8 and Task 7 Step 5 — **with one gap made explicit rather than papered over: §6 has no branch for K1 red at an N other than 480.** [DESIGN §7]'s risks: fixed dispatch (Task 2 TSDoc), readback size (Task 5 records it), the two capability bits (deferred to 2a-3 with the four sites named). **Deliberately out of scope, each with an owner in the deferral table:** §4.4 (landed in 2a-1), §4.5, §4.6, §4.7, T5, P2a, `G5-confirm`; and [DESIGN §5]'s informational perEdge T4 row, which is not run and not owed.

**2. Placeholder scan.** **None.** The `'<checksum>'` placeholder and its "FILL IN from the first green run" instruction are gone — the digest is measured (`4318905e`, Bun/JSC) and written in, with D4's engine caveat beside it. `<date>` in results paths is `drive.ts`'s own `new Date().toISOString().slice(0,10)`, not a TBD. No "add appropriate error handling", no "similar to Task N", no undefined types.

**3. Type consistency.** `GpuFactorResult` is declared in Task 3 and consumed in Tasks 5–6 by the same field names (`L`, `status`, `wallMs`, `metaReadbackMs`, `factorReadbackMs`). `T4Case` is declared in Task 1 and consumed in Tasks 1, 5 and 6 (`Asigma`, `sigma`, `A`, `C`, `n`, `k`, `gradientRhs`, `projectionRhs`). `backwardErrorF`/`forwardMetric` are declared once in Task 1 and consumed by Task 1's tests and Task 5's spike — **not redefined inline**, which is what makes them testable. `k1Verdict`/`t4Verdict`/`bootVerdict` keep one signature across Tasks 4 and 7. `loadK1Calibration` takes `string | URL` so every caller can resolve against `import.meta.url`. `triIndexFromSeed` is used by `triIndex` and by the Task 2 test only. `nodeCount` is declared in Task 3's interface block and asserted in its test.

**4. Expected test counts, end to end.** Baseline **359** across 40 files. Task 1 **+12** (4 fixtures + 3 metrics + 5 emulation) → 371, 43 files. Task 2 **+8** → 379, 44 files. Task 3 **+3** → 382. Task 4 **+7** → 389, 45 files. Tasks 5 and 6 **+0** (measurements, not unit tests) → 389. Task 7 **+2** → **391 across 45 files**. Per-file at the end: `test/gpu/cholesky.test.ts` 11, `test/gpu/phase2aVerdicts.test.ts` 9, `test/gpu/phase2aFixtures.test.ts` 4, `test/gpu/choleskyEmulate.test.ts` 5, `test/gpu/phase2aMetrics.test.ts` 3.

**5. Known residual risks, stated rather than hidden.**
- **The WGSL kernels are still never executed until Task 5 runs on hardware** — but the gap is now bounded rather than open. Task 1's CPU transliteration checks the *algorithm* (exact against `choleskyF64`, a per-entry write census proving no entry is updated twice or zero times, the status encoding) and *measures its f32 precision*, with no GPU. What remains untested is **the transliteration itself** — that `bench/gpu/choleskyEmulate.ts` still says what the WGSL strings say. That is narrower than "the whole algorithm" and it is not zero: the index-agreement test catches index drift, the parameter-name and one-bare-`fn` tests catch binding drift, and nothing catches a changed arithmetic statement except review and K1. **Say this, do not let Task 1 be read as "the kernel is verified".**
- **Two of the emulation's approximations run in the OPTIMISTIC direction**: `Math.sqrt` and `/` are correctly rounded in JS while WGSL requires neither. A hardware K1 *worse* than the emulated one is therefore expected behaviour, not a contradiction; a hardware K1 *better* is the surprise worth chasing.
- `deriveVerdicts`' parameter types are `Record<string, unknown>` rows with casts at the point of use. A stricter schema would be better; it is not worth a Zod dependency for a bench emitter, and the two emitter tests pin the fields that matter.
- **The largest risk in this plan is not in this plan.** The kernel choice is unresolved (BLOCKING notice). Tasks 2, 3, 5 and 6 are written against [DESIGN §4.3]'s kernel and will need their WGSL and Task 1's emulation changed together if the amendment moves it. Nothing else in the plan depends on which kernel wins.
