# Orchestrator handoff — WebGPU solver Phase 1 implementation

You are the orchestrator for implementing an already-finalized, already-reviewed plan. Do NOT re-plan, re-spec, or re-review the plan. Read `CLAUDE.md` first (its "## Review loop", "## Models", "## Inline-fix gate" and "## Subagent briefing" sections govern everything below), then the plan, then the spec.

## Artifacts
- Branch: `feat/webgpu-phase1` (off `main` @ 3ea1b3c). Two commits on it: `b926ac6` = the plan (one commit per plan doc — never stack follow-ups on it), `4e30546` = review-loop research doc.
- Plan: `docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md` — Tasks 1, 2, 3a, 3b, 3c, 4, 5, 6, 7a, 7b, 8. Decisions D1–D5 and the "Global Constraints" / "Stop conditions" sections are binding.
- Spec: `docs/superpowers/specs/2026-08-13-webgpu-solver-design.md` (Task 5 amends §1 (iii), §2.3, §6 in its own commit — the plan gives the exact wording).
- Review ledger (untracked, do not commit): `docs/superpowers/plans/phase1-review-ledger/` — r1 (REJECT, 13 findings), r1 fix diff, r2 (APPROVE, all 13 ADDRESSED, no Important+ open). Open nits parked there: N4 (seam-diff ceiling is a judgment call, set to 40), N5 (D2 "divide by zero" wording, inert). Nothing else is open.
- Phase 0 baseline: `bench/gpu/README.md` gate report; spike harness `bench/gpu/spikes.ts`, runner `bench/gpu/drive.ts` (hardware-adapter INVALID rule lives there).

## Execution protocol
- `superpowers:subagent-driven-development`, one fresh implementer per task, in a worktree on `feat/webgpu-phase1`. TDD per task exactly as the plan's steps specify; one commit per task with the plan's commit message.
- Model routing (plan "Execution & review protocol" + CLAUDE.md "## Models"): Tasks 1, 3b, 3c, 7b → orchestrator-tier implementer; 2, 3a, 4, 5, 6, 7a, 8 → Sonnet unless they fail the CLAUDE.md four-part "simple" test on inspection.
- Per-task review: round 1 blind + full scope (one reviewer, two verdicts: spec/plan compliance + quality). Tasks 1, 3b, 3c use TWO blind reviewers (numerics-critical: kernel math / guard semantics / gate methodology). Re-reviews are informed + scoped (open findings verbatim + fix diff, ADDRESSED/NOT ADDRESSED per finding). Cap 3 rounds; at the cap adjudicate each open finding in writing. Never coach a reviewer on what to ignore or pre-rate severity.
- Nits → CLAUDE.md inline-fix gate, else `gh issue create` with tags. Blockers fixed on branch regardless of size.
- Subagent delivery: have every subagent WRITE its report to a file (`/tmp/<task>-<role>-rN.md`) and wait with a background `until [ -f … ]` loop — in the previous session, subagent chat returns were sometimes lost ("idle" with no output). Paste the CLAUDE.md "Subagent briefing" block verbatim into every prompt.
- After Task 8: exactly one blind whole-branch review → one consolidated fix dispatch → one scoped re-review → surface residuals to the human. Then STOP and report; the human opens the PR.

## Gate commands (all need `bun run dev` in a separate shell; Chrome via CDP per memory `headless-webgpu-boot-check`)
- `bun bench/gpu/drive.ts phase1Reduce --out phase1-reduce` (3a)
- `bun bench/gpu/drive.ts phase1KernelSmoke --out phase1-kernel-smoke` (3b)
- `bun bench/gpu/drive.ts phase1Tolerance --out phase1-tolerance` (3c: T1 <1e-5 on nearTouchPair gaps 1e-3..1e-6, T2 <1e-6, T3 cosine >1−1e-6, every dE finite on every fixture incl. `crossing`)
- `bun bench/gpu/drive.ts phase1DeGate --out phase1-de-gate` (Task 4: ≥5× at N=480 and N=960, wall-clock incl. pack/upload/readback; pre-registered relErr band per D4)
- `bun bench/gpu/drive.ts phase1DriverSmoke --out phase1-driver-smoke` (Task 8)
- Static: `bun test && bunx tsc --noEmit && bun run lint && bun run knip && git diff --check`; plus the Task 8 negative-constraint proofs (no diff in `src/core/tangentPointEnergy.ts`, `src/core/sobolev`, `package.json`, `bun.lock`).

## Known hazards (each is anchored in the plan — do not let an implementer "clean up")
- three r185.1 TSL: `wgslFn` accepts ONE bare `fn` (regex in `WGSLNodeFunction.js:4`); shared code goes through `wgsl(src)` includes. Storage pointer access mode must equal the node's access (`read` ⇔ `.toReadOnly()`, `read_write` ⇔ plain `instancedArray`) — `WGSLNodeBuilder.js:2155`.
- Degeneracy guards are strict `>` in WGSL / `<=` in the TS mirrors; guard may be exactly 0 (`crossing` preset collinear pair). `>=` NaNs the whole gradient.
- ε=1e-10 is added after every norm like the CPU (NOT inert: 5e-4 rel at gap 1e-6). Reciprocal-power form (`invD`, `coeff_d`) avoids the d^7 denormal.
- `renderer.onDeviceLost` / `onError` are unbound prototype methods → `.call(renderer, info)`. `onError` receives an object despite the d.ts saying string.
- `GpuDriver.step` copies `st.live` once and hands the SAME array to kernel and step (`buildStepArgs` passes `live` by reference; `PinControls.tsx:169-174` mutates it in place).
- `dE?` length check is the first statement of `sobolevStepSet`, before `timingsBegin()` (`optimizer.ts:262`).
- Viewer: `dispatchGeneration.current++; inFlight.current = null;` are the first statements of the worker-lifecycle effect, before the `'main'` early return.
- Machine is a real NVIDIA Quadro RTX 3000 (`nvidia-smi`), the `vboxuser` naming is a scarecrow. Adapter must be hardware or the run is INVALID.
- Never touch memory files without explicit approval. One commit per spec/plan doc; amend, don't stack.

## Stop conditions (from the plan)
Any T1/T2/T3 red, INVALID adapter, or speedup < 5 at either N ⇒ Phase 1 does not merge; record the result in the README gate report and report to the human. Any `src/core` diff beyond the Task 5 seam lines, any dependency/lockfile change ⇒ spec violation, remove.
