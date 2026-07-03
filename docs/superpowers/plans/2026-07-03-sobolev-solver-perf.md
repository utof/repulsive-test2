# Sobolev Solver-Perf Milestone (briefing §5A + bench harness) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the two §5A solver-perf changes (per-step factorization/assembly reuse + typed-array dense core) plus a permanent per-phase benchmark harness and a UI step-timing readout, each oracle-gated and bench-measured before/after.

**Architecture:** A module-scoped phase-timing collector instruments the existing sobolev step pipeline (opt-in, zero effect on numerics); a standalone Bun bench script drives `sobolevStepSet` on parametric trefoils and saves per-phase medians as committed JSON keyed by git SHA. The dense core moves to flat `Float64Array` row-major internals behind the existing `number[][]` exported APIs (which become thin wrappers kept as the reference/golden surface). Factorization reuse freezes Ā, C, and one LU factorization at the step base point and reuses it for the gradient solve and every projection iterate of every τ-trial (paper-sanctioned: SelfAvoiding.tex line 734), as an opt-in `projectionMode: 'frozen'` — default stays `'reassemble'` so every existing golden/test is untouched.

**Tech Stack:** Bun + TypeScript (no new deps; hand-rolled timing via `performance.now()`), Python oracle (`uv run --with numpy --with scipy`), existing golden/property-gate pattern.

**Spec:** `local_files/2026-07-03-next-steps-briefing.md` §1 (measured profile), §2 (corrections — read before "optimizing" anything already measured), §5A. Anchors that bind this plan: `src/core/sobolev/lineSearch.ts` (reassembly TSDoc — may only be superseded WITH the measurement + paper cite), `src/core/sobolev/linsolve.ts` (self-certifying residual on EVERY solve — non-negotiable), CLAUDE.md (TSDoc anchor rule, one commit per plan doc).

**Branch:** `feat/sobolev-solver-perf` (cut from the post-merge `main`). Nothing is pushed without explicit user approval.

**Commit sequence (each bench-measured, results JSON committed with it):**
1. `docs: implementation plan for sobolev solver-perf milestone (bench + 5A)` — this file, iterate with `--amend`.
2. `feat(bench): sobolev per-phase benchmark harness + UI step timings` (+ baseline results JSON).
3. `perf(sobolev): reuse E₀ across descent steps (bit-identical)`.
4. `perf(sobolev): typed-array dense core (flat assembly + LU factor/solve split)`.
5. `perf(sobolev): frozen-projection factorization reuse (oracle-gated, paper line 734)`.

**Global gates (apply to every task):**
- `bun test` fully green (165 pre-existing tests + new ones); golden tolerances stay at their committed values (1e-9 vertex/g̃ rel, 1e-10 residual, 1e-12 energy rel) — never loosened.
- Every new exported symbol carries TSDoc with `@see`/`Why:` (CLAUDE.md).
- Bench run before + after each perf commit on the same machine/session; numbers go into the commit message and `bench/results/`.

---

## Task 1: Phase-timing collector + step timings surface

**Files:**
- Create: `src/core/sobolev/phaseTimings.ts`
- Modify: `src/core/optimizer.ts` (begin/end + attach timings), `src/core/sobolev/innerProduct.ts`, `src/core/sobolev/gradient.ts`, `src/core/sobolev/lineSearch.ts`, `src/core/tangentPointEnergy.ts` — NO: energy/dE are timed at their **call sites** in optimizer/lineSearch (the kernel files stay untouched).
- Test: `test/sobolev/phaseTimings.test.ts`

**Design (locked):**

```ts
// src/core/sobolev/phaseTimings.ts
/**
 * Stable top-level phase keys. Sub-phases (bHigh/bLow inside assembleA;
 * later 'factor' inside 'saddle') OVERLAP their parents — sums across keys
 * double-count by design; compare like-to-like across bench runs.
 * Why: keys are the bench ledger's schema — renaming one breaks baseline
 * comparability in bench/results/. Add keys; never rename.
 * @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 1)
 */
export type SobolevPhaseKey =
    | 'dE'          // differential (analytical or FD), timed at call site
    | 'energy'      // calculateEnergy calls (E₀ + Armijo trials), timed at call sites
    | 'bHigh'       // assembleBHigh body
    | 'bLow'        // assembleBLow body
    | 'assembleA'   // assembleA total (parent of bHigh/bLow + the sum loop)
    | 'expand'      // expandBlockDiag (drops to 0 calls after Task 4)
    | 'saddle'      // one whole saddle solve (build K + factor + backsolve + residual)
    | 'factor'      // LU factorization only (appears after Task 4's split)
    | 'projection'  // projectOntoConstraintSet total
    | 'lineSearch'  // lineSearchStepSet total
    | 'step';       // sobolevStepSet total

export interface PhaseSample { ms: number; calls: number }
export type SobolevStepTimings = Partial<Record<SobolevPhaseKey, PhaseSample>>;

let acc: SobolevStepTimings | null = null; // null = collection off (default)

export function timingsBegin(): void { acc = {}; }
export function timingsEnd(): SobolevStepTimings | null { const r = acc; acc = null; return r; }

/** Wrap a phase. When collection is off this is a plain call — zero overhead
 * beyond one null check; numerics are NEVER affected (pure timing). */
export function timed<T>(key: SobolevPhaseKey, fn: () => T): T {
    if (acc === null) return fn();
    const t0 = performance.now();
    const r = fn();
    const dt = performance.now() - t0;
    const s = acc[key] ?? (acc[key] = { ms: 0, calls: 0 });
    s.ms += dt;
    s.calls += 1;
    return r;
}
```

Threading (no signature changes below `sobolevStepSet`):
- `SobolevStepOptions` gains `collectTimings?: boolean` (default false).
- `sobolevStepSet` result gains a TOP-LEVEL `timings?: SobolevStepTimings` field (NOT inside `stats` — `test/sobolev/constraintSetFlow.test.ts` does `expect(viaSet.stats).toEqual(viaX0.stats)` across two separate calls, which would never match on wall-clock).
- Call-site wraps: `timed('dE', ...)` and the convergence-path `timed('energy', ...)` in `sobolevStepSet`; `timed('energy'|'projection'|…)` inside `lineSearchStepSet`/`projectOntoConstraintSet`; `timed('bHigh'|'bLow')` inside `assembleA` around the two assembler calls, with the whole body wrapped in `timed('assembleA', ...)`; `timed('expand'|'saddle', ...)` at the `expandBlockDiag`/`solveSaddle` call sites in `gradient.ts` and `lineSearch.ts`.
- Reentrancy: none — the frame loop and bench are single-threaded and `sobolevStepSet` never nests.

**Steps:**

- [ ] 1.1 Write `test/sobolev/phaseTimings.test.ts`:
  - default off: `sobolevStepSet(...)` without the flag → `timings` is `undefined`;
  - opt-in: with `{collectTimings: true}` on the crossing fixture → `timings.step.calls === 1`, every present phase has `ms >= 0`, `timings.energy.calls >= 2` (E₀ + ≥1 Armijo eval), `timings.assembleA.calls >= 2` (gradient solve + ≥1 projection iterate), `timings.saddle.calls === timings.assembleA.calls`;
  - numerics unaffected: run with and without the flag → `flatten(vertices)` arrays `toEqual`, `energy` `toBe`-equal, `stats` `toEqual`.
- [ ] 1.2 Run it, confirm it fails (`bun test test/sobolev/phaseTimings.test.ts`).
- [ ] 1.3 Implement `phaseTimings.ts` + the wraps listed above.
- [ ] 1.4 `bun test` — everything green (existing suites prove the off-path is inert).

## Task 2: Bench harness + baseline

**Files:**
- Create: `bench/sobolev.bench.ts`, `bench/README.md`, `bench/results/` (first JSON), package.json script `"bench": "bun bench/sobolev.bench.ts"`.

**Design (locked):**
- Cases: parametric closed trefoil generated IN the bench file — `p(t) = (sin t + 2 sin 2t, cos t − 2 cos 2t, −sin 3t)`, `t = 2πi/N`, edges `[i, (i+1)%N]` — at N=60 and N=120 (flag `--big` adds N=240), each under two constraint sets: `[barycenter, totalLength]` and `[barycenter, edgeLengths]` (frozen targets from the initial geometry, mirroring the store lifecycle). Matches the briefing §1 table for continuity.
- Per case: 2 warmup full steps (discarded), then K=5 measured `sobolevStepSet` calls **each from the same initial vertices** (the step is pure and deterministic — identical work every repeat), `collectTimings: true`; report per-phase MEDIAN ms + calls. Plus isolated micro-medians (of 7) for `calculateEnergy`, `gradientAnalytical`, `assembleA`, `expandBlockDiag`, `solveSaddle` on the same geometry.
- Output: printed markdown table; `--save <label>` writes `bench/results/<YYYY-MM-DD>-<label>.json` `{label, date, gitShaShort, bunVersion, cases:[{name, nV, nE, constraintMode, projectionMode, phases, isolated, fullStepMsMedian}]}`; `--baseline <path>` adds a Δ% column vs that file's matching cases.
- `bench/results/*.json` are COMMITTED — they are the "what improved with each iteration" ledger the user asked for. VM noise is ±5% (briefing §1); treat |Δ| < 10% as noise unless reproduced.

**Steps:**

- [ ] 2.1 Write `bench/sobolev.bench.ts` + `bench/README.md` (usage, methodology, noise caveat, ledger convention) + the package.json script.
- [ ] 2.2 Run `bun bench/sobolev.bench.ts --save baseline` → sanity-check the table against the briefing §1 numbers (same order of magnitude; note we're on the same VM).
- [ ] 2.3 Commit baseline JSON.

## Task 3: UI step-timings line

**Files:**
- Modify: `src/store.ts` (field `sobolevTimings: SobolevStepTimings | null`, cleared wherever `sobolevStats` is cleared; `dispatchDescentStep` gains optional `collectTimings` passthrough and returns `timings` in `DescentStepOutcome`), `src/scene/Viewer.tsx` (pass `collectTimings: true` in sobolev mode; publish `sobolevTimings` in BOTH the throttled and the auto-pause `setState`), `src/ui/Stats.tsx` (second monospace line).
- Test: extend `test/sobolev/phaseTimings.test.ts` with a `dispatchDescentStep` case (sobolev mode + flag → `timings` non-null; raw mode → `timings` null). Check `test/store-constraints*.test.ts` for whole-outcome `toEqual`s before adding the field; if any compare the full outcome object, extend their expected shape knowingly (milestone contract change, not a regression).

**UI format (Stats.tsx, sobolev mode only, when timings present):**
`Δt 53.9ms — dE 7.0 · A 14.4×2 · saddle 18.4×2 · E 12.0×2 · proj 9.5 · LS 31.2`
(`×n` = calls; values `.toFixed(1)` ms; omit absent keys.)

**Steps:**

- [ ] 3.1 Extend the test; run; fails.
- [ ] 3.2 Implement store + Viewer + Stats wiring.
- [ ] 3.3 `bun test` green; `bun run dev` + headless boot check (memory: SwiftShader errors are environmental) to eyeball the line.
- [ ] 3.4 Commit Tasks 1–3 together: `feat(bench): sobolev per-phase benchmark harness + UI step timings`.

## Task 4: E₀ reuse across steps (bit-identical)

**Files:**
- Modify: `src/core/sobolev/lineSearch.ts` (`LineSearchOptions.energyBefore?: number`; `const e0 = opts?.energyBefore ?? calculateEnergy(...)`), `src/core/optimizer.ts` (`SobolevStepOptions.energyBefore?: number`; use it on the converged/singular paths too; pass down to the line search), `src/store.ts` (`dispatchDescentStep` passthrough), `src/scene/Viewer.tsx` (a `lastEnergy` ref).
- Test: `test/sobolev/energyReuse.test.ts`.

**Correctness invariant (the ONLY subtle point):** `energyBefore` may be passed **iff it is exactly `calculateEnergy(current vertices, …)`**. The previous accepted step's returned `energy` was computed at exactly the vertices that become this step's input, so within a continuous run the reuse is bit-identical. Frame-loop rule: `lastEnergy.current = null` whenever `!running` (covers run start, user pause, auto-pause, preset rebuild — the store re-anchors targets at those same boundaries); set it to `result.energy` after every accepted step; pass `energyBefore: lastEnergy.current ?? undefined`. A stale E₀ would corrupt the Armijo gate — the null-on-pause rule makes staleness structurally impossible.

**Steps:**

- [ ] 4.1 Test: on crossing + linked-rings fixtures, `sobolevStepSet(v, …)` vs `sobolevStepSet(v, …, {energyBefore: calculateEnergy(v, …)})` → `flatten(vertices)` `toEqual`, `energy` `toBe`, `stats` `toEqual` (bit-identity); and with `collectTimings` the second call has `timings.energy.calls` one lower than the first.
- [ ] 4.2 Run; fails; implement; `bun test` green.
- [ ] 4.3 Bench: `bun bench/sobolev.bench.ts --baseline bench/results/<date>-baseline.json --save e0-reuse`. Expected: full-step median drops by roughly one energy eval (~10% at N=60); phase table shows `energy.calls` 2→1.
- [ ] 4.4 Commit with the numbers.

## Task 5: Typed-array dense core

**Files:**
- Modify: `src/core/sobolev/innerProduct.ts` (flat internals + wrappers), `src/core/sobolev/linsolve.ts` (factor/solve split + scalar-A saddle path), `src/core/sobolev/gradient.ts` + `src/core/sobolev/lineSearch.ts` (switch hot path to the new entry points).
- Test: `test/sobolev/linsolveFlat.test.ts`; every existing suite is the main gate.

**Design (locked signatures):**

```ts
// innerProduct.ts — flat row-major n×n internals; the existing number[][]
// exports become thin wrappers over these (reference/golden surface unchanged).
export function assembleBHighFlat(vertices, edges, disjointPairs, alpha, beta, epsilon): Float64Array
export function assembleBLowFlat(...): Float64Array
export function assembleAFlat(...): Float64Array   // bHighFlat + bLowFlat, summed
// Internals rule: ONLY the accumulation matrix goes flat (B[ia*n+ib] += …);
// vertices stay Vec3[] — the 07-01 log measured vertex-layout flattening as a
// dead end on inlined kernels (briefing §2.1). Kernel arithmetic, ε placement,
// symmetrization, and comment anchors are copied VERBATIM from the nested
// versions; per-matrix symmetrization stays per-matrix (bHigh and bLow each),
// preserving today's op order as closely as the layout allows.

// linsolve.ts
export interface LuFactorization { m: Float64Array; piv: Int32Array; n: number }
/** Factors IN PLACE on the caller's buffer (caller must not reuse it). Same
 * partial-pivoting algorithm, same singular/non-finite throws as luSolve. */
export function luFactor(k: Float64Array, n: number): LuFactorization
/** Forward/back substitution; keeps luSolve's non-finite-rhs throw. */
export function luSolveFactored(fac: LuFactorization, rhs: number[]): number[]
/** Saddle solve from the SCALAR n×n A (flat) — builds flat K directly with the
 * three diagonal Ā blocks + C/Cᵀ (expandBlockDiag is never materialized),
 * factors, solves, and computes the self-certifying residual via the
 * STRUCTURED matvec K·z = [A·x per coord block + Cᵀλ; C·x] against the
 * original A and C (the factorization destroyed K). Residual definition is
 * unchanged: ‖K·z − rhs‖₂ / max(1, ‖rhs‖₂), computed on EVERY solve.
 * Returns fac for Task 6's reuse. */
export function solveSaddleFromA(
    a: Float64Array, n: number, C: number[][], rhsTop: number[], rhsBottom?: number[],
): { x: number[]; lambda: number[]; residual: number; fac: LuFactorization }
// luSolve(number[][], …) and solveSaddle(A3, …) REMAIN with their current
// bodies as the slow reference implementations (golden-tested, cross-checked
// against the fast path by the new test). buildSaddleMatrix stays.
```

`gradient.ts` and `lineSearch.ts` switch to `assembleAFlat` + `solveSaddleFromA` (timed via the same `'assembleA'`/`'saddle'` keys; `'expand'` naturally drops to zero calls; `'factor'` wraps the `luFactor` call inside `solveSaddleFromA`).

**Gate:** the committed goldens at committed tolerances are the contract (briefing §5A.2 — op-order changes may break bit-identity; 1e-9/1e-10 gates decide). If any golden fails: fix the code, never the tolerance.

**Steps:**

- [ ] 5.1 `test/sobolev/linsolveFlat.test.ts`:
  - `luFactor`+`luSolveFactored` vs `luSolve` on a deterministic well-conditioned 12×12 system → solutions `toEqual` (identical algorithm & op order on a copy);
  - singular (zero pivot column) and non-finite matrix/rhs inputs throw with "singular" in the message, matching `luSolve`'s contract;
  - `solveSaddleFromA(flatten(A), n, C, dE)` vs `solveSaddle(expandBlockDiag(A), C, dE)` on the crossing fixture's real system → `x`/`lambda` rel-diff ≤ 1e-12, both residuals ≤ 1e-10;
  - `assembleAFlat` vs `assembleA` on two fixtures → entrywise identical (`toEqual` after unflattening; the per-matrix symmetrization + verbatim kernel makes this exact — if it is not exact, entries must still be ≤1e-15 rel and the test documents which op reordered, but aim for exact).
- [ ] 5.2 Run; fails; implement `innerProduct` flat internals + wrappers.
- [ ] 5.3 Implement `luFactor`/`luSolveFactored`/`solveSaddleFromA`; switch `gradient.ts`/`lineSearch.ts`.
- [ ] 5.4 `bun test` — ALL suites green (goldens are the gate). Run the oracle property script is NOT needed (no oracle change).
- [ ] 5.5 Bench vs baseline + e0-reuse; expected 2–6× on `saddle`/`assembleA` phases (briefing §2.1 — but MEASURE; the log's reverts are the cautionary tale). Save `--save typed-array`. If the win is <1.2× overall, say so plainly in the commit message — do not spin it.
- [ ] 5.6 Commit with numbers. If LDLᵀ curiosity remains AND time permits, it is a separate measure-first A/B AFTER this commit — kill on any golden failure or <1.3× factor win (briefing §2.2: not paper-mandated; residual self-certification must survive).

## Task 6: Frozen-projection factorization reuse (oracle first, then TS)

**Files:**
- Modify: `oracle/tpe_constraints_oracle.py` (projection-variant arg), `oracle/README.md` (frozen section + regen commands), `src/core/sobolev/lineSearch.ts` (frozen path + SUPERSEDE the reassembly anchor TSDoc), `src/core/optimizer.ts` (`projectionMode` option + operator handoff), `src/store.ts` + `src/ui/ControlPanel.tsx` (A/B select, default `'frozen'`), `src/scene/Viewer.tsx` (passthrough).
- Create: `oracle/golden/{crossing,linked-rings,knot}-length-frozen.json`, `oracle/golden/{crossing,junction-y}-edgelengths-frozen.json`, `oracle/golden/crossing-point-frozen.json`; `test/sobolev/frozenProjection.test.ts`.

**Semantics (locked):** at step start, assemble `A(γ₀)`, evaluate `C(γ₀)`, factor `K(γ₀)` ONCE (`solveSaddleFromA` already returns `fac`). The gradient solve consumes that solve directly. In `projectionMode: 'frozen'`, every projection iterate of every τ-trial solves `fac · [x; μ] = [0; −Φ(γ^q)]` with Φ evaluated FRESH at the current iterate (quasi-Newton: frozen metric + frozen Jacobian, live residual); `γ^{q+1} = γ^q + x`. The per-block §3.3 stopping rule, tolerances, `maxIter = 8`, non-finite→`ok:false`, and throw→`ok:false` semantics are IDENTICAL to the reassemble path. The projection solves' self-certifying residual is computed per solve against the FROZEN A/C via the structured matvec (the anchor in `linsolve.ts` — never skipped). Default stays `'reassemble'`: all committed goldens and every existing test run the old path unchanged; the app opts into `'frozen'` via the store (that is the measured decision the old lineSearch anchor demanded — update that TSDoc to cite SelfAvoiding.tex line 734, the frozen goldens, and this plan).

**Oracle protocol (BEFORE any TS change — this is the kill switch):**
- `tpe_constraints_oracle.py` gains optional 5th arg `projection` ∈ `{reassemble, frozen}` (default `reassemble`; existing invocations byte-stable). Frozen mode: `scipy.linalg.lu_factor` on `K(γ₀)` once; `lu_solve` per projection iterate; same fresh-Φ semantics as above.
- New embedded property checks in frozen mode: everything the mode already checks PLUS `projection_iterations <= 3` (briefing/paper expectation) on the accepted step.
- Generate the 6 frozen goldens (commands mirror README's loops with the extra arg). **Kill criterion:** if `knot-length-frozen` or `linked-rings-length-frozen` fails the iteration gate or any drift gate, STOP — commit only the oracle-mode extension + a briefing-doc note; do not port to TS.

**Steps:**

- [ ] 6.1 Extend the oracle + README; regenerate ONLY the 6 new `-frozen` goldens (existing goldens must be byte-identical — verify with `git status`).
- [ ] 6.2 Gate check: all frozen-mode property checks green on all 6, iterations ≤ 3. Record iteration counts.
- [ ] 6.3 `test/sobolev/frozenProjection.test.ts`: for each frozen golden — TS frozen step reproduces `accepted`, `tau` exact, `projection_iterations` exact, vertices rel ≤ 1e-9, energy rel ≤ 1e-12; drift gates (total-length / per-edge / pin distance ≤ 1e-8 vs the golden's targets); plus a default-mode guard: `projectionMode` omitted ⇒ output `toEqual` the pre-change path on one fixture (bit-identity of the default).
- [ ] 6.4 Run; fails; implement the TS frozen path (operator built in `sobolevStepSet`, passed through `lineSearchStepSet` into `projectOntoConstraintSet`; direct `projectOntoConstraintSet` callers without an operator assemble it once at entry).
- [ ] 6.5 Store field + ControlPanel select (`projection: frozen | reassemble`, default frozen) + Viewer passthrough; extend a store test in the `store-constraints-m2` style.
- [ ] 6.6 `bun test` green. Bench both modes (`projectionMode` is a bench case dimension) vs previous results; expected: per accepted step, assembly count drops from `1 + Σ projIters` to 1 and factorizations to 1. Save `--save frozen-reuse`.
- [ ] 6.7 Commit with numbers. Update the briefing doc's §5A status line (done/kill) and comment measured results on issue #1 (`gh issue comment 1`).

---

## Self-review checklist (run after writing, before executing)

- Spec coverage: §5A.1 reuse (Task 6 + E₀ in Task 4) ✔; §5A.2 typed-array (Task 5) ✔; bench+UI (user ask, Tasks 1–3) ✔; measure-each-on-§1-bench ✔ (per-task bench steps); LDLᵀ optional A/B ✔ (5.6); anchors respected ✔ (lineSearch anchor superseded only in Task 6 with cite+measurement; linsolve residual anchor preserved in both new solve paths).
- No placeholders: interfaces locked; mechanical ports specify exact invariants (verbatim kernels, per-matrix symmetrization, throw contracts) instead of duplicated code — the executor must copy from the named existing bodies, never re-derive.
- Type consistency: `SobolevStepTimings` (Tasks 1/2/3), `LuFactorization`/`solveSaddleFromA` (Tasks 5/6), `projectionMode` (Task 6) cross-referenced consistently.
