# WebGPU Solver — Design Spec

**Status:** REVIEWED ×3 (three independent adversarial reviews 2026-08-13,
each APPROVE-WITH-FIXES; all 14 + 11 + 10 findings applied — review 3 ran
fresh-eyes with no knowledge of rounds 1–2 and found defects only in forward
promises, none in present-code claims). Fresh-session entry point: read §0,
then the two research docs this spec is built on.

**Goal:** realtime (>15 fps) Sobolev-descent untangling at N≈1000 vertices by
moving the O(E²) pairwise kernels *and* the dominant saddle solve onto the GPU
already used for rendering, with pre-registered kill gates at every phase.

**Evidence base (read before disputing any number here):**
- `docs/2026-08-13-ai-research-webgpu-compute.md` — implementation patterns,
  verified against installed `three@0.185.1`. Cited below as **[IMPL §n]**.
- `docs/2026-08-13-ai-research-gpu-precision.md` — precision measurements
  against the real kernel. Cited below as **[PREC Qn]**.
- `bench/results/2026-07-06-ldlt-ab.json` — CPU per-phase baseline at N=60/120.
- Prior art: no WebGPU/WGSL port of tangent-point energy exists anywhere;
  every reference implementation is f64 [PREC Q4]. We are first; expect zero
  copyable code.

---

## 0. State detection (for a fresh session)

- If `src/gpu/` does not exist → nothing implemented; start at Phase 0.
- If `src/gpu/` exists but `solverDriver` in `src/store.ts` has no `'gpu'`
  value → Phase 0 spikes landed, Phase 1 integration not started.
- Phase gate results are committed as `bench/results/*-gpu-*.json` and a
  `docs/superpowers/plans/2026-08-13-webgpu-solver.md` checklist (plan doc,
  written after this spec converged).

## 1. Scope

**In scope:**
- GPU pairwise kernels: energy E, analytic gradient dE, and the A-matrix
  *action* (matvec) — all O(E²) work.
- GPU/mixed-precision replacement for the dense saddle solve at large N.
- A third solver driver `'gpu'` alongside `'main'`/`'worker'`, with automatic
  fallback to the existing drivers on any gate failure at boot.
- Tolerance-based verification gates replacing bit-identity for GPU paths.
- Zero-readback rendering spike (compute writes positions the renderer reads).

**Out of scope (explicit non-goals):**
- Barnes–Hut / BCT / multigrid on GPU. The CPU-side stage-2 science just
  stabilized (#5/#6 closed 2026-08-13); its TS port is a separate milestone.
  GPU brute-force O(E²) is the bet for N up to roughly ~2000 (this spec's own
  extrapolation from [PREC headline; IMPL §4] — neither doc states the bound).
- Full df64 emulation (~4× cost, kills the win) [PREC Q3].
- Firefox (per-dispatch overhead ~1 ms — disqualified) [IMPL §4].
- f16 anywhere (ε=1e-10 flushes to zero in f16) [PREC Q5].
- Touching `src/core/**` **numerics**: the CPU f64 compute path remains the
  reference and the fallback, byte-for-byte unchanged. Two sanctioned public
  type changes exist and are named here so they cannot drift in silently
  (review-3 F1): (i) the store's `SolverDriver` union widens with `'gpu'`;
  (ii) the CORE type `DescentStepOutcome` (`src/core/dispatch.ts:75-98`,
  re-exported by the store) gains an explicit stats-only variant in Phase 3
  (§2.8), which also touches `SolverWorkerResponse`'s carried type and
  requires a knowing update to the worker plan's §T2 deep-equality tests.
- GPU support for `descentMode: 'raw'`, `mode: 'finiteDiff'`, or any nonzero
  penalty config (review-3 F4): the `'gpu'` driver serves
  **sobolev + analytical gradient + penalties-off only**; any of those
  configs active forces per-step fallback to `'worker'`. (Penalties enter the
  Armijo objective — `lineSearch.ts:439-451` — and the GPU line search does
  not evaluate them; serving them is optional follow-up, not this milestone.)
  T5 and τ-agreement fixtures are declared penalties-off.

## 2. Architecture (decided; changing any of these requires re-review)

### 2.1 Device, thread, and library surface
- **One GPU device**, owned by the existing `WebGPURenderer`
  (`src/scene/Viewer.tsx`). The solver uses `renderer.compute()` /
  `renderer.getArrayBufferAsync()`; never a second adapter [IMPL §2].
- **Main thread**, beside the renderer. The existing worker stays as the CPU
  fallback driver, contract untouched — GPU dispatch is ~30 µs of CPU per
  call, not CPU-bound work [IMPL §5].
- **TSL for plumbing, `wgslFn` for the kernel body.** three's TSL owns buffers,
  bind groups, and dispatch; the ~200-flop pair kernel is hand-written WGSL via
  the `wgslFn` escape hatch so the node compiler cannot reassociate the
  precision-critical expressions [IMPL §1]. (Reassociation is still *legal* for
  the downstream WGSL compiler — that risk is Gate G2, not a TSL choice.)

### 2.2 Kernel formulation
- **Gather, no atomics.** Kernel keyed on edge I evaluates BOTH roles of each
  ordered pair and accumulates only into I's own vertices — pure gather,
  deterministic, ~2× flops (cheap on GPU vs atomic contention). WGSL has no
  f32 atomicAdd anyway [IMPL §3].
- **Reductions** (total energy, dot products): fixed-fanout workgroup-memory
  tree, two passes, modeled on three's `webgpu_compute_reduce` example. No
  `subgroupAdd` (vendor-dependent tree shape), no Kahan/two-sum (not portable
  in WGSL — §15.7.5 permits unconditional reassociation) [PREC Q3, Q5].
- **Topology** (edges + CSR-flattened `disjointPairs`) uploaded once per
  `graphVersion`, mirroring the worker's topology-cache contract.

### 2.3 Precision model [PREC recommended architecture]
- **Positions: two-float (hi/lo f32 pairs).** All coordinate differences
  computed `(hi_i − hi_j) + (lo_i − lo_j)`; everything downstream plain f32.
  This is the near-touch cancellation fix (up to 10⁶× at gap 1e-6) at ~1.1×
  cost. NOT a two-sum; still gated per-adapter (G2).
- **Integer-exponent powers via repeated multiplication**; `pow` only where the
  exponent is genuinely non-integer. WGSL `pow` = `exp2(y·log2 x)`, measured
  harmless at α=3/β=6 — re-verify if exponents ever change [PREC Q1].
- **Degeneracy guards re-derived for f32.** The CPU `< 1e-14` pre-ε length
  guards are meaningless in f32; the GPU kernel is mathematically the ε=0
  energy (ε=1e-10 is provably inert in f32), so degeneracy protection is an
  explicit branch with an f32-scale threshold, derived in the plan [PREC Q1].
- **Never mix CPU and GPU energies within one descent run** — they differ at
  the 1e-9 level (ε-inertness); *within a run, every energy COMPARISON uses
  energies from one source*. (Phase 1 satisfies this by keeping ALL energies
  CPU-side — §5.) [PREC end]

### 2.4 The solve (the actual bottleneck — ~45–80% of a step at N=120, O(N³))
Two candidate paths, decided by pre-registered experiment (Gate G5):
- **(a) Mixed-precision direct: GPU f32 factorization + CPU f64 iterative
  refinement.** Measured 1–3 refine iterations to the 1e-10 residual gate at
  N≤128; κ(K) ∝ N^3.3 leaves ~10× headroom at N=1000, GMRES-IR as documented
  fallback [PREC Q2]. **Honesty note:** a CPU f32 factor cannot reach G7 at
  N=1000 — n≈3000 LDLᵀ is ~n³/3 ≈ 9e9 flops; per-factorization at n≈363 is
  ≈13–15 ms measured [LDLᵀ plan doc's 3-run table: 14.8/12.9/13.2 ms at
  N120-total-frozen], ×(3003/363)³ ≈ 565 →
  ~7 s against a 66 ms budget. Therefore (a) at target N **means a GPU dense
  factorization — a large, named deliverable with no prior art in this stack,
  and this variant's principal risk**; the CPU-f32 factor+refine combination
  serves only the kernels-only fallback tier at small N, not G7.
- **(b) f32 GPU CG inside a f64 iterative-refinement outer loop** (CG replaces
  the triangular solves, GMRES-IR style [PREC Q2]): inner fixed-count f32 CG
  with GPU matvec, batched into ONE `renderer.compute([...])` submission per
  inner solve [IMPL §4]; outer loop computes f64 residuals on CPU — **one
  readback per outer iteration, and that readback cost is part of (b)'s
  measured cost in G5**. Pure f32 CG alone has a residual floor ~n·u_f32 ≈
  1e-6..1e-5 and can NEVER meet T4; (b) is only defined with the outer f64
  loop. Preconditioning and inner-iteration counts are a plan-level
  experiment; the iterative-cost yardstick is the closed-loop ρ-vs-N table in
  issue #5's closing comment (10/20/31 V-cycles at N=60/240/960, Galerkin +
  nested init — public record of a session run; `check_mg_stage2.py` itself
  covers only n≤61 fixtures) (review-3 F6).
- **Serving the line search's projection solves (review-2 F2 — the coupling
  the variants must both answer):** every Armijo trial embeds a projection
  correction loop whose every iteration is itself a solve against the step-frozen
  K(γ₀) (`lineSearch.ts:253-262`). Per variant:
  - under **(a)**: projection corrections are **GPU backsolves against the
    resident f32 factor**, batched into the same submission as the trial
    evaluations (only coherent with the GPU-resident line search §2.5/§5
    place in Phase 2).
  - under **(b)**: no factorization exists; each projection correction is a
    **fixed-count inner f32 CG solve** reusing (b)'s preconditioner, batched
    likewise.
  In BOTH variants these projection solves are part of G5's measured/estimated
  **per-STEP** cost — a variant that wins the gradient solve but loses the
  line search's projections has not won.
- The `perEdge` constraint set (adds N constraint rows) is measured in Phase 0
  — its conditioning is the untested branch of the κ extrapolation [PREC Q2].

### 2.5 Descent loop and line search (revised per review-1 F1, review-2 F2/F3/F4)
- **Reality check first:** every Armijo trial in `src/core` embeds a
  constraint projection — an iterative quasi-Newton loop with its own saddle
  solves (`src/core/sobolev/lineSearch.ts:511-530`), and projection dominates
  the line-search phase (24.8 of 26.4 ms at N120-total reassemble [bench]). A
  "GPU-resident line search" that ignores projection would be a different
  algorithm. Consequences:
  - **Phases 0–1: the line search stays CPU-side, f64, unchanged** — trivially
    coherent because the solve is also still CPU-side.
  - **Phase 2: the frozen-mode line search moves GPU-resident TOGETHER WITH
    the chosen solve variant** — the projection solves inside it are served by
    that variant (§2.4: (a) → factored backsolves; (b) → fixed-count inner
    CG), so line search and solve are one inseparable GPU stage. Honest
    naming (review-2 F4): the GPU port is a **fixed-count quasi-Newton
    correction loop — in-shader Φ evaluation for the barycenter/length/pin
    blocks plus the variant's solve — replacing the CPU loop's up-to-8
    iterations with early exit** (`lineSearch.ts:218-294`); dropping the
    early exit is a deliberate behavior change with its own verification
    surface. It ships behind a **τ-agreement gate**: on T5's fixtures, the
    GPU-chosen τ matches the CPU line search's τ on ≥95% of steps, else the
    driver stays on the CPU line search (paying the §2.4 sync costs and
    likely failing G7 — i.e., τ-agreement is effectively load-bearing for
    the milestone and is called out as such). **Measurement protocol
    (review-3 F2 — shadow evaluation, not trajectory comparison):** at each
    GPU step, the CPU f64 line search is additionally run ONCE from the SAME
    input state (positions, gradient, frozen factor) as the GPU search;
    "agreement" means equal backtracking index k (τ = τ₀·ρᵏ makes k the
    discrete statistic); gate = k-agreement on ≥95% of the K=50 steps of each
    T5 fixture, penalties off. This sidesteps trajectory divergence — after
    any step the runs would otherwise compare different states.
  - **`'reassemble'` mode keeps the CPU f64 line search and CPU f64 factor
    permanently** — it re-factors at every trial by design and is documented
    as the small-N/reference mode; G7 is gated on frozen mode only (matching
    the store default, `store.ts:359`).
  - **Deliberate deviation from [PREC Stage 3]** (review-1 F11): the
    frozen-mode in-shader Armijo comparison uses f32 energies, justified by
    the measured 4–6 orders of margin — evidence basis N≤128 / 3000 steps
    [PREC Q1]; T5 is the backstop and the margin is re-measured at N=1000
    before any default flip.
- **One `renderer.compute([...])` array per GPU stage of a step.** Never one
  call per kernel — the submit overhead (~13 µs of the ~30 µs per-dispatch
  cost) only amortizes inside a batched array [IMPL §4]. Single
  highest-leverage rule in the design.
- Steady-state readback: at most one small stats readback per frame (energy,
  step outcome) — plus the mandatory position-authority readbacks of §2.8.

### 2.6 Driver integration
- `solverDriver: 'gpu' | 'worker' | 'main'` in the store; `'gpu'` selectable
  only after boot gates pass, with automatic fallback `'gpu'→'worker'` and the
  existing `'worker'→'main'` chain. Boot gates (G2 self-test + feature checks)
  run once per adapter at renderer init.
- The dispatch/step-arg contract (`src/core/dispatch.ts`) is the integration
  seam, same as the worker driver used. **No `src/core` numeric changes; the
  only public-type deltas are the two sanctioned in §1** (SolverDriver
  widening; DescentStepOutcome stats-only variant).

### 2.7 Rendering integration (Phase 3, spiked in Phase 0)
- Target: compute output written into the fat-line geometry's interleaved
  `instanceStart`/`instanceEnd` storage buffer — usage flags verified
  compatible [IMPL §2]. This requires a dedicated **combine/scatter kernel**
  (review finding 13): per-vertex hi/lo pairs → `hi+lo` f32 → per-edge
  interleaved instance layout, one cheap pass per frame, included in the G4
  spike so the spike proves what Phase 3 needs.
- This is a rewrite of `src/scene/Curve.tsx` (today: fresh Float32Array +
  `setPositions()` per frame) against a material (`Line2NodeMaterial`) that
  exposes no position hook. **Spiked in Phase 0** because failure changes the
  architecture: the fallback (one `getArrayBufferAsync` of positions per
  frame, ~0.5 ms + a frame of latency) is acceptable for ≥15 fps but erodes
  the ceiling.
- **Vertex spheres (review-2 F6):** `Curve.tsx` also renders per-vertex
  spheres via an InstancedMesh whose matrices are CPU-written from `live`
  every frame — under Phase 3 stale-`live` they would freeze while the lines
  move. Disposition: **spheres are hidden under the `'gpu'` driver in
  Phase 3** (cheapest honest option); a second scatter kernel writing
  `instanceMatrix` is noted as optional follow-up, not scoped here. The G4
  spike needs only the line path.
- **GradientArrows / descent-field consumers (review-2 F11):** the §D14
  arrows-during-run path consumes the step outcome's field; under the
  stats-only Phase 3 outcome it has no data. Disposition: arrows are hidden
  under the `'gpu'` driver (same rationale as spheres); re-enabling them via
  a field readback at HUD cadence is optional follow-up.

### 2.8 Position authority & live-buffer coherence (new; review finding 5)
The store's entire main-thread contract assumes CPU-visible positions:
`applyStepOutcome` mutates `live` in place, `Curve.tsx` reads `live` per
frame, commit-on-pause folds `live` into `graph.vertices` and recomputes
energy (`src/store.ts:554-578`), pins snapshot `live`, and
`DescentStepOutcome` carries vertices per step (`src/core/dispatch.ts`).
Authority model per phase:

- **Phases 0–2: CPU `live` remains the single position authority.** The GPU
  driver uploads positions per step and returns vertices in its outcome like
  the worker driver does. No coherence risk exists yet.
- **Phase 3, frozen mode only: GPU buffer becomes the steady-state authority;
  `live` becomes a lazily-synced mirror.** (`'reassemble'` mode never gets
  the authority flip — it keeps the Phase 2-style per-step position exchange
  permanently, since its CPU line search consumes and produces positions
  every step; review-2 F10.) Mandatory readback points (each a full positions
  readback + `live` refresh, f64-recomputed energy): **pause/commit**,
  **pin add/remove/drag-start**, **driver fallback**, **auto-pause on
  rejected step**, **device loss (§6)**. The GPU driver's step outcome
  becomes stats-only in steady state; the outcome type gains an explicit
  variant for "positions resident on GPU" rather than silently-absent
  vertices. UI stats between readback points come from the per-frame stats
  readback, labeled as f32.

## 3. Verification: tolerance gates replace bit-identity

Bit-identity cannot cross into WebGPU — cross-vendor reproducibility is
explicitly not guaranteed by the spec (it's modeled as a fingerprinting
vector) [PREC Q5]. The GPU path is therefore verified against the **unchanged
CPU f64 path** with pre-registered tolerances:

| Gate | Quantity | Tolerance | Fixture set |
|---|---|---|---|
| T1 | per-pair kernel value vs f64 oracle | rel err < 1e-5 | committed unit pair fixtures incl. near-touch (gap ≥1e-6, extent/gap ≤4e6) |
| T2 | total energy vs f64 | rel err < 1e-6 | all committed presets + trefoil N=240/960 |
| T3 | gradient direction vs f64 | cosine > 1 − 1e-6 | same |
| T4 | solve residual (projected, rel, f64-measured) | ≤ 1e-10 after refinement; ≤4 outer refine iters | crossing (fixed N=8) + trefoil sweep N=60..960, both constraint modes |
| T5 | descent trajectory | energy after K=50 steps within ±1% of the f64 trajectory; no accepted step increases energy — **both judged in f64 recomputed from readback positions** (review finding 10); statistic = median of 3 runs, flaky if any run disagrees on pass/fail → investigate before merging | trefoil N=120 (total-length), trefoil N=120 (perEdge) |

Tolerances T1–T3 are from [PREC risk 3]; T4 from [PREC Q2]; T5 band is this
spec's addition. Gates run in the browser harness (headless CDP recipe in
project memory) with a **hard adapter precondition** (review finding 8): the
harness asserts `adapter.info` identifies the hardware adapter (Quadro RTX
3000 class, not SwiftShader) — otherwise the run is **INVALID**, not FAIL.
Results are committed JSON keyed by git SHA + adapter info, same culture as
`bench/results/`.

**Fixture debt (review-1 F9, review-2 F7/F8):** a parameterizable trefoil
generator exists only as a private function in `bench/sobolev.bench.ts`; the
presets have only a fixed-N=50 trefoil (`testConfigs.ts` id `'knot'`) and no
near-touch pair fixtures. Phase 0 extracts and commits the generator plus the
**full fixture matrix every gate names: trefoil N=60/120/240/480/960/1000
(both constraint modes where a gate says so) + the committed near-touch pair
fixtures (T1/G2)** — BEFORE any gate that names them can run.

## 4. Pre-registered kill gates (feasibility first, cheapest first)

Phase 0 is deliberately all-gates-no-product: every gate is falsifiable in
hours and each failure kills or reshapes the milestone before kernel work.

- **G0a — hardware adapter under the harness (FIRST Phase-0 task; review-3
  F7).** The known headless CDP recipe boots SwiftShader (project memory);
  every browser gate below is INVALID on a software adapter. PASS: headless
  (or, failing that, headed) Chrome launch flags that yield
  `adapter.info` identifying the Quadro RTX 3000-class hardware Vulkan
  adapter, committed as the harness's launch recipe. FAIL after honest
  effort → all gates run headed on the dev box; recipe documented either way.
- **G0t — GPU throughput probe.** f32 FMA and dense-matvec (n≈3000)
  microbenchmarks under G3 methodology; results committed to
  `bench/results/`. Not pass/fail — it is G5's estimator input and exists so
  the (a)-vs-(b) decision binds to measured numbers.
- **G1 — dispatch batching.** 250 no-op compute nodes: batched
  `renderer.compute([...])` vs 250 separate calls. PASS: batched total CPU
  cost < 2 ms and <0.25× the unbatched cost on this machine (Chrome/Vulkan,
  Quadro RTX 3000). FAIL → the per-frame iterative solve loop is dead;
  milestone reduces to kernels-only (accepting the ~1.3–1.5× Amdahl cap
  [PREC headline]) or is dropped.
- **G2 — two-float survives the compiler, tested through the PRODUCTION
  kernel** (review finding 6): the boot self-test runs the actual `wgslFn`
  pair kernel on a committed synthetic near-touch pair fixture
  (extent/gap = 10⁶) and asserts the T1 tolerance (rel err < 1e-5 vs the f64
  CPU value) — a standalone expression test proves nothing about the
  production dataflow. Runs per adapter at boot; failure ⇒ `'gpu'` driver
  falls back to `'worker'` [PREC risk 1].
- **G3 — timestamp benchmarking viable.** PASS: coefficient of variation of
  GPU-timestamp totals < 10% across 5 runs of a fixed ≥100-iteration workload
  (amortizing Dawn's 100 µs quantization; headless runs may instead disable
  quantization via `--enable-webgpu-developer-features` per [PREC Q5] — fits
  the existing CDP recipe). FAIL → all perf gates fall back to CPU wall-clock
  with single terminal sync; methodology note becomes mandatory in every
  result file [IMPL §6].
- **G4 — zero-readback render spike.** The wave samples are **CPU-computed
  and uploaded**; the compute pass does ONLY the §2.7 combine (hi+lo→f32) and
  per-edge interleave scatter into a fat-line `instanceStart/End` storage
  buffer rendered by the existing scene (review-3 F5 — a GPU-evaluated `sin`
  could never match CPU f32 exactly under WGSL's graphics-grade accuracy, and
  the plumbing is what's under test anyway). PASS, machine-checked: a one-off
  verification readback of the instance buffer matches the uploaded wave
  (f32 exact per element), and
  the steady-state loop issues zero per-frame readbacks (asserted by
  instrumentation counter). FAIL → Phase 3 replaced by the 1-readback/frame
  fallback permanently; ceiling noted in the plan.
- **G5 — solve-path decision experiment (Phase 2 entry; review-2 F1 —
  asymmetric by necessity).** Variant (b) is cheap to build (CG = matvec +
  axpy + reductions, all Phase 1 kernels) and is **measured**: p50 full-STEP
  time (gradient solve + line-search projection solves per §2.4, readbacks
  included) at N=480/960, both constraint modes, vs the CPU f64 LDLᵀ
  baseline, T4 green. Variant (a)'s GPU factorization does NOT exist at Phase
  2 entry and is **estimated** by a pre-registered method: counted
  factor/backsolve FLOPs × the **Phase-0 G0t throughput probe's** measured
  f32 FMA + dense-matvec rates (review-3 F3 — G1/G3 measure overhead and
  timing variance, not FLOP/s; without G0t this estimate is unexecutable) +
  the dispatch-cost model, plus an implementation-cost judgment (§6). **Bias policy: measured
  beats estimated — (b) is chosen unless (a)'s estimate wins by >1.5× at
  N=960**, in which case a minimal GPU-factorization spike (single f32
  blocked-LDLᵀ of a committed N=480 system, T4-checked) is budgeted to
  confirm the estimate BEFORE Phase 2 commits to (a). If (b) measured fails
  to beat CPU LDLᵀ at N=960 AND (a)'s confirmed estimate also fails → Phase 2
  FAILS; milestone ships kernels-only with the measured cap stated in the
  README.
- **G6 — perEdge conditioning.** Measure κ(K) growth for the perEdge
  constraint set at N=60..960 (power-iteration estimate acceptable). If
  κ·u_f32 > 0.1 at target N, perEdge mode falls back to CPU f64 solve and the
  spec's claims are narrowed to total-length mode [PREC risk 2].
- **G7 — end-state perf.** Frozen mode, trefoil, barycenter+totalLength.
  Two clauses (review-2 F5 — any N=1000 ratio baseline is vacuous because
  every alternative is factor-dominated into seconds there):
  (i) **absolute:** full descent step p50 ≥ 15 fps (≤66 ms) at N=1000;
  (ii) **earned-complexity ratio:** end-state full-step p50 at **N=480** ≥3×
  better than the kernels-only Phase 1 configuration's full-step p50 at the
  same N=480 (both sides measured at N=480 — the largest size where
  kernels-only is still frame-viable per the Phase 0 baselines; adjust to the
  measured viability edge if baselines say otherwise, recorded before Phase 2
  starts). Measured with G3 methodology, 5 runs, medians, same statistic
  culture as the LDLᵀ gate. FAIL → milestone does not flip any default;
  `'gpu'` stays opt-in experimental.

Perf-gate baselines at N=240/480/960/1000 do not exist yet (bench stops at
N=120); Phase 0 extends `bench/` to record them BEFORE any GPU kernel lands,
so G5/G7 compare against pre-registered numbers, not retrofitted ones.

## 5. Phasing

- **Phase 0 — gates & baselines (no product code):** G0a FIRST (hardware
  adapter under the harness), then G0t throughput probe, G1–G4 spikes, G6
  measurement, CPU baselines to N=1000, fixture extraction (§3 fixture debt),
  `trackTimestamp` Viewer boot change (§6), tolerance-gate harness skeleton
  (T1–T3 runnable against a trivially-wrong kernel to prove they can fail).
- **Phase 1 — gradient kernel:** dE gather kernel (TSL+wgslFn, two-float,
  tree reductions) behind the `'gpu'` driver. **Energy-source policy (review
  finding 2): the GPU computes dE ONLY; ALL energies — E₀ and every Armijo
  trial — stay CPU f64**, so §2.3's never-mix rule holds trivially and the
  line search is untouched. Interim per step: one positions upload + one
  gradient readback. Phase gate: T1/T3 green + GPU dE phase ≥5× the CPU dE
  phase at N≥480, **with the per-step gradient readback counted inside the
  GPU dE phase time** (anti-gaming; review-3 F9) — dE only; full-step speedup
  here is Amdahl-capped and NOT a gate. The GPU energy kernel is still written and
  T2-verified in this phase, but consumed only by gates, not by descent.
- **Phase 2 — the solve + frozen-mode GPU line search (one inseparable
  stage, §2.4/§2.5):** G5 experiment → chosen variant implemented, serving
  both the gradient solve and the line search's projection solves; τ-agreement
  gate on the GPU line search; T4 + T5 green; full-step speedup honestly
  re-measured. Positions still CPU-authoritative (per-step exchange).
- **Phase 3 — zero-readback rendering:** Curve.tsx rewrite per G4 spike
  (lines only; spheres/arrows hidden under `'gpu'` per §2.7); §2.8
  position-authority flip (frozen mode only) with its mandatory readback
  points; steady-state readback reduced to stats-only; G7 measured and
  recorded.

Each phase is a branch + PR with its own plan tasks; the spec's gates are the
merge criteria. Default `solverDriver` flips to `'gpu'` only after G7 passes
AND a week of dogfooding without fallback triggers (same caution as the LDLᵀ
default flip).

## 6. Risks not covered by gates

- **three.js API drift:** `trackTimestamp`/`resolveTimestampsAsync` are
  experimental; TSL compute surface is younger than the render surface. Pin
  `three@0.185.x`; re-run G1–G4 on any upgrade (add to the upgrade checklist).
- **6 GB VRAM (Quadro RTX 3000):** dense f32 A at N=1000 is 3000² × 4 B ≈ 36 MB
  — trivial; CSR pair lists at E≈1000 are ~8 MB. Memory is a non-issue at
  target sizes; revisit only if N>10⁴ ever becomes a target.
- **Interactive pin-drag** mutates positions mid-run from the UI; §2.8 makes
  pin operations mandatory readback points in Phase 3, and the GPU driver
  needs a small pin-target upload path (plan-level design).
- **Mid-run device loss** (review finding 14): `GPUDevice.lost` while the GPU
  buffer is the position authority strands the run. Recovery: fall back to
  `'worker'` from the last §2.8 readback point (pause/commit or stats-cadence
  snapshot), accepting loss of progress since then; surfaced to the UI as an
  auto-pause, never a silent restart.
- **GPU dense factorization complexity** (if G5 picks variant (a)): a WGSL
  blocked LDLᵀ/LU is the largest single deliverable in that branch and has no
  prior art in this stack; G5's estimate must include an implementation-cost
  judgment, not just projected solve times.
- **ε-semantics drift:** the GPU energy is the ε=0 energy [PREC Q1]. All
  fixtures with intentionally-degenerate edges must go through the explicit
  f32 degeneracy branch tests, not rely on ε.
- **`trackTimestamp` is a Viewer change** (review-3 F8): G3/G7 need
  `trackTimestamp: true` at `WebGPURenderer` construction; the current gl
  factory (`src/scene/Viewer.tsx:376-381`) doesn't pass it. Small, but it
  touches the render boot path — named here so it lands in Phase 0, not
  mid-phase.
- **Phase-3 pin PICKING staleness** (review-3 F10): the raycast that decides
  WHICH vertex a pin grabs runs against the stale `live` mirror before the
  pin readback point fires — under fast motion it can pick the wrong vertex.
  Disposition: pointer-down triggers the readback FIRST, then the raycast
  runs against fresh positions (one extra ~0.5 ms sync on a user gesture is
  free); recorded so the plan tests it.

## 7. Open questions (deferred to the plan, not blockers)

- Matrix-free vs materialized-A matvec for CG (G5 may answer implicitly).
- Whether Phase 1's gradient readback uses `getArrayBufferAsync` per step or
  double-buffered async with one frame of pipelining.
- Exact f32 degeneracy threshold derivation (needs a small error analysis in
  the plan; [PREC Q1] gives the scale argument).
- Stats/HUD cadence for the one-readback budget, and the Phase 3 stats-label
  UX (f32 numbers between f64 readback points).
