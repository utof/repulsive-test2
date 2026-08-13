# AI research: GPU floating-point precision strategy for the tangent-point WebGPU port

> **Provenance:** deep-research subagent report, delivered 2026-08-13, saved
> verbatim below (only this banner added). The agent measured against the real
> `src/core/tangentPointEnergy.ts` kernel via exact f32 emulation
> (`Math.fround`), built the real saddle system from `assembleAFlat` +
> `buildSaddleMatrix`, and had a sub-agent clone/grep the reference
> implementations. Scratch scripts referenced in the report lived in the
> session scratchpad and are ephemeral; the numbers are inlined. Companion doc:
> `docs/2026-08-13-ai-research-webgpu-compute.md` (implementation patterns —
> the two agree on the pre-registered tolerance-gate prerequisite).
> Context: issue utof/repulsive-test2#1; target realtime (>15 fps) at N>1000.

---

# GPU precision strategy for the tangent-point WebGPU port

## Headline

**Plain f32 is sufficient for the energy/gradient kernels, but only with two cheap changes: tree/pairwise summation (free on GPU) and two-float position differences (~6 extra flops/pair).** What nearly kills f32 is not summation error and not the line search — it is *catastrophic cancellation in vertex coordinate differences for near-touching strands*, amplified β=6-fold. Two-float positions fix that by up to 10⁶×.

**The bigger finding is strategic: precision is not your bottleneck, and the O(E²) kernels are not either.** Your own bench data shows the dense saddle solve already consumes 53–80% of a step at N=120, and it scales O(N³) against the kernels' O(E²). Porting only the pairwise kernels to GPU is Amdahl-capped at roughly 1.3–1.5× at N=120 and much worse at N=1000.

All numbers below are measured against the real kernel, not estimated. Scripts (read-only, nothing in the repo was modified): session-scratchpad `{f32sim,armijo,tail,refine,kap,twofloat}.ts`. They emulate f32 with `Math.fround` on f64 intermediates, which is *exact* f32 for `+ - * / sqrt` (f64 carries more than 2p+2 bits of the f32 format, so no double-rounding error).

---

## Q1 — Is f32 enough for energy/gradient? What breaks first?

I ranked the four candidate failure modes you named by measuring each against `src/core/tangentPointEnergy.ts`.

**Summation error: real but entirely fixable.** Naive left-to-right accumulation over disjoint pairs degrades with N as theory predicts; a binary tree reduction recovers essentially full f32 accuracy:

| N | pairs | naive f32 rel err | pairwise f32 rel err |
|---|---|---|---|
| 128 | 16,000 | 1.9e-6 | 6.2e-8 |
| 512 | 260,608 | 2.3e-5 | 1.9e-9 |
| 1000 | 997,000 | 2.8e-5 | 5.1e-8 |

A ~500× improvement, landing at the f32 unit roundoff floor (2⁻²⁴ = 6.0e-8). Costs nothing to adopt — a workgroup tree reduction is the natural GPU form anyway. **Do not use Kahan or two-sum compensation**; see Q3, it does not survive WGSL.

**WGSL's weaker `pow` is a non-issue here.** WGSL specifies `pow(x,y)` as `exp2(y*log2(x))` rather than a correctly-rounded pow. I emulated that chain in f32 and it was indistinguishable at your α=3, β=6 with O(1) operands (1.9e-6 vs 1.9e-6 naive at N=128; 6.5e-8 vs 5.1e-8 pairwise at N=1000). Re-check if α/β ever grow — the error scales with |y·log₂x|.

**The ε=1e-10 regularization is a confirmed no-op in f32.** Measured at N=128:

```
eps=0e+0    E64=1.6617842578e+1   E32=1.6617843628e+1
eps=1e-10   E64=1.6617842597e+1   E32=1.6617843628e+1   <- bit-identical to eps=0
```

ε changes the f64 energy in the 9th digit and changes the f32 energy not at all. Not fatal — ε's job is regularization against exact degeneracy, not accuracy — but it means **the f32 kernel is mathematically the ε=0 energy**, and degeneracy protection must come from an explicit branch. Related hazard at `src/core/tangentPointEnergy.ts:173` and `:211`: the `safeUnit` guards test pre-ε lengths against `< 1e-14`. In f64 a near-degenerate length lands near 1e-16 and can trip the guard; in f32 it lands near 1e-7·scale and never trips. Those thresholds must be re-derived for f32, not copied.

**The Armijo line search is safe by five orders of magnitude — this surprised me.** I instrumented real `sobolevStep` descent on a trefoil. Relative energy drop per accepted step:

- N=128, 60 steps: min 9.8e-3, median 1.7e-2
- N=64, 3000 steps: starts at 9.7e-2, decays roughly as 1/iteration, still **2.4e-4 at iteration 2800**, never once dips below 1e-6

Against a pairwise-f32 noise floor of ~6e-8 that is 4–6 orders of margin; reaching the floor would take ~10⁶ iterations. Even naive summation at N=1000 (2.8e-5) retains ~10× margin, though I would not rely on it. Secondary result: a uniformly-scaled "tight" start (0.05×) behaved identically, because the error depends on the *ratio* of extent to closest approach, which uniform scaling preserves.

**The actual first failure is coordinate cancellation at near-touching strands.** Two circles of extent ~4 at controlled separation, plain f32 throughout:

| gap | extent/gap | rel err (plain f32) |
|---|---|---|
| 1e-2 | 4e2 | 2.9e-6 |
| 1e-3 | 4e3 | 1.4e-4 |
| 1e-5 | 4e5 | 4.1e-3 |
| 1e-6 | 4e6 | **1.5e-1** |

Mechanism: `d = p_i − p_j` between coordinates of magnitude ~2 known only to 6e-8 relative, so a gap of 1e-6 is itself known only to ~6% — and β=6 amplifies that to ~36%. Error scales as roughly `β·u_f·(extent/gap)`. This is a property of *storing positions in f32*, not of the arithmetic: rounding only the input positions and computing in exact f64 still gave gradient cosine similarity 0.999999991 against the f64 gradient — that is the floor you pay for f32 storage alone.

**The fix, and it is dramatic.** Store positions as two f32 (`hi = f32(p)`, `lo = f32(p − hi)`) and compute differences as `(hi_i − hi_j) + (lo_i − lo_j)`, leaving everything downstream plain f32. This is Godot's large-world rendering trick (https://godotengine.org/article/emulating-double-precision-gpu-render-large-worlds/), which likewise emulates *only* the translation and leaves rotation/scale single:

| gap | plain f32 | two-float positions | gain |
|---|---|---|---|
| 1e-3 | 1.4e-4 | 2.0e-7 | 695× |
| 1e-5 | 4.1e-3 | 2.3e-7 | 17,487× |
| 1e-6 | 1.5e-1 | 1.5e-7 | **1,031,628×** |
| 1e-7 | 4.1e-1 | 1.2e-4 | 3,533× |

Cost is one extra subtract and add per coordinate difference — ~6 flops against ~50 per pair. Crucially this is **not** a two-sum: it never needs to capture the rounding error of an operation, so it is far more robust to compiler transformation than Kahan (caveat in Q3).

One reassuring dynamic: during descent the minimum disjoint-pair distance *grows* monotonically (0.21 → 0.82 over 55 steps at N=128) — the flow moves away from the dangerous regime. The risk is concentrated in the user's *initial* tangled configuration.

## Q2 — Mixed-precision iterative refinement for the saddle solve

Standard practice, and it works on your actual system. Governing theory is Carson & Higham's three-precision analysis (https://eprints.maths.manchester.ac.uk/2562/, and Higham's summary https://nhigham.com/2017/07/26/accelerating-the-solution-of-linear-systems-by-iterative-refinement-in-three-precisions/): factor in u_f, work in u, compute residuals in u_r. Standard LU-IR converges while κ(A)·u_f ≲ 1; substituting a GMRES solve preconditioned by the LU factors (GMRES-IR) relaxes this to κ ≤ 10⁸ for (working, residual) = (f32, f64). Broader context: https://arxiv.org/pdf/2007.06674.

I built the real saddle system K from `assembleAFlat` + `buildSaddleMatrix` with the barycenter constraint block, factored and solved entirely in f32, and refined with f64 residuals:

| N | size | κ₂(K) | it0 rel resid | it1 | it2 |
|---|---|---|---|---|---|
| 32 | 99 | 3.0e1 | 6.7e-7 | 4.9e-13 | — |
| 64 | 195 | 1.9e2 | 6.0e-6 | 6.6e-11 | — |
| 128 | 387 | 1.8e3 | 1.6e-4 | 8.0e-8 | 5.1e-11 |

**One to three refinement iterations reach your 1e-10 relative-residual gate.** The saddle structure caused no trouble at these sizes; partial pivoting handled the indefiniteness.

The concern is extrapolation. κ₂(K) at N = 32/64/128/192/256 gives 3.0e1 / 1.9e2 / 1.7e3 / 6.7e3 / 1.7e4 — a clean power law **κ ∝ N^3.3** in the tail. Extrapolating to N=1000 gives κ ≈ 1.6e6, so κ·u_f ≈ 0.09: inside the standard LU-IR convergence condition, but with only about one order of magnitude of headroom. At N=1000 f32 LU-IR is *marginal*, and GMRES-IR is the documented fallback. Two caveats I could not close: this is the barycenter-only constraint set (the `perEdge` set adds N constraints and is plausibly worse-conditioned, untested), and my κ values are power-iteration estimates, not exact SVDs.

## Q3 — Emulated double precision in WGSL

**Compensated arithmetic is not portable in WGSL — the single most important spec finding.** WGSL §15.7.5 (W3C CR Draft, 6 Aug 2026, https://www.w3.org/TR/WGSL/):

> "An implementation **may reassociate** operations. An implementation may fuse operations if the transformed expression is at least as accurate as the original formulation."

The asymmetry is deliberate — fusion is gated on accuracy, reassociation is unconditional. That was the resolution of https://github.com/gpuweb/gpuweb/issues/2402, where the editor argued you cannot statically determine whether a reassociation is at least as accurate. Dekker/Knuth two-sum, TwoProduct, Kahan and Neumaier all depend on `(a+b)` being *rounded* and then `((a+b)−a)` evaluated on that rounded value; reassociation or FMA fusion collapses the error term to zero.

No escape hatch exists. WGSL has no `precise` keyword; `@invariant` is restricted to the vertex `position` builtin. The standardization request is https://github.com/gpuweb/gpuweb/issues/2076, open since 2021 and filed for *exactly* this problem ("The algorithm in the example above is emulated double precision"). Unlikely to land — dneto0's objection is that Vulkan/D3D/GL all behave as if fast-math is always on, so a strict mode would need a CTS nobody can pass.

Worse, **Metal defaults to fast-math under WebGPU**. Dawn's `ShaderModuleMTL.mm` uses `GetStrictMath().value_or(false)`, emitting `#pragma METAL fp math_mode(relaxed)` on macOS 15+ or `fastMathEnabled = true` on older systems. D3D is IEEE-strict by default; Vulkan has no handling at all. Chrome 131 added a `strictMath` flag on `GPUShaderModuleDescriptor`, but it is `[RuntimeEnabled=WebGPUDeveloperFeatures]` — behind `chrome://flags`, Metal/D3D only, not in the W3C spec. Useful to *measure* the cost of fast-math locally; not shippable.

Cost: the commonly cited figure for full df64 emulation is roughly **4× slower than f32**; technique documented in https://arxiv.org/pdf/2408.09699 and http://blog.hvidtfeldts.net/index.php/2012/07/double-precision-in-opengl-and-webgl/. I do not recommend it. The targeted two-float-*positions* trick from Q1 buys the accuracy that actually matters at ~1.1× cost instead of 4×.

**Residual risk on the two-float trick:** `(hi_i − hi_j) + (lo_i − lo_j)` could in principle be reassociated to `(hi_i + lo_i) − (hi_j + lo_j)`, silently collapsing it to plain f32. Much less fragile than two-sum (needs no captured rounding error), but permitted. Must be verified at runtime per device, not assumed — see kill gates.

## Q4 — Prior art: everyone uses double, nobody has tried f32

Delegated to a subagent that cloned and grepped actual sources. Unusually clean result: **every implementation in the repulsive-curves lineage is double-precision, and the choice appears never to have been examined.**

- **icethrush/repulsive-curves** — no scalar typedef; `double` spelled out in every signature (`include/tpe_energy_sc.h:27-41`). `float` appears only in marching-cubes visualization. Guards at `src/tpe_energy_sc.cpp:309,346,387` are `if (proj_len < 1e-10)`; flow thresholds `1e-15` and `1e-10` — all meaningless in f32. Zero mentions of precision across the README and all 11 issues.
- **HenrikSchumacher/Repulsor** — templated on `Real`, but `Example/main.cpp:38` says it outright: `using Real = double;  // Everything else but double won't work.` Not mechanically enforced (the static_assert would pass float), so treat it as the author's empirical claim. The precision-critical constant is `src/GJK.hpp:79`, `eps = sqrt(machine_eps)` — 1.49e-8 in double, **3.45e-4 in float**, a 23,000× looser geometric predicate. Notably his `Power` already trades accuracy for speed *in double* via `exp2(y*log2(x))` — the same formula WGSL mandates.
- **repulsive-surfaces** — `typedef double mreal` plus a cache-line layout literally sized in doubles (`optimized_bct_types.h:7,40-43`). **repulsive-shells** likewise; its vendored Repulsor snapshot contains an *entirely commented-out* `float` specialization of `pow` (`deps/repulsion/src/MyMath/MyMath.h:197-290`) — a float path was written and abandoned, reason unrecorded.
- **GPU ports** — exactly one exists, `pauljoohyunkim/partC_Dissertation` (Oxford dissertation, CUDA), double end to end (`norm3d` not `norm3df`; zero `float` tokens in the release kernels). The dissertation never discusses the choice. No WebGPU/WGSL/compute-shader TPE port exists anywhere on GitHub.

Transferable expertise lives in the FMM/n-body world instead, and it validates the architecture recommended here. **exafmm** has an explicit `--enable-single` switch (default double) with paired epsilons `1e-8f`/`1e-16`, and — tellingly — computes its *double* rsqrt from an f32 seed plus two Newton–Raphson steps (`include/vec.h:766-776`). **Bonsai** (GPU Barnes-Hut) is textbook mixed precision: forces entirely f32 with `rsqrtf` and Plummer softening, but multipole accumulation, bounding boxes and all energy diagnostics in f64 (`compute_propertiesD.cu`, `octree.h:339-341`). Consensus pattern: *f32 for the pairwise kernel with explicit softening or a mask, f64 for anything summed over many terms, f32-seed + Newton where you want f64 accuracy at f32 throughput.*

## Q5 — WebGPU spec status

- **f64: not happening.** https://github.com/gpuweb/gpuweb/issues/2805 open since 2022. Corentin Wallez, Aug 2025: "This hasn't started yet, IDK when the group will get to it… on consumer GPU the rate of computation with f64 is often much slower than for f32 (8x slower is usual)."
- **f32 accuracy is graphics-grade, not IEEE.** `+ − *` correctly rounded, but `/` is 2.5 ULP, `inverseSqrt` 2 ULP, **`sqrt` is defined as `1.0/inverseSqrt(x)`** (~4.5 ULP), `pow` as `exp2(y*log2(x))`, `exp2` as `3 + 2|x|` ULP, and **`dot` has no specified summation order**. Your `length`/`normalize` calls inherit all of it.
- **Bit-identity across vendors is not guaranteed and is explicitly modeled as a fingerprinting vector** (WebGPU CR §2.2.2: "machine-specific rasterization/precision artifacts… precision fingerprints are identical across most or all of the devices of each vendor"). **Your CPU-side bit-identity discipline cannot cross to WGSL.** Every GPU A/B gate must be tolerance-based.
- **Flush-to-zero is permitted** on inputs *and* outputs of essentially every arithmetic op (§15.7.2). ε=1e-10 is comfortably normal in f32 (min normal 1.18e-38), so the constant is safe; the exposure is *products* — `d*d` goes subnormal below |d|≈1.1e-19 and hard zero below 1.2e-22. Also `min`/`max` may return either operand if both are subnormal, and implementations may assume no NaN/Inf, so NaN-based guards are not portable.
- **Subgroups: shipped in Chrome stable since 134** (March 2025), `enable subgroups;`, with `subgroupAdd`, ballot, shuffle. But no summation order is specified and subgroup size varies by vendor (32 NVIDIA, 32/64 AMD, 8/16/32 Intel), so a float `subgroupAdd` yields a different reduction tree per vendor. For reproducibility prefer a fixed-fanout workgroup-memory tree.
- **Timestamp queries: stable and flag-free since Chrome 121**, but **quantized to 100 µs** by default (Dawn `timestamp_quantization`). Amortize over ≥100 iterations, or disable quantization in headless CI via `--enable-webgpu-developer-features` — fits the existing CDP recipe in memory.
- **shader-f16**: shipping since Chrome 120, irrelevant and harmful here — f16's smallest normal is 6.1e-5, so ε=1e-10 is subnormal and flushes to zero.

---

## Recommended precision architecture

**Stage 1 — pairwise kernels (energy, gradient, A assembly): f32 on GPU.**
- Positions uploaded as two f32 buffers (`hi`, `lo`); all coordinate differences computed as `(hi_i − hi_j) + (lo_i − lo_j)`. Everything downstream plain f32.
- Accumulation via fixed-fanout workgroup tree reduction. No Kahan, no two-sum — they do not survive WGSL reassociation.
- Prefer repeated multiplication over `pow` for integer exponents; keep `pow` only for the α=3 / 1−α non-integer paths.
- Re-derive the `< 1e-14` degeneracy guards for f32 scale; do not port the constants.
- Final energy scalar reduced to f64 on the CPU (cheap last stage, removes any residual summation concern for the Armijo comparison).

**Stage 2 — saddle solve: f32 factor with f64 iterative refinement.**
- Factor and triangular-solve in f32 (GPU or CPU); residual `r = b − Kx` and update `x ← x + d` in f64 on the CPU. Measured 1–3 iterations to your 1e-10 gate at N ≤ 128.
- Keep GMRES-IR in reserve for N ≳ 500, where κ·u_f approaches the LU-IR limit.

**Stage 3 — stays CPU f64, permanently.**
- The Armijo comparison itself, the constraint projection to 1e-10, the residual/update arithmetic in refinement, and the barycenter/length constraint targets. All O(N) or O(N²) and cost nothing.

**Sequencing.** Do Stage 2 before or alongside Stage 1. `bench/results/2026-07-06-ldlt-ab.json` shows `saddle` at 53–80% of a step at N=120 while the O(E²) phases are 19–47%; the solve is O(N³) against the kernels' O(E²), so at N=1000 the gap widens by roughly two orders of magnitude. A kernels-only GPU port is Amdahl-capped near 1.3–1.5×.

One consolation that de-risks the whole precision question: `docs/2026-07-04-ai-research-multigrid.md` reports BCT matvec relative errors of **1.8e-2 (B) and 8.1e-2 (B⁰) at θ=0.5**. If Barnes-Hut/BCT is on the roadmap at all, its approximation error exceeds f32 rounding error by five orders of magnitude. **f32 is numerically free relative to the approximation you are already planning to accept.**

## Top 3 risks to pre-register kill gates for

**1. Reassociation silently collapsing the two-float position trick.** The entire near-touch accuracy argument (10⁶× at gap=1e-6) rests on `(hi_i − hi_j) + (lo_i − lo_j)` not being regrouped, and WGSL §15.7.5 permits regrouping unconditionally with Metal defaulting to fast-math. *Gate:* a boot-time GPU self-test computing a known difference with extent/gap = 10⁶, compared against the f64 CPU value; require relative error < 1e-5, else fall back to CPU. Run per adapter, not once — it is a per-vendor, per-backend property. Cheap, and it is the difference between the port working and being subtly wrong only on Apple hardware.

**2. κ(K) outrunning f32 iterative refinement at large N.** Measured κ ∝ N^3.3 extrapolates to ≈1.6e6 at N=1000, giving κ·u_f ≈ 0.09 — inside the convergence condition but with only ~10× headroom, and untested for the `perEdge` constraint set which adds N constraints. *Gate:* instrument the refinement loop; require the 1e-10 relative residual within ≤4 iterations, and abort the f32-factor path to CPU f64 if a condition estimate exceeds κ·u_f > 0.1. Measure `perEdge` explicitly before committing.

**3. Loss of the bit-identity regression discipline.** The repo's entire verification story — Python/numpy oracles, golden suites, the LDLT swap proven bit-identical — depends on reproducibility that WebGPU explicitly does not provide and which the spec treats as a *vendor fingerprint*. *Gate:* before any GPU code lands, define the tolerance-based replacements: per-pair kernel relative error < 1e-5 against the f64 oracle, total energy relative error < 1e-6, gradient cosine similarity > 1−1e-6, and a descent-trajectory gate (energy after K steps within a stated band of the f64 trajectory). Without these written down first, there is no way to tell a working port from a broken one.

Secondary item worth watching but not gating: ε=1e-10 is provably inert in f32, so the GPU kernel computes the ε=0 energy. Fine for accuracy, but degeneracy protection must be an explicit branch, and it makes the GPU and CPU kernels *different energies* at the 1e-9 level — which matters if you ever mix them within one descent run.
