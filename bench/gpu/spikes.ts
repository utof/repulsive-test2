// bench/gpu/spikes.ts — spike registry; each spike returns JSON-serializable data.
// @see docs/superpowers/plans/2026-08-13-webgpu-solver-phase0.md Task 1

import * as THREE from 'three/webgpu';
import { nearTouchPair, splitHiLo } from '../../src/core/fixtures';
import { DEFAULTS } from '../../src/core/optimizer';
import type { Vec3 } from '../../src/core/testConfigs';

type SpikeResult = Record<string, unknown>;
const spikes: Record<string, () => Promise<SpikeResult>> = {};

/**
 * Minimal shape of the fields this file reads off `renderer.backend.device`.
 * NOT `GPUDevice` from lib: the repo has no `@webgpu/types` dependency and
 * `@types/three`'s ambient `GPUDevice` is an intentionally-empty stub
 * (node_modules/@types/three/src/renderers/webgpu/WebGPUBackend.d.ts:4), so
 * `.adapterInfo`/`.limits` would not typecheck against it. This local type
 * documents the real runtime shape (WebGPU spec `GPUDevice.adapterInfo` /
 * `GPUAdapterInfo`) instead of widening to `any`.
 * @see docs/superpowers/plans/2026-08-13-webgpu-solver-phase0.md Task 1
 */
type DeviceWithAdapterInfo = {
    adapterInfo: {
        vendor: string;
        architecture: string;
        device: string;
        description: string;
    };
    limits: {
        maxComputeWorkgroupSizeX: number;
    };
};

/** G0a: report the adapter we actually got. INVALID-vs-FAIL decisions happen
 * in the driver, not here. @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §4 G0a */
spikes.adapterInfo = async () => {
    const renderer = new THREE.WebGPURenderer();
    await renderer.init();
    // renderer.backend.device is the one and only GPUDevice — verified at
    // WebGPUBackend.js:111 ("A reference to the device." / `this.device`)
    // and WebGPUBackend.js:292 (`this.device = device;`) [plan Task 1 verify step].
    const device = (renderer.backend as unknown as { device: DeviceWithAdapterInfo }).device;
    const info = device.adapterInfo; // GPUAdapterInfo: vendor/architecture/device/description
    return {
        vendor: info.vendor,
        architecture: info.architecture,
        device: info.device,
        description: info.description,
        limits: { maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX },
    };
};

/**
 * G3: CV of GPU-timestamp totals over 5 runs of a fixed ≥100-dispatch
 * workload — establishes whether `trackTimestamp`/`resolveTimestampsAsync`
 * are viable for the rest of the perf gates, or whether Dawn's 100 µs
 * timestamp quantization forces a CPU-wall-clock fallback.
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §4 G3
 * @see docs/superpowers/plans/2026-08-13-webgpu-solver-phase0.md Task 5
 */
spikes.g3 = async () => {
    const renderer = new THREE.WebGPURenderer({ trackTimestamp: true });
    await renderer.init();
    const { Fn, instancedArray, instanceIndex, Loop } = await import('three/tsl');
    const buf = instancedArray(65536, 'float');
    // Inner-loop the FMA chain so the fixed 100-dispatch workload totals
    // ≥~10 ms of GPU time — Dawn quantizes timestamps to 100 µs, so CV<10%
    // needs total ≫ the quantum [IMPL §6, PREC Q5]. Loop count tuned
    // empirically against the real Quadro RTX 3000 (phase0 T5); see
    // bench/gpu/README.md for the measured total and the flag-recipe
    // decision if quantization still dominated.
    const kernel = Fn(() => {
        const e = buf.element(instanceIndex);
        const acc = e.toVar();
        Loop(4096, () => {
            acc.assign(acc.mul(1.000001).add(0.5));
        });
        e.assign(acc);
    })().compute(65536);
    const totals: number[] = [];
    for (let run = 0; run < 5; run++) {
        // 100 DISTINCT dispatches of the same pipeline, batched into one
        // renderer.compute() array/one command-encoder submit — verified
        // against Renderer.js:2718-2758 and WebGPUBackend.js:1600-1683 that
        // this dispatches 100 times (the shared-node pipeline/bindings cache
        // does not dedupe the dispatchWorkgroups() calls) [plan Task 5 verify
        // step].
        renderer.compute(Array.from({ length: 100 }, () => kernel));
        await renderer.resolveTimestampsAsync(THREE.TimestampQuery.COMPUTE);
        // renderer.info.compute.timestamp is overwritten (not accumulated)
        // on each resolveTimestampsAsync call — verified at
        // WebGPUTimestampQueryPool.js:94-121 (currentQueryIndex/queryOffsets
        // reset inside _resolveQueries before the next batch of queries can
        // be allocated) — so no renderer.info.reset() is needed between runs
        // [plan Task 5 verify step].
        totals.push(renderer.info.compute.timestamp);
    }
    const mean = totals.reduce((a, b) => a + b) / totals.length;
    const sd = Math.sqrt(
        totals.map((t) => (t - mean) ** 2).reduce((a, b) => a + b) / totals.length,
    );
    return { totalsMs: totals, mean, cv: sd / mean, pass: sd / mean < 0.1 };
};

/**
 * Deterministic xorshift32 PRNG — no `Math.random`, so seeded data (the
 * matvec matrix in particular) is reproducible across runs/machines.
 * @see docs/superpowers/plans/2026-08-13-webgpu-solver-phase0.md Task 6
 */
function xorshift32(seed: number): () => number {
    let s = seed >>> 0 || 1;
    return () => {
        s ^= s << 13;
        s >>>= 0;
        s ^= s >>> 17;
        s ^= s << 5;
        s >>>= 0;
        return s / 4294967296;
    };
}

function median(xs: number[]): number {
    const sorted = [...xs].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

/**
 * Minimal shape of `GPUDevice.queue` this file reads — see
 * `DeviceWithAdapterInfo` above for why a local type instead of the (empty
 * stub) ambient `@types/three` `GPUDevice`.
 * @see docs/superpowers/plans/2026-08-13-webgpu-solver-phase0.md Task 7
 */
type GPUQueueLike = {
    onSubmittedWorkDone: () => Promise<void>;
};

/**
 * G0t: GPU throughput probe (FMA rate + dense f32 matvec), under G3
 * methodology (batched dispatches, GPU timestamps, 5 runs/medians) since G3
 * PASSed. Not pass/fail — feeds the G5 kernel-cost estimator.
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §4 G0t
 * @see docs/superpowers/plans/2026-08-13-webgpu-solver-phase0.md Task 6
 */
spikes.g0t = async () => {
    const renderer = new THREE.WebGPURenderer({ trackTimestamp: true });
    await renderer.init();
    const { instancedArray, attributeArray, instanceIndex, wgslFn } = await import('three/tsl');

    // ---- 1. FMA rate: raw-WGSL loop (not TSL Loop) so the multiply-add
    // chain can't be folded by the TSL node compiler, plus a per-thread
    // data-dependent seed (buf[index], not a literal) so the driver
    // compiler can't constant-fold it across threads either
    // [plan Task 6 Step 1]. wgslFn storage-buffer-parameter syntax
    // (`ptr<storage, array<f32>, read_write>` + named-object call args)
    // verified against https://discourse.threejs.org/t/how-to-use-storagebufferattribute-as-a-input-to-wgslfn/73006
    // and https://blog.maximeheckel.com/posts/field-guide-to-tsl-and-webgpu/
    // (fetched this session — no local three.js example ships this pattern).
    const FMA_N = 1 << 22; // 2**22 threads
    const FMA_ITERS = 64;
    const rngFma = xorshift32(0x9e3779b9);
    const fmaSeed = new Float32Array(FMA_N);
    for (let i = 0; i < FMA_N; i++) fmaSeed[i] = rngFma();
    const fmaBuf = attributeArray(fmaSeed, 'float');

    const fmaKernelFn = wgslFn(`
        fn fmaKernel(buf: ptr<storage, array<f32>, read_write>, index: u32) -> void {
            var acc = buf[index];
            for (var i = 0u; i < 64u; i = i + 1u) {
                acc = acc * 1.0000001 + 0.5;
            }
            buf[index] = acc;
        }
    `);
    const fmaCompute = fmaKernelFn({ buf: fmaBuf, index: instanceIndex }).compute(FMA_N);

    // Batched like G3 (one renderer.compute() array/one submit per run) so a
    // sub-ms single dispatch doesn't fall prey to Dawn's 100 µs timestamp
    // quantization [IMPL §6]; repeat count tuned during implementation.
    const FMA_REPEATS = 20;
    const fmaTotalsMs: number[] = [];
    for (let run = 0; run < 5; run++) {
        renderer.compute(Array.from({ length: FMA_REPEATS }, () => fmaCompute));
        await renderer.resolveTimestampsAsync(THREE.TimestampQuery.COMPUTE);
        fmaTotalsMs.push(renderer.info.compute.timestamp);
    }
    const fmaMedianMs = median(fmaTotalsMs);
    const fmaFlopsPerDispatch = FMA_N * FMA_ITERS * 2; // 1 mul + 1 add per iter
    const fmaGflops = (FMA_REPEATS * fmaFlopsPerDispatch) / (fmaMedianMs * 1e6);

    // ---- 2. Dense matvec: y = M*x, one thread per row, 3072x3072 f32 M
    // seeded deterministically CPU-side and uploaded [plan Task 6 Step 1].
    const MV_N = 3072;
    const rngMv = xorshift32(0xc2b2ae35);
    const matData = new Float32Array(MV_N * MV_N);
    for (let i = 0; i < matData.length; i++) matData[i] = rngMv() * 2 - 1;
    const xData = new Float32Array(MV_N);
    for (let i = 0; i < MV_N; i++) xData[i] = rngMv() * 2 - 1;

    const matBuf = attributeArray(matData, 'float').toReadOnly();
    const xBuf = attributeArray(xData, 'float').toReadOnly();
    const yBuf = instancedArray(MV_N, 'float');

    const matvecFn = wgslFn(`
        fn matvecKernel(
            M: ptr<storage, array<f32>, read>,
            x: ptr<storage, array<f32>, read>,
            y: ptr<storage, array<f32>, read_write>,
            row: u32
        ) -> void {
            var acc = 0.0;
            let base = row * 3072u;
            for (var col = 0u; col < 3072u; col = col + 1u) {
                acc = acc + M[base + col] * x[col];
            }
            y[row] = acc;
        }
    `);
    const matvecCompute = matvecFn({ M: matBuf, x: xBuf, y: yBuf, row: instanceIndex }).compute(
        MV_N,
    );

    // A single matvec is bandwidth-bound and sub-ms [plan Task 6 sanity
    // band]; batch like the FMA probe above so timestamps clear the
    // quantization floor, then divide back out to ms/matvec.
    const MV_REPEATS = 200;
    const matvecTotalsMs: number[] = [];
    for (let run = 0; run < 5; run++) {
        renderer.compute(Array.from({ length: MV_REPEATS }, () => matvecCompute));
        await renderer.resolveTimestampsAsync(THREE.TimestampQuery.COMPUTE);
        matvecTotalsMs.push(renderer.info.compute.timestamp);
    }
    const mvMedianMs = median(matvecTotalsMs);
    const matvecMs = mvMedianMs / MV_REPEATS;
    const matvecFlops = 2 * MV_N * MV_N; // 1 mul + 1 add per (row, col)
    const matvecGflops = matvecFlops / (matvecMs * 1e6);

    return {
        fmaGflops,
        fmaTotalsMs,
        matvecMs,
        matvecGflops,
        matvecTotalsMs,
        method: 'timestamp',
    };
};

/**
 * G1 — dispatch-batching kill gate. 250 DISTINCT no-op compute nodes (each
 * `Fn(() => {...})()` call below closes over a different `i`, so this is a
 * genuinely different node graph per index, not the same node object reused
 * — required so the pipeline/bindings cache can't dedupe the unbatched arm
 * into something artificially cheap) [spec §4 G1 "DISTINCT node objects"].
 * Arm A: one `renderer.compute([...])` call with all 250. Arm B: 250
 * separate `renderer.compute(node)` calls. Both arms submit ALL dispatches
 * before the single terminal `device.queue.onSubmittedWorkDone()` sync per
 * arm/run — "sequential dispatches with a single terminal sync", never a
 * sync per dispatch, or the ~20x sync-conflation artifact makes batching
 * look miraculous and non-batching look catastrophic for the wrong reason.
 * The timed region is the submission loop only; the terminal sync is a
 * barrier between runs (drains the queue so run N+1's CPU submission timing
 * isn't polluted by run N's backlog), not part of the measured CPU cost.
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §4 G1
 * @see docs/2026-08-13-ai-research-webgpu-compute.md §4 (single-terminal-sync methodology), (c)2
 * @see docs/superpowers/plans/2026-08-13-webgpu-solver-phase0.md Task 7
 */
spikes.g1 = async () => {
    const renderer = new THREE.WebGPURenderer();
    await renderer.init();
    const { Fn, instancedArray } = await import('three/tsl');

    const N = 250;
    const buf = instancedArray(N, 'float');
    const nodes = Array.from({ length: N }, (_, i) =>
        Fn(() => {
            const e = buf.element(i);
            e.assign(e.add(1));
        })().compute(1),
    ); // compute(1): 1 thread == 1 workgroup per node [plan Task 7].

    const device = (renderer.backend as unknown as { device: { queue: GPUQueueLike } }).device;

    // Warm-up (both arms compile/cache all 250 pipelines once here) so
    // neither arm's timed runs pay first-use shader-compile cost unevenly.
    renderer.compute(nodes);
    await device.queue.onSubmittedWorkDone();
    for (const node of nodes) renderer.compute(node);
    await device.queue.onSubmittedWorkDone();

    const batchedRuns: number[] = [];
    const unbatchedRuns: number[] = [];
    for (let run = 0; run < 5; run++) {
        const t0 = performance.now();
        renderer.compute(nodes);
        const t1 = performance.now();
        await device.queue.onSubmittedWorkDone(); // terminal sync, outside the timed region
        batchedRuns.push(t1 - t0);

        const t2 = performance.now();
        for (const node of nodes) renderer.compute(node);
        const t3 = performance.now();
        await device.queue.onSubmittedWorkDone(); // terminal sync, outside the timed region
        unbatchedRuns.push(t3 - t2);
    }

    const batchedMs = median(batchedRuns);
    const unbatchedMs = median(unbatchedRuns);
    const ratio = batchedMs / unbatchedMs;
    return {
        batchedMs,
        unbatchedMs,
        ratio,
        batchedRuns,
        unbatchedRuns,
        pass: batchedMs < 2 && batchedMs < 0.25 * unbatchedMs,
    };
};

/**
 * CPU f64 reference quantity for the G2 spike: the sumK term of ONE (I,J)
 * edge pair (the 4 endpoint-combination sum, PRE the `0.25 * ell_I^(1-a) *
 * ell_J` scaling) — ported VERBATIM (op order + ε-after-norm placement) from
 * `calculateEnergy`'s inner loop body [tangentPointEnergy.ts:60-100], scoped
 * to a single (I,J) instead of the full O(E²) sum. This is what the GPU
 * kernel below reproduces; NOT the full curve energy.
 * @see docs/superpowers/specs/2026-07-01-tangent-point-hotpath-optimization-design.md
 * @see docs/superpowers/plans/2026-08-13-webgpu-solver-phase0.md Task 8
 */
function tangentPointKernelPieceCPU(
    vertices: Vec3[],
    i1: number,
    i2: number,
    j1: number,
    j2: number,
    alpha: number,
    beta: number,
    epsilon: number,
): number {
    const eIx = vertices[i2][0] - vertices[i1][0];
    const eIy = vertices[i2][1] - vertices[i1][1];
    const eIz = vertices[i2][2] - vertices[i1][2];

    let sumK = 0;
    for (const i of [i1, i2]) {
        for (const j of [j1, j2]) {
            const dx = vertices[i][0] - vertices[j][0];
            const dy = vertices[i][1] - vertices[j][1];
            const dz = vertices[i][2] - vertices[j][2];
            const d_norm = Math.sqrt(dx * dx + dy * dy + dz * dz) + epsilon; // ε after norm — tangentPointEnergy.ts:90
            const cx = eIy * dz - eIz * dy;
            const cy = eIz * dx - eIx * dz;
            const cz = eIx * dy - eIy * dx;
            const c_norm = Math.sqrt(cx * cx + cy * cy + cz * cz) + epsilon; // ε after norm — tangentPointEnergy.ts:97
            sumK += Math.pow(c_norm, alpha) / Math.pow(d_norm, beta);
        }
    }
    return sumK;
}

/**
 * G2 — two-float reassociation spike, tested through the production tangent-
 * point kernel arithmetic (not a standalone expression), per spec §4 G2.
 * Fixture: `nearTouchPair(1e-6)` — edge I=[0,1], edge J=[2,3], disjoint. The
 * pairs (i=0,j=2) and (i=1,j=3) have `d = v_i − v_j` along the near-touch
 * axis (‖d‖ = gap = 1e-6), and that ‖d‖ is raised to `-beta` (=-6) in the
 * kernel — the exact catastrophic-cancellation-under-f32 case T1/G2 exist to
 * catch, amplified sharply by the β=6 exponent.
 *
 * WGSL kernel mirrors `tangentPointKernelPieceCPU` above (== calculateEnergy
 * inner loop) EXACTLY: same (i,j) traversal order, same cross-product
 * component order, same ε-after-norm placement. `pow()` (not repeated
 * multiplication) is used for the integer exponents here — PREC Q1 measured
 * WGSL's `pow` = `exp2(y·log2 x)` as harmless at this α=3/β=6, and this
 * spike's job is isolating the two-float reassociation question, not the
 * separate pow-vs-repeated-multiply choice §2.3 makes for the production
 * kernel.
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.3, §4 G2
 * @see docs/2026-08-13-ai-research-gpu-precision.md Q1, Q3 (residual reassociation risk)
 * @see docs/superpowers/plans/2026-08-13-webgpu-solver-phase0.md Task 8
 */
spikes.g2 = async () => {
    const renderer = new THREE.WebGPURenderer();
    await renderer.init();
    const { attributeArray, instancedArray, wgslFn } = await import('three/tsl');

    const { vertices } = nearTouchPair(1e-6);
    const { alpha, beta, epsilon } = DEFAULTS;

    const cpu64 = tangentPointKernelPieceCPU(vertices, 0, 1, 2, 3, alpha, beta, epsilon);

    const { hi, lo } = splitHiLo(vertices);
    const hiBuf = attributeArray(hi, 'float').toReadOnly();
    const loBuf = attributeArray(lo, 'float').toReadOnly();
    const outTwoBuf = instancedArray(1, 'float');
    const outPlainBuf = instancedArray(1, 'float');

    // wgslFn storage-pointer-param syntax verified against the g0t spike
    // above (same session-verified pattern: discourse.threejs.org
    // "storagebufferattribute as input to wgslfn" / Maxime Heckel's TSL
    // field guide).
    const g2KernelFn = wgslFn(`
        fn g2Kernel(
            hi: ptr<storage, array<f32>, read>,
            lo: ptr<storage, array<f32>, read>,
            outTwo: ptr<storage, array<f32>, read_write>,
            outPlain: ptr<storage, array<f32>, read_write>
        ) -> void {
            let alpha = ${alpha}.0;
            let beta = ${beta}.0;
            let eps = ${epsilon};

            // Two-float coordinate differences: (hi_i - hi_j) + (lo_i - lo_j)
            // BEFORE any other arithmetic — the reassociation under test.
            // A legal-but-fatal compiler move would collapse this back to
            // (hi_i + lo_i) - (hi_j + lo_j), i.e. plain f32 [PREC Q3].
            // @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.3
            let eIx_two = (hi[3u] - hi[0u]) + (lo[3u] - lo[0u]);
            let eIy_two = (hi[4u] - hi[1u]) + (lo[4u] - lo[1u]);
            let eIz_two = (hi[5u] - hi[2u]) + (lo[5u] - lo[2u]);
            // Plain-f32 control (separate output slot): hi only, lo ignored.
            let eIx_pl = hi[3u] - hi[0u];
            let eIy_pl = hi[4u] - hi[1u];
            let eIz_pl = hi[5u] - hi[2u];

            var sumTwo = 0.0;
            var sumPlain = 0.0;

            // 4 endpoint combinations: i in {v0,v1} (edge I), j in {v2,v3}
            // (edge J) — mirrors calculateEnergy's (i,j) inner loop and c/d/
            // pow op order + ε-after-norm placement EXACTLY
            // [tangentPointEnergy.ts:84-99].
            for (var ii = 0u; ii < 2u; ii = ii + 1u) {
                for (var jj = 0u; jj < 2u; jj = jj + 1u) {
                    let i = ii;
                    let j = 2u + jj;

                    let dx_two = (hi[3u * i + 0u] - hi[3u * j + 0u]) + (lo[3u * i + 0u] - lo[3u * j + 0u]);
                    let dy_two = (hi[3u * i + 1u] - hi[3u * j + 1u]) + (lo[3u * i + 1u] - lo[3u * j + 1u]);
                    let dz_two = (hi[3u * i + 2u] - hi[3u * j + 2u]) + (lo[3u * i + 2u] - lo[3u * j + 2u]);
                    let dx_pl = hi[3u * i + 0u] - hi[3u * j + 0u];
                    let dy_pl = hi[3u * i + 1u] - hi[3u * j + 1u];
                    let dz_pl = hi[3u * i + 2u] - hi[3u * j + 2u];

                    let d_norm_two = sqrt(dx_two * dx_two + dy_two * dy_two + dz_two * dz_two) + eps;
                    let cx_two = eIy_two * dz_two - eIz_two * dy_two;
                    let cy_two = eIz_two * dx_two - eIx_two * dz_two;
                    let cz_two = eIx_two * dy_two - eIy_two * dx_two;
                    let c_norm_two = sqrt(cx_two * cx_two + cy_two * cy_two + cz_two * cz_two) + eps;
                    sumTwo = sumTwo + pow(c_norm_two, alpha) / pow(d_norm_two, beta);

                    let d_norm_pl = sqrt(dx_pl * dx_pl + dy_pl * dy_pl + dz_pl * dz_pl) + eps;
                    let cx_pl = eIy_pl * dz_pl - eIz_pl * dy_pl;
                    let cy_pl = eIz_pl * dx_pl - eIx_pl * dz_pl;
                    let cz_pl = eIx_pl * dy_pl - eIy_pl * dx_pl;
                    let c_norm_pl = sqrt(cx_pl * cx_pl + cy_pl * cy_pl + cz_pl * cz_pl) + eps;
                    sumPlain = sumPlain + pow(c_norm_pl, alpha) / pow(d_norm_pl, beta);
                }
            }

            outTwo[0u] = sumTwo;
            outPlain[0u] = sumPlain;
        }
    `);

    const g2Compute = g2KernelFn({
        hi: hiBuf,
        lo: loBuf,
        outTwo: outTwoBuf,
        outPlain: outPlainBuf,
    }).compute(1);

    renderer.compute(g2Compute);
    const outTwoData = new Float32Array(await renderer.getArrayBufferAsync(outTwoBuf.value));
    const outPlainData = new Float32Array(await renderer.getArrayBufferAsync(outPlainBuf.value));
    const gpu = outTwoData[0];
    const gpuPlain = outPlainData[0];

    const relErr = Math.abs(gpu - cpu64) / Math.abs(cpu64);
    const relErrPlain = Math.abs(gpuPlain - cpu64) / Math.abs(cpu64);

    return {
        gpu,
        gpuPlain,
        cpu64,
        relErr,
        relErrPlain,
        pass: relErr < 1e-5,
    };
};

/**
 * G4 — zero-readback render spike. Stands in for "solver output lands in a
 * storage buffer, gets rendered with zero per-frame readbacks": 120 frames of
 * a 64-edge open polyline (`y = sin(x + t)`) are CPU-precomputed as f32
 * hi/lo pairs (`splitHiLo`) and uploaded ONE buffer-write per frame — the
 * wave itself is NOT GPU-evaluated (review-3 F5: a GPU `sin` could never
 * bit-match CPU f32 exactly, and the render plumbing is what's under test).
 * Each frame then runs the §2.7 combine/scatter kernel entirely on GPU and
 * renders the result through the SAME fat-line material the app already
 * uses (`Line2NodeMaterial`, `src/scene/Curve.tsx`) — but with
 * `instanceStart`/`instanceEnd` bound to `StorageBufferAttribute`s (usage
 * `STORAGE | VERTEX`, `WebGPUBackend.js:2564`) instead of the production
 * per-frame `setPositions()` CPU path. `Line2NodeMaterial` reads
 * `attribute('instanceStart'/'instanceEnd')` off whatever geometry
 * attribute has that name (`Line2NodeMaterial.js:117-118`) — it does not
 * care whether the backing buffer is CPU- or GPU-authored, so no material
 * changes are needed, only the geometry's attribute wiring (Phase 3 will
 * make this same swap in `Curve.tsx`).
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.7, §4 G4
 * @see docs/2026-08-13-ai-research-webgpu-compute.md §2 (StorageBufferAttribute usage flags; Line2NodeMaterial hardcoded attribute names)
 * @see docs/superpowers/plans/2026-08-13-webgpu-solver-phase0.md Task 9
 */
spikes.g4 = async () => {
    const { LineSegmentsGeometry } = await import('three/addons/lines/LineSegmentsGeometry.js');
    const { LineSegments2 } = await import('three/addons/lines/webgpu/LineSegments2.js');
    const { Fn, instanceIndex, storage } = await import('three/tsl');

    const renderer = new THREE.WebGPURenderer();
    await renderer.init();
    renderer.setSize(256, 256);
    document.body.appendChild(renderer.domElement);

    // Render into an offscreen RenderTarget instead of presenting the
    // default canvas swapchain. First attempt presented directly and hit a
    // genuine `VK_ERROR_OUT_OF_DEVICE_MEMORY` device loss after all 120
    // frames rendered successfully (confirmed via a temporary per-frame
    // console.log: the loop completed, the crash surfaced only in the
    // POST-loop verification readback) — headless Chrome's canvas-present
    // path is documented (README "Hardware-headless canvas-present noise")
    // to spam a benign-looking Dawn `ImportMemory` validation chain on every
    // presented frame; at 120 presented frames it apparently escalates to a
    // real leak, not just log noise. Rendering off-canvas exercises the
    // IDENTICAL vertex/fragment pipeline (same geometry, same
    // StorageBufferAttribute-backed `instanceStart`/`instanceEnd`, same
    // Line2NodeMaterial) without touching the leaking swapchain-export path
    // — legitimate given G4 verifies via buffer readback, not the presented
    // image [spec §4 G4; README "G4 ... verifies via a buffer readback, not
    // the presented image"].
    const renderTarget = new THREE.RenderTarget(256, 256);
    renderer.setRenderTarget(renderTarget);

    // Readback-entrypoint counter (spec §4 G4: "asserted by instrumentation
    // counter"). Wraps `renderer.getArrayBufferAsync` — the only readback
    // entry point any spike in this file uses (see g2 above) — AND the
    // backend method it forwards to (`Renderer.js:1980` ->
    // `backend.getArrayBufferAsync`), so a hypothetical internal call that
    // bypasses the renderer-level wrapper is still counted.
    let readbackCount = 0;
    const origRendererReadback = renderer.getArrayBufferAsync.bind(renderer);
    renderer.getArrayBufferAsync = (async (...args: Parameters<typeof origRendererReadback>) => {
        readbackCount++;
        return origRendererReadback(...args);
    }) as typeof renderer.getArrayBufferAsync;
    const backend = renderer.backend as unknown as {
        getArrayBufferAsync: (...args: unknown[]) => Promise<ArrayBuffer>;
    };
    const origBackendReadback = backend.getArrayBufferAsync.bind(backend);
    backend.getArrayBufferAsync = async (...args: unknown[]) => {
        readbackCount++;
        return origBackendReadback(...args);
    };

    // ---- geometry: 64-edge open polyline (65 vertices). instanceStart/End
    // are StorageBufferAttributes instead of the production Curve.tsx's
    // setPositions()-derived InterleavedBufferAttribute [research doc §2].
    const V = 65;
    const E = 64;
    const geometry = new LineSegmentsGeometry();
    const instanceStartAttr = new THREE.StorageBufferAttribute(E, 3);
    const instanceEndAttr = new THREE.StorageBufferAttribute(E, 3);
    geometry.setAttribute('instanceStart', instanceStartAttr);
    geometry.setAttribute('instanceEnd', instanceEndAttr);
    geometry.instanceCount = E;
    // GPU compute writes land ONLY in the GPU-side buffer, never mirrored
    // back into these attributes' CPU-side `.array` (doing so would itself
    // be a readback) — so geometry.computeBoundingSphere()/BoundingBox(),
    // which read that CPU-side array, would see stale zeros. Set fixed
    // bounds from the known wave extent instead, and disable frustum
    // culling so a wrong/degenerate CPU-derived bound can never hide the
    // mesh (this is the honest GPU-authored-attribute adaptation, not a
    // workaround for a spike-only bug).
    geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(Math.PI, 0, 0), 4);
    geometry.boundingBox = new THREE.Box3(
        new THREE.Vector3(0, -1.5, -1.5),
        new THREE.Vector3(2 * Math.PI, 1.5, 1.5),
    );

    const material = new THREE.Line2NodeMaterial({
        color: 0x4a9eff,
        linewidth: 3,
        worldUnits: false,
    });
    const lineMesh = new LineSegments2(geometry, material);
    lineMesh.frustumCulled = false;

    const scene = new THREE.Scene();
    scene.add(lineMesh);
    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
    camera.position.set(Math.PI, 0, 5);
    camera.lookAt(Math.PI, 0, 0);

    // ---- CPU-precomputed wave frames: y = sin(x + t), x uniform over
    // [0, 2π] — NOT GPU-evaluated [spec §4 G4, review-3 F5].
    const FRAMES = 120;
    const xs = Array.from({ length: V }, (_, v) => (v / E) * 2 * Math.PI);
    const frameHiLo = Array.from({ length: FRAMES }, (_, f) => {
        const t = (f / FRAMES) * 2 * Math.PI;
        const verts: Vec3[] = xs.map((x) => [x, Math.sin(x + t), 0]);
        return splitHiLo(verts);
    });

    // Per-vertex hi/lo storage buffers, updated (not recreated) each frame.
    const hiAttr = new THREE.StorageBufferAttribute(V, 3);
    const loAttr = new THREE.StorageBufferAttribute(V, 3);
    const hiBuf = storage(hiAttr, 'vec3', V).toReadOnly();
    const loBuf = storage(loAttr, 'vec3', V).toReadOnly();
    const instanceStartNode = storage(instanceStartAttr, 'vec3', E);
    const instanceEndNode = storage(instanceEndAttr, 'vec3', E);

    /**
     * Combine/scatter kernel — spec §2.7: `pos = hi + lo` per vertex (the
     * combine), then scatter vertex i, i+1 into instance slot i's start/end
     * (the scatter). Ships unmodified into Phase 3 against the production
     * geometry. Interior vertices are read twice (once as an edge's "end",
     * once as the next edge's "start") instead of through a separate
     * combine pass into an intermediate buffer — same result, one dispatch,
     * no atomics, no cross-thread write conflicts.
     * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.7
     */
    const combineScatterKernel = Fn(() => {
        const e = instanceIndex;
        instanceStartNode.element(e).assign(hiBuf.element(e).add(loBuf.element(e)));
        instanceEndNode.element(e).assign(hiBuf.element(e.add(1)).add(loBuf.element(e.add(1))));
    })().compute(E);

    for (let f = 0; f < FRAMES; f++) {
        // Upload — a buffer write + needsUpdate flag, NOT a readback.
        (hiAttr.array as Float32Array).set(frameHiLo[f].hi);
        (loAttr.array as Float32Array).set(frameHiLo[f].lo);
        hiAttr.needsUpdate = true;
        loAttr.needsUpdate = true;

        renderer.compute(combineScatterKernel);
        renderer.render(scene, camera);
    }

    const readbacksDuringLoop = readbackCount;

    // ONE verification readback, post-loop, of the last frame's instance
    // buffers — f32-exact per element [spec §4 G4]. Compared at
    // `instanceStartAttr.itemSize` (the buffer's ACTUAL current stride),
    // not the itemSize=3 it was constructed with: three r185 silently pads
    // any itemSize-3 StorageBufferAttribute to itemSize 4 in-place on first
    // GPU use (WGSL forbids packed vec3 in storage buffers —
    // `WebGPUAttributeUtils.js:114-119` "WGSL does not support packed vec3
    // data in storage buffers, pad to vec4"), and mutates the attribute's
    // own `.itemSize`/`.array` to match. Comparing at a hardcoded stride of
    // 3 against that mutated, 4-wide buffer would misalign every element
    // after the first and report a false mismatch — that would be a bug in
    // THIS verification, not evidence about the GPU data.
    const lastHiLo = frameHiLo[FRAMES - 1];
    const expectedStart = new Float32Array(E * 3);
    const expectedEnd = new Float32Array(E * 3);
    for (let e = 0; e < E; e++) {
        for (let d = 0; d < 3; d++) {
            expectedStart[e * 3 + d] = Math.fround(lastHiLo.hi[e * 3 + d] + lastHiLo.lo[e * 3 + d]);
            expectedEnd[e * 3 + d] = Math.fround(
                lastHiLo.hi[(e + 1) * 3 + d] + lastHiLo.lo[(e + 1) * 3 + d],
            );
        }
    }
    const actualStart = new Float32Array(await renderer.getArrayBufferAsync(instanceStartAttr));
    const actualEnd = new Float32Array(await renderer.getArrayBufferAsync(instanceEndAttr));
    const stride = instanceStartAttr.itemSize; // 3 if unpadded, 4 if three.js padded it

    let mismatches = 0;
    for (let e = 0; e < E; e++) {
        for (let d = 0; d < 3; d++) {
            if (!Object.is(expectedStart[e * 3 + d], actualStart[e * stride + d])) mismatches++;
            if (!Object.is(expectedEnd[e * 3 + d], actualEnd[e * stride + d])) mismatches++;
        }
    }

    // SECOND check, independent of the buffer-contents question above:
    // whether Line2NodeMaterial actually painted the line anywhere. The
    // buffer padding above is silently applied to the SAME GPU buffer
    // that's also bound as the `instanceStart`/`instanceEnd` VERTEX
    // attribute, and Line2NodeMaterial hardcodes a vec3 read for it
    // (`vec4(instanceStart, 1.0)`, `Line2NodeMaterial.js:120-121`) — the
    // itemSize-4 mutation desyncs that assumption (surfaced by the "Length
    // of parameters exceeds maximum length of function 'vec4()'" console
    // warning this run also captures), so the buffer contents can be
    // correct while the RENDER is still broken. One-off pixel readback,
    // same "post-loop only" allowance as the buffer verification above.
    const pixels = await renderer.readRenderTargetPixelsAsync(renderTarget, 0, 0, 256, 256);
    let coloredPixels = 0;
    for (let i = 0; i < pixels.length; i += 4) {
        if (pixels[i + 2] > 100 && pixels[i + 0] < 150) coloredPixels++; // bluish (line color 0x4a9eff)
    }
    const rendered = coloredPixels > 0;

    return {
        frames: FRAMES,
        readbacksDuringLoop,
        mismatches,
        rendered,
        coloredPixels,
        pass: readbacksDuringLoop === 0 && mismatches === 0 && rendered,
        // Root-cause note for the record (spec §4 G4 FAIL disposition: "do
        // NOT hack around it ... record the honest FAIL"). `mismatches`
        // above proves the zero-readback combine/scatter DATA pipeline is
        // correct (f32-exact) — the failure is `rendered === false`:
        // WGSL's mandatory vec3-storage padding
        // (`WebGPUAttributeUtils.js:114-119`) mutates the SAME buffer
        // object's `itemSize` that Line2NodeMaterial hardcodes as vec3 for
        // its `instanceStart`/`instanceEnd` vertex read
        // (`Line2NodeMaterial.js:117-121`), so nothing is painted (0 of
        // 65536 pixels matched the line color) even though the underlying
        // storage buffer holds the exactly-correct wave data. This is a
        // structural collision between a WGSL requirement and
        // Line2NodeMaterial's fixed shader graph, not a spike plumbing
        // mistake — "Line2NodeMaterial fights it" per spec §4 G4's own FAIL
        // disposition, now confirmed on real hardware.
        note: 'zero-readback combine/scatter DATA is proven correct (mismatches counted at the buffer\'s actual post-pad stride), but the RENDER fails: three r185 silently pads any itemSize-3 StorageBufferAttribute to itemSize 4 in-place (WGSL forbids packed vec3 in storage buffers, WebGPUAttributeUtils.js:114-119) — the SAME buffer is also the instanceStart/instanceEnd vertex attribute, and Line2NodeMaterial hardcodes a vec3 read for it (Line2NodeMaterial.js:117-121), so the mutation desyncs the vertex shader and nothing is painted (0 colored pixels) — "Line2NodeMaterial fights it" per spec §4 G4 FAIL disposition',
    };
};

declare global {
    interface Window {
        __runSpike: (name: string) => Promise<string>;
    }
}
window.__runSpike = async (name) => JSON.stringify(await spikes[name]());
