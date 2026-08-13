// bench/gpu/spikes.ts — spike registry; each spike returns JSON-serializable data.
// @see docs/superpowers/plans/2026-08-13-webgpu-solver-phase0.md Task 1
import * as THREE from 'three/webgpu';

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

// Later tasks append spikes here (g1, g2, g4).

declare global {
    interface Window {
        __runSpike: (name: string) => Promise<string>;
    }
}
window.__runSpike = async (name) => JSON.stringify(await spikes[name]());
