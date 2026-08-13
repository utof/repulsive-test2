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

// Later tasks append spikes here (g0t, g1, g2, g4).

declare global {
    interface Window {
        __runSpike: (name: string) => Promise<string>;
    }
}
window.__runSpike = async (name) => JSON.stringify(await spikes[name]());
