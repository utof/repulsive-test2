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

// Later tasks append spikes here (g0t, g1, g3, g2, g4).

declare global {
    interface Window {
        __runSpike: (name: string) => Promise<string>;
    }
}
window.__runSpike = async (name) => JSON.stringify(await spikes[name]());
