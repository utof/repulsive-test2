/**
 * Out-of-band identification of the GPU a browser run was pinned to.
 *
 * Why this module exists: Firefox (naga/wgpu) reports a blank `GPUAdapterInfo`
 * — measured 2026-09-11 on Firefox 155.0.1, all four of
 * `vendor`/`architecture`/`device`/`description` are the empty string — so the
 * in-page channel cannot name the adapter, and `classifyAdapter` correctly
 * refuses to certify it (`adapterClass.ts:39-47`). This module supplies the
 * missing identity from *outside* the browser: pin the Vulkan loader to exactly
 * one ICD manifest, ask `vulkaninfo` what that ICD resolves to, and hand the
 * browser the same environment. The device the browser gets is then the device
 * `vulkaninfo` named, and that name is recorded in the result JSON.
 *
 * This is not a heuristic. Measured on this box (see the gate report at
 * `.superpowers/sdd/2026-09-11-phase2h-queue/reports/54-firefox-host.md`):
 * without a pin, `nvidia-smi --query-compute-apps` lists no firefox process
 * during a WebGPU compute spike; with `VK_DRIVER_FILES` pinned to
 * `nvidia_icd.json` it lists `/usr/lib/firefox/firefox` holding device memory.
 *
 * @issue utof/repulsive-test2#53
 * @issue utof/repulsive-test2#54
 */
import { existsSync } from 'node:fs';

import type { DevicePin } from './hosts/types';

/** Where Mesa/NVIDIA drop their ICD manifests on a Debian/Ubuntu box. */
const ICD_DIR = '/usr/share/vulkan/icd.d';

/**
 * `vulkaninfo`'s `deviceType` for a CPU (software) rasterizer. llvmpipe and
 * lavapipe both report this; it is the out-of-band twin of the `swiftshader` /
 * `llvmpipe` substrings `classifyAdapter` looks for in-page.
 */
export const CPU_DEVICE_TYPE = 'PHYSICAL_DEVICE_TYPE_CPU';

/**
 * The environment that pins the Vulkan loader to a single ICD manifest.
 *
 * Both variables are set: `VK_ICD_FILENAMES` is the legacy name and
 * `VK_DRIVER_FILES` the current one, and which the installed loader honours
 * depends on its version — setting both is the portable spelling.
 * @see https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderInterfaceArchitecture.md
 */
export function icdPinEnv(icdPath: string): Record<string, string> {
    return { VK_DRIVER_FILES: icdPath, VK_ICD_FILENAMES: icdPath };
}

/** Absolute path of a named ICD manifest, e.g. `icdPath('nvidia')`. */
export function icdPath(name: string): string {
    return `${ICD_DIR}/${name}_icd.json`;
}

/**
 * Ask `vulkaninfo` which physical device an ICD manifest resolves to.
 *
 * Returns `null` when `vulkaninfo` is absent, the manifest does not exist, or
 * the summary names no device — all of which mean "un-identified", which the
 * certification path must treat as *not* certified rather than as a pass.
 *
 * Why parse `--summary` rather than `--json`: `--json` emits one object per
 * device and is not stable across loader versions, while the `deviceName` /
 * `deviceType` lines of `--summary` have been stable for years and are the two
 * fields the certification actually needs.
 */
export function describeIcdPin(icdManifestPath: string): DevicePin | null {
    if (!existsSync(icdManifestPath)) return null;
    let stdout: string;
    try {
        const proc = Bun.spawnSync(['vulkaninfo', '--summary'], {
            env: { ...process.env, ...icdPinEnv(icdManifestPath) },
        });
        stdout = proc.stdout.toString();
    } catch {
        return null; // vulkaninfo not installed
    }
    // `--summary` prints an instance section first; the device fields only
    // appear under `Devices:`, so anchor there or an instance-level line could
    // be mistaken for a device.
    const devicesIdx = stdout.indexOf('Devices:');
    if (devicesIdx < 0) return null;
    const devices = stdout.slice(devicesIdx);
    const nameMatch = devices.match(/deviceName\s*=\s*(.+)/);
    const typeMatch = devices.match(/deviceType\s*=\s*(\S+)/);
    if (!nameMatch || !typeMatch) return null;
    return {
        icdPath: icdManifestPath,
        deviceName: nameMatch[1].trim(),
        deviceType: typeMatch[1].trim(),
    };
}

/**
 * The ICD pins the Firefox host tries, in order: the discrete NVIDIA GPU first
 * (what `CLAUDE.md` means by "GPU perf can be benchmarked locally"), then the
 * Intel iGPU as a still-hardware fallback.
 *
 * llvmpipe is deliberately absent — it is reachable only through
 * `GPU_VK_ICD`, which is how the knock-out test forces the software path.
 */
export const HARDWARE_ICD_CANDIDATES = ['nvidia', 'intel', 'radeon'] as const;
