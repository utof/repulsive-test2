import { describe, expect, test } from 'bun:test';

import { HOSTS, selectHost } from '../../bench/gpu/hosts';
import { icdPath, icdPinEnv } from '../../bench/gpu/vulkanPin';

/**
 * The `GPU_HOST` variant switch and the launch recipes it hands the driver.
 * These are pure/structural checks — actually launching a browser is the gate
 * harness's job, not a unit test's.
 *
 * @see bench/gpu/hosts/index.ts
 * @issue utof/repulsive-test2#54
 */
describe('selectHost', () => {
    // Firefox is the default because it is the browser installed on the dev
    // box; `drive.ts` defaulting to `google-chrome` is what made every GPU kill
    // gate unrunnable in the first place.
    test('defaults to firefox when GPU_HOST is unset', () => {
        expect(selectHost(undefined).id).toBe('firefox');
    });

    test('selects each known host by id', () => {
        expect(selectHost('firefox').id).toBe('firefox');
        expect(selectHost('chrome').id).toBe('chrome');
    });

    // Throwing rather than falling back matters: a silent fallback on a typo
    // would move a numerics measurement to the other WGSL backend without
    // saying so, and naga/Dawn disagree on compensated arithmetic.
    // @see https://github.com/gpuweb/gpuweb/issues/2076
    test('throws on an unknown host instead of falling back', () => {
        expect(() => selectHost('safari')).toThrow(/Unknown GPU_HOST/);
    });
});

describe('shader backend provenance', () => {
    // The field that stops a compensated-arithmetic result taken on one host
    // from being read as evidence about the other. @issue utof/repulsive-test2#51
    test('firefox reports naga and chrome reports dawn', () => {
        expect(HOSTS.firefox.shaderBackend).toBe('naga');
        expect(HOSTS.chrome.shaderBackend).toBe('dawn');
    });

    test('the two hosts never share a backend', () => {
        expect(HOSTS.firefox.shaderBackend).not.toBe(HOSTS.chrome.shaderBackend);
    });
});

describe('firefox launch recipes', () => {
    const variants = HOSTS.firefox.launchVariants;

    // Without this pref `navigator.gpu` is undefined on Firefox 155/Linux —
    // measured, WebGPU is not on by default on this platform.
    test('every recipe enables dom.webgpu.enabled', () => {
        expect(variants.length).toBeGreaterThan(0);
        for (const v of variants) expect(v.prefs['dom.webgpu.enabled']).toBe(true);
    });

    test('every recipe is headless', () => {
        for (const v of variants) expect(v.argv).toContain('--headless');
    });

    // The discrete GPU must be tried first: unpinned Firefox runs WebGPU on the
    // Intel iGPU on this box and never touches the NVIDIA Quadro.
    test('the first recipe pins the Vulkan loader to the discrete NVIDIA ICD', () => {
        expect(variants[0].env.VK_DRIVER_FILES).toBe(icdPath('nvidia'));
    });

    // Kept so the harness still RUNS on a box with no vulkaninfo — it will fail
    // certification and report INVALID, which is honest, rather than crashing.
    test('an unpinned recipe exists as the last resort only', () => {
        const unpinned = variants.filter((v) => v.env.VK_DRIVER_FILES === undefined);
        expect(unpinned).toHaveLength(1);
        expect(variants[variants.length - 1]).toBe(unpinned[0]);
    });

    // No recipe may pin a CPU rasterizer: llvmpipe is reachable only through
    // the explicit GPU_VK_ICD knock-out lever.
    test('no default recipe pins a software rasterizer', () => {
        for (const v of variants) {
            expect(v.env.VK_DRIVER_FILES ?? '').not.toContain('lvp');
        }
    });
});

describe('chrome launch recipes', () => {
    // The flag sets that passed the Phase 0 / Phase 1 gates, preserved verbatim
    // across the move behind the host seam.
    // @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §4 G0a
    test('preserves the three historical flag sets in order', () => {
        const v = HOSTS.chrome.launchVariants;
        expect(v).toHaveLength(3);
        expect(v[0].argv).toEqual([
            '--headless=new',
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan',
            '--use-angle=vulkan',
            '--enable-gpu',
            '--no-sandbox',
        ]);
        expect(v[1].argv).toContain('--ignore-gpu-blocklist');
        // headed fallback last (spec G0a: allowed after an honest headless effort)
        expect(v[2].argv).not.toContain('--headless=new');
    });

    test('resolveVariant is the identity — Chrome has no env override', () => {
        const v = HOSTS.chrome.launchVariants[0];
        expect(HOSTS.chrome.resolveVariant(v)).toBe(v);
    });
});

describe('icdPinEnv', () => {
    // Both spellings: VK_ICD_FILENAMES is the legacy name, VK_DRIVER_FILES the
    // current one, and which the installed loader honours depends on its version.
    test('sets both the legacy and current loader variables', () => {
        expect(icdPinEnv('/x/y.json')).toEqual({
            VK_DRIVER_FILES: '/x/y.json',
            VK_ICD_FILENAMES: '/x/y.json',
        });
    });
});
