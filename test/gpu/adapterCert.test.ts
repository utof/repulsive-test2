import { describe, expect, test } from 'bun:test';

import { certifyAdapter } from '../../bench/gpu/adapterCert';
import type { HostEvidence } from '../../bench/gpu/hosts/types';
import { CPU_DEVICE_TYPE } from '../../bench/gpu/vulkanPin';

/**
 * The certification layer is what lets Firefox gates run at all without letting
 * an uncertifiable adapter produce a passing gate. `classifyAdapter` rules a
 * blank `GPUAdapterInfo` `'software'` on purpose, and Firefox blanks all four
 * fields — so without a second, positively-evidenced route every Firefox gate
 * is INVALID, and with a careless route every software run is a false PASS.
 * These tests pin both edges.
 *
 * @see bench/gpu/adapterCert.ts
 * @issue utof/repulsive-test2#53
 * @issue utof/repulsive-test2#54
 */
const BLANK = { vendor: '', architecture: '', device: '', description: '' };

const NVIDIA_PIN = {
    icdPath: '/usr/share/vulkan/icd.d/nvidia_icd.json',
    deviceName: 'Quadro RTX 3000',
    deviceType: 'PHYSICAL_DEVICE_TYPE_DISCRETE_GPU',
};

const LLVMPIPE_PIN = {
    icdPath: '/usr/share/vulkan/icd.d/lvp_icd.json',
    deviceName: 'llvmpipe (LLVM 20.1.2, 256 bits)',
    deviceType: CPU_DEVICE_TYPE,
};

const noEvidence: HostEvidence = { devicePin: null };

describe('certifyAdapter — the adapter-info route (Chrome)', () => {
    test('a named hardware adapter certifies through GPUAdapterInfo', () => {
        const cert = certifyAdapter(
            { vendor: 'NVIDIA', architecture: 'turing', device: '', description: '' },
            noEvidence,
        );
        expect(cert.classification).toBe('hardware');
        expect(cert.method).toBe('adapter-info');
    });

    test('a named software rasterizer stays software', () => {
        const cert = certifyAdapter(
            { vendor: 'Google Inc.', architecture: 'swiftshader', device: '', description: '' },
            noEvidence,
        );
        expect(cert.classification).toBe('software');
        expect(cert.method).toBe('adapter-info');
    });

    // The precedence rule that makes the pin route safe: a pin may only ever
    // supply an identity the page failed to supply, never overrule one it did.
    // Without this ordering, pinning the loader to a hardware ICD would launder
    // a self-identified SwiftShader run into a hardware PASS.
    test('a device pin can NOT launder a named software adapter into hardware', () => {
        const cert = certifyAdapter(
            { vendor: 'Mesa', architecture: '', device: 'llvmpipe', description: '' },
            { isFallbackAdapter: false, devicePin: NVIDIA_PIN },
        );
        expect(cert.classification).toBe('software');
        expect(cert.method).toBe('adapter-info');
    });
});

describe('certifyAdapter — the fallback-flag+device-pin route (Firefox)', () => {
    // The measured Firefox-155-on-this-box case: blank info, non-fallback
    // adapter, pinned to the discrete NVIDIA GPU.
    test('blank info certifies as hardware when BOTH signals agree', () => {
        const cert = certifyAdapter(BLANK, { isFallbackAdapter: false, devicePin: NVIDIA_PIN });
        expect(cert.classification).toBe('hardware');
        expect(cert.method).toBe('fallback-flag+device-pin');
        expect(cert.rationale).toContain('Quadro RTX 3000');
    });

    test('the evidence records the blank-info verdict alongside the override', () => {
        const cert = certifyAdapter(BLANK, { isFallbackAdapter: false, devicePin: NVIDIA_PIN });
        // A reader must be able to see that GPUAdapterInfo alone said 'software'
        // and that the hardware verdict came from the pin, not from the page.
        expect(cert.evidence.adapterInfoVerdict).toBe('software');
        expect(cert.evidence.devicePin).toEqual(NVIDIA_PIN);
        expect(cert.evidence.isFallbackAdapter).toBe(false);
    });

    // The knock-out, as a unit test: this is the configuration the live
    // `GPU_VK_ICD=lvp` run produces, and it must be INVALID-yielding.
    test('a CPU-type pin is software even though the flag alone would not say so', () => {
        const cert = certifyAdapter(BLANK, { isFallbackAdapter: true, devicePin: LLVMPIPE_PIN });
        expect(cert.classification).toBe('software');
        expect(cert.method).toBe('undetermined');
        expect(cert.rationale).toContain(CPU_DEVICE_TYPE);
    });

    test('a CPU-type pin is software even if isFallbackAdapter lies and says false', () => {
        const cert = certifyAdapter(BLANK, { isFallbackAdapter: false, devicePin: LLVMPIPE_PIN });
        expect(cert.classification).toBe('software');
        expect(cert.method).toBe('undetermined');
    });
});

describe('certifyAdapter — every way the route must refuse', () => {
    // Each conjunct is individually necessary. `isFallbackAdapter === false`
    // alone does not say WHICH gpu (the Intel iGPU and the NVIDIA dGPU both
    // report false), and a pin alone does not prove the browser honoured it.
    test('no pin at all is software, even with isFallbackAdapter false', () => {
        const cert = certifyAdapter(BLANK, { isFallbackAdapter: false, devicePin: null });
        expect(cert.classification).toBe('software');
        expect(cert.method).toBe('undetermined');
        expect(cert.rationale).toContain('no Vulkan ICD pin was applied');
    });

    test('a fallback adapter is software, even with a discrete-GPU pin', () => {
        const cert = certifyAdapter(BLANK, { isFallbackAdapter: true, devicePin: NVIDIA_PIN });
        expect(cert.classification).toBe('software');
        expect(cert.method).toBe('undetermined');
        expect(cert.rationale).toContain('isFallbackAdapter=true');
    });

    test('an unreadable isFallbackAdapter is software, not an assumed false', () => {
        const cert = certifyAdapter(BLANK, { devicePin: NVIDIA_PIN });
        expect(cert.classification).toBe('software');
        expect(cert.rationale).toContain('unreadable');
    });

    test('no evidence whatsoever is software (the launch-failure path)', () => {
        const cert = certifyAdapter(BLANK, noEvidence);
        expect(cert.classification).toBe('software');
        expect(cert.method).toBe('undetermined');
    });

    test('whitespace-only fields count as blank, not as a name', () => {
        const cert = certifyAdapter(
            { vendor: '   ', architecture: '\t', device: '', description: '\n' },
            { isFallbackAdapter: false, devicePin: NVIDIA_PIN },
        );
        expect(cert.method).toBe('fallback-flag+device-pin');
    });

    test('missing fields do not throw and are treated as blank', () => {
        const cert = certifyAdapter({}, noEvidence);
        expect(cert.classification).toBe('software');
    });
});
