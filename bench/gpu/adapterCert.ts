/**
 * How a gate run's adapter was certified — the provenance record that sits
 * beside `classifyAdapter`'s verdict in every result JSON.
 *
 * **The invariant this module protects:** an adapter we cannot certify as
 * hardware must not yield a passing gate. `classifyAdapter` enforces that from
 * the four `GPUAdapterInfo` identity strings alone (`adapterClass.ts:39-47`).
 * Firefox blanks all four — measured, see `vulkanPin.ts` — so under that
 * predicate alone every Firefox gate is INVALID: a harness that runs and
 * certifies nothing.
 *
 * The fix is **not** to relax the blank guard. `classifyAdapter` is untouched
 * and still rules a blank record `'software'`. Instead this module adds a
 * second, narrower route that requires *positive* evidence from two mutually
 * independent channels, and records which route was taken so a reader of the
 * JSON can always tell a hardware run from a software one:
 *
 * 1. `adapter-info` — the four fields named the adapter. This is Chrome's route
 *    and is the only route that is self-describing from the payload.
 * 2. `fallback-flag+device-pin` — the fields were blank, AND the page reported
 *    `GPUAdapterInfo.isFallbackAdapter === false`, AND the browser process was
 *    pinned to a single Vulkan ICD whose device `vulkaninfo` independently
 *    names as a non-CPU device. Both signals must agree.
 * 3. `undetermined` — anything else. Always `'software'`.
 *
 * Route 2 is falsifiable, and is falsified in the knock-out transcript: pinning
 * the same browser to `lvp_icd.json` flips `isFallbackAdapter` to `true` *and*
 * makes `vulkaninfo` report `PHYSICAL_DEVICE_TYPE_CPU`, so both conjuncts fail
 * and the run is ruled `'software'` → INVALID.
 *
 * @issue utof/repulsive-test2#53
 * @issue utof/repulsive-test2#54
 * @see bench/gpu/README.md — "Hardware certification without adapterInfo"
 */
import { type AdapterIdentity, classifyAdapter } from './adapterClass';
import type { HostEvidence } from './hosts/types';
import { CPU_DEVICE_TYPE } from './vulkanPin';

/** The route by which `classification` was reached. @see module docstring */
export type CertificationMethod = 'adapter-info' | 'fallback-flag+device-pin' | 'undetermined';

/**
 * The provenance record written into every gate result. `classification` is
 * what the gate's PASS/INVALID decision reads; `method` and `evidence` are what
 * a human (or a later agent) reads to decide whether to believe it.
 */
export type AdapterCertification = {
    classification: 'hardware' | 'software';
    method: CertificationMethod;
    /** Why this method was chosen — one sentence, safe to quote into a report. */
    rationale: string;
    evidence: {
        adapterInfo: AdapterIdentity;
        /** `classifyAdapter`'s verdict on `adapterInfo` alone, always recorded. */
        adapterInfoVerdict: 'hardware' | 'software';
        isFallbackAdapter?: boolean;
        devicePin: HostEvidence['devicePin'];
    };
};

/**
 * Certify the adapter a run used.
 *
 * Note the ordering: the `adapter-info` route is tried first, so a host that
 * *does* identify its adapter (Chrome) is never routed through the pin channel,
 * and the Firefox route can never overrule a `'software'` verdict that came
 * from a *named* software rasterizer. A record that positively says
 * "swiftshader" stays software no matter what the pin says.
 *
 * @issue utof/repulsive-test2#53
 * @issue utof/repulsive-test2#54
 */
export function certifyAdapter(
    adapterInfo: AdapterIdentity,
    evidence: HostEvidence,
): AdapterCertification {
    const adapterInfoVerdict = classifyAdapter(adapterInfo);
    const base = {
        adapterInfo,
        adapterInfoVerdict,
        isFallbackAdapter: evidence.isFallbackAdapter,
        devicePin: evidence.devicePin,
    };

    const named = isNamed(adapterInfo);

    // Route 1: the adapter named itself. Trust that, either way — a named
    // software rasterizer is software, and no out-of-band pin may override it.
    if (named) {
        return {
            classification: adapterInfoVerdict,
            method: 'adapter-info',
            rationale: `GPUAdapterInfo named the adapter; classifyAdapter ruled '${adapterInfoVerdict}'.`,
            evidence: base,
        };
    }

    // Route 2: blank GPUAdapterInfo (Firefox). Requires BOTH the in-page
    // fallback flag and an out-of-band, non-CPU device pin. Either one alone is
    // insufficient: `isFallbackAdapter === false` does not say *which* GPU (the
    // Intel iGPU and the NVIDIA dGPU both report false), and a pin alone does
    // not prove the browser honoured it.
    const pin = evidence.devicePin;
    if (
        evidence.isFallbackAdapter === false &&
        pin !== null &&
        pin.deviceType !== CPU_DEVICE_TYPE
    ) {
        return {
            classification: 'hardware',
            method: 'fallback-flag+device-pin',
            rationale: `GPUAdapterInfo was blank, but isFallbackAdapter=false and the browser was pinned to ${pin.icdPath}, which vulkaninfo names '${pin.deviceName}' (${pin.deviceType}).`,
            evidence: base,
        };
    }

    return {
        classification: 'software',
        method: 'undetermined',
        rationale: blankRationale(evidence),
        evidence: base,
    };
}

/** True when any identity field carries a non-blank value. */
function isNamed(info: AdapterIdentity): boolean {
    return (
        `${info?.vendor ?? ''}${info?.architecture ?? ''}${info?.device ?? ''}${info?.description ?? ''}`.trim() !==
        ''
    );
}

/** Spell out which conjunct of route 2 failed, so the JSON is self-diagnosing. */
function blankRationale(evidence: HostEvidence): string {
    const reasons: string[] = ['GPUAdapterInfo was blank'];
    if (evidence.isFallbackAdapter === undefined) reasons.push('isFallbackAdapter was unreadable');
    else if (evidence.isFallbackAdapter)
        reasons.push('isFallbackAdapter=true (a fallback adapter)');
    if (evidence.devicePin === null) reasons.push('no Vulkan ICD pin was applied');
    else if (evidence.devicePin.deviceType === CPU_DEVICE_TYPE) {
        reasons.push(
            `the pin resolves to '${evidence.devicePin.deviceName}' (${evidence.devicePin.deviceType}), a CPU rasterizer`,
        );
    }
    return `${reasons.join('; ')} — the adapter could not be certified as hardware.`;
}
