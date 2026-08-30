// bench/gpu/phase1.ts — Phase 1 gates through the PRODUCTION kernel pieces.
// Registered into the Phase 0 spike registry by spikes.ts (Object.assign).
// @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.2, §3 (T1–T3), §5 Phase 1
// @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (Task 3)
import { attributeArray } from 'three/tsl';
import * as THREE from 'three/webgpu';
import { makeReduceSum, REDUCE_MAX_EDGES } from '../../src/gpu/reduce';

/**
 * Phase 1 spikes, merged into `spikes.ts`'s registry so `drive.ts` can run
 * them by name with the same adapter classification (software ⇒ INVALID).
 * @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (Task 3)
 */
export const phase1Spikes: Record<string, () => Promise<Record<string, unknown>>> = {};

/**
 * Reduction-only check for `makeReduceSum`: fill `input` with 1..n and
 * require `out[0] === n(n+1)/2` EXACTLY. Every integer below 2^24 is exact
 * in f32; the largest sum here, 16384·16385/2 = 134,225,920, lies in
 * [2^27, 2^28), where the f32 spacing is 16 — it is a multiple of 16
 * (134,225,920 / 16 = 8,389,120) and so still exactly representable.
 * `exactInF32` re-checks that per row rather than trusting this arithmetic.
 * A wrong tree/guard/barrier therefore appears as a wrong integer, never as
 * noise.
 * The n set brackets every guard boundary: 1 (single lane), 63/64/65 (the
 * lane count), 128/129 (one group vs. two), 960 (T2's largest fixture) and
 * 16384 (`REDUCE_MAX_EDGES`, both passes fully populated).
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.2
 * @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (Task 3a)
 */
phase1Spikes.phase1Reduce = async () => {
    const renderer = new THREE.WebGPURenderer();
    await renderer.init();
    const sizes = [1, 63, 64, 65, 128, 129, 960, REDUCE_MAX_EDGES];
    const rows = [];
    for (const n of sizes) {
        const data = new Float32Array(n);
        for (let i = 0; i < n; i++) data[i] = i + 1;
        const input = attributeArray(data, 'float').toReadOnly();
        const { nodes, out } = makeReduceSum(input, n);
        renderer.compute(nodes as never);
        const got = new Float32Array(await renderer.getArrayBufferAsync(out.value))[0];
        const expected = (n * (n + 1)) / 2;
        const exactInF32 = Math.fround(expected) === expected;
        rows.push({ n, expected, got, exact: got === expected, exactInF32 });
    }
    return {
        gate: 'phase1Reduce',
        rows,
        pass: rows.every((r) => r.exact && r.exactInF32),
    };
};
