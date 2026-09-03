// bench/gpu/phase1.ts — Phase 1 gates through the PRODUCTION kernel pieces.
// Registered into the Phase 0 spike registry by spikes.ts (Object.assign).
// @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.2, §3 (T1–T3), §5 Phase 1
// @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (Task 3)
import { attributeArray } from 'three/tsl';
import * as THREE from 'three/webgpu';
import { nearTouchPair, trefoil } from '../../src/core/fixtures';
import { DEFAULTS } from '../../src/core/optimizer';
import {
    calculateDisjointPairs,
    calculateEnergy,
    gradientAnalytical,
} from '../../src/core/tangentPointEnergy';
import { type Edge, testConfigs, type Vec3 } from '../../src/core/testConfigs';
import { makeReduceSum, REDUCE_MAX_EDGES } from '../../src/gpu/reduce';
import { GpuTangentPoint } from '../../src/gpu/tangentPoint';
import { packTopology } from '../../src/gpu/topology';
import { relErrComparator } from './compare';

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

/**
 * Per-vertex triage row of `phase1KernelSmoke`: GPU vs CPU-f64 dE at one
 * vertex. `relErr` = |dE_gpu − dE_cpu| / max(|dE_cpu|, 1e-12) (vector norms).
 * @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (Task 3b, D1 triage)
 */
type VertexRow = { v: number; gpu: Vec3; cpu: Vec3; relErr: number; finite: boolean };

/**
 * Kernel-compile smoke check: the FIRST time the Task 1 WGSL is compiled on
 * the real GPU through `GpuTangentPoint`. Three fixtures — `trefoil(12)`,
 * the `crossing` preset (an exactly-collinear disjoint pair, the D3 strict
 * `>` guard path where guard == rc == 0) and `nearTouchPair(1e-6)` (the T1
 * cancellation fixture). Per fixture: build, `gradient()`, `energy()`.
 * Assertions: (a) no exception (shader compiled and bound), (b) every
 * gradient component finite, (c) per-vertex relErr < 1e-4 against
 * `gradientAnalytical`, (d) energy relErr < 1e-5 against `calculateEnergy`.
 * PASS = (a) ∧ (b) ∧ (d); (c) is INFORMATIONAL triage output (per-vertex
 * relative error is not a pre-registered gate — the gates are T1–T3 in
 * `phase1Tolerance`). Every per-vertex row is returned so a wrong-role /
 * wrong-sign term is visible immediately (plan D1 triage).
 * NOTE: a WGSL compile failure surfaces as a Dawn validation error on the
 * console (captured in the result's `consoleLines` by drive.ts) plus a
 * zero/garbage readback — so (b)/(d) catch it even when nothing throws.
 * @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (Task 3b, Step 4½; D1, D3)
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.2, §2.3
 */
phase1Spikes.phase1KernelSmoke = async () => {
    const renderer = new THREE.WebGPURenderer();
    await renderer.init();
    const { alpha, beta, epsilon } = DEFAULTS;
    const crossing = testConfigs.find((c) => c.id === 'crossing');
    if (!crossing) throw new Error('phase1KernelSmoke: crossing preset missing');
    const fixtures: { name: string; vertices: Vec3[]; edges: Edge[] }[] = [
        { name: 'trefoil12', ...trefoil(12) },
        { name: 'crossing', ...crossing.generate() },
        { name: 'nearTouch1e-6', ...nearTouchPair(1e-6) },
    ];
    const results = [];
    for (const f of fixtures) {
        // Same disjoint-pair source as the CPU reference ⇒ identical (I,J) set and order (plan D1).
        const dp = calculateDisjointPairs(f.edges);
        const cpuE = calculateEnergy(f.vertices, f.edges, dp, alpha, beta, epsilon);
        const cpuG = gradientAnalytical(f.vertices, f.edges, dp, alpha, beta, epsilon);
        let gpu: GpuTangentPoint | undefined;
        try {
            gpu = new GpuTangentPoint(renderer, packTopology(f.vertices.length, f.edges, dp));
            const grad = await gpu.gradient(f.vertices);
            const energy = await gpu.energy(f.vertices);
            const rows: VertexRow[] = grad.dE.map((g, v) => {
                const c = cpuG[v];
                const diff = Math.hypot(g[0] - c[0], g[1] - c[1], g[2] - c[2]);
                const relErr = diff / Math.max(Math.hypot(c[0], c[1], c[2]), 1e-12);
                return { v, gpu: g, cpu: c, relErr, finite: g.every(Number.isFinite) };
            });
            const finite = rows.every((r) => r.finite);
            const energyRelErr = relErrComparator(energy.energy, cpuE);
            const energyPass = energyRelErr < 1e-5;
            const maxRelErr = Math.max(...rows.map((r) => r.relErr));
            results.push({
                name: f.name,
                nV: f.vertices.length,
                nE: f.edges.length,
                compiled: true,
                finite,
                energy: { gpu: energy.energy, cpu64: cpuE, relErr: energyRelErr, pass: energyPass },
                // (c) informational — NOT part of `pass`.
                perVertex: { maxRelErr, allBelow1e4: maxRelErr < 1e-4 },
                wallMs: { gradient: grad.wallMs, energy: energy.wallMs },
                rows,
                pass: finite && energyPass,
            });
        } catch (e) {
            results.push({
                name: f.name,
                nV: f.vertices.length,
                nE: f.edges.length,
                compiled: false,
                error: e instanceof Error ? `${e.message}\n${e.stack ?? ''}` : String(e),
                pass: false,
            });
        } finally {
            gpu?.dispose();
        }
    }
    return {
        gate: 'phase1KernelSmoke',
        params: { alpha, beta, epsilon },
        fixtures: results,
        pass: results.every((r) => r.pass),
    };
};
