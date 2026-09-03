// bench/gpu/phase1.ts — Phase 1 gates through the PRODUCTION kernel pieces.
// Registered into the Phase 0 spike registry by spikes.ts (Object.assign).
// @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.2, §3 (T1–T3), §4 G2, §5 Phase 1
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
import { cosineComparator, relErrComparator, withSeed } from './compare';

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

/**
 * Build one `GpuTangentPoint` for a fixture, using the SAME disjoint-pair
 * source as the CPU reference so GPU and CPU evaluate an identical (I,J)
 * set in the same order (plan D1). The caller owns `gpu` (dispose in
 * `finally`).
 * @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (Task 3c, Step 5)
 */
const engine = (r: THREE.WebGPURenderer, vertices: Vec3[], edges: Edge[]) => {
    const dp = calculateDisjointPairs(edges);
    return { dp, gpu: new GpuTangentPoint(r, packTopology(vertices.length, edges, dp)) };
};

/**
 * A preset's `params` at their declared defaults — the plan's "every
 * `testConfigs` preset at default params" (global-constraints, Fixtures).
 * @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (Task 3c, Step 5)
 */
const defaults = (c: (typeof testConfigs)[number]) =>
    Object.fromEntries((c.params ?? []).map((q) => [q.name, q.default]));

/**
 * The pre-registered Phase 1 tolerance gate through the PRODUCTION kernel.
 * T1 = single-pair energy on the 2-edge `nearTouchPair(gap)` graph for
 * gap ∈ {1e-3, 1e-4, 1e-5, 1e-6}, relErr < 1e-5 vs CPU-f64 `calculateEnergy`;
 * the gap=1e-6 row doubles as the G2-production datapoint (`g2Production`).
 * T2 = total energy relErr < 1e-6 and T3 = gradient cosine > 1 − 1e-6 on
 * every `testConfigs` preset at default params (Math.random seeded with
 * 0x5eed for the whole fixture list, so `random`/`chain` are reproducible)
 * plus `trefoil(240)` and `trefoil(960)`. Every fixture must also return an
 * all-finite gradient (plan D3 — `crossing`'s exactly-collinear disjoint
 * pair exercises the strict `>` guard). Tolerances, fixtures and the seed
 * are pre-registered and NOT negotiable; a red row stops the branch.
 * T2 error budget: the pre-registered expected band is [5e-8, 1e-6]; a
 * value below 1e-9 is investigated as a vacuous comparison (plan D4).
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §3 (T1–T3), §4 G2, §5 Phase 1
 * @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (Task 3c; D1, D3, D4)
 */
phase1Spikes.phase1Tolerance = async () => {
    const renderer = new THREE.WebGPURenderer();
    await renderer.init();
    const { alpha, beta, epsilon } = DEFAULTS;
    const t1 = [];
    for (const gap of [1e-3, 1e-4, 1e-5, 1e-6]) {
        const { vertices, edges } = nearTouchPair(gap);
        const { dp, gpu } = engine(renderer, vertices, edges);
        try {
            const cpu64 = calculateEnergy(vertices, edges, dp, alpha, beta, epsilon);
            const { energy } = await gpu.energy(vertices);
            const relErr = relErrComparator(energy, cpu64);
            t1.push({ gap, gpu: energy, cpu64, relErr, pass: relErr < 1e-5 });
        } finally {
            gpu.dispose();
        }
    }
    const fixtures = withSeed(0x5eed, () => [
        ...testConfigs.map((c) => ({ name: c.id, ...c.generate(defaults(c)) })),
        { name: 'trefoil240', ...trefoil(240) },
        { name: 'trefoil960', ...trefoil(960) },
    ]);
    const t23 = [];
    for (const f of fixtures) {
        const { dp, gpu } = engine(renderer, f.vertices, f.edges);
        try {
            const e64 = calculateEnergy(f.vertices, f.edges, dp, alpha, beta, epsilon);
            const g64 = gradientAnalytical(f.vertices, f.edges, dp, alpha, beta, epsilon);
            const { energy } = await gpu.energy(f.vertices);
            const { dE } = await gpu.gradient(f.vertices);
            const t2 = relErrComparator(energy, e64);
            const t3 = cosineComparator(dE.flat(), g64.flat());
            // D3: a NaN anywhere (e.g. `crossing`'s collinear pair) fails the fixture.
            const finite = dE.every((v) => v.every(Number.isFinite));
            t23.push({
                name: f.name,
                nV: f.vertices.length,
                t2,
                t2pass: t2 < 1e-6,
                t3,
                t3pass: t3 > 1 - 1e-6,
                finite,
            });
        } finally {
            gpu.dispose();
        }
    }
    const pass = t1.every((r) => r.pass) && t23.every((r) => r.t2pass && r.t3pass && r.finite);
    return {
        gate: 'phase1Tolerance',
        seed: 0x5eed,
        t1,
        t23,
        g2Production: t1.find((r) => r.gap === 1e-6),
        worst: {
            t1: Math.max(...t1.map((r) => r.relErr)),
            t2: Math.max(...t23.map((r) => r.t2)),
            t3: Math.min(...t23.map((r) => r.t3)),
        },
        pass,
    };
};

/**
 * Median of `xs` (upper middle for an even count). Used instead of the mean
 * for the perf gate because a single scheduler/driver hiccup in a 5-run
 * sample moves a mean but not a median, and the gate is pre-registered as
 * "medians" — averaging or cherry-picking runs is explicitly forbidden.
 * Sorts a copy so the caller's raw sample array stays in run order in the
 * committed JSON.
 * @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (D4 timing method; Task 4)
 */
const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];

/**
 * The pre-registered Phase 1 PERF gate: GPU dE must be >= 5x faster than CPU
 * dE at N = 480 and N = 960. Method D4, and every part of it is load-bearing:
 * `gpu.gradient()`'s `wallMs` spans pack -> upload -> ONE `renderer.compute`
 * -> `getArrayBufferAsync` readback -> `Vec3[]` construction, so the readback
 * the solver actually pays for is INSIDE the measured window (readback-
 * inclusive timing is not negotiable, global-constraints "Gates"). CPU dE is
 * `gradientAnalytical` timed in the SAME browser on the SAME arrays, so the
 * ratio is not contaminated by a Bun-vs-Chrome JIT difference. One CPU and
 * one GPU warm-up precede the timed runs; the GPU warm-up exists to pay the
 * shader compile + pipeline creation once, which would otherwise dominate
 * run 1. Then 5 timed runs each, interleaved, and the medians are compared.
 * `gpuComputeOnlyP50Ms` (WebGPU timestamp queries, requires the renderer's
 * `trackTimestamp: true`) is INFORMATIONAL only — the gate is wall-clock; a
 * 0/undefined timestamp on some driver is recorded as-is and changes nothing.
 * `fullStepGate: false` records that this measures dE alone, not a whole
 * descent step (Phase 1 computes only dE on the GPU, global-constraints).
 * A red row STOPS THE BRANCH (spec §5); the 5x threshold is pre-registered
 * and NOT negotiable.
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §5 (Phase 1 stop condition)
 * @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (Task 4; D4 timing method)
 */
phase1Spikes.phase1DeGate = async () => {
    // trackTimestamp: the ONLY reason for it here is `gpuComputeOnlyP50Ms`
    // (informational). Without it `resolveTimestampsAsync` warns and returns
    // undefined (three r185.1, Backend.js:599-604).
    const renderer = new THREE.WebGPURenderer({ trackTimestamp: true });
    await renderer.init();
    const { alpha, beta, epsilon } = DEFAULTS;
    const rows = [];
    for (const N of [480, 960]) {
        const { vertices, edges } = trefoil(N);
        const { dp, gpu } = engine(renderer, vertices, edges);
        try {
            gradientAnalytical(vertices, edges, dp, alpha, beta, epsilon); // CPU warm-up
            await gpu.gradient(vertices); // GPU warm-up (pipeline compile) — NOT timed
            const cpuMs: number[] = [];
            const gpuMs: number[] = [];
            const gpuOnly: number[] = [];
            for (let r = 0; r < 5; r++) {
                const { wallMs } = await gpu.gradient(vertices);
                gpuMs.push(wallMs); // D4: pack+upload+compute+readback+Vec3[]
                await renderer.resolveTimestampsAsync(THREE.TimestampQuery.COMPUTE);
                gpuOnly.push(renderer.info.compute.timestamp);
                const t0 = performance.now();
                gradientAnalytical(vertices, edges, dp, alpha, beta, epsilon);
                cpuMs.push(performance.now() - t0);
            }
            const speedup = median(cpuMs) / median(gpuMs);
            rows.push({
                N,
                cpuMs,
                gpuMs,
                cpuP50Ms: median(cpuMs),
                gpuP50Ms: median(gpuMs),
                gpuComputeOnlyP50Ms: median(gpuOnly),
                speedup,
                pass: speedup >= 5,
            });
        } finally {
            gpu.dispose(); // per-N: each N owns its buffers; leaking N=480's would skew N=960.
        }
    }
    return {
        gate: 'phase1DeGate',
        methodology:
            'wall-clock incl. pack+upload+compute+readback+Vec3[]; 1 warm-up, 5 runs, medians; same browser for CPU',
        fullStepGate: false,
        rows,
        pass: rows.every((r) => r.pass),
    };
};
