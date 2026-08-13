import type { Edge, Vec3 } from './testConfigs';

/**
 * Parametric closed trefoil at arbitrary N — the canonical perf/verification
 * fixture for the WebGPU milestone's fixture matrix (N=60/120/240/480/960/1000).
 * Body moved VERBATIM from bench/sobolev.bench.ts so existing baseline results
 * stay comparable. Deterministic: no Math.random.
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §3 (fixture debt)
 */
export function trefoil(n: number): { vertices: Vec3[]; edges: Edge[] } {
    const vertices: Vec3[] = [];
    const edges: Edge[] = [];
    for (let i = 0; i < n; i++) {
        const t = (2 * Math.PI * i) / n;
        vertices.push([
            Math.sin(t) + 2 * Math.sin(2 * t),
            Math.cos(t) - 2 * Math.cos(2 * t),
            -Math.sin(3 * t),
        ]);
        edges.push([i, (i + 1) % n]);
    }
    return { vertices, edges };
}

/**
 * Two parallel edges at controlled closest-approach `gap`, positioned at O(1)
 * coordinates that are NOT exactly representable in f32 (0.1/2.1/0.3/4.1) —
 * REQUIRED so that f32 position storage corrupts the gap via catastrophic
 * cancellation, which is the failure mode T1/G2 exist to detect. An
 * axis-aligned layout through the origin (exact f32 coords) would cancel
 * nothing and make every gate built on this fixture vacuous (plan review F1).
 * Layout: A (0.1,2.1,0.3)→(4.1,2.1,0.3); B (0.1,2.1+gap,0.3)→(4.1,2.1+gap,0.3).
 * Disjoint (no shared vertices) → the pair contributes energy.
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §3 (T1), §4 (G2)
 * @see docs/2026-08-13-ai-research-gpu-precision.md Q1 (extent/gap error law)
 */
export function nearTouchPair(gap: number): {
    vertices: Vec3[];
    edges: Edge[];
    gap: number;
} {
    const vertices: Vec3[] = [
        [0.1, 2.1, 0.3],
        [4.1, 2.1, 0.3],
        [0.1, 2.1 + gap, 0.3],
        [4.1, 2.1 + gap, 0.3],
    ];
    return {
        vertices,
        edges: [
            [0, 1],
            [2, 3],
        ],
        gap,
    };
}

/**
 * Split each vertex coordinate into an f32 hi/lo pair: `hi = fround(c)`,
 * `lo = fround(c - hi)` (the residual of the f32 rounding, itself rounded to
 * f32) — the two-float positions trick that survives near-touch cancellation
 * GPU-side, since `(hi_i − hi_j) + (lo_i − lo_j)` recovers precision the
 * plain-f32 `hi_i − hi_j` alone loses. Flat xyz layout (index `3*i+0/1/2`),
 * matching the storage-buffer layout the G2 spike/production kernel upload.
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.3 (two-float positions)
 * @see docs/2026-08-13-ai-research-gpu-precision.md Q1 (extent/gap error law), Q3 (residual reassociation risk)
 */
export function splitHiLo(v: Vec3[]): { hi: Float32Array; lo: Float32Array } {
    const hi = new Float32Array(v.length * 3);
    const lo = new Float32Array(v.length * 3);
    for (let i = 0; i < v.length; i++) {
        for (let d = 0; d < 3; d++) {
            const c = v[i][d];
            const h = Math.fround(c);
            hi[3 * i + d] = h;
            lo[3 * i + d] = Math.fround(c - h);
        }
    }
    return { hi, lo };
}
