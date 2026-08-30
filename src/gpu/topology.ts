// src/gpu/topology.ts — pure CPU-side packing + D3 guard mirrors.
// @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (D1, D3)
import type { Edge, Vec3 } from '../core/testConfigs';
import { F32_U, GUARD_FACTOR } from './wgsl';

/**
 * CSR-packed topology uploaded once per graphVersion; consumed by the WGSL
 * kernels in `src/gpu/wgsl.ts` (`tpGradient`, `tpVertexGather`, `tpEnergy`).
 * `incidentSlots` entries are s = 2·edge + endpoint(0|1) — the slot contract
 * of `gradientKernelWgsl` / `vertexGatherWgsl`.
 * @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (D1)
 */
export interface PackedTopology {
    vertexCount: number;
    edgeCount: number;
    /** [i1, i2] per edge (2E). */
    edges: Uint32Array;
    /** E+1, CSR. */
    pairOffsets: Uint32Array;
    /** J's in disjointPairs[I] order. */
    pairIndices: Uint32Array;
    /** V+1, CSR. */
    incidentOffsets: Uint32Array;
    /** slot = 2*edge + endpoint(0|1). */
    incidentSlots: Uint32Array;
}

/**
 * CSR-flatten edges, disjoint pairs (calculateDisjointPairs order preserved —
 * the GPU loops the same (I,J) sequence the CPU does) and vertex→endpoint-slot
 * incidence (slot = 2·edge + endpoint). Uploaded once per graphVersion
 * (spec §2.2). Never sorts: stored order is part of reproducibility.
 * Malformed input is a programmer error (throws), not a runtime fallback.
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.2
 * @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (D1)
 */
export function packTopology(
    vertexCount: number,
    edges: Edge[],
    disjointPairs: number[][],
): PackedTopology {
    const E = edges.length;
    if (disjointPairs.length !== E)
        throw new Error('packTopology: disjointPairs.length !== edges.length');
    const flat = new Uint32Array(2 * E);
    const deg = new Uint32Array(vertexCount + 1);
    for (let I = 0; I < E; I++) {
        const [a, b] = edges[I];
        // Non-integers are rejected too: a fractional index silently truncates in the Uint32Array
        // write below and would corrupt the CSR (same message — it is one "bad index" class). @see plan D1
        if (
            !Number.isInteger(a) ||
            !Number.isInteger(b) ||
            a < 0 ||
            b < 0 ||
            a >= vertexCount ||
            b >= vertexCount
        )
            throw new Error('packTopology: vertex index out of range');
        flat[2 * I] = a;
        flat[2 * I + 1] = b;
        deg[a + 1]++;
        deg[b + 1]++;
    }
    const pairOffsets = new Uint32Array(E + 1);
    for (let I = 0; I < E; I++) pairOffsets[I + 1] = pairOffsets[I] + disjointPairs[I].length;
    const pairIndices = new Uint32Array(pairOffsets[E]);
    for (let I = 0; I < E; I++) {
        for (let k = 0; k < disjointPairs[I].length; k++) {
            const J = disjointPairs[I][k];
            if (J === I) throw new Error('packTopology: self-pair');
            if (J < 0 || J >= E) throw new Error('packTopology: pair index out of range');
            const [a, b] = edges[I];
            const [c, d] = edges[J];
            if (a === c || a === d || b === c || b === d)
                throw new Error('packTopology: listed pair shares a vertex');
            pairIndices[pairOffsets[I] + k] = J;
        }
    }
    const incidentOffsets = new Uint32Array(vertexCount + 1);
    for (let v = 0; v < vertexCount; v++) incidentOffsets[v + 1] = incidentOffsets[v] + deg[v + 1];
    const fill = incidentOffsets.slice(0, vertexCount);
    const incidentSlots = new Uint32Array(incidentOffsets[vertexCount]);
    for (let I = 0; I < E; I++) {
        // slot = 2·I + endpoint — must match tpGradient's write layout (wgsl.ts gradientKernelWgsl TSDoc). @see plan D1
        incidentSlots[fill[flat[2 * I]]++] = 2 * I;
        incidentSlots[fill[flat[2 * I + 1]]++] = 2 * I + 1;
    }
    return {
        vertexCount,
        edgeCount: E,
        edges: flat,
        pairOffsets,
        pairIndices,
        incidentOffsets,
        incidentSlots,
    };
}

/**
 * In-place interleaved [hi.xyz, lo.xyz] write (zero per-step allocation):
 * hi = fround(c), lo = fround(c − hi) — same split as fixtures.ts splitHiLo.
 * @see src/core/fixtures.ts (splitHiLo), spec §2.3
 */
export function writeHiLoInterleaved(vertices: Vec3[], target: Float32Array): void {
    if (target.length !== 6 * vertices.length)
        throw new Error(
            `writeHiLoInterleaved: expected ${6 * vertices.length} floats, got ${target.length}`,
        );
    for (let i = 0; i < vertices.length; i++)
        for (let d = 0; d < 3; d++) {
            const c = vertices[i][d];
            const h = Math.fround(c);
            target[6 * i + d] = h;
            target[6 * i + 3 + d] = Math.fround(c - h);
        }
}

/**
 * D3 mirror of WGSL `diffGuard`: |a−b| below the two-float noise floor 32·u²·max|hi|.
 * This mirror evaluates the difference in f64, whereas the kernel forms it as the f32
 * two-float sum `diff2`; it is therefore a classification oracle for tests, not a
 * bit-identical twin of the WGSL at the guard boundary — boundary cases are decided by
 * the WGSL, so do not "tighten" this to match a GPU result.
 * @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (D3)
 * @see src/gpu/wgsl.ts (pairKernelWgsl → diffGuard)
 */
export function diffDegenerate(a: Vec3, b: Vec3): boolean {
    const k = GUARD_FACTOR * F32_U * F32_U;
    let g2 = 0;
    let n2 = 0;
    for (let d = 0; d < 3; d++) {
        const g = k * Math.max(Math.abs(Math.fround(a[d])), Math.abs(Math.fround(b[d])));
        g2 += g * g;
        const diff = a[d] - b[d];
        n2 += diff * diff;
    }
    // `<=`, not `<`: a guard of exactly 0 with a length of exactly 0 IS degenerate (mirrors WGSL `if (r > guard)`). @see plan D3
    return Math.sqrt(n2) <= Math.sqrt(g2);
}

/**
 * D3 mirror of WGSL `crossGuard`: |e×d| at or below 32·u·‖per-component product scale‖.
 * Like `diffDegenerate`, the products and the cross are evaluated in f64 here while the
 * kernel computes them in f32 (its `d` coming from the two-float `diff2`); it is a
 * classification oracle for tests, not bit-identical to the WGSL at the guard boundary —
 * boundary cases are decided by the WGSL.
 * @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (D3)
 * @see src/gpu/wgsl.ts (pairKernelWgsl → crossGuard)
 */
export function crossDegenerate(e: Vec3, d: Vec3): boolean {
    const c = [e[1] * d[2] - e[2] * d[1], e[2] * d[0] - e[0] * d[2], e[0] * d[1] - e[1] * d[0]];
    const s = [
        Math.abs(e[1] * d[2]) + Math.abs(e[2] * d[1]),
        Math.abs(e[2] * d[0]) + Math.abs(e[0] * d[2]),
        Math.abs(e[0] * d[1]) + Math.abs(e[1] * d[0]),
    ];
    // `<=` — guard may be exactly 0 (collinear axis-aligned, `crossing` preset); see diffDegenerate. @see plan D3
    return Math.hypot(...c) <= GUARD_FACTOR * F32_U * Math.hypot(...s);
}
