/**
 * Task 2 of Phase 2H: the R^6 tangent-point BVH (a PORT of the spike; its measured constants are oracles).
 *
 * Oracle table: `.superpowers/sdd/2026-09-11-phase2h-queue/scratch/out-bct-torusKnot-theta0.5-leaf8-endpoint4-struct.json`
 * (git-ignored spike artefact), key `rows[].stats` (`leafCount`, `depth`, `nNodes`), params
 * `{LEAF: 8, THETA: 0.5, QUAD: "endpoint4", ALPHA: 3, BETA: 6, EPS: 1e-10}`, closed curve (m = n).
 * The three integers are fixture-independent (median split => perfectly balanced), so a trefoil-ish
 * closed curve stands in for the artefact's torus knot.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.5, §11.2 task 2, §11.3
 */
import { describe, expect, test } from 'bun:test';
import { buildBvh, edgeData } from '../../../src/core/sobolev/hierarchical/bvh';
import type { Edge, Vec3 } from '../../../src/core/testConfigs';

const EPS = 1e-10;

function knot(n: number): { vertices: Vec3[]; edges: Edge[] } {
    const vertices: Vec3[] = [];
    for (let i = 0; i < n; i++) {
        const t = (2 * Math.PI * i) / n;
        vertices.push([
            Math.sin(t) + 2 * Math.sin(2 * t),
            Math.cos(t) - 2 * Math.cos(2 * t),
            -Math.sin(3 * t),
        ]);
    }
    const edges: Edge[] = [];
    for (let i = 0; i < n; i++) edges.push([i, (i + 1) % n]);
    return { vertices, edges };
}

const build = (n: number, leaf = 8) => {
    const { vertices, edges } = knot(n);
    return { vertices, edges, bvh: buildBvh(vertices, edges, leaf, EPS) };
};

describe('oracle table (out-bct-...-struct.json rows[].stats)', () => {
    const rows: [number, number, number, number][] = [
        [960, 128, 7, 255],
        [2000, 256, 8, 511],
        [5000, 1024, 10, 2047],
        [10000, 2048, 11, 4095],
        [30000, 4096, 12, 8191],
    ];
    for (const [n, leafCount, depth, nNodes] of rows) {
        test(`n=${n}`, () => {
            const { bvh } = build(n);
            expect(bvh.leafCount).toBe(leafCount);
            expect(bvh.depth).toBe(depth);
            expect(bvh.nNodes).toBe(nNodes);
            const mean = n / bvh.leafCount;
            expect(mean).toBeGreaterThanOrEqual(4.88);
            expect(mean).toBeLessThanOrEqual(7.82);
        });
    }
});

describe('partition invariants', () => {
    for (const n of [64, 100, 256]) {
        test(`n=${n}`, () => {
            const { bvh } = build(n);
            const seen = new Uint8Array(n);
            for (const I of bvh.order) seen[I]++;
            expect(seen.every((v) => v === 1)).toBe(true);
            expect(bvh.nNodes).toBe(2 * bvh.leafCount - 1);
            for (let N = 0; N < bvh.nNodes; N++) {
                const b = bvh.begin[N];
                const e = bvh.end[N];
                if (bvh.left[N] < 0) {
                    expect(bvh.right[N]).toBe(-1);
                    expect(e - b).toBeLessThanOrEqual(8);
                    expect(e - b).toBeGreaterThan(0);
                } else {
                    const l = bvh.left[N];
                    const r = bvh.right[N];
                    expect(bvh.begin[l]).toBe(b);
                    expect(bvh.end[l]).toBe(bvh.begin[r]);
                    expect(bvh.end[r]).toBe(e);
                }
            }
        });
    }
});

describe('cluster data', () => {
    const n = 100;
    const { vertices, edges, bvh } = build(n);
    const ed = edgeData(vertices, edges, EPS);

    test('radii are exact max distance from barycentre and <= box half-diagonal', () => {
        for (let N = 0; N < bvh.nNodes; N++) {
            let mrx = 0;
            let mrt = 0;
            const lo = [Infinity, Infinity, Infinity];
            const hi = [-Infinity, -Infinity, -Infinity];
            for (let s = bvh.begin[N]; s < bvh.end[N]; s++) {
                const I = bvh.order[s];
                let dx2 = 0;
                let dt2 = 0;
                for (let c = 0; c < 3; c++) {
                    dx2 += (ed.P[3 * I + c] - bvh.C[3 * N + c]) ** 2;
                    dt2 += (ed.T[3 * I + c] - bvh.TN[3 * N + c]) ** 2;
                    lo[c] = Math.min(lo[c], ed.P[3 * I + c]);
                    hi[c] = Math.max(hi[c], ed.P[3 * I + c]);
                }
                mrx = Math.max(mrx, dx2);
                mrt = Math.max(mrt, dt2);
            }
            expect(Math.abs(bvh.rx[N] - Math.sqrt(mrx))).toBeLessThanOrEqual(
                1e-15 * (1 + bvh.rx[N]),
            );
            expect(Math.abs(bvh.rt[N] - Math.sqrt(mrt))).toBeLessThanOrEqual(
                1e-15 * (1 + bvh.rt[N]),
            );
            const half = 0.5 * Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
            // Barycentre radius <= box diagonal (not half): barycentre lies in the box, so the bound is the full diagonal.
            expect(bvh.rx[N]).toBeLessThanOrEqual(2 * half + 1e-15);
        }
    });

    test('TN is the length-weighted average and NOT renormalised', () => {
        let sawShort = false;
        for (let N = 0; N < bvh.nNodes; N++) {
            let L = 0;
            const t = [0, 0, 0];
            for (let s = bvh.begin[N]; s < bvh.end[N]; s++) {
                const I = bvh.order[s];
                L += ed.ell[I];
                for (let c = 0; c < 3; c++) t[c] += ed.ell[I] * ed.T[3 * I + c];
            }
            for (let c = 0; c < 3; c++) {
                expect(Math.abs(bvh.TN[3 * N + c] - t[c] / L)).toBeLessThanOrEqual(1e-14);
            }
            expect(Math.abs(bvh.L[N] - L)).toBeLessThanOrEqual(1e-13);
            const norm = Math.hypot(bvh.TN[3 * N], bvh.TN[3 * N + 1], bvh.TN[3 * N + 2]);
            // The value that must NOT appear: renormalised ||TN|| == 1 on a curved node.
            if (bvh.end[N] - bvh.begin[N] > n / 2) {
                expect(norm).toBeLessThan(1 - 1e-6);
                sawShort = true;
            }
        }
        expect(sawShort).toBe(true);
    });

    test('maxEll is the largest edge length in the node', () => {
        for (let N = 0; N < bvh.nNodes; N++) {
            let me = 0;
            for (let s = bvh.begin[N]; s < bvh.end[N]; s++) me = Math.max(me, ed.ell[bvh.order[s]]);
            expect(bvh.maxEll[N]).toBe(me);
        }
    });

    test('slot-order copies follow `order`, ell has eps after the norm', () => {
        for (let s = 0; s < n; s++) {
            const I = bvh.order[s];
            const [a, b] = edges[I];
            expect(bvh.sE[2 * s]).toBe(a);
            expect(bvh.sE[2 * s + 1]).toBe(b);
            for (let c = 0; c < 3; c++) {
                expect(bvh.sXa[3 * s + c]).toBe(vertices[a][c]);
                expect(bvh.sXb[3 * s + c]).toBe(vertices[b][c]);
                expect(bvh.sP[3 * s + c]).toBe(ed.P[3 * I + c]);
                expect(bvh.sT[3 * s + c]).toBe(ed.T[3 * I + c]);
            }
            expect(bvh.sEll[s]).toBe(ed.ell[I]);
            const ex = vertices[b][0] - vertices[a][0];
            const ey = vertices[b][1] - vertices[a][1];
            const ez = vertices[b][2] - vertices[a][2];
            const r = Math.sqrt(ex * ex + ey * ey + ez * ez); // same op order as innerProduct.ts:69-70
            expect(bvh.sEll[s]).toBe(r + EPS);
        }
    });
});

test('determinism: identical input gives bit-identical arrays', () => {
    const a = build(256).bvh;
    const b = build(256).bvh;
    for (const k of [
        'left',
        'right',
        'begin',
        'end',
        'order',
        'L',
        'C',
        'TN',
        'rx',
        'rt',
        'maxEll',
        'sP',
        'sT',
        'sEll',
        'sE',
        'sXa',
        'sXb',
    ] as const) {
        expect(
            Buffer.from(a[k].buffer, a[k].byteOffset, a[k].byteLength).equals(
                Buffer.from(b[k].buffer, b[k].byteOffset, b[k].byteLength),
            ),
        ).toBe(true);
    }
});

test('bytes is the hand-computed sum of retained arrays (n=64, leaf 8 => 15 nodes)', () => {
    const { bvh } = build(64);
    const nodes = 15;
    const m = 64;
    // left/right/begin/end are retained as subarray VIEWS of maxNodes-capacity buffers (spike counts the
    // buffers): maxNodes = 4*ceil(m/leaf)+64 = 96. 4 B: those four + order (m) + sE (2m);
    // 8 B: L,rx,rt,maxEll (nodes) + C,TN (3 nodes each) + sP,sT,sXa,sXb (3m each) + sEll (m).
    const maxNodes = 4 * Math.ceil(m / 8) + 64;
    const expected = 4 * (4 * maxNodes + m + 2 * m) + 8 * (4 * nodes + 6 * nodes + 12 * m + m);
    expect(bvh.nNodes).toBe(nodes);
    expect(bvh.bytes).toBe(expected);
});
