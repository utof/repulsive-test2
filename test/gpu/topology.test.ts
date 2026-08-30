// test/gpu/topology.test.ts
import { expect, test } from 'bun:test';
import { nearTouchPair, splitHiLo, trefoil } from '../../src/core/fixtures';
import { calculateDisjointPairs } from '../../src/core/tangentPointEnergy';
import type { Vec3 } from '../../src/core/testConfigs';
import { testConfigs } from '../../src/core/testConfigs';
import {
    crossDegenerate,
    diffDegenerate,
    packTopology,
    writeHiLoInterleaved,
} from '../../src/gpu/topology';

test('packTopology: exact layout on a 4-vertex chain', () => {
    const edges: [number, number][] = [
        [0, 1],
        [1, 2],
        [2, 3],
    ];
    const dp = calculateDisjointPairs(edges); // [[2],[],[0]]
    const t = packTopology(4, edges, dp);
    expect(Array.from(t.edges)).toEqual([0, 1, 1, 2, 2, 3]);
    expect(Array.from(t.pairOffsets)).toEqual([0, 1, 1, 2]);
    expect(Array.from(t.pairIndices)).toEqual([2, 0]);
    expect(Array.from(t.incidentOffsets)).toEqual([0, 1, 3, 5, 6]);
    expect(Array.from(t.incidentSlots)).toEqual([0, 1, 2, 3, 4, 5]);
});

test('packTopology: CSR mirrors calculateDisjointPairs order on trefoil(12)', () => {
    const { vertices, edges } = trefoil(12);
    const dp = calculateDisjointPairs(edges);
    const t = packTopology(vertices.length, edges, dp);
    for (let I = 0; I < 12; I++)
        expect(Array.from(t.pairIndices.slice(t.pairOffsets[I], t.pairOffsets[I + 1]))).toEqual(
            dp[I],
        );
});

test('packTopology: rejects malformed topology with exact messages', () => {
    expect(() => packTopology(2, [[0, 5]], [[]])).toThrow(
        'packTopology: vertex index out of range',
    );
    expect(() => packTopology(2, [[0, 1]], [])).toThrow(
        'packTopology: disjointPairs.length !== edges.length',
    );
    expect(() => packTopology(2, [[0, 0.5]], [[]])).toThrow(
        'packTopology: vertex index out of range',
    );
    expect(() => packTopology(2, [[0, 1]], [[7]])).toThrow('packTopology: pair index out of range');
    expect(() => packTopology(2, [[0, 1]], [[0]])).toThrow('packTopology: self-pair');
    expect(() =>
        packTopology(
            3,
            [
                [0, 1],
                [1, 2],
            ],
            [[1], [0]],
        ),
    ).toThrow('packTopology: listed pair shares a vertex');
});

test('writeHiLoInterleaved: [hi.xyz, lo.xyz] per vertex, equals splitHiLo, exact-length only', () => {
    const { vertices } = trefoil(5);
    const ref = splitHiLo(vertices);
    const out = new Float32Array(30);
    writeHiLoInterleaved(vertices, out);
    for (let i = 0; i < 5; i++)
        for (let d = 0; d < 3; d++) {
            expect(out[6 * i + d]).toBe(ref.hi[3 * i + d]);
            expect(out[6 * i + 3 + d]).toBe(ref.lo[3 * i + d]);
        }
    expect(() => writeHiLoInterleaved(vertices, new Float32Array(29))).toThrow(
        'writeHiLoInterleaved: expected 30 floats, got 29',
    );
});

test('D3 guards: coincident ⇒ degenerate; near-touch(1e-6) ⇒ not; collinear cross ⇒ degenerate (incl. guard exactly 0); scale-invariant', () => {
    const { vertices } = nearTouchPair(1e-6);
    expect(diffDegenerate(vertices[0], vertices[0])).toBe(true);
    expect(diffDegenerate([0, 0, 0], [0, 0, 0])).toBe(true); // guard exactly 0 AND length 0 ⇒ degenerate (≤)
    expect(diffDegenerate(vertices[0], vertices[2])).toBe(false);
    const e: Vec3 = [4, 0, 0];
    const d: Vec3 = [0, -1e-6, 0];
    expect(crossDegenerate(e, d)).toBe(false);
    expect(crossDegenerate(e, [2, 0, 0])).toBe(true); // axis-aligned collinear: every product 0 ⇒ guard 0 ⇒ must still be degenerate
    expect(crossDegenerate([1, 1, 0], [2, 2, 0])).toBe(true); // collinear, non-zero products
    const s = (v: Vec3): Vec3 => [v[0] * 0.05, v[1] * 0.05, v[2] * 0.05];
    expect(diffDegenerate(s(vertices[0]), s(vertices[2]))).toBe(false);
    expect(crossDegenerate(s(e), s(d))).toBe(false);
});

test('D3 guards on the crossing preset: its collinear disjoint pair (edges 0 and 2) is degenerate, its non-collinear pairs are not', () => {
    const crossing = testConfigs.find((c) => c.id === 'crossing');
    if (!crossing) throw new Error('crossing preset missing');
    const { vertices, edges } = crossing.generate();
    const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    const e0 = sub(vertices[edges[0][1]], vertices[edges[0][0]]);
    const d02 = sub(vertices[edges[0][0]], vertices[edges[2][0]]); // v0 − v2, on the same line
    expect(crossDegenerate(e0, d02)).toBe(true);
    const d03 = sub(vertices[edges[0][0]], vertices[edges[3][0]]); // v0 − v4, off the line
    expect(crossDegenerate(e0, d03)).toBe(false);
});
