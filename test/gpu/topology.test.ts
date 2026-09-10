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

// --- packTopology validation: one endpoint / one bound at a time -------------
// The malformed-topology test above trips each guard, but always via a single
// witness that satisfies several clauses at once (index 5 against vertexCount 2
// is both `>= vertexCount` and `> vertexCount`; the shares-a-vertex pair is
// listed in BOTH directions, so `b === c` on edge 0 and `a === d` on edge 1
// each throw on their own). That leaves the per-endpoint and per-bound halves
// of every guard unpinned. The cases below split them: each input violates
// exactly ONE clause of one guard, and every out-of-range index is at the
// inclusive boundary where `>=` and `>` disagree.
// @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (D1)

test('packTopology: vertex-index guard rejects each endpoint separately, and vertexCount itself', () => {
    const msg = 'packTopology: vertex index out of range';
    // Negative: Number.isInteger(-1) is true, so only the `< 0` clause can fire.
    expect(() => packTopology(2, [[-1, 1]], [[]])).toThrow(msg); // a < 0
    expect(() => packTopology(2, [[0, -1]], [[]])).toThrow(msg); // b < 0
    expect(() => packTopology(2, [[5, 0]], [[]])).toThrow(msg); // a >= vertexCount
    // a === vertexCount / b === vertexCount: the boundary `>=` must reject and `>` would not.
    expect(() => packTopology(2, [[2, 0]], [[]])).toThrow(msg);
    expect(() => packTopology(2, [[0, 2]], [[]])).toThrow(msg);
});

test('packTopology: pair-index guard rejects negative J and J === edgeCount', () => {
    const msg = 'packTopology: pair index out of range';
    expect(() => packTopology(2, [[0, 1]], [[-1]])).toThrow(msg); // J < 0
    expect(() => packTopology(2, [[0, 1]], [[1]])).toThrow(msg); // J === E, `>=` boundary
});

test('packTopology: shares-a-vertex guard fires on each of the four endpoint identities', () => {
    const msg = 'packTopology: listed pair shares a vertex';
    // Each pairing is listed one-way ([[1], []]) so only edge 0's check can throw,
    // and the shared vertex sits in exactly one of the four positions.
    expect(() =>
        packTopology(
            3,
            [
                [1, 0],
                [1, 2],
            ],
            [[1], []],
        ),
    ).toThrow(msg); // a === c
    expect(() =>
        packTopology(
            3,
            [
                [1, 0],
                [2, 1],
            ],
            [[1], []],
        ),
    ).toThrow(msg); // a === d
    expect(() =>
        packTopology(
            3,
            [
                [0, 1],
                [1, 2],
            ],
            [[1], []],
        ),
    ).toThrow(msg); // b === c
    expect(() =>
        packTopology(
            3,
            [
                [0, 1],
                [2, 1],
            ],
            [[1], []],
        ),
    ).toThrow(msg); // b === d
});

test('packTopology: a vertex that is the FIRST endpoint of two edges gets both slots', () => {
    // Chain and trefoil fixtures are 2-regular with every vertex appearing once as
    // endpoint 0 and once as endpoint 1, so the `fill[...]++` bump on the endpoint-0
    // write is never read back by a second endpoint-0 write. A fan (vertex 0 first in
    // both edges) is the smallest topology that reads it.
    // @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (D1)
    const t = packTopology(
        3,
        [
            [0, 1],
            [0, 2],
        ],
        [[], []],
    );
    expect(Array.from(t.incidentOffsets)).toEqual([0, 2, 3, 4]);
    expect(Array.from(t.incidentSlots)).toEqual([0, 2, 1, 3]); // v0 → slots 0 and 2
});

// --- D3 guards: magnitudes, signs and cancellation ---------------------------
// The guards are RELATIVE: diffGuard is k·max|hi| per axis with k = 32·u², and
// crossGuard is 32·u·‖per-axis sum of |products|‖. Every D3 case above uses
// O(1) coordinates and either exactly coincident or plainly separated inputs,
// where k·x ≈ k/x, max ≈ min, and every cross component is an exactly-zero
// product — so the scaling, the max/min choice, the accumulation sign and the
// cancellation-sensitive scale rows are all invisible.
// @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (D3)
// @see src/gpu/wgsl.ts (pairKernelWgsl → diffGuard, crossGuard)

test('diffDegenerate: the guard shrinks with the coordinate scale (k·max, not k/max)', () => {
    // At 1e-9 the guard is k·1e-9 ≈ 1.1e-22, far below the 1e-9 separation ⇒ not
    // degenerate. Had the guard been k/max it would be ≈1.1e-4 and swallow it.
    expect(diffDegenerate([1e-9, 0, 0], [0, 0, 0])).toBe(false);
});

test('diffDegenerate: the guard follows the LARGER endpoint (max, not min)', () => {
    // 1e39 overflows f32, so Math.fround gives Infinity: the mirror reports what the
    // kernel would see once the coordinate is uploaded as f32 — an infinite noise
    // floor, hence degenerate. min(|hi_a|, |hi_b|) would be 1 and report separated.
    expect(Math.fround(1e39)).toBe(Number.POSITIVE_INFINITY);
    expect(diffDegenerate([1e39, 0, 0], [1, 0, 0])).toBe(true);
});

test('diffDegenerate: a non-zero separation strictly inside the guard is degenerate', () => {
    // 1e-14 apart at |x| = 1: guard = 32·u² ≈ 1.14e-13 > 9.99e-15 ⇒ degenerate.
    // Every other degenerate case in the suite has an identically zero difference,
    // which cannot tell an accumulating ‖·‖² from one that cancels itself out.
    expect(1 + 1e-14 - 1).toBeGreaterThan(0);
    expect(diffDegenerate([1, 0, 0], [1 + 1e-14, 0, 0])).toBe(true);
});

test('crossDegenerate: collinear with all three components non-zero ⇒ every cross row cancels', () => {
    // [1,1,1] × [2,2,2]: each row is p − p with p = 2 ≠ 0, so a row that added
    // instead of subtracting, or that divided instead of multiplying, yields a
    // cross of magnitude ≫ the guard. Every earlier collinear case has at least
    // one identically-zero factor in rows 0 and 1.
    expect(crossDegenerate([1, 1, 1], [2, 2, 2])).toBe(true);
});

test('crossDegenerate: near-collinear at large magnitude is degenerate on each scale row', () => {
    // The scale rows exist to size the guard by the magnitude of the products that
    // CANCEL, not by the residue. Each case below is a near-collinear pair in one
    // coordinate plane: |c| ≈ 0.1 while the surviving scale row is ≈2e8, so the
    // guard 32·u·2e8 ≈ 381 covers it. A row that subtracted its two |products|
    // instead of adding them (or an empty scale vector) collapses to ≈0.1 or 0.
    expect(crossDegenerate([1e8, 1, 0], [1e8, 1 + 1e-9, 0])).toBe(true); // row 2
    expect(crossDegenerate([0, 1e8, 1], [0, 1e8, 1 + 1e-9])).toBe(true); // row 0
    expect(crossDegenerate([1, 0, 1e8], [1 + 1e-9, 0, 1e8])).toBe(true); // row 1
});

test('crossDegenerate: a small factor must shrink its scale row, never inflate it', () => {
    // Near-orthogonal vectors with a 1e-8 x-component: |e×d| ≈ 1 against a guard of
    // 32·u·‖s‖ ≈ 1.9e-6 ⇒ separated. Replacing a product by a quotient in a scale
    // row turns that 1e-8 into 1e8 and inflates the guard to ≈191, swallowing the
    // whole cross — the mirror would then call every crossing pair degenerate.
    expect(crossDegenerate([0, 0, 1], [1e-8, 1, 0])).toBe(false); // row 1, e[2]·d[0]
    expect(crossDegenerate([0, 1, 0], [1e-8, 0, 1])).toBe(false); // row 2, e[1]·d[0]
});

// No test pins the two CSR loop bounds `v < vertexCount` and `I < E` against their
// `<=` off-by-one. Both extra iterations are unobservable, not untested: the `v <=`
// iteration reads deg[vertexCount + 1] (past a Uint32Array of length vertexCount + 1
// ⇒ undefined ⇒ NaN) and writes incidentOffsets[vertexCount + 1], also past the end;
// the `I <= E` iteration reads flat[2E] (past the end), so both fill[] lookups key on
// "undefined" — an expando on a slice-local array — and both incidentSlots writes land
// on index NaN. Out-of-range typed-array writes are silent no-ops even under ESM strict
// mode, so neither iteration can change a returned field or throw. Verified
// byte-identical to clean source over a 14848-row topology battery.
// @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (D1)
