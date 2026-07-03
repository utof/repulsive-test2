import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import {
    assertValidConstraintSet,
    barycenterBlock,
    type ConstraintSet,
    edgeLengths,
    edgeLengthsBlock,
    evaluateConstraintSet,
    pointBlock,
    totalLength,
    totalLengthBlock,
} from '../../src/core/sobolev/constraintSet';
import { barycenterTarget } from '../../src/core/sobolev/constraints';
import { blockIndex, unflatten } from '../../src/core/sobolev/layout';
import type { Edge, Vec3 } from '../../src/core/testConfigs';

interface Fixture {
    name: string;
    vertices: Vec3[];
    edges: Edge[];
    alpha: number;
    beta: number;
    epsilon: number;
}

// Load at runtime (avoids needing resolveJsonModule in tsconfig; test/** is
// typechecked), mirroring test/sobolev/constraintSet.test.ts.
function loadFixture(name: string): Fixture {
    return JSON.parse(
        readFileSync(new URL(`../../oracle/fixtures/${name}.json`, import.meta.url), 'utf8'),
    ) as Fixture;
}

function euclideanNorm(a: number[]): number {
    let sumSq = 0;
    for (const x of a) sumSq += x * x;
    return Math.sqrt(sumSq);
}

function euclideanDiff(a: number[], b: number[]): number {
    let sumSq = 0;
    for (let i = 0; i < a.length; i++) {
        const d = a[i] - b[i];
        sumSq += d * d;
    }
    return Math.sqrt(sumSq);
}

// Central-difference FD check of a stacked set's Jacobian on a deterministic
// direction — the same pattern as constraintSet.test.ts's stacked FD check;
// gate 1e-6 rel (see oracle/README.md "Known tolerance caveats" for why not
// tighter).
// @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §5.4
function fdJacobianRel(set: ConstraintSet, vertices: Vec3[], edges: Edge[]): number {
    const n = vertices.length;
    const eta = 1e-6;
    const h = Array.from({ length: 3 * n }, (_, k) => Math.sin(0.11 + 0.37 * k));
    const offsetsPlus = unflatten(h.map((x) => eta * x));
    const offsetsMinus = unflatten(h.map((x) => -eta * x));
    const plus: Vec3[] = vertices.map((v, i) => [
        v[0] + offsetsPlus[i][0],
        v[1] + offsetsPlus[i][1],
        v[2] + offsetsPlus[i][2],
    ]);
    const minus: Vec3[] = vertices.map((v, i) => [
        v[0] + offsetsMinus[i][0],
        v[1] + offsetsMinus[i][1],
        v[2] + offsetsMinus[i][2],
    ]);
    const { C } = evaluateConstraintSet(set, vertices, edges);
    const { phi: phiPlus } = evaluateConstraintSet(set, plus, edges);
    const { phi: phiMinus } = evaluateConstraintSet(set, minus, edges);
    const fd = phiPlus.map((p, r) => (p - phiMinus[r]) / (2 * eta));
    const Ch = C.map((row) => row.reduce((s, x, k) => s + x * h[k], 0));
    return euclideanDiff(fd, Ch) / Math.max(1, euclideanNorm(Ch));
}

// Edge-length rows on a junction fixture too (spec §5.4.1) — junction-y
// exercises shared-vertex accumulation across a degree-3 vertex.
for (const name of ['crossing', 'junction-y'] as const) {
    test(`edgeLengthsBlock: stacked [barycenter, edgeLengths] FD Jacobian on ${name} ≤ 1e-6 rel (spec §5.4.1)`, () => {
        const { vertices, edges } = loadFixture(name);
        const x0 = barycenterTarget(vertices, edges);
        const ell0 = edgeLengths(vertices, edges);
        const set: ConstraintSet = [barycenterBlock(x0), edgeLengthsBlock(ell0)];
        const { phi, C } = evaluateConstraintSet(set, vertices, edges);
        expect(phi.length).toBe(3 + edges.length);
        expect(C.length).toBe(3 + edges.length);
        const rel = fdJacobianRel(set, vertices, edges);
        console.log(
            `[constraintSetM2] ${name}: [barycenter, edgeLengths] FD-Jacobian rel = ${rel.toExponential(3)}`,
        );
        expect(rel).toBeLessThanOrEqual(1e-6);
    });
}

test('pointBlock: [barycenter, point] FD Jacobian on crossing ≤ 1e-6 rel; rows are the exact identity block (spec §2)', () => {
    const { vertices, edges } = loadFixture('crossing');
    const x0 = barycenterTarget(vertices, edges);
    // Off-vertex target → non-trivial Φ; the point Jacobian is target-independent.
    const target: Vec3 = [vertices[0][0] + 0.1, vertices[0][1] - 0.2, vertices[0][2] + 0.3];
    const set: ConstraintSet = [barycenterBlock(x0), pointBlock(0, target)];
    const rel = fdJacobianRel(set, vertices, edges);
    console.log(
        `[constraintSetM2] crossing: [barycenter, point] FD-Jacobian rel = ${rel.toExponential(3)}`,
    );
    expect(rel).toBeLessThanOrEqual(1e-6);

    // Exact structure on a different vertex: Φ = γ_i − x_i (paper sign:
    // CURRENT minus target, spec §2), C[r] has a single 1 at blockIndex(r, i, n).
    const n = vertices.length;
    const { phi, C } = pointBlock(2, target).evaluate(vertices, edges);
    expect(phi).toEqual([
        vertices[2][0] - target[0],
        vertices[2][1] - target[1],
        vertices[2][2] - target[2],
    ]);
    expect(C.length).toBe(3);
    for (let r = 0; r < 3; r++) {
        for (let k = 0; k < 3 * n; k++) {
            expect(C[r][k]).toBe(k === blockIndex(r, 2, n) ? 1 : 0);
        }
    }
});

test('edgeLengthsBlock: Φ is exactly ℓ⁰ − ℓ per edge (zero at the anchored targets)', () => {
    const { vertices, edges } = loadFixture('crossing');
    const ell0 = edgeLengths(vertices, edges);
    const atAnchor = edgeLengthsBlock(ell0).evaluate(vertices, edges);
    expect(atAnchor.phi.length).toBe(edges.length);
    for (const v of atAnchor.phi) expect(v).toBe(0);

    const shifted = edgeLengthsBlock(ell0.map((l) => l + 0.25)).evaluate(vertices, edges);
    const now = edgeLengths(vertices, edges);
    for (let r = 0; r < edges.length; r++) {
        expect(shifted.phi[r]).toBe(ell0[r] + 0.25 - now[r]);
    }
});

// The §3.4 rank-dependence rule made falsifiable: the totalLength row is the
// SUM of the edgeLengths rows — bit-identically, since per column both sides
// perform the same additions in the same (edge) order.
test('edgeLengthsBlock: rows sum EXACTLY to the totalLength row (spec §3.4 rank-dependence)', () => {
    for (const name of ['crossing', 'junction-y'] as const) {
        const { vertices, edges } = loadFixture(name);
        const ell0 = edgeLengths(vertices, edges);
        const L0 = totalLength(vertices, edges);
        const edgeEval = edgeLengthsBlock(ell0).evaluate(vertices, edges);
        const totalEval = totalLengthBlock(L0).evaluate(vertices, edges);
        const n3 = 3 * vertices.length;
        for (let k = 0; k < n3; k++) {
            let s = 0;
            for (const row of edgeEval.C) s += row[k];
            expect(s).toBe(totalEval.C[0][k]);
        }
        // Φ sums too, but association order differs → approximate gate.
        const phiSum = edgeEval.phi.reduce((a, b) => a + b, 0);
        expect(Math.abs(phiSum - totalEval.phi[0])).toBeLessThanOrEqual(1e-12);
    }
});

test('edgeLengthsBlock: zero-length edge (degenerate guard) yields a finite all-zero row, no NaN', () => {
    // Edge (0,1) is zero-length → T = [0,0,0] (1e-14 guard); edge (1,2) has
    // length 1 along +x → T = [1,0,0]. Same setup as the M1 degenerate test.
    const degenerateVertices: Vec3[] = [
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
    ];
    const degenerateEdges: Edge[] = [
        [0, 1],
        [1, 2],
    ];
    const ell0 = edgeLengths(degenerateVertices, degenerateEdges);
    const { phi, C } = edgeLengthsBlock(ell0).evaluate(degenerateVertices, degenerateEdges);
    expect(phi[0]).toBe(0);
    expect(phi[1]).toBe(0);
    const n = degenerateVertices.length;
    for (const row of C) {
        for (const v of row) expect(Number.isFinite(v)).toBe(true);
    }
    for (const v of C[0]) expect(v).toBe(0);
    expect(C[1][blockIndex(0, 1, n)]).toBe(1);
    expect(C[1][blockIndex(0, 2, n)]).toBe(-1);
});

test('rank rule (spec §3.4): totalLength + edgeLengths (REAL blocks) throws at construction; valid M2 compositions pass', () => {
    const { vertices, edges } = loadFixture('crossing');
    const x0 = barycenterTarget(vertices, edges);
    const L0 = totalLength(vertices, edges);
    const ell0 = edgeLengths(vertices, edges);
    const pinTarget: Vec3 = [vertices[0][0], vertices[0][1], vertices[0][2]];
    expect(() => assertValidConstraintSet([totalLengthBlock(L0), edgeLengthsBlock(ell0)])).toThrow(
        /§3\.4/,
    );
    expect(() =>
        assertValidConstraintSet([
            barycenterBlock(x0),
            edgeLengthsBlock(ell0),
            pointBlock(0, pinTarget),
        ]),
    ).not.toThrow();
    expect(() =>
        assertValidConstraintSet([edgeLengthsBlock(ell0), pointBlock(0, pinTarget)]),
    ).not.toThrow();
});

// Never-throw backstops: malformed inputs must surface as NaN Φ (→ the
// existing projection_failed rejection), never as a frame-loop throw.
test('edgeLengthsBlock: mismatched ℓ⁰ length yields NaN Φ rows, never throws', () => {
    const { vertices, edges } = loadFixture('crossing');
    const { phi, C } = edgeLengthsBlock([1]).evaluate(vertices, edges);
    expect(phi.length).toBe(edges.length);
    expect(Number.isNaN(phi[0])).toBe(false);
    expect(Number.isNaN(phi[1])).toBe(true);
    for (const row of C) {
        for (const v of row) expect(Number.isFinite(v)).toBe(true);
    }
});

test('pointBlock: out-of-range vertexIndex yields NaN Φ and zero C rows, never throws', () => {
    const { vertices, edges } = loadFixture('crossing');
    const { phi, C } = pointBlock(999, [0, 0, 0]).evaluate(vertices, edges);
    expect(phi.length).toBe(3);
    expect(phi.every((v) => Number.isNaN(v))).toBe(true);
    for (const row of C) {
        for (const v of row) expect(v).toBe(0);
    }
});
