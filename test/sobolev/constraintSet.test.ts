import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import {
    assertValidConstraintSet,
    barycenterBlock,
    type ConstraintBlock,
    type ConstraintSet,
    evaluateConstraintSet,
    totalLength,
    totalLengthBlock,
} from '../../src/core/sobolev/constraintSet';
import { barycenterPhiAndC, barycenterTarget } from '../../src/core/sobolev/constraints';
import { blockIndex, unflatten } from '../../src/core/sobolev/layout';
import { barycenterScale } from '../../src/core/sobolev/lineSearch';
import type { Edge, Vec3 } from '../../src/core/testConfigs';

interface Fixture {
    name: string;
    vertices: Vec3[];
    edges: Edge[];
    alpha: number;
    beta: number;
    epsilon: number;
}

// Load at runtime (avoids needing resolveJsonModule in tsconfig; test/** is typechecked),
// mirroring test/sobolev/gradient.test.ts.
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

const fixture = loadFixture('crossing');
const { vertices, edges } = fixture;

test('constraintSet: stacked [barycenter, totalLength] Jacobian matches central-difference FD to 1e-6 rel (spec §4.4.1)', () => {
    const x0 = barycenterTarget(vertices, edges);
    const L0 = totalLength(vertices, edges);
    const set: ConstraintSet = [barycenterBlock(x0), totalLengthBlock(L0)];

    const { phi, C } = evaluateConstraintSet(set, vertices, edges);
    expect(phi.length).toBe(4);
    expect(C.length).toBe(4);

    const n = vertices.length;
    const eta = 1e-6;
    // Deterministic, non-trivial perturbation direction over the flat 3n
    // coordinate-block layout — same pattern as constraints.test.ts's FD check.
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
    const { phi: phiPlus } = evaluateConstraintSet(set, plus, edges);
    const { phi: phiMinus } = evaluateConstraintSet(set, minus, edges);

    // Central differences (spec §4.4.1) — see oracle/README.md "Known tolerance
    // caveats" for why the gate is 1e-6 and not tighter.
    const fd = phiPlus.map((p, r) => (p - phiMinus[r]) / (2 * eta));
    const Ch = C.map((row) => row.reduce((s, x, k) => s + x * h[k], 0));
    const rel = euclideanDiff(fd, Ch) / Math.max(1, euclideanNorm(Ch));
    console.log(
        `[constraintSet] crossing: stacked FD-Jacobian relative diff = ${rel.toExponential(3)}`,
    );
    expect(rel).toBeLessThanOrEqual(1e-6);
});

test('constraintSet: barycenterBlock evaluate/scale is a bit-identical passthrough of barycenterPhiAndC/barycenterScale', () => {
    const x0 = barycenterTarget(vertices, edges);
    const block = barycenterBlock(x0);

    const { phi, C } = block.evaluate(vertices, edges);
    const direct = barycenterPhiAndC(vertices, edges, x0);
    expect(phi).toEqual([...direct.phi]);
    expect(C).toEqual(direct.C);
    for (let i = 0; i < phi.length; i++) {
        expect(phi[i]).toBe(direct.phi[i]);
    }

    expect(block.scale(vertices, edges)).toBe(barycenterScale(vertices, edges, x0));
});

test('constraintSet: totalLengthBlock phi is exactly L0 minus current total length', () => {
    const currentLength = totalLength(vertices, edges);

    const atZero = totalLengthBlock(currentLength).evaluate(vertices, edges);
    expect(atZero.phi[0]).toBe(0);

    const L0 = currentLength + 1;
    const atOne = totalLengthBlock(L0).evaluate(vertices, edges);
    // Safest exactness check (per task spec): compare against the identical
    // expression computed independently, rather than assuming (L+1)-L === 1.
    expect(atOne.phi[0]).toBe(L0 - totalLength(vertices, edges));
});

test('constraintSet: evaluateConstraintSet stacks block rows in array order; empty set yields empty phi/C', () => {
    const x0 = barycenterTarget(vertices, edges);
    const L0 = totalLength(vertices, edges);
    const barBlock = barycenterBlock(x0);
    const lenBlock = totalLengthBlock(L0);
    const barEval = barBlock.evaluate(vertices, edges);
    const lenEval = lenBlock.evaluate(vertices, edges);

    const stacked = evaluateConstraintSet([barBlock, lenBlock], vertices, edges);
    expect(stacked.phi.length).toBe(4);
    expect(stacked.phi.slice(0, 3)).toEqual(barEval.phi);
    expect(stacked.phi.slice(3, 4)).toEqual(lenEval.phi);
    expect(stacked.C.slice(0, 3)).toEqual(barEval.C);
    expect(stacked.C.slice(3, 4)).toEqual(lenEval.C);

    // Empty set — spec §9a: the saddle system degenerates to k = 0 rows.
    const empty = evaluateConstraintSet([], vertices, edges);
    expect(empty.phi).toEqual([]);
    expect(empty.C).toEqual([]);
});

test('constraintSet: totalLengthBlock handles a zero-length edge (degenerate guard) without NaN', () => {
    // Edge (0,1) is zero-length (both vertices coincide) -> T = [0,0,0].
    // Edge (1,2) has length 1 along +x -> T = [1,0,0].
    const degenerateVertices: Vec3[] = [
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
    ];
    const degenerateEdges: Edge[] = [
        [0, 1],
        [1, 2],
    ];
    const L0 = totalLength(degenerateVertices, degenerateEdges);
    const { phi, C } = totalLengthBlock(L0).evaluate(degenerateVertices, degenerateEdges);
    expect(phi[0]).toBe(0);

    const n = degenerateVertices.length;
    const row = C[0];
    for (const v of row) {
        expect(Number.isFinite(v)).toBe(true);
    }
    // Zero-length edge (0,1) contributes T = [0,0,0]: vertex 0's x-column is untouched.
    expect(row[blockIndex(0, 0, n)]).toBe(0);
    // Edge (1,2): +T at vertex 1, -T at vertex 2, in the x-column.
    expect(row[blockIndex(0, 1, n)]).toBe(1);
    expect(row[blockIndex(0, 2, n)]).toBe(-1);
});

test('constraintSet: assertValidConstraintSet passes for [barycenter, totalLength], throws for totalLength+edgeLengths (spec §3.4)', () => {
    const x0 = barycenterTarget(vertices, edges);
    const L0 = totalLength(vertices, edges);
    expect(() =>
        assertValidConstraintSet([barycenterBlock(x0), totalLengthBlock(L0)]),
    ).not.toThrow();

    // Minimal inline stub for the M2 edgeLengths block — only `kind` matters here.
    const edgeLengthsStub: ConstraintBlock = {
        kind: 'edgeLengths',
        evaluate: () => ({ phi: [], C: [] }),
        scale: () => 1,
    };
    expect(() => assertValidConstraintSet([totalLengthBlock(L0), edgeLengthsStub])).toThrow(
        /§3\.4/,
    );
});
