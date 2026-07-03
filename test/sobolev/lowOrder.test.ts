import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { assembleA, assembleBLow } from '../../src/core/sobolev/innerProduct';
import { expandBlockDiag } from '../../src/core/sobolev/layout';
import { calculateDisjointPairs } from '../../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../../src/core/testConfigs';

// All 5 oracle fixture/golden pairs (Stage-1 Sobolev oracle harness).
// @see oracle/README.md
const FIXTURE_NAMES = ['crossing', 'junction-y', 'helix', 'linked-rings', 'knot'] as const;

interface Fixture {
    name: string;
    vertices: Vec3[];
    edges: Edge[];
    alpha: number;
    beta: number;
    epsilon: number;
}

interface Golden {
    B0: number[][];
    A: number[][];
    A3: number[][];
}

// Load at runtime (avoids needing resolveJsonModule in tsconfig; test/** is typechecked),
// mirroring test/golden.test.ts and the other sobolev tests.
function loadFixture(name: string): Fixture {
    return JSON.parse(
        readFileSync(new URL(`../../oracle/fixtures/${name}.json`, import.meta.url), 'utf8'),
    ) as Fixture;
}

function loadGolden(name: string): Golden {
    return JSON.parse(
        readFileSync(new URL(`../../oracle/golden/${name}.json`, import.meta.url), 'utf8'),
    ) as Golden;
}

function frobeniusNorm(M: number[][]): number {
    let sumSq = 0;
    for (const row of M) {
        for (const x of row) sumSq += x * x;
    }
    return Math.sqrt(sumSq);
}

function frobeniusDiff(a: number[][], b: number[][]): number {
    let sumSq = 0;
    for (let i = 0; i < a.length; i++) {
        for (let j = 0; j < a[i].length; j++) {
            const d = a[i][j] - b[i][j];
            sumSq += d * d;
        }
    }
    return Math.sqrt(sumSq);
}

// Frobenius-relative diff with the max(1,.) floor for near-zero golden norms.
// @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §E (general "max(1, norm)" form)
function relDiff(actual: number[][], golden: number[][]): number {
    return frobeniusDiff(actual, golden) / Math.max(1, frobeniusNorm(golden));
}

for (const name of FIXTURE_NAMES) {
    const fixture = loadFixture(name);
    const golden = loadGolden(name);
    const disjointPairs = calculateDisjointPairs(fixture.edges);
    const n = fixture.vertices.length;

    const B0 = assembleBLow(
        fixture.vertices,
        fixture.edges,
        disjointPairs,
        fixture.alpha,
        fixture.beta,
        fixture.epsilon,
    );
    const A = assembleA(
        fixture.vertices,
        fixture.edges,
        disjointPairs,
        fixture.alpha,
        fixture.beta,
        fixture.epsilon,
    );
    const A3 = expandBlockDiag(A);

    // Cross-language gates vs the independent Python oracle.
    // @see oracle/README.md ("verified by diffing against an independent implementation")
    test(`lowOrder: ${name} — B0 matches oracle to 1e-12 relative (Frobenius)`, () => {
        const rel = relDiff(B0, golden.B0);
        console.log(`[lowOrder] ${name}: B0 Frobenius-relative diff = ${rel.toExponential(3)}`);
        expect(rel).toBeLessThanOrEqual(1e-12);
    });

    test(`lowOrder: ${name} — A matches oracle to 1e-12 relative (Frobenius)`, () => {
        const rel = relDiff(A, golden.A);
        console.log(`[lowOrder] ${name}: A Frobenius-relative diff = ${rel.toExponential(3)}`);
        expect(rel).toBeLessThanOrEqual(1e-12);
    });

    test(`lowOrder: ${name} — expandBlockDiag(A) matches oracle A3 to 1e-12 relative (Frobenius)`, () => {
        const rel = relDiff(A3, golden.A3);
        console.log(`[lowOrder] ${name}: A3 Frobenius-relative diff = ${rel.toExponential(3)}`);
        expect(rel).toBeLessThanOrEqual(1e-12);
    });

    // Guaranteed by the explicit final symmetrization in assembleBLow — exact equality, not
    // just close. @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §E ("Symmetry")
    test(`lowOrder: ${name} — B0 is exactly symmetric`, () => {
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                expect(Object.is(B0[i][j], B0[j][i])).toBe(true);
            }
        }
    });

    // Constant nullspace: A = B + B0 is built entirely from differences, so A·1 ≈ 0.
    // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §E ("Constant nullspace")
    test(`lowOrder: ${name} — A annihilates the constant vector`, () => {
        const rowSums = A.map((row) => row.reduce((s, x) => s + x, 0));
        const normA1 = Math.sqrt(rowSums.reduce((s, x) => s + x * x, 0));
        const normA = frobeniusNorm(A);
        expect(normA1).toBeLessThanOrEqual(1e-10 * Math.max(1, normA));
    });

    // Off-diagonal blocks of expandBlockDiag are exact zero; diagonal blocks are Object.is-equal
    // to A's own entries (pure data movement, no arithmetic).
    // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §A ("Ā = diag(A, A, A)")
    test(`lowOrder: ${name} — expandBlockDiag(A) off-diagonal blocks are exactly zero, diagonal blocks equal A`, () => {
        for (let bi = 0; bi < 3; bi++) {
            for (let bj = 0; bj < 3; bj++) {
                const rowOff = bi * n;
                const colOff = bj * n;
                for (let i = 0; i < n; i++) {
                    for (let j = 0; j < n; j++) {
                        const entry = A3[rowOff + i][colOff + j];
                        if (bi === bj) {
                            expect(Object.is(entry, A[i][j])).toBe(true);
                        } else {
                            expect(Object.is(entry, 0)).toBe(true);
                        }
                    }
                }
            }
        }
    });
}
