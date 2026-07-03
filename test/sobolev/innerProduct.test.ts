import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { assembleBHigh } from '../../src/core/sobolev/innerProduct';
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
    B: number[][];
}

// Load at runtime (avoids needing resolveJsonModule in tsconfig; test/** is typechecked),
// mirroring test/golden.test.ts.
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

for (const name of FIXTURE_NAMES) {
    const fixture = loadFixture(name);
    const golden = loadGolden(name);
    const disjointPairs = calculateDisjointPairs(fixture.edges);
    const B = assembleBHigh(
        fixture.vertices,
        fixture.edges,
        disjointPairs,
        fixture.alpha,
        fixture.beta,
        fixture.epsilon,
    );

    // Cross-language gate vs the independent Python oracle.
    // @see oracle/README.md ("verified by diffing against an independent implementation")
    test(`innerProduct: ${name} — B matches oracle to 1e-12 relative (Frobenius)`, () => {
        const goldenNorm = frobeniusNorm(golden.B);
        const rel = frobeniusDiff(B, golden.B) / goldenNorm;
        console.log(`[innerProduct] ${name}: Frobenius-relative diff = ${rel.toExponential(3)}`);
        expect(rel).toBeLessThanOrEqual(1e-12);
    });

    // Guaranteed by the explicit final symmetrization in assembleBHigh — exact equality, not
    // just close. @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §E ("Symmetry")
    test(`innerProduct: ${name} — B is exactly symmetric`, () => {
        const n = B.length;
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                expect(Object.is(B[i][j], B[j][i])).toBe(true);
            }
        }
    });

    // Constant nullspace: B is built entirely from differences (D_I u - D_J u), so B·1 ≈ 0.
    // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §E ("Constant nullspace")
    test(`innerProduct: ${name} — B annihilates the constant vector`, () => {
        const rowSums = B.map((row) => row.reduce((s, x) => s + x, 0));
        const normB1 = Math.sqrt(rowSums.reduce((s, x) => s + x * x, 0));
        const normB = frobeniusNorm(B);
        expect(normB1).toBeLessThanOrEqual(1e-10 * normB);
    });
}
