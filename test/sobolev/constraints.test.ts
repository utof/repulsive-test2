import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { barycenterPhiAndC, barycenterTarget } from '../../src/core/sobolev/constraints';
import { unflatten } from '../../src/core/sobolev/layout';
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
    x0_barycenter_target: number[];
    Phi_barycenter: number[];
    C_barycenter: number[][];
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

function euclideanDiff(a: number[], b: number[]): number {
    let sumSq = 0;
    for (let i = 0; i < a.length; i++) {
        const d = a[i] - b[i];
        sumSq += d * d;
    }
    return Math.sqrt(sumSq);
}

for (const name of FIXTURE_NAMES) {
    const fixture = loadFixture(name);
    const golden = loadGolden(name);
    const x0 = barycenterTarget(fixture.vertices, fixture.edges);
    const { phi, C } = barycenterPhiAndC(fixture.vertices, fixture.edges, x0);

    // Cross-language gate vs the independent Python oracle.
    // @see oracle/README.md ("verified by diffing against an independent implementation")
    test(`constraints: ${name} — x0 matches oracle to 1e-12 (max abs component)`, () => {
        const diffs = x0.map((v, k) => Math.abs(v - golden.x0_barycenter_target[k]));
        const maxDiff = Math.max(...diffs);
        console.log(`[constraints] ${name}: x0 max-abs diff = ${maxDiff.toExponential(3)}`);
        expect(maxDiff).toBeLessThanOrEqual(1e-12);
    });

    // Golden Phi values are ~1e-16 (initialization satisfies the constraint by
    // construction of x0), so this is an absolute-norm gate.
    // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B
    test(`constraints: ${name} — phi at x0 matches oracle to 1e-12 (norm)`, () => {
        const diff = euclideanDiff(phi, golden.Phi_barycenter);
        console.log(`[constraints] ${name}: phi norm diff = ${diff.toExponential(3)}`);
        expect(diff).toBeLessThanOrEqual(1e-12);
    });

    test(`constraints: ${name} — C matches oracle to 1e-12 (Frobenius, rel to max(1, ||golden||))`, () => {
        const rel =
            frobeniusDiff(C, golden.C_barycenter) / Math.max(1, frobeniusNorm(golden.C_barycenter));
        console.log(`[constraints] ${name}: C Frobenius-relative diff = ${rel.toExponential(3)}`);
        expect(rel).toBeLessThanOrEqual(1e-12);
    });

    // Finite-difference Jacobian property test — independent of the oracle: checks
    // that C really is dΦ, dℓ terms included (the full Jacobian, no frozen-length
    // approximation). @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B
    // ("Use the full Jacobian, including length dependence"), §E (FD Jacobian check)
    test(`constraints: ${name} — C matches finite-difference dΦ to 1e-5`, () => {
        const n = fixture.vertices.length;
        const eta = 1e-6;
        // Deterministic direction over the flat 3n coordinate-block layout.
        const h = Array.from({ length: 3 * n }, (_, k) => Math.sin(0.11 + 0.37 * k));
        const offsets = unflatten(h.map((x) => eta * x));
        const perturbed: Vec3[] = fixture.vertices.map((v, i) => [
            v[0] + offsets[i][0],
            v[1] + offsets[i][1],
            v[2] + offsets[i][2],
        ]);
        const { phi: phiPerturbed } = barycenterPhiAndC(perturbed, fixture.edges, x0);
        const fd = phi.map((p, r) => (phiPerturbed[r] - p) / eta);
        const Ch = C.map((row) => row.reduce((s, x, k) => s + x * h[k], 0));
        const fdNorm = Math.sqrt(fd.reduce((s, x) => s + x * x, 0));
        const rel = euclideanDiff(fd, Ch) / Math.max(1, fdNorm);
        console.log(`[constraints] ${name}: FD-Jacobian relative diff = ${rel.toExponential(3)}`);
        expect(rel).toBeLessThanOrEqual(1e-5);
    });

    // Orientation invariance: flipping an edge's stored order negates T AND swaps the
    // endpoint roles, so Φ and C must be unchanged (audited algebra, Item 8).
    // @see local_files/2026-07-02-sobolev-formula-audit.md (Item 8 — "edge-orientation-invariant")
    test(`constraints: ${name} — phi and C invariant under flipping edge 0`, () => {
        const flippedEdges: Edge[] = fixture.edges.map((e, i) => (i === 0 ? [e[1], e[0]] : e));
        const { phi: phiFlipped, C: CFlipped } = barycenterPhiAndC(
            fixture.vertices,
            flippedEdges,
            x0,
        );
        const phiDiff = euclideanDiff(phi, phiFlipped);
        const cDiff = frobeniusDiff(C, CFlipped);
        console.log(
            `[constraints] ${name}: orientation-flip phi diff = ${phiDiff.toExponential(3)}, C diff = ${cDiff.toExponential(3)}`,
        );
        expect(phiDiff).toBeLessThanOrEqual(1e-12);
        expect(cDiff).toBeLessThanOrEqual(1e-12);
    });
}
