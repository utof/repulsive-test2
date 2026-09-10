import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { barycenterPhiAndC, barycenterTarget } from '../../src/core/sobolev/constraints';
import { blockIndex, unflatten } from '../../src/core/sobolev/layout';
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

// ---------------------------------------------------------------------------
// Degenerate-geometry guards.
//
// Both guards in src/core/sobolev/constraints.ts (the `total < 1e-14` fallback
// at :38 and the `ell < 1e-14` zero-tangent branch at :93) are unreached by the
// five oracle fixtures above, which are all non-degenerate curves. The cases
// below drive them directly, and every expected value is cross-checked against
// the Python oracle so these stay cross-language gates rather than
// change-detectors.
// @see oracle/tpe_stage1_oracle.py (length_weighted_barycenter / safe_unit / DEGENERATE_TOL)
// @see oracle/README.md ("verified by diffing against an independent implementation")
// ---------------------------------------------------------------------------

/**
 * The degenerate tolerance shared by both guards, mirroring the oracle's
 * `DEGENERATE_TOL` (oracle/tpe_stage1_oracle.py:33).
 */
const DEGENERATE_TOL = 1e-14;

function maxAbsDiff(a: number[], b: number[]): number {
    let m = 0;
    for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
    return m;
}

// A graph whose every edge has collapsed to a point: `total` is exactly 0, so
// the weighted formula would be 0/0. The fallback is the PLAIN VERTEX MEAN over
// all vertices (not the midpoint mean), matching `np.mean(vertices, axis=0)`.
// Vertices are chosen so all three components of the mean are non-zero and
// mutually distinct, and so mean != sum: that separates the vertex mean from
// dropping the loop body, from negating an accumulation, and from summing
// v[k]*n instead of v[k]/n.
// @see oracle/tpe_stage1_oracle.py (length_weighted_barycenter, `if total < DEGENERATE_TOL`)
// @see src/core/sobolev/constraints.ts:38
test('constraints: fully collapsed graph — barycenterTarget returns the plain vertex mean', () => {
    const vertices: Vec3[] = [
        [1, -2, 3],
        [1, -2, 3],
        [5, 7, -11],
        [5, 7, -11],
    ];
    const edges: Edge[] = [
        [0, 1],
        [2, 3],
    ];
    const x0 = barycenterTarget(vertices, edges);
    // Oracle: length_weighted_barycenter -> array([3., 2.5, -4.]).
    const expected = [3, 2.5, -4];
    console.log(`[constraints] collapsed-graph x0 = [${x0.join(', ')}]`);
    for (const v of x0) expect(Number.isFinite(v)).toBe(true);
    expect(maxAbsDiff(x0, expected)).toBeLessThanOrEqual(1e-15);
});

// Boundary case for the guard's strictness. `total` is EXACTLY the double
// 1e-14 (Math.sqrt(1e-14 * 1e-14) === 1e-14 in IEEE-754 double, asserted below),
// so `total < 1e-14` is false and the length-weighted branch must run. Relaxing
// the guard to `<=` would take the vertex-mean fallback here and silently
// diverge from the oracle, which uses a strict `<` (tpe_stage1_oracle.py:261).
// The chain is v0 -- v1 == v2: one edge of length exactly 1e-14 plus one
// collapsed edge, so the vertex mean (2/3 * 1e-14) differs from the
// length-weighted barycenter (the midpoint of the single live edge, 5e-15).
// @see src/core/sobolev/constraints.ts:38
test('constraints: total exactly at DEGENERATE_TOL — strict < keeps the length-weighted branch', () => {
    // Precondition: the construction really lands on the boundary. If a future
    // engine's Math.sqrt is not correctly rounded this fails loudly instead of
    // turning the test into a no-op.
    expect(Math.sqrt(DEGENERATE_TOL * DEGENERATE_TOL)).toBe(DEGENERATE_TOL);

    const vertices: Vec3[] = [
        [0, 0, 0],
        [DEGENERATE_TOL, 0, 0],
        [DEGENERATE_TOL, 0, 0],
    ];
    const edges: Edge[] = [
        [0, 1],
        [1, 2],
    ];
    const x0 = barycenterTarget(vertices, edges);
    // Oracle: array([5.e-15, 0., 0.]). The `<=` variant would return the vertex
    // mean, [6.666...e-15, 0, 0].
    const vertexMean = (2 * DEGENERATE_TOL) / 3;
    console.log(
        `[constraints] boundary x0[0] = ${x0[0].toExponential(3)} (vertex mean would be ${vertexMean.toExponential(3)})`,
    );
    expect(maxAbsDiff(x0, [5e-15, 0, 0])).toBeLessThanOrEqual(1e-30);
});

// The zero-tangent guard, exercised end-to-end: a collapsed edge has ell = 0 and
// T = 0, so BOTH of its Jacobian terms (-T[c]*rvec[r] and 0.5*ell*delta) vanish
// and it must contribute exactly nothing to phi or C. Appending it to an
// otherwise identical problem is therefore a no-op. Without the guard, 1/ell is
// Infinity and T becomes [NaN, NaN, NaN], poisoning the columns of both
// endpoints; the vertex list is shared between the two calls so the two C
// matrices are directly comparable.
// @see oracle/tpe_stage1_oracle.py (safe_unit, `if n < DEGENERATE_TOL`)
// @see src/core/sobolev/constraints.ts:93
test('constraints: a collapsed edge contributes nothing to phi or C (T = 0)', () => {
    const vertices: Vec3[] = [
        [0, 0, 0],
        [1, 0.5, 0],
        [2, -0.25, 1],
        [2, -0.25, 1], // duplicate of vertex 2 — edge [2,3] is collapsed
    ];
    const live: Edge[] = [
        [0, 1],
        [1, 2],
    ];
    const withCollapsed: Edge[] = [
        [0, 1],
        [1, 2],
        [2, 3],
    ];
    const x0: Vec3 = [0.3, -0.7, 0.2];

    const a = barycenterPhiAndC(vertices, live, x0);
    const b = barycenterPhiAndC(vertices, withCollapsed, x0);

    for (const row of b.C) for (const v of row) expect(Number.isFinite(v)).toBe(true);
    for (const v of b.phi) expect(Number.isFinite(v)).toBe(true);

    const cDiff = frobeniusDiff(a.C, b.C);
    const phiDiff = euclideanDiff(a.phi, b.phi);
    console.log(
        `[constraints] collapsed-edge phi diff = ${phiDiff.toExponential(3)}, C diff = ${cDiff.toExponential(3)}`,
    );
    // Oracle: max|C1 - C2| = 0.0, phi1 - phi2 = [0, 0, 0] — an exact no-op, not
    // merely a small one.
    expect(cDiff).toBe(0);
    expect(phiDiff).toBe(0);
});

// Boundary case for the tangent guard's strictness, with an amplifier: the edge
// length is exactly 1e-14 but x0 sits a full unit away, so rvec ~ [-1, 0, 0] and
// the -T[c]*rvec[r] term is O(1) while the 0.5*ell*delta term is O(1e-14).
// Strict `<` keeps the real unit tangent T = [1, 0, 0] and gives C[0][x-col of
// v0] = 1; a `<=` guard would zero T and leave 5e-15 there. The oracle's
// safe_unit is likewise strict (tpe_stage1_oracle.py:70).
// @see src/core/sobolev/constraints.ts:93
test('constraints: edge length exactly at DEGENERATE_TOL — strict < keeps the unit tangent', () => {
    expect(Math.sqrt(DEGENERATE_TOL * DEGENERATE_TOL)).toBe(DEGENERATE_TOL);

    const vertices: Vec3[] = [
        [0, 0, 0],
        [DEGENERATE_TOL, 0, 0],
    ];
    const edges: Edge[] = [[0, 1]];
    const x0: Vec3 = [1, 0, 0];
    const { C } = barycenterPhiAndC(vertices, edges, x0);

    const n = vertices.length;
    const c00 = C[0][blockIndex(0, 0, n)];
    const c01 = C[0][blockIndex(0, 1, n)];
    console.log(`[constraints] boundary tangent: C[0][x,v0] = ${c00}, C[0][x,v1] = ${c01}`);
    // Oracle: C[0,0] = 1.0, C[0,1] = -0.99999999999999. With T zeroed both would
    // collapse to +5e-15.
    expect(Math.abs(c00 - 1)).toBeLessThanOrEqual(1e-14);
    expect(Math.abs(c01 - -0.99999999999999)).toBeLessThanOrEqual(1e-14);
});
