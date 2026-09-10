import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import {
    assembleA,
    assembleAFlat,
    assembleBHigh,
    assembleBHighFlat,
    assembleBLow,
    assembleBLowFlat,
} from '../../src/core/sobolev/innerProduct';
import { timingsBegin, timingsEnd } from '../../src/core/sobolev/phaseTimings';
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

// ---------------------------------------------------------------------------
// Guard, boundary and partner-list-contract coverage.
//
// The five-fixture sweep above only ever feeds assembleBHigh well-formed,
// non-degenerate geometry, so it cannot tell the intentional guards in
// innerProduct.ts apart from their absence. Each block below pins one of them
// by the source line it defends; if a line named here is "cleaned up" as
// redundant, exactly one of these fails.
// @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md
// ---------------------------------------------------------------------------

const GUARD_ALPHA = 3;
const GUARD_BETA = 6;
const GUARD_EPSILON = 1e-3;

// Two edges with no shared vertex, i.e. exactly one ordered disjoint pair each
// way — the smallest configuration in which the tangent T_I reaches the kernel.
const TWO_EDGES: Edge[] = [
    [0, 1],
    [2, 3],
];

function expectAllFinite(M: Float64Array, label: string): void {
    const bad = [...M].findIndex((x) => !Number.isFinite(x));
    expect(`${label}: first non-finite index=${bad}`).toBe(`${label}: first non-finite index=-1`);
}

function frobenius(M: Float64Array): number {
    let s = 0;
    for (const x of M) s += x * x;
    return Math.sqrt(s);
}

function frobeniusDiffFlat(a: Float64Array, b: Float64Array): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) {
        const d = a[i] - b[i];
        s += d * d;
    }
    return Math.sqrt(s);
}

// A zero-length edge (both endpoints at the same point) has no direction, so
// `1 / r` is Infinity and `ex * inv` is NaN. innerProduct.ts:71 (B) and :209
// (B⁰) intercept that with `tangent[I] = [0, 0, 0]`. Dropping the branch, or
// leaving `tangent[I]` unset, poisons every entry of the assembled matrix
// (NaN) or throws on `TI[0]`.
test('innerProduct: a zero-length edge takes the degenerate-tangent branch (finite B/B⁰, zero T·T coupling)', () => {
    const P: Vec3 = [0.25, -0.5, 0.75];
    const vertices: Vec3[] = [P, [0.25, -0.5, 0.75], [1, 1, 0], [2, 1, 0]];
    const n = vertices.length;
    const disjointPairs = calculateDisjointPairs(TWO_EDGES);
    expect(disjointPairs).toEqual([[1], [0]]); // both directions of the pair are visited

    const B = assembleBHighFlat(
        vertices,
        TWO_EDGES,
        disjointPairs,
        GUARD_ALPHA,
        GUARD_BETA,
        GUARD_EPSILON,
    );
    const B0 = assembleBLowFlat(
        vertices,
        TWO_EDGES,
        disjointPairs,
        GUARD_ALPHA,
        GUARD_BETA,
        GUARD_EPSILON,
    );
    expectAllFinite(B, 'B');
    expectAllFinite(B0, 'B0');

    // T_0 = [0,0,0] ⇒ dotT = T_0·T_1 = 0 exactly ⇒ B's whole edge-0/edge-1 cross
    // block (the `-= sign*w*dotT/(ellI*ellJ)` terms) is exactly zero.
    for (const i of [0, 1]) {
        for (const j of [2, 3]) {
            expect(B[i * n + j] === 0).toBe(true);
            expect(B[j * n + i] === 0).toBe(true);
        }
    }
    // Non-vacuous: the same-edge blocks and B⁰ still carry the pair's weight.
    expect(B[0 * n + 0]).toBeGreaterThan(0);
    expect([...B0].filter((x) => x !== 0).length).toBeGreaterThan(0);
});

// The guard is `r < 1e-14`, strictly: an edge whose length is EXACTLY the
// threshold still gets a real unit tangent. `Math.sqrt(1e-14 * 1e-14)` is
// exactly the double 1e-14, so this lands on the boundary bit-for-bit.
// @see src/core/sobolev/innerProduct.ts:71 and :209
test('innerProduct: an edge of length exactly 1e-14 is NOT degenerate (guard is strict <)', () => {
    expect(Math.sqrt(1e-14 * 1e-14)).toBe(1e-14); // premise of this test

    const boundary: Vec3[] = [
        [0, 0, 0],
        [1e-14, 0, 0],
        [1, 1, 0],
        [2, 1, 0],
    ];
    const zeroLength: Vec3[] = [
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 0],
        [2, 1, 0],
    ];
    const n = boundary.length;
    const disjointPairs = calculateDisjointPairs(TWO_EDGES);
    const args = [TWO_EDGES, disjointPairs, GUARD_ALPHA, GUARD_BETA, GUARD_EPSILON] as const;

    // B: a real tangent gives non-zero T_0·T_1 coupling; the zero tangent gives
    // exactly zero (asserted in the test above).
    const B = assembleBHighFlat(boundary, ...args);
    expectAllFinite(B, 'B');
    expect(B[0 * n + 2] === 0).toBe(false);

    // B⁰: with T_0 = 0 the cross-norm collapses to ε, which changes k²₄ by ~six
    // orders of magnitude here — so the boundary edge must NOT resemble the
    // zero-length edge. Clean ≈ 1.0; treating 1e-14 as degenerate gives ~1e-11.
    const B0boundary = assembleBLowFlat(boundary, ...args);
    const B0zero = assembleBLowFlat(zeroLength, ...args);
    const rel = frobeniusDiffFlat(B0boundary, B0zero) / frobenius(B0zero);
    expect(rel).toBeGreaterThan(0.5);
});

// `if (!Js) continue` (innerProduct.ts:81 and :219) tolerates a partner list
// with fewer entries than `edges`: a missing entry means "no partners", not a
// crash on `for (const J of undefined)`.
test('innerProduct: a disjointPairs list shorter than edges is tolerated (missing entry == no partners)', () => {
    const vertices: Vec3[] = [
        [0, 0, 0],
        [1, 0.25, 0],
        [0, 1, 0],
        [1, 1.25, 0.5],
    ];
    const full = calculateDisjointPairs(TWO_EDGES);
    const short = [full[0]]; // length 1 < edges.length === 2
    const explicitEmpty = [full[0], []];
    const rest = [GUARD_ALPHA, GUARD_BETA, GUARD_EPSILON] as const;

    expect([...assembleBHighFlat(vertices, TWO_EDGES, short, ...rest)]).toEqual([
        ...assembleBHighFlat(vertices, TWO_EDGES, explicitEmpty, ...rest),
    ]);
    expect([...assembleBLowFlat(vertices, TWO_EDGES, short, ...rest)]).toEqual([
        ...assembleBLowFlat(vertices, TWO_EDGES, explicitEmpty, ...rest),
    ]);
});

// The assembly loops are bounded by `edges.length` (innerProduct.ts:79 and
// :217), not by `disjointPairs.length`: a longer partner list is ignored
// rather than indexing `edges` out of range.
test('innerProduct: a disjointPairs list longer than edges is ignored (loop is bounded by edges.length)', () => {
    const vertices: Vec3[] = [
        [0, 0, 0],
        [1, 0.25, 0],
        [0, 1, 0],
        [1, 1.25, 0.5],
    ];
    const full = calculateDisjointPairs(TWO_EDGES);
    const long = [...full, [0]]; // one trailing entry past the last edge
    const rest = [GUARD_ALPHA, GUARD_BETA, GUARD_EPSILON] as const;

    expect([...assembleBHighFlat(vertices, TWO_EDGES, long, ...rest)]).toEqual([
        ...assembleBHighFlat(vertices, TWO_EDGES, full, ...rest),
    ]);
    expect([...assembleBLowFlat(vertices, TWO_EDGES, long, ...rest)]).toEqual([
        ...assembleBLowFlat(vertices, TWO_EDGES, full, ...rest),
    ]);
});

// An empty curve must assemble to an empty matrix instead of spinning: every
// loop bound here is `< length`, so all of them are entered zero times.
// @see src/core/sobolev/innerProduct.ts:131 and :276 (symmetrization bounds)
test('innerProduct: an empty vertex/edge set assembles to an empty matrix', () => {
    const rest = [GUARD_ALPHA, GUARD_BETA, GUARD_EPSILON] as const;
    expect(assembleBHigh([], [], [], ...rest)).toEqual([]);
    expect(assembleBLow([], [], [], ...rest)).toEqual([]);
    expect(assembleA([], [], [], ...rest)).toEqual([]);
    expect(assembleAFlat([], [], [], ...rest).length).toBe(0);
});

// assembleAFlat's two sub-phases are recorded under the ledger keys 'bHigh' and
// 'bLow' (innerProduct.ts:337 and :340). The keys are the bench ledger's
// schema — renaming one silently breaks baseline comparability in bench/results/.
// @see src/core/sobolev/phaseTimings.ts (SobolevPhaseKey — "Add keys; never rename")
test("innerProduct: assembleAFlat records its sub-phases as 'bHigh' and 'bLow'", () => {
    const fixture = loadFixture('crossing');
    const disjointPairs = calculateDisjointPairs(fixture.edges);

    timingsBegin();
    let ledger: ReturnType<typeof timingsEnd>;
    try {
        assembleAFlat(
            fixture.vertices,
            fixture.edges,
            disjointPairs,
            fixture.alpha,
            fixture.beta,
            fixture.epsilon,
        );
    } finally {
        ledger = timingsEnd();
    }

    expect(ledger).not.toBeNull();
    // Exactly these three keys — no empty-string or renamed sub-phase key.
    expect(Object.keys(ledger ?? {}).sort()).toEqual(['assembleA', 'bHigh', 'bLow']);
    expect(ledger?.bHigh?.calls).toBe(1);
    expect(ledger?.bLow?.calls).toBe(1);
    expect(ledger?.assembleA?.calls).toBe(1);
});
