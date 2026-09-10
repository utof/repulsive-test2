import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { blockIndex, flatten, unflatten } from '../../src/core/sobolev/layout';
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
    dE: Vec3[];
    dE_flat: number[];
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

for (const name of FIXTURE_NAMES) {
    const fixture = loadFixture(name);
    const golden = loadGolden(name);

    // Pure data movement — must be bit-exact (Object.is), same gate as test/golden.test.ts.
    test(`layout: ${name} — flatten(dE) matches oracle dE_flat exactly`, () => {
        const flat = flatten(golden.dE);
        expect(flat.length).toBe(golden.dE_flat.length);
        for (let i = 0; i < flat.length; i++) {
            expect(Object.is(flat[i], golden.dE_flat[i])).toBe(true);
        }
    });

    test(`layout: ${name} — unflatten(flatten(vertices)) round-trips exactly`, () => {
        const roundTripped = unflatten(flatten(fixture.vertices));
        expect(roundTripped.length).toBe(fixture.vertices.length);
        for (let i = 0; i < fixture.vertices.length; i++) {
            for (let d = 0; d < 3; d++) {
                expect(Object.is(roundTripped[i][d], fixture.vertices[i][d])).toBe(true);
            }
        }
    });

    test(`layout: ${name} — blockIndex spot checks`, () => {
        const n = fixture.vertices.length;
        expect(blockIndex(0, 0, n)).toBe(0);
        expect(blockIndex(0, n - 1, n)).toBe(n - 1);
        expect(blockIndex(1, 0, n)).toBe(n);
        expect(blockIndex(1, n - 1, n)).toBe(2 * n - 1);
        expect(blockIndex(2, 0, n)).toBe(2 * n);
        expect(blockIndex(2, n - 1, n)).toBe(3 * n - 1);
    });
}

// --- Contract edges the oracle-fixture loop above does not reach ------------------------
// The loop only exercises n = |V| of the 5 oracle meshes, and only through round-trips, so
// three things stay unpinned: the small-n index convention, n = 0, and the length guard in
// `unflatten`. All three are load-bearing — every consumer (inner-product assembly, saddle
// solve, barycenter Jacobian) must agree on this one flat layout or indices silently
// misalign, which is exactly the failure the module header warns about.
// @see src/core/sobolev/layout.ts (module header)
// @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §A ("Use coordinate-block flattening")

test('layout: flatten/unflatten are coordinate-block ordered (hand-checked n=2)', () => {
    const vecs: Vec3[] = [
        [1, 2, 3],
        [4, 5, 6],
    ];
    // [x0, x1, y0, y1, z0, z1] — three contiguous blocks, NOT per-vertex interleaved.
    expect(flatten(vecs)).toEqual([1, 4, 2, 5, 3, 6]);
    expect(unflatten([1, 4, 2, 5, 3, 6])).toEqual(vecs);
});

test('layout: flatten/unflatten handle the empty mesh (n = 0)', () => {
    expect(flatten([])).toEqual([]);
    expect(unflatten([])).toEqual([]);
});

// The `flat.length % 3 !== 0` guard must fire BEFORE `const n = flat.length / 3` reaches
// `new Array(n)`: with a fractional n that allocation throws a bare RangeError('Array length
// must be a positive integer of safe magnitude'), which blames the allocator instead of the
// caller's mis-sized vector. So the assertion pins the concrete Error and its message, not
// merely "something threw" — a bare `.toThrow()` passes for either error.
// @see src/core/sobolev/layout.ts (unflatten length guard)
test('layout: unflatten rejects a flat length that is not a multiple of 3', () => {
    for (const bad of [[0], [0, 0], [0, 0, 0, 0], [0, 0, 0, 0, 0]]) {
        let caught: unknown;
        try {
            unflatten(bad);
        } catch (e) {
            caught = e;
        }
        expect(caught).toBeInstanceOf(Error);
        expect(caught).not.toBeInstanceOf(RangeError);
        expect((caught as Error).message).toBe('flat vector length must be divisible by 3');
    }
});

test('layout: unflatten accepts every multiple of 3 (the guard does not over-fire)', () => {
    for (const n of [0, 1, 2, 5, 17]) {
        const flat = Array.from({ length: 3 * n }, (_, i) => i);
        expect(unflatten(flat).length).toBe(n);
    }
});
