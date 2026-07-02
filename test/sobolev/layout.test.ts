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
