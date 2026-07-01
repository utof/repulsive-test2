import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import {
    calculateDisjointPairs,
    calculateEnergy,
    gradientAnalytical,
} from '../src/tangentPointEnergy';
import type { Edge, Vec3 } from '../src/testConfigs';

interface GoldenCase {
    name: string;
    alpha: number;
    beta: number;
    epsilon: number;
    vertices: Vec3[];
    edges: Edge[];
    energy: number;
    gradient: number[][];
}

// Load at runtime (avoids needing resolveJsonModule in tsconfig; test/** is typechecked).
const golden = JSON.parse(
    readFileSync(new URL('./golden.json', import.meta.url), 'utf8'),
) as GoldenCase[];

// STRICT uses toBe (Object.is): valid only while results are bit-identical AND finite
// (JSON has no -0/NaN/Infinity). Flip to false once opt #3 (integer-pow) perturbs low bits.
const STRICT = true;
const ATOL = 1e-6;
const RTOL = 1e-5;

for (const c of golden) {
    test(`golden: ${c.name}`, () => {
        const vertices = c.vertices;
        const edges = c.edges;
        const disjoint = calculateDisjointPairs(edges);

        const energy = calculateEnergy(vertices, edges, disjoint, c.alpha, c.beta, c.epsilon);
        const grad = gradientAnalytical(vertices, edges, disjoint, c.alpha, c.beta, c.epsilon);

        if (STRICT) {
            expect(energy).toBe(c.energy);
        } else {
            expect(Math.abs(energy - c.energy)).toBeLessThanOrEqual(
                1e-9 * Math.max(1, Math.abs(c.energy)),
            );
        }

        expect(grad.length).toBe(c.gradient.length);

        for (let v = 0; v < grad.length; v++) {
            for (let d = 0; d < 3; d++) {
                const got = grad[v][d];
                const want = c.gradient[v][d];
                if (STRICT) {
                    expect(got).toBe(want);
                } else {
                    expect(Math.abs(got - want)).toBeLessThanOrEqual(
                        ATOL + RTOL * Math.max(Math.abs(got), Math.abs(want)),
                    );
                }
            }
        }
    });
}
