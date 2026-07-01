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

// Bit-identical gate: the golden fixture holds energy/gradient from the PRE-optimization code;
// the scalar-inlining preserved exact IEEE-754 op order, so results must match to the bit
// (Object.is). @see docs/superpowers/specs/2026-07-01-tangent-point-hotpath-optimization-design.md — "Verification (the safety net)"
for (const c of golden) {
    test(`golden: ${c.name}`, () => {
        const vertices = c.vertices;
        const edges = c.edges;
        const disjoint = calculateDisjointPairs(edges);

        const energy = calculateEnergy(vertices, edges, disjoint, c.alpha, c.beta, c.epsilon);
        const grad = gradientAnalytical(vertices, edges, disjoint, c.alpha, c.beta, c.epsilon);

        expect(energy).toBe(c.energy);
        expect(grad.length).toBe(c.gradient.length);

        for (let v = 0; v < grad.length; v++) {
            for (let d = 0; d < 3; d++) {
                const got = grad[v][d];
                const want = c.gradient[v][d];
                expect(got).toBe(want);
            }
        }
    });
}
