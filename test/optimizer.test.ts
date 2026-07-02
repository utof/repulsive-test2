import { expect, test } from 'bun:test';
import { step } from '../src/core/optimizer';
import {
    calculateDisjointPairs,
    calculateEnergy,
    gradientAnalytical,
} from '../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../src/core/testConfigs';

const vertices: Vec3[] = [
    [-1, 0, 0.3],
    [1, 0, 0.3],
    [0, -1, -0.3],
    [0, 1, -0.3],
];
const edges: Edge[] = [
    [0, 1],
    [2, 3],
];

test('step applies v - stepSize*grad and reports energy AT the new vertices (analytical)', () => {
    const pairs = calculateDisjointPairs(edges);
    const stepSize = 0.001;
    const grad = gradientAnalytical(vertices, edges, pairs, 3, 6, 1e-10);
    const expected: Vec3[] = vertices.map((v, i) => [
        v[0] - stepSize * grad[i][0],
        v[1] - stepSize * grad[i][1],
        v[2] - stepSize * grad[i][2],
    ]);
    const expectedEnergy = calculateEnergy(expected, edges, pairs, 3, 6, 1e-10);

    const out = step(vertices, edges, pairs, { mode: 'analytical', stepSize });

    expect(out.vertices).toEqual(expected);
    expect(Object.is(out.energy, expectedEnergy)).toBe(true);
});

test('step does not mutate its input vertices', () => {
    const pairs = calculateDisjointPairs(edges);
    step(vertices, edges, pairs, { mode: 'analytical', stepSize: 0.001 });
    expect(vertices[0]).toEqual([-1, 0, 0.3]);
});
