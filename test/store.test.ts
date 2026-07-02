import { expect, test } from 'bun:test';
import { DEFAULTS } from '../src/core/optimizer';
import { calculateDisjointPairs, calculateEnergy } from '../src/core/tangentPointEnergy';
import { testConfigs } from '../src/core/testConfigs';
import { buildGraphState } from '../src/store';

test('buildGraphState builds graph, disjoint pairs, and initial energy for a preset', () => {
    const crossing = testConfigs.find((t) => t.id === 'crossing')!;
    const built = buildGraphState(crossing, {});

    const expectedGraph = crossing.generate({});
    const expectedPairs = calculateDisjointPairs(expectedGraph.edges);
    const expectedEnergy = calculateEnergy(
        expectedGraph.vertices,
        expectedGraph.edges,
        expectedPairs,
        DEFAULTS.alpha,
        DEFAULTS.beta,
        DEFAULTS.epsilon,
    );

    expect(built.graph.vertices.length).toBe(expectedGraph.vertices.length);
    expect(built.graph.edges).toEqual(expectedGraph.edges);
    expect(built.disjointPairs).toEqual(expectedPairs);
    expect(Object.is(built.energy, expectedEnergy)).toBe(true);
});
