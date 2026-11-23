// Edge subdivision and planar crossing counting. @see src/core/subdivide.ts
import { describe, expect, test } from 'bun:test';
import { countPlanarCrossings, subdivideGraph } from '../../src/core/subdivide';
import type { Edge, Vec3 } from '../../src/core/testConfigs';

const V: Vec3[] = [
    [0, 0, 0],
    [4, 0, 0],
    [4, 4, 0],
    [0, 4, 0],
];
const SQUARE: Edge[] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
];

describe('subdivideGraph', () => {
    test('pieces = 1 returns the graph itself', () => {
        const g = subdivideGraph(V, SQUARE, 1);
        expect(g.vertices).toBe(V);
        expect(g.edges).toBe(SQUARE);
        expect(g.nodeCount).toBe(4);
    });

    test('nodes keep their indices; each edge becomes a chain of equal segments', () => {
        const g = subdivideGraph(V, SQUARE, 4);
        expect(g.nodeCount).toBe(4);
        expect(g.vertices.length).toBe(4 + 4 * 3);
        expect(g.edges.length).toBe(16);
        expect(g.vertices.slice(0, 4)).toEqual(V);
        // Edge 0 = (0,1): its 4 segments run from node 0 to node 1 through new points 4, 5, 6.
        expect(g.edges.slice(0, 4)).toEqual([
            [0, 4],
            [4, 5],
            [5, 6],
            [6, 1],
        ]);
        expect(g.vertices[4]).toEqual([1, 0, 0]);
        expect(g.vertices[6]).toEqual([3, 0, 0]);
        // The input is not mutated, and the output does not alias it.
        g.vertices[0][0] = 99;
        expect(V[0][0]).toBe(0);
    });

    test('total length is unchanged', () => {
        const len = (W: Vec3[], E: Edge[]) =>
            E.reduce((s, [a, b]) => s + Math.hypot(W[b][0] - W[a][0], W[b][1] - W[a][1]), 0);
        const g = subdivideGraph(V, SQUARE, 7);
        expect(len(g.vertices, g.edges)).toBeCloseTo(len(V, SQUARE), 12);
    });
});

describe('countPlanarCrossings', () => {
    const X: Edge[] = [
        [0, 2],
        [1, 3],
    ];
    test('two diagonals cross once; the square’s sides never do', () => {
        expect(countPlanarCrossings(V, X)).toBe(1);
        expect(countPlanarCrossings(V, SQUARE)).toBe(0);
        // Sides and diagonals share endpoints: touching at a shared vertex is not a crossing.
        expect(countPlanarCrossings(V, [...SQUARE, ...X])).toBe(1);
    });

    test('the count is the same after subdivision, given the pieces', () => {
        // Two segments that cross away from any k-th point of either (the square's diagonals
        // would meet exactly AT their midpoints, which is a touching, not a proper crossing).
        const W: Vec3[] = [
            [0, 0, 0],
            [4, 3, 0],
            [0, 2, 0],
            [4, 0.5, 0],
        ];
        const pair: Edge[] = [
            [0, 1],
            [2, 3],
        ];
        expect(countPlanarCrossings(W, pair)).toBe(1);
        for (const k of [2, 3, 8]) {
            const g = subdivideGraph(W, pair, k);
            expect(countPlanarCrossings(g.vertices, g.edges, k)).toBe(1);
        }
    });

    test('with pieces, an edge folding over itself is not counted', () => {
        // One edge as a self-crossing chain of 3 segments (a bow): 0 crossings between EDGES.
        const W: Vec3[] = [
            [0, 0, 0],
            [0, 1, 0],
            [2, 2, 0],
            [2, -1, 0],
        ];
        const bow: Edge[] = [
            [0, 2],
            [2, 3],
            [3, 1],
        ];
        expect(countPlanarCrossings(W, bow, 1)).toBe(1);
        expect(countPlanarCrossings(W, bow, 3)).toBe(0);
    });
});
