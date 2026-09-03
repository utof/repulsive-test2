// test/gpu/tangentPoint.test.ts — constructor-domain (no renderer access before the check)
import { expect, test } from 'bun:test';
import { GpuTangentPoint, REDUCE_MAX_EDGES } from '../../src/gpu/tangentPoint';
import { packTopology } from '../../src/gpu/topology';

test('GpuTangentPoint rejects edge counts beyond the two-pass reduction domain before touching the renderer', () => {
    const E = REDUCE_MAX_EDGES + 1;
    const edges = Array.from({ length: E }, (_, i) => [i, (i + 1) % E] as [number, number]);
    const dp = edges.map(() => []);
    const renderer = new Proxy(
        {},
        {
            get() {
                throw new Error('renderer touched');
            },
        },
    );
    expect(() => new GpuTangentPoint(renderer as never, packTopology(E, edges, dp))).toThrow(
        'GpuTangentPoint: edge count exceeds two-pass reduction limit',
    );
});

// Zero-length storage buffers are rejected up front: Dawn refuses a 0-byte binding at bind-group
// creation and the dispatch is silently dropped (all-zero finite readback) — task-3b review B #2,
// verified on hardware in .superpowers/sdd/2026-08-29-webgpu-solver-phase1/reports/task3b-fix-r1.md.
const throwingRenderer = () =>
    new Proxy(
        {},
        {
            get() {
                throw new Error('renderer touched');
            },
        },
    );

test('GpuTangentPoint rejects a topology with no edges before touching the renderer', () => {
    expect(() => new GpuTangentPoint(throwingRenderer() as never, packTopology(3, [], []))).toThrow(
        'GpuTangentPoint: topology has no edges',
    );
});

test('GpuTangentPoint rejects a topology with no disjoint pairs before touching the renderer', () => {
    // A single edge and a 2-edge path: every edge shares a vertex with every other ⇒ pairIndices is empty.
    for (const edges of [
        [[0, 1]],
        [
            [0, 1],
            [1, 2],
        ],
    ] as [number, number][][]) {
        const dp = edges.map(() => []);
        expect(
            () => new GpuTangentPoint(throwingRenderer() as never, packTopology(3, edges, dp)),
        ).toThrow('GpuTangentPoint: topology has no disjoint pairs');
    }
});
