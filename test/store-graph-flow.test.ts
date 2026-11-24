// The local graph flow as the app's layout engine: lifecycle in the store, the drag through a
// pin, and the fallback when the graph is not flat. @see src/store.ts (the graphFlow field)
import { afterAll, beforeEach, describe, expect, test } from 'bun:test';
import { countPlanarCrossings } from '../src/core/subdivide';
import type { Vec3 } from '../src/core/testConfigs';
import { useSimStore } from '../src/store';

const st = () => useSimStore.getState();
const crossings = () => countPlanarCrossings(st().live, st().graph.edges, st().edgeSubdivision);
const snapshot = (): Vec3[] => st().live.map((p) => [p[0], p[1], p[2]] as Vec3);

describe('graph flow (store)', () => {
    beforeEach(() => {
        st().setRunning(false);
        st().setGraphFlow(false);
        st().setEdgeSubdivision(1);
        st().setPreset('flat-grid');
    });
    // The store is one module-level object shared by every test file in a run: leave it as found.
    afterAll(() => {
        st().setRunning(false);
        st().setGraphFlow(false);
        st().setEdgeSubdivision(1);
        st().setPreset('random');
    });

    test('the flat-grid preset is flat and has no crossings', () => {
        expect(st().live.every((p) => p[2] === 0)).toBe(true);
        expect(st().live.length).toBe(64);
        expect(crossings()).toBe(0);
    });

    test('off by default; on, the first tick builds the run from the live geometry', () => {
        expect(st().graphFlow).toBe(false);
        st().setEdgeSubdivision(2);
        st().setGraphFlow(true);
        st().setRunning(true);
        expect(st().graphFlowRun).toBeNull(); // lazily, by the first tick
        st().tickGraphFlow(1, Number.POSITIVE_INFINITY);
        const run = st().graphFlowRun;
        expect(run).not.toBeNull();
        expect(run?.g.n).toBe(st().live.length);
        expect(run?.g.pieces).toBe(2);
        expect(st().graphFlowInfo?.phase).toBe('running');
        expect(st().step).toBe(1);
        st().setRunning(false);
        expect(st().graphFlowRun).toBeNull(); // nothing carried across a pause
    });

    test('dragging a pinned corner node into the grid: it follows, nothing crosses, the rest makes room', () => {
        st().setEdgeSubdivision(2);
        st().setGraphFlow(true);
        const before = snapshot();
        // Pin the bottom-left corner (node 0) and ask for it at the grid's centre.
        st().addPin(0);
        st().setRunning(true);
        // The drag happens DURING the run (play re-anchors every pin to where its node is).
        st().setPinTarget(0, [0, 0, 0]);
        const start: Vec3 = [before[0][0], before[0][1], 0];
        const distToTarget = () => Math.hypot(st().live[0][0], st().live[0][1]);
        const d0 = distToTarget();
        let worst = 0;
        for (let k = 0; k < 150; k++) {
            st().tickGraphFlow(1, Number.POSITIVE_INFINITY);
            worst = Math.max(worst, crossings());
        }
        expect(worst).toBe(0); // never, at any step
        expect(distToTarget()).toBeLessThan(0.6 * d0); // it went a good part of the way
        // The rest made room: nodes other than the dragged one moved too.
        let moved = 0;
        for (let i = 1; i < 64; i++)
            if (Math.hypot(st().live[i][0] - before[i][0], st().live[i][1] - before[i][1]) > 0.02)
                moved++;
        expect(moved).toBeGreaterThan(5);
        expect(start[0]).not.toBe(st().live[0][0]);
        st().setRunning(false);
    });

    test('a pinned node without a new target does not move', () => {
        st().setEdgeSubdivision(2);
        st().setGraphFlow(true);
        st().addPin(27);
        const p = [...st().live[27]];
        st().setRunning(true);
        for (let k = 0; k < 20; k++) st().tickGraphFlow(1, Number.POSITIVE_INFINITY);
        expect(st().live[27][0]).toBeCloseTo(p[0], 12);
        expect(st().live[27][1]).toBeCloseTo(p[1], 12);
        st().setRunning(false);
    });

    test('a graph that is not flat: skipped with the reason, and the ordinary descent is left to run', () => {
        st().setPreset('knot');
        st().setGraphFlow(true);
        st().setRunning(true);
        st().tickGraphFlow();
        expect(st().graphFlowRun).toBeNull();
        expect(st().graphFlowSkipped).toBe(true);
        expect(st().graphFlowInfo).toEqual({ phase: 'skipped', reason: 'not a flat graph' });
        st().setRunning(false);
        expect(st().graphFlowSkipped).toBe(false); // re-evaluated by the next run
    });
});
