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
        st().setGraphFlowMoveThrough(false);
        st().setGraphFlowTurn(true);
        st().setGraphFlowHang(false);
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

    test('move-through: the carried group turns to face its links unless told not to, the dragged node under the pointer', () => {
        const carry = (turn: boolean): { tilt: number; off: number } => {
            st().setRunning(false);
            st().setPreset('flat-grid');
            st().setEdgeSubdivision(2);
            st().setGraphFlow(true);
            st().setGraphFlowMoveThrough(true);
            st().setGraphFlowTurn(turn);
            for (const i of [0, 1, 8, 9]) st().addPin(i); // the corner square
            st().setRunning(true);
            const base = [...st().live[0]];
            const side = [st().live[1][0] - base[0], st().live[1][1] - base[1]];
            st().beginPinDrag(0);
            st().setPinTarget(0, [base[0] + 3, base[1] + 1, 0]);
            for (let k = 0; k < 3; k++) st().tickGraphFlow(1, Number.POSITIVE_INFINITY);
            const now = [st().live[1][0] - st().live[0][0], st().live[1][1] - st().live[0][1]];
            const out = {
                tilt: Math.atan2(
                    side[0] * now[1] - side[1] * now[0],
                    side[0] * now[0] + side[1] * now[1],
                ),
                off: Math.hypot(st().live[0][0] - base[0] - 3, st().live[0][1] - base[1] - 1),
            };
            st().endPinDrag();
            return out;
        };
        const kept = carry(false);
        expect(kept.tilt).toBe(0);
        expect(kept.off).toBeLessThan(1e-12);
        const turned = carry(true);
        expect(Math.abs(turned.tilt)).toBeGreaterThan(0.05);
        expect(turned.off).toBeLessThan(1e-9);
    });

    test("hang: a branch released next to another node is taken off its parent and joined there; the app's edges follow", () => {
        st().setPreset('flat-tree');
        st().setEdgeSubdivision(2);
        st().setGraphFlow(true);
        st().setGraphFlowMoveThrough(true);
        st().setGraphFlowHang(true);
        const P = 2;
        const N = st().nodeCount;
        // The tree as the app holds it: original edge e = segments e·P … e·P+P−1.
        const ends = (): [number, number][] => {
            const E = st().graph.edges;
            const out: [number, number][] = [];
            for (let e = 0; e < E.length / P; e++) out.push([E[e * P][0], E[e * P + P - 1][1]]);
            return out;
        };
        const before = ends();
        const kids = new Map<number, number[]>();
        for (const [a, b] of before) kids.set(a, [...(kids.get(a) ?? []), b]);
        const sub = (i: number): number[] => [i, ...(kids.get(i) ?? []).flatMap(sub)];
        const main = kids.get(0) as number[];
        const branch = sub(main[1]);
        const root = main[1];
        expect(branch.length).toBeGreaterThan(5);
        for (const i of branch) st().addPin(i);
        st().setRunning(true);
        // Dragged by its root to just beyond a node of the opposite main branch.
        const there = (kids.get(main[4]) as number[])[0];
        const to = st().live[there];
        st().beginPinDrag(root);
        st().setPinTarget(root, [1.25 * to[0], 1.25 * to[1], 0]);
        for (let k = 0; k < 4; k++) st().tickGraphFlow(1, Number.POSITIVE_INFINITY);
        expect(st().graphFlowRun?.move).not.toBeNull();
        st().endPinDrag();
        let ticks = 0;
        do {
            st().tickGraphFlow(1, Number.POSITIVE_INFINITY);
            ticks++;
        } while (st().graphFlowRun?.growth && ticks < 600);
        expect(st().graphFlowRun?.growth).toBeNull();
        // Still a tree on the same nodes; the branch's root has a new parent outside the branch.
        const after = ends();
        expect(after.length).toBe(before.length);
        const link = after.filter(([a, b]) => branch.includes(a) !== branch.includes(b));
        expect(link.length).toBe(1);
        const parentNow = branch.includes(link[0][0]) ? link[0][1] : link[0][0];
        expect(link[0]).toContain(root);
        expect(parentNow).not.toBe(0);
        expect(parentNow).toBeLessThan(N);
        // The app's edges are the flow's segments, and the drawing has no crossing.
        const g = st().graphFlowRun?.g;
        expect(
            st().graph.edges.every(([a, b], k) => a === g?.segs.a[k] && b === g?.segs.b[k]),
        ).toBe(true);
        expect(crossings()).toBe(0);
        expect(st().live.every((p) => p.every(Number.isFinite))).toBe(true);
        st().setRunning(false);
        st().setGraphFlowMoveThrough(false);
    });

    test('move-through: the pinned group is carried over the grid, set down, and crossings never grow', () => {
        st().setEdgeSubdivision(2);
        st().setGraphFlow(true);
        st().setGraphFlowMoveThrough(true);
        // Without turning: this test is about WHERE the group is put (to the last digits);
        // turning, which moves it about its centroid, has its own test below.
        st().setGraphFlowTurn(false);
        const blob = [0, 1, 8, 9]; // the corner square
        for (const i of blob) st().addPin(i);
        st().setRunning(true);
        const base = [...st().live[0]];
        const side = Math.hypot(st().live[1][0] - base[0], st().live[1][1] - base[1]);
        // A click alone carries nothing.
        st().beginPinDrag(0);
        st().tickGraphFlow(1, Number.POSITIVE_INFINITY);
        expect(st().graphFlowRun?.move).toBeNull();
        // Drag node 0 to the middle of the grid: the whole square goes with it, rigidly, over
        // everything in between.
        st().setPinTarget(0, [base[0] + 2, base[1] + 2, 0]);
        for (let k = 0; k < 5; k++) st().tickGraphFlow(1, Number.POSITIVE_INFINITY);
        const info = st().graphFlowInfo;
        expect(info?.phase === 'running' && info.carrying).toBe(true);
        expect(st().live[0][0]).toBeCloseTo(base[0] + 2, 12);
        expect(
            Math.hypot(st().live[1][0] - st().live[0][0], st().live[1][1] - st().live[0][1]),
        ).toBeCloseTo(side, 12);
        // Release: set down smaller than it is (it lands among the grid's edges), links not yet.
        st().endPinDrag();
        expect(st().pinDrag).toBeNull();
        expect(st().graphFlowRun?.move).toBeNull();
        expect(st().graphFlowRun?.growth).not.toBeNull();
        const info2 = () => {
            const i = st().graphFlowInfo;
            if (i?.phase !== 'running') throw new Error('graph flow is not running');
            return i;
        };
        // (Its links are still carried — drawn, but not part of the drawing yet.)
        st().tickGraphFlow(1, Number.POSITIVE_INFINITY);
        expect(info2().crossings).toBe(0);
        // It grows back as the rest makes room.
        let ticks = 0;
        do {
            st().tickGraphFlow(1, Number.POSITIVE_INFINITY);
            ticks++;
        } while (info2().growing !== null && ticks < 400);
        expect(info2().growing).toBeNull();
        // Full size, exactly where it was put, held there by its pins.
        expect(st().live[0][0]).toBeCloseTo(base[0] + 2, 9);
        expect(
            Math.hypot(st().live[1][0] - st().live[0][0], st().live[1][1] - st().live[0][1]),
        ).toBeCloseTo(side, 9);
        const pin1 = st().pins.find((p) => p.vertexIndex === 1);
        expect(pin1?.target[0]).toBeCloseTo(st().live[1][0], 12);
        // Its links are down now: they cross what lies between, and that is all that crosses.
        const at = crossings();
        expect(at).toBeGreaterThan(0);
        expect(info2().crossings).toBe(at);
        let worst = at;
        for (let k = 0; k < 120; k++) {
            st().tickGraphFlow(1, Number.POSITIVE_INFINITY);
            worst = Math.max(worst, crossings());
        }
        expect(worst).toBe(at);
        expect(st().live[0][0]).toBeCloseTo(base[0] + 2, 9); // held
        expect(info2().carrying).toBe(false);
        expect(st().live.every((p) => p.every(Number.isFinite))).toBe(true);
        st().setRunning(false);
        st().setGraphFlowMoveThrough(false);
    });

    test('pausing while a group is in the air sets it down first', () => {
        st().setEdgeSubdivision(2);
        st().setGraphFlow(true);
        st().setGraphFlowMoveThrough(true);
        st().addPin(0);
        st().setRunning(true);
        const base = [...st().live[0]];
        st().beginPinDrag(0);
        st().setPinTarget(0, [base[0] + 2, base[1] + 2, 0]);
        st().tickGraphFlow(1, Number.POSITIVE_INFINITY);
        expect(st().graphFlowRun?.move).not.toBeNull();
        st().setRunning(false);
        expect(st().pinDrag).toBeNull();
        // Set down (a single node has no size to grow back to): nothing is in the air, the
        // run is dropped as on any pause, and the drawing has the crossings of its two links.
        expect(st().graphFlowRun).toBeNull();
        const at = crossings();
        expect(at).toBeGreaterThan(0);
        // The next run takes the drawing as it is, crossings included, and can move.
        st().setRunning(true);
        st().tickGraphFlow(1, Number.POSITIVE_INFINITY);
        const info = st().graphFlowInfo;
        expect(info?.phase === 'running' && info.crossings).toBe(at);
        st().setRunning(false);
        st().setGraphFlowMoveThrough(false);
    });
});
