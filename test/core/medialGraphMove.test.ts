// Moving a group across a flat drawing: pick up, carry, set down inside one face, grow back; the
// crossings its links make are allowed and their number cannot grow. @see src/core/medialGraphMove.ts
import { describe, expect, test } from 'bun:test';
import {
    allowedCrossingsNow,
    beginGroupMove,
    cancelGroupMove,
    countCrossings,
    crossingPairs,
    dropGroupMove,
    growGroup,
    prunePassPairs,
    roomRadius,
    setGroupOffset,
} from '../../src/core/medialGraphMove';
import { medialGraphFrom, medialGraphStep, minClearance } from '../../src/core/medialSphereGraph';
import type { Edge, Vec3 } from '../../src/core/testConfigs';

const R = 0.4;

/** A jittered size×size grid with unit spacing (deterministic). */
function grid(size: number, pieces: number) {
    let s = 7;
    const rnd = () => {
        s = (1664525 * s + 1013904223) % 4294967296;
        return s / 4294967296;
    };
    const V: Vec3[] = [];
    const E: Edge[] = [];
    for (let y = 0; y < size; y++)
        for (let x = 0; x < size; x++)
            V.push([x + 0.3 * (rnd() - 0.5), y + 0.3 * (rnd() - 0.5), 0]);
    for (let y = 0; y < size; y++)
        for (let x = 0; x < size; x++) {
            if (x + 1 < size) E.push([y * size + x, y * size + x + 1]);
            if (y + 1 < size) E.push([y * size + x, (y + 1) * size + x]);
        }
    return medialGraphFrom(V, E, pieces);
}
const finite = (x: Float64Array) => x.every(Number.isFinite);

/** A radial tree: 6 branches from the root, every node two children, 4 levels (91 nodes);
 *  `branch` = the nodes of the first branch (15). */
function radialTree(pieces: number) {
    const V: Vec3[] = [[0, 0, 0]];
    const E: Edge[] = [];
    const branch: number[] = [];
    const lay = (parent: number, level: number, a0: number, a1: number, mine: boolean): void => {
        const a = 0.5 * (a0 + a1);
        const i = V.length;
        V.push([1.2 * level * Math.cos(a), 1.2 * level * Math.sin(a), 0]);
        E.push([parent, i]);
        if (mine) branch.push(i);
        if (level === 4) return;
        lay(i, level + 1, a0, a, mine);
        lay(i, level + 1, a, a1, mine);
    };
    for (let b = 0; b < 6; b++) lay(0, 1, (b * Math.PI) / 3, ((b + 1) * Math.PI) / 3, b === 0);
    return { g: medialGraphFrom(V, E, pieces), branch };
}

describe('group move', () => {
    test('a carried group is invisible to the flow and stays where the caller puts it', () => {
        const g = grid(8, 2);
        const blob = [0, 1, 8, 9];
        const move = beginGroupMove(g, blob);
        expect(move.internal.length).toBe(4); // the square's four sides
        expect(move.connecting.length).toBe(4);
        setGroupOffset(g, move, 3.5, 3.5); // right on top of the grid's middle
        const placed = blob.map((i) => [g.x[3 * i], g.x[3 * i + 1]]);
        let limited = 0;
        for (let k = 0; k < 20; k++) limited += medialGraphStep(g, { r: R }).cappedNodes;
        blob.forEach((i, k) => {
            expect(g.x[3 * i]).toBe(placed[k][0]);
            expect(g.x[3 * i + 1]).toBe(placed[k][1]);
        });
        // The rest is not frozen by the group lying across it (a guarded pair that crosses
        // has a motion limit of zero).
        expect(limited).toBe(0);
        expect(() => beginGroupMove(g, [20])).toThrow(); // one group at a time
    });

    test('set down on top of the grid: shrunk into one face, touching nothing, links still carried', () => {
        const g = grid(8, 2);
        const blob = [0, 1, 8, 9];
        const move = beginGroupMove(g, blob);
        setGroupOffset(g, move, 3.5, 3.5);
        const drop = dropGroupMove(g, move, { r: R });
        expect(drop.scale).toBeLessThan(1);
        expect(drop.scale).toBeGreaterThan(0.05);
        expect(drop.growth).not.toBeNull();
        // The square is in the drawing and crosses nothing; its four links are not down yet.
        expect(drop.crossings).toBe(0);
        expect(countCrossings(g)).toBe(0);
        for (const e of move.connecting) expect(g.ghost?.edge[e]).toBe(1);
        for (const e of move.internal) expect(g.ghost?.edge[e]).toBe(0);
        for (const i of blob) expect(g.ghost?.node[i]).toBe(1);
        expect(drop.targets.size).toBe(4);
    });

    test('it grows back to full size as the rest makes room; then its links land and crossings never grow', () => {
        const g = grid(8, 2);
        const blob = [0, 1, 8, 9];
        const move = beginGroupMove(g, blob);
        setGroupOffset(g, move, 3.5, 3.5);
        const drop = dropGroupMove(g, move, { r: R });
        const growth = drop.growth;
        if (!growth) throw new Error('expected a set-down below full size');
        let steps = 0;
        let last = growth.scale;
        while (growth.scale < 1 && steps < 300) {
            const sc = growGroup(g, growth, R);
            expect(sc).toBeGreaterThanOrEqual(last); // never shrinks
            last = sc;
            // While it grows nothing crosses: the group takes only room it has been given.
            if (sc < 1) expect(countCrossings(g)).toBe(0);
            medialGraphStep(g, { r: R });
            steps++;
        }
        expect(growth.scale).toBe(1);
        expect(g.ghost).toBeUndefined();
        // (The flow step after the release has moved the nodes a little; pin them as the app does.)
        for (const i of blob) g.pinned[i] = 1;
        // Now the links are down, straight across what lies between: those are the crossings.
        const at = countCrossings(g);
        expect(at).toBeGreaterThan(0);
        expect(allowedCrossingsNow(g)).toBe(at);
        const E = g.edgeA.length;
        const link = new Set(move.connecting);
        for (const key of g.passPairs?.keys() ?? [])
            expect(link.has(Math.floor(key / E)) || link.has(key % E)).toBe(true);
        // The links took their new length: no spring pulls the group back home.
        for (const e of move.connecting) expect(g.rest[e * 2]).toBeGreaterThan(1);
        let worst = at;
        for (let k = 0; k < 150; k++) {
            medialGraphStep(g, { r: R });
            prunePassPairs(g, 0.05 * R);
            worst = Math.max(worst, countCrossings(g));
        }
        expect(finite(g.x)).toBe(true);
        expect(worst).toBe(at); // not one more, at any step
        // Everything that is not allowed to cross keeps a real distance.
        expect(minClearance(g, 2)).toBeGreaterThan(0.02);
    });

    test('the room map is smooth, increasing, enlarges the disc and keeps areas far outside', () => {
        const c = 2;
        const k = 1.2;
        // Inside the disc: a plain enlargement.
        expect(roomRadius(0.7, c, k)).toBeCloseTo(0.84, 12);
        // Continuous, with a continuous slope, at the rim and at the end of the band (1.5·c).
        const slope = (rho: number, h: number) =>
            (roomRadius(rho + h, c, k) - roomRadius(rho, c, k)) / h;
        for (const at of [c, 1.5 * c]) {
            expect(roomRadius(at + 1e-9, c, k)).toBeCloseTo(roomRadius(at - 1e-9, c, k), 7);
            expect(slope(at, 1e-5)).toBeCloseTo(slope(at - 1e-5, 1e-5), 3);
        }
        // Increasing everywhere, and nothing is pulled inwards.
        let last = 0;
        for (let rho = 0.05; rho < 20; rho += 0.05) {
            const to = roomRadius(rho, c, k);
            expect(to).toBeGreaterThan(last);
            expect(to).toBeGreaterThanOrEqual(rho);
            last = to;
        }
        // Beyond the band every ring keeps its area: to² − rho² is one constant.
        const gain = (rho: number) => roomRadius(rho, c, k) ** 2 - rho * rho;
        expect(gain(9)).toBeCloseTo(gain(4), 9);
        expect(gain(40)).toBeCloseTo(gain(4), 9);
        // The shift dies off with distance.
        expect(roomRadius(40, c, k) - 40).toBeLessThan(0.2 * (roomRadius(4, c, k) - 4));
    });

    test('a big branch grows back by making room directly; waiting for the flow is far slower', () => {
        const grow = (makeRoom: 'warp' | 'flow', budget: number) => {
            const { g, branch } = radialTree(2);
            const r = 0.25;
            for (let k = 0; k < 30; k++) medialGraphStep(g, { r });
            const move = beginGroupMove(g, branch);
            // Half a turn round the root, as a translation: on top of the opposite branch.
            let cx = 0;
            let cy = 0;
            for (const i of branch) {
                cx += g.x[3 * i] / branch.length;
                cy += g.x[3 * i + 1] / branch.length;
            }
            setGroupOffset(g, move, -2 * cx, -2 * cy);
            const drop = dropGroupMove(g, move, { r });
            const growth = drop.growth;
            if (!growth) throw new Error('expected a set-down below full size');
            let steps = 0;
            let crossed = 0;
            while (growth.scale < 1 && steps < budget) {
                const sc = growGroup(g, growth, r, { makeRoom });
                // The rest is moved out of the way without anything passing through anything.
                if (sc < 1) crossed = Math.max(crossed, countCrossings(g));
                medialGraphStep(g, { r });
                steps++;
            }
            return { g, steps, scale: growth.scale, setDown: drop.scale, crossed };
        };
        const direct = grow('warp', 60); // measured: full size after 10 steps
        expect(direct.setDown).toBeLessThanOrEqual(0.5);
        expect(direct.scale).toBe(1);
        expect(direct.crossed).toBe(0);
        expect(finite(direct.g.x)).toBe(true);
        // The comparison: the same move, the rest pushed away by the flow's contact forces only.
        const waiting = grow('flow', direct.steps); // measured: 0.56 of its size by then
        expect(waiting.scale).toBeLessThan(0.8);
    });

    test('an allowed pair cannot get more crossings than allowed: the step undoes the move', () => {
        // Bar 2 lies 0.3 above bar 1 and its middle is dragged to well below bar 1.
        const bars = () =>
            medialGraphFrom(
                [
                    [0, 0, 0],
                    [4, 0, 0],
                    [0.5, 0.3, 0],
                    [3.5, 0.3, 0],
                ],
                [
                    [0, 1],
                    [2, 3],
                ] as Edge[],
                8,
            );
        const mid = 4 + 7 + 3; // interior point 4 of bar 2
        const run = (holdCrossings: boolean): number => {
            const g = bars();
            // The pair (edge 0, edge 1), key 0·2 + 1, may cross 0 times. Being in the map, it
            // is NOT held apart by the motion limits and its points do not repel.
            g.passPairs = new Map([[1, 0]]);
            g.pinned[2] = 1; // bar 2's ends are held, so its middle has to bow through bar 1
            g.pinned[3] = 1;
            const target = new Map<number, [number, number]>([[mid, [g.x[3 * mid], -1.5]]]);
            let worst = 0;
            for (let k = 0; k < 300; k++) {
                medialGraphStep(g, { r: R, pinTargets: target, pinForceMax: 1e6, holdCrossings });
                worst = Math.max(worst, countCrossings(g));
            }
            return worst;
        };
        expect(run(true)).toBe(0);
        expect(run(false)).toBeGreaterThan(0); // without the check it goes straight through
    });

    test('set down in empty space: full size, and a leaf that crosses nothing is not allowed to', () => {
        const g = grid(6, 2);
        // Corner node 0 moved straight out, away from the grid: its two links cross nothing.
        const move = beginGroupMove(g, [0]);
        setGroupOffset(g, move, -1.5, -1.5);
        const drop = dropGroupMove(g, move, { r: R });
        expect(drop.scale).toBe(1);
        expect(drop.crossings).toBe(0);
        expect(countCrossings(g)).toBe(0);
        expect(g.x[0]).toBeCloseTo(drop.targets.get(0)?.[0] ?? Number.NaN, 12);
    });

    test('cancel puts the group back, real again, with no crossings', () => {
        const g = grid(6, 2);
        const before = Array.from(g.x);
        const move = beginGroupMove(g, [14, 15]);
        setGroupOffset(g, move, 2, 1);
        const drop = cancelGroupMove(g, move, { r: R });
        expect(g.ghost).toBeUndefined();
        expect(drop.crossings).toBe(0);
        expect(countCrossings(g)).toBe(0);
        // Full-size positions are the old ones (the set-down may have shrunk it meanwhile).
        expect(drop.targets.get(14)?.[0]).toBeCloseTo(before[3 * 14], 12);
        expect(drop.targets.get(15)?.[1]).toBeCloseTo(before[3 * 15 + 1], 12);
    });

    test('a drawing that already has a crossing can flow once the pair is allowed', () => {
        // Two bars forming a lopsided X (their middle points do not coincide), each drawn 20%
        // longer than its rest length.
        const V: Vec3[] = [
            [-1, -1, 0],
            [1, 1, 0],
            [-1, 0.6, 0],
            [1, -0.9, 0],
        ];
        const make = () => {
            const g = medialGraphFrom(
                V,
                [
                    [0, 1],
                    [2, 3],
                ] as Edge[],
                1,
            );
            for (let k = 0; k < g.rest.length; k++) g.rest[k] /= 1.2;
            return g;
        };
        const length = (g: ReturnType<typeof make>) => Math.hypot(g.x[3] - g.x[0], g.x[4] - g.x[1]);
        // Guarded, the pair's distance is zero, so its motion limit is zero: frozen.
        const frozen = make();
        for (let k = 0; k < 30; k++) medialGraphStep(frozen, { r: R });
        expect(length(frozen)).toBeCloseTo(2 * Math.SQRT2, 9);
        // Allowed, the springs shorten both bars.
        const g = make();
        g.passPairs = crossingPairs(g);
        expect(g.passPairs.size).toBe(1);
        for (let k = 0; k < 30; k++) medialGraphStep(g, { r: R });
        expect(length(g)).toBeLessThan(0.9 * 2 * Math.SQRT2);
        expect(countCrossings(g)).toBe(1);
    });

    test('a pair that has come apart is guarded again', () => {
        const V: Vec3[] = [
            [-1, 0, 0],
            [1, 0, 0],
            [0, -1, 0],
            [0, 1, 0],
        ];
        const g = medialGraphFrom(
            V,
            [
                [0, 1],
                [2, 3],
            ] as Edge[],
            1,
        );
        g.passPairs = crossingPairs(g);
        expect(g.passPairs.size).toBe(1);
        expect(prunePassPairs(g, 0.02)).toBe(0); // still crossing
        g.x[3 * 2 + 1] = 0.5; // lift the vertical bar clear of the horizontal one
        g.x[3 * 3 + 1] = 2.5;
        expect(prunePassPairs(g, 1)).toBe(0); // apart, but not by that much
        expect(prunePassPairs(g, 0.02)).toBe(1);
        expect(g.passPairs.size).toBe(0);
    });
});
