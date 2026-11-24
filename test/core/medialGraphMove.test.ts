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
    facingAngle,
    growGroup,
    hangGroupOn,
    nearestOutsideNode,
    prunePassPairs,
    reattachGroup,
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

    describe('turning and re-attaching', () => {
        /** The radial tree, settled; `branch` is one of its six main branches. */
        const settled = () => {
            const { g, branch } = radialTree(2);
            for (let k = 0; k < 30; k++) medialGraphStep(g, { r: 0.25 });
            return { g, branch };
        };
        const centroid = (g: ReturnType<typeof radialTree>['g'], nodes: number[]) => {
            let cx = 0;
            let cy = 0;
            for (const i of nodes) {
                cx += g.x[3 * i] / nodes.length;
                cy += g.x[3 * i + 1] / nodes.length;
            }
            return [cx, cy] as const;
        };
        test('a carried branch turns to face its parent the way it did where it stood', () => {
            const { g, branch } = settled();
            const root = branch[0]; // linked to node 0, the tree's root
            const [cx, cy] = centroid(g, branch);
            /** Angle at the branch's centroid from "towards the parent" to "towards its own root". */
            const bearing = (): number => {
                const [nx, ny] = centroid(g, branch);
                const p = [g.x[0] - nx, g.x[1] - ny];
                const q = [g.x[3 * root] - nx, g.x[3 * root + 1] - ny];
                return Math.atan2(p[0] * q[1] - p[1] * q[0], p[0] * q[0] + p[1] * q[1]);
            };
            const before = bearing();
            expect(Math.abs(before)).toBeLessThan(0.5); // it faces its parent to begin with
            const move = beginGroupMove(g, branch);
            // Picking up turns nothing: every point is where it was, to the last bit.
            const x0 = Float64Array.from(g.x);
            setGroupOffset(g, move, 0, 0, { turn: 'links' });
            expect(Math.abs(move.angle)).toBeLessThan(1e-12);
            for (const i of branch) expect(Math.abs(g.x[3 * i] - x0[3 * i])).toBeLessThan(1e-12);
            // To the far side of the root: unturned, the branch would point back at it.
            setGroupOffset(g, move, -2 * cx, -2 * cy, { turn: 'links' });
            expect(bearing()).toBeCloseTo(before, 9);
            expect(Math.abs(Math.abs(move.angle) - Math.PI)).toBeLessThan(0.2); // about half a turn
            // A turn, not a deformation: every distance inside the group is what it was.
            const home = (i: number) => {
                const k = move.nodes.indexOf(i);
                return [move.home[2 * k], move.home[2 * k + 1]];
            };
            for (const a of branch.slice(0, 5))
                for (const b of branch.slice(5, 10))
                    expect(
                        Math.hypot(g.x[3 * a] - g.x[3 * b], g.x[3 * a + 1] - g.x[3 * b + 1]),
                    ).toBeCloseTo(Math.hypot(home(a)[0] - home(b)[0], home(a)[1] - home(b)[1]), 9);
        });
        test('without the option nothing turns: a plain carry is home + offset exactly', () => {
            const { g, branch } = settled();
            const move = beginGroupMove(g, branch);
            setGroupOffset(g, move, 3, 1);
            expect(move.angle).toBe(0);
            for (let k = 0; k < move.points.length; k++) {
                expect(g.x[3 * move.points[k]]).toBe(move.home[2 * k] + 3);
                expect(g.x[3 * move.points[k] + 1]).toBe(move.home[2 * k + 1] + 1);
            }
        });
        test('the node under the pointer stays under it while the group turns', () => {
            const { g, branch } = settled();
            const held = branch[branch.length - 1]; // a leaf, far from the centroid
            const hx = g.x[3 * held];
            const hy = g.x[3 * held + 1];
            const [cx, cy] = centroid(g, branch);
            const move = beginGroupMove(g, branch);
            for (const t of [0.25, 0.5, 0.75, 1]) {
                const dx = -2 * cx * t + 0.3 * t;
                const dy = -2 * cy * t - 0.2 * t;
                setGroupOffset(g, move, dx, dy, { turn: 'links', hold: held });
                expect(
                    Math.hypot(g.x[3 * held] - hx - dx, g.x[3 * held + 1] - hy - dy),
                ).toBeLessThan(1e-3);
            }
            expect(Math.abs(move.angle)).toBeGreaterThan(1); // and it did turn
        });
        test('with several links the turn is the one that keeps them shortest', () => {
            const g = grid(8, 2);
            const blob = [27, 28, 35, 36]; // a 2×2 block in the middle: 8 links to the rest
            const move = beginGroupMove(g, blob);
            setGroupOffset(g, move, 0.4, -0.3, { turn: 'links' });
            const cost = (angle: number): number => {
                move.angle = angle;
                setGroupOffset(g, move, 0.4, -0.3);
                let sum = 0;
                for (const e of move.connecting)
                    sum +=
                        (g.x[3 * g.edgeA[e]] - g.x[3 * g.edgeB[e]]) ** 2 +
                        (g.x[3 * g.edgeA[e] + 1] - g.x[3 * g.edgeB[e] + 1]) ** 2;
                return sum;
            };
            const best = facingAngle(g, move);
            const atBest = cost(best);
            for (let k = 0; k < 720; k++)
                expect(cost((k * Math.PI) / 360)).toBeGreaterThanOrEqual(atBest - 1e-9);
        });
        test('a branch hung on another node lands turned towards it, on a visible node, with no crossing', () => {
            const { g, branch } = settled();
            const [cx, cy] = centroid(g, branch);
            const oldParent = 0;
            const move = beginGroupMove(g, branch);
            expect(move.connecting.length).toBe(1);
            const link = move.connecting[0];
            // Carried a third of a turn round the root: on top of another branch.
            const c = Math.cos((2 * Math.PI) / 3);
            const s = Math.sin((2 * Math.PI) / 3);
            setGroupOffset(g, move, c * cx - s * cy - cx, s * cx + c * cy - cy, { turn: 'links' });
            const to = nearestOutsideNode(g, move, move.cx + move.dx, move.cy + move.dy);
            expect(to).not.toBe(oldParent);
            hangGroupOn(g, move, to);
            const drop = dropGroupMove(g, move, { r: 0.25 });
            const growth = drop.growth;
            let steps = 0;
            while (growth && growth.scale < 1 && steps < 200) {
                growGroup(g, growth, 0.25);
                medialGraphStep(g, { r: 0.25 });
                steps++;
            }
            expect(growth === null || growth.scale === 1).toBe(true);
            // The link no longer goes to the old parent, and the tree is drawn without crossings.
            const ends = [g.edgeA[link], g.edgeB[link]];
            expect(ends).toContain(branch[0]);
            expect(ends).not.toContain(oldParent);
            expect(countCrossings(g)).toBe(0);
            // The flow runs on the changed graph and keeps it so.
            for (const i of branch) g.pinned[i] = 1;
            for (let k = 0; k < 60; k++) medialGraphStep(g, { r: 0.25 });
            expect(countCrossings(g)).toBe(0);
            expect(finite(g.x)).toBe(true);
        });
        test('a landed link takes up its slack as the drawing settles, down to its old length and no further', () => {
            const { g, branch } = settled();
            const [cx, cy] = centroid(g, branch);
            const move = beginGroupMove(g, branch);
            const link = move.connecting[0];
            const restOf = (): number => {
                let sum = 0;
                for (let q = 0; q < g.pieces; q++) sum += g.rest[link * g.pieces + q];
                return sum;
            };
            const before = restOf();
            setGroupOffset(g, move, 1.5 * cx, 1.5 * cy); // straight out: the link gets longer
            const drop = dropGroupMove(g, move, { r: 0.25 });
            expect(drop.growth).toBeNull(); // out in the open: full size at once, links landed
            const landed = restOf();
            expect(landed).toBeGreaterThan(1.5 * before);
            expect(g.slack?.get(link)).toBeCloseTo(before, 12);
            // The ends come together (here: the branch's root is put half way back): the rest
            // length follows, so the link has no excess to fold up.
            const root = branch[0];
            g.x[3 * root] -= 0.5 * (g.x[3 * root] - g.x[0]);
            g.x[3 * root + 1] -= 0.5 * (g.x[3 * root + 1] - g.x[1]);
            const chord = Math.hypot(g.x[3 * root] - g.x[0], g.x[3 * root + 1] - g.x[1]);
            for (const i of branch) g.pinned[i] = 1;
            g.pinned[0] = 1;
            medialGraphStep(g, { r: 0.25 });
            expect(restOf()).toBeCloseTo(Math.max(before, chord), 9);
            expect(restOf()).toBeLessThan(landed);
            // Pulled right up to its parent: it stops at its old length and the rule retires.
            g.x[3 * root] = g.x[0] + 0.3 * before * (cx / Math.hypot(cx, cy));
            g.x[3 * root + 1] = g.x[1] + 0.3 * before * (cy / Math.hypot(cx, cy));
            medialGraphStep(g, { r: 0.25 });
            expect(restOf()).toBeCloseTo(before, 9);
            expect(g.slack?.has(link)).toBe(false);
            // … and stretching it again does not lengthen it: slack is taken up, never paid out.
            g.x[3 * root] = g.x[0] + 3 * before * (cx / Math.hypot(cx, cy));
            g.x[3 * root + 1] = g.x[1] + 3 * before * (cy / Math.hypot(cx, cy));
            medialGraphStep(g, { r: 0.25 });
            expect(restOf()).toBeCloseTo(before, 9);
        });
        test('re-attaching refuses what would not be a graph', () => {
            const { g, branch } = settled();
            const move = beginGroupMove(g, branch);
            const link = move.connecting[0];
            expect(() => reattachGroup(g, move, link, branch[3])).toThrow(); // into the group
            expect(() => reattachGroup(g, move, 9999, 5)).toThrow(); // not a link of the group
            const g2 = grid(4, 2);
            const m2 = beginGroupMove(g2, [5]);
            // Node 5 is already linked to node 6: a second link there would be a double edge.
            const toFour = m2.connecting.find((e) => g2.edgeA[e] === 4 || g2.edgeB[e] === 4);
            expect(() => reattachGroup(g2, m2, toFour as number, 6)).toThrow();
            expect(() => hangGroupOn(g2, m2, 10)).toThrow(); // four links: which one?
        });
    });
});
