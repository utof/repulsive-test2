// bench/app/treeLayouts.ts — two drawings of a rooted tree, shared by the benchmarks on real
// note collections (pkmGraphs.ts, planarSkeleton.ts).
import { runForceLayout } from '../../src/core/forceLayout';
import type { Edge, Vec3 } from '../../src/core/testConfigs';

/**
 * Mind-map layout of a rooted tree: depth along x (the root's children alternate right and
 * left), leaves one unit apart along y, every inner node at the middle of its children.
 * Straight edges of such a layout never cross.
 * The step in x from a node to its children is 3, or a third of the children's spread in y if
 * that is more. Why: real maps have hubs with 200 children; at a fixed step their edges are a
 * fan a hundredth of a unit apart (mind-map programs draw them as curves for that reason).
 */
export function tidyTree(n: number, parent: number[], root: number, order?: number[][]): Vec3[] {
    // `order`: every node's children in the order to draw them (`orderSiblings`); then the
    // root's first half goes right and its second half left, so that neighbours in the
    // order are neighbours in the drawing. Without it: by index, alternating sides.
    const kids: number[][] = order ?? Array.from({ length: n }, () => []);
    if (!order) for (let i = 0; i < n; i++) if (parent[i] >= 0) kids[parent[i]].push(i);
    const half = Math.ceil(kids[root].length / 2);
    const sideAt = (k: number): number => (order ? (k < half ? 0 : 1) : k % 2);
    const V: Vec3[] = Array.from({ length: n }, () => [0, 0, 0] as Vec3);
    const next = [0, 0]; // next free leaf slot on the right / left side
    // First y (leaf slots, inner nodes in the middle of their children) …
    const placeY = (i: number, side: number): void => {
        if (kids[i].length === 0) {
            V[i][1] = next[side]++;
            return;
        }
        for (const c of kids[i]) placeY(c, side);
        V[i][1] = 0.5 * (V[kids[i][0]][1] + V[kids[i][kids[i].length - 1]][1]);
    };
    for (let k = 0; k < kids[root].length; k++) placeY(kids[root][k], sideAt(k));
    const shift = [0.5 * (next[0] - 1), 0.5 * (next[1] - 1)];
    const sideOf = new Int8Array(n);
    const mark = (i: number, side: number): void => {
        sideOf[i] = side;
        V[i][1] -= shift[side];
        for (const c of kids[i]) mark(c, side);
    };
    for (let k = 0; k < kids[root].length; k++) mark(kids[root][k], sideAt(k));
    // … then x, outwards from the root.
    const placeX = (i: number, side: number): void => {
        if (kids[i].length === 0) return;
        const spread = Math.abs(V[kids[i][kids[i].length - 1]][1] - V[kids[i][0]][1]);
        const step = Math.max(3, spread / 3);
        for (const c of kids[i]) {
            V[c][0] = V[i][0] + (side === 0 ? step : -step);
            placeX(c, side);
        }
    };
    let rootSpread = [0, 0];
    for (let k = 0; k < kids[root].length; k++)
        rootSpread[sideAt(k)] = Math.max(rootSpread[sideAt(k)], Math.abs(V[kids[root][k]][1]));
    rootSpread = rootSpread.map((v) => Math.max(3, (2 * v) / 3));
    for (let k = 0; k < kids[root].length; k++) {
        const c = kids[root][k];
        V[c][0] = sideAt(k) === 0 ? rootSpread[0] : -rootSpread[1];
        placeX(c, sideAt(k));
    }
    return V;
}

/**
 * The tree as a graph view would show it: a d3-style force layout of the tree's own edges,
 * started from a radial drawing (so branches start out untangled) and scaled to a mean edge
 * length of 2. Unlike the tidy tree it may have a few crossings; they are allowed ones.
 */
export function organicTree(n: number, parent: number[], root: number): Vec3[] {
    const tidy = tidyTree(n, parent, root);
    // Radial start: the tidy tree's leaf order as the angle, its depth as the radius.
    let lo = Infinity;
    let hi = -Infinity;
    for (const p of tidy) {
        lo = Math.min(lo, p[1]);
        hi = Math.max(hi, p[1]);
    }
    const depth = new Int32Array(n);
    const order = [root];
    for (let q = 0; q < order.length; q++)
        for (let i = 0; i < n; i++)
            if (parent[i] === order[q]) {
                depth[i] = depth[order[q]] + 1;
                order.push(i);
            }
    const start: Vec3[] = tidy.map((p, i) => {
        const d = depth[i];
        const turn = (Math.PI * (p[1] - lo)) / (hi - lo + 1) + (p[0] < 0 ? Math.PI : 0);
        return [d * Math.cos(turn), d * Math.sin(turn), 0] as Vec3;
    });
    const E: Edge[] = [];
    for (let i = 0; i < n; i++) if (parent[i] >= 0) E.push([parent[i], i]);
    const out = runForceLayout(start, E).vertices;
    let mean = 0;
    for (const [a, b] of E)
        mean += Math.hypot(out[a][0] - out[b][0], out[a][1] - out[b][1]) / E.length;
    return out.map((p) => [(2 * p[0]) / mean, (2 * p[1]) / mean, 0] as Vec3);
}
