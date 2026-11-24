// bench/app/pkmGraphs.ts — bun bench/app/pkmGraphs.ts [dir] [steps=400] [maxNodes=5000] [out] [makeRoom=warp|flow] [turn=none|links|hang|scored]
// The graph flow and the group move on REAL note collections: mind maps (FreeMind / Freeplane
// files) and note vaults (Obsidian-style links), as extracted by scripts/pkmExtract.ts into
// local_files/pkm-graphs/graphs/ (git-ignored; see its README for where they come from).
// One JSON line per (graph, layout):
//   mind map      twice: as mind-map programs draw it (root in the middle, branches right and
//                 left, every subtree a vertical stack: "tidy tree"), and as a graph view
//                 would (d3-style force layout of the tree). Cross links (arrowlinks) and any
//                 crossings of the force layout are allowed crossings.
//   vault, links  every link an edge, force layout: what a "graph view" shows. Only its
//                 crossings are counted — it is a hairball, not a drawing to keep planar.
//   vault as a tree  the same notes as a mind map: a breadth-first spanning tree from the
//                 most linked note, the other links left out; both layouts as above.
// For every tree layout: the flow settles it, then a mid-sized main branch is carried to the
// other side of the root, set down on top of what is there, and grows back.
// @see src/core/medialGraphMove.ts, bench/app/medialGroupMove.ts (the synthetic counterpart)
import { appendFileSync, existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { runForceLayout } from '../../src/core/forceLayout';
import {
    allowedCrossingsNow,
    beginGroupMove,
    countCrossings,
    crossingPairs,
    dropGroupMove,
    growGroup,
    hangGroupOn,
    nearestOutsideNode,
    prunePassPairs,
    setGroupOffset,
    settleTurn,
} from '../../src/core/medialGraphMove';
import { medialGraphFrom, medialGraphStep, minClearance } from '../../src/core/medialSphereGraph';
import type { Edge, Vec3 } from '../../src/core/testConfigs';

const DIR = process.argv[2] ?? 'local_files/pkm-graphs/graphs';
const STEPS = Number(process.argv[3] ?? 400);
const MAX_NODES = Number(process.argv[4] ?? 5000);
const OUT = process.argv[5] ?? 'bench/results/2026-11-24-pkm-graphs.jsonl';
// How the rest makes room for a group growing back (see `growGroup`). 'flow' is the
// comparison run: it repeats the moves only, not the hairball count.
const MAKE_ROOM = (process.argv[6] ?? 'warp') as 'warp' | 'flow';
// How the carried group is oriented: 'none' = as picked up; 'links' = turned to face what it is
// linked to; 'hang' = also re-attached to the node nearest to where it is held (only a group
// with exactly one link; otherwise as 'links').
const TURN = (process.argv[7] ?? 'none') as 'none' | 'links' | 'hang' | 'scored';
const DUMP = process.env.DUMP;
// FRAMES=<prefix>: also write every step of each group move as
// `<prefix>-<graph>-<layout>-<makeRoom>.json`, a panel for bench/app/renderGraphFlow.py.
const FRAMES = process.env.FRAMES;
const PIECES = 2;
const R = 0.3; // clearance, in units of the spacing between neighbouring leaves (1)

if (!existsSync(DIR)) {
    console.error(`no graphs in ${DIR}: run scripts/pkmExtract.ts first (see the file header)`);
    process.exit(0);
}

interface Source {
    name: string;
    kind: 'mm' | 'vault';
    nodes: number;
    edges: [number, number][];
    parent?: number[];
}

/**
 * Mind-map layout of a rooted tree: depth along x (the root's children alternate right and
 * left), leaves one unit apart along y, every inner node at the middle of its children.
 * Straight edges of such a layout never cross.
 * The step in x from a node to its children is 3, or a third of the children's spread in y if
 * that is more. Why: real maps have hubs with 200 children; at a fixed step their edges are a
 * fan a hundredth of a unit apart (mind-map programs draw them as curves for that reason).
 */
function tidyTree(n: number, parent: number[], root: number): Vec3[] {
    const kids: number[][] = Array.from({ length: n }, () => []);
    for (let i = 0; i < n; i++) if (parent[i] >= 0) kids[parent[i]].push(i);
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
    for (let k = 0; k < kids[root].length; k++) placeY(kids[root][k], k % 2);
    const shift = [0.5 * (next[0] - 1), 0.5 * (next[1] - 1)];
    const sideOf = new Int8Array(n);
    const mark = (i: number, side: number): void => {
        sideOf[i] = side;
        V[i][1] -= shift[side];
        for (const c of kids[i]) mark(c, side);
    };
    for (let k = 0; k < kids[root].length; k++) mark(kids[root][k], k % 2);
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
        rootSpread[k % 2] = Math.max(rootSpread[k % 2], Math.abs(V[kids[root][k]][1]));
    rootSpread = rootSpread.map((v) => Math.max(3, (2 * v) / 3));
    for (let k = 0; k < kids[root].length; k++) {
        const c = kids[root][k];
        V[c][0] = k % 2 === 0 ? rootSpread[0] : -rootSpread[1];
        placeX(c, k % 2);
    }
    return V;
}

/**
 * The tree as a graph view would show it: a d3-style force layout of the tree's own edges,
 * started from a radial drawing (so branches start out untangled) and scaled to a mean edge
 * length of 2. Unlike the tidy tree it may have a few crossings; they are allowed ones.
 */
function organicTree(n: number, parent: number[], root: number): Vec3[] {
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

/** Breadth-first spanning tree from the node with the most links. */
function spanningTree(n: number, edges: [number, number][]): { parent: number[]; root: number } {
    const adj: number[][] = Array.from({ length: n }, () => []);
    for (const [a, b] of edges) {
        adj[a].push(b);
        adj[b].push(a);
    }
    let root = 0;
    for (let i = 1; i < n; i++) if (adj[i].length > adj[root].length) root = i;
    const parent = new Array<number>(n).fill(-2);
    parent[root] = -1;
    const queue = [root];
    for (let q = 0; q < queue.length; q++)
        for (const w of adj[queue[q]])
            if (parent[w] === -2) {
                parent[w] = queue[q];
                queue.push(w);
            }
    return { parent, root };
}

function emit(row: Record<string, unknown>): void {
    console.log(JSON.stringify(row));
    appendFileSync(OUT, `${JSON.stringify(row)}\n`);
}

/** Settle a tree layout with the flow, then move one main branch across the root. */
function runTree(
    name: string,
    layout: string,
    n: number,
    parent: number[],
    root: number,
    extra: Edge[],
    V: Vec3[],
): void {
    const E: Edge[] = [];
    for (let i = 0; i < n; i++) if (parent[i] >= 0) E.push([parent[i], i]);
    for (const e of extra) E.push(e);
    const g = medialGraphFrom(V, E, PIECES);
    const dump = (stage: number, hi: number[]): void => {
        if (!DUMP) return;
        writeFileSync(
            `${DUMP}-${name}-${layout}-${stage}.json`,
            JSON.stringify({
                x: Array.from(g.x),
                a: Array.from(g.segs.a),
                b: Array.from(g.segs.b),
                nodeCount: g.nodeCount,
                hi,
            }),
        );
    };
    g.passPairs = crossingPairs(g, 1e-9 * R);
    const crossingsStart = allowedCrossingsNow(g);
    const clearanceStart = minClearance(g, 2 * R);
    let ms = 0;
    let cgMax = 0;
    let move = 0;
    let limited = 0;
    let t = performance.now();
    for (let k = 0; k < STEPS; k++) {
        const st = medialGraphStep(g, { r: R });
        prunePassPairs(g, 0.05 * R);
        cgMax = Math.max(cgMax, st.cgIterations);
        move = st.movePerNode;
        limited = st.cappedNodes;
    }
    ms = (performance.now() - t) / STEPS;
    const crossingsSettled = allowedCrossingsNow(g);
    const clearanceSettled = minClearance(g, 2 * R);

    // The branch to move: the main branch nearest a tenth of the map in size.
    const kids: number[][] = Array.from({ length: n }, () => []);
    for (let i = 0; i < n; i++) if (parent[i] >= 0) kids[parent[i]].push(i);
    const sub = (i: number): number[] => {
        const out = [i];
        for (let q = 0; q < out.length; q++) for (const c of kids[out[q]]) out.push(c);
        return out;
    };
    const mains = kids[root]
        .map(sub)
        .sort((a, b) => Math.abs(a.length - n / 10) - Math.abs(b.length - n / 10));
    const blob = mains[0];
    dump(0, blob);
    let cx = 0;
    for (const i of blob) cx += g.x[3 * i] / blob.length;
    const frames: { x: number[]; label: string }[] = [];
    const frame = (label: string): void => {
        if (!FRAMES) return;
        const xy = new Array<number>(2 * g.n);
        for (let i = 0; i < g.n; i++) {
            xy[2 * i] = Math.round(g.x[3 * i] * 1000) / 1000;
            xy[2 * i + 1] = Math.round(g.x[3 * i + 1] * 1000) / 1000;
        }
        frames.push({ x: xy, label });
    };
    const grab = beginGroupMove(g, blob);
    for (let k = 1; k <= 10; k++) {
        setGroupOffset(g, grab, (-2 * cx * k) / 10, 0, {
            turn: TURN === 'none' ? 'none' : 'links',
        }); // to the mirror place across the root
        medialGraphStep(g, { r: R });
        frame('picked up: carried over the map');
    }
    // 'scored': carried turned to its links, then set down at the angle with fewest crossings.
    if (TURN === 'scored') settleTurn(g, grab);
    if (TURN === 'hang' && grab.connecting.length === 1)
        hangGroupOn(g, grab, nearestOutsideNode(g, grab, grab.cx + grab.dx, grab.cy + grab.dy));
    const drop = dropGroupMove(g, grab, { r: R });
    dump(1, blob);
    let stepsToFullSize: number | null = drop.growth ? null : 0;
    let crossingsAtRelease: number | null = drop.growth ? null : countCrossings(g);
    let worst = crossingsAtRelease ?? 0;
    let scale = drop.scale;
    if (!drop.growth) for (const i of blob) g.pinned[i] = 1;
    t = performance.now();
    let after = 0;
    let ran = 0;
    for (let k = 0; k < 3 * STEPS && after < STEPS; k++) {
        ran++;
        if (drop.growth && stepsToFullSize === null) {
            scale = growGroup(g, drop.growth, R, { makeRoom: MAKE_ROOM });
            if (scale >= 1) {
                stepsToFullSize = k + 1;
                for (const i of blob) g.pinned[i] = 1;
                crossingsAtRelease = countCrossings(g);
                worst = crossingsAtRelease;
            }
        } else after++;
        medialGraphStep(g, { r: R });
        prunePassPairs(g, 0.05 * R);
        if (stepsToFullSize !== null && after % 10 === 0)
            worst = Math.max(worst, allowedCrossingsNow(g));
        frame(
            stepsToFullSize === null
                ? `step ${k + 1}: set down small, growing back, ${Math.round(100 * scale)}% of its size`
                : `full size after ${stepsToFullSize} steps; links landed, ${crossingsAtRelease} crossings in the map, held`,
        );
    }
    dump(2, blob);
    if (FRAMES)
        writeFileSync(
            `${FRAMES}-${name}-${layout}-${MAKE_ROOM}.json`
                .replaceAll(' ', '_')
                .replaceAll(',', ''),
            JSON.stringify({
                scene: 'pkm',
                panels: [
                    {
                        title:
                            MAKE_ROOM === 'flow'
                                ? 'waiting: the rest is pushed away by contact forces only'
                                : 'making room: the rest is moved out of the way directly',
                        a: Array.from(g.segs.a),
                        b: Array.from(g.segs.b),
                        nodeCount: g.nodeCount,
                        hi: blob,
                        frames,
                    },
                ],
            }),
        );
    emit({
        bench: 'pkm-graphs',
        graph: name,
        layout,
        nodes: n,
        edges: E.length,
        crossLinks: extra.length,
        segments: g.segs.a.length,
        r: R,
        steps: STEPS,
        crossingsStart,
        crossingsSettled,
        clearanceStart: Number(clearanceStart.toFixed(3)),
        clearanceSettled: Number(clearanceSettled.toFixed(3)),
        msPerStep: Number(ms.toFixed(1)),
        cgMax,
        movePerNodeSettled: Number(move.toExponential(2)),
        limitedSettled: limited,
        moveGroupNodes: blob.length,
        makeRoom: MAKE_ROOM,
        turn: TURN,
        moveSetDownScale: Number(drop.scale.toFixed(3)),
        moveScaleEnd: Number(scale.toFixed(3)),
        moveStepsToFullSize: stepsToFullSize,
        moveCrossingsAtRelease: crossingsAtRelease,
        moveCrossingsWorst: worst,
        moveCrossingsEnd: countCrossings(g),
        // While the group is still growing its links are carried straight over the map, so a
        // clearance would only measure those; null says "not landed yet".
        moveClearanceEnd:
            stepsToFullSize === null ? null : Number(minClearance(g, 2 * R).toFixed(3)),
        moveMsPerStep: Number(((performance.now() - t) / Math.max(1, ran)).toFixed(1)),
        finite: g.x.every(Number.isFinite),
    });
}

const LAYOUTS: [string, (n: number, parent: number[], root: number) => Vec3[]][] = [
    ['tidy tree', tidyTree],
    ['force layout', organicTree],
];
const files = readdirSync(DIR)
    .filter((f) => f.endsWith('.json'))
    .sort();
for (const f of files) {
    const src = JSON.parse(readFileSync(join(DIR, f), 'utf8')) as Source;
    if (src.nodes > MAX_NODES || src.nodes < 20) {
        console.error(
            `[pkmGraphs] ${src.name}: ${src.nodes} nodes, skipped (limits 20 … ${MAX_NODES})`,
        );
        continue;
    }
    if (src.kind === 'mm' && src.parent) {
        const root = src.parent.indexOf(-1);
        const tree = new Set(
            src.parent.map((p, i) => (p < 0 ? '' : `${Math.min(p, i)},${Math.max(p, i)}`)),
        );
        const extra = src.edges.filter(
            ([a, b]) => !tree.has(`${Math.min(a, b)},${Math.max(a, b)}`),
        );
        for (const [layout, make] of LAYOUTS)
            runTree(
                src.name,
                `mind map, ${layout}`,
                src.nodes,
                src.parent,
                root,
                extra as Edge[],
                make(src.nodes, src.parent, root),
            );
        continue;
    }
    // A vault. First what a graph view shows: every link, force layout, crossings counted.
    let seed = 5;
    const rnd = (): number => {
        seed = (1664525 * seed + 1013904223) % 4294967296;
        return seed / 4294967296;
    };
    const spread = Math.sqrt(src.nodes);
    const V0: Vec3[] = Array.from(
        { length: src.nodes },
        () => [spread * (rnd() - 0.5), spread * (rnd() - 0.5), 0] as Vec3,
    );
    if (MAKE_ROOM !== 'flow') {
        let t = performance.now();
        const forced = runForceLayout(V0, src.edges as Edge[]);
        const layoutMs = performance.now() - t;
        t = performance.now();
        const hairball = medialGraphFrom(forced.vertices, src.edges as Edge[], 1);
        emit({
            bench: 'pkm-graphs',
            graph: src.name,
            layout: 'vault, links (force layout)',
            nodes: src.nodes,
            edges: src.edges.length,
            forceLayoutMs: Math.round(layoutMs),
            crossings: countCrossings(hairball),
            countMs: Math.round(performance.now() - t),
        });
    }
    // Then the same notes as a mind map.
    const { parent, root } = spanningTree(src.nodes, src.edges);
    for (const [layout, make] of LAYOUTS)
        runTree(
            src.name,
            `vault as a tree, ${layout}`,
            src.nodes,
            parent,
            root,
            [],
            make(src.nodes, parent, root),
        );
}
