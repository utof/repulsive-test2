// bench/app/recordGraphFlow.ts — bun bench/app/recordGraphFlow.ts <scene> <out.json>
// Replays what the flat-graph flow can do, from the core's own functions, and writes every
// step's positions for `renderGraphFlow.py` to turn into a side-by-side clip. Each scene is a
// COMPARISON on one fixture; the panels differ in exactly one thing:
//   drag-vs-move      12×12 grid, its corner 2×2 block wanted in the middle.
//                     left: the block is DRAGGED (the flow keeps the embedding, so the map is
//                     pulled along); right: it is MOVED THROUGH (carried, set down, grown back).
//   waiting-vs-room   radial tree of 150 nodes, a 40-node branch moved a third of a turn.
//                     left: the rest is pushed away by the flow's contact forces only;
//                     right: the rest is moved out of the way directly (`makeRoom`).
//   keep-vs-turn      the same tree and branch (rest moved out of the way in both).
//                     left: the branch keeps its direction; right: it turns while carried
//                     so that its root faces its parent.
//   keep-vs-hang      left: as before (it keeps its parent: a link across the map);
//                     right: it is HUNG on the node it is dropped next to and turned to face it.
//   sleeping          60×60 grid at rest, its middle node dragged 1 unit and then held.
//                     left: every node is stepped; right: only the nodes that are awake
//                     (the waking spreads from the drag and dies away again).
//   squeezed          18×18 grid with its middle third squeezed to a fifth, opening (one panel).
// Why a recorder and not a screen capture of the app: deterministic, no headless browser.
// @see src/core/medialGraphMove.ts, src/core/medialSphereGraph.ts
// @see docs/plans/2026-11-24-medial-sphere-method.md
import { writeFileSync } from 'node:fs';
import {
    beginGroupMove,
    countCrossings,
    dropGroupMove,
    type GroupGrowth,
    growGroup,
    hangGroupOn,
    nearestOutsideNode,
    prunePassPairs,
    setGroupOffset,
} from '../../src/core/medialGraphMove';
import {
    type MedialGraph,
    medialGraphFrom,
    medialGraphStep,
    minClearance,
} from '../../src/core/medialSphereGraph';
import type { Edge, Vec3 } from '../../src/core/testConfigs';

interface Frame {
    /** x0, y0, x1, y1, … of every point, rounded. */
    x: number[];
    label: string;
    /** Indices of the nodes that are asleep (the `sleeping` scene only). */
    asleep?: number[];
    /** Segment ends that differ from the panel's lists in this frame (a re-attached link):
     *  [0 = `a` / 1 = `b`, segment index, node]. */
    patch?: [number, number, number][];
}
interface Panel {
    title: string;
    a: number[];
    b: number[];
    /** Points below this index are the graph's own nodes (drawn as dots). */
    nodeCount: number;
    /** Nodes to draw highlighted (the group being moved, the node being dragged). */
    hi: number[];
    frames: Frame[];
}

function rng(seed: number): () => number {
    let s = seed;
    return () => {
        s = (1664525 * s + 1013904223) % 4294967296;
        return s / 4294967296;
    };
}
const snap = (g: MedialGraph): number[] => {
    const out = new Array<number>(2 * g.n);
    for (let i = 0; i < g.n; i++) {
        out[2 * i] = Math.round(g.x[3 * i] * 1000) / 1000;
        out[2 * i + 1] = Math.round(g.x[3 * i + 1] * 1000) / 1000;
    }
    return out;
};
const panelOf = (title: string, g: MedialGraph, hi: number[]): Panel => ({
    title,
    a: Array.from(g.segs.a),
    b: Array.from(g.segs.b),
    nodeCount: g.nodeCount,
    hi,
    frames: [],
});

function grid(size: number, seed = 7): MedialGraph {
    const rnd = rng(seed);
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
    return medialGraphFrom(V, E, 2);
}

/** The radial tree of `bench/app/medialGroupMove.ts` and its second-largest branch. */
function tree(target: number): { g: MedialGraph; blob: number[]; offset: [number, number] } {
    const rnd = rng(11);
    const parent = [-1];
    const kids: number[][] = [[]];
    const depth = [0];
    let frontier = [0];
    while (parent.length < target && frontier.length > 0) {
        const next: number[] = [];
        for (const p of frontier) {
            const c = p === 0 ? 6 : 1 + Math.floor(rnd() * 2.4);
            for (let k = 0; k < c && parent.length < target; k++) {
                parent.push(p);
                kids.push([]);
                depth.push(depth[p] + 1);
                kids[p].push(parent.length - 1);
                next.push(parent.length - 1);
            }
        }
        frontier = next;
    }
    const N = parent.length;
    const leaves = new Array<number>(N).fill(0);
    for (let i = N - 1; i >= 0; i--) {
        if (kids[i].length === 0) leaves[i] = 1;
        if (parent[i] >= 0) leaves[parent[i]] += leaves[i];
    }
    const maxDepth = Math.max(...depth);
    const ring = Math.max(1, (0.8 * leaves[0]) / (2 * Math.PI * maxDepth));
    const V: Vec3[] = new Array(N);
    const E: Edge[] = [];
    const lay = (i: number, a0: number, a1: number): void => {
        const a = 0.5 * (a0 + a1);
        V[i] = [depth[i] * ring * Math.cos(a), depth[i] * ring * Math.sin(a), 0];
        let at = a0;
        for (const c of kids[i]) {
            const span = ((a1 - a0) * leaves[c]) / leaves[i];
            E.push([i, c]);
            lay(c, at, at + span);
            at += span;
        }
    };
    lay(0, 0, 2 * Math.PI);
    const sub = (i: number): number[] => [i, ...kids[i].flatMap(sub)];
    const main = kids[0].slice().sort((a, b) => leaves[b] - leaves[a]);
    const blob = sub(main[1]);
    let cx = 0;
    let cy = 0;
    for (const i of blob) {
        cx += V[i][0] / blob.length;
        cy += V[i][1] / blob.length;
    }
    const c = Math.cos((2 * Math.PI) / 3);
    const s = Math.sin((2 * Math.PI) / 3);
    return {
        g: medialGraphFrom(V, E, 4),
        blob,
        offset: [c * cx - s * cy - cx, s * cx + c * cy - cy],
    };
}

/** A group move, recorded: carry (20 steps), set down, grow back, then rest. */
function recordMove(
    panel: Panel,
    g: MedialGraph,
    r: number,
    blob: number[],
    offset: [number, number],
    makeRoom: 'warp' | 'flow',
    frames: number,
    how: { turn?: 'none' | 'links'; hang?: boolean } = {},
): void {
    const move = beginGroupMove(g, blob);
    for (let k = 1; k <= 20; k++) {
        setGroupOffset(g, move, (offset[0] * k) / 20, (offset[1] * k) / 20, { turn: how.turn });
        medialGraphStep(g, { r });
        panel.frames.push({ x: snap(g), label: 'picked up: carried over the map' });
    }
    if (how.hang) {
        // Hung on the node nearest to where it is held.
        const to = nearestOutsideNode(g, move, move.cx + move.dx, move.cy + move.dy);
        hangGroupOn(g, move, to);
    }
    // The re-attached link's two ends, as they are now (they change once more when it lands).
    const hungEnds = (): [number, number, number][] | undefined => {
        const e = move.hung;
        if (e === undefined) return undefined;
        const first = e * g.pieces;
        const last = first + g.pieces - 1;
        return [
            [0, first, g.segs.a[first]],
            [1, last, g.segs.b[last]],
        ];
    };
    const drop = dropGroupMove(g, move, { r });
    let growth: GroupGrowth | null = drop.growth;
    let landedAt: number | null = growth ? null : 0;
    let crossings = growth ? 0 : countCrossings(g);
    if (!growth) for (const i of blob) g.pinned[i] = 1;
    for (let k = 1; panel.frames.length < frames; k++) {
        if (growth) {
            const scale = growGroup(g, growth, r, { makeRoom });
            if (scale >= 1) {
                growth = null;
                landedAt = k;
                for (const i of blob) g.pinned[i] = 1;
                crossings = countCrossings(g);
            }
        }
        medialGraphStep(g, { r });
        prunePassPairs(g, 0.05 * r);
        const label = growth
            ? `step ${k}: set down small, growing back, ${Math.round(100 * growth.scale)}% of its size`
            : `full size after ${landedAt} steps; links landed, ${crossings} crossings, held`;
        panel.frames.push({ x: snap(g), label, patch: hungEnds() });
    }
}

const [, , scene, out] = process.argv;
const panels: Panel[] = [];
if (scene === 'drag-vs-move') {
    const r = 0.4;
    const size = 12;
    const blob = [0, 1, size, size + 1];
    const offset: [number, number] = [5, 5];
    const FRAMES = 300;
    {
        const g = grid(size);
        const p = panelOf(
            'dragged: the flow keeps the embedding, the map is pulled along',
            g,
            blob,
        );
        const home = blob.map((i) => [g.x[3 * i], g.x[3 * i + 1]] as [number, number]);
        for (let k = 1; k <= FRAMES; k++) {
            const f = Math.min(1, k / 40);
            const targets = new Map<number, [number, number]>();
            blob.forEach((i, q) => {
                targets.set(i, [home[q][0] + f * offset[0], home[q][1] + f * offset[1]]);
            });
            medialGraphStep(g, { r, pinTargets: targets });
            let away = 0;
            blob.forEach((i, q) => {
                away = Math.max(
                    away,
                    Math.hypot(
                        g.x[3 * i] - home[q][0] - offset[0],
                        g.x[3 * i + 1] - home[q][1] - offset[1],
                    ),
                );
            });
            p.frames.push({
                x: snap(g),
                label: `step ${k}: pulled towards the middle, still ${away.toFixed(1)} away`,
            });
        }
        panels.push(p);
    }
    {
        const g = grid(size);
        const p = panelOf('moved through: carried, set down small, the rest makes room', g, blob);
        recordMove(p, g, r, blob, offset, 'warp', FRAMES);
        panels.push(p);
    }
} else if (scene === 'waiting-vs-room') {
    const r = 0.25;
    const FRAMES = 330;
    for (const mode of ['flow', 'warp'] as const) {
        const { g, blob, offset } = tree(150);
        for (let k = 0; k < 100; k++) medialGraphStep(g, { r });
        const p = panelOf(
            mode === 'flow'
                ? 'waiting: the rest is pushed away by contact forces only'
                : 'making room: the rest is moved out of the way directly',
            g,
            blob,
        );
        recordMove(p, g, r, blob, offset, mode, FRAMES);
        panels.push(p);
    }
} else if (scene === 'keep-vs-turn' || scene === 'keep-vs-hang') {
    const r = 0.25;
    const FRAMES = 240;
    const hang = scene === 'keep-vs-hang';
    for (const changed of [false, true]) {
        const { g, blob, offset } = tree(150);
        for (let k = 0; k < 100; k++) medialGraphStep(g, { r });
        const p = panelOf(
            !changed
                ? 'as before: it keeps its direction and its parent'
                : hang
                  ? 'hung on the node it is dropped next to, turned to face it'
                  : 'turned while carried, to face its parent',
            g,
            blob,
        );
        recordMove(p, g, r, blob, offset, 'warp', FRAMES, changed ? { turn: 'links', hang } : {});
        panels.push(p);
    }
} else if (scene === 'sleeping') {
    const r = 0.4;
    const size = 60;
    for (const sleep of [false, true]) {
        const g = grid(size);
        // No tension, so that the grid comes to rest (see bench/app/medialLocalDrag.ts).
        const opts = sleep ? { r, tension: 0, sleep: {} } : { r, tension: 0 };
        for (let k = 0; k < 100; k++) medialGraphStep(g, opts);
        const mid = Math.floor(size / 2) * size + Math.floor(size / 2);
        const from: [number, number] = [g.x[3 * mid], g.x[3 * mid + 1]];
        const p = panelOf(
            sleep ? 'sleeping: only what moves is stepped' : 'all awake: every node, every step',
            g,
            [mid],
        );
        let ms = 0;
        for (let k = 1; k <= 420; k++) {
            const f = Math.min(1, k / 50);
            const target = new Map<number, [number, number]>([
                [mid, [from[0] + 0.7 * f, from[1] + 0.7 * f]],
            ]);
            const t = performance.now();
            const st = medialGraphStep(g, { ...opts, pinTargets: target });
            ms += performance.now() - t;
            const asleep: number[] = [];
            const s = g.sleepState;
            if (sleep && s) for (let i = 0; i < g.n; i++) if (s.asleep[i]) asleep.push(i);
            p.frames.push({
                x: snap(g),
                asleep,
                label: `step ${k}: ${st.awakeNodes} of ${g.n} nodes stepped, ${(ms / k).toFixed(0)} ms per step so far`,
            });
        }
        panels.push(p);
    }
} else if (scene === 'squeezed') {
    const r = 0.4;
    const size = 18;
    const clean = grid(size);
    const rest = Float64Array.from(clean.rest);
    const lo = (size - 1) / 3;
    const hi = (2 * (size - 1)) / 3;
    const f = (y: number): number =>
        y <= lo ? y : y <= hi ? lo + 0.2 * (y - lo) : lo + 0.2 * (hi - lo) + (y - hi);
    const g = clean;
    for (let i = 0; i < g.n; i++) g.x[3 * i + 1] = f(g.x[3 * i + 1]);
    g.rest.set(rest); // rest lengths of the un-squeezed grid: it wants to open
    const p = panelOf('a grid squeezed to a fifth in the middle opens; nothing crosses', g, []);
    let clearance = minClearance(g, 2);
    p.frames.push({ x: snap(g), label: `start: tightest gap ${clearance.toFixed(2)}` });
    for (let k = 1; k <= 900; k++) {
        medialGraphStep(g, { r });
        if (k % 3 !== 0) continue;
        if (k % 15 === 0) clearance = minClearance(g, 2);
        p.frames.push({
            x: snap(g),
            label: `step ${k}: tightest gap ${clearance.toFixed(2)}, crossings ${countCrossings(g)}`,
        });
    }
    panels.push(p);
} else {
    console.error(
        'usage: bun bench/app/recordGraphFlow.ts <drag-vs-move|waiting-vs-room|keep-vs-turn|keep-vs-hang|sleeping|squeezed> <out.json>',
    );
    process.exit(2);
}
writeFileSync(out, JSON.stringify({ scene, panels }));
console.log(`${scene}: ${panels.length} panel(s), ${panels[0].frames.length} frames -> ${out}`);
