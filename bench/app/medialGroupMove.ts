// bench/app/medialGroupMove.ts — bun bench/app/medialGroupMove.ts [fixtures=grid12,grid36,tree150,tree600] [steps=600] [out] [makeRoom=warp|flow]
// The mind-map move: carry a group of nodes across a flat drawing, set it down, let the flow
// make room. One JSON line per fixture: how many crossings the move leaves (and that the
// number never grows afterwards), how many steps the group needs to grow back to full size
// where it was put, and the cost.
//   gridN    jittered N×N grid, 2 segments per edge; the corner 2×2 square goes to the middle.
//   treeN    a radial tree of about N nodes (a mind map), 4 segments per edge; the second
//            largest main branch is carried a third of a turn round the root, on top of others.
// @see src/core/medialGraphMove.ts, docs/plans/2026-11-24-medial-sphere-method.md
import { appendFileSync, writeFileSync } from 'node:fs';
import {
    allowedCrossingsNow,
    beginGroupMove,
    countCrossings,
    dropGroupMove,
    growGroup,
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

const fixtures = (process.argv[2] ?? 'grid12,grid36,tree150,tree600').split(',');
const STEPS = Number(process.argv[3] ?? 600);
const OUT = process.argv[4] ?? 'bench/results/2026-11-24-group-move.jsonl';
// How the rest makes room while the group grows back (see `growGroup`): both are kept so the
// two can be compared on the same fixtures.
const MAKE_ROOM = (process.argv[5] ?? 'warp') as 'warp' | 'flow';
// DUMP=<prefix> writes <prefix>-<fixture>-{0,1,2}.json (before, at the set-down, at the end):
// {x, a, b, nodeCount, hi} for a drawing script.
const DUMP = process.env.DUMP;

function rng(seed: number): () => number {
    let s = seed;
    return () => {
        s = (1664525 * s + 1013904223) % 4294967296;
        return s / 4294967296;
    };
}

interface Fixture {
    g: MedialGraph;
    r: number;
    blob: number[];
    offset: [number, number];
    settle: number;
}

function gridFixture(size: number): Fixture {
    const rnd = rng(7);
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
    const to = Math.floor(size / 2) - 1;
    return {
        g: medialGraphFrom(V, E, 2),
        r: 0.4,
        blob: [0, 1, size, size + 1],
        offset: [to, to],
        settle: 0,
    };
}

function treeFixture(target: number): Fixture {
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
    // Radial tidy layout: a wedge per subtree, as wide as its share of the leaves; rings far
    // enough apart that the outermost leaves are about 0.8 apart.
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
    // A third of a turn round the root (a translation: the branch keeps its orientation).
    const c = Math.cos((2 * Math.PI) / 3);
    const s = Math.sin((2 * Math.PI) / 3);
    return {
        g: medialGraphFrom(V, E, 4),
        r: 0.25,
        blob,
        offset: [c * cx - s * cy - cx, s * cx + c * cy - cy],
        settle: 100,
    };
}

for (const name of fixtures) {
    const size = Number(name.replace(/^[a-z]+/, ''));
    const fx = name.startsWith('grid') ? gridFixture(size) : treeFixture(size);
    const { g, r, blob } = fx;
    let t = performance.now();
    for (let k = 0; k < fx.settle; k++) medialGraphStep(g, { r });
    // Cost of a step before the move, for comparison.
    t = performance.now();
    for (let k = 0; k < 10; k++) medialGraphStep(g, { r });
    const msBefore = (performance.now() - t) / 10;
    const crossingsBefore = countCrossings(g);
    const clearanceBefore = minClearance(g, 2 * r);

    const dump = (stage: number): void => {
        if (!DUMP) return;
        writeFileSync(
            `${DUMP}-${name}-${stage}.json`,
            JSON.stringify({
                x: Array.from(g.x),
                a: Array.from(g.segs.a),
                b: Array.from(g.segs.b),
                nodeCount: g.nodeCount,
                hi: blob,
            }),
        );
    };
    dump(0);
    const move = beginGroupMove(g, blob);
    // How far the REST of the drawing is from its own shape: mean |length / rest length − 1|
    // over the segments of edges that do not touch the group. The price of making room.
    const touched = new Uint8Array(g.edgeA.length);
    for (const e of move.internal) touched[e] = 1;
    for (const e of move.connecting) touched[e] = 1;
    const restStrain = (): number => {
        let sum = 0;
        let count = 0;
        for (let k = 0; k < g.segs.a.length; k++) {
            if (touched[Math.floor(k / g.pieces)]) continue;
            const a = g.segs.a[k];
            const b = g.segs.b[k];
            const len = Math.hypot(g.x[3 * b] - g.x[3 * a], g.x[3 * b + 1] - g.x[3 * a + 1]);
            sum += Math.abs(len / g.rest[k] - 1);
            count++;
        }
        return sum / Math.max(1, count);
    };
    const strainBefore = restStrain();
    let strainAtRelease: number | null = null;
    for (let k = 1; k <= 20; k++) {
        setGroupOffset(g, move, (fx.offset[0] * k) / 20, (fx.offset[1] * k) / 20);
        medialGraphStep(g, { r });
    }
    const drop = dropGroupMove(g, move, { r });
    dump(1);
    let crossingsAtRelease: number | null = drop.growth ? null : countCrossings(g);
    if (!drop.growth) for (const i of blob) g.pinned[i] = 1;
    let worst = crossingsAtRelease ?? 0;
    let ms = 0;
    let movePerNode = 0;
    let limited = 0;
    let cgMax = 0;
    let stepsToFullSize: number | null = drop.growth ? null : 0;
    let scale = drop.scale;
    for (let k = 0; k < STEPS; k++) {
        t = performance.now();
        if (drop.growth && stepsToFullSize === null) {
            scale = growGroup(g, drop.growth, r, { makeRoom: MAKE_ROOM });
            if (scale >= 1) {
                stepsToFullSize = k + 1;
                for (const i of blob) g.pinned[i] = 1; // held where it was put, as the app does
                crossingsAtRelease = countCrossings(g);
                worst = crossingsAtRelease;
                strainAtRelease = restStrain();
            }
        }
        const st = medialGraphStep(g, { r });
        prunePassPairs(g, 0.05 * r);
        ms += performance.now() - t;
        movePerNode = st.movePerNode;
        limited = st.cappedNodes;
        cgMax = Math.max(cgMax, st.cgIterations);
        // Every 5 steps the count among the pairs allowed to cross (cheap, and where any
        // new crossing would have to be); the independent all-pairs count is taken at the
        // release and at the end.
        if (k % 5 === 4) worst = Math.max(worst, allowedCrossingsNow(g));
        if (k === STEPS - 1) worst = Math.max(worst, countCrossings(g));
    }
    const away = (): number =>
        Math.max(
            ...blob.map((i) => {
                const to = drop.targets.get(i) as [number, number];
                return Math.hypot(g.x[3 * i] - to[0], g.x[3 * i + 1] - to[1]);
            }),
        );
    dump(2);
    const row = {
        bench: 'medial-group-move',
        fixture: name,
        nodes: g.nodeCount,
        segments: g.segs.a.length,
        r,
        groupNodes: blob.length,
        links: move.connecting.length,
        makeRoom: MAKE_ROOM,
        restStrainBefore: Number(strainBefore.toFixed(3)),
        restStrainAtRelease: strainAtRelease === null ? null : Number(strainAtRelease.toFixed(3)),
        restStrainEnd: Number(restStrain().toFixed(3)),
        setDownScale: Number(drop.scale.toFixed(3)),
        crossingsBefore,
        crossingsAtRelease,
        crossingsWorst: worst,
        crossingsEnd: countCrossings(g),
        scaleEnd: Number(scale.toFixed(3)),
        stepsToFullSize,
        groupFromTargetEnd: Number(away().toFixed(3)),
        clearanceBefore: Number(clearanceBefore.toFixed(3)),
        clearanceEnd: Number(minClearance(g, 2 * r).toFixed(3)),
        steps: STEPS,
        movePerNodeEnd: Number(movePerNode.toExponential(2)),
        limitedEnd: limited,
        cgMax,
        msPerStepBefore: Number(msBefore.toFixed(2)),
        msPerStepAfter: Number((ms / STEPS).toFixed(2)),
        finite: g.x.every(Number.isFinite),
    };
    console.log(JSON.stringify(row));
    appendFileSync(OUT, `${JSON.stringify(row)}\n`);
}
