/**
 * bench/app/forcePrePass.ts — bun bench/app/forcePrePass.ts [maxSteps=150] [maxSeconds=90] [recordDir]
 *
 * Does a conventional force-directed pre-pass (src/core/forceLayout.ts) make the tangent-point
 * flow untangle a GRAPH faster? Two arms over the same seeded fixtures, both stepping with the
 * app's own `dispatchDescentStep` (Sobolev descent, barycenter + total length):
 *   flow        the flow from the scrambled input
 *   force+flow  force layout until it settles (rescaled to the input's total length, so both
 *               arms carry the same length constraint and their energies are comparable),
 *               then the flow
 * Reported per fixture: energy at the start, after the pre-pass, at the end of each arm; steps
 * and seconds; seconds for each arm to first reach the HIGHER of the two final energies (+2%),
 * i.e. "time to the same energy"; and, for planar fixtures, the number of edge crossings.
 * One JSON line per (fixture, arm) is appended to bench/results/<date>-force-prepass.jsonl.
 * With `recordDir`, nothing is appended; instead every arm's trajectory is written there as
 * `<fixture index>-<arm>.json` in the format bench/app/renderFlow.py draws (pre-pass frames
 * every 25 ticks, then one frame per descent step).
 * Why a bench and not an argument: the pre-pass passes edges through each other, so whether it
 * helps or hands the flow a worse start is an empirical question per kind of graph.
 */
import { appendFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { dispatchDescentStep } from '../../src/core/dispatch';
import {
    createForceLayout,
    forceLayoutSettled,
    forceLayoutTick,
    rescaleForceLayout,
    writeForceLayout,
} from '../../src/core/forceLayout';
import { edgeLengths, totalLength } from '../../src/core/sobolev/constraintSet';
import { barycenterTarget } from '../../src/core/sobolev/constraints';
import { calculateDisjointPairs, calculateEnergy } from '../../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../../src/core/testConfigs';

const maxSteps = Number(process.argv[2] ?? 150);
const maxSeconds = Number(process.argv[3] ?? 90);
const recordDir = process.argv[4];
const OUT = `bench/results/${new Date().toISOString().slice(0, 10)}-force-prepass.jsonl`;
if (recordDir) mkdirSync(recordDir, { recursive: true });

function rng(seed: number): () => number {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}
const inBall = (r: () => number, R: number): Vec3 => {
    for (;;) {
        const p: Vec3 = [2 * r() - 1, 2 * r() - 1, 2 * r() - 1];
        if (p[0] ** 2 + p[1] ** 2 + p[2] ** 2 <= 1) return [p[0] * R, p[1] * R, p[2] * R];
    }
};

interface Fixture {
    name: string;
    vertices: Vec3[];
    edges: Edge[];
    planar: boolean;
}

/** A connected random graph: a random spanning tree plus extra random edges, random positions. */
function randomGraph(n: number, m: number, seed: number): Fixture {
    const r = rng(seed);
    const vertices = Array.from({ length: n }, () => inBall(r, 2));
    const edges: Edge[] = [];
    const seen = new Set<string>();
    const add = (a: number, b: number) => {
        const key = a < b ? `${a}-${b}` : `${b}-${a}`;
        if (a === b || seen.has(key)) return;
        seen.add(key);
        edges.push([a, b]);
    };
    for (let i = 1; i < n; i++) add(i, Math.floor(r() * i));
    while (edges.length < m) add(Math.floor(r() * n), Math.floor(r() * n));
    return { name: `random graph, ${n} nodes / ${m} edges, 3-D`, vertices, edges, planar: false };
}
/** A g×g grid graph whose nodes are thrown at random into a square in the plane z = 0. */
function scrambledGrid2D(g: number, seed: number): Fixture {
    const r = rng(seed);
    const vertices: Vec3[] = [];
    const edges: Edge[] = [];
    for (let i = 0; i < g * g; i++) vertices.push([4 * r() - 2, 4 * r() - 2, 0]);
    for (let y = 0; y < g; y++)
        for (let x = 0; x < g; x++) {
            if (x + 1 < g) edges.push([y * g + x, y * g + x + 1]);
            if (y + 1 < g) edges.push([y * g + x, (y + 1) * g + x]);
        }
    return { name: `${g}x${g} grid, scrambled in the plane`, vertices, edges, planar: true };
}
/** A g×g×g grid graph whose nodes are thrown at random into a ball. */
function scrambledGrid3D(g: number, seed: number): Fixture {
    const r = rng(seed);
    const vertices: Vec3[] = [];
    const edges: Edge[] = [];
    const id = (x: number, y: number, z: number) => (z * g + y) * g + x;
    for (let i = 0; i < g ** 3; i++) vertices.push(inBall(r, 2));
    for (let z = 0; z < g; z++)
        for (let y = 0; y < g; y++)
            for (let x = 0; x < g; x++) {
                if (x + 1 < g) edges.push([id(x, y, z), id(x + 1, y, z)]);
                if (y + 1 < g) edges.push([id(x, y, z), id(x, y + 1, z)]);
                if (z + 1 < g) edges.push([id(x, y, z), id(x, y, z + 1)]);
            }
    return { name: `${g}x${g}x${g} grid, scrambled in a ball`, vertices, edges, planar: false };
}
/** A binary tree with random node positions in the plane. */
function scrambledTree2D(n: number, seed: number): Fixture {
    const r = rng(seed);
    const vertices: Vec3[] = Array.from({ length: n }, () => [4 * r() - 2, 4 * r() - 2, 0]);
    const edges: Edge[] = [];
    for (let i = 1; i < n; i++) edges.push([Math.floor((i - 1) / 2), i]);
    return {
        name: `binary tree, ${n} nodes, scrambled in the plane`,
        vertices,
        edges,
        planar: true,
    };
}

/** Proper crossings between edges that share no endpoint, in the xy-plane. */
function crossings(V: Vec3[], edges: Edge[]): number {
    const o = (a: Vec3, b: Vec3, c: Vec3) =>
        Math.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]));
    let count = 0;
    for (let i = 0; i < edges.length; i++)
        for (let j = i + 1; j < edges.length; j++) {
            const [a, b] = edges[i];
            const [c, d] = edges[j];
            if (a === c || a === d || b === c || b === d) continue;
            if (
                o(V[a], V[b], V[c]) * o(V[a], V[b], V[d]) < 0 &&
                o(V[c], V[d], V[a]) * o(V[c], V[d], V[b]) < 0
            )
                count++;
        }
    return count;
}

interface Trace {
    t: number[];
    e: number[];
    steps: number;
    stopped: string;
    vertices: Vec3[];
    /** Every iterate, kept only when recording. */
    path: Vec3[][];
}
function flow(V0: Vec3[], edges: Edge[], pairs: number[][], tOffset: number): Trace {
    const exact = (V: Vec3[]) => calculateEnergy(V, edges, pairs, 3, 6, 1e-10);
    let V = V0;
    let energy: number | undefined;
    let elapsed = tOffset;
    const t = [elapsed];
    const e = [exact(V)];
    let stopped = 'steps';
    let steps = 0;
    const path: Vec3[][] = [V0];
    const base = {
        descentMode: 'sobolev' as const,
        edges,
        disjointPairs: pairs,
        mode: 'analytical' as const,
        stepSize: 0.001,
        x0: barycenterTarget(V0, edges),
        sobolevL0: totalLength(V0, edges),
        sobolevEll0: edgeLengths(V0, edges),
        barycenterConstraint: true,
        lengthMode: 'total' as const,
        projectionMode: 'frozen' as const,
    };
    while (steps < maxSteps && elapsed - tOffset < maxSeconds * 1000) {
        const t0 = performance.now();
        const r = dispatchDescentStep({
            ...base,
            vertices: V,
            ...(energy === undefined ? {} : { energyBefore: energy }),
        });
        elapsed += performance.now() - t0;
        if (!r.accepted) {
            stopped = r.converged ? 'converged' : (r.stats?.reason ?? 'rejected');
            break;
        }
        V = r.vertices;
        energy = r.energy;
        steps++;
        t.push(elapsed);
        e.push(r.energy);
        if (recordDir) path.push(V);
    }
    if (stopped === 'steps' && steps < maxSteps) stopped = 'time';
    return { t, e, steps, stopped, vertices: V, path };
}
const firstBelow = (tr: Trace, target: number): number | null => {
    const k = tr.e.findIndex((x) => x <= target);
    return k < 0 ? null : tr.t[k] / 1000;
};

const fixtures: Fixture[] = [
    scrambledGrid2D(6, 1),
    scrambledGrid2D(10, 2),
    scrambledTree2D(63, 3),
    randomGraph(30, 50, 4),
    randomGraph(100, 150, 5),
    scrambledGrid3D(4, 6),
    scrambledGrid3D(6, 7),
];
for (const f of fixtures) {
    const pairs = calculateDisjointPairs(f.edges);
    // The pre-pass, exactly as `runForceLayout` does it, with the low-level calls so that a
    // recording can keep intermediate layouts (kept only when recording: it costs time).
    const exactE = (V: Vec3[]) => calculateEnergy(V, f.edges, pairs, 3, 6, 1e-10);
    const snapshot = (): Vec3[] => {
        const out: Vec3[] = f.vertices.map(() => [0, 0, 0]);
        writeForceLayout(layout, out);
        return out;
    };
    const t0 = performance.now();
    const layout = createForceLayout(f.vertices, f.edges);
    const preFrames: { tick: number; vertices: Vec3[] }[] = [];
    while (!forceLayoutSettled(layout)) {
        forceLayoutTick(layout);
        if (recordDir && layout.ticks % 25 === 0 && layout.ticks < 300)
            preFrames.push({ tick: layout.ticks, vertices: snapshot() });
    }
    rescaleForceLayout(layout, totalLength(f.vertices, f.edges));
    const pre = { vertices: snapshot(), ticks: layout.ticks };
    const preMs = performance.now() - t0;
    const a = flow(f.vertices, f.edges, pairs, 0);
    const b = flow(pre.vertices, f.edges, pairs, preMs);
    const target = Math.max(a.e[a.e.length - 1], b.e[b.e.length - 1]) * 1.02;
    const row = (arm: string, tr: Trace) => ({
        fixture: f.name,
        nodes: f.vertices.length,
        edges: f.edges.length,
        arm,
        prePassMs: arm === 'flow' ? 0 : preMs,
        prePassTicks: arm === 'flow' ? 0 : pre.ticks,
        energyStart: a.e[0],
        energyAtFlowStart: tr.e[0],
        energyEnd: tr.e[tr.e.length - 1],
        steps: tr.steps,
        stopped: tr.stopped,
        seconds: tr.t[tr.t.length - 1] / 1000,
        secondsToCommonEnergy: firstBelow(tr, target),
        commonEnergy: target,
        crossingsStart: f.planar
            ? crossings(arm === 'flow' ? f.vertices : pre.vertices, f.edges)
            : null,
        crossingsEnd: f.planar ? crossings(tr.vertices, f.edges) : null,
        maxSteps,
        maxSeconds,
    });
    for (const [arm, tr] of [
        ['flow', a],
        ['force+flow', b],
    ] as const) {
        const r = row(arm, tr);
        if (recordDir) {
            const flat = (V: Vec3[]) => V.flatMap((p) => p.map((c) => Number(c.toFixed(5))));
            const frames = [
                ...(arm === 'flow'
                    ? []
                    : [
                          { step: 0, phase: 'start', energy: a.e[0], ms: 0, xyz: flat(f.vertices) },
                          ...preFrames.map((p) => ({
                              step: 0,
                              phase: `force layout, tick ${p.tick}`,
                              energy: exactE(p.vertices),
                              ms: (preMs * p.tick) / pre.ticks,
                              xyz: flat(p.vertices),
                          })),
                      ]),
                ...tr.path.map((V, k) => ({
                    step: k,
                    phase: 'tangent-point flow',
                    energy: tr.e[k],
                    ms: tr.t[k],
                    xyz: flat(V),
                })),
            ];
            writeFileSync(
                `${recordDir}/${fixtures.indexOf(f)}-${arm === 'flow' ? 'flow' : 'force-flow'}.json`,
                JSON.stringify({
                    source: f.name,
                    n: f.vertices.length,
                    edges: f.edges,
                    route: 'dense',
                    arm,
                    frames,
                }),
            );
        } else appendFileSync(OUT, `${JSON.stringify(r)}\n`);
        console.log(
            `${f.name} | ${arm.padEnd(10)} | E ${r.energyAtFlowStart.toPrecision(4)} -> ${r.energyEnd.toPrecision(4)} | ${r.steps} steps, ${r.stopped}, ${r.seconds.toFixed(1)} s | to common E: ${r.secondsToCommonEnergy?.toFixed(1) ?? 'never'} s${f.planar ? ` | crossings ${r.crossingsStart} -> ${r.crossingsEnd}` : ''}`,
        );
    }
}
