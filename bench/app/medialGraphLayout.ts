/**
 * bench/app/medialGraphLayout.ts — bun bench/app/medialGraphLayout.ts [sizes=8,18,36] [maxSteps=300] [budgetSec=120] [tpSteps=10] [tpMaxNodes=6000] [pieces=8]
 *
 * The flat-graph spike of docs/plans/2026-11-24-medial-sphere-method.md: the medial-sphere
 * graph flow (src/core/medialSphereGraph.ts) against the tangent-point flow with the crossing
 * guard (`dispatchDescentStep`, `preventCrossings`), on the SAME subdivided graphs.
 *
 * Fixture "squeezed grid": a jittered g×g lattice (spacing 1, crossing-free) whose middle third
 * is squeezed vertically to a fifth, so rows of edges lie 0.2 apart — a drawing that must make
 * room. Every edge is cut into 8 segments (h = 1/8). g = 8, 18, 36 gives about 0.9k, 4.9k and
 * 20k segments.
 * Measured per method: time per step, crossings before/after (must stay 0), and the smallest
 * distance between two segments that share no node, before/after.
 * The two flows minimise different things (springs + local clearance vs the tangent-point
 * energy at fixed total length), so "who spreads further" is not comparable; time per step,
 * topology held and clearance gained per second are.
 * One JSON line per (size, method) is appended to bench/results/<date>-medial-graph-layout.jsonl.
 */
import { appendFileSync } from 'node:fs';
import { dispatchDescentStep } from '../../src/core/dispatch';
import {
    type MedialGraph,
    medialGraphFrom,
    medialGraphStep,
    minClearance,
} from '../../src/core/medialSphereGraph';
import { edgeLengths, totalLength } from '../../src/core/sobolev/constraintSet';
import { barycenterTarget } from '../../src/core/sobolev/constraints';
import { pairListFits } from '../../src/core/sobolevRoute';
import { countPlanarCrossings } from '../../src/core/subdivide';
import { calculateDisjointPairs } from '../../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../../src/core/testConfigs';

const sizes = (process.argv[2] ?? '8,18,36').split(',').map(Number);
const maxSteps = Number(process.argv[3] ?? 300);
const budgetMs = 1000 * Number(process.argv[4] ?? 120);
const tpSteps = Number(process.argv[5] ?? 10);
const tpMaxNodes = Number(process.argv[6] ?? 6000);
// Segments per edge. 8 was needed while probes tested node against node; with segment probes
// fewer will do, and the cost is proportional. Sixth argument, default 8.
const PIECES = Number(process.argv[7] ?? 8);
const R = 0.4;
const OUT = `bench/results/${new Date().toISOString().slice(0, 10)}-medial-graph-layout.jsonl`;

function rng(seed: number): () => number {
    let s = seed;
    return () => {
        s = (1664525 * s + 1013904223) % 4294967296;
        return s / 4294967296;
    };
}
function gridEdges(g: number): Edge[] {
    const E: Edge[] = [];
    for (let y = 0; y < g; y++)
        for (let x = 0; x < g; x++) {
            if (x + 1 < g) E.push([y * g + x, y * g + x + 1]);
            if (y + 1 < g) E.push([y * g + x, (y + 1) * g + x]);
        }
    return E;
}
function lattice(g: number, seed: number): Vec3[] {
    const r = rng(seed);
    const V: Vec3[] = [];
    for (let y = 0; y < g; y++)
        for (let x = 0; x < g; x++) V.push([x + 0.3 * (r() - 0.5), y + 0.3 * (r() - 0.5), 0]);
    return V;
}
/** Squeeze the middle third of the drawing vertically to a fifth (a continuous map of y). */
function squeeze(V: Vec3[], g: number): Vec3[] {
    const lo = (g - 1) / 3;
    const hi = (2 * (g - 1)) / 3;
    const f = (y: number): number =>
        y <= lo ? y : y <= hi ? lo + 0.2 * (y - lo) : lo + 0.2 * (hi - lo) + (y - hi);
    return V.map((p) => [p[0], f(p[1]), 0] as Vec3);
}
const toVec3 = (g: MedialGraph): Vec3[] =>
    Array.from({ length: g.n }, (_, i) => [g.x[3 * i], g.x[3 * i + 1], 0] as Vec3);
const edgesOf = (g: MedialGraph): Edge[] =>
    Array.from(g.segs.a, (a, k) => [a, g.segs.b[k]] as Edge);

for (const size of sizes) {
    const E = gridEdges(size);
    const clean = lattice(size, 7);
    const rest = medialGraphFrom(clean, E, PIECES).rest;
    const start = medialGraphFrom(squeeze(clean, size), E, PIECES);
    start.rest = rest;
    const segEdges = edgesOf(start);
    const base = {
        fixture: `squeezed ${size}x${size} grid`,
        pieces: PIECES,
        nodes: start.n,
        segments: segEdges.length,
        crossingsStart: countPlanarCrossings(toVec3(start), segEdges, PIECES),
        clearanceStart: +minClearance(start, 2).toFixed(4),
    };

    // --- medial-sphere graph flow ---------------------------------------------------------------
    {
        const g: MedialGraph = { ...start, x: Float64Array.from(start.x) };
        let steps = 0;
        let stopped = 'steps';
        let ms = 0;
        let cgMax = 0;
        let limited = 0;
        let move = Number.NaN;
        const clearanceAt: Record<number, number> = {};
        while (steps < maxSteps) {
            if (ms > budgetMs) {
                stopped = 'time';
                break;
            }
            const t = performance.now();
            const r = medialGraphStep(g, { r: R });
            ms += performance.now() - t;
            steps++;
            cgMax = Math.max(cgMax, r.cgIterations);
            limited += r.cappedNodes;
            move = r.movePerNode;
            if (steps === tpSteps || steps === 50)
                clearanceAt[steps] = +minClearance(g, 2).toFixed(4);
            if (move < 3e-4) {
                stopped = 'converged';
                break;
            }
        }
        const row = {
            ...base,
            method: 'medial-sphere graph',
            steps,
            stopped,
            msPerStep: +(ms / Math.max(steps, 1)).toFixed(2),
            seconds: +(ms / 1000).toFixed(2),
            crossingsEnd: countPlanarCrossings(toVec3(g), segEdges, PIECES),
            clearanceEnd: +minClearance(g, 2).toFixed(4),
            clearanceAt,
            movePerNode: +move.toExponential(2),
            cgMax,
            cappedNodeSteps: limited,
        };
        appendFileSync(OUT, `${JSON.stringify(row)}\n`);
        console.log(JSON.stringify(row));
    }

    // --- tangent-point flow with the crossing guard --------------------------------------------
    // Skipped above `tpMaxNodes`: one CPU step there is minutes (all-pairs gradient), and a
    // row that only says "too slow" is recorded instead.
    if (start.n > tpMaxNodes) {
        const row = {
            ...base,
            method: 'tangent-point + guard',
            steps: 0,
            stopped: `skipped: ${start.n} nodes > ${tpMaxNodes}`,
            msPerStep: null,
            seconds: 0,
            crossingsEnd: base.crossingsStart,
            clearanceEnd: base.clearanceStart,
        };
        appendFileSync(OUT, `${JSON.stringify(row)}\n`);
        console.log(JSON.stringify(row));
    } else {
        let V = toVec3(start);
        const pairs = pairListFits(V.length) ? calculateDisjointPairs(segEdges) : [];
        const args = {
            descentMode: 'sobolev' as const,
            edges: segEdges,
            disjointPairs: pairs,
            mode: 'analytical' as const,
            stepSize: 0.001,
            x0: barycenterTarget(V, segEdges),
            sobolevL0: totalLength(V, segEdges),
            sobolevEll0: edgeLengths(V, segEdges),
            barycenterConstraint: true,
            lengthMode: 'total' as const,
            projectionMode: 'frozen' as const,
            sobolevSolver: 'auto' as const,
            preventCrossings: true,
        };
        let energy: number | undefined;
        let steps = 0;
        let stopped = 'steps';
        let ms = 0;
        while (steps < tpSteps) {
            if (ms > budgetMs) {
                stopped = 'time';
                break;
            }
            const t = performance.now();
            let r: ReturnType<typeof dispatchDescentStep>;
            try {
                r = dispatchDescentStep({
                    ...args,
                    vertices: V,
                    ...(energy === undefined ? {} : { energyBefore: energy }),
                });
            } catch (err) {
                stopped = `threw: ${String(err).slice(0, 80)}`;
                break;
            }
            ms += performance.now() - t;
            if (!r.accepted) {
                stopped = r.converged ? 'converged' : (r.stats?.reason ?? 'rejected');
                break;
            }
            V = r.vertices;
            energy = r.energy;
            steps++;
        }
        const end: MedialGraph = { ...start, x: Float64Array.from(V.flat()) };
        const row = {
            ...base,
            method: 'tangent-point + guard',
            steps,
            stopped,
            msPerStep: steps > 0 ? +(ms / steps).toFixed(1) : null,
            seconds: +(ms / 1000).toFixed(2),
            crossingsEnd: countPlanarCrossings(V, segEdges, PIECES),
            clearanceEnd: +minClearance(end, 2).toFixed(4),
        };
        appendFileSync(OUT, `${JSON.stringify(row)}\n`);
        console.log(JSON.stringify(row));
    }
}
