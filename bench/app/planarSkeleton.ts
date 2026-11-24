// bench/app/planarSkeleton.ts — bun bench/app/planarSkeleton.ts [dir] [flowSteps=40] [out] [drawings=both|tidy]
// What to cut in a note vault that cannot be drawn without crossings: for every vault in
// `dir` (local_files/pkm-graphs/graphs/, see scripts/pkmExtract.ts), every way of choosing
// the tree (`TreeChoice`) and both tree drawings (tidy tree, force layout of the tree):
//   - how many links the crossing-free drawing keeps: the tree's, plus the ones taken back in
//     (`readmitLinks`), against all links and against Euler's bound 3n − 6;
//   - how much of the links' total SCORE is kept (the overlay should get the jumps);
//   - whether the kept drawing really has no crossing beyond the tree drawing's own, and
//     what the flow makes of it in `flowSteps` steps (crossings, tightest gap, time per step).
// One JSON line per (vault, tree, drawing).
// @see src/core/planarSkeleton.ts, docs/plans/2026-11-24-medial-sphere-method.md ("What to cut")
import { appendFileSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { countCrossings, crossingPairs } from '../../src/core/medialGraphMove';
import { medialGraphFrom, medialGraphStep, minClearance } from '../../src/core/medialSphereGraph';
import {
    linkScores,
    orderSiblings,
    readmitLinks,
    spanningTreeBy,
    type TreeChoice,
} from '../../src/core/planarSkeleton';
import type { Edge, Vec3 } from '../../src/core/testConfigs';
import { organicTree, tidyTree } from './treeLayouts';

const DIR = process.argv[2] ?? 'local_files/pkm-graphs/graphs';
const FLOW_STEPS = Number(process.argv[3] ?? 40);
const OUT = process.argv[4] ?? 'bench/results/2026-11-24-planar-skeleton.jsonl';
// Both drawings have neighbouring leaves about 1 apart; a link taken back in keeps this far
// from everything it does not end at.
const CLEARANCE = 0.2;
const R = 0.25;

const LAYOUTS: [string, (n: number, parent: number[], root: number) => Vec3[]][] = [
    ['tidy tree', tidyTree],
    ['force layout', organicTree],
];
// 'tidy': without the force layout of the tree (minutes on a vault of several thousand notes).
if (process.argv[5] === 'tidy') LAYOUTS.length = 1;
const CHOICES: TreeChoice[] = ['bfs', 'bfs-best', 'max-weight'];

for (const file of readdirSync(DIR)
    .filter((f) => f.endsWith('.json'))
    .sort()) {
    const src = JSON.parse(readFileSync(join(DIR, file), 'utf8')) as {
        name: string;
        kind: string;
        nodes: number;
        edges: [number, number][];
    };
    if (src.kind !== 'vault') continue;
    const n = src.nodes;
    const score = linkScores(n, src.edges);
    let links = 0;
    let scoreAll = 0;
    for (let e = 0; e < src.edges.length; e++)
        if (score[e] >= 0) {
            links++;
            scoreAll += score[e];
        }
    for (const choice of CHOICES) {
        const { parent, via, root } = spanningTreeBy(n, src.edges, score, choice);
        const joined = parent.filter((p) => p !== -2).length;
        if (joined < n) {
            console.error(`${src.name}: ${n - joined} notes not joined to the root: skipped`);
            continue;
        }
        const depth = new Int32Array(n);
        let maxDepth = 0;
        const order = [root];
        const kids: number[][] = Array.from({ length: n }, () => []);
        for (let i = 0; i < n; i++) if (parent[i] >= 0) kids[parent[i]].push(i);
        for (let q = 0; q < order.length; q++)
            for (const c of kids[order[q]]) {
                depth[c] = depth[order[q]] + 1;
                maxDepth = Math.max(maxDepth, depth[c]);
                order.push(c);
            }
        for (const [layout, make] of LAYOUTS)
            for (const siblings of layout === 'tidy tree'
                ? (['as found', 'linked together'] as const)
                : (['as found'] as const)) {
                const t0 = performance.now();
                const V =
                    siblings === 'linked together'
                        ? tidyTree(n, parent, root, orderSiblings(n, parent, src.edges, score))
                        : make(n, parent, root);
                const layoutMs = performance.now() - t0;
                const x = new Float64Array(3 * n);
                for (let i = 0; i < n; i++) {
                    x[3 * i] = V[i][0];
                    x[3 * i + 1] = V[i][1];
                }
                const kept = new Uint8Array(src.edges.length);
                for (let i = 0; i < n; i++) if (via[i] >= 0) kept[via[i]] = 1;
                const treeEdges = src.edges.filter((_, e) => kept[e]) as Edge[];
                const treeCrossings = countCrossings(medialGraphFrom(V, treeEdges, 1));
                const t1 = performance.now();
                const taken = readmitLinks(x, src.edges, kept, score, CLEARANCE);
                const readmitMs = performance.now() - t1;
                const keptEdges = src.edges.filter((_, e) => kept[e]) as Edge[];
                let scoreKept = 0;
                for (let e = 0; e < src.edges.length; e++) if (kept[e]) scoreKept += score[e];
                // The kept drawing under the flow.
                const g = medialGraphFrom(V, keptEdges, 2);
                const crossingsKept = countCrossings(g);
                g.passPairs = crossingPairs(g, 1e-9 * R);
                const t2 = performance.now();
                for (let k = 0; k < FLOW_STEPS; k++) medialGraphStep(g, { r: R });
                const msPerStep = FLOW_STEPS > 0 ? (performance.now() - t2) / FLOW_STEPS : null;
                const row = {
                    bench: 'planar-skeleton',
                    graph: src.name,
                    tree: choice,
                    layout,
                    siblings,
                    nodes: n,
                    links,
                    eulerBound: 3 * n - 6,
                    treeLinks: n - 1,
                    takenBackIn: taken,
                    kept: n - 1 + taken,
                    keptShare: +((n - 1 + taken) / links).toFixed(4),
                    keptShareOfBound: +((n - 1 + taken) / Math.min(links, 3 * n - 6)).toFixed(4),
                    overlay: links - (n - 1) - taken,
                    scoreShareKept: +(scoreKept / (scoreAll || 1)).toFixed(4),
                    treeDepth: maxDepth,
                    treeCrossings,
                    crossingsKept,
                    flowSteps: FLOW_STEPS,
                    crossingsAfterFlow: countCrossings(g),
                    clearanceAfterFlow: +minClearance(g, R).toFixed(4),
                    finite: g.x.every(Number.isFinite),
                    layoutMs: Math.round(layoutMs),
                    readmitMs: Math.round(readmitMs),
                    msPerStep: msPerStep === null ? null : +msPerStep.toFixed(1),
                };
                appendFileSync(OUT, `${JSON.stringify(row)}\n`);
                console.log(JSON.stringify(row));
            }
    }
}
