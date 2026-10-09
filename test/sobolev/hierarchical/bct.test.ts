/**
 * Task 3 of Phase 2H: the block cluster tree (a PORT of the spike; its measured constants are
 * ORACLES, reproduced exactly) plus the one thing the spike lacks — a streaming chi=0 path.
 *
 * Oracle artefacts (git-ignored spike outputs under
 * `.superpowers/sdd/2026-09-11-phase2h-queue/scratch/`, all produced by `run-bct.ts` /
 * `run-accuracy.ts` over `scratch/bvh.ts` + `scratch/bct.ts` on the fixtures of `scratch/curves.ts`):
 *
 * - `out-bct-torusKnot-theta0.5-leaf8-endpoint4-struct.json`, key `rows[].stats` +
 *   `rows[].nearPairsPerEdge`, params `{THETA: 0.5, THETA_T: null, LEAF: 8, QUAD: "endpoint4",
 *   ALPHA: 3, BETA: 6, EPS: 1e-10}`.
 * - `out-bct-tangle_circle-theta0.5-leaf8-endpoint4-struct.json`, same params, fixtures `tangle`
 *   and `circle`.
 * - `out-accuracy.json`, params `{QUAD: "endpoint4", ALPHA: 3, BETA: 6, EPS: 1e-10}`, keys
 *   `rows[].{nFar,nNear,nearPairs,adjacencyVetoed}` — the theta ladder and the leaf sweep on
 *   `torusKnot` n=960, plus `openHelix`.
 *
 * The fixtures below are the spike's generators transcribed with the SAME arithmetic (same
 * constants, same op order, same bbox normalisation): the oracles are integer combinatorics over a
 * deterministic tree over these exact coordinates, so exact equality is the honest assertion.
 *
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.5, §7.1, §7.2, §11.2 task 3, §11.3
 */
import { describe, expect, test } from 'bun:test';
import {
    type Bct,
    type BctParams,
    buildBct,
    DEFAULT_THETA,
    DEFAULT_THETA_T,
    thetaLadder,
    traverseBct,
} from '../../../src/core/sobolev/hierarchical/bct';
import { type Bvh, buildBvh } from '../../../src/core/sobolev/hierarchical/bvh';
import type { Edge, Vec3 } from '../../../src/core/testConfigs';

/** `params.EPS` of every oracle artefact. */
const EPS = 1e-10;

// ---------------------------------------------------------------------------------------------
// Fixtures — transcribed from `scratch/curves.ts` (same arithmetic, same op order), INCLUDING its
// bbox normalisation (`curves.ts:49-66`, applied by `closedFrom`/`openFrom`): every oracle below
// was measured on curves rescaled to bounding-box max extent 1, centred at the origin. That is a
// CONDITION of the oracles, not cosmetics: the R^6 split score adds squared diameters of three
// position and three unit-tangent coordinates in one number (`curves.ts:13-17`), so a different
// spatial scale changes the chosen split axes, the tree, and therefore kappa_near.
// ---------------------------------------------------------------------------------------------

interface Curve {
    vertices: Vec3[];
    edges: Edge[];
}

/** `scratch/curves.ts` `lcg`. */
function lcg(seed: number): () => number {
    let s = seed >>> 0;
    return () => {
        s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
        return s / 4294967296;
    };
}

/** `scratch/curves.ts` `normalize`: bbox max extent 1, bbox centred at the origin. */
function normalize(X: Float64Array, n: number): void {
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n; i++) {
        for (let c = 0; c < 3; c++) {
            const v = X[3 * i + c];
            if (v < lo[c]) lo[c] = v;
            if (v > hi[c]) hi[c] = v;
        }
    }
    const ext = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
    const s = ext > 0 ? 1 / ext : 1;
    for (let i = 0; i < n; i++) {
        for (let c = 0; c < 3; c++) {
            X[3 * i + c] = (X[3 * i + c] - 0.5 * (lo[c] + hi[c])) * s;
        }
    }
}

function toCurve(X: Float64Array, n: number, closed: boolean): Curve {
    normalize(X, n);
    const vertices: Vec3[] = [];
    for (let i = 0; i < n; i++) vertices.push([X[3 * i], X[3 * i + 1], X[3 * i + 2]]);
    const edges: Edge[] = [];
    const m = closed ? n : n - 1;
    for (let I = 0; I < m; I++) edges.push([I, closed ? (I + 1) % n : I + 1]);
    return { vertices, edges };
}

/** `scratch/curves.ts` `circle`. */
function circle(n: number): Curve {
    const X = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) {
        const t = (2 * Math.PI * i) / n;
        X[3 * i] = Math.cos(t);
        X[3 * i + 1] = Math.sin(t);
        X[3 * i + 2] = 0;
    }
    return toCurve(X, n, true);
}

/** `scratch/curves.ts` `torusKnot(n, 2, 3)`: R=2, r=0.8. */
function torusKnot(n: number): Curve {
    const p = 2;
    const q = 3;
    const X = new Float64Array(3 * n);
    const R = 2;
    const r = 0.8;
    for (let i = 0; i < n; i++) {
        const t = (2 * Math.PI * i) / n;
        const rad = R + r * Math.cos(q * t);
        X[3 * i] = rad * Math.cos(p * t);
        X[3 * i + 1] = rad * Math.sin(p * t);
        X[3 * i + 2] = r * Math.sin(q * t);
    }
    return toCurve(X, n, true);
}

/** `scratch/curves.ts` `tangle(n, 20260911)`. */
function tangle(n: number): Curve {
    const rnd = lcg(20260911);
    const H = 7;
    const a: number[] = [];
    const ph: number[] = [];
    for (let k = 0; k < 3 * H; k++) {
        a.push(0.35 * (rnd() - 0.5));
        ph.push(2 * Math.PI * rnd());
    }
    const X = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) {
        const t = (2 * Math.PI * i) / n;
        const base = [Math.cos(t), Math.sin(t), 0.3 * Math.sin(3 * t)];
        for (let c = 0; c < 3; c++) {
            let v = base[c];
            for (let k = 0; k < H; k++) {
                const h = k + 2;
                v += a[c * H + k] * Math.sin(h * t + ph[c * H + k]);
            }
            X[3 * i + c] = v;
        }
    }
    return toCurve(X, n, true);
}

/** `scratch/curves.ts` `openHelix(n, 4)` — open chain, `m = n - 1`. */
function openHelix(n: number): Curve {
    const turns = 4;
    const X = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) {
        const t = (turns * 2 * Math.PI * i) / (n - 1);
        X[3 * i] = Math.cos(t);
        X[3 * i + 1] = Math.sin(t);
        X[3 * i + 2] = (3 * i) / (n - 1);
    }
    return toCurve(X, n, false);
}

const FIXTURES = { circle, torusKnot, tangle, openHelix } as const;
type FixtureName = keyof typeof FIXTURES;

const bvhCache = new Map<string, Bvh>();
/** BVH builds dominate this file's runtime (270 ms at n=30 000), so cache per (fixture, n, leaf). */
function tree(fixture: FixtureName, n: number, leafSize = 8): Bvh {
    const key = `${fixture}/${n}/${leafSize}`;
    let bvh = bvhCache.get(key);
    if (!bvh) {
        const { vertices, edges } = FIXTURES[fixture](n);
        bvh = buildBvh(vertices, edges, leafSize, EPS);
        bvhCache.set(key, bvh);
    }
    return bvh;
}

// ---------------------------------------------------------------------------------------------
// Brute-force helpers.
// ---------------------------------------------------------------------------------------------

/** Leaf node ids under `node`, by DFS. */
function leavesUnder(bvh: Bvh, node: number, out: number[] = []): number[] {
    if (bvh.left[node] < 0) out.push(node);
    else {
        leavesUnder(bvh, bvh.left[node], out);
        leavesUnder(bvh, bvh.right[node], out);
    }
    return out;
}

/**
 * Coverage count of every ORDERED leaf pair by the union of far and near blocks. A correct BCT is
 * a partition: every entry is exactly 1. This is the test that catches a traversal silently
 * dropping blocks — the failure that makes every kappa_near LOOK better.
 */
function leafPairCoverage(bvh: Bvh, bct: Bct): Uint32Array {
    const leafIndex = new Int32Array(bvh.nNodes).fill(-1);
    const leaves = leavesUnder(bvh, 0);
    leaves.forEach((id, k) => {
        leafIndex[id] = k;
    });
    const L = leaves.length;
    const cover = new Uint32Array(L * L);
    const mark = (A: number, B: number) => {
        for (const la of leavesUnder(bvh, A)) {
            for (const lb of leavesUnder(bvh, B)) cover[leafIndex[la] * L + leafIndex[lb]]++;
        }
    };
    for (let k = 0; k < bct.stats.nFar; k++) mark(bct.farA[k], bct.farB[k]);
    for (let k = 0; k < bct.stats.nNear; k++) mark(bct.nearA[k], bct.nearB[k]);
    return cover;
}

/** Does a far block `(A, B)` contain an edge pair the energy omits (shared vertex, or I == J)? */
function farBlockHasAdjacentPair(bvh: Bvh, A: number, B: number): boolean {
    const { begin, end, sE } = bvh;
    for (let s = begin[A]; s < end[A]; s++) {
        const ia = sE[2 * s];
        const ib = sE[2 * s + 1];
        for (let t = begin[B]; t < end[B]; t++) {
            const ja = sE[2 * t];
            const jb = sE[2 * t + 1];
            // The exact predicate of `calculateDisjointPairs` (src/core/tangentPointEnergy.ts:52).
            if (ia === ja || ia === jb || ib === ja || ib === jb) return true;
        }
    }
    return false;
}

function countFarBlocksWithAdjacentPairs(bvh: Bvh, bct: Bct): number {
    let bad = 0;
    for (let k = 0; k < bct.stats.nFar; k++) {
        if (farBlockHasAdjacentPair(bvh, bct.farA[k], bct.farB[k])) bad++;
    }
    return bad;
}

function bufEq(a: Int32Array, b: Int32Array): boolean {
    return Buffer.from(a.buffer, a.byteOffset, a.byteLength).equals(
        Buffer.from(b.buffer, b.byteOffset, b.byteLength),
    );
}

// ---------------------------------------------------------------------------------------------
// 1. The oracle table — exact.
// ---------------------------------------------------------------------------------------------

/** One struct-artefact row: `[n, nFar, nNear, nearPairs, adjacencyVetoed, maxStackDepth]`. */
type StructRow = [number, number, number, number, number, number];

// Conditions (§11.3): theta=0.5, thetaT=null (off), leaf=8, EPS=1e-10, fixtures at unit bbox
// extent (see the fixtures header), closed curves m=n except openHelix m=n-1.
describe('oracle table: out-bct-*-theta0.5-leaf8-endpoint4-struct.json rows[].stats (theta=0.5, thetaT=null, leaf=8, unit-bbox fixtures)', () => {
    const table: Record<FixtureName, { rows: StructRow[]; kappaNear: Record<number, number> }> = {
        torusKnot: {
            rows: [
                [960, 3530, 1622, 89538, 248, 44],
                [2000, 7698, 3520, 208994, 648, 50],
                [5000, 34534, 14622, 318382, 3506, 62],
                [10000, 77610, 28444, 604436, 7400, 68],
                [30000, 154540, 62730, 3245868, 15240, 74],
            ],
            // rows[].nearPairsPerEdge — §11.2 names 93.3 at n=960 and 108.20 at n=30 000.
            kappaNear: {
                960: 93.26875,
                2000: 104.497,
                5000: 63.6764,
                10000: 60.4436,
                30000: 108.1956,
            },
        },
        tangle: {
            rows: [
                [960, 6562, 4782, 273630, 80, 44],
                [30000, 412220, 184526, 9681766, 13498, 74],
            ],
            kappaNear: { 960: 285.03125, 30000: 322.7255333333333 },
        },
        circle: {
            rows: [
                [960, 748, 384, 21598, 480, 44],
                [30000, 24556, 12288, 659598, 16256, 74],
            ],
            kappaNear: { 960: 22.497916666666665, 30000: 21.9866 },
        },
        // out-accuracy.json (same params); open chain so m = n - 1 = 959.
        openHelix: {
            rows: [[960, 1424, 1106, 64463, 72, 44]],
            kappaNear: { 960: 67.21897810218978 },
        },
    };
    for (const fixture of Object.keys(table) as FixtureName[]) {
        for (const [n, nFar, nNear, nearPairs, vetoed, maxStackDepth] of table[fixture].rows) {
            test(`${fixture} n=${n}`, () => {
                const bvh = tree(fixture, n);
                const bct = buildBct(bvh, { theta: 0.5 });
                expect(bct.stats.nFar).toBe(nFar);
                expect(bct.stats.nNear).toBe(nNear);
                expect(bct.stats.nearPairs).toBe(nearPairs);
                expect(bct.stats.adjacencyVetoed).toBe(vetoed);
                // Pins the push ORDER of the traversal, not only its block set: task 4's
                // accumulation order (and so its f32 noise oracle) follows the list order.
                expect(bct.stats.maxStackDepth).toBe(maxStackDepth);
                const m = bvh.order.length;
                expect(bct.stats.nearPairs / m).toBe(table[fixture].kappaNear[n]);
                expect(bct.farA.length).toBe(nFar);
                expect(bct.nearA.length).toBe(nNear);
                expect(bct.bytes).toBe(8 * (nFar + nNear));
            });
        }
    }

    test('torusKnot n=960 meanFarRows / meanNearRows (floats, rows[].stats)', () => {
        const bct = buildBct(tree('torusKnot', 960), { theta: 0.5 });
        expect(bct.stats.meanFarRows).toBeCloseTo(12.33371104815864, 12);
        expect(bct.stats.meanNearRows).toBeCloseTo(7.430332922318126, 12);
    });

    test('out-accuracy.json: torusKnot n=400 and n=4000 (theta=0.5, leaf=8)', () => {
        const r400 = buildBct(tree('torusKnot', 400), { theta: 0.5 }).stats;
        expect([r400.nFar, r400.nNear, r400.nearPairs, r400.adjacencyVetoed]).toEqual([
            1222, 714, 29108, 90,
        ]);
        const r4000 = buildBct(tree('torusKnot', 4000), { theta: 0.5 }).stats;
        expect([r4000.nFar, r4000.nNear, r4000.nearPairs, r4000.adjacencyVetoed]).toEqual([
            14808, 7186, 410074, 1576,
        ]);
    });

    test('out-accuracy.json: leaf sweep 4/16/32 on torusKnot n=960, theta=0.5', () => {
        const sweep: [number, number, number, number, number][] = [
            [4, 7334, 2684, 34706, 706],
            [16, 1384, 942, 211950, 60],
            [32, 302, 506, 455400, 8],
        ];
        for (const [leaf, nFar, nNear, nearPairs, vetoed] of sweep) {
            const s = buildBct(tree('torusKnot', 960, leaf), { theta: 0.5 }).stats;
            expect([s.nFar, s.nNear, s.nearPairs, s.adjacencyVetoed]).toEqual([
                nFar,
                nNear,
                nearPairs,
                vetoed,
            ]);
        }
    });
});

// ---------------------------------------------------------------------------------------------
// 2. The theta ladder — out-accuracy.json, torusKnot n=960 leaf=8.
// ---------------------------------------------------------------------------------------------

// Conditions (§11.3): torusKnot n=960 at unit bbox extent, leaf=8, thetaT=null, EPS=1e-10.
describe('theta ladder: out-accuracy.json torusKnot n=960 leaf=8 rows[].{nFar,nNear,nearPairs,adjacencyVetoed} (unit-bbox fixture)', () => {
    /** `[theta, nFar, nNear, nearPairs, adjacencyVetoed]`, descending theta. */
    const ladder: [number, number, number, number, number][] = [
        [1, 2482, 918, 50718, 724],
        [0.8, 2466, 1018, 56162, 454],
        [0.6, 3098, 1334, 73514, 288],
        [0.5, 3530, 1622, 89538, 248],
        [0.4, 4524, 2152, 118672, 0],
        [0.3, 5548, 2976, 163908, 0],
        [0.25, 5716, 3228, 178320, 0],
        [0.2, 5690, 3440, 190292, 0],
        [0.15, 6356, 3584, 198600, 0],
        [0.1, 7742, 3944, 218844, 0],
    ];
    const thetas = ladder.map((r) => r[0]);

    test('every rung reproduces exactly; kappa_near is monotone non-decreasing as theta falls', () => {
        const out = thetaLadder(tree('torusKnot', 960), thetas, {});
        expect(out.rungs.length).toBe(ladder.length);
        let prev = -1;
        ladder.forEach(([theta, nFar, nNear, nearPairs, vetoed], k) => {
            const r = out.rungs[k];
            expect(r.theta).toBe(theta);
            expect([
                r.stats.nFar,
                r.stats.nNear,
                r.stats.nearPairs,
                r.stats.adjacencyVetoed,
            ]).toEqual([nFar, nNear, nearPairs, vetoed]);
            expect(r.stats.nearPairs).toBeGreaterThanOrEqual(prev);
            prev = r.stats.nearPairs;
        });
        expect(out.anyTie).toBe(false);
    });

    test('adjacencyVetoed = 248 at theta=0.5 and 0 at theta=0.4 (the pair that tells a working certificate from a disabled one)', () => {
        const bvh = tree('torusKnot', 960);
        // NOT zero at 0.5 — a test asserting zero there is wrong (brief: "that asymmetry is the
        // measurement"); and zero at <=0.4 alone cannot distinguish a certificate from no certificate.
        expect(buildBct(bvh, { theta: 0.5 }).stats.adjacencyVetoed).toBe(248);
        expect(buildBct(bvh, { theta: 0.4 }).stats.adjacencyVetoed).toBe(0);
        // The structural reason 0 is the right value below 0.4 on this fixture: with the
        // certificate OFF the far set is identical, i.e. nothing was there to veto.
        const on = buildBct(bvh, { theta: 0.4 });
        const off = buildBct(bvh, { theta: 0.4, adjacencyCertificate: false });
        expect(off.stats.nFar).toBe(on.stats.nFar);
        expect(off.stats.adjacencyVetoed).toBe(0);
    });
});

// ---------------------------------------------------------------------------------------------
// 3. chi=0 is exact and streams.
// ---------------------------------------------------------------------------------------------

describe('chi=0 (theta=0): the exact all-inadmissible path', () => {
    test('leafSize=1: theta=0 gives nFar=0 — and the guard is load-bearing, a tiny positive theta does NOT', () => {
        // leafSize=1 is the setting where Eq. 28 written `max(r) <= theta*d` is degenerate:
        // single-primitive nodes have r == 0 exactly, so without the explicit guard theta=0
        // admits every non-adjacent leaf pair (0 <= 0) and "exact" becomes fully rank-1 lumped.
        // At leafSize=8 the assertion would pass without the guard (memory:
        // gates-name-their-impossible-values), which is why this test uses leafSize=1.
        const bvh = tree('torusKnot', 256, 1);
        const m = 256;
        const exact = buildBct(bvh, { theta: 0 });
        expect(exact.stats.exact).toBe(true);
        expect(exact.stats.nFar).toBe(0); // structural: the guard forbids admissibility outright
        expect(exact.stats.nNear).toBe(m * m); // every ordered leaf pair, diagonal included
        expect(exact.stats.nearPairs).toBe(m * m);
        expect(exact.stats.adjacencyVetoed).toBe(0); // nothing passed Eq. 28, so nothing was vetoed
        // The trap, demonstrated: the smallest positive theta still admits r == 0 pairs.
        const trap = buildBct(bvh, { theta: Number.MIN_VALUE });
        expect(trap.stats.exact).toBe(false);
        expect(trap.stats.nFar).toBeGreaterThan(0);
    });

    test('theta=0 at leafSize=8: nFar=0, every ordered leaf pair is one near block', () => {
        const bvh = tree('torusKnot', 960, 8);
        const s = buildBct(bvh, { theta: 0 }).stats;
        expect(s.nFar).toBe(0);
        expect(s.nNear).toBe(bvh.leafCount * bvh.leafCount);
        expect(s.nearPairs).toBe(960 * 960);
    });

    test('negative / NaN theta and NaN thetaT are rejected (chi=0 is the minimum, not "anything <= 0")', () => {
        const bvh = tree('torusKnot', 64, 8);
        expect(() => buildBct(bvh, { theta: -0.1 })).toThrow(RangeError);
        expect(() => buildBct(bvh, { theta: Number.NaN })).toThrow(RangeError);
        expect(() => buildBct(bvh, { theta: Number.POSITIVE_INFINITY })).toThrow(RangeError);
        expect(() => buildBct(bvh, { thetaT: Number.NaN })).toThrow(RangeError);
        expect(() => buildBct(bvh, { thetaT: 0 })).toThrow(RangeError);
    });

    test('streaming: chi=0 at n=30 000, leaf=8 allocates no O(N^2) near array', () => {
        // Bug this catches: a traversal that collects near blocks into a list (the spike's
        // `PairList`) before handing them to the consumer. At this setting that list is
        // leafCount^2 = 4096^2 = 16 777 216 blocks x 8 B = 134 MB (§7.1: "the dense storage this
        // whole phase exists to escape"), which the RSS delta below would see. The streaming
        // traversal's only allocation is its O(depth) stack.
        const bvh = tree('torusKnot', 30000, 8);
        const expectNear = bvh.leafCount * bvh.leafCount;
        Bun.gc(true);
        const rss0 = process.memoryUsage().rss;
        let visited = 0;
        const t = traverseBct(
            bvh,
            { theta: 0 },
            () => {
                throw new Error('chi=0 must never emit a far block');
            },
            () => {
                visited++;
            },
        );
        const rssDelta = process.memoryUsage().rss - rss0;
        expect(visited).toBe(expectNear);
        expect(t.stats.nNear).toBe(expectNear);
        expect(t.stats.nFar).toBe(0);
        expect(t.stats.nearPairs).toBe(30000 * 30000);
        expect(t.bytes).toBeLessThan(4096); // the stack: 2 ints per frame, O(depth) frames
        expect(rssDelta).toBeLessThan(48 * 1024 * 1024); // vs. >= 134 MB materialised
    });

    test('the memory instrument can trip: materialising the same lists costs >= 100 MB', () => {
        // Without this the previous test's RSS bound could be an instrument that never moves.
        const bvh = tree('torusKnot', 30000, 8);
        Bun.gc(true);
        const rss0 = process.memoryUsage().rss;
        const bct = buildBct(bvh, { theta: 0 });
        const rssDelta = process.memoryUsage().rss - rss0;
        expect(bct.bytes).toBe(8 * bvh.leafCount * bvh.leafCount); // 134 217 728
        expect(bct.bytes).toBeGreaterThanOrEqual(100 * 1024 * 1024);
        expect(rssDelta).toBeGreaterThanOrEqual(100 * 1024 * 1024);
    });
});

// ---------------------------------------------------------------------------------------------
// 4. The adjacency certificate is load-bearing.
// ---------------------------------------------------------------------------------------------

describe('adjacency certificate (spec §5.5; proof in bct.ts module TSDoc)', () => {
    test('torusKnot n=960 theta=0.5: disabled => some far block contains an adjacent pair; enabled => none', () => {
        const bvh = tree('torusKnot', 960);
        const off = buildBct(bvh, { theta: 0.5, adjacencyCertificate: false });
        expect(countFarBlocksWithAdjacentPairs(bvh, off)).toBeGreaterThan(0);
        const on = buildBct(bvh, { theta: 0.5 });
        expect(countFarBlocksWithAdjacentPairs(bvh, on)).toBe(0);
        // The vetoed blocks are exactly the far blocks the certificate removed; they were split
        // further, so the near side grows and the total is still a partition (see section 6).
        expect(on.stats.adjacencyVetoed).toBe(248);
        expect(off.stats.adjacencyVetoed).toBe(0);
        expect(on.stats.nearPairs).toBeGreaterThan(off.stats.nearPairs);
    });

    for (const [fixture, n, leaf] of [
        ['circle', 960, 8],
        ['tangle', 960, 8],
        ['openHelix', 960, 8],
        ['torusKnot', 512, 1],
        ['torusKnot', 512, 2],
    ] as [FixtureName, number, number][]) {
        test(`${fixture} n=${n} leaf=${leaf}: no far block contains an adjacent pair at theta in {1, 0.5, 0.25}`, () => {
            const bvh = tree(fixture, n, leaf);
            for (const theta of [1, 0.5, 0.25]) {
                expect(countFarBlocksWithAdjacentPairs(bvh, buildBct(bvh, { theta }))).toBe(0);
            }
        });
    }
});

// ---------------------------------------------------------------------------------------------
// 5. The inverted-admissibility knock-out (spec §7.2 H1-a knock-out 2) exists and is reachable.
// ---------------------------------------------------------------------------------------------

describe('invertAdmissibility knock-out (spec §7.2 H1-a, knock-out 2)', () => {
    test('one sign flip changes the block classification at theta=0.5, n=960', () => {
        const bvh = tree('torusKnot', 960);
        const normal = buildBct(bvh, { theta: 0.5 });
        const inverted = buildBct(bvh, { theta: 0.5, invertAdmissibility: true });
        expect(inverted.stats.nFar).not.toBe(normal.stats.nFar);
        expect(inverted.stats.nearPairs).not.toBe(normal.stats.nearPairs);
        expect(inverted.params.invertAdmissibility).toBe(true);
        // Still a partition — the knock-out must corrupt the OPERATOR, not drop blocks.
        const cover = leafPairCoverage(bvh, inverted);
        expect(cover.every((v) => v === 1)).toBe(true);
    });

    test('inverted at chi=0 is still exact: the guard precedes the (inverted) test', () => {
        const s = buildBct(tree('torusKnot', 256, 1), {
            theta: 0,
            invertAdmissibility: true,
        }).stats;
        expect(s.nFar).toBe(0);
    });
});

// ---------------------------------------------------------------------------------------------
// 6. Partition correctness, brute force.
// ---------------------------------------------------------------------------------------------

describe('partition: every ordered leaf pair is covered exactly once by far ∪ near', () => {
    const cases: [FixtureName, number, number, BctParams][] = [
        ['torusKnot', 64, 8, { theta: 0.5 }],
        ['torusKnot', 128, 8, { theta: 0.5 }],
        ['torusKnot', 128, 1, { theta: 0.5 }],
        ['torusKnot', 128, 1, { theta: 0.25 }],
        ['torusKnot', 128, 1, { theta: 0 }],
        ['torusKnot', 128, 8, { theta: 1, adjacencyCertificate: false }],
        ['tangle', 100, 1, { theta: 0.5 }],
        ['circle', 96, 8, { theta: 0.5, thetaT: 0.25 }],
        ['openHelix', 65, 1, { theta: 0.5 }],
    ];
    for (const [fixture, n, leaf, params] of cases) {
        test(`${fixture} n=${n} leaf=${leaf} ${JSON.stringify(params)}`, () => {
            const bvh = tree(fixture, n, leaf);
            const bct = buildBct(bvh, params);
            const cover = leafPairCoverage(bvh, bct);
            expect(cover.length).toBe(bvh.leafCount * bvh.leafCount);
            expect(cover.every((v) => v === 1)).toBe(true);
            // nearPairs is the near blocks' |A|*|B| sum — cross-check against the lists.
            let pairs = 0;
            for (let k = 0; k < bct.stats.nNear; k++) {
                const A = bct.nearA[k];
                const B = bct.nearB[k];
                pairs += (bvh.end[A] - bvh.begin[A]) * (bvh.end[B] - bvh.begin[B]);
            }
            expect(bct.stats.nearPairs).toBe(pairs);
        });
    }

    test('one traversal, two consumers: the streaming visitor sees exactly the materialised lists, in order', () => {
        const bvh = tree('torusKnot', 2000);
        const bct = buildBct(bvh, { theta: 0.5 });
        const farA: number[] = [];
        const farB: number[] = [];
        const nearA: number[] = [];
        const nearB: number[] = [];
        const t = traverseBct(
            bvh,
            { theta: 0.5 },
            (A, B) => {
                farA.push(A);
                farB.push(B);
            },
            (A, B) => {
                nearA.push(A);
                nearB.push(B);
            },
        );
        expect(t.stats).toEqual(bct.stats);
        expect(farA).toEqual(Array.from(bct.farA));
        expect(farB).toEqual(Array.from(bct.farB));
        expect(nearA).toEqual(Array.from(bct.nearA));
        expect(nearB).toEqual(Array.from(bct.nearB));
    });
});

// ---------------------------------------------------------------------------------------------
// 7. Determinism, defaults, and the theta-ladder tie flag.
// ---------------------------------------------------------------------------------------------

test('determinism: identical input gives bit-identical lists', () => {
    const bvh = tree('tangle', 2000);
    const a = buildBct(bvh, { theta: 0.5 });
    const b = buildBct(bvh, { theta: 0.5 });
    expect(bufEq(a.farA, b.farA)).toBe(true);
    expect(bufEq(a.farB, b.farB)).toBe(true);
    expect(bufEq(a.nearA, b.nearA)).toBe(true);
    expect(bufEq(a.nearB, b.nearB)).toBe(true);
    expect(a.stats).toEqual(b.stats);
});

test('defaults: omitted theta/thetaT resolve to the named constants (value NOT pinned here — gate H6 decides it)', () => {
    const bvh = tree('torusKnot', 960);
    const dflt = buildBct(bvh, {});
    const explicit = buildBct(bvh, { theta: DEFAULT_THETA, thetaT: DEFAULT_THETA_T });
    expect(dflt.params).toEqual(explicit.params);
    expect(dflt.stats).toEqual(explicit.stats);
    expect(dflt.params.adjacencyCertificate).toBe(true);
    expect(dflt.params.invertAdmissibility).toBe(false);
    expect(DEFAULT_THETA_T).toBe(Number.POSITIVE_INFINITY); // theta_T off (spec §5.5)
});

test('thetaT bound is live: tangle n=960 theta=0.5 thetaT=0.5 (out-accuracy.json)', () => {
    const s = buildBct(tree('tangle', 960), { theta: 0.5, thetaT: 0.5 }).stats;
    expect([s.nFar, s.nNear, s.nearPairs, s.adjacencyVetoed]).toEqual([6886, 5022, 287862, 80]);
    const s1 = buildBct(tree('tangle', 960), { theta: 0.5, thetaT: 1 }).stats;
    expect([s1.nFar, s1.nNear, s1.nearPairs]).toEqual([6562, 4782, 273630]); // == thetaT off
});

describe('theta-ladder tie flag (spec §7.2 H1-a knock-out 1 / must-NOT-trip 3)', () => {
    test('a repeated theta is a tie, and is flagged — never swallowed', () => {
        const out = thetaLadder(tree('torusKnot', 400), [0.5, 0.5, 0.25], {});
        expect(out.rungs[0].tiedWithPrevious).toBe(false);
        expect(out.rungs[1].tiedWithPrevious).toBe(true);
        expect(out.rungs[2].tiedWithPrevious).toBe(false);
        expect(out.anyTie).toBe(true);
        expect(out.ties).toEqual([[0.5, 0.5]]);
    });

    test('circle n=960: the 0.5 -> 0.25 kappa_near tie (spec §7.2 must-NOT-trip 3) reproduces and is FLAGGED', () => {
        const bvh = tree('circle', 960);
        const out = thetaLadder(bvh, [0.5, 0.25], {});
        const [a, b] = out.rungs;
        expect(a.stats.nearPairs).toBe(21598); // out-accuracy.json circle theta=0.5 leaf=8
        // theta=0.25 has NO artefact (out-accuracy.json carries circle at 0.5 only); MEASURED by
        // this task on 2026-10-09: nearPairs 21598, nNear 384 — the kappa_near tie the spec
        // claims — BUT nFar 904 vs 748 and adjacencyVetoed 0 vs 480. The NEAR lists are
        // bit-identical; the FAR lists are not: blocks vetoed at 0.5 are merely inadmissible at
        // 0.25 and split one level deeper into more, smaller far blocks. So the rank-1 lumping
        // differs and delta_compress need NOT tie exactly (task 4 measures); what ties, and what
        // the flag reports, is kappa_near — exactly §7.2's wording.
        expect(b.stats.nearPairs).toBe(21598);
        expect(b.stats.nNear).toBe(384);
        expect(b.stats.nFar).toBe(904);
        expect(b.stats.adjacencyVetoed).toBe(0);
        expect(b.tiedWithPrevious).toBe(true);
        expect(out.anyTie).toBe(true);
        expect(out.ties).toEqual([[0.5, 0.25]]);
        const l5 = buildBct(bvh, { theta: 0.5 });
        const l25 = buildBct(bvh, { theta: 0.25 });
        expect(bufEq(l5.nearA, l25.nearA) && bufEq(l5.nearB, l25.nearB)).toBe(true);
        expect(l5.farA.length).not.toBe(l25.farA.length);
        // And the flag does NOT fire where kappa_near differs (torusKnot 0.5 -> 0.25: 89538 vs 178320).
        expect(thetaLadder(tree('torusKnot', 960), [0.5, 0.25], {}).anyTie).toBe(false);
    });
});
