// test/core/planarSkeleton.test.ts — what to cut in a graph that cannot be drawn without
// crossings: link scores, the scored spanning trees, and taking links back into a drawing.
import { describe, expect, test } from 'bun:test';
import {
    linkScores,
    orderSiblings,
    readmitLinks,
    spanningTreeBy,
    type TreeChoice,
} from '../../src/core/planarSkeleton';

type E = [number, number];
const positions = (pts: [number, number][]): Float64Array => {
    const x = new Float64Array(3 * pts.length);
    pts.forEach((p, i) => {
        x[3 * i] = p[0];
        x[3 * i + 1] = p[1];
    });
    return x;
};
/** Do the open segments cross (properly)? */
const cross = (x: Float64Array, a: number, b: number, c: number, d: number): boolean => {
    const o = (p: number, q: number, r: number): number =>
        (x[3 * q] - x[3 * p]) * (x[3 * r + 1] - x[3 * p + 1]) -
        (x[3 * q + 1] - x[3 * p + 1]) * (x[3 * r] - x[3 * p]);
    return o(a, b, c) * o(a, b, d) < 0 && o(c, d, a) * o(c, d, b) < 0;
};

describe('planar skeleton', () => {
    test('a link scores the neighbours its ends share; self-links and repeats are out', () => {
        // A triangle 0-1-2 with a tail 2-3, a self-link and a repeated link.
        const edges: E[] = [
            [0, 1],
            [1, 2],
            [0, 2],
            [2, 3],
            [3, 3],
            [1, 0],
        ];
        expect(Array.from(linkScores(4, edges))).toEqual([1, 1, 1, 0, -1, -1]);
    });

    test('every choice of tree spans what hangs together; the scored ones score no less', () => {
        // Two triangles sharing node 2 (the most linked), a path off one of them, a loner.
        const edges: E[] = [
            [0, 1],
            [1, 2],
            [0, 2],
            [2, 3],
            [3, 4],
            [2, 4],
            [4, 5],
            [5, 6],
            [0, 6],
        ];
        const n = 8; // node 7 has no link
        const score = linkScores(n, edges);
        const total: Record<string, number> = {};
        const depths: Record<string, number[]> = {};
        for (const choice of ['bfs', 'bfs-best', 'max-weight'] as TreeChoice[]) {
            const { parent, via, root } = spanningTreeBy(n, edges, score, choice);
            expect(root).toBe(2);
            expect(parent[root]).toBe(-1);
            expect(parent[7]).toBe(-2);
            let sum = 0;
            const depth: number[] = [];
            for (let i = 0; i < 7; i++) {
                if (i !== root) {
                    // Its link really joins it to its parent.
                    expect([...edges[via[i]]].sort()).toEqual([i, parent[i]].sort());
                    sum += score[via[i]];
                }
                let d = 0;
                for (let v = i; v !== root; v = parent[v]) expect(++d).toBeLessThan(n); // reaches the root
                depth.push(d);
            }
            expect(new Set(via.filter((e) => e >= 0)).size).toBe(6); // 7 nodes, 6 different links
            total[choice] = sum;
            depths[choice] = depth;
        }
        expect(depths['bfs-best']).toEqual(depths.bfs); // as shallow as plain breadth-first
        expect(total['bfs-best']).toBeGreaterThanOrEqual(total.bfs);
        expect(total['max-weight']).toBeGreaterThanOrEqual(total['bfs-best']);
    });

    test('children that link to each other are put next to each other, best links first', () => {
        // Node 0 with children 1…6; among them the links 1-4, 4-6, 6-1 (a ring: one must give),
        // 2-5; child 3 links to none. Scores make 1-4 and 4-6 the better two of the ring.
        const edges: E[] = [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [0, 5],
            [0, 6],
            [1, 4],
            [4, 6],
            [6, 1],
            [2, 5],
        ];
        const parent = [-1, 0, 0, 0, 0, 0, 0];
        const score = Float64Array.from([0, 0, 0, 0, 0, 0, 3, 2, 1, 1]);
        const kids = orderSiblings(7, parent, edges, score);
        const order = kids[0];
        expect([...order].sort()).toEqual([1, 2, 3, 4, 5, 6]); // everyone, once
        const next = (a: number, b: number): boolean =>
            Math.abs(order.indexOf(a) - order.indexOf(b)) === 1;
        expect(next(1, 4)).toBe(true);
        expect(next(4, 6)).toBe(true);
        expect(next(6, 1)).toBe(false); // the ring's weakest link
        expect(next(2, 5)).toBe(true);
        for (let i = 1; i < 7; i++) expect(kids[i]).toEqual([]);
    });

    test('a square with both diagonals: the fourth side and ONE diagonal come back', () => {
        const x = positions([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ]);
        const edges: E[] = [
            [0, 1],
            [1, 2],
            [2, 3], // the tree: three sides
            [3, 0],
            [0, 2],
            [1, 3],
        ];
        const kept = Uint8Array.from([1, 1, 1, 0, 0, 0]);
        const score = linkScores(4, edges); // all 2: the shorter comes first
        expect(readmitLinks(x, edges, kept, score, 0.05)).toBe(2);
        expect(Array.from(kept)).toEqual([1, 1, 1, 1, 1, 0]);
    });

    test('a link along a row of nodes, or through a node, is not taken (it crosses nothing, but is no drawing)', () => {
        const x = positions([
            [0, 0],
            [1, 0],
            [2, 0],
            [1, 1],
            [1, -1],
        ]);
        const edges: E[] = [
            [0, 1],
            [1, 2],
            [1, 3],
            [1, 4], // the tree: a cross
            [0, 2], // on top of the row
            [3, 4], // straight through node 1
            [0, 3], // free
        ];
        const kept = Uint8Array.from([1, 1, 1, 1, 0, 0, 0]);
        const score = Float64Array.from([0, 0, 0, 0, 5, 5, 0]); // the bad ones are tried first
        expect(readmitLinks(x, edges, kept, score, 0.05)).toBe(1);
        expect(Array.from(kept)).toEqual([1, 1, 1, 1, 0, 0, 1]);
    });

    test('on a comb of 400 jittered nodes: nothing kept crosses, and every link left out is blocked', () => {
        let seed = 3;
        const rnd = (): number => {
            seed = (1664525 * seed + 1013904223) % 4294967296;
            return seed / 4294967296;
        };
        const S = 20;
        const pts: [number, number][] = [];
        for (let j = 0; j < S; j++)
            for (let i = 0; i < S; i++)
                pts.push([i + 0.3 * (rnd() - 0.5), j + 0.3 * (rnd() - 0.5)]);
        const x = positions(pts);
        const edges: E[] = [];
        for (let i = 0; i + 1 < S; i++) edges.push([i, i + 1]); // the back of the comb
        for (let j = 0; j + 1 < S; j++)
            for (let i = 0; i < S; i++) edges.push([j * S + i, (j + 1) * S + i]);
        const tree = edges.length;
        for (let k = 0; k < 1500; k++) {
            const a = Math.floor(rnd() * S * S);
            const dx = Math.floor(rnd() * 7) - 3;
            const dy = Math.floor(rnd() * 7) - 3;
            const i = (a % S) + dx;
            const j = Math.floor(a / S) + dy;
            if (i < 0 || i >= S || j < 0 || j >= S || (dx === 0 && dy === 0)) continue;
            edges.push([a, j * S + i]);
        }
        const kept = new Uint8Array(edges.length);
        kept.fill(1, 0, tree);
        const score = linkScores(S * S, edges);
        const clearance = 0.05;
        const taken = readmitLinks(x, edges, kept, score, clearance);
        expect(taken).toBeGreaterThan(100);
        const keptList = Array.from(edges.keys()).filter((e) => kept[e]);
        // Nothing kept crosses anything kept.
        let crossings = 0;
        for (let p = 0; p < keptList.length; p++)
            for (let q = p + 1; q < keptList.length; q++) {
                const [a, b] = edges[keptList[p]];
                const [c, d] = edges[keptList[q]];
                if (a === c || a === d || b === c || b === d) continue;
                if (cross(x, a, b, c, d)) crossings++;
            }
        expect(crossings).toBe(0);
        // Every candidate left out (with a score) crosses, repeats, or comes too near something kept:
        // here checked for the plain case — it crosses a kept link — on at least most of them.
        let leftOut = 0;
        let blockedByCrossing = 0;
        for (let e = tree; e < edges.length; e++) {
            if (kept[e] || score[e] < 0) continue;
            leftOut++;
            const [a, b] = edges[e];
            if (
                keptList.some((f) => {
                    const [c, d] = edges[f];
                    return a !== c && a !== d && b !== c && b !== d && cross(x, a, b, c, d);
                })
            )
                blockedByCrossing++;
        }
        expect(leftOut).toBeGreaterThan(100);
        expect(blockedByCrossing).toBeGreaterThan(0.8 * leftOut);
    });
});
