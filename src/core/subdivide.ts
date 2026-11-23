/**
 * Edge subdivision and planar crossing counting for graphs.
 *
 * Why subdivision exists. The discrete tangent-point energy is evaluated at VERTICES only. A
 * graph drawn with one straight segment per edge has no vertex along an edge, so two edges can
 * approach and cross mid-span while every evaluated distance stays large: the energy's barrier,
 * which is what forbids crossings in the smooth theory, is simply not sampled there. Measured
 * on scrambled planar graphs (2026-11-23, `bench/results/2026-11-23-force-prepass.jsonl`), the
 * flow alone changed the number of crossings (383 → 113 on a 6×6 grid): edges passed through
 * each other. Putting extra vertices along every edge samples the barrier where it is needed;
 * the paper draws its graph examples with subdivided, curved edges for the same reason.
 * @see local_files/repulsive_orig_paper/SelfAvoiding.tex (line 803, "Graph Drawing")
 *
 * Why the crossing counter lives here too: "did the flow keep the drawing's crossings fixed"
 * is the measurement that says whether a subdivision level is fine enough, and the user-facing
 * readability number for a flat graph.
 */
import type { Edge, Vec3 } from './testConfigs';

/** A subdivided graph and how it maps back to the graph it came from. */
export interface SubdividedGraph {
    /** The original vertices, in their original order and at indices `0 .. nodeCount-1`,
     *  followed by the new interior points. */
    vertices: Vec3[];
    /** `pieces` segments per original edge, consecutive: original edge `e` owns
     *  `edges[e·pieces .. (e+1)·pieces)`, ordered from its first endpoint to its second. */
    edges: Edge[];
    /** Number of vertices of the original graph. Why kept: pins, node spheres and anything else
     *  that means "a node" must not mistake an interior point for one. */
    nodeCount: number;
    pieces: number;
}

/**
 * Cut every edge into `pieces` equal straight segments. `pieces = 1` returns the graph as it is
 * (same arrays). Original vertices keep their indices, so anything addressed by vertex index —
 * pins, selections — stays valid.
 */
export function subdivideGraph(vertices: Vec3[], edges: Edge[], pieces: number): SubdividedGraph {
    const k = Math.max(1, Math.floor(pieces));
    if (k === 1) return { vertices, edges, nodeCount: vertices.length, pieces: 1 };
    const outV: Vec3[] = vertices.map((p) => [p[0], p[1], p[2]]);
    const outE: Edge[] = [];
    for (const [a, b] of edges) {
        const pa = vertices[a];
        const pb = vertices[b];
        let prev = a;
        for (let j = 1; j < k; j++) {
            const u = j / k;
            outV.push([
                pa[0] + (pb[0] - pa[0]) * u,
                pa[1] + (pb[1] - pa[1]) * u,
                pa[2] + (pb[2] - pa[2]) * u,
            ]);
            outE.push([prev, outV.length - 1]);
            prev = outV.length - 1;
        }
        outE.push([prev, b]);
    }
    return { vertices: outV, edges: outE, nodeCount: vertices.length, pieces: k };
}

/**
 * Number of proper crossings, in the xy-plane, between segments that share no vertex.
 * With `pieces > 1` (a graph from {@link subdivideGraph}), two segments of the SAME original
 * edge are not counted: an edge folding over itself is not a crossing between edges.
 * Exact O(|E|²); meant for measurements and for a status line on graphs of a few thousand
 * segments, not for an inner loop.
 */
export function countPlanarCrossings(
    vertices: readonly Vec3[],
    edges: readonly Edge[],
    pieces = 1,
): number {
    const m = edges.length;
    const orient = (a: Vec3, b: Vec3, c: Vec3): number =>
        Math.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]));
    let count = 0;
    for (let i = 0; i < m; i++) {
        const [a, b] = edges[i];
        const va = vertices[a];
        const vb = vertices[b];
        const minX = Math.min(va[0], vb[0]);
        const maxX = Math.max(va[0], vb[0]);
        const minY = Math.min(va[1], vb[1]);
        const maxY = Math.max(va[1], vb[1]);
        const ownerI = Math.floor(i / pieces);
        for (let j = i + 1; j < m; j++) {
            if (pieces > 1 && Math.floor(j / pieces) === ownerI) continue;
            const [c, d] = edges[j];
            if (a === c || a === d || b === c || b === d) continue;
            const vc = vertices[c];
            const vd = vertices[d];
            // Bounding boxes first: most pairs are nowhere near each other.
            if (
                Math.max(vc[0], vd[0]) < minX ||
                Math.min(vc[0], vd[0]) > maxX ||
                Math.max(vc[1], vd[1]) < minY ||
                Math.min(vc[1], vd[1]) > maxY
            )
                continue;
            if (
                orient(va, vb, vc) * orient(va, vb, vd) < 0 &&
                orient(vc, vd, va) * orient(vc, vd, vb) < 0
            )
                count++;
        }
    }
    return count;
}
