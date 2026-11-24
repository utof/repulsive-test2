/**
 * The medial-sphere idea carried over to FLAT GRAPHS WITH JUNCTIONS — an experiment, not a
 * method from any paper. The 2025 paper covers closed curves only; a graph layout needs three
 * things it does not have, and each is a choice made here:
 *
 *  1. Junctions. A node of degree ≠ 2 has no normal. Here a node grows one circle into each
 *     angular SECTOR between consecutive incident segments, along the sector's bisector. For a
 *     degree-2 node that is exactly the paper's ±normal; for a junction it is one probe per
 *     gap between its edges.
 *  2. No shrinking. The paper's length term makes the curve contract (that is how it
 *     untangles). A layout must keep its size, so the length term is replaced by a spring per
 *     segment with a rest length.
 *  3. Pins. A pinned node is held: zero right-hand side and an identity row in the solve.
 *
 * What is kept from the paper: the pull of each node towards the centres of its empty circles
 * (capped at `rMax`), which pushes a strand away from anything nearer than `2·rMax` and does
 * nothing in open space; the preconditioner as the graph Laplacian plus weak links between
 * nodes that share a circle; and a move limited only by a collision bound (here per node,
 * `nodeMotionCaps`), with no energy line search. Every interaction is local, so a step costs O(n).
 *
 * Along its own edge a node reacts only to a FOLD (the circumcircle of an ordinary bend is not
 * contact: pulling towards it makes the edge buckle outward, the paper's space-FILLING
 * behaviour, which a layout does not want). Two edges that meet at a junction interact only
 * when the angle between them falls under `minAngleDeg`. Both rules sit between two measured
 * failures — permanent repulsion that swells the drawing, and no repulsion at all, which lets
 * strands close to a rounding error — and are explained where they are applied in the step.
 *
 * @see docs/plans/2026-11-24-medial-sphere-method.md (the flat-graph spike and its bar)
 * @see src/core/medialSphere.ts (the closed-curve method this is derived from)
 */
import { forNearSegmentPairs, makeGrid, nodeMotionCaps, type SegmentList } from './medialSphere';
import { subdivideGraph } from './subdivide';
import type { Edge, Vec3 } from './testConfigs';

/** A subdivided flat graph ready for the flow. Coordinates are xyz with z = 0. */
export interface MedialGraph {
    n: number;
    x: Float64Array;
    segs: SegmentList;
    /** Rest length of every segment. */
    rest: Float64Array;
    /** Number of original graph nodes (they keep indices `0 .. nodeCount-1`). */
    nodeCount: number;
    /** Segments per original edge. */
    pieces: number;
    /** Endpoints of the original edges. */
    edgeA: Int32Array;
    edgeB: Int32Array;
    /** 1 = held in place. */
    pinned: Uint8Array;
    /**
     * A group being carried over the drawing (see src/core/medialGraphMove.ts). `edge[e] = 1`:
     * original edge `e` takes no part in the flow — no forces, no collisions, invisible to
     * every probe. `node[i] = 1`: the node is placed by the caller and the flow leaves it alone.
     */
    ghost?: { edge: Uint8Array; node: Uint8Array };
    /**
     * Pairs of original edges that are allowed to cross, and how many times: key
     * `min·E + max` (E = number of original edges; see `passKey`) → the number of crossings
     * the pair may have. Such a pair is not held apart by the motion limits, so a drawing
     * that has crossings can still flow; instead the step UNDOES the move of both edges
     * whenever it would give them more crossings than allowed, and lowers the allowance when
     * they have fewer. Every other pair keeps the no-crossing guarantee. So crossings can
     * disappear and never appear.
     * Two simpler rules were tried and measured: (1) the pair exempt from everything — the
     * two edges wove through each other, 3 crossings became 7; (2) the exemption per pair of
     * SEGMENTS — a crossing was then locked between two nodes of each edge, slid up to one of
     * them and jammed there (gap 1e-16), because a node only looks sideways from its own
     * strand and a strand that crosses it comes at it lengthways.
     */
    passPairs?: Map<number, number>;
}

/** Key of the unordered pair of original edges `(e, f)` in `MedialGraph.passPairs`. */
export function passKey(g: MedialGraph, e: number, f: number): number {
    return Math.min(e, f) * g.edgeA.length + Math.max(e, f);
}

/**
 * How many times original edges `e` and `f` cross right now (pairs of their segments that
 * share no node and intersect). Why exported: the group move counts with the same rule the
 * step enforces.
 */
export function crossingsBetween(g: MedialGraph, e: number, f: number): number {
    const { x, segs, pieces } = g;
    let c = 0;
    for (let p = 0; p < pieces; p++)
        for (let q = e === f ? p + 1 : 0; q < pieces; q++) {
            const k1 = e * pieces + p;
            const k2 = f * pieces + q;
            const a1 = segs.a[k1];
            const b1 = segs.b[k1];
            const a2 = segs.a[k2];
            const b2 = segs.b[k2];
            if (a1 === a2 || a1 === b2 || b1 === a2 || b1 === b2) continue;
            if (planarSegmentDistance(x, a1, b1, a2, b2) <= 0) c++;
        }
    return c;
}

/** Build the flow's graph: every edge cut into `pieces` segments, rest lengths as drawn. */
export function medialGraphFrom(vertices: Vec3[], edges: Edge[], pieces: number): MedialGraph {
    const g = subdivideGraph(vertices, edges, pieces);
    return medialGraphFromSubdivided(g.vertices, g.edges, g.nodeCount, g.pieces);
}

/**
 * Build the flow's graph from one that is ALREADY subdivided the way `subdivideGraph` does it
 * (original nodes first, then each edge's interior points; edge `e` owns segments
 * `e·pieces … (e+1)·pieces`). Why: the app's store keeps its graph in that form, and the flow
 * must address the same vertex indices the pins and the renderer use.
 * @see src/core/subdivide.ts (the layout this relies on)
 */
export function medialGraphFromSubdivided(
    vertices: readonly Vec3[],
    edges: readonly Edge[],
    nodeCount: number,
    pieces: number,
): MedialGraph {
    const n = vertices.length;
    const x = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) {
        x[3 * i] = vertices[i][0];
        x[3 * i + 1] = vertices[i][1];
    }
    const m = edges.length;
    const a = new Int32Array(m);
    const b = new Int32Array(m);
    const rest = new Float64Array(m);
    for (let k = 0; k < m; k++) {
        a[k] = edges[k][0];
        b[k] = edges[k][1];
        rest[k] = Math.hypot(x[3 * b[k]] - x[3 * a[k]], x[3 * b[k] + 1] - x[3 * a[k] + 1]);
    }
    const original = m / pieces;
    const edgeA = new Int32Array(original);
    const edgeB = new Int32Array(original);
    for (let e = 0; e < original; e++) {
        edgeA[e] = a[e * pieces];
        edgeB[e] = b[(e + 1) * pieces - 1];
    }
    return {
        n,
        x,
        segs: { a, b },
        rest,
        nodeCount,
        pieces,
        edgeA,
        edgeB,
        pinned: new Uint8Array(n),
    };
}

/** Parameters of the flat-graph flow. */
export interface MedialGraphOptions {
    /** Wanted clearance scale: a strand is pushed by anything nearer than `2·capRatio·r`. */
    r: number;
    /** `rMax / r`. Default 1. */
    capRatio?: number;
    /** Spring stiffness relative to the paper's unit tension. Default 4. */
    spring?: number;
    /** Constant tension along every segment (the paper's length term has 1). Default 0.5;
     *  edges settle about `tension / spring` short of their rest length. */
    tension?: number;
    /** Stiffness of the bending springs (across two consecutive segments of an edge), in
     *  units of `spring`. Default 1; 0 switches them off. */
    bend?: number;
    /** Smallest angle, in degrees, that two edges meeting at a junction are left alone at;
     *  below it they push each other apart. Default 30. */
    minAngleDeg?: number;
    /** Exponent of the contact barrier on the push (0 = the paper's bounded pull). Default 1. */
    barrier?: number;
    /** Weight of the Laplacian links between nodes sharing a circle. Paper: 0.01. */
    beta?: number;
    /** Floor of the mass weight, in units of α, for nodes whose probes hit nothing (nodes in
     *  contact get their own stiffness on top). Default 0.02. */
    lambda?: number;
    /** Step length before the per-node limits. Default 1 (a quasi-Newton step). */
    tau?: number;
    /** Undo the move of two edges that are allowed to cross when it would give them more
     *  crossings than allowed (see `MedialGraph.passPairs`). Default true; false exists so a
     *  test can show that this check, and nothing else, is what holds the count. */
    holdCrossings?: boolean;
    /** Give the segment that a node pushes off the opposite push (see the step).
     *  `'auto'` (default): only at the contacts that need it — where the probing node is held
     *  (pinned or carried: it cannot give way itself), or where the segment, or the probing
     *  node's own segment, is a long one (over 3× the typical length). `'all'`: every
     *  contact; measured to halve the opening speed of a finely divided grid (18×18 at 8
     *  points per edge: clearance 0.28 after 600 steps against 0.68), so not the default.
     *  `'none'`: the node-only pull of the paper, kept for comparison. */
    reaction?: 'auto' | 'all' | 'none';
    /** The drag: node index → (x, y) it is being pulled to. Such a node is NOT held; it gets a
     *  spring to its target (`pinStiffness`, force capped at `pinForceMax`) and otherwise
     *  takes part in the flow like any other — so the contact barrier acts on it too.
     *  Why a spring and not a prescribed move: a node marched towards its target at the speed
     *  its crossing-free limit allows closes every gap ahead of it by 90% per step and jams
     *  against the first strand in its way (measured: a 2×2 blob dragged into a 12×12 grid
     *  stopped after a third of the way with zero clearance, for good). Pulled by a bounded
     *  force it pushes what is ahead out of the way and advances as room opens. */
    pinTargets?: ReadonlyMap<number, readonly [number, number]>;
    /** Drag spring stiffness: the pull is `pinStiffness · distance / r`. Default 4. */
    pinStiffness?: number;
    /** Largest drag pull, in units of the tension. Default 6. */
    pinForceMax?: number;
}

/** What one step did. */
export interface MedialGraphStepResult {
    /** Nodes whose move was cut to their crossing-free limit this step. */
    cappedNodes: number;
    /** Mean node displacement of this step, in units of `r` — the convergence measure
     *  (the 2024 paper stops at 3e-4). */
    movePerNode: number;
    cgIterations: number;
    links: number;
    /** Smallest circle radius found, in units of `r`: half the tightest clearance seen. */
    minRadius: number;
}

/** Per-node topology that does not change during a flow. */
interface Topology {
    /** CSR of each node's neighbours along segments. */
    nbrStart: Int32Array;
    nbr: Int32Array;
    /** Original edge owning an interior node, or −1 for an original node. */
    own: Int32Array;
    /** Keys `min·nodeCount + max` of the original edges, for junction–junction adjacency. */
    edgeKeys: Set<number>;
}

const topologies = new WeakMap<SegmentList, Topology>();

function topologyOf(g: MedialGraph): Topology {
    const cached = topologies.get(g.segs);
    if (cached) return cached;
    const { n, segs } = g;
    const m = segs.a.length;
    const nbrStart = new Int32Array(n + 1);
    for (let k = 0; k < m; k++) {
        nbrStart[segs.a[k] + 1]++;
        nbrStart[segs.b[k] + 1]++;
    }
    for (let i = 0; i < n; i++) nbrStart[i + 1] += nbrStart[i];
    const fill = Int32Array.from(nbrStart.subarray(0, n));
    const nbr = new Int32Array(2 * m);
    for (let k = 0; k < m; k++) {
        nbr[fill[segs.a[k]]++] = segs.b[k];
        nbr[fill[segs.b[k]]++] = segs.a[k];
    }
    const own = new Int32Array(n).fill(-1);
    // Interior points of edge e are nodeCount + e·(pieces−1) + 0..pieces−2 (subdivideGraph).
    if (g.pieces > 1)
        for (let i = g.nodeCount; i < n; i++)
            own[i] = Math.floor((i - g.nodeCount) / (g.pieces - 1));
    const edgeKeys = new Set<number>();
    for (let e = 0; e < g.edgeA.length; e++)
        edgeKeys.add(
            Math.min(g.edgeA[e], g.edgeB[e]) * g.nodeCount + Math.max(g.edgeA[e], g.edgeB[e]),
        );
    const t = { nbrStart, nbr, own, edgeKeys };
    topologies.set(g.segs, t);
    return t;
}

/** Largest number of probe directions a node can have (its degree, at least 3). */
function maxDirections(t: Topology, n: number): number {
    let d = 3;
    for (let i = 0; i < n; i++) d = Math.max(d, t.nbrStart[i + 1] - t.nbrStart[i]);
    return d;
}

/**
 * One step of the flat-graph flow, in place on `g.x`.
 * @see docs/plans/2026-11-24-medial-sphere-method.md
 */
export function medialGraphStep(g: MedialGraph, opts: MedialGraphOptions): MedialGraphStepResult {
    const { n, x, segs, rest, pinned } = g;
    const m = segs.a.length;
    const t = topologyOf(g);
    const r = opts.r;
    const rMax = (opts.capRatio ?? 1) * r;
    const ks = opts.spring ?? 4;
    const tension = opts.tension ?? 0.5;
    const barrier = opts.barrier ?? 1;
    const bend = opts.bend ?? 1;
    const reaction = opts.reaction ?? 'auto';
    const tanHalfMin = Math.tan(((opts.minAngleDeg ?? 30) * Math.PI) / 360);
    const beta = opts.beta ?? 0.01;
    const lambda = opts.lambda ?? 0.02;
    const tau = opts.tau ?? 1;
    // The paper's empirical thickness–weight relation (its reference code: r ≈ 3.36/√α).
    const alpha = (3.36 / r) ** 2;
    // Carried group and allowed crossings (both usually absent). `fixed` = not moved by the flow.
    const ghostEdge = g.ghost?.edge;
    const pass = g.passPairs && g.passPairs.size > 0 ? g.passPairs : undefined;
    const nEdges = g.edgeA.length;
    const fixed = g.ghost
        ? pinned.map((v, i) => v | (g.ghost as { node: Uint8Array }).node[i])
        : pinned;

    const len = new Float64Array(m);
    const w = new Float64Array(n);
    for (let k = 0; k < m; k++) {
        const a = segs.a[k];
        const b = segs.b[k];
        len[k] = Math.hypot(x[3 * b] - x[3 * a], x[3 * b + 1] - x[3 * a + 1]) || 1e-300;
        w[a] += 0.5 * len[k];
        w[b] += 0.5 * len[k];
    }

    // --- the pull: one circle per angular sector of every node --------------------------------
    const G = new Float64Array(3 * n);
    // LONG segments (over 3× the typical length) are kept out of the grid search and tested
    // against every node instead. Why: the search radius has to cover the longest segment it
    // serves, and one edge drawn across the whole map (a moved group's link) would make every
    // node search the whole drawing — measured 3.4× slower per step on a 12×12 grid with four
    // such edges. With no long segment this is exactly the old search. "Typical" = the mean
    // of the segments not over twice the plain mean, so a few long ones do not hide themselves.
    let meanAll = 0;
    for (let k = 0; k < m; k++) meanAll += len[k] / m;
    let typical = 0;
    let nTypical = 0;
    for (let k = 0; k < m; k++)
        if (len[k] <= 2 * meanAll) {
            typical += len[k];
            nTypical++;
        }
    typical /= nTypical || 1;
    const longSegs: number[] = [];
    let maxSeg = 0;
    for (let k = 0; k < m; k++)
        if (len[k] > 3 * typical) longSegs.push(k);
        else maxSeg = Math.max(maxSeg, len[k]);
    const isLong = longSegs.length > 0 ? new Uint8Array(m) : undefined;
    for (const k of longSegs) (isLong as Uint8Array)[k] = 1;
    // Nodes at the ends of a long segment: their contacts get the reaction force (see there).
    const onLong = longSegs.length > 0 ? new Uint8Array(n) : undefined;
    for (const k of longSegs) {
        (onLong as Uint8Array)[segs.a[k]] = 1;
        (onLong as Uint8Array)[segs.b[k]] = 1;
    }
    // The probes' search: a grid of segment MIDPOINTS. A segment can cut a node's capped circle
    // only if its midpoint is within 2·rMax + half its length of the node. Why midpoints and
    // not "segments starting at a node within reach" (the first version): the reach is half a
    // segment shorter and the test is per segment; the search was a third of a step.
    const segMid = new Float64Array(3 * m);
    for (let k = 0; k < m; k++) {
        segMid[3 * k] = 0.5 * (x[3 * segs.a[k]] + x[3 * segs.b[k]]);
        segMid[3 * k + 1] = 0.5 * (x[3 * segs.a[k] + 1] + x[3 * segs.b[k] + 1]);
    }
    const segReach = 2 * rMax + 0.5 * maxSeg;
    const grid = makeGrid(segMid, m, segReach);
    const reach2 = (2 * rMax) ** 2;
    const maxDir = maxDirections(t, n);
    const dirX = new Float64Array(maxDir);
    const dirY = new Float64Array(maxDir);
    const rad = new Float64Array(maxDir);
    const hit = new Int32Array(maxDir);
    const hitSeg = new Int32Array(maxDir);
    const hitAt = new Float64Array(maxDir);
    // Reaction of long segments (see below): force and stiffness handed to their end nodes.
    const reactDiag = new Float64Array(n);
    const angles: number[] = [];
    const linkA: number[] = [];
    const linkB: number[] = [];
    const linkW: number[] = [];
    const linkSeen = new Set<number>();
    // Current length of every original edge, and a node's position along its own edge in
    // segments (1 … pieces−1 for interior points; subdivideGraph's index layout).
    const edgeLen = new Float64Array(g.edgeA.length);
    for (let k = 0; k < m; k++) edgeLen[Math.floor(k / g.pieces)] += len[k];
    const hopIndex = (i: number): number => ((i - g.nodeCount) % (g.pieces - 1)) + 1;
    // Per node: the stiffness of its pull (per probe in contact, barrier included), averaged
    // over its probes — what the preconditioner's diagonal has to carry.
    const contact = new Float64Array(n);
    let minRadius = Infinity;
    for (let i = 0; i < n; i++) {
        const deg = t.nbrStart[i + 1] - t.nbrStart[i];
        const xi = x[3 * i];
        const yi = x[3 * i + 1];
        let nd = 0;
        if (deg === 0) continue;
        // A point of a carried edge neither pushes nor is pushed.
        if (ghostEdge && t.own[i] >= 0 && ghostEdge[t.own[i]]) continue;
        angles.length = 0;
        for (let q = t.nbrStart[i]; q < t.nbrStart[i + 1]; q++) {
            const j = t.nbr[q];
            angles.push(Math.atan2(x[3 * j + 1] - yi, x[3 * j] - xi));
        }
        if (deg === 1) {
            // A leaf: straight ahead and to both sides.
            for (const off of [Math.PI, Math.PI / 2, -Math.PI / 2]) {
                dirX[nd] = Math.cos(angles[0] + off);
                dirY[nd] = Math.sin(angles[0] + off);
                nd++;
            }
        } else {
            angles.sort((p, q) => p - q);
            for (let q = 0; q < deg; q++) {
                const a0 = angles[q];
                const a1 = q + 1 < deg ? angles[q + 1] : angles[0] + 2 * Math.PI;
                const mid = 0.5 * (a0 + a1);
                dirX[nd] = Math.cos(mid);
                dirY[nd] = Math.sin(mid);
                nd++;
            }
        }
        for (let l = 0; l < nd; l++) {
            rad[l] = rMax;
            hit[l] = -1;
        }
        const ownI = t.own[i];
        const hopI = ownI >= 0 ? hopIndex(i) : -1;
        // Every segment whose midpoint is within 2·rMax + half its length (see `segMid`).
        grid.forNearPoint(xi, yi, 0, segReach, (k) => {
            if (isLong?.[k]) return;
            const mx = segMid[3 * k] - xi;
            const my = segMid[3 * k + 1] - yi;
            const lim = 2 * rMax + 0.5 * len[k];
            if (mx * mx + my * my < lim * lim) testSegment(k);
        });
        for (const k of longSegs) testSegment(k);
        // forNear skips the node itself: segments starting AT node i contain it and are
        // skipped by the rule below anyway.
        function testSegment(k: number): void {
            const sa = segs.a[k];
            const sb = segs.b[k];
            if (sa === i || sb === i) return;
            // Cheap rejection first (most candidates end here): the segment's nearest point is
            // beyond reach of any capped circle.
            const px = x[3 * sa] - xi;
            const py = x[3 * sa + 1] - yi;
            const ux = x[3 * sb] - x[3 * sa];
            const uy = x[3 * sb + 1] - x[3 * sa + 1];
            const A = ux * ux + uy * uy;
            const B = 2 * (px * ux + py * uy);
            const C = px * px + py * py;
            const sNear = A > 0 ? Math.max(0, Math.min(1, -B / (2 * A))) : 0;
            if (A * sNear * sNear + B * sNear + C >= reach2) return;
            const e = Math.floor(k / g.pieces);
            if (ghostEdge?.[e]) return;
            // Two edges that are allowed to cross do not push each other's points away: a point
            // of one ignores the other. Why: with the push, the two points either side of every
            // crossing backed off to a full clearance from the other edge — an edge crossed
            // ten times became a band of zigzags. (A junction is not a point of ONE edge and
            // still sees everything: it has to stay off the edges its own edges cross.)
            if (pass && ownI >= 0 && pass.has(Math.min(ownI, e) * nEdges + Math.max(ownI, e)))
                return;
            const pos = (k % g.pieces) + 0.5; // the segment's place along its edge, in hops
            // Which rule applies (see the two measured failures each sits between):
            //  - a segment of the node's own edge (for a junction: of one of its edges): the
            //    FOLD rule — the circle is compared with a quarter of the arc between them,
            //    so an ordinary bend's circumcircle is not contact but a hairpin is;
            //  - a segment of an edge that meets the node's edge at a junction J: the
            //    MINIMUM-ANGLE rule — compared with what `minAngle` gives at this distance
            //    from J, so they push apart only when the angle closes below it;
            //  - anything else: the plain rule, compared with rMax.
            // "Compared with ref" means radii are rescaled by rMax/ref before the usual test.
            let ref = rMax;
            if (ownI >= 0) {
                const a1 = g.edgeA[ownI];
                const b1 = g.edgeB[ownI];
                if (e === ownI) {
                    const arc = ((Math.abs(hopI - pos) - 0.5) * edgeLen[e]) / g.pieces;
                    ref = Math.min(rMax, arc / 4);
                } else {
                    const a2 = g.edgeA[e];
                    const b2 = g.edgeB[e];
                    const shared = a1 === a2 || a1 === b2 ? a1 : b1 === a2 || b1 === b2 ? b1 : -1;
                    if (shared >= 0) {
                        const sJ = Math.hypot(x[3 * shared] - xi, x[3 * shared + 1] - yi);
                        ref = Math.min(rMax, sJ * tanHalfMin);
                    }
                }
            } else if (g.edgeA[e] === i || g.edgeB[e] === i) {
                const hop = g.edgeA[e] === i ? 0 : g.pieces;
                const arc = ((Math.abs(hop - pos) - 0.5) * edgeLen[e]) / g.pieces;
                ref = Math.min(rMax, arc / 4);
            }
            if (ref <= 0) return;
            const scale = rMax / ref;
            // The largest circle tangent at the node along x that stays clear of the WHOLE
            // segment P(s) = p + s·u: min over s∈[0,1] of |P|² / (2 x·P). A node-to-node test
            // (the paper's, and this file's first version) is blind to a strand closer than
            // about one segment length — its nearest node is off to the side, so the circle
            // through it is large — which is how strands closed to a rounding error unseen.
            for (let l = 0; l < nd; l++) {
                const E = dirX[l] * ux + dirY[l] * uy;
                const F = dirX[l] * px + dirY[l] * py;
                // R(s) = (A s² + B s + C) / (2 (E s + F)) where E s + F > 0. Stationary points:
                // A E s² + 2 A F s + (B F − E C) = 0. Candidates: those roots and the ends.
                let best = Infinity;
                let bestS = 0;
                const consider = (sv: number): void => {
                    if (sv < 0 || sv > 1) return;
                    const den = E * sv + F;
                    if (den <= 0) return;
                    const R = (A * sv * sv + B * sv + C) / (2 * den);
                    if (R < best) {
                        best = R;
                        bestS = sv;
                    }
                };
                consider(0);
                consider(1);
                const qa = A * E;
                const qb = 2 * A * F;
                const qc = B * F - E * C;
                if (Math.abs(qa) > 1e-300) {
                    const disc = qb * qb - 4 * qa * qc;
                    if (disc >= 0) {
                        const sq = Math.sqrt(disc);
                        consider((-qb + sq) / (2 * qa));
                        consider((-qb - sq) / (2 * qa));
                    }
                } else if (Math.abs(qb) > 1e-300) consider(-qc / qb);
                const R = scale * best;
                if (R < rad[l]) {
                    rad[l] = R;
                    hit[l] = bestS < 0.5 ? sa : sb;
                    hitSeg[l] = k;
                    hitAt[l] = bestS;
                }
            }
        }
        // G_i = −α w_i (1/N_i) Σ_l (R_l − rMax) x_l: descent moves the node away from whatever
        // cut a circle short. For the two opposite probes of a degree-2 node this IS the
        // paper's pull towards the circle centres (the rMax terms cancel); written as a
        // deficit it is also exactly zero in open space for ANY set of probe directions,
        // which sector bisectors of a junction are not (their plain sum does not vanish, and
        // the first version of this dragged every junction and leaf across empty space).
        let px = 0;
        let py = 0;
        let stiffSum = 0;
        for (let l = 0; l < nd; l++) {
            // Contact barrier: the deficit is multiplied by (rMax/R)^barrier, so the push grows
            // without bound as a circle shrinks to nothing. Why: the paper's pull saturates at
            // rMax, and a spring under 5× compression simply out-pushes it — strands were
            // pressed together until they were a rounding error apart (4e-17) and the sign of
            // an orientation test flipped: a "crossing" that no motion limit can prevent,
            // because nothing ever moved through anything. A force that wins at contact keeps
            // them apart at a finite distance instead.
            const Rl = Math.max(rad[l], 1e-12 * rMax);
            const gain = barrier > 0 ? (rMax / Rl) ** barrier : 1;
            px += gain * (rad[l] - rMax) * dirX[l];
            py += gain * (rad[l] - rMax) * dirY[l];
            // Its stiffness, relative to the plain pull's: −d/dR of gain·(rMax − R).
            const stiffL = hit[l] >= 0 ? gain * (1 + (barrier * (rMax - Rl)) / Rl) : 0;
            stiffSum += stiffL;
            // REACTION: the segment that cut the circle short gets the opposite push, shared
            // between its two ends by where the contact is. Why: the pull acts on the node
            // that probes, and a node that meets the INSIDE of a segment is felt by nobody on
            // the other side — the segment's ends see a point a good way off, keep moving as
            // the rest of the drawing drives them, and are only held back by their motion
            // limit, 0.45 of the gap per step: the gap closed geometrically to a rounding
            // error. Measured twice after a group was set down: its long link against a side
            // of the group (5e-2 → 2e-16 in 200 steps), and the link's middle point against
            // an ordinary edge of the grid (1e-7 → 2e-16). With the reaction every node of a
            // contact feels it, and the push is a pair of equal and opposite forces.
            // By default only where it is needed: a held node (it cannot give way, so without
            // the reaction an edge pressed against a pinned group's corner closed to the gap
            // floor and stayed: 0.008 where the same run with the reaction keeps 0.15), or a
            // long segment on either side. Between ordinary free segments both sides probe
            // each other anyway, and doubling the push there slowed a finely divided grid to
            // half its speed (see `reaction` in the options).
            if (
                hit[l] >= 0 &&
                (reaction === 'all' ||
                    (reaction === 'auto' && (fixed[i] || isLong?.[hitSeg[l]] || onLong?.[i])))
            ) {
                const sa = segs.a[hitSeg[l]];
                const sb = segs.b[hitSeg[l]];
                const f = (alpha * w[i] * gain * (rad[l] - rMax)) / nd;
                const kk = (alpha * w[i] * stiffL) / nd;
                const wb = hitAt[l];
                const wa = 1 - wb;
                G[3 * sa] += wa * f * dirX[l];
                G[3 * sa + 1] += wa * f * dirY[l];
                G[3 * sb] += wb * f * dirX[l];
                G[3 * sb + 1] += wb * f * dirY[l];
                reactDiag[sa] += wa * kk;
                reactDiag[sb] += wb * kk;
            }
            minRadius = Math.min(minRadius, rad[l]);
            const j = hit[l];
            if (j >= 0) {
                const id = Math.min(i, j) * n + Math.max(i, j);
                if (!linkSeen.has(id)) {
                    linkSeen.add(id);
                    linkA.push(i);
                    linkB.push(j);
                    linkW.push(
                        (beta * ks) /
                            Math.max(Math.hypot(x[3 * j] - xi, x[3 * j + 1] - yi), 1e-300),
                    );
                }
            }
        }
        contact[i] = stiffSum / nd;
        G[3 * i] -= (alpha * w[i] * px) / nd;
        G[3 * i + 1] -= (alpha * w[i] * py) / nd;
    }

    // --- springs and tension: E = Σ ks (len − L)² / (2L) + tension · len ----------------------------------------------------
    const stiff = new Float64Array(m);
    for (let k = 0; k < m; k++) {
        if (ghostEdge?.[Math.floor(k / g.pieces)]) continue; // a carried edge pulls on nobody
        const a = segs.a[k];
        const b = segs.b[k];
        const L = rest[k] || 1e-300;
        // Spring plus a constant tension (the paper's length term, kept small): the spring
        // holds the size, the tension pulls kinks straight — springs alone leave an edge that
        // was squeezed and released as a zigzag, because nothing resists bending.
        const f = ((ks * (len[k] - L)) / L + tension) / len[k];
        const fx = f * (x[3 * a] - x[3 * b]);
        const fy = f * (x[3 * a + 1] - x[3 * b + 1]);
        G[3 * a] += fx;
        G[3 * a + 1] += fy;
        G[3 * b] -= fx;
        G[3 * b + 1] -= fy;
        stiff[k] = ks / L + tension / len[k];
    }
    // --- bending: a spring across every two consecutive segments of an edge -----------------------
    // Rest length = the two segments' rest lengths added, i.e. "straight". Why: the probes look
    // sideways from an edge, so an edge that folds up ALONG its own line — an accordion, which
    // is how a chain under compression buckles — approaches itself where no probe looks.
    // Measured at 4,896 segments: a junction ended 1e-15 from the second segment of its own
    // edge while unrelated edges kept a clearance of 0.33. A spring between a point's two
    // neighbours resists the fold from the start, and as a link in P it costs no stability.
    const contactLinks = linkA.length; // reported; the bending links below are not contacts
    if (bend > 0 && g.pieces > 1) {
        const nodeAt = (e: number, q: number): number =>
            q === 0
                ? g.edgeA[e]
                : q === g.pieces
                  ? g.edgeB[e]
                  : g.nodeCount + e * (g.pieces - 1) + q - 1;
        for (let e = 0; e < g.edgeA.length; e++)
            for (let q = 1; q < g.pieces; q++) {
                if (ghostEdge?.[e]) break;
                const a = nodeAt(e, q - 1);
                const b = nodeAt(e, q + 1);
                const L = rest[e * g.pieces + q - 1] + rest[e * g.pieces + q] || 1e-300;
                const dx = x[3 * a] - x[3 * b];
                const dy = x[3 * a + 1] - x[3 * b + 1];
                const d = Math.hypot(dx, dy) || 1e-300;
                const f = (bend * ks * (d - L)) / L / d;
                G[3 * a] += f * dx;
                G[3 * a + 1] += f * dy;
                G[3 * b] -= f * dx;
                G[3 * b + 1] -= f * dy;
                linkA.push(a);
                linkB.push(b);
                linkW.push((bend * ks) / L);
            }
    }
    // --- the drag: a capped spring from each dragged node to its target ---------------------------
    const dragDiag = new Float64Array(n);
    if (opts.pinTargets) {
        const kp = (opts.pinStiffness ?? 4) / r;
        const fMax = opts.pinForceMax ?? 6;
        for (const [i, target] of opts.pinTargets) {
            if (i >= n || fixed[i]) continue;
            const dx = x[3 * i] - target[0];
            const dy = x[3 * i + 1] - target[1];
            const dist = Math.hypot(dx, dy);
            if (dist === 0) continue;
            const f = Math.min(kp * dist, fMax) / dist;
            G[3 * i] += f * dx;
            G[3 * i + 1] += f * dy;
            dragDiag[i] = kp; // its stiffness, for the preconditioner (more damping when capped)
        }
    }
    for (let i = 0; i < n; i++)
        if (fixed[i]) {
            G[3 * i] = 0;
            G[3 * i + 1] = 0;
        }

    // --- P = spring Laplacian + links + λ α M; Jacobi-preconditioned conjugate gradients ------
    const diag = new Float64Array(n);
    // Mass term = the pull's frozen-centre Hessian, α·w·(share of probes in contact), plus a
    // floor λ·α·w. Why not a uniform λ: a step of length 1 is stable only where P is at least
    // half as stiff as the energy. A uniform λ = 1 is stable but makes whole-drawing motion
    // crawl; λ = 0.1 moves fast but is 10× too soft exactly at the nodes in contact, which then
    // overshoot every step (measured at 4,896 segments: the move per step stayed at 3.6e-2·r
    // for 300 steps; with this term it decays to 1.6e-3). Free nodes get (almost) the pure
    // Laplacian, nodes in contact their full stiffness.
    // Tried and worse: the same stiffness as LINKS instead of a diagonal (free for two
    // strands moving together) — strands ended pressed to zero clearance.
    for (let i = 0; i < n; i++)
        diag[i] = (lambda + contact[i]) * alpha * w[i] + dragDiag[i] + reactDiag[i];
    for (let k = 0; k < m; k++) {
        diag[segs.a[k]] += stiff[k];
        diag[segs.b[k]] += stiff[k];
    }
    for (let k = 0; k < linkA.length; k++) {
        diag[linkA[k]] += linkW[k];
        diag[linkB[k]] += linkW[k];
    }
    const applyP = (v: Float64Array, out: Float64Array): void => {
        for (let i = 0; i < n; i++) out[i] = diag[i] * v[i];
        for (let k = 0; k < m; k++) {
            const a = segs.a[k];
            const b = segs.b[k];
            out[a] -= stiff[k] * v[b];
            out[b] -= stiff[k] * v[a];
        }
        for (let k = 0; k < linkA.length; k++) {
            const a = linkA[k];
            const b = linkB[k];
            out[a] -= linkW[k] * v[b];
            out[b] -= linkW[k] * v[a];
        }
        // A pinned node's unknown is fixed at 0: identity row (its v stays 0 throughout).
        for (let i = 0; i < n; i++) if (fixed[i]) out[i] = v[i];
    };
    const d = new Float64Array(3 * n);
    const rhs = new Float64Array(n);
    const sol = new Float64Array(n);
    let cgIterations = 0;
    for (let c = 0; c < 2; c++) {
        for (let i = 0; i < n; i++) rhs[i] = G[3 * i + c];
        cgIterations = Math.max(cgIterations, jacobiCg(applyP, diag, rhs, sol));
        for (let i = 0; i < n; i++) d[3 * i + c] = fixed[i] ? 0 : sol[i];
    }

    // --- move, each node within its own crossing-free limit ------------------------------------
    // Not one global step: see `nodeMotionCaps` for why (a single tight spot froze the graph).
    // Pairs that may cross (a carried edge with anything; two edges allowed to cross) limit
    // nobody; every other pair keeps the guarantee.
    const pieces = g.pieces;
    const skip =
        ghostEdge || pass
            ? (k1: number, k2: number): boolean => {
                  const e1 = Math.floor(k1 / pieces);
                  const e2 = Math.floor(k2 / pieces);
                  if (ghostEdge && (ghostEdge[e1] || ghostEdge[e2])) return true;
                  return !!pass && pass.has(Math.min(e1, e2) * nEdges + Math.max(e1, e2));
              }
            : undefined;
    const close = {
        below: 0.02 * r,
        rel: 0.02,
        k1: [] as number[],
        k2: [] as number[],
        d: [] as number[],
    };
    const caps = nodeMotionCaps(x, n, segs, 2 * rMax, 0.45, skip, close);
    let move = 0;
    let capped = 0;
    const xOld = pass || close.d.length > 0 ? Float64Array.from(x) : undefined;
    for (let i = 0; i < n; i++) {
        const dx = tau * d[3 * i];
        const dy = tau * d[3 * i + 1];
        const dl = Math.hypot(dx, dy);
        let sc = 1;
        if (dl > caps[i]) {
            sc = caps[i] / dl;
            capped++;
        }
        x[3 * i] -= sc * dx;
        x[3 * i + 1] -= sc * dy;
        move += sc * dl;
    }
    // --- a floor under every gap --------------------------------------------------------------
    // A pair of segments already nearer than 0.02·r, and nearer than 0.02 of the shorter of
    // the two (so that the ordinary gaps next to a junction of a finely divided drawing do
    // not count: see `nodeMotionCaps`), may not get nearer: where the move would do that, the
    // pair's four nodes stay where they were. Why: the motion limit lets a gap
    // shrink by up to 90% a step, so anything that keeps pressing closes it geometrically —
    // to a rounding error, where the sign of an orientation test flips and a "crossing"
    // appears that nothing moved through. Forces were added for each case found (barrier,
    // reaction, bending), but a real mind map has hubs with 200 children drawn as a fan of
    // edges a hundredth of a clearance apart, and there the forces cannot all be satisfied
    // (measured: 1,237-node map, a gap went 4e-3 → 6e-15 in 60 steps, then 44 "crossings").
    // With the floor no gap that starts above that threshold can fall below a tenth of it.
    // Holding nodes back is always safe: a node that moves less stays within its limit.
    // --- edges allowed to cross: no more crossings than allowed ---------------------------------
    // The motion limits do not hold these pairs apart, so the count is checked after the move
    // and the move of BOTH edges undone where it went up (see `MedialGraph.passPairs`).
    // One undo can change another pair that shares a node with it, hence the loop; a pair
    // whose nodes are all back where they were is as it was, so the loop ends.
    if (xOld) {
        const held = new Uint8Array(n);
        const hold = (i: number): void => {
            x[3 * i] = xOld[3 * i];
            x[3 * i + 1] = xOld[3 * i + 1];
            held[i] = 1;
        };
        const undo = (e: number): void => {
            for (let q = 0; q <= pieces; q++)
                hold(
                    q === 0
                        ? g.edgeA[e]
                        : q === pieces
                          ? g.edgeB[e]
                          : g.nodeCount + e * (pieces - 1) + q - 1,
                );
        };
        const holdCrossings = pass !== undefined && (opts.holdCrossings ?? true);
        // An edge is undone at most once: a count that is still too high after both edges are
        // back where they were was not raised by this step, and undoing again changes nothing.
        const undone = pass ? new Uint8Array(nEdges) : undefined;
        for (let again = true; again; ) {
            again = false;
            for (let q = 0; q < close.d.length; q++) {
                const a1 = segs.a[close.k1[q]];
                const b1 = segs.b[close.k1[q]];
                const a2 = segs.a[close.k2[q]];
                const b2 = segs.b[close.k2[q]];
                if (held[a1] && held[b1] && held[a2] && held[b2]) continue;
                if (planarSegmentDistance(x, a1, b1, a2, b2) >= close.d[q]) continue;
                hold(a1);
                hold(b1);
                hold(a2);
                hold(b2);
                again = true;
            }
            if (pass && undone && holdCrossings)
                for (const [key, allowed] of pass) {
                    const e = Math.floor(key / nEdges);
                    const f = key % nEdges;
                    if ((undone[e] && undone[f]) || crossingsBetween(g, e, f) <= allowed) continue;
                    undo(e);
                    undo(f);
                    undone[e] = 1;
                    undone[f] = 1;
                    again = true;
                }
        }
        if (pass && holdCrossings)
            for (const [key, allowed] of pass) {
                const c = crossingsBetween(g, Math.floor(key / nEdges), key % nEdges);
                if (c < allowed) pass.set(key, c);
            }
    }
    return {
        cappedNodes: capped,
        movePerNode: move / n / r,
        cgIterations,
        links: contactLinks,
        minRadius: minRadius / r,
    };
}

/**
 * Conjugate gradients for the symmetric positive-definite `P x = b` with the diagonal as
 * preconditioner; relative residual 1e-6 or 300 iterations. Returns the iteration count.
 * Why no factorization: this repo has no sparse Cholesky, and the mass term keeps P well
 * enough conditioned for a spike; if the iteration counts in the benchmark grow with the
 * graph, that is the first thing to replace.
 */
function jacobiCg(
    applyP: (v: Float64Array, out: Float64Array) => void,
    diag: Float64Array,
    b: Float64Array,
    x: Float64Array,
): number {
    const n = b.length;
    const res = new Float64Array(n);
    const z = new Float64Array(n);
    const p = new Float64Array(n);
    const Ap = new Float64Array(n);
    let bNorm = 0;
    let rz = 0;
    for (let i = 0; i < n; i++) {
        x[i] = 0;
        res[i] = b[i];
        z[i] = b[i] / diag[i];
        p[i] = z[i];
        bNorm += b[i] * b[i];
        rz += res[i] * z[i];
    }
    if (bNorm === 0) return 0;
    for (let it = 1; it <= 300; it++) {
        applyP(p, Ap);
        let pAp = 0;
        for (let i = 0; i < n; i++) pAp += p[i] * Ap[i];
        const a = rz / pAp;
        let rr = 0;
        for (let i = 0; i < n; i++) {
            x[i] += a * p[i];
            res[i] -= a * Ap[i];
            rr += res[i] * res[i];
        }
        if (rr <= 1e-12 * bNorm) return it;
        let rzNew = 0;
        for (let i = 0; i < n; i++) {
            z[i] = res[i] / diag[i];
            rzNew += res[i] * z[i];
        }
        const bt = rzNew / rz;
        rz = rzNew;
        for (let i = 0; i < n; i++) p[i] = z[i] + bt * p[i];
    }
    return 300;
}

/**
 * Smallest distance between two segments of NON-INCIDENT original edges (exact,
 * grid-accelerated), or `searchWithin` if none is nearer. Why not all pairs: segments of one
 * edge are a segment-length apart by construction, and two edges that meet at a junction come
 * arbitrarily close near it — neither says anything about the drawing's clearance. Pairs
 * that are allowed to cross (`passPairs`) are left out for the same reason.
 */
export function minClearance(g: MedialGraph, searchWithin: number): number {
    const incident = (e: number, f: number): boolean =>
        e === f ||
        g.edgeA[e] === g.edgeA[f] ||
        g.edgeA[e] === g.edgeB[f] ||
        g.edgeB[e] === g.edgeA[f] ||
        g.edgeB[e] === g.edgeB[f];
    const { n, x, segs } = g;
    let best = searchWithin;
    forNearSegmentPairs(x, n, segs, searchWithin, (e, f) => {
        const e1 = Math.floor(e / g.pieces);
        const e2 = Math.floor(f / g.pieces);
        if (incident(e1, e2)) return;
        if (g.passPairs?.has(passKey(g, e1, e2))) return;
        best = Math.min(best, planarSegmentDistance(x, segs.a[e], segs.b[e], segs.a[f], segs.b[f]));
    });
    return best;
}

/**
 * Distance between two segments in the plane: 0 if they cross, else the least
 * endpoint–segment distance. Why exported: the group move decides which edge pairs cross.
 */
export function planarSegmentDistance(
    x: Float64Array,
    a: number,
    b: number,
    c: number,
    d: number,
): number {
    const orient = (p: number, q: number, s: number): number =>
        (x[3 * q] - x[3 * p]) * (x[3 * s + 1] - x[3 * p + 1]) -
        (x[3 * q + 1] - x[3 * p + 1]) * (x[3 * s] - x[3 * p]);
    if (orient(a, b, c) * orient(a, b, d) < 0 && orient(c, d, a) * orient(c, d, b) < 0) return 0;
    const pointSeg = (p: number, s0: number, s1: number): number => {
        const ux = x[3 * s1] - x[3 * s0];
        const uy = x[3 * s1 + 1] - x[3 * s0 + 1];
        const l2 = ux * ux + uy * uy;
        const tt =
            l2 > 0
                ? Math.max(
                      0,
                      Math.min(
                          1,
                          ((x[3 * p] - x[3 * s0]) * ux + (x[3 * p + 1] - x[3 * s0 + 1]) * uy) / l2,
                      ),
                  )
                : 0;
        return Math.hypot(x[3 * p] - x[3 * s0] - tt * ux, x[3 * p + 1] - x[3 * s0 + 1] - tt * uy);
    };
    return Math.min(pointSeg(a, c, d), pointSeg(b, c, d), pointSeg(c, a, b), pointSeg(d, a, b));
}
