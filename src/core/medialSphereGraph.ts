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
import {
    forNearSegmentPairs,
    makeGrid,
    nodeMotionCaps,
    type SegmentList,
    segmentPieces,
} from './medialSphere';
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
    /**
     * Edges that TAKE UP SLACK: original edge → the shortest total rest length it may shrink
     * to. While such an edge's ends are nearer than its rest length, the rest length follows
     * them down (never up, never below the floor); at the floor the entry is dropped.
     * Why: a moved group's link is measured when it lands, and at that moment the drawing
     * around it is still pushed apart by the room that was made. It then settles back, the
     * link's ends come together, and a link that keeps its landing length has nowhere to put
     * the excess: measured on a 150-node tree, a link landed 6.3 long between ends that
     * settled 2.4 apart and stood as a Z. The floor (its length before the move, set by
     * `landLinks`) is what keeps this from being a ratchet that reels the group in.
     * @see src/core/medialGraphMove.ts (`landLinks`)
     */
    slack?: Map<number, number>;
    /**
     * Which nodes are at rest and left alone by the step (only kept when the step is run with
     * `sleep`): `asleep[i]`, the number of consecutive steps node `i` has hardly moved, and
     * the positions the last step left — a node found somewhere else has been moved by the
     * caller and is woken. Set to `undefined` to wake everything (after changing rest
     * lengths, pins or anything else the positions do not show).
     */
    sleepState?: { quiet: Uint16Array; asleep: Uint8Array; last: Float64Array };
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
     *  `'auto'` (default): at every contact between two UNRELATED edges, and besides where
     *  the probing node is held (pinned or carried: it cannot give way itself) or the
     *  segment, or the probing node's own segment, is a long one (over 3× the typical
     *  length). Not at the contacts of an edge with itself or with an edge it meets at a
     *  junction (the fold rule, the minimum angle). `'all'`: those too; measured to halve
     *  the opening speed of a finely divided grid (18×18 at 8 points per edge: clearance
     *  0.28 after 600 steps against 0.68), so not the default. Without the unrelated-edge
     *  part a real 88-node mind map settled with two strands 0.007 of a clearance apart
     *  (0.34 with it).
     *  `'none'`: the node-only pull of the paper, kept for comparison. */
    reaction?: 'auto' | 'all' | 'none';
    /** Leave the part of the drawing that is at rest alone. A node that has moved less than
     *  `below·r` (default 2e-4) in each of `after` steps (default 10) in a row goes to sleep:
     *  no probes, no motion limits, held in place. It wakes when anything within reach of it
     *  moves more than that, or when the caller moves it or drags it.
     *  Why: every interaction is local, so a step should cost what is MOVING, not the size
     *  of the drawing — a mind map of 30,000 nodes with one node being dragged.
     *  Off by default: the benchmarks of the flow itself measure every node. */
    sleep?: { below?: number; after?: number };
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
    /** Nodes the step worked on: all that are not held, or with `sleep` those awake. */
    awakeNodes: number;
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
 * Every segment's length, and per node half the length of the segments at it.
 * Why a function of its own (as the other plain passes of `medialGraphStep`): inside that
 * very large function the same loop ran several times slower — the engine's optimising tiers
 * do not take a function of that size.
 */
function segmentLengths(
    x: Float64Array,
    segs: SegmentList,
    len: Float64Array,
    w: Float64Array,
): void {
    const m = segs.a.length;
    for (let k = 0; k < m; k++) {
        const a = segs.a[k];
        const b = segs.b[k];
        len[k] = Math.hypot(x[3 * b] - x[3 * a], x[3 * b + 1] - x[3 * a + 1]) || 1e-300;
        w[a] += 0.5 * len[k];
        w[b] += 0.5 * len[k];
    }
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
    if (g.slack?.size) {
        // See `MedialGraph.slack`. An edge being carried is only drawn: left alone.
        const P = g.pieces;
        for (const [e, floor] of g.slack) {
            if (ghostEdge?.[e]) continue;
            const ia = g.edgeA[e];
            const ib = g.edgeB[e];
            const chord = Math.hypot(x[3 * ia] - x[3 * ib], x[3 * ia + 1] - x[3 * ib + 1]);
            let total = 0;
            for (let q = 0; q < P; q++) total += rest[e * P + q];
            if (chord < total) {
                const next = Math.max(floor, chord);
                for (let q = 0; q < P; q++) rest[e * P + q] *= next / total;
                total = next;
            }
            if (total <= floor) g.slack.delete(e);
        }
    }
    const pass = g.passPairs && g.passPairs.size > 0 ? g.passPairs : undefined;
    const nEdges = g.edgeA.length;
    const fixed = g.ghost
        ? pinned.map((v, i) => v | (g.ghost as { node: Uint8Array }).node[i])
        : pinned;

    // --- sleeping, the cheap way out: nothing awake and nothing touched → nothing to do --------
    // Why here, before anything is built: a drawing at rest is the common state of the app,
    // and even the linear passes below (lengths, grids, the solve's setup) were a tenth of a
    // full step (20 ms of 200 for 10,680 nodes).
    if (opts.sleep && g.sleepState && g.sleepState.asleep.length === n) {
        const s = g.sleepState;
        let busy = false;
        for (let i = 0; i < n && !busy; i++)
            busy =
                (!s.asleep[i] && !fixed[i]) ||
                x[3 * i] !== s.last[3 * i] ||
                x[3 * i + 1] !== s.last[3 * i + 1];
        if (!busy && opts.pinTargets)
            for (const [i, target] of opts.pinTargets)
                if (i < n && !fixed[i] && (x[3 * i] !== target[0] || x[3 * i + 1] !== target[1]))
                    busy = true;
        if (!busy)
            return {
                cappedNodes: 0,
                movePerNode: 0,
                cgIterations: 0,
                links: 0,
                minRadius: Number.POSITIVE_INFINITY,
                awakeNodes: 0,
            };
    }

    const len = new Float64Array(m);
    const w = new Float64Array(n);
    segmentLengths(x, segs, len, w);

    // --- the pull: one circle per angular sector of every node --------------------------------
    const G = new Float64Array(3 * n);
    // LONG segments (over 3× the typical length; "typical" = the mean of the segments not
    // over twice the plain mean, so a few long ones do not hide themselves): the contacts
    // that get the reaction force by default (see there). The neighbour search no longer
    // treats them specially — it cuts them into pieces (`segmentPieces`).
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
    for (let k = 0; k < m; k++) if (len[k] > 3 * typical) longSegs.push(k);
    const isLong = longSegs.length > 0 ? new Uint8Array(m) : undefined;
    for (const k of longSegs) (isLong as Uint8Array)[k] = 1;
    // Nodes at the ends of a long segment: their contacts get the reaction force (see there).
    const onLong = longSegs.length > 0 ? new Uint8Array(n) : undefined;
    for (const k of longSegs) {
        (onLong as Uint8Array)[segs.a[k]] = 1;
        (onLong as Uint8Array)[segs.b[k]] = 1;
    }
    // The probes' search: a grid of the midpoints of the segments' pieces (`segmentPieces`;
    // a long segment is several pieces). A segment can cut a node's capped circle only if one
    // of its pieces has its midpoint within 2·rMax + half the piece's length of the node.
    // Why midpoints and not "segments starting at a node within reach" (the first version):
    // the reach is half a segment shorter and the test is per piece; the search was a third
    // of a step.
    // With most of the drawing asleep, only the segments near something that can move, probe
    // or wake are cut and indexed (`relevantSegments`).
    const sleeping =
        opts.sleep && g.sleepState && g.sleepState.asleep.length === n ? g.sleepState : undefined;
    const pcs = segmentPieces(x, segs, 2 * rMax, (maxHalf) =>
        sleeping
            ? relevantSegments(x, segs, sleeping, fixed, opts.pinTargets, 2 * rMax + 2 * maxHalf)
            : undefined,
    );
    const segReach = 2 * rMax + pcs.maxHalf;
    const grid = makeGrid(pcs.mid, pcs.count, segReach, pcs.origin);
    // tested[k] = i: segment k has already been tested for node i (found once per piece).
    const tested = new Int32Array(m).fill(-1);
    const reach2 = (2 * rMax) ** 2;
    const maxDir = maxDirections(t, n);
    const dirX = new Float64Array(maxDir);
    const dirY = new Float64Array(maxDir);
    const rad = new Float64Array(maxDir);
    const hit = new Int32Array(maxDir);
    const hitSeg = new Int32Array(maxDir);
    const hitPlain = new Uint8Array(maxDir);
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
    // --- sleeping: which nodes this step leaves alone (see `sleep` in the options) -------------
    let sl = opts.sleep ? g.sleepState : undefined;
    if (opts.sleep && (!sl || sl.asleep.length !== n))
        sl = g.sleepState = {
            quiet: new Uint16Array(n),
            asleep: new Uint8Array(n),
            last: Float64Array.from(x),
        };
    const asleep = sl?.asleep;
    const xStart = sl ? Float64Array.from(x) : undefined;
    /** Wake every node of a segment within reach of (px, py). */
    const wakeNear = (px: number, py: number): void => {
        if (!sl) return;
        const s = sl;
        grid.forNearPoint(px, py, 0, segReach, (q) => {
            const k = pcs.seg[q];
            s.asleep[segs.a[k]] = 0;
            s.quiet[segs.a[k]] = 0;
            s.asleep[segs.b[k]] = 0;
            s.quiet[segs.b[k]] = 0;
        });
    };
    if (sl) {
        // Moved by the caller since the last step (a drag target, a group set down, an edit):
        // awake, with everything around where it was and where it is now.
        for (let i = 0; i < n; i++)
            if (x[3 * i] !== sl.last[3 * i] || x[3 * i + 1] !== sl.last[3 * i + 1]) {
                sl.asleep[i] = 0;
                sl.quiet[i] = 0;
                wakeNear(x[3 * i], x[3 * i + 1]);
                wakeNear(sl.last[3 * i], sl.last[3 * i + 1]);
            }
        if (opts.pinTargets)
            for (const [i, target] of opts.pinTargets)
                if (i < n && (x[3 * i] !== target[0] || x[3 * i + 1] !== target[1])) {
                    sl.asleep[i] = 0;
                    sl.quiet[i] = 0;
                }
        // A node the caller holds is not "asleep": when it is let go it must be awake.
        for (let i = 0; i < n; i++)
            if (fixed[i]) {
                sl.asleep[i] = 0;
                sl.quiet[i] = 0;
            }
    }
    // Held = not moved by this step: fixed by the caller, or asleep.
    const held = asleep ? fixed.map((v, i) => v | asleep[i]) : fixed;
    let awakeNodes = 0;
    for (let i = 0; i < n; i++) if (!held[i]) awakeNodes++;
    for (let i = 0; i < n; i++) {
        if (asleep?.[i]) continue;
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
        // Every segment with a piece whose midpoint is within 2·rMax + half the piece (see `pcs`).
        grid.forNearPoint(xi, yi, 0, segReach, (q) => {
            const k = pcs.seg[q];
            if (tested[k] === i) return;
            const mx = pcs.mid[3 * q] - xi;
            const my = pcs.mid[3 * q + 1] - yi;
            const lim = 2 * rMax + pcs.half[q];
            if (mx * mx + my * my >= lim * lim) return;
            tested[k] = i;
            testSegment(k);
        });
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
            let plain = true; // a segment of an unrelated edge: the plain rule applies
            if (ownI >= 0) {
                const a1 = g.edgeA[ownI];
                const b1 = g.edgeB[ownI];
                if (e === ownI) {
                    const arc = ((Math.abs(hopI - pos) - 0.5) * edgeLen[e]) / g.pieces;
                    ref = Math.min(rMax, arc / 4);
                    plain = false;
                } else {
                    const a2 = g.edgeA[e];
                    const b2 = g.edgeB[e];
                    const shared = a1 === a2 || a1 === b2 ? a1 : b1 === a2 || b1 === b2 ? b1 : -1;
                    if (shared >= 0) {
                        const sJ = Math.hypot(x[3 * shared] - xi, x[3 * shared + 1] - yi);
                        ref = Math.min(rMax, sJ * tanHalfMin);
                        plain = false;
                    }
                }
            } else if (g.edgeA[e] === i || g.edgeB[e] === i) {
                const hop = g.edgeA[e] === i ? 0 : g.pieces;
                const arc = ((Math.abs(hop - pos) - 0.5) * edgeLen[e]) / g.pieces;
                ref = Math.min(rMax, arc / 4);
                plain = false;
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
                    hitPlain[l] = plain ? 1 : 0;
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
            // By default: every contact between UNRELATED edges (the plain rule), plus a held
            // node (it cannot give way, so without the reaction an edge pressed against a
            // pinned group's corner closed to the gap floor and stayed: 0.008 where the same
            // run with the reaction keeps 0.15) and a long segment on either side.
            // Not for the two rules about an edge's own shape (the fold rule, the minimum
            // angle at a junction): those are one-sided by design, and reacting to them is
            // what slowed a finely divided grid to half its speed (see `reaction`).
            if (
                hit[l] >= 0 &&
                (reaction === 'all' ||
                    (reaction === 'auto' &&
                        (hitPlain[l] === 1 || fixed[i] || isLong?.[hitSeg[l]] || onLong?.[i])))
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
        // Between two held nodes: everything it would add goes to rows that are dropped (the
        // solve is over the free nodes). Skipped so that a sleeping drawing costs nothing here.
        if (held[a] && held[b]) continue;
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
                if (held[a] && held[b]) continue; // as for the springs above
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
        if (held[i]) {
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
    // The system is solved over the FREE nodes only (not held: not pinned, carried or asleep).
    // A held node's unknown is 0 and its right-hand side is 0 (zeroed above), so its row and
    // column drop out: every term they would add to a sum is an exact 0, which is why this
    // gives the same numbers, bit for bit, as solving over all nodes with identity rows — as
    // long as free nodes and segments are visited in their original order (they are).
    // Why: with most of a large drawing asleep the solve was the largest part of a step and
    // the one part that did not get cheaper (measured at 42,960 nodes with 13 % awake: 103 of
    // 240 ms).
    // @see docs/plans/2026-11-24-medial-sphere-method.md ("A step costs what is awake")
    let nFree = 0;
    const slot = new Int32Array(n).fill(-1);
    for (let i = 0; i < n; i++) if (!held[i]) slot[i] = nFree++;
    const freeNode = new Int32Array(nFree);
    const diagF = new Float64Array(nFree);
    for (let i = 0; i < n; i++)
        if (slot[i] >= 0) {
            freeNode[slot[i]] = i;
            diagF[slot[i]] = diag[i];
        }
    // Couplings between two free nodes (one with a held end multiplies that end's 0).
    let nc = 0;
    const coA = new Int32Array(m + linkA.length);
    const coB = new Int32Array(m + linkA.length);
    const coW = new Float64Array(m + linkA.length);
    for (let k = 0; k < m; k++) {
        const a = slot[segs.a[k]];
        const b = slot[segs.b[k]];
        if (a < 0 || b < 0) continue;
        coA[nc] = a;
        coB[nc] = b;
        coW[nc++] = stiff[k];
    }
    for (let k = 0; k < linkA.length; k++) {
        const a = slot[linkA[k]];
        const b = slot[linkB[k]];
        if (a < 0 || b < 0) continue;
        coA[nc] = a;
        coB[nc] = b;
        coW[nc++] = linkW[k];
    }
    const applyP = (v: Float64Array, out: Float64Array): void => {
        for (let i = 0; i < nFree; i++) out[i] = diagF[i] * v[i];
        for (let k = 0; k < nc; k++) {
            const a = coA[k];
            const b = coB[k];
            out[a] -= coW[k] * v[b];
            out[b] -= coW[k] * v[a];
        }
    };
    const d = new Float64Array(3 * n);
    const rhs = new Float64Array(nFree);
    const sol = new Float64Array(nFree);
    let cgIterations = 0;
    for (let c = 0; c < 2; c++) {
        for (let i = 0; i < nFree; i++) rhs[i] = G[3 * freeNode[i] + c];
        cgIterations = Math.max(cgIterations, jacobiCg(applyP, diagF, rhs, sol));
        for (let i = 0; i < nFree; i++) d[3 * freeNode[i] + c] = sol[i];
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
    // With sleeping nodes: only the pairs that have a segment with a node that can move.
    let activeSeg: Uint8Array | undefined;
    if (asleep) {
        activeSeg = new Uint8Array(m);
        for (let k = 0; k < m; k++) activeSeg[k] = held[segs.a[k]] && held[segs.b[k]] ? 0 : 1;
    }
    // `pcs`: the probes' cut of the segments (same reach, positions not yet moved).
    const caps = nodeMotionCaps(x, n, segs, 2 * rMax, 0.45, skip, close, activeSeg, pcs);
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
    // --- sleeping: who moved, who is woken by it, who goes to sleep -----------------------------
    if (sl && xStart && opts.sleep) {
        const eps = (opts.sleep.below ?? 2e-4) * r;
        const after = opts.sleep.after ?? 10;
        for (let i = 0; i < n; i++) {
            if (held[i]) continue;
            const dx = x[3 * i] - xStart[3 * i];
            const dy = x[3 * i + 1] - xStart[3 * i + 1];
            if (dx * dx + dy * dy > eps * eps) {
                sl.quiet[i] = 0;
                // (The grid is of the positions at the start of the step; its reach covers
                // one step's motion, which is under a clearance.)
                wakeNear(xStart[3 * i], xStart[3 * i + 1]);
            } else if (sl.quiet[i] < 65535) sl.quiet[i]++;
        }
        for (let i = 0; i < n; i++) if (!held[i] && sl.quiet[i] >= after) sl.asleep[i] = 1;
        sl.last.set(x);
    }
    return {
        cappedNodes: capped,
        movePerNode: move / n / r,
        cgIterations,
        links: contactLinks,
        minRadius: minRadius / r,
        awakeNodes,
    };
}

/**
 * Which segments this step's searches have to know about when most nodes are asleep: 1 per
 * segment that can come within `reach` of
 *  - a node that is, or can become, awake this step: not asleep, held by the caller, pulled
 *    by a pin target, moved by the caller since the last step — or an end of a segment near
 *    where such a moved node is or was (those are woken by it and then probe);
 *  - any point of a segment with such a node at an end (the pair search starts from those).
 * A superset, never less: the step's results are the same as with every segment indexed
 * (checked bit for bit against the unrestricted step).
 * How: a coarse mask of cells of side 2·reach. Every point that searches is within one cell
 * side of a SAMPLE (a node, or points along a segment no farther apart than a cell), the
 * 3×3 cells around each sample are marked, and a segment counts if a cell its bounding box
 * touches is marked: a segment within `reach` of a searching point has a point at most
 * reach + half a cell = one cell side from a sample, so in one of the nine.
 * Why: cutting every segment into pieces and indexing them twice per step was, with 13 % of
 * a 42,960-node drawing awake, 30 of 165 ms; this pass is two loops of arithmetic.
 * @see docs/plans/2026-11-24-medial-sphere-method.md ("A step costs what is awake")
 */
export function relevantSegments(
    x: Float64Array,
    segs: SegmentList,
    sl: { asleep: Uint8Array; last: Float64Array },
    fixed: ArrayLike<number>,
    pinTargets: ReadonlyMap<number, readonly [number, number]> | undefined,
    reach: number,
): Uint8Array | undefined {
    const n = sl.asleep.length;
    const m = segs.a.length;
    const may = new Uint8Array(n); // may be awake this step
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (let i = 0; i < n; i++) {
        if (!sl.asleep[i] || fixed[i]) may[i] = 1;
        // Where it is, and where it was at the last step (the same for all but moved nodes).
        const px = x[3 * i];
        const py = x[3 * i + 1];
        if (px < minX) minX = px;
        if (px > maxX) maxX = px;
        if (py < minY) minY = py;
        if (py > maxY) maxY = py;
        const lx = sl.last[3 * i];
        const ly = sl.last[3 * i + 1];
        if (lx !== px || ly !== py) {
            if (lx < minX) minX = lx;
            if (lx > maxX) maxX = lx;
            if (ly < minY) minY = ly;
            if (ly > maxY) maxY = ly;
        }
    }
    if (pinTargets) for (const i of pinTargets.keys()) if (i < n) may[i] = 1;
    if (!(maxX >= minX) || !(reach > 0)) return undefined;
    let C = 2 * reach;
    while ((Math.floor((maxX - minX) / C) + 1) * (Math.floor((maxY - minY) / C) + 1) > 1 << 22)
        C *= 2;
    const nx = Math.floor((maxX - minX) / C) + 1;
    const ny = Math.floor((maxY - minY) / C) + 1;
    const mask = new Uint8Array(nx * ny);
    const mark = (px: number, py: number): void => {
        const cx = Math.floor((px - minX) / C);
        const cy = Math.floor((py - minY) / C);
        for (let v = Math.max(0, cy - 1); v <= Math.min(ny - 1, cy + 1); v++)
            for (let u = Math.max(0, cx - 1); u <= Math.min(nx - 1, cx + 1); u++)
                mask[u + nx * v] = 1;
    };
    /** Does a cell the segment's bounding box touches carry a mark? (A long one: yes, unasked.) */
    const hits = (k: number): boolean => {
        const ax = x[3 * segs.a[k]];
        const ay = x[3 * segs.a[k] + 1];
        const bx = x[3 * segs.b[k]];
        const by = x[3 * segs.b[k] + 1];
        const u0 = Math.floor((Math.min(ax, bx) - minX) / C);
        const u1 = Math.floor((Math.max(ax, bx) - minX) / C);
        const v0 = Math.floor((Math.min(ay, by) - minY) / C);
        const v1 = Math.floor((Math.max(ay, by) - minY) / C);
        if ((u1 - u0 + 1) * (v1 - v0 + 1) > 256) return true;
        for (let v = v0; v <= v1; v++)
            for (let u = u0; u <= u1; u++) if (mask[u + nx * v]) return true;
        return false;
    };
    // Moved by the caller: awake, and it wakes the segments near where it is and where it was.
    let moved = false;
    for (let i = 0; i < n; i++)
        if (x[3 * i] !== sl.last[3 * i] || x[3 * i + 1] !== sl.last[3 * i + 1]) {
            moved = true;
            may[i] = 1;
            mark(x[3 * i], x[3 * i + 1]);
            mark(sl.last[3 * i], sl.last[3 * i + 1]);
        }
    if (moved)
        for (let k = 0; k < m; k++)
            if (hits(k)) {
                may[segs.a[k]] = 1;
                may[segs.b[k]] = 1;
            }
    // Every segment with an end that may be awake: marked along its whole length.
    for (let k = 0; k < m; k++) {
        const a = segs.a[k];
        const b = segs.b[k];
        if (!may[a] && !may[b]) continue;
        const ax = x[3 * a];
        const ay = x[3 * a + 1];
        const bx = x[3 * b];
        const by = x[3 * b + 1];
        const steps = Math.max(1, Math.ceil(Math.hypot(bx - ax, by - ay) / C));
        for (let j = 0; j <= steps; j++)
            mark(ax + (j / steps) * (bx - ax), ay + (j / steps) * (by - ay));
    }
    // (The test of `hits`, written out: this loop is over every segment of the drawing.)
    const only = new Uint8Array(m);
    const inv = 1 / C;
    for (let k = 0; k < m; k++) {
        const a = segs.a[k];
        const b = segs.b[k];
        if (may[a] || may[b]) {
            only[k] = 1;
            continue;
        }
        const ax = x[3 * a];
        const ay = x[3 * a + 1];
        const bx = x[3 * b];
        const by = x[3 * b + 1];
        const u0 = Math.floor(((ax < bx ? ax : bx) - minX) * inv);
        const u1 = Math.floor(((ax < bx ? bx : ax) - minX) * inv);
        const v0 = Math.floor(((ay < by ? ay : by) - minY) * inv);
        const v1 = Math.floor(((ay < by ? by : ay) - minY) * inv);
        if (u0 === u1 && v0 === v1) {
            only[k] = mask[u0 + nx * v0];
            continue;
        }
        if ((u1 - u0 + 1) * (v1 - v0 + 1) > 256) {
            only[k] = 1;
            continue;
        }
        for (let v = v0; v <= v1 && !only[k]; v++)
            for (let u = u0; u <= u1; u++)
                if (mask[u + nx * v]) {
                    only[k] = 1;
                    break;
                }
    }
    return only;
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
