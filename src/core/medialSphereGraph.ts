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
import { makeGrid, nodeMotionCaps, type SegmentList } from './medialSphere';
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
}

/** Build the flow's graph: every edge cut into `pieces` segments, rest lengths as drawn. */
export function medialGraphFrom(vertices: Vec3[], edges: Edge[], pieces: number): MedialGraph {
    const g = subdivideGraph(vertices, edges, pieces);
    const n = g.vertices.length;
    const x = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) {
        x[3 * i] = g.vertices[i][0];
        x[3 * i + 1] = g.vertices[i][1];
    }
    const m = g.edges.length;
    const a = new Int32Array(m);
    const b = new Int32Array(m);
    const rest = new Float64Array(m);
    for (let k = 0; k < m; k++) {
        a[k] = g.edges[k][0];
        b[k] = g.edges[k][1];
        rest[k] = Math.hypot(x[3 * b[k]] - x[3 * a[k]], x[3 * b[k] + 1] - x[3 * a[k] + 1]);
    }
    return {
        n,
        x,
        segs: { a, b },
        rest,
        nodeCount: g.nodeCount,
        pieces: g.pieces,
        edgeA: Int32Array.from(edges, (e) => e[0]),
        edgeB: Int32Array.from(edges, (e) => e[1]),
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
    /** CSR of the segments by their first endpoint. */
    segStart: Int32Array;
    segByFirst: Int32Array;
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
    const segStart = new Int32Array(n + 1);
    for (let k = 0; k < m; k++) segStart[segs.a[k] + 1]++;
    for (let i = 0; i < n; i++) segStart[i + 1] += segStart[i];
    const segFill = Int32Array.from(segStart.subarray(0, n));
    const segByFirst = new Int32Array(m);
    for (let k = 0; k < m; k++) segByFirst[segFill[segs.a[k]]++] = k;
    const t = { nbrStart, nbr, own, edgeKeys, segStart, segByFirst };
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
    const tanHalfMin = Math.tan(((opts.minAngleDeg ?? 30) * Math.PI) / 360);
    const beta = opts.beta ?? 0.01;
    const lambda = opts.lambda ?? 0.02;
    const tau = opts.tau ?? 1;
    // The paper's empirical thickness–weight relation (its reference code: r ≈ 3.36/√α).
    const alpha = (3.36 / r) ** 2;

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
    let maxSeg = 0;
    for (let k = 0; k < m; k++) maxSeg = Math.max(maxSeg, len[k]);
    const segReach = 2 * rMax + maxSeg;
    const grid = makeGrid(x, n, segReach);
    const reach2 = (2 * rMax) ** 2;
    const maxDir = maxDirections(t, n);
    const dirX = new Float64Array(maxDir);
    const dirY = new Float64Array(maxDir);
    const rad = new Float64Array(maxDir);
    const hit = new Int32Array(maxDir);
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
        // Every segment that starts at a node within reach: its first endpoint is at most
        // 2·rMax + (its own length) from anything it can affect.
        grid.forNear(i, segReach, (j0) => {
            for (let q = t.segStart[j0]; q < t.segStart[j0 + 1]; q++) testSegment(t.segByFirst[q]);
        });
        // forNear skips the node itself: segments starting AT node i contain it and are
        // skipped by the rule below anyway.
        function testSegment(k: number): void {
            const sa = segs.a[k];
            const sb = segs.b[k];
            if (sa === i || sb === i) return;
            const e = Math.floor(k / g.pieces);
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
            const px = x[3 * sa] - xi;
            const py = x[3 * sa + 1] - yi;
            const ux = x[3 * sb] - x[3 * sa];
            const uy = x[3 * sb + 1] - x[3 * sa + 1];
            const A = ux * ux + uy * uy;
            const B = 2 * (px * ux + py * uy);
            const C = px * px + py * py;
            // Cheap rejection: the segment's nearest point is beyond reach of any capped circle.
            const sNear = A > 0 ? Math.max(0, Math.min(1, -B / (2 * A))) : 0;
            if (A * sNear * sNear + B * sNear + C >= reach2) return;
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
            if (hit[l] >= 0) stiffSum += gain * (1 + (barrier * (rMax - Rl)) / Rl);
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
    for (let i = 0; i < n; i++)
        if (pinned[i]) {
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
    for (let i = 0; i < n; i++) diag[i] = (lambda + contact[i]) * alpha * w[i];
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
        for (let i = 0; i < n; i++) if (pinned[i]) out[i] = v[i];
    };
    const d = new Float64Array(3 * n);
    const rhs = new Float64Array(n);
    const sol = new Float64Array(n);
    let cgIterations = 0;
    for (let c = 0; c < 2; c++) {
        for (let i = 0; i < n; i++) rhs[i] = G[3 * i + c];
        cgIterations = Math.max(cgIterations, jacobiCg(applyP, diag, rhs, sol));
        for (let i = 0; i < n; i++) d[3 * i + c] = pinned[i] ? 0 : sol[i];
    }

    // --- move, each node within its own crossing-free limit ------------------------------------
    // Not one global step: see `nodeMotionCaps` for why (a single tight spot froze the graph).
    const caps = nodeMotionCaps(x, n, segs, 2 * rMax);
    let move = 0;
    let capped = 0;
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
 * arbitrarily close near it — neither says anything about the drawing's clearance.
 */
export function minClearance(g: MedialGraph, searchWithin: number): number {
    const incident = (e: number, f: number): boolean =>
        e === f ||
        g.edgeA[e] === g.edgeA[f] ||
        g.edgeA[e] === g.edgeB[f] ||
        g.edgeB[e] === g.edgeA[f] ||
        g.edgeB[e] === g.edgeB[f];
    const { n, x, segs } = g;
    const m = segs.a.length;
    let maxEdge = 0;
    for (let k = 0; k < m; k++)
        maxEdge = Math.max(
            maxEdge,
            Math.hypot(
                x[3 * segs.b[k]] - x[3 * segs.a[k]],
                x[3 * segs.b[k] + 1] - x[3 * segs.a[k] + 1],
            ),
        );
    const reach = searchWithin + 2 * maxEdge;
    const grid = makeGrid(x, n, reach);
    const start = new Int32Array(n + 1);
    for (let k = 0; k < m; k++) start[segs.a[k] + 1]++;
    for (let i = 0; i < n; i++) start[i + 1] += start[i];
    const fill = Int32Array.from(start.subarray(0, n));
    const byFirst = new Int32Array(m);
    for (let k = 0; k < m; k++) byFirst[fill[segs.a[k]]++] = k;
    let best = searchWithin;
    for (let e = 0; e < m; e++) {
        const a1 = segs.a[e];
        const b1 = segs.b[e];
        grid.forNear(a1, reach, (j) => {
            for (let q = start[j]; q < start[j + 1]; q++) {
                const f = byFirst[q];
                if (f <= e) continue;
                const a2 = segs.a[f];
                const b2 = segs.b[f];
                if (incident(Math.floor(e / g.pieces), Math.floor(f / g.pieces))) continue;
                best = Math.min(best, planarSegmentDistance(x, a1, b1, a2, b2));
            }
        });
    }
    return best;
}

/** Distance between two segments in the plane: 0 if they cross, else the least endpoint–segment distance. */
function planarSegmentDistance(
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
