/**
 * Moving a group of nodes ACROSS a flat drawing — the mind-map move "this blob does not belong
 * here, put it over there and let the rest make room".
 *
 * Why this is not a drag: the graph flow never lets one edge pass through another, so it keeps
 * the drawing's embedding — which face every node sits in. A group pulled through the flow
 * therefore drags the whole map along and can never get to another region (measured: a corner
 * block pulled to the centre of a 12×12 grid sheared the grid into a diamond). To change region
 * the group has to leave the plane for a moment:
 *
 *  1. PICK UP (`beginGroupMove`): the group's nodes and every edge touching them become
 *     "carried" — invisible to the flow, exerting no force. The rest keeps flowing.
 *  2. CARRY (`setGroupOffset`): the caller puts the group where the pointer is.
 *  3. SET DOWN (`dropGroupMove`): the group is shrunk about a point until it fits inside ONE
 *     face of the rest of the drawing, and becomes real again, still held by the caller.
 *  4. GROW BACK (`growGroup`, once per flow step): the group is scaled up rigidly, each time
 *     by as much as the room around it allows; the flow's contact forces push the rest away
 *     from it, which gives it more room — "the rest makes room" is the existing flow. At
 *     full size the group is released, exactly where and as it was put, and only then do its
 *     links to the rest become real.
 *     Why the links wait: from a group shrunk to a dot they leave as a tight fan, a few
 *     hundredths apart, each asking for a full clearance from the others and from everything
 *     near — forces that nothing can satisfy while the group is small (measured on a grid:
 *     the room around the group SHRANK from 0.09 to 0.02 and it never grew).
 *     Why rigid and not "let its springs inflate it" (the first version): edges squeezed to
 *     an eighth of their length do not stretch back, they buckle — after 600 steps a moved
 *     branch of a tree was a tangle of zigzags a third of the way out.
 *
 * The links then run straight across whatever lies between. The pairs of edges that cross at
 * that moment are recorded as ALLOWED to cross, that many times (`MedialGraph.passPairs`);
 * every other pair stays guarded. So the crossings left are the ones the move made, and their
 * number cannot grow.
 * The connecting edges take their new length as rest length: the user redrew them.
 * @see docs/plans/2026-11-24-medial-sphere-method.md ("The graph flow in the app, and the drag")
 * @see src/core/medialSphereGraph.ts (the flow; `ghost` and `passPairs` on its graph)
 */
import { forNearSegmentPairs, nodeMotionCaps } from './medialSphere';
import {
    crossingsBetween,
    type MedialGraph,
    passKey,
    planarSegmentDistance,
} from './medialSphereGraph';

/** A group that has been picked up and not yet set down. */
export interface GroupMove {
    /** The group's original nodes. */
    nodes: number[];
    /** Original edges with both ends in the group (they keep their shape and rest lengths). */
    internal: number[];
    /** Original edges with exactly one end in the group, and which end (0 = edgeA) is inside. */
    connecting: number[];
    /** Points that move rigidly with the group: its nodes and the interior points of its
     *  internal edges; `home` is their xy at pick-up. */
    points: Int32Array;
    home: Float64Array;
    /** Current offset of the group from `home`. */
    dx: number;
    dy: number;
}

/** Index of point `q` (0 … pieces) along original edge `e` (subdivideGraph's layout). */
function nodeAt(g: MedialGraph, e: number, q: number): number {
    return q === 0
        ? g.edgeA[e]
        : q === g.pieces
          ? g.edgeB[e]
          : g.nodeCount + e * (g.pieces - 1) + q - 1;
}

/**
 * Pick the group up: from now until `dropGroupMove` (or `cancelGroupMove`) the flow ignores
 * its nodes and every edge touching them. One group at a time.
 * Why the edges too: an edge from the group to the rest has to sweep over the drawing while
 * the group travels; guarded, it would stop the move at the first edge in its way.
 */
export function beginGroupMove(g: MedialGraph, nodes: readonly number[]): GroupMove {
    if (g.ghost) throw new Error('medialGraphMove: a group is already being moved');
    const inGroup = new Uint8Array(g.nodeCount);
    const list: number[] = [];
    for (const v of nodes) {
        if (!(v >= 0 && v < g.nodeCount))
            throw new Error(`medialGraphMove: ${v} is not an original node`);
        if (!inGroup[v]) list.push(v);
        inGroup[v] = 1;
    }
    const edge = new Uint8Array(g.edgeA.length);
    const node = new Uint8Array(g.n);
    const internal: number[] = [];
    const connecting: number[] = [];
    const pts: number[] = [...list];
    for (const v of list) node[v] = 1;
    for (let e = 0; e < g.edgeA.length; e++) {
        const ina = inGroup[g.edgeA[e]];
        const inb = inGroup[g.edgeB[e]];
        if (!ina && !inb) continue;
        edge[e] = 1;
        (ina && inb ? internal : connecting).push(e);
        for (let q = 1; q < g.pieces; q++) {
            node[nodeAt(g, e, q)] = 1;
            if (ina && inb) pts.push(nodeAt(g, e, q));
        }
    }
    const points = Int32Array.from(pts);
    const home = new Float64Array(2 * points.length);
    for (let k = 0; k < points.length; k++) {
        home[2 * k] = g.x[3 * points[k]];
        home[2 * k + 1] = g.x[3 * points[k] + 1];
    }
    g.ghost = { edge, node };
    return { nodes: list, internal, connecting, points, home, dx: 0, dy: 0 };
}

/** Put a connecting edge's interior points on the straight line between its ends. */
function straighten(g: MedialGraph, e: number): void {
    const a = g.edgeA[e];
    const b = g.edgeB[e];
    for (let q = 1; q < g.pieces; q++) {
        const i = nodeAt(g, e, q);
        const s = q / g.pieces;
        g.x[3 * i] = (1 - s) * g.x[3 * a] + s * g.x[3 * b];
        g.x[3 * i + 1] = (1 - s) * g.x[3 * a + 1] + s * g.x[3 * b + 1];
    }
}

/**
 * Carry the group to `home + (dx, dy)`: rigid for the group, straight lines for the edges that
 * tie it to the rest (whose other ends are wherever the flow has them now).
 * Why straight: while carried these edges are only drawn; any shape would do.
 */
export function setGroupOffset(g: MedialGraph, move: GroupMove, dx: number, dy: number): void {
    move.dx = dx;
    move.dy = dy;
    for (let k = 0; k < move.points.length; k++) {
        g.x[3 * move.points[k]] = move.home[2 * k] + dx;
        g.x[3 * move.points[k] + 1] = move.home[2 * k + 1] + dy;
    }
    for (const e of move.connecting) straighten(g, e);
}

/**
 * Abandon the move: the group goes back to where it was picked up.
 * Why this is a set-down and not a plain restore: the rest kept flowing while the group was
 * away and may have moved into its place, so the same care is needed as for any other spot.
 */
export function cancelGroupMove(g: MedialGraph, move: GroupMove, opts: { r: number }): GroupDrop {
    setGroupOffset(g, move, 0, 0);
    return dropGroupMove(g, move, opts);
}

/** A group that was set down smaller than it is and is growing back (see `growGroup`). */
export interface GroupGrowth {
    /** The group's points, their full-size xy, and the point the scaling is about. */
    points: Int32Array;
    full: Float64Array;
    ax: number;
    ay: number;
    /** Current scale, below 1 until done. */
    scale: number;
    /** Per segment: 1 = of an edge inside the group. */
    internalSeg: Uint8Array;
    /** The group's links to the rest (still carried), and each group node's full-size place. */
    connecting: number[];
    targets: Map<number, [number, number]>;
    tol: number;
}

/** What `dropGroupMove` did. */
export interface GroupDrop {
    /** Scale the group was set down at (1 = full size, it fitted as it was). */
    scale: number;
    /** Where each group node belongs at full size: node → (x, y). */
    targets: Map<number, [number, number]>;
    /** The group still has to grow back to full size: pass this to `growGroup` once per flow
     *  step until it returns 1. `null` if it was set down at full size. */
    growth: GroupGrowth | null;
    /** Crossings the drawing has after the set-down, and how many of them involve the group.
     *  With `growth` pending the links are not down yet and neither number counts them: read
     *  `allowedCrossingsNow` once `growGroup` has returned 1. */
    crossings: number;
    groupCrossings: number;
}

/** Distance from the point (px, py) to segment `k`. */
function pointToSegment(g: MedialGraph, px: number, py: number, k: number): number {
    const a = g.segs.a[k];
    const b = g.segs.b[k];
    const ux = g.x[3 * b] - g.x[3 * a];
    const uy = g.x[3 * b + 1] - g.x[3 * a + 1];
    const l2 = ux * ux + uy * uy;
    const t =
        l2 > 0
            ? Math.max(0, Math.min(1, ((px - g.x[3 * a]) * ux + (py - g.x[3 * a + 1]) * uy) / l2))
            : 0;
    return Math.hypot(px - g.x[3 * a] - t * ux, py - g.x[3 * a + 1] - t * uy);
}

/**
 * Set the group down at its current offset. See the file header for the method.
 * `opts.r` is the flow's clearance scale (only used for tolerances; NaN → the drawing's size).
 *
 * How the set-down scale is chosen: scale 1 if the group's own edges then cross nothing;
 * otherwise halved until they do not, and at the latest at the scale where the whole group
 * fits in the empty disc around its anchor point (half the anchor's distance to the nearest
 * edge of the rest), which crosses nothing by construction. The anchor is whichever of the
 * group's centre and its nodes has the most room.
 */
export function dropGroupMove(g: MedialGraph, move: GroupMove, opts: { r: number }): GroupDrop {
    const ghost = g.ghost;
    if (!ghost) throw new Error('medialGraphMove: no group is being moved');
    const { x, segs, pieces } = g;
    const m = segs.a.length;
    const np = move.points.length;
    // Full-size positions of the group's points.
    const full = new Float64Array(2 * np);
    let cx = 0;
    let cy = 0;
    for (let k = 0; k < np; k++) {
        full[2 * k] = move.home[2 * k] + move.dx;
        full[2 * k + 1] = move.home[2 * k + 1] + move.dy;
    }
    const ng = move.nodes.length; // the group's nodes come first in `points`
    for (let k = 0; k < ng; k++) {
        cx += full[2 * k] / ng;
        cy += full[2 * k + 1] / ng;
    }
    let extent = 0;
    for (let i = 0; i < g.n; i++)
        extent = Math.max(extent, Math.abs(x[3 * i]), Math.abs(x[3 * i + 1]));
    const tol = 1e-9 * (Number.isFinite(opts.r) ? opts.r : extent || 1);

    // The rest of the drawing: every segment of an edge that is not carried.
    const restSegs: number[] = [];
    for (let k = 0; k < m; k++) if (!ghost.edge[Math.floor(k / pieces)]) restSegs.push(k);
    const roomAt = (px: number, py: number): number => {
        let best = Infinity;
        for (const k of restSegs) best = Math.min(best, pointToSegment(g, px, py, k));
        return best;
    };
    // Anchor: the candidate with the most room; if every one sits on an edge, step off it.
    let ax = cx;
    let ay = cy;
    let room = roomAt(cx, cy);
    for (let k = 0; k < ng; k++) {
        const c = roomAt(full[2 * k], full[2 * k + 1]);
        if (c > room) {
            room = c;
            ax = full[2 * k];
            ay = full[2 * k + 1];
        }
    }
    if (room <= 1e3 * tol) {
        const step = Number.isFinite(opts.r) ? 0.05 * opts.r : 1e-3 * (extent || 1);
        for (let q = 0; q < 8; q++) {
            const px = cx + step * Math.cos((q * Math.PI) / 4 + 0.3);
            const py = cy + step * Math.sin((q * Math.PI) / 4 + 0.3);
            const c = roomAt(px, py);
            if (c > room) {
                room = c;
                ax = px;
                ay = py;
            }
        }
    }
    let radius = 0;
    for (let k = 0; k < np; k++)
        radius = Math.max(radius, Math.hypot(full[2 * k] - ax, full[2 * k + 1] - ay));
    const place = (s: number): void => {
        for (let k = 0; k < np; k++) {
            x[3 * move.points[k]] = ax + s * (full[2 * k] - ax);
            x[3 * move.points[k] + 1] = ay + s * (full[2 * k + 1] - ay);
        }
    };
    // Does the group (its internal edges; its nodes if it has none) touch the rest at scale s?
    const groupSegs: number[] = [];
    for (const e of move.internal) for (let q = 0; q < pieces; q++) groupSegs.push(e * pieces + q);
    const touches = (): boolean => {
        for (const k of groupSegs)
            for (const f of restSegs) {
                if (
                    segs.a[k] === segs.a[f] ||
                    segs.a[k] === segs.b[f] ||
                    segs.b[k] === segs.a[f] ||
                    segs.b[k] === segs.b[f]
                )
                    continue;
                if (planarSegmentDistance(x, segs.a[k], segs.b[k], segs.a[f], segs.b[f]) <= tol)
                    return true;
            }
        for (let k = 0; k < ng; k++)
            for (const f of restSegs)
                if (pointToSegment(g, x[3 * move.points[k]], x[3 * move.points[k] + 1], f) <= tol)
                    return true;
        return false;
    };
    const fits = radius > 0 ? Math.min(1, (0.5 * room) / radius) : 1;
    let scale = 1;
    place(scale);
    while (scale > fits && touches()) {
        scale = Math.max(fits, 0.5 * scale);
        place(scale);
    }

    const fullOf = new Map<number, [number, number]>();
    for (let k = 0; k < ng; k++) fullOf.set(move.nodes[k], [full[2 * k], full[2 * k + 1]]);
    const E = g.edgeA.length;
    if (scale < 1) {
        // Held by the caller until it is back at full size: its own edges are real (the rest
        // is pushed off them), its points fixed, its links still carried.
        const edge = new Uint8Array(E);
        const node = new Uint8Array(g.n);
        for (let k = 0; k < np; k++) node[move.points[k]] = 1;
        for (const e of move.connecting) {
            edge[e] = 1;
            straighten(g, e);
            for (let q = 1; q < pieces; q++) node[nodeAt(g, e, q)] = 1;
        }
        g.ghost = { edge, node };
        refreshPassPairs(g, tol);
        const internalSeg = new Uint8Array(m);
        for (const e of move.internal) internalSeg.fill(1, e * pieces, (e + 1) * pieces);
        const growth: GroupGrowth = {
            points: move.points,
            full,
            ax,
            ay,
            scale,
            internalSeg,
            connecting: move.connecting,
            targets: fullOf,
            tol,
        };
        return { scale, targets: fullOf, crossings: countCrossings(g), groupCrossings: 0, growth };
    }
    g.ghost = undefined;
    const groupCrossings = landLinks(g, move.connecting, fullOf, tol);
    return { scale, targets: fullOf, crossings: countCrossings(g), groupCrossings, growth: null };
}

/**
 * Make a group's links real: straight, as long as drawn from now on (measured to the
 * full-size place of the group's end), and allowed exactly the crossings they have at this
 * moment. Returns how many crossings involve them. `g.ghost` must already be cleared.
 */
function landLinks(
    g: MedialGraph,
    connecting: readonly number[],
    fullOf: Map<number, [number, number]>,
    tol: number,
): number {
    const { x, pieces } = g;
    const E = g.edgeA.length;
    const link = new Uint8Array(E);
    for (const e of connecting) {
        link[e] = 1;
        straighten(g, e);
        const inside = fullOf.get(g.edgeA[e]) ?? (fullOf.get(g.edgeB[e]) as [number, number]);
        const outside = fullOf.has(g.edgeA[e]) ? g.edgeB[e] : g.edgeA[e];
        const L = Math.hypot(inside[0] - x[3 * outside], inside[1] - x[3 * outside + 1]) / pieces;
        for (let q = 0; q < pieces; q++) g.rest[e * pieces + q] = L;
    }
    // The crossings that exist NOW are the allowed ones (for pairs already allowed this can
    // only lower the count).
    const pairs = refreshPassPairs(g, tol);
    let mine = 0;
    for (const [key, count] of pairs) if (link[Math.floor(key / E)] || link[key % E]) mine += count;
    return mine;
}

/**
 * One step of growing a set-down group back to full size; call it before each flow step.
 * Returns the new scale; at 1 the group is released (the flow moves it again), its links are
 * put down, and `growth` is finished.
 *
 * How far it grows in one call: every point of the group moves outwards from the anchor in
 * proportion to its distance from it, and no point may cover more than
 * `0.45 · gap · min(1, gap / (margin·r))` of the gap between the group's edges and the rest
 * (nor the scale grow by more than `rate` of itself). 0.45·gap is the flow's own motion limit,
 * so nothing is crossed. Why the second factor: a group marched outwards at the full limit
 * closes every gap ahead of it by 90% a step and is pressed against the first edge in no time
 * (measured with a dragged node); with it the gap closes ever more slowly the smaller it is,
 * while the push on the rest keeps growing — the rest gives way before they touch. Why not
 * simply stop at a fixed gap: on a grid the rest settled closer than any margin worth having,
 * and the group never grew at all.
 */
export function growGroup(
    g: MedialGraph,
    growth: GroupGrowth,
    r: number,
    opts: { rate?: number; margin?: number } = {},
): number {
    if (growth.scale >= 1) return 1;
    const rate = opts.rate ?? 0.25;
    const margin = (opts.margin ?? 0.5) * r;
    const within = 2 * r;
    const { points, full, ax, ay, internalSeg } = growth;
    const { x, segs, pieces } = g;
    const ghostEdge = g.ghost?.edge;
    // Gap of each point of the group to the rest, through the group's own edges (frac = 1
    // makes the "limit" the distance itself, capped at `within`).
    const gap = nodeMotionCaps(x, g.n, segs, within, 1, (k1, k2) => {
        if (internalSeg[k1] === internalSeg[k2]) return true;
        return (
            !!ghostEdge &&
            (ghostEdge[Math.floor(k1 / pieces)] || ghostEdge[Math.floor(k2 / pieces)]) === 1
        );
    });
    let ds = Math.min(1 - growth.scale, rate * growth.scale + 1e-3);
    for (let k = 0; k < points.length; k++) {
        const rho = Math.hypot(full[2 * k] - ax, full[2 * k + 1] - ay);
        if (rho === 0) continue;
        const d = gap[points[k]];
        ds = Math.min(ds, (0.45 * d * Math.min(1, d / margin)) / rho);
    }
    growth.scale = growth.scale + ds >= 1 - 1e-12 ? 1 : growth.scale + ds;
    for (let k = 0; k < points.length; k++) {
        x[3 * points[k]] = ax + growth.scale * (full[2 * k] - ax);
        x[3 * points[k] + 1] = ay + growth.scale * (full[2 * k + 1] - ay);
    }
    if (growth.scale < 1) {
        for (const e of growth.connecting) straighten(g, e); // carried: only drawn
        return growth.scale;
    }
    g.ghost = undefined;
    landLinks(g, growth.connecting, growth.targets, growth.tol);
    return 1;
}

/** Visit every pair of segments sharing no node whose distance is at most `tol`. */
function forTouchingSegments(
    g: MedialGraph,
    tol: number,
    visit: (k1: number, k2: number) => void,
): void {
    const { x, segs, pieces } = g;
    const ghostEdge = g.ghost?.edge;
    forNearSegmentPairs(x, g.n, segs, tol, (e, f) => {
        // Carried edges are not in the drawing.
        if (ghostEdge && (ghostEdge[Math.floor(e / pieces)] || ghostEdge[Math.floor(f / pieces)]))
            return;
        if (planarSegmentDistance(x, segs.a[e], segs.b[e], segs.a[f], segs.b[f]) <= tol)
            visit(e, f);
    });
}

/**
 * Set `g.passPairs` to the crossings the drawing has now. Pairs that were allowed before and no
 * longer cross stay in it with an allowance of 0 until `prunePassPairs` finds them apart —
 * dropped at once, a pair still touching would be frozen by its motion limit.
 */
function refreshPassPairs(g: MedialGraph, tol: number): Map<number, number> {
    const now = crossingPairs(g, tol);
    for (const key of g.passPairs?.keys() ?? []) if (!now.has(key)) now.set(key, 0);
    g.passPairs = now;
    return now;
}

/**
 * The pairs of original edges that cross (or touch within `tol`) right now, with how many
 * times, as a `passPairs` map. Why "touch" counts: a pair at distance zero that is held apart
 * by the motion limits would have a limit of zero, and its nodes would never move again.
 * Use it once when a drawing that already has crossings enters the flow, and at a set-down.
 */
export function crossingPairs(g: MedialGraph, tol = 0): Map<number, number> {
    const out = new Map<number, number>();
    forTouchingSegments(g, tol, (k1, k2) => {
        const key = passKey(g, Math.floor(k1 / g.pieces), Math.floor(k2 / g.pieces));
        out.set(key, (out.get(key) ?? 0) + 1);
    });
    return out;
}

/**
 * The drawing's crossing count, from the allowed pairs alone (every crossing is between an
 * allowed pair). Why next to `countCrossings`: this is cheap enough to show every frame.
 */
export function allowedCrossingsNow(g: MedialGraph): number {
    if (!g.passPairs) return 0;
    const E = g.edgeA.length;
    let c = 0;
    for (const key of g.passPairs.keys()) c += crossingsBetween(g, Math.floor(key / E), key % E);
    return c;
}

/** Number of segment pairs that cross right now (the drawing's crossing count). */
export function countCrossings(g: MedialGraph): number {
    let c = 0;
    forTouchingSegments(g, 0, () => {
        c++;
    });
    return c;
}

/**
 * Forget pairs that no longer cross and have come at least `minDistance` apart; returns how
 * many. Why: from then on the motion limits hold them apart like any other pair, and the
 * crossing is gone for good. Why a distance and not "as soon as they part": a pair handed to
 * the motion limits at distance zero would be frozen by them.
 */
export function prunePassPairs(g: MedialGraph, minDistance: number): number {
    if (!g.passPairs || g.passPairs.size === 0) return 0;
    const E = g.edgeA.length;
    const { x, segs, pieces } = g;
    let removed = 0;
    for (const key of [...g.passPairs.keys()]) {
        const e = Math.floor(key / E);
        const f = key % E;
        if (crossingsBetween(g, e, f) > 0) continue;
        let d = Infinity;
        for (let p = 0; p < pieces && d > minDistance; p++)
            for (let q = 0; q < pieces; q++) {
                const k1 = e * pieces + p;
                const k2 = f * pieces + q;
                if (k1 === k2) continue;
                const a1 = segs.a[k1];
                const b1 = segs.b[k1];
                const a2 = segs.a[k2];
                const b2 = segs.b[k2];
                if (a1 === a2 || a1 === b2 || b1 === a2 || b1 === b2) continue;
                d = Math.min(d, planarSegmentDistance(x, a1, b1, a2, b2));
            }
        if (d > minDistance) {
            g.passPairs.delete(key);
            removed++;
        }
    }
    return removed;
}
