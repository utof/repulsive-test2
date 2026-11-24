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
    /** Centroid of the group's nodes at pick-up: what it is turned about. */
    cx: number;
    cy: number;
    /** Current turn of the group about its centroid, in radians (0 = as picked up). */
    angle: number;
    /** `facingAngle` at pick-up: how far the group was from squarely facing its links where it
     *  stood. Turning keeps that much (see `setGroupOffset`), so a pick-up turns nothing. */
    angle0: number;
    /** Set by `hangGroupOn`: the link that was re-attached. When the links land it goes to the
     *  nearest node its end can SEE, if the node chosen while carrying is not one. */
    hung?: number;
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
    let cx = 0;
    let cy = 0;
    for (let k = 0; k < list.length; k++) {
        cx += home[2 * k] / list.length;
        cy += home[2 * k + 1] / list.length;
    }
    const move: GroupMove = {
        nodes: list,
        internal,
        connecting,
        points,
        home,
        dx: 0,
        dy: 0,
        cx,
        cy,
        angle: 0,
        angle0: 0,
    };
    move.angle0 = facingAngle(g, move);
    return move;
}

/**
 * Where point `k` of a carried group is: its place at pick-up, turned by `move.angle` about the
 * group's centroid and shifted by the offset. With no turn this is `home + offset` to the last
 * bit (a turn of exactly 0 must not perturb a plain move by rounding).
 */
function placed(move: GroupMove, k: number): [number, number] {
    if (move.angle === 0) return [move.home[2 * k] + move.dx, move.home[2 * k + 1] + move.dy];
    const c = Math.cos(move.angle);
    const s = Math.sin(move.angle);
    const qx = move.home[2 * k] - move.cx;
    const qy = move.home[2 * k + 1] - move.cy;
    return [move.cx + move.dx + c * qx - s * qy, move.cy + move.dy + s * qx + c * qy];
}

/**
 * The turn that makes a carried group FACE WHAT IT IS LINKED TO: the angle about its centroid
 * that brings the nodes carrying its links as close as possible to those links' other ends
 * (least sum of squared link lengths; in the plane that is one `atan2` — the 2-D case of the
 * orthogonal Procrustes problem). When there is nothing to go by (no links, or every linked
 * node sits at the centroid) it returns the answer that leaves the group turned as it is.
 * `setGroupOffset` and `hangGroupOn` apply it RELATIVE to its value at pick-up.
 * Why: a branch carried across a map keeps the direction it had, so set down on the other
 * side of its parent it points back INTO the map, its root on the far side and its link
 * through its own body. Turned by this rule its root is the point nearest the parent and the
 * branch opens away from it — what a mind map looks like. With several links it is the
 * compromise that keeps them shortest, which is also where they cross least.
 * @see docs/plans/2026-11-24-medial-sphere-method.md ("Turning and re-attaching")
 */
export function facingAngle(g: MedialGraph, move: GroupMove): number {
    const inGroup = new Map<number, number>();
    for (let k = 0; k < move.nodes.length; k++) inGroup.set(move.nodes[k], k);
    const cx = move.cx + move.dx;
    const cy = move.cy + move.dy;
    let cross = 0;
    let dot = 0;
    for (const e of move.connecting) {
        const ka = inGroup.get(g.edgeA[e]);
        const k = ka ?? (inGroup.get(g.edgeB[e]) as number);
        const outside = ka === undefined ? g.edgeA[e] : g.edgeB[e];
        const qx = move.home[2 * k] - move.cx;
        const qy = move.home[2 * k + 1] - move.cy;
        const ox = g.x[3 * outside] - cx;
        const oy = g.x[3 * outside + 1] - cy;
        cross += qx * oy - qy * ox;
        dot += qx * ox + qy * oy;
    }
    // Nothing to go by: the answer that leaves the group as it is turned now.
    if (Math.hypot(cross, dot) < 1e-12) return move.angle + (move.angle0 ?? 0);
    return Math.atan2(cross, dot);
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
export function setGroupOffset(
    g: MedialGraph,
    move: GroupMove,
    dx: number,
    dy: number,
    opts: { turn?: 'none' | 'links'; hold?: number } = {},
): void {
    move.dx = dx;
    move.dy = dy;
    // `turn: 'links'`: the group also turns so that it faces what it is linked to the way it
    // did where it was picked up (`facingAngle` now, minus `facingAngle` then), so what is
    // seen while carrying is how it will be set down. Default: it keeps its direction.
    // Why relative to the pick-up and not the best fit itself: a branch drawn a little askew
    // would otherwise jump straight the moment it is picked up; this way nothing turns until
    // the group has gone somewhere.
    if (opts.turn === 'links') {
        move.angle = facingAngle(g, move) - move.angle0;
        // `hold`: node of the group that (dx, dy) is the offset OF (the one under the pointer),
        // instead of the centroid's.
        const k = opts.hold === undefined ? -1 : move.nodes.indexOf(opts.hold);
        if (k >= 0) turnHolding(g, move, k, dx, dy);
    }
    for (let k = 0; k < move.points.length; k++) {
        const [px, py] = placed(move, k);
        g.x[3 * move.points[k]] = px;
        g.x[3 * move.points[k] + 1] = py;
    }
    for (const e of move.connecting) straighten(g, e);
}

/**
 * Turn a carried group to face its links (as `setGroupOffset` with `turn: 'links'`) while its
 * node number `k` stays at its pick-up place + (dx, dy) — under the pointer that holds it.
 * Why this needs a search: the turn depends on where the centroid is, and with a node held
 * the centroid is where the turn puts it. So the angle is a root of
 *     f(angle) = facing angle with the centroid placed for `angle` − angle at pick-up − angle
 * found by sampling the circle (90 samples) and bisecting the sign change nearest to the
 * current angle, so the group turns steadily instead of jumping between solutions. Plain
 * repeated substitution diverges as soon as the held node is farther from the centroid than
 * the centroid is from what the group is linked to (measured: 3.0 units off the pointer).
 * No root (the held node swings the centroid right round the link's other end): the group
 * keeps its current angle.
 */
function turnHolding(g: MedialGraph, move: GroupMove, k: number, dx: number, dy: number): void {
    const qx = move.home[2 * k] - move.cx;
    const qy = move.home[2 * k + 1] - move.cy;
    const place = (angle: number): void => {
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        move.dx = dx + qx - (c * qx - s * qy);
        move.dy = dy + qy - (s * qx + c * qy);
    };
    const wrap = (a: number): number => a - 2 * Math.PI * Math.round(a / (2 * Math.PI));
    const f = (angle: number): number => {
        place(angle);
        return wrap(facingAngle(g, move) - move.angle0 - angle);
    };
    const start = move.angle;
    const N = 90;
    const h = (2 * Math.PI) / N;
    let best = Number.NaN;
    let bestFar = Infinity;
    let fa = f(start);
    for (let i = 0; i < N; i++) {
        const a = start + i * h;
        const fb = f(a + h);
        // A sign change between two small values is a root; between two large ones it is the
        // jump of `wrap` at half a turn.
        if (fa * fb <= 0 && Math.abs(fa) < 1 && Math.abs(fb) < 1) {
            let lo = a;
            let hi = a + h;
            let flo = fa;
            for (let it = 0; it < 40; it++) {
                const mid = 0.5 * (lo + hi);
                const fm = f(mid);
                if (flo * fm <= 0) hi = mid;
                else {
                    lo = mid;
                    flo = fm;
                }
            }
            const root = 0.5 * (lo + hi);
            const far = Math.abs(wrap(root - start));
            if (far < bestFar) {
                bestFar = far;
                best = start + wrap(root - start);
            }
        }
        fa = fb;
    }
    move.angle = Number.isNaN(best) ? start : best;
    place(move.angle);
}

/**
 * RE-ATTACH a carried group: its link `edge` (one of `move.connecting`) leaves the node it
 * went to and goes to `to` instead — "take this branch off its parent and hang it on that
 * node". The graph itself changes (`edgeA`/`edgeB` and the segment list).
 * Why: moved far, a branch that keeps its parent has to keep a link across everything in
 * between, and each thing it crosses is a crossing for good. Hung on a node next to where it
 * is set down, a tree stays a tree drawn without crossings.
 * Throws if `to` is in the group, or already linked to the same node of the group.
 */
export function reattachGroup(g: MedialGraph, move: GroupMove, edge: number, to: number): void {
    if (!move.connecting.includes(edge))
        throw new Error(`medialGraphMove: edge ${edge} is not a link of the carried group`);
    if (!(to >= 0 && to < g.nodeCount)) throw new Error(`medialGraphMove: ${to} is not a node`);
    if (move.nodes.includes(to)) throw new Error('medialGraphMove: the new end is in the group');
    const aInside = move.nodes.includes(g.edgeA[edge]);
    const inside = aInside ? g.edgeA[edge] : g.edgeB[edge];
    for (let e = 0; e < g.edgeA.length; e++)
        if (
            e !== edge &&
            ((g.edgeA[e] === inside && g.edgeB[e] === to) ||
                (g.edgeB[e] === inside && g.edgeA[e] === to))
        )
            throw new Error('medialGraphMove: the group is already linked to that node');
    retarget(g, edge, aInside, to);
    straighten(g, edge);
}

/** Make the outside end of `edge` the node `to` (`aInside`: its `edgeA` end is the group's). */
function retarget(g: MedialGraph, edge: number, aInside: boolean, to: number): void {
    // A NEW segment list object: the flow keeps what it derives from the topology per list.
    const a = Int32Array.from(g.segs.a);
    const b = Int32Array.from(g.segs.b);
    if (aInside) {
        g.edgeB[edge] = to;
        b[edge * g.pieces + g.pieces - 1] = to;
    } else {
        g.edgeA[edge] = to;
        a[edge * g.pieces] = to;
    }
    g.segs = { a, b };
}

/**
 * At the moment a hung group's links land: if the straight link to the node chosen while
 * carrying would cross something, re-attach it to the NEAREST NODE ITS END CAN SEE (a straight
 * link that touches nothing but its two ends), among the 40 nearest. Nothing changes if the
 * chosen node is visible, or none is.
 * Why at landing and not when it is dropped: only now is the group at full size in the room
 * the rest has made, so only now is it known what its end can see. A branch dropped "next to"
 * a node is set down in a face of the drawing that need not touch that node; hung on the
 * nearest node regardless, a tree branch landed with 2 crossings, with this rule with none.
 */
function rehangOnVisible(
    g: MedialGraph,
    edge: number,
    carried: readonly number[],
    fullOf: Map<number, [number, number]>,
    tol: number,
): void {
    const { x, segs, pieces } = g;
    const aInside = fullOf.has(g.edgeA[edge]);
    const inside = aInside ? g.edgeA[edge] : g.edgeB[edge];
    const chosen = aInside ? g.edgeB[edge] : g.edgeA[edge];
    const skipEdge = new Uint8Array(g.edgeA.length);
    for (const e of carried) skipEdge[e] = 1;
    const linked = new Set<number>();
    for (let e = 0; e < g.edgeA.length; e++) {
        if (e === edge) continue;
        if (g.edgeA[e] === inside) linked.add(g.edgeB[e]);
        if (g.edgeB[e] === inside) linked.add(g.edgeA[e]);
    }
    const visible = (v: number): boolean => {
        for (let k = 0; k < segs.a.length; k++) {
            if (skipEdge[Math.floor(k / pieces)]) continue;
            const a = segs.a[k];
            const b = segs.b[k];
            if (a === inside || b === inside || a === v || b === v) continue;
            if (planarSegmentDistance(x, inside, v, a, b) <= tol) return false;
        }
        return true;
    };
    if (visible(chosen)) return;
    const near: [number, number][] = [];
    for (let v = 0; v < g.nodeCount; v++) {
        if (fullOf.has(v) || v === chosen || linked.has(v)) continue;
        near.push([(x[3 * v] - x[3 * inside]) ** 2 + (x[3 * v + 1] - x[3 * inside + 1]) ** 2, v]);
    }
    near.sort((p, q) => p[0] - q[0]);
    for (const [, v] of near.slice(0, 40))
        if (visible(v)) {
            retarget(g, edge, aInside, v);
            return;
        }
}

/**
 * HANG a carried group on node `to`: re-attach its link (`reattachGroup`), turn it to face
 * `to` the way it faced its old parent (`facingAngle`, less its value at pick-up), and — if it was brought closer than that — back it off along the line
 * from `to` through where it is held, until the node carrying the link is one link length
 * (the link's rest length) from `to`. So the side of `to` the group is dropped on is the side
 * it hangs on, and it never lands on top of its new parent.
 * `edge` defaults to the group's only link; a group with several links has to name one.
 */
export function hangGroupOn(g: MedialGraph, move: GroupMove, to: number, edge?: number): void {
    const e = edge ?? (move.connecting.length === 1 ? move.connecting[0] : -1);
    if (e < 0) throw new Error('medialGraphMove: name the link to re-attach (several, or none)');
    let natural = 0;
    for (let q = 0; q < g.pieces; q++) natural += g.rest[e * g.pieces + q];
    if (g.edgeA[e] !== to && g.edgeB[e] !== to) reattachGroup(g, move, e, to);
    move.hung = e;
    move.angle = facingAngle(g, move) - move.angle0;
    const k = move.nodes.indexOf(move.nodes.includes(g.edgeA[e]) ? g.edgeA[e] : g.edgeB[e]);
    const reach = Math.hypot(move.home[2 * k] - move.cx, move.home[2 * k + 1] - move.cy);
    const cx = move.cx + move.dx;
    const cy = move.cy + move.dy;
    const ox = g.x[3 * to] - cx;
    const oy = g.x[3 * to + 1] - cy;
    const far = Math.hypot(ox, oy);
    if (far > 0 && far < reach + natural) {
        const back = reach + natural - far;
        move.dx -= (back * ox) / far;
        move.dy -= (back * oy) / far;
    }
    for (let q = 0; q < move.points.length; q++) {
        const [px, py] = placed(move, q);
        g.x[3 * move.points[q]] = px;
        g.x[3 * move.points[q] + 1] = py;
    }
    for (const c of move.connecting) straighten(g, c);
}

/**
 * The node outside a carried group that is nearest to the point (x, y) — the natural node to
 * re-attach to when the group is set down there. −1 if the group is the whole graph.
 */
export function nearestOutsideNode(g: MedialGraph, move: GroupMove, x: number, y: number): number {
    const inGroup = new Uint8Array(g.nodeCount);
    for (const v of move.nodes) inGroup[v] = 1;
    let best = -1;
    let bestD = Infinity;
    for (let i = 0; i < g.nodeCount; i++) {
        if (inGroup[i]) continue;
        const d = (g.x[3 * i] - x) ** 2 + (g.x[3 * i + 1] - y) ** 2;
        if (d < bestD) {
            bestD = d;
            best = i;
        }
    }
    return best;
}

/**
 * Abandon the move: the group goes back to where it was picked up.
 * Why this is a set-down and not a plain restore: the rest kept flowing while the group was
 * away and may have moved into its place, so the same care is needed as for any other spot.
 */
export function cancelGroupMove(g: MedialGraph, move: GroupMove, opts: { r: number }): GroupDrop {
    move.angle = 0;
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
    /** The re-attached link, if the group was hung on a node (see `GroupMove.hung`). */
    hung?: number;
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
        const [px, py] = placed(move, k);
        full[2 * k] = px;
        full[2 * k + 1] = py;
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
            hung: move.hung,
        };
        return { scale, targets: fullOf, crossings: countCrossings(g), groupCrossings: 0, growth };
    }
    g.ghost = undefined;
    const groupCrossings = landLinks(g, move.connecting, fullOf, tol, move.hung);
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
    hung?: number,
): number {
    if (hung !== undefined) rehangOnVisible(g, hung, connecting, fullOf, tol);
    const { x, pieces } = g;
    const E = g.edgeA.length;
    const link = new Uint8Array(E);
    for (const e of connecting) {
        link[e] = 1;
        straighten(g, e);
        const inside = fullOf.get(g.edgeA[e]) ?? (fullOf.get(g.edgeB[e]) as [number, number]);
        const outside = fullOf.has(g.edgeA[e]) ? g.edgeB[e] : g.edgeA[e];
        const L = Math.hypot(inside[0] - x[3 * outside], inside[1] - x[3 * outside + 1]) / pieces;
        // What it may shrink back to as the drawing settles (`MedialGraph.slack`): the length
        // it had before the move, if that is shorter than where it lands.
        let before = 0;
        for (let q = 0; q < pieces; q++) before += g.rest[e * pieces + q];
        g.slack ??= new Map();
        if (before < pieces * L) g.slack.set(e, before);
        else g.slack.delete(e);
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
 *
 * `makeRoom` — what happens when the gaps allow less than the wanted rate: `'warp'` (default)
 * moves the rest of the drawing out of the way directly, by at most `roomStep·r` per call
 * (see `makeRoom` below); `'flow'` only waits for the flow's contact forces — the first
 * version, kept as the comparison.
 */
export function growGroup(
    g: MedialGraph,
    growth: GroupGrowth,
    r: number,
    opts: {
        rate?: number;
        margin?: number;
        makeRoom?: 'warp' | 'flow';
        roomStep?: number;
    } = {},
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
    // MAKE ROOM (default): where the gaps allow less than the wanted rate, move the REST out of
    // the way directly (see `makeRoom`) and grow by what that allows instead.
    const dsWanted = Math.min(1 - growth.scale, rate * growth.scale + 1e-3);
    if ((opts.makeRoom ?? 'warp') === 'warp' && ds < 0.999 * dsWanted) {
        const made = makeRoom(g, growth, r, (opts.roomStep ?? 1) * r, dsWanted, ds);
        if (made > 0) ds = made;
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
    landLinks(g, growth.connecting, growth.targets, growth.tol, growth.hung);
    return 1;
}

/**
 * Where a point at distance `rho` from the anchor goes when a disc of radius `c` around the
 * anchor is enlarged `k` times (k ≥ 1) and everything else has to make room for it.
 *
 *  - inside the disc (`rho ≤ c`): `k·rho` — a plain enlargement, so whatever is in there
 *    (the growing group, and anything lying between its branches) keeps its shape exactly;
 *  - far outside (`rho ≥ beta·c`): `sqrt(rho² + const)` — every ring keeps its AREA and just
 *    moves outwards, so nothing far away is squeezed; the shift dies off like 1/rho;
 *  - in between: the area magnification falls linearly from k² to 1.
 * Why the band in between: an enlargement stretches radially by k, an area-keeping push
 * compresses radially by 1/k; glued directly the map has a crease at the disc's rim, and a
 * straight segment across a crease does not stay where the map puts its ends. With the band
 * the map is smooth (continuous first derivative), increasing, and never shrinks an area.
 * @see docs/plans/2026-11-24-medial-sphere-method.md ("Making room directly")
 */
export function roomRadius(rho: number, c: number, k: number, beta = 1.5): number {
    if (rho <= c) return k * rho;
    const b = beta * c;
    const k2 = k * k;
    // ∫ 2ρ·m(ρ) dρ from 0, with m = k² up to c and linear from k² to 1 between c and b.
    const band = (q: number): number =>
        k2 * q * q + ((1 - k2) / (b - c)) * ((2 / 3) * (q ** 3 - c ** 3) - c * (q * q - c * c));
    return Math.sqrt(rho <= b ? band(rho) : band(b) + rho * rho - b * b);
}

/**
 * Does point `p` land on segment (`q`, `r`) at some moment while all three move in straight
 * lines from their place in `x0` to their place in `x1`? Exact for that motion: the signed
 * area of (q, r, p) is a quadratic in time; at each of its zeros in (0, 1] the point is on
 * the segment's line, and it is a hit if it is between the ends then.
 * Why exact and not the flow's "nobody moves more than 0.45 of the gap": making room moves
 * neighbours TOGETHER by many times the gap between them; a limit on each one's own motion
 * would forbid nearly all of it.
 */
function meetsWhileMoving(
    x0: Float64Array,
    x1: Float64Array,
    p: number,
    q: number,
    r: number,
): boolean {
    const e0x = x0[3 * r] - x0[3 * q];
    const e0y = x0[3 * r + 1] - x0[3 * q + 1];
    const w0x = x0[3 * p] - x0[3 * q];
    const w0y = x0[3 * p + 1] - x0[3 * q + 1];
    const e1x = x1[3 * r] - x1[3 * q] - e0x;
    const e1y = x1[3 * r + 1] - x1[3 * q + 1] - e0y;
    const w1x = x1[3 * p] - x1[3 * q] - w0x;
    const w1y = x1[3 * p + 1] - x1[3 * q + 1] - w0y;
    const f0 = e0x * w0y - e0y * w0x;
    const f1 = e0x * w1y - e0y * w1x + e1x * w0y - e1y * w0x;
    const f2 = e1x * w1y - e1y * w1x;
    const onSegmentAt = (t: number): boolean => {
        if (!(t > 0 && t <= 1)) return false;
        const ex = e0x + t * e1x;
        const ey = e0y + t * e1y;
        const along = (w0x + t * w1x) * ex + (w0y + t * w1y) * ey;
        const L = ex * ex + ey * ey;
        return along >= -1e-9 * L && along <= (1 + 1e-9) * L;
    };
    const scale = Math.abs(f0) + Math.abs(f1) + Math.abs(f2);
    if (scale === 0) return false;
    if (Math.abs(f2) <= 1e-14 * scale) return f1 !== 0 && onSegmentAt(-f0 / f1);
    const disc = f1 * f1 - 4 * f2 * f0;
    if (disc < 0) return false;
    // The numerically stable pair of roots.
    const qv = -0.5 * (f1 + (f1 >= 0 ? 1 : -1) * Math.sqrt(disc));
    return onSegmentAt(qv / f2) || (qv !== 0 && onSegmentAt(f0 / qv));
}

/**
 * MAKING ROOM DIRECTLY. Move everything that is not held along `roomRadius` so that the group
 * can grow from its scale `s` to `s + ds`, and report the `ds` that was possible (0 = none;
 * then nothing was moved). The group's own points are NOT moved here (the caller places them
 * by the scale); they are only taken into account.
 *
 * Why this exists: waiting for the flow to push the rest away grows a group only as fast as
 * the TIGHTEST gap around it opens, and a big group has a long boundary — a 134-node branch
 * of a 600-node tree went from 0.25 to 0.45 of its size in 400 steps and stopped. The room a
 * group needs is known in advance (a disc around its anchor, `k` times bigger), so the rest
 * can be moved there in one smooth motion of the whole drawing.
 *
 * Why it cannot create a crossing: the trial positions are accepted only if, moving every
 * node in a straight line from old to new, no node meets a segment (`meetsWhileMoving`, for
 * every pair of segments within reach of each other; pairs farther apart than twice the
 * largest displacement cannot meet). On top of that no gap under half a clearance may lose
 * more than 30% (a looser bound let a gap close geometrically over a few calls), and pairs
 * that are allowed to cross may not cross more often. A trial that fails is halved, down to
 * `dsFloor` (what the gaps allow anyway).
 * Held nodes (pinned by the caller) stay where they are; the check then simply limits how
 * far their neighbours may go.
 *
 * Tried and dropped: clearing a region of the group's own OUTLINE (its reach from the anchor
 * per direction) instead of a whole disc, so that a group that is not round moves less of
 * the drawing. Measured: the rest was a little less distorted (mean length error 0.059
 * against 0.071 on the 600-node tree), but the sheared band along the outline's flanks made
 * the check refuse — the 2×2 block on the 12×12 grid stopped at 0.61 of its size — and left
 * more crossings and a tenth of the clearance on the tree. The disc is the robust one.
 */
function makeRoom(
    g: MedialGraph,
    growth: GroupGrowth,
    r: number,
    maxMove: number,
    dsWanted: number,
    dsFloor: number,
): number {
    const { points, full, ax, ay, internalSeg } = growth;
    const { x, segs, pieces } = g;
    const n = g.n;
    const ghost = g.ghost;
    let radius = 0;
    for (let k = 0; k < points.length; k++)
        radius = Math.max(radius, Math.hypot(full[2 * k] - ax, full[2 * k + 1] - ay));
    if (radius === 0) return 0;
    const s = growth.scale;
    // The enlarged disc reaches 2r beyond the group. Why: the group's farthest point would
    // otherwise sit exactly on the rim, where the map changes character; a segment of the rest
    // lying across the rim next to it lags a little behind it at every call, and the gap
    // between the two closed geometrically (measured: 5e-7 after 40 calls, and the group
    // stopped at 0.998 of its size). With the margin everything within the flow's reach of
    // the group is enlarged WITH it, and only the rest's own segments cross the rim.
    const c = s * radius + 2 * r;
    const isGroup = new Uint8Array(n);
    for (let k = 0; k < points.length; k++) isGroup[points[k]] = 1;
    const x1 = new Float64Array(x.length);
    let pairs: Int32Array | null = null;
    let gaps: Float64Array | null = null;
    // 1.06: the largest displacement is a little beyond the disc's rim (see `roomRadius`).
    let ds = Math.min(dsWanted, (maxMove * s) / (1.06 * c));
    for (let trial = 0; trial < 6 && ds > dsFloor; trial++, ds *= 0.5) {
        const k = (s + ds) / s;
        x1.set(x);
        let moved = 0;
        for (let i = 0; i < n; i++) {
            if (isGroup[i]) continue;
            if (g.pinned[i] || ghost?.node[i]) continue; // held, or a carried link's point
            const dx = x[3 * i] - ax;
            const dy = x[3 * i + 1] - ay;
            const rho = Math.hypot(dx, dy);
            if (rho === 0) continue;
            const f = roomRadius(rho, c, k) / rho;
            x1[3 * i] = ax + f * dx;
            x1[3 * i + 1] = ay + f * dy;
            moved = Math.max(moved, (f - 1) * rho);
        }
        for (let q = 0; q < points.length; q++) {
            x1[3 * points[q]] = ax + (s + ds) * (full[2 * q] - ax);
            x1[3 * points[q] + 1] = ay + (s + ds) * (full[2 * q + 1] - ay);
            moved = Math.max(moved, ds * Math.hypot(full[2 * q] - ax, full[2 * q + 1] - ay));
        }
        if (!pairs || !gaps) {
            // Every pair that could meet in the FIRST (largest) trial; smaller trials move
            // less, so the same list serves them.
            const list: number[] = [];
            const dist: number[] = [];
            const E = g.edgeA.length;
            forNearSegmentPairs(x, n, segs, 2.2 * moved, (e, f) => {
                if (internalSeg[e] && internalSeg[f]) return; // both of the group: same shape
                const e1 = Math.floor(e / pieces);
                const e2 = Math.floor(f / pieces);
                if (ghost && (ghost.edge[e1] || ghost.edge[e2])) return;
                if (g.passPairs?.has(Math.min(e1, e2) * E + Math.max(e1, e2))) return;
                list.push(e, f);
                dist.push(planarSegmentDistance(x, segs.a[e], segs.b[e], segs.a[f], segs.b[f]));
            });
            pairs = Int32Array.from(list);
            gaps = Float64Array.from(dist);
        }
        let ok = true;
        for (let q = 0; ok && q < gaps.length; q++) {
            const a1 = segs.a[pairs[2 * q]];
            const b1 = segs.b[pairs[2 * q]];
            const a2 = segs.a[pairs[2 * q + 1]];
            const b2 = segs.b[pairs[2 * q + 1]];
            ok =
                !meetsWhileMoving(x, x1, a1, a2, b2) &&
                !meetsWhileMoving(x, x1, b1, a2, b2) &&
                !meetsWhileMoving(x, x1, a2, a1, b1) &&
                !meetsWhileMoving(x, x1, b2, a1, b1) &&
                planarSegmentDistance(x1, a1, b1, a2, b2) >= Math.min(0.7 * gaps[q], 0.5 * r);
        }
        if (!ok) continue;
        const old = Float64Array.from(x);
        x.set(x1);
        let within = true;
        if (g.passPairs) {
            const E = g.edgeA.length;
            for (const [key, allowed] of g.passPairs)
                if (crossingsBetween(g, Math.floor(key / E), key % E) > allowed) {
                    within = false;
                    break;
                }
        }
        if (within) return ds;
        x.set(old);
    }
    return 0;
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
