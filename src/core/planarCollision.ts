/**
 * Continuous collision test for a flat graph: does moving every vertex in a straight line from
 * `from` to `to` make any vertex pass through any edge?
 *
 * Why it exists. The tangent-point energy is infinite when two strands touch, and that barrier
 * is what keeps a curve in its isotopy class — but only for a CONTINUOUS motion. A descent step
 * is a finite jump, and a jump can land on the far side of a thin barrier with a lower energy.
 * The paper's guarantee comes from the time step, not from the energy: "use the first time step
 * τ_max at which a collision occurs as the starting point for a line search, which guarantees
 * that the curve remains in the same isotopy class"; the simpler rule we shipped first "comes
 * with no collision guarantees" (its words). Measured on flat graphs with that simpler rule
 * (`bench/results/2026-11-23-subdivision-crossings.jsonl`): the number of crossings changed on
 * the first step from a drawing that already had crossings, at 1, 2, 4 AND 8 segments per edge
 * — so subdividing edges alone does not hold the topology.
 * @see local_files/repulsive_orig_paper/SelfAvoiding.tex (line 639, "Time Stepping")
 *
 * How it is used: as the line search's third acceptance test (`LineSearchOptions.guard`), on
 * the PROJECTED candidate — the motion actually applied, constraint projection included. A
 * trial that fails is backtracked like a failed Armijo test, which bounds the step by the
 * first collision without computing τ_max explicitly.
 *
 * Scope: the plane only (every vertex of both configurations at one z). In the plane the only
 * way two edges that share no vertex can start or stop crossing is a vertex of one passing
 * through the other, which is a root of a quadratic. The 3-D edge–edge case is a cubic and is
 * NOT here: on a curve that leaves the plane this returns `false` (no opinion).
 * Not covered either: two edges at the same node swapping their order around it.
 */
import type { Edge, Vec3 } from './testConfigs';

/** Tolerance on "the same z" for a configuration to count as flat. */
const FLAT_TOL = 1e-12;
/** A root this close to t = 0 is the starting configuration itself, not an event. */
const T_START = 1e-12;
/**
 * Smallest distance, as a fraction of the mean edge length, that a candidate may leave between
 * a vertex and an edge it does not belong to.
 * Why a floor at all: where a drawing ALREADY has a crossing, the energy pushes the offending
 * vertex back through the edge it crossed. The pass-through test refuses that, the line search
 * halves the step, and the vertex creeps up to the edge geometrically — measured on a
 * 63-node tree, it got within rounding distance in 21 steps, and from there the event is
 * indistinguishable from "already touching at t = 0" and went through. With the floor the creep
 * stops while the event is still well resolved, and the line search reports `collision`.
 * A pair that is already closer than the floor at the start is exempt (it must be allowed to
 * move apart).
 */
const CLEARANCE_REL = 1e-6;

/** Are both configurations in one plane z = const? */
export function isFlat(from: readonly Vec3[], to: readonly Vec3[]): boolean {
    if (from.length === 0) return false;
    const z = from[0][2];
    for (let i = 0; i < from.length; i++) {
        if (Math.abs(from[i][2] - z) > FLAT_TOL || Math.abs(to[i][2] - z) > FLAT_TOL) return false;
    }
    return true;
}

/**
 * `true` when some vertex passes through some edge that it is not an endpoint of (and that
 * shares no vertex with the edge the test pairs it with) during the linear motion, or ends
 * closer to such an edge than the clearance floor (`CLEARANCE_REL`). Exact
 * O(|E|²) pairs with a swept bounding-box rejection first; a flat graph of ~1 500 segments
 * costs about 10–20 ms per call.
 */
export function planarMotionCollides(
    from: readonly Vec3[],
    to: readonly Vec3[],
    edges: readonly Edge[],
): boolean {
    if (!isFlat(from, to)) return false;
    const m = edges.length;
    let meanLen = 0;
    for (const [a, b] of edges)
        meanLen += Math.hypot(from[b][0] - from[a][0], from[b][1] - from[a][1]);
    const clearance2 = (CLEARANCE_REL * (m > 0 ? meanLen / m : 0)) ** 2;
    /** Squared distance from vertex `p` to segment (a, b) in configuration `V`. */
    const dist2 = (V: readonly Vec3[], p: number, a: number, b: number): number => {
        const sx = V[b][0] - V[a][0];
        const sy = V[b][1] - V[a][1];
        const qx = V[p][0] - V[a][0];
        const qy = V[p][1] - V[a][1];
        const len2 = sx * sx + sy * sy;
        const u = len2 === 0 ? 0 : Math.max(0, Math.min(1, (qx * sx + qy * sy) / len2));
        const ex = qx - u * sx;
        const ey = qy - u * sy;
        return ex * ex + ey * ey;
    };
    /** Does the candidate bring `p` under the clearance floor of (a, b), from above it? */
    const tooClose = (p: number, a: number, b: number): boolean =>
        dist2(to, p, a, b) < clearance2 && dist2(from, p, a, b) >= clearance2;
    // Swept bounding box of every segment (both endpoints, both configurations), grown by the
    // clearance floor: two boxes that merely come within the floor of each other must still be
    // examined, or the floor would never be tested for a pair that stops just short.
    const pad = Math.sqrt(clearance2);
    const box = new Float64Array(4 * m);
    for (let e = 0; e < m; e++) {
        const [a, b] = edges[e];
        box[4 * e] = Math.min(from[a][0], from[b][0], to[a][0], to[b][0]) - pad;
        box[4 * e + 1] = Math.max(from[a][0], from[b][0], to[a][0], to[b][0]) + pad;
        box[4 * e + 2] = Math.min(from[a][1], from[b][1], to[a][1], to[b][1]) - pad;
        box[4 * e + 3] = Math.max(from[a][1], from[b][1], to[a][1], to[b][1]) + pad;
    }
    /** Does vertex `p` pass through segment (a, b) for some t in (0, 1]? */
    const hits = (p: number, a: number, b: number): boolean => {
        // u(t) = B − A, w(t) = P − A, both linear in t; the vertex is on the LINE of the
        // segment when cross(u, w) = c0 + c1·t + c2·t² vanishes.
        const ux = from[b][0] - from[a][0];
        const uy = from[b][1] - from[a][1];
        const wx = from[p][0] - from[a][0];
        const wy = from[p][1] - from[a][1];
        const dux = to[b][0] - to[a][0] - ux;
        const duy = to[b][1] - to[a][1] - uy;
        const dwx = to[p][0] - to[a][0] - wx;
        const dwy = to[p][1] - to[a][1] - wy;
        const c0 = ux * wy - uy * wx;
        const c1 = ux * dwy - uy * dwx + dux * wy - duy * wx;
        const c2 = dux * dwy - duy * dwx;
        const onSegmentAt = (t: number): boolean => {
            if (!(t > T_START && t <= 1)) return false;
            const sx = ux + t * dux;
            const sy = uy + t * duy;
            const qx = wx + t * dwx;
            const qy = wy + t * dwy;
            const len2 = sx * sx + sy * sy;
            if (len2 === 0) return qx === 0 && qy === 0;
            const s = (qx * sx + qy * sy) / len2;
            return s >= 0 && s <= 1;
        };
        if (c2 === 0) return c1 !== 0 && onSegmentAt(-c0 / c1);
        const disc = c1 * c1 - 4 * c2 * c0;
        if (disc < 0) return false;
        // The numerically stable pair of roots (no cancellation in −c1 ± √disc).
        const q = -0.5 * (c1 + (c1 >= 0 ? 1 : -1) * Math.sqrt(disc));
        return onSegmentAt(q / c2) || (q !== 0 && onSegmentAt(c0 / q));
    };
    for (let i = 0; i < m; i++) {
        const [a, b] = edges[i];
        for (let j = i + 1; j < m; j++) {
            if (
                box[4 * j + 1] < box[4 * i] ||
                box[4 * j] > box[4 * i + 1] ||
                box[4 * j + 3] < box[4 * i + 2] ||
                box[4 * j + 2] > box[4 * i + 3]
            )
                continue;
            const [c, d] = edges[j];
            if (a === c || a === d || b === c || b === d) continue;
            if (hits(c, a, b) || hits(d, a, b) || hits(a, c, d) || hits(b, c, d)) return true;
            if (tooClose(c, a, b) || tooClose(d, a, b) || tooClose(a, c, d) || tooClose(b, c, d))
                return true;
        }
    }
    return false;
}
