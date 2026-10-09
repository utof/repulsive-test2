/**
 * R^6 tangent-point BVH for the Phase 2H BCT spike (issue utof/repulsive-test2#57).
 *
 * Port of the spike that produced every `out-bct-*.json` oracle (tasks 3-4 test against them);
 * keep the structure exactly -- "cleaning up" a field invalidates those oracles.
 *
 * Faithful to Yu/Schumacher/Crane App. B.1.1: primitives are tangent-points
 * `p_I = (x_I, T_I) in R^6`; a node is split by "cycling through all six coordinates,
 * choosing a splitting plane that minimizes the sum of squared diameters of the two
 * child bounding boxes"; each node stores
 *     L_N = sum l_I,  x_N = sum l_I x_I / L_N,  T_N = sum l_I T_I / L_N
 * plus spatial and tangential radii r_x^N, r_T^N.
 *
 * TWO DELIBERATE DEVIATIONS, both recorded so the 2H spec does not re-derive them:
 *
 * 1. The splitting plane is the per-axis MEDIAN (via quickselect), not the box centre.
 *    The paper does not say which candidate plane it scores. Median guarantees a
 *    balanced tree (depth = ceil(log2(m/leafSize))) and therefore an honest O(n log n)
 *    build; a centre split can degenerate to O(n) depth on a non-uniform curve and
 *    would make the measured constant a property of the fixture, not the algorithm.
 *
 * 2. r_x^N is the EXACT `max_{I in N} |x_I - x_N|` (distance from the barycentre),
 *    not the bounding-box half-diagonal. It is the quantity the Taylor argument
 *    behind Eq. 27/28 actually needs, it is never larger than the box radius, and it
 *    is O(1) extra work at build time. Same for r_T^N with `|T_I - T_N|`.
 *
 * `maxEll` per node exists for the adjacency exclusion in the block cluster tree (task 3). Do not delete it: without it, a two-edge leaf pair of
 * ADJACENT edges can satisfy Eq. 28 and get lumped into the far field, silently
 * adding an interaction the discrete energy explicitly omits (arXiv:2006.07859
 * line "we simply omit neighboring edge" pairs; repo: `calculateDisjointPairs`,
 * src/core/tangentPointEnergy.ts:44).
 *
 * @see https://arxiv.org/abs/2006.07859 App. B.1.1
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.5 (recorded deviations)
 * @issue utof/repulsive-test2#57
 */
import type { Edge, Vec3 } from '../../testConfigs';

/**
 * Per-edge primitives, edge-id indexed: midpoint `P`, unit tangent `T`, regularised length `ell`.
 * Why: shared by the BVH and the task-4 near field so both see one discretisation.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.5
 */
export interface EdgeData {
    m: number;
    P: Float64Array;
    T: Float64Array;
    ell: Float64Array;
}

/**
 * Builds {@link EdgeData}. `ell = ||e|| + epsilon` with epsilon added AFTER the norm; `T` is the
 * UNregularised direction, zero when `||e|| < 1e-14`.
 * Why: bit-level agreement with `assembleAFlat` (task 4 pin, spec §5.4 item 4).
 * @see src/core/sobolev/innerProduct.ts:70 (`ell[I] = r + epsilon`, confirmed) and :71-76 (tangent guard)
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.5
 */
export function edgeData(vertices: Vec3[], edges: Edge[], epsilon: number): EdgeData {
    const m = edges.length;
    const P = new Float64Array(3 * m);
    const T = new Float64Array(3 * m);
    const ell = new Float64Array(m);
    for (let I = 0; I < m; I++) {
        const a = vertices[edges[I][0]];
        const b = vertices[edges[I][1]];
        const ex = b[0] - a[0];
        const ey = b[1] - a[1];
        const ez = b[2] - a[2];
        const r = Math.sqrt(ex * ex + ey * ey + ez * ez);
        ell[I] = r + epsilon; // ε AFTER the norm, as innerProduct.ts:70 (spec §5.4 item 4).
        P[3 * I] = 0.5 * (a[0] + b[0]);
        P[3 * I + 1] = 0.5 * (a[1] + b[1]);
        P[3 * I + 2] = 0.5 * (a[2] + b[2]);
        if (r >= 1e-14) {
            const inv = 1 / r;
            T[3 * I] = ex * inv;
            T[3 * I + 1] = ey * inv;
            T[3 * I + 2] = ez * inv;
        } // else degenerate direction: zero vector, as innerProduct.ts:71.
    }
    return { m, P, T, ell };
}

/**
 * The tree and slot-ordered edge data. Fields `maxEll`, `sXa`, `sXb` must NOT be deleted as unused.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.5
 * @issue utof/repulsive-test2#57
 */
export interface Bvh {
    nNodes: number;
    /** `-1` for a leaf. */
    left: Int32Array;
    right: Int32Array;
    /** Slot range `[begin, end)` into `order`. */
    begin: Int32Array;
    end: Int32Array;
    /** `order[slot] = original edge index`. */
    order: Int32Array;
    /** Total mass `L_N`. */
    L: Float64Array;
    /** Barycentre `x_N`, flat xyz per node. */
    C: Float64Array;
    /** Length-weighted average tangent `T_N` (NOT renormalised — per the paper). */
    TN: Float64Array;
    /** Spatial radius `max |x_I - x_N|`. */
    rx: Float64Array;
    /** Tangential radius `max |T_I - T_N|`. */
    rt: Float64Array;
    /** Largest edge length in the node — adjacency-exclusion certificate. Do not delete (spec §5.5). */
    maxEll: Float64Array;
    depth: number;
    leafCount: number;
    leafSize: number;
    /** Edge data permuted into slot order (cache locality for the near field). */
    sP: Float64Array;
    sT: Float64Array;
    sEll: Float64Array;
    /** `sE[2*slot], sE[2*slot+1]` — endpoint vertex ids, in slot order. */
    sE: Int32Array;
    /**
     * Endpoint COORDINATES in slot order (`sXa` = first endpoint, `sXb` = second).
     * Needed by the `endpoint4` near-field quadrature, which is the one the repo's
     * `assembleBHighFlat`/`assembleBLowFlat` actually use (4 endpoint pairs per edge
     * pair, not the midpoint). Do not drop these to save 48 B/edge: without them the
     * near field silently uses a different discretization than `src/`.
     * @see src/core/sobolev/innerProduct.ts:88-101
     * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.5 (mandated, measured 33.9 % error)
     */
    sXa: Float64Array;
    sXb: Float64Array;
    buildMs: number;
    /** Deterministic byte count of every array retained by the tree. */
    bytes: number;
}

/** Quickselect `a[lo..hi]` (inclusive) about position `k`, keyed by `key6[a[i]*6+axis]`. */
function selectK(
    a: Int32Array,
    lo: number,
    hi: number,
    k: number,
    key6: Float64Array,
    axis: number,
): void {
    let l = lo;
    let h = hi;
    while (l < h) {
        // Median-of-three pivot: guards against the already-sorted input that a
        // sequentially-indexed polyline hands us on the very first split.
        const mid = (l + h) >> 1;
        const kl = key6[a[l] * 6 + axis];
        const km = key6[a[mid] * 6 + axis];
        const kh = key6[a[h] * 6 + axis];
        let pIdx = mid;
        if ((kl <= km && km <= kh) || (kh <= km && km <= kl)) pIdx = mid;
        else if ((km <= kl && kl <= kh) || (kh <= kl && kl <= km)) pIdx = l;
        else pIdx = h;
        const pivot = key6[a[pIdx] * 6 + axis];
        let i = l;
        let j = h;
        while (i <= j) {
            while (key6[a[i] * 6 + axis] < pivot) i++;
            while (key6[a[j] * 6 + axis] > pivot) j--;
            if (i <= j) {
                const t = a[i];
                a[i] = a[j];
                a[j] = t;
                i++;
                j--;
            }
        }
        if (k <= j) h = j;
        else if (k >= i) l = i;
        else return;
    }
}

/** Sum of squared bbox diameters over all six coordinates for slots `[b, e)`. */
function boxScore(a: Int32Array, b: number, e: number, key6: Float64Array): number {
    const lo = [Infinity, Infinity, Infinity, Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity, -Infinity, -Infinity, -Infinity];
    for (let s = b; s < e; s++) {
        const o = a[s] * 6;
        for (let c = 0; c < 6; c++) {
            const v = key6[o + c];
            if (v < lo[c]) lo[c] = v;
            if (v > hi[c]) hi[c] = v;
        }
    }
    let sum = 0;
    for (let c = 0; c < 6; c++) {
        const d = hi[c] - lo[c];
        sum += d * d;
    }
    return sum;
}

/**
 * Builds the R^6 tangent-point BVH.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.5
 * @issue utof/repulsive-test2#57
 * @param epsilon regularisation added after the edge norm (see {@link edgeData}).
 * @param leafSize maximum primitives in a leaf; a first-class cost knob (it trades
 *   near-field pair count against tree size) and therefore swept by the bench.
 */
export function buildBvh(vertices: Vec3[], edges: Edge[], leafSize: number, epsilon: number): Bvh {
    const t0 = performance.now(); // not Bun.nanoseconds: src/ runs in browsers/workers.
    const ed = edgeData(vertices, edges, epsilon);
    const m = ed.m;
    // R^6 tangent-points, edge-id indexed.
    const key6 = new Float64Array(6 * m);
    for (let I = 0; I < m; I++) {
        key6[6 * I] = ed.P[3 * I];
        key6[6 * I + 1] = ed.P[3 * I + 1];
        key6[6 * I + 2] = ed.P[3 * I + 2];
        key6[6 * I + 3] = ed.T[3 * I];
        key6[6 * I + 4] = ed.T[3 * I + 1];
        key6[6 * I + 5] = ed.T[3 * I + 2];
    }

    // Median splitting stops at count <= leafSize, so every leaf holds > leafSize/2
    // primitives => fewer than 2m/leafSize leaves => fewer than 4m/leafSize nodes.
    const maxNodes = 4 * Math.ceil(m / Math.max(1, leafSize)) + 64;
    const left = new Int32Array(maxNodes).fill(-1);
    const right = new Int32Array(maxNodes).fill(-1);
    const begin = new Int32Array(maxNodes);
    const end = new Int32Array(maxNodes);
    const order = new Int32Array(m);
    for (let i = 0; i < m; i++) order[i] = i;

    let nNodes = 0;
    let depth = 0;
    let leafCount = 0;
    const scratch = new Int32Array(m);

    const alloc = (b: number, e: number): number => {
        const id = nNodes++;
        begin[id] = b;
        end[id] = e;
        return id;
    };

    const build = (b: number, e: number, d: number): number => {
        const id = alloc(b, e);
        if (d > depth) depth = d;
        if (e - b <= leafSize) {
            leafCount++;
            return id;
        }
        const mid = (b + e) >> 1;
        let bestAxis = 0;
        let bestScore = Infinity;
        for (let axis = 0; axis < 6; axis++) {
            scratch.set(order.subarray(b, e), b);
            selectK(scratch, b, e - 1, mid, key6, axis);
            const sc = boxScore(scratch, b, mid, key6) + boxScore(scratch, mid, e, key6);
            if (sc < bestScore) {
                bestScore = sc;
                bestAxis = axis;
            }
        }
        selectK(order, b, e - 1, mid, key6, bestAxis);
        const l = build(b, mid, d + 1);
        const r = build(mid, e, d + 1);
        left[id] = l;
        right[id] = r;
        return id;
    };

    build(0, m, 0);

    // Cluster data, bottom-up over the (post-order-compatible) node array.
    const L = new Float64Array(nNodes);
    const C = new Float64Array(3 * nNodes);
    const TN = new Float64Array(3 * nNodes);
    const rx = new Float64Array(nNodes);
    const rt = new Float64Array(nNodes);
    const maxEll = new Float64Array(nNodes);

    // Permuted (slot-order) primitive arrays.
    const sP = new Float64Array(3 * m);
    const sT = new Float64Array(3 * m);
    const sEll = new Float64Array(m);
    const sE = new Int32Array(2 * m);
    const sXa = new Float64Array(3 * m);
    const sXb = new Float64Array(3 * m);
    for (let s = 0; s < m; s++) {
        const I = order[s];
        sP[3 * s] = ed.P[3 * I];
        sP[3 * s + 1] = ed.P[3 * I + 1];
        sP[3 * s + 2] = ed.P[3 * I + 2];
        sT[3 * s] = ed.T[3 * I];
        sT[3 * s + 1] = ed.T[3 * I + 1];
        sT[3 * s + 2] = ed.T[3 * I + 2];
        sEll[s] = ed.ell[I];
        const i1 = edges[I][0];
        const i2 = edges[I][1];
        sE[2 * s] = i1;
        sE[2 * s + 1] = i2;
        for (let c = 0; c < 3; c++) {
            sXa[3 * s + c] = vertices[i1][c];
            sXb[3 * s + c] = vertices[i2][c];
        }
    }

    for (let id = nNodes - 1; id >= 0; id--) {
        const b = begin[id];
        const e = end[id];
        let mass = 0;
        let cx = 0;
        let cy = 0;
        let cz = 0;
        let tx = 0;
        let ty = 0;
        let tz = 0;
        let me = 0;
        for (let s = b; s < e; s++) {
            const w = sEll[s];
            mass += w;
            cx += w * sP[3 * s];
            cy += w * sP[3 * s + 1];
            cz += w * sP[3 * s + 2];
            tx += w * sT[3 * s];
            ty += w * sT[3 * s + 1];
            tz += w * sT[3 * s + 2];
            if (w > me) me = w;
        }
        const inv = mass > 0 ? 1 / mass : 0;
        cx *= inv;
        cy *= inv;
        cz *= inv;
        tx *= inv;
        ty *= inv;
        tz *= inv;
        let mrx = 0;
        let mrt = 0;
        for (let s = b; s < e; s++) {
            const dx = sP[3 * s] - cx;
            const dy = sP[3 * s + 1] - cy;
            const dz = sP[3 * s + 2] - cz;
            const d2 = dx * dx + dy * dy + dz * dz;
            if (d2 > mrx) mrx = d2;
            const ex = sT[3 * s] - tx;
            const ey = sT[3 * s + 1] - ty;
            const ez = sT[3 * s + 2] - tz;
            const e2 = ex * ex + ey * ey + ez * ez;
            if (e2 > mrt) mrt = e2;
        }
        L[id] = mass;
        C[3 * id] = cx;
        C[3 * id + 1] = cy;
        C[3 * id + 2] = cz;
        TN[3 * id] = tx;
        TN[3 * id + 1] = ty;
        TN[3 * id + 2] = tz;
        rx[id] = Math.sqrt(mrx);
        rt[id] = Math.sqrt(mrt);
        maxEll[id] = me;
    }

    const buildMs = performance.now() - t0;
    const bytes =
        4 * (left.length + right.length + begin.length + end.length + order.length + sE.length) +
        8 *
            (L.length +
                C.length +
                TN.length +
                rx.length +
                rt.length +
                maxEll.length +
                sP.length +
                sT.length +
                sEll.length +
                sXa.length +
                sXb.length);

    return {
        nNodes,
        left: left.subarray(0, nNodes),
        right: right.subarray(0, nNodes),
        begin: begin.subarray(0, nNodes),
        end: end.subarray(0, nNodes),
        order,
        L,
        C,
        TN,
        rx,
        rt,
        maxEll,
        depth,
        leafCount,
        leafSize,
        sP,
        sT,
        sEll,
        sE,
        sXa,
        sXb,
        buildMs,
        bytes,
    };
}
