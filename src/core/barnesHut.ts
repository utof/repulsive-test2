/**
 * Barnes–Hut evaluation of the tangent-point energy and its differential: O(n log n) instead of
 * the all-pairs O(n²) of `tangentPointEnergy.ts`, and with NO `disjointPairs` list.
 *
 * Why this exists: measured on the shipped code (`bench/results/2026-11-23-h-step-scale.jsonl`),
 * a descent step on the hierarchical solver at n = 4 000 spends 16.8 s of 17.5 s in the all-pairs
 * differential and energy, and at n = 30 000 the step cannot be set up at all because
 * `disjointPairs` is n rows of n − 3 entries. The solver (Phase 2H) already scales; this is the
 * other half of the paper's acceleration scheme.
 *
 * Method (Yu–Schumacher–Crane, arXiv:2006.07859 §6.1 and App. B.1; local TeX
 * `local_files/repulsive_orig_paper/SelfAvoiding.tex:1001-1054`). For each target edge `I` the
 * R⁶ tangent-point BVH is walked from the root:
 *  - a LEAF is summed exactly, edge by edge, skipping edges that share a vertex with `I`
 *    (the discrete energy omits neighbouring pairs — `calculateDisjointPairs`);
 *  - an ADMISSIBLE node `N` is replaced by its mass `L_N`, barycentre `x_N` and average tangent
 *    `T_N`;
 *  - anything else is split into its two children.
 *
 * What is approximated, precisely. The repo's energy is
 *     E = ½ Σ_{I,J disjoint, ordered} w_IJ,
 *     w_IJ = ¼ ℓ_I^{1−α} ℓ_J Σ_{i∈I, j∈J} f(γ_i − γ_j, e_I),   f(d, e) = (‖e×d‖+ε)^α / (‖d‖+ε)^β.
 * The far field keeps the target's two endpoints and its tangent EXACT and lumps only the source:
 *     w_IN = ½ ℓ_I^{1−α} L_N Σ_{i∈I} f(γ_i − x_N, e_I).
 * The first-order error vanishes because `x_N` is the mass-weighted mean of the edge midpoints, so
 * the error is O((r_x/d)²). This is one step more accurate than the paper's Eq. 22, which also
 * collapses the target to its midpoint; it costs one extra kernel per far node and keeps the far
 * field on the same vertex quadrature as the near field.
 *
 * THE DIFFERENTIAL IS NOT THE GRADIENT OF THE APPROXIMATE ENERGY, by the paper's own choice
 * (App. B.1, "Differential": "Rather than differentiate our Barnes-Hut approximation of E, we
 * approximate the differential of the (full) discrete energy directly"). Each target edge
 * contributes only the derivatives with respect to ITS OWN two vertices, of both orderings
 * `w_IN + w_NI`; cluster data is treated as constant, which is exact for the true energy because
 * `I` is never inside an admissible node. Consequence: `barnesHutEnergy` and `barnesHutGradient`
 * agree with each other only to the approximation error, so an Armijo test fed both sees that
 * error as noise. @issue utof/repulsive-test2#17 records the same fact for the Python oracle.
 *
 * This file is NOT under the golden bit-identity suites (those pin `tangentPointEnergy.ts`), so it
 * is free to use `**` and its own summation order. At `theta = 0` nothing is admissible and both
 * functions reproduce the all-pairs values to rounding (pinned in `test/core/barnesHut.test.ts`).
 */
import { type Bvh, buildBvh } from './sobolev/hierarchical/bvh';
import type { Edge, Vec3 } from './testConfigs';

/**
 * Default opening parameter.
 * Why 0.25: measured in `bench/results/2026-11-23-bh-step-scale.jsonl` (trefoil, per-vertex
 * differential), one descent step at this θ is accepted at full step length at every size from
 * n = 960 to n = 30 000, and the EXACT energy falls by 1.55 at n = 960–8 000 and by 1.48 at
 * n = 30 000 — the all-pairs step at n = 960 gives 1.545. It is not a converged constant: the
 * differential's relative L² error at fixed θ grows with n (1.45e-2 at n = 960, 3.8e-2 at
 * n = 4 000), and the energy carries a +0.3–0.5 % offset. Lower it if a line search starts
 * rejecting steps near convergence; the bench takes θ as an argument.
 */
export const DEFAULT_BH_THETA = 0.25;

/**
 * Default BVH leaf size. Why 8: it is the value the Phase 2H solver builds its tree with
 * (`hierarchicalSaddle.ts` `DEFAULT_LEAF_SIZE`), so one tree can serve both.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.5
 */
export const DEFAULT_BH_LEAF_SIZE = 8;

/** Options of {@link prepareBarnesHut}. Why: θ and leaf size are the two cost/accuracy knobs. */
export interface BarnesHutOptions {
    /** Opening parameter θ ≥ 0. `0` disables the far field entirely (exact, O(n²)). */
    theta?: number;
    /**
     * Bound on a node's tangential radius `r_T`, used by the differential only. Defaults to
     * `theta`, which is the paper's single-ε form of Eq. 23.
     */
    thetaT?: number;
    /**
     * How the differential walks the tree. Default `'vertex'`; `'edge'` is the paper's literal
     * scheme, kept as a measured comparison. See {@link barnesHutGradient} for why they differ.
     * `'auto'` is `'vertex'` unless a sampled count says the tree hardly engages, in which case
     * the differential is summed EXACTLY, pair by pair (see `exactWalkIsCheaper`).
     */
    traversal?: 'vertex' | 'edge' | 'auto';
    leafSize?: number;
    /**
     * Test-only defect switch, in the style of `SchurOptions.knockOut`.
     * `'no-adjacency-check'` lets a node that contains an edge sharing a vertex with the target be
     * lumped into the far field — the defect the adjacency check exists to prevent. Why it is
     * here: a guard never observed failing is not a verified guard.
     */
    knockOut?: 'no-adjacency-check';
}

/** Work counters filled by a traversal. Why: proves a test exercised the far field at all. */
export interface BarnesHutStats {
    /** (target edge, admissible node) pairs replaced by a cluster. */
    farNodes: number;
    /** (target edge, source edge) pairs summed exactly. */
    nearPairs: number;
}

/**
 * Everything one geometry needs for any number of energy / differential evaluations.
 * Why a context object: E₀ and dE are evaluated at the same vertices, and the tree is the same
 * tree the Phase 2H solver builds, so it should be built once and passed around.
 */
export interface BarnesHutContext {
    /** Vertex count. */
    n: number;
    /** Edge count. */
    m: number;
    /** Vertex coordinates, flat xyz. */
    X: Float64Array;
    /** Edge endpoints in edge-id order: `E[2I]`, `E[2I+1]`. */
    E: Int32Array;
    bvh: Bvh;
    /** `slotOf[edge] = slot`, the inverse of `bvh.order`. */
    slotOf: Int32Array;
    /** CSR of edges incident to each vertex: `incEdges[incStart[v] .. incStart[v+1])`. */
    incStart: Int32Array;
    incEdges: Int32Array;
    theta: number;
    thetaT: number;
    traversal: 'vertex' | 'edge' | 'auto';
    epsilon: number;
    checkAdjacency: boolean;
}

/**
 * Builds the tree and the adjacency index for one geometry.
 * @see https://arxiv.org/abs/2006.07859 App. B.1.1 (the BVH), and `bvh.ts` for its two deviations
 */
export function prepareBarnesHut(
    vertices: Vec3[],
    edges: Edge[],
    epsilon: number,
    opts: BarnesHutOptions = {},
): BarnesHutContext {
    const n = vertices.length;
    const m = edges.length;
    const theta = opts.theta ?? DEFAULT_BH_THETA;
    if (!(theta >= 0)) throw new RangeError(`prepareBarnesHut: theta must be >= 0, got ${theta}`);

    const X = new Float64Array(3 * n);
    for (let v = 0; v < n; v++) {
        X[3 * v] = vertices[v][0];
        X[3 * v + 1] = vertices[v][1];
        X[3 * v + 2] = vertices[v][2];
    }
    const E = new Int32Array(2 * m);
    const incStart = new Int32Array(n + 1);
    for (let I = 0; I < m; I++) {
        E[2 * I] = edges[I][0];
        E[2 * I + 1] = edges[I][1];
        incStart[edges[I][0] + 1]++;
        incStart[edges[I][1] + 1]++;
    }
    for (let v = 0; v < n; v++) incStart[v + 1] += incStart[v];
    const incEdges = new Int32Array(2 * m);
    const fill = incStart.slice(0, n);
    for (let I = 0; I < m; I++) {
        incEdges[fill[E[2 * I]]++] = I;
        incEdges[fill[E[2 * I + 1]]++] = I;
    }

    const bvh = buildBvh(vertices, edges, opts.leafSize ?? DEFAULT_BH_LEAF_SIZE, epsilon);
    const slotOf = new Int32Array(m);
    for (let s = 0; s < m; s++) slotOf[bvh.order[s]] = s;

    return {
        n,
        m,
        X,
        E,
        bvh,
        slotOf,
        incStart,
        incEdges,
        theta,
        thetaT: opts.thetaT ?? theta,
        traversal: opts.traversal ?? 'vertex',
        epsilon,
        checkAdjacency: opts.knockOut !== 'no-adjacency-check',
    };
}

/**
 * May node `N` stand in for its edges, as seen from target edge `(i1, i2)` with midpoint `c`?
 *
 * Three conditions, all required:
 *  1. spatial (paper Eq. 23, first half): `r_x ≤ θ · (|c − x_N| − ℓ_I/2)`. The `ℓ_I/2` is there
 *     because the far field is evaluated at the target's ENDPOINTS, which sit up to half an edge
 *     closer to the cluster than the midpoint does.
 *  2. tangential (Eq. 23, second half): `r_T ≤ θ`, only when `needTangent`. The energy's far term
 *     uses the target's own tangent and never the cluster's, so it does not need this; the
 *     differential's reverse term `w_NI` uses `T_N` and does.
 *  3. adjacency: `N` contains neither the target nor any edge sharing a vertex with it. Geometry
 *     alone does not guarantee this — a node of very short edges next to a long target edge has a
 *     small `r_x` and passes (1). Lumping it would add an interaction the discrete energy omits.
 *     Checked exactly, through the slot range, so it holds for junctions and open ends too.
 *
 * `theta === 0` is never admissible, including the `r_x = 0` nodes condition (1) would let through:
 * it is the exact-reference setting and must stay structurally exact.
 */
function admissible(
    ctx: BarnesHutContext,
    N: number,
    cx: number,
    cy: number,
    cz: number,
    halfEll: number,
    i1: number,
    i2: number,
    needTangent: boolean,
): boolean {
    const theta = ctx.theta;
    if (theta === 0) return false;
    const bvh = ctx.bvh;
    if (needTangent && bvh.rt[N] > ctx.thetaT) return false;
    const dx = cx - bvh.C[3 * N];
    const dy = cy - bvh.C[3 * N + 1];
    const dz = cz - bvh.C[3 * N + 2];
    const reach = Math.sqrt(dx * dx + dy * dy + dz * dz) - halfEll;
    if (!(reach > 0 && bvh.rx[N] <= theta * reach)) return false;
    if (!ctx.checkAdjacency) return true;
    const b = bvh.begin[N];
    const e = bvh.end[N];
    const { incStart, incEdges, slotOf } = ctx;
    for (let k = incStart[i1]; k < incStart[i1 + 1]; k++) {
        const s = slotOf[incEdges[k]];
        if (s >= b && s < e) return false;
    }
    for (let k = incStart[i2]; k < incStart[i2 + 1]; k++) {
        const s = slotOf[incEdges[k]];
        if (s >= b && s < e) return false;
    }
    return true;
}

/**
 * Are the exponents the app's defaults, (α, β) = (3, 6)? Then `x³` and `x⁶` are formed by
 * multiplication instead of `Math.pow`.
 * Why it matters: on a tight tangle the far field hardly engages, so an evaluation is ~n² exact
 * terms and `pow` was most of each one — measured on the 1 998-vertex dataset knot, the
 * gradient's kernel time fell by about a sixth (3.9 s → 3.3 s) and an energy evaluation
 * from 0.27 s to 0.18 s. Why it is allowed HERE and not in
 * `tangentPointEnergy.ts`: that file is the bit-exact reference (its per-exponent `pow` is
 * pinned by goldens); this module is an approximation whose results are compared with
 * tolerances, and the two forms differ in the last bit or two only.
 */
const CUBE_OVER_SIXTH = (alpha: number, beta: number): boolean => alpha === 3 && beta === 6;

/**
 * Barnes–Hut tangent-point energy. Same definition as `calculateEnergy` (ε after every norm, the
 * final `/2` for the doubly-counted ordered pairs); only the far field is approximated.
 * @see src/core/tangentPointEnergy.ts (calculateEnergy — the exact reference)
 * @see https://arxiv.org/abs/2006.07859 App. B.1.2
 */
export function barnesHutEnergy(
    ctx: BarnesHutContext,
    alpha: number,
    beta: number,
    stats?: BarnesHutStats,
): number {
    const { X, E, bvh, epsilon, m } = ctx;
    const { left, right, begin, end, sE, sEll, L, C } = bvh;
    const stack = new Int32Array(2 * bvh.depth + 8);
    let farNodes = 0;
    let nearPairs = 0;
    const cubeOverSixth = CUBE_OVER_SIXTH(alpha, beta);

    // f(d, e) = (‖e×d‖+ε)^α / (‖d‖+ε)^β — ε after the norm, as in calculateEnergy.
    const f = (ex: number, ey: number, ez: number, dx: number, dy: number, dz: number): number => {
        const cx = ey * dz - ez * dy;
        const cy = ez * dx - ex * dz;
        const cz = ex * dy - ey * dx;
        const cNorm = Math.sqrt(cx * cx + cy * cy + cz * cz) + epsilon;
        const dNorm = Math.sqrt(dx * dx + dy * dy + dz * dz) + epsilon;
        if (cubeOverSixth) {
            // (α, β) = (3, 6) by multiplication — see CUBE_OVER_SIXTH.
            const d2 = dNorm * dNorm;
            return (cNorm * cNorm * cNorm) / (d2 * d2 * d2);
        }
        return cNorm ** alpha / dNorm ** beta;
    };

    let total = 0;
    for (let I = 0; I < m; I++) {
        const i1 = E[2 * I];
        const i2 = E[2 * I + 1];
        const ax = X[3 * i1];
        const ay = X[3 * i1 + 1];
        const az = X[3 * i1 + 2];
        const bx = X[3 * i2];
        const by = X[3 * i2 + 1];
        const bz = X[3 * i2 + 2];
        const ex = bx - ax;
        const ey = by - ay;
        const ez = bz - az;
        const ellI = Math.sqrt(ex * ex + ey * ey + ez * ez) + epsilon;
        const mx = 0.5 * (ax + bx);
        const my = 0.5 * (ay + by);
        const mz = 0.5 * (az + bz);
        const halfEll = 0.5 * ellI;

        let acc = 0;
        let sp = 0;
        stack[sp++] = 0;
        while (sp > 0) {
            const N = stack[--sp];
            if (left[N] < 0) {
                for (let s = begin[N]; s < end[N]; s++) {
                    const j1 = sE[2 * s];
                    const j2 = sE[2 * s + 1];
                    // Not disjoint (this also drops I itself): the discrete energy omits the pair.
                    if (j1 === i1 || j1 === i2 || j2 === i1 || j2 === i2) continue;
                    const px = X[3 * j1];
                    const py = X[3 * j1 + 1];
                    const pz = X[3 * j1 + 2];
                    const qx = X[3 * j2];
                    const qy = X[3 * j2 + 1];
                    const qz = X[3 * j2 + 2];
                    const sumK =
                        f(ex, ey, ez, ax - px, ay - py, az - pz) +
                        f(ex, ey, ez, ax - qx, ay - qy, az - qz) +
                        f(ex, ey, ez, bx - px, by - py, bz - pz) +
                        f(ex, ey, ez, bx - qx, by - qy, bz - qz);
                    acc += 0.25 * sEll[s] * sumK;
                    nearPairs++;
                }
            } else if (admissible(ctx, N, mx, my, mz, halfEll, i1, i2, false)) {
                const xx = C[3 * N];
                const xy = C[3 * N + 1];
                const xz = C[3 * N + 2];
                // ½ and not ¼: the source's two endpoints have collapsed onto x_N.
                acc +=
                    0.5 *
                    L[N] *
                    (f(ex, ey, ez, ax - xx, ay - xy, az - xz) +
                        f(ex, ey, ez, bx - xx, by - xy, bz - xz));
                farNodes++;
            } else {
                stack[sp++] = left[N];
                stack[sp++] = right[N];
            }
        }
        total += ellI ** (1 - alpha) * acc;
    }
    if (stats) {
        stats.farNodes = farNodes;
        stats.nearPairs = nearPairs;
    }
    // /2: every unordered pair was visited from both of its edges, as in calculateEnergy.
    return total / 2;
}

/**
 * {@link admissible} for a target VERTEX `v` instead of a target edge: the same three conditions,
 * with the target's reach and adjacency widened to cover every edge incident to `v`.
 *  - spatial: `r_x ≤ θ · (|γ_v − x_N| − pad)`, `pad` = the longest edge at `v`, because the kernels
 *    of an incident edge are also evaluated at its OTHER endpoint, up to one edge away;
 *  - adjacency: `N` contains no edge that shares a vertex with any edge incident to `v`.
 * @see barnesHutGradient (why the differential is walked per vertex)
 */
function admissibleForVertex(ctx: BarnesHutContext, N: number, v: number, pad: number): boolean {
    const theta = ctx.theta;
    if (theta === 0) return false;
    const bvh = ctx.bvh;
    if (bvh.rt[N] > ctx.thetaT) return false;
    const X = ctx.X;
    const dx = X[3 * v] - bvh.C[3 * N];
    const dy = X[3 * v + 1] - bvh.C[3 * N + 1];
    const dz = X[3 * v + 2] - bvh.C[3 * N + 2];
    const reach = Math.sqrt(dx * dx + dy * dy + dz * dz) - pad;
    if (!(reach > 0 && bvh.rx[N] <= theta * reach)) return false;
    if (!ctx.checkAdjacency) return true;
    const b = bvh.begin[N];
    const e = bvh.end[N];
    const { incStart, incEdges, slotOf, E } = ctx;
    for (let k = incStart[v]; k < incStart[v + 1]; k++) {
        const I = incEdges[k];
        for (let side = 0; side < 2; side++) {
            const u = E[2 * I + side];
            for (let kk = incStart[u]; kk < incStart[u + 1]; kk++) {
                const s = slotOf[incEdges[kk]];
                if (s >= b && s < e) return false;
            }
        }
    }
    return true;
}

/**
 * Would the per-vertex walk do more kernel work than summing every pair exactly?
 *
 * Why the question arises: on a tight tangle almost no node is admissible, so the per-vertex
 * walk degenerates to all pairs — and it visits every edge from both of its ends, i.e. TWICE
 * the exact sum's work, for an answer that is the exact one anyway. Measured on the 1 998-vertex
 * dataset knot (2026-11-23): 3.7 s per-vertex against 1.8 s exact at the start of its run
 * (far nodes: 217), 2.0 s against 1.7 s after 20 steps, 0.49 s against 1.9 s after 40.
 *
 * The count is SAMPLED (every `stride`-th vertex, at most ~128 walks, no kernel evaluated) and
 * scaled up: (edge at the vertex, source edge) leaf pairs plus (edge, far node) lumps, each
 * costing about one kernel evaluation, against the `m²` evaluations of the exact sum.
 */
function exactWalkIsCheaper(ctx: BarnesHutContext): boolean {
    const { bvh, m, n, incStart, incEdges } = ctx;
    const { left, right, begin, end } = bvh;
    if (ctx.theta === 0) return true;
    const ell = (I: number): number => {
        const a = ctx.E[2 * I];
        const b = ctx.E[2 * I + 1];
        return (
            Math.hypot(
                ctx.X[3 * b] - ctx.X[3 * a],
                ctx.X[3 * b + 1] - ctx.X[3 * a + 1],
                ctx.X[3 * b + 2] - ctx.X[3 * a + 2],
            ) + ctx.epsilon
        );
    };
    const stack = new Int32Array(2 * bvh.depth + 8);
    const stride = Math.max(1, Math.floor(n / 128));
    let work = 0;
    let sampled = 0;
    for (let v = 0; v < n; v += stride) {
        const k0 = incStart[v];
        const k1 = incStart[v + 1];
        if (k0 === k1) continue;
        sampled++;
        let pad = 0;
        for (let k = k0; k < k1; k++) pad = Math.max(pad, ell(incEdges[k]));
        let sp = 0;
        stack[sp++] = 0;
        while (sp > 0) {
            const N = stack[--sp];
            if (left[N] < 0) work += (k1 - k0) * (end[N] - begin[N]);
            else if (admissibleForVertex(ctx, N, v, pad)) work += k1 - k0;
            else {
                stack[sp++] = left[N];
                stack[sp++] = right[N];
            }
        }
    }
    if (sampled === 0) return false;
    return (work / sampled) * n > m * m;
}

/**
 * Barnes–Hut differential of the tangent-point energy, one `Vec3` per vertex.
 *
 * For each target edge `I` and each of its two vertices `a`, the exact differential is
 *     ½ · ∂/∂γ_a Σ_J ( w_IJ + w_JI ),
 * and the far part of the sum over `J` is replaced node by node with
 *     w_IN = ½ ℓ_I^{1−α} L_N Σ_{i∈I} f(γ_i − x_N, e_I),   w_NI = ½ ℓ_I L_N Σ_{i∈I} f(x_N − γ_i, T_N).
 * A vertex's derivative only ever involves pairs in which one of ITS edges takes part, so summing
 * over targets gives the full differential. See the module header for why this is not the
 * gradient of {@link barnesHutEnergy}.
 *
 * WHY THE DEFAULT WALK IS PER VERTEX AND NOT PER EDGE (the paper walks per edge). `w_IJ` depends
 * on the target's direction `e_I`, and `∂e_I/∂γ` is ±1 at the two endpoints: each edge receives a
 * TORQUE of size O(1), independent of the edge length, with opposite signs at its two ends. At a
 * vertex the torques of the two incident edges cancel down to the O(1/n) nodal value. That
 * cancellation survives approximation only if both edges lump the SAME nodes. Walking per edge,
 * two neighbouring edges make different admissibility decisions at the threshold, the mismatch is
 * a θ²-relative error on an O(1) term, and the nodal error relative to the O(1/n) answer grows
 * like n·θ³. Measured with `traversal: 'edge'` on the trefoil: 4.97e-2 at n = 960 and 9.87e-2 at
 * n = 2 000 for θ = 0.25, and 3.1e-3 at n = 960 for θ = 0.1 ((0.1/0.25)³ predicts 3.2e-3); at
 * n = 30 000 the resulting direction is rejected by the line search
 * (`bench/results/2026-11-23-bh-step-scale.jsonl`, the `edge` rows). Walking per vertex gives both
 * incident edges one node set by construction. It costs about twice the kernel evaluations, since
 * every edge is visited from both of its ends.
 *
 * @see src/core/tangentPointEnergy.ts (gradientAnalytical — the exact reference; the kernel
 *   derivative below mirrors its `kernelDerivs`, guards included)
 * @see https://arxiv.org/abs/2006.07859 App. B.1.2, paragraph "Differential"
 */
export function barnesHutGradient(
    ctx: BarnesHutContext,
    alpha: number,
    beta: number,
    stats?: BarnesHutStats,
): Vec3[] {
    const { X, E, bvh, epsilon, m, n, incStart, incEdges } = ctx;
    const { left, right, begin, end, sE, sEll, L, C, TN } = bvh;
    const stack = new Int32Array(2 * bvh.depth + 8);
    const G = new Float64Array(3 * n);
    let farNodes = 0;
    let nearPairs = 0;

    // Per-edge target data, computed once: raw edge vector, ℓ = ‖e‖+ε, unit direction (zero for a
    // degenerate edge — same `< 1e-14` guard as the reference), ℓ^{1−α} and its derivative in ℓ.
    const EV = new Float64Array(3 * m);
    const EU = new Float64Array(3 * m);
    const ELL = new Float64Array(m);
    const EP = new Float64Array(m);
    const EDP = new Float64Array(m);
    for (let I = 0; I < m; I++) {
        const a = E[2 * I];
        const b = E[2 * I + 1];
        const x = X[3 * b] - X[3 * a];
        const y = X[3 * b + 1] - X[3 * a + 1];
        const z = X[3 * b + 2] - X[3 * a + 2];
        const re = Math.sqrt(x * x + y * y + z * z);
        const inv = re < 1e-14 ? 0 : 1 / re;
        EV[3 * I] = x;
        EV[3 * I + 1] = y;
        EV[3 * I + 2] = z;
        EU[3 * I] = inv * x;
        EU[3 * I + 1] = inv * y;
        EU[3 * I + 2] = inv * z;
        ELL[I] = re + epsilon;
        EP[I] = ELL[I] ** (1 - alpha);
        EDP[I] = (1 - alpha) * ELL[I] ** -alpha;
    }

    const cubeOverSixth = CUBE_OVER_SIXTH(alpha, beta);
    // Outputs of `kd`, written to captured locals instead of a returned object: this runs
    // ~10⁷–10⁸ times per evaluation and must not allocate.
    let kF = 0;
    let kDdx = 0;
    let kDdy = 0;
    let kDdz = 0;
    let kDex = 0;
    let kDey = 0;
    let kDez = 0;
    // f(d, e) and its derivatives with respect to d and e. The `< 1e-14` guards test the PRE-ε
    // lengths, exactly as `kernelDerivs` does: a zero-length vector has no unit direction.
    const kd = (ex: number, ey: number, ez: number, dx: number, dy: number, dz: number): void => {
        const rd = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const dEps = rd + epsilon;
        const cx = ey * dz - ez * dy;
        const cy = ez * dx - ex * dz;
        const cz = ex * dy - ey * dx;
        const rc = Math.sqrt(cx * cx + cy * cy + cz * cz);
        const cEps = rc + epsilon;
        let cPowA1: number; // cEps^(α−1)
        let dPowB: number; // dEps^β
        if (cubeOverSixth) {
            // (α, β) = (3, 6) by multiplication — see CUBE_OVER_SIXTH.
            const d2 = dEps * dEps;
            cPowA1 = cEps * cEps;
            dPowB = d2 * d2 * d2;
        } else {
            cPowA1 = cEps ** (alpha - 1);
            dPowB = dEps ** beta;
        }
        // The neighbouring powers by one multiplication each, on both branches: two `pow`
        // calls per term instead of four.
        const cPowA = cPowA1 * cEps;
        kF = cPowA / dPowB;
        const coeffC = (alpha * cPowA1) / dPowB;
        const coeffD = rd < 1e-14 ? 0 : (-beta * cPowA) / (dPowB * dEps * rd);
        if (rc >= 1e-14) {
            const s = coeffC / rc;
            // ∂‖c‖/∂d = (c × e)/‖c‖,  ∂‖c‖/∂e = (d × c)/‖c‖
            kDdx = s * (cy * ez - cz * ey) + coeffD * dx;
            kDdy = s * (cz * ex - cx * ez) + coeffD * dy;
            kDdz = s * (cx * ey - cy * ex) + coeffD * dz;
            kDex = s * (dy * cz - dz * cy);
            kDey = s * (dz * cx - dx * cz);
            kDez = s * (dx * cy - dy * cx);
        } else {
            kDdx = coeffD * dx;
            kDdy = coeffD * dy;
            kDdz = coeffD * dz;
            kDex = 0;
            kDey = 0;
            kDez = 0;
        }
    };

    // The current target edge (endpoints a = i1, b = i2), loaded by `setTarget`.
    let i1 = 0;
    let i2 = 0;
    let ax = 0;
    let ay = 0;
    let az = 0;
    let bx = 0;
    let by = 0;
    let bz = 0;
    let ex = 0;
    let ey = 0;
    let ez = 0;
    let ux = 0;
    let uy = 0;
    let uz = 0;
    let ellI = 0;
    let p = 0;
    let dp = 0;
    const setTarget = (I: number): void => {
        i1 = E[2 * I];
        i2 = E[2 * I + 1];
        ax = X[3 * i1];
        ay = X[3 * i1 + 1];
        az = X[3 * i1 + 2];
        bx = X[3 * i2];
        by = X[3 * i2 + 1];
        bz = X[3 * i2 + 2];
        ex = EV[3 * I];
        ey = EV[3 * I + 1];
        ez = EV[3 * I + 2];
        ux = EU[3 * I];
        uy = EU[3 * I + 1];
        uz = EU[3 * I + 2];
        ellI = ELL[I];
        p = EP[I];
        dp = EDP[I];
    };

    // Derivative of one (target edge, source) interaction with respect to the target's endpoints
    // a and b — set by `nearPair` / `farNode`, WITHOUT the final ½.
    let cax = 0;
    let cay = 0;
    let caz = 0;
    let cbx = 0;
    let cby = 0;
    let cbz = 0;

    // Exact pair with the source edge in slot `s`. Returns false (and sets nothing) when the two
    // edges share a vertex: the discrete energy omits that pair.
    const nearPair = (s: number): boolean => {
        const j1 = sE[2 * s];
        const j2 = sE[2 * s + 1];
        if (j1 === i1 || j1 === i2 || j2 === i1 || j2 === i2) return false;
        const px = X[3 * j1];
        const py = X[3 * j1 + 1];
        const pz = X[3 * j1 + 2];
        const qx = X[3 * j2];
        const qy = X[3 * j2 + 1];
        const qz = X[3 * j2 + 2];
        const ellJ = sEll[s];

        // Forward, w_IJ = ¼ ℓ_I^{1−α} ℓ_J Σ f(γ_i − γ_j, e_I).
        kd(ex, ey, ez, ax - px, ay - py, az - pz);
        let S = kF;
        let dax = kDdx;
        let day = kDdy;
        let daz = kDdz;
        let fex = kDex;
        let fey = kDey;
        let fez = kDez;
        kd(ex, ey, ez, ax - qx, ay - qy, az - qz);
        S += kF;
        dax += kDdx;
        day += kDdy;
        daz += kDdz;
        fex += kDex;
        fey += kDey;
        fez += kDez;
        kd(ex, ey, ez, bx - px, by - py, bz - pz);
        S += kF;
        let dbx = kDdx;
        let dby = kDdy;
        let dbz = kDdz;
        fex += kDex;
        fey += kDey;
        fez += kDez;
        kd(ex, ey, ez, bx - qx, by - qy, bz - qz);
        S += kF;
        dbx += kDdx;
        dby += kDdy;
        dbz += kDdz;
        fex += kDex;
        fey += kDey;
        fez += kDez;
        const wf = 0.25 * ellJ;
        const sdp = S * dp;
        cax = wf * (-sdp * ux + p * (dax - fex));
        cay = wf * (-sdp * uy + p * (day - fey));
        caz = wf * (-sdp * uz + p * (daz - fez));
        cbx = wf * (sdp * ux + p * (dbx + fex));
        cby = wf * (sdp * uy + p * (dby + fey));
        cbz = wf * (sdp * uz + p * (dbz + fez));

        // Reverse, w_JI = ¼ ℓ_J^{1−α} ℓ_I Σ f(γ_j − γ_i, e_J): I enters through ℓ_I and through
        // d = γ_j − γ_i, whose derivative in γ_i is −f_d.
        const jx = qx - px;
        const jy = qy - py;
        const jz = qz - pz;
        kd(jx, jy, jz, px - ax, py - ay, pz - az);
        let R = kF;
        dax = kDdx;
        day = kDdy;
        daz = kDdz;
        kd(jx, jy, jz, qx - ax, qy - ay, qz - az);
        R += kF;
        dax += kDdx;
        day += kDdy;
        daz += kDdz;
        kd(jx, jy, jz, px - bx, py - by, pz - bz);
        R += kF;
        dbx = kDdx;
        dby = kDdy;
        dbz = kDdz;
        kd(jx, jy, jz, qx - bx, qy - by, qz - bz);
        R += kF;
        dbx += kDdx;
        dby += kDdy;
        dbz += kDdz;
        const wr = 0.25 * ellJ ** (1 - alpha);
        cax += wr * (-R * ux - ellI * dax);
        cay += wr * (-R * uy - ellI * day);
        caz += wr * (-R * uz - ellI * daz);
        cbx += wr * (R * ux - ellI * dbx);
        cby += wr * (R * uy - ellI * dby);
        cbz += wr * (R * uz - ellI * dbz);
        return true;
    };

    // Lumped interaction with node `N`.
    const farNode = (N: number): void => {
        const xx = C[3 * N];
        const xy = C[3 * N + 1];
        const xz = C[3 * N + 2];

        // Forward, w_IN = ½ ℓ_I^{1−α} L_N Σ_i f(γ_i − x_N, e_I).
        kd(ex, ey, ez, ax - xx, ay - xy, az - xz);
        let S = kF;
        let dax = kDdx;
        let day = kDdy;
        let daz = kDdz;
        let fex = kDex;
        let fey = kDey;
        let fez = kDez;
        kd(ex, ey, ez, bx - xx, by - xy, bz - xz);
        S += kF;
        let dbx = kDdx;
        let dby = kDdy;
        let dbz = kDdz;
        fex += kDex;
        fey += kDey;
        fez += kDez;
        const w = 0.5 * L[N];
        const sdp = S * dp;
        cax = w * (-sdp * ux + p * (dax - fex));
        cay = w * (-sdp * uy + p * (day - fey));
        caz = w * (-sdp * uz + p * (daz - fez));
        cbx = w * (sdp * ux + p * (dbx + fex));
        cby = w * (sdp * uy + p * (dby + fey));
        cbz = w * (sdp * uz + p * (dbz + fez));

        // Reverse, w_NI = ½ ℓ_I L_N Σ_i f(x_N − γ_i, T_N). T_N is the length-weighted mean tangent,
        // NOT renormalised (paper App. B.1.1): Σ_J ℓ_J^{1−α}(‖e_J×d‖+ε)^α is ℓ_J‖T_J×d‖^α up to ε,
        // and lumping that over the node gives L_N‖T_N×d‖^α.
        const tx = TN[3 * N];
        const ty = TN[3 * N + 1];
        const tz = TN[3 * N + 2];
        kd(tx, ty, tz, xx - ax, xy - ay, xz - az);
        let R = kF;
        dax = kDdx;
        day = kDdy;
        daz = kDdz;
        kd(tx, ty, tz, xx - bx, xy - by, xz - bz);
        R += kF;
        dbx = kDdx;
        dby = kDdy;
        dbz = kDdz;
        cax += w * (-R * ux - ellI * dax);
        cay += w * (-R * uy - ellI * day);
        caz += w * (-R * uz - ellI * daz);
        cbx += w * (R * ux - ellI * dbx);
        cby += w * (R * uy - ellI * dby);
        cbz += w * (R * uz - ellI * dbz);
    };

    // `'auto'`: when the tree hardly engages, sum every pair exactly with the per-edge walk
    // (no node is ever lumped, so the per-edge walk's far-field weakness does not apply).
    const exactWalk = ctx.traversal === 'auto' && exactWalkIsCheaper(ctx);
    if (ctx.traversal === 'edge' || exactWalk) {
        // The paper's walk: one traversal per target edge, both endpoints updated together.
        for (let I = 0; I < m; I++) {
            setTarget(I);
            const mx = 0.5 * (ax + bx);
            const my = 0.5 * (ay + by);
            const mz = 0.5 * (az + bz);
            const halfEll = 0.5 * ellI;
            let gax = 0;
            let gay = 0;
            let gaz = 0;
            let gbx = 0;
            let gby = 0;
            let gbz = 0;
            let sp = 0;
            stack[sp++] = 0;
            while (sp > 0) {
                const N = stack[--sp];
                if (left[N] < 0) {
                    for (let s = begin[N]; s < end[N]; s++) {
                        if (!nearPair(s)) continue;
                        gax += cax;
                        gay += cay;
                        gaz += caz;
                        gbx += cbx;
                        gby += cby;
                        gbz += cbz;
                        nearPairs++;
                    }
                } else if (!exactWalk && admissible(ctx, N, mx, my, mz, halfEll, i1, i2, true)) {
                    farNode(N);
                    gax += cax;
                    gay += cay;
                    gaz += caz;
                    gbx += cbx;
                    gby += cby;
                    gbz += cbz;
                    farNodes++;
                } else {
                    stack[sp++] = left[N];
                    stack[sp++] = right[N];
                }
            }
            // ½: E = ½ Σ over ORDERED pairs, and both orderings were just differentiated.
            G[3 * i1] += 0.5 * gax;
            G[3 * i1 + 1] += 0.5 * gay;
            G[3 * i1 + 2] += 0.5 * gaz;
            G[3 * i2] += 0.5 * gbx;
            G[3 * i2 + 1] += 0.5 * gby;
            G[3 * i2 + 2] += 0.5 * gbz;
        }
    } else {
        // One traversal per target VERTEX; every edge at the vertex sees the same nodes.
        for (let v = 0; v < n; v++) {
            const k0 = incStart[v];
            const k1 = incStart[v + 1];
            if (k0 === k1) continue; // isolated vertex: no edge, no energy.
            let pad = 0;
            for (let k = k0; k < k1; k++) pad = Math.max(pad, ELL[incEdges[k]]);
            let gx = 0;
            let gy = 0;
            let gz = 0;
            let sp = 0;
            stack[sp++] = 0;
            while (sp > 0) {
                const N = stack[--sp];
                if (left[N] < 0) {
                    for (let k = k0; k < k1; k++) {
                        setTarget(incEdges[k]);
                        const isA = i1 === v;
                        for (let s = begin[N]; s < end[N]; s++) {
                            if (!nearPair(s)) continue;
                            if (isA) {
                                gx += cax;
                                gy += cay;
                                gz += caz;
                            } else {
                                gx += cbx;
                                gy += cby;
                                gz += cbz;
                            }
                            nearPairs++;
                        }
                    }
                } else if (admissibleForVertex(ctx, N, v, pad)) {
                    for (let k = k0; k < k1; k++) {
                        setTarget(incEdges[k]);
                        farNode(N);
                        if (i1 === v) {
                            gx += cax;
                            gy += cay;
                            gz += caz;
                        } else {
                            gx += cbx;
                            gy += cby;
                            gz += cbz;
                        }
                    }
                    farNodes++;
                } else {
                    stack[sp++] = left[N];
                    stack[sp++] = right[N];
                }
            }
            G[3 * v] = 0.5 * gx;
            G[3 * v + 1] = 0.5 * gy;
            G[3 * v + 2] = 0.5 * gz;
        }
    }
    if (stats) {
        stats.farNodes = farNodes;
        stats.nearPairs = nearPairs;
    }

    const out: Vec3[] = new Array(n);
    for (let v = 0; v < n; v++) out[v] = [G[3 * v], G[3 * v + 1], G[3 * v + 2]];
    return out;
}
