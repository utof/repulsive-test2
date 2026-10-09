/**
 * `L^{2−s}` — the fractional (low-order) operator of the Repulsive Surfaces preconditioner
 * `Ã⁻¹ = (−Δ)⁻¹ L^{2−s} (−Δ)⁻¹`, applied hierarchically over the SAME BVH and the SAME block
 * lists as `A` (spec §5.2: *"3 columns on the same tree and the same block lists as `A`"*).
 *
 * THE QUADRATIC FORM (`[RS §5.1.1]`, arXiv:2107.01664), vertex-collocated:
 *
 *     vᵀ L^σ v = sum_{i != j} a_i a_j |V_i − V_j|^{−(1 + 2σ)} (v_i − v_j)²
 *
 * with `σ = 2 − s` and `a_i = (ell_{i−1} + ell_i)/2` the dual-edge weight. That is a weighted
 * graph Laplacian `L = 2 (diag(W·1) − W)`, `W_ij = a_i a_j K_ij`.
 *
 * TWO DIFFERENT EXPONENTS, and a reader has already confused them (spec §5.2). `s = (β−1)/α =
 * 5/3` and `σ_lo = 2 − s = 1/3`, so `L^{2−s}`'s distance exponent is `1 + 2σ_lo = ` **5/3** —
 * while the high-order ENERGY kernel's is `2(s−1) + 1 = ` **7/3** (`nearField.ts` `exponents`).
 * At α=3, β=6 the number 5/3 also equals `s`; that is a coincidence of these α, β, not an
 * identity. {@link lowOrderExponent} derives it from α, β so the two can never be swapped.
 *
 * WHY VERTEX-COLLOCATED IS THE PRODUCTION VARIANT, and why the literal `[RS §5.1.1]` identity is
 * a KNOCK-OUT and not an alternative. With every `W_ij > 0` on a connected curve the nullspace of
 * `L` is EXACTLY the constants — nullity 1. The literal identity
 * `L^σ v = 2Uᵀ[diag(diag(a)⁻¹Ha) − H]Uv` loses that for even `n`: on a closed polyline
 * `|E| = |V|`, so `U` is square and annihilates the checkerboard `(−1)^i`, giving nullity **2** —
 * and it kills the highest-frequency mode, the one `A_σ` weights most. MEASURED at n ∈
 * {64,120,240,480} (`.superpowers/sdd/2026-09-11-phase2h-queue/scratch/e4-closer-0b.json`):
 * nullity 1 vs 2, `κ_eig(Ã⁻¹A_σ)` 1.0977e4 vs 4.5450e15. Both variants live here behind
 * {@link LowOrderVariant} because H5 needs the knock-out to be reachable (CLAUDE.md: prefer a
 * variant switch over one hardcoded kernel); `'rs-identity'` must never be chosen in production.
 *
 * WHY `W·1` GOES THROUGH THIS SAME APPLY (do not "simplify" it into a separate diagonal sum).
 * Nullity 1 holds *because the diagonal is exactly the row sum*. {@link prepareLowOrder} computes
 * `rowSum` by applying this operator to the ones vector — exactly as `prepareA` does for
 * `K1`/`K⁰1` (spec §2.2; `apply.ts` module TSDoc) — so `L·1 = 2(rowSum ⊙ 1 − W·1)` is `x − x` on
 * identical bits and is **0 bit-exactly and structurally**, not to a tolerance. Computing the
 * diagonal any other way breaks H5 property 1 for an implementation-artefact reason.
 *
 * THE BIT-EXACTNESS IS SCOPED TO THE ONES VECTOR, and a reader must not widen it. `x − x` cancels
 * because `rowSum` is literally the `Σ_t W_st` this apply produces for `psi = 1`. For a general
 * constant `c·1` the two sides are `c·rowSum` and `W(c·1)`, which round differently. MEASURED
 * `max|L·(c·1)|` at n ∈ {64, 240, 480, 960}, theta=0.25: **0 exactly** for `c = 1`, but
 * 5.3e-14 .. 9.9e-14 for `c = 7.5`, 2.1e-14 .. 3.2e-14 for `c = -3` and 6.7e-16 .. 9.4e-16 for
 * `c = 0.1` — i.e. a RELATIVE roundoff of the row sum that scales with `|c|`, not an absolute
 * floor, and FLAT in n. Nullity 1 is unaffected (it is a property of the exact operator), but a
 * gate asserting `L·(c·1) == 0` for `c ≠ 1` would be asserting something false.
 *
 * THE TREE IS EDGE-BASED AND THIS OPERATOR IS VERTEX-BASED. `bvh.ts` clusters edge primitives
 * (midpoints `P`, unit tangents `T` in R⁶); `L^{2−s}` sums over vertex pairs. So "the same tree
 * and the same block lists" means: the block LISTS and the tree TOPOLOGY are reused unchanged,
 * and the per-node geometry — `nodeC` for the far kernel, plus the diagnostic `nodeMass` and
 * `nodeRx` — is RECOMPUTED from the collocation points, since each vertex sits `ell/2` off its
 * edge's midpoint.
 *
 * HOW MUCH THAT RECOMPUTE IS WORTH, measured rather than asserted: substituting `bvh.C` for
 * `nodeC` moves `δ_compress` by **+2.5 % at n=240 and +0.16 % at n=480**, and leaves the nullity
 * and `λ_max` UNCHANGED. So the recompute is the correct thing to do and stays — it evaluates
 * the rank-1 lump at the points the kernel is actually collocated on — but it is a small
 * accuracy improvement, not a correctness guard, and the first draft of this comment over-stated
 * it. It does NOT affect admissibility: the block lists are reused exactly as `bct.ts` built
 * them, nothing here re-decides Eq. 28, and `nodeRx` is never read by the apply.
 *
 * SLOT ↔ VERTEX. The vertex variant collocates vertex `sE[2·slot]` (the slot's edge's FIRST
 * endpoint) on each slot, which is a bijection exactly on a closed curve (`|E| = |V|`); the
 * vertex's coordinates are then `bvh.sXa`, already in slot order. An open chain has `m = n − 1`
 * and no slot for the last vertex, so it throws {@link CurveTopologyError} rather than silently
 * dropping a vertex. (`laplacian.ts` restricts the preconditioner's topology anyway.)
 *
 * NO NEAR PRECOMPUTE, DELIBERATELY DEFERRED. `prepareA` caches its near kernel values
 * (`precomputeNear`); this operator evaluates them inline inside {@link applyW}. The measured
 * price: one 3-column `L^{2−s}` apply costs about as much as one whole 12-column `A` matvec
 * (`msL / msPre` = 1.08 .. 1.36 across runs at n=960), so a left-preconditioned GMRES iteration
 * costs roughly 2x an unpreconditioned one. The comparison is LOAD-DEPENDENT and the cost test
 * logs the matched inline-vs-inline ratio (0.10, i.e. per kernel evaluation this operator is
 * ~10x cheaper than `A`) next to it. The storage a precompute would cost at N=30 000 is
 * `8 B x nearPairs` =
 * **51.8 MiB** in f64 at the production theta=0.25 (spec §3.4's near-pair table:
 * `nearPairs = 6 785 670` for the trefoil, so `8 B x nearPairs / 2²⁰`), 189.9 MiB for the worst
 * case in that table (tangle at theta=0.1, 24 886 020 pairs), half of each in f32. And `L^{2−s}`
 * has ONE kernel, so unlike `A` it pays this once, not twice. It is **NOT 3.6 GB**: that figure is
 * §7.1's `chi = 0` ALL-PAIRS case (`3.6e9/8 B = 4.5e8 = 30 000²/2` exactly, VERIFIED), and reading
 * it as the near-pair cost over-states the storage 66x. Adding the precompute is a performance
 * feature with its own storage/accuracy trade-off and its own gates; whoever sizes it should start
 * from 51.8 MiB.
 *
 * NO EPSILON BY DEFAULT. `gmres_lab.py:148-171`, the code that produced every H5 number, uses raw
 * edge lengths and raw vertex distances with no regularisation; `A`'s `epsilon` regularises the
 * ENERGY and has no business in a preconditioner. `epsilon` stays a parameter (default 0).
 *
 * f64 ONLY — there is no {@link AccumulatorVariant} switch here, unlike `apply.ts`. The whole
 * preconditioner is f64 because `laplacian.ts`'s `solveLaplacian` is (`Float64Array` signature),
 * and spec §6's mixed-precision scheme is f32 INNER / f64 OUTER with the preconditioner on the
 * outer side. Spec §11.2 task 10 owns that question.
 *
 * @see https://arxiv.org/abs/2107.01664 §5.1.1, §5.2
 * @see .superpowers/sdd/2026-09-11-phase2h-queue/scratch/gmres_lab.py:128-171 (both variants)
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.2, §5.6, §7.8, §11.2 task 6
 * @issue utof/repulsive-test2#59
 */
import type { Edge, Vec3 } from '../../testConfigs';
import { type ApplyWorkspace, type BlockSource, forEachBlock } from './apply';
import { type BctStats, resolveBctParams } from './bct';
import type { Bvh } from './bvh';
import { CurveTopologyError } from './laplacian';

/**
 * `'vertex'`: the production vertex-collocated discretisation (nullity 1). `'rs-identity'`: the
 * literal `[RS §5.1.1]` edge-midpoint identity, a KNOCK-OUT with nullity 2 on every even-`n`
 * closed curve (module TSDoc; spec §7.8 knock-out 1).
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §7.8
 */
export type LowOrderVariant = 'vertex' | 'rs-identity';

/**
 * The fractional order `σ_lo = 2 − s` and the kernel's distance exponent `p = 1 + 2σ_lo`, from the
 * same `s = (β−1)/α` as `innerProduct.ts:52-54` and `nearField.ts` `exponents`.
 * Why a function rather than a constant: `p = 5/3` here against the high-order energy kernel's
 * `7/3` is the confusion spec §5.2 records, and deriving both from α, β is what makes them
 * impossible to swap. `σ_lo ∈ (0, 2)` is `[RS §5.2]`'s admissibility condition (1/3 at α=3, β=6).
 * @see src/core/sobolev/innerProduct.ts:52-54
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.2
 */
export function lowOrderExponent(alpha: number, beta: number): { sigmaOrder: number; p: number } {
    const s = (beta - 1) / alpha;
    const sigmaOrder = 2 - s;
    return { sigmaOrder, p: 1 + 2 * sigmaOrder };
}

/**
 * Options of {@link prepareLowOrder}.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.2, §7.8
 */
export interface LowOrderOptions {
    alpha: number;
    beta: number;
    /** Default `'vertex'`. `'rs-identity'` is a knock-out — see {@link LowOrderVariant}. */
    variant?: LowOrderVariant;
    /**
     * Added AFTER each distance norm. Default **0**, which is what `gmres_lab.py:148-171`
     * measured; `A`'s epsilon regularises the energy, not the preconditioner (module TSDoc).
     */
    epsilon?: number;
    /**
     * TESTS ONLY, same contract as `prepareA`: lets a knock-out block source through (inverted
     * admissibility, or the adjacency certificate off). Default `false`.
     * @see src/core/sobolev/hierarchical/apply.ts (`PrepareOptions.knockOut`)
     */
    knockOut?: boolean;
}

/**
 * A prepared hierarchical `L^{2−s}`. NOT RE-ENTRANT, for the same reason `HierarchicalA` is not
 * (`apply.ts`): `ws` and `scratch` are shared, so one prepared instance serves one caller at a
 * time — and task 7's GMRES calls it inside its loop.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.2, §5.6
 */
export interface LowOrderA {
    /** Vertex count `|V|`. */
    n: number;
    /** Edge count `m` = slot count. */
    m: number;
    bvh: Bvh;
    source: BlockSource;
    variant: LowOrderVariant;
    /** Kernel distance exponent `1 + 2(2−s)` = 5/3, NOT the energy's 7/3 (module TSDoc). */
    p: number;
    epsilon: number;
    /** Collocation weight per slot: `a_i` (vertex) or `1` (rs-identity, `H` is unweighted). */
    w: Float64Array;
    /** Collocation point per slot, flat xyz: the vertex (vertex) or the edge midpoint. */
    X: Float64Array;
    /** `vOf[slot]` = the collocated vertex id; `-1` for `'rs-identity'` (edge-collocated). */
    vOf: Int32Array;
    /** RAW edge length per slot, no epsilon — `rs-identity`'s `U` and its `diag(a)⁻¹Ha`. */
    ell: Float64Array;
    /**
     * RECOMPUTED node mass `sum_{slot in N} w` over the collocation points. DIAGNOSTIC, like
     * {@link LowOrderA.nodeRx}: the far lump's weight is the PERCOLATED `s[B] = sum_t w_t psi_t`
     * ({@link applyW}), which depends on `psi`, so a precomputed per-node mass cannot stand in
     * for it and nothing in the apply reads this. KEPT rather than deleted because it is the
     * `psi = 1` case of exactly that percolation and is therefore the cheapest independent check
     * that the bottom-up pass is right: `nodeMass[root] = total curve length` and
     * `nodeMass[parent] = nodeMass[left] + nodeMass[right]`, both asserted in `lowOrder.test.ts`.
     * An earlier docstring called it "the far lump's `s[B]` weight", which is what made a
     * reviewer's mutant on it vacuous (fix round 1, finding m1).
     */
    nodeMass: Float64Array;
    /** RECOMPUTED node barycentre of the collocation points, `w`-weighted, flat xyz. */
    nodeC: Float64Array;
    /**
     * RECOMPUTED node radius `max |X_slot − nodeC|`. DIAGNOSTIC: the block lists are reused as
     * built, so nothing here re-decides Eq. 28 — this exists so a test can MEASURE how far the
     * vertex geometry departs from `bvh.rx` instead of assuming it is close.
     */
    nodeRx: Float64Array;
    /** `k(nodeC_A, nodeC_B)` per far block; `null` for a `'stream'` source (nothing cached). */
    far: Float64Array | null;
    /**
     * `W·1` (vertex) or `H·ell` (rs-identity), slot order, produced by {@link applyLowOrder}'s own
     * apply on the ones vector. Do NOT recompute it any other way — module TSDoc.
     */
    rowSum: Float64Array;
    stats: BctStats;
    /** `i == j` pairs skipped by the near field — exactly `m` per apply (one diagonal per slot). */
    diagSkipped: number;
    /** Per-column-count workspaces, keyed by `nc`. */
    ws: Map<number, F64Workspace>;
    /** Slot-space scratch, 3 columns wide: `psi` (gathered input), `phi` (`W psi`). */
    scratch: { psi: Float64Array; phi: Float64Array };
}

/** `1 / (|X_s − X_t| + eps)^p`, eps AFTER the norm (the convention of `nearField.ts`). */
function pairKernel(X: Float64Array, s: number, t: number, p: number, eps: number): number {
    const dx = X[3 * s] - X[3 * t];
    const dy = X[3 * s + 1] - X[3 * t + 1];
    const dz = X[3 * s + 2] - X[3 * t + 2];
    // Per-exponent `pow` (p = 5/3 is not a small integer); `**` matches `apply.ts`'s `d ** p`.
    return 1 / (Math.sqrt(dx * dx + dy * dy + dz * dz) + eps) ** p;
}

/** The rank-1 far-block kernel, at the RECOMPUTED collocation barycentres (module TSDoc). */
function farKernel(C: Float64Array, A: number, B: number, p: number, eps: number): number {
    return pairKernel(C, A, B, p, eps);
}

/**
 * The percolation scratch, f64 and TYPED so {@link applyW} needs no cast. `apply.ts`'s
 * {@link ApplyWorkspace} is `RealArray` because it carries the f32/f16 accumulator variants;
 * this operator is f64-only (module TSDoc), so it declares the narrow type instead of casting
 * one away — the cast was true only because `makeWorkspace(..., F64)` happened to be the caller.
 */
interface F64Workspace {
    /** Upward sums `s[node*nc + c]`. */
    s: Float64Array;
    /** Downward accumulators `acc[node*nc + c]`. */
    acc: Float64Array;
}

function workspace(Lo: LowOrderA, nc: number): F64Workspace {
    let ws = Lo.ws.get(nc);
    if (!ws) {
        const len = Lo.bvh.nNodes * nc;
        ws = { s: new Float64Array(len), acc: new Float64Array(len) };
        Lo.ws.set(nc, ws);
    }
    return ws;
}

/**
 * `phi = W psi` for `nc` columns in SLOT order, `W_st = w_s w_t k(X_s, X_t)` and `W_ss = 0`.
 *
 * THE PERCOLATION IS LOAD-BEARING, exactly as in `apply.ts`: a far block is one scalar
 * multiply-accumulate per column against `s[B] = sum_{t in B} w_t psi_t`, not a loop over `B`'s
 * primitives. The accumulation order (upward by descending node id, far blocks in list order,
 * near blocks in list order, downward by ascending id) mirrors `applyK` so that the two operators
 * are the same traversal with different kernels.
 *
 * `i == j` is excluded in the near field only. Far blocks cannot contain it: the adjacency
 * certificate proves no two edges of `A`, `B` share a vertex (`bct.ts`), hence `A` and `B` hold
 * disjoint slots; independently, `max(r_A, r_B) <= theta·d` with `theta < 1` already forbids one
 * node containing the other. {@link prepareLowOrder} refuses a source with the certificate off.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.6
 * @see https://arxiv.org/abs/2006.07859 App. B.3.4 ("we percolate these sums up and down the BVH")
 */
function applyW(Lo: LowOrderA, psi: Float64Array, phi: Float64Array, nc: number): BctStats {
    const { bvh, source, w, X, p, epsilon, far, nodeC } = Lo;
    const { left, right, begin, end, nNodes } = bvh;
    const { s: sAcc, acc } = workspace(Lo, nc);

    phi.fill(0, 0, Lo.m * nc);
    sAcc.fill(0);
    acc.fill(0);

    // --- upward percolation: s[N] = sum_{slot in N} w_slot psi_slot ---
    // Node ids are allocated pre-order (bvh.ts), so descending the id range visits children first.
    for (let id = nNodes - 1; id >= 0; id--) {
        const base = id * nc;
        if (left[id] < 0) {
            for (let slot = begin[id]; slot < end[id]; slot++) {
                const ww = w[slot];
                const pb = slot * nc;
                for (let c = 0; c < nc; c++) sAcc[base + c] += ww * psi[pb + c];
            }
        } else {
            const lb = left[id] * nc;
            const rb = right[id] * nc;
            for (let c = 0; c < nc; c++) sAcc[base + c] = sAcc[lb + c] + sAcc[rb + c];
        }
    }

    let diagSkipped = 0;
    const stats = forEachBlock(
        bvh,
        source,
        (k, A, B) => {
            const kv = far ? far[k] : farKernel(nodeC, A, B, p, epsilon);
            const ab = A * nc;
            const bb = B * nc;
            for (let c = 0; c < nc; c++) acc[ab + c] += kv * sAcc[bb + c];
        },
        (_k, A, B) => {
            const bB = begin[B];
            const eB = end[B];
            for (let s = begin[A], eA = end[A]; s < eA; s++) {
                const ob = s * nc;
                for (let t = bB; t < eB; t++) {
                    if (s === t) {
                        // `K_ii = 0` (gmres_lab.py:169's `fill_diagonal(K, 0)`): the Gagliardo
                        // sum is over `i != j`. Counted, because a skip counter pinned at 0 reads
                        // exactly like a passing one (memory `gates-name-their-impossible-values`).
                        diagSkipped++;
                        continue;
                    }
                    const kv = w[s] * w[t] * pairKernel(X, s, t, p, epsilon);
                    const tb = t * nc;
                    for (let c = 0; c < nc; c++) phi[ob + c] += kv * psi[tb + c];
                }
            }
        },
    );

    // --- downward percolation, then scatter to the primitives ---
    // Near first, then far: the same order as `applyK` (apply.ts's ACCUMULATION ORDER note).
    for (let id = 0; id < nNodes; id++) {
        const base = id * nc;
        if (left[id] < 0) {
            for (let slot = begin[id]; slot < end[id]; slot++) {
                const ww = w[slot];
                const pb = slot * nc;
                for (let c = 0; c < nc; c++) phi[pb + c] += ww * acc[base + c];
            }
        } else {
            const lb = left[id] * nc;
            const rb = right[id] * nc;
            for (let c = 0; c < nc; c++) {
                acc[lb + c] += acc[base + c];
                acc[rb + c] += acc[base + c];
            }
        }
    }

    Lo.diagSkipped = diagSkipped;
    return stats;
}

/**
 * Builds a {@link LowOrderA} over an existing {@link Bvh} and {@link BlockSource} — the SAME tree
 * and the SAME block lists as the `A` prepared from them (spec §5.2). Recomputes the collocation
 * weights, points and node data (module TSDoc), precomputes the far values for a `'lists'` source,
 * then computes `rowSum` through {@link applyW} on the ones vector.
 * @param vertices/edges needed for the dual weights `a_i`, which the BVH does not carry.
 * @throws CurveTopologyError if the vertex variant has no slot/vertex bijection (open chain, or a
 *   topology whose edges' first endpoints are not a permutation of the vertices).
 * @throws Error if the block source is a knock-out instrument and `knockOut` was not passed.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §2.2, §5.2, §5.6
 */
export function prepareLowOrder(
    vertices: Vec3[],
    edges: Edge[],
    bvh: Bvh,
    source: BlockSource,
    opts: LowOrderOptions,
): LowOrderA {
    const variant = opts.variant ?? 'vertex';
    const epsilon = opts.epsilon ?? 0;
    const { p } = lowOrderExponent(opts.alpha, opts.beta);
    const n = vertices.length;
    const m = bvh.order.length;
    const lists = source.kind === 'lists';

    // Same guard and same wording as `prepareA`: a knock-out block source reaching a production
    // path is silent corruption, so it must be asked for explicitly.
    const resolved = lists ? source.bct.params : resolveBctParams(source.params);
    if (!opts.knockOut && (resolved.invertAdmissibility || !resolved.adjacencyCertificate)) {
        throw new Error(
            `prepareLowOrder: knock-out instrument in a production apply (invertAdmissibility=${resolved.invertAdmissibility}, adjacencyCertificate=${resolved.adjacencyCertificate}); pass knockOut: true in a test`,
        );
    }

    // RAW slot-order edge lengths, from the endpoint coordinates the BVH already carries.
    // NOT `bvh.sEll`, which is `||e|| + epsilon` (bvh.ts `edgeData`) — `gmres_lab.py:150` uses the
    // unregularised norm and every H5 number was measured with it.
    const ell = new Float64Array(m);
    for (let s = 0; s < m; s++) {
        const dx = bvh.sXb[3 * s] - bvh.sXa[3 * s];
        const dy = bvh.sXb[3 * s + 1] - bvh.sXa[3 * s + 1];
        const dz = bvh.sXb[3 * s + 2] - bvh.sXa[3 * s + 2];
        ell[s] = Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    const w = new Float64Array(m);
    const X = new Float64Array(3 * m);
    const vOf = new Int32Array(m).fill(-1);
    if (variant === 'vertex') {
        // Dual-edge weights `a_i = (ell_{i−1} + ell_i)/2`, accumulated in EDGE-LIST order and
        // from unregularised lengths — `gmres_lab.py:163-166` verbatim, so the f64 sum matches.
        const a = new Float64Array(n);
        for (const [q, r] of edges) {
            const u = vertices[q];
            const v = vertices[r];
            const dx = v[0] - u[0];
            const dy = v[1] - u[1];
            const dz = v[2] - u[2];
            const l = Math.sqrt(dx * dx + dy * dy + dz * dz);
            a[q] += 0.5 * l;
            a[r] += 0.5 * l;
        }
        // Slot -> vertex: the slot's edge's FIRST endpoint. A bijection exactly when |E| = |V|
        // and every vertex starts exactly one edge (module TSDoc); verified, never assumed.
        const seen = new Uint8Array(n);
        for (let s = 0; s < m; s++) {
            const i = bvh.sE[2 * s];
            if (seen[i]) {
                throw new CurveTopologyError(
                    `prepareLowOrder: vertex ${i} starts more than one edge, so slot->vertex is not a bijection; the 'vertex' variant needs a closed curve (|E| = |V|)`,
                );
            }
            seen[i] = 1;
            vOf[s] = i;
            w[s] = a[i];
            // `bvh.sXa` IS the first endpoint's coordinates in slot order (bvh.ts), raw.
            X[3 * s] = bvh.sXa[3 * s];
            X[3 * s + 1] = bvh.sXa[3 * s + 1];
            X[3 * s + 2] = bvh.sXa[3 * s + 2];
        }
        for (let i = 0; i < n; i++) {
            if (!seen[i]) {
                throw new CurveTopologyError(
                    `prepareLowOrder: vertex ${i} starts no edge, so it has no slot to collocate on; the 'vertex' variant needs a closed curve (|E| = |V|, got |E| = ${m}, |V| = ${n})`,
                );
            }
        }
    } else {
        // `H` is UNWEIGHTED (`gmres_lab.py:141`: `H = D ** -(2σ+1)`, no length factors) — the
        // lengths enter through `U` and `diag(a)⁻¹Ha` instead, in applyLowOrderColumns.
        w.fill(1);
        X.set(bvh.sP);
    }

    // --- the RECOMPUTED vertex-side node data (module TSDoc, "the tree is edge-based") ---
    const nodeMass = new Float64Array(bvh.nNodes);
    const nodeC = new Float64Array(3 * bvh.nNodes);
    const nodeRx = new Float64Array(bvh.nNodes);
    for (let id = bvh.nNodes - 1; id >= 0; id--) {
        let mass = 0;
        let cx = 0;
        let cy = 0;
        let cz = 0;
        for (let s = bvh.begin[id], e = bvh.end[id]; s < e; s++) {
            const ww = w[s];
            mass += ww;
            cx += ww * X[3 * s];
            cy += ww * X[3 * s + 1];
            cz += ww * X[3 * s + 2];
        }
        const inv = mass > 0 ? 1 / mass : 0;
        cx *= inv;
        cy *= inv;
        cz *= inv;
        let r2 = 0;
        for (let s = bvh.begin[id], e = bvh.end[id]; s < e; s++) {
            const dx = X[3 * s] - cx;
            const dy = X[3 * s + 1] - cy;
            const dz = X[3 * s + 2] - cz;
            const d2 = dx * dx + dy * dy + dz * dz;
            if (d2 > r2) r2 = d2;
        }
        nodeMass[id] = mass;
        nodeC[3 * id] = cx;
        nodeC[3 * id + 1] = cy;
        nodeC[3 * id + 2] = cz;
        nodeRx[id] = Math.sqrt(r2);
    }

    let far: Float64Array | null = null;
    if (source.kind === 'lists') {
        const { farA, farB } = source.bct;
        far = new Float64Array(source.bct.stats.nFar);
        for (let k = 0; k < far.length; k++) {
            far[k] = farKernel(nodeC, farA[k], farB[k], p, epsilon);
        }
    }

    const Lo: LowOrderA = {
        n,
        m,
        bvh,
        source,
        variant,
        p,
        epsilon,
        w,
        X,
        vOf,
        ell,
        nodeMass,
        nodeC,
        nodeRx,
        far,
        rowSum: new Float64Array(m),
        stats: {} as BctStats,
        diagSkipped: 0,
        ws: new Map(),
        scratch: { psi: new Float64Array(3 * m), phi: new Float64Array(3 * m) },
    };

    // `rowSum` THROUGH THE SAME APPLY (spec §2.2) — this is the line a future agent would
    // "simplify" into a separate diagonal sum, which breaks the bit-exact `L·1 = 0` and with it
    // H5 property 1. The input is the ones vector for the vertex variant (`W·1`) and `ell` for
    // the rs-identity variant (`H·ell`, which is `U·1` pushed through `H`).
    const probe = new Float64Array(3 * m);
    if (variant === 'vertex') probe.fill(1, 0, m);
    else probe.set(ell.subarray(0, m), 0);
    Lo.stats = applyW(Lo, probe, Lo.rowSum, 1);
    return Lo;
}

/**
 * `out = L^{2−s} v` for a SCALAR vertex function (length `n`); `out` is overwritten.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.2
 */
export function applyLowOrder(Lo: LowOrderA, v: ArrayLike<number>, out: Float64Array): void {
    applyLowOrderColumns(Lo, v, out, 1);
}

/**
 * `out = L^{2−s} V` for `nb` coordinate blocks of a vector field in the repo's coordinate-block
 * layout `V[c*n + i]` (src/core/sobolev/layout.ts:9) — `nb = 3` is spec §5.2's *"3 columns on the
 * same tree"*, which is what `precond.ts` applies per GMRES iteration. Column `c` is bit-identical
 * to {@link applyLowOrder} on block `c` (columns never mix; tested).
 *
 * The `2×` in `L = 2(diag(W·1) − W)` is `[RS §5.1.1]`'s Gagliardo form written as a matrix: the
 * double sum runs over ORDERED pairs `i != j`, so each unordered pair appears twice. Removing it
 * rescales `Ã⁻¹` by 1/2 and silently changes every `κ_eig` the gate reads.
 * @see .superpowers/sdd/2026-09-11-phase2h-queue/scratch/gmres_lab.py:171, :146
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.2
 */
export function applyLowOrderColumns(
    Lo: LowOrderA,
    V: ArrayLike<number>,
    out: Float64Array,
    nb: number,
): void {
    const { n, m, vOf, ell, rowSum, bvh } = Lo;
    const { psi, phi } = Lo.scratch;

    if (Lo.variant === 'vertex') {
        for (let s = 0; s < m; s++) {
            const i = vOf[s];
            for (let b = 0; b < nb; b++) psi[s * nb + b] = V[b * n + i];
        }
        applyW(Lo, psi, phi, nb);
        out.fill(0, 0, nb * n);
        for (let s = 0; s < m; s++) {
            const i = vOf[s];
            for (let b = 0; b < nb; b++) {
                // `rowSum_s * psi − phi_s` is `x − x` ON THE ONES VECTOR, because `rowSum` came
                // from this same apply (module TSDoc) — that is why `L·1 = 0` bit-exactly. For a
                // general constant `c·1` it is not (MEASURED `max|L·(7.5·1)|` = 5.3e-14 .. 9.9e-14,
                // scaling with `|c|`); see the module TSDoc.
                out[b * n + i] = 2 * (rowSum[s] * psi[s * nb + b] - phi[s * nb + b]);
            }
        }
        return;
    }

    // --- 'rs-identity' knock-out: 2 Uᵀ [diag(diag(a)⁻¹Ha) − H] U v, a = ell (module TSDoc) ---
    const sE = bvh.sE;
    for (let s = 0; s < m; s++) {
        const i1 = sE[2 * s];
        const i2 = sE[2 * s + 1];
        for (let b = 0; b < nb; b++) {
            // (U v)_I = ell_I (v_a + v_b)/2 — the |E|x|V| "average onto edges, times length" map.
            psi[s * nb + b] = ell[s] * (0.5 * (V[b * n + i1] + V[b * n + i2]));
        }
    }
    applyW(Lo, psi, phi, nb);
    out.fill(0, 0, nb * n);
    for (let s = 0; s < m; s++) {
        const i1 = sE[2 * s];
        const i2 = sE[2 * s + 1];
        for (let b = 0; b < nb; b++) {
            // Written as `(H ell)_I * (z_I / ell_I)` and NOT as `((H ell)_I / ell_I) * z_I`: on
            // `z = U·1 = ell` the first form is `(H ell)_I * 1 − (H ell)_I = 0` on identical bits,
            // so the knock-out's nullspace still contains the constants EXACTLY and its only
            // defect is the one it exists to exhibit (the singular `U`, nullity 2). The algebra is
            // identical; only the rounding differs.
            const mid = rowSum[s] * (psi[s * nb + b] / ell[s]) - phi[s * nb + b];
            const c = ell[s] * mid; // 2 * (0.5 ell_I) mid_I — the `2` of L = 2(...) and Uᵀ's 0.5.
            out[b * n + i1] += c;
            out[b * n + i2] += c;
        }
    }
}
