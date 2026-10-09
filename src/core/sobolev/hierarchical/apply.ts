/**
 * The hierarchical apply of the Phase 2H Sobolev operator: `phi = K psi` for `nc` columns at
 * once over a block cluster tree (far blocks rank-1 through the BVH percolation, near blocks
 * exact through `nearField.ts`), and on top of it the 12-column `A` matvec
 *
 *     A u = Mᵀ (diag(K1) − K) M u  +  Eᵀ (diag(K⁰1) − K⁰) E u                     (Eq. 31)
 *
 * with `(M u)_I = ((u_i2 − u_i1) / l_I) T_I ∈ R³` and `(E u)_I = (u_i1 + u_i2) / 2`.
 *
 * PORT of `.superpowers/sdd/2026-09-11-phase2h-queue/scratch/bct.ts` (`applyK`, `applyKF32`,
 * `precomputeFar`, `makeWorkspace`) and `scratch/operators.ts` (`prepare`, `bctScalarA`,
 * `bctVectorA`, `exactA`), the code that produced every `out-accuracy.json` oracle. The
 * accumulation ORDER (upward percolation by descending node id, far blocks in list order,
 * downward percolation by ascending id, near blocks in list order, `psi` columns innermost) is
 * part of what the oracles measure; keep it.
 *
 * ONE TRAVERSAL (spec §5.6, non-negotiable). There is one block walker, {@link forEachBlock}:
 * over materialised lists (`{kind: 'lists'}`, theta > 0, reused across applies) or over `bct.ts`'s
 * streaming `traverseBct` (`{kind: 'stream'}`, which at theta = 0 IS the exact all-inadmissible
 * reference and materialises nothing — spec §7.1: its near list would be 112 MB and its near
 * values 3.60 GB at N=30 000). The far and near block bodies are the same functions either way,
 * so lists-vs-stream is bit-identical (tested). f64 and f32 are {@link AccumulatorVariant}s of
 * this one code path: the variant supplies the rounding of every arithmetic result and the
 * width of every state array (`nearField.ts` module TSDoc). The f32 variant is ACCURACY-ONLY —
 * `Math.fround` per operation makes its timing meaningless; never quote a wall-clock from it.
 * Spec §3.5's 194.19 ms 12-column matvec is the f64 number.
 *
 * THE PERCOLATION IS LOAD-BEARING (`scratch/bct.ts:19-23`). A far block `(A, B)` is applied as
 * `acc[A] += k(x_A, x_B) * s[B]` where `s[B] = sum_{J in B} l_J psi_J` was accumulated bottom-up
 * ONCE per apply and `acc` is pushed top-down ONCE to the primitives. A "simpler" far apply that
 * touches every primitive in A for every far block costs `sum_blocks |A|` instead of `#blocks` —
 * about two orders of magnitude more on these fixtures — and measuring THAT and calling it the
 * BCT constant is the failure utof/repulsive-test2#57 exists to prevent. Do not inline the
 * scatter.
 *
 * K1 AND K⁰1 come from THIS apply on the ones vector (spec §2.2): `diag(K1) − K` then annihilates
 * constants BIT-EXACTLY, because `K · 1` in the apply is the same sequence of operations that
 * produced `K1`. That is why `A_bct · 1 = 0` is asserted EXACTLY (`apply.test.ts` section 4) and
 * why the `SobolevOperator` wrapper (spec §11.2 task 5) must not recompute `K1` any other way (a dense row sum would break the exact zero
 * and, worse, the PSD-by-construction argument of §2.2).
 *
 * LAYOUT of the vector apply: the repo's coordinate-block layout `[x0..x(n-1), y0.., z0..]`
 * (src/core/sobolev/layout.ts:9, `expandBlockDiag`), NOT the spike's interleaved `U[3i + c]`.
 * `A` acts block-diagonally, so one `A` matvec is 9 high-order columns (3 components × 3 tangent
 * directions of `M u`) + 3 low-order columns = 12 columns of `K` (spec §3.5). `nc` is a parameter
 * of {@link applyK} because the `L_{2−s}` metric (spec §11.2 task 6) is 3 columns on the same tree.
 *
 * @see https://arxiv.org/abs/2006.07859 Eq. 31-32, App. B.2-B.3 ("we percolate these sums up and down the BVH")
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §2.2, §3.5, §5.6, §7.1, §7.2, §11.2 task 4
 * @see src/core/sobolev/layout.ts (coordinate-block layout)
 * @issue utof/repulsive-test2#57
 */
import {
    type Bct,
    type BctParams,
    type BctStats,
    type BlockVisitor,
    resolveBctParams,
    traverseBct,
} from './bct';
import type { Bvh } from './bvh';
import {
    type AccumulatorVariant,
    exponents,
    F64,
    type KernelId,
    type NearGeometry,
    type NearPre,
    nearBlockInline,
    nearBlockPre,
    nearGeometry,
    precomputeNear,
    type RealArray,
    type Rounding,
} from './nearField';

/**
 * Where the blocks come from. `'lists'`: a {@link Bct} built once and walked per apply (theta > 0,
 * the production path). `'stream'`: `traverseBct` is run inside every apply with the given
 * params and nothing is materialised — the chi=0 exact reference (spec §7.1) and any
 * no-cache mode. A `'stream'` source never carries precomputed far or near values.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.6, §7.1
 */
export type BlockSource = { kind: 'lists'; bct: Bct } | { kind: 'stream'; params: BctParams };

/**
 * Per-node accumulators of one apply, `ncols` wide, at the variant's width.
 * `s[N] = sum_{I in N} l_I psi_I` (upward), `acc[N]` = far-field contributions (downward).
 * Why per node and not per primitive: the percolation (module TSDoc, "load-bearing").
 * @see https://arxiv.org/abs/2006.07859 App. B.3.4
 * @issue utof/repulsive-test2#57
 */
export interface ApplyWorkspace {
    s: RealArray;
    acc: RealArray;
    ncols: number;
}

/**
 * Allocates an {@link ApplyWorkspace} for `ncols` columns. Why exported: `apply.test.ts` drives
 * {@link applyK} directly for the per-kernel oracles; {@link prepareA} caches one per `nc`.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.6
 */
export function makeWorkspace(
    bvh: Bvh,
    ncols: number,
    variant: AccumulatorVariant,
): ApplyWorkspace {
    return { s: variant.alloc(bvh.nNodes * ncols), acc: variant.alloc(bvh.nNodes * ncols), ncols };
}

/**
 * Everything one kernel's apply needs besides the vectors. `far[k]` is `k(p_A, p_B)` of far
 * block `k` (lists source only; `null` => evaluated inline), `near` the precomputed near field
 * (lists only; `null` => inline). `epsilon` is already rounded to the variant.
 * Why `far`/`near` are nullable rather than always present: the stream source (spec §7.1)
 * materialises nothing, and {@link applyK} refuses precomputed values with it.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.6, §7.1
 */
export interface KernelData {
    kernel: KernelId;
    /** Exponent `pHi` or `pLow` of {@link exponents}. */
    p: number;
    epsilon: number;
    far: RealArray | null;
    near: NearPre | null;
}

/**
 * What one {@link applyK} reports besides `phi`. Why `adjacentSkipped` is reported at all: a
 * counter the tests pin at `3m` / `3m − 2` (nearField.ts module TSDoc, "a counter pinned at 0
 * would read like a passing gate").
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.5, §7.1
 */
export interface ApplyStats {
    /** Adjacent pairs skipped by the near field — `3m` on a closed curve (nearField.ts). */
    adjacentSkipped: number;
    /** The traversal's counts (lists: the Bct's; stream: this walk's). */
    stats: BctStats;
}

/** Node geometry (barycentres, averaged tangents) at the variant's width, for the far kernel. */
interface NodeGeometry {
    C: RealArray;
    TN: RealArray;
    epsilon: number;
}

/**
 * The rank-1 far-block kernel `k(p_A, p_B)` at the cluster barycentres: `2 / (d + eps)^pHi`
 * (high; `k^0` does not involve T, so the `+` of the two orientations is a factor 2) and
 * `(|T_A × d| + eps)^2 + (|T_B × d| + eps)^2) / (d + eps)^pLow` (low), eps after EACH norm as
 * innerProduct.ts:216-222. A rank-1 lump IS a midpoint rule; the far field has no endpoint
 * choice (spec §5.4 item 2). `T_N` is the length-weighted average tangent, NOT renormalised
 * (per the paper; bvh.ts).
 * @see https://arxiv.org/abs/2006.07859 Eq. 32 and App. B.3.4
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.4
 */
function farKernel(
    g: NodeGeometry,
    A: number,
    B: number,
    kernel: KernelId,
    p: number,
    fr: Rounding,
): number {
    const { C, TN, epsilon } = g;
    const dx = fr(C[3 * A] - C[3 * B]);
    const dy = fr(C[3 * A + 1] - C[3 * B + 1]);
    const dz = fr(C[3 * A + 2] - C[3 * B + 2]);
    const d = fr(fr(Math.sqrt(fr(fr(fr(dx * dx) + fr(dy * dy)) + fr(dz * dz)))) + epsilon);
    if (kernel === 'hi') return fr(2 / fr(d ** p));
    const ax = TN[3 * A];
    const ay = TN[3 * A + 1];
    const az = TN[3 * A + 2];
    const bx = TN[3 * B];
    const by = TN[3 * B + 1];
    const bz = TN[3 * B + 2];
    const c1x = fr(fr(ay * dz) - fr(az * dy));
    const c1y = fr(fr(az * dx) - fr(ax * dz));
    const c1z = fr(fr(ax * dy) - fr(ay * dx));
    const c2x = fr(fr(by * dz) - fr(bz * dy));
    const c2y = fr(fr(bz * dx) - fr(bx * dz));
    const c2z = fr(fr(bx * dy) - fr(by * dx));
    const n1 = fr(fr(Math.sqrt(fr(fr(fr(c1x * c1x) + fr(c1y * c1y)) + fr(c1z * c1z)))) + epsilon);
    const n2 = fr(fr(Math.sqrt(fr(fr(fr(c2x * c2x) + fr(c2y * c2y)) + fr(c2z * c2z)))) + epsilon);
    return fr(fr(fr(n1 * n1) + fr(n2 * n2)) / fr(d ** p));
}

function nodeGeometry(bvh: Bvh, epsilon: number, variant: AccumulatorVariant): NodeGeometry {
    return { C: variant.cast(bvh.C), TN: variant.cast(bvh.TN), epsilon: variant.round(epsilon) };
}

/**
 * Precomputes `k(p_A, p_B)` for every far block of a {@link Bct}, at the variant's width and
 * arithmetic. Geometry-only: one-time BUILD cost, reused by every apply at a fixed configuration.
 * Why at the variant's precision rather than f64-then-cast (the spike's construction,
 * `run-accuracy.ts:104`): so that lists-with-precomputed-far and streaming-with-inline-far are
 * bit-identical in EVERY variant, and so the f32 operator contains no f64 arithmetic anywhere.
 * `apply.test.ts` reproduces the spike's f32 noise oracle with the cast construction to 0.31 %
 * (hi) / 0.036 % (low) — not bit-for-bit, see the accumulation-order note in {@link applyK} —
 * and shows this construction lands within 2x of it.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.6, §7.2
 */
export function precomputeFar(
    bvh: Bvh,
    bct: Bct,
    kernel: KernelId,
    alpha: number,
    beta: number,
    epsilon: number,
    variant: AccumulatorVariant,
): RealArray {
    const { pHi, pLow } = exponents(alpha, beta);
    const p = kernel === 'hi' ? pHi : pLow;
    const g = nodeGeometry(bvh, epsilon, variant);
    const out = variant.alloc(bct.stats.nFar);
    for (let k = 0; k < bct.stats.nFar; k++) {
        out[k] = farKernel(g, bct.farA[k], bct.farB[k], kernel, p, variant.round);
    }
    return out;
}

/**
 * THE one block walker. `onFar(k, A, B)` / `onNear(k, A, B)` with `k` the block's index in its
 * list (the index into `far[]` / `near.offsets`), in the traversal's order for both sources.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.6 ("ONE traversal"), §7.1
 */
function forEachBlock(
    bvh: Bvh,
    source: BlockSource,
    onFar: (k: number, A: number, B: number) => void,
    onNear: (k: number, A: number, B: number) => void,
): BctStats {
    if (source.kind === 'lists') {
        const { bct } = source;
        const { nFar, nNear } = bct.stats;
        for (let k = 0; k < nFar; k++) onFar(k, bct.farA[k], bct.farB[k]);
        for (let k = 0; k < nNear; k++) onNear(k, bct.nearA[k], bct.nearB[k]);
        return bct.stats;
    }
    let kf = 0;
    let kn = 0;
    const far: BlockVisitor = (A, B) => onFar(kf++, A, B);
    const near: BlockVisitor = (A, B) => onNear(kn++, A, B);
    return traverseBct(bvh, source.params, far, near).stats;
}

/**
 * `phi = K psi` for one kernel, `ws.ncols` columns at once, in SLOT order (`psi[slot*nc + c]`,
 * already permuted by `bvh.order`). `phi` is zeroed first. Batching columns is not cosmetic:
 * the traversal, the adjacency test and the kernel `pow` are shared across columns, and one
 * `A` matvec is 12 of them (module TSDoc).
 *
 * Far field (lists: `kd.far[k]`; stream or `far === null`: inline) through the percolation;
 * near field from `kd.near` when present (lists only), else inline through
 * `nearField.ts`. Throws if `kd.near.kernel !== kd.kernel` (the silent-wrong-operator trap) and
 * if a `'stream'` source carries precomputed values (spec §7.1).
 *
 * PER-APPLY ALLOCATION: `variant.cast(bvh.sEll)` (and, when inline, the node/near geometry) is
 * cast on every call — a copy for f32, free for f64. Accepted here (one `Float32Array.from` of
 * `m` floats against an O(nearPairs) apply); a GPU port holds the cast geometry ONCE per
 * prepared operator and never re-uploads it per apply.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §3.5, §5.6, §7.1
 * @issue utof/repulsive-test2#57
 */
export function applyK(
    bvh: Bvh,
    source: BlockSource,
    kd: KernelData,
    psi: RealArray,
    phi: RealArray,
    ws: ApplyWorkspace,
    variant: AccumulatorVariant,
): ApplyStats {
    const nc = ws.ncols;
    const fr = variant.round;
    const frNear = variant.roundNear;
    const { left, right, begin, end, nNodes } = bvh;
    const sEll = variant.cast(bvh.sEll);
    const { s, acc } = ws;
    const { kernel, p } = kd;
    const near = kd.near;
    if (near && near.kernel !== kernel) {
        throw new Error(`applyK: precomputed near field is '${near.kernel}', apply is '${kernel}'`);
    }
    if (source.kind === 'stream' && (kd.far || near)) {
        throw new Error('applyK: a streaming source carries no precomputed values (spec §7.1)');
    }
    const far = kd.far;
    const nodeG = far ? null : nodeGeometry(bvh, kd.epsilon, variant);
    const nearG: NearGeometry | null = near ? null : nearGeometry(bvh, kd.epsilon, variant);

    phi.fill(0);
    s.fill(0);
    acc.fill(0);

    // --- upward percolation: s[N] = sum_{I in N} l_I psi_I ---
    // Node ids are allocated pre-order (bvh.ts), so children have LARGER ids than their parent;
    // descending the id range visits children first.
    for (let id = nNodes - 1; id >= 0; id--) {
        const base = id * nc;
        if (left[id] < 0) {
            for (let slot = begin[id]; slot < end[id]; slot++) {
                const w = sEll[slot];
                const pb = slot * nc;
                for (let c = 0; c < nc; c++) s[base + c] = fr(s[base + c] + fr(w * psi[pb + c]));
            }
        } else {
            const lb = left[id] * nc;
            const rb = right[id] * nc;
            for (let c = 0; c < nc; c++) s[base + c] = fr(s[lb + c] + s[rb + c]);
        }
    }

    // --- far blocks: one scalar multiply-accumulate per block per column (the percolation) ---
    let adjacentSkipped = 0;
    const stats = forEachBlock(
        bvh,
        source,
        (k, A, B) => {
            const kv = far ? far[k] : farKernel(nodeG as NodeGeometry, A, B, kernel, p, fr);
            const ab = A * nc;
            const bb = B * nc;
            for (let c = 0; c < nc; c++) acc[ab + c] = fr(acc[ab + c] + fr(kv * s[bb + c]));
        },
        (k, A, B) => {
            // --- near blocks: exact pair sums over disjoint pairs (nearField.ts) ---
            adjacentSkipped += near
                ? nearBlockPre(near, k, begin, end, A, B, psi, phi, nc, fr, frNear)
                : nearBlockInline(
                      nearG as NearGeometry,
                      begin,
                      end,
                      A,
                      B,
                      kernel,
                      p,
                      psi,
                      phi,
                      nc,
                      fr,
                      frNear,
                  );
        },
    );

    // --- downward percolation, then scatter to primitives ---
    // ACCUMULATION ORDER — near first, then far (the spike, run-accuracy.ts, added far FIRST).
    // For f64 the order only moves reduction noise (every f64 oracle reproduces, delta_compress
    // to 2.39e-15); for f32 it reorders one addition per entry, so the archived f32 noise oracle
    // is reproduced to 0.31 % (hi) / 0.036 % (low) in apply.test.ts, NOT bit-for-bit. This is a
    // §5.6-vs-§11.2 conflict ruled for §5.6 (task-4 fix round 1, controller ruling): reproducing
    // the spike's order needs a far-only pass, the scatter, then a near-only pass — TWO traversals,
    // which §5.6 forbids — and the `'stream'` source has no materialised near list to replay
    // (§7.1: 112 MB of lists, 3.60 GB of near values at N = 30 000). Buffering the near sum and
    // adding it after the scatter is one traversal but computes `far + sum(near)` instead of
    // `((far + near_1) + near_2) + ...`, a different f32 rounding again. Do NOT "fix" the order
    // back to far-first: it cannot be done on the stream path and buys no bit-identity on lists.
    for (let id = 0; id < nNodes; id++) {
        const base = id * nc;
        if (left[id] < 0) {
            for (let slot = begin[id]; slot < end[id]; slot++) {
                const w = sEll[slot];
                const pb = slot * nc;
                for (let c = 0; c < nc; c++) phi[pb + c] = fr(phi[pb + c] + fr(w * acc[base + c]));
            }
        } else {
            const lb = left[id] * nc;
            const rb = right[id] * nc;
            for (let c = 0; c < nc; c++) {
                acc[lb + c] = fr(acc[lb + c] + acc[base + c]);
                acc[rb + c] = fr(acc[rb + c] + acc[base + c]);
            }
        }
    }

    return { adjacentSkipped, stats };
}

/**
 * The prepared hierarchical `A` (spec §5.6 `apply.ts`; the `SobolevOperator` wrapper of spec
 * §11.2 task 5 takes it with `sigma = tr(A_bct)/n`). Holds the tree, the block source, both
 * kernels' data, the geometry-only row sums `K1` / `K⁰1` (computed by {@link applyK} on the ones
 * vector — module TSDoc), and the workspaces.
 *
 * NOT RE-ENTRANT: `mv` and `ws` are shared scratch, so two concurrent {@link applyA} /
 * {@link applyAScalar} calls on one `HierarchicalA` corrupt each other — a GMRES that overlaps
 * matvecs (or a worker pool) needs one prepared `A` per in-flight apply, or `prepareA` per
 * worker. Sequential reuse is the design (the scratch exists to avoid per-apply allocation).
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §2.2, §3.5, §5.6, §11.2 task 5
 */
export interface HierarchicalA {
    bvh: Bvh;
    /** Vertex count `|V|`. */
    n: number;
    /** Edge count `m` (slots). */
    m: number;
    source: BlockSource;
    variant: AccumulatorVariant;
    alpha: number;
    beta: number;
    epsilon: number;
    hi: KernelData;
    low: KernelData;
    /** `K 1`, slot order, at the variant's width. */
    k1Hi: RealArray;
    /** `K⁰ 1`, slot order. */
    k1Low: RealArray;
    /** Stats of the traversal that produced `k1Hi` (lists: the Bct's; stream: the first walk). */
    stats: BctStats;
    /** Near-field adjacent pairs skipped per single-kernel apply (`3m` closed, `3m − 2` open). */
    adjacentSkipped: number;
    /** Per-column-count workspaces, keyed by `nc`. */
    ws: Map<number, ApplyWorkspace>;
    /** Edge-space scratch: `g`/`kg` 9m (high), `ev`/`ke` 3m (low), `uc` 3n (rounded input). */
    mv: { g: RealArray; kg: RealArray; ev: RealArray; ke: RealArray; uc: RealArray };
    /** Slot-order geometry at the variant's width, for M, E and their transposes. */
    geom: NearGeometry;
}

/**
 * Options of {@link prepareA}.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.6, §7.1, §7.2
 */
export interface PrepareOptions {
    alpha: number;
    beta: number;
    epsilon: number;
    /** Default {@link F64}. */
    variant?: AccumulatorVariant;
    /**
     * Assemble the near field once (`nearField.ts` `precomputeNear`) — lists source only; a
     * streaming source ignores it (spec §7.1). Default `true`.
     */
    precomputeNear?: boolean;
    /**
     * TESTS ONLY. Allows a knock-out instrument through: a block source built with
     * `invertAdmissibility` (spec §7.2 H1-a knock-out 2) or the `F16_NEAR_KNOCKOUT` variant
     * (H1-b knock-out 2). Without it {@link prepareA} throws, so an inverted or truncated
     * operator can never be applied by accident — reaching a knock-out from a production path
     * would be silent corruption (`.superpowers/sdd/2026-10-09-phase2h-bct-gmres-design/task-3-review.md`;
     * guarded since this file's first review). Default `false`.
     */
    knockOut?: boolean;
}

function workspace(H: HierarchicalA, nc: number): ApplyWorkspace {
    let ws = H.ws.get(nc);
    if (!ws) {
        ws = makeWorkspace(H.bvh, nc, H.variant);
        H.ws.set(nc, ws);
    }
    return ws;
}

/**
 * Builds a {@link HierarchicalA}: far values and (optionally) near values for both kernels, then
 * `K1` and `K⁰1` by applying `K` / `K⁰` to the ones vector through the SAME apply (spec §2.2).
 * A `'stream'` source precomputes nothing — at theta = 0 this is the exact operator, and it must
 * not allocate anything proportional to the pair count (spec §7.1).
 * @param n vertex count `|V|` (the BVH carries edges only).
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §2.2, §5.6, §7.1
 * @issue utof/repulsive-test2#57
 */
export function prepareA(
    bvh: Bvh,
    n: number,
    source: BlockSource,
    opts: PrepareOptions,
): HierarchicalA {
    const variant = opts.variant ?? F64;
    const { alpha, beta, epsilon } = opts;
    const { pHi, pLow } = exponents(alpha, beta);
    const m = bvh.order.length;
    const eps = variant.round(epsilon);
    const lists = source.kind === 'lists';
    const pre = lists && (opts.precomputeNear ?? true);
    // Knock-out instruments are refused unless explicitly let through (PrepareOptions.knockOut).
    const inverted = lists
        ? source.bct.params.invertAdmissibility
        : resolveBctParams(source.params).invertAdmissibility;
    if (!opts.knockOut && (inverted || variant.name === 'f16near')) {
        throw new Error(
            `prepareA: knock-out instrument in a production apply (invertAdmissibility=${inverted}, variant=${variant.name}); pass knockOut: true in a test`,
        );
    }
    const kernelData = (kernel: KernelId): KernelData => ({
        kernel,
        p: kernel === 'hi' ? pHi : pLow,
        epsilon: eps,
        far: lists ? precomputeFar(bvh, source.bct, kernel, alpha, beta, epsilon, variant) : null,
        near: pre ? precomputeNear(bvh, source.bct, kernel, alpha, beta, epsilon, variant) : null,
    });
    const hi = kernelData('hi');
    const low = kernelData('low');
    // K1 / K⁰1: geometry-only row sums, ONE-TIME build cost, through the same apply (§2.2).
    const ones = variant.alloc(m).fill(1);
    const ws1 = makeWorkspace(bvh, 1, variant);
    const k1Hi = variant.alloc(m);
    const k1Low = variant.alloc(m);
    const st = applyK(bvh, source, hi, ones, k1Hi, ws1, variant);
    applyK(bvh, source, low, ones, k1Low, ws1, variant);
    return {
        bvh,
        n,
        m,
        source,
        variant,
        alpha,
        beta,
        epsilon,
        hi,
        low,
        k1Hi,
        k1Low,
        stats: st.stats,
        adjacentSkipped: st.adjacentSkipped,
        ws: new Map([[1, ws1]]),
        mv: {
            g: variant.alloc(9 * m),
            kg: variant.alloc(9 * m),
            ev: variant.alloc(3 * m),
            ke: variant.alloc(3 * m),
            uc: variant.alloc(3 * n),
        },
        geom: nearGeometry(bvh, epsilon, variant),
    };
}

/**
 * `out = A u` for a SCALAR vertex function `u` (length n): 3 high-order columns (the three
 * components of `M u`) + 1 low-order column. `out` is overwritten (variant-rounded values; a
 * `Float64Array` holds f32 values exactly). Used by the `sigma = tr(A_bct)/n` trace (spec §11.2
 * task 5) and by the gates; the production matvec is {@link applyA}.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §3.5, §5.6
 */
export function applyAScalar(H: HierarchicalA, u: ArrayLike<number>, out: RealArray): void {
    applyColumns(H, u, out, 1);
}

/**
 * `out = A U` for a vector field `U ∈ R^{3|V|}` in the repo's COORDINATE-BLOCK layout
 * `U[c*n + i]` (src/core/sobolev/layout.ts:9) — the matvec a GMRES iteration performs. `A` is
 * block-diagonal over the three components (`expandBlockDiag`), so this is 9 high-order + 3
 * low-order columns of `K` in ONE traversal each (spec §3.5): the kernel work is amortised
 * across columns, which is why it is NOT 3x the scalar apply. Column `c` of the result is
 * bit-identical to {@link applyAScalar} on block `c` (columns never mix; tested).
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §3.5, §5.6
 * @see src/core/sobolev/layout.ts (`expandBlockDiag`)
 */
export function applyA(H: HierarchicalA, U: ArrayLike<number>, out: RealArray): void {
    applyColumns(H, U, out, 3);
}

/**
 * The shared body of {@link applyAScalar} (`nb = 1` block) and {@link applyA} (`nb = 3` blocks):
 * M / E, the two K applies with `3 nb` and `nb` columns, then Mᵀ / Eᵀ, every arithmetic result
 * rounded by the variant. Layout of the edge-space scratch: `g[s*(3nb) + 3b + k]` = component
 * `b` of `u`, tangent direction `k`; `ev[s*nb + b]`.
 */
function applyColumns(H: HierarchicalA, U: ArrayLike<number>, out: RealArray, nb: number): void {
    const { n, m, variant, geom, mv } = H;
    const fr = variant.round;
    const { sE, sT, sEll } = geom;
    const ncH = 3 * nb;
    const { g, kg, ev, ke, uc } = mv;

    // Inputs at the variant's width (an f32 GPU would receive f32 vectors).
    for (let i = 0; i < nb * n; i++) uc[i] = fr(U[i]);

    // M u (R³ per edge per block) and E u.
    for (let s = 0; s < m; s++) {
        const i1 = sE[2 * s];
        const i2 = sE[2 * s + 1];
        const tx = sT[3 * s];
        const ty = sT[3 * s + 1];
        const tz = sT[3 * s + 2];
        const ell = sEll[s];
        for (let b = 0; b < nb; b++) {
            const o = b * n;
            // du = (u[i2] − u[i1]) / l_I — on a constant this is (x − x)/l = 0 EXACTLY, the first
            // half of why A·1 = 0 holds bit-exactly (spec §2.2 anchor).
            const d = fr(fr(uc[o + i2] - uc[o + i1]) / ell);
            const gb = s * ncH + 3 * b;
            g[gb] = fr(d * tx);
            g[gb + 1] = fr(d * ty);
            g[gb + 2] = fr(d * tz);
            ev[s * nb + b] = fr(0.5 * fr(uc[o + i1] + uc[o + i2]));
        }
    }

    applyK(H.bvh, H.source, H.hi, g, kg, workspace(H, ncH), variant);
    applyK(H.bvh, H.source, H.low, ev, ke, workspace(H, nb), variant);

    // Mᵀ (diag(K1) − K) M u  +  Eᵀ (diag(K⁰1) − K⁰) E u.
    out.fill(0);
    for (let s = 0; s < m; s++) {
        const i1 = sE[2 * s];
        const i2 = sE[2 * s + 1];
        const tx = sT[3 * s];
        const ty = sT[3 * s + 1];
        const tz = sT[3 * s + 2];
        const ell = sEll[s];
        const w = H.k1Hi[s];
        const w0 = H.k1Low[s];
        for (let b = 0; b < nb; b++) {
            const o = b * n;
            const gb = s * ncH + 3 * b;
            const hx = fr(fr(w * g[gb]) - kg[gb]);
            const hy = fr(fr(w * g[gb + 1]) - kg[gb + 1]);
            const hz = fr(fr(w * g[gb + 2]) - kg[gb + 2]);
            const r = fr(fr(fr(fr(hx * tx) + fr(hy * ty)) + fr(hz * tz)) / ell);
            out[o + i2] = fr(out[o + i2] + r);
            out[o + i1] = fr(out[o + i1] - r);
            // k1Low · ev − ke: on the ones vector ke IS k1Low (same apply), so this is x − x = 0
            // exactly — the second half of the §2.2 anchor.
            const h = fr(0.5 * fr(fr(w0 * ev[s * nb + b]) - ke[s * nb + b]));
            out[o + i1] = fr(out[o + i1] + h);
            out[o + i2] = fr(out[o + i2] + h);
        }
    }
}
