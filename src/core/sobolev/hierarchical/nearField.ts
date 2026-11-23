/**
 * Near field of the Phase 2H hierarchical Sobolev operator: the repo's `endpoint4` quadrature
 * for one edge pair, the per-near-block accumulation (inline or from a precomputed array), and
 * the accumulator VARIANTS that make f32 and f64 one code path.
 *
 * PORT of `.superpowers/sdd/2026-09-11-phase2h-queue/scratch/bct.ts` (`quad4`, `quad4f32`,
 * `precomputeNear`, the near loop of `applyK`/`applyKF32`, `exponents`) — the code that produced
 * every `out-accuracy.json` oracle. The op ORDER of every kernel chain is kept exactly: the f32
 * noise oracles (`f32NoiseHi = 1.0639e-06`) are properties of that order.
 *
 * QUADRATURE (spec §5.4, mandated). The near field uses the repo's 4-point ENDPOINT rule —
 * `w_IJ = 1/4 l_I l_J sum over the four endpoint pairs 1/(d + eps)^{2sigma+1}` of
 * `assembleBHighFlat` (src/core/sobolev/innerProduct.ts:88-101), and the matching
 * `w0_IJ` of `assembleBLowFlat` (:236-252) — NOT the one-point midpoint rule of Eq. 32 of
 * Yu/Schumacher/Crane. The two disagree by ~34 % at n=960 with no BCT involved
 * (re-measured by `test/sobolev/hierarchical/apply.test.ts`, section 2): for a nearly adjacent
 * pair the closest ENDPOINT pair is about half the midpoint distance and the kernel goes as
 * `d^{-7/3}`. There is deliberately NO midpoint switch in `src/`.
 *
 * KERNEL VALUES returned (spec §5.2; `scratch/operators.ts:18-26` verified the constants against
 * the repo): `K_IJ = 2 w_IJ = 0.5 l_I l_J sum_ab 1/(d_ab + eps)^pHi` for the high-order kernel and
 * `K0_IJ = w0_IJ + w0_JI = 0.25 l_I l_J sum_ab [k2(d_ab, T_I) + k2(d_ab, T_J)]` for the low-order
 * one, so that `sum_{I<J} K_IJ (.)^2` equals the repo's ORDERED-pair accumulation. The factor 2
 * is REQUIRED (`assembleBHighFlat` visits both `(I,J)` and `(J,I)` with no extra 1/2 — see the
 * "do NOT add another 1/2" comment at innerProduct.ts:103-106); dropping it halves A.
 *
 * ADJACENT PAIRS. The discrete energy omits `J == I` and every `J` sharing a vertex with `I`
 * (`calculateDisjointPairs`, src/core/tangentPointEnergy.ts:44-58). The near field skips those
 * pairs (stored as an exact 0 when precomputed) and COUNTS them in `adjacentSkipped` — a counter
 * that is structurally `3m` on a closed curve (`3m - 2` open), which is what the tests assert; a
 * counter pinned at 0 would read like a passing gate (spec §11.2 task 4; ruling 1 of
 * `.superpowers/sdd/2026-10-09-phase2h-bct-gmres-design/task-4-brief.md`).
 *
 * eps PLACEMENT: `+ epsilon` AFTER each norm separately (distance, each cross-product norm),
 * mirroring innerProduct.ts:96 and :242-246. Per-exponent `pow` (`d ** p`) is used rather than a
 * combined `d^4 * d^p`: the repo's low-order kernel writes `k24 / dNorm ** distExp` with
 * `k24 = cross^2 / dNorm^4`, and this file evaluates `cross^2 / d ** pLow` with
 * `pLow = pHi + 4` — algebraically identical. **Keep whichever form is here. Do not "simplify" to
 * the other without re-running the pin in `nearField.test.ts`'s `endpoint4Kernel` f64 test, which
 * builds its oracle with the OTHER form (`k24 / dNorm ** pHi`) and pins this file's `d ** pLow`
 * against it at 4e-15** — a hard, executed, cross-form bar, which is the reason not to touch the
 * form.
 *
 * CORRECTED 2026-10-12. This paragraph previously justified the instruction with *"because the two
 * differ in f32 by more than the H1-b band is wide"*. That magnitude claim is WRONG and is
 * withdrawn: MEASURED in f32 over 12 004 samples (`d ∈ [1e-3, 1]` × four numerator magnitudes), the
 * two forms differ by **4.1e-8 mean / 2.4e-7 max per evaluation**, against an H1-b band at
 * torusKnot n=960 θ=0.5 that is **5.768e-5 wide** — i.e. the difference is ~240× NARROWER than the
 * band, not wider, and swapping the form would not move H1-b out of band on this evidence. The
 * INSTRUCTION is unchanged and correct; only the quantity it was anchored to was.
 * @see docs/superpowers/plans/phase2h-review-ledger/probes/powform.ts (the 12 004-sample f32 measurement)
 *
 * PRECONDITION of the f32 variant (per-exponent `pow` underflow): `fr(d ** pLow)` with
 * `pLow = 19/3` flushes to 0 for `d` below ~8e-8 (binary32 min normal 1.18e-38), and the low
 * kernel then returns `Infinity` — MEASURED 2026-10-09: `d = 5e-8` gives `Inf` in F32 against
 * `2.2e34` in F64. Unreachable for the bbox-normalised fixtures in scope (edge length ~1.6e-4 at
 * N = 30 000, and `d` is an endpoint distance of a NON-adjacent pair), but an inner solve on a
 * curve with near-coincident non-adjacent endpoints would go NaN silently. Keep the SCALE
 * PRECONDITION of `bvh.ts` (max extent 1) and the disjoint-pair skip; do not lower the scale.
 *
 * ONE TRAVERSAL, THREE ACCUMULATOR VARIANTS (spec §5.6, "non-negotiable"). JavaScript arithmetic
 * is f64; an f32 path is genuine only if EVERY intermediate is rounded. Rather than two bodies
 * (the spike's `applyK`/`applyKF32`, "change one, change the other"), every function here takes a
 * {@link Rounding} `fr` and applies it to every arithmetic result, and every piece of state and
 * geometry is allocated by the variant ({@link AccumulatorVariant}). `F64` is the identity, `F32`
 * is `Math.fround` with `Float32Array` state and DOWN-CAST geometry (the GPU would never see the
 * f64 coordinates), `F16_NEAR_KNOCKOUT` is the f32 path with the near accumulator truncated to
 * binary16 — spec §7.2 H1-b's knock-out (2), a test instrument. The f32 variant is ACCURACY-ONLY:
 * `Math.fround` per operation makes its timing meaningless, so never quote a wall-clock from it
 * (spec §3.5's 194.19 ms matvec is the f64 number).
 *
 * @see https://arxiv.org/abs/2006.07859 Eq. 31-32 (the operator), §4 (disjoint pairs)
 * @see src/core/sobolev/innerProduct.ts:38-140, :177-285 (the production assemblers this pins to)
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.2, §5.4, §5.6, §7.2, §11.2 task 4
 * @issue utof/repulsive-test2#57
 */
import type { Bct } from './bct';
import type { Bvh } from './bvh';

/**
 * `'hi'` = the high-order kernel `k^0_{2s+1}` (B); `'low'` = the low-order `k^2_{2s+5}` (B⁰).
 * @see https://arxiv.org/abs/2006.07859 Eq. 31-32
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.2
 */
export type KernelId = 'hi' | 'low';

/**
 * Rounding applied to every arithmetic result: identity (f64) or `Math.fround` (f32).
 * Why a function and not a flag: the variant must be able to round at EVERY site (module TSDoc).
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.6
 */
export type Rounding = (x: number) => number;

/**
 * State/geometry array at the variant's width.
 * Why a union and not `Float64Array`: a `Float32Array` holding the f32 state is what makes the
 * f32 variant genuine rather than f64-until-stored (memory `simulate-bugs-at-target-precision`).
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.6
 */
export type RealArray = Float64Array | Float32Array;

/**
 * An accumulator variant of the one traversal (spec §5.6). All four members must agree: a
 * `Float32Array` with an identity `round` would still be an f64 simulation (products and sums
 * are f64 until stored), which is exactly the defect memory `simulate-bugs-at-target-precision`
 * records. EXPONENT WIDTH: the exponent `p` of `d ** p` stays an f64 constant in every variant
 * (only the base and the result are rounded); a GPU port that holds `p` as an f32 literal
 * (7/3, 19/3 are not representable) evaluates a slightly different power — compare against
 * {@link F32} with that in mind, not as a bit-identity.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.6
 */
export interface AccumulatorVariant {
    readonly name: 'f64' | 'f32' | 'f16near';
    /** Applied to EVERY arithmetic result: kernel chain, percolation, far and near accumulation. */
    readonly round: Rounding;
    /**
     * Applied to the running NEAR accumulator `phi += k * psi` in place of `round`. Equal to
     * `round` except in the f16 knock-out. Why a separate hook: spec §11.2 task 4 words the
     * knock-out as "truncate the near accumulator to f16", nothing else.
     */
    readonly roundNear: Rounding;
    /** Allocates a zeroed state array of the variant's width. */
    readonly alloc: (n: number) => RealArray;
    /** Casts f64 geometry to the variant's width (identity for f64). */
    readonly cast: (a: Float64Array) => RealArray;
}

const identity: Rounding = (x) => x;

/**
 * f64: the reference accumulator. The identity rounding costs a call per op; accepted.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.6
 */
export const F64: AccumulatorVariant = {
    name: 'f64',
    round: identity,
    roundNear: identity,
    alloc: (n) => new Float64Array(n),
    cast: (a) => a,
};

/**
 * f32: every result through `Math.fround`, `Float32Array` state, down-cast geometry.
 * ACCURACY-ONLY — never quote a wall-clock from this variant (module TSDoc). Rounding placement
 * is pinned bit-exactly by `nearField.test.ts` / `apply.test.ts` ("ROUNDING-PLACEMENT PIN").
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.6, §7.2
 */
export const F32: AccumulatorVariant = {
    name: 'f32',
    round: Math.fround,
    roundNear: Math.fround,
    alloc: (n) => new Float32Array(n),
    cast: (a) => Float32Array.from(a),
};

/**
 * KNOCK-OUT INSTRUMENT, test/calibration only — never use in production. The f32 path with the
 * near accumulator truncated to binary16 (`Math.f16round`, Bun >= 1.1 / ES2025). Spec §7.2 H1-b
 * knock-out (2) predicts `97 * 4.88e-04 = 4.7e-02`, ~820x over the bar; `apply.test.ts` measures
 * it. Do not delete as dead code: a knock-out that cannot fire is the defect this phase has hit
 * twice (memory `gates-name-their-impossible-values`).
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §7.2, §7.11 item 7
 */
export const F16_NEAR_KNOCKOUT: AccumulatorVariant = {
    name: 'f16near',
    round: Math.fround,
    roundNear: Math.f16round,
    alloc: (n) => new Float32Array(n),
    cast: (a) => Float32Array.from(a),
};

/**
 * Kernel exponents. `s = (beta - 1) / alpha`, `sigma = s - 1`; `pHi = 2 sigma + 1` (7/3 at
 * alpha=3, beta=6) and `pLow = 2 sigma + 5` (19/3). Same arithmetic as innerProduct.ts:52-54.
 * @see src/core/sobolev/innerProduct.ts:52-54
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.2
 */
export function exponents(alpha: number, beta: number): { pHi: number; pLow: number } {
    const s = (beta - 1) / alpha;
    const sigma = s - 1;
    const pHi = 2 * sigma + 1;
    const pLow = 2 * sigma + 5;
    return { pHi, pLow };
}

/**
 * Slot-ordered edge geometry at a variant's width: unit tangents, regularised lengths, both
 * endpoint coordinate sets, endpoint vertex ids, and the (rounded) epsilon.
 * Why a struct: the kernel reads five arrays per pair; passing them once keeps the kernel's
 * signature identical across variants.
 */
export interface NearGeometry {
    sT: RealArray;
    sEll: RealArray;
    sXa: RealArray;
    sXb: RealArray;
    sE: Int32Array;
    epsilon: number;
}

/**
 * Builds {@link NearGeometry} from a {@link Bvh}. For `F64` the BVH's own arrays are returned
 * (no copy); for narrower variants the geometry is DOWN-CAST (`scratch/bct.ts:659-664`: "the GPU
 * would never see the f64 values").
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.6
 */
export function nearGeometry(bvh: Bvh, epsilon: number, variant: AccumulatorVariant): NearGeometry {
    return {
        sT: variant.cast(bvh.sT),
        sEll: variant.cast(bvh.sEll),
        sXa: variant.cast(bvh.sXa),
        sXb: variant.cast(bvh.sXb),
        sE: bvh.sE,
        epsilon: variant.round(epsilon),
    };
}

/**
 * The repo's 4-point endpoint quadrature for ONE edge pair (slots `s`, `t`), returning `K_IJ`
 * (high) or `K0_IJ` (low) directly — including the factor that makes `sum_{I<J} K_IJ (.)^2`
 * equal the repo's ordered-pair accumulation (module TSDoc). Every arithmetic result goes
 * through `fr`; with `fr = identity` this is bit-for-bit the spike's `quad4`, with
 * `Math.fround` its `quad4f32`.
 *
 * Endpoint pair order `(a, b)` for `a` in {Xa, Xb} of `s`, `b` in {Xa, Xb} of `t` — the loop
 * order of innerProduct.ts:91-92 (`for ia of idxI, for jb of idxJ`); keep it, the f32 sum
 * depends on it.
 * @param p the kernel exponent (`pHi` or `pLow` from {@link exponents}) — per-exponent `pow`.
 * @see src/core/sobolev/innerProduct.ts:88-101, :236-252
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.4
 */
export function endpoint4Kernel(
    g: NearGeometry,
    s: number,
    t: number,
    kernel: KernelId,
    p: number,
    fr: Rounding,
): number {
    const { sT, sXa, sXb, epsilon } = g;
    const tix = sT[3 * s];
    const tiy = sT[3 * s + 1];
    const tiz = sT[3 * s + 2];
    const tjx = sT[3 * t];
    const tjy = sT[3 * t + 1];
    const tjz = sT[3 * t + 2];
    const lij = fr(g.sEll[s] * g.sEll[t]);
    let sum = 0;
    for (let a = 0; a < 2; a++) {
        const A = a === 0 ? sXa : sXb;
        const ax = A[3 * s];
        const ay = A[3 * s + 1];
        const az = A[3 * s + 2];
        for (let b = 0; b < 2; b++) {
            const B = b === 0 ? sXa : sXb;
            const dx = fr(ax - B[3 * t]);
            const dy = fr(ay - B[3 * t + 1]);
            const dz = fr(az - B[3 * t + 2]);
            // eps AFTER the norm (innerProduct.ts:96).
            const d = fr(fr(Math.sqrt(fr(fr(fr(dx * dx) + fr(dy * dy)) + fr(dz * dz)))) + epsilon);
            if (kernel === 'hi') {
                sum = fr(sum + fr(1 / fr(d ** p)));
            } else {
                // k^2(x, y, T) = (|T x (x - y)| + eps)^2 / (|x - y| + eps)^{2s+5}, for T_I and T_J.
                const c1x = fr(fr(tiy * dz) - fr(tiz * dy));
                const c1y = fr(fr(tiz * dx) - fr(tix * dz));
                const c1z = fr(fr(tix * dy) - fr(tiy * dx));
                const c2x = fr(fr(tjy * dz) - fr(tjz * dy));
                const c2y = fr(fr(tjz * dx) - fr(tjx * dz));
                const c2z = fr(fr(tjx * dy) - fr(tjy * dx));
                const n1 = fr(
                    fr(Math.sqrt(fr(fr(fr(c1x * c1x) + fr(c1y * c1y)) + fr(c1z * c1z)))) + epsilon,
                );
                const n2 = fr(
                    fr(Math.sqrt(fr(fr(fr(c2x * c2x) + fr(c2y * c2y)) + fr(c2z * c2z)))) + epsilon,
                );
                sum = fr(sum + fr(fr(fr(n1 * n1) + fr(n2 * n2)) / fr(d ** p)));
            }
        }
    }
    // K = 2w = 0.5 l_I l_J sum; K0 = w0_IJ + w0_JI = 0.25 l_I l_J sum (module TSDoc).
    return kernel === 'hi' ? fr(fr(0.5 * lij) * sum) : fr(fr(0.25 * lij) * sum);
}

/** `true` iff the ordered pair of slots is one the energy omits (`J == I` or a shared vertex). */
function isAdjacent(sE: Int32Array, s: number, t: number): boolean {
    const ia = sE[2 * s];
    const ib = sE[2 * s + 1];
    const ja = sE[2 * t];
    const jb = sE[2 * t + 1];
    // The exact predicate of `calculateDisjointPairs` (src/core/tangentPointEnergy.ts:52).
    return ia === ja || ia === jb || ib === ja || ib === jb;
}

/**
 * Precomputed near-field entries for ONE kernel: `vals[offsets[k] + (s - begin[A]) * |B| + (t -
 * begin[B])]` is `K_st` for near block `k = (A, B)`, 0 for adjacent pairs. `kernel` is carried so
 * the apply can refuse a mismatch — the high- and low-order arrays have different values and
 * handing one to the other kernel's apply silently computes a plausible-looking wrong operator
 * (`scratch/operators.ts:74-82`).
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.6, §7.1
 */
export interface NearPre {
    kernel: KernelId;
    vals: RealArray;
    offsets: Int32Array;
    /** Pairs stored as 0 because the energy omits them (module TSDoc). `3m` on a closed curve. */
    adjacentSkipped: number;
}

/**
 * Assembles the near field once per gradient step (spec §5.6 `nearField.ts`): every near-block
 * entry of one kernel into a flat array, at the variant's width and arithmetic, so that
 * `applyK` with this array and `applyK` evaluating the kernel inline are bit-identical.
 * MEASURED in the spike: precomputing is 12.75x faster per apply; memory is `8 B * nearPairs`
 * (3.60 GB at chi=0, N=30 000 — which is why the chi=0 path must NOT use this, spec §7.1).
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.6, §7.1
 * @issue utof/repulsive-test2#57
 */
export function precomputeNear(
    bvh: Bvh,
    bct: Bct,
    kernel: KernelId,
    alpha: number,
    beta: number,
    epsilon: number,
    variant: AccumulatorVariant,
): NearPre {
    const { pHi, pLow } = exponents(alpha, beta);
    const p = kernel === 'hi' ? pHi : pLow;
    const fr = variant.round;
    const g = nearGeometry(bvh, epsilon, variant);
    const { begin, end, sE } = bvh;
    const nNear = bct.stats.nNear;
    const offsets = new Int32Array(nNear + 1);
    for (let k = 0; k < nNear; k++) {
        const A = bct.nearA[k];
        const B = bct.nearB[k];
        offsets[k + 1] = offsets[k] + (end[A] - begin[A]) * (end[B] - begin[B]);
    }
    const vals = variant.alloc(offsets[nNear]);
    let adjacentSkipped = 0;
    for (let k = 0; k < nNear; k++) {
        const A = bct.nearA[k];
        const B = bct.nearB[k];
        const bB = begin[B];
        const eB = end[B];
        let o = offsets[k];
        for (let s = begin[A], eA = end[A]; s < eA; s++) {
            for (let t = bB; t < eB; t++, o++) {
                if (isAdjacent(sE, s, t)) {
                    vals[o] = 0;
                    adjacentSkipped++;
                    continue;
                }
                vals[o] = endpoint4Kernel(g, s, t, kernel, p, fr);
            }
        }
    }
    return { kernel, vals, offsets, adjacentSkipped };
}

/**
 * Accumulates one near block `(A, B)` with the kernel evaluated INLINE:
 * `phi[s*nc + c] += K_st * psi[t*nc + c]` over the block's non-adjacent pairs, `nc` columns.
 * Returns the number of adjacent pairs skipped. `fr` rounds the product, `frNear` the running sum.
 * The chi=0 streaming reference uses this body (nothing precomputed, spec §7.1).
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.6, §7.1
 */
export function nearBlockInline(
    g: NearGeometry,
    begin: Int32Array,
    end: Int32Array,
    A: number,
    B: number,
    kernel: KernelId,
    p: number,
    psi: RealArray,
    phi: RealArray,
    nc: number,
    fr: Rounding,
    frNear: Rounding,
): number {
    const { sE } = g;
    const bB = begin[B];
    const eB = end[B];
    let skipped = 0;
    for (let s = begin[A], eA = end[A]; s < eA; s++) {
        const ob = s * nc;
        for (let t = bB; t < eB; t++) {
            if (isAdjacent(sE, s, t)) {
                skipped++;
                continue;
            }
            const kv = endpoint4Kernel(g, s, t, kernel, p, fr);
            const tb = t * nc;
            // FMA CONTRACT: the reference is UNFUSED — the product is rounded (`fr`) BEFORE the add,
            // here, in `nearBlockPre`, in the far scatter (apply.ts) and in the kernel chain. WGSL
            // permits contracting `a*b+c` into one rounding; a GPU port that reproduces f32 values
            // bit-for-bit must disable contraction (or expect last-bit differences, not a bug).
            for (let c = 0; c < nc; c++) phi[ob + c] = frNear(phi[ob + c] + fr(kv * psi[tb + c]));
        }
    }
    return skipped;
}

/**
 * Accumulates near block `k = (A, B)` from a {@link NearPre} array — same arithmetic as
 * {@link nearBlockInline} (an exact-0 entry is skipped exactly as an adjacent pair is), so the
 * two are bit-identical under the same variant. Returns the number of zero entries skipped.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.6
 */
export function nearBlockPre(
    pre: NearPre,
    k: number,
    begin: Int32Array,
    end: Int32Array,
    A: number,
    B: number,
    psi: RealArray,
    phi: RealArray,
    nc: number,
    fr: Rounding,
    frNear: Rounding,
): number {
    const { vals } = pre;
    const bB = begin[B];
    const eB = end[B];
    let o = pre.offsets[k];
    let skipped = 0;
    for (let s = begin[A], eA = end[A]; s < eA; s++) {
        const ob = s * nc;
        for (let t = bB; t < eB; t++, o++) {
            const kv = vals[o];
            if (kv === 0) {
                skipped++;
                continue;
            }
            const tb = t * nc;
            for (let c = 0; c < nc; c++) phi[ob + c] = frNear(phi[ob + c] + fr(kv * psi[tb + c]));
        }
    }
    return skipped;
}
