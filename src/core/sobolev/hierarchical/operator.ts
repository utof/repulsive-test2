/**
 * The `SobolevOperator` wrapper of the hierarchical apply (Phase 2H) and the structural
 * `sigma = tr(A_bct)/n` of the shift `A_sigma = A + (sigma/n) 1 1^T`.
 *
 * `A_bct = M^T (diag(K1) - K) M + E^T (diag(K⁰1) - K⁰) E` (spec §2.2). `K1` / `K⁰1` come from
 * {@link prepareA}'s own apply on the ones vector and are consumed as they are: recomputing them
 * any other way breaks the bit-exact `A_bct 1 = 0` (apply.ts module TSDoc).
 *
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §2.2, §2.3, §5.1, §11.2 task 5
 * @see bench/gpu/k1-calibration.py:228-230 (the shift's form)
 */
import { applyA, type HierarchicalA } from './apply';

/**
 * Matrix-free operator on one flat `3n` vector in the coordinate-block layout.
 * NOT RE-ENTRANT: one instance = one caller at a time (the wrapped `HierarchicalA` owns shared
 * scratch, apply.ts `HierarchicalA` TSDoc). Why: the interface is spec §5.1:601-610 verbatim.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.1
 */
export interface SobolevOperator {
    applyA(u: ArrayLike<number>, out: Float64Array): void; // 3n -> 3n
    n: number;
}

/** A {@link SobolevOperator} that also carries its shift `sigma = tr(A_bct)/n` (scalar `A`). */
export interface HierarchicalOperator extends SobolevOperator {
    /** `tr(A_bct)/n` of the SCALAR `A` (the 3n operator is 3 copies of it). */
    sigma: number;
}

/**
 * `tr(A_bct)` of the scalar operator, in O(m) from `K1` / `K⁰1` — no basis-vector applies.
 *
 * Derivation: `tr(M^T L M) = sum_{I,J} L_IJ <M_I, M_J>_F`, nonzero only if edges I, J share a
 * vertex. The BCT sets `K_IJ = 0` exactly on shared-vertex pairs and on I = J (the energy omits
 * them; apply.ts `adjacentSkipped`), so `L_IJ = 0` for I != J there and `L_II = K1_I`. With
 * `|M_I|_F^2 = 2 |T_I|^2 / l_I^2` and `|E_I|_F^2 = 1/2` the trace is
 * `sum_I K1_I 2|T_I|^2/l_I^2 + K⁰1_I / 2`. The vanishing of the cross terms is VERIFIED, not
 * assumed: operator.test.ts compares against the densified diagonal at 6 (n, theta) cells.
 * Why: spec §2.3 forbids an exact trace in production; densifying is n applies (~64 min at 30k).
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §2.2, §2.3
 */
export function traceA(H: HierarchicalA): number {
    const { sT, sEll } = H.geom;
    let tr = 0;
    for (let s = 0; s < H.m; s++) {
        const tx = sT[3 * s];
        const ty = sT[3 * s + 1];
        const tz = sT[3 * s + 2];
        const ell = sEll[s];
        tr += (H.k1Hi[s] * 2 * (tx * tx + ty * ty + tz * tz)) / (ell * ell) + 0.5 * H.k1Low[s];
    }
    return tr;
}

/**
 * Wraps a prepared {@link HierarchicalA}; `sigma = traceA(H) / n` (the `/n` is the shift's
 * `sigma/n` scaling's inner factor: `A_sigma = A + (sigma/n) 11^T`, spec §2.3). A production
 * path only: a knock-out `HierarchicalA` cannot exist unless `prepareA` was given `knockOut`.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §2.3, §5.1
 */
export function makeHierarchicalOperator(H: HierarchicalA): HierarchicalOperator {
    return {
        n: H.n,
        sigma: traceA(H) / H.n,
        applyA: (u, out) => applyA(H, u, out),
    };
}
