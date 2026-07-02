/**
 * Barycenter constraint Φ and its full Jacobian C = dΦ for the fractional
 * Sobolev-gradient saddle solve (Repulsive Curves, Yu/Schumacher/Crane 2021).
 *
 * Φ_bar(γ) = Σ_{I∈E} ℓ_I·(m_I − x₀) ∈ R³, with m_I the edge midpoint and x₀ a
 * fixed target point (set once at initialization — see {@link barycenterTarget}).
 * C ∈ R^{3×3|V|} indexes columns via the coordinate-block layout of `./layout`.
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B
 * @see oracle/tpe_stage1_oracle.py (length_weighted_barycenter / barycenter_phi_and_C)
 */
import type { Edge, Vec3 } from '../testConfigs';
import { blockIndex } from './layout';

/**
 * The length-weighted edge-midpoint barycenter x₀ = (Σ_I ℓ_I·m_I)/(Σ_I ℓ_I),
 * evaluated once at initialization as the constraint target.
 *
 * Uses RAW geometric lengths ℓ_I = ‖e_I‖ (no +ε) — see the ε note on
 * {@link barycenterPhiAndC}, which shares this convention.
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("set x₀ once at initialization")
 * @see oracle/tpe_stage1_oracle.py (length_weighted_barycenter)
 */
export function barycenterTarget(vertices: Vec3[], edges: Edge[]): Vec3 {
    let total = 0;
    const accum: Vec3 = [0, 0, 0];
    for (const [i1, i2] of edges) {
        const p1 = vertices[i1];
        const p2 = vertices[i2];
        const ex = p2[0] - p1[0];
        const ey = p2[1] - p1[1];
        const ez = p2[2] - p1[2];
        const ell = Math.sqrt(ex * ex + ey * ey + ez * ez);
        total += ell;
        accum[0] += ell * 0.5 * (p1[0] + p2[0]);
        accum[1] += ell * 0.5 * (p1[1] + p2[1]);
        accum[2] += ell * 0.5 * (p1[2] + p2[2]);
    }
    if (total < 1e-14) {
        // Last-resort fallback for a completely degenerate graph (all edges shorter
        // than the degenerate tolerance): plain vertex mean, mirroring the oracle.
        // @see oracle/tpe_stage1_oracle.py (length_weighted_barycenter, DEGENERATE_TOL)
        const n = vertices.length;
        const mean: Vec3 = [0, 0, 0];
        for (const v of vertices) {
            mean[0] += v[0] / n;
            mean[1] += v[1] / n;
            mean[2] += v[2] / n;
        }
        return mean;
    }
    return [accum[0] / total, accum[1] / total, accum[2] / total];
}

/**
 * Evaluates Φ = Σ_I ℓ_I·(m_I − x₀) and its FULL Jacobian C = dΦ ∈ R^{3×3|V|}
 * (rows = output coords, columns = coordinate-block flat layout via `blockIndex`).
 *
 * The Jacobian includes the dℓ_I terms — the paper defines C := dΦ with no
 * frozen-length approximation, so per edge (a,b) with rvec = m_I − x₀:
 * ∂Φ_r/∂γ_{a,c} += −T_{I,c}·rvec_r + (ℓ_I/2)·δ_rc, and +T_{I,c}·rvec_r + (ℓ_I/2)·δ_rc
 * for endpoint b. Endpoints and junctions need no special handling: every incident
 * edge contributes its term to the same vertex columns.
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("Use the full Jacobian, including length dependence")
 * @see local_files/2026-07-02-sobolev-formula-audit.md (Item 8 — Jacobian algebra verified)
 * @see oracle/tpe_stage1_oracle.py (barycenter_phi_and_C)
 */
export function barycenterPhiAndC(
    vertices: Vec3[],
    edges: Edge[],
    x0: Vec3,
): { phi: Vec3; C: number[][] } {
    const n = vertices.length;
    const phi: Vec3 = [0, 0, 0];
    const C: number[][] = Array.from({ length: 3 }, () => new Array<number>(3 * n).fill(0));

    for (const [i1, i2] of edges) {
        const p1 = vertices[i1];
        const p2 = vertices[i2];
        const ex = p2[0] - p1[0];
        const ey = p2[1] - p1[1];
        const ez = p2[2] - p1[2];
        // ℓ_I here is the RAW geometric length ‖e_I‖ — NO +ε. The constraint is
        // geometric, not part of the regularized energy; the handoff's ε rule covers
        // only lengths entering the energy. Do NOT "unify" this with the ℓ^ε used in
        // innerProduct.ts.
        // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("Use raw geometric lengths ... not ℓ^ε")
        const ell = Math.sqrt(ex * ex + ey * ey + ez * ez);
        // Unit tangent from the unregularized direction; zero vector when
        // ‖e‖ < 1e-14 (same degenerate guard as innerProduct.ts / the spec's
        // "If ‖e_I‖ < 1e-14, use T_I = 0 in its length derivative").
        // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B
        let T: Vec3;
        if (ell < 1e-14) {
            T = [0, 0, 0];
        } else {
            const inv = 1 / ell;
            T = [ex * inv, ey * inv, ez * inv];
        }
        const rvec: Vec3 = [
            0.5 * (p1[0] + p2[0]) - x0[0],
            0.5 * (p1[1] + p2[1]) - x0[1],
            0.5 * (p1[2] + p2[2]) - x0[2],
        ];
        phi[0] += ell * rvec[0];
        phi[1] += ell * rvec[1];
        phi[2] += ell * rvec[2];

        // dφ_I = dℓ_I·(m_I − x₀) + ℓ_I·dm_I, with dℓ_I = T_I·(dγ_b − dγ_a) and
        // dm_I = (dγ_a + dγ_b)/2.
        // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("For perturbations")
        for (let r = 0; r < 3; r++) {
            for (let c = 0; c < 3; c++) {
                const delta = r === c ? 1 : 0;
                C[r][blockIndex(c, i1, n)] += -T[c] * rvec[r] + 0.5 * ell * delta;
                C[r][blockIndex(c, i2, n)] += +T[c] * rvec[r] + 0.5 * ell * delta;
            }
        }
    }
    return { phi, C };
}
