/**
 * Assembly of the high-order Sobolev inner-product matrix B (Repulsive Curves,
 * Yu/Schumacher/Crane 2021), specialized to the app's data model and
 * ε/ordered-pair conventions.
 *
 * B is the |V|×|V| scalar matrix such that, for scalar vertex functions u, v,
 * `uᵀBv = Σ_{I,J disjoint} w_IJ ⟨D_I u − D_J u, D_I v − D_J v⟩`. It is one of
 * the two summands of A = B + B⁰ (B⁰, the low-order term, is out of scope
 * here — milestone 3).
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §A ("High-order matrix (B)")
 * @see oracle/tpe_stage1_oracle.py (assemble_inner_product — the B-only slice of it)
 */
import type { Edge, Vec3 } from '../testConfigs';

/**
 * Assembles the |V|×|V| high-order matrix B.
 *
 * `disjointPairs` must be the ORDERED per-edge disjoint-partner lists from
 * {@link import('../tangentPointEnergy').calculateDisjointPairs} — both (I,J)
 * and (J,I) are visited, with NO extra 1/2: the app energy's ÷2 belongs to
 * E/dE only, the metric B is not halved.
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §A ("Ordered-pair convention")
 */
export function assembleBHigh(
    vertices: Vec3[],
    edges: Edge[],
    disjointPairs: number[][],
    alpha: number,
    beta: number,
    epsilon: number,
): number[][] {
    const n = vertices.length;
    const B: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));

    // sigma = (beta-1)/alpha - 1; distance exponent 2*sigma+1 (= 7/3 at alpha=3, beta=6).
    // Computed from alpha/beta at runtime rather than hardcoded, per spec.
    // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §A
    const s = (beta - 1) / alpha;
    const sigma = s - 1;
    const distExp = 2 * sigma + 1;

    // Per-edge geometry, computed once. ell^eps = ||e|| + eps with eps added AFTER the
    // norm (mirrors calculateEnergy's ell_I convention). The unit tangent T_I uses the
    // UNregularized direction e/||e||; zero vector when ||e|| < 1e-14, matching the
    // degenerate-direction guard used throughout the analytical gradient.
    // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §A ("The unit tangent uses the unregularized direction")
    // @see src/core/tangentPointEnergy.ts (ell_I / safeUnit convention)
    const ell = new Array<number>(edges.length);
    const tangent: Vec3[] = new Array(edges.length);
    for (let I = 0; I < edges.length; I++) {
        const [i1, i2] = edges[I];
        const ex = vertices[i2][0] - vertices[i1][0];
        const ey = vertices[i2][1] - vertices[i1][1];
        const ez = vertices[i2][2] - vertices[i1][2];
        const r = Math.sqrt(ex * ex + ey * ey + ez * ez);
        ell[I] = r + epsilon; // ε after norm — see module header / spec §A.
        if (r < 1e-14) {
            tangent[I] = [0, 0, 0]; // degenerate direction: zero unit vector, per spec §A.
        } else {
            const inv = 1 / r;
            tangent[I] = [ex * inv, ey * inv, ez * inv];
        }
    }

    for (let I = 0; I < edges.length; I++) {
        const Js = disjointPairs[I];
        if (!Js) continue;
        const [i1, i2] = edges[I];
        const idxI: [number, number] = [i1, i2];
        const ellI = ell[I];
        const TI = tangent[I];

        for (const J of Js) {
            const [j1, j2] = edges[J];
            const idxJ: [number, number] = [j1, j2];
            const ellJ = ell[J];
            const TJ = tangent[J];

            // w_IJ = (1/4) * ell_I^eps * ell_J^eps * sum_{i in I, j in J} 1/(||gamma_i-gamma_j||+eps)^(2sigma+1)
            let sumDist = 0;
            for (const ia of idxI) {
                for (const jb of idxJ) {
                    const dx = vertices[ia][0] - vertices[jb][0];
                    const dy = vertices[ia][1] - vertices[jb][1];
                    const dz = vertices[ia][2] - vertices[jb][2];
                    const dNorm = Math.sqrt(dx * dx + dy * dy + dz * dz) + epsilon; // ε after norm.
                    sumDist += 1 / dNorm ** distExp;
                }
            }
            const w = 0.25 * ellI * ellJ * sumDist;
            const dotT = TI[0] * TJ[0] + TI[1] * TJ[1] + TI[2] * TJ[2];

            // Index-level increment table, s_ab = (-1)^(a+b) over local endpoint indices a,b.
            // Ordered disjoint pairs (both (I,J) and (J,I) visited via disjointPairs) — do NOT
            // add another 1/2 here, that factor belongs to the app energy's E/dE, not to B.
            // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §A ("Index-level assembly")
            for (let a = 0; a < 2; a++) {
                for (let b = 0; b < 2; b++) {
                    const sign = (a + b) % 2 === 0 ? 1 : -1;
                    const ia = idxI[a];
                    const ib = idxI[b];
                    const ja = idxJ[a];
                    const jb = idxJ[b];

                    B[ia][ib] += (sign * w) / (ellI * ellI);
                    B[ia][jb] -= (sign * w * dotT) / (ellI * ellJ);
                    B[ja][jb] += (sign * w) / (ellJ * ellJ);
                    B[ja][ib] -= (sign * w * dotT) / (ellI * ellJ);
                }
            }
        }
    }

    // Final explicit symmetrization: remove roundoff-level asymmetry from accumulation order.
    // Off-diagonal pairs only — diagonal entries are unaffected by 0.5*(x+x) == x.
    // @see oracle/tpe_stage1_oracle.py ~line 238 ("Remove roundoff-level asymmetry from accumulation order")
    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            const avg = 0.5 * (B[i][j] + B[j][i]);
            B[i][j] = avg;
            B[j][i] = avg;
        }
    }

    return B;
}
