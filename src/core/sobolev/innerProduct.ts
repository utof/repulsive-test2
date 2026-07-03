/**
 * Assembly of the high-order Sobolev inner-product matrix B (Repulsive Curves,
 * Yu/Schumacher/Crane 2021), specialized to the app's data model and
 * ε/ordered-pair conventions.
 *
 * B is the |V|×|V| scalar matrix such that, for scalar vertex functions u, v,
 * `uᵀBv = Σ_{I,J disjoint} w_IJ ⟨D_I u − D_J u, D_I v − D_J v⟩`. It is one of
 * the two summands of A = B + B⁰ ({@link assembleBLow} assembles the low-order
 * term B⁰; {@link assembleA} sums the two).
 *
 * Storage layout (solver-perf Task 5): the REAL bodies are the `*Flat`
 * functions accumulating into a flat row-major n×n `Float64Array`
 * (`B[i*n + j]`); the `number[][]` exports are thin allocation wrappers kept as
 * the reference/golden surface. ONLY the accumulation matrix went flat —
 * vertices stay `Vec3[]` (vertex-layout flattening was measured as a dead end
 * on the inlined kernels, briefing §2.1). Kernel arithmetic, ε placement, and
 * accumulation order are verbatim from the pre-flat nested bodies, so flat and
 * nested results are bit-identical.
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §A ("High-order matrix (B)")
 * @see oracle/tpe_stage1_oracle.py (assemble_inner_product — the B-only slice of it)
 * @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 5)
 */
import type { Edge, Vec3 } from '../testConfigs';
import { timed } from './phaseTimings';

/**
 * Assembles the |V|×|V| high-order matrix B into a flat row-major
 * `Float64Array` (`B[i*n + j]`) — the hot-path form consumed by
 * `solveSaddleFromA` (see `./linsolve`).
 *
 * `disjointPairs` must be the ORDERED per-edge disjoint-partner lists from
 * {@link import('../tangentPointEnergy').calculateDisjointPairs} — both (I,J)
 * and (J,I) are visited, with NO extra 1/2: the app energy's ÷2 belongs to
 * E/dE only, the metric B is not halved.
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §A ("Ordered-pair convention")
 * @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 5)
 */
export function assembleBHighFlat(
    vertices: Vec3[],
    edges: Edge[],
    disjointPairs: number[][],
    alpha: number,
    beta: number,
    epsilon: number,
): Float64Array {
    const n = vertices.length;
    const B = new Float64Array(n * n); // zero-initialized, like the nested fill(0)

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

                    B[ia * n + ib] += (sign * w) / (ellI * ellI);
                    B[ia * n + jb] -= (sign * w * dotT) / (ellI * ellJ);
                    B[ja * n + jb] += (sign * w) / (ellJ * ellJ);
                    B[ja * n + ib] -= (sign * w * dotT) / (ellI * ellJ);
                }
            }
        }
    }

    // Final explicit symmetrization: remove roundoff-level asymmetry from accumulation order.
    // Off-diagonal pairs only — diagonal entries are unaffected by 0.5*(x+x) == x.
    // @see oracle/tpe_stage1_oracle.py ~line 238 ("Remove roundoff-level asymmetry from accumulation order")
    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            const avg = 0.5 * (B[i * n + j] + B[j * n + i]);
            B[i * n + j] = avg;
            B[j * n + i] = avg;
        }
    }

    return B;
}

/**
 * Assembles the |V|×|V| high-order matrix B as `number[][]`.
 *
 * Reference/golden surface: thin allocation wrapper over
 * {@link assembleBHighFlat} (which holds the real body and all convention
 * anchors) — numerically bit-identical, only the storage layout differs.
 * Untimed, like before the flat split: 'bHigh' fires only at assembleAFlat's
 * internal call site.
 * @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 5)
 */
export function assembleBHigh(
    vertices: Vec3[],
    edges: Edge[],
    disjointPairs: number[][],
    alpha: number,
    beta: number,
    epsilon: number,
): number[][] {
    return unflattenSquare(
        assembleBHighFlat(vertices, edges, disjointPairs, alpha, beta, epsilon),
        vertices.length,
    );
}

/**
 * Assembles the |V|×|V| low-order matrix B⁰ into a flat row-major
 * `Float64Array` (Repulsive Curves, Yu/Schumacher/Crane 2021).
 *
 * B⁰ is the other summand of A = B + B⁰ ({@link assembleBHighFlat} assembles B and documents
 * the shared ε/ordered-pair conventions used here too). For edge averages
 * `u_I = (u_{i1}+u_{i2})/2`, `uᵀB⁰v = Σ_{I,J disjoint} w⁰_IJ (u_I − u_J)(v_I − v_J)`.
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §A ("Low-order matrix (B⁰)")
 * @see oracle/tpe_stage1_oracle.py (assemble_inner_product — the B0 slice, and tangent_point_kernel)
 * @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 5)
 */
export function assembleBLowFlat(
    vertices: Vec3[],
    edges: Edge[],
    disjointPairs: number[][],
    alpha: number,
    beta: number,
    epsilon: number,
): Float64Array {
    const n = vertices.length;
    const B0 = new Float64Array(n * n); // zero-initialized, like the nested fill(0)

    // Same sigma / distance-exponent convention as assembleBHighFlat, computed at runtime
    // from alpha/beta (not hardcoded), per spec.
    // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §A
    const s = (beta - 1) / alpha;
    const sigma = s - 1;
    const distExp = 2 * sigma + 1;

    // Per-edge geometry — identical convention to assembleBHighFlat (ell^eps with eps after
    // the norm; unregularized unit tangent with the 1e-14 degenerate guard). Recomputed here
    // rather than shared/cached across assemblers: assembleA calls both assemblers
    // independently rather than fusing their loops, for reviewability.
    // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §A
    const ell = new Array<number>(edges.length);
    const tangent: Vec3[] = new Array(edges.length);
    for (let I = 0; I < edges.length; I++) {
        const [i1, i2] = edges[I];
        const ex = vertices[i2][0] - vertices[i1][0];
        const ey = vertices[i2][1] - vertices[i1][1];
        const ez = vertices[i2][2] - vertices[i1][2];
        const r = Math.sqrt(ex * ex + ey * ey + ez * ez);
        ell[I] = r + epsilon; // ε after norm.
        if (r < 1e-14) {
            tangent[I] = [0, 0, 0]; // degenerate direction guard, matches assembleBHighFlat.
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
        const TI = tangent[I]; // ONLY I's tangent is used below — see the asymmetry note.

        for (const J of Js) {
            const [j1, j2] = edges[J];
            const idxJ: [number, number] = [j1, j2];
            const ellJ = ell[J];

            // k^2_4(gamma_i, gamma_j, T_I) = (||T_I x (gamma_i-gamma_j)|| + eps)^2 / (||gamma_i-gamma_j|| + eps)^4,
            // eps added after EACH norm separately (numerator cross-norm and denominator distance),
            // mirroring the oracle's tangent_point_kernel(p, q, T, alpha=2, beta=4, eps). The kernel
            // uses T_I (the FIRST edge's tangent) only — NOT a symmetrized T_I/T_J combination — so
            // w0_IJ != w0_JI in general; that asymmetry is expected/correct, not a bug.
            // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §A ("Low-order matrix (B0)")
            // @see oracle/tpe_stage1_oracle.py (tangent_point_kernel)
            let sumLow = 0;
            for (const ia of idxI) {
                for (const jb of idxJ) {
                    const dx = vertices[ia][0] - vertices[jb][0];
                    const dy = vertices[ia][1] - vertices[jb][1];
                    const dz = vertices[ia][2] - vertices[jb][2];
                    const dNorm = Math.sqrt(dx * dx + dy * dy + dz * dz) + epsilon; // ε after norm.
                    const cx = TI[1] * dz - TI[2] * dy;
                    const cy = TI[2] * dx - TI[0] * dz;
                    const cz = TI[0] * dy - TI[1] * dx;
                    const crossNorm = Math.sqrt(cx * cx + cy * cy + cz * cz) + epsilon; // ε after norm, separately.
                    const k24 = (crossNorm * crossNorm) / (dNorm * dNorm * dNorm * dNorm);
                    sumLow += k24 / dNorm ** distExp;
                }
            }
            const wLow = 0.25 * ellI * ellJ * sumLow;
            const q = 0.25 * wLow;

            // Increment table: q is constant across all four (a,b) — unlike B's table, there is NO
            // (-1)^(a+b) sign here. Ordered disjoint pairs (both (I,J) and (J,I) visited via
            // disjointPairs) with no extra 1/2, same convention as assembleBHighFlat.
            // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §A ("The index increments are...")
            for (let a = 0; a < 2; a++) {
                for (let b = 0; b < 2; b++) {
                    const ia = idxI[a];
                    const ib = idxI[b];
                    const ja = idxJ[a];
                    const jb = idxJ[b];
                    B0[ia * n + ib] += q;
                    B0[ia * n + jb] -= q;
                    B0[ja * n + ib] -= q;
                    B0[ja * n + jb] += q;
                }
            }
        }
    }

    // Final explicit symmetrization — same rationale as assembleBHighFlat.
    // @see oracle/tpe_stage1_oracle.py ~line 239 ("Remove roundoff-level asymmetry from accumulation order")
    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            const avg = 0.5 * (B0[i * n + j] + B0[j * n + i]);
            B0[i * n + j] = avg;
            B0[j * n + i] = avg;
        }
    }

    return B0;
}

/**
 * Assembles the |V|×|V| low-order matrix B⁰ as `number[][]`.
 *
 * Reference/golden surface: thin allocation wrapper over
 * {@link assembleBLowFlat} (which holds the real body and all convention
 * anchors) — numerically bit-identical, only the storage layout differs.
 * Untimed, like before the flat split: 'bLow' fires only at assembleAFlat's
 * internal call site.
 * @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 5)
 */
export function assembleBLow(
    vertices: Vec3[],
    edges: Edge[],
    disjointPairs: number[][],
    alpha: number,
    beta: number,
    epsilon: number,
): number[][] {
    return unflattenSquare(
        assembleBLowFlat(vertices, edges, disjointPairs, alpha, beta, epsilon),
        vertices.length,
    );
}

/**
 * Assembles A = B + B⁰ as a flat row-major |V|×|V| `Float64Array` — the
 * hot-path entry point consumed by `solveSaddleFromA` (see `./linsolve`),
 * which expands the three coordinate-diagonal blocks implicitly instead of
 * materializing `expandBlockDiag`.
 *
 * Calls the two assemblers separately rather than fusing their loops:
 * reviewability first, perf later (per milestone-3 spec). Carries the
 * 'assembleA'/'bHigh'/'bLow' phase-timing wraps — the nested {@link assembleA}
 * wrapper must NOT re-wrap, or calls would double-count.
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §A ("A = B + B0")
 * @see oracle/tpe_stage1_oracle.py (assemble_inner_product — "A = B + B0")
 * @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 5)
 */
export function assembleAFlat(
    vertices: Vec3[],
    edges: Edge[],
    disjointPairs: number[][],
    alpha: number,
    beta: number,
    epsilon: number,
): Float64Array {
    // Phase-timing wraps (opt-in, default-inert): the whole body is 'assembleA';
    // the two assembler calls are the 'bHigh'/'bLow' sub-phases that overlap it.
    // @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 1)
    return timed('assembleA', () => {
        const B = timed('bHigh', () =>
            assembleBHighFlat(vertices, edges, disjointPairs, alpha, beta, epsilon),
        );
        const B0 = timed('bLow', () =>
            assembleBLowFlat(vertices, edges, disjointPairs, alpha, beta, epsilon),
        );
        const n = vertices.length;
        const A = new Float64Array(n * n);
        for (let i = 0; i < n * n; i++) {
            A[i] = B[i] + B0[i];
        }
        return A;
    });
}

/**
 * Assembles A = B + B⁰, the |V|×|V| scalar Sobolev inner-product matrix as
 * `number[][]` (before the 3|V|×3|V| block-diagonal expansion — see
 * `expandBlockDiag` in `./layout`).
 *
 * Reference/golden surface: thin allocation wrapper over {@link assembleAFlat}
 * — numerically bit-identical, only the storage layout differs. No timed()
 * wrap here: assembleAFlat already records 'assembleA' internally, and a
 * second wrap would double-count calls.
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §A ("A = B + B0")
 * @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 5)
 */
export function assembleA(
    vertices: Vec3[],
    edges: Edge[],
    disjointPairs: number[][],
    alpha: number,
    beta: number,
    epsilon: number,
): number[][] {
    return unflattenSquare(
        assembleAFlat(vertices, edges, disjointPairs, alpha, beta, epsilon),
        vertices.length,
    );
}

// Flat row-major n×n → number[][] for the reference/golden wrapper surface.
// Pure copy — values (and their bit patterns) are untouched.
function unflattenSquare(flat: Float64Array, n: number): number[][] {
    const out: number[][] = new Array(n);
    for (let i = 0; i < n; i++) {
        const row = new Array<number>(n);
        for (let j = 0; j < n; j++) {
            row[j] = flat[i * n + j];
        }
        out[i] = row;
    }
    return out;
}
