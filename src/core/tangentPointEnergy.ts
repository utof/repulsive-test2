/**
 * Tangent-point (repulsive-curves) energy and its analytical gradient.
 *
 * SINGLE SOURCE OF TRUTH — imported by src/core/optimizer.ts, src/store.ts, and
 * test_gradient.ts (which verifies gradientAnalytical against central finite differences).
 * @see docs/superpowers/specs/2026-07-01-tangent-point-hotpath-optimization-design.md
 *
 * PERF: the vector helpers are inlined to x/y/z scalars to avoid per-edge-pair array
 * allocation in the O(E²) loop (measured ~6× on the gradient hot path — the one optimization
 * that paid off). A flat-Float64Array variant and an integer-exponent Math.pow fast path
 * were both evaluated and REVERTED (≈5% and ≈0% respectively; not worth the complexity /
 * loss of bit-identity). Further speedups (flat data model through the UI, Barnes–Hut/BVH,
 * WASM-SIMD, WebGPU) are deferred to issue #1.
 *
 * 3D ONLY BY DESIGN. 2D configs embed as z=0: ‖cross3D‖ with z=0 equals |cross2D|, and the
 * gradient's z-components are identically 0. Do NOT re-introduce a 2D branch.
 * @see docs/superpowers/specs/2026-07-01-tangent-point-hotpath-optimization-design.md — "3D-only; 2D embeds as z=0"
 */
import type { Edge, Vec3 } from './testConfigs';

// Vector math helpers.
// cross3D/subtract/scale/add are inlined to scalar x/y/z math in the energy/gradient hot loops
// (see module header PERF note) to remove per-edge-pair `number[]` allocations. `norm` is kept
// because it is exported and consumed by the viewer (src/index.tsx, for the gradient-arrow
// magnitude); `dot` is kept because `norm` calls it. The inlined scalar expressions below
// preserve dot's left-assoc reduce-from-0 op order bit-identically (`x*x + y*y + z*z`).
// @see docs/superpowers/specs/2026-07-01-tangent-point-hotpath-optimization-design.md — "Optimizations (#1–3)"
const dot = (a: number[], b: number[]): number => a.reduce((sum, val, i) => sum + val * b[i], 0);

export const norm = (v: number[]): number => Math.sqrt(dot(v, v));

// Calculate disjoint edge pairs (edges sharing no vertices)
export function calculateDisjointPairs(edges: Edge[]): number[][] {
    const disjoint: number[][] = [];
    for (let i = 0; i < edges.length; i++) {
        disjoint[i] = [];
        for (let j = 0; j < edges.length; j++) {
            if (i === j) continue;
            const [a1, a2] = edges[i];
            const [b1, b2] = edges[j];
            if (a1 !== b1 && a1 !== b2 && a2 !== b1 && a2 !== b2) {
                disjoint[i].push(j);
            }
        }
    }
    return disjoint;
}

// Calculate energy
export function calculateEnergy(
    vertices: Vec3[],
    edges: Edge[],
    disjointPairs: number[][],
    alpha: number,
    beta: number,
    epsilon: number,
): number {
    let totalEnergy = 0;

    for (let I = 0; I < edges.length; I++) {
        if (!disjointPairs[I]) continue;

        const [i1, i2] = edges[I];
        // e_I = subtract(vertices[i2], vertices[i1]) — inlined to scalars.
        const eIx = vertices[i2][0] - vertices[i1][0];
        const eIy = vertices[i2][1] - vertices[i1][1];
        const eIz = vertices[i2][2] - vertices[i1][2];
        // ell_I = ||e_I|| + epsilon. ε is added AFTER the norm: kernel = (‖e×d‖ + ε)^α / (‖d‖ + ε)^β.
        // This is part of the energy DEFINITION (regularization), not a guard — moving ε
        // inside/outside the norm changes the energy. Same placement applies at every
        // ell_J/d_norm/c_norm below, and at d_eps/c_eps/ell_I/ell_J in gradientAnalytical.
        // @see docs/superpowers/specs/2026-07-01-tangent-point-hotpath-optimization-design.md — "Invariants the inlined kernel must preserve"
        const ell_I = Math.sqrt(eIx * eIx + eIy * eIy + eIz * eIz) + epsilon;

        for (const J of disjointPairs[I]) {
            const [j1, j2] = edges[J];
            // ell_J = norm(subtract(vertices[j2], vertices[j1])) + epsilon — ε after norm, see above.
            const eJx = vertices[j2][0] - vertices[j1][0];
            const eJy = vertices[j2][1] - vertices[j1][1];
            const eJz = vertices[j2][2] - vertices[j1][2];
            const ell_J = Math.sqrt(eJx * eJx + eJy * eJy + eJz * eJz) + epsilon;

            let sumK = 0;
            for (const i of [i1, i2]) {
                for (const j of [j1, j2]) {
                    // d = subtract(vertices[i], vertices[j]) — inlined to scalars.
                    const dx = vertices[i][0] - vertices[j][0];
                    const dy = vertices[i][1] - vertices[j][1];
                    const dz = vertices[i][2] - vertices[j][2];
                    const d_norm = Math.sqrt(dx * dx + dy * dy + dz * dz) + epsilon; // ε after norm, see above.
                    // c = cross3D(e_I, d); component term order matches cross3D, and ε is added
                    // AFTER the norm below — both are load-bearing invariants of the inlined kernel.
                    // @see docs/superpowers/specs/2026-07-01-tangent-point-hotpath-optimization-design.md — "Invariants the inlined kernel must preserve"
                    const cx = eIy * dz - eIz * dy;
                    const cy = eIz * dx - eIx * dz;
                    const cz = eIx * dy - eIy * dx;
                    const c_norm = Math.sqrt(cx * cx + cy * cy + cz * cz) + epsilon;
                    sumK += Math.pow(c_norm, alpha) / Math.pow(d_norm, beta);
                }
            }

            totalEnergy += 0.25 * Math.pow(ell_I, 1 - alpha) * ell_J * sumK;
        }
    }

    // /2 because disjointPairs lists BOTH (I,J) and (J,I) — every unordered pair is summed twice.
    // gradientAnalytical's final *0.5 loop divides for the SAME reason. Keep energy and gradient
    // in lockstep, or the analytical gradient stops matching the energy (test_gradient.ts fails).
    // @see docs/superpowers/specs/2026-07-01-tangent-point-hotpath-optimization-design.md — "Invariants the inlined kernel must preserve"
    return totalEnergy / 2;
}

// Finite difference gradient
export function gradientFiniteDiff(
    vertices: Vec3[],
    edges: Edge[],
    disjointPairs: number[][],
    alpha: number,
    beta: number,
    epsilon: number,
    h: number,
): Vec3[] {
    const gradient: Vec3[] = vertices.map(() => [0, 0, 0]);
    const E0 = calculateEnergy(vertices, edges, disjointPairs, alpha, beta, epsilon);

    for (let v = 0; v < vertices.length; v++) {
        for (let d = 0; d < 3; d++) {
            const perturbed: Vec3[] = vertices.map((vtx) => [...vtx] as Vec3);
            perturbed[v][d] += h;
            const E1 = calculateEnergy(perturbed, edges, disjointPairs, alpha, beta, epsilon);
            gradient[v][d] = (E1 - E0) / h;
        }
    }

    return gradient;
}

// Analytical gradient (matches calculateEnergy exactly, including /2)
export function gradientAnalytical(
    vertices: Vec3[],
    edges: Edge[],
    disjointPairs: number[][],
    alpha: number,
    beta: number,
    epsilon: number,
): Vec3[] {
    const gradient: Vec3[] = vertices.map(() => [0, 0, 0]);

    // addToGrad: scalar-component form of the original `(idx, v: number[])` version.
    // Keep the `|| 0` guards EXACTLY — preserved verbatim from the original addToGrad(idx, v[]):
    // coerces a missing/undefined component to 0.
    const addToGrad = (idx: number, x: number, y: number, z: number) => {
        gradient[idx][0] += x || 0;
        gradient[idx][1] += y || 0;
        gradient[idx][2] += z || 0;
    };

    // f(d,e) = (||e x d|| + eps)^alpha / (||d|| + eps)^beta
    // returns f, df/dd (w.r.t dvec), df/de (w.r.t evec) as scalar triples.
    // Vector math is inlined to x/y/z scalars here (subtract/scale/add/cross3D + safeUnit) to
    // avoid per-iteration array allocation (most of the measured speedup). Keep the x→y→z
    // accumulation order — it's what makes the inlining bit-identical to the pre-optimization
    // baseline (golden.test.ts asserts exact equality).
    // @see docs/superpowers/specs/2026-07-01-tangent-point-hotpath-optimization-design.md — "Invariants the inlined kernel must preserve"
    const kernelDerivs = (
        ex: number,
        ey: number,
        ez: number,
        dx: number,
        dy: number,
        dz: number,
    ) => {
        // safeUnit(dvec): guard tests the PRE-ε length `rd` (not rd+epsilon) with `< 1e-14` — a
        // ~0-length vector has no defined unit direction, so we zero that derivative. Keep the
        // `< 1e-14` / `>= 1e-14` exactly (see also the `rc` guard below).
        // @see docs/superpowers/specs/2026-07-01-tangent-point-hotpath-optimization-design.md — "Invariants the inlined kernel must preserve"
        const rd = Math.sqrt(dx * dx + dy * dy + dz * dz);
        let dHat_x = 0;
        let dHat_y = 0;
        let dHat_z = 0;
        if (!(rd < 1e-14)) {
            const invRd = 1 / rd;
            dHat_x = invRd * dx;
            dHat_y = invRd * dy;
            dHat_z = invRd * dz;
        }
        const d_eps = rd + epsilon; // ε after norm — see the canonical note at ell_I in calculateEnergy.

        // cvec = cross3D(e, dvec); component term order matches cross3D — see the kernelDerivs
        // op-order note above.
        const cx = ey * dz - ez * dy;
        const cy = ez * dx - ex * dz;
        const cz = ex * dy - ey * dx;
        // safeUnit(cvec): only the length (rc) is used here.
        const rc = Math.sqrt(cx * cx + cy * cy + cz * cz);
        const c_eps = rc + epsilon; // ε after norm — see the canonical note at ell_I in calculateEnergy.

        const cPowA = Math.pow(c_eps, alpha);
        const dPowB = Math.pow(d_eps, beta);
        const f = cPowA / dPowB;

        // dc/dd = ((e x d) x e) / ||e x d||
        // dc/de = (d x (e x d)) / ||e x d||
        let dc_dd_x = 0;
        let dc_dd_y = 0;
        let dc_dd_z = 0;
        let dc_de_x = 0;
        let dc_de_y = 0;
        let dc_de_z = 0;
        if (rc >= 1e-14) {
            // Same PRE-ε guard rule as `rd` above (rc, not rc+epsilon). Keep `>=` and 1e-14 exactly.
            const invRc = 1 / rc;
            // cross3D(cvec, e)
            const cce_x = cy * ez - cz * ey;
            const cce_y = cz * ex - cx * ez;
            const cce_z = cx * ey - cy * ex;
            dc_dd_x = invRc * cce_x;
            dc_dd_y = invRc * cce_y;
            dc_dd_z = invRc * cce_z;
            // cross3D(dvec, cvec)
            const dcc_x = dy * cz - dz * cy;
            const dcc_y = dz * cx - dx * cz;
            const dcc_z = dx * cy - dy * cx;
            dc_de_x = invRc * dcc_x;
            dc_de_y = invRc * dcc_y;
            dc_de_z = invRc * dcc_z;
        }

        // df/dd = alpha*(c_eps)^(a-1)/d_eps^b * dc/dd  - beta*(c_eps)^a/d_eps^(b+1) * dHat
        const coeff_c = (alpha * Math.pow(c_eps, alpha - 1)) / dPowB;
        const coeff_d = (-beta * cPowA) / Math.pow(d_eps, beta + 1);

        // df_dd = add(scale(coeff_c, dc_dd), scale(coeff_d, dHat)) — per component.
        const df_dd_x = coeff_c * dc_dd_x + coeff_d * dHat_x;
        const df_dd_y = coeff_c * dc_dd_y + coeff_d * dHat_y;
        const df_dd_z = coeff_c * dc_dd_z + coeff_d * dHat_z;

        // df/de = alpha*(c_eps)^(a-1)/d_eps^b * dc/de  →  scale(coeff_c, dc_de) per component.
        const df_de_x = coeff_c * dc_de_x;
        const df_de_y = coeff_c * dc_de_y;
        const df_de_z = coeff_c * dc_de_z;

        return { f, df_dd_x, df_dd_y, df_dd_z, df_de_x, df_de_y, df_de_z };
    };

    for (let I = 0; I < edges.length; I++) {
        const dis = disjointPairs[I];
        if (!dis || dis.length === 0) continue;

        const [i1, i2] = edges[I];
        // e_I = subtract(vertices[i2], vertices[i1]) — inlined to scalars.
        const eIx = vertices[i2][0] - vertices[i1][0];
        const eIy = vertices[i2][1] - vertices[i1][1];
        const eIz = vertices[i2][2] - vertices[i1][2];

        // safeUnit(e_I): same PRE-ε `< 1e-14` guard rule as `rd`/`rc` in kernelDerivs above.
        const reI = Math.sqrt(eIx * eIx + eIy * eIy + eIz * eIz);
        let eI_hat_x = 0;
        let eI_hat_y = 0;
        let eI_hat_z = 0;
        if (!(reI < 1e-14)) {
            const invReI = 1 / reI;
            eI_hat_x = invReI * eIx;
            eI_hat_y = invReI * eIy;
            eI_hat_z = invReI * eIz;
        }
        const ell_I = reI + epsilon; // ε after norm — see the canonical note at ell_I in calculateEnergy.
        const ell_I_pow = Math.pow(ell_I, 1 - alpha);

        // d(ell_I^(1-a))/dv = (1-a)*ell_I^(-a) * d(ell_I)/dv
        // d(ell_I)/dv_i1 = -eI_hat, d(ell_I)/dv_i2 = +eI_hat
        // scale((1-a)*ell_I^(-a), ·) shared across i1/i2; `1-alpha`/pow are identical both sites.
        const dPowCoeff = (1 - alpha) * Math.pow(ell_I, -alpha);
        const dPow_dv_i1_x = dPowCoeff * (-1 * eI_hat_x);
        const dPow_dv_i1_y = dPowCoeff * (-1 * eI_hat_y);
        const dPow_dv_i1_z = dPowCoeff * (-1 * eI_hat_z);
        const dPow_dv_i2_x = dPowCoeff * eI_hat_x;
        const dPow_dv_i2_y = dPowCoeff * eI_hat_y;
        const dPow_dv_i2_z = dPowCoeff * eI_hat_z;

        for (const J of dis) {
            const [j1, j2] = edges[J];

            // e_J = subtract(vertices[j2], vertices[j1]) — inlined to scalars.
            const eJx = vertices[j2][0] - vertices[j1][0];
            const eJy = vertices[j2][1] - vertices[j1][1];
            const eJz = vertices[j2][2] - vertices[j1][2];
            // safeUnit(e_J): same PRE-ε `< 1e-14` guard rule as `rd`/`rc` in kernelDerivs above.
            const reJ = Math.sqrt(eJx * eJx + eJy * eJy + eJz * eJz);
            let eJ_hat_x = 0;
            let eJ_hat_y = 0;
            let eJ_hat_z = 0;
            if (!(reJ < 1e-14)) {
                const invReJ = 1 / reJ;
                eJ_hat_x = invReJ * eJx;
                eJ_hat_y = invReJ * eJy;
                eJ_hat_z = invReJ * eJz;
            }
            const ell_J = reJ + epsilon; // ε after norm — see the canonical note at ell_I in calculateEnergy.

            // d(ell_J)/dv_j1 = -eJ_hat, d(ell_J)/dv_j2 = +eJ_hat
            // dEllJ_dv_j1 = scale(-1, eJ_hat); dEllJ_dv_j2 = eJ_hat (used directly below).
            const dEllJ_dv_j1_x = -1 * eJ_hat_x;
            const dEllJ_dv_j1_y = -1 * eJ_hat_y;
            const dEllJ_dv_j1_z = -1 * eJ_hat_z;

            // Precompute all 4 endpoint pairs for this (I,J)
            const pairs = [
                { i: i1, j: j1 },
                { i: i1, j: j2 },
                { i: i2, j: j1 },
                { i: i2, j: j2 },
            ];

            type Term = {
                i: number;
                j: number;
                f: number;
                df_dd_x: number;
                df_dd_y: number;
                df_dd_z: number;
                df_de_x: number;
                df_de_y: number;
                df_de_z: number;
            };

            const terms: Term[] = [];
            let sumF = 0;

            for (const { i, j } of pairs) {
                // dvec = subtract(vertices[i], vertices[j]) — inlined to scalars.
                const dx = vertices[i][0] - vertices[j][0];
                const dy = vertices[i][1] - vertices[j][1];
                const dz = vertices[i][2] - vertices[j][2];
                const k = kernelDerivs(eIx, eIy, eIz, dx, dy, dz);
                sumF += k.f;
                terms.push({
                    i,
                    j,
                    f: k.f,
                    df_dd_x: k.df_dd_x,
                    df_dd_y: k.df_dd_y,
                    df_dd_z: k.df_dd_z,
                    df_de_x: k.df_de_x,
                    df_de_y: k.df_de_y,
                    df_de_z: k.df_de_z,
                });
            }

            // Energy contribution (ordered pair):
            // E_IJ = 0.25 * ell_I^(1-a) * ell_J * sumF
            // So:
            // dE via ell_I^(1-a): 0.25 * ell_J * sumF * d(ell_I^(1-a))
            const sI = 0.25 * ell_J * sumF;
            addToGrad(i1, sI * dPow_dv_i1_x, sI * dPow_dv_i1_y, sI * dPow_dv_i1_z);
            addToGrad(i2, sI * dPow_dv_i2_x, sI * dPow_dv_i2_y, sI * dPow_dv_i2_z);

            // dE via ell_J: 0.25 * ell_I^(1-a) * sumF * d(ell_J)
            const sJ = 0.25 * ell_I_pow * sumF;
            addToGrad(j1, sJ * dEllJ_dv_j1_x, sJ * dEllJ_dv_j1_y, sJ * dEllJ_dv_j1_z);
            addToGrad(j2, sJ * eJ_hat_x, sJ * eJ_hat_y, sJ * eJ_hat_z);

            const base = 0.25 * ell_I_pow * ell_J;

            // Per-term derivatives:
            // - dvec part goes to the specific endpoints (i and j) of that term
            // - e_I part goes to BOTH i1 and i2 (since e_I = v_i2 - v_i1) for every term
            for (const t of terms) {
                // dvec = v_i - v_j
                addToGrad(t.i, base * t.df_dd_x, base * t.df_dd_y, base * t.df_dd_z);
                addToGrad(t.j, -base * t.df_dd_x, -base * t.df_dd_y, -base * t.df_dd_z);

                // e_I = v_i2 - v_i1
                addToGrad(i1, -base * t.df_de_x, -base * t.df_de_y, -base * t.df_de_z);
                addToGrad(i2, base * t.df_de_x, base * t.df_de_y, base * t.df_de_z);
            }
        }
    }

    // /2 because disjointPairs lists BOTH (I,J) and (J,I) — every unordered pair is summed twice.
    // Same reason calculateEnergy divides by 2 (see its `return totalEnergy / 2` above). Keep
    // energy and gradient in lockstep, or the analytical gradient stops matching the energy
    // (test_gradient.ts fails).
    // @see docs/superpowers/specs/2026-07-01-tangent-point-hotpath-optimization-design.md — "Invariants the inlined kernel must preserve"
    for (let v = 0; v < gradient.length; v++) {
        gradient[v][0] *= 0.5;
        gradient[v][1] *= 0.5;
        gradient[v][2] *= 0.5;
    }

    return gradient;
}
