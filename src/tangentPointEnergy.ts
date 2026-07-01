import type { Edge, Vec3 } from './testConfigs';

// Vector math helpers.
// NOTE (Opt #1, Task 3): cross3D/subtract/scale/add were inlined to scalar x/y/z math in the
// energy/gradient hot loops to remove per-edge-pair `number[]` allocations. `norm` is kept
// because it is exported and consumed by the viewer (src/index.tsx); `dot` is kept because
// `norm` calls it. The inlined scalar expressions preserve op order bit-identically
// (dot's left-assoc reduce from 0 == `x*x + y*y + z*z`; see golden.test.ts STRICT).
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
        // ell_I = ||e_I|| + epsilon. epsilon is added AFTER the norm (spec §Invariants).
        const ell_I = Math.sqrt(eIx * eIx + eIy * eIy + eIz * eIz) + epsilon;

        for (const J of disjointPairs[I]) {
            const [j1, j2] = edges[J];
            // ell_J = norm(subtract(vertices[j2], vertices[j1])) + epsilon.
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
                    const d_norm = Math.sqrt(dx * dx + dy * dy + dz * dz) + epsilon;
                    // c = cross3D(e_I, d); component term order matches cross3D (spec §Invariants).
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
    // Vector helpers (subtract/scale/add/cross3D + safeUnit) are inlined to scalar math here;
    // op order is preserved bit-identically (spec §Invariants).
    const kernelDerivs = (
        ex: number,
        ey: number,
        ez: number,
        dx: number,
        dy: number,
        dz: number,
    ) => {
        // safeUnit(dvec): guard tests the PRE-epsilon norm with `< 1e-14` (spec §Invariants).
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
        const d_eps = rd + epsilon; // epsilon added AFTER the norm (spec §Invariants).

        // cvec = cross3D(e, dvec); component term order matches cross3D (spec §Invariants).
        const cx = ey * dz - ez * dy;
        const cy = ez * dx - ex * dz;
        const cz = ex * dy - ey * dx;
        // safeUnit(cvec): only the length (rc) is used here.
        const rc = Math.sqrt(cx * cx + cy * cy + cz * cz);
        const c_eps = rc + epsilon; // epsilon added AFTER the norm (spec §Invariants).

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
            // keep `>=` and 1e-14 exactly (spec §Invariants).
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

        // safeUnit(e_I): guard tests PRE-epsilon norm with `< 1e-14` (spec §Invariants).
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
        const ell_I = reI + epsilon; // epsilon added AFTER the norm (spec §Invariants).
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
            // safeUnit(e_J): guard tests PRE-epsilon norm with `< 1e-14` (spec §Invariants).
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
            const ell_J = reJ + epsilon; // epsilon added AFTER the norm (spec §Invariants).

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

    // calculateEnergy divides by 2 (because disjointPairs contains both (I,J) and (J,I)),
    // so gradient must also be divided by 2 to match finite-diff.
    for (let v = 0; v < gradient.length; v++) {
        gradient[v][0] *= 0.5;
        gradient[v][1] *= 0.5;
        gradient[v][2] *= 0.5;
    }

    return gradient;
}
