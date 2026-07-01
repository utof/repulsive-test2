import type { Edge, Vec3 } from './testConfigs';

// Vector math helpers
const cross3D = (a: number[], b: number[]): number[] => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
];

const dot = (a: number[], b: number[]): number => a.reduce((sum, val, i) => sum + val * b[i], 0);

const subtract = (a: number[], b: number[]): number[] => a.map((val, i) => val - b[i]);

const scale = (s: number, v: number[]): number[] => v.map((x) => s * x);

export const norm = (v: number[]): number => Math.sqrt(dot(v, v));

const add = (a: number[], b: number[]): number[] => a.map((val, i) => val + b[i]);

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
        const e_I = subtract(vertices[i2], vertices[i1]);
        const ell_I = norm(e_I) + epsilon;

        for (const J of disjointPairs[I]) {
            const [j1, j2] = edges[J];
            const ell_J = norm(subtract(vertices[j2], vertices[j1])) + epsilon;

            let sumK = 0;
            for (const i of [i1, i2]) {
                for (const j of [j1, j2]) {
                    const d = subtract(vertices[i], vertices[j]);
                    const d_norm = norm(d) + epsilon;
                    const c_norm = norm(cross3D(e_I, d)) + epsilon;
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

    const zero: Vec3 = [0, 0, 0];

    const addToGrad = (idx: number, v: number[]) => {
        gradient[idx][0] += v[0] || 0;
        gradient[idx][1] += v[1] || 0;
        gradient[idx][2] += v[2] || 0;
    };

    const safeUnit = (v: number[]): { len: number; unit: Vec3 } => {
        const r = norm(v);
        if (r < 1e-14) return { len: r, unit: [0, 0, 0] };
        return { len: r, unit: scale(1 / r, v) as Vec3 };
    };

    // f(d,e) = (||e x d|| + eps)^alpha / (||d|| + eps)^beta
    // returns f, df/dd (w.r.t dvec), df/de (w.r.t evec)
    const kernelDerivs = (e: Vec3, dvec: Vec3) => {
        const { len: rd, unit: dHat } = safeUnit(dvec);
        const d_eps = rd + epsilon;

        const cvec = cross3D(e, dvec) as Vec3;
        const { len: rc } = safeUnit(cvec);
        const c_eps = rc + epsilon;

        const cPowA = Math.pow(c_eps, alpha);
        const dPowB = Math.pow(d_eps, beta);
        const f = cPowA / dPowB;

        // dc/dd = ((e x d) x e) / ||e x d||
        // dc/de = (d x (e x d)) / ||e x d||
        let dc_dd: Vec3 = zero;
        let dc_de: Vec3 = zero;
        if (rc >= 1e-14) {
            dc_dd = scale(1 / rc, cross3D(cvec, e)) as Vec3;
            dc_de = scale(1 / rc, cross3D(dvec, cvec)) as Vec3;
        }

        // df/dd = alpha*(c_eps)^(a-1)/d_eps^b * dc/dd  - beta*(c_eps)^a/d_eps^(b+1) * dHat
        const coeff_c = (alpha * Math.pow(c_eps, alpha - 1)) / dPowB;
        const coeff_d = (-beta * cPowA) / Math.pow(d_eps, beta + 1);

        const df_dd = add(scale(coeff_c, dc_dd), scale(coeff_d, dHat)) as Vec3;

        // df/de = alpha*(c_eps)^(a-1)/d_eps^b * dc/de
        const df_de = scale(coeff_c, dc_de) as Vec3;

        return { f, df_dd, df_de };
    };

    for (let I = 0; I < edges.length; I++) {
        const dis = disjointPairs[I];
        if (!dis || dis.length === 0) continue;

        const [i1, i2] = edges[I];
        const e_I = subtract(vertices[i2], vertices[i1]) as Vec3;

        const { len: reI, unit: eI_hat } = safeUnit(e_I);
        const ell_I = reI + epsilon;
        const ell_I_pow = Math.pow(ell_I, 1 - alpha);

        // d(ell_I^(1-a))/dv = (1-a)*ell_I^(-a) * d(ell_I)/dv
        // d(ell_I)/dv_i1 = -eI_hat, d(ell_I)/dv_i2 = +eI_hat
        const dPow_dv_i1 = scale((1 - alpha) * Math.pow(ell_I, -alpha), scale(-1, eI_hat)) as Vec3;
        const dPow_dv_i2 = scale((1 - alpha) * Math.pow(ell_I, -alpha), eI_hat) as Vec3;

        for (const J of dis) {
            const [j1, j2] = edges[J];

            const e_J = subtract(vertices[j2], vertices[j1]) as Vec3;
            const { len: reJ, unit: eJ_hat } = safeUnit(e_J);
            const ell_J = reJ + epsilon;

            // d(ell_J)/dv_j1 = -eJ_hat, d(ell_J)/dv_j2 = +eJ_hat
            const dEllJ_dv_j1 = scale(-1, eJ_hat) as Vec3;
            const dEllJ_dv_j2 = eJ_hat;

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
                df_dd: Vec3;
                df_de: Vec3;
            };

            const terms: Term[] = [];
            let sumF = 0;

            for (const { i, j } of pairs) {
                const dvec = subtract(vertices[i], vertices[j]) as Vec3;
                const { f, df_dd, df_de } = kernelDerivs(e_I, dvec);
                sumF += f;
                terms.push({ i, j, f, df_dd, df_de });
            }

            // Energy contribution (ordered pair):
            // E_IJ = 0.25 * ell_I^(1-a) * ell_J * sumF
            // So:
            // dE via ell_I^(1-a): 0.25 * ell_J * sumF * d(ell_I^(1-a))
            addToGrad(i1, scale(0.25 * ell_J * sumF, dPow_dv_i1));
            addToGrad(i2, scale(0.25 * ell_J * sumF, dPow_dv_i2));

            // dE via ell_J: 0.25 * ell_I^(1-a) * sumF * d(ell_J)
            addToGrad(j1, scale(0.25 * ell_I_pow * sumF, dEllJ_dv_j1));
            addToGrad(j2, scale(0.25 * ell_I_pow * sumF, dEllJ_dv_j2));

            const base = 0.25 * ell_I_pow * ell_J;

            // Per-term derivatives:
            // - dvec part goes to the specific endpoints (i and j) of that term
            // - e_I part goes to BOTH i1 and i2 (since e_I = v_i2 - v_i1) for every term
            for (const t of terms) {
                // dvec = v_i - v_j
                addToGrad(t.i, scale(base, t.df_dd));
                addToGrad(t.j, scale(-base, t.df_dd));

                // e_I = v_i2 - v_i1
                addToGrad(i1, scale(-base, t.df_de));
                addToGrad(i2, scale(base, t.df_de));
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
