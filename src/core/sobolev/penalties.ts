/**
 * Soft-constraint penalties for the Sobolev descent objective (5C): the
 * paper's penalty catalog — total length Σℓ_I, length difference
 * Σ_{v∈V_int}(ℓ_{I_v}−ℓ_{J_v})², field alignment Σℓ_I·|T_I×X|² with a
 * CONSTANT unit field X.
 *
 * Penalties enter the OBJECTIVE only: their analytic gradients ADD to dE
 * before the saddle solve and their energies ADD to the Armijo gate — they
 * never become constraint rows, and the fractional H^s inner product is
 * unchanged (paper-verbatim: "the energies considered here involve
 * lower-order derivatives … we can continue to use the fractional Sobolev
 * inner product without modification").
 *
 * Conventions (mirrored 1:1 with the oracle so goldens compare at 1e-12):
 * RAW edge lengths (no +ε — penalties are geometric, same rule as the
 * constraint rows); safe-unit guard T := 0 when ℓ < 1e-14 (same constant as
 * constraintSet.ts) so degenerate edges contribute ZERO everywhere; edges
 * and V_int vertices iterated in index order.
 * @see docs/superpowers/plans/2026-07-03-sobolev-penalties.md §2–§3
 * @see local_files/repulsive_orig_paper/SelfAvoiding.tex (lines 762–767 catalog, 769 H^s remark)
 * @see oracle/tpe_constraints_oracle.py (penalty_energy / penalty_gradient)
 */
import type { Edge, Vec3 } from '../testConfigs';

/**
 * Field-alignment penalty parameters: weight w and the constant field X.
 * X is normalized ONCE per evaluation; ‖X‖ < 1e-14 deactivates the field
 * penalty (plan §3 "X handling"). Spatially varying X(c_I) is deliberately
 * out of scope — it adds a ∂X/∂c chain-rule term the gradient below does not
 * carry (plan §1 OUT).
 * @see docs/superpowers/plans/2026-07-03-sobolev-penalties.md §2.3, §3
 */
export interface FieldPenalty {
    weight: number;
    X: Vec3;
}

/**
 * Penalty weights for the descent objective; absent/0 = off. All-off configs
 * are inert: callers gate on {@link penaltiesActive} so the descent code
 * paths stay BIT-IDENTICAL to the penalty-free build (plan §2.4 gate).
 * Weights are free knobs (paper gives no values — plan §3 ledger).
 * @see docs/superpowers/plans/2026-07-03-sobolev-penalties.md §2.4, §3
 */
export interface PenaltyConfig {
    totalLength?: number;
    lengthDiff?: number;
    field?: FieldPenalty;
}

// Resolved weights + unit field, computed once per evaluate call. The
// 1e-14 degenerate-X guard mirrors the oracle's normalize_penalties.
interface ResolvedPenalties {
    wLength: number;
    wDiff: number;
    wField: number;
    X: Vec3;
}

function resolve(config: PenaltyConfig): ResolvedPenalties {
    const wLength = config.totalLength ?? 0;
    const wDiff = config.lengthDiff ?? 0;
    let wField = config.field?.weight ?? 0;
    let X: Vec3 = [0, 0, 0];
    if (wField !== 0 && config.field) {
        const [x, y, z] = config.field.X;
        const n = Math.sqrt(x * x + y * y + z * z);
        // ‖X‖ < 1e-14 ⇒ field inactive (plan §3), same constant as the
        // degenerate-tangent guard below.
        if (n < 1e-14) {
            wField = 0;
        } else {
            X = [x / n, y / n, z / n];
        }
    }
    return { wLength, wDiff, wField, X };
}

/**
 * True iff the config carries at least one live penalty term. The descent
 * integration gates EVERY penalty branch on this so that absent/zero configs
 * leave all code paths bit-identical to the penalty-free build (plan §2.4;
 * same guard pattern as the frozen-operator forwarding in optimizer.ts).
 * @see docs/superpowers/plans/2026-07-03-sobolev-penalties.md §2.4
 */
export function penaltiesActive(config?: PenaltyConfig): config is PenaltyConfig {
    if (!config) return false;
    const r = resolve(config);
    return r.wLength !== 0 || r.wDiff !== 0 || r.wField !== 0;
}

// Raw lengths + safe-unit tangents per edge (T = 0 when ℓ < 1e-14 — same
// guard constant as constraintSet.ts rows), mirroring the oracle's
// _edge_lengths_and_tangents so downstream loops match term-for-term.
function edgeGeometry(vertices: Vec3[], edges: Edge[]): { ell: number[]; T: Vec3[] } {
    const ell: number[] = new Array(edges.length);
    const T: Vec3[] = new Array(edges.length);
    for (let r = 0; r < edges.length; r++) {
        const [i1, i2] = edges[r];
        const p1 = vertices[i1];
        const p2 = vertices[i2];
        const ex = p2[0] - p1[0];
        const ey = p2[1] - p1[1];
        const ez = p2[2] - p1[2];
        const len = Math.sqrt(ex * ex + ey * ey + ez * ez);
        ell[r] = len;
        T[r] = len < 1e-14 ? [0, 0, 0] : [ex / len, ey / len, ez / len];
    }
    return { ell, T };
}

// V_int for the length-difference penalty: (v, I_v, J_v) for every vertex of
// degree EXACTLY 2 — paper line 764 "interior" vertices; degree-1 endpoints
// and degree≥3 junctions are EXCLUDED. Incident edges by ascending edge index
// (plan §3: the term is swap-symmetric, the order is fixed for determinism).
function interiorVertices(n: number, edges: Edge[]): Array<[number, number, number]> {
    const incident: number[][] = Array.from({ length: n }, () => []);
    for (let r = 0; r < edges.length; r++) {
        incident[edges[r][0]].push(r);
        incident[edges[r][1]].push(r);
    }
    const out: Array<[number, number, number]> = [];
    for (let v = 0; v < n; v++) {
        if (incident[v].length === 2) out.push([v, incident[v][0], incident[v][1]]);
    }
    return out;
}

/**
 * E_pen(γ) = w_len·Σ_I ℓ_I + w_diff·Σ_{V_int}(ℓ_{I_v}−ℓ_{J_v})²
 * + w_field·Σ_I ℓ_I·|T_I×X|² (plan §2.1–§2.3; paper lines 763/764/766).
 * The field term uses an explicit cross product so a degenerate T = 0
 * contributes 0 (the 1−(T·X)² identity would wrongly give 1 there).
 * Per-penalty accumulators are multiplied by their weight at the end,
 * mirroring the oracle's op order.
 * @see docs/superpowers/plans/2026-07-03-sobolev-penalties.md §2
 * @see oracle/tpe_constraints_oracle.py (penalty_energy)
 */
export function penaltyEnergy(vertices: Vec3[], edges: Edge[], config: PenaltyConfig): number {
    const { wLength, wDiff, wField, X } = resolve(config);
    const { ell, T } = edgeGeometry(vertices, edges);
    let total = 0;
    if (wLength !== 0) {
        let acc = 0;
        for (let r = 0; r < edges.length; r++) acc += ell[r];
        total += wLength * acc;
    }
    if (wDiff !== 0) {
        let acc = 0;
        for (const [, rI, rJ] of interiorVertices(vertices.length, edges)) {
            const d = ell[rI] - ell[rJ];
            acc += d * d;
        }
        total += wDiff * acc;
    }
    if (wField !== 0) {
        let acc = 0;
        for (let r = 0; r < edges.length; r++) {
            const t = T[r];
            const cx = t[1] * X[2] - t[2] * X[1];
            const cy = t[2] * X[0] - t[0] * X[2];
            const cz = t[0] * X[1] - t[1] * X[0];
            acc += ell[r] * (cx * cx + cy * cy + cz * cz);
        }
        total += wField * acc;
    }
    return total;
}

/**
 * Analytic d(E_pen)/dγ, ascent orientation — ADDS to dE before the saddle
 * solve (same orientation as `gradientAnalytical`). Per-edge stencils for
 * edge I=(a,b), e = γ_b−γ_a, T = e/ℓ (plan §2):
 *   total length:  −w·T at a, +w·T at b                          (§2.1)
 *   length diff:   ±2w·(ℓ_{I_v}−ℓ_{J_v})·T on each edge's ends   (§2.2)
 *   field:         g_I = (1+(T·X)²)·T − 2(T·X)·X, ∓w·g_I         (§2.3;
 *                  bounded as ℓ→0, aligned edge ⇒ g_I = 0)
 * FD-verified at rel ≤ 1e-6 on both sides (oracle property check + the
 * penalties test suite); vs the oracle's recorded gradient at 1e-12.
 * @see docs/superpowers/plans/2026-07-03-sobolev-penalties.md §2, §5
 * @see oracle/tpe_constraints_oracle.py (penalty_gradient)
 */
export function penaltyGradient(vertices: Vec3[], edges: Edge[], config: PenaltyConfig): Vec3[] {
    const { wLength, wDiff, wField, X } = resolve(config);
    const { ell, T } = edgeGeometry(vertices, edges);
    const g: Vec3[] = vertices.map(() => [0, 0, 0]);
    if (wLength !== 0) {
        for (let r = 0; r < edges.length; r++) {
            const [a, b] = edges[r];
            const t = T[r];
            g[a][0] -= wLength * t[0];
            g[a][1] -= wLength * t[1];
            g[a][2] -= wLength * t[2];
            g[b][0] += wLength * t[0];
            g[b][1] += wLength * t[1];
            g[b][2] += wLength * t[2];
        }
    }
    if (wDiff !== 0) {
        for (const [, rI, rJ] of interiorVertices(vertices.length, edges)) {
            const s = 2 * wDiff * (ell[rI] - ell[rJ]);
            const tI = T[rI];
            const [aI, bI] = edges[rI];
            g[aI][0] -= s * tI[0];
            g[aI][1] -= s * tI[1];
            g[aI][2] -= s * tI[2];
            g[bI][0] += s * tI[0];
            g[bI][1] += s * tI[1];
            g[bI][2] += s * tI[2];
            const tJ = T[rJ];
            const [aJ, bJ] = edges[rJ];
            g[aJ][0] += s * tJ[0];
            g[aJ][1] += s * tJ[1];
            g[aJ][2] += s * tJ[2];
            g[bJ][0] -= s * tJ[0];
            g[bJ][1] -= s * tJ[1];
            g[bJ][2] -= s * tJ[2];
        }
    }
    if (wField !== 0) {
        for (let r = 0; r < edges.length; r++) {
            // Degenerate edge: zero contribution (plan §2 conventions) — T is
            // already 0 here, but skip explicitly to mirror the oracle.
            if (ell[r] < 1e-14) continue;
            const t = T[r];
            const u = t[0] * X[0] + t[1] * X[1] + t[2] * X[2];
            const c = 1 + u * u;
            const gx = c * t[0] - 2 * u * X[0];
            const gy = c * t[1] - 2 * u * X[1];
            const gz = c * t[2] - 2 * u * X[2];
            const [a, b] = edges[r];
            g[a][0] -= wField * gx;
            g[a][1] -= wField * gy;
            g[a][2] -= wField * gz;
            g[b][0] += wField * gx;
            g[b][1] += wField * gy;
            g[b][2] += wField * gz;
        }
    }
    return g;
}
