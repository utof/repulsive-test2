/**
 * Stacked-blocks ConstraintSet abstraction for the constrained Sobolev-gradient
 * saddle solve (Repulsive Curves, Yu/Schumacher/Crane 2021): a `ConstraintSet`
 * is an ordered array of `ConstraintBlock`s, each contributing some rows of
 * Φ (target) and C = dΦ (Jacobian) to the stacked saddle system
 * `[[Ā, Cᵀ], [C, 0]]·[x; μ] = [dE; −Φ]`.
 *
 * M1 ships two builders: {@link barycenterBlock} (wraps the existing
 * `barycenterPhiAndC`/`barycenterScale`, unchanged) and {@link totalLengthBlock}
 * (new — total-length equality constraint). `edgeLengths` and `point` are
 * catalogued in the `kind` union now so the type is future-proof, but their
 * builders arrive in M2.
 * @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §3.1
 */
import type { Edge, Vec3 } from '../testConfigs';
import { barycenterPhiAndC } from './constraints';
import { blockIndex } from './layout';
import { barycenterScale } from './lineSearch';

/**
 * One constraint block's evaluation: `phi` (k values) and `C` (k rows ×
 * 3|V| columns, coordinate-block layout via {@link blockIndex}). Multiple
 * blocks' evaluations stack (row-concatenate) into the full saddle-system
 * constraint via {@link evaluateConstraintSet}.
 * @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §3.1
 */
export interface ConstraintEval {
    phi: number[];
    C: number[][];
}

/**
 * One constraint block in a {@link ConstraintSet}: knows its own Φ rows, its
 * Jacobian rows, and its projection-tolerance scale (§3.3). `kind` names all
 * four catalog constraints (barycenter, total length, per-edge length, point)
 * even though only `barycenter`/`totalLength` have builders in M1 — `kind` is
 * also what {@link assertValidConstraintSet} inspects for the §3.4 rank rule.
 * @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §3.1, §3.4
 */
export interface ConstraintBlock {
    kind: 'barycenter' | 'totalLength' | 'edgeLengths' | 'point';
    evaluate(vertices: Vec3[], edges: Edge[]): ConstraintEval;
    scale(vertices: Vec3[], edges: Edge[]): number;
}

/**
 * An ordered stack of {@link ConstraintBlock}s; row order in the assembled
 * saddle system follows array order (barycenter block first WHEN present, per
 * §3.2). The empty set (`[]`) is valid — the saddle system degenerates to
 * k = 0 constraint rows (§9a).
 * @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §3.1, §9a
 */
export type ConstraintSet = ConstraintBlock[];

/**
 * Raw total curve length L = Σ_{I∈E} ℓ_I, with ℓ_I = ‖γ_{i2} − γ_{i1}‖ the RAW
 * geometric edge length — NO +ε. Constraints are geometric, not part of the
 * regularized energy; same convention as `barycenterPhiAndC`'s ℓ_I (do NOT
 * "unify" this with the ℓ^ε used in innerProduct.ts).
 * Shared by {@link totalLengthBlock}'s Φ/scale and (later) the store/Stats
 * "current length" readout.
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("Use raw geometric lengths ... not ℓ^ε")
 * @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §2
 */
export function totalLength(vertices: Vec3[], edges: Edge[]): number {
    let L = 0;
    for (const [i1, i2] of edges) {
        const p1 = vertices[i1];
        const p2 = vertices[i2];
        const ex = p2[0] - p1[0];
        const ey = p2[1] - p1[1];
        const ez = p2[2] - p1[2];
        L += Math.sqrt(ex * ex + ey * ey + ez * ez);
    }
    return L;
}

/**
 * Barycenter constraint block: Φ_bar(γ) = Σ_I ℓ_I·(m_I − x₀) ∈ R³ (3 rows).
 * `evaluate`/`scale` are a bit-identical passthrough of the existing
 * `barycenterPhiAndC`/`barycenterScale` — the math is NOT reimplemented here,
 * only wrapped, so the barycenter-only ConstraintSet path stays numerically
 * identical to the pre-ConstraintSet code (back-compat requirement).
 * @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §3.1, §3.2
 * @see src/core/sobolev/constraints.ts (barycenterPhiAndC)
 * @see src/core/sobolev/lineSearch.ts (barycenterScale)
 */
export function barycenterBlock(x0: Vec3): ConstraintBlock {
    return {
        kind: 'barycenter',
        evaluate(vertices, edges) {
            const { phi, C } = barycenterPhiAndC(vertices, edges, x0);
            return { phi: [...phi], C };
        },
        scale(vertices, edges) {
            return barycenterScale(vertices, edges, x0);
        },
    };
}

/**
 * Total-length constraint block: Φ_len(γ) = L⁰ − Σ_I ℓ_I ∈ R (1 row), paper
 * sign convention (target minus current). Jacobian row: for every edge
 * I=(i1,i2) with unit tangent T_I = e_I/ℓ_I, accumulate `+T_I` into vertex
 * i1's columns and `−T_I` into vertex i2's columns — signs follow from
 * dΦ = −Σ dℓ_I, dℓ_I = T_I·(dγ_{i2} − dγ_{i1}). Junctions/endpoints need no
 * special case: every incident edge adds its term to the same vertex columns,
 * exactly as in `barycenterPhiAndC`.
 *
 * Degenerate guard: T_I = [0,0,0] when ℓ_I < 1e-14 — same guard, same
 * constant, same rationale as `barycenterPhiAndC` (constraints.ts).
 *
 * Projection-tolerance scale: max(1, L) with L = Σℓ_I raw — OUR tunable
 * choice, NOT paper-sourced; same flagging convention as `barycenterScale` in
 * lineSearch.ts. Do not treat it as paper ground truth.
 * @see local_files/2026-07-02-sobolev-formula-audit.md (Item 9 — "Unstated inventions")
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B (degenerate-edge guard)
 * @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §2, §3.1, §3.3
 */
export function totalLengthBlock(L0: number): ConstraintBlock {
    return {
        kind: 'totalLength',
        evaluate(vertices, edges) {
            const n = vertices.length;
            const row = new Array<number>(3 * n).fill(0);
            let L = 0;
            for (const [i1, i2] of edges) {
                const p1 = vertices[i1];
                const p2 = vertices[i2];
                const ex = p2[0] - p1[0];
                const ey = p2[1] - p1[1];
                const ez = p2[2] - p1[2];
                // RAW geometric length, no +ε — see the module-level totalLength anchor.
                const ell = Math.sqrt(ex * ex + ey * ey + ez * ez);
                L += ell;
                // Degenerate guard: T_I = 0 when ‖e_I‖ < 1e-14 (same constant as
                // barycenterPhiAndC's guard, constraints.ts:92-98).
                // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B
                let T: Vec3;
                if (ell < 1e-14) {
                    T = [0, 0, 0];
                } else {
                    const inv = 1 / ell;
                    T = [ex * inv, ey * inv, ez * inv];
                }
                for (let c = 0; c < 3; c++) {
                    row[blockIndex(c, i1, n)] += T[c];
                    row[blockIndex(c, i2, n)] += -T[c];
                }
            }
            // Φ = L0 − Σℓ_I — paper sign, target minus current.
            // @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §2
            return { phi: [L0 - L], C: [row] };
        },
        scale(vertices, edges) {
            // OUR tunable choice, NOT paper-sourced — same flagging convention as
            // lineSearch.ts's barycenterScale.
            // @see local_files/2026-07-02-sobolev-formula-audit.md (Item 9 — "Unstated inventions")
            // @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §3.3
            return Math.max(1, totalLength(vertices, edges));
        },
    };
}

/**
 * Stacks every block's Φ rows and C rows, in `set` array order, into one
 * `ConstraintEval` for the saddle solve. Works for the empty set (`phi: []`,
 * `C: []`) — the k = 0 case the saddle solver already handles (§9a).
 * @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §3.1, §9a
 */
export function evaluateConstraintSet(
    set: ConstraintSet,
    vertices: Vec3[],
    edges: Edge[],
): ConstraintEval {
    const phi: number[] = [];
    const C: number[][] = [];
    for (const block of set) {
        const res = block.evaluate(vertices, edges);
        phi.push(...res.phi);
        C.push(...res.C);
    }
    return { phi, C };
}

/**
 * Construction-time rank-rule guard (§3.4): `totalLength` and `edgeLengths`
 * blocks are mutually exclusive in one set, because the total-length row is
 * EXACTLY the sum of the edge-length rows — composing both makes C exactly
 * rank-deficient, which produces a singular saddle system rather than a
 * meaningful solve-time failure. Throwing here, at construction, surfaces the
 * mistake immediately instead of downstream in the solver.
 * @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §3.4
 */
export function assertValidConstraintSet(set: ConstraintSet): void {
    const hasTotalLength = set.some((b) => b.kind === 'totalLength');
    const hasEdgeLengths = set.some((b) => b.kind === 'edgeLengths');
    if (hasTotalLength && hasEdgeLengths) {
        throw new Error(
            'ConstraintSet: totalLength and edgeLengths are mutually exclusive — ' +
                'the total-length row is exactly the sum of the edge-length rows, ' +
                'causing exact rank deficiency in the saddle system ' +
                '(see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §3.4).',
        );
    }
}
