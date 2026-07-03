/**
 * The constrained fractional Sobolev gradient g̃ (Repulsive Curves,
 * Yu/Schumacher/Crane 2021): composition of the verified Stage-1 pieces —
 * inner-product assembly (`./innerProduct`), block-diagonal expansion
 * (`./layout`), stacked constraint Jacobians (`./constraintSet`), and the
 * saddle solve (`./linsolve`) — into the one call the descent loop needs.
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("Gradient saddle system")
 * @see oracle/tpe_stage1_oracle.py (solve_constrained_gradient)
 * @see oracle/tpe_constraints_oracle.py (solve_constrained_gradient_set)
 */
import type { Edge, Vec3 } from '../testConfigs';
import { barycenterBlock, type ConstraintSet, evaluateConstraintSet } from './constraintSet';
import { assembleA } from './innerProduct';
import { expandBlockDiag, flatten, unflatten } from './layout';
import { solveSaddle } from './linsolve';

/**
 * Solves the constrained Sobolev-gradient saddle system
 * `[[Ā, Cᵀ], [C, 0]]·[g̃; λ] = [dE; 0]` for an arbitrary stacked
 * {@link ConstraintSet} (k = Σ block rows; the EMPTY set is valid — k = 0
 * degenerates to Ā·g̃ = dE, spec §9a) and returns g̃ per-vertex, the
 * multipliers λ (one per stacked constraint row, in set order), and the
 * solve's self-certifying relative residual (gated at ≤1e-10 by spec §E
 * prop 8).
 *
 * `dE` is a PARAMETER (per-vertex differential), never computed here: the
 * solve must be pluggable w.r.t. how dE was produced — finite-difference
 * first, analytical later, with identical solve behavior either way.
 * @see local_files/sobolev-gradient-handoff.md §1 ("The dE fed into the solve must be pluggable")
 *
 * Constraint TARGETS (x₀, L⁰, …) live inside the blocks, frozen at
 * construction — pass the same set every solve of a run; never rebuild the
 * blocks from the current iterate (that makes the constraints vacuous).
 * Programmatic set composition should be validated once with
 * `assertValidConstraintSet` at construction time (spec §3.4).
 * @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §3.1, §3.5
 * @see oracle/tpe_constraints_oracle.py (solve_constrained_gradient_set)
 */
export function solveConstrainedGradientSet(
    vertices: Vec3[],
    edges: Edge[],
    disjointPairs: number[][],
    alpha: number,
    beta: number,
    epsilon: number,
    dE: Vec3[],
    set: ConstraintSet,
): { gTilde: Vec3[]; lambda: number[]; residual: number } {
    const A = assembleA(vertices, edges, disjointPairs, alpha, beta, epsilon);
    const A3 = expandBlockDiag(A);
    // Only the Jacobian C enters the gradient solve. Φ itself does NOT: the
    // saddle RHS bottom block is 0 (solveSaddle's default), unlike the
    // constraint-projection solve which passes −Φ there.
    // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("Gradient saddle system" — RHS [dE; 0])
    const { C } = evaluateConstraintSet(set, vertices, edges);
    const { x, lambda, residual } = solveSaddle(A3, C, flatten(dE));
    return { gTilde: unflatten(x), lambda, residual };
}

/**
 * Back-compat x₀ signature: delegates to {@link solveConstrainedGradientSet}
 * with the barycenter-only set `[barycenterBlock(x0)]` — numerically
 * bit-identical to the pre-ConstraintSet implementation (the block is a pure
 * passthrough of `barycenterPhiAndC`; regression-proven by the unmodified
 * stage-1 golden tests, spec §3.2).
 *
 * `x0` is the frozen barycenter target, set ONCE at initialization (see
 * `barycenterTarget` in `./constraints`) and passed unchanged on every solve —
 * do not recompute it from the current vertices.
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("Gradient saddle system")
 * @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §3.2
 * @see oracle/tpe_stage1_oracle.py (solve_constrained_gradient)
 */
export function solveConstrainedGradient(
    vertices: Vec3[],
    edges: Edge[],
    disjointPairs: number[][],
    alpha: number,
    beta: number,
    epsilon: number,
    dE: Vec3[],
    x0: Vec3,
): { gTilde: Vec3[]; lambda: number[]; residual: number } {
    return solveConstrainedGradientSet(vertices, edges, disjointPairs, alpha, beta, epsilon, dE, [
        barycenterBlock(x0),
    ]);
}
