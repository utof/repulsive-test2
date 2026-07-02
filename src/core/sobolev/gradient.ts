/**
 * The constrained fractional Sobolev gradient g̃ (Repulsive Curves,
 * Yu/Schumacher/Crane 2021): composition of the verified Stage-1 pieces —
 * inner-product assembly (`./innerProduct`), block-diagonal expansion
 * (`./layout`), barycenter constraint Jacobian (`./constraints`), and the
 * saddle solve (`./linsolve`) — into the one call the descent loop needs.
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("Gradient saddle system")
 * @see oracle/tpe_stage1_oracle.py (solve_constrained_gradient)
 */
import type { Edge, Vec3 } from '../testConfigs';
import { barycenterPhiAndC } from './constraints';
import { assembleA } from './innerProduct';
import { expandBlockDiag, flatten, unflatten } from './layout';
import { solveSaddle } from './linsolve';

/**
 * Solves the constrained Sobolev-gradient saddle system
 * `[[Ā, Cᵀ], [C, 0]]·[g̃; λ] = [dE; 0]` for the current curve state and
 * returns g̃ per-vertex, the barycenter multipliers λ, and the solve's
 * self-certifying relative residual (gated at ≤1e-10 by spec §E prop 8).
 *
 * `dE` is a PARAMETER (per-vertex differential), never computed here: the
 * solve must be pluggable w.r.t. how dE was produced — finite-difference
 * first, analytical later, with identical solve behavior either way.
 * @see local_files/sobolev-gradient-handoff.md §1 ("The dE fed into the solve must be pluggable")
 *
 * `x0` is the frozen barycenter target, set ONCE at initialization (see
 * `barycenterTarget` in `./constraints`) and passed unchanged on every solve —
 * do not recompute it from the current vertices.
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("Gradient saddle system")
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
    const A = assembleA(vertices, edges, disjointPairs, alpha, beta, epsilon);
    const A3 = expandBlockDiag(A);
    // Only the Jacobian C enters the gradient solve. Φ itself does NOT: the
    // saddle RHS bottom block is 0 (solveSaddle's default), unlike the stage-2
    // constraint-projection solve which passes −Φ there.
    // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("Gradient saddle system" — RHS [dE; 0])
    const { C } = barycenterPhiAndC(vertices, edges, x0);
    const { x, lambda, residual } = solveSaddle(A3, C, flatten(dE));
    return { gTilde: unflatten(x), lambda, residual };
}
