/**
 * Which machinery a descent step uses, decided in ONE place so the store, the worker, the frame
 * loop and `dispatchDescentStep` cannot disagree about it.
 *
 * Two independent facts are decided here:
 *  1. whether the O(n²) `disjointPairs` list exists at all for a graph of this size
 *     ({@link pairListFits}) — the store and the worker both key their allocation on it;
 *  2. whether a Sobolev step runs the DENSE route (assemble Ā, factor it, exact all-pairs energy)
 *     or the HIERARCHICAL route (matrix-free BCT solve + Barnes–Hut energy and differential)
 *     ({@link resolveSobolevRoute}).
 *
 * This module is worker-bundle pure: no store, no React.
 * @see bench/results/2026-11-23-h-step-scale.jsonl, bench/results/2026-11-23-bh-step-scale.jsonl
 */
import type { Edge } from './testConfigs';

/** The user's preference. `'auto'` picks by size and topology. */
export type SobolevSolver = 'auto' | 'dense' | 'hierarchical';

/** What a step actually runs. */
export type SobolevRoute = 'dense' | 'hierarchical';

/**
 * Largest vertex count for which `disjointPairs` is built.
 * Why 4 000: the list is n rows of about n entries; measured, one copy brings the process to
 * 492 MB rss at n = 4 000 (`2026-11-23-h-step-scale.jsonl`, the `step` rows), and the app holds
 * two copies (store and worker). Above this, every consumer must work without the list.
 */
export const PAIR_LIST_MAX_N = 4000;

/**
 * Smallest closed ring that `'auto'` sends to the hierarchical route.
 * Why 240: measured on the trefoil, one step costs 0.27 s hierarchical against 0.67 s dense at
 * n = 240, and 0.49 s against 2.06 s at n = 480 (`bench/scale/hStepScale.ts`, modes `bh` and
 * `dense`, 2026-11-23, loaded machine). Below 240 the two were not compared, and the dense route
 * is exact there, so it stays the default.
 */
export const HIERARCHICAL_AUTO_MIN_N = 240;

/**
 * GMRES stopping tolerance of the hierarchical route in the app.
 * Why 1e-2 and not the solver's own default 1e-12: a descent step needs a good direction, not a
 * small residual (spec §3.5). Measured, 1e-2 stops after 7 iterations at every n from 960 to
 * 30 000 where 1e-12 takes 35–47, and the step is accepted at the same step length with the same
 * energy to four digits (14.91663 against 14.91675 at n = 960; 14.89311 against 14.89318 at
 * n = 4 000 — `2026-11-23-h-step-scale.jsonl`). The line search remains the safeguard: a poor
 * direction is backtracked or rejected there.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §3.5
 */
export const APP_HIERARCHICAL_TOL = 1e-2;

/** Does the O(n²) pair list exist for a graph with this many vertices? */
export function pairListFits(nVertices: number): boolean {
    return nVertices <= PAIR_LIST_MAX_N;
}

/**
 * Is the graph one closed loop whose edges are exactly `(i, i+1)` and the wrap `(n−1, 0)`?
 * Why this exact shape: it is the only topology the hierarchical solver's preconditioner supports
 * (its curve Laplacian must be cyclic-tridiagonal), and `makeHierarchicalSaddle` throws on
 * anything else. This predicate mirrors that check so routing can ask first.
 * @see src/core/sobolev/hierarchical/hierarchicalSaddle.ts (assertClosedNaturalRing)
 * @issue utof/repulsive-test2#93
 */
export function isClosedNaturalRing(nVertices: number, edges: Edge[]): boolean {
    const n = nVertices;
    if (n < 3 || edges.length !== n) return false;
    const seen = new Uint8Array(n);
    for (const [p, q] of edges) {
        const slot = q === p + 1 && p >= 0 && q < n ? p : p === n - 1 && q === 0 ? n - 1 : -1;
        if (slot < 0 || seen[slot]) return false;
        seen[slot] = 1;
    }
    return true;
}

/**
 * The route a Sobolev step takes, or `null` when no route can run it.
 *
 *  - The hierarchical route needs a closed natural ring and the analytical differential
 *    (a finite-difference differential would cost n energy evaluations).
 *  - The dense route needs the pair list.
 *  - `'auto'` prefers hierarchical from {@link HIERARCHICAL_AUTO_MIN_N} up; an explicit
 *    preference is honoured when that route can run and falls back to the other when it cannot.
 *  - `null`: too large for the pair list AND not a closed ring. Nothing can step it yet (#93).
 */
export function resolveSobolevRoute(
    preference: SobolevSolver,
    nVertices: number,
    edges: Edge[],
    analytical: boolean,
): SobolevRoute | null {
    const canDense = pairListFits(nVertices);
    const canHier = analytical && isClosedNaturalRing(nVertices, edges);
    if (!canDense) return canHier ? 'hierarchical' : null;
    if (!canHier) return 'dense';
    if (preference === 'dense') return 'dense';
    if (preference === 'hierarchical') return 'hierarchical';
    return nVertices >= HIERARCHICAL_AUTO_MIN_N ? 'hierarchical' : 'dense';
}
