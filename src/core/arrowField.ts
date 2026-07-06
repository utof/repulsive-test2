import type { DescentMode, Mode } from './dispatch';
import { DEFAULTS } from './optimizer';
import { solveConstrainedGradient } from './sobolev/gradient';
import { gradientAnalytical, gradientFiniteDiff } from './tangentPointEnergy';
import type { Edge, Vec3 } from './testConfigs';

/**
 * Pure descent-field computation shared by the GradientArrows diagnostic's
 * worker path and its synchronous fallback (§D13-a). Returns the RAW field
 * (dE in raw mode, g̃ in sobolev mode) — NOT negated; the caller negates when
 * orienting cones. Bit-identical to the field previously inlined in
 * GradientArrows.tsx:77-117: raw uses `gradientAnalytical`/`gradientFiniteDiff`
 * under DEFAULTS.alpha/beta/epsilon (and DEFAULTS.h for finiteDiff); sobolev
 * feeds that dE into `solveConstrainedGradient` and returns `.gTilde`.
 *
 * Returns `null` iff the sobolev saddle solve throws (singular system): the
 * caller HIDES the arrows rather than showing −dE mislabelled as the sobolev
 * field — the exact contract of the old try/catch at GradientArrows.tsx:109-115.
 *
 * Imports only from `src/core/**` so the worker bundle never pulls React/zustand
 * (worker-bundle purity, plan §D2).
 * @see docs/superpowers/plans/2026-07-04-worker-solver.md §D13
 */
export function computeArrowField(
    vertices: Vec3[],
    edges: Edge[],
    disjointPairs: number[][],
    mode: Mode,
    descentMode: DescentMode,
    x0: Vec3,
): Vec3[] | null {
    const dE =
        mode === 'analytical'
            ? gradientAnalytical(
                  vertices,
                  edges,
                  disjointPairs,
                  DEFAULTS.alpha,
                  DEFAULTS.beta,
                  DEFAULTS.epsilon,
              )
            : gradientFiniteDiff(
                  vertices,
                  edges,
                  disjointPairs,
                  DEFAULTS.alpha,
                  DEFAULTS.beta,
                  DEFAULTS.epsilon,
                  DEFAULTS.h,
              );
    if (descentMode === 'sobolev') {
        try {
            return solveConstrainedGradient(
                vertices,
                edges,
                disjointPairs,
                DEFAULTS.alpha,
                DEFAULTS.beta,
                DEFAULTS.epsilon,
                dE,
                x0,
            ).gTilde;
        } catch {
            // Singular saddle system (sobolevStep's 'singular_system' contract):
            // no defined Sobolev direction — return null so the caller hides the
            // arrows rather than show −dE mislabelled as the sobolev field. This
            // preserves the exact GradientArrows.tsx:109-115 behavior.
            // @see docs/superpowers/plans/2026-07-04-worker-solver.md §D13
            return null;
        }
    }
    return dE;
}
