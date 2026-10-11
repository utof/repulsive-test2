/**
 * `ext` — the length scale of the preconditioner's default mass weight: the MAXIMUM bounding-box
 * side of the curve, `max over x,y,z of (hi − lo)`.
 *
 * Why this and not total length or mean edge length: amendment A20 measured all three over
 * 8 curve shapes x 4-6 sizes x 19-27 mass values (1089 cells). Total length is REFUTED (the
 * per-shape optimum spans 96x), mean edge length is KILLED (past the conditioning optimum
 * `kappa_excl` grows like `n^{+4}`, so a fixed constant only delays the cliff), and extent is a
 * property of the CURVE rather than of its discretisation.
 *
 * A SECOND, INDEPENDENT COPY of this loop is `extentOf` in `test/sobolev/hierarchical/precond.test.ts`.
 * That is deliberate and must stay: a test asserting `massWeight === 30/ext²` that took `ext` from
 * this module would be checking production against itself, and a defect shared by both paths
 * (say, `hi` instead of `hi − lo`) would pass. Two ten-line bbox loops are the cheap cross-check
 * (memory `common-mode-blindness`; issue utof/repulsive-test2#72, comment 2).
 *
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §11.2 (mass-weight ruling, SUPERSEDED by amendment A20 -- see §10.2's retracted-claims row)
 * @see .superpowers/sdd/2026-10-09-phase2h-bct-gmres-design/spec-amendments.md A20
 * @issue utof/repulsive-test2#72
 */
import type { Vec3 } from '../../testConfigs';

/**
 * Maximum bounding-box side of `vertices`; `0` for an empty or single-point set.
 *
 * Does NOT validate: NaN/Infinity coordinates propagate into the result (`Math.max` with a NaN
 * operand is NaN), and a coincident point set gives exactly 0. The caller that divides by `ext²`
 * must reject both, as `makePreconditioner` does. Why: a silent clamp here would turn a degenerate
 * curve into a plausible-looking mass weight.
 *
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §11.2 (mass-weight ruling, A20)
 * @issue utof/repulsive-test2#72
 */
export function curveExtent(vertices: Vec3[]): number {
    let ext = 0;
    for (let d = 0; d < 3; d++) {
        let lo = Number.POSITIVE_INFINITY;
        let hi = Number.NEGATIVE_INFINITY;
        for (const v of vertices) {
            lo = Math.min(lo, v[d]);
            hi = Math.max(hi, v[d]);
        }
        ext = Math.max(ext, hi - lo);
    }
    return ext;
}
