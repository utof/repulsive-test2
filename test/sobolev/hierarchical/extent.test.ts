/**
 * `curveExtent` — the length scale `ext` of the preconditioner's default mass weight `30/ext²`.
 * Every expectation below is a HAND-COMPUTED constant, not a second bbox loop: the independent
 * cross-check against production's mass weight lives in `precond.test.ts` (`extentOf`), so the
 * two never share a defect (memory `common-mode-blindness`).
 *
 * Why the boxes are asymmetric and offset: the max-vs-min-span, `hi`-vs-`hi − lo` and
 * single-axis mistakes each pass on a unit cube at the origin.
 *
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §11.2 (mass-weight ruling, SUPERSEDED by amendment A20)
 * @issue utof/repulsive-test2#72
 */
import { describe, expect, test } from 'bun:test';
import { curveExtent } from '../../../src/core/sobolev/hierarchical/extent';
import type { Vec3 } from '../../../src/core/testConfigs';

describe('curveExtent: max bounding-box side (issue #72)', () => {
    test('is the MAX span over the three axes, whichever axis carries it', () => {
        // Spans (x, y, z) = (1, 5, 2) / (7, 1, 2) / (1, 2, 9). A min-span or first-axis-only
        // implementation returns 1 / 7 / 1 here.
        expect(
            curveExtent([
                [0, 0, 0],
                [1, 5, 2],
            ]),
        ).toBe(5);
        expect(
            curveExtent([
                [0, 0, 0],
                [7, 1, 2],
            ]),
        ).toBe(7);
        expect(
            curveExtent([
                [0, 0, 0],
                [1, 2, 9],
            ]),
        ).toBe(9);
    });

    test('is hi − lo, so it is translation-invariant and handles negative coordinates', () => {
        // A box [10, 13] x [-4, -1] x [100, 102] has spans (3, 3, 2): `hi` alone would give 102.
        const box: Vec3[] = [
            [10, -4, 100],
            [13, -1, 102],
            [11.5, -2.5, 101],
        ];
        expect(curveExtent(box)).toBe(3);
        const shifted = box.map((v): Vec3 => [v[0] - 1000, v[1] + 1000, v[2] - 1000]);
        expect(curveExtent(shifted)).toBe(3);
        // An interior point never changes the box.
        expect(curveExtent([...box, [11, -3, 100.5]])).toBe(3);
    });

    test('scales linearly: ext(λV) = λ·ext(V)', () => {
        const pts: Vec3[] = [
            [0.25, -1, 2],
            [3, 0.5, -2],
            [1, 1, 1],
        ];
        // Spans (2.75, 1.5, 4): ext = 4.
        expect(curveExtent(pts)).toBe(4);
        for (const lam of [0.125, 8, 1024]) {
            expect(curveExtent(pts.map((v): Vec3 => [lam * v[0], lam * v[1], lam * v[2]]))).toBe(
                4 * lam,
            );
        }
    });

    test('is 0 for an empty or coincident point set, and does NOT launder NaN/Infinity', () => {
        // The caller (`makePreconditioner`) divides by ext², so it must see these values as they
        // are: a clamp here would turn a degenerate curve into a plausible mass weight.
        expect(curveExtent([])).toBe(0);
        expect(curveExtent([[1, 2, 3]])).toBe(0);
        expect(
            curveExtent([
                [1, 2, 3],
                [1, 2, 3],
            ]),
        ).toBe(0);
        expect(
            curveExtent([
                [0, 0, 0],
                [1, Number.NaN, 0],
            ]),
        ).toBeNaN();
        expect(
            curveExtent([
                [0, 0, 0],
                [0, 0, Number.POSITIVE_INFINITY],
            ]),
        ).toBe(Number.POSITIVE_INFINITY);
    });
});
