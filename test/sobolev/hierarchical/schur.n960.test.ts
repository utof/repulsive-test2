/**
 * The n = 960 half of `schur.test.ts` (spec §8.2 / §11.2 row 9b): the six n = 960 cells against the
 * refined dense truth, and the tol sweep behind `DEFAULT_SCHUR_TOL`. FULL TIER ONLY: every test
 * here is `fullTest` / `fullDescribe` (test/tier.ts), so a default run reports all eight as skipped.
 * Test names and describe names are unchanged from when they lived in `schur.test.ts`.
 *
 * Why a separate file: the runner runs one process per file, four at a time, and the full tier's
 * wall time was the schur file alone (431 s against 139 s for the next file, measured 2026-11-24 at
 * load ~15). Split, the two halves run in parallel. The harness is `schurHarness.ts`; read
 * `schur.test.ts`'s module TSDoc for what the oracle can and cannot certify.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §8.2, §8.3, §11.2 row 9b
 * @issue utof/repulsive-test2#62
 */
import { describe, expect, setDefaultTimeout, test } from 'bun:test';
import { DEFAULT_THETA } from '../../../src/core/sobolev/hierarchical/bct';
import { DEFAULT_SCHUR_TOL, prepareSchur } from '../../../src/core/sobolev/hierarchical/schur';
import { fullDescribe, fullTest } from '../../tier';
import { BAR, checkCells, denseRefs, log, relErr, SET_NAMES, setup } from './schurHarness';

setDefaultTimeout(900_000);

describe('Schur vs dense: same operator (lists at DEFAULT_THETA), spec §11.2 row 9b', () => {
    for (const name of SET_NAMES) {
        // n = 960: the literal agreement is LOGGED (it plateaus at the oracle's own error); the 1e-10
        // bar is asserted against the refined truth.
        // FULL TIER ONLY (test/tier.ts): one dense LDLᵀ + refinement of the 2880-dim saddle per set
        // (~8.5 s each). Why: the n = 240 cells in schur.test.ts assert the literal AND the truth bar
        // on every set in the default tier; what waits for a full run is a defect that only shows at
        // n = 960 (deeper tree, more far blocks, κ·u near the bar) (README.md "Tests").
        fullTest(`n=960 ${name}: 1e-10 vs refined truth (literal agreement logged)`, () => {
            checkCells(960, DEFAULT_THETA, 'op', name, { literalBar: false, truthBar: true });
        });
    }
});

// ===========================================================================
// The tol sweep: the measurement behind DEFAULT_SCHUR_TOL (pinned by a default test in
// schur.test.ts), and the truth bar's non-vacuity. Shares the n = 960 dense references of
// `barylen` and `pin` with the cells above through `denseRefs`' per-process cache.
// ===========================================================================

// FULL TIER ONLY (test/tier.ts): a calibration measurement, five prepareSchur + solves per set at
// n = 960 (~21 s). Why: it is the evidence behind DEFAULT_SCHUR_TOL (pinned by the default test
// above), not a per-merge check; re-run it (`bun run test:full`) whenever the tol, the GMRES
// stopping rule or the operator changes (README.md "Tests").
fullDescribe('tol sweep (spec §8.3; the knee behind DEFAULT_SCHUR_TOL), n = 960', () => {
    // MEASURED (n = 960, barylen and pin, grad/0 and gen/-phi), error vs refined truth:
    //   tol 1e-6: 2.5e-6..3.2e-6   1e-8: 3.3e-8..7.4e-8   1e-10: 2.2e-10..5.7e-10   1e-11: 2.1e-11..7.3e-11
    //   tol 1e-12: 3.7e-12..7.6e-12   1e-13: 2.2e-12..3.2e-12   1e-14: 2.2e-12..3.2e-12 (identical)
    // i.e. error ≈ 2-4 × the TRUE backward error η, with a floor of ~2-3e-12 (κ·u) that tol 1e-13
    // reaches and tol 1e-14 cannot improve. 1e-12 is within 2-3x of the floor, 13x under the bar, and
    // costs 46 iterations against 51 for 1e-13.
    for (const name of ['barylen', 'pin'] as const) {
        test(`${name}: error vs truth falls with tol to a floor; the chosen tol is within 4x of it`, () => {
            const { fx, C, cells } = setup(960, DEFAULT_THETA, name);
            const refs = denseRefs(fx, DEFAULT_THETA, 'op', name, C, cells);
            const errAt = (tol: number): { err: number; eta: number } => {
                // etaAccept: Infinity so a loose tol is observable instead of refused.
                const solver = prepareSchur(fx.op, fx.P, C, {
                    tol,
                    etaAccept: Number.POSITIVE_INFINITY,
                });
                let err = 0;
                let eta = 0;
                for (const rhs of cells) {
                    if (rhs.name === 'zero/-phi') continue; // no rhs GMRES: tests prepare only
                    const r = solver.solve(rhs.top, rhs.bottom);
                    const ref = refs.get(rhs.name);
                    if (!ref) throw new Error('missing ref');
                    err = Math.max(err, relErr(r.x, ref.x));
                    eta = Math.max(eta, r.gmres.eta, ...solver.report.columns.map((c) => c.eta));
                }
                return { err, eta };
            };
            const sweep = [1e-8, 1e-10, 1e-12, 1e-13, 1e-14].map((tol) => ({ tol, ...errAt(tol) }));
            for (const s of sweep) {
                log(
                    `tol sweep ${name}`,
                    `tol=${s.tol.toExponential(0)} err=${s.err.toExponential(2)} eta=${s.eta.toExponential(1)} err/eta=${(s.err / s.eta).toFixed(1)}`,
                );
            }
            const at = (t: number) => sweep.find((s) => s.tol === t) as (typeof sweep)[number];
            // Non-vacuity: a loose tol DOES violate the bar, so the truth bar above is a gate that can trip.
            expect(at(1e-8).err).toBeGreaterThan(BAR * 100);
            expect(at(1e-10).err).toBeGreaterThan(BAR);
            // The default clears the bar by >10x ...
            expect(DEFAULT_SCHUR_TOL).toBe(1e-12);
            expect(at(1e-12).err).toBeLessThan(BAR / 10);
            // ... and the knee: 1e-13 and 1e-14 are the same floor, 1e-12 is within 4x of it.
            const floor = at(1e-14).err;
            expect(at(1e-13).err / floor).toBeLessThan(1.5);
            expect(at(1e-12).err / floor).toBeLessThan(4);
            // Above the floor the forward error tracks the true backward error (the quantity etaAccept
            // certifies); below it η keeps falling while the error cannot, so the ratio is unbounded.
            for (const s of sweep.filter((q) => q.tol >= 1e-12))
                expect(s.err / s.eta).toBeLessThan(30);
        });
    }
});
