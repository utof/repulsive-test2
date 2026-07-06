import { expect, test } from 'bun:test';
import { useSimStore } from '../src/store';

// §D14 field reuse (issue #9): `arrowField` publishes the descent field the STEP
// actually computed (raw dE / full-set g̃) so GradientArrows renders it directly
// while running — no redundant second-worker recompute. The store owns only the
// field + its null default; the publish path (Viewer.applyStepOutcome) and the
// running-consume path (GradientArrows) need a Canvas/Worker and so live outside
// the store test, exactly like the solverDriver split in store-solver-driver.test.ts.
// @see docs/superpowers/plans/2026-07-04-worker-solver.md §D14
test('store: arrowField defaults to null', () => {
    // Null = no step has published a field yet (and singular-saddle steps publish
    // null too, hiding the arrows) — GradientArrows falls back to its on-demand path.
    expect(useSimStore.getState().arrowField).toBeNull();
});
