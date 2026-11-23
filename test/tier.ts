/**
 * Test tiers: the default tier runs on every `bun run test`; the full tier adds the heavy
 * scaling gates, calibration sweeps and artifact writers, and runs only with
 * `TEST_TIER=full` (`bun run test:full`).
 *
 * Why: 90 % of the suite's time was ~60 tests, mostly dense f64 reference solves at n = 960
 * (a 2880-3843-dim saddle per reference), run before every merge. Those checks still matter,
 * so they are skipped, not deleted, and bun reports them as `skip`: the runner prints the
 * count in the default tier and fails the full tier if any test was skipped (a check that
 * silently did not run reads exactly like one that passed, @issue utof/repulsive-test2#70).
 * When the full tier is mandatory before a merge: README.md "Tests".
 *
 * A `fullDescribe` body still RUNS when skipped (bun 1.3.11 executes a skipped describe's
 * body to collect its tests; only `beforeAll`/tests are skipped), so heavy work inside a
 * full-tier describe must live in a lazy getter, a `beforeAll`, or a test body, never in the
 * describe body itself.
 */
import { describe, test } from 'bun:test';

/** True under `TEST_TIER=full`. Why: selects the heavy variants (README.md "Tests"). */
export const FULL = process.env.TEST_TIER === 'full';

/** `test` that runs only in the full tier, reported as skipped otherwise. Why: see module doc. */
export const fullTest = test.if(FULL);

/** `describe` whose tests run only in the full tier. Why: see module doc (body still runs). */
export const fullDescribe = describe.if(FULL);
