import { expect, test } from 'bun:test';
import {
    type DispatchDescentStepArgs,
    dispatchDescentStep,
    type SolverWorkerResponse,
} from '../../src/core/dispatch';
import { trefoil } from '../../src/core/fixtures';
import { DEFAULTS } from '../../src/core/optimizer';
import { assembleAFlat } from '../../src/core/sobolev/innerProduct';
import {
    ExternalSolveError,
    luSolve,
    type PreparedSaddle,
    type SaddleFactorization,
} from '../../src/core/sobolev/linsolve';
import { timingsEnd } from '../../src/core/sobolev/phaseTimings';
import { calculateDisjointPairs } from '../../src/core/tangentPointEnergy';
import type { Vec3 } from '../../src/core/testConfigs';

function args(n = 24): DispatchDescentStepArgs {
    const { vertices, edges } = trefoil(n);
    const x0: Vec3 = [0, 0, 0];
    for (const v of vertices) {
        x0[0] += v[0] / n;
        x0[1] += v[1] / n;
        x0[2] += v[2] / n;
    }
    const L0 = edges.reduce(
        (s, [a, b]) =>
            s +
            Math.hypot(
                vertices[a][0] - vertices[b][0],
                vertices[a][1] - vertices[b][1],
                vertices[a][2] - vertices[b][2],
            ),
        0,
    );
    return {
        descentMode: 'sobolev',
        vertices,
        edges,
        disjointPairs: calculateDisjointPairs(edges),
        mode: 'analytical',
        stepSize: 1,
        x0,
        barycenterConstraint: true,
        lengthMode: 'total',
        sobolevL0: L0,
        projectionMode: 'frozen',
        collectTimings: false,
        collectField: true,
    };
}

// What this proves, exactly: with `saddle` ABSENT the step is DETERMINISTIC and leaks
// nothing across calls, with all three of this task's default-path control-flow edits
// in place (the saddle.A shape guard at optimizer.ts:265-266, the ExternalSolveError
// rethrow at :361, the timing-collector disarm at :282).
// What it does NOT prove: identity with the PRE-seam code. Both arms are the post-seam
// build, so a regression the seam introduced is present on both sides and cancels. The
// pre/post backstop is the committed golden suites (test/golden.test.ts,
// test/sobolev/constraintSetFlow.test.ts vs oracle/golden/*.json), which is why the full
// suite is run against the 292 baseline. Determinism is a PRECONDITION for that
// comparison, which is why this test is still worth having.
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4
test('saddle seam ABSENT: the step is deterministic and leaks nothing between calls', () => {
    const a = args();
    // Guard the comparison's meaning: a rejected/converged step would compare two echoes.
    expect(dispatchDescentStep(a).accepted).toBe(true);
    expect(dispatchDescentStep(args())).toEqual(dispatchDescentStep(a));
});

// D1's SECOND half — the one the `finally` spelling fails. A collected step that does
// NOT throw must still return a populated ledger; an unconditional finally would have
// nulled `acc` before optimizer.ts:461 reads it.
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4 (D1)
test('saddle seam ABSENT: collectTimings still returns a ledger (the disarm is throw-path only)', () => {
    const r = dispatchDescentStep({ ...args(), collectTimings: true });
    expect(r.timings).toBeDefined();
    expect(r.timings?.step?.calls).toBe(1);
    // NOT an exact count: `saddle` fires once for the gradient solve (gradient.ts:106)
    // plus once per frozen projection Newton iteration (lineSearch.ts:260), and that
    // iteration count is fixture- and step-size-dependent. Asserting a literal here
    // would be pinning an incidental number, not the ledger's presence.
    expect(r.timings?.saddle?.calls).toBeGreaterThanOrEqual(1);
});
