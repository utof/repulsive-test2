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

// A PreparedSaddle that delegates to a plain LU of the SAME K the core would have
// built, so `saddle` present and `saddle` absent must agree to round-off. This is the
// routing probe; the real numerics are the GPU Cholesky's ([DESIGN §4.1]). Note the
// coordinate-major block write `K[b*n+i][b*n+j]` — Ā = I₃ ⊗ A.
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1
function identityPrepared(a: DispatchDescentStepArgs): PreparedSaddle {
    const A = assembleAFlat(
        a.vertices,
        a.edges,
        a.disjointPairs,
        DEFAULTS.alpha,
        DEFAULTS.beta,
        DEFAULTS.epsilon,
    );
    return {
        A,
        factor(C: number[][]): SaddleFactorization {
            const n = a.vertices.length;
            const m = 3 * n;
            const k = C.length;
            const size = m + k;
            const K: number[][] = Array.from({ length: size }, () => new Array(size).fill(0));
            for (let b = 0; b < 3; b++)
                for (let i = 0; i < n; i++)
                    for (let j = 0; j < n; j++) K[b * n + i][b * n + j] = A[i * n + j];
            for (let r = 0; r < k; r++)
                for (let c = 0; c < m; c++) {
                    K[m + r][c] = C[r][c];
                    K[c][m + r] = C[r][c];
                }
            return {
                kind: 'external',
                solve: (rhs: number[]) => luSolve(K, rhs),
            };
        },
    };
}

function boomPrepared(a: DispatchDescentStepArgs): PreparedSaddle {
    return {
        A: assembleAFlat(
            a.vertices,
            a.edges,
            a.disjointPairs,
            DEFAULTS.alpha,
            DEFAULTS.beta,
            DEFAULTS.epsilon,
        ),
        factor: () => ({
            kind: 'external',
            solve: () => {
                throw new ExternalSolveError('IR did not converge');
            },
        }),
    };
}

test('saddle seam PRESENT: routes through the external solve and agrees with the CPU path', () => {
    const a = args();
    const base = dispatchDescentStep(a);
    const viaSeam = dispatchDescentStep({ ...args(), saddle: identityPrepared(a) });
    expect(viaSeam.accepted).toBe(base.accepted);
    const b = base.descentField as Vec3[];
    const t = viaSeam.descentField as Vec3[];
    let checked = 0;
    for (let i = 0; i < b.length; i++)
        for (let c = 0; c < 3; c++) {
            if (Math.abs(b[i][c]) < 1e-12) continue;
            expect(t[i][c] / b[i][c]).toBeCloseTo(1, 8);
            checked++;
        }
    // Non-vacuity: if every component were filtered out the loop would assert nothing.
    expect(checked).toBeGreaterThan(0);
});

test('saddle seam: a thrown ExternalSolveError propagates out of dispatchDescentStep', () => {
    // NOT swallowed into 'singular_system': the driver must be able to see it and
    // re-run the step on the CPU path ([DESIGN §4.5] fallback trigger 3).
    // @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.5
    expect(() => dispatchDescentStep({ ...args(), saddle: boomPrepared(args()) })).toThrow(
        ExternalSolveError,
    );
});

// D1's FIRST half. Asserts the collector's OWN state, NOT a later step's ledger: the
// obvious spelling — run another collected step and check `timings.step.calls === 1` —
// CANNOT FAIL, because timingsBegin() is `acc = {}` (phaseTimings.ts:48-50), an
// UNCONDITIONAL reset, so the next collected step destroys any leaked ledger before
// recording a phase and `step.calls` is 1 with or without the disarm. timingsEnd()
// returns the leaked partial ledger when the disarm is missing and null when it fired,
// so this discriminates. Do not "simplify" it back — see D1.
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4 (D1)
test('saddle seam: a throw disarms the timing collector', () => {
    expect(() =>
        dispatchDescentStep({
            ...args(),
            saddle: boomPrepared(args()),
            collectTimings: true,
        }),
    ).toThrow(ExternalSolveError);
    expect(timingsEnd()).toBeNull();
});

// D1's first half again, at the hazard the previous test cannot reach: the step AFTER
// the throw has `collectTimings: false`, so it never calls timingsBegin() and a leaked
// `acc` is never reset. Without the disarm, timed() accumulates that step's phases into
// the THROWING step's ledger — for the process lifetime, paying performance.now() on
// every phase of every later step and handing a cross-step ledger to whoever collects
// next. This is the failure the vacuous spelling above would have shipped.
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4 (D1)
test('saddle seam: a leaked collector cannot accumulate into an UNCOLLECTED step', () => {
    expect(() =>
        dispatchDescentStep({
            ...args(),
            saddle: boomPrepared(args()),
            collectTimings: true,
        }),
    ).toThrow(ExternalSolveError);
    dispatchDescentStep({ ...args(), collectTimings: false });
    expect(timingsEnd()).toBeNull();
});

// The `saddle.A` contract ([DESIGN §4.4]: "the matrix of the SAME vertices the step
// receives") enforced, not merely documented. solveSaddleFrozen (linsolve.ts:882-905)
// validates rhsTop/rhsBottom lengths ONLY, while the path it replaces does validate A
// (solveSaddleFromA, :755-757: `if (a.length !== n * n) throw`) — so without this guard
// the seam path is strictly LESS checked than the CPU path. A short A reads out of
// range → undefined → NaN → structuredSaddleResidual returns NaN → nothing in src/core/
// thresholds `residual`, so the line search rejects a NaN direction as `armijo_failed`
// and the run auto-pauses with usedGpuSolve still true: the same silent-failure class
// [DESIGN §4.1] step 4's throw exists to prevent.
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4
test('saddle seam: a wrong-sized saddle.A throws before anything is solved', () => {
    const a = args();
    const bad: PreparedSaddle = { ...identityPrepared(a), A: new Float64Array(4) };
    expect(() => dispatchDescentStep({ ...a, saddle: bad })).toThrow(/saddle\.A length/);
    // The guard is on the PRE-ARM path (before timingsBegin(), like the dE guard at
    // optimizer.ts:265-266), so it needs no disarm — and this proves it, since a guard
    // moved below timingsBegin() would leave the collector armed here (D1).
    expect(() => dispatchDescentStep({ ...a, saddle: bad, collectTimings: true })).toThrow(
        /saddle\.A length/,
    );
    expect(timingsEnd()).toBeNull();
});

// The worker guard's falsifier. Without it the guard is dead code that READS as a live
// safety property: `saddle` rides `msg.args`, never `msg` (D4). `topology` must be posted
// first, or the worker throws 'step received before topology' and this test would pass
// for the wrong reason — hence the message assertion, not just `type === 'error'`.
// Round-trip shape mirrors test/worker-solver.test.ts:71-92.
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4
test('solverWorker: a saddle on the step payload is rejected', async () => {
    const a = args();
    const response = await new Promise<SolverWorkerResponse>((resolve, reject) => {
        const worker = new Worker(
            new URL('../../src/worker/solverWorker.ts', import.meta.url).href,
        );
        const timer = setTimeout(() => {
            worker.terminate();
            reject(new Error('worker timed out'));
        }, 5000);
        worker.onmessage = (event: MessageEvent<SolverWorkerResponse>) => {
            clearTimeout(timer);
            worker.terminate();
            resolve(event.data);
        };
        worker.onerror = (event: ErrorEvent) => {
            clearTimeout(timer);
            worker.terminate();
            reject(new Error(`worker error: ${event.message}`));
        };
        worker.postMessage({ type: 'topology', graphVersion: 0, edges: a.edges });
        // A PLAIN {}, deliberately: a real PreparedSaddle carries closures, so
        // postMessage would fail with a DataCloneError before the guard ever ran. The
        // cast is what makes the guard reachable from a test at all. `edges` /
        // `disjointPairs` ride along harmlessly — solverWorker.ts:75-79 overwrites both
        // from the topology cache.
        worker.postMessage({
            type: 'step',
            graphVersion: 0,
            args: { ...a, saddle: {} as unknown as PreparedSaddle },
        });
    });
    if (response.type !== 'error') throw new Error(`expected an error, got ${response.type}`);
    // The GUARD's own message, not a loose /saddle/. Measured: with the guard
    // mis-spelled as `'saddle' in msg` (D4's dead-code form, which can never fire)
    // the step still errors — the saddle.A guard at optimizer.ts:290-291 rejects the
    // plain {} with a TypeError naming `opts.saddle.A`, whose text ALSO contains
    // "saddle". A /saddle/ assertion therefore passes on the dead spelling and
    // falsifies nothing; this one fails on it.
    expect(response.message).toMatch(/`saddle` cannot cross the worker boundary/);
}, 10_000);
