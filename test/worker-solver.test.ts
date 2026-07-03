import { expect, test } from 'bun:test';
import {
    type DescentMode,
    type DescentStepOutcome,
    type DispatchStepArgs,
    dispatchDescentStep,
    type SolverWorkerRequest,
    type SolverWorkerResponse,
} from '../src/core/dispatch';
import { edgeLengths, totalLength } from '../src/core/sobolev/constraintSet';
import { barycenterTarget } from '../src/core/sobolev/constraints';
import { calculateDisjointPairs } from '../src/core/tangentPointEnergy';
import { testConfigs } from '../src/core/testConfigs';

// Round-trip bit-identity gate (§T2, §6.2): the real solver worker runs the SAME
// pure dispatchDescentStep on the SAME inputs as the synchronous main-thread
// path, so per-step results must be EXACTLY equal (bit identity), never
// approximate. If exact equality fails, the drivers diverged and THAT is the bug.
// The worker is spawned via the new URL(...) form — Bun runs TS module workers
// natively; §D3's dev-server path string is the BROWSER call site only.
// @see docs/superpowers/plans/2026-07-04-worker-solver.md §T2, §6, §D3

function makeFixture() {
    // Reuse a small deterministic preset (§T2: "reuse a config from testConfigs").
    const cfg = testConfigs.find((t) => t.id === 'crossing');
    if (!cfg) throw new Error('crossing preset missing');
    const { vertices, edges } = cfg.generate();
    const disjointPairs = calculateDisjointPairs(edges);
    return {
        vertices,
        edges,
        disjointPairs,
        x0: barycenterTarget(vertices, edges),
        L0: totalLength(vertices, edges),
        ell0: edgeLengths(vertices, edges),
    };
}

function stepArgs(descentMode: DescentMode, f: ReturnType<typeof makeFixture>): DispatchStepArgs {
    // collectTimings is deliberately OMITTED: per-phase timings are wall-clock
    // measurements that differ run-to-run, so they are not part of the numeric
    // bit-identity contract. With it absent both paths report `timings: null`,
    // and toEqual on the whole outcome stays exact.
    return {
        descentMode,
        vertices: f.vertices,
        mode: 'analytical',
        stepSize: 0.001,
        x0: f.x0,
        sobolevL0: f.L0,
        barycenterConstraint: true,
        lengthMode: 'total',
        sobolevEll0: f.ell0,
        projectionMode: 'frozen',
    };
}

/**
 * Spawn the real worker, replay `topology` messages (which get no reply), then
 * the `step`, and resolve with the single response. Handler is attached before
 * any post so no message is missed.
 */
function roundTrip(
    topologies: SolverWorkerRequest[],
    stepReq: SolverWorkerRequest,
): Promise<SolverWorkerResponse> {
    return new Promise((resolve, reject) => {
        const worker = new Worker(new URL('../src/worker/solverWorker.ts', import.meta.url).href);
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
        for (const t of topologies) worker.postMessage(t);
        worker.postMessage(stepReq);
    });
}

function expectBitIdentical(got: DescentStepOutcome, expected: DescentStepOutcome) {
    // Object.is on every scalar makes the bit-identity intent explicit (and pins
    // ±0/NaN, which toEqual is lax about); the whole-object toEqual is the
    // backstop over stats/timings.
    expect(got.vertices.length).toBe(expected.vertices.length);
    for (let i = 0; i < expected.vertices.length; i++) {
        for (let k = 0; k < 3; k++) {
            expect(Object.is(got.vertices[i][k], expected.vertices[i][k])).toBe(true);
        }
    }
    expect(Object.is(got.energy, expected.energy)).toBe(true);
    expect(got.accepted).toBe(expected.accepted);
    expect(got.converged).toBe(expected.converged);
    expect(got).toEqual(expected);
}

test('worker round-trip: raw mode is bit-identical to synchronous dispatchDescentStep', async () => {
    const f = makeFixture();
    const graphVersion = 3;
    const args = stepArgs('raw', f);
    const expected = dispatchDescentStep({
        ...args,
        edges: f.edges,
        disjointPairs: f.disjointPairs,
    });

    const resp = await roundTrip([{ type: 'topology', graphVersion, edges: f.edges }], {
        type: 'step',
        graphVersion,
        args,
    });

    expect(resp.type).toBe('result');
    if (resp.type !== 'result') return;
    expect(resp.graphVersion).toBe(graphVersion);
    expectBitIdentical(resp.result, expected);
});

test('worker round-trip: sobolev mode is bit-identical to synchronous dispatchDescentStep', async () => {
    const f = makeFixture();
    const graphVersion = 7;
    const args = stepArgs('sobolev', f);
    const expected = dispatchDescentStep({
        ...args,
        edges: f.edges,
        disjointPairs: f.disjointPairs,
    });

    const resp = await roundTrip([{ type: 'topology', graphVersion, edges: f.edges }], {
        type: 'step',
        graphVersion,
        args,
    });

    expect(resp.type).toBe('result');
    if (resp.type !== 'result') return;
    expect(resp.graphVersion).toBe(graphVersion);
    // Sobolev actually moved the curve (τ ≈ 1 scale), so this exercises real
    // numeric output, not a no-op step.
    expect(resp.result.vertices).not.toEqual(f.vertices);
    expectBitIdentical(resp.result, expected);
});

test('worker echoes the request graphVersion untouched (§D5 mismatch-drop contract)', async () => {
    const f = makeFixture();
    // topology carries gv=1; the step carries a DIFFERENT gv. The worker computes
    // from its cached topology but must echo the STEP's gv verbatim — the main
    // thread relies on that tag to DROP a stale result whose gv no longer matches
    // the store (§D5). If the worker rewrote the tag, the drop test would be moot.
    const resp = await roundTrip([{ type: 'topology', graphVersion: 1, edges: f.edges }], {
        type: 'step',
        graphVersion: 999,
        args: stepArgs('sobolev', f),
    });
    expect(resp.type).toBe('result');
    if (resp.type !== 'result') return;
    expect(resp.graphVersion).toBe(999);
});

test('worker posts an error (never a silent crash) when a step arrives before topology (§D6/§D11)', async () => {
    const f = makeFixture();
    // No topology sent: the try/catch must convert the missing-cache throw into a
    // protocol error message, which the main driver turns into the §D6 fallback.
    const resp = await roundTrip([], { type: 'step', graphVersion: 1, args: stepArgs('raw', f) });
    expect(resp.type).toBe('error');
    if (resp.type !== 'error') return;
    expect(resp.message).toContain('before topology');
});
