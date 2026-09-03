import { expect, test } from 'bun:test';
import type { Edge, Vec3 } from '../../src/core/testConfigs';
import {
    allFinite,
    GpuDriver,
    type GpuStepSource,
    gpuStepSupported,
    gpuTopologySupported,
} from '../../src/gpu/driver';
import { GpuTangentPoint, REDUCE_MAX_EDGES } from '../../src/gpu/tangentPoint';
import { packTopology } from '../../src/gpu/topology';
import { useSimStore } from '../../src/store';

// Pure routing predicates of the 'gpu' driver, plus the two `GpuDriver.step`
// behaviours reachable without a device (the single-flight guard and the
// predicate/constructor cross-check). A step that actually READS BACK from the
// GPU needs a renderer and is exercised by the browser driver smoke (plan Task 8).
// @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §1, §2.6

const pen0 = useSimStore.getState().penalties; // the store's all-zero default

/**
 * A renderer that throws on ANY property read — the idiom from
 * `test/gpu/tangentPoint.test.ts`. `GpuTangentPoint`'s constructor never reads a
 * renderer property (it only stores the reference and builds TSL nodes), so a
 * SUPPORTED topology constructs fine here and the throw lands where the kernel
 * first dispatches, inside `gradient()`. That is exactly what the in-flight test
 * below needs: `step` suspends at `await this.engine.gradient(...)` before the
 * rejection surfaces.
 */
const throwingRenderer = () =>
    new Proxy(
        {},
        {
            get() {
                throw new Error('renderer touched');
            },
        },
    );

test('gpuStepSupported: sobolev + analytical + penalties-off only (spec §1)', () => {
    expect(gpuStepSupported({ descentMode: 'sobolev', mode: 'analytical', penalties: pen0 })).toBe(
        true,
    );
    expect(gpuStepSupported({ descentMode: 'raw', mode: 'analytical', penalties: pen0 })).toBe(
        false,
    );
    expect(gpuStepSupported({ descentMode: 'sobolev', mode: 'finiteDiff', penalties: pen0 })).toBe(
        false,
    );
    expect(
        gpuStepSupported({
            descentMode: 'sobolev',
            mode: 'analytical',
            penalties: { ...pen0, totalLength: 1 },
        }),
    ).toBe(false);
});

test('allFinite rejects NaN/inf', () => {
    expect(allFinite([[0, 1, 2]])).toBe(true);
    expect(allFinite([[0, Number.NaN, 2]])).toBe(false);
    expect(allFinite([[Number.POSITIVE_INFINITY, 0, 0]])).toBe(false);
});

// Mirrors the GpuTangentPoint constructor domain (src/gpu/tangentPoint.ts): the
// two MUST agree, else a "supported" topology would throw in the constructor
// and flip the driver (spec §2.6) instead of routing that step to the CPU (spec §1).
test('gpuTopologySupported: 1..REDUCE_MAX_EDGES edges with ≥1 disjoint pair', () => {
    expect(gpuTopologySupported(0, 0)).toBe(false); // no edges
    expect(gpuTopologySupported(1, 0)).toBe(false); // one edge, no disjoint pairs
    expect(gpuTopologySupported(3, 0)).toBe(false); // triangle: every edge touches every other
    expect(gpuTopologySupported(1, 1)).toBe(true);
    expect(gpuTopologySupported(4, 4)).toBe(true);
    expect(gpuTopologySupported(REDUCE_MAX_EDGES, 2)).toBe(true);
    expect(gpuTopologySupported(REDUCE_MAX_EDGES + 1, 2)).toBe(false);
});

// The prose "MUST agree" above, made executable: for every topology the predicate
// rejects, the real constructor must also throw — so drift between the two fails
// here instead of flipping the driver at runtime. The constructor's domain checks
// run before any renderer access, so a throwing Proxy renderer is enough.
const overLimitCycle = (): { V: number; edges: Edge[]; dp: number[][] } => {
    const E = REDUCE_MAX_EDGES + 1;
    const edges = Array.from({ length: E }, (_, i) => [i, (i + 1) % E] as Edge);
    const dp: number[][] = edges.map(() => []);
    // Two genuinely disjoint edges, so pairCount > 0 and ONLY the edge bound rejects.
    dp[0] = [2];
    dp[2] = [0];
    return { V: E, edges, dp };
};

test('gpuTopologySupported agrees with the GpuTangentPoint constructor on every reject case', () => {
    const cases: { name: string; V: number; edges: Edge[]; dp: number[][] }[] = [
        { name: 'no edges', V: 3, edges: [], dp: [] },
        { name: 'single edge', V: 2, edges: [[0, 1]], dp: [[]] },
        {
            name: '2-edge path',
            V: 3,
            edges: [
                [0, 1],
                [1, 2],
            ],
            dp: [[], []],
        },
        {
            name: 'triangle',
            V: 3,
            edges: [
                [0, 1],
                [1, 2],
                [2, 0],
            ],
            dp: [[], [], []],
        },
        { name: 'over the reduction limit', ...overLimitCycle() },
    ];
    for (const { name, V, edges, dp } of cases) {
        const pairCount = dp.reduce((s, l) => s + l.length, 0);
        expect(gpuTopologySupported(edges.length, pairCount), name).toBe(false);
        const topo = packTopology(V, edges, dp);
        expect(() => new GpuTangentPoint(throwingRenderer() as never, topo), name).toThrow(
            /^GpuTangentPoint: /,
        );
    }
});

// A 4-cycle: edges 0 and 2 are disjoint, as are 1 and 3.
const CYCLE4_EDGES: Edge[] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
];
const CYCLE4_DP: number[][] = [[2], [3], [0], [1]];
const CYCLE4_LIVE: Vec3[] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
];

test('gpuTopologySupported accepts a 4-cycle (predicate side of the accept case)', () => {
    // Predicate only: the constructor's ACCEPT side needs no renderer either (the
    // in-flight test below constructs this very topology), but a full accept path —
    // dispatch + readback — needs a device and is covered by the Task 8 smoke.
    expect(gpuTopologySupported(CYCLE4_EDGES.length, 4)).toBe(true);
});

/**
 * Store state with the topology and config a GPU step is actually served for
 * (spec §1). The sobolev frozen targets are overridden to match the 4-cycle so
 * the CPU route below is a real step, not a length-mismatch throw.
 */
const gpuStepSource = (over: Partial<GpuStepSource> = {}): GpuStepSource =>
    ({
        ...useSimStore.getState(),
        descentMode: 'sobolev',
        mode: 'analytical',
        penalties: pen0,
        live: CYCLE4_LIVE.map((v) => [...v] as Vec3),
        graph: { edges: CYCLE4_EDGES },
        disjointPairs: CYCLE4_DP,
        graphVersion: 1,
        sobolevX0: [0.5, 0.5, 0] as Vec3,
        sobolevEll0: [1, 1, 1, 1],
        sobolevL0: 4,
        pins: [],
        ...over,
    }) as unknown as GpuStepSource;

// I1: the driver is the LAST line of defense for spec §1, not just the Viewer.
// The core's `opts.dE ?? …` seam takes a supplied dE regardless of `mode`, so a
// mis-routed finiteDiff step would silently be served the GPU ANALYTICAL dE and
// a raw step would report `usedGpu` for a dE the dispatch discards. The throwing
// Proxy renderer is the falsifier: if the config check were removed, `step` would
// build the engine and blow up on `gradient()` instead of returning a CPU step.
test('GpuDriver.step routes an unsupported CONFIG to the CPU dE without touching the renderer', async () => {
    const unsupported: Partial<GpuStepSource>[] = [
        { mode: 'finiteDiff' },
        { descentMode: 'raw' },
        { penalties: { ...pen0, totalLength: 1 } },
    ];
    for (const over of unsupported) {
        const label = JSON.stringify(over);
        const driver = new GpuDriver(throwingRenderer() as never);
        const { usedGpu, outcome } = await driver.step(gpuStepSource(over), undefined);
        expect(usedGpu, label).toBe(false);
        expect(outcome.vertices.length, label).toBe(CYCLE4_LIVE.length);
    }
});

// Executable half of `step`'s single-flight contract (see its TSDoc): a second
// call while one is outstanding must be REJECTED, not served — the graphVersion
// rebuild would otherwise dispose an engine whose readback is still pending and
// make the FIRST call fail on what was only a preset change.
test('GpuDriver.step rejects an overlapping call and re-arms afterwards', async () => {
    const st = gpuStepSource();
    expect(gpuStepSupported(st)).toBe(true); // guard: we are on the GPU path, not the CPU route
    const driver = new GpuDriver(throwingRenderer() as never);

    const first = driver.step(st, undefined); // suspends at `await engine.gradient(...)`
    await expect(driver.step(st, undefined)).rejects.toThrow('GpuDriver.step: overlapping call');
    // The outstanding call fails on the Proxy renderer, NOT on the guard.
    await expect(first).rejects.toThrow('renderer touched');
    // `finally` cleared the latch, so the driver is usable again.
    await expect(driver.step(st, undefined)).rejects.toThrow('renderer touched');
    // No `driver.dispose()`: it would reach into the Proxy renderer and throw. Real
    // callers dispose against a real device (Viewer unmount / driver switch).
});
