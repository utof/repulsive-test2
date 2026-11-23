// The force-layout pre-pass in the store: lifecycle of `prePass` / `prePassDone`, and the
// re-anchoring of the frozen targets when it ends. @see src/store.ts (the forcePrePass field)
import { beforeEach, describe, expect, test } from 'bun:test';
import { edgeLengths, totalLength } from '../src/core/sobolev/constraintSet';
import type { Vec3 } from '../src/core/testConfigs';
import { useSimStore } from '../src/store';

const st = () => useSimStore.getState();
/** Tick with no time budget until the pre-pass phase ends. */
function finishPrePass(): number {
    let calls = 0;
    while (st().prePass) {
        st().tickPrePass(50, Number.POSITIVE_INFINITY);
        calls++;
    }
    return calls;
}

describe('force-layout pre-pass (store)', () => {
    beforeEach(() => {
        st().setRunning(false);
        st().setForcePrePass(false);
        st().setPreset('random');
    });

    test('off by default: a run starts with no pre-pass', () => {
        expect(st().forcePrePass).toBe(false);
        st().setRunning(true);
        expect(st().prePass).toBeNull();
        st().setRunning(false);
    });

    test('on: the run starts in the pre-pass phase, and ends it with re-anchored targets', () => {
        st().setForcePrePass(true);
        const before = st().live.map((p) => [...p]);
        const length0 = totalLength(st().live, st().graph.edges);
        st().setRunning(true);
        expect(st().prePass).not.toBeNull();
        expect(st().prePassAlpha).toBe(1);

        st().tickPrePass(1, Number.POSITIVE_INFINITY);
        expect(st().prePass?.ticks).toBe(1);
        expect(st().live).not.toEqual(before); // the live buffer moves during the phase

        finishPrePass();
        expect(st().prePassDone).toBe(true);
        expect(st().prePassAlpha).toBeNull();
        expect(st().running).toBe(true); // the descent takes over; the run does not stop
        // Same total length as the run started with, and the targets describe the NEW geometry.
        expect(totalLength(st().live, st().graph.edges)).toBeCloseTo(length0, 9);
        expect(st().sobolevL0).toBeCloseTo(length0, 9);
        expect(st().sobolevEll0).toEqual(edgeLengths(st().live, st().graph.edges));
        expect(Number.isFinite(st().energy)).toBe(true);
    });

    test('once per graph: pause/play does not repeat it; a rebuild or a toggle re-arms it', () => {
        st().setForcePrePass(true);
        st().setRunning(true);
        finishPrePass();
        st().setRunning(false);
        st().setRunning(true);
        expect(st().prePass).toBeNull();
        st().setRunning(false);

        st().regenerate();
        expect(st().prePassDone).toBe(false);
        st().setRunning(true);
        expect(st().prePass).not.toBeNull();
        st().setRunning(false);

        st().setRunning(true);
        finishPrePass();
        st().setRunning(false);
        st().setForcePrePass(false);
        st().setForcePrePass(true);
        st().setRunning(true);
        expect(st().prePass).not.toBeNull();
        st().setRunning(false);
    });

    test('a pause during the pre-pass abandons it and commits where the nodes are', () => {
        st().setForcePrePass(true);
        st().setRunning(true);
        st().tickPrePass(5, Number.POSITIVE_INFINITY);
        st().setRunning(false);
        expect(st().prePass).toBeNull();
        expect(st().prePassDone).toBe(false);
        expect(st().graph.vertices).toEqual(st().live);
        expect(st().sobolevL0).toBeCloseTo(totalLength(st().live, st().graph.edges), 12);
    });

    test('an enabled pin holds its vertex through the pre-pass', () => {
        st().addPin(0);
        const pinned: Vec3 = [...st().live[0]];
        st().setForcePrePass(true);
        st().setRunning(true);
        finishPrePass();
        expect(st().live[0]).toEqual(pinned);
        expect(st().pins[0].target).toEqual(pinned);
        st().setRunning(false);
    });
});
