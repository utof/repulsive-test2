// The untangle pass in the store: lifecycle of `untangle` / `untangleDone`, what the stats
// line is told, and the re-anchoring of the frozen targets. @see src/store.ts (untangleFirst)
import { afterAll, beforeEach, describe, expect, test } from 'bun:test';
import { knotSignature } from '../src/core/knotDeterminant';
import { totalLength } from '../src/core/sobolev/constraintSet';
import type { Vec3 } from '../src/core/testConfigs';
import { useSimStore } from '../src/store';

const st = () => useSimStore.getState();
const flat = (V: Vec3[]) => ({ n: V.length, x: Float64Array.from(V.flat()) });
function finishPass(): void {
    while (st().untangle) st().tickUntangle(50, Number.POSITIVE_INFINITY);
}

describe('untangle pass (store)', () => {
    beforeEach(() => {
        st().setRunning(false);
        st().setUntangleFirst(false);
        st().setPreset('knot');
    });
    // The store is one module-level object shared by every test file in a run: leave it as found.
    afterAll(() => {
        st().setRunning(false);
        st().setUntangleFirst(false);
        st().setPreset('random');
    });

    test('off by default: a run starts with no pass', () => {
        expect(st().untangleFirst).toBe(false);
        st().setRunning(true);
        expect(st().untangle).toBeNull();
        expect(st().untangleInfo).toBeNull();
    });

    test('on: the run starts in the pass, and ends it with the knot kept and targets re-anchored', () => {
        st().setUntangleFirst(true);
        const n = st().live.length;
        const length0 = totalLength(st().live, st().graph.edges);
        const sig0 = knotSignature(flat(st().live));
        st().setRunning(true);
        expect(st().untangle).not.toBeNull();
        expect(st().untangleInfo?.phase).toBe('running');

        finishPass();
        expect(st().untangleDone).toBe(true);
        expect(st().running).toBe(true); // the descent takes over
        const info = st().untangleInfo;
        expect(info?.phase).toBe('done');
        if (info?.phase === 'done') expect(info.audit).toBe('kept');
        expect(st().live.length).toBe(n);
        expect(knotSignature(flat(st().live))).toEqual(sig0);
        expect(totalLength(st().live, st().graph.edges)).toBeCloseTo(length0, 9);
        expect(st().sobolevL0).toBeCloseTo(length0, 9);
        expect(Number.isFinite(st().energy)).toBe(true);
    });

    test('once per graph: pause/play does not repeat it; a rebuild re-arms it', () => {
        st().setUntangleFirst(true);
        st().setRunning(true);
        finishPass();
        st().setRunning(false);
        st().setRunning(true);
        expect(st().untangle).toBeNull();
        st().setRunning(false);
        st().regenerate();
        expect(st().untangleDone).toBe(false);
        st().setRunning(true);
        expect(st().untangle).not.toBeNull();
    });

    test('a pause mid-pass ends it properly: an audited curve in `live`, and the pass can resume', () => {
        st().setUntangleFirst(true);
        const sig0 = knotSignature(flat(st().live));
        st().setRunning(true);
        st().tickUntangle(5, Number.POSITIVE_INFINITY);
        expect(st().untangle).not.toBeNull();
        st().setRunning(false);
        expect(st().untangle).toBeNull();
        expect(st().untangleDone).toBe(false);
        const info = st().untangleInfo;
        expect(info?.phase).toBe('done');
        if (info?.phase === 'done') {
            expect(info.stopped).toBe('abandoned');
            expect(info.audit).not.toBe('failed');
        }
        expect(knotSignature(flat(st().live))).toEqual(sig0);
        expect(st().graph.vertices).toEqual(st().live); // committed on pause
    });

    test('skipped, and says why: not one closed curve; pins set', () => {
        st().setPreset('linked-rings');
        st().setUntangleFirst(true);
        st().setRunning(true);
        expect(st().untangle).toBeNull();
        expect(st().untangleInfo).toEqual({
            phase: 'skipped',
            reason: 'not a single closed curve',
        });
        st().setRunning(false);

        st().setPreset('knot');
        st().addPin(0);
        st().setRunning(true);
        expect(st().untangle).toBeNull();
        expect(st().untangleInfo).toEqual({ phase: 'skipped', reason: 'pins are set' });
    });
});
