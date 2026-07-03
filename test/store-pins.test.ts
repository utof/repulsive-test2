import { expect, test } from 'bun:test';
import { sobolevStepSet } from '../src/core/optimizer';
import {
    barycenterBlock,
    edgeLengths,
    pointBlock,
    totalLength,
    totalLengthBlock,
} from '../src/core/sobolev/constraintSet';
import { barycenterTarget } from '../src/core/sobolev/constraints';
import { flatten } from '../src/core/sobolev/layout';
import { dispatchDescentStep, useSimStore } from '../src/store';

// Pin-drag milestone store/dispatch coverage (briefing §5B): the pin actions
// (add/dedupe/remove/enable/target), the frozen-target lifecycle mirroring the
// sobolevEll0 anchor (clear on rebuild, re-anchor on play/commit, frame loop
// never mutates), and the pins → pointBlock dispatch wiring.
// NOTE: the store is a module-level singleton shared across test files —
// every test restores what it changes (pins are cleared by setPreset, so
// starting each test with setPreset('crossing') is a clean reset).
// @see docs/superpowers/plans/2026-07-03-pin-drag-ui.md (Decisions D2, D5, D6)
// @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §2, §3.5, §5.3

test('store: addPin snapshots the current live position, dedupes by vertexIndex; remove/enable/target actions', () => {
    useSimStore.getState().setPreset('crossing');
    expect(useSimStore.getState().pins).toEqual([]);

    // addPin snapshots live[2] as a COPY (later live motion must not move it).
    const live = useSimStore.getState().live;
    const snapshot: [number, number, number] = [live[2][0], live[2][1], live[2][2]];
    useSimStore.getState().addPin(2);
    const pins = useSimStore.getState().pins;
    expect(pins.length).toBe(1);
    expect(pins[0].vertexIndex).toBe(2);
    expect(pins[0].enabled).toBe(true);
    expect(pins[0].target).toEqual(snapshot);

    // The target is a copy: mutating live must not move it.
    useSimStore.getState().live[2][0] += 5;
    expect(useSimStore.getState().pins[0].target).toEqual(snapshot);

    // addPin on the same vertex is idempotent (no duplicate, keeps original target).
    useSimStore.getState().addPin(2);
    expect(useSimStore.getState().pins.length).toBe(1);
    expect(useSimStore.getState().pins[0].target).toEqual(snapshot);

    // A second, distinct pin appends.
    useSimStore.getState().addPin(0);
    expect(useSimStore.getState().pins.map((p) => p.vertexIndex)).toEqual([2, 0]);

    // setPinEnabled flips only that pin; clears the stale converged verdict.
    useSimStore.setState({ sobolevConverged: true });
    useSimStore.getState().setPinEnabled(2, false);
    expect(useSimStore.getState().pins.find((p) => p.vertexIndex === 2)?.enabled).toBe(false);
    expect(useSimStore.getState().sobolevConverged).toBe(false);

    // setPinTarget updates only the target of the named pin.
    useSimStore.getState().setPinTarget(0, [1.25, -0.5, 0.75]);
    expect(useSimStore.getState().pins.find((p) => p.vertexIndex === 0)?.target).toEqual([
        1.25, -0.5, 0.75,
    ]);
    expect(useSimStore.getState().pins.find((p) => p.vertexIndex === 2)?.target).toEqual(snapshot);

    // removePin drops just that pin.
    useSimStore.getState().removePin(2);
    expect(useSimStore.getState().pins.map((p) => p.vertexIndex)).toEqual([0]);

    // Restore: setPreset clears pins (verified in the lifecycle test below).
    useSimStore.getState().setPreset('crossing');
    expect(useSimStore.getState().pins).toEqual([]);
});

test('store: pins clear on rebuild and re-anchor on play/commit from live positions, never mid-run (frozen-targets lifecycle)', () => {
    useSimStore.getState().setPreset('crossing');
    useSimStore.getState().addPin(1);
    expect(useSimStore.getState().pins.length).toBe(1);

    // Rebuild (regenerate) is a re-anchor boundary where indices become
    // meaningless → pins CLEAR (the faithful mirror of "recompute from the new
    // graph"). @see plan D5.
    useSimStore.getState().regenerate();
    expect(useSimStore.getState().pins).toEqual([]);

    // Re-pin, then perturb live and play: the pin target must re-anchor to the
    // CURRENT live position (mirrors sobolevEll0's play re-anchor).
    useSimStore.getState().addPin(1);
    useSimStore.getState().live[1][0] += 0.321;
    const expected: [number, number, number] = [
        useSimStore.getState().live[1][0],
        useSimStore.getState().live[1][1],
        useSimStore.getState().live[1][2],
    ];
    useSimStore.getState().setRunning(true);
    const pinsAtPlay = useSimStore.getState().pins;
    expect(pinsAtPlay[0].target).toEqual(expected);

    // Mid-run motion of live must NOT move the frozen target (same array
    // identity — nothing re-anchored it). Mirror of the sobolevEll0 test.
    useSimStore.getState().live[1][1] += 0.5;
    expect(useSimStore.getState().pins).toBe(pinsAtPlay);
    expect(useSimStore.getState().pins[0].target).toEqual(expected);

    // Pause = vertex commit → re-anchor from the committed positions.
    useSimStore.getState().setRunning(false);
    const committed: [number, number, number] = [
        useSimStore.getState().live[1][0],
        useSimStore.getState().live[1][1],
        useSimStore.getState().live[1][2],
    ];
    expect(useSimStore.getState().pins[0].target).toEqual(committed);
    expect(useSimStore.getState().pins[0].target).not.toEqual(expected);

    useSimStore.getState().setPreset('crossing');
});

test('dispatchDescentStep: enabled in-range pins append pointBlocks after the length block; disabled/out-of-range excluded; absent ≡ no pins', () => {
    useSimStore.getState().setPreset('crossing');
    const st = useSimStore.getState();
    const vertices = st.graph.vertices;
    const edges = st.graph.edges;
    const disjointPairs = st.disjointPairs;
    const x0 = barycenterTarget(vertices, edges);
    const L0 = totalLength(vertices, edges);
    const target: [number, number, number] = [
        vertices[0][0] + 0.1,
        vertices[0][1] - 0.2,
        vertices[0][2] + 0.3,
    ];
    const base = {
        descentMode: 'sobolev' as const,
        vertices,
        edges,
        disjointPairs,
        mode: 'analytical' as const,
        stepSize: 0.001,
        x0,
        sobolevL0: L0,
        sobolevEll0: edgeLengths(vertices, edges),
        barycenterConstraint: true,
        lengthMode: 'total' as const,
    };
    const opts = { mode: 'analytical' as const };

    // One enabled pin → [barycenter, totalLength, point(0)] (pins appended after
    // the length block, plan D6 row order).
    const withPin = dispatchDescentStep({
        ...base,
        pins: [{ vertexIndex: 0, target, enabled: true }],
    });
    const withPinRef = sobolevStepSet(
        vertices,
        edges,
        disjointPairs,
        [barycenterBlock(x0), totalLengthBlock(L0), pointBlock(0, target)],
        opts,
    );
    expect(withPin.energy).toBe(withPinRef.energy);
    expect(flatten(withPin.vertices)).toEqual(flatten(withPinRef.vertices));

    // No pins ≡ the M2 [barycenter, totalLength] path.
    const noPinRef = sobolevStepSet(
        vertices,
        edges,
        disjointPairs,
        [barycenterBlock(x0), totalLengthBlock(L0)],
        opts,
    );

    // A disabled pin contributes NO block.
    const disabled = dispatchDescentStep({
        ...base,
        pins: [{ vertexIndex: 0, target, enabled: false }],
    });
    expect(disabled.energy).toBe(noPinRef.energy);
    expect(flatten(disabled.vertices)).toEqual(flatten(noPinRef.vertices));

    // An out-of-range pin contributes NO block (never breaks the descent).
    const stale = dispatchDescentStep({
        ...base,
        pins: [{ vertexIndex: 999, target, enabled: true }],
    });
    expect(stale.energy).toBe(noPinRef.energy);
    expect(flatten(stale.vertices)).toEqual(flatten(noPinRef.vertices));

    // pins field ABSENT ≡ no pins (existing call shape unaffected).
    const absent = dispatchDescentStep(base);
    expect(absent.energy).toBe(noPinRef.energy);
    expect(flatten(absent.vertices)).toEqual(flatten(noPinRef.vertices));

    useSimStore.getState().setPreset('crossing');
});
