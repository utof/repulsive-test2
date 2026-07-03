import { expect, test } from 'bun:test';
import { sobolevStep, sobolevStepSet } from '../src/core/optimizer';
import {
    barycenterBlock,
    edgeLengths,
    edgeLengthsBlock,
    totalLength,
    totalLengthBlock,
} from '../src/core/sobolev/constraintSet';
import { barycenterTarget } from '../src/core/sobolev/constraints';
import { flatten } from '../src/core/sobolev/layout';
import { dispatchDescentStep, useSimStore } from '../src/store';

// Constraints-M2 store/dispatch coverage: the 3-way lengthMode (spec §5.3)
// with its legacy-boolean write-through mirror, the frozen-ℓ⁰ lifecycle
// (mirrors the sobolevL0 tests in test/store-constraints.test.ts — same three
// re-anchor points), and the lengthMode-driven ConstraintSet dispatch.
// NOTE: the store is a module-level singleton shared across test files —
// every test here restores the defaults it changes (lengthMode 'total'),
// because test/store-constraints.test.ts asserts those defaults.
// @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §3.5, §5.3, §9a

test('store: lengthMode defaults to total; setLengthMode syncs the legacy boolean mirror and clears converged', () => {
    useSimStore.getState().setPreset('crossing');
    expect(useSimStore.getState().lengthMode).toBe('total');
    expect(useSimStore.getState().lengthConstraint).toBe(true);

    useSimStore.setState({ sobolevConverged: true });
    useSimStore.getState().setLengthMode('perEdge');
    expect(useSimStore.getState().lengthMode).toBe('perEdge');
    // perEdge is still "a length constraint" for the legacy mirror.
    expect(useSimStore.getState().lengthConstraint).toBe(true);
    // A converged verdict is per-constraint-set: toggling invalidates it.
    expect(useSimStore.getState().sobolevConverged).toBe(false);

    useSimStore.getState().setLengthMode('none');
    expect(useSimStore.getState().lengthConstraint).toBe(false);

    // The legacy M1 setter writes THROUGH lengthMode (mirror can't diverge).
    useSimStore.getState().setLengthConstraint(true);
    expect(useSimStore.getState().lengthMode).toBe('total');
    useSimStore.getState().setLengthConstraint(false);
    expect(useSimStore.getState().lengthMode).toBe('none');

    useSimStore.getState().setLengthMode('total');
});

test('store: ℓ⁰ re-anchors on play and on commit from live positions, never mid-run (frozen-targets lifecycle)', () => {
    useSimStore.getState().setPreset('crossing');
    const st = useSimStore.getState();
    // Rebuild (setPreset) is re-anchor point 1: ℓ⁰ = current graph edge lengths.
    expect(st.sobolevEll0).toEqual(edgeLengths(st.graph.vertices, st.graph.edges));

    // Perturb the live buffer, then play: ℓ⁰ must re-anchor to the CURRENT
    // geometry, not the preset.
    st.live[0][0] += 0.123;
    useSimStore.getState().setRunning(true);
    const ell0AtPlay = useSimStore.getState().sobolevEll0;
    expect(ell0AtPlay).toEqual(
        edgeLengths(useSimStore.getState().live, useSimStore.getState().graph.edges),
    );

    // Mid-run motion of the live buffer must NOT move the frozen target
    // (same array identity — nothing recomputed it).
    useSimStore.getState().live[1][1] += 0.5;
    expect(useSimStore.getState().sobolevEll0).toBe(ell0AtPlay);

    // Pause = vertex commit → re-anchor from the committed positions.
    useSimStore.getState().setRunning(false);
    const ell0AtCommit = useSimStore.getState().sobolevEll0;
    expect(ell0AtCommit).toEqual(
        edgeLengths(useSimStore.getState().live, useSimStore.getState().graph.edges),
    );
    expect(ell0AtCommit).not.toEqual(ell0AtPlay);
});

test('dispatchDescentStep: lengthMode builds the ConstraintSet — perEdge matches the explicit set, mode wins over the legacy boolean, legacy shape unchanged', () => {
    useSimStore.getState().setPreset('crossing');
    const st = useSimStore.getState();
    const vertices = st.graph.vertices;
    const edges = st.graph.edges;
    const disjointPairs = st.disjointPairs;
    const x0 = barycenterTarget(vertices, edges);
    const L0 = totalLength(vertices, edges);
    const ell0 = edgeLengths(vertices, edges);
    const base = {
        descentMode: 'sobolev' as const,
        vertices,
        edges,
        disjointPairs,
        mode: 'analytical' as const,
        stepSize: 0.001,
        x0,
        sobolevL0: L0,
        sobolevEll0: ell0,
    };
    const opts = { mode: 'analytical' as const };

    // perEdge: [barycenter, edgeLengths] — barycenter first (spec §3.2 row order).
    const perEdge = dispatchDescentStep({
        ...base,
        barycenterConstraint: true,
        lengthMode: 'perEdge',
    });
    const perEdgeRef = sobolevStepSet(
        vertices,
        edges,
        disjointPairs,
        [barycenterBlock(x0), edgeLengthsBlock(ell0)],
        opts,
    );
    expect(perEdge.energy).toBe(perEdgeRef.energy);
    expect(flatten(perEdge.vertices)).toEqual(flatten(perEdgeRef.vertices));

    // lengthMode 'total' is bit-identical to the M1 boolean path.
    const totalViaMode = dispatchDescentStep({ ...base, lengthMode: 'total' });
    const totalViaBool = dispatchDescentStep({ ...base, lengthConstraint: true });
    expect(totalViaMode.energy).toBe(totalViaBool.energy);
    expect(flatten(totalViaMode.vertices)).toEqual(flatten(totalViaBool.vertices));

    // lengthMode WINS over a contradictory legacy boolean.
    const modeWins = dispatchDescentStep({
        ...base,
        barycenterConstraint: true,
        lengthConstraint: true,
        lengthMode: 'none',
    });
    const noneRef = sobolevStepSet(vertices, edges, disjointPairs, [barycenterBlock(x0)], opts);
    expect(modeWins.energy).toBe(noneRef.energy);
    expect(flatten(modeWins.vertices)).toEqual(flatten(noneRef.vertices));

    // All toggle fields ABSENT (pre-M1 call shape): still bit-identical to the
    // legacy sobolevStep(x0) path (extra sobolevEll0 data alone must not
    // change the dispatch decision).
    const legacy = dispatchDescentStep(base);
    const legacyRef = sobolevStep(vertices, edges, disjointPairs, x0, opts);
    expect(legacy.energy).toBe(legacyRef.energy);
    expect(flatten(legacy.vertices)).toEqual(flatten(legacyRef.vertices));
});

// Solver-perf Task 6: projectionMode passthrough — dispatch('frozen') must
// match the explicit sobolevStepSet frozen call, dispatch with the field
// ABSENT must stay bit-identical to explicit 'reassemble' (the default is
// strictly opt-in; store default 'frozen' only enters via Viewer passthrough).
// @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 6)
test('dispatchDescentStep: projectionMode passthrough — frozen matches explicit set call, absent ≡ reassemble', () => {
    useSimStore.getState().setPreset('crossing');
    const st = useSimStore.getState();
    const vertices = st.graph.vertices;
    const edges = st.graph.edges;
    const disjointPairs = st.disjointPairs;
    const x0 = barycenterTarget(vertices, edges);
    const L0 = totalLength(vertices, edges);
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
    const set = [barycenterBlock(x0), totalLengthBlock(L0)];

    const frozenDispatch = dispatchDescentStep({ ...base, projectionMode: 'frozen' });
    const frozenRef = sobolevStepSet(vertices, edges, disjointPairs, set, {
        mode: 'analytical',
        projectionMode: 'frozen',
    });
    expect(frozenDispatch.energy).toBe(frozenRef.energy);
    expect(flatten(frozenDispatch.vertices)).toEqual(flatten(frozenRef.vertices));

    const absent = dispatchDescentStep(base);
    const reassembleDispatch = dispatchDescentStep({ ...base, projectionMode: 'reassemble' });
    expect(reassembleDispatch.energy).toBe(absent.energy);
    expect(flatten(reassembleDispatch.vertices)).toEqual(flatten(absent.vertices));

    // The store defaults to 'frozen' (reference-impl scheme) with a
    // no-side-effect setter.
    expect(useSimStore.getState().projectionMode).toBe('frozen');
    useSimStore.getState().setProjectionMode('reassemble');
    expect(useSimStore.getState().projectionMode).toBe('reassemble');
    useSimStore.getState().setProjectionMode('frozen');
});
