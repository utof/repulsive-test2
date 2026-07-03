import { expect, test } from 'bun:test';
import { sobolevStep, sobolevStepSet } from '../src/core/optimizer';
import { barycenterBlock, totalLength, totalLengthBlock } from '../src/core/sobolev/constraintSet';
import { barycenterTarget } from '../src/core/sobolev/constraints';
import { flatten } from '../src/core/sobolev/layout';
import { dispatchDescentStep, useSimStore } from '../src/store';

// Constraints-M1 store/dispatch coverage: frozen-L⁰ lifecycle (mirrors the
// sobolevX0 tests in test/optimizer-sobolev.test.ts — same three re-anchor
// points), constraint toggles (spec §4.2 + §9a all-constraint toggles), and
// the ConstraintSet-building dispatch.
// @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §3.5, §4.2, §9a

test('store: constraint toggles default ON; setters toggle and clear the stale converged flag', () => {
    useSimStore.getState().setPreset('crossing');
    const st = useSimStore.getState();
    // Defaults per spec §4.2 (+ §9a): both constraints enabled — sobolev runs
    // have an equilibrium and current behavior is preserved.
    expect(st.lengthConstraint).toBe(true);
    expect(st.barycenterConstraint).toBe(true);

    useSimStore.setState({ sobolevConverged: true });
    useSimStore.getState().setLengthConstraint(false);
    expect(useSimStore.getState().lengthConstraint).toBe(false);
    // A converged verdict is per-constraint-set: toggling invalidates it.
    expect(useSimStore.getState().sobolevConverged).toBe(false);

    useSimStore.setState({ sobolevConverged: true });
    useSimStore.getState().setBarycenterConstraint(false);
    expect(useSimStore.getState().barycenterConstraint).toBe(false);
    expect(useSimStore.getState().sobolevConverged).toBe(false);

    useSimStore.getState().setLengthConstraint(true);
    useSimStore.getState().setBarycenterConstraint(true);
});

test('store: L0 re-anchors on play and on commit from live positions, never mid-run (frozen-targets lifecycle)', () => {
    useSimStore.getState().setPreset('crossing');
    const st = useSimStore.getState();
    // Rebuild (setPreset) is re-anchor point 1: L0 = current graph length.
    expect(st.sobolevL0).toBe(totalLength(st.graph.vertices, st.graph.edges));

    // Perturb the live buffer (as if the user had run and paused elsewhere),
    // then play: L0 must re-anchor to the CURRENT geometry, not the preset.
    st.live[0][0] += 0.123;
    useSimStore.getState().setRunning(true);
    const l0AtPlay = useSimStore.getState().sobolevL0;
    expect(l0AtPlay).toBe(
        totalLength(useSimStore.getState().live, useSimStore.getState().graph.edges),
    );

    // Mid-run motion of the live buffer must NOT move the frozen target: only
    // play / commit / rebuild recompute it (spec §3.5).
    useSimStore.getState().live[1][1] += 0.5;
    expect(useSimStore.getState().sobolevL0).toBe(l0AtPlay);

    // Pause = vertex commit → re-anchor from the committed positions.
    useSimStore.getState().setRunning(false);
    const l0AtCommit = useSimStore.getState().sobolevL0;
    expect(l0AtCommit).toBe(
        totalLength(useSimStore.getState().live, useSimStore.getState().graph.edges),
    );
    expect(l0AtCommit).not.toBe(l0AtPlay);
});

test('dispatchDescentStep: toggles build the ConstraintSet — each combination matches the explicit sobolevStepSet', () => {
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
    };
    const opts = { mode: 'analytical' as const };

    // Both ON (store defaults): [barycenter, totalLength] — barycenter first
    // (spec §3.2 row order).
    const both = dispatchDescentStep({
        ...base,
        barycenterConstraint: true,
        lengthConstraint: true,
    });
    const bothRef = sobolevStepSet(
        vertices,
        edges,
        disjointPairs,
        [barycenterBlock(x0), totalLengthBlock(L0)],
        opts,
    );
    expect(both.energy).toBe(bothRef.energy);
    expect(flatten(both.vertices)).toEqual(flatten(bothRef.vertices));

    // Barycenter OFF, length ON: [totalLength] only (§9a: sets without the
    // barycenter block are valid).
    const lenOnly = dispatchDescentStep({
        ...base,
        barycenterConstraint: false,
        lengthConstraint: true,
    });
    const lenOnlyRef = sobolevStepSet(vertices, edges, disjointPairs, [totalLengthBlock(L0)], opts);
    expect(lenOnly.energy).toBe(lenOnlyRef.energy);
    expect(flatten(lenOnly.vertices)).toEqual(flatten(lenOnlyRef.vertices));

    // Both OFF: the EMPTY set (unconstrained Sobolev flow, §9a).
    const none = dispatchDescentStep({
        ...base,
        barycenterConstraint: false,
        lengthConstraint: false,
    });
    const noneRef = sobolevStepSet(vertices, edges, disjointPairs, [], opts);
    expect(none.energy).toBe(noneRef.energy);
    expect(flatten(none.vertices)).toEqual(flatten(noneRef.vertices));
    expect(none.accepted).toBe(true);

    // Toggle fields ABSENT (pre-M1 call shape): barycenter-only — bit-identical
    // to the legacy sobolevStep(x0) path, so pre-existing dispatch tests and
    // call sites are unaffected.
    const legacy = dispatchDescentStep(base);
    const legacyRef = sobolevStep(vertices, edges, disjointPairs, x0, opts);
    expect(legacy.energy).toBe(legacyRef.energy);
    expect(flatten(legacy.vertices)).toEqual(flatten(legacyRef.vertices));
});
