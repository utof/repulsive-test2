import { expect, test } from 'bun:test';
import { sobolevStepSet } from '../src/core/optimizer';
import {
    barycenterBlock,
    edgeLengths,
    totalLength,
    totalLengthBlock,
} from '../src/core/sobolev/constraintSet';
import { barycenterTarget } from '../src/core/sobolev/constraints';
import { flatten } from '../src/core/sobolev/layout';
import { dispatchDescentStep, type PenaltyConfig, useSimStore } from '../src/store';

// Sobolev penalties store/UI milestone (plan §4 Task 5): the penalty-config
// setters + threading into dispatchDescentStep, the E₀-cache invalidation nonce
// (penaltyEpoch, plan §2.4), and the target-length animation schedule
// (advanceLengthSchedule, paper SelfAvoiding.tex line 760). Store-only coverage
// in the store-pins.test.ts style (the frame-loop wiring in Viewer.tsx consumes
// penaltyEpoch / calls advanceLengthSchedule on accepted steps — not unit-
// tested here).
// NOTE: the store is a module-level singleton SHARED across test files (see
// store-pins.test.ts). Penalties/growth/projection/stepSize/lengthMode do NOT
// reset on setPreset, so every test restores them via resetSobolevConfig().
// @see docs/superpowers/plans/2026-07-03-sobolev-penalties.md §2.4, §4 Task 5
// @see local_files/repulsive_orig_paper/SelfAvoiding.tex (line 760 target-length animation, 769 H^s remark)

// Restore every knob this file mutates back to the store defaults so later
// files start from a clean singleton.
function resetSobolevConfig() {
    const s = useSimStore.getState();
    s.setPenaltyTotalLength(0);
    s.setPenaltyLengthDiff(0);
    s.setPenaltyFieldWeight(0);
    s.setPenaltyFieldX([1, 0, 0]);
    s.setLengthGrowthRate(1);
    s.setLengthMode('total');
    s.setProjectionMode('frozen');
    s.setStepSize(0.001);
    s.setPreset('crossing');
}

test('dispatchDescentStep: penalty config threads verbatim into sobolevStepSet opts.penalties (bit-identical), and is not silently dropped', () => {
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
    const penalties: PenaltyConfig = {
        totalLength: 0.5,
        lengthDiff: 0.25,
        field: { weight: 0.1, X: [0, 0, 1] },
    };

    // dispatch(withPenalty) ≡ an explicit sobolevStepSet with the SAME penalties
    // in opts — proves the config reaches the step verbatim (pins-test pattern).
    const withPen = dispatchDescentStep({ ...base, penalties });
    const withPenRef = sobolevStepSet(vertices, edges, disjointPairs, set, {
        mode: 'analytical',
        penalties,
    });
    expect(withPen.energy).toBe(withPenRef.energy);
    expect(flatten(withPen.vertices)).toEqual(flatten(withPenRef.vertices));

    // The penalty is LIVE, not silently dropped: the total objective differs
    // from the penalty-free step.
    const noPenRef = sobolevStepSet(vertices, edges, disjointPairs, set, { mode: 'analytical' });
    expect(withPen.energy).not.toBe(noPenRef.energy);

    resetSobolevConfig();
});

test('dispatchDescentStep: all-off penalty config ≡ absent ≡ penalty-free path (bit-identical default behavior)', () => {
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

    // The store's default all-off config (every weight 0) must be inert.
    const allOff: PenaltyConfig = {
        totalLength: 0,
        lengthDiff: 0,
        field: { weight: 0, X: [1, 0, 0] },
    };
    const withDefault = dispatchDescentStep({ ...base, penalties: allOff });
    const absent = dispatchDescentStep({ ...base });
    const ref = sobolevStepSet(vertices, edges, disjointPairs, set, { mode: 'analytical' });

    expect(withDefault.energy).toBe(absent.energy);
    expect(withDefault.energy).toBe(ref.energy);
    expect(flatten(withDefault.vertices)).toEqual(flatten(absent.vertices));
    expect(flatten(withDefault.vertices)).toEqual(flatten(ref.vertices));

    resetSobolevConfig();
});

test('penaltyEpoch: every penalty setter bumps it (E₀ invalidation) and clears sobolevConverged; unrelated changes leave it untouched', () => {
    const g = useSimStore.getState;
    g().setPreset('crossing');

    // Each penalty setter bumps the E₀-invalidation nonce and writes its knob.
    let e = g().penaltyEpoch;
    g().setPenaltyTotalLength(0.5);
    expect(g().penaltyEpoch).toBe(e + 1);
    expect(g().penalties.totalLength).toBe(0.5);

    e = g().penaltyEpoch;
    g().setPenaltyLengthDiff(0.25);
    expect(g().penaltyEpoch).toBe(e + 1);
    expect(g().penalties.lengthDiff).toBe(0.25);

    e = g().penaltyEpoch;
    g().setPenaltyFieldWeight(0.1);
    expect(g().penaltyEpoch).toBe(e + 1);
    expect(g().penalties.field?.weight).toBe(0.1);

    e = g().penaltyEpoch;
    g().setPenaltyFieldX([0, 1, 0]);
    expect(g().penaltyEpoch).toBe(e + 1);
    expect(g().penalties.field?.X).toEqual([0, 1, 0]);

    // A penalty change clears the per-config converged verdict but KEEPS
    // sobolevStats (the last step actually taken), like the constraint toggles.
    useSimStore.setState({
        sobolevConverged: true,
        sobolevStats: { tau: 1, residual: 0, gradientL2Norm: 0, projectionIterations: 1 },
    });
    g().setPenaltyTotalLength(0.7);
    expect(g().sobolevConverged).toBe(false);
    expect(g().sobolevStats).not.toBeNull();

    // Unrelated changes must NOT bump the epoch — the E₀ chain stays valid
    // across them (step size, projection strategy, pins, length mode, growth
    // rate, and a schedule advance are all objective-invariant, plan §2.4).
    e = g().penaltyEpoch;
    g().setStepSize(0.002);
    g().setProjectionMode('reassemble');
    g().addPin(0);
    g().setLengthMode('perEdge');
    g().setLengthGrowthRate(1.05);
    g().advanceLengthSchedule();
    expect(g().penaltyEpoch).toBe(e);

    resetSobolevConfig();
});

test('advanceLengthSchedule: scales frozen targets from the SCHEDULE not the geometry; per length-mode; rate 1.0 no-op; setLengthGrowthRate clamps; no E₀ bump', () => {
    const g = useSimStore.getState;
    g().setPreset('crossing');
    g().setLengthMode('total');

    // Seed a KNOWN schedule value ≠ the live geometry, then perturb live: the
    // advance must scale the STORED target, never re-read totalLength(live).
    useSimStore.setState({ sobolevL0: 10 });
    g().live[0][0] += 3; // move geometry so totalLength(live) ≠ 10
    g().setLengthGrowthRate(1.05);
    g().advanceLengthSchedule();
    expect(g().sobolevL0).toBe(10 * 1.05);

    // Follows the schedule across steps: a second advance compounds (rate²),
    // from the previously-stored target — not from geometry.
    g().advanceLengthSchedule();
    expect(g().sobolevL0).toBe(10 * 1.05 * 1.05);

    // rate 1.0 is a strict no-op (off).
    g().setLengthGrowthRate(1);
    const held = g().sobolevL0;
    g().advanceLengthSchedule();
    expect(g().sobolevL0).toBe(held);

    // Per-edge mode scales EVERY ℓ⁰_I from the schedule (same op as l => l*rate).
    g().setLengthMode('perEdge');
    useSimStore.setState({ sobolevEll0: [1, 2, 3, 4] });
    g().setLengthGrowthRate(1.1);
    g().advanceLengthSchedule();
    expect(g().sobolevEll0).toEqual([1 * 1.1, 2 * 1.1, 3 * 1.1, 4 * 1.1]);

    // lengthMode 'none' → no length target to animate (both arrays untouched,
    // same identity).
    g().setLengthMode('none');
    const l0Ref = g().sobolevL0;
    const ellRef = g().sobolevEll0;
    g().advanceLengthSchedule();
    expect(g().sobolevL0).toBe(l0Ref);
    expect(g().sobolevEll0).toBe(ellRef);

    // setLengthGrowthRate clamps to [0.9, 1.1] (strictly positive, gentle).
    g().setLengthGrowthRate(5);
    expect(g().lengthGrowthRate).toBe(1.1);
    g().setLengthGrowthRate(0);
    expect(g().lengthGrowthRate).toBe(0.9);
    g().setLengthGrowthRate(-2);
    expect(g().lengthGrowthRate).toBe(0.9);

    // The schedule advance does NOT touch the E₀ cache — constraint targets are
    // not in the objective (plan §2.4).
    g().setLengthMode('total');
    g().setLengthGrowthRate(1.05);
    const e = g().penaltyEpoch;
    g().advanceLengthSchedule();
    expect(g().penaltyEpoch).toBe(e);

    resetSobolevConfig();
});
