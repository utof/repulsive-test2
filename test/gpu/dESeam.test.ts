import { expect, test } from 'bun:test';
import { type DispatchDescentStepArgs, dispatchDescentStep } from '../../src/core/dispatch';
import { trefoil } from '../../src/core/fixtures';
import { DEFAULTS } from '../../src/core/optimizer';
import { calculateDisjointPairs, gradientAnalytical } from '../../src/core/tangentPointEnergy';
import type { Vec3 } from '../../src/core/testConfigs';

function args(): DispatchDescentStepArgs {
    const { vertices, edges } = trefoil(24);
    const x0: Vec3 = [0, 0, 0];
    for (const v of vertices) {
        x0[0] += v[0] / 24;
        x0[1] += v[1] / 24;
        x0[2] += v[2] / 24;
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

test('dE seam: supplying the CPU analytical dE is toEqual-identical to omitting it', () => {
    const a = args();
    const dE = gradientAnalytical(
        a.vertices,
        a.edges,
        a.disjointPairs,
        DEFAULTS.alpha,
        DEFAULTS.beta,
        DEFAULTS.epsilon,
    );
    // Guard the comparison's meaning: if a future fixture change made this step
    // rejected/converged, toEqual would compare two echoes and prove nothing.
    expect(dispatchDescentStep(a).accepted).toBe(true);
    expect(dispatchDescentStep({ ...args(), dE })).toEqual(dispatchDescentStep(a));
});

test('dE seam: a perturbed dE reaches the saddle RHS (1.5x dE ⇒ different energy)', () => {
    const a = args();
    const dE = gradientAnalytical(
        a.vertices,
        a.edges,
        a.disjointPairs,
        DEFAULTS.alpha,
        DEFAULTS.beta,
        DEFAULTS.epsilon,
    );
    const baseline = dispatchDescentStep(a);
    const scaled = dispatchDescentStep({
        ...args(),
        dE: dE.map((v) => v.map((c) => 1.5 * c) as Vec3),
    });
    // The STRONG probe that dE feeds the SOLVE: g̃ = Ā⁻¹dE is linear in dE, so a
    // 1.5x dE must give exactly 1.5x descentField. `energy` alone is a weak probe
    // here — the line search normalizes g̃ in the L²ₕ norm (sobolev/lineSearch.ts,
    // "normalize g̃ in the L²ₕ norm"), so a UNIFORM scale cancels and only ~3 ULP
    // of rounding survives. Both are asserted; only the ratio has real margin.
    const b = baseline.descentField as Vec3[];
    const t = scaled.descentField as Vec3[];
    let checked = 0;
    for (let i = 0; i < b.length; i++) {
        for (let c = 0; c < 3; c++) {
            // Skip components at cancellation noise (~1e-16) where the ratio is junk.
            if (Math.abs(b[i][c]) < 1e-12) continue;
            expect(t[i][c] / b[i][c]).toBeCloseTo(1.5, 10);
            checked++;
        }
    }
    expect(checked).toBeGreaterThan(0);
    expect(scaled.energy).not.toBe(baseline.energy);
});

test('dE seam: a supplied dE is actually used (zero field ⇒ converged, vertices echoed)', () => {
    const a = args();
    const r = dispatchDescentStep({ ...a, dE: a.vertices.map(() => [0, 0, 0] as Vec3) });
    expect(r.converged).toBe(true);
    expect(r.vertices).toEqual(a.vertices);
});

test('dE seam: length mismatch throws', () => {
    expect(() => dispatchDescentStep({ ...args(), dE: [[0, 0, 0]] })).toThrow('dE length mismatch');
});
