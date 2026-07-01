import { expect, test } from 'bun:test';
import type { Vec3 } from '../src/core/testConfigs';
import { clampPolar, POLAR_LIMIT, project } from '../src/viewRotation';

// Regression test for the "dragging in 3D is unintuitive / vertical drag reverses
// when upside-down" bug.
//
// Root cause: rotationX (view pitch) was an UNBOUNDED world-frame Euler angle, so
// dragging past |rotationX| = PI/2 tipped the scene upside-down and silently
// reversed the vertical-drag direction (drag up -> scene moves DOWN). Re-grabbing
// could not fix it because the reversal is a pure function of the persistent pitch,
// not of any per-drag state.
//
// Fix: clampPolar() bounds the pitch to three.js OrbitControls' default polar range
// (phi in [0, PI]  <=>  rotationX in [-PI/2, PI/2]), making the upside-down regime
// unreachable. @see src/viewRotation.ts (clampPolar).

const rY = 0.5;
// An equatorial reference vertex (y = 0): for such a point the raw vertical-drag
// reversal happens exactly when |rotationX| > PI/2, which gives a clean invariant.
const EQ: Vec3 = [0.5, 0, 0.8];

const screenY = (rx: number) => project(EQ, 800, 600, rY, rx, 1)[1];

// "Drag up" nudges rotationX by ~-0.02 rad; the real handler clamps the result,
// so the probe must clamp too. Reversal := the point moved DOWN (larger screenY).
const dragUpReversesClamped = (rx: number) => screenY(clampPolar(rx - 0.02)) > screenY(rx) + 1e-9;

test('bug is real: a fully tipped-over (upside-down) view reverses vertical drag', () => {
    // Raw math, no clamp: at rotationX = PI the scene is upside-down and drag-up
    // moves points DOWN. This is exactly what the clamp must make unreachable.
    expect(screenY(Math.PI - 0.02) > screenY(Math.PI) + 1e-9).toBe(true);
    // ...while a normal orientation behaves correctly (drag up -> moves up).
    expect(screenY(0.3 - 0.02) > screenY(0.3) + 1e-9).toBe(false);
});

test('clampPolar bounds pitch to OrbitControls polar range [-PI/2, PI/2]', () => {
    expect(POLAR_LIMIT).toBeCloseTo(Math.PI / 2);
    expect(clampPolar(0.3)).toBe(0.3); // in range: unchanged
    expect(clampPolar(10)).toBe(POLAR_LIMIT); // over the top pole -> clamped
    expect(clampPolar(-10)).toBe(-POLAR_LIMIT);
});

test('no accumulated drag can escape the clamped range', () => {
    let rx = 0.3;
    for (let i = 0; i < 1000; i++) rx = clampPolar(rx + 0.1); // hammer "drag down"
    expect(rx).toBeLessThanOrEqual(POLAR_LIMIT + 1e-12);
    rx = 0.3;
    for (let i = 0; i < 1000; i++) rx = clampPolar(rx - 0.1); // hammer "drag up"
    expect(rx).toBeGreaterThanOrEqual(-POLAR_LIMIT - 1e-12);
});

test('within the clamped range, vertical drag never reverses (symptom fixed)', () => {
    for (let rx = -POLAR_LIMIT; rx <= POLAR_LIMIT; rx += 0.02) {
        expect(dragUpReversesClamped(clampPolar(rx))).toBe(false);
    }
});
