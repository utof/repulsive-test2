import { expect, test } from 'bun:test';
import type { Vec3 } from '../../src/core/testConfigs';
import { gradientVerdict, runGpuSelfTest, selfTestVerdict } from '../../src/gpu/selfTest';

// Pure verdict of the boot self-test (spec §2.6 boot gates; T1 tolerance spec §3).
// `runGpuSelfTest` needs a renderer and is exercised by the browser smoke (plan
// docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md —
// "Task 8: Driver smoke gate (T5 preview) + final verification + gate report").
// @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.6, §3

test('selfTestVerdict: relErr < 1e-5 passes; NaN/inf/≥1e-5 fail', () => {
    expect(selfTestVerdict(1 + 5e-6, 1).ok).toBe(true);
    expect(selfTestVerdict(1.001, 1).ok).toBe(false);
    expect(selfTestVerdict(Number.NaN, 1).ok).toBe(false);
    expect(selfTestVerdict(Number.POSITIVE_INFINITY, 1).ok).toBe(false);
});

// The gradient leg of the boot gate. `runGpuSelfTest` now compiles and runs BOTH
// entry points, so a gradient-side compile/binding failure fails the boot verdict
// instead of surfacing as a mid-run fallback.
// @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.6, §3

test('gradientVerdict: scale-normalised deviation under 1e-5 passes', () => {
    const cpu: Vec3[] = [
        [1e6, -2e6, 3e6],
        [0, 1e-3, -1e6],
    ];
    // worst absolute deviation 1 against scale 3e6 ⇒ relErr 3.3e-7
    const gpu: Vec3[] = [
        [1e6 + 1, -2e6, 3e6],
        [0, 1e-3, -1e6],
    ];
    const v = gradientVerdict(gpu, cpu);
    expect(v.ok).toBe(true);
    expect(v.relErr).toBeLessThan(1e-5);
});

test('gradientVerdict: a deviation above the tolerance fails', () => {
    const cpu: Vec3[] = [[1, 0, 0]];
    expect(gradientVerdict([[1.001, 0, 0]], cpu).ok).toBe(false);
});

test('gradientVerdict: any non-finite component fails', () => {
    const cpu: Vec3[] = [[1, 2, 3]];
    expect(gradientVerdict([[1, 2, Number.NaN]], cpu).ok).toBe(false);
    expect(gradientVerdict([[1, Number.POSITIVE_INFINITY, 3]], cpu).ok).toBe(false);
});

test('gradientVerdict: a length mismatch or empty readback fails', () => {
    // A dropped dispatch can return a differently shaped / all-zero readback
    // (@issue utof/repulsive-test2#25) — that must never read as a pass.
    expect(gradientVerdict([], [[1, 2, 3]]).ok).toBe(false);
    expect(gradientVerdict([[1, 2, 3]], []).ok).toBe(false);
    expect(gradientVerdict([], []).ok).toBe(false);
});

test('runGpuSelfTest never throws: a renderer that throws is a failed gate', async () => {
    // Same throwing-Proxy pattern as test/gpu/tangentPoint.test.ts: the first
    // renderer access (the compute submit inside `energy()`) throws, and the
    // `finally`'s `dispose()` throws too — the documented contract is that the
    // verdict survives both. @see src/gpu/selfTest.ts (runGpuSelfTest TSDoc)
    const renderer = new Proxy(
        {},
        {
            get() {
                throw new Error('renderer touched');
            },
        },
    );
    const r = await runGpuSelfTest(renderer as never);
    expect(r.ok).toBe(false);
    expect(r.error).toBe('renderer touched');
});
