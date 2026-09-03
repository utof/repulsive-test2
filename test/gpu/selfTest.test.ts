import { expect, test } from 'bun:test';
import { selfTestVerdict } from '../../src/gpu/selfTest';

// Pure verdict of the boot self-test (spec §2.6 boot gates; T1 tolerance spec §3).
// `runGpuSelfTest` needs a renderer and is exercised by the browser smoke (plan Task 8).
// @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.6, §3

test('selfTestVerdict: relErr < 1e-5 passes; NaN/inf/≥1e-5 fail', () => {
    expect(selfTestVerdict(1 + 5e-6, 1).ok).toBe(true);
    expect(selfTestVerdict(1.001, 1).ok).toBe(false);
    expect(selfTestVerdict(Number.NaN, 1).ok).toBe(false);
    expect(selfTestVerdict(Number.POSITIVE_INFINITY, 1).ok).toBe(false);
});
