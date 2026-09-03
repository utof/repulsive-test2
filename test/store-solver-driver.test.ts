import { expect, test } from 'bun:test';
import { useSimStore } from '../src/store';

// solverDriver toggle + default (worker-solver §D6/§T3). The field selects which
// driver runs dispatchDescentStep each frame: default 'worker' (off-main-thread),
// with a 'main' synchronous fallback the setter also drives on auto-fallback
// (§D6). The store owns only the field + setter; the fallback TRIGGER lives in
// the frame loop (Viewer) and can't be exercised without a DOM/Worker, so the
// store test covers the field default + the toggle, matching the store-test style.
// @see docs/superpowers/plans/2026-07-04-worker-solver.md §D6

test('store: solverDriver defaults to worker; setSolverDriver toggles to main and back', () => {
    // Default 'worker' — the whole milestone's point is smooth interaction by
    // default; 'main' is the opt-in/fallback A/B baseline (§D6).
    expect(useSimStore.getState().solverDriver).toBe('worker');

    // The auto-fallback path calls exactly this with 'main'.
    useSimStore.getState().setSolverDriver('main');
    expect(useSimStore.getState().solverDriver).toBe('main');

    useSimStore.getState().setSolverDriver('worker');
    expect(useSimStore.getState().solverDriver).toBe('worker');
});

// 'gpu' driver widening + the boot-gate flag (WebGPU Phase 1, spec §2.6): the
// store owns `gpuAvailable` (default false — the Viewer sets it after the boot
// self-test) and accepts 'gpu' in setSolverDriver; the default driver is unchanged.
// @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.6

test('store: gpuAvailable defaults false; setSolverDriver accepts gpu; default driver unchanged', () => {
    expect(useSimStore.getState().solverDriver).toBe('worker');
    expect(useSimStore.getState().gpuAvailable).toBe(false);
    useSimStore.getState().setGpuAvailable(true);
    expect(useSimStore.getState().gpuAvailable).toBe(true);
    useSimStore.getState().setSolverDriver('gpu');
    expect(useSimStore.getState().solverDriver).toBe('gpu');
    useSimStore.getState().setSolverDriver('worker');
    useSimStore.getState().setGpuAvailable(false);
});
