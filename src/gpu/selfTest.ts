// src/gpu/selfTest.ts — boot gate: G2 through the production kernel, once per renderer/adapter.
// @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.6 (boot gates), §4 G2; PREC risk 1 (per-adapter)
import type { WebGPURenderer } from 'three/webgpu';
import { nearTouchPair } from '../core/fixtures';
import { DEFAULTS } from '../core/optimizer';
import { calculateDisjointPairs, calculateEnergy } from '../core/tangentPointEnergy';
import { GpuTangentPoint } from './tangentPoint';
import { packTopology } from './topology';

/**
 * Pure verdict: T1 tolerance (spec §3); non-finite ⇒ fail (a NaN `relErr`
 * compares false against the bound, but `isFinite` is stated so the intent is
 * not lost if the comparison is ever rewritten).
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.6, §3
 */
export function selfTestVerdict(gpu: number, cpu64: number): { ok: boolean; relErr: number } {
    const relErr = Math.abs(gpu - cpu64) / Math.abs(cpu64);
    return { ok: Number.isFinite(relErr) && relErr < 1e-5, relErr };
}

/**
 * Runs the G2 self-test (energy at `nearTouchPair(1e-6)`) through the production
 * kernel on the Viewer's renderer and compares with the CPU f64 energy. Never
 * throws — any exception is a failed gate (⇒ `gpuAvailable=false`, spec §2.6).
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.6, §4 G2
 */
export async function runGpuSelfTest(
    renderer: WebGPURenderer,
): Promise<{ ok: boolean; relErr: number; error?: string }> {
    let gpu: GpuTangentPoint | null = null;
    try {
        const { vertices, edges } = nearTouchPair(1e-6);
        const dp = calculateDisjointPairs(edges);
        gpu = new GpuTangentPoint(renderer, packTopology(vertices.length, edges, dp));
        const { energy } = await gpu.energy(vertices);
        return selfTestVerdict(
            energy,
            calculateEnergy(vertices, edges, dp, DEFAULTS.alpha, DEFAULTS.beta, DEFAULTS.epsilon),
        );
    } catch (e) {
        return { ok: false, relErr: Number.NaN, error: e instanceof Error ? e.message : String(e) };
    } finally {
        // Why: this function is documented "never throws", and a `finally` that throws
        // would REPLACE the verdict already computed above with an exception at the
        // boot gate. `dispose()` reaches into the renderer backend, which can throw
        // after a device loss — exactly the case where the verdict is already decided.
        try {
            gpu?.dispose();
        } catch {
            /* verdict already decided; a dispose failure cannot change it */
        }
    }
}
