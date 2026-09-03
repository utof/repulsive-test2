// src/gpu/selfTest.ts — boot gate: G2 through the production kernel, once per renderer/adapter.
// @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.6 (boot gates), §4 G2; PREC risk 1 (per-adapter)
import type { WebGPURenderer } from 'three/webgpu';
import { nearTouchPair } from '../core/fixtures';
import { DEFAULTS } from '../core/optimizer';
import {
    calculateDisjointPairs,
    calculateEnergy,
    gradientAnalytical,
} from '../core/tangentPointEnergy';
import type { Vec3 } from '../core/testConfigs';
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
 * Pure verdict for the GRADIENT leg of the boot gate: every component finite AND
 * the worst absolute component deviation, normalised by the largest CPU-f64
 * component, under the same T1 tolerance the energy leg uses (spec §3).
 *
 * Why scale-normalised rather than per-component relative: at `nearTouchPair`
 * scale individual components of the field are near-cancelling while the field
 * itself is O(1e6), so a per-component ratio would be dominated by the small
 * components' cancellation noise and would fail on a correct kernel. A length
 * mismatch is a hard fail — a dropped dispatch returns a differently shaped or
 * all-zero readback (@issue utof/repulsive-test2#25).
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.6, §3
 */
export function gradientVerdict(gpu: Vec3[], cpu64: Vec3[]): { ok: boolean; relErr: number } {
    if (gpu.length !== cpu64.length || gpu.length === 0) return { ok: false, relErr: Number.NaN };
    let scale = 0;
    let worst = 0;
    let finite = true;
    for (let i = 0; i < gpu.length; i++) {
        for (let k = 0; k < 3; k++) {
            const a = gpu[i][k];
            const b = cpu64[i][k];
            if (!Number.isFinite(a)) finite = false;
            if (Math.abs(b) > scale) scale = Math.abs(b);
            if (Math.abs(a - b) > worst) worst = Math.abs(a - b);
        }
    }
    const relErr = scale > 0 ? worst / scale : worst;
    return { ok: finite && Number.isFinite(relErr) && relErr < 1e-5, relErr };
}

/**
 * Runs the G2 self-test at `nearTouchPair(1e-6)` through the production kernel on
 * the Viewer's renderer and compares with the CPU f64 reference. BOTH entry points
 * are exercised: `energy()` (the G2 number) and `gradient()` — the gradient entry
 * (`tpGradient` + `tpVertexGather`) is the only kernel descent actually uses, so a
 * compile or binding failure there must fail the BOOT verdict rather than surface
 * later as a mid-run fallback. Never throws — any exception is a failed gate
 * (⇒ `gpuAvailable=false`, spec §2.6).
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.6, §4 G2
 * @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md — "Task 6: Store, boot self-test, driver module"
 */
export async function runGpuSelfTest(
    renderer: WebGPURenderer,
): Promise<{ ok: boolean; relErr: number; gradRelErr?: number; error?: string }> {
    let gpu: GpuTangentPoint | null = null;
    try {
        const { vertices, edges } = nearTouchPair(1e-6);
        const dp = calculateDisjointPairs(edges);
        const { alpha, beta, epsilon } = DEFAULTS;
        gpu = new GpuTangentPoint(renderer, packTopology(vertices.length, edges, dp));
        const { energy } = await gpu.energy(vertices);
        const e = selfTestVerdict(
            energy,
            calculateEnergy(vertices, edges, dp, alpha, beta, epsilon),
        );
        const { dE } = await gpu.gradient(vertices);
        const g = gradientVerdict(
            dE,
            gradientAnalytical(vertices, edges, dp, alpha, beta, epsilon),
        );
        return { ok: e.ok && g.ok, relErr: e.relErr, gradRelErr: g.relErr };
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
