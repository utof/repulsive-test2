// src/gpu/driver.ts — the 'gpu' solver driver (Phase 1: GPU dE, CPU everything else).
// @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §1 (served configs), §2.6, §5 Phase 1
import type { WebGPURenderer } from 'three/webgpu';
import {
    buildStepArgs,
    type DescentStepOutcome,
    dispatchDescentStep,
    type StepArgsSource,
} from '../core/dispatch';
import { DEFAULTS } from '../core/optimizer';
import { penaltiesActive } from '../core/sobolev/penalties';
import type { Vec3 } from '../core/testConfigs';
import { GpuTangentPoint, REDUCE_MAX_EDGES } from './tangentPoint';
import { packTopology } from './topology';

/**
 * Spec §1: 'gpu' serves sobolev + analytical + penalties-off ONLY. Config-only
 * predicate (the Viewer routes a `false` step to the 'worker' path without
 * changing the selected driver); topology is a separate question, see
 * {@link gpuTopologySupported}. {@link GpuDriver.step} re-checks this itself —
 * see its TSDoc for why the Viewer's routing is not the only line of defense.
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §1
 */
export function gpuStepSupported(
    s: Pick<StepArgsSource, 'descentMode' | 'mode' | 'penalties'>,
): boolean {
    return s.descentMode === 'sobolev' && s.mode === 'analytical' && !penaltiesActive(s.penalties);
}

/**
 * Per-topology "can the kernel be built at all". MUST agree with the
 * `GpuTangentPoint` constructor's domain checks (`src/gpu/tangentPoint.ts`,
 * constructor): `1 ≤ edgeCount ≤ REDUCE_MAX_EDGES` (two-pass reduction bound,
 * `src/gpu/reduce.ts`) and at least one disjoint pair (a single edge, a 2-edge
 * path, a triangle, a star have none — ordinary user-edited graphs). These are
 * unsupported topologies, not device failures: {@link GpuDriver.step} routes
 * such a step to the CPU dE path without changing the selected driver (spec §1),
 * instead of letting the constructor throw and flip the driver (spec §2.6).
 * If the constructor's domain changes, change this too (test/gpu/driver.test.ts
 * cross-checks the two against each other).
 * @see src/gpu/tangentPoint.ts (GpuTangentPoint constructor domain checks)
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §1, §2.6
 */
export function gpuTopologySupported(edgeCount: number, pairCount: number): boolean {
    return edgeCount >= 1 && edgeCount <= REDUCE_MAX_EDGES && pairCount > 0;
}

/**
 * Non-finite GPU output (f32 overflow below gap≈4e-7, plan D2) must never reach
 * the solve: the driver falls back to the CPU dE for that step.
 * @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (D2)
 */
export function allFinite(v: Vec3[]): boolean {
    for (const p of v)
        if (!Number.isFinite(p[0]) || !Number.isFinite(p[1]) || !Number.isFinite(p[2]))
            return false;
    return true;
}

/**
 * What {@link GpuDriver.step} reads from the store: the shared step-assembly
 * view plus the topology cache key and the arrows toggle.
 * @see src/core/dispatch.ts (StepArgsSource, §D7)
 */
export type GpuStepSource = StepArgsSource & { graphVersion: number; showArrows: boolean };

/**
 * Per-step orchestration. Engine rebuilt on graphVersion change (mirrors the
 * worker's §D4 topology cache). Unsupported config or topology ⇒ this step runs
 * with CPU dE (`usedGpu=false`, driver unchanged). Non-finite dE ⇒ likewise
 * (`usedGpu=false`, Viewer warns). Thrown errors propagate ⇒ Viewer flips the
 * driver to 'worker' (§2.6). `timings.dE` is overwritten with the GPU wall time
 * (upload→compute→readback→Vec3[]; packing is per topology at construction and
 * is NOT in this number — D4's gate definition includes pack, the per-step one
 * does not) so Stats shows the honest phase cost; `timings.step` gains the same
 * interval so `step ⊇ dE` stays true.
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §1, §2.6, §5 Phase 1
 * @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (D2, D4, D5)
 */
export class GpuDriver {
    private engine: GpuTangentPoint | null = null;
    private topoVersion = -1;
    // Cached together with `topoVersion` so {@link gpuTopologySupported} has ONE
    // source of truth per topology: mixing a fresh `st.graph.edges.length` with a
    // cached pair count would disagree if `graph` were ever mutated without
    // bumping `graphVersion`. -1 = not yet computed.
    private edgeCount = -1;
    private pairCount = -1;
    // Executable half of the single-flight contract documented on `step`: set for
    // the whole body of `step`, cleared in its `finally`.
    private inFlight = false;
    constructor(private readonly renderer: WebGPURenderer) {}

    /**
     * Runs one descent step, GPU dE when the config AND topology are served.
     *
     * NOT re-entrant. The caller single-flights it — one outstanding `step` per
     * driver, the mirror of the Viewer's `inFlight` latch — and must drop the
     * outcome if `graphVersion` changed across the await (the store may have been
     * re-seeded meanwhile). `dispose()` mid-flight may reject the pending step
     * (its buffers are destroyed under the readback's `mapAsync`); the caller
     * swallows that rejection rather than flipping the driver. Overlapping calls
     * are rejected here rather than silently corrupting state: the graphVersion
     * rebuild below would otherwise dispose an engine whose readback is still
     * pending, and the FIRST call would reject on what was only a preset change.
     *
     * The config predicate is re-checked here even though the Viewer routes
     * unsupported configs to the worker — this is the last line of defense. The
     * core's `opts.dE ?? …` seam (`src/core/optimizer.ts`, `dE?: Vec3[]`) accepts
     * a supplied dE regardless of `mode`, so a finiteDiff / raw / penalty step
     * handed to this driver would otherwise be served the GPU ANALYTICAL dE
     * (silently wrong for finiteDiff; discarded but reported `usedGpu` for raw).
     * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §1, §2.6
     * @see src/core/optimizer.ts (`dE?: Vec3[]` — the Phase 1 seam TSDoc)
     */
    async step(
        st: GpuStepSource,
        energyBefore: number | undefined,
    ): Promise<{ outcome: DescentStepOutcome; usedGpu: boolean }> {
        if (this.inFlight) throw new Error('GpuDriver.step: overlapping call');
        this.inFlight = true;
        try {
            if (st.graphVersion !== this.topoVersion) {
                // New topology: drop the old engine, recount edges+pairs; the engine
                // itself is (re)built lazily below only if the topology is supported.
                this.engine?.dispose();
                this.engine = null;
                this.topoVersion = st.graphVersion;
                this.edgeCount = st.graph.edges.length;
                this.pairCount = st.disjointPairs.reduce((s, l) => s + l.length, 0);
            }
            // Unsupported config or topology (spec §1 routing, NOT a device failure —
            // spec §2.6): this step runs with CPU dE, driver unchanged, timings as the
            // core produced. Config half: see the `step` TSDoc (the `opts.dE ??` seam
            // would otherwise accept a GPU analytical dE for a finiteDiff/raw/penalty
            // step). Topology half mirrors the GpuTangentPoint constructor domain (see
            // its TSDoc) so an unbuildable topology never reaches the constructor.
            if (!gpuStepSupported(st) || !gpuTopologySupported(this.edgeCount, this.pairCount)) {
                const outcome = dispatchDescentStep({
                    ...buildStepArgs(st, energyBefore),
                    collectTimings: true,
                    collectField: st.showArrows,
                });
                return { outcome, usedGpu: false };
            }
            if (!this.engine) {
                // Kernel α/β/ε MUST equal the α/β/ε the CPU step uses: `dispatch.ts`
                // forwards none of them, so the core falls back to `DEFAULTS`
                // (src/core/optimizer.ts) — pass the SAME object explicitly instead of
                // leaning on the constructor's default. A store-level α/β would need a
                // new engine here.
                this.engine = new GpuTangentPoint(
                    this.renderer,
                    packTopology(st.live.length, st.graph.edges, st.disjointPairs),
                    DEFAULTS,
                );
            }
            // `buildStepArgs` passes `st.live` by REFERENCE (dispatch.ts:333) and PinControls.tsx:169-174 mutates
            // `live` in place across our `await` below — so copy ONCE here and hand the SAME copy to the kernel
            // and to the step; otherwise dE would be evaluated at different vertices than the step uses
            // (dE seam contract, Task 5 TSDoc). Do not "simplify" this to `args.vertices`.
            const vertices = st.live.map((v) => [v[0], v[1], v[2]] as Vec3);
            const args = { ...buildStepArgs(st, energyBefore), vertices };
            const { dE, wallMs } = await this.engine.gradient(vertices);
            // D2: non-finite readback (f32 overflow below gap≈4e-7) ⇒ CPU dE for this step.
            const usedGpu = allFinite(dE);
            const outcome = dispatchDescentStep({
                ...args,
                ...(usedGpu ? { dE } : {}),
                collectTimings: true,
                collectField: st.showArrows,
            });
            if (usedGpu && outcome.timings) {
                // With a supplied dE the core's 'dE' phase is ≈0; report the GPU wall time
                // (readback included, D4) and keep `step ⊇ dE`.
                outcome.timings.dE = { ms: wallMs, calls: 1 };
                const s = outcome.timings.step;
                if (s) outcome.timings.step = { ms: s.ms + wallMs, calls: s.calls };
            }
            return { outcome, usedGpu };
        } finally {
            this.inFlight = false;
        }
    }

    dispose(): void {
        this.engine?.dispose();
        this.engine = null;
        this.topoVersion = -1;
        this.edgeCount = -1;
        this.pairCount = -1;
    }
}
