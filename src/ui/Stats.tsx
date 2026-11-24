import { totalLength } from '../core/sobolev/constraintSet';
import type { SobolevStepTimings } from '../core/sobolev/phaseTimings';
import { useSimStore } from '../store';

// Curated (label, key) subset shown on the timings line, in this order; the
// leading `step` becomes the `Δt …ms —` prefix. `×n` marks calls>1 (single
// calls print no ×); values are `.toFixed(1)` ms; absent keys are omitted.
// @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 3 — UI format)
const TIMING_LINE: Array<[keyof SobolevStepTimings, string]> = [
    ['dE', 'dE'],
    ['assembleA', 'A'],
    ['saddle', 'saddle'],
    ['energy', 'E'],
    ['projection', 'proj'],
    ['lineSearch', 'LS'],
];

function formatStepTimings(t: SobolevStepTimings): string {
    const parts: string[] = [];
    for (const [key, label] of TIMING_LINE) {
        const s = t[key];
        if (!s) continue;
        parts.push(`${label} ${s.ms.toFixed(1)}${s.calls > 1 ? `×${s.calls}` : ''}`);
    }
    const prefix = t.step ? `Δt ${t.step.ms.toFixed(1)}ms — ` : '';
    return `${prefix}${parts.join(' · ')}`;
}

export function Stats() {
    const step = useSimStore((s) => s.step);
    const energy = useSimStore((s) => s.energy);
    const zoom = useSimStore((s) => s.zoom);
    const mode = useSimStore((s) => s.mode);
    const descentMode = useSimStore((s) => s.descentMode);
    const sobolevStats = useSimStore((s) => s.sobolevStats);
    const sobolevConverged = useSimStore((s) => s.sobolevConverged);
    const sobolevTimings = useSimStore((s) => s.sobolevTimings);
    const sobolevL0 = useSimStore((s) => s.sobolevL0);
    const prePassAlpha = useSimStore((s) => s.prePassAlpha);
    const untangleInfo = useSimStore((s) => s.untangleInfo);
    const graphFlowInfo = useSimStore((s) => s.graphFlowInfo);
    const vertices = useSimStore((s) => s.graph.vertices.length);
    const edges = useSimStore((s) => s.graph.edges.length);
    // Current total length from the LIVE buffer. `live` is mutated in place and
    // deliberately never React-subscribed (its identity never changes — see the
    // store's `live` comment), so read it via getState() at render time: the
    // throttled step/energy publishes above are what trigger the re-renders that
    // refresh this readout while running.
    // @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §4.3
    const liveNow = useSimStore.getState();
    const L = totalLength(liveNow.live, liveNow.graph.edges);

    return (
        <div style={{ marginBottom: 10, fontFamily: 'monospace' }}>
            <span style={{ marginRight: 20 }}>Step: {step}</span>
            <span style={{ marginRight: 20 }}>Energy: {energy.toFixed(6)}</span>
            <span style={{ marginRight: 20 }}>Vertices: {vertices}</span>
            <span style={{ marginRight: 20 }}>Edges: {edges}</span>
            <span style={{ marginRight: 20 }}>Zoom: {zoom.toFixed(2)}x</span>
            {/* Why: these hexes must stay in sync with the 3D gradient-cone colors in
                scene/GradientArrows.tsx (analytical=green, finiteDiff=orange); editing one
                without the other silently desyncs the label from the visual. */}
            <span style={{ marginRight: 20, color: mode === 'analytical' ? '#00ff88' : '#ffaa00' }}>
                Gradient: {mode === 'analytical' ? 'Analytical (green)' : 'Finite Diff (orange)'}
            </span>
            {/* Force-layout pre-pass in progress: its temperature falls from 1 to 0.001, at
                which point the descent takes over. @see src/core/forceLayout.ts */}
            {prePassAlpha !== null && (
                <span style={{ marginRight: 20, color: '#66ccff' }}>
                    pre-pass: force layout, cooling {prePassAlpha.toFixed(3)}
                </span>
            )}
            {/* Untangle pass: progress while it runs (the flow's own point count falls as the
                knot tightens), then how it ended. A skipped pass and a failed knot-type audit
                are shown as limits, not hidden. @see src/core/untanglePass.ts */}
            {untangleInfo?.phase === 'running' && (
                <span style={{ marginRight: 20, color: '#66ccff' }}>
                    untangle pass: step {untangleInfo.steps}, {untangleInfo.nodes} points
                </span>
            )}
            {untangleInfo?.phase === 'done' && (
                <span
                    style={{
                        marginRight: 20,
                        color: untangleInfo.audit === 'failed' ? '#ff6666' : '#66ccff',
                    }}
                >
                    untangled in {untangleInfo.steps} steps ({untangleInfo.stopped}); knot type{' '}
                    {untangleInfo.audit === 'kept'
                        ? 'kept'
                        : untangleInfo.audit === 'unverified'
                          ? 'not verified'
                          : 'CHANGED — input restored'}
                </span>
            )}
            {/* Local graph flow: how far the nodes still move per step (in units of the
                clearance), how many are held back by their crossing-free limit, and the cost of
                a step; or why it is not running. @see src/core/medialSphereGraph.ts */}
            {graphFlowInfo?.phase === 'running' && (
                <span style={{ marginRight: 20, color: '#66ccff' }}>
                    graph flow: move {graphFlowInfo.movePerNode.toExponential(1)}, limited{' '}
                    {graphFlowInfo.cappedNodes}, {graphFlowInfo.msPerStep.toFixed(0)} ms/step
                    {graphFlowInfo.awake === 0 ? ', at rest' : `, awake ${graphFlowInfo.awake}`}
                    {graphFlowInfo.crossings > 0 && `, crossings ${graphFlowInfo.crossings}`}
                    {graphFlowInfo.growing !== null &&
                        `, group growing back ${(100 * graphFlowInfo.growing).toFixed(0)}%`}
                    {graphFlowInfo.carrying && ', carrying a group'}
                </span>
            )}
            {graphFlowInfo?.phase === 'skipped' && (
                <span style={{ marginRight: 20, color: '#ffaa00' }}>
                    graph flow not used: {graphFlowInfo.reason}
                </span>
            )}
            {untangleInfo?.phase === 'skipped' && (
                <span style={{ marginRight: 20, color: '#ffaa00' }}>
                    untangle pass skipped: {untangleInfo.reason}
                </span>
            )}
            {/* Sobolev-only diagnostics of the last step: τ, saddle residual, ‖g̃‖_L²ₕ,
                projection iterations, and the terminal outcome (converged / reject reason).
                @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §C (steps 5, 10) */}
            {/* Sobolev-only length readout: current L and drift |L − L⁰|/L⁰ against
                the frozen target (shown even before the first step, when
                sobolevStats is still null). Drift ≤ ~1e-4 with "Fix length" on is
                projection tolerance (reference backproj_threshold), not error.
                @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §4.3, §3.5 */}
            {descentMode === 'sobolev' && (
                <>
                    <span style={{ marginRight: 20 }}>L: {L.toFixed(6)}</span>
                    <span style={{ marginRight: 20 }}>
                        L drift: {(Math.abs(L - sobolevL0) / sobolevL0).toExponential(2)}
                    </span>
                </>
            )}
            {descentMode === 'sobolev' && sobolevStats && (
                <>
                    <span style={{ marginRight: 20 }}>τ: {sobolevStats.tau.toExponential(2)}</span>
                    <span style={{ marginRight: 20 }}>
                        residual: {sobolevStats.residual.toExponential(2)}
                    </span>
                    <span style={{ marginRight: 20 }}>
                        ‖g̃‖: {sobolevStats.gradientL2Norm.toExponential(2)}
                    </span>
                    <span style={{ marginRight: 20 }}>
                        proj iters: {sobolevStats.projectionIterations ?? '—'}
                    </span>
                    {sobolevStats.reason === 'accuracy_floor' ? (
                        // Not an error: the hierarchical route stopped within a decade of the
                        // convergence tolerance, where its approximate energy can no longer
                        // tell a descent step from noise. @see src/core/sobolevRoute.ts
                        <span style={{ marginRight: 20, color: '#ffcc66' }}>
                            stopped: accuracy limit of the hierarchical route
                        </span>
                    ) : (
                        sobolevStats.reason && (
                            <span style={{ marginRight: 20, color: '#ff4444' }}>
                                rejected: {sobolevStats.reason}
                            </span>
                        )
                    )}
                </>
            )}
            {descentMode === 'sobolev' && sobolevConverged && (
                <span style={{ color: '#00ff88' }}>converged</span>
            )}
            {/* Second line: per-phase step timings (sobolev mode only, when
                collected). Δt is the whole-step wall clock; the rest are the
                curated hot phases. @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 3) */}
            {descentMode === 'sobolev' && sobolevTimings?.step && (
                <div style={{ marginTop: 4, color: '#88aabb' }}>
                    {formatStepTimings(sobolevTimings)}
                </div>
            )}
        </div>
    );
}
