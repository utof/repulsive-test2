import { useSimStore } from '../store';

export function Stats() {
    const step = useSimStore((s) => s.step);
    const energy = useSimStore((s) => s.energy);
    const zoom = useSimStore((s) => s.zoom);
    const mode = useSimStore((s) => s.mode);
    const descentMode = useSimStore((s) => s.descentMode);
    const sobolevStats = useSimStore((s) => s.sobolevStats);
    const sobolevConverged = useSimStore((s) => s.sobolevConverged);
    const vertices = useSimStore((s) => s.graph.vertices.length);
    const edges = useSimStore((s) => s.graph.edges.length);

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
            {/* Sobolev-only diagnostics of the last step: τ, saddle residual, ‖g̃‖_L²ₕ,
                projection iterations, and the terminal outcome (converged / reject reason).
                @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §C (steps 5, 10) */}
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
                    {sobolevStats.reason && (
                        <span style={{ marginRight: 20, color: '#ff4444' }}>
                            rejected: {sobolevStats.reason}
                        </span>
                    )}
                </>
            )}
            {descentMode === 'sobolev' && sobolevConverged && (
                <span style={{ color: '#00ff88' }}>converged</span>
            )}
        </div>
    );
}
