import { useSimStore } from '../store';

export function Stats() {
    const step = useSimStore((s) => s.step);
    const energy = useSimStore((s) => s.energy);
    const zoom = useSimStore((s) => s.zoom);
    const mode = useSimStore((s) => s.mode);
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
            <span style={{ color: mode === 'analytical' ? '#00ff88' : '#ffaa00' }}>
                Gradient: {mode === 'analytical' ? 'Analytical (green)' : 'Finite Diff (orange)'}
            </span>
        </div>
    );
}
