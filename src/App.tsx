import { ControlPanel } from './ui/ControlPanel';
import { Stats } from './ui/Stats';

export function App() {
    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', padding: 20 }}>
            <h1 style={{ fontSize: 24, marginBottom: 10 }}>Repulsive Energy Gradient Descent</h1>
            <ControlPanel />
            <Stats />
            <div
                style={{ flex: 1, position: 'relative', border: '2px solid #333', borderRadius: 8 }}
            >
                {/* <Viewer/> mounts here in Task 6 */}
            </div>
            <p style={{ marginTop: 10, color: '#888', fontSize: 14 }}>
                Drag to rotate. Scroll to zoom. Arrows show negative gradient (descent direction).
            </p>
        </div>
    );
}
