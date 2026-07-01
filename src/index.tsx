import type React from 'react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
    calculateDisjointPairs,
    calculateEnergy,
    gradientAnalytical,
    gradientFiniteDiff,
    norm,
} from './tangentPointEnergy';
import { type GraphState, testConfigs, type Vec3 } from './testConfigs';
import { clampPolar, project, projectArrow } from './viewRotation';

// LocalStorage keys
const STORAGE_KEY = 'repulsive-test-config';

function loadSavedConfig(): { testId: string; params: Record<string, number> } {
    try {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) return JSON.parse(saved);
    } catch {}
    return { testId: 'crossing', params: {} };
}

function saveConfig(testId: string, params: Record<string, number>) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ testId, params }));
}

// View math (rotate3D / project / projectArrow) and the OrbitControls-style pitch
// clamp live in ./viewRotation so they are unit-testable and easy to swap for
// react-three-fiber later. @see src/viewRotation.ts

// --- Gradient-arrow display knobs -------------------------------------------
// Arrows are the true perspective projection of the (log-compressed) 3D gradient
// (see projectArrow), so they foreshorten correctly instead of whipping around.
// These knobs control size + visibility. @see src/viewRotation.ts (projectArrow)
const ARROW_SCALE = 0.2; // world-units of arrow per log-unit of |gradient| (overall size)
const MIN_ARROW_LEN = 14; // px: floor so small / foreshortened arrows stay visible
const MAX_ARROW_LEN = 90; // px: cap so huge (near-intersection) gradients don't shoot off-canvas
const ARROW_WIDTH = 2.5; // px: arrow line thickness
const ARROW_HEAD = 8; // px: max arrowhead size

function App() {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    // Load saved config
    const savedConfig = loadSavedConfig();
    const initialTest = testConfigs.find((t) => t.id === savedConfig.testId) || testConfigs[0];
    const initialParams: Record<string, number> = {};
    if (initialTest.params) {
        for (const p of initialTest.params) {
            initialParams[p.name] = savedConfig.params[p.name] ?? p.default;
        }
    }

    const [selectedTestId, setSelectedTestId] = useState(initialTest.id);
    const [testParams, setTestParams] = useState<Record<string, number>>(initialParams);
    const [graph, setGraph] = useState<GraphState>(() => initialTest.generate(initialParams));
    const [running, setRunning] = useState(false);
    const [mode, setMode] = useState<'analytical' | 'finiteDiff'>('analytical');
    const [step, setStep] = useState(0);
    const [energy, setEnergy] = useState(0);
    const [rotationY, setRotationY] = useState(0.5);
    const [rotationX, setRotationX] = useState(0.3);
    const [stepSize, setStepSize] = useState(0.001);
    const [zoom, setZoom] = useState(1);

    const disjointPairs = useRef(calculateDisjointPairs(graph.edges));
    const animationRef = useRef<number>(0);

    const selectedTest = testConfigs.find((t) => t.id === selectedTestId) || testConfigs[0];

    // Save config when test or params change
    useEffect(() => {
        saveConfig(selectedTestId, testParams);
    }, [selectedTestId, testParams]);

    // Handle test change
    const handleTestChange = (newTestId: string) => {
        setRunning(false);
        setSelectedTestId(newTestId);
        const test = testConfigs.find((t) => t.id === newTestId)!;

        // Initialize params with defaults
        const newParams: Record<string, number> = {};
        if (test.params) {
            for (const p of test.params) {
                newParams[p.name] = p.default;
            }
        }
        setTestParams(newParams);

        const newGraph = test.generate(newParams);
        setGraph(newGraph);
        disjointPairs.current = calculateDisjointPairs(newGraph.edges);
        setStep(0);
    };

    // Handle param change
    const handleParamChange = (name: string, value: number) => {
        const newParams = { ...testParams, [name]: value };
        setTestParams(newParams);
    };

    // Regenerate graph with current params
    const regenerate = () => {
        setRunning(false);
        const newGraph = selectedTest.generate(testParams);
        setGraph(newGraph);
        disjointPairs.current = calculateDisjointPairs(newGraph.edges);
        setStep(0);
    };

    const alpha = 3;
    const beta = 6;
    const epsilon = 1e-10;
    const h = 1e-6;

    // Draw function
    const draw = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const width = canvas.width;
        const height = canvas.height;

        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, width, height);

        // Draw edges
        ctx.strokeStyle = '#4a9eff';
        ctx.lineWidth = 3;
        for (const [v1, v2] of graph.edges) {
            const [x1, y1] = project(graph.vertices[v1], width, height, rotationY, rotationX, zoom);
            const [x2, y2] = project(graph.vertices[v2], width, height, rotationY, rotationX, zoom);
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
        }

        // Draw vertices
        for (let i = 0; i < graph.vertices.length; i++) {
            const [x, y] = project(graph.vertices[i], width, height, rotationY, rotationX, zoom);
            ctx.beginPath();
            ctx.arc(x, y, 8, 0, Math.PI * 2);
            ctx.fillStyle = '#ff6b6b';
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        // Draw gradient vectors if not running
        if (!running) {
            const grad =
                mode === 'analytical'
                    ? gradientAnalytical(
                          graph.vertices,
                          graph.edges,
                          disjointPairs.current,
                          alpha,
                          beta,
                          epsilon,
                      )
                    : gradientFiniteDiff(
                          graph.vertices,
                          graph.edges,
                          disjointPairs.current,
                          alpha,
                          beta,
                          epsilon,
                          h,
                      );

            ctx.strokeStyle = mode === 'analytical' ? '#00ff88' : '#ffaa00';
            ctx.lineWidth = ARROW_WIDTH;

            for (let i = 0; i < graph.vertices.length; i++) {
                const gradVec = grad[i];
                const gradNorm = norm(gradVec);
                if (gradNorm <= 1e-6) continue;

                // Negative gradient = descent direction. Draw it as the true perspective
                // projection of the 3D vector so it foreshortens and keeps a consistent
                // direction across views. @see src/viewRotation.ts (projectArrow)
                const negGrad: Vec3 = [-gradVec[0], -gradVec[1], -gradVec[2]];
                const { baseX, baseY, tipX, tipY, len } = projectArrow(
                    graph.vertices[i],
                    negGrad,
                    gradNorm,
                    width,
                    height,
                    rotationY,
                    rotationX,
                    zoom,
                    ARROW_SCALE,
                    MIN_ARROW_LEN,
                    MAX_ARROW_LEN,
                );

                ctx.beginPath();
                ctx.moveTo(baseX, baseY);
                ctx.lineTo(tipX, tipY);
                ctx.stroke();

                // Arrow head
                const angle = Math.atan2(tipY - baseY, tipX - baseX);
                const headSize = Math.min(ARROW_HEAD, len * 0.4);
                ctx.beginPath();
                ctx.moveTo(tipX, tipY);
                ctx.lineTo(
                    tipX - headSize * Math.cos(angle - 0.4),
                    tipY - headSize * Math.sin(angle - 0.4),
                );
                ctx.moveTo(tipX, tipY);
                ctx.lineTo(
                    tipX - headSize * Math.cos(angle + 0.4),
                    tipY - headSize * Math.sin(angle + 0.4),
                );
                ctx.stroke();
            }
        }
    }, [graph, rotationY, rotationX, zoom, mode, running]);

    // Animation step
    const animate = useCallback(() => {
        if (!running) return;

        const grad =
            mode === 'analytical'
                ? gradientAnalytical(
                      graph.vertices,
                      graph.edges,
                      disjointPairs.current,
                      alpha,
                      beta,
                      epsilon,
                  )
                : gradientFiniteDiff(
                      graph.vertices,
                      graph.edges,
                      disjointPairs.current,
                      alpha,
                      beta,
                      epsilon,
                      h,
                  );

        const newVertices: Vec3[] = graph.vertices.map((v, i) => [
            v[0] - stepSize * grad[i][0],
            v[1] - stepSize * grad[i][1],
            v[2] - stepSize * grad[i][2],
        ]);

        const newEnergy = calculateEnergy(
            newVertices,
            graph.edges,
            disjointPairs.current,
            alpha,
            beta,
            epsilon,
        );

        setGraph({ ...graph, vertices: newVertices });
        setEnergy(newEnergy);
        setStep((s) => s + 1);

        animationRef.current = requestAnimationFrame(animate);
    }, [running, mode, graph, stepSize]);

    useEffect(() => {
        if (running) {
            animationRef.current = requestAnimationFrame(animate);
        }
        return () => cancelAnimationFrame(animationRef.current);
    }, [running, animate]);

    useEffect(() => {
        draw();
    }, [draw]);

    useEffect(() => {
        const e = calculateEnergy(
            graph.vertices,
            graph.edges,
            disjointPairs.current,
            alpha,
            beta,
            epsilon,
        );
        setEnergy(e);
    }, [graph]);

    // Mouse drag for rotation
    const [dragging, setDragging] = useState(false);
    const lastPos = useRef({ x: 0, y: 0 });

    const handleMouseDown = (e: React.MouseEvent) => {
        setDragging(true);
        lastPos.current = { x: e.clientX, y: e.clientY };
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        if (!dragging) return;
        const dx = e.clientX - lastPos.current.x;
        const dy = e.clientY - lastPos.current.y;
        setRotationY((r) => r + dx * 0.01);
        // Clamp pitch to OrbitControls' polar range so the view can't tip
        // upside-down and silently reverse the vertical drag. @see src/viewRotation.ts (clampPolar)
        setRotationX((r) => clampPolar(r + dy * 0.01));
        lastPos.current = { x: e.clientX, y: e.clientY };
    };

    const handleMouseUp = () => setDragging(false);

    const handleWheel = (e: React.WheelEvent) => {
        e.preventDefault();
        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        setZoom((z) => Math.max(0.1, Math.min(10, z * delta)));
    };

    const reset = () => {
        setRunning(false);
        const newGraph = selectedTest.generate(testParams);
        setGraph(newGraph);
        disjointPairs.current = calculateDisjointPairs(newGraph.edges);
        setStep(0);
        setZoom(1);
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', padding: 20 }}>
            <h1 style={{ fontSize: 24, marginBottom: 10 }}>Repulsive Energy Gradient Descent</h1>

            {/* Test Selection Row */}
            <div
                style={{
                    marginBottom: 10,
                    display: 'flex',
                    gap: 15,
                    flexWrap: 'wrap',
                    alignItems: 'center',
                }}
            >
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span>Test:</span>
                    <select
                        value={selectedTestId}
                        onChange={(e) => handleTestChange(e.target.value)}
                        style={{ padding: 8, fontSize: 14, minWidth: 200 }}
                    >
                        {testConfigs.map((t) => (
                            <option key={t.id} value={t.id}>
                                {t.name}
                            </option>
                        ))}
                    </select>
                </label>

                {/* Parameter sliders for configurable tests */}
                {selectedTest.params &&
                    selectedTest.params.map((p) => (
                        <label
                            key={p.name}
                            style={{ display: 'flex', alignItems: 'center', gap: 8 }}
                        >
                            <span>{p.name}:</span>
                            <input
                                type="range"
                                min={p.min}
                                max={p.max}
                                value={testParams[p.name] ?? p.default}
                                onChange={(e) =>
                                    handleParamChange(p.name, parseInt(e.target.value))
                                }
                                style={{ width: 80 }}
                            />
                            <span style={{ fontFamily: 'monospace', width: 40 }}>
                                {testParams[p.name] ?? p.default}
                            </span>
                        </label>
                    ))}

                {selectedTest.params && (
                    <button
                        onClick={regenerate}
                        style={{
                            padding: '8px 16px',
                            fontSize: 14,
                            cursor: 'pointer',
                            background: '#5577cc',
                            color: '#fff',
                            border: 'none',
                            borderRadius: 5,
                        }}
                    >
                        Regenerate
                    </button>
                )}
            </div>

            {/* Controls Row */}
            <div
                style={{
                    marginBottom: 15,
                    display: 'flex',
                    gap: 15,
                    flexWrap: 'wrap',
                    alignItems: 'center',
                }}
            >
                <button
                    onClick={() => setRunning(!running)}
                    style={{
                        padding: '10px 20px',
                        fontSize: 16,
                        cursor: 'pointer',
                        background: running ? '#ff4444' : '#44aa44',
                        color: '#fff',
                        border: 'none',
                        borderRadius: 5,
                    }}
                >
                    {running ? 'Stop' : 'Start'}
                </button>
                <button
                    onClick={reset}
                    style={{
                        padding: '10px 20px',
                        fontSize: 16,
                        cursor: 'pointer',
                        background: '#666',
                        color: '#fff',
                        border: 'none',
                        borderRadius: 5,
                    }}
                >
                    Reset
                </button>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span>Mode:</span>
                    <select
                        value={mode}
                        onChange={(e) => setMode(e.target.value as 'analytical' | 'finiteDiff')}
                        style={{ padding: 8, fontSize: 14 }}
                    >
                        <option value="analytical">Analytical</option>
                        <option value="finiteDiff">Finite Diff</option>
                    </select>
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span>Step Size:</span>
                    <input
                        type="range"
                        min="-5"
                        max="-2"
                        step="0.1"
                        value={Math.log10(stepSize)}
                        onChange={(e) => setStepSize(10 ** parseFloat(e.target.value))}
                        style={{ width: 100 }}
                    />
                    <span style={{ fontFamily: 'monospace', width: 60 }}>
                        {stepSize.toExponential(0)}
                    </span>
                </label>
            </div>

            {/* Stats Row */}
            <div style={{ marginBottom: 10, fontFamily: 'monospace' }}>
                <span style={{ marginRight: 20 }}>Step: {step}</span>
                <span style={{ marginRight: 20 }}>Energy: {energy.toFixed(6)}</span>
                <span style={{ marginRight: 20 }}>Vertices: {graph.vertices.length}</span>
                <span style={{ marginRight: 20 }}>Edges: {graph.edges.length}</span>
                <span style={{ marginRight: 20 }}>Zoom: {zoom.toFixed(2)}x</span>
                <span
                    style={{
                        color: mode === 'analytical' ? '#00ff88' : '#ffaa00',
                    }}
                >
                    Gradient:{' '}
                    {mode === 'analytical' ? 'Analytical (green)' : 'Finite Diff (orange)'}
                </span>
            </div>
            <div style={{ flex: 1, position: 'relative' }}>
                <canvas
                    ref={canvasRef}
                    width={800}
                    height={600}
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseUp}
                    onWheel={handleWheel}
                    style={{
                        border: '2px solid #333',
                        borderRadius: 8,
                        cursor: dragging ? 'grabbing' : 'grab',
                        maxWidth: '100%',
                    }}
                />
            </div>
            <p style={{ marginTop: 10, color: '#888', fontSize: 14 }}>
                Drag to rotate. Scroll to zoom. Arrows show negative gradient (descent direction).
            </p>
        </div>
    );
}

createRoot(document.getElementById('root')!).render(<App />);
