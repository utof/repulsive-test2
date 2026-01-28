import React, { useState, useRef, useEffect, useCallback } from 'react';
import { createRoot } from 'react-dom/client';

// Vector math helpers
const cross3D = (a: number[], b: number[]): number[] => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]
];

const dot = (a: number[], b: number[]): number =>
    a.reduce((sum, val, i) => sum + val * b[i], 0);

const subtract = (a: number[], b: number[]): number[] =>
    a.map((val, i) => val - b[i]);

const scale = (s: number, v: number[]): number[] =>
    v.map(x => s * x);

const norm = (v: number[]): number =>
    Math.sqrt(dot(v, v));

const add = (a: number[], b: number[]): number[] =>
    a.map((val, i) => val + b[i]);

type Vec3 = [number, number, number];
type Edge = [number, number];

interface GraphState {
    vertices: Vec3[];
    edges: Edge[];
}

// Calculate disjoint edge pairs (edges sharing no vertices)
function calculateDisjointPairs(edges: Edge[]): number[][] {
    const disjoint: number[][] = [];
    for (let i = 0; i < edges.length; i++) {
        disjoint[i] = [];
        for (let j = 0; j < edges.length; j++) {
            if (i === j) continue;
            const [a1, a2] = edges[i];
            const [b1, b2] = edges[j];
            if (a1 !== b1 && a1 !== b2 && a2 !== b1 && a2 !== b2) {
                disjoint[i].push(j);
            }
        }
    }
    return disjoint;
}

// Calculate energy
function calculateEnergy(
    vertices: Vec3[],
    edges: Edge[],
    disjointPairs: number[][],
    alpha: number,
    beta: number,
    epsilon: number
): number {
    let totalEnergy = 0;

    for (let I = 0; I < edges.length; I++) {
        if (!disjointPairs[I]) continue;

        const [i1, i2] = edges[I];
        const e_I = subtract(vertices[i2], vertices[i1]);
        const ell_I = norm(e_I) + epsilon;

        for (const J of disjointPairs[I]) {
            const [j1, j2] = edges[J];
            const ell_J = norm(subtract(vertices[j2], vertices[j1])) + epsilon;

            let sumK = 0;
            for (const i of [i1, i2]) {
                for (const j of [j1, j2]) {
                    const d = subtract(vertices[i], vertices[j]);
                    const d_norm = norm(d) + epsilon;
                    const c_norm = norm(cross3D(e_I, d)) + epsilon;
                    sumK += Math.pow(c_norm, alpha) / Math.pow(d_norm, beta);
                }
            }

            totalEnergy += 0.25 * Math.pow(ell_I, 1 - alpha) * ell_J * sumK;
        }
    }

    return totalEnergy / 2;
}

// Finite difference gradient
function gradientFiniteDiff(
    vertices: Vec3[],
    edges: Edge[],
    disjointPairs: number[][],
    alpha: number,
    beta: number,
    epsilon: number,
    h: number
): Vec3[] {
    const gradient: Vec3[] = vertices.map(() => [0, 0, 0]);
    const E0 = calculateEnergy(vertices, edges, disjointPairs, alpha, beta, epsilon);

    for (let v = 0; v < vertices.length; v++) {
        for (let d = 0; d < 3; d++) {
            const perturbed: Vec3[] = vertices.map(vtx => [...vtx] as Vec3);
            perturbed[v][d] += h;
            const E1 = calculateEnergy(perturbed, edges, disjointPairs, alpha, beta, epsilon);
            gradient[v][d] = (E1 - E0) / h;
        }
    }

    return gradient;
}

// Analytical gradient
function gradientAnalytical(
    vertices: Vec3[],
    edges: Edge[],
    disjointPairs: number[][],
    alpha: number,
    beta: number,
    epsilon: number
): Vec3[] {
    const gradient: Vec3[] = vertices.map(() => [0, 0, 0]);

    const addToGrad = (idx: number, contrib: number[]) => {
        for (let d = 0; d < 3; d++) {
            gradient[idx][d] += contrib[d] || 0;
        }
    };

    for (let I = 0; I < edges.length; I++) {
        if (!disjointPairs[I]) continue;

        const [i1, i2] = edges[I];
        const e_I = subtract(vertices[i2], vertices[i1]);
        const ell_I = norm(e_I) + epsilon;
        const T_J_unused = scale(1 / ell_I, e_I);

        for (const J of disjointPairs[I]) {
            const [j1, j2] = edges[J];
            const e_J = subtract(vertices[j2], vertices[j1]);
            const ell_J = norm(e_J) + epsilon;
            const T_J = scale(1 / ell_J, e_J);

            const pairs = [
                { i: i1, j: j1 }, { i: i1, j: j2 },
                { i: i2, j: j1 }, { i: i2, j: j2 }
            ];

            let sumK = 0;
            const kernelInfo = pairs.map(({ i, j }) => {
                const d = subtract(vertices[i], vertices[j]);
                const d_norm = norm(d) + epsilon;
                const c_norm = norm(cross3D(e_I, d)) + epsilon;
                const K = Math.pow(c_norm, alpha) / Math.pow(d_norm, beta);
                sumK += K;
                return { i, j, d, d_norm, c_norm, K };
            });

            const ell_I_pow = Math.pow(ell_I, 1 - alpha);

            // i1 contributions
            addToGrad(i1, scale(
                (1 - alpha) * Math.pow(ell_I, -alpha - 1) * sumK * 0.25 * ell_J,
                subtract(vertices[i1], vertices[i2])
            ));

            for (const { i, j, d, d_norm, c_norm } of kernelInfo) {
                const d_cross_ed = subtract(scale(d_norm * d_norm, e_I), scale(dot(d, e_I), d));
                const factor = alpha * Math.pow(c_norm, alpha - 2) / Math.pow(d_norm, beta);
                addToGrad(i1, scale(0.25 * ell_J * ell_I_pow * factor, d_cross_ed));

                if (i === i1) {
                    const e_cross_ed = subtract(scale(dot(e_I, d), e_I), scale(ell_I * ell_I, d));
                    addToGrad(i1, scale(0.25 * ell_J * ell_I_pow * factor, e_cross_ed));
                    const d_factor = -beta * Math.pow(c_norm, alpha) / Math.pow(d_norm, beta + 2);
                    addToGrad(i1, scale(0.25 * ell_J * ell_I_pow * d_factor, d));
                }
            }

            // i2 contributions
            addToGrad(i2, scale(
                (1 - alpha) * Math.pow(ell_I, -alpha - 1) * sumK * 0.25 * ell_J,
                subtract(vertices[i2], vertices[i1])
            ));

            for (const { i, j, d, d_norm, c_norm } of kernelInfo) {
                const d_cross_ed = subtract(scale(d_norm * d_norm, e_I), scale(dot(d, e_I), d));
                const factor = alpha * Math.pow(c_norm, alpha - 2) / Math.pow(d_norm, beta);
                addToGrad(i2, scale(-0.25 * ell_J * ell_I_pow * factor, d_cross_ed));

                if (i === i2) {
                    const e_cross_ed = subtract(scale(dot(e_I, d), e_I), scale(ell_I * ell_I, d));
                    addToGrad(i2, scale(0.25 * ell_J * ell_I_pow * factor, e_cross_ed));
                    const d_factor = -beta * Math.pow(c_norm, alpha) / Math.pow(d_norm, beta + 2);
                    addToGrad(i2, scale(0.25 * ell_J * ell_I_pow * d_factor, d));
                }
            }

            // j1 contributions
            addToGrad(j1, scale(-0.25 * ell_I_pow * sumK, T_J));

            for (const { i, j, d, d_norm, c_norm } of kernelInfo) {
                if (j !== j1) continue;
                const e_cross_ed = subtract(scale(dot(e_I, d), e_I), scale(ell_I * ell_I, d));
                const factor = alpha * Math.pow(c_norm, alpha - 2) / Math.pow(d_norm, beta);
                addToGrad(j1, scale(-0.25 * ell_J * ell_I_pow * factor, e_cross_ed));
                const d_factor = -beta * Math.pow(c_norm, alpha) / Math.pow(d_norm, beta + 2);
                addToGrad(j1, scale(-0.25 * ell_J * ell_I_pow * d_factor, d));
            }

            // j2 contributions
            addToGrad(j2, scale(0.25 * ell_I_pow * sumK, T_J));

            for (const { i, j, d, d_norm, c_norm } of kernelInfo) {
                if (j !== j2) continue;
                const e_cross_ed = subtract(scale(dot(e_I, d), e_I), scale(ell_I * ell_I, d));
                const factor = alpha * Math.pow(c_norm, alpha - 2) / Math.pow(d_norm, beta);
                addToGrad(j2, scale(-0.25 * ell_J * ell_I_pow * factor, e_cross_ed));
                const d_factor = -beta * Math.pow(c_norm, alpha) / Math.pow(d_norm, beta + 2);
                addToGrad(j2, scale(-0.25 * ell_J * ell_I_pow * d_factor, d));
            }
        }
    }

    return gradient;
}

// Initial graph - two crossing lines in 3D
function createInitialGraph(): GraphState {
    return {
        vertices: [
            [-1.5, 0, 0.3],
            [-0.5, 0, 0.3],
            [0.5, 0, 0.3],
            [1.5, 0, 0.3],
            [0, -1.5, -0.3],
            [0, -0.5, -0.3],
            [0, 0.5, -0.3],
            [0, 1.5, -0.3],
        ],
        edges: [
            [0, 1], [1, 2], [2, 3],  // horizontal line
            [4, 5], [5, 6], [6, 7],  // vertical line (crossing)
        ]
    };
}

// 3D projection
function project(
    point: Vec3,
    width: number,
    height: number,
    rotationY: number,
    rotationX: number
): [number, number] {
    const [x, y, z] = point;

    // Rotate around Y
    const cosY = Math.cos(rotationY);
    const sinY = Math.sin(rotationY);
    const x1 = x * cosY - z * sinY;
    const z1 = x * sinY + z * cosY;

    // Rotate around X
    const cosX = Math.cos(rotationX);
    const sinX = Math.sin(rotationX);
    const y1 = y * cosX - z1 * sinX;
    const z2 = y * sinX + z1 * cosX;

    // Simple perspective
    const fov = 3;
    const scale = fov / (fov + z2 + 3);
    const screenX = width / 2 + x1 * scale * 150;
    const screenY = height / 2 - y1 * scale * 150;

    return [screenX, screenY];
}

function App() {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [graph, setGraph] = useState<GraphState>(createInitialGraph);
    const [running, setRunning] = useState(false);
    const [mode, setMode] = useState<'analytical' | 'finiteDiff'>('analytical');
    const [step, setStep] = useState(0);
    const [energy, setEnergy] = useState(0);
    const [rotationY, setRotationY] = useState(0.5);
    const [rotationX, setRotationX] = useState(0.3);
    const [stepSize, setStepSize] = useState(0.001);

    const disjointPairs = useRef(calculateDisjointPairs(graph.edges));
    const animationRef = useRef<number>(0);

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
            const [x1, y1] = project(graph.vertices[v1], width, height, rotationY, rotationX);
            const [x2, y2] = project(graph.vertices[v2], width, height, rotationY, rotationX);
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
        }

        // Draw vertices
        for (let i = 0; i < graph.vertices.length; i++) {
            const [x, y] = project(graph.vertices[i], width, height, rotationY, rotationX);
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
            const grad = mode === 'analytical'
                ? gradientAnalytical(graph.vertices, graph.edges, disjointPairs.current, alpha, beta, epsilon)
                : gradientFiniteDiff(graph.vertices, graph.edges, disjointPairs.current, alpha, beta, epsilon, h);

            ctx.strokeStyle = mode === 'analytical' ? '#00ff88' : '#ffaa00';
            ctx.lineWidth = 2;

            for (let i = 0; i < graph.vertices.length; i++) {
                const [x, y] = project(graph.vertices[i], width, height, rotationY, rotationX);
                const gradVec = grad[i];
                const gradNorm = norm(gradVec);
                if (gradNorm > 0.001) {
                    const scaled = scale(-50 / Math.max(gradNorm, 1), gradVec);
                    const target: Vec3 = [
                        graph.vertices[i][0] + scaled[0],
                        graph.vertices[i][1] + scaled[1],
                        graph.vertices[i][2] + scaled[2]
                    ];
                    const [tx, ty] = project(target, width, height, rotationY, rotationX);

                    ctx.beginPath();
                    ctx.moveTo(x, y);
                    ctx.lineTo(tx, ty);
                    ctx.stroke();

                    // Arrow head
                    const angle = Math.atan2(ty - y, tx - x);
                    ctx.beginPath();
                    ctx.moveTo(tx, ty);
                    ctx.lineTo(tx - 10 * Math.cos(angle - 0.3), ty - 10 * Math.sin(angle - 0.3));
                    ctx.moveTo(tx, ty);
                    ctx.lineTo(tx - 10 * Math.cos(angle + 0.3), ty - 10 * Math.sin(angle + 0.3));
                    ctx.stroke();
                }
            }
        }
    }, [graph, rotationY, rotationX, mode, running]);

    // Animation step
    const animate = useCallback(() => {
        if (!running) return;

        const grad = mode === 'analytical'
            ? gradientAnalytical(graph.vertices, graph.edges, disjointPairs.current, alpha, beta, epsilon)
            : gradientFiniteDiff(graph.vertices, graph.edges, disjointPairs.current, alpha, beta, epsilon, h);

        const newVertices: Vec3[] = graph.vertices.map((v, i) => [
            v[0] - stepSize * grad[i][0],
            v[1] - stepSize * grad[i][1],
            v[2] - stepSize * grad[i][2]
        ]);

        const newEnergy = calculateEnergy(newVertices, graph.edges, disjointPairs.current, alpha, beta, epsilon);

        setGraph({ ...graph, vertices: newVertices });
        setEnergy(newEnergy);
        setStep(s => s + 1);

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
        const e = calculateEnergy(graph.vertices, graph.edges, disjointPairs.current, alpha, beta, epsilon);
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
        setRotationY(r => r + dx * 0.01);
        setRotationX(r => r + dy * 0.01);
        lastPos.current = { x: e.clientX, y: e.clientY };
    };

    const handleMouseUp = () => setDragging(false);

    const reset = () => {
        setGraph(createInitialGraph());
        setStep(0);
        setRunning(false);
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', padding: 20 }}>
            <h1 style={{ fontSize: 24, marginBottom: 10 }}>Repulsive Energy Gradient Descent</h1>
            <div style={{ marginBottom: 15, display: 'flex', gap: 15, flexWrap: 'wrap', alignItems: 'center' }}>
                <button
                    onClick={() => setRunning(!running)}
                    style={{
                        padding: '10px 20px',
                        fontSize: 16,
                        cursor: 'pointer',
                        background: running ? '#ff4444' : '#44aa44',
                        color: '#fff',
                        border: 'none',
                        borderRadius: 5
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
                        borderRadius: 5
                    }}
                >
                    Reset
                </button>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span>Mode:</span>
                    <select
                        value={mode}
                        onChange={e => setMode(e.target.value as 'analytical' | 'finiteDiff')}
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
                        onChange={e => setStepSize(Math.pow(10, parseFloat(e.target.value)))}
                        style={{ width: 100 }}
                    />
                    <span style={{ fontFamily: 'monospace', width: 60 }}>{stepSize.toExponential(0)}</span>
                </label>
            </div>
            <div style={{ marginBottom: 10, fontFamily: 'monospace' }}>
                <span style={{ marginRight: 20 }}>Step: {step}</span>
                <span style={{ marginRight: 20 }}>Energy: {energy.toFixed(6)}</span>
                <span style={{
                    color: mode === 'analytical' ? '#00ff88' : '#ffaa00'
                }}>
                    Gradient: {mode === 'analytical' ? 'Analytical (green)' : 'Finite Diff (orange)'}
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
                    style={{
                        border: '2px solid #333',
                        borderRadius: 8,
                        cursor: dragging ? 'grabbing' : 'grab',
                        maxWidth: '100%'
                    }}
                />
            </div>
            <p style={{ marginTop: 10, color: '#888', fontSize: 14 }}>
                Drag to rotate. Arrows show negative gradient direction (descent direction).
            </p>
        </div>
    );
}

createRoot(document.getElementById('root')!).render(<App />);
