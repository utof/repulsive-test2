import React, { useState, useRef, useEffect, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import { testConfigs, type GraphState, type Vec3, type Edge } from './testConfigs';

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

// Analytical gradient (matches calculateEnergy exactly, including /2)
function gradientAnalytical(
  vertices: Vec3[],
  edges: Edge[],
  disjointPairs: number[][],
  alpha: number,
  beta: number,
  epsilon: number
): Vec3[] {
  const gradient: Vec3[] = vertices.map(() => [0, 0, 0]);

  const zero: Vec3 = [0, 0, 0];

  const addToGrad = (idx: number, v: number[]) => {
    gradient[idx][0] += v[0] || 0;
    gradient[idx][1] += v[1] || 0;
    gradient[idx][2] += v[2] || 0;
  };

  const safeUnit = (v: number[]): { len: number; unit: Vec3 } => {
    const r = norm(v);
    if (r < 1e-14) return { len: r, unit: [0, 0, 0] };
    return { len: r, unit: scale(1 / r, v) as Vec3 };
  };

  // f(d,e) = (||e x d|| + eps)^alpha / (||d|| + eps)^beta
  // returns f, df/dd (w.r.t dvec), df/de (w.r.t evec)
  const kernelDerivs = (e: Vec3, dvec: Vec3) => {
    const { len: rd, unit: dHat } = safeUnit(dvec);
    const d_eps = rd + epsilon;

    const cvec = cross3D(e, dvec) as Vec3;
    const { len: rc } = safeUnit(cvec);
    const c_eps = rc + epsilon;

    const cPowA = Math.pow(c_eps, alpha);
    const dPowB = Math.pow(d_eps, beta);
    const f = cPowA / dPowB;

    // dc/dd = ((e x d) x e) / ||e x d||
    // dc/de = (d x (e x d)) / ||e x d||
    let dc_dd: Vec3 = zero;
    let dc_de: Vec3 = zero;
    if (rc >= 1e-14) {
      dc_dd = scale(1 / rc, cross3D(cvec, e)) as Vec3;
      dc_de = scale(1 / rc, cross3D(dvec, cvec)) as Vec3;
    }

    // df/dd = alpha*(c_eps)^(a-1)/d_eps^b * dc/dd  - beta*(c_eps)^a/d_eps^(b+1) * dHat
    const coeff_c = alpha * Math.pow(c_eps, alpha - 1) / dPowB;
    const coeff_d = -beta * cPowA / Math.pow(d_eps, beta + 1);

    const df_dd = add(scale(coeff_c, dc_dd), scale(coeff_d, dHat)) as Vec3;

    // df/de = alpha*(c_eps)^(a-1)/d_eps^b * dc/de
    const df_de = scale(coeff_c, dc_de) as Vec3;

    return { f, df_dd, df_de };
  };

  for (let I = 0; I < edges.length; I++) {
    const dis = disjointPairs[I];
    if (!dis || dis.length === 0) continue;

    const [i1, i2] = edges[I];
    const e_I = subtract(vertices[i2], vertices[i1]) as Vec3;

    const { len: reI, unit: eI_hat } = safeUnit(e_I);
    const ell_I = reI + epsilon;
    const ell_I_pow = Math.pow(ell_I, 1 - alpha);

    // d(ell_I^(1-a))/dv = (1-a)*ell_I^(-a) * d(ell_I)/dv
    // d(ell_I)/dv_i1 = -eI_hat, d(ell_I)/dv_i2 = +eI_hat
    const dPow_dv_i1 = scale((1 - alpha) * Math.pow(ell_I, -alpha), scale(-1, eI_hat)) as Vec3;
    const dPow_dv_i2 = scale((1 - alpha) * Math.pow(ell_I, -alpha), eI_hat) as Vec3;

    for (const J of dis) {
      const [j1, j2] = edges[J];

      const e_J = subtract(vertices[j2], vertices[j1]) as Vec3;
      const { len: reJ, unit: eJ_hat } = safeUnit(e_J);
      const ell_J = reJ + epsilon;

      // d(ell_J)/dv_j1 = -eJ_hat, d(ell_J)/dv_j2 = +eJ_hat
      const dEllJ_dv_j1 = scale(-1, eJ_hat) as Vec3;
      const dEllJ_dv_j2 = eJ_hat;

      // Precompute all 4 endpoint pairs for this (I,J)
      const pairs = [
        { i: i1, j: j1 },
        { i: i1, j: j2 },
        { i: i2, j: j1 },
        { i: i2, j: j2 },
      ];

      type Term = {
        i: number;
        j: number;
        f: number;
        df_dd: Vec3;
        df_de: Vec3;
      };

      const terms: Term[] = [];
      let sumF = 0;

      for (const { i, j } of pairs) {
        const dvec = subtract(vertices[i], vertices[j]) as Vec3;
        const { f, df_dd, df_de } = kernelDerivs(e_I, dvec);
        sumF += f;
        terms.push({ i, j, f, df_dd, df_de });
      }

      // Energy contribution (ordered pair):
      // E_IJ = 0.25 * ell_I^(1-a) * ell_J * sumF
      // So:
      // dE via ell_I^(1-a): 0.25 * ell_J * sumF * d(ell_I^(1-a))
      addToGrad(i1, scale(0.25 * ell_J * sumF, dPow_dv_i1));
      addToGrad(i2, scale(0.25 * ell_J * sumF, dPow_dv_i2));

      // dE via ell_J: 0.25 * ell_I^(1-a) * sumF * d(ell_J)
      addToGrad(j1, scale(0.25 * ell_I_pow * sumF, dEllJ_dv_j1));
      addToGrad(j2, scale(0.25 * ell_I_pow * sumF, dEllJ_dv_j2));

      const base = 0.25 * ell_I_pow * ell_J;

      // Per-term derivatives:
      // - dvec part goes to the specific endpoints (i and j) of that term
      // - e_I part goes to BOTH i1 and i2 (since e_I = v_i2 - v_i1) for every term
      for (const t of terms) {
        // dvec = v_i - v_j
        addToGrad(t.i, scale(base, t.df_dd));
        addToGrad(t.j, scale(-base, t.df_dd));

        // e_I = v_i2 - v_i1
        addToGrad(i1, scale(-base, t.df_de));
        addToGrad(i2, scale(base, t.df_de));
      }
    }
  }

  // calculateEnergy divides by 2 (because disjointPairs contains both (I,J) and (J,I)),
  // so gradient must also be divided by 2 to match finite-diff.
  for (let v = 0; v < gradient.length; v++) {
    gradient[v][0] *= 0.5;
    gradient[v][1] *= 0.5;
    gradient[v][2] *= 0.5;
  }

  return gradient;
}


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

// 3D rotation (returns rotated point)
function rotate3D(
    point: Vec3,
    rotationY: number,
    rotationX: number
): Vec3 {
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

    return [x1, y1, z2];
}

// 3D projection
function project(
    point: Vec3,
    width: number,
    height: number,
    rotationY: number,
    rotationX: number,
    zoom: number = 1
): [number, number] {
    const [x1, y1, z2] = rotate3D(point, rotationY, rotationX);

    // Simple perspective
    const fov = 3;
    const s = fov / (fov + z2 + 3);
    const screenX = width / 2 + x1 * s * 150 * zoom;
    const screenY = height / 2 - y1 * s * 150 * zoom;

    return [screenX, screenY];
}

// Project a direction vector from a point (for arrows)
// Returns screen-space direction that's consistent regardless of view
function projectDirection(
    origin: Vec3,
    direction: Vec3,
    rotationY: number,
    rotationX: number
): [number, number] {
    // Rotate the direction vector (not a point, so no translation)
    const rotated = rotate3D(direction, rotationY, rotationX);
    // For screen space: x goes right, y goes down (but we flip y in project)
    // So direction in screen space is (rotated.x, -rotated.y)
    const len = Math.sqrt(rotated[0] * rotated[0] + rotated[1] * rotated[1]);
    if (len < 1e-10) return [0, 0];
    return [rotated[0] / len, -rotated[1] / len];
}

// Logarithmic scaling for arrow length
function logScale(magnitude: number): number {
    if (magnitude < 0.001) return 0;
    // log(1 + x) gives nice compression of large values
    // Scale so small gradients are visible but large ones don't explode
    return Math.sign(magnitude) * Math.log(1 + Math.abs(magnitude)) * 15;
}

function App() {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    // Load saved config
    const savedConfig = loadSavedConfig();
    const initialTest = testConfigs.find(t => t.id === savedConfig.testId) || testConfigs[0];
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

    const selectedTest = testConfigs.find(t => t.id === selectedTestId) || testConfigs[0];

    // Save config when test or params change
    useEffect(() => {
        saveConfig(selectedTestId, testParams);
    }, [selectedTestId, testParams]);

    // Handle test change
    const handleTestChange = (newTestId: string) => {
        setRunning(false);
        setSelectedTestId(newTestId);
        const test = testConfigs.find(t => t.id === newTestId)!;

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
            const grad = mode === 'analytical'
                ? gradientAnalytical(graph.vertices, graph.edges, disjointPairs.current, alpha, beta, epsilon)
                : gradientFiniteDiff(graph.vertices, graph.edges, disjointPairs.current, alpha, beta, epsilon, h);

            ctx.strokeStyle = mode === 'analytical' ? '#00ff88' : '#ffaa00';
            ctx.lineWidth = 2;

            for (let i = 0; i < graph.vertices.length; i++) {
                const [x, y] = project(graph.vertices[i], width, height, rotationY, rotationX, zoom);
                const gradVec = grad[i];
                const gradNorm = norm(gradVec);

                if (gradNorm > 0.001) {
                    // Negative gradient = descent direction
                    const negGrad: Vec3 = [-gradVec[0], -gradVec[1], -gradVec[2]];

                    // Get consistent screen-space direction from 3D gradient
                    const [dirX, dirY] = projectDirection(graph.vertices[i], negGrad, rotationY, rotationX);

                    // Logarithmic length scaling
                    const arrowLen = logScale(gradNorm) * zoom;

                    if (arrowLen > 2) {
                        const tx = x + dirX * arrowLen;
                        const ty = y + dirY * arrowLen;

                        ctx.beginPath();
                        ctx.moveTo(x, y);
                        ctx.lineTo(tx, ty);
                        ctx.stroke();

                        // Arrow head
                        const angle = Math.atan2(dirY, dirX);
                        const headSize = Math.min(8, arrowLen * 0.3);
                        ctx.beginPath();
                        ctx.moveTo(tx, ty);
                        ctx.lineTo(tx - headSize * Math.cos(angle - 0.4), ty - headSize * Math.sin(angle - 0.4));
                        ctx.moveTo(tx, ty);
                        ctx.lineTo(tx - headSize * Math.cos(angle + 0.4), ty - headSize * Math.sin(angle + 0.4));
                        ctx.stroke();
                    }
                }
            }
        }
    }, [graph, rotationY, rotationX, zoom, mode, running]);

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

    const handleWheel = (e: React.WheelEvent) => {
        e.preventDefault();
        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        setZoom(z => Math.max(0.1, Math.min(10, z * delta)));
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
            <div style={{ marginBottom: 10, display: 'flex', gap: 15, flexWrap: 'wrap', alignItems: 'center' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span>Test:</span>
                    <select
                        value={selectedTestId}
                        onChange={e => handleTestChange(e.target.value)}
                        style={{ padding: 8, fontSize: 14, minWidth: 200 }}
                    >
                        {testConfigs.map(t => (
                            <option key={t.id} value={t.id}>{t.name}</option>
                        ))}
                    </select>
                </label>

                {/* Parameter sliders for configurable tests */}
                {selectedTest.params && selectedTest.params.map(p => (
                    <label key={p.name} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span>{p.name}:</span>
                        <input
                            type="range"
                            min={p.min}
                            max={p.max}
                            value={testParams[p.name] ?? p.default}
                            onChange={e => handleParamChange(p.name, parseInt(e.target.value))}
                            style={{ width: 80 }}
                        />
                        <span style={{ fontFamily: 'monospace', width: 40 }}>{testParams[p.name] ?? p.default}</span>
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
                            borderRadius: 5
                        }}
                    >
                        Regenerate
                    </button>
                )}
            </div>

            {/* Controls Row */}
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

            {/* Stats Row */}
            <div style={{ marginBottom: 10, fontFamily: 'monospace' }}>
                <span style={{ marginRight: 20 }}>Step: {step}</span>
                <span style={{ marginRight: 20 }}>Energy: {energy.toFixed(6)}</span>
                <span style={{ marginRight: 20 }}>Vertices: {graph.vertices.length}</span>
                <span style={{ marginRight: 20 }}>Edges: {graph.edges.length}</span>
                <span style={{ marginRight: 20 }}>Zoom: {zoom.toFixed(2)}x</span>
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
                    onWheel={handleWheel}
                    style={{
                        border: '2px solid #333',
                        borderRadius: 8,
                        cursor: dragging ? 'grabbing' : 'grab',
                        maxWidth: '100%'
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
