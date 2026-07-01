export type Vec3 = [number, number, number];
export type Edge = [number, number];

export interface GraphState {
    vertices: Vec3[];
    edges: Edge[];
}

export interface TestConfig {
    id: string;
    name: string;
    generate: (params?: Record<string, number>) => GraphState;
    params?: { name: string; min: number; max: number; default: number }[];
}

// Test 1: Original crossing lines
function createCrossingLines(): GraphState {
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
            [0, 1],
            [1, 2],
            [2, 3],
            [4, 5],
            [5, 6],
            [6, 7],
        ],
    };
}

// Test 2: 3D Helix that crosses itself
function createHelix(): GraphState {
    const vertices: Vec3[] = [];
    const edges: Edge[] = [];
    const n = 30;

    for (let i = 0; i < n; i++) {
        const t = (i / n) * Math.PI * 4;
        const r = 1.0;
        vertices.push([r * Math.cos(t), (i / n) * 3 - 1.5, r * Math.sin(t)]);
        if (i > 0) edges.push([i - 1, i]);
    }

    return { vertices, edges };
}

// Test 3: Two interlinked rings (like chain links)
function createLinkedRings(): GraphState {
    const vertices: Vec3[] = [];
    const edges: Edge[] = [];
    const n = 16;

    // First ring in XY plane
    for (let i = 0; i < n; i++) {
        const t = (i / n) * Math.PI * 2;
        vertices.push([Math.cos(t), Math.sin(t), 0]);
        edges.push([i, (i + 1) % n]);
    }

    // Second ring in XZ plane, offset and rotated
    const offset = n;
    for (let i = 0; i < n; i++) {
        const t = (i / n) * Math.PI * 2;
        vertices.push([0.5 + Math.cos(t) * 0.8, 0, Math.sin(t) * 0.8]);
        edges.push([offset + i, offset + ((i + 1) % n)]);
    }

    return { vertices, edges };
}

// Test 4: Knot-like structure
function createKnot(): GraphState {
    const vertices: Vec3[] = [];
    const edges: Edge[] = [];
    const n = 50;

    // Trefoil knot parametrization
    for (let i = 0; i < n; i++) {
        const t = (i / n) * Math.PI * 2;
        vertices.push(
            [
                Math.sin(t) + 2 * Math.sin(2 * t),
                Math.cos(t) - 2 * Math.cos(2 * t),
                -Math.sin(3 * t),
            ].map((x) => x * 0.4) as Vec3,
        );
        edges.push([i, (i + 1) % n]);
    }

    return { vertices, edges };
}

// Test 5: Stress test - 200 vertices grid with connections
function createStressTest(): GraphState {
    const vertices: Vec3[] = [];
    const edges: Edge[] = [];

    // Create a 3D grid of points
    const gridSize = 6; // 6x6x6 ≈ 216 points, we'll use ~200
    const spacing = 0.5;
    let idx = 0;
    const indexMap: number[][][] = [];

    for (let x = 0; x < gridSize; x++) {
        indexMap[x] = [];
        for (let y = 0; y < gridSize; y++) {
            indexMap[x][y] = [];
            for (let z = 0; z < gridSize; z++) {
                if (idx >= 200) break;
                vertices.push([
                    (x - gridSize / 2) * spacing + (Math.random() - 0.5) * 0.2,
                    (y - gridSize / 2) * spacing + (Math.random() - 0.5) * 0.2,
                    (z - gridSize / 2) * spacing + (Math.random() - 0.5) * 0.2,
                ]);
                indexMap[x][y][z] = idx;
                idx++;
            }
            if (idx >= 200) break;
        }
        if (idx >= 200) break;
    }

    // Connect adjacent vertices in the grid
    for (let x = 0; x < gridSize; x++) {
        for (let y = 0; y < gridSize; y++) {
            for (let z = 0; z < gridSize; z++) {
                const i = indexMap[x]?.[y]?.[z];
                if (i === undefined) continue;

                // Connect to neighbors
                if (x + 1 < gridSize && indexMap[x + 1]?.[y]?.[z] !== undefined) {
                    edges.push([i, indexMap[x + 1][y][z]]);
                }
                if (y + 1 < gridSize && indexMap[x]?.[y + 1]?.[z] !== undefined) {
                    edges.push([i, indexMap[x][y + 1][z]]);
                }
                if (z + 1 < gridSize && indexMap[x]?.[y]?.[z + 1] !== undefined) {
                    edges.push([i, indexMap[x][y][z + 1]]);
                }
            }
        }
    }

    return { vertices, edges };
}

// Test 6: Random configurable graph
function createRandomGraph(params?: Record<string, number>): GraphState {
    const numVertices = params?.vertices ?? 20;
    const numEdges = params?.edges ?? 30;

    const vertices: Vec3[] = [];
    const edges: Edge[] = [];

    // Random vertices in a sphere
    for (let i = 0; i < numVertices; i++) {
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        const r = Math.pow(Math.random(), 1 / 3) * 2; // Uniform in volume
        vertices.push([
            r * Math.sin(phi) * Math.cos(theta),
            r * Math.sin(phi) * Math.sin(theta),
            r * Math.cos(phi),
        ]);
    }

    // Random edges (avoiding duplicates and self-loops)
    const edgeSet = new Set<string>();
    let attempts = 0;
    while (edges.length < numEdges && attempts < numEdges * 10) {
        const a = Math.floor(Math.random() * numVertices);
        const b = Math.floor(Math.random() * numVertices);
        if (a !== b) {
            const key = a < b ? `${a}-${b}` : `${b}-${a}`;
            if (!edgeSet.has(key)) {
                edgeSet.add(key);
                edges.push([a, b]);
            }
        }
        attempts++;
    }

    return { vertices, edges };
}

// Test 7: Random chain (connected path)
function createRandomChain(params?: Record<string, number>): GraphState {
    const numVertices = params?.vertices ?? 50;

    const vertices: Vec3[] = [];
    const edges: Edge[] = [];

    // Start at origin
    let pos: Vec3 = [0, 0, 0];
    vertices.push([...pos]);

    // Random walk
    for (let i = 1; i < numVertices; i++) {
        const dir: Vec3 = [
            (Math.random() - 0.5) * 0.5,
            (Math.random() - 0.5) * 0.5,
            (Math.random() - 0.5) * 0.5,
        ];
        pos = [pos[0] + dir[0], pos[1] + dir[1], pos[2] + dir[2]];
        vertices.push([...pos]);
        edges.push([i - 1, i]);
    }

    // Center the vertices
    const center: Vec3 = [0, 0, 0];
    for (const v of vertices) {
        center[0] += v[0];
        center[1] += v[1];
        center[2] += v[2];
    }
    center[0] /= numVertices;
    center[1] /= numVertices;
    center[2] /= numVertices;

    for (const v of vertices) {
        v[0] -= center[0];
        v[1] -= center[1];
        v[2] -= center[2];
    }

    return { vertices, edges };
}

export const testConfigs: TestConfig[] = [
    {
        id: 'crossing',
        name: 'Crossing Lines',
        generate: createCrossingLines,
    },
    {
        id: 'helix',
        name: 'Helix Spiral',
        generate: createHelix,
    },
    {
        id: 'linked-rings',
        name: 'Linked Rings',
        generate: createLinkedRings,
    },
    {
        id: 'knot',
        name: 'Trefoil Knot',
        generate: createKnot,
    },
    {
        id: 'stress',
        name: 'Stress Test (200 vertices)',
        generate: createStressTest,
    },
    {
        id: 'random',
        name: 'Random Graph (configurable)',
        generate: createRandomGraph,
        params: [
            { name: 'vertices', min: 5, max: 500, default: 30 },
            { name: 'edges', min: 5, max: 1000, default: 50 },
        ],
    },
    {
        id: 'chain',
        name: 'Random Chain (configurable)',
        generate: createRandomChain,
        params: [{ name: 'vertices', min: 5, max: 500, default: 50 }],
    },
];
