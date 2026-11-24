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

/**
 * The same trefoil at a chosen resolution, up to the project's scale target. Why a separate
 * preset: `createKnot` is pinned at 50 vertices by the golden suites; this one exists to put a
 * large closed loop in the app, which the hierarchical route (`sobolevRoute.ts`) needs.
 */
function createLargeKnot(params?: Record<string, number>): GraphState {
    const n = Math.max(3, Math.round(params?.vertices ?? 2000));
    const vertices: Vec3[] = [];
    const edges: Edge[] = [];
    for (let i = 0; i < n; i++) {
        const t = (i / n) * Math.PI * 2;
        vertices.push([
            0.4 * (Math.sin(t) + 2 * Math.sin(2 * t)),
            0.4 * (Math.cos(t) - 2 * Math.cos(2 * t)),
            0.4 * -Math.sin(3 * t),
        ]);
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
        const r = Math.random() ** (1 / 3) * 2; // Uniform in volume
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

    // Connect vertices the sampler left in no edge (routine at 20v/30e). An
    // isolated vertex has all-zero rows in the Sobolev saddle system — no edge
    // assembles into its Ā rows or C columns — so the solve is exactly singular
    // and sobolev descent rejects every step. Nearest-neighbor keeps the repair
    // edge short. Guaranteed duplicate-free: no existing edge contains v.
    // @see src/core/sobolev/linsolve.ts (luSolve singularity throw)
    const used = new Set<number>();
    for (const [a, b] of edges) {
        used.add(a);
        used.add(b);
    }
    for (let v = 0; v < numVertices; v++) {
        if (used.has(v)) continue;
        let nearest = -1;
        let bestDistSq = Infinity;
        for (let u = 0; u < numVertices; u++) {
            if (u === v) continue;
            const dx = vertices[u][0] - vertices[v][0];
            const dy = vertices[u][1] - vertices[v][1];
            const dz = vertices[u][2] - vertices[v][2];
            const distSq = dx * dx + dy * dy + dz * dz;
            if (distSq < bestDistSq) {
                bestDistSq = distSq;
                nearest = u;
            }
        }
        edges.push([v, nearest]);
        used.add(v);
        used.add(nearest);
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

/**
 * A flat (z = 0) tree drawn radially: six main branches, one to three children per node,
 * each subtree in a wedge as wide as its share of the leaves — a mind map without crossings.
 * Why it exists: moving a BRANCH (turning it, hanging it on another node) is the mind-map
 * gesture, and the grid has no branch: every block of it has links on several sides.
 * Deterministic (same tree for the same size); overall extent about 4, like the others.
 * @see src/core/medialGraphMove.ts
 */
function createFlatTree(params?: Record<string, number>): GraphState {
    const target = Math.max(8, Math.floor(params?.nodes ?? 150));
    let seed = 11;
    const rnd = (): number => {
        seed = (1664525 * seed + 1013904223) % 4294967296;
        return seed / 4294967296;
    };
    const parent = [-1];
    const kids: number[][] = [[]];
    const depth = [0];
    let frontier = [0];
    while (parent.length < target && frontier.length > 0) {
        const next: number[] = [];
        for (const p of frontier) {
            const c = p === 0 ? 6 : 1 + Math.floor(rnd() * 2.4);
            for (let k = 0; k < c && parent.length < target; k++) {
                parent.push(p);
                kids.push([]);
                depth.push(depth[p] + 1);
                kids[p].push(parent.length - 1);
                next.push(parent.length - 1);
            }
        }
        frontier = next;
    }
    const n = parent.length;
    const leaves = new Array<number>(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
        if (kids[i].length === 0) leaves[i] = 1;
        if (parent[i] >= 0) leaves[parent[i]] += leaves[i];
    }
    const ring = 2 / Math.max(...depth); // outermost ring at radius 2
    const vertices: Vec3[] = new Array(n);
    const edges: Edge[] = [];
    const lay = (i: number, a0: number, a1: number): void => {
        const a = 0.5 * (a0 + a1);
        vertices[i] = [depth[i] * ring * Math.cos(a), depth[i] * ring * Math.sin(a), 0];
        let at = a0;
        for (const c of kids[i]) {
            const span = ((a1 - a0) * leaves[c]) / leaves[i];
            edges.push([i, c]);
            lay(c, at, at + span);
            at += span;
        }
    };
    lay(0, 0, 2 * Math.PI);
    return { vertices, edges };
}

/**
 * A flat (z = 0) jittered grid: a planar graph with junctions, the stand-in for a mind map.
 * Why it exists: the local graph flow and the crossing guard work in the plane only, and no
 * other preset is flat. The jitter is deterministic (same graph for the same size).
 * @see src/core/medialSphereGraph.ts
 */
function createFlatGrid(params?: Record<string, number>): GraphState {
    const g = Math.max(2, Math.floor(params?.size ?? 8));
    let seed = 7;
    const rnd = (): number => {
        seed = (1664525 * seed + 1013904223) % 4294967296;
        return seed / 4294967296;
    };
    const vertices: Vec3[] = [];
    const edges: Edge[] = [];
    const scale = 4 / (g - 1); // overall extent 4, like the other presets
    for (let y = 0; y < g; y++)
        for (let x = 0; x < g; x++)
            vertices.push([
                (x - (g - 1) / 2 + 0.3 * (rnd() - 0.5)) * scale,
                (y - (g - 1) / 2 + 0.3 * (rnd() - 0.5)) * scale,
                0,
            ]);
    for (let y = 0; y < g; y++)
        for (let x = 0; x < g; x++) {
            if (x + 1 < g) edges.push([y * g + x, y * g + x + 1]);
            if (y + 1 < g) edges.push([y * g + x, (y + 1) * g + x]);
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
        id: 'knot-large',
        name: 'Trefoil Knot (large, configurable)',
        generate: createLargeKnot,
        params: [{ name: 'vertices', min: 100, max: 30000, default: 2000 }],
    },
    {
        id: 'flat-grid',
        name: 'Flat Grid (planar graph, configurable)',
        generate: createFlatGrid,
        params: [{ name: 'size', min: 2, max: 40, default: 8 }],
    },
    {
        id: 'flat-tree',
        name: 'Flat Tree (planar mind map, configurable)',
        generate: createFlatTree,
        params: [{ name: 'nodes', min: 8, max: 3000, default: 150 }],
    },
    {
        id: 'chain',
        name: 'Random Chain (configurable)',
        generate: createRandomChain,
        params: [{ name: 'vertices', min: 5, max: 500, default: 50 }],
    },
];
