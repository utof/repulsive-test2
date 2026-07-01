import { calculateDisjointPairs, calculateEnergy } from '../src/tangentPointEnergy';
import type { Edge, Vec3 } from '../src/testConfigs';

// Fixed pseudo-trefoil-ish chain, N points, deterministic (index-based, no Math.random).
function makeChain(n: number): { vertices: Vec3[]; edges: Edge[] } {
    const vertices: Vec3[] = [];
    const edges: Edge[] = [];
    for (let i = 0; i < n; i++) {
        const t = (i / n) * Math.PI * 4;
        vertices.push([Math.cos(t), Math.sin(t), Math.sin(2 * t) * 0.5]);
        if (i > 0) edges.push([i - 1, i]);
    }
    return { vertices, edges };
}

for (const n of [50, 128, 256]) {
    const { vertices, edges } = makeChain(n);
    const dis = calculateDisjointPairs(edges);
    const iters = 20;
    const t0 = performance.now();
    for (let k = 0; k < iters; k++) calculateEnergy(vertices, edges, dis, 3, 6, 1e-10);
    const ms = (performance.now() - t0) / iters;
    console.log(`N=${n}: ${ms.toFixed(3)} ms/energy`);
}
