import { calculateEnergy, gradientAnalytical, gradientFiniteDiff } from './tangentPointEnergy';
import type { Edge, Vec3 } from './testConfigs';

// Physical/numeric constants for the tangent-point descent. These were hardcoded
// in the old src/index.tsx; centralised here so the store and scene share one source.
// @see docs/superpowers/specs/2026-07-02-react-three-webgpu-switch-design.md §4.1
export const DEFAULTS = { alpha: 3, beta: 6, epsilon: 1e-10, h: 1e-6 } as const;

export interface StepOptions {
    mode: 'analytical' | 'finiteDiff';
    stepSize: number;
    alpha?: number;
    beta?: number;
    epsilon?: number;
    h?: number;
}

// One gradient-descent step. Pure: returns NEW arrays, never mutates inputs.
// Mirrors the old animate() sequence exactly: grad -> v - stepSize*grad -> energy(new v).
export function step(
    vertices: Vec3[],
    edges: Edge[],
    disjointPairs: number[][],
    opts: StepOptions,
): { vertices: Vec3[]; energy: number } {
    const alpha = opts.alpha ?? DEFAULTS.alpha;
    const beta = opts.beta ?? DEFAULTS.beta;
    const epsilon = opts.epsilon ?? DEFAULTS.epsilon;
    const h = opts.h ?? DEFAULTS.h;

    const grad =
        opts.mode === 'analytical'
            ? gradientAnalytical(vertices, edges, disjointPairs, alpha, beta, epsilon)
            : gradientFiniteDiff(vertices, edges, disjointPairs, alpha, beta, epsilon, h);

    const next: Vec3[] = vertices.map((v, i) => [
        v[0] - opts.stepSize * grad[i][0],
        v[1] - opts.stepSize * grad[i][1],
        v[2] - opts.stepSize * grad[i][2],
    ]);

    const energy = calculateEnergy(next, edges, disjointPairs, alpha, beta, epsilon);
    return { vertices: next, energy };
}
