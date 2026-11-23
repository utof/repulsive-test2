/**
 * hStepScale — what one descent step on the SHIPPED hierarchical path costs as n grows.
 *
 * Why: Phase 2H's gates certify the solver at n <= 960 against the dense reference; the figures
 * quoted for n = 30 000 (spec §3.5: 194 ms per matvec, 3.5 s precompute) come from the prototype in
 * `.superpowers/sdd/2026-09-11-phase2h-queue/scratch/`, not from `src/`. This script times the
 * shipped modules themselves, one n per process so each run starts with a clean heap.
 *
 *   bun bench/scale/hStepScale.ts <n> solve [tol]   build phases + prepare(C) + one solve; O(n) memory
 *   bun bench/scale/hStepScale.ts <n> step  [tol]   one full sobolevStepSet on the hierarchical route
 *   bun bench/scale/hStepScale.ts <n> dense         the same step on the dense route (reference)
 *
 * `solve` never builds `disjointPairs` (n rows of n-3 entries, the O(n²) object the energy and its
 * gradient still need), so it is the only mode that reaches n = 30 000 on a 16 GB machine. Its rhs
 * is a seeded mean-free random vector, NOT an energy gradient: iteration counts are for that rhs.
 *
 * Wall-clock is load-dependent: run with the machine quiet and read `loadAvg` in the output.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §3.5, §7.10
 * @issue utof/repulsive-test2#51
 */
import { loadavg } from 'node:os';
import { trefoil } from '../../src/core/fixtures';
import { DEFAULTS, sobolevStepSet } from '../../src/core/optimizer';
import { barycenterBlock, type ConstraintSet } from '../../src/core/sobolev/constraintSet';
import { barycenterPhiAndC, barycenterTarget } from '../../src/core/sobolev/constraints';
import { prepareA } from '../../src/core/sobolev/hierarchical/apply';
import { buildBct } from '../../src/core/sobolev/hierarchical/bct';
import { buildBvh } from '../../src/core/sobolev/hierarchical/bvh';
import { applyAsigma, estimateNorm2 } from '../../src/core/sobolev/hierarchical/gmres';
import {
    DEFAULT_LEAF_SIZE,
    makeHierarchicalSaddle,
} from '../../src/core/sobolev/hierarchical/hierarchicalSaddle';
import { prepareLowOrder } from '../../src/core/sobolev/hierarchical/lowOrder';
import { makeHierarchicalOperator } from '../../src/core/sobolev/hierarchical/operator';
import { makePreconditioner } from '../../src/core/sobolev/hierarchical/precond';
import { prepareSchur } from '../../src/core/sobolev/hierarchical/schur';
import { calculateDisjointPairs, calculateEnergy } from '../../src/core/tangentPointEnergy';

const n = Number(process.argv[2]);
const mode = process.argv[3] as 'solve' | 'step' | 'dense';
const tol = process.argv[4] === undefined ? undefined : Number(process.argv[4]);
if (!Number.isInteger(n) || n < 8 || !['solve', 'step', 'dense'].includes(mode)) {
    console.error('usage: bun bench/scale/hStepScale.ts <n> solve|step|dense [tol]');
    process.exit(2);
}

const ms: Record<string, number> = {};
function timed<T>(label: string, f: () => T): T {
    const t0 = performance.now();
    const r = f();
    ms[label] = Math.round((performance.now() - t0) * 10) / 10;
    return r;
}
const rssMb = () => Math.round(process.memoryUsage().rss / 2 ** 20);

const { alpha, beta, epsilon } = DEFAULTS;
const { vertices, edges } = trefoil(n);
const out: Record<string, unknown> = { n, mode, tol: tol ?? 'default', loadAvgStart: loadavg() };

if (mode === 'solve') {
    // The factory's own sequence (hierarchicalSaddle.ts, makeHierarchicalSaddle), phase by phase.
    const bvh = timed('bvh', () => buildBvh(vertices, edges, DEFAULT_LEAF_SIZE, epsilon));
    const bct = timed('bct', () => buildBct(bvh, {}));
    const source = { kind: 'lists', bct } as const;
    const A = timed('nearFieldA', () => prepareA(bvh, n, source, { alpha, beta, epsilon }));
    const op = makeHierarchicalOperator(A);
    const lowOrder = timed('lowOrder', () =>
        prepareLowOrder(vertices, edges, bvh, source, { alpha, beta, variant: 'vertex' }),
    );
    const precond = timed('precond', () =>
        makePreconditioner(vertices, edges, lowOrder, op.sigma, {}),
    );
    const normA2 = timed('normA2', () =>
        estimateNorm2((u, o) => applyAsigma(op, u, o), 3 * n),
    ).norm;

    // One matvec and one preconditioner apply, each the median of 5.
    let seed = 12345;
    const rnd = () => {
        seed = (seed * 1664525 + 1013904223) >>> 0;
        return seed / 2 ** 32 - 0.5;
    };
    const b = new Float64Array(3 * n);
    for (let c = 0; c < 3; c++) {
        let mean = 0;
        for (let i = 0; i < n; i++) {
            b[3 * i + c] = rnd();
            mean += b[3 * i + c];
        }
        mean /= n;
        for (let i = 0; i < n; i++) b[3 * i + c] -= mean;
    }
    const med = (f: () => void): number => {
        const t: number[] = [];
        for (let r = 0; r < 5; r++) {
            const t0 = performance.now();
            f();
            t.push(performance.now() - t0);
        }
        t.sort((x, y) => x - y);
        return Math.round(t[2] * 100) / 100;
    };
    const scratch = new Float64Array(3 * n);
    ms.matvecMedian = med(() => applyAsigma(op, b, scratch));
    ms.precondApplyMedian = med(() => precond.apply(b, scratch));

    const { C } = barycenterPhiAndC(vertices, edges, barycenterTarget(vertices, edges));
    const solver = timed('prepareC', () =>
        prepareSchur(op, precond, C, {
            normA2,
            ...(tol === undefined ? {} : { tol, etaAccept: 1 }),
        }),
    );
    const res = timed('solve', () => solver.solve(b));
    out.bctStats = bct.stats;
    out.prepareIters = solver.report.columns.map((c) => c.iters);
    out.solveIters = res.gmres.iters;
    out.solveMatvecs = res.gmres.matvecs;
    out.solveStop = res.gmres.stop;
    out.solveEta = res.gmres.eta;
    out.residual = res.residual;
} else {
    const dp = timed('disjointPairs', () => calculateDisjointPairs(edges));
    const set: ConstraintSet = [barycenterBlock(barycenterTarget(vertices, edges))];
    // `collectTimings` is the step's own opt-in phase breakdown (sobolev/phaseTimings.ts): it is what
    // shows whether the solve or the O(n²) energy/differential dominates.
    const base = {
        mode: 'analytical' as const,
        projectionMode: 'frozen' as const,
        collectTimings: true,
    };
    out.energyBefore = timed('oneEnergyEval', () =>
        calculateEnergy(vertices, edges, dp, alpha, beta, epsilon),
    );
    if (mode === 'step') {
        const saddle = timed('build', () =>
            makeHierarchicalSaddle(
                vertices,
                edges,
                dp,
                tol === undefined ? {} : { schur: { tol, etaAccept: 1 } },
            ),
        );
        const r = timed('step', () =>
            sobolevStepSet(vertices, edges, dp, set, { ...base, saddle }),
        );
        out.accepted = r.accepted;
        out.energy = r.energy;
        out.stats = r.stats;
        out.timings = r.timings;
    } else {
        const r = timed('step', () => sobolevStepSet(vertices, edges, dp, set, base));
        out.accepted = r.accepted;
        out.energy = r.energy;
        out.stats = r.stats;
        out.timings = r.timings;
    }
}

out.ms = ms;
out.rssMb = rssMb();
out.loadAvgEnd = loadavg();
console.log(JSON.stringify(out));
