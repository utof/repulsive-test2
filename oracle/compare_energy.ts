/**
 * Cross-language check: the repo's verified TS energy/gradients vs the Python
 * oracle's goldens on the same fixtures.
 *
 * Why: the oracle claims to replicate OUR energy convention (ordered pairs,
 * epsilon-after-norm, the extra 1/2). The TS side is the verified ground truth
 * for E and dE, so agreement here validates the oracle's transcription of §2–3
 * of the handoff — independent of any Sobolev math.
 *
 * Tolerances: energy is a plain double sum -> tight (1e-12 rel; languages
 * differ only in rounding order). Forward-difference dE divides ~1e-15-level
 * energy noise by h=1e-6, so FD-vs-FD agreement is capped near 1e-9 rel —
 * gate at 1e-6 (norm-relative). Analytical-vs-oracle-FD is bounded by the FD
 * truncation error itself: measured O(h) scaling (helix 3.9e-5 @ h=1e-6 ->
 * 3.95e-6 @ h=1e-7 -> noise floor @ 1e-8) confirms truncation, so gate 1e-4 at
 * h=1e-6. The results doc's §E prop 14 suggestion of 1e-5 is too tight for
 * forward differences on high-curvature configs (helix, knot).
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §E prop 14
 *
 * Run: bun oracle/compare_energy.ts
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import {
    calculateDisjointPairs,
    calculateEnergy,
    gradientAnalytical,
    gradientFiniteDiff,
} from '../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../src/core/testConfigs';

const FIXTURES = ['crossing', 'junction-y', 'helix', 'linked-rings', 'knot'];

function norm(v: number[]): number {
    return Math.sqrt(v.reduce((s, x) => s + x * x, 0));
}
function flatDiffNorm(a: Vec3[], b: number[][]): { diff: number; ref: number } {
    const d: number[] = [];
    const r: number[] = [];
    for (let i = 0; i < a.length; i++)
        for (let c = 0; c < 3; c++) {
            d.push(a[i][c] - b[i][c]);
            r.push(b[i][c]);
        }
    return { diff: norm(d), ref: norm(r) };
}

let failed = false;
function gate(name: string, rel: number, tol: number) {
    const ok = rel <= tol;
    if (!ok) failed = true;
    console.log(
        `  [${ok ? 'PASS' : 'FAIL'}] ${name}: rel ${rel.toExponential(2)} (tol ${tol.toExponential(0)})`,
    );
}

for (const name of FIXTURES) {
    const fx = JSON.parse(readFileSync(join(import.meta.dir, 'fixtures', `${name}.json`), 'utf8'));
    const golden = JSON.parse(
        readFileSync(join(import.meta.dir, 'golden', `${name}.json`), 'utf8'),
    );
    const vertices = fx.vertices as Vec3[];
    const edges = fx.edges as Edge[];
    const disjoint = calculateDisjointPairs(edges);

    console.log(`== ${name} ==`);

    const e = calculateEnergy(vertices, edges, disjoint, fx.alpha, fx.beta, fx.epsilon);
    gate(
        'TS energy == oracle energy',
        Math.abs(e - golden.energy) / Math.abs(golden.energy),
        1e-12,
    );

    const fd = gradientFiniteDiff(
        vertices,
        edges,
        disjoint,
        fx.alpha,
        fx.beta,
        fx.epsilon,
        golden.finite_difference_h,
    );
    const { diff: dFd, ref } = flatDiffNorm(fd, golden.dE);
    gate('TS FD dE == oracle FD dE', dFd / ref, 1e-6);

    const an = gradientAnalytical(vertices, edges, disjoint, fx.alpha, fx.beta, fx.epsilon);
    const { diff: dAn } = flatDiffNorm(an, golden.dE);
    gate('TS analytical dE ~ oracle FD dE', dAn / ref, 1e-4);
}

console.log(failed ? 'FAILED' : 'ALL CHECKS PASSED');
process.exit(failed ? 1 : 0);
