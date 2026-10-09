/**
 * Spec §11.2 task 5 of Phase 2H: the `SobolevOperator` wrapper of the hierarchical apply, the
 * structural `sigma = tr(A_bct)/n`, and the gates SPD / symmetry / trace ratio (§2.2, §2.3).
 *
 * Oracle: `.superpowers/sdd/2026-09-11-phase2h-queue/scratch/out-64-direction.json` `rows[]`
 * (@issue utof/repulsive-test2#64), fixture = the repo's `src/core/fixtures.ts trefoil(n)`,
 * UNnormalised (`run-64-dense.ts:50-56`), leaf=8, ALPHA=3, BETA=6, EPS=1e-10, endpoint4 near.
 * Its `sigma` column is `tr(A_exact)/n` (`run64_direction.py:148`), NOT the production sigma.
 *
 * Quantities are NOT interchangeable: `symDefect` here is `max|A-A^T|/max|A|` on the DENSIFIED
 * matrix (`run64_direction.py:223`), not the bilinear-form check of apply.test.ts (~4e-14).
 *
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §2.2, §2.3, §5.1, §11.2 task 5
 * @issue utof/repulsive-test2#64
 */
import { describe, expect, setDefaultTimeout, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { trefoil } from '../../../src/core/fixtures';
import { applyAScalar, prepareA } from '../../../src/core/sobolev/hierarchical/apply';
import { buildBct } from '../../../src/core/sobolev/hierarchical/bct';
import { buildBvh } from '../../../src/core/sobolev/hierarchical/bvh';
import { makeHierarchicalOperator, traceA } from '../../../src/core/sobolev/hierarchical/operator';
import { assembleAFlat } from '../../../src/core/sobolev/innerProduct';
import { calculateDisjointPairs } from '../../../src/core/tangentPointEnergy';

setDefaultTimeout(300_000);

const ALPHA = 3;
const BETA = 6;
const EPS = 1e-10;
const LEAF = 8;
const U_F64 = 2.220446049250313e-16;
const ORACLE = `${import.meta.dir}/../../../.superpowers/sdd/2026-09-11-phase2h-queue/scratch/out-64-direction.json`;

interface OracleRow {
    n: number;
    theta: number;
    sigma: number;
    traceRatio: number;
    symDefect: number;
    lamMinBctSigma: number;
}
const rows: OracleRow[] = JSON.parse(readFileSync(ORACLE, 'utf8')).rows;
const row = (n: number, theta: number): OracleRow => {
    const r = rows.find((x) => x.n === n && x.theta === theta);
    if (!r) throw new Error(`no oracle row ${n}/${theta}`);
    return r;
};
const log = (name: string, v: unknown) => console.log(`[task5] ${name} = ${v}`);
const sig3 = (x: number) => Number(x.toPrecision(3));

const cache = new Map<string, ReturnType<typeof build>>();
function build(n: number, theta: number) {
    const { vertices, edges } = trefoil(n);
    const bvh = buildBvh(vertices, edges, LEAF, EPS);
    const H = prepareA(
        bvh,
        n,
        { kind: 'lists', bct: buildBct(bvh, { theta }) },
        {
            alpha: ALPHA,
            beta: BETA,
            epsilon: EPS,
        },
    );
    return { H, vertices, edges };
}
function prep(n: number, theta: number) {
    const k = `${n}/${theta}`;
    let v = cache.get(k);
    if (!v) cache.set(k, (v = build(n, theta)));
    return v;
}

/** TEST ONLY, O(n) applies: the densified scalar `A_bct`, row-major, from basis-vector applies. */
const denseCache = new Map<string, Float64Array>();
function dense(n: number, theta: number): Float64Array {
    const k = `${n}/${theta}`;
    let A = denseCache.get(k);
    if (A) return A;
    const { H } = prep(n, theta);
    A = new Float64Array(n * n);
    const e = new Float64Array(n);
    const col = new Float64Array(n);
    for (let j = 0; j < n; j++) {
        e[j] = 1;
        applyAScalar(H, e, col);
        e[j] = 0;
        for (let i = 0; i < n; i++) A[i * n + j] = col[i];
    }
    denseCache.set(k, A);
    return A;
}

function symDefect(M: Float64Array, n: number): number {
    let num = 0;
    let den = 0;
    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) {
            num = Math.max(num, Math.abs(M[i * n + j] - M[j * n + i]));
            den = Math.max(den, Math.abs(M[i * n + j]));
        }
    return num / den;
}

/** TEST-ONLY dense Cholesky; throws on a non-positive pivot (the knock-out direction). */
function cholesky(A: Float64Array, n: number): Float64Array {
    const L = new Float64Array(n * n);
    for (let j = 0; j < n; j++) {
        let d = A[j * n + j];
        for (let k = 0; k < j; k++) d -= L[j * n + k] * L[j * n + k];
        if (!(d > 0)) throw new Error(`cholesky: non-positive pivot ${d} at ${j}`);
        const ljj = Math.sqrt(d);
        L[j * n + j] = ljj;
        for (let i = j + 1; i < n; i++) {
            let s = A[i * n + j];
            for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
            L[i * n + j] = s / ljj;
        }
    }
    return L;
}

const THETAS = [0.5, 0.25, 0.1];

describe('RULING 1: structural trace == densified diagonal sum', () => {
    for (const n of [240, 480])
        for (const theta of THETAS)
            test(`n=${n} theta=${theta}`, () => {
                const A = dense(n, theta);
                let ref = 0;
                for (let i = 0; i < n; i++) ref += A[i * n + i];
                const t = traceA(prep(n, theta).H);
                const rel = Math.abs(t - ref) / ref;
                log(`structuralTraceRel n=${n} theta=${theta}`, rel);
                expect(rel).toBeLessThanOrEqual(1e-13);
            });
});

describe('gate 5a: trace ratio tr(A_bct)/tr(A)', () => {
    const ratios: number[] = [];
    for (const n of [240, 480, 960])
        for (const theta of THETAS)
            test(`n=${n} theta=${theta}`, () => {
                const { H, vertices, edges } = prep(n, theta);
                const A = assembleAFlat(
                    vertices,
                    edges,
                    calculateDisjointPairs(edges),
                    ALPHA,
                    BETA,
                    EPS,
                );
                let trA = 0;
                for (let i = 0; i < n; i++) trA += A[i * n + i];
                const ratio = traceA(H) / trA;
                ratios.push(ratio);
                log(`traceRatio n=${n} theta=${theta}`, ratio);
                expect(sig3(ratio)).toBe(sig3(row(n, theta).traceRatio));
                expect(ratio).toBeGreaterThanOrEqual(0.989);
                expect(ratio).toBeLessThanOrEqual(1.0);
            });
});

describe('gate 5b: SPD after the sigma shift', () => {
    for (const n of [240, 480])
        for (const theta of THETAS)
            test(`cholesky n=${n} theta=${theta}`, () => {
                const A = dense(n, theta);
                const op = makeHierarchicalOperator(prep(n, theta).H);
                // A_sigma = A + (sigma/n) 11^T (spec §2.3; k1-calibration.py:228-230).
                const As = Float64Array.from(A);
                for (let i = 0; i < n * n; i++) As[i] += op.sigma / n;
                cholesky(As, n); // throws if not SPD
                // lamMinBct is genuinely negative in the oracle (RULING 2) -> never asserted > 0.
                // Knock-out: one negated diagonal entry must make the Cholesky fail.
                const bad = Float64Array.from(As);
                bad[5 * n + 5] = -bad[5 * n + 5];
                let threw = false;
                try {
                    cholesky(bad, n);
                } catch {
                    threw = true;
                }
                log(`choleskyOK/knockOutThrew n=${n} theta=${theta}`, `true/${threw}`);
                expect(threw).toBe(true);
            });
});

describe('gate 5c: symmetry defect (densified)', () => {
    for (const n of [240, 480])
        for (const theta of THETAS)
            test(`fidelity + 4u bar n=${n} theta=${theta}`, () => {
                const d = symDefect(dense(n, theta), n);
                log(
                    `symDefect n=${n} theta=${theta}`,
                    `${d} (bar ${4 * U_F64}, oracle ${row(n, theta).symDefect})`,
                );
                // Fidelity: 4 of 6 cells reproduce the artefact to 3 s.f. (in fact bit-equal); n=240
                // theta=0.5/0.25 differ by ~0.5e-16 (< 1 u_f64, one ulp of one entry: the port's op
                // order vs the spike's). 3 s.f. on a ~1-ulp roundoff quantity is not a stable bar,
                // so fidelity is |d - oracle| <= u_f64 (brief RULING 3 amended; see task-5 report).
                expect(Math.abs(d - row(n, theta).symDefect)).toBeLessThanOrEqual(U_F64);
                expect(d).toBeLessThanOrEqual(4 * U_F64);
            });
});

describe('gate 5d: sigma is the HIERARCHICAL trace', () => {
    test('n=960 theta=0.5 differs from tr(A_exact)/n by >= 0.5%', () => {
        const { H, vertices, edges } = prep(960, 0.5);
        const op = makeHierarchicalOperator(H);
        const A = assembleAFlat(vertices, edges, calculateDisjointPairs(edges), ALPHA, BETA, EPS);
        let trA = 0;
        for (let i = 0; i < 960; i++) trA += A[i * 960 + i];
        const exactSigma = trA / 960;
        const diff = (exactSigma - op.sigma) / exactSigma;
        log('sigma_bct n=960 theta=0.5', op.sigma);
        log('sigma_exact n=960 theta=0.5', exactSigma);
        log('sigma relative difference', diff);
        // The artefact's `sigma` column IS the exact one (RULING 4); production = traceRatio * it.
        expect(Math.abs(exactSigma - row(960, 0.5).sigma) / exactSigma).toBeLessThan(1e-9);
        expect(sig3(op.sigma)).toBe(sig3(row(960, 0.5).traceRatio * row(960, 0.5).sigma));
        expect(diff).toBeGreaterThanOrEqual(0.005);
    });
});

describe('SobolevOperator.applyA', () => {
    test('block-diagonal over 3 components, equals applyA of apply.ts', () => {
        const n = 240;
        const { H } = prep(n, 0.25);
        const op = makeHierarchicalOperator(H);
        expect(op.n).toBe(n);
        const u = Float64Array.from({ length: 3 * n }, (_, i) => Math.sin(i * 0.37));
        const out = new Float64Array(3 * n);
        op.applyA(u, out);
        const s = new Float64Array(n);
        for (let c = 0; c < 3; c++) {
            applyAScalar(H, u.subarray(c * n, (c + 1) * n), s);
            for (let i = 0; i < n; i++) expect(out[c * n + i]).toBe(s[i]);
        }
    });
});
