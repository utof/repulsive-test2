/**
 * Spec §11.2 task 4 of Phase 2H, part 2: the 12-column hierarchical apply, f64 and f32 as accumulator
 * variants of ONE traversal, and the four gates this task carries:
 *
 *   1. spec §5.4 item 4 — the bit-level pin of the chi=0 `endpoint4` apply against the repo's
 *      `assembleAFlat` (the ONLY standing check that the near field is the production quadrature);
 *   2. spec §11.1 constraint 3 — linearity of the apply, before anything is built from
 *      basis-vector applies;
 *   3. spec §2.2 — `A_bct · 1 = 0` EXACTLY for the apply (structural, see the test);
 *   4. spec §7.2 H1-a (delta_compress monotone down the theta ladder; inverted-admissibility
 *      knock-out RED; the circle tie flagged) and H1-b (the precision band, TWO-sided per the
 *      brief `.superpowers/sdd/2026-10-09-phase2h-bct-gmres-design/task-4-brief.md`, and the f64 / f32 / f16 three-rung ladder).
 *
 * Oracle artefact: `.superpowers/sdd/2026-09-11-phase2h-queue/scratch/out-accuracy.json`
 * (git-ignored spike output of `run-accuracy.ts`), `params {ALPHA: 3, BETA: 6, EPS: 1e-10,
 * QUAD: "endpoint4"}`, keys `rows[].{relErrRandom, f32NoiseHi, f32NoiseLow, nearPairs, nFar}`.
 * Its `reference` line: "exactA(quad='endpoint4'), pinned to src/ assembleAFlat at ~2e-15".
 * Conditions of every oracle: fixtures at bbox max extent 1 centred at the origin
 * (`scratch/curves.ts:49-66`; the SCALE PRECONDITION of `bvh.ts`), leaf=8, thetaT off, the
 * random rhs `seeded(n, 12345)` of `run-accuracy.ts:37-45`.
 *
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §2.2, §3.5, §5.4, §5.6, §7.2, §11.1, §11.2 task 4, §11.3
 */
import { describe, expect, setDefaultTimeout, test } from 'bun:test';
import {
    applyA,
    applyAScalar,
    applyK,
    type BlockSource,
    type HierarchicalA,
    makeWorkspace,
    precomputeFar,
    prepareA,
} from '../../../src/core/sobolev/hierarchical/apply';
import { buildBct, thetaLadder } from '../../../src/core/sobolev/hierarchical/bct';
import { type Bvh, buildBvh, edgeData } from '../../../src/core/sobolev/hierarchical/bvh';
import {
    type AccumulatorVariant,
    exponents,
    F16_NEAR_KNOCKOUT,
    F32,
    F64,
    precomputeNear,
} from '../../../src/core/sobolev/hierarchical/nearField';
import { assembleAFlat } from '../../../src/core/sobolev/innerProduct';
import { calculateDisjointPairs } from '../../../src/core/tangentPointEnergy';
import { type Curve, FIXTURES, type FixtureName } from './fixtures';

/** `params` of `out-accuracy.json`. */
const EPS = 1e-10;
const ALPHA = 3;
const BETA = 6;
const LEAF = 8;
/** IEEE binary32 unit roundoff, as spec §7.2 H1-b writes it. */
const U_F32 = 5.96e-8;

// The chi=0 reference is an O(m^2) apply; the first test to touch one for a fixture pays ~5 s at
// n=960 (per-op rounding calls, apply.ts module TSDoc), well past bun's 5 s default.
setDefaultTimeout(60_000);

// ---------------------------------------------------------------------------------------------
// Harness helpers.
// ---------------------------------------------------------------------------------------------

const curveCache = new Map<string, Curve>();
function curve(fixture: FixtureName, n: number): Curve {
    const key = `${fixture}/${n}`;
    let c = curveCache.get(key);
    if (!c) {
        c = FIXTURES[fixture](n);
        curveCache.set(key, c);
    }
    return c;
}

const bvhCache = new Map<string, Bvh>();
function tree(fixture: FixtureName, n: number, leafSize = LEAF): Bvh {
    const key = `${fixture}/${n}/${leafSize}`;
    let bvh = bvhCache.get(key);
    if (!bvh) {
        const { vertices, edges } = curve(fixture, n);
        bvh = buildBvh(vertices, edges, leafSize, EPS);
        bvhCache.set(key, bvh);
    }
    return bvh;
}

/** `run-accuracy.ts:37-45` / `run-verify.ts:37-45` — the random rhs of every `relErrRandom`. */
function seeded(n: number, seed: number): Float64Array {
    let s = seed >>> 0;
    const u = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
        u[i] = s / 4294967296 - 0.5;
    }
    return u;
}

/** `scratch/operators.ts:339` `relErr`: `||a - b||_2 / ||a||_2`, the reference in the first slot. */
function relErr(a: ArrayLike<number>, b: ArrayLike<number>): number {
    let num = 0;
    let den = 0;
    for (let i = 0; i < a.length; i++) {
        const d = a[i] - b[i];
        num += d * d;
        den += a[i] * a[i];
    }
    return den > 0 ? Math.sqrt(num / den) : Math.sqrt(num);
}

function maxAbs(a: ArrayLike<number>): number {
    let mx = 0;
    for (let i = 0; i < a.length; i++) if (Math.abs(a[i]) > mx) mx = Math.abs(a[i]);
    return mx;
}

/** The production operator, densely: `assembleAFlat(...)` on the fixture. Cached — O(n^2) build. */
const denseCache = new Map<string, Float64Array>();
function denseA(fixture: FixtureName, n: number): Float64Array {
    const key = `${fixture}/${n}`;
    let A = denseCache.get(key);
    if (!A) {
        const { vertices, edges } = curve(fixture, n);
        A = assembleAFlat(vertices, edges, calculateDisjointPairs(edges), ALPHA, BETA, EPS);
        denseCache.set(key, A);
    }
    return A;
}

function denseMv(A: Float64Array, n: number, u: ArrayLike<number>): Float64Array {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        let acc = 0;
        const b = i * n;
        for (let j = 0; j < n; j++) acc += A[b + j] * u[j];
        out[i] = acc;
    }
    return out;
}

type Mode = 'pre' | 'inline';

const hCache = new Map<string, HierarchicalA>();
/**
 * Prepared operator, cached per configuration. `theta === 0` streams (`traverseBct`, nothing
 * materialised — spec §7.1); `theta > 0` walks `buildBct`'s lists, with the near field
 * precomputed (`'pre'`) or evaluated inline (`'inline'`).
 */
function prepared(
    fixture: FixtureName,
    n: number,
    theta: number,
    variant: AccumulatorVariant = F64,
    mode: Mode = 'pre',
    extra: { invertAdmissibility?: boolean } = {},
): HierarchicalA {
    const key = `${fixture}/${n}/${theta}/${variant.name}/${mode}/${JSON.stringify(extra)}`;
    let H = hCache.get(key);
    if (!H) {
        const bvh = tree(fixture, n);
        const source: BlockSource =
            theta === 0
                ? { kind: 'stream', params: { theta: 0, ...extra } }
                : { kind: 'lists', bct: buildBct(bvh, { theta, ...extra }) };
        H = prepareA(bvh, n, source, {
            alpha: ALPHA,
            beta: BETA,
            epsilon: EPS,
            variant,
            precomputeNear: mode === 'pre',
            // The knock-out instruments are opted into HERE, per configuration, never by default.
            knockOut: extra.invertAdmissibility === true || variant === F16_NEAR_KNOCKOUT,
        });
        hCache.set(key, H);
    }
    return H;
}

const uCache = new Map<string, Float64Array>();
/** The random rhs of `out-accuracy.json` (`seeded(n, 12345)`), per n. */
function rhs(n: number, seed = 12345): Float64Array {
    const key = `${n}/${seed}`;
    let u = uCache.get(key);
    if (!u) {
        u = seeded(n, seed);
        uCache.set(key, u);
    }
    return u;
}

const exactCache = new Map<string, Float64Array>();
/** `A_exact u` — the chi=0 streaming apply, f64, on `rhs(n)`. Cached: O(m^2) per call. */
function exactAu(fixture: FixtureName, n: number): Float64Array {
    const key = `${fixture}/${n}`;
    let out = exactCache.get(key);
    if (!out) {
        out = new Float64Array(n);
        applyAScalar(prepared(fixture, n, 0), rhs(n), out);
        exactCache.set(key, out);
    }
    return out;
}

function scalarApply(H: HierarchicalA, u: ArrayLike<number>): Float64Array {
    const out = new Float64Array(H.n);
    applyAScalar(H, u, out);
    return out;
}

function vectorApply(H: HierarchicalA, U: ArrayLike<number>): Float64Array {
    const out = new Float64Array(3 * H.n);
    applyA(H, U, out);
    return out;
}

/**
 * `delta_compress(theta)` of spec §7.2: `||A_exact u - A_theta u|| / ||A_exact u||`, both f64,
 * the chi=0 streaming apply as the reference — exactly `run-accuracy.ts:119`'s `relErrRandom`.
 */
function deltaCompress(
    fixture: FixtureName,
    n: number,
    theta: number,
    extra: { invertAdmissibility?: boolean } = {},
): number {
    return relErr(
        exactAu(fixture, n),
        scalarApply(prepared(fixture, n, theta, F64, 'pre', extra), rhs(n)),
    );
}

/**
 * TEST-ONLY midpoint (Eq. 32 one-point) O(m^2) double sum, transcribed from
 * `scratch/operators.ts:295-306` (`exactA(quad='midpoint')`). It exists ONLY to re-measure spec
 * §5.4's "33.9 %" and lives here, not in `src/`, so the production near field has no midpoint
 * switch to flip (§5.4 item 1 mandates `endpoint4`).
 */
function midpointAu(fixture: FixtureName, n: number, u: Float64Array): Float64Array {
    const { vertices, edges } = curve(fixture, n);
    const { m, P, T, ell } = edgeData(vertices, edges, EPS);
    const { pHi, pLow } = exponents(ALPHA, BETA);
    const g = new Float64Array(3 * m);
    const ev = new Float64Array(m);
    for (let I = 0; I < m; I++) {
        const [i1, i2] = edges[I];
        const d = (u[i2] - u[i1]) / ell[I];
        g[3 * I] = d * T[3 * I];
        g[3 * I + 1] = d * T[3 * I + 1];
        g[3 * I + 2] = d * T[3 * I + 2];
        ev[I] = 0.5 * (u[i1] + u[i2]);
    }
    const out = new Float64Array(n);
    for (let I = 0; I < m; I++) {
        const [ia, ib] = edges[I];
        let a1 = 0;
        let gx = 0;
        let gy = 0;
        let gz = 0;
        let b1 = 0;
        let e1 = 0;
        for (let J = 0; J < m; J++) {
            const [ja, jb] = edges[J];
            if (ia === ja || ia === jb || ib === ja || ib === jb) continue;
            const dx = P[3 * I] - P[3 * J];
            const dy = P[3 * I + 1] - P[3 * J + 1];
            const dz = P[3 * I + 2] - P[3 * J + 2];
            const lij = ell[I] * ell[J];
            const d = Math.sqrt(dx * dx + dy * dy + dz * dz) + EPS;
            const kHi = (2 * lij) / d ** pHi;
            const c1x = T[3 * I + 1] * dz - T[3 * I + 2] * dy;
            const c1y = T[3 * I + 2] * dx - T[3 * I] * dz;
            const c1z = T[3 * I] * dy - T[3 * I + 1] * dx;
            const c2x = T[3 * J + 1] * dz - T[3 * J + 2] * dy;
            const c2y = T[3 * J + 2] * dx - T[3 * J] * dz;
            const c2z = T[3 * J] * dy - T[3 * J + 1] * dx;
            const n1 = Math.sqrt(c1x * c1x + c1y * c1y + c1z * c1z) + EPS;
            const n2 = Math.sqrt(c2x * c2x + c2y * c2y + c2z * c2z) + EPS;
            const kLow = (lij * (n1 * n1 + n2 * n2)) / d ** pLow;
            a1 += kHi;
            gx += kHi * g[3 * J];
            gy += kHi * g[3 * J + 1];
            gz += kHi * g[3 * J + 2];
            b1 += kLow;
            e1 += kLow * ev[J];
        }
        const hx = a1 * g[3 * I] - gx;
        const hy = a1 * g[3 * I + 1] - gy;
        const hz = a1 * g[3 * I + 2] - gz;
        const r = (hx * T[3 * I] + hy * T[3 * I + 1] + hz * T[3 * I + 2]) / ell[I];
        out[ib] += r;
        out[ia] -= r;
        const h = b1 * ev[I] - e1;
        out[ia] += 0.5 * h;
        out[ib] += 0.5 * h;
    }
    return out;
}

/** Measured values this file prints, so the report can quote them (spec §7.12: value, not verdict). */
function record(label: string, value: number | string): void {
    console.log(`[task4] ${label} = ${typeof value === 'number' ? value.toExponential(6) : value}`);
}

// ---------------------------------------------------------------------------------------------
// 1. Spec §5.4 item 4 — the bit-level pin (§11.1 constraint 2: before any accuracy number).
// ---------------------------------------------------------------------------------------------

describe('§5.4 item 4: chi=0 endpoint4 apply vs assembleAFlat (unit gate, unit-bbox fixtures, EPS=1e-10)', () => {
    // Bar: the artefact's own reference line says the spike pinned at ~2e-15 (and §5.4's prose
    // says ~1.4e-15); 4e-15 is twice the archived order. Both sides are f64 sums of the same
    // terms in different orders, so the value is f64 reduction noise, NOT an approximation.
    const BAR = 4e-15;
    for (const [fixture, n] of [
        ['torusKnot', 400],
        ['torusKnot', 960],
        ['openHelix', 400],
    ] as [FixtureName, number][]) {
        test(`${fixture} n=${n}: relErr(assembleAFlat*u, exactA(endpoint4)*u) < ${BAR}`, () => {
            const u = rhs(n);
            const repo = denseMv(denseA(fixture, n), n, u);
            const ex = exactAu(fixture, n);
            const e = relErr(repo, ex);
            record(`pin ${fixture} n=${n} relErr(assembleAFlat u, exactA u)`, e);
            expect(e).toBeLessThan(BAR);
            // The impossible value: exactly 0 would mean the dense product and the hierarchical
            // sum ran the same arithmetic in the same order, which they do not.
            expect(e).toBeGreaterThan(0);
            expect(maxAbs(ex)).toBeGreaterThan(0);
        });
    }

    test('torusKnot n=400: the 12-column vector apply is diag(A, A, A) in the coordinate-block layout of layout.ts (expandBlockDiag)', () => {
        const n = 400;
        const A = denseA('torusKnot', n);
        const U = new Float64Array(3 * n);
        for (let c = 0; c < 3; c++) U.set(seeded(n, 100 + c), c * n);
        const out = vectorApply(prepared('torusKnot', n, 0), U);
        for (let c = 0; c < 3; c++) {
            const ref = denseMv(A, n, U.subarray(c * n, (c + 1) * n));
            const e = relErr(ref, out.subarray(c * n, (c + 1) * n));
            record(`pin vector block c=${c} n=${n}`, e);
            expect(e).toBeLessThan(BAR);
        }
    });
});

// ---------------------------------------------------------------------------------------------
// 2. Spec §5.4 — re-measure the endpoint4-vs-midpoint gap and RECORD it (no artefact exists).
// ---------------------------------------------------------------------------------------------

describe('§5.4: endpoint4 vs midpoint on the O(m^2) double sum, no BCT (re-measured, recorded, not pinned to 33.9 %)', () => {
    // The spec's 33.9 % (n=960) / 32.6 % (n=400) are console-only prose; the values this test
    // prints are the first archived ones. Only the ORDER is asserted: a double-digit percentage
    // is what makes §5.4 item 1 (endpoint4 mandated) load-bearing. A midpoint near field would
    // also fail the pin above by this same amount.
    for (const n of [400, 960]) {
        test(`torusKnot n=${n}`, () => {
            const u = rhs(n);
            const ex4 = exactAu('torusKnot', n);
            const mid = midpointAu('torusKnot', n, u);
            const gap = relErr(ex4, mid);
            record(
                `endpoint4-vs-midpoint gap torusKnot n=${n} (relErr, endpoint4 as reference)`,
                gap,
            );
            record(
                `endpoint4-vs-midpoint gap torusKnot n=${n} (percent)`,
                `${(100 * gap).toFixed(2)} %`,
            );
            expect(gap).toBeGreaterThan(0.1);
            expect(gap).toBeLessThan(1);
        });
    }
});

// ---------------------------------------------------------------------------------------------
// 3. Spec §11.1 constraint 3 — linearity, before anything built from basis-vector applies.
// ---------------------------------------------------------------------------------------------

describe('§11.1 constraint 3: the apply is linear, A(a u + b v) = a A u + b A v', () => {
    const n = 960;
    const a = 0.7;
    const b = -1.3;
    // f64 bar: reduction-order noise of ~100 near terms + the far percolation per entry, far
    // below 1e-13; a data-dependent branch or a per-vector renormalisation would show at O(1).
    const F64_BAR = 1e-13;

    function combo(u: Float64Array, v: Float64Array): Float64Array {
        const w = new Float64Array(u.length);
        for (let i = 0; i < u.length; i++) w[i] = a * u[i] + b * v[i];
        return w;
    }
    function axpby(x: Float64Array, y: Float64Array): Float64Array {
        const w = new Float64Array(x.length);
        for (let i = 0; i < x.length; i++) w[i] = a * x[i] + b * y[i];
        return w;
    }

    for (const [label, theta] of [
        ['theta=0.5 lists+precomputed', 0.5],
        ['chi=0 streaming', 0],
    ] as [string, number][]) {
        test(`${label}, f64, scalar and vector apply, torusKnot n=${n}`, () => {
            const H = prepared('torusKnot', n, theta);
            const u = rhs(n);
            const v = rhs(n, 999);
            const lhs = scalarApply(H, combo(u, v));
            const rhsv = axpby(scalarApply(H, u), scalarApply(H, v));
            const e = relErr(lhs, rhsv);
            record(`linearity scalar ${label}`, e);
            expect(e).toBeLessThan(F64_BAR);
            if (theta === 0) return; // the vector chi=0 apply is 12 O(m^2) columns; one is enough
            const U = new Float64Array(3 * n);
            const V = new Float64Array(3 * n);
            for (let c = 0; c < 3; c++) {
                U.set(seeded(n, 10 + c), c * n);
                V.set(seeded(n, 20 + c), c * n);
            }
            const L = vectorApply(H, combo(U, V));
            const R = axpby(vectorApply(H, U), vectorApply(H, V));
            const ev = relErr(L, R);
            record(`linearity vector ${label}`, ev);
            expect(ev).toBeLessThan(F64_BAR);
        });
    }

    test('f32 variant: linear to f32 roundoff (reported; bar 1e-4 is ~10x the H1-b band ceiling)', () => {
        const H = prepared('torusKnot', n, 0.5, F32);
        const u = rhs(n);
        const v = rhs(n, 999);
        const e = relErr(scalarApply(H, combo(u, v)), axpby(scalarApply(H, u), scalarApply(H, v)));
        record('linearity scalar theta=0.5 f32', e);
        expect(e).toBeLessThan(1e-4);
        expect(e).toBeGreaterThan(0); // f32 cannot be exactly linear
    });
});

// ---------------------------------------------------------------------------------------------
// 4. Spec §2.2 — A_bct * 1 = 0 EXACTLY for the apply (quantity (i) of §2.2's three).
// ---------------------------------------------------------------------------------------------

describe('§2.2: A_bct * 1 = 0 EXACTLY for the apply — all variants, all block sources', () => {
    // WHY AN EXACT ZERO IS THE RIGHT ASSERTION HERE (§11.3 rule 2): it is structural, not
    // luck. High order: `du = (u[i2] - u[i1]) / ell` is `(1 - 1) / ell = 0` exactly, so `g = 0`
    // and every product in K g is `k * 0 = 0`, and `diag(K1) g - K g = 0 - 0`. Low order:
    // `ev = 0.5 * (1 + 1) = 1` exactly, so `K0 ev` is the SAME sequence of operations as the
    // `K0 1` that produced `k1Low` in `prepareA` — bit-identical — and `k1Low * 1 - ke` is `x - x`.
    // That holds in f32 and in the f16 knock-out for the same reason (same variant both times).
    // §2.2 quantity (ii), the densified row sum, is NOT zero — see the last test.
    const n = 960;
    const cases: [string, number, AccumulatorVariant, Mode][] = [
        ['theta=0.5 lists precomputed f64', 0.5, F64, 'pre'],
        ['theta=0.5 lists inline f64', 0.5, F64, 'inline'],
        ['theta=0.5 lists precomputed f32', 0.5, F32, 'pre'],
        ['theta=0.5 lists precomputed f16-near knock-out', 0.5, F16_NEAR_KNOCKOUT, 'pre'],
        ['theta=0.25 lists precomputed f64', 0.25, F64, 'pre'],
        ['chi=0 streaming f64', 0, F64, 'inline'],
    ];
    for (const [label, theta, variant, mode] of cases) {
        test(`${label}: max_i |(A 1)_i| === 0, scalar and vector apply`, () => {
            const H = prepared('torusKnot', n, theta, variant, mode);
            const ones = new Float64Array(n).fill(1);
            const o = scalarApply(H, ones);
            expect(maxAbs(o)).toBe(0);
            if (theta === 0) return;
            const O = vectorApply(H, new Float64Array(3 * n).fill(1));
            expect(maxAbs(O)).toBe(0);
            // And the operator is not the zero operator: the same apply on the random rhs is O(1).
            expect(maxAbs(scalarApply(H, rhs(n)))).toBeGreaterThan(1e-3);
        });
    }

    test('openHelix n=400 (open chain, m = n - 1): exact zero too', () => {
        const H = prepared('openHelix', 400, 0.5);
        expect(maxAbs(scalarApply(H, new Float64Array(400).fill(1)))).toBe(0);
    });

    test('§2.2 quantity (ii): row sums of the DENSIFIED operator are NOT zero (n=128, theta=0.5) — the densifier is not faking it', () => {
        // Densify by 128 basis-vector applies (legitimate only because section 3 passed).
        const n2 = 128;
        const H = prepared('torusKnot', n2, 0.5);
        const cols = new Float64Array(n2 * n2);
        const e = new Float64Array(n2);
        for (let j = 0; j < n2; j++) {
            e.fill(0);
            e[j] = 1;
            const col = scalarApply(H, e);
            for (let i = 0; i < n2; i++) cols[i * n2 + j] = col[i];
        }
        let rowMax = 0;
        let normMax = 0;
        for (let i = 0; i < n2; i++) {
            let s = 0;
            for (let j = 0; j < n2; j++) {
                s += cols[i * n2 + j];
                if (Math.abs(cols[i * n2 + j]) > normMax) normMax = Math.abs(cols[i * n2 + j]);
            }
            if (Math.abs(s) > rowMax) rowMax = Math.abs(s);
        }
        record('densified max|row sum| n=128 theta=0.5', rowMax);
        record('densified max|row sum| / max|A_ij|', rowMax / normMax);
        expect(rowMax).toBeGreaterThan(0);
        expect(rowMax / normMax).toBeLessThan(1e-12); // roundoff, not structure
        // While the apply on the ones vector is exactly 0 on the same operator:
        expect(maxAbs(scalarApply(H, new Float64Array(n2).fill(1)))).toBe(0);
    });
});

// ---------------------------------------------------------------------------------------------
// 5. Spec §7.2 H1-a — delta_compress down the theta ladder; knock-outs; the circle tie.
// ---------------------------------------------------------------------------------------------

describe('§7.2 H1-a: delta_compress — out-accuracy.json relErrRandom (torusKnot n=960 leaf=8, thetaT off, unit-bbox, rhs seeded(960, 12345))', () => {
    /** `rows[].relErrRandom` at `rows[].theta`, descending. */
    const ladder: [number, number][] = [
        [1, 0.04130197819344654],
        [0.8, 0.040773887456738066],
        [0.6, 0.04046258813571584],
        [0.5, 0.040387640897987256],
        [0.4, 0.01354350714410973],
        [0.3, 0.007533558332768233],
        [0.25, 0.00675058883802985],
        [0.2, 0.0016126636887620503],
        [0.15, 0.0008575484106994909],
        [0.1, 0.0003829798999774701],
    ];
    // Tolerance: both the artefact and this port are f64 evaluations of the same operator on the
    // same rhs with the same block ORDER (bct.test.ts pins maxStackDepth), differing only in
    // reduction order against an O(m^2) reference summed in a different order — so the agreement
    // is at f64 reduction noise (~1e-13 relative on delta). 1e-6 relative keeps a wide margin
    // while still rejecting any real change (a wrong kernel factor or quadrature moves delta
    // by 10-100 %).
    const REL_TOL = 1e-6;

    for (const [theta, oracle] of ladder) {
        test(`theta=${theta}: delta_compress reproduces ${oracle.toExponential(4)}`, () => {
            const d = deltaCompress('torusKnot', 960, theta);
            record(`delta_compress torusKnot n=960 theta=${theta}`, d);
            expect(Math.abs(d - oracle) / oracle).toBeLessThan(REL_TOL);
        });
    }

    test('error vs n at theta=0.5: torusKnot n=400 (row: 0.06834912244082188)', () => {
        const d400 = deltaCompress('torusKnot', 400, 0.5);
        record('delta_compress torusKnot n=400 theta=0.5', d400);
        expect(Math.abs(d400 - 0.06834912244082188) / 0.06834912244082188).toBeLessThan(REL_TOL);
    });

    test('error vs n at theta=0.5: torusKnot n=2000 (row: 0.015248951384356373; ~4 s, O(m^2) reference)', () => {
        const d2000 = deltaCompress('torusKnot', 2000, 0.5);
        record('delta_compress torusKnot n=2000 theta=0.5', d2000);
        expect(Math.abs(d2000 - 0.015248951384356373) / 0.015248951384356373).toBeLessThan(REL_TOL);
    });

    test('H1-a bar: delta_compress(theta_{k+1}) <= 1.001 * delta_compress(theta_k) on {0.5, 0.25, 0.1}; the INVALID tie predicate does not fire', () => {
        const thetas = [0.5, 0.25, 0.1];
        const d = thetas.map((t) => deltaCompress('torusKnot', 960, t));
        for (let k = 0; k + 1 < thetas.length; k++) {
            record(`H1-a ratio ${thetas[k + 1]}/${thetas[k]}`, d[k + 1] / d[k]);
            expect(d[k + 1]).toBeLessThanOrEqual(1.001 * d[k]);
        }
        // INVALID predicate (§7.2 table, row 1): equal delta at 0.5 and 0.1 where kappa_near
        // differs (89 538 vs 218 844 near pairs) would mean theta was never plumbed.
        const lad = thetaLadder(tree('torusKnot', 960), thetas, {});
        expect(lad.rungs[0].stats.nearPairs).toBe(89538);
        expect(lad.rungs[2].stats.nearPairs).toBe(218844);
        expect(lad.anyTie).toBe(false);
        expect(d[0]).not.toBe(d[2]);
    });

    test('knock-out (2): invertAdmissibility => delta_compress RISES as theta falls => H1-a RED', () => {
        // The switch is bct.ts's (`invertAdmissibility`, spec §7.2 knock-out 2); the RED is asserted here because
        // delta_compress is a ratio of two applies. The knock-out must TRIP: a gate never seen
        // failing is not a verified gate (memory gates-name-their-impossible-values).
        const thetas = [0.5, 0.25, 0.1];
        const d = thetas.map((t) =>
            deltaCompress('torusKnot', 960, t, { invertAdmissibility: true }),
        );
        thetas.forEach((t, k) => record(`delta_compress INVERTED theta=${t}`, d[k]));
        const green = d[1] <= 1.001 * d[0] && d[2] <= 1.001 * d[1];
        expect(green).toBe(false);
        // "rises as theta falls": the last rung is worse than the first.
        expect(d[2]).toBeGreaterThan(d[0]);
        // And the normal operator at the same thetas is monotone (same code, flag off).
        const ok = thetas.map((t) => deltaCompress('torusKnot', 960, t));
        expect(ok[1] <= 1.001 * ok[0] && ok[2] <= 1.001 * ok[1]).toBe(true);
    });

    test('must-NOT-trip (3): circle n=960, 0.5 -> 0.25 — kappa_near tie FLAGGED, delta_compress GREEN, both values recorded', () => {
        // PROVENANCE: spec §7.2 labels this case "MEASURED", but no artefact carries circle at
        // theta=0.25 (out-accuracy.json has circle at 0.5 only; H1-a's "15/15 cells" are
        // torusKnot). Task 3 produced the first theta=0.25 block counts; the delta_compress values
        // below are the first ones, MEASURED by this task on 2026-10-09: 8.640671e-03 at 0.5 (which
        // is the artefact's 0.5 row) and 8.461128e-03 at 0.25, ratio 0.979. The tie is on the
        // NEAR side only — nearPairs 21 598 at both theta, near lists bit-identical — while the
        // far side differs (nFar 748 -> 904, adjacencyVetoed 480 -> 0), which is why delta_compress
        // moves at all. Task-4 brief ruling 2: it is MEASURED, not asserted equal.
        const lad = thetaLadder(tree('circle', 960), [0.5, 0.25], {});
        expect(lad.rungs[0].stats.nearPairs).toBe(21598);
        expect(lad.rungs[1].stats.nearPairs).toBe(21598);
        expect([lad.rungs[0].stats.nFar, lad.rungs[1].stats.nFar]).toEqual([748, 904]);
        expect(lad.ties).toEqual([[0.5, 0.25]]);
        const d5 = deltaCompress('circle', 960, 0.5);
        const d25 = deltaCompress('circle', 960, 0.25);
        record('delta_compress circle n=960 theta=0.5', d5);
        record('delta_compress circle n=960 theta=0.25', d25);
        record('circle 0.25/0.5 ratio', d25 / d5);
        // out-accuracy.json circle theta=0.5 leaf=8 relErrRandom:
        expect(Math.abs(d5 - 0.008640671046366358) / 0.008640671046366358).toBeLessThan(REL_TOL);
        expect(d25).toBeLessThanOrEqual(1.001 * d5); // GREEN with the tie flag
    });
});

// ---------------------------------------------------------------------------------------------
// 6. Spec §7.2 H1-b — the precision band (two-sided, per the brief
//    `.superpowers/sdd/2026-10-09-phase2h-bct-gmres-design/task-4-brief.md`) and the ladder.
// ---------------------------------------------------------------------------------------------

describe('§7.2 H1-b: delta_precision band and the f64 / f32 / f16 ladder (torusKnot n=960 leaf=8, theta=0.5 AND theta=0.25)', () => {
    const n = 960;

    /** `delta_precision` = relErr(A_f64,theta u, A_variant,theta u) at the SAME theta. */
    function deltaPrecision(
        theta: number,
        variant: AccumulatorVariant,
        mode: Mode = 'pre',
    ): number {
        const ref = scalarApply(prepared('torusKnot', n, theta), rhs(n));
        return relErr(ref, scalarApply(prepared('torusKnot', n, theta, variant, mode), rhs(n)));
    }

    /** `10 * (kappa_near + kappa_far) * u_f32` from the prepared operator's OWN counts. */
    function h1bBar(theta: number): number {
        const H = prepared('torusKnot', n, theta);
        return 10 * (H.stats.nearPairs / H.m + H.stats.nFar / H.m) * U_F32;
    }

    // theta=0.5 is the anchor the MEASURED f32 noise was taken against (f32NoiseHi 1.0639e-06,
    // f32NoiseLow 3.0024e-06, out-accuracy.json); theta=0.25 is PRODUCTION after gate H6 read RED
    // (spec §7.9, 2026-10-09). A precision gate anchored only at a theta the product no longer
    // uses is a gate on nothing, so the band is asserted at both.
    const BARS: [number, number, number, number][] = [
        // [theta, nearPairs, nFar, bar]:
        //   0.5:  kappa_near = 89538/960 = 93.26875 (§7.2 "93.3"), kappa_far = 3530/960 = 3.677
        //         (§7.2 "3.68") => 10 * 96.946 * 5.96e-08 = 5.778e-05, written 5.78e-05.
        //   0.25: kappa_near = 178320/960 = 185.75 (out-accuracy.json nearPairsPerEdge), kappa_far
        //         = 5716/960 = 5.954 => 10 * 191.704 * 5.96e-08 = 1.1426e-04.
        [0.5, 89538, 3530, 5.78e-5],
        [0.25, 178320, 5716, 1.1426e-4],
    ];

    for (const [theta, nearPairs, nFar, expectedBar] of BARS) {
        test(`theta=${theta} — the bar: 10 * (kappa_near + kappa_far) * u_f32 from the MEASURED counts = ${expectedBar}`, () => {
            const H = prepared('torusKnot', n, theta);
            expect(H.stats.nearPairs).toBe(nearPairs);
            expect(H.stats.nFar).toBe(nFar);
            const bar = h1bBar(theta);
            record(`H1-b bar theta=${theta}`, bar);
            expect(Math.abs(bar - expectedBar) / expectedBar).toBeLessThan(1e-3);
        });

        test(`theta=${theta} — rung 2, f32 accumulator: 1e-07 <= delta_precision <= bar (two-sided; the lower bound names the unplumbed-f32 value)`, () => {
            // Upper bound: §7.2's bar. Lower bound (the brief, see section header): a decade under the MEASURED f32
            // apply noise at n=960, because an "f32" path reporting ~1e-16 is an f64 simulation
            // that was never rounded — the defect memory simulate-bugs-at-target-precision
            // records. One-sided bars cannot see it.
            // SCOPE: the band bounds ACCUMULATION noise (kappa_near + kappa_far terms per row). It
            // is BLIND to kernel-chain rounding: evaluating the whole hi chain in f64 with one
            // fround on the return moved delta_precision by ~1 % (-1.2 %, MEASURED 2026-10-09)
            // and this test passed. An H1-b pass is NOT evidence that the chain is rounded —
            // the ROUNDING-PLACEMENT PINs (nearField.test.ts, and below for the far kernel) are.
            const bar = h1bBar(theta);
            const d = deltaPrecision(theta, F32);
            record(`delta_precision f32 (scalar A apply) theta=${theta}`, d);
            record(`delta_precision f32 / bar theta=${theta}`, d / bar);
            expect(d).toBeGreaterThanOrEqual(1e-7);
            expect(d).toBeLessThanOrEqual(bar);
            // Inline and precomputed near fields are the same f32 arithmetic: bit-identical.
            expect(deltaPrecision(theta, F32, 'inline')).toBe(d);
            // The vector apply, same band.
            const U = new Float64Array(3 * n);
            for (let c = 0; c < 3; c++) U.set(seeded(n, 300 + c), c * n);
            const dv = relErr(
                vectorApply(prepared('torusKnot', n, theta), U),
                vectorApply(prepared('torusKnot', n, theta, F32), U),
            );
            record(`delta_precision f32 (12-column A apply) theta=${theta}`, dv);
            expect(dv).toBeGreaterThanOrEqual(1e-7);
            expect(dv).toBeLessThanOrEqual(bar);
        });

        test(`theta=${theta} — rung 3, f16 near accumulator (knock-out, §7.11 item 7): delta_precision EXCEEDS the bar`, () => {
            // Model (§7.2): m * u_f16 = 97 * 4.88e-04 = 4.7e-02, ~820x over the bar — a worst-case
            // LINEAR-growth model. NOT previously measured; first measurement 2026-10-09 at
            // theta=0.5: 2.635e-03, 45.6x over the bar (random-walk growth, ~sqrt(m) * u_f16).
            // The knock-out must trip, and it does, decisively.
            expect(typeof Math.f16round).toBe('function'); // Bun 1.3.11; the variant depends on it
            const bar = h1bBar(theta);
            const d = deltaPrecision(theta, F16_NEAR_KNOCKOUT);
            record(`delta_precision f16-near knock-out theta=${theta}`, d);
            record(`f16 knock-out / bar theta=${theta}`, d / bar);
            expect(d).toBeGreaterThan(bar);
            expect(d).toBeGreaterThan(10 * bar); // not a marginal trip
        });
    }

    test('rung 1, f64 accumulator: delta_precision is IDENTICALLY 0 at both theta (must-NOT-trip consistency check)', () => {
        // Structural: the same deterministic traversal with the same variant on the same inputs.
        // A second prepared instance (inline near field) is used so this is not an object
        // compared with itself.
        for (const theta of [0.5, 0.25]) {
            const d = deltaPrecision(theta, F64, 'inline');
            record(`delta_precision f64-vs-f64 theta=${theta}`, d);
            expect(d).toBe(0);
        }
    });

    test('INVALID predicates (§7.2 table rows 2-3): the triangle holds and delta_total is not below delta_precision / 2', () => {
        const dc = deltaCompress('torusKnot', n, 0.5);
        const dp = deltaPrecision(0.5, F32);
        const dt = relErr(
            exactAu('torusKnot', n),
            scalarApply(prepared('torusKnot', n, 0.5, F32), rhs(n)),
        );
        record('delta_total f32 theta=0.5 vs f64 chi=0', dt);
        // PRESENT, NOT EXERCISED: at n=960 delta_compress ~8.6e-3 dominates delta_precision
        // ~1.05e-6 by four orders of magnitude, so both predicates hold with that margin and a
        // pass here is not evidence of anything beyond "the instruments were wired". They would
        // bite only where delta_compress and delta_precision are comparable (theta -> 0, or f16).
        expect(dt).toBeLessThanOrEqual(1.01 * (dc + dp));
        expect(dt).toBeGreaterThanOrEqual(dp / 2);
    });

    test('f32 ROUNDING-PLACEMENT PIN for the far kernel: precomputeFar(F32) is bit-identical to a per-op Math.fround transcription, both kernels', () => {
        // Same guardian as nearField.test.ts's pin, for the far (rank-1, barycentre) kernel —
        // through `precomputeFar(…, F32)` rather than a new export of `farKernel` (nothing else
        // needs it). Unfused binary32 contract: every product rounded before the add.
        const bvh = tree('torusKnot', 400);
        const bct = buildBct(bvh, { theta: 0.5 });
        const f = Math.fround;
        const C = Float32Array.from(bvh.C);
        const TN = Float32Array.from(bvh.TN);
        const eps = f(EPS);
        const { pHi, pLow } = exponents(ALPHA, BETA);
        const ref32Far = (A: number, B: number, kernel: 'hi' | 'low', p: number): number => {
            const dx = f(C[3 * A] - C[3 * B]);
            const dy = f(C[3 * A + 1] - C[3 * B + 1]);
            const dz = f(C[3 * A + 2] - C[3 * B + 2]);
            const d = f(f(Math.sqrt(f(f(f(dx * dx) + f(dy * dy)) + f(dz * dz)))) + eps);
            const dp = f(d ** p);
            if (kernel === 'hi') return f(2 / dp);
            const [ax, ay, az] = [TN[3 * A], TN[3 * A + 1], TN[3 * A + 2]];
            const [bx, by, bz] = [TN[3 * B], TN[3 * B + 1], TN[3 * B + 2]];
            const c1x = f(f(ay * dz) - f(az * dy));
            const c1y = f(f(az * dx) - f(ax * dz));
            const c1z = f(f(ax * dy) - f(ay * dx));
            const c2x = f(f(by * dz) - f(bz * dy));
            const c2y = f(f(bz * dx) - f(bx * dz));
            const c2z = f(f(bx * dy) - f(by * dx));
            const n1 = f(f(Math.sqrt(f(f(f(c1x * c1x) + f(c1y * c1y)) + f(c1z * c1z)))) + eps);
            const n2 = f(f(Math.sqrt(f(f(f(c2x * c2x) + f(c2y * c2y)) + f(c2z * c2z)))) + eps);
            return f(f(f(n1 * n1) + f(n2 * n2)) / dp);
        };
        let differFromRoundOnce = 0;
        for (const kernel of ['hi', 'low'] as const) {
            const p = kernel === 'hi' ? pHi : pLow;
            const far32 = precomputeFar(bvh, bct, kernel, ALPHA, BETA, EPS, F32);
            const far64 = precomputeFar(bvh, bct, kernel, ALPHA, BETA, EPS, F64);
            expect(far32.length).toBe(bct.stats.nFar);
            for (let k = 0; k < bct.stats.nFar; k++) {
                const ref = ref32Far(bct.farA[k], bct.farB[k], kernel, p);
                expect(far32[k]).toBe(ref);
                if (f(far64[k]) !== ref) differFromRoundOnce++;
            }
        }
        expect(bct.stats.nFar).toBe(1222);
        // The pin's own impossible value: round-once must disagree with per-op on most blocks.
        expect(differFromRoundOnce).toBeGreaterThan(bct.stats.nFar);
    });

    test('K-apply f32 noise oracle: f32NoiseHi 1.0638571324665979e-06, f32NoiseLow 3.002378984962962e-06 (nc=1, psi = sin(0.013 s) + 0.5, run-accuracy.ts:94-106)', () => {
        // The spike computed far values in f64 and down-cast them (`Float32Array.from(precomputeFar(...))`,
        // run-accuracy.ts:104). Reproduced with that construction first (port fidelity), then with
        // the production f32 construction (far values evaluated at f32), which must land within
        // 2x — the far field is 3.68 terms per row against 93.3 near terms.
        const bvh = tree('torusKnot', n);
        const bct = buildBct(bvh, { theta: 0.5 });
        const src: BlockSource = { kind: 'lists', bct };
        const m = bvh.order.length;
        const psi = new Float64Array(m);
        for (let s = 0; s < m; s++) psi[s] = Math.sin(0.013 * s) + 0.5;
        const { pHi, pLow } = exponents(ALPHA, BETA);
        const oracle = { hi: 1.0638571324665979e-6, low: 3.002378984962962e-6 };
        for (const kernel of ['hi', 'low'] as const) {
            const p = kernel === 'hi' ? pHi : pLow;
            const far64 = precomputeFar(bvh, bct, kernel, ALPHA, BETA, EPS, F64);
            const p64 = new Float64Array(m);
            applyK(
                bvh,
                src,
                { kernel, p, epsilon: EPS, far: far64, near: null },
                psi,
                p64,
                makeWorkspace(bvh, 1, F64),
                F64,
            );
            // (a) oracle provenance: f64 far values cast to f32.
            const p32a = new Float32Array(m);
            applyK(
                bvh,
                src,
                { kernel, p, epsilon: Math.fround(EPS), far: Float32Array.from(far64), near: null },
                Float32Array.from(psi),
                p32a,
                makeWorkspace(bvh, 1, F32),
                F32,
            );
            const ea = relErr(p64, p32a);
            record(`f32 K-apply noise ${kernel} (f64-cast far values, oracle construction)`, ea);
            // Not bit-identical to the spike: this port adds the far scatter AFTER the near sum
            // (one walker for lists and stream — apply.ts, "ACCUMULATION ORDER"), the spike added
            // it before. One f32 addition per entry is reordered. MEASURED 2026-10-09 (n=960,
            // theta=0.5): construction (a) hi 1.060521e-06 vs oracle 1.0638571e-06 (-0.31 %),
            // low 3.001293e-06 vs 3.0023790e-06 (-0.036 %); construction (b) hi 1.061746e-06,
            // low 3.001233e-06. Bar: 0.5 % relative = the hi deviation plus 0.19 points of
            // headroom (x1.6), 14x the low deviation — a measured floor, not a chosen round
            // number (§11.3 rule 3; replaces the 2 % of the first review). This bar bounds
            // ACCUMULATION noise only: an unrounded kernel chain moves the hi value by 0.15 %
            // (M3b, 2026-10-09) and passes here — the ROUNDING-PLACEMENT PINs are the guardian
            // of chain rounding, not this test.
            expect(Math.abs(ea - oracle[kernel]) / oracle[kernel]).toBeLessThan(0.005);
            // (b) production construction: far values at f32.
            const p32b = new Float32Array(m);
            applyK(
                bvh,
                src,
                {
                    kernel,
                    p,
                    epsilon: Math.fround(EPS),
                    far: precomputeFar(bvh, bct, kernel, ALPHA, BETA, EPS, F32),
                    near: null,
                },
                Float32Array.from(psi),
                p32b,
                makeWorkspace(bvh, 1, F32),
                F32,
            );
            const eb = relErr(p64, p32b);
            record(`f32 K-apply noise ${kernel} (f32 far values, production construction)`, eb);
            expect(eb).toBeGreaterThan(oracle[kernel] / 2);
            expect(eb).toBeLessThan(oracle[kernel] * 2);
        }
    });
});

// ---------------------------------------------------------------------------------------------
// 7. ONE traversal (§5.6): lists vs stream, precomputed vs inline, columns vs scalar — bit-identical.
// ---------------------------------------------------------------------------------------------

describe('§5.6 one traversal: every block source and column count is the same arithmetic', () => {
    const n = 960;

    for (const variant of [F64, F32]) {
        test(`${variant.name}: lists(theta=0.5, inline near) == stream(theta=0.5) bit-for-bit, and == lists precomputed`, () => {
            const u = rhs(n);
            const bvh = tree('torusKnot', n);
            const stream = prepareA(
                bvh,
                n,
                { kind: 'stream', params: { theta: 0.5 } },
                {
                    alpha: ALPHA,
                    beta: BETA,
                    epsilon: EPS,
                    variant,
                    precomputeNear: false,
                },
            );
            const a = scalarApply(prepared('torusKnot', n, 0.5, variant, 'inline'), u);
            const b = scalarApply(stream, u);
            const c = scalarApply(prepared('torusKnot', n, 0.5, variant, 'pre'), u);
            expect(Array.from(b)).toEqual(Array.from(a));
            expect(Array.from(c)).toEqual(Array.from(a));
            expect(stream.stats).toEqual(prepared('torusKnot', n, 0.5, variant, 'inline').stats);
            expect(stream.hi.near).toBeNull();
            expect(stream.hi.far).toBeNull();
        });
    }

    test('f64: vector apply block c == scalar apply of block c, bit-for-bit (columns never mix)', () => {
        const H = prepared('torusKnot', n, 0.5);
        const U = new Float64Array(3 * n);
        for (let c = 0; c < 3; c++) U.set(seeded(n, 500 + c), c * n);
        const O = vectorApply(H, U);
        for (let c = 0; c < 3; c++) {
            const o = scalarApply(H, U.subarray(c * n, (c + 1) * n));
            expect(Array.from(O.subarray(c * n, (c + 1) * n))).toEqual(Array.from(o));
        }
    });

    test('applyK with nc=1 on column j == column j of the nc=3 apply, bit-for-bit', () => {
        const bvh = tree('torusKnot', 400);
        const bct = buildBct(bvh, { theta: 0.5 });
        const src: BlockSource = { kind: 'lists', bct };
        const m = bvh.order.length;
        const { pHi } = exponents(ALPHA, BETA);
        const kd = {
            kernel: 'hi' as const,
            p: pHi,
            epsilon: EPS,
            far: precomputeFar(bvh, bct, 'hi', ALPHA, BETA, EPS, F64),
            near: null,
        };
        const psi3 = new Float64Array(3 * m);
        for (let j = 0; j < 3; j++) {
            const col = seeded(m, 700 + j);
            for (let s = 0; s < m; s++) psi3[3 * s + j] = col[s];
        }
        const phi3 = new Float64Array(3 * m);
        const st = applyK(bvh, src, kd, psi3, phi3, makeWorkspace(bvh, 3, F64), F64);
        expect(st.adjacentSkipped).toBe(3 * 400);
        expect(st.stats).toBe(bct.stats);
        for (let j = 0; j < 3; j++) {
            const psi1 = new Float64Array(m);
            for (let s = 0; s < m; s++) psi1[s] = psi3[3 * s + j];
            const phi1 = new Float64Array(m);
            applyK(bvh, src, kd, psi1, phi1, makeWorkspace(bvh, 1, F64), F64);
            for (let s = 0; s < m; s++) expect(phi3[3 * s + j]).toBe(phi1[s]);
        }
    });

    test('applyK throws on a precomputed near field of the OTHER kernel (the silent-wrong-operator trap)', () => {
        const bvh = tree('torusKnot', 128);
        const bct = buildBct(bvh, { theta: 0.5 });
        const { pHi } = exponents(ALPHA, BETA);
        const near = precomputeNear(bvh, bct, 'low', ALPHA, BETA, EPS, F64);
        const m = bvh.order.length;
        expect(() =>
            applyK(
                bvh,
                { kind: 'lists', bct },
                { kernel: 'hi', p: pHi, epsilon: EPS, far: null, near },
                new Float64Array(m),
                new Float64Array(m),
                makeWorkspace(bvh, 1, F64),
                F64,
            ),
        ).toThrow(/precomputed near field is 'low', apply is 'hi'/);
    });

    test('applyK throws when a streaming source carries precomputed far or near values (spec §7.1)', () => {
        const bvh = tree('torusKnot', 128);
        const bct = buildBct(bvh, { theta: 0.5 });
        const { pHi } = exponents(ALPHA, BETA);
        const m = bvh.order.length;
        const stream: BlockSource = { kind: 'stream', params: { theta: 0.5 } };
        const run = (far: Float64Array | null, near: ReturnType<typeof precomputeNear> | null) =>
            applyK(
                bvh,
                stream,
                { kernel: 'hi', p: pHi, epsilon: EPS, far, near },
                new Float64Array(m),
                new Float64Array(m),
                makeWorkspace(bvh, 1, F64),
                F64,
            );
        expect(() =>
            run(precomputeFar(bvh, bct, 'hi', ALPHA, BETA, EPS, F64) as Float64Array, null),
        ).toThrow(/streaming source carries no precomputed values/);
        expect(() => run(null, precomputeNear(bvh, bct, 'hi', ALPHA, BETA, EPS, F64))).toThrow(
            /streaming source carries no precomputed values/,
        );
        expect(() => run(null, null)).not.toThrow();
    });

    test('symmetry: u^T A v == v^T A u to f64 roundoff (theta=0.5)', () => {
        const H = prepared('torusKnot', n, 0.5);
        const u = rhs(n);
        const v = rhs(n, 999);
        const Au = scalarApply(H, u);
        const Av = scalarApply(H, v);
        let uAv = 0;
        let vAu = 0;
        for (let i = 0; i < n; i++) {
            uAv += u[i] * Av[i];
            vAu += v[i] * Au[i];
        }
        const e = Math.abs(uAv - vAu) / Math.abs(uAv);
        record('symmetry rel', e);
        expect(e).toBeLessThan(1e-13);
    });

    test('knock-out instruments are refused by prepareA unless opted in (production can never apply an inverted or truncated operator; task-3-review.md)', () => {
        const bvh = tree('torusKnot', 128);
        const base = { alpha: ALPHA, beta: BETA, epsilon: EPS };
        const inverted = buildBct(bvh, { theta: 0.5, invertAdmissibility: true });
        expect(() => prepareA(bvh, 128, { kind: 'lists', bct: inverted }, base)).toThrow(
            /knock-out instrument/,
        );
        expect(() =>
            prepareA(
                bvh,
                128,
                { kind: 'stream', params: { theta: 0.5, invertAdmissibility: true } },
                base,
            ),
        ).toThrow(/knock-out instrument/);
        expect(() =>
            prepareA(
                bvh,
                128,
                { kind: 'lists', bct: buildBct(bvh, { theta: 0.5 }) },
                {
                    ...base,
                    variant: F16_NEAR_KNOCKOUT,
                },
            ),
        ).toThrow(/knock-out instrument/);
        // The honest configurations are untouched, and the opt-in lets the knock-outs through.
        expect(() =>
            prepareA(bvh, 128, { kind: 'lists', bct: buildBct(bvh, { theta: 0.5 }) }, base),
        ).not.toThrow();
        expect(() =>
            prepareA(bvh, 128, { kind: 'lists', bct: inverted }, { ...base, knockOut: true }),
        ).not.toThrow();
    });

    test('prepareA reports the traversal stats and the near-field skip count (brief ruling 1)', () => {
        const H = prepared('torusKnot', n, 0.5);
        expect(H.stats.nearPairs).toBe(89538);
        expect(H.adjacentSkipped).toBe(3 * n); // closed curve: I == J plus two neighbours per edge
        expect(H.k1Hi.length).toBe(H.m);
        expect(H.k1Low.length).toBe(H.m);
        // K1 is geometry-only and positive: every row has at least one non-adjacent partner.
        for (let s = 0; s < H.m; s++) {
            expect(H.k1Hi[s]).toBeGreaterThan(0);
            expect(H.k1Low[s]).toBeGreaterThan(0);
        }
    });
});
