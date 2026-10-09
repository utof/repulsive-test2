/**
 * Spec §11.2 task 6 of Phase 2H, second half: `Ã⁻¹ = (−Δ)⁻¹ L^{2−s} (−Δ)⁻¹` with the
 * constant-mode correction, and spec §7.8's knock-out (2) — dropping that correction, which §7.8
 * records as *"NOT MEASURED, cheap"*.
 *
 * FIXTURE AND SCALE, stated because the oracle corpus straddles two scales (@issue
 * utof/repulsive-test2#69): the UNnormalised `src/core/fixtures.ts trefoil(n)`, bbox max extent
 * 5.4716 — the curve `gmres_lab.py`'s `trefoil` builds and every `e4-closer-0b.json` /
 * `e0-closer-0b.json` row was measured on. Nothing here is compared against an `out-bct-*.json`
 * or `out-accuracy.json` number, which belong to extent-1.0 curves.
 *
 * AND NO `κ` FIGURE HERE MAY BE STATED WITHOUT ITS EXTENT. The lab's `Ã⁻¹` is scale-invariant to
 * 7-9 significant digits; the shipped `massWeight = 1` one is NOT, because stiffness element
 * matrices scale as `length⁻¹` and mass as `length⁺¹`, so `massWeight` carries units of
 * `length⁻²`. MEASURED (`task-6d-report.md`, "Follow-up: scale dependence"): `κ_excl` moves 62x at
 * n=64 (8.87 -> 553.1) and 53x at n=1920 (20.3 -> 1081.4) between extent 5.4716 and extent 1.0. So
 * "the shipped operator conditions ~150x better than the lab" is true ONLY at extent 5.4716; at
 * extent 1.0 it is ~2.8x. Every κ test below ASSERTS the extent it measured at.
 *
 * `κ_eig(Ã⁻¹A_σ)` IS MEASURED HERE, on the operator `src/` ships. `Ã⁻¹A_σ` is a product of two
 * SPD matrices and is therefore not symmetric, but with `A_σ = R Rᵀ` it is SIMILAR to `Rᵀ Ã⁻¹ R`,
 * which is — so the Householder+Sturm eigensolver in `denseSpectrum.ts` reaches its spectrum with
 * no non-symmetric solver anywhere (derivation in {@link cholesky}'s TSDoc). The instrument is
 * validated against `e4-closer-0b.json` by rebuilding the LAB's `Ã⁻¹` in TypeScript, and
 * reproduces its `kappaPrecond` to 8 significant digits before it is pointed at production. That
 * matters because spec §11.2 row 11's ruling — H5's dense-eigenvalue instruments stay in
 * `scratch/*.py` — left properties 2 and 3 QUOTED from a Python artefact of a DIFFERENT `(−Δ)`
 * (the lab's pure-stiffness pseudo-inverse; see `precond.ts`'s provenance note).
 *
 * TWO `κ`s, LABELLED SEPARATELY, because the repo has been conflating them. `λ_min = 1` is ALWAYS
 * the constant mode pinned by `+11ᵀ/(nσ)` (`A_σ1 = σ1`, `Π1 = 0` ⇒ `Ã⁻¹A_σ1 = 1` exactly), and it
 * sits BELOW every other eigenvalue, so the corpus's `kappaPrecond = max|λ|/min|λ|` is just
 * `λ_max`. `kappaInclConst` is that number; `kappaExclConst = λ_max/λ₂` is the conditioning GMRES
 * actually sees after it disposes of the pinned mode. They have DIFFERENT gate power: `κ_incl`
 * catches §7.8's knock-out (2) (it makes `λ₁ = 0`) and `κ_excl` is blind to it — both are recorded
 * for every case below. No flatness bar is asserted on either; see the test's own comment.
 *
 * The rest of the file measures what a matvec-only instrument can decide: the constant-mode
 * eigenvalue exactly, the three-application O(N) structure, symmetry, positive-definiteness on
 * `1⊥`, the knock-outs, and the per-application cost.
 *
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.2, §5.3, §7.8, §11.2 task 6
 * @see .superpowers/sdd/2026-10-09-phase2h-bct-gmres-design/task-6-brief.md
 * @issue utof/repulsive-test2#59
 */
import { describe, expect, setDefaultTimeout, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { trefoil } from '../../../src/core/fixtures';
import { applyA, applyAScalar, prepareA } from '../../../src/core/sobolev/hierarchical/apply';
import { buildBct } from '../../../src/core/sobolev/hierarchical/bct';
import { buildBvh } from '../../../src/core/sobolev/hierarchical/bvh';
import {
    CurveTopologyError,
    makeLaplacianWorkspace,
    solveLaplacian,
    triResidual,
} from '../../../src/core/sobolev/hierarchical/laplacian';
import {
    applyLowOrder,
    type LowOrderVariant,
    prepareLowOrder,
} from '../../../src/core/sobolev/hierarchical/lowOrder';
import { makeHierarchicalOperator } from '../../../src/core/sobolev/hierarchical/operator';
import { makePreconditioner } from '../../../src/core/sobolev/hierarchical/precond';
import type { Edge, Vec3 } from '../../../src/core/testConfigs';
import { cholesky, cholSolve, densify, matmul, spectrum, transpose } from './denseSpectrum';

setDefaultTimeout(900_000);

const ALPHA = 3;
const BETA = 6;
const EPS_A = 1e-10;
const LEAF = 8;
const THETA = 0.25;
const U_F64 = 2.220446049250313e-16;
/** `gmres_lab.py:21` `U_F32`, H5 property 3's unit roundoff. */
const U_F32 = 5.96e-8;
/** The oracle corpus's extent (module TSDoc); every κ test asserts the curve it got matches. */
const EXTENT = 5.4716;
const SCRATCH = `${import.meta.dir}/../../../.superpowers/sdd/2026-09-11-phase2h-queue/scratch`;

const log = (name: string, v: unknown) => console.log(`[task6] ${name} = ${v}`);

/** `scratch/curves.ts` `lcg` — deterministic, no `Math.random`. */
function lcg(seed: number): () => number {
    let s = seed >>> 0;
    return () => {
        s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
        return s / 4294967296;
    };
}
const randVec = (n: number, seed: number): Float64Array => {
    const rnd = lcg(seed);
    const v = new Float64Array(n);
    for (let i = 0; i < n; i++) v[i] = rnd() - 0.5;
    return v;
};

interface BuildOpts {
    massWeight?: number;
    dropConstantMode?: boolean;
    /** Default `'vertex'`, the production variant; `'rs-identity'` needs `knockOut` (precond.ts). */
    variant?: LowOrderVariant;
}

const cache = new Map<string, ReturnType<typeof build>>();
function build(n: number, opts: BuildOpts = {}) {
    const { vertices, edges } = trefoil(n);
    const bvh = buildBvh(vertices, edges, LEAF, EPS_A);
    const source = { kind: 'lists', bct: buildBct(bvh, { theta: THETA }) } as const;
    const H = prepareA(bvh, n, source, { alpha: ALPHA, beta: BETA, epsilon: EPS_A });
    const op = makeHierarchicalOperator(H);
    const variant = opts.variant ?? 'vertex';
    const Lo = prepareLowOrder(vertices, edges, bvh, source, {
        alpha: ALPHA,
        beta: BETA,
        variant,
    });
    // ASSERTED, not assumed: every test in this file reads as a pass on the `'rs-identity'`
    // knock-out otherwise (it is a one-token default change in `lowOrder.ts`, and nothing at the
    // preconditioner level used to depend on nullity). The κ tests below now do — but this
    // builder is what the other 13 tests go through.
    expect(Lo.variant).toBe(variant);
    const P = makePreconditioner(vertices, edges, Lo, op.sigma, {
        massWeight: opts.massWeight,
        dropConstantMode: opts.dropConstantMode,
        knockOut: variant !== 'vertex',
    });
    return { vertices, edges, bvh, H, op, Lo, P };
}
function prep(n: number, opts: BuildOpts = {}) {
    const k = `${n}/${opts.massWeight ?? 1}/${opts.dropConstantMode ?? false}/${opts.variant ?? 'vertex'}`;
    let v = cache.get(k);
    if (!v) {
        v = build(n, opts);
        cache.set(k, v);
    }
    return v;
}

/** The fixture's bbox max extent — the number every oracle κ is conditional on (module TSDoc). */
function extentOf(vertices: Vec3[]): number {
    let ext = 0;
    for (let d = 0; d < 3; d++) {
        let lo = Number.POSITIVE_INFINITY;
        let hi = Number.NEGATIVE_INFINITY;
        for (const v of vertices) {
            lo = Math.min(lo, v[d]);
            hi = Math.max(hi, v[d]);
        }
        ext = Math.max(ext, hi - lo);
    }
    return ext;
}

/** `A_σ u = A_bct u + (σ/n)(1ᵀu) 1`, scalar — spec §2.3's shift. */
function applyASigma(
    H: Parameters<typeof applyAScalar>[0],
    sigma: number,
    n: number,
    u: ArrayLike<number>,
    out: Float64Array,
): void {
    applyAScalar(H, u, out);
    let s = 0;
    for (let i = 0; i < n; i++) s += u[i];
    const c = (sigma / n) * s;
    for (let i = 0; i < n; i++) out[i] += c;
}

const l2 = (x: ArrayLike<number>, n: number): number => {
    let s = 0;
    for (let i = 0; i < n; i++) s += x[i] * x[i];
    return Math.sqrt(s);
};

// ---------------------------------------------------------------------------
// The κ_eig instrument. TEST ONLY and O(n³) — `denseSpectrum.ts`'s module TSDoc carries the
// provenance; this part is the two operators it is pointed at.
// ---------------------------------------------------------------------------

/** Row of `e4-closer-0b.json`, the H5 property-1/2 oracle (`e4_precond_integrity.py:50-56`). */
interface E4Row {
    n: number;
    variant: string;
    nullDimLfrac: number;
    kappaPrecond: number;
}
const e4: E4Row[] = JSON.parse(readFileSync(`${SCRATCH}/e4-closer-0b.json`, 'utf8')).rows;
const e4row = (n: number, variant: string): E4Row => {
    const r = e4.find((x) => x.n === n && x.variant === variant);
    if (!r) throw new Error(`e4-closer-0b.json has no row n=${n} variant=${variant}`);
    return r;
};

/**
 * `gmres_lab.py:95-110` `polyline_laplacian`, transcribed: PURE stiffness, `w = 1/ell`, singular
 * with nullspace the constants. NOT `assembleCurveLaplacian` — that one is
 * `stiffness·L + mass·M` and refuses `massWeight <= 0` (`laplacian.ts`), which is exactly the
 * provenance boundary `precond.ts`'s TSDoc records. This is the LAB's operator, kept here so the
 * instrument can be validated against the LAB's numbers before it is trusted on production's.
 */
function labStiffness(vertices: Vec3[], edges: Edge[]): Float64Array {
    const n = vertices.length;
    const L = new Float64Array(n * n);
    for (const [a, b] of edges) {
        const dx = vertices[b][0] - vertices[a][0];
        const dy = vertices[b][1] - vertices[a][1];
        const dz = vertices[b][2] - vertices[a][2];
        const w = 1 / Math.sqrt(dx * dx + dy * dy + dz * dz);
        L[a * n + a] += w;
        L[b * n + b] += w;
        L[a * n + b] -= w;
        L[b * n + a] -= w;
    }
    return L;
}

/**
 * The LAB's `Ã⁻¹ = Π D⁺ L^{2−s} D⁺ Π + 11ᵀ/(nσ)` (`gmres_lab.py:174-194`), dense.
 *
 * `D⁺ = pinv(Π Lap Π)` is reproduced WITHOUT an SVD: for `b ⊥ 1` the pseudo-inverse is the unique
 * `x ⊥ 1` with `Lap x = b`, and `(Lap + c·11ᵀ/n) x = b` delivers exactly that for any `c > 0`
 * (taking `1ᵀ` of both sides gives `c·1ᵀx = 1ᵀb = 0`), while being SPD and hence Cholesky-able.
 * Both `D⁺` operands here are `⊥ 1` already (`L^{2−s}` is symmetric with `L·1 = 0`, so its range
 * is `1⊥`), and `1ᵀcore = 0` follows from `G⁻¹1 = 1/c`, so the outer `Π`s are exact no-ops and are
 * omitted rather than simulated. VERIFIED by the 8-digit agreement with `kappaPrecond`.
 */
function labPreconditioner(
    vertices: Vec3[],
    edges: Edge[],
    Lf: Float64Array,
    sigma: number,
): Float64Array {
    const n = vertices.length;
    const Lap = labStiffness(vertices, edges);
    let dsum = 0;
    for (let i = 0; i < n; i++) dsum += Lap[i * n + i];
    const c = dsum / n;
    const G = new Float64Array(Lap);
    for (let i = 0; i < n * n; i++) G[i] += c / n;
    const LG = cholesky(G, n);
    const mulLf = (u: Float64Array, out: Float64Array) => {
        for (let i = 0; i < n; i++) {
            let acc = 0;
            for (let k = 0; k < n; k++) acc += Lf[i * n + k] * u[k];
            out[i] = acc;
        }
    };
    // `core = G⁻¹ Lf G⁻¹`, built as `Y = G⁻¹ Lf` then `core = G⁻¹ Yᵀ` — `Lf G⁻¹ = (G⁻¹ Lf)ᵀ`
    // because both factors are symmetric.
    const Y = densify(n, (u, out) => {
        mulLf(u, out);
        cholSolve(LG, n, out);
    });
    const Yt = transpose(Y, n);
    const core = densify(n, (u, out) => {
        for (let i = 0; i < n; i++) {
            let acc = 0;
            for (let k = 0; k < n; k++) acc += Yt[i * n + k] * u[k];
            out[i] = acc;
        }
        cholSolve(LG, n, out);
    });
    for (let i = 0; i < n * n; i++) core[i] += 1 / (n * sigma);
    return core;
}

/** Dense `A_σ = A_bct + (σ/n)11ᵀ` from `n` hierarchical applies. */
const densifyASigma = (
    H: Parameters<typeof applyAScalar>[0],
    sigma: number,
    n: number,
): Float64Array => densify(n, (u, out) => applyASigma(H, sigma, n, u, out));

/**
 * The spectrum of `M A_σ` for symmetric `M`, via the similarity `Rᵀ M R` with `A_σ = R Rᵀ`
 * (`denseSpectrum.ts` {@link cholesky}). The Cholesky is also the SPD check on `A_σ` itself: at
 * θ=0.25 the BCT approximation could in principle lose definiteness, and this would throw.
 */
function kappaOf(M: Float64Array, Asig: Float64Array, n: number) {
    const R = cholesky(Asig, n);
    return spectrum(matmul(transpose(R, n), matmul(M, R, n), n), n);
}

/** Least-squares slope of `log y` against `log n` — H5 property 2's exponent `p`. */
function fitExponent(ns: number[], ys: number[]): number {
    const k = ns.length;
    let sx = 0;
    let sy = 0;
    let sxx = 0;
    let sxy = 0;
    for (let i = 0; i < k; i++) {
        const X = Math.log(ns[i]);
        const Y = Math.log(ys[i]);
        sx += X;
        sy += Y;
        sxx += X * X;
        sxy += X * Y;
    }
    return (k * sxy - sx * sy) / (k * sxx - sx * sx);
}

// ===========================================================================

describe('precond: H5 κ_eig(Ã⁻¹A_σ) on the operator src/ ships (spec §7.8 properties 2 and 3)', () => {
    test('the instrument reproduces e4-closer-0b.json kappaPrecond on the LAB construction', () => {
        // A harness that cannot reproduce the corpus cannot be trusted to EXTEND it, so this runs
        // first and on the lab's three ingredients: `D⁺ = pinv(Π Lap Π)` of the pure-stiffness
        // Laplacian ({@link labPreconditioner}), the EXACT `L^{2−s}`, and the EXACT `A_σ`. Both
        // "exact" operators come from a θ=0 block source, where every block is inadmissible, so
        // the hierarchical apply degenerates to the all-pairs sum — the same matrix the oracle's
        // numpy cache holds, reached through `src/` instead of a second transcription.
        for (const n of [64, 120, 240]) {
            const { vertices, edges } = trefoil(n);
            expect(extentOf(vertices) / EXTENT).toBeCloseTo(1, 2);
            const bvh = buildBvh(vertices, edges, LEAF, EPS_A);
            const exact = { kind: 'lists', bct: buildBct(bvh, { theta: 0 }) } as const;
            const H0 = prepareA(bvh, n, exact, { alpha: ALPHA, beta: BETA, epsilon: EPS_A });
            const sigma = makeHierarchicalOperator(H0).sigma;
            const Lo0 = prepareLowOrder(vertices, edges, bvh, exact, { alpha: ALPHA, beta: BETA });
            const Lf = densify(n, (u, out) => applyLowOrder(Lo0, u, out));
            const Mlab = labPreconditioner(vertices, edges, Lf, sigma);
            const sp = kappaOf(Mlab, densifyASigma(H0, sigma, n), n);
            const kInc = sp.lambdaMax / sp.lambda1;
            const want = e4row(n, 'vertex').kappaPrecond;
            const rel = Math.abs(kInc / want - 1);
            log(`LAB λ₁ n=${n}`, sp.lambda1);
            log(`LAB λ₂ n=${n}`, sp.lambda2);
            log(`LAB λ_max n=${n}`, sp.lambdaMax);
            log(`LAB kappaInclConst n=${n}`, kInc);
            log(`LAB kappaExclConst n=${n}`, sp.lambdaMax / sp.lambda2);
            log(`LAB vs e4-closer-0b.json kappaPrecond n=${n}`, `${kInc} vs ${want}, rel ${rel}`);
            // MEASURED 3.0e-10 / 4.5e-11 / 1.8e-11 — i.e. 10 significant digits, where the
            // dispatch asked for 4. The bar is 1e-6 so a transcription slip cannot hide inside it.
            expect(rel).toBeLessThan(1e-6);
            // `λ₁ = 1` EXACTLY is structural, not fitted: `A_σ 1 = σ 1` (`A·1 = 0` bit-exactly),
            // `Π 1 = 0`, so `Ã⁻¹A_σ 1 = 11ᵀ(σ 1)/(nσ) = 1`. The bar scales with `λ_max`, NOT with
            // `u_f64`: bisection resolves ABSOLUTELY, to `2.3e-16 · Gershgorin span` (`eigAt`), so a
            // structural 1 next to a `λ_max` of 1.1e4 cannot be resolved better than ~1e-9.
            // MEASURED |λ₁ − 1|/λ_max: 1.8e-13 here, 9e-14 on production — bar set 50-100x above.
            expect(Math.abs(sp.lambda1 - 1)).toBeLessThan(1e-11 * sp.lambdaMax);
            expect(sp.lambda2).toBeGreaterThan(1);
        }
    });

    test('κ_eig of the SHIPPED Ã⁻¹A_σ, n=64..480, both definitions', () => {
        const ns = [64, 120, 240, 480];
        const kInc: number[] = [];
        const kExc: number[] = [];
        for (const n of ns) {
            const { vertices, H, op, P } = prep(n);
            // A-SCALE: the κ of the shipped operator is NOT scale-invariant (module TSDoc, 53-62x
            // between extent 5.4716 and 1.0), so a number without its extent is not a measurement.
            const ext = extentOf(vertices);
            log(`extent n=${n}`, ext);
            expect(ext / EXTENT).toBeCloseTo(1, 2);
            const sp = kappaOf(
                densify(n, (u, out) => P.applyScalar(u, out)),
                densifyASigma(H, op.sigma, n),
                n,
            );
            kInc.push(sp.lambdaMax / sp.lambda1);
            kExc.push(sp.lambdaMax / sp.lambda2);
            log(`λ₁ n=${n}`, sp.lambda1);
            log(`λ₂ n=${n}`, sp.lambda2);
            log(`λ_max n=${n}`, sp.lambdaMax);
            log(`λ_max/λ₂ n=${n}`, sp.lambdaMax / sp.lambda2);
            log(`kappaInclConst n=${n}`, sp.lambdaMax / sp.lambda1);
            log(`kappaExclConst n=${n}`, sp.lambdaMax / sp.lambda2);
            log(`nullDim(Ã⁻¹A_σ) n=${n}`, sp.nullDim);
            // The pinned constant mode, again exactly 1, and it is the SMALLEST eigenvalue — which
            // is why the corpus's `max|λ|/min|λ|` is just `λ_max`.
            expect(Math.abs(sp.lambda1 - 1)).toBeLessThan(1e-11 * sp.lambdaMax);
            expect(sp.lambda2).toBeGreaterThan(1.5);
            expect(sp.nullDim).toBe(0);
            // H5 property 3 on the SHIPPED operator for the first time (it was quoted from
            // `e0-closer-0b.json` before): `κ_eig · u_f32 <= 1e-2`. RECORDED, not a licence to
            // solve in f32 — a bounded condition number is necessary, not sufficient.
            log(`kappaInclConst · u_f32 n=${n}`, (sp.lambdaMax / sp.lambda1) * U_F32);
            expect((sp.lambdaMax / sp.lambda1) * U_F32).toBeLessThan(1e-2);
        }
        // H5 property 2's exponent. LOGGED, NOT ASSERTED, and deliberately: the `|p| <= 0.05` bar
        // reads RED on this operator (p = 0.0988 over all four points) AND on the lab reference
        // under the excluded-constant definition, so it is a bar a correct operator fails.
        // Re-registering it is an orchestrator/user decision, not a test's. The POINT SET is named
        // because `p` depends on it: dropping n=64 moves it from 0.0988 to 0.0776.
        log('p(kappaInclConst) over n=64,120,240,480', fitExponent(ns, kInc));
        log('p(kappaInclConst) over n=120,240,480', fitExponent(ns.slice(1), kInc.slice(1)));
        log('p(kappaExclConst) over n=64,120,240,480', fitExponent(ns, kExc));
        // Cross-operator, with the extent that makes it true (A-SCALE): at extent 5.4716 the
        // shipped `κ_incl` is ~150x below the lab's ~1.1e4; at extent 1.0 it is ~2.8x.
        log(
            'kappaInclConst, shipped vs LAB at extent 5.4716, n=240',
            `${kInc[2]} vs ${e4row(240, 'vertex').kappaPrecond}`,
        );
    });

    test('the knock-outs show up in the NULLITY of Ã⁻¹A_σ, and κ_excl is blind to one of them', () => {
        // This is what H5 property 1 is FOR, finally measured at the preconditioner level rather
        // than on `L^{2−s}` alone: a knock-out that cannot be seen in the gated quantity is a gate
        // that reads GREEN on a broken operator (memory `gates-name-their-impossible-values`).
        const n = 64;
        const base = prep(n);
        const Asig = densifyASigma(base.H, base.op.sigma, n);
        const good = kappaOf(
            densify(n, (u, o) => base.P.applyScalar(u, o)),
            Asig,
            n,
        );
        const kExcGood = good.lambdaMax / good.lambda2;
        // `nullDim` is of `Ã⁻¹A_σ`, NOT of `L^{2−s}`, and the two differ by exactly the rank-1
        // correction: `'rs-identity'` loses the CHECKERBOARD (nullity 2 in `L^{2−s}`,
        // `e4-closer-0b.json`'s `nullDimLfrac`), but `+11ᵀ/(nσ)` still pins the constants, so one
        // null direction survives into the product. Only the two knock-outs TOGETHER give 2.
        const cases: { label: string; opts: BuildOpts; nullDim: number }[] = [
            {
                label: 'knock-out (2), dropConstantMode',
                opts: { dropConstantMode: true },
                nullDim: 1,
            },
            { label: 'rs-identity, correction ON', opts: { variant: 'rs-identity' }, nullDim: 1 },
            {
                label: 'rs-identity + dropConstantMode',
                opts: { variant: 'rs-identity', dropConstantMode: true },
                nullDim: 2,
            },
        ];
        for (const { label, opts, nullDim } of cases) {
            const bad = prep(n, opts);
            const sp = kappaOf(
                densify(n, (u, o) => bad.P.applyScalar(u, o)),
                densifyASigma(bad.H, bad.op.sigma, n),
                n,
            );
            log(`${label}: λ₁`, sp.lambda1);
            log(`${label}: λ₂`, sp.lambda2);
            log(`${label}: λ_max`, sp.lambdaMax);
            log(`${label}: nullDim`, sp.nullDim);
            log(`${label}: kappaInclConst`, sp.lambdaMax / sp.lambda1);
            log(`${label}: kappaExclConst`, sp.lambdaMax / sp.lambda2);
            expect(sp.nullDim).toBe(nullDim);
            expect(Math.abs(sp.lambda1)).toBeLessThan(sp.rankTol);
        }
        // The `rs-identity` nullity is the oracle's own verdict on the same defect, reached above
        // through the BCT instead of a dense numpy `eigvalsh`.
        expect(e4row(n, 'rs-identity').nullDimLfrac).toBe(2);
        expect(e4row(n, 'vertex').nullDimLfrac).toBe(1);
        // WHICH κ HAS THE GATE POWER, measured rather than assumed. `κ_incl = max|λ|/min|λ|` (what
        // the corpus computes) catches BOTH knock-outs — each drives `min|λ|` to ~1e-14, so
        // `κ_incl` ~ 1e15-1e16. `κ_excl = λ_max/λ₂` catches only the `rs-identity` one (2425 vs
        // 8.87, 273x): knock-out (2) is RANK-1 ON THE CONSTANTS, which are below `λ₂`, so it leaves
        // `λ₂` and `λ_max` untouched to 1e-15 and `κ_excl` cannot see it AT ALL. Hence H5 must gate
        // `κ_incl`; `κ_excl` is reported as the number GMRES experiences, not as the gate.
        const dropped = kappaOf(
            densify(n, (u, o) => prep(n, { dropConstantMode: true }).P.applyScalar(u, o)),
            Asig,
            n,
        );
        const rsid = kappaOf(
            densify(n, (u, o) => prep(n, { variant: 'rs-identity' }).P.applyScalar(u, o)),
            densifyASigma(prep(n, { variant: 'rs-identity' }).H, base.op.sigma, n),
            n,
        );
        log('κ_excl production', kExcGood);
        log('κ_excl knock-out (2) — BLIND', dropped.lambdaMax / dropped.lambda2);
        log('κ_excl rs-identity — caught', rsid.lambdaMax / rsid.lambda2);
        log('κ_incl knock-out (2)', Math.abs(dropped.lambdaMax / dropped.lambda1));
        log('κ_incl rs-identity', Math.abs(rsid.lambdaMax / rsid.lambda1));
        expect(dropped.lambdaMax / dropped.lambda2 / kExcGood).toBeCloseTo(1, 10);
        expect(rsid.lambdaMax / rsid.lambda2 / kExcGood).toBeGreaterThan(100);
        expect(Math.abs(dropped.lambdaMax / dropped.lambda1)).toBeGreaterThan(1e12);
        expect(Math.abs(rsid.lambdaMax / rsid.lambda1)).toBeGreaterThan(1e12);
    });
});

describe('precond: the constant mode is EXACT, and that is what +11ᵀ/(nσ) buys', () => {
    for (const n of [64, 240]) {
        test(`Ã⁻¹A_σ · 1 = 1 at n=${n}`, () => {
            const { H, op, P } = prep(n);
            const ones = new Float64Array(n).fill(1);
            const as = new Float64Array(n);
            applyASigma(H, op.sigma, n, ones, as);
            // `A_bct·1 = 0` BIT-EXACTLY (spec §2.2, apply.ts), so `A_σ 1` is `σ·1` to one rounding
            // of `(σ/n)·n`. Never asserted as "small": it is structural.
            let worstA = 0;
            for (let i = 0; i < n; i++) worstA = Math.max(worstA, Math.abs(as[i] / op.sigma - 1));
            log(`max|A_σ1/σ − 1| n=${n}`, worstA);
            expect(worstA).toBeLessThan(4 * U_F64);

            const got = new Float64Array(n);
            P.applyScalar(as, got);
            let worst = 0;
            for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(got[i] - 1));
            log(`max|Ã⁻¹A_σ1 − 1| n=${n}`, worst);
            log(`sigma n=${n}`, op.sigma);
            // The correction term alone gives `(1ᵀ(σ1))/(nσ) = 1` to three roundings. What is NOT
            // exact is `Π(σ·1)`: the mean of a constant vector round-trips through the sum, so any
            // residual enters `core` and is amplified by `(−Δ)⁻¹L(−Δ)⁻¹`.
            // BAR: `100 u_f64` ABSOLUTE — σ-free and n-free, because HEAD reads 1.11e-15 at n=64
            // and 2.22e-15 at n=240, i.e. 5-10 u and FLAT in n. The previous bar was `1e4 · σ ·
            // u_f64` = 1.8e-9 at n=240, which the naive-projector mutant PASSED at 2.07e-11: it
            // scaled with σ (806 at n=240) and so grew faster than the quantity it guarded (fix
            // round 1, finding m2). `project`'s compensated two-pass sum is what keeps this
            // bounded — see its TSDoc.
            expect(worst).toBeLessThan(100 * U_F64);
        });
    }

    test('Ã⁻¹ maps the constants onto the constants with eigenvalue 1/σ exactly', () => {
        const n = 240;
        const { op, P } = prep(n);
        const ones = new Float64Array(n).fill(1);
        const got = new Float64Array(n);
        P.applyScalar(ones, got);
        const want = 1 / op.sigma;
        let worst = 0;
        for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(got[i] / want - 1));
        log('max|Ã⁻¹1 / (1/σ) − 1|', worst);
        expect(worst).toBeLessThan(4 * U_F64);
    });
});

describe('precond: KNOCK-OUT (2) — drop the constant-mode correction (spec §7.8, NEVER MEASURED)', () => {
    test('Ã⁻¹ becomes singular on the constants, so Ã⁻¹A_σ loses rank 1', () => {
        const n = 240;
        const good = prep(n);
        const bad = prep(n, { dropConstantMode: true });
        expect(bad.P.dropConstantMode).toBe(true);
        const ones = new Float64Array(n).fill(1);
        const as = new Float64Array(n);
        applyASigma(bad.H, bad.op.sigma, n, ones, as);

        const outGood = new Float64Array(n);
        const outBad = new Float64Array(n);
        good.P.applyScalar(ones, outGood);
        bad.P.applyScalar(ones, outBad);
        log('‖Ã⁻¹·1‖ WITH correction', l2(outGood, n));
        log('‖Ã⁻¹·1‖ WITHOUT correction (knock-out)', l2(outBad, n));
        // EXACTLY zero, not small: `Π·1 = 0` on identical bits and `L^{2−s}·1 = 0` bit-exactly
        // (lowOrder.ts), so with the correction removed there is nothing left on this mode.
        expect(l2(outBad, n)).toBe(0);
        expect(l2(outGood, n)).toBeCloseTo(Math.sqrt(n) / good.op.sigma, 10);

        const pGood = new Float64Array(n);
        const pBad = new Float64Array(n);
        good.P.applyScalar(as, pGood);
        bad.P.applyScalar(as, pBad);
        // The constant mode's eigenvalue of the PRECONDITIONED operator: 1 with the correction,
        // 0 without. λ_min = 0 ⇒ κ_eig = ∞ ⇒ H5 property 1 RED, and properties 2 and 3 with it.
        const lamGood = pGood.reduce((s, x) => s + x, 0) / n;
        const lamBad = pBad.reduce((s, x) => s + x, 0) / n;
        log('constant-mode eigenvalue of Ã⁻¹A_σ, WITH correction', lamGood);
        log('constant-mode eigenvalue of Ã⁻¹A_σ, WITHOUT (knock-out)', lamBad);
        log('‖Ã⁻¹A_σ·1‖ WITH correction', l2(pGood, n));
        log('‖Ã⁻¹A_σ·1‖ WITHOUT correction', l2(pBad, n));
        log('knock-out / production ‖Ã⁻¹A_σ·1‖', l2(pBad, n) / l2(pGood, n));
        expect(lamGood).toBeCloseTo(1, 12);
        // EXACTLY 0, and that it is exact is itself structural rather than lucky: `A_σ·1 = σ·1`
        // exactly (`A_bct·1 = 0` bit-exactly, spec §2.2), `Π(σ·1) = 0` exactly because `project`
        // uses a compensated two-pass sum (its TSDoc; with a naive sum this read 1.74e-10), and
        // `L^{2−s}·1 = 0` exactly (lowOrder.ts). So removing `+11ᵀ/(nσ)` leaves literally nothing
        // on this mode: λ_min(Ã⁻¹A_σ) = 0 ⇒ κ_eig = ∞ ⇒ H5 RED on all three properties.
        expect(lamBad).toBe(0);
        expect(l2(pBad, n)).toBe(0);
        expect(l2(pGood, n)).toBeCloseTo(Math.sqrt(n), 10);
    });

    test('the knock-out leaves the complement of the constants untouched', () => {
        // The defect is RANK-1 and nothing else: on a rhs already in `1⊥` the two preconditioners
        // are bit-identical. That is what makes it a clean knock-out rather than a perturbation,
        // and it is why H5 is the only gate in the set that catches it (spec §7.8: H3 within 1
        // iteration, H4a/H4b′ green, ε_A degraded only 14-75 %).
        const n = 240;
        const good = prep(n);
        const bad = prep(n, { dropConstantMode: true });
        const r = randVec(n, 11);
        const mean = r.reduce((s, x) => s + x, 0) / n;
        for (let i = 0; i < n; i++) r[i] -= mean;
        const a = new Float64Array(n);
        const b = new Float64Array(n);
        good.P.applyScalar(r, a);
        bad.P.applyScalar(r, b);
        let worst = 0;
        for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
        log('max|Ã⁻¹r − Ã⁻¹_knockout r| on r ⊥ 1', worst);
        // Not bit-exact: `Π r` leaves a residual mean of O(u)·‖r‖, which the correction scales by
        // 1/σ. Bounded by the correction's own size on a zero-mean vector.
        expect(worst).toBeLessThan(1e-12 * l2(a, n));
    });
});

describe('precond: THREE operator applications, all O(N) (spec §5.2)', () => {
    test('exactly 2 Laplacian solves per block and exactly 1 L^{2−s} apply', () => {
        const n = 240;
        const { P } = prep(n, { massWeight: 0.5 }); // a fresh instance, so the counters start at 0
        expect(P.solves).toBe(0);
        expect(P.lowOrderApplies).toBe(0);
        const out1 = new Float64Array(n);
        P.applyScalar(randVec(n, 1), out1);
        log('solves after one scalar apply', P.solves);
        log('lowOrderApplies after one scalar apply', P.lowOrderApplies);
        expect(P.solves).toBe(2);
        expect(P.lowOrderApplies).toBe(1);
        const out3 = new Float64Array(3 * n);
        P.apply(randVec(3 * n, 2), out3);
        log('solves after one 3n apply', P.solves);
        log('lowOrderApplies after one 3n apply', P.lowOrderApplies);
        // 3 blocks x 2 solves, and still ONE `L^{2−s}` apply: the 3 columns share the traversal,
        // the block lists and the kernel `pow` (spec §5.2's "3 columns on the same tree").
        expect(P.solves).toBe(2 + 6);
        expect(P.lowOrderApplies).toBe(2);
    });

    test('the 3n apply is bit-identical to 3 scalar applies', () => {
        const n = 240;
        const { P } = prep(n);
        const R = randVec(3 * n, 21);
        const got = new Float64Array(3 * n);
        P.apply(R, got);
        for (let b = 0; b < 3; b++) {
            const one = new Float64Array(n);
            P.applyScalar(R.subarray(b * n, (b + 1) * n), one);
            for (let i = 0; i < n; i++) expect(got[b * n + i]).toBe(one[i]);
        }
    });

    test('the Laplacian solves actually solve (triResidual), closed curve => 2 Thomas sweeps', () => {
        const n = 480;
        const { P } = prep(n);
        expect(P.sys.closed).toBe(true);
        const r = randVec(n, 31);
        const out = new Float64Array(n);
        P.applyScalar(r, out);
        // `solveLaplacian`'s own honesty check on the SAME assembled bands the preconditioner
        // uses: a fast solve that does not solve is not a result (laplacian.ts). Re-solved with a
        // local workspace because the preconditioner's is internal; the system is `P.sys` itself,
        // so this gates the operator the preconditioner actually inverts.
        const x = new Float64Array(n);
        const rhs = randVec(n, 32);
        solveLaplacian(P.sys, rhs, x, makeLaplacianWorkspace(n));
        const resid = triResidual(P.sys, x, rhs);
        log(`triResidual of (−Δ) solve n=${n}`, resid);
        // laplacian.ts's own derived stability bar, `1000 * N * u_f64`.
        expect(resid).toBeLessThan(1000 * n * U_F64);
    });
});

describe('precond: symmetry and definiteness on 1⊥', () => {
    test('Ã⁻¹ is symmetric to roundoff and positive on 1⊥', () => {
        const n = 240;
        const { P } = prep(n);
        const mk = (seed: number) => {
            const v = randVec(n, seed);
            const mean = v.reduce((s, x) => s + x, 0) / n;
            for (let i = 0; i < n; i++) v[i] -= mean;
            return v;
        };
        const v = mk(41);
        const w = mk(42);
        const Mv = new Float64Array(n);
        const Mw = new Float64Array(n);
        P.applyScalar(v, Mv);
        P.applyScalar(w, Mw);
        let a = 0;
        let b = 0;
        let q = 0;
        let scale = 0;
        for (let i = 0; i < n; i++) {
            a += w[i] * Mv[i];
            b += v[i] * Mw[i];
            q += v[i] * Mv[i];
            scale += Math.abs(Mv[i] * w[i]);
        }
        log('Ã⁻¹ symmetry defect (relative), mean-free v and w', Math.abs(a - b) / scale);
        log('Ã⁻¹ Rayleigh quotient on 1⊥', q);
        // `Ã⁻¹ = Π S Π L Π S Π` with S and L symmetric, so symmetry is structural; it is measured
        // because an operand-order slip would break it silently and GMRES does not check.
        expect(Math.abs(a - b) / scale).toBeLessThan(1e-12);
        expect(q).toBeGreaterThan(0);
        // ON MEAN-FREE VECTORS THIS TEST IS BLIND TO THE OUTER `Π`. Deleting `precond.ts:195`'s
        // final `project` left 44/0 passing: on `1⊥` the operator WITHOUT it genuinely IS
        // symmetric (it reads 4.9e-18 either way), because the defect it admits is a constant
        // component, which is orthogonal to every `w` here. MEASURED on GENERAL `v, w`: 5.0e-17
        // with the projector, 3.3e-3 without (fix round 1, finding I2). Same bar, different draw.
        const gv = randVec(n, 43);
        const gw = randVec(n, 44);
        const Mgv = new Float64Array(n);
        const Mgw = new Float64Array(n);
        P.applyScalar(gv, Mgv);
        P.applyScalar(gw, Mgw);
        let ga = 0;
        let gb = 0;
        let gscale = 0;
        for (let i = 0; i < n; i++) {
            ga += gw[i] * Mgv[i];
            gb += gv[i] * Mgw[i];
            gscale += Math.abs(Mgv[i] * gw[i]);
        }
        log(
            'mean(v), mean(w) of the general pair',
            `${gv.reduce((x, y) => x + y, 0) / n}, ${gw.reduce((x, y) => x + y, 0) / n}`,
        );
        log('Ã⁻¹ symmetry defect (relative), GENERAL v and w', Math.abs(ga - gb) / gscale);
        expect(Math.abs(ga - gb) / gscale).toBeLessThan(1e-12);
        // Positive on 10 random directions in 1⊥ — the matvec-only stand-in for SPD.
        for (let s = 0; s < 10; s++) {
            const u = mk(100 + s);
            const Mu = new Float64Array(n);
            P.applyScalar(u, Mu);
            let qq = 0;
            for (let i = 0; i < n; i++) qq += u[i] * Mu[i];
            expect(qq).toBeGreaterThan(0);
        }
    });

    test('Ã⁻¹ leaves no constant component on a mean-free rhs (the OUTER Π)', () => {
        // The second half of finding I2, and the direct statement of what `precond.ts:195` buys:
        // `core = Π S Π L Π S Π` must land in `1⊥`, so for `r ⊥ 1` the whole result is
        // `Ã⁻¹r ⊥ 1` — the `+11ᵀ/(nσ)` term contributes `(1ᵀr)/(nσ) = 0`. Both halves of I2 are
        // implemented rather than one: this one states the invariant directly (a reader does not
        // have to reason about which draw exposes it), the symmetry one is the discriminator the
        // reviewer actually measured, and together they pin the projector from both sides.
        // MEASURED, this metric: 1.5e-16 with the projector, 1.94e-2 without.
        for (const n of [64, 240, 480]) {
            const { P } = prep(n);
            const r = randVec(n, 11);
            const mean = r.reduce((x, y) => x + y, 0) / n;
            for (let i = 0; i < n; i++) r[i] -= mean;
            const out = new Float64Array(n);
            P.applyScalar(r, out);
            let sum = 0;
            for (let i = 0; i < n; i++) sum += out[i];
            const leak = (Math.abs(sum / n) * Math.sqrt(n)) / l2(out, n);
            log(`|mean(Ã⁻¹r)|√n / ‖Ã⁻¹r‖, r ⊥ 1, n=${n}`, leak);
            // BAR `32 u_f64`, absolute. MEASURED 0.140 u / 0.667 u / 1.27 u at n=64/240/480 —
            // roughly LINEAR in n (it is one `project` residual per solve), so at the n=30 000
            // target this bar would need revisiting; it is not a constant of the algorithm.
            expect(leak).toBeLessThan(32 * U_F64);
        }
    });

    test('Ã⁻¹A_σ is close to the identity in the sense GMRES needs (right-hand residual)', () => {
        // Not a gate: `κ_eig` is the gated quantity and it lives in `scratch/*.py` (§11.2 row 11).
        // This records the one thing a matvec instrument can say about conditioning: how far
        // `Ã⁻¹A_σ v` is from `v` on a random direction, for the production operator and for the
        // knock-out side by side.
        const n = 240;
        for (const drop of [false, true]) {
            const { H, op, P } = prep(n, { dropConstantMode: drop });
            const v = randVec(n, 51);
            const av = new Float64Array(n);
            const mav = new Float64Array(n);
            applyASigma(H, op.sigma, n, v, av);
            P.applyScalar(av, mav);
            let num = 0;
            let dot = 0;
            for (let i = 0; i < n; i++) {
                num += (mav[i] - v[i]) ** 2;
                dot += mav[i] * v[i];
            }
            const cos = dot / (l2(mav, n) * l2(v, n));
            // `Ã⁻¹` is RS-faithful only up to an overall SCALE (the ~29x below), which Krylov does
            // not see; the scale-invariant number is the angle. Both recorded so the 29 cannot be
            // misread as the preconditioner failing.
            log(`‖Ã⁻¹A_σv − v‖/‖v‖ dropConstantMode=${drop}`, Math.sqrt(num) / l2(v, n));
            log(`‖Ã⁻¹A_σv‖/‖v‖ (the scale) dropConstantMode=${drop}`, l2(mav, n) / l2(v, n));
            log(`cos∠(Ã⁻¹A_σv, v) dropConstantMode=${drop}`, cos);
        }
    });
});

describe('precond: the mass-weight provenance gap (module TSDoc)', () => {
    test('how far the production Ã⁻¹ moves as massWeight varies', () => {
        // The lab's `D⁺` is `pinv(Π Lap Π)` of the PURE-STIFFNESS Laplacian (`gmres_lab.py:95-110`,
        // singular). Production cannot use it — `massWeight <= 0` throws (laplacian.ts) — so
        // `massWeight = 1` is the ruling and the two operators differ. This measures by how much,
        // instead of assuming the oracle's `κ_eig` transfers.
        const n = 240;
        const base = prep(n);
        const r = randVec(n, 61);
        const mean = r.reduce((s, x) => s + x, 0) / n;
        for (let i = 0; i < n; i++) r[i] -= mean;
        const ref = new Float64Array(n);
        base.P.applyScalar(r, ref);
        for (const massWeight of [1, 0.1, 0.01, 1e-3]) {
            const { P } = prep(n, { massWeight });
            const got = new Float64Array(n);
            P.applyScalar(r, got);
            let num = 0;
            let dot = 0;
            for (let i = 0; i < n; i++) {
                num += (got[i] - ref[i]) ** 2;
                dot += got[i] * ref[i];
            }
            // The NORM ratio says how differently the two operators SCALE, and the scale is
            // irrelevant to Krylov (`κ_eig(cM A) = κ_eig(M A)`). The scale-invariant quantity —
            // does the preconditioner point the same way — is the angle, so both are recorded and
            // the norm one must not be read as "the operators disagree by 234x".
            log(`mass=${massWeight}: ‖ΔÃ⁻¹‖/‖Ã⁻¹(mass=1)‖ on r⊥1`, Math.sqrt(num) / l2(ref, n));
            log(`mass=${massWeight}: ‖Ã⁻¹‖/‖Ã⁻¹(mass=1)‖`, l2(got, n) / l2(ref, n));
            log(`mass=${massWeight}: cos∠(Ã⁻¹r, Ã⁻¹(mass=1)r)`, dot / (l2(got, n) * l2(ref, n)));
        }
        // `massWeight <= 0` is the lab's pure-stiffness limit and is SINGULAR on a closed curve
        // (constants in the kernel of pure stiffness); `laplacian.ts` refuses it rather than
        // reaching for a pseudo-inverse, which is why production and the oracle differ at all.
        expect(() => prep(n, { massWeight: 0 })).toThrow(CurveTopologyError);
    });

    test('sigma must be the structural tr(A_bct)/n, and a bad one is refused', () => {
        const n = 64;
        const { vertices, edges, Lo, op } = prep(n);
        log('sigma = tr(A_bct)/n at n=64', op.sigma);
        for (const bad of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
            expect(() => makePreconditioner(vertices, edges, Lo, bad)).toThrow(/sigma/);
        }
        // A mismatched `L^{2−s}` is refused too: a preconditioner built from another curve's
        // operator would read as a plausible but wrong one.
        const other = prep(240).Lo;
        expect(() => makePreconditioner(vertices, edges, other, op.sigma)).toThrow(/n=240/);
    });

    test("a knock-out L^{2-s} is refused unless let through, like prepareA's", () => {
        // GREPPED before adding this (fix round 1, finding I3): `makePreconditioner` has exactly
        // two call sites in the repo, both in this file, and `'rs-identity'` appears in `src/`
        // only as the type literal and its docstrings — nothing legitimate builds a
        // preconditioner on it. What made the guard Important anyway is that flipping
        // `lowOrder.ts`'s default variant to `'rs-identity'` used to leave ALL 13 tests in this
        // file passing, while `Ã⁻¹` built on it annihilates the checkerboard — the mode `A_σ`
        // weights MOST (`e4_precond_integrity.py`, `gmres_lab.py`'s KNOWN DEFECT note). Same
        // reasoning and same shape as `apply.ts:482-490` (commit `2a8b88b`).
        const n = 64;
        const { vertices, edges, bvh, op } = prep(n);
        const source = { kind: 'lists', bct: buildBct(bvh, { theta: THETA }) } as const;
        const bad = prepareLowOrder(vertices, edges, bvh, source, {
            alpha: ALPHA,
            beta: BETA,
            variant: 'rs-identity',
        });
        expect(bad.variant).toBe('rs-identity');
        expect(() => makePreconditioner(vertices, edges, bad, op.sigma)).toThrow(/knock-out/);
        expect(() =>
            makePreconditioner(vertices, edges, bad, op.sigma, { dropConstantMode: true }),
        ).toThrow(/rs-identity/);
        // The escape, which the κ tests above use: explicit, in-test, and it returns a working
        // object rather than a half-built one.
        const escaped = makePreconditioner(vertices, edges, bad, op.sigma, { knockOut: true });
        expect(escaped.lowOrder.variant).toBe('rs-identity');
        const out = new Float64Array(n);
        escaped.applyScalar(randVec(n, 81), out);
        expect(Number.isFinite(out[0])).toBe(true);
        // And the production variant still goes through with no flag.
        expect(() => makePreconditioner(vertices, edges, prep(n).Lo, op.sigma)).not.toThrow();
    });
});

describe('precond: cost per application (spec §5.2 trap 1, the half that is measurable here)', () => {
    test('Ã⁻¹ vs A_σ wall-clock, n=64 and n=960', () => {
        // Spec §5.2 trap 1, VERBATIM: "The RS preconditioner is **worse than no
        // preconditioner below n ≈ 223.** Unpreconditioned GMRES needs 45 iterations at n=64 where
        // RS needs 29, but the crossover is real and the dense path stays the right answer at
        // small n." Those two sentences disagree on ITERATIONS (29 < 45 makes RS better), so the
        // only reading on which "worse" holds is WALL-CLOCK: per iteration, left-preconditioned
        // GMRES pays one `A_σ` matvec PLUS one `Ã⁻¹`.
        //
        // THE RATIO BELOW IS LOGGED, NOT ASSERTED, AND IT DOES NOT SETTLE THAT READING. Break-even
        // is `29 · ratio = 45`, i.e. ratio = 1.5517, and the measurement STRADDLES it under load:
        // three independent observers read 1.65-2.63 isolated and 1.15-1.20 in-suite, a 2.2x swing
        // across a 1.55 threshold. A load-dependent quantity cannot decide it. The crossover is
        // task 7's to decide, on ITERATION COUNTS, where the same §5.2 numbers are unambiguous.
        for (const n of [64, 960]) {
            const { H, P } = prep(n);
            const U = randVec(3 * n, 71);
            const out = new Float64Array(3 * n);
            const timeIt = (f: () => void, reps: number) => {
                f();
                const t0 = performance.now();
                for (let i = 0; i < reps; i++) f();
                return (performance.now() - t0) / reps;
            };
            const msA = timeIt(() => applyA(H, U, out), 10);
            const msM = timeIt(() => P.apply(U, out), 10);
            log(`applyA (12-col) ms n=${n}`, msA);
            log(`Ã⁻¹ (3-col) ms n=${n}`, msM);
            log(`per-iteration cost ratio (A+Ã⁻¹)/A n=${n}`, (msA + msM) / msA);
            // 45 unpreconditioned iterations vs 29 RS iterations (spec §5.2, MEASURED) in units
            // of one unpreconditioned iteration:
            const ratio = (msA + msM) / msA;
            log(`DERIVED wall-clock, 29 RS iters vs 45 plain, n=${n}`, `${29 * ratio} vs 45`);
            expect(msA).toBeGreaterThan(0);
            expect(msM).toBeGreaterThan(0);
        }
    });
});
