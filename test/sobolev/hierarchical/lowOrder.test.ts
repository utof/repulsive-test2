/**
 * Spec §11.2 task 6 of Phase 2H, first half: the fractional operator `L^{2−s}` on the SAME tree
 * and the SAME block lists as `A`, and gate H5 property 1 (`dim null(L^{2−s}) == 1`).
 *
 * ORACLE / PROVENANCE. H5's artefacts are
 * `.superpowers/sdd/2026-09-11-phase2h-queue/scratch/e4-closer-0b.json` (property 1, 8 rows) and
 * `e0-closer-0b.json` (properties 2-3, 6 rows). Both were produced by `e4_precond_integrity.py` /
 * `e0_sanity.py` on `gmres_lab.py`'s `trefoil`, which is the repo's `src/core/fixtures.ts`
 * `trefoil(n)` VERBATIM (same formula, same `[i, (i+1)%n]` edges) and **UNnormalised** — bbox max
 * extent 5.4716. That is OUTSIDE the SCALE PRECONDITION `bvh.ts:32-38` documents for itself, and
 * it is deliberate: rescaling one curve moves `κ_near` 1.70x and `nFar` 1.76x, so an
 * `out-bct-*.json` number (extent 1.0, from `test/sobolev/hierarchical/fixtures.ts`) and an
 * `e4-closer-0b.json` number are NOT comparable. Every test here uses the unnormalised
 * `src/core/fixtures.ts trefoil` and nothing here quotes a `test/.../fixtures.ts` curve — the one
 * cross-operator comparison (`δ_compress` of `A` vs of `L^{2−s}`) is MEASURED here, on this
 * fixture at this scale, rather than read off `out-accuracy.json`.
 * @issue utof/repulsive-test2#69
 *
 * The dense `L^{2−s}` and the dense eigensolver below are TEST ONLY and O(n²)/O(n³): `src/` never
 * allocates a dense `W` (7.2 GB at n=30 000, spec §7.1).
 *
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.2, §7.8, §11.2 task 6
 * @see .superpowers/sdd/2026-10-09-phase2h-bct-gmres-design/task-6-brief.md
 * @issue utof/repulsive-test2#59
 *
 * REVIEW PROVENANCE. Comments below that cite a measured counter-example come from this task's
 * two blind reviews and the scoped re-review, kept at
 * `.superpowers/sdd/2026-10-09-phase2h-bct-gmres-design/` as `task-6-review-gates.md`,
 * `task-6-review-math.md` and `task-6-rereview.md`. Cited here once rather than inline, so the
 * individual comments carry the measurement and not an ephemeral finding label.
 */
import { describe, expect, setDefaultTimeout, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { trefoil } from '../../../src/core/fixtures';
import { applyA, prepareA } from '../../../src/core/sobolev/hierarchical/apply';
import { buildBct } from '../../../src/core/sobolev/hierarchical/bct';
import { buildBvh } from '../../../src/core/sobolev/hierarchical/bvh';
import { CurveTopologyError } from '../../../src/core/sobolev/hierarchical/laplacian';
import {
    applyLowOrder,
    applyLowOrderColumns,
    type LowOrderA,
    type LowOrderVariant,
    lowOrderExponent,
    prepareLowOrder,
} from '../../../src/core/sobolev/hierarchical/lowOrder';
import type { Edge, Vec3 } from '../../../src/core/testConfigs';
import { eigAt, RANK_TOL_REL, spectrum, type Tridiag } from './denseSpectrum';

setDefaultTimeout(900_000);

const ALPHA = 3;
const BETA = 6;
/** `A`'s regularisation. `L^{2−s}`'s own epsilon is 0 — see the `prepareLowOrder` TSDoc. */
const EPS_A = 1e-10;
const LEAF = 8;
const U_F32 = 5.96e-8;
/** The production default (`bct.ts` `DEFAULT_THETA`, H6 RED, commit 4c7af4d). */
const THETA = 0.25;
const SCRATCH = `${import.meta.dir}/../../../.superpowers/sdd/2026-09-11-phase2h-queue/scratch`;

const log = (name: string, v: unknown) => console.log(`[task6] ${name} = ${v}`);

interface E4Row {
    n: number;
    variant: LowOrderVariant;
    nullDimLfrac: number;
    lambda1: number;
    lambda2: number;
    lambdaMax: number;
    rankTolAbs: number;
    rankTolRel: number;
    kappaPrecond: number;
}
const e4: E4Row[] = JSON.parse(readFileSync(`${SCRATCH}/e4-closer-0b.json`, 'utf8')).rows;
const e4row = (n: number, variant: LowOrderVariant): E4Row => {
    const r = e4.find((x) => x.n === n && x.variant === variant);
    if (!r) throw new Error(`no e4-closer-0b row ${n}/${variant}`);
    return r;
};

// ---------------------------------------------------------------------------
// The dense references. TEST ONLY, O(n²) — `src/` never forms these (spec §7.1).
// Transcribed from `gmres_lab.py` with its op order: raw `ell` and raw distances (NO epsilon
// anywhere), `K_ii = 0`. `Math.sqrt` of the dot product, not `Math.hypot`, so the distances
// agree with `np.linalg.norm` to the last bits.
// ---------------------------------------------------------------------------

const dist = (X: ArrayLike<number>, i: number, Y: ArrayLike<number>, j: number): number => {
    const dx = X[3 * i] - Y[3 * j];
    const dy = X[3 * i + 1] - Y[3 * j + 1];
    const dz = X[3 * i + 2] - Y[3 * j + 2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
};

const flat = (vertices: Vec3[]): Float64Array => {
    const X = new Float64Array(3 * vertices.length);
    for (let i = 0; i < vertices.length; i++) {
        X[3 * i] = vertices[i][0];
        X[3 * i + 1] = vertices[i][1];
        X[3 * i + 2] = vertices[i][2];
    }
    return X;
};

/** `gmres_lab.py:163-166`: `a[p] += 0.5*ell[I]; a[q] += 0.5*ell[I]`, in EDGE-LIST order. */
function dualWeights(vertices: Vec3[], edges: Edge[]): Float64Array {
    const X = flat(vertices);
    const a = new Float64Array(vertices.length);
    for (const [p, q] of edges) {
        const l = dist(X, p, X, q);
        a[p] += 0.5 * l;
        a[q] += 0.5 * l;
    }
    return a;
}

/**
 * `gmres_lab.py:148-171` `frac_laplacian_curve`, row-major `n x n`. TEST ONLY, O(n²).
 *
 * `eps` defaults to 0, which is the lab and the production default. It is a parameter because
 * `LowOrderOptions.epsilon` is one, and the regularisation's PLACEMENT — `(‖d‖ + ε)^p`, not
 * `(‖d‖² + ε)^{p/2}` — is otherwise untested at every ε that could distinguish them.
 */
function denseVertex(vertices: Vec3[], edges: Edge[], p: number, eps = 0): Float64Array {
    const n = vertices.length;
    const X = flat(vertices);
    const a = dualWeights(vertices, edges);
    const L = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
        let row = 0;
        for (let j = 0; j < n; j++) {
            if (i === j) continue;
            const w = a[i] * a[j] * (dist(X, i, X, j) + eps) ** -p;
            L[i * n + j] = -2 * w;
            row += w;
        }
        L[i * n + i] = 2 * row;
    }
    return L;
}

/** `gmres_lab.py:128-146` `frac_laplacian_curve_rs_identity`, row-major. TEST ONLY, O(n²). */
function denseRsIdentity(vertices: Vec3[], edges: Edge[], p: number): Float64Array {
    const n = vertices.length;
    const m = edges.length;
    const V = flat(vertices);
    const ell = new Float64Array(m);
    const X = new Float64Array(3 * m);
    for (let I = 0; I < m; I++) {
        ell[I] = dist(V, edges[I][0], V, edges[I][1]);
        for (let c = 0; c < 3; c++) {
            X[3 * I + c] = 0.5 * (V[3 * edges[I][0] + c] + V[3 * edges[I][1] + c]);
        }
    }
    const H = new Float64Array(m * m);
    for (let I = 0; I < m; I++) {
        for (let J = 0; J < m; J++) if (I !== J) H[I * m + J] = dist(X, I, X, J) ** -p;
    }
    const Hell = new Float64Array(m);
    for (let I = 0; I < m; I++) {
        let s = 0;
        for (let J = 0; J < m; J++) s += H[I * m + J] * ell[J];
        Hell[I] = s;
    }
    // Mid = diag((H ell)/ell) − H, then 2 Uᵀ Mid U with U[I, a] = U[I, b] = 0.5 ell_I.
    const L = new Float64Array(n * n);
    for (let I = 0; I < m; I++) {
        for (let J = 0; J < m; J++) {
            const mid = (I === J ? Hell[I] / ell[I] : 0) - H[I * m + J];
            const c = 2 * 0.25 * ell[I] * ell[J] * mid;
            for (const r of edges[I]) for (const s of edges[J]) L[r * n + s] += c;
        }
    }
    return L;
}

const denseOf = (
    variant: LowOrderVariant,
    vertices: Vec3[],
    edges: Edge[],
    p: number,
): Float64Array =>
    variant === 'vertex' ? denseVertex(vertices, edges, p) : denseRsIdentity(vertices, edges, p);

function denseMv(L: Float64Array, n: number, v: ArrayLike<number>): Float64Array {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        let s = 0;
        for (let j = 0; j < n; j++) s += L[i * n + j] * v[j];
        out[i] = s;
    }
    return out;
}

/** TEST ONLY, O(n) applies + O(n²) storage: the densified hierarchical `L^{2−s}`, row-major. */
function densifyLowOrder(Lo: LowOrderA): Float64Array {
    const n = Lo.n;
    const A = new Float64Array(n * n);
    const e = new Float64Array(n);
    const col = new Float64Array(n);
    for (let j = 0; j < n; j++) {
        e.fill(0);
        e[j] = 1;
        applyLowOrder(Lo, e, col);
        for (let i = 0; i < n; i++) A[i * n + j] = col[i];
    }
    return A;
}

// The dense symmetric eigensolver H5 property 1 rests on lives in `denseSpectrum.ts`, shared
// with `precond.test.ts` so the two gates cannot drift apart. Its module TSDoc carries the
// provenance (NR §11.2/§11.4, 8-digit agreement with `e4-closer-0b.json`) and the reason it is
// NOT the matrix-free inverse iteration spec §7.8 names. The matrix-free alternative that DOES
// work on the production variant is at the bottom of this file.

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

const { sigmaOrder, p: P_LOW } = lowOrderExponent(ALPHA, BETA);

const cache = new Map<string, ReturnType<typeof build>>();
function build(n: number, theta: number, variant: LowOrderVariant) {
    const { vertices, edges } = trefoil(n);
    const bvh = buildBvh(vertices, edges, LEAF, EPS_A);
    const source =
        theta === 0
            ? ({ kind: 'stream', params: { theta: 0 } } as const)
            : ({ kind: 'lists', bct: buildBct(bvh, { theta }) } as const);
    const Lo = prepareLowOrder(vertices, edges, bvh, source, { alpha: ALPHA, beta: BETA, variant });
    return { vertices, edges, bvh, source, Lo };
}
function prep(n: number, theta: number, variant: LowOrderVariant) {
    const k = `${n}/${theta}/${variant}`;
    let v = cache.get(k);
    if (!v) {
        v = build(n, theta, variant);
        cache.set(k, v);
    }
    return v;
}

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

/** `‖x − y‖₂ / ‖y‖₂`. */
function relErr(x: ArrayLike<number>, y: ArrayLike<number>): number {
    let num = 0;
    let den = 0;
    for (let i = 0; i < y.length; i++) {
        const dd = x[i] - y[i];
        num += dd * dd;
        den += y[i] * y[i];
    }
    return Math.sqrt(num / den);
}

// ===========================================================================

describe('lowOrder: the exponent', () => {
    test('2 − s = 1/3 and the kernel exponent is 5/3, not 7/3', () => {
        // The high-order ENERGY kernel's distance exponent is 2σ+1 = 7/3 (`nearField.ts`
        // `exponents`); `L^{2−s}`'s is 1 + 2(2−s) = 5/3. A reader has already confused them
        // (spec §5.2). They also coincide with `s` at alpha=3, beta=6 (5 − 2s = 5/3 = s), which
        // is why this is pinned rather than eyeballed.
        log('sigmaOrder (2-s)', sigmaOrder);
        log('kernel exponent p', P_LOW);
        expect(sigmaOrder).toBeCloseTo(1 / 3, 15);
        expect(P_LOW).toBeCloseTo(5 / 3, 15);
        // `2 − s ∈ (0, 2)` is `[RS §5.2]`'s admissibility condition for the fractional operator.
        expect(sigmaOrder).toBeGreaterThan(0);
        expect(sigmaOrder).toBeLessThan(2);
    });
});

describe('lowOrder: the hierarchical operator IS the dense reference (spec §5.6)', () => {
    // theta = 0 is the exact all-inadmissible stream path (spec §7.1) — every pair summed
    // exactly. Agreement there pins the TRANSCRIPTION and the slot->vertex collocation map at f64
    // roundoff. theta > 0 is the COMPRESSION error, a different quantity with a different
    // provenance; the two must not share a bar.
    for (const n of [240, 480]) {
        for (const variant of ['vertex', 'rs-identity'] as const) {
            test(`theta=0 reproduces the dense ${variant} transcription at n=${n}`, () => {
                const { vertices, edges, Lo } = prep(n, 0, variant);
                const L = denseOf(variant, vertices, edges, P_LOW);
                const v = randVec(n, 7 + n);
                const got = new Float64Array(n);
                applyLowOrder(Lo, v, got);
                const err = relErr(got, denseMv(L, n, v));
                log(`exact-vs-dense relErr ${variant} n=${n}`, err);
                expect(err).toBeLessThan(1e-13);
            });
        }
    }

    test('compression error of L^{2−s} vs that of A, SAME fixture, scale, theta and rhs', () => {
        // This is spec §5.6's real open question ("the same tree and the same block lists" for a
        // VERTEX-collocated kernel on an EDGE tree). The brief asked for <= 1e-6 relative against the dense
        // reference at theta > 0 and says a miss means "same tree, same block lists is not viable
        // for a vertex-collocated kernel". MEASURED, that inference does not hold: at theta = 0
        // the transcription agrees to ~1e-15 (above), so the mapping is exact; what remains at
        // theta > 0 is the ordinary rank-1 lumping error, the same `δ_compress` gate H1-a measures
        // for `A` — and `A`'s own is the SAME ORDER on this fixture. A bar of 1e-6 would reject
        // the operator this phase ships.
        for (const n of [240, 480]) {
            const { vertices, edges, bvh } = prep(n, THETA, 'vertex');
            const dense = denseOf('vertex', vertices, edges, P_LOW);
            const v = randVec(n, 7 + n);
            const gotL = new Float64Array(n);
            applyLowOrder(prep(n, THETA, 'vertex').Lo, v, gotL);
            const dL = relErr(gotL, denseMv(dense, n, v));

            // `δ_compress(A)`: theta=0.25 lists against the theta=0 stream (the exact operator),
            // one scalar apply, same curve, same rhs. Spec §7.2's definition.
            const mk = (theta: number) =>
                prepareA(
                    bvh,
                    n,
                    theta === 0
                        ? { kind: 'stream', params: { theta: 0 } }
                        : { kind: 'lists', bct: buildBct(bvh, { theta }) },
                    { alpha: ALPHA, beta: BETA, epsilon: EPS_A },
                );
            const U = new Float64Array(3 * n);
            for (let i = 0; i < n; i++) U[i] = v[i];
            const exactA = new Float64Array(3 * n);
            const bctA = new Float64Array(3 * n);
            applyA(mk(0), U, exactA);
            applyA(mk(THETA), U, bctA);
            const dA = relErr(bctA.subarray(0, n), exactA.subarray(0, n));

            log(`delta_compress L^{2-s} theta=${THETA} n=${n}`, dL);
            log(`delta_compress A       theta=${THETA} n=${n}`, dA);
            log(`ratio L/A n=${n}`, dL / dA);
            // The claim: `L^{2−s}` compresses AT LEAST AS WELL as `A` does on the same tree.
            expect(dL).toBeLessThan(2 * dA);
            // And neither is anywhere near 1e-6 — that bar belongs to a precision quantity
            // (task 4's f32-vs-f64 apply, 1.5e-6), not to a compression quantity.
            expect(dA).toBeGreaterThan(1e-5);
        }
    });

    test('the BCT apply at theta=0.25 IS the block-list transcription (the FAR path, exactly)', () => {
        // THE EXACT PIN ON THE FAR PATH. The theta=0 test above pins the transcription where
        // every block is INADMISSIBLE, so it never evaluates `farKernel`, `far[]`, `nodeC` or the
        // percolation at all; and `delta_compress` is non-monotone in a far-field defect (a
        // uniform far-kernel x1.01 IMPROVES it on this geometry, 6.477e-3 -> 6.281e-3 at n=240),
        // so no ratio bar can close that hole. This closes it: an INDEPENDENT transcription of the
        // compressed operator at the production theta, built in this file from `bct.farA/farB`,
        // `bct.nearA/nearB` and `bvh` alone, with its own dual weights, its own collocation points
        // and its own w-weighted node barycentres. Nothing on the FAR PATH is shared with
        // `lowOrder.ts`: not `farKernel`, `far[]`, the `s` percolation, the `acc` down-scatter, nor
        // `forEachBlock`. The one shared input is the exponent `P_LOW` via `lowOrderExponent`, so a
        // defect in `p` is common-mode HERE and is caught instead by the 15-digit exponent pin and
        // the 10-digit `λ_max`-vs-oracle pins (measured: `p × 1.01` leaves this test green at
        // 6.35e-16 and fails 9 others).
        //
        //   far block (A,B):  W~[s][t] += w_s w_t k(nodeC_A, nodeC_B)   for ALL s in A, t in B
        //                     (the `acc` percolation scatters down to every descendant slot)
        //   near block (A,B): W~[s][t] += w_s w_t k(X_s, X_t)           for s != t
        //   then              L = 2(diag(W~ 1) - W~), read out at vertex `vOf[s]`
        //
        // Fix round 1, finding I1.
        for (const n of [240, 480]) {
            const { vertices, edges, bvh, Lo } = prep(n, THETA, 'vertex');
            const bct = buildBct(bvh, { theta: THETA });
            const m = bvh.order.length;
            expect(m).toBe(n);
            // --- independent slot data ---
            const a = dualWeights(vertices, edges);
            const vOf = new Int32Array(m);
            const w = new Float64Array(m);
            const Xs = new Float64Array(3 * m);
            for (let slot = 0; slot < m; slot++) {
                const i = bvh.sE[2 * slot];
                vOf[slot] = i;
                w[slot] = a[i];
                for (let c = 0; c < 3; c++) Xs[3 * slot + c] = bvh.sXa[3 * slot + c];
            }
            // --- independent w-weighted node barycentres ---
            const C = new Float64Array(3 * bvh.nNodes);
            for (let id = 0; id < bvh.nNodes; id++) {
                let mass = 0;
                const c3 = [0, 0, 0];
                for (let slot = bvh.begin[id]; slot < bvh.end[id]; slot++) {
                    mass += w[slot];
                    for (let c = 0; c < 3; c++) c3[c] += w[slot] * Xs[3 * slot + c];
                }
                for (let c = 0; c < 3; c++) C[3 * id + c] = mass > 0 ? c3[c] / mass : 0;
            }
            let worstC = 0;
            for (let id = 0; id < 3 * bvh.nNodes; id++) {
                worstC = Math.max(worstC, Math.abs(C[id] - Lo.nodeC[id]));
            }
            log(`independent nodeC vs Lo.nodeC, max abs diff n=${n}`, worstC);
            expect(worstC).toBeLessThan(1e-14);
            // --- the compressed matrix, in SLOT space ---
            const Wt = new Float64Array(m * m);
            const kern = (P: Float64Array, i: number, Q: Float64Array, j: number) =>
                dist(P, i, Q, j) ** -P_LOW;
            for (let k = 0; k < bct.stats.nFar; k++) {
                const A = bct.farA[k];
                const B = bct.farB[k];
                const kv = kern(C, A, C, B);
                for (let st = bvh.begin[A]; st < bvh.end[A]; st++) {
                    for (let t = bvh.begin[B]; t < bvh.end[B]; t++) {
                        Wt[st * m + t] += w[st] * w[t] * kv;
                    }
                }
            }
            let nearDiag = 0;
            for (let k = 0; k < bct.stats.nNear; k++) {
                const A = bct.nearA[k];
                const B = bct.nearB[k];
                for (let st = bvh.begin[A]; st < bvh.end[A]; st++) {
                    for (let t = bvh.begin[B]; t < bvh.end[B]; t++) {
                        if (st === t) {
                            nearDiag++;
                            continue;
                        }
                        Wt[st * m + t] += w[st] * w[t] * kern(Xs, st, Xs, t);
                    }
                }
            }
            // The same diagonal count the operator reports, from an independent walk of the lists.
            expect(nearDiag).toBe(Lo.m);
            expect(Lo.diagSkipped).toBe(Lo.m);
            // --- L = 2(diag(W~ 1) - W~), applied to a random vertex vector ---
            const v = randVec(n, 13 + n);
            const want = new Float64Array(n);
            for (let st = 0; st < m; st++) {
                let row = 0;
                let dot = 0;
                for (let t = 0; t < m; t++) {
                    row += Wt[st * m + t];
                    dot += Wt[st * m + t] * v[vOf[t]];
                }
                want[vOf[st]] = 2 * (row * v[vOf[st]] - dot);
            }
            const got = new Float64Array(n);
            applyLowOrder(Lo, v, got);
            const err = relErr(got, want);
            log(`bct-vs-block-list-transcription relErr theta=${THETA} n=${n}`, err);
            // ROUNDOFF, not compression: both sides evaluate the SAME compressed operator, only
            // the summation order differs (percolation vs row-major). MEASURED 7e-16 / 1e-15.
            // A uniform far-kernel x1.01 moves this to ~1e-2.
            expect(err).toBeLessThan(1e-13);
            log(`nFar / nNear theta=${THETA} n=${n}`, `${bct.stats.nFar} / ${bct.stats.nNear}`);
            expect(bct.stats.nFar).toBeGreaterThan(0);
        }
    });

    test('lists(theta=0.25) == stream(theta=0.25) bit-for-bit, so far[] matches the inline kernel', () => {
        // Spec §5.6's ONE-TRAVERSAL requirement, the same pin `apply.test.ts` carries for `A`:
        // a `'lists'` source precomputes `far[k] = farKernel(nodeC, farA[k], farB[k], ...)` while
        // a `'stream'` source evaluates it inline mid-traversal, so bit-identity pins the
        // precompute's INDEXING (which `k` is which pair) and the traversal order against each
        // other at theta > 0.
        //
        // WHAT IT CANNOT CATCH, stated because the dispatch expected it to: both paths call the
        // SAME `farKernel`, so a defect INSIDE that function (a x1.01 factor, a wrong exponent, a
        // wrong `nodeC` index) moves both sides identically and this test stays GREEN. VERIFIED
        // by running it under the x1.01 mutant (fix round 1 report). The test above is the one
        // that fails there; this one is the indexing pin.
        const n = 240;
        const { vertices, edges, bvh, Lo } = prep(n, THETA, 'vertex');
        const stream = prepareLowOrder(
            vertices,
            edges,
            bvh,
            { kind: 'stream', params: { theta: THETA } },
            { alpha: ALPHA, beta: BETA },
        );
        expect(Lo.far).not.toBeNull();
        expect(stream.far).toBeNull();
        const v = randVec(n, 23);
        const a = new Float64Array(n);
        const b = new Float64Array(n);
        applyLowOrder(Lo, v, a);
        applyLowOrder(stream, v, b);
        for (let i = 0; i < n; i++) expect(b[i]).toBe(a[i]);
        // And the row sums, which `prepareLowOrder` computed through the same traversal.
        for (let st = 0; st < Lo.m; st++) expect(stream.rowSum[st]).toBe(Lo.rowSum[st]);
    });

    for (const theta of [0.5, 0.25, 0.1]) {
        test(`L^{2−s} compression falls with theta (theta=${theta}, n=480)`, () => {
            const n = 480;
            const { vertices, edges, Lo } = prep(n, theta, 'vertex');
            const dense = denseOf('vertex', vertices, edges, P_LOW);
            const v = randVec(n, 7 + n);
            const got = new Float64Array(n);
            applyLowOrder(Lo, v, got);
            const err = relErr(got, denseMv(dense, n, v));
            log(`compress relErr vertex theta=${theta} n=${n}`, err);
            // MEASURED: 1.6346e-2 / 6.9997e-3 / 1.4483e-3 at theta = 0.5 / 0.25 / 0.1, i.e.
            // `err / theta^1.5` = 0.0462 / 0.0560 / 0.0458 — the O(theta^(3/2)) rate of a rank-1
            // (degree-0) far lump, with a coefficient flat to 22 % over a 5x range of theta.
            // The bars BRACKET THAT COEFFICIENT, 0.02 to 0.12: ~2.3x below the smallest and ~2.1x
            // above the largest. The previous bars (`10 theta^1.5` / `0.01 theta^1.5`) sat 179-218x
            // above and 4.6-5.6x below the measurement, so the upper one could not fire — and the
            // 9.75e-4 they quoted at theta=0.1 was wrong.
            expect(err).toBeLessThan(0.12 * theta ** 1.5);
            expect(err).toBeGreaterThan(0.02 * theta ** 1.5);
        });
    }
});

describe('lowOrder: L·1 = 0 bit-exactly and structurally (spec §7.8 property 1)', () => {
    for (const variant of ['vertex', 'rs-identity'] as const) {
        for (const theta of [0, THETA]) {
            test(`L·1 == 0 EXACTLY, ${variant}, theta=${theta}`, () => {
                const { Lo } = prep(240, theta, variant);
                const ones = new Float64Array(240).fill(1);
                const out = new Float64Array(240);
                applyLowOrder(Lo, ones, out);
                let worst = 0;
                for (const x of out) worst = Math.max(worst, Math.abs(x));
                log(`max|L·1| ${variant} theta=${theta}`, worst);
                // NOT a tolerance. `rowSum` is produced by the SAME apply on the same input
                // (spec §2.2; `apply.ts` does this for `K1`/`K⁰1`), so `rowSum_i * 1 − (W·1)_i`
                // is `x − x` on identical bits. A scheme that computes the diagonal separately
                // breaks this and then H5 property 1 fails for an implementation artefact.
                expect(worst).toBe(0);
            });
        }
    }

    test('the 3-column apply is bit-identical to 3 scalar applies', () => {
        const n = 240;
        const { Lo } = prep(n, THETA, 'vertex');
        const U = new Float64Array(3 * n);
        for (let b = 0; b < 3; b++) {
            const v = randVec(n, 101 + b);
            for (let i = 0; i < n; i++) U[b * n + i] = v[i];
        }
        const got = new Float64Array(3 * n);
        applyLowOrderColumns(Lo, U, got, 3);
        for (let b = 0; b < 3; b++) {
            const one = new Float64Array(n);
            applyLowOrder(Lo, U.subarray(b * n, (b + 1) * n), one);
            for (let i = 0; i < n; i++) expect(got[b * n + i]).toBe(one[i]);
        }
    });

    test('the rs-identity knock-out also annihilates the checkerboard EXACTLY', () => {
        // The defect it exists to exhibit: on a closed polyline |E| = |V|, so `U` is square and
        // for even n annihilates `(−1)^i` (`gmres_lab.py:117-124`). `(U v)_I = ell_I (v_a+v_b)/2`
        // and `v_a + v_b = 0` there, exactly — so this is 0 on identical bits, not a tolerance,
        // and the resulting nullity is 2 rather than 1.
        const n = 240;
        const { Lo } = prep(n, THETA, 'rs-identity');
        const cb = new Float64Array(n);
        for (let i = 0; i < n; i++) cb[i] = i % 2 === 0 ? 1 : -1;
        const out = new Float64Array(n);
        applyLowOrder(Lo, cb, out);
        let worst = 0;
        for (const x of out) worst = Math.max(worst, Math.abs(x));
        log('max|L_rs · checkerboard|', worst);
        expect(worst).toBe(0);
        // The production variant does NOT kill it — that is the whole point of the variant.
        const prod = prep(n, THETA, 'vertex');
        applyLowOrder(prod.Lo, cb, out);
        let q = 0;
        for (let i = 0; i < n; i++) q += cb[i] * out[i];
        log('checkerboard Rayleigh quotient, vertex variant', q / n);
        expect(q / n).toBeGreaterThan(1);
    });
});

describe('lowOrder: structure of the hierarchical W', () => {
    test('symmetric and PSD (the two properties nullity 1 rests on)', () => {
        const n = 240;
        const { Lo } = prep(n, THETA, 'vertex');
        const v = randVec(n, 3);
        const w = randVec(n, 4);
        const Lv = new Float64Array(n);
        const Lw = new Float64Array(n);
        applyLowOrder(Lo, v, Lv);
        applyLowOrder(Lo, w, Lw);
        let a = 0;
        let b = 0;
        let q = 0;
        let scale = 0;
        for (let i = 0; i < n; i++) {
            a += w[i] * Lv[i];
            b += v[i] * Lw[i];
            q += v[i] * Lv[i];
            scale += Math.abs(Lv[i] * w[i]);
        }
        log('symmetry defect |wᵀLv − vᵀLw| (relative)', Math.abs(a - b) / scale);
        log('Rayleigh quotient vᵀLv', q);
        expect(Math.abs(a - b) / scale).toBeLessThan(1e-13);
        expect(q).toBeGreaterThan(0);
    });

    test('the diagonal skip counter is m per apply, not 0', () => {
        // A counter pinned at 0 reads exactly like a pass (memory `gates-name-their-impossible-
        // values`). Every slot sits in exactly one diagonal near block, so `i == j` is skipped
        // exactly m times per apply at any theta — and never in a far block, which is what the
        // adjacency certificate buys here.
        for (const theta of [0, THETA]) {
            const { Lo } = prep(240, theta, 'vertex');
            const out = new Float64Array(240);
            applyLowOrder(Lo, randVec(240, 5), out);
            log(`diagSkipped / m theta=${theta}`, `${Lo.diagSkipped} / ${Lo.m}`);
            expect(Lo.diagSkipped).toBe(Lo.m);
        }
    });

    test('the vertex side needs its OWN node mass, barycentre and radius (spec §5.6)', () => {
        // The tree is built over EDGE primitives (midpoints + unit tangents in R⁶); `L^{2−s}` is
        // collocated at VERTICES. "The same tree and the same block lists" therefore cannot mean
        // identical per-node data: the block LISTS and the topology are reused, the node DATA is
        // recomputed. These two numbers say how much that matters.
        const n = 480;
        const { vertices, edges, bvh, Lo } = prep(n, THETA, 'vertex');
        let worstR = 0;
        let worstC = 0;
        for (let id = 0; id < bvh.nNodes; id++) {
            if (bvh.rx[id] > 0) worstR = Math.max(worstR, Lo.nodeRx[id] / bvh.rx[id]);
            const dc = dist(Lo.nodeC, id, bvh.C, id);
            worstC = Math.max(worstC, dc / Math.max(bvh.rx[id], 1e-300));
        }
        log(`max vertex-rx / edge-rx n=${n}`, worstR);
        log(`max |vertexC − edgeC| / edge-rx n=${n}`, worstC);
        // Each vertex sits `ell/2` off its edge's midpoint, so the vertex radius EXCEEDS the edge
        // radius and the barycentre moves. What that is worth is MEASURED, not asserted: swapping
        // `bvh.C` in for `nodeC` moves `δ_compress` by +2.5 % at n=240 and +0.16 % at n=480 and
        // leaves the nullity and `λ_max` unchanged. The recompute is right and stays — the lump
        // belongs at the points the kernel is collocated on — but it is an accuracy improvement,
        // NOT an admissibility guard: the block lists are reused as `bct.ts` built them and
        // `nodeRx` is never read by the apply.
        expect(worstR).toBeGreaterThan(1);
        expect(worstC).toBeGreaterThan(0.01);

        // `nodeMass` is DIAGNOSTIC (its TSDoc) — the far lump's weight is the percolated `s[B]`,
        // which depends on `psi`. It is the `psi = 1` case of that percolation, which makes it the
        // cheapest independent check that the bottom-up pass is right, so it is CHECKED here
        // rather than left dead: an un-read field with no test is a field a mutant cannot move
        //.
        let total = 0;
        for (const [a, b] of edges) total += dist(flat(vertices), a, flat(vertices), b);
        log(`nodeMass[root] vs total curve length n=${n}`, `${Lo.nodeMass[0]} vs ${total}`);
        expect(Math.abs(Lo.nodeMass[0] / total - 1)).toBeLessThan(1e-14);
        let internal = 0;
        let worstSplit = 0;
        for (let id = 0; id < bvh.nNodes; id++) {
            const l = bvh.left[id];
            const r = bvh.right[id];
            if (l <= 0 || r <= 0) continue; // a leaf; only the root has child id 0
            internal++;
            const sum = Lo.nodeMass[l] + Lo.nodeMass[r];
            worstSplit = Math.max(worstSplit, Math.abs(Lo.nodeMass[id] / sum - 1));
        }
        log(`internal nodes checked n=${n}`, internal);
        log('max |nodeMass[parent]/(left+right) − 1|', worstSplit);
        expect(internal).toBeGreaterThan(10);
        // Not bit-exact, and that is arithmetic rather than a defect: the parent sums its whole
        // slot range in order, the children sum two sub-ranges, so the associations differ.
        expect(worstSplit).toBeLessThan(1e-14);
    });

    test('epsilon regularises `(‖d‖ + ε)^p`, and that placement is pinned', () => {
        // `epsilon` defaults to 0 and production never sets it (module TSDoc: `A`'s regularisation
        // regularises the ENERGY and "has no business in a preconditioner"). KEPT rather than
        // dropped, because `LowOrderOptions` is exported and a caller composing this with `A`'s
        // own `EPS_A` is a reasonable thing to try — but a parameter never exercised non-zero is a
        // parameter whose placement is untested: a mutant moving `(‖d‖ + ε)^p` to
        // `(‖d‖² + ε)^{p/2}` survives vacuously at ε = 0. ε is taken
        // LARGE (0.05, ~4 % of the mean edge length at n=240) so the two spellings cannot agree.
        const n = 240;
        const eps = 0.05;
        const { vertices, edges } = prep(n, 0, 'vertex');
        const bvh = buildBvh(vertices, edges, LEAF, EPS_A);
        const src = { kind: 'lists', bct: buildBct(bvh, { theta: 0 }) } as const;
        const Lo = prepareLowOrder(vertices, edges, bvh, src, {
            alpha: ALPHA,
            beta: BETA,
            epsilon: eps,
        });
        expect(Lo.epsilon).toBe(eps);
        const v = randVec(n, 97);
        const got = new Float64Array(n);
        applyLowOrder(Lo, v, got);
        // Same ε on both sides, θ=0 so the comparison is exact-vs-exact.
        const want = denseMv(denseVertex(vertices, edges, P_LOW, eps), n, v);
        log(`eps=${eps} relErr vs dense WITH the same eps, n=${n}`, relErr(got, want));
        expect(relErr(got, want)).toBeLessThan(1e-13);
        // And ε actually CHANGES the operator by far more than that bar, so the agreement above
        // is not vacuous: MEASURED 0.272 relative against the ε = 0 reference (ε = 0.05 is ~42 %
        // of the mean edge length at n=240), so the two ε spellings cannot coincide either.
        const wantEps0 = denseMv(denseVertex(vertices, edges, P_LOW, 0), n, v);
        log(`eps=${eps} relErr vs the eps=0 dense reference, n=${n}`, relErr(got, wantEps0));
        expect(relErr(got, wantEps0)).toBeGreaterThan(1e-3);
        // `L·1 = 0` survives regularisation — the diagonal is still the row sum of the SAME sum.
        const ones = new Float64Array(n).fill(1);
        const l1 = new Float64Array(n);
        applyLowOrder(Lo, ones, l1);
        let worst = 0;
        for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(l1[i]));
        log(`max|L·1| with eps=${eps}`, worst);
        expect(worst).toBe(0);
    });

    test('an open chain is refused for the vertex variant (no slot/vertex bijection)', () => {
        // On an open chain `m = n − 1`, so the edges' first endpoints miss vertex `n − 1` and
        // there is no slot to collocate it on. Silently dropping a vertex would make `L^{2−s}` a
        // different operator; `CurveTopologyError` names it instead. (`laplacian.ts` restricts the
        // preconditioner's topology anyway, so this is not a lost capability.)
        const n = 32;
        const vertices: Vec3[] = [];
        const edges: Edge[] = [];
        for (let i = 0; i < n; i++) vertices.push([i * 0.1, Math.sin(i * 0.3), 0]);
        for (let i = 0; i < n - 1; i++) edges.push([i, i + 1]);
        const bvh = buildBvh(vertices, edges, LEAF, EPS_A);
        const source = { kind: 'lists', bct: buildBct(bvh, { theta: THETA }) } as const;
        expect(() =>
            prepareLowOrder(vertices, edges, bvh, source, { alpha: ALPHA, beta: BETA }),
        ).toThrow(CurveTopologyError);
        // The edge-collocated knock-out has no such restriction (|E| need not equal |V|).
        expect(() =>
            prepareLowOrder(vertices, edges, bvh, source, {
                alpha: ALPHA,
                beta: BETA,
                variant: 'rs-identity',
            }),
        ).not.toThrow();
    });

    test('a knock-out block source is refused unless let through', () => {
        const n = 64;
        const { vertices, edges } = trefoil(n);
        const bvh = buildBvh(vertices, edges, LEAF, EPS_A);
        for (const params of [
            { theta: THETA, adjacencyCertificate: false },
            { theta: THETA, invertAdmissibility: true },
        ]) {
            const bct = buildBct(bvh, params);
            const src = { kind: 'lists', bct } as const;
            expect(() =>
                prepareLowOrder(vertices, edges, bvh, src, { alpha: ALPHA, beta: BETA }),
            ).toThrow(/knock-out/);
            expect(() =>
                prepareLowOrder(vertices, edges, bvh, src, {
                    alpha: ALPHA,
                    beta: BETA,
                    knockOut: true,
                }),
            ).not.toThrow();
        }
    });
});

describe('H5 property 1: dim null(L^{2−s}) == 1 (vertex) / 2 (rs-identity)', () => {
    for (const n of [64, 120, 240, 480]) {
        for (const variant of ['vertex', 'rs-identity'] as const) {
            test(`n=${n} ${variant}`, () => {
                const or = e4row(n, variant);
                const { vertices, edges } = prep(n, THETA, variant);

                // (a) The DENSE transcription — reproduces the oracle's own instrument.
                const sD = spectrum(denseOf(variant, vertices, edges, P_LOW), n);
                // (b) The HIERARCHICAL operator at the production theta — the thing `src/` ships,
                //     and a number that has never existed before. Densified by n applies.
                const sH = spectrum(densifyLowOrder(prep(n, THETA, variant).Lo), n);

                for (const [tag, s] of [
                    ['dense', sD],
                    [`bct-theta${THETA}`, sH],
                ] as const) {
                    log(
                        `nullDim ${variant} n=${n} ${tag}`,
                        `${s.nullDim} (oracle ${or.nullDimLfrac})`,
                    );
                    log(`lambda1 ${variant} n=${n} ${tag}`, `${s.lambda1} (oracle ${or.lambda1})`);
                    log(`lambda2 ${variant} n=${n} ${tag}`, `${s.lambda2} (oracle ${or.lambda2})`);
                    log(
                        `lambdaMax ${variant} n=${n} ${tag}`,
                        `${s.lambdaMax} (oracle ${or.lambdaMax})`,
                    );
                    log(
                        `rankTol ${variant} n=${n} ${tag}`,
                        `${s.rankTol} (oracle ${or.rankTolAbs})`,
                    );
                    log(`lambda2/lambdaMax ${variant} n=${n} ${tag}`, s.lambda2 / s.lambdaMax);

                    // --- the gate itself ---
                    expect(s.nullDim).toBe(or.nullDimLfrac);

                    // --- the three INVALID predicates, per memory `gates-name-their-impossible-values` ---
                    // (1) `dim null == 0` ⇒ INVALID: structurally impossible, the constants are
                    //     in the nullspace by construction for EVERY variant, so 0 would mean the
                    //     rank tolerance is mis-set rather than that the operator is non-singular.
                    expect(s.nullDim).not.toBe(0);
                    // (2) `dim null >= 3` ⇒ INVALID, not RED: the known defect gives exactly 2
                    //     (the checkerboard), so 3+ is not a known failure shape and must not be
                    //     scored as one.
                    expect(s.nullDim).toBeLessThan(3);
                    // (3) `λ₂/λ_max < 10 × rankTolRel` ⇒ INVALID — SCOPED TO THE PRODUCTION
                    //     VERTEX VARIANT ONLY. Unscoped it fires on the knock-out at every n
                    //     (MEASURED λ₂/λ_max ≈ 1e-15…1e-16), reporting it INVALID instead of RED
                    //     and defeating it — spec §7.8: "the identical error class as #59
                    //     classifying H4b's q <= 0 as INVALID". The knock-out's tiny ratio is
                    //     CORRECT behaviour: its nullity genuinely is 2, so λ₂ genuinely is a
                    //     null mode.
                    if (variant === 'vertex') {
                        expect(s.lambda2 / s.lambdaMax).toBeGreaterThan(10 * RANK_TOL_REL);
                    } else {
                        expect(s.lambda2 / s.lambdaMax).toBeLessThan(10 * RANK_TOL_REL);
                    }
                }

                // The dense transcription must BE the oracle's matrix, not merely agree in
                // verdict: λ_max and (for the vertex variant) λ₂ to 10 digits.
                expect(sD.lambdaMax / or.lambdaMax).toBeCloseTo(1, 10);
                if (variant === 'vertex') expect(sD.lambda2 / or.lambda2).toBeCloseTo(1, 10);
                // λ₁ is a FLOATING-POINT ZERO in the oracle (±3.5e-15 at n=64, sign varying with
                // n) because `eigvalsh` of a dense matrix cannot return an exact 0. It is
                // STRUCTURALLY zero here, and `max|L·1| == 0` above proves it on the operator
                // itself, so it is gated against the rank tolerance rather than against a value.
                expect(Math.abs(sD.lambda1)).toBeLessThan(sD.rankTol);
                expect(Math.abs(sH.lambda1)).toBeLessThan(sH.rankTol);
                // Compression moves λ_max by O(δ_compress) and must NOT move the nullity. ASSERTED,
                // not just logged: a far-field error that `δ_compress` cannot see (it is
                // non-monotone in the defect, fix round 1 finding I1) may still move `λ_max`, so
                // this is a second, independent reading of the same compression. MEASURED across
                // all 8 cells: 0.99627 .. 1.00085, i.e. within 3.8e-3 — the bar is 1e-2.
                log(`lambdaMax bct/dense ${variant} n=${n}`, sH.lambdaMax / sD.lambdaMax);
                expect(Math.abs(sH.lambdaMax / sD.lambdaMax - 1)).toBeLessThan(1e-2);
            });
        }
    }

    test('matrix-free λ₂, production variant, n=960 — beyond the dense range', () => {
        // §7.8 asserts "property 1 is an integer at any n (inverse iteration, matrix-free, O(N)
        // storage)" but names an instrument this phase cannot build (no factorization, §1.1).
        // What DOES work matrix-free on the production variant: Lanczos on
        //     M = L + (λ_max/n) 1 1ᵀ,
        // which moves the constant mode from 0 to λ_max EXACTLY (`M 1 = L1 + λ_max·1`) and leaves
        // `M = L` on 1⊥. So `λ_min(M) = λ₂`. Shifting the OPERATOR, not projecting the VECTORS,
        // is what keeps the constant direction from leaking back in as a spurious zero — a
        // mean-subtracting projection measured a ghost λ₂ of −8.1e-15 at n=64 (task-6 report).
        //
        // DIRECTION OF THE BOUND, stated because it decides what this can conclude: Ritz values
        // interlace from above, so the smallest Ritz value is an UPPER bound on λ₂. This can
        // FALSIFY "nullity 1"; the PROOF is structural (every `W̃_ij > 0` ⇒ connected weighted
        // graph Laplacian ⇒ nullity exactly 1, `lowOrder.ts` TSDoc).
        const n = 960;
        const { Lo } = prep(n, THETA, 'vertex');
        const Lv = new Float64Array(n);
        // λ_max by plain power iteration (converges from below; only its scale matters here).
        let v = randVec(n, 77);
        let lmax = 0;
        for (let it = 0; it < 120; it++) {
            applyLowOrder(Lo, v, Lv);
            let nrm = 0;
            for (let i = 0; i < n; i++) nrm += Lv[i] * Lv[i];
            nrm = Math.sqrt(nrm);
            lmax = 0;
            for (let i = 0; i < n; i++) lmax += v[i] * Lv[i];
            for (let i = 0; i < n; i++) v[i] = Lv[i] / nrm;
        }
        const shift = lmax / n;
        const mv = (x: Float64Array, out: Float64Array) => {
            applyLowOrder(Lo, x, out);
            let s = 0;
            for (let i = 0; i < n; i++) s += x[i];
            const c = shift * s;
            for (let i = 0; i < n; i++) out[i] += c;
        };
        // Lanczos with full reorthogonalisation, 80 steps: O(80 n) storage.
        const K = 80;
        const d = new Float64Array(K);
        const e = new Float64Array(K);
        const Q: Float64Array[] = [];
        const w = new Float64Array(n);
        v = randVec(n, 99);
        let nv = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
        for (let i = 0; i < n; i++) v[i] /= nv;
        let steps = 0;
        for (let j = 0; j < K; j++) {
            mv(v, w);
            let a = 0;
            for (let i = 0; i < n; i++) a += v[i] * w[i];
            d[j] = a;
            steps = j + 1;
            for (let i = 0; i < n; i++) w[i] -= a * v[i];
            if (j > 0) for (let i = 0; i < n; i++) w[i] -= e[j - 1] * Q[j - 1][i];
            for (let pass = 0; pass < 2; pass++) {
                for (const q of [...Q, v]) {
                    let dot = 0;
                    for (let i = 0; i < n; i++) dot += q[i] * w[i];
                    for (let i = 0; i < n; i++) w[i] -= dot * q[i];
                }
            }
            nv = Math.sqrt(w.reduce((s, x) => s + x * x, 0));
            Q.push(v);
            if (!(nv > 1e-13 * Math.abs(a))) break;
            e[j] = nv;
            v = new Float64Array(n);
            for (let i = 0; i < n; i++) v[i] = w[i] / nv;
        }
        const t: Tridiag = { d: d.subarray(0, steps), e: e.subarray(0, Math.max(0, steps - 1)) };
        const lambda2 = eigAt(t, 0);
        const rankTol = RANK_TOL_REL * lmax;
        log(`matrix-free lambdaMax n=${n}`, lmax);
        log(`matrix-free lambda2 (upper bound) n=${n}`, lambda2);
        log(`matrix-free lambda2/lambdaMax n=${n}`, lambda2 / lmax);
        log(`matrix-free Lanczos steps n=${n}`, steps);
        // The predicate that would have fired had the constant mode not been the ONLY null mode.
        expect(lambda2).toBeGreaterThan(10 * rankTol);
        // §7.8: λ₂/λ_max decays only as n^-0.53 (0.204 at n=64 -> 0.0702 at n=480), so at n=960
        // it must still be ~5e-2, i.e. ~6 orders above the 1e-7 threshold.
        expect(lambda2 / lmax).toBeGreaterThan(1e-2);
    });
});

describe('lowOrder: cost, against spec §3’s DERIVED +25 %', () => {
    test('one 3-column L^{2−s} apply vs one 12-column A matvec at n=960', () => {
        const n = 960;
        const { bvh, Lo } = prep(n, THETA, 'vertex');
        const bct = buildBct(bvh, { theta: THETA });
        const opts = { alpha: ALPHA, beta: BETA, epsilon: EPS_A };
        const src = { kind: 'lists', bct } as const;
        const Hpre = prepareA(bvh, n, src, opts);
        const Hinline = prepareA(bvh, n, src, { ...opts, precomputeNear: false });
        const U = randVec(3 * n, 9);
        const out = new Float64Array(3 * n);
        const timeIt = (f: () => void, reps: number) => {
            f();
            const t0 = performance.now();
            for (let i = 0; i < reps; i++) f();
            return (performance.now() - t0) / reps;
        };
        const msPre = timeIt(() => applyA(Hpre, U, out), 5);
        const msInline = timeIt(() => applyA(Hinline, U, out), 5);
        const msL = timeIt(() => applyLowOrderColumns(Lo, U, out, 3), 5);
        log(`applyA 12-col ms, near PREcomputed, n=${n}`, msPre);
        log(`applyA 12-col ms, near INLINE, n=${n}`, msInline);
        log(`L^{2-s} 3-col ms, near INLINE, n=${n}`, msL);
        log('L/A ratio vs production A (near precomputed)', msL / msPre);
        log('L/A ratio vs inline A (matched kernel evaluation)', msL / msInline);
        log('nearPairs', bct.stats.nearPairs);
        log('nFar', bct.stats.nFar);
        // WHY `L^{2−s}` HAS NO NEAR PRECOMPUTE, and the real price of that. `prepareA` caches its
        // near kernel values (`precomputeNear`); this operator evaluates them inline, which is
        // the whole of the `msL / msPre` gap above (`msL / msInline` is the matched comparison).
        // The precompute was DELIBERATELY DEFERRED, not ruled out: it is a performance feature
        // with its own storage/accuracy trade-off and its own gates. The honest price of deferring
        // it is right above: `msL / msPre` means one 3-column `L^{2−s}` apply costs about as much
        // as one whole 12-column `A` matvec (1.08 .. 1.36 across runs), so a preconditioned
        // iteration costs ~2x an unpreconditioned one — while `msL / msInline` (~0.10) shows that
        // per kernel evaluation this operator is the cheaper of the two. The storage it would cost is
        // `8 B x nearPairs` = **51.8 MiB f64** at the production theta=0.25 at N=30 000 (spec
        // §3.4's near-pair table: 6 785 670 pairs for the trefoil), 189.9 MiB for the worst case
        // there (tangle at theta=0.1, 24 886 020 pairs), half of each in f32. NOT 3.6 GB: that is
        // §7.1's `chi = 0` ALL-PAIRS case (`3.6e9/8 = 4.5e8 = 30 000²/2` exactly) and reading it as
        // the near-pair cost over-states the storage 66x — whoever sizes that task should start
        // from 51.8 MiB.
        //
        // Non-gate: a wall-clock on one machine, f64. Asserted only so the measurement cannot
        // silently become a no-op. `msInline > msPre` was asserted here and is now logged only:
        // the gap is 9.6x so it rarely flips, but one GC pause inside the 85 ms `msPre` window
        // can flip it, and the log already carries everything the assertion did (finding I4).
        expect(msL).toBeGreaterThan(0);
        expect(msPre).toBeGreaterThan(0);
        expect(msInline).toBeGreaterThan(0);
    });
});

describe('lowOrder: property 2 and 3 oracles (spec §11.2 row 11 — the LAB operator)', () => {
    test('e0-closer-0b.json / e4-closer-0b.json carry properties 2 and 3', () => {
        // §11.2 row 11: "H5 and H6 stay in `scratch/*.py`, where their dense-eigenvalue
        // instruments already live". THESE ROWS CHARACTERISE THE LAB'S `Ã⁻¹`, whose `(−Δ)⁻¹` is
        // the pseudo-inverse of the PURE-STIFFNESS Laplacian — a different operator from the one
        // `src/` ships (`precond.ts`'s provenance note). They are kept, and read as the lab's,
        // because they are what the gate was registered against and what `p` was fitted on.
        //
        // The earlier claim here — that `Ã⁻¹A_σ` is non-symmetric so "nothing else in this repo
        // can" reach its eigenvalues — was WRONG, and it blocked the obvious closer for a round.
        // `Ã⁻¹A_σ` is indeed not symmetric, but with `A_σ = R Rᵀ` it is SIMILAR to `Rᵀ Ã⁻¹ R`,
        // which is, so the same Householder+Sturm instrument reaches it. `precond.test.ts` now
        // measures properties 2 and 3 on the SHIPPED operator (κ_incl 59.0 / 64.6 / 69.7 / 71.9 at
        // n=64/120/240/480, extent 5.4716) and validates the harness against these very rows.
        const e0 = JSON.parse(readFileSync(`${SCRATCH}/e0-closer-0b.json`, 'utf8'));
        const kap: number[] = e0.rows.map((r: { kappaPrecond: number }) => r.kappaPrecond);
        const ns: number[] = e0.rows.map((r: { n: number }) => r.n);
        const fit = (idx: number[]) => {
            let sx = 0;
            let sy = 0;
            let sxx = 0;
            let sxy = 0;
            for (const i of idx) {
                const x = Math.log(ns[i]);
                const y = Math.log(kap[i]);
                sx += x;
                sy += y;
                sxx += x * x;
                sxy += x * y;
            }
            const k = idx.length;
            return (k * sxy - sx * sy) / (k * sxx - sx * sx);
        };
        // EVERY FIT MUST NAME ITS POINT SET (spec §7.8's provenance note: #59's `p = 0.0010`
        // reproduces only over the four LARGEST points, and "every number was sound and only the
        // archive was missing — which is indistinguishable from fabrication until someone
        // re-runs it").
        const pAll = fit([0, 1, 2, 3, 4, 5]);
        const pSmall4 = fit([0, 1, 2, 3]);
        const pLarge4 = fit([2, 3, 4, 5]);
        log('property2 p, all six points n=64..1920 (e0-closer-0b.json)', pAll);
        log('property2 p, four smallest n=64..480', pSmall4);
        log('property2 p, four largest n=240..1920', pLarge4);
        log('property3 kappa_eig(n=1920) (e0-closer-0b.json rows[5])', kap[5]);
        log('property3 kappa_eig * u_f32 at n=1920', kap[5] * U_F32);
        expect(pAll).toBeCloseTo(0.00645, 6);
        expect(pSmall4).toBeCloseTo(0.012152, 6);
        expect(pLarge4).toBeCloseTo(0.00099, 5);
        expect(Math.abs(pAll)).toBeLessThan(0.05);
        expect(kap[5] * U_F32).toBeLessThan(1e-2);
        // Property 3 is RECORDED, not a licence (spec §7.8): `[CH18]` must NOT be cited
        // for the 1e-2 threshold — its conditions are on `κ∞(A)` of the ORIGINAL matrix and its
        // relaxation is derived for GMRES preconditioned by low-precision LU factors, and this
        // phase has no factorization at all (§1.1). The applicable theory is inexact Krylov
        // (Simoncini & Szyld; van den Eshof & Sleijpen), which bounds the per-iteration matvec
        // perturbation — H1-b's `δ_precision`, measured in task 4. The threshold is CHOSEN.

        // The defect variant, from e4-closer-0b.json: 11 orders high, RED on property 3.
        for (const n of [64, 120, 240, 480]) {
            const d = e4row(n, 'rs-identity');
            log(`defect kappa_eig n=${n}`, d.kappaPrecond);
            log(`defect kappa_eig*u_f32 n=${n}`, d.kappaPrecond * U_F32);
            expect(d.kappaPrecond * U_F32).toBeGreaterThan(1e-2);
            const g = e4row(n, 'vertex');
            log(`vertex kappa_eig n=${n}`, g.kappaPrecond);
            log(`vertex kappa_eig*u_f32 n=${n}`, g.kappaPrecond * U_F32);
            expect(g.kappaPrecond * U_F32).toBeLessThan(1e-2);
        }
        // §11.2 row 6 writes the gated quantity as `κ₂(Ã⁻¹A_σ)·u_inner ≤ 1e-2`. That notation is
        // a known typo corrected by §7.8: the gated quantity is `κ_eig = λ_max/λ_min`, what
        // `e4_precond_integrity.py:37-38` computes (`|eigvals(Minv @ A)|`, max/min), while `κ₂`
        // denotes `cond(Ã⁻¹A_σ, 2)` — a DIFFERENT number, MEASURED 1.4544e4 vs 1.0977e4 at n=64,
        // a 1.33x gap persisting across the range. `kappaPrecond` in both artefacts is `κ_eig`.
        expect(e4row(64, 'vertex').kappaPrecond).toBeCloseTo(1.09765e4, -1);
    });
});
