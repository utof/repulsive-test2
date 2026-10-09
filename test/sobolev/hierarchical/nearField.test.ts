/**
 * Task 4 of Phase 2H, part 1: the `endpoint4` near field and its precomputed arrays.
 *
 * What is pinned here is STRUCTURE (exponents, which pairs are skipped, that the f32 variant is
 * really f32); the bit-level pin against `assembleAFlat`, the fidelity ladder and the precision
 * band are in `apply.test.ts`, because they are properties of the whole matvec.
 *
 * Conditions of every oracle-derived number below: fixtures transcribed from `scratch/curves.ts`
 * INCLUDING its bbox normalisation (max extent 1, centred at the origin — `curves.ts:49-66`), the
 * SCALE PRECONDITION `bvh.ts` states; `EPS = 1e-10`, `ALPHA = 3`, `BETA = 6`, `leaf = 8`, quadrature
 * `endpoint4`, `thetaT` off (`params` of `out-accuracy.json`).
 *
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.4, §5.6, §11.2 task 4, §11.3
 */
import { describe, expect, test } from 'bun:test';
import { buildBct } from '../../../src/core/sobolev/hierarchical/bct';
import { type Bvh, buildBvh } from '../../../src/core/sobolev/hierarchical/bvh';
import {
    endpoint4Kernel,
    exponents,
    F16_NEAR_KNOCKOUT,
    F32,
    F64,
    nearGeometry,
    precomputeNear,
} from '../../../src/core/sobolev/hierarchical/nearField';
import type { Edge, Vec3 } from '../../../src/core/testConfigs';

const EPS = 1e-10;
const ALPHA = 3;
const BETA = 6;

// ---------------------------------------------------------------------------------------------
// Fixtures — transcribed from `scratch/curves.ts` (same arithmetic, same op order), INCLUDING its
// bbox normalisation (`curves.ts:49-66`, applied by `closedFrom`/`openFrom`): every oracle below
// was measured on curves rescaled to bounding-box max extent 1, centred at the origin. That is a
// CONDITION of the oracles, not cosmetics: the R^6 split score adds squared diameters of three
// position and three unit-tangent coordinates in one number (`curves.ts:13-17`), so a different
// spatial scale changes the chosen split axes, the tree, and therefore kappa_near.
// ---------------------------------------------------------------------------------------------

interface Curve {
    vertices: Vec3[];
    edges: Edge[];
}

/** `scratch/curves.ts` `normalize`: bbox max extent 1, bbox centred at the origin. */
function normalize(X: Float64Array, n: number): void {
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n; i++) {
        for (let c = 0; c < 3; c++) {
            const v = X[3 * i + c];
            if (v < lo[c]) lo[c] = v;
            if (v > hi[c]) hi[c] = v;
        }
    }
    const ext = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
    const s = ext > 0 ? 1 / ext : 1;
    for (let i = 0; i < n; i++) {
        for (let c = 0; c < 3; c++) {
            X[3 * i + c] = (X[3 * i + c] - 0.5 * (lo[c] + hi[c])) * s;
        }
    }
}

function toCurve(X: Float64Array, n: number, closed: boolean): Curve {
    normalize(X, n);
    const vertices: Vec3[] = [];
    for (let i = 0; i < n; i++) vertices.push([X[3 * i], X[3 * i + 1], X[3 * i + 2]]);
    const edges: Edge[] = [];
    const m = closed ? n : n - 1;
    for (let I = 0; I < m; I++) edges.push([I, closed ? (I + 1) % n : I + 1]);
    return { vertices, edges };
}

/** `scratch/curves.ts` `torusKnot(n, 2, 3)`: R=2, r=0.8. */
function torusKnot(n: number): Curve {
    const p = 2;
    const q = 3;
    const X = new Float64Array(3 * n);
    const R = 2;
    const r = 0.8;
    for (let i = 0; i < n; i++) {
        const t = (2 * Math.PI * i) / n;
        const rad = R + r * Math.cos(q * t);
        X[3 * i] = rad * Math.cos(p * t);
        X[3 * i + 1] = rad * Math.sin(p * t);
        X[3 * i + 2] = r * Math.sin(q * t);
    }
    return toCurve(X, n, true);
}

/** `scratch/curves.ts` `openHelix(n, 4)` — open chain, `m = n - 1`. */
function openHelix(n: number): Curve {
    const turns = 4;
    const X = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) {
        const t = (turns * 2 * Math.PI * i) / (n - 1);
        X[3 * i] = Math.cos(t);
        X[3 * i + 1] = Math.sin(t);
        X[3 * i + 2] = (3 * i) / (n - 1);
    }
    return toCurve(X, n, false);
}

const FIXTURES = { torusKnot, openHelix } as const;
type FixtureName = keyof typeof FIXTURES;

const bvhCache = new Map<string, Bvh>();
function tree(fixture: FixtureName, n: number, leafSize = 8): Bvh {
    const key = `${fixture}/${n}/${leafSize}`;
    let bvh = bvhCache.get(key);
    if (!bvh) {
        const { vertices, edges } = FIXTURES[fixture](n);
        bvh = buildBvh(vertices, edges, leafSize, EPS);
        bvhCache.set(key, bvh);
    }
    return bvh;
}

// ---------------------------------------------------------------------------------------------
// 1. Exponents.
// ---------------------------------------------------------------------------------------------

test('exponents: pHi = 2*sigma+1 = 7/3, pLow = 2*sigma+5 = 19/3 at alpha=3, beta=6 (innerProduct.ts:52-54)', () => {
    const { pHi, pLow } = exponents(ALPHA, BETA);
    // Same arithmetic as src/core/sobolev/innerProduct.ts: s = (beta-1)/alpha; sigma = s-1.
    const sigma = (BETA - 1) / ALPHA - 1;
    expect(pHi).toBe(2 * sigma + 1);
    expect(pLow).toBe(2 * sigma + 5);
    // 2*sigma+5 and 19/3 differ by one ulp (6.333333333333334 vs ...333); the production
    // assembler's own arithmetic is the binding form, pinned exactly above.
    expect(pHi).toBeCloseTo(7 / 3, 14);
    expect(pLow).toBeCloseTo(19 / 3, 14);
});

// ---------------------------------------------------------------------------------------------
// 2. The near field skips exactly the pairs the energy omits, and nothing else.
// ---------------------------------------------------------------------------------------------

describe('precomputeNear: adjacentSkipped and the zero entries (spec §2.2, §5.5; task-4 brief ruling 1)', () => {
    // Structural count: `calculateDisjointPairs` (src/core/tangentPointEnergy.ts:44-58) omits
    // J == I and every J sharing a vertex with I. On a closed polyline every edge has exactly two
    // neighbours, so each ROW I meets 3 skipped columns; the BCT is a partition of ordered leaf
    // pairs (bct.test.ts §6) and the adjacency certificate keeps every adjacent pair out of the
    // far field (bct.test.ts §4), so every one of those 3m pairs is visited by the near field
    // exactly once. Open chain: the two end edges have one neighbour each => 3m - 2.
    // A pinned always-zero counter would read like a passing gate (memory
    // gates-name-their-impossible-values); 3m is the value a correctly plumbed counter MUST show.
    for (const [fixture, n, expected] of [
        ['torusKnot', 960, 3 * 960],
        ['torusKnot', 400, 3 * 400],
        ['openHelix', 400, 3 * 399 - 2],
    ] as [FixtureName, number, number][]) {
        test(`${fixture} n=${n} theta=0.5: adjacentSkipped = ${expected} for both kernels, and those entries are exactly 0`, () => {
            const bvh = tree(fixture, n);
            const bct = buildBct(bvh, { theta: 0.5 });
            for (const kernel of ['hi', 'low'] as const) {
                const pre = precomputeNear(bvh, bct, kernel, ALPHA, BETA, EPS, F64);
                expect(pre.kernel).toBe(kernel);
                expect(pre.adjacentSkipped).toBe(expected);
                expect(pre.offsets.length).toBe(bct.stats.nNear + 1);
                expect(pre.offsets[bct.stats.nNear]).toBe(bct.stats.nearPairs);
                expect(pre.vals.length).toBe(bct.stats.nearPairs);
                let zeros = 0;
                let positive = 0;
                for (const v of pre.vals) {
                    if (v === 0) zeros++;
                    else if (v > 0) positive++;
                }
                // Every non-skipped entry is a kernel value > 0: that is what makes
                // diag(K1) - K a non-negatively weighted graph Laplacian (spec §2.2).
                expect(zeros).toBe(expected);
                expect(positive).toBe(bct.stats.nearPairs - expected);
            }
        });
    }

    test('chi=0 at leafSize=1: adjacentSkipped = 3m with every ordered pair visited', () => {
        // The exact path visits every ordered edge pair once, so the counter is 3m here too —
        // but reached through m*m single-pair near blocks instead of leaf-leaf blocks.
        const bvh = tree('torusKnot', 128, 1);
        const bct = buildBct(bvh, { theta: 0 });
        expect(bct.stats.nearPairs).toBe(128 * 128);
        const pre = precomputeNear(bvh, bct, 'hi', ALPHA, BETA, EPS, F64);
        expect(pre.adjacentSkipped).toBe(3 * 128);
    });
});

// ---------------------------------------------------------------------------------------------
// 3. The endpoint4 kernel: the repo's 4-point rule, in the three accumulator variants.
// ---------------------------------------------------------------------------------------------

describe('endpoint4Kernel', () => {
    const { pHi, pLow } = exponents(ALPHA, BETA);

    test('f64: K_IJ = 2 w_IJ (high) and K0_IJ = w0_IJ + w0_JI (low), against a verbatim transcription of innerProduct.ts:88-101 / :236-252', () => {
        const bvh = tree('torusKnot', 64);
        const g = nearGeometry(bvh, EPS, F64);
        // Direct transcription of the repo's per-pair sums (innerProduct.ts), evaluated on the
        // slot-ordered endpoint coordinates. `w` = 0.25 * ellI * ellJ * sumDist; `wLow` likewise
        // with k24/dNorm^distExp; K = 2w (ordered pairs, both orientations); K0 = w0_IJ + w0_JI.
        const repoW = (s: number, t: number): { w: number; wLow: number } => {
            const T = [g.sT[3 * s], g.sT[3 * s + 1], g.sT[3 * s + 2]];
            let sumDist = 0;
            let sumLow = 0;
            for (const Xi of [g.sXa, g.sXb]) {
                for (const Xj of [g.sXa, g.sXb]) {
                    const dx = Xi[3 * s] - Xj[3 * t];
                    const dy = Xi[3 * s + 1] - Xj[3 * t + 1];
                    const dz = Xi[3 * s + 2] - Xj[3 * t + 2];
                    const dNorm = Math.sqrt(dx * dx + dy * dy + dz * dz) + EPS;
                    sumDist += 1 / dNorm ** pHi;
                    const cx = T[1] * dz - T[2] * dy;
                    const cy = T[2] * dx - T[0] * dz;
                    const cz = T[0] * dy - T[1] * dx;
                    const crossNorm = Math.sqrt(cx * cx + cy * cy + cz * cz) + EPS;
                    const k24 = (crossNorm * crossNorm) / (dNorm * dNorm * dNorm * dNorm);
                    sumLow += k24 / dNorm ** pHi;
                }
            }
            return {
                w: 0.25 * g.sEll[s] * g.sEll[t] * sumDist,
                wLow: 0.25 * g.sEll[s] * g.sEll[t] * sumLow,
            };
        };
        let checked = 0;
        for (let s = 0; s < 64; s += 7) {
            for (let t = 0; t < 64; t += 5) {
                const ia = g.sE[2 * s];
                const ib = g.sE[2 * s + 1];
                const ja = g.sE[2 * t];
                const jb = g.sE[2 * t + 1];
                if (ia === ja || ia === jb || ib === ja || ib === jb) continue;
                const kHi = endpoint4Kernel(g, s, t, 'hi', pHi, F64.round);
                const kLow = endpoint4Kernel(g, s, t, 'low', pLow, F64.round);
                const ij = repoW(s, t);
                const ji = repoW(t, s);
                expect(Math.abs(kHi - 2 * ij.w) / (2 * ij.w)).toBeLessThan(4e-15);
                expect(Math.abs(kLow - (ij.wLow + ji.wLow)) / (ij.wLow + ji.wLow)).toBeLessThan(
                    4e-15,
                );
                checked++;
            }
        }
        expect(checked).toBeGreaterThan(50);
    });

    test('f32 variant: every result is f32-representable, differs from f64, and the precomputed array equals the inline kernel bit-for-bit', () => {
        const bvh = tree('torusKnot', 400);
        const bct = buildBct(bvh, { theta: 0.5 });
        const g64 = nearGeometry(bvh, EPS, F64);
        const g32 = nearGeometry(bvh, EPS, F32);
        expect(g32.sXa).toBeInstanceOf(Float32Array);
        expect(g32.sEll).toBeInstanceOf(Float32Array);
        expect(g32.epsilon).toBe(Math.fround(EPS));
        for (const kernel of ['hi', 'low'] as const) {
            const p = kernel === 'hi' ? pHi : pLow;
            const pre32 = precomputeNear(bvh, bct, kernel, ALPHA, BETA, EPS, F32);
            expect(pre32.vals).toBeInstanceOf(Float32Array);
            let differ = 0;
            let o = 0;
            for (let k = 0; k < bct.stats.nNear; k++) {
                const A = bct.nearA[k];
                const B = bct.nearB[k];
                for (let s = bvh.begin[A]; s < bvh.end[A]; s++) {
                    for (let t = bvh.begin[B]; t < bvh.end[B]; t++, o++) {
                        const v = pre32.vals[o];
                        expect(Math.fround(v)).toBe(v);
                        if (v === 0) continue;
                        // Inline evaluation under the same variant is the same arithmetic.
                        expect(endpoint4Kernel(g32, s, t, kernel, p, F32.round)).toBe(v);
                        if (endpoint4Kernel(g64, s, t, kernel, p, F64.round) !== v) differ++;
                    }
                }
            }
            expect(o).toBe(bct.stats.nearPairs);
            // The impossible value: an "f32" kernel that agrees with f64 on every pair has not
            // been rounded (memory simulate-bugs-at-target-precision).
            expect(differ).toBeGreaterThan(0.9 * (bct.stats.nearPairs - pre32.adjacentSkipped));
        }
    });

    test('the three variants are distinct objects with the documented rounding', () => {
        expect(F64.round(0.1)).toBe(0.1);
        expect(F32.round(0.1)).toBe(Math.fround(0.1));
        expect(F16_NEAR_KNOCKOUT.round(0.1)).toBe(Math.fround(0.1));
        // The knock-out truncates ONLY the near accumulator (spec §11.2 task 4): everything else
        // is the f32 path.
        expect(F16_NEAR_KNOCKOUT.roundNear(0.1)).toBe(Math.f16round(0.1));
        expect(F16_NEAR_KNOCKOUT.roundNear(0.1)).not.toBe(Math.fround(0.1));
        expect(F64.roundNear).toBe(F64.round);
        expect(F32.roundNear).toBe(F32.round);
        expect(F64.alloc(3)).toBeInstanceOf(Float64Array);
        expect(F32.alloc(3)).toBeInstanceOf(Float32Array);
        expect(F16_NEAR_KNOCKOUT.alloc(3)).toBeInstanceOf(Float32Array);
    });
});
