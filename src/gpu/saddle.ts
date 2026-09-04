// src/gpu/saddle.ts — the CPU f64 half of the Phase 2a solve: shift, Schur, Woodbury,
// iterative refinement, on a factor produced elsewhere (the GPU Cholesky in slice 2a-2,
// or the f64 reference used by gate CR). NO `three` import — this file must run under
// `bun test` so CR can gate the algebra BEFORE any WGSL exists ([DESIGN §5]).
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1, §4.2, §5 (CR)
import {
    type ExternalFactorization,
    ExternalSolveError,
    luSolve,
    type PreparedSaddle,
    type SaddleFactorization,
} from '../core/sobolev/linsolve';

/**
 * Narrow a {@link SaddleFactorization} to its `'external'` member, throwing if it is
 * anything else.
 * Why this exists at all: `PreparedSaddle.factor` returns the WIDENED union and
 * [DESIGN §4.4] pins that signature verbatim, so it cannot be narrowed at the source —
 * but `solve` is a member of only ONE of the three arms, so `factor(C).solve(rhs)` is a
 * `TS2339` and every `bunx tsc --noEmit` in this plan would fail on it. The test suites
 * and gate CR all need `solve`, so the narrowing lives here once rather than being
 * re-spelt in three files. Uses the SAME `'kind' in fac && fac.kind === 'external'` test
 * D3 pins for `solveFactored`: `LuFactorization` (`linsolve.ts:219-223`) has no `kind`
 * field, so `fac.kind === …` alone is a TS error on this union.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4
 * @see docs/superpowers/plans/2026-09-04-webgpu-solver-phase2a-1.md (## Decisions, D3)
 */
export function asExternal(fac: SaddleFactorization): ExternalFactorization {
    if (!('kind' in fac && fac.kind === 'external'))
        throw new Error('asExternal: expected an external factorization');
    return fac;
}

/**
 * IR stops at this PURE relative residual `‖r_i‖₂/‖r‖₂`. Deliberately NOT the core's
 * own `residual` stat: `structuredSaddleResidual` (`linsolve.ts:860`) normalises by
 * `max(1, ‖r‖₂)` and is therefore ABSOLUTE for any rhs with `‖r‖₂ ≤ 1` — which the
 * projection rhs `[0; −Φ]` always is — so reusing it would accept a solve ~1e4× worse
 * than this criterion claims.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1 step 4, §4.2
 */
export const SADDLE_IR_TOL = 1e-10;

/**
 * Refinement budget: z₀ plus this many corrections.
 * Why 4, and why exceeding it throws rather than warns: [CAL §E.5] measures that a
 * factor sitting exactly AT K1's green bar needs all four refinements with ZERO spare,
 * and that the model's exhaustion level sits above the green bar — so K1 is NOT what
 * catches a factor between the two. The [DESIGN §4.1] step-4 throw is, per step, at run
 * time. ([CAL §E.6]'s `solvesTo1e10` of 4 at n=960 is z₀ + THREE refinements: the
 * `residualCurve[0] = 1.0` entry is the z=0 artefact the consumer compensates for.)
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-gate-calibration.md §E.5, §E.6
 */
export const SADDLE_IR_MAX_REFINEMENTS = 4;

/**
 * σ = trace(A)/n and A_σ = A + (σ/n)·𝟙𝟙ᵀ (row-major, n×n).
 * Why this exact shift: null(A) = span(𝟙) and is ONE-dimensional at every N measured,
 * including the disconnected `crossing` and `linked-rings` presets ([DESIGN §2]), so a
 * rank-1 shift is enough to make A_σ SPD — and it is positive exactly on the null
 * direction, `A_σ(𝟙/√n) = σ(𝟙/√n)`. It is removed EXACTLY by the Woodbury step, so it
 * does not perturb the metric (do not confuse with `buildSaddleMatrix`'s "no
 * regularizing identity" rule, `linsolve.ts:133-154`, which protects the METRIC).
 * `out` lets a caller reuse one buffer across steps — a per-step N×N allocation is a
 * measured throughput regression, not a style question ([CAL §H.1]: the "A_σ build +
 * f32 cast" constant spreads 10.7× at N=480 between the churn and reuse regimes).
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1 step 1
 */
export function shiftSobolev(
    A: Float64Array,
    n: number,
    out?: Float64Array,
): { Asigma: Float64Array; sigma: number } {
    let tr = 0;
    for (let i = 0; i < n; i++) tr += A[i * n + i];
    const sigma = tr / n;
    const s = sigma / n;
    const Asigma = out ?? new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) Asigma[i] = A[i] + s;
    return { Asigma, sigma };
}

/**
 * Lower-triangular Cholesky factor L of a row-major SPD n×n matrix, L Lᵀ = A_σ.
 * The f64 REFERENCE — gate CR runs the whole [DESIGN §4.1] reformulation on this factor
 * so a CR red localises to the algebra, never to a kernel; slice 2a-2's f32 GPU result
 * is promoted into the same shape. Left-looking (the GPU kernel is right-looking,
 * [DESIGN §4.3]): the op orders differ deliberately and CR's bar is κ-relative, so they
 * must NOT be "unified".
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1, §4.3, §5 (CR)
 */
export function choleskyF64(Asigma: Float64Array, n: number): Float64Array {
    const L = new Float64Array(n * n);
    for (let j = 0; j < n; j++) {
        let d = Asigma[j * n + j];
        for (let k = 0; k < j; k++) d -= L[j * n + k] * L[j * n + k];
        // Same guard the GPU `finalize` kernel applies ([DESIGN §4.3]): `!(d > 0)` also
        // rejects NaN, which `d <= 0` would let through.
        if (!(d > 0) || !Number.isFinite(d))
            throw new Error(`choleskyF64: non-positive pivot ${d} at column ${j}`);
        const r = Math.sqrt(d);
        L[j * n + j] = r;
        for (let i = j + 1; i < n; i++) {
            let s = Asigma[i * n + j];
            for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
            L[i * n + j] = s / r;
        }
    }
    return L;
}

/** Forward/back substitution against L Lᵀ, one n-vector, in place into `out`. */
function solveWithL(L: Float64Array, n: number, rhs: ArrayLike<number>, out: Float64Array): void {
    for (let i = 0; i < n; i++) {
        let s = rhs[i];
        for (let k = 0; k < i; k++) s -= L[i * n + k] * out[k];
        out[i] = s / L[i * n + i];
    }
    for (let i = n - 1; i >= 0; i--) {
        let s = out[i];
        for (let k = i + 1; k < n; k++) s -= L[k * n + i] * out[k];
        out[i] = s / L[i * n + i];
    }
}

/**
 * Â⁻¹ applied to a 3n vector — three INDEPENDENT n-backsolves, one per coordinate block.
 * COORDINATE-MAJOR, and that is load-bearing: `Ā = I₃ ⊗ A`, so block `b` is coordinate
 * `b` over ALL vertices and occupies rows `[b·n, (b+1)·n)`. This is the layout the core
 * writes (`linsolve.ts:823-827`, `off = b * n` / `dst = (off + i) * size + off`), the
 * layout `flatten` produces (`layout.ts:23-32`), and the layout the oracle uses
 * (`oracle/tpe_stage1_oracle.py:245-248`, `np.block([[A,0,0],[0,A,0],[0,0,A]])`).
 * The interleaved vertex-major spelling `A ⊗ I₃` — indexing `v[3*i + b]` instead of
 * `v[b*n + i]` — is the OPPOSITE convention and would make every test in this file fail;
 * an earlier draft of [DESIGN §4.1] had it backwards, which is why it is anchored here.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1
 */
function applyAhatInv(L: Float64Array, n: number, v: ArrayLike<number>): Float64Array {
    const out = new Float64Array(3 * n);
    const buf = new Float64Array(n);
    const rhs = new Float64Array(n);
    for (let b = 0; b < 3; b++) {
        for (let i = 0; i < n; i++) rhs[i] = v[b * n + i];
        solveWithL(L, n, rhs, buf);
        for (let i = 0; i < n; i++) out[b * n + i] = buf[i];
    }
    return out;
}

/** IR diagnostics for one solve — the convergence curve CR and (in 2a-2) T4 record. */
export interface IrTrace {
    refinements: number;
    relResiduals: number[];
}

/**
 * The Phase 2a external solve: [DESIGN §4.1] steps 2–4 on a factor produced elsewhere.
 * `A` is the UNSHIFTED f64 Sobolev matrix (step 4's residuals are measured against the
 * f64 K, which is what makes T4 hold as [P §3] wrote it); `L` is the lower-triangular
 * Cholesky factor of A_σ (f64-promoted from f32 on the GPU path); `sigma` is the shift
 * the factor was built with.
 *
 * Per-step cost (k = C.length): 3k+1 setup n-backsolves, then 3 per solve plus O(nk)
 * ([DESIGN §4.1] step 4). With P pins k = 4+3P and the setup is 13+9P ([DESIGN §7]).
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1, §4.2
 */
export function makePreparedSaddle(
    A: Float64Array,
    L: Float64Array,
    sigma: number,
    n: number,
    onTrace?: (t: IrTrace) => void,
): PreparedSaddle {
    return {
        A,
        factor(C: number[][]): SaddleFactorization {
            const m = 3 * n;
            const k = C.length;
            const invSqrtN = 1 / Math.sqrt(n);

            // Setup, ONCE per step (the factor is frozen, [P §2.5]).
            // Z = Â⁻¹Cᵀ — 3k n-backsolves.
            const Z: Float64Array[] = C.map((row) => applyAhatInv(L, n, row));
            // S = C Z, k×k, SPD. κ(S) = 5.06…5.28 across N=120…960 ([DESIGN §4.1]), so a
            // plain LU is ample; luSolve is the core's own, already gated.
            const S: number[][] = Array.from({ length: k }, (_, r) =>
                Array.from({ length: k }, (_, c) => {
                    let s = 0;
                    for (let i = 0; i < m; i++) s += C[r][i] * Z[c][i];
                    return s;
                }),
            );
            // W = K̂⁻¹U, three columns. Each column of P = I₃⊗(𝟙/√n) is supported on ONE
            // contiguous n-block, so Â⁻¹p_b needs exactly ONE n-backsolve of 𝟙/√n —
            // computed here and reused for all three coordinates ([DESIGN §4.1] step 3).
            const onesRhs = new Float64Array(n).fill(invSqrtN);
            const u = new Float64Array(n);
            solveWithL(L, n, onesRhs, u);
            const W: Float64Array[] = [];
            for (let b = 0; b < 3; b++) {
                const y = new Float64Array(m);
                for (let i = 0; i < n; i++) y[b * n + i] = u[i];
                // K̂⁻¹[p_b; 0] through the same Schur machinery: λ = S⁻¹(C y), x = y − Zλ.
                const Cy = C.map((row) => {
                    let s = 0;
                    for (let i = 0; i < m; i++) s += row[i] * y[i];
                    return s;
                });
                const lam = luSolve(S, Cy);
                const w = new Float64Array(m + k);
                for (let i = 0; i < m; i++) {
                    let s = y[i];
                    for (let r = 0; r < k; r++) s -= Z[r][i] * lam[r];
                    w[i] = s;
                }
                for (let r = 0; r < k; r++) w[m + r] = lam[r];
                W.push(w);
            }
            // M = σ⁻¹I₃ − UᵀW, 3×3. HEAVILY CANCELLING: measured diag(M) is 9.2e4×
            // smaller than σ⁻¹ at N=960 ([DESIGN §7]), so its formation destroys ~5
            // decimal digits and the loss grows ∝ κ. Form and invert it in f64, NEVER
            // f32 and NEVER on the GPU — a hard constraint that also binds slice 2c
            // ([DESIGN §4.7]). CR records diag(M) and κ(M) so this margin is a number.
            const M: number[][] = Array.from({ length: 3 }, (_, a) =>
                Array.from({ length: 3 }, (_, b) => {
                    let s = 0;
                    for (let i = 0; i < n; i++) s += invSqrtN * W[b][a * n + i];
                    return (a === b ? 1 / sigma : 0) - s;
                }),
            );

            /** ẑ = K̂⁻¹r via Schur, then Woodbury back to K⁻¹r. [DESIGN §4.1] steps 2–3. */
            const applyKinv = (rhs: ArrayLike<number>): Float64Array => {
                const y = applyAhatInv(L, n, rhs);
                const rhsBottom = new Array<number>(k);
                for (let r = 0; r < k; r++) rhsBottom[r] = rhs[m + r];
                const Cy = C.map((row, r) => {
                    let s = 0;
                    for (let i = 0; i < m; i++) s += row[i] * y[i];
                    return s - rhsBottom[r];
                });
                const lam = luSolve(S, Cy);
                const xh = new Float64Array(m);
                for (let i = 0; i < m; i++) {
                    let s = y[i];
                    for (let r = 0; r < k; r++) s -= Z[r][i] * lam[r];
                    xh[i] = s;
                }
                // Woodbury: z = ẑ + W M⁻¹ (Pᵀx̂). UᵀK̂⁻¹r = Uᵀẑ = Pᵀx̂ because U = [P; 0].
                const Ptx = [0, 0, 0];
                for (let a = 0; a < 3; a++) {
                    let s = 0;
                    for (let i = 0; i < n; i++) s += invSqrtN * xh[a * n + i];
                    Ptx[a] = s;
                }
                const mu = luSolve(M, Ptx);
                const z = new Float64Array(m + k);
                for (let i = 0; i < m; i++) {
                    let s = xh[i];
                    for (let a = 0; a < 3; a++) s += W[a][i] * mu[a];
                    z[i] = s;
                }
                for (let r = 0; r < k; r++) {
                    let s = lam[r];
                    for (let a = 0; a < 3; a++) s += W[a][m + r] * mu[a];
                    z[m + r] = s;
                }
                return z;
            };

            /** r = rhs − K z against the f64 A and C — K is never materialized. */
            const residualOf = (rhs: ArrayLike<number>, z: Float64Array): Float64Array => {
                const r = new Float64Array(m + k);
                for (let b = 0; b < 3; b++)
                    for (let i = 0; i < n; i++) {
                        let s = 0;
                        for (let j = 0; j < n; j++) s += A[i * n + j] * z[b * n + j];
                        r[b * n + i] = s;
                    }
                for (let q = 0; q < k; q++) {
                    const lamq = z[m + q];
                    for (let i = 0; i < m; i++) r[i] += C[q][i] * lamq;
                }
                for (let q = 0; q < k; q++) {
                    let s = 0;
                    for (let i = 0; i < m; i++) s += C[q][i] * z[i];
                    r[m + q] = s;
                }
                for (let i = 0; i < m + k; i++) r[i] = rhs[i] - r[i];
                return r;
            };

            return {
                kind: 'external',
                solve(rhs: number[]): number[] {
                    let rhsNorm = 0;
                    for (let i = 0; i < rhs.length; i++) rhsNorm += rhs[i] * rhs[i];
                    rhsNorm = Math.sqrt(rhsNorm);
                    const z = applyKinv(rhs);
                    const relResiduals: number[] = [];
                    // A zero rhs has the exact solution z = 0 and no relative residual to
                    // measure; returning here avoids 0/0 becoming a spurious
                    // ExternalSolveError. (CR's projection fixtures assert a NON-zero rhs
                    // so this branch can never make that gate vacuous — D6.)
                    // @see docs/superpowers/plans/2026-09-04-webgpu-solver-phase2a-1.md
                    //      (## Decisions, D6)
                    if (rhsNorm === 0) {
                        onTrace?.({ refinements: 0, relResiduals: [0] });
                        return Array.from(z);
                    }
                    for (let it = 0; ; it++) {
                        const r = residualOf(rhs, z);
                        let rn = 0;
                        for (let i = 0; i < r.length; i++) rn += r[i] * r[i];
                        // The PURE relative residual — divided by ‖r‖₂, NEVER by
                        // max(1, ‖r‖₂). The core's own stat uses the latter
                        // (`structuredSaddleResidual`, `linsolve.ts:860`) and is therefore
                        // ABSOLUTE for the projection rhs `[0; −Φ]`, whose ‖r‖₂ is always
                        // ≤ 1; delegating to it would certify a solve ~1e4× worse than
                        // this criterion claims, in the PERMISSIVE direction.
                        // @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1 step 4, §4.2
                        const rel = Math.sqrt(rn) / rhsNorm;
                        relResiduals.push(rel);
                        if (rel <= SADDLE_IR_TOL) {
                            onTrace?.({ refinements: it, relResiduals });
                            return Array.from(z);
                        }
                        // `it` counts refinements ALREADY applied to z₀, so the budget is
                        // z₀ + SADDLE_IR_MAX_REFINEMENTS corrections — the same counting
                        // `ir_iters` pins (`bench/gpu/k1-calibration.py:256`, whose
                        // `_ir_budget_selfcheck` at `:420` asserts it). Off-by-one here
                        // understates every derived bar.
                        if (it >= SADDLE_IR_MAX_REFINEMENTS) {
                            onTrace?.({ refinements: it, relResiduals });
                            // The ONLY failure detectable during the step. Returning an
                            // unconverged z is a contract violation, not a degraded mode:
                            // nothing in src/core/ thresholds `residual`, so the line
                            // search would reject the wrong g̃ as `armijo_failed` and the
                            // run would auto-pause with usedGpuSolve still true.
                            // @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1 step 4, §4.5
                            throw new ExternalSolveError(
                                `saddle IR: relative residual ${rel.toExponential(3)} after ` +
                                    `${SADDLE_IR_MAX_REFINEMENTS} refinements (bar ${SADDLE_IR_TOL})`,
                            );
                        }
                        const d = applyKinv(r);
                        for (let i = 0; i < z.length; i++) z[i] += d[i];
                    }
                },
            };
        },
    };
}
