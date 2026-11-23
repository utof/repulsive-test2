/**
 * Spec §11.2 row 8b: `makeHierarchicalSaddle` — the PRODUCER, against a REAL compressed operator
 * and the real `schur.ts` solver, end to end through `sobolevStepSet`.
 *
 * This file is deliberately NOT an extension of `hierarchicalSeam.test.ts`. Those stubs are
 * dense-EXACT by design (`denseOp` is a dense `Ā·u` matvec in `structuredSaddleResidual`'s own
 * accumulation order, and `exactSolver` refactors the dense `K` on every call), which is what lets
 * them assert plumbing with `.toBe`. A real `A_bct` at `DEFAULT_THETA` is compressed, so none of
 * those bit-level assertions can survive here; the two files measure different things and both are
 * needed.
 *
 * WHAT `cos∠_Ā ≥ 0.9` DOES AND DOES NOT MEASURE — read this before trusting a green run.
 * The row-8b gate is `cos∠_Ā(g̃_hier, g̃_dense) ≥ 0.9`, in the DENSE `Ā` seminorm. MEASURED on this
 * fixture (n = 960, trefoil, barycentre set): **0.999840**, i.e. 11× the margin. That number is
 * about the OPERATOR, not about the linear solver (spec A35 item 8): the θ = 0.25 compression moves
 * `g̃` by 2.2e-3 while the Schur solver is ~5e-12 from the truth of its own operator. Worse, and
 * measured here rather than assumed:
 *  1. **No `theta` can trip the 0.9 bar.** The ladder saturates — `cos∠_Ā` = 0.999963 / 0.999307 /
 *     0.998749 at θ = 0.25 / 0.5 / 1, and θ = 1, 2, 4, 8, 16, 32 are all the SAME tree
 *     (`nFar = 274, nNear = 126` at n = 240) because the adjacency certificate floors the near
 *     field. The worst `cos∠_Ā` any θ can produce on a gradient rhs is 0.9987.
 *  2. **The bar is blind to a saddle built for the WRONG GEOMETRY.** A saddle built from
 *     `2 × vertices` at the same `n` passes `op.n` and reads `cos∠_Ā = 0.999839` — a hair from the
 *     clean 0.999840 — while its `g̃` is **4.04 relative** away from the dense one. Direction in a
 *     seminorm is simply not sensitive to that.
 *  3. What DOES trip it: `alpha = 2` against the step's 3 → **0.790**, `beta = 8` → **0.878**,
 *     `maxit = 1` → **0.778** (all at n = 960). Those are asserted below, so the gate is known to
 *     be trippable rather than assumed to be (CLAUDE.md: a gate that cannot trip reads exactly like
 *     one that passed).
 * Every `cos∠_Ā` assertion below therefore carries a MOD-TRANSLATION relative-error companion,
 * which is the sharp instrument: clean 2.2e-3, and it trips on every defect in the list including
 * the wrong geometry.
 *
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.1, §7.5, §11.2 row 8b, A33, A35
 * @see .superpowers/sdd/2026-10-09-phase2h-bct-gmres-design/task-8b-report.md (every figure above, with its command)
 * @issue utof/repulsive-test2#62
 * @issue utof/repulsive-test2#69
 */
import { describe, expect, setDefaultTimeout, test } from 'bun:test';
import { trefoil } from '../../src/core/fixtures';
import { DEFAULTS, sobolevStepSet } from '../../src/core/optimizer';
import {
    barycenterBlock,
    type ConstraintSet,
    edgeLengthsBlock,
    evaluateConstraintSet,
    pointBlock,
    totalLength,
    totalLengthBlock,
} from '../../src/core/sobolev/constraintSet';
import { barycenterTarget } from '../../src/core/sobolev/constraints';
import { applyAsigma, estimateNorm2 } from '../../src/core/sobolev/hierarchical/gmres';
import {
    DEFAULT_LEAF_SIZE,
    HierarchicalSaddleUnsupportedError,
    makeHierarchicalSaddle,
} from '../../src/core/sobolev/hierarchical/hierarchicalSaddle';
import type { NullspaceSolver } from '../../src/core/sobolev/hierarchical/nullspace';
import {
    SCHUR_MAX_K,
    SchurConvergenceError,
    SchurRouteError,
    type SchurSolver,
} from '../../src/core/sobolev/hierarchical/schur';
import { assembleAFlat } from '../../src/core/sobolev/innerProduct';
import { flatten } from '../../src/core/sobolev/layout';
import { l2CurveNorm } from '../../src/core/sobolev/lineSearch';
import {
    ExternalSolveError,
    type HierarchicalPreparedSaddle,
    SolveNotCertifiedError,
} from '../../src/core/sobolev/linsolve';
import { calculateDisjointPairs } from '../../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../../src/core/testConfigs';

setDefaultTimeout(900_000);

/** `dispatch.ts` forwards no α/β/ε, so the step falls back to these; the saddle MUST match (A33). */
const { alpha: ALPHA, beta: BETA, epsilon: EPS } = DEFAULTS;
/** Row 8b's headline bar. */
const COS_BAR = 0.9;
/**
 * Companion bar on the MOD-TRANSLATION relative error of `g̃`. CHOSEN as 1e-2, from the measured
 * clean value 2.23e-3 at n = 960 / 2.86e-3 at n = 240 (4.5× margin) and the measured θ = 0.5 value
 * 2.86e-2 — i.e. loose enough for the default θ, tight enough that every defect in the module
 * TSDoc's list (0.25 … 4.04) trips it. NOT an accuracy claim about the solver: the solver is ~5e-12
 * from the truth of its own operator (9b). This bar measures θ's compression, like `cos∠_Ā` does,
 * but unlike `cos∠_Ā` it is also sensitive to magnitude and geometry.
 */
const MOD_TRANS_BAR = 1e-2;

const log = (name: string, v: unknown) => console.log(`[8b] ${name} = ${v}`);

const l2 = (u: ArrayLike<number>): number => {
    let s = 0;
    for (let i = 0; i < u.length; i++) s += u[i] * u[i];
    return Math.sqrt(s);
};
const relErr = (u: ArrayLike<number>, ref: ArrayLike<number>): number => {
    let s = 0;
    for (let i = 0; i < u.length; i++) s += (u[i] - ref[i]) ** 2;
    return Math.sqrt(s) / l2(ref);
};
/** Per-coordinate-block means — the translation `schur.ts` gauges and the dense path does not. */
function blockMeans(u: ArrayLike<number>, n: number): [number, number, number] {
    const m: number[] = [];
    for (let c = 0; c < 3; c++) {
        let s = 0;
        for (let i = 0; i < n; i++) s += u[c * n + i];
        m.push(s / n);
    }
    return m as [number, number, number];
}
/** Removes each coordinate block's mean: the `Nᵀ` gauge quotient (spec A35 item 3). */
function demean(u: ArrayLike<number>, n: number): number[] {
    const m = blockMeans(u, n);
    const out: number[] = [];
    for (let c = 0; c < 3; c++) for (let i = 0; i < n; i++) out.push(u[c * n + i] - m[c]);
    return out;
}

/**
 * `⟨u, v⟩_Ā` over the flat n×n DENSE `Ā` (`assembleAFlat`), applied to each of the three
 * coordinate blocks — Ā is block-diagonal with three identical scalar blocks (`linsolve.ts`).
 *
 * The metric is the UNSHIFTED `Ā`, deliberately and not `A_σ`: `Ā·1 = 0`, so `⟨·,·⟩_Ā` annihilates
 * a translation, which is what makes `cos∠_Ā` well-defined on the k = 0 cell where the dense saddle
 * is translation-singular and the two arms return different translations (spec A35 item 3). With
 * `A_σ` the same comparison would read a gauge difference as a direction error.
 */
function makeIpA(Aflat: Float64Array, n: number) {
    return (u: ArrayLike<number>, v: ArrayLike<number>): number => {
        let s = 0;
        for (let c = 0; c < 3; c++) {
            for (let i = 0; i < n; i++) {
                let row = 0;
                for (let j = 0; j < n; j++) row += Aflat[i * n + j] * v[c * n + j];
                s += u[c * n + i] * row;
            }
        }
        return s;
    };
}

/** The reusable n=960 / n=240 fixture: the curve, its disjoint pairs, and the dense `Ā` metric. */
function fixture(n: number) {
    const { vertices, edges } = trefoil(n);
    const disjointPairs = calculateDisjointPairs(edges);
    const Aflat = assembleAFlat(vertices, edges, disjointPairs, ALPHA, BETA, EPS);
    const ipA = makeIpA(Aflat, n);
    /**
     * `cos∠_Ā(u, v)`. Both A-seminorms are asserted FINITE and `> 0` first: a vector in `null(Ā)`
     * (a pure translation) would give 0/0 = NaN, and `NaN >= 0.9` is false, so the gate would read
     * RED for the wrong reason — name the impossible value instead of letting it look like a
     * failure (CLAUDE.md).
     */
    const cosA = (u: ArrayLike<number>, v: ArrayLike<number>): number => {
        const uu = ipA(u, u);
        const vv = ipA(v, v);
        expect(Number.isFinite(uu) && uu > 0).toBe(true);
        expect(Number.isFinite(vv) && vv > 0).toBe(true);
        return ipA(u, v) / Math.sqrt(uu * vv);
    };
    return { n, vertices, edges, disjointPairs, Aflat, ipA, cosA };
}

/** Offset so `Φ ≠ 0` — the Z₃-symmetric trefoil at its OWN barycentre is a degenerate cell (A35 item 5). */
function offsetBaryTarget(vertices: Vec3[], edges: Edge[]): Vec3 {
    const t = barycenterTarget(vertices, edges);
    return [t[0] + 0.05, t[1] + 0.05, t[2] + 0.05];
}

/**
 * Which of §8.2's two routes `prepare(C)` picked. `report.projector` is the discriminant —
 * `NullspacePrepareReport` is the only member of the union that carries one — and this is written as
 * a TYPE PREDICATE on purpose: used as `if (!isNullspaceRoute(s)) throw`, the route claim and the
 * narrowing are the same statement, so no `as` cast sits between the assertion and the reads that
 * depend on it (a wrong route then fails the test instead of silently mistyping the body).
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §8.2, §11.2 row 9b
 */
function isNullspaceRoute(s: SchurSolver | NullspaceSolver): s is NullspaceSolver {
    return 'projector' in s.report;
}

/** `projectionMode: 'frozen'` is MANDATORY on this arm (cross-task contract from task 8, `a8569d8`). */
const STEP = { mode: 'analytical' as const, projectionMode: 'frozen' as const, collectField: true };

// ---------------------------------------------------------------------------
// A. End-to-end at n = 960: row 8b's headline cell, plus the three tripwires that
//    show the bar can fail and the one that shows what it cannot see.
// ---------------------------------------------------------------------------

describe('row 8b end-to-end: sobolevStepSet on the hierarchical route, n = 960, trefoil', () => {
    const f = fixture(960);
    const set: ConstraintSet = [barycenterBlock(offsetBaryTarget(f.vertices, f.edges))];
    const dense = sobolevStepSet(f.vertices, f.edges, f.disjointPairs, set, STEP);
    const gd = flatten(dense.descentField as Vec3[]);
    const gdm = demean(gd, f.n);

    const runHier = (opts?: Parameters<typeof makeHierarchicalSaddle>[3], verts?: Vec3[]) => {
        const saddle = makeHierarchicalSaddle(verts ?? f.vertices, f.edges, f.disjointPairs, opts);
        const out = sobolevStepSet(f.vertices, f.edges, f.disjointPairs, set, { ...STEP, saddle });
        const g = flatten(out.descentField as Vec3[]);
        return {
            saddle,
            out,
            cos: f.cosA(g, gd),
            modTrans: relErr(demean(g, f.n), gdm),
        };
    };

    test('cos∠_Ā >= 0.9 against the dense f64 reference step, and the step agrees mod translation', () => {
        const h = runHier();
        log('n=960 bary build', JSON.stringify(h.saddle.build));
        log('n=960 bary bct.stats', JSON.stringify(h.saddle.parts.bct.stats));
        log('n=960 bary cos∠_Ā', h.cos);
        log('n=960 bary modTrans relErr', h.modTrans);
        log('n=960 bary residual dense/hier', `${dense.stats.residual} / ${h.out.stats.residual}`);
        log('n=960 bary tau dense/hier', `${dense.stats.tau} / ${h.out.stats.tau}`);
        log(
            'n=960 bary projIters dense/hier',
            `${dense.stats.projectionIterations} / ${h.out.stats.projectionIterations}`,
        );
        log('n=960 bary energy dense/hier', `${dense.energy} / ${h.out.energy}`);
        // MEASURED 0.999840. The gate, and it is about θ, not about the solver (module TSDoc).
        expect(h.cos).toBeGreaterThanOrEqual(COS_BAR);
        // MEASURED 2.226e-3. The sharp companion.
        expect(h.modTrans).toBeLessThan(MOD_TRANS_BAR);
        // The step itself must RUN, not merely produce a direction: both arms accept at the same τ
        // and take the same number of projection iterates, so the line search saw the same shape.
        expect(h.out.accepted).toBe(true);
        expect(dense.accepted).toBe(true);
        expect(h.out.stats.tau).toBe(dense.stats.tau);
        expect(h.out.stats.projectionIterations).toBe(dense.stats.projectionIterations);
        // `x` must be a `number[]`: `unflatten` is typed `(flat: number[])` (`layout.ts:39`) and
        // `gmres` returns a `Float64Array`. A Float64Array would survive `unflatten` by duck-typing
        // and only surface later, so pin the type the seam contract names.
        const { C } = evaluateConstraintSet(set, f.vertices, f.edges);
        const r = h.saddle.prepare(C).solve(new Array<number>(3 * f.n).fill(0), [0, 0, 0]);
        expect(Array.isArray(r.x)).toBe(true);
        expect(Array.isArray(r.lambda)).toBe(true);
        expect(typeof r.residual).toBe('number');
    });

    test('the 0.9 bar CAN fail: alpha/beta mismatched with the step, and maxit = 1', () => {
        // The exact trap A33's third fact names: `dispatch.ts` forwards no α/β/ε, so a saddle built
        // with different ones is an operator that is not the Ā of the step consuming it, and
        // nothing downstream can see the disagreement. These are reachable through the shipped
        // options with no knock-out flag, which is why they are the right falsifiers.
        const bad = runHier({ alpha: 2 });
        log('n=960 alpha=2 cos∠_Ā', bad.cos);
        log('n=960 alpha=2 modTrans', bad.modTrans);
        expect(bad.cos).toBeLessThan(COS_BAR); // MEASURED 0.790
        expect(bad.modTrans).toBeGreaterThan(MOD_TRANS_BAR); // MEASURED 9.23e-1

        const bad2 = runHier({ beta: 8 });
        log('n=960 beta=8 cos∠_Ā', bad2.cos);
        expect(bad2.cos).toBeLessThan(COS_BAR); // MEASURED 0.878
        expect(bad2.modTrans).toBeGreaterThan(MOD_TRANS_BAR); // MEASURED 7.78e-1

        // A truncated GMRES. `etaAccept: Infinity` is REQUIRED to observe it: at the shipped
        // certificate this very configuration throws `SchurConvergenceError` instead (asserted in
        // the 'typed errors' block), which is the production behaviour.
        const bad3 = runHier({ schur: { maxit: 1, etaAccept: Number.POSITIVE_INFINITY } });
        log('n=960 maxit=1 cos∠_Ā', bad3.cos);
        log('n=960 maxit=1 residual', bad3.out.stats.residual);
        expect(bad3.cos).toBeLessThan(COS_BAR); // MEASURED 0.778
        expect(bad3.out.stats.residual).toBeGreaterThan(1); // MEASURED 1.82 — the certificate's job
    });

    test('the 0.9 bar is BLIND to a saddle built for the wrong geometry; modTrans is not', () => {
        // `op.n` is the only shape the seam checks (`optimizer.ts:314`), so a saddle built from a
        // DIFFERENT curve of the same n passes every guard. This is the Phase 1 "same vertices"
        // rule (`DensePreparedSaddle` TSDoc, `src/gpu/driver.ts:164`) with no enforcement, and it
        // is the reason `cos∠_Ā` alone is not an adequate gate for row 8b.
        const scaled = f.vertices.map((p) => [2 * p[0], 2 * p[1], 2 * p[2]] as Vec3);
        const h = runHier(undefined, scaled);
        log('n=960 wrong-geometry (2x) cos∠_Ā', h.cos);
        log('n=960 wrong-geometry (2x) modTrans', h.modTrans);
        // The gate PASSES on a wrong operator — that is the finding, asserted so it cannot rot.
        expect(h.cos).toBeGreaterThanOrEqual(COS_BAR); // MEASURED 0.999839, clean is 0.999840
        expect(Math.abs(h.cos - 0.9998)).toBeLessThan(1e-3);
        // And the companion catches it.
        expect(h.modTrans).toBeGreaterThan(1); // MEASURED 4.04
    });
});

// ---------------------------------------------------------------------------
// B. k = 0, the one constraint cell row 8b owns (A33 item 1 / A35 item 3).
// ---------------------------------------------------------------------------

describe('k = 0 (empty constraint set) against the dense reference, MOD TRANSLATION', () => {
    const f = fixture(960);
    const dense = sobolevStepSet(f.vertices, f.edges, f.disjointPairs, [], STEP);
    const saddle = makeHierarchicalSaddle(f.vertices, f.edges, f.disjointPairs);
    const hier = sobolevStepSet(f.vertices, f.edges, f.disjointPairs, [], { ...STEP, saddle });
    const gd = flatten(dense.descentField as Vec3[]);
    const gh = flatten(hier.descentField as Vec3[]);

    test('the two arms agree to 2.2e-3 mod translation, and NOT AT ALL literally', () => {
        const literal = relErr(gh, gd);
        const modTrans = relErr(demean(gh, f.n), demean(gd, f.n));
        log('n=960 k0 cos∠_Ā', f.cosA(gh, gd));
        log('n=960 k0 LITERAL relErr', literal);
        log('n=960 k0 MOD-TRANSLATION relErr', modTrans);
        log('n=960 k0 blockMeans(dense)', JSON.stringify(blockMeans(gd, f.n)));
        log('n=960 k0 blockMeans(hier)', JSON.stringify(blockMeans(gh, f.n)));
        // The bar, mod translation. MEASURED 2.226e-3 — the SAME number as the k = 3 barycentre
        // cell's 2.226e-3, because what is left after the gauge is exactly θ's compression.
        expect(modTrans).toBeLessThan(MOD_TRANS_BAR);
        // NON-VACUITY of the demeaning, which is the whole point of A35 item 3: the literal
        // comparison fails by 370×. If a future change made the dense k = 0 path return the A_σ
        // gauge too, THIS assertion goes red and the demeaning becomes unnecessary — that is the
        // intended signal, not a regression.
        expect(literal).toBeGreaterThan(0.5); // MEASURED 8.203e-1
        // Why they differ: `schur.ts` returns the A_σ gauge (`Nᵀx = Nᵀb/σ`, ≈ 0 for a mean-free
        // energy gradient) while dense LDLᵀ on a translation-singular saddle returns an ARBITRARY
        // translation (A35 item 3 measured LDLᵀ vs LU at 6.0e-2 vs 8.0e-2 for the same system).
        for (const m of blockMeans(gh, f.n)) expect(Math.abs(m)).toBeLessThan(1e-14);
        expect(l2(blockMeans(gd, f.n))).toBeGreaterThan(1e-3); // MEASURED 0.21
        // And `cos∠_Ā` cannot see any of it: Ā annihilates the translation, so this reads the same
        // 0.99984 as the k = 3 cell.
        expect(f.cosA(gh, gd)).toBeGreaterThanOrEqual(COS_BAR);
    });

    test('the gauge is NOT cosmetic: it changes the step the line search takes', () => {
        // FINDING, measured here and not in the brief. `lineSearchStepSet:499-517` normalises the
        // direction by `l2CurveNorm(gTilde)` before stepping, so a translation component in `g̃`
        // INFLATES that norm and SHRINKS the realised step. At k = 0 the dense `g̃` is 82 %
        // translation, so its curve norm is 1.77× the hierarchical one and the dense arm takes a
        // 1.77× shorter step along an identical direction — landing at an energy 0.615 (4 %) away.
        // `cos∠_Ā` is scale-invariant in each argument by construction and cannot see it, and
        // neither can the mod-translation bar (both arms' MEAN-FREE parts agree to 2.2e-3).
        // Consequence for the spec: on a translation-singular set the dense path is not a reference
        // for the STEP either, only for `x` mod translation. Do not read this cell as "the two arms
        // produce the same step".
        const nd = l2CurveNorm(dense.descentField as Vec3[], f.vertices, f.edges);
        const nh = l2CurveNorm(hier.descentField as Vec3[], f.vertices, f.edges);
        log('n=960 k0 l2CurveNorm dense/hier', `${nd} / ${nh}`);
        log('n=960 k0 curve-norm ratio', nd / nh);
        log('n=960 k0 energy dense/hier', `${dense.energy} / ${hier.energy}`);
        expect(nd / nh).toBeGreaterThan(1.5); // MEASURED 1.7686
        expect(Math.abs(hier.energy - dense.energy)).toBeGreaterThan(1e-2); // MEASURED 0.615
        // Both still ACCEPT, which is why nothing downstream flags the divergence.
        expect(dense.accepted).toBe(true);
        expect(hier.accepted).toBe(true);
        expect(dense.stats.projectionIterations).toBe(0);
        expect(hier.stats.projectionIterations).toBe(0);
    });
});

// ---------------------------------------------------------------------------
// C. The exactness differential — the cell that CAN separate the solver from θ.
// ---------------------------------------------------------------------------

test('theta = 0 makes the operator exact (nFar = 0) and the whole stack reproduces the dense step to 2.6e-12', () => {
    // Why this cell exists: `cos∠_Ā ≥ 0.9` and the mod-translation bar both measure θ's
    // compression (spec A35 item 8), so neither can tell a correct composition from one with a
    // plumbing defect of up to ~1e-3. At θ = 0 every block is inadmissible (spec §7.1's exact
    // path), so `A_bct` IS `Ā` up to summation order and the SAME end-to-end path must reproduce
    // the dense step to the conditioning floor. It does: 2.611e-12 at n = 240 and 2.533e-10 at
    // n = 960 — the latter being exactly A35 item 7's floor (two f64 evaluations of the same
    // operator have true solutions 1.8e-10..2.2e-10 apart at n = 960), reproduced here from an
    // independent direction. A `leafSize` large enough to give `nFar = 0` (64 at n = 240) returns
    // the identical 2.611e-12, which cross-checks that it is the compression and not θ's code path.
    const f = fixture(240);
    const set: ConstraintSet = [barycenterBlock(offsetBaryTarget(f.vertices, f.edges))];
    const dense = sobolevStepSet(f.vertices, f.edges, f.disjointPairs, set, STEP);
    const gd = flatten(dense.descentField as Vec3[]);

    const exact = makeHierarchicalSaddle(f.vertices, f.edges, f.disjointPairs, {
        bct: { theta: 0 },
    });
    expect(exact.parts.bct.stats.nFar).toBe(0);
    const hier = sobolevStepSet(f.vertices, f.edges, f.disjointPairs, set, {
        ...STEP,
        saddle: exact,
    });
    const gh = flatten(hier.descentField as Vec3[]);
    const modTrans = relErr(demean(gh, f.n), demean(gd, f.n));
    log(
        'n=240 theta=0 nFar/nNear',
        `${exact.parts.bct.stats.nFar} / ${exact.parts.bct.stats.nNear}`,
    );
    log('n=240 theta=0 cos∠_Ā', f.cosA(gh, gd));
    log('n=240 theta=0 modTrans relErr', modTrans);
    // MEASURED 2.611e-12. The bar is 1e-10: three decades of room over the measurement and still
    // 1e8 tighter than the compressed cell's 2.9e-3, so it discriminates what that one cannot.
    expect(modTrans).toBeLessThan(1e-10);
    expect(f.cosA(gh, gd)).toBeGreaterThan(1 - 1e-12);

    // The same exactness reached a different way: no far blocks because the tree has 4 leaves.
    const big = makeHierarchicalSaddle(f.vertices, f.edges, f.disjointPairs, { leafSize: 64 });
    expect(big.parts.bct.stats.nFar).toBe(0);
    const hier2 = sobolevStepSet(f.vertices, f.edges, f.disjointPairs, set, {
        ...STEP,
        saddle: big,
    });
    const gh2 = flatten(hier2.descentField as Vec3[]);
    log('n=240 leafSize=64 modTrans relErr', relErr(demean(gh2, f.n), demean(gd, f.n)));
    expect(relErr(demean(gh2, f.n), demean(gd, f.n))).toBeLessThan(1e-10);

    // NON-VACUITY: the shipped DEFAULT_THETA on the SAME fixture is 1e9 times worse, so the 1e-10
    // bar is not something any configuration passes.
    const compressed = makeHierarchicalSaddle(f.vertices, f.edges, f.disjointPairs);
    expect(compressed.parts.bct.stats.nFar).toBeGreaterThan(0);
    const hier3 = sobolevStepSet(f.vertices, f.edges, f.disjointPairs, set, {
        ...STEP,
        saddle: compressed,
    });
    const gh3 = flatten(hier3.descentField as Vec3[]);
    const compressedErr = relErr(demean(gh3, f.n), demean(gd, f.n));
    log('n=240 DEFAULT_THETA modTrans relErr', compressedErr);
    expect(compressedErr).toBeGreaterThan(1e-4); // MEASURED 2.859e-3
    expect(compressedErr).toBeLessThan(MOD_TRANS_BAR);
});

// ---------------------------------------------------------------------------
// D. The two pre-arm guards (`optimizer.ts:308-327`).
// ---------------------------------------------------------------------------

describe('the two pre-arm guards fire on a REAL hierarchical saddle', () => {
    const n = 48;
    const { vertices, edges } = trefoil(n);
    const disjointPairs = calculateDisjointPairs(edges);
    const set: ConstraintSet = [barycenterBlock(offsetBaryTarget(vertices, edges))];
    const saddle = makeHierarchicalSaddle(vertices, edges, disjointPairs);

    test('the clean configuration passes both guards (so the two below are not passing vacuously)', () => {
        const r = sobolevStepSet(vertices, edges, disjointPairs, set, { ...STEP, saddle });
        expect(r.accepted).toBe(true);
        expect(r.stats.reason).toBeUndefined();
    });

    test("op.n mismatch throws 'sobolevStepSet: saddle.op.n mismatch'", () => {
        // The value that trips it, and it EXISTS: a saddle whose operator was built for n-1
        // vertices. `op.n` comes from `H.n` (`operator.ts`), the vertex count — the same quantity
        // the dense arm checks `A.length` against.
        const wrong = { ...saddle, op: { ...saddle.op, n: n - 1 } };
        expect(() =>
            sobolevStepSet(vertices, edges, disjointPairs, set, { ...STEP, saddle: wrong }),
        ).toThrow('sobolevStepSet: saddle.op.n mismatch');
        // ...and a genuinely differently-sized saddle, not just a doctored field.
        const small = trefoil(n - 1);
        const realSmall = makeHierarchicalSaddle(
            small.vertices,
            small.edges,
            calculateDisjointPairs(small.edges),
        );
        expect(() =>
            sobolevStepSet(vertices, edges, disjointPairs, set, { ...STEP, saddle: realSmall }),
        ).toThrow('sobolevStepSet: saddle.op.n mismatch');
    });

    test("projectionMode other than 'frozen' is refused", () => {
        for (const projectionMode of ['reassemble' as const, undefined]) {
            expect(() =>
                sobolevStepSet(vertices, edges, disjointPairs, set, {
                    ...STEP,
                    projectionMode,
                    saddle,
                }),
            ).toThrow("sobolevStepSet: a hierarchical saddle requires projectionMode 'frozen'");
        }
    });
});

// ---------------------------------------------------------------------------
// E. The worker route. Reachability is MAIN-THREAD / INLINE ONLY (A33 item 2).
// ---------------------------------------------------------------------------

describe('the worker route cannot carry this saddle', () => {
    const n = 48;
    const { vertices, edges } = trefoil(n);
    const disjointPairs = calculateDisjointPairs(edges);

    test('a REAL hierarchical saddle is not structured-cloneable, so it can never reach the worker', () => {
        // The limitation pinned at its root rather than at the guard. `structuredClone` is the SAME
        // serialisation `postMessage` runs, so this is the measurement behind A33 item 2's claim
        // that the object is "not structured-cloneable by construction": it holds closures
        // (`prepare`, `op.applyA`, `precond.apply`). `test/gpu/saddleSeam.test.ts:312` pins the
        // worker's own guard with a PLAIN `{}` precisely because a real saddle dies here first.
        const saddle = makeHierarchicalSaddle(vertices, edges, disjointPairs);
        expect(() => structuredClone({ saddle })).toThrow(/can ?not be cloned|DataCloneError/i);
        // The parts that WOULD be cloneable (the typed arrays) are not the problem — name the
        // closure that is, so a future attempt to "just clone the data" reads this first.
        expect(typeof saddle.prepare).toBe('function');
        expect(typeof saddle.op.applyA).toBe('function');
        expect(typeof saddle.parts.precond.apply).toBe('function');
    });

    test('the worker rejects a hierarchical saddle on the step payload with its documented error', async () => {
        const a = {
            descentMode: 'sobolev' as const,
            vertices,
            mode: 'analytical' as const,
            stepSize: 0.01,
            x0: barycenterTarget(vertices, edges),
            barycenterConstraint: true,
            lengthMode: 'none' as const,
            projectionMode: 'frozen' as const,
        };
        const response = await new Promise<{ type: string; message?: string }>(
            (resolve, reject) => {
                const worker = new Worker(
                    new URL('../../src/worker/solverWorker.ts', import.meta.url).href,
                );
                const timer = setTimeout(() => {
                    worker.terminate();
                    reject(new Error('worker timed out'));
                }, 5000);
                worker.onmessage = (event: MessageEvent<{ type: string; message?: string }>) => {
                    clearTimeout(timer);
                    worker.terminate();
                    resolve(event.data);
                };
                worker.onerror = (event: ErrorEvent) => {
                    clearTimeout(timer);
                    worker.terminate();
                    reject(new Error(`worker error: ${event.message}`));
                };
                worker.postMessage({ type: 'topology', graphVersion: 0, edges });
                // A CLONEABLE stand-in carrying THIS arm's discriminant, because the real object
                // cannot be posted at all (test above). It is what makes the guard reachable, and
                // tagging it `'hierarchical'` additionally shows the worker guard runs BEFORE the
                // pre-arm guards in `optimizer.ts` — with the guard removed, this payload would
                // instead die on `opts.saddle.op.n` with a message that never mentions the worker.
                worker.postMessage({
                    type: 'step',
                    graphVersion: 0,
                    args: { ...a, saddle: { kind: 'hierarchical', op: { n } } },
                });
            },
        );
        expect(response.type).toBe('error');
        expect(response.message).toMatch(/`saddle` cannot cross the worker boundary/);
    }, 10_000);
});

// ---------------------------------------------------------------------------
// F. prepare() once per step; the solver reused across projection iterates.
// ---------------------------------------------------------------------------

test('prepare is called ONCE per step and its solver serves every projection iterate', () => {
    // The dense-stub precedent is `hierarchicalSeam.test.ts:268`; this is the same property against
    // the real factory. It is what "frozen" means on this arm: the tree, the near-field precompute
    // and `Ã⁻¹` are built once (in the factory) and `prepare` is the only per-step work, so a
    // second `prepare` would mean the k GMRES runs of the Schur columns paid twice per step.
    const n = 48;
    const { vertices, edges } = trefoil(n);
    const disjointPairs = calculateDisjointPairs(edges);
    const set: ConstraintSet = [barycenterBlock(offsetBaryTarget(vertices, edges))];
    const real = makeHierarchicalSaddle(vertices, edges, disjointPairs);
    let prepares = 0;
    let solves = 0;
    // The counter lives HERE and not in `src/`: counting is the test's business, and a `src` field
    // for it would be an unreachable-in-production surface.
    const counted: HierarchicalPreparedSaddle = {
        kind: 'hierarchical',
        op: real.op,
        prepare(C) {
            prepares++;
            const inner = real.prepare(C);
            return {
                solve(rhsTop, rhsBottom) {
                    solves++;
                    return inner.solve(rhsTop, rhsBottom);
                },
            };
        },
    };
    const r = sobolevStepSet(vertices, edges, disjointPairs, set, { ...STEP, saddle: counted });
    log(
        'n=48 prepares/solves/projIters',
        `${prepares} / ${solves} / ${r.stats.projectionIterations}`,
    );
    expect(r.accepted).toBe(true);
    expect(prepares).toBe(1);
    // 1 gradient solve + one per projection iterate. MEASURED 4 with projectionIterations = 3.
    expect(solves).toBeGreaterThanOrEqual(2);
    expect(r.stats.projectionIterations).not.toBeNull();
    expect(solves).toBe(1 + (r.stats.projectionIterations as number));
    // The projection really did iterate, so `>= 2` above is not vacuous.
    expect(r.stats.projectionIterations as number).toBeGreaterThanOrEqual(1);
});

// ---------------------------------------------------------------------------
// G. The factory's own refusals.
// ---------------------------------------------------------------------------

describe('the factory refuses what the stack cannot build, before the deep throw', () => {
    const n = 24;
    const ring = trefoil(n);
    const dp = calculateDisjointPairs(ring.edges);

    test('a closed natural ring is accepted (the baseline the refusals below are measured against)', () => {
        const s = makeHierarchicalSaddle(ring.vertices, ring.edges, dp);
        expect(s.build.n).toBe(n);
        expect(s.op.n).toBe(n);
        expect(s.build.leafSize).toBe(DEFAULT_LEAF_SIZE);
        expect(s.build.theta).toBe(0.25);
        expect(s.build.alpha).toBe(ALPHA);
        expect(s.build.beta).toBe(BETA);
        expect(s.build.epsilon).toBe(EPS);
    });

    test.each([
        ['open chain (no wrap edge)', ring.edges.slice(0, n - 1), /CLOSED curve \(\|E\| = \|V\|\)/],
        [
            'a junction / permuted edge',
            ring.edges.map((e, i) => (i === 3 ? ([3, 9] as Edge) : e)),
            /not \(i,i\+1\) or the wrap/,
        ],
        [
            'a duplicated edge in place of the wrap',
            ring.edges.map((e, i) => (i === n - 1 ? ([0, 1] as Edge) : e)),
            /duplicated/,
        ],
    ])('refuses %s with reason "topology"', (_name, edges, pattern) => {
        let caught: unknown;
        try {
            makeHierarchicalSaddle(ring.vertices, edges, calculateDisjointPairs(edges));
        } catch (e) {
            caught = e;
        }
        expect(caught).toBeInstanceOf(HierarchicalSaddleUnsupportedError);
        expect((caught as HierarchicalSaddleUnsupportedError).reason).toBe('topology');
        expect((caught as Error).message).toMatch(pattern);
        // Every refusal must name the fallback, which is the whole reason it exists rather than
        // letting `CurveTopologyError` surface from three frames down (laplacian.ts:13-14).
        expect((caught as Error).message).toMatch(/DENSE path/);
    });

    test('fewer than 3 vertices is refused', () => {
        expect(() => makeHierarchicalSaddle([[0, 0, 0]], [], [])).toThrow(
            HierarchicalSaddleUnsupportedError,
        );
    });

    test('disjointPairs that do not belong to these edges is a RangeError', () => {
        // The value that trips it: the pair list of a DIFFERENT curve. Accepting it would mean the
        // saddle was not built for the step's inputs, which nothing downstream can detect.
        const other = trefoil(n + 4);
        expect(() =>
            makeHierarchicalSaddle(ring.vertices, ring.edges, calculateDisjointPairs(other.edges)),
        ).toThrow(RangeError);
        // A row of the wrong length, same row count.
        const mangled = dp.map((row, i) => (i === 2 ? row.slice(0, row.length - 1) : row));
        expect(() => makeHierarchicalSaddle(ring.vertices, ring.edges, mangled)).toThrow(
            /disjointPairs\[2\]/,
        );
        // ...and the correct one is accepted, so the gate is not simply always-throwing.
        expect(makeHierarchicalSaddle(ring.vertices, ring.edges, dp).build.n).toBe(n);
    });

    // -----------------------------------------------------------------------
    // SUPERSEDED, kept visible rather than deleted (CLAUDE.md: a retracted claim stays on the page
    // with its amendment marker, the way `lowOrder.test.ts:1026-1040` keeps §7.8's `p = 0.0010`).
    //
    // > test('an edgeLengths set is refused by prepare() as a ROUTE error, not solved badly')
    // >   // Row 9's territory (`nullspace.ts`, k = |E| = n). Until it exists there is no
    // >   // hierarchical route for it, so this throw IS the documented dense fallback. A33
    // >   // corrected row 8b: the `edgeLengths` cell is NOT 8b's.
    // >   expect(caught).toBeInstanceOf(SchurRouteError);
    //
    // WITHDRAWN as a statement about `edgeLengths` sets. The premise "until it exists" expired:
    // `nullspace.ts` shipped (row 9) and §11.2 row 9b's dispatch now routes the `|E|` rows to it in
    // `makeHierarchicalSaddle.prepare`, so an `edgeLengths` set is SOLVED here, not refused. What
    // the old test actually pinned — a route refusal escaping the step's `'singular_system'` fold —
    // is NOT withdrawn and is re-asserted below on the set the refusal really belongs to: an
    // all-`O(1)` set above `maxK`, which has no `O(n)` block for a projection to absorb.
    // -----------------------------------------------------------------------

    test("§8.2's hybrid set {barycenter, edgeLengths, pins} is SOLVED, not refused", () => {
        // The combination §8.2 names as one of exactly two that "must actually work", at the size
        // this describe block already builds (n = 24 ⇒ k = 3 + 24 + 3 = 30, i.e. above `maxK`, so
        // the all-Schur route would throw `SchurRouteError` here and did until row 9b's dispatch
        // landed). MEASURED on this fixture, all five numbers below from one run.
        // @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §8.2, §11.2 row 9b
        const s = makeHierarchicalSaddle(ring.vertices, ring.edges, dp);
        const ell0 = ring.edges.map(([p, q]) => {
            const a = ring.vertices[p];
            const b = ring.vertices[q];
            return Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) * 1.01;
        });
        const pinTo: Vec3 = [ring.vertices[0][0] + 0.01, ring.vertices[0][1], ring.vertices[0][2]];
        const set: ConstraintSet = [
            barycenterBlock(offsetBaryTarget(ring.vertices, ring.edges)),
            edgeLengthsBlock(ell0),
            pointBlock(0, pinTo),
        ];
        const { C, phi } = evaluateConstraintSet(set, ring.vertices, ring.edges);
        expect(C.length).toBe(n + 6);
        expect(C.length).toBeGreaterThan(SCHUR_MAX_K);

        const sol = s.prepare(C);
        // It took the NULLSPACE route, and the dispatch found the block where it really is: the
        // `|E|` rows start at 3 (behind `barycenterBlock`) and the 6 `O(1)` rows went to
        // `prepareSchur`. A dispatch that guessed `rowStart = 0` would project onto the wrong
        // subspace and still report a self-consistent residual, so these three are the assertion.
        if (!isNullspaceRoute(sol)) {
            throw new Error("the hybrid set took the SCHUR route, not §8.2's nullspace route");
        }
        const rep = sol.report;
        expect(rep.kEdge).toBe(n);
        expect(rep.kOther).toBe(6);
        expect(rep.edgeRowStart).toBe(3);
        expect(rep.schur.k).toBe(6);
        // The knobs are the factory's, not re-chosen by the route (A35 item 4's ruled `tol`).
        expect(rep.schur.tol).toBe(1e-12);
        // `‖P A_σ P‖₂` is estimated on the PROJECTED operator, NOT taken from the cached
        // unprojected `build.normA2` — substituting the larger norm would shrink `η` and loosen the
        // certificate (`NullspaceOptions.normA2`). MEASURED 17.898 vs the build's 18.0-ish.
        expect(rep.normA2).toBeGreaterThan(0);
        expect(rep.normA2).toBeLessThan(s.build.normA2);
        // The projector is a projector: MEASURED idempotence 1.1e-16, annihilation 1.5e-16,
        // self-adjointness 2.0e-17, so 1e-12 is five decades of margin and still not vacuous (a
        // `precond-unprojected` knock-out reads 4.1e-1, `nullspace.ts`).
        expect(rep.projector.idempotenceDefect).toBeLessThan(1e-12);
        expect(rep.projector.annihilationDefect).toBeLessThan(1e-12);
        expect(rep.projector.selfAdjointDefect).toBeLessThan(1e-12);

        const rhsTop = Array.from({ length: 3 * n }, (_, i) => Math.sin(1 + 7 * i));
        const r = sol.solve(
            rhsTop,
            phi.map((v) => -v),
        );
        // Certified end to end: MEASURED residual 5.84e-13, edge-row satisfaction 1.07e-14,
        // η = 8.76e-14 in 26 iterations. Bars are three decades above the measured values, which is
        // the gap `schur.test.ts` uses for the same quantity.
        expect(r.residual).toBeLessThan(1e-9);
        expect(r.edgeConstraintResidual).toBeLessThan(1e-11);
        expect(r.gmres.eta).toBeLessThan(1e-10);
        expect(r.gmres.censored).toBe(false);
        // Non-vacuity: the solve really ran, and `lambda` comes back in the STACKED row order with
        // the `|E|` multipliers at `rowStart`, not as a bare 6-vector of the `O(1)` rows.
        expect(r.gmres.iters).toBeGreaterThan(0);
        expect(r.lambda.length).toBe(n + 6);
        const edgeLambda = r.lambda.slice(3, 3 + n);
        expect(Math.max(...edgeLambda.map(Math.abs))).toBeGreaterThan(0);
    });

    test('the PURE edgeLengths set also routes to the nullspace route (gauge, not refusal)', () => {
        // Supersedes the withdrawn test quoted above. §8.2 routes `edgeLengthsBlock` by BLOCK, so a
        // set with no `O(1)` rows beside it is `kOther = 0` — `prepareSchur`'s `k = 0` path, a
        // first-class case. It is TRANSLATION-SINGULAR by construction (`C_edge N = 0`), so the
        // route returns `schur.ts`'s gauge `Nᵀx = Nᵀb/σ` with `pinned: false`, and the rhs must be
        // mean-free or the dense system is INCONSISTENT rather than merely gauged (`nullspace.ts`
        // TRANSLATION GAUGE paragraph). MEASURED with a mean-free rhs: residual 4.82e-12.
        const s = makeHierarchicalSaddle(ring.vertices, ring.edges, dp);
        const set: ConstraintSet = [edgeLengthsBlock(ring.edges.map(() => 0.3))];
        const { C, phi } = evaluateConstraintSet(set, ring.vertices, ring.edges);
        expect(C.length).toBe(n);
        expect(C.length).toBeGreaterThan(SCHUR_MAX_K);
        const sol = s.prepare(C);
        if (!isNullspaceRoute(sol)) {
            throw new Error(
                "the pure edgeLengths set took the SCHUR route, not §8.2's nullspace route",
            );
        }
        const rep = sol.report;
        expect(rep.kEdge).toBe(n);
        expect(rep.kOther).toBe(0);
        expect(rep.edgeRowStart).toBe(0);
        expect(rep.schur.k).toBe(0);
        expect(rep.schur.pinned).toBe(false);
        const b = new Array<number>(3 * n).fill(0);
        for (let c = 0; c < 3; c++) {
            let mean = 0;
            for (let i = 0; i < n; i++) {
                b[c * n + i] = Math.sin(1 + 7 * (c * n + i));
                mean += b[c * n + i];
            }
            mean /= n;
            for (let i = 0; i < n; i++) b[c * n + i] -= mean;
        }
        const r = sol.solve(
            b,
            phi.map((v) => -v),
        );
        expect(r.residual).toBeLessThan(1e-9);
        expect(r.edgeConstraintResidual).toBeLessThan(1e-11);
    });

    test('an all-O(1) set is still routed to prepareSchur, and above maxK still refused', () => {
        // The control half of the dispatch: the route is chosen by BLOCK, so a set with no
        // `edgeLengths` rows reaches `prepareSchur` exactly as before — `report` is a
        // `SchurPrepareReport` (no `projector`), which is what pins that the dispatch did not
        // quietly send everything through the projection.
        const s = makeHierarchicalSaddle(ring.vertices, ring.edges, dp);
        const ok: ConstraintSet = [
            barycenterBlock(offsetBaryTarget(ring.vertices, ring.edges)),
            totalLengthBlock(totalLength(ring.vertices, ring.edges) * 1.01),
        ];
        const okC = evaluateConstraintSet(ok, ring.vertices, ring.edges).C;
        expect(okC.length).toBe(4);
        const okSolver = s.prepare(okC);
        if (isNullspaceRoute(okSolver)) {
            throw new Error('an all-O(1) set took the nullspace route; the dispatch over-reaches');
        }
        expect(okSolver.report.k).toBe(4);

        // And the refusal the withdrawn test was really pinning, on the set it belongs to: 6 pins
        // is 18 `O(1)` rows, above `maxK = 16`, with NO `O(n)` block for a projection to absorb. It
        // must still throw, because `SchurRouteError`'s own advice ("raise maxK only if k really is
        // O(1)") is the advice that applies — rerouting it would hide a real cost decision.
        const tooMany: ConstraintSet = Array.from({ length: 6 }, (_, i) =>
            pointBlock(i * 3, [
                ring.vertices[i * 3][0] + 0.01,
                ring.vertices[i * 3][1],
                ring.vertices[i * 3][2],
            ]),
        );
        const bigC = evaluateConstraintSet(tooMany, ring.vertices, ring.edges).C;
        expect(bigC.length).toBe(18);
        expect(bigC.length).toBeGreaterThan(SCHUR_MAX_K);
        let caught: unknown;
        try {
            s.prepare(bigC);
        } catch (e) {
            caught = e;
        }
        expect(caught).toBeInstanceOf(SchurRouteError);
        // It must ESCAPE the step's 'singular_system' fold: a dispatch bug is not a singular matrix.
        expect(caught).toBeInstanceOf(SolveNotCertifiedError);
        expect(caught).not.toBeInstanceOf(ExternalSolveError);
    });
});

// ---------------------------------------------------------------------------
// H. Typed errors across the seam: the `SolveNotCertifiedError` split (A35 item 6).
// ---------------------------------------------------------------------------

describe('a solve that cannot certify itself propagates; a singular one does not', () => {
    const n = 48;
    const { vertices, edges } = trefoil(n);
    const disjointPairs = calculateDisjointPairs(edges);
    const set: ConstraintSet = [barycenterBlock(offsetBaryTarget(vertices, edges))];

    test('GMRES non-convergence in the GRADIENT solve escapes the singular_system fold', () => {
        // `maxit: 1` at the SHIPPED `etaAccept`, i.e. the production certificate doing its job.
        const s = makeHierarchicalSaddle(vertices, edges, disjointPairs, { schur: { maxit: 1 } });
        let caught: unknown;
        try {
            sobolevStepSet(vertices, edges, disjointPairs, set, { ...STEP, saddle: s });
        } catch (e) {
            caught = e;
        }
        expect(caught).toBeInstanceOf(SchurConvergenceError);
        // The point of the whole `SolveNotCertifiedError` change: this must reach the caller...
        expect(caught).toBeInstanceOf(SolveNotCertifiedError);
        // ...and must NOT look like the EXTERNAL (GPU) arm, whose documented recovery is "re-run on
        // the CPU path" ([DESIGN §4.5] trigger 3) — re-running this CPU GMRES fails identically.
        expect(caught).not.toBeInstanceOf(ExternalSolveError);
        // A loose `tol` reaches the same throw, which is why no `tol` knob can quietly degrade the
        // step: MEASURED η = 2.49e-4 at tol = 1e-2 against etaAccept = 1e-10.
        const loose = makeHierarchicalSaddle(vertices, edges, disjointPairs, {
            schur: { tol: 1e-2 },
        });
        expect(() =>
            sobolevStepSet(vertices, edges, disjointPairs, set, { ...STEP, saddle: loose }),
        ).toThrow(SchurConvergenceError);
    });

    test('a PLAIN Error from prepare is still folded into singular_system (the contrast)', () => {
        // Without this the assertion above would not show that the TYPE is what decides: every
        // throw would propagate and `'singular_system'` would be dead.
        const real = makeHierarchicalSaddle(vertices, edges, disjointPairs);
        const bad: HierarchicalPreparedSaddle = {
            kind: 'hierarchical',
            op: real.op,
            prepare: () => {
                throw new Error('a genuinely singular saddle');
            },
        };
        const r = sobolevStepSet(vertices, edges, disjointPairs, set, { ...STEP, saddle: bad });
        expect(r.accepted).toBe(false);
        expect(r.stats.reason).toBe('singular_system');
        expect(Number.isNaN(r.stats.residual)).toBe(true);
    });

    test('the PROJECTION solve has the same two behaviours (lineSearch.ts:329)', () => {
        // Three of the four solves of a normal step are projection solves, so the gradient-side
        // rethrow alone would leave the limitation latent on most of the step.
        const real = makeHierarchicalSaddle(vertices, edges, disjointPairs);
        const flaky = (err: () => Error): HierarchicalPreparedSaddle => {
            let call = 0;
            return {
                kind: 'hierarchical',
                op: real.op,
                prepare(C) {
                    const inner = real.prepare(C);
                    return {
                        solve(rhsTop, rhsBottom) {
                            call++;
                            if (call >= 2) throw err();
                            return inner.solve(rhsTop, rhsBottom);
                        },
                    };
                },
            };
        };
        expect(() =>
            sobolevStepSet(vertices, edges, disjointPairs, set, {
                ...STEP,
                saddle: flaky(() => new SolveNotCertifiedError('projection could not certify')),
            }),
        ).toThrow('projection could not certify');
        const r = sobolevStepSet(vertices, edges, disjointPairs, set, {
            ...STEP,
            saddle: flaky(() => new Error('plain projection failure')),
        });
        expect(r.accepted).toBe(false);
        expect(r.stats.reason).toBe('projection_failed');
    });
});

// ---------------------------------------------------------------------------
// I. `factorMode` is silently inert on this arm (spec §11.2 row 8b, R2).
// ---------------------------------------------------------------------------

test('factorMode moves bits on the dense arm and NONE on the hierarchical arm', () => {
    // Makes the TSDoc caveat added to `optimizer.ts`'s `factorMode?` and `gradient.ts`'s
    // `solveConstrainedGradientSetFrozen` executable. The dense A/B is the control: if 'lu' and
    // 'ldlt' were bit-equal there too, the hierarchical result would prove nothing.
    const n = 48;
    const { vertices, edges } = trefoil(n);
    const disjointPairs = calculateDisjointPairs(edges);
    const set: ConstraintSet = [barycenterBlock(offsetBaryTarget(vertices, edges))];
    const countDiff = (a: number[], b: number[]): number => {
        let d = 0;
        for (let i = 0; i < a.length; i++) if (!Object.is(a[i], b[i])) d++;
        return d;
    };

    const dLu = flatten(
        sobolevStepSet(vertices, edges, disjointPairs, set, { ...STEP, factorMode: 'lu' }).vertices,
    );
    const dLd = flatten(
        sobolevStepSet(vertices, edges, disjointPairs, set, { ...STEP, factorMode: 'ldlt' })
            .vertices,
    );
    const denseDiff = countDiff(dLu, dLd);
    log('n=48 dense lu-vs-ldlt differing coords', `${denseDiff} / ${dLu.length}`);
    expect(denseDiff).toBeGreaterThan(0); // MEASURED 118 / 144

    // Two independently built saddles, so a shared mutable cache cannot fake the agreement.
    const hLu = flatten(
        sobolevStepSet(vertices, edges, disjointPairs, set, {
            ...STEP,
            saddle: makeHierarchicalSaddle(vertices, edges, disjointPairs),
            factorMode: 'lu',
        }).vertices,
    );
    const hLd = flatten(
        sobolevStepSet(vertices, edges, disjointPairs, set, {
            ...STEP,
            saddle: makeHierarchicalSaddle(vertices, edges, disjointPairs),
            factorMode: 'ldlt',
        }).vertices,
    );
    log('n=48 hierarchical lu-vs-ldlt differing coords', `${countDiff(hLu, hLd)} / ${hLu.length}`);
    expect(countDiff(hLu, hLd)).toBe(0); // MEASURED 0 / 144 — the option selects nothing here
});

// ---------------------------------------------------------------------------
// J. The cached `‖A_σ‖₂` is bit-identical to recomputing it per prepare.
// ---------------------------------------------------------------------------

test('build.normA2 is bit-identical to estimateNorm2 and to what prepare would compute', () => {
    // `estimateNorm2` is deterministic (fixed LCG seed, fixed iteration count, `gmres.ts:273-296`)
    // and `op` is frozen, so caching it in the factory is a pure saving of 30 matvecs per
    // `prepare` — but only if it really is the same number. `Object.is`, not a tolerance: a
    // tolerance here would hide exactly the drift the caching could introduce.
    const n = 48;
    const { vertices, edges } = trefoil(n);
    const disjointPairs = calculateDisjointPairs(edges);
    const s = makeHierarchicalSaddle(vertices, edges, disjointPairs);
    const recomputed = estimateNorm2((u, out) => applyAsigma(s.op, u, out), 3 * n).norm;
    log('n=48 build.normA2 / recomputed', `${s.build.normA2} / ${recomputed}`);
    expect(Object.is(s.build.normA2, recomputed)).toBe(true);
    expect(s.build.normA2).toBeGreaterThan(0);
    // The report `prepare` publishes must carry the SAME value, i.e. it really is being used.
    const set: ConstraintSet = [barycenterBlock(offsetBaryTarget(vertices, edges))];
    const C = evaluateConstraintSet(set, vertices, edges).C;
    expect(Object.is(s.prepare(C).report.normA2, s.build.normA2)).toBe(true);
    // And an explicit override still wins, so the cache is not a hard-coding.
    expect(s.prepare(C).report.normA2).not.toBe(123.5);
    const o = makeHierarchicalSaddle(vertices, edges, disjointPairs, {
        schur: { normA2: 123.5 },
    });
    expect(o.prepare(C).report.normA2).toBe(123.5);
    // The default `tol` is 9b's ruled 1e-12 and is NOT re-chosen here (A35 item 4). The narrowing
    // is also the claim that a barycentre-only set is still an all-Schur set after §8.2's dispatch:
    // the cached `normA2` above is the SCHUR route's input and would not be used on the other one.
    const prepared = s.prepare(C);
    if (isNullspaceRoute(prepared)) {
        throw new Error('a barycentre-only set took the nullspace route');
    }
    expect(prepared.report.tol).toBe(1e-12);
    // The curve extent is RECORDED, not normalised (issue #69). Checked against a SECOND,
    // INDEPENDENT bbox loop rather than against `curveExtent` itself: importing the production
    // helper here would be checking production against itself and a shared defect (`hi` instead of
    // `hi − lo`) would pass — the same deliberate duplication `extent.ts`'s TSDoc requires of
    // `precond.test.ts` (memory `common-mode-blindness`, issue #72 comment 2).
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
    log('n=48 build.curveExtent / independent bbox', `${s.build.curveExtent} / ${ext}`);
    expect(s.build.curveExtent).toBeCloseTo(ext, 12);
    // The load-bearing fact, not the digits: the shipped fixture is NOT at extent ~1, which is the
    // scale every `out-bct-*.json` oracle was measured at (κ_near moves 1.70× between the two).
    expect(s.build.curveExtent).toBeGreaterThan(5);
    // ...and the preconditioner computed its own `ext`, so no scale factor is applied twice.
    expect(s.parts.precond.sigma).toBe(s.op.sigma);
});
