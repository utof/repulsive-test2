/**
 * Row 8's HIERARCHICAL arm of the saddle seam: `HierarchicalPreparedSaddle`,
 * `HierarchicalSaddleSolver`, the `kind`-tagged `FrozenSaddleOperator`,
 * `operatorSaddleResidual`, and the two guards `sobolevStepSet` grew for them.
 *
 * **What this file can prove, and what it cannot.** The production producer is task 8b
 * (`makeHierarchicalSaddle`), so every hierarchical operator here is a STUB whose `applyA` /
 * `solve` are the DENSE ones. The equality tests are therefore PLUMBING differentials: they
 * prove the hierarchical route is wired to the right inputs and that its outputs reach the step —
 * not that GMRES converges. The common-mode hazard (CLAUDE.md: a two-path differential cannot
 * pin anything the two paths share) is answered by two knock-outs: a stub solver that is wrong on
 * purpose MUST move the step, and a stub operator that is wrong in its last bit MUST move the
 * residual. Without those, a silent fall-back to the dense path would read exactly like a pass.
 *
 * Bit-identity is claimed, not tolerated, because the stub performs the dense path's OWN calls:
 * `solveSaddleFromA` on the same `a`/`C`/rhs, and re-factoring a deterministic `K` reproduces the
 * frozen route's `solveFactored` bits exactly. A tolerance here would hide the wiring defects this
 * file exists to catch.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.1, §11.2 rows 8 / 8b
 */
import { expect, test } from 'bun:test';
import { trefoil } from '../../src/core/fixtures';
import { DEFAULTS, sobolevStepSet } from '../../src/core/optimizer';
import {
    barycenterBlock,
    type ConstraintSet,
    evaluateConstraintSet,
    totalLength,
    totalLengthBlock,
} from '../../src/core/sobolev/constraintSet';
import { barycenterTarget } from '../../src/core/sobolev/constraints';
import type { SobolevOperator } from '../../src/core/sobolev/hierarchical/operator';
import { assembleAFlat } from '../../src/core/sobolev/innerProduct';
import { projectOntoConstraintSet } from '../../src/core/sobolev/lineSearch';
import {
    type FrozenSaddleOperator,
    type HierarchicalPreparedSaddle,
    type HierarchicalSaddleSolver,
    operatorSaddleResidual,
    solveSaddleFromA,
    solveSaddleFrozen,
} from '../../src/core/sobolev/linsolve';
import { timingsEnd } from '../../src/core/sobolev/phaseTimings';
import { calculateDisjointPairs } from '../../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../../src/core/testConfigs';

const N = 24;

function fixture() {
    const { vertices, edges } = trefoil(N);
    const dp = calculateDisjointPairs(edges);
    const set: ConstraintSet = [
        barycenterBlock(barycenterTarget(vertices, edges)),
        totalLengthBlock(totalLength(vertices, edges)),
    ];
    const a = assembleAFlat(vertices, edges, dp, DEFAULTS.alpha, DEFAULTS.beta, DEFAULTS.epsilon);
    return { vertices, edges, dp, set, a, n: vertices.length };
}

/**
 * `out = Ā u` in `structuredSaddleResidual`'s OWN loop and accumulation order (linsolve.ts: block
 * `b`, row `i`, `s = 0` then `s += a[i*n+j] * u[b*n+j]`). Why that order exactly:
 * `operatorSaddleResidual` starts each row accumulator AT the operator's output, so an operator
 * reproducing this order makes the two residual branches bit-identical — the only tolerance-free
 * check that the sibling branch is faithful to the dense one.
 */
function denseOp(a: Float64Array, n: number): SobolevOperator {
    return {
        n,
        applyA(u, out) {
            for (let b = 0; b < 3; b++) {
                const off = b * n;
                for (let i = 0; i < n; i++) {
                    let s = 0;
                    const base = i * n;
                    for (let j = 0; j < n; j++) {
                        s += a[base + j] * u[off + j];
                    }
                    out[off + i] = s;
                }
            }
        },
    };
}

/** The same operator, wrong in one last bit — so the equality above cannot be an accident. */
function perturbedOp(a: Float64Array, n: number): SobolevOperator {
    const exact = denseOp(a, n);
    return {
        n,
        applyA(u, out) {
            exact.applyA(u, out);
            out[0] = out[0] * (1 + Number.EPSILON);
        },
    };
}

interface CountingSolver {
    solver: HierarchicalSaddleSolver;
    calls: () => number;
}

/**
 * A `HierarchicalSaddleSolver` that solves the DENSE system the step would have solved, re-doing
 * the whole factorization per solve — which is exactly the hierarchical contract ("reuse the
 * build, run a fresh solve", spec §5.1) and is bit-identical to the dense frozen route because
 * factoring the same `K` twice yields the same bits. `perturb` is the knock-out dial.
 */
function exactSolver(a: Float64Array, n: number, C: number[][], perturb: number): CountingSolver {
    let calls = 0;
    return {
        calls: () => calls,
        solver: {
            solve(rhsTop, rhsBottom) {
                calls += 1;
                const r = solveSaddleFromA(
                    a,
                    n,
                    C,
                    Array.from(rhsTop),
                    rhsBottom === undefined ? undefined : Array.from(rhsBottom),
                );
                return {
                    x: perturb === 0 ? r.x : r.x.map((v) => v * (1 + perturb)),
                    lambda: r.lambda,
                    residual: r.residual,
                };
            },
        },
    };
}

function hierPrepared(v: Vec3[], edges: Edge[], dp: number[][], perturb = 0) {
    const n = v.length;
    const a = assembleAFlat(v, edges, dp, DEFAULTS.alpha, DEFAULTS.beta, DEFAULTS.epsilon);
    let prepares = 0;
    let last: CountingSolver | undefined;
    const prepared: HierarchicalPreparedSaddle = {
        kind: 'hierarchical',
        op: denseOp(a, n),
        prepare(C: number[][]): HierarchicalSaddleSolver {
            prepares += 1;
            last = exactSolver(a, n, C, perturb);
            return last.solver;
        },
    };
    return { prepared, prepares: () => prepares, solverCalls: () => last?.calls() ?? 0 };
}

const STEP_OPTS = {
    mode: 'analytical' as const,
    alpha: DEFAULTS.alpha,
    beta: DEFAULTS.beta,
    epsilon: DEFAULTS.epsilon,
    collectField: true,
    // The hierarchical route REQUIRES this (the reassemble projection has no matrix-free path and
    // would silently fall back to a per-iterate dense assembly). Guard-tested below.
    projectionMode: 'frozen' as const,
};

// ---------------------------------------------------------------------------------------------
// operatorSaddleResidual — the operator-shaped sibling of the (module-private) dense residual.
// ---------------------------------------------------------------------------------------------

test('operatorSaddleResidual: bit-identical to the dense structured residual on the same z/rhs', () => {
    const f = fixture();
    const { C } = evaluateConstraintSet(f.set, f.vertices, f.edges);
    // A literal rhs: no transcendental, so this assertion carries no Math.pow engine dependence.
    const rhsTop = Array.from({ length: 3 * f.n }, (_, i) => ((i % 7) - 3) / 8);
    const rhsBottom = C.map((_, r) => (r + 1) / 16);
    const r = solveSaddleFromA(f.a, f.n, C, rhsTop, rhsBottom);
    // Non-vacuity: a zero residual would make the comparison below pass on any formula.
    expect(r.residual).toBeGreaterThan(0);
    const z = [...r.x, ...r.lambda];
    const rhs = [...rhsTop, ...rhsBottom];
    expect(operatorSaddleResidual(denseOp(f.a, f.n), C, z, rhs)).toBe(r.residual);
    // Knock-out: the operator IS consulted — a one-ULP-wrong apply gives a different number.
    expect(operatorSaddleResidual(perturbedOp(f.a, f.n), C, z, rhs)).not.toBe(r.residual);
});

test('operatorSaddleResidual: rejects an operator whose n disagrees with the vectors', () => {
    const f = fixture();
    const { C } = evaluateConstraintSet(f.set, f.vertices, f.edges);
    const z = new Array<number>(3 * f.n + C.length).fill(0);
    const rhs = new Array<number>(3 * f.n + C.length).fill(0);
    expect(() => operatorSaddleResidual(denseOp(f.a, f.n), C, z.slice(1), rhs)).toThrow(
        /operatorSaddleResidual/,
    );
});

// ---------------------------------------------------------------------------------------------
// solveSaddleFrozen — the kind dispatch.
// ---------------------------------------------------------------------------------------------

test('solveSaddleFrozen: the hierarchical arm delegates to its solver and forwards the result', () => {
    const f = fixture();
    const { C } = evaluateConstraintSet(f.set, f.vertices, f.edges);
    const s = exactSolver(f.a, f.n, C, 0);
    const op: FrozenSaddleOperator = {
        kind: 'hierarchical',
        n: f.n,
        C,
        op: denseOp(f.a, f.n),
        solver: s.solver,
    };
    const rhsTop = Array.from({ length: 3 * f.n }, (_, i) => ((i % 5) - 2) / 4);
    const rhsBottom = C.map((_, r) => -(r + 1) / 32);
    const viaFrozen = solveSaddleFrozen(op, rhsTop, rhsBottom);
    expect(s.calls()).toBe(1);
    // The dense solve of the SAME system, for free: the stub is that call, so this pins
    // "forwarded verbatim", not the arithmetic.
    const direct = solveSaddleFromA(f.a, f.n, C, rhsTop, rhsBottom);
    expect(viaFrozen.residual).toBe(direct.residual);
    for (let i = 0; i < direct.x.length; i++) expect(viaFrozen.x[i]).toBe(direct.x[i]);
    for (let i = 0; i < direct.lambda.length; i++)
        expect(viaFrozen.lambda[i]).toBe(direct.lambda[i]);
});

test('solveSaddleFrozen: the rhs length guards run BEFORE the hierarchical solver is called', () => {
    const f = fixture();
    const { C } = evaluateConstraintSet(f.set, f.vertices, f.edges);
    const s = exactSolver(f.a, f.n, C, 0);
    const op: FrozenSaddleOperator = {
        kind: 'hierarchical',
        n: f.n,
        C,
        op: denseOp(f.a, f.n),
        solver: s.solver,
    };
    expect(() => solveSaddleFrozen(op, new Array<number>(3 * f.n - 1).fill(0))).toThrow(
        /rhsTop length/,
    );
    expect(() =>
        solveSaddleFrozen(op, new Array<number>(3 * f.n).fill(0), new Array<number>(99).fill(0)),
    ).toThrow(/rhsBottom length/);
    // A delegated call on a mis-shaped rhs is the failure this ordering prevents: the solver
    // cannot validate what it never sees the length of (its `C` is frozen, the rhs is not).
    expect(s.calls()).toBe(0);
});

// ---------------------------------------------------------------------------------------------
// The step, end to end.
// ---------------------------------------------------------------------------------------------

test('hierarchical step: an exactly-dense stub reproduces the dense frozen step BIT-for-bit', () => {
    const f = fixture();
    const dense = sobolevStepSet(f.vertices, f.edges, f.dp, f.set, STEP_OPTS);
    const h = hierPrepared(f.vertices, f.edges, f.dp);
    const hier = sobolevStepSet(f.vertices, f.edges, f.dp, f.set, {
        ...STEP_OPTS,
        saddle: h.prepared,
    });
    // Non-vacuity: two REJECTED steps both echo the input vertices and would compare equal.
    expect(dense.accepted).toBe(true);
    expect(hier.accepted).toBe(true);
    expect(hier.energy).toBe(dense.energy);
    expect(hier.stats.tau).toBe(dense.stats.tau);
    expect(hier.stats.residual).toBe(dense.stats.residual);
    expect(hier.stats.gradientL2Norm).toBe(dense.stats.gradientL2Norm);
    expect(hier.stats.projectionIterations).toBe(dense.stats.projectionIterations);
    for (let i = 0; i < dense.vertices.length; i++)
        for (let c = 0; c < 3; c++) expect(hier.vertices[i][c]).toBe(dense.vertices[i][c]);
    const d = dense.descentField as Vec3[];
    const t = hier.descentField as Vec3[];
    for (let i = 0; i < d.length; i++) for (let c = 0; c < 3; c++) expect(t[i][c]).toBe(d[i][c]);
});

test('hierarchical step: the PROJECTION solves route through the same prepared solver', () => {
    const f = fixture();
    const h = hierPrepared(f.vertices, f.edges, f.dp);
    const r = sobolevStepSet(f.vertices, f.edges, f.dp, f.set, {
        ...STEP_OPTS,
        saddle: h.prepared,
    });
    expect(r.accepted).toBe(true);
    // The census's open question (task-8-seam-census §1): with projectionMode 'frozen' the
    // projection reaches the hierarchical solver through solveSaddleFrozen's dispatch, so it
    // never assembles a dense Ā. One prepare per step; ≥1 gradient solve + ≥1 projection solve.
    expect(h.prepares()).toBe(1);
    // A projection that converged at iteration 0 would make the count below vacuous.
    expect(r.stats.projectionIterations).toBeGreaterThanOrEqual(1);
    expect(h.solverCalls()).toBeGreaterThanOrEqual(2);
});

test('hierarchical step: a stub solver that is WRONG moves the step (no silent dense fallback)', () => {
    const f = fixture();
    const dense = sobolevStepSet(f.vertices, f.edges, f.dp, f.set, STEP_OPTS);
    const bad = hierPrepared(f.vertices, f.edges, f.dp, 1e-6);
    const hier = sobolevStepSet(f.vertices, f.edges, f.dp, f.set, {
        ...STEP_OPTS,
        saddle: bad.prepared,
    });
    expect(hier.accepted).toBe(true);
    let moved = 0;
    for (let i = 0; i < dense.vertices.length; i++)
        for (let c = 0; c < 3; c++)
            if (!Object.is(hier.vertices[i][c], dense.vertices[i][c])) moved++;
    expect(moved).toBeGreaterThan(0);
});

// ---------------------------------------------------------------------------------------------
// The guards. Both are on the PRE-ARM path, like the dense `saddle.A` guard.
// ---------------------------------------------------------------------------------------------

test('sobolevStepSet: a hierarchical saddle whose op.n disagrees throws before anything is solved', () => {
    const f = fixture();
    const h = hierPrepared(f.vertices, f.edges, f.dp);
    const wrong = { ...h.prepared, op: denseOp(f.a, f.n - 1) };
    expect(() =>
        sobolevStepSet(f.vertices, f.edges, f.dp, f.set, { ...STEP_OPTS, saddle: wrong }),
    ).toThrow(/saddle\.op\.n/);
    expect(() =>
        sobolevStepSet(f.vertices, f.edges, f.dp, f.set, {
            ...STEP_OPTS,
            saddle: wrong,
            collectTimings: true,
        }),
    ).toThrow(/saddle\.op\.n/);
    // Pre-arm, so no disarm is needed — a guard moved below timingsBegin() leaves this armed.
    expect(timingsEnd()).toBeNull();
    expect(h.prepares()).toBe(0);
});

test('sobolevStepSet: a hierarchical saddle without projectionMode frozen is REFUSED, not silently densified', () => {
    const f = fixture();
    const h = hierPrepared(f.vertices, f.edges, f.dp);
    const { projectionMode: _drop, ...noMode } = STEP_OPTS;
    expect(() =>
        sobolevStepSet(f.vertices, f.edges, f.dp, f.set, { ...noMode, saddle: h.prepared }),
    ).toThrow(/projectionMode/);
    expect(() =>
        sobolevStepSet(f.vertices, f.edges, f.dp, f.set, {
            ...noMode,
            projectionMode: 'reassemble',
            saddle: h.prepared,
        }),
    ).toThrow(/projectionMode/);
    expect(timingsEnd()).toBeNull();
    expect(h.prepares()).toBe(0);
});

// ---------------------------------------------------------------------------------------------
// projectOntoConstraintSet, called DIRECTLY — the half of the step that spec §5.1 found had no
// hierarchical route at all. Its body is byte-for-byte unchanged (§1.4); the route exists only
// because solveSaddleFrozen dispatches on `kind`, which is what this pins.
// ---------------------------------------------------------------------------------------------

test('projectOntoConstraintSet: the hierarchical frozen operator projects, bit-identically to the dense one', () => {
    const f = fixture();
    const { C } = evaluateConstraintSet(f.set, f.vertices, f.edges);
    // A candidate OFF the constraints, deterministically (no RNG, no transcendental): the
    // projection must actually correct it, else `iterations` is 0 and this test pins nothing.
    const off = f.vertices.map(
        (p, i): Vec3 => [
            p[0] + 0.05 * ((i % 3) - 1),
            p[1] - 0.04 * ((i % 5) - 2),
            p[2] + 0.03 * ((i % 7) - 3),
        ],
    );
    const common = [
        off,
        f.edges,
        f.dp,
        DEFAULTS.alpha,
        DEFAULTS.beta,
        DEFAULTS.epsilon,
        f.set,
    ] as const;

    const first = solveSaddleFromA(f.a, f.n, C, new Array<number>(3 * f.n).fill(0));
    const dense = projectOntoConstraintSet(...common, {
        frozen: { kind: 'dense', a: f.a, n: f.n, C, fac: first.fac },
    });
    const s = exactSolver(f.a, f.n, C, 0);
    const hier = projectOntoConstraintSet(...common, {
        frozen: { kind: 'hierarchical', n: f.n, C, op: denseOp(f.a, f.n), solver: s.solver },
    });

    expect(dense.ok).toBe(true);
    // A projection that converged at iteration 0 would never reach a solve at all.
    expect(dense.iterations).toBeGreaterThanOrEqual(1);
    expect(s.calls()).toBe(dense.iterations);
    expect(hier.ok).toBe(dense.ok);
    expect(hier.iterations).toBe(dense.iterations);
    expect(hier.phiNorm).toBe(dense.phiNorm);
    for (let i = 0; i < dense.vertices.length; i++)
        for (let c = 0; c < 3; c++) expect(hier.vertices[i][c]).toBe(dense.vertices[i][c]);
});
