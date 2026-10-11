/**
 * DENSE SADDLE PATH — BIT-LEVEL GOLDEN. The instrument for spec §11.2 row 8's gate
 * *"the dense branch stays byte-identical"* (the row-8 seam change touches `linsolve.ts`,
 * `gradient.ts`, `lineSearch.ts`, `optimizer.ts`, `src/gpu/saddle.ts`).
 *
 * Why: nothing else in the repo can measure that gate. Every exact assertion near the saddle
 * solve is on a bool, a small int, a power-of-two `tau`, a `projection_iterations` count or a
 * same-build A/B comparison, and every vertex/solution comparison is relative ≤1e-9, so a solve
 * that moves by one ULP is invisible to all of them. This file pins the floating-point arrays
 * THEMSELVES with `Object.is`, in the shape of `test/golden.test.ts`.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §11.2 (the A24 box, 2026-10-11), §11.4
 * @see test/golden.test.ts (the bit-gate shape this copies)
 *
 * What it pins (all DENSE path, all bit-exact — x/g̃, λ, residual, vertices, energy):
 *  T1  `solveSaddleFromA` and `solveSaddleFrozen` on LITERAL inputs (A, C, rhs stored in the
 *      fixture), × `factorMode` {ldlt, lu} × rhs shape {gradient, projection, mixed}. The frozen
 *      solver is also asserted bit-equal to `solveSaddleFromA` on the same `fac`.
 *  T1b `gradient.ts`'s entry points (`solveConstrainedGradientSetFrozen` with and without a
 *      dense `saddle`, `solveConstrainedGradientSet`, `solveConstrainedGradient`) against the
 *      T1 gradient solve — no extra storage, so the Prepared arm and the dense arm are held to
 *      the same stored bits.
 *  T2  `projectOntoConstraintSet` called DIRECTLY from a perturbed iterate (several correction
 *      solves), reassemble route AND frozen route, × `factorMode`.
 *  T3  `sobolevStepSet` end to end — a 3-step chain per case so a one-ULP difference compounds —
 *      on both projection routes (`projectionMode` omitted ⇒ reassemble; `'frozen'`), × `factorMode`
 *      omitted/`'lu'`, over every constraint shape that changes `C`'s shape: empty (k=0),
 *      barycenter (3), +totalLength (4), +point (6), +edgeLengths (9 / 15). Each chain is re-run with a
 *      dense `PreparedSaddle` supplied via `opts.saddle` and held to the SAME stored bits.
 *
 * WHAT THIS ESTABLISHES: the dense path's outputs on these inputs are bit-identical to what the
 * code produced when this golden was captured (commit + seam-file hashes in `provenance`).
 *
 * WHAT THIS DOES NOT ESTABLISH — read before citing it: the golden was generated FROM the code it
 * gates, so it is self-consistent by construction. It can only detect a FUTURE change. It is a
 * regression / byte-identity gate, NOT a correctness gate: it does not say the dense solve is
 * right. Correctness is the `oracle/golden/*.json` numpy comparisons' job (test/sobolev/*), and
 * those keep their tolerances. Likewise it does not cover the GPU / `kind: 'external'`
 * factorization path, `ExternalSolveError`, the hierarchical route, or timings.
 *
 * Platform: T1 reads its inputs literally from the fixture, so it depends on no transcendental.
 * T1b/T2/T3 recompute `assembleAFlat` live and T3 also `gradientAnalytical` (Math.pow & friends
 * are implementation-approximated in ECMAScript), exactly as `test/golden.test.ts` already does;
 * `provenance.runtime` records the engine. A red T1b/T2/T3 with a green T1 localises the change
 * OUTSIDE the factor/solve in `linsolve.ts`.
 *
 * FORBIDDEN: regenerating this golden to make a failing test pass. If the row-8 seam change (or
 * anything else) turns this red, either the diff is wrong or the numerical change is intentional and needs a
 * WRITTEN RULING (spec §11.2, A24) — never a re-capture. Regeneration exists (below) only because a
 * golden nobody can legitimately regenerate rots; it is guarded, logged in the fixture, and always
 * ends RED so it can never masquerade as a pass.
 *
 * Regenerate (guarded):
 *   DENSE_SADDLE_GOLDEN_REGENERATE='<why, ≥20 chars>' bun test test/sobolev/denseSaddleBitGolden.test.ts
 * and, if any seam file differs from `main` (committed or not), ALSO
 *   DENSE_SADDLE_GOLDEN_RULING='<path of the written ruling, must exist>'.
 * Then re-run WITHOUT the env vars; the fixture diff must go in the same commit as the ruling.
 *
 * Row-8 edit allowance: the spec makes `PreparedSaddle` and `FrozenSaddleOperator` `kind`-tagged
 * unions, which breaks this file at COMPILE time only, at three sites: the `densePrepared()`
 * literal, the `op` literal in `runSolves()`, and the `.frozen.a/.C/.n` reads in T1b. The only
 * edits this file may need are `kind: 'dense'` on those literals and a narrowing on those reads
 * (`tsc` is the oracle for the list). The fixture must stay byte-identical (`cmp`); an edit that
 * changes an expected value is a red, not an allowance.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.1, §11.2 row 8
 */
import { expect, test } from 'bun:test';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { sobolevStepSet } from '../../src/core/optimizer';
import {
    barycenterBlock,
    type ConstraintSet,
    edgeLengths,
    edgeLengthsBlock,
    evaluateConstraintSet,
    pointBlock,
    totalLength,
    totalLengthBlock,
} from '../../src/core/sobolev/constraintSet';
import { barycenterTarget } from '../../src/core/sobolev/constraints';
import {
    solveConstrainedGradient,
    solveConstrainedGradientSet,
    solveConstrainedGradientSetFrozen,
} from '../../src/core/sobolev/gradient';
import { assembleAFlat } from '../../src/core/sobolev/innerProduct';
import { flatten, unflatten } from '../../src/core/sobolev/layout';
import { projectOntoConstraintSet } from '../../src/core/sobolev/lineSearch';
import {
    type FactorMode,
    type FrozenSaddleOperator,
    type PreparedSaddle,
    solveSaddleFromA,
    solveSaddleFrozen,
} from '../../src/core/sobolev/linsolve';
import { calculateDisjointPairs, gradientAnalytical } from '../../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../../src/core/testConfigs';

// ---------------------------------------------------------------------------------------------
// The case matrix. Shared by capture and verification, so the two can never drift apart; the
// verification asserts the fixture holds EXACTLY these ids (a truncated fixture is a red test,
// not a smaller gate).
// ---------------------------------------------------------------------------------------------

const FIXTURE_NAMES = ['crossing', 'junction-y', 'helix', 'linked-rings', 'knot'] as const;
type FixtureName = (typeof FIXTURE_NAMES)[number];

/** Each shape changes C's row count and sparsity, hence the pivot sequence of the factorization. */
type ConstraintKind = 'empty' | 'bary' | 'bary+length' | 'bary+point' | 'bary+edgeLengths';

const FACTOR_MODES: readonly FactorMode[] = ['ldlt', 'lu'];
const RHS_KINDS = ['gradient', 'projection', 'mixed'] as const;
type RhsKind = (typeof RHS_KINDS)[number];
const ROUTES = ['reassemble', 'frozen'] as const;
type Route = (typeof ROUTES)[number];

const SOLVER_CASES: readonly { fixture: FixtureName; constraints: ConstraintKind }[] = [
    { fixture: 'crossing', constraints: 'empty' },
    { fixture: 'crossing', constraints: 'bary' },
    { fixture: 'crossing', constraints: 'bary+length' },
    { fixture: 'crossing', constraints: 'bary+point' },
    { fixture: 'crossing', constraints: 'bary+edgeLengths' },
    { fixture: 'junction-y', constraints: 'bary' },
    { fixture: 'junction-y', constraints: 'bary+edgeLengths' },
    { fixture: 'helix', constraints: 'bary' },
    { fixture: 'linked-rings', constraints: 'bary+length' },
    { fixture: 'linked-rings', constraints: 'bary+point' },
    { fixture: 'knot', constraints: 'bary+length' },
];

type ProjMode = 'default' | 'frozen'; // `default` ⇒ `projectionMode` OMITTED (⇒ reassemble route)
type FactorOpt = 'default' | 'lu'; // `default` ⇒ `factorMode` OMITTED (⇒ solveSaddleFromA's 'ldlt')

const STEP_CASES: readonly {
    fixture: FixtureName;
    constraints: ConstraintKind;
    proj: ProjMode;
    factor: FactorOpt;
}[] = [
    ...(['default', 'frozen'] as const).flatMap((proj) =>
        (['default', 'lu'] as const).flatMap((factor) =>
            (
                [
                    ['crossing', 'bary+length'],
                    ['crossing', 'bary+edgeLengths'],
                    ['crossing', 'bary+point'],
                    ['junction-y', 'bary+edgeLengths'],
                    ['linked-rings', 'bary+length'],
                ] as const
            ).map(([fixture, constraints]) => ({ fixture, constraints, proj, factor })),
        ),
    ),
    ...(['default', 'frozen'] as const).flatMap((proj) =>
        (
            [
                ['crossing', 'empty'],
                ['junction-y', 'bary'],
                ['helix', 'bary'],
                ['linked-rings', 'bary+point'],
                ['knot', 'bary+length'],
            ] as const
        ).map(([fixture, constraints]) => ({
            fixture,
            constraints,
            proj,
            factor: 'default' as const,
        })),
    ),
];
const CHAIN_STEPS = 3;

const solverId = (c: { fixture: string; constraints: string }) => `${c.fixture}/${c.constraints}`;
const stepId = (c: (typeof STEP_CASES)[number]) =>
    `${c.fixture}/${c.constraints}/proj=${c.proj}/factorMode=${c.factor}`;

// ---------------------------------------------------------------------------------------------
// Bit-exact codec. JSON cannot carry -0 (JSON.stringify(-0) === '0'), and `Object.is(0, -0)` is
// false, so a bare-number encoding would silently pin the WRONG zero. -0 is escaped as '-0'.
// Every other finite double round-trips exactly through ECMAScript's shortest-repr Number→string.
// NaN/±Infinity are refused at capture: a healthy dense fixture has none, and a NaN would make
// `Object.is` pass for a solve that had silently degenerated.
// ---------------------------------------------------------------------------------------------

type Enc = number | '-0';

function enc(x: number): Enc {
    if (!Number.isFinite(x)) {
        throw new Error(`denseSaddleBitGolden: refusing to pin non-finite value ${x}`);
    }
    return Object.is(x, -0) ? '-0' : x;
}
const dec = (x: Enc): number => (x === '-0' ? -0 : x);
const encVec = (xs: ArrayLike<number>): Enc[] => Array.from(xs, (x) => enc(x));
const decVec = (xs: Enc[]): number[] => xs.map(dec);

const f64 = new DataView(new ArrayBuffer(8));
function hex(x: number): string {
    f64.setFloat64(0, x);
    return f64.getBigUint64(0).toString(16).padStart(16, '0');
}

/** Appends one line per bitwise mismatch (`Object.is`) — never a tolerance. */
function diffVec(diffs: string[], label: string, got: ArrayLike<number>, want: Enc[]): void {
    if (got.length !== want.length) {
        diffs.push(`${label}: length ${got.length} != pinned ${want.length}`);
        return;
    }
    for (let i = 0; i < want.length; i++) {
        const w = dec(want[i]);
        if (!Object.is(got[i], w)) {
            diffs.push(
                `${label}[${i}]: got ${got[i]} (0x${hex(got[i])}) != pinned ${w} (0x${hex(w)})`,
            );
        }
    }
}
const diffNum = (diffs: string[], label: string, got: number, want: Enc) =>
    diffVec(diffs, label, [got], [want]);

function expectNoDiffs(diffs: string[]): void {
    const shown =
        diffs.length > 8 ? [...diffs.slice(0, 8), `… and ${diffs.length - 8} more`] : diffs;
    expect(shown).toEqual([]);
}

// ---------------------------------------------------------------------------------------------
// Fixture schema
// ---------------------------------------------------------------------------------------------

interface SolveEnc {
    x: Enc[];
    lambda: Enc[];
    residual: Enc;
}
interface ProjEnc {
    vertices: Enc[];
    ok: boolean;
    iterations: number;
    phiNorm: Enc;
}
interface StepEnc {
    accepted: boolean;
    converged: boolean;
    energy: Enc;
    tau: Enc;
    residual: Enc;
    gradientL2Norm: Enc;
    projectionIterations: number | null;
    reason: string | null;
    vertices: Enc[];
    descentField: Enc[] | null;
}
interface FixtureData {
    vertices: Enc[][];
    edges: Edge[];
    alpha: number;
    beta: number;
    epsilon: number;
    /** The iterate the projection surfaces start from (seeded perturbation of `vertices`). */
    perturbed: Enc[][];
    /** `assembleAFlat(vertices)` — the n×n single-component Ā that `solveSaddleFromA` expands. */
    a: Enc[];
}
interface SolverCaseData {
    id: string;
    fixture: FixtureName;
    constraints: ConstraintKind;
    C: Enc[][];
    dE: Enc[];
    /** −Φ at the perturbed iterate — the projection solve's rhsBottom. */
    negPhi: Enc[];
    solve: Record<FactorMode, Record<RhsKind, SolveEnc>>;
    projection: Record<FactorMode, Record<Route, ProjEnc>>;
}
interface StepCaseData {
    id: string;
    steps: StepEnc[];
}
interface Provenance {
    capturedCommit: string;
    /** `git diff --numstat main...HEAD -- <seam files>` + the uncommitted equivalent. '' ⇒ untouched. */
    seamDiffVsMain: string;
    seamFileSha256: Record<string, string>;
    runtime: string;
    /** Append-only. Entry 0 is the initial capture. Every later entry is a regeneration. */
    regenerations: { reason: string; ruling: string | null; commit: string }[];
}
interface Golden {
    provenance: Provenance;
    fixtures: Record<FixtureName, FixtureData>;
    solverCases: SolverCaseData[];
    stepCases: StepCaseData[];
}

const FIXTURE_URL = new URL('./denseSaddleBitGolden.json', import.meta.url);
let cached: Golden | undefined;
function golden(): Golden {
    // Load at runtime (avoids needing resolveJsonModule in tsconfig; test/** is typechecked).
    cached ??= JSON.parse(readFileSync(FIXTURE_URL, 'utf8')) as Golden;
    return cached;
}

// ---------------------------------------------------------------------------------------------
// Shared builders — used by capture AND verification
// ---------------------------------------------------------------------------------------------

/** Arbitrary but fixed: a non-axis-aligned offset so all three pin rows carry a nonzero Φ. */
const PIN_VERTEX = 2;
const PIN_OFFSET: Vec3 = [0.05, -0.04, 0.03];

/**
 * Builds the set with its targets frozen at `vertices`, the way a run does once at start
 * (spec 2026-07-03-sobolev-constraints-design.md §3.5).
 */
function buildSet(kind: ConstraintKind, vertices: Vec3[], edges: Edge[]): ConstraintSet {
    const bary = barycenterBlock(barycenterTarget(vertices, edges));
    switch (kind) {
        case 'empty':
            return [];
        case 'bary':
            return [bary];
        case 'bary+length':
            return [bary, totalLengthBlock(totalLength(vertices, edges))];
        case 'bary+edgeLengths':
            return [bary, edgeLengthsBlock(edgeLengths(vertices, edges))];
        case 'bary+point': {
            const p = vertices[PIN_VERTEX];
            return [
                bary,
                pointBlock(PIN_VERTEX, [
                    p[0] + PIN_OFFSET[0],
                    p[1] + PIN_OFFSET[1],
                    p[2] + PIN_OFFSET[2],
                ]),
            ];
        }
    }
}

const zeros = (len: number): number[] => new Array<number>(len).fill(0);

/** The three call shapes: the gradient solve, the projection solve, and the general one. */
function rhsFor(kind: RhsKind, n: number, dE: number[], negPhi: number[]) {
    switch (kind) {
        case 'gradient': // solveSaddleFromA(A, n, C, flatten(dE), undefined, mode) — rhsBottom omitted
            return { top: dE, bottom: undefined };
        case 'projection': // lineSearch's projection solve: (zeros(3n), −Φ)
            return { top: zeros(3 * n), bottom: negPhi };
        case 'mixed':
            return { top: dE, bottom: negPhi };
    }
}

/**
 * A dense `PreparedSaddle` whose factorization is the SAME one `solveSaddleFromA` would build, so a
 * step run through `opts.saddle` must land on the SAME bits as one without. (The shipping producer
 * returns a `kind: 'external'` factorization; that path is deliberately not pinned here.)
 * The row-8 `kind` discriminant lands on THIS literal (and on `runSolves()`'s `op`) — see the file header.
 */
function densePrepared(
    vertices: Vec3[],
    edges: Edge[],
    dp: number[][],
    f: { alpha: number; beta: number; epsilon: number },
    mode: FactorMode | undefined,
): PreparedSaddle {
    const n = vertices.length;
    const A = assembleAFlat(vertices, edges, dp, f.alpha, f.beta, f.epsilon);
    return {
        kind: 'dense',
        A,
        factor: (C) => solveSaddleFromA(A, n, C, zeros(3 * n), undefined, mode).fac,
    };
}

interface SolveRaw {
    x: number[];
    lambda: number[];
    residual: number;
}
const encSolve = (s: SolveRaw): SolveEnc => ({
    x: encVec(s.x),
    lambda: encVec(s.lambda),
    residual: enc(s.residual),
});
function diffSolve(diffs: string[], label: string, got: SolveRaw, want: SolveEnc): void {
    diffVec(diffs, `${label}.x`, got.x, want.x);
    diffVec(diffs, `${label}.lambda`, got.lambda, want.lambda);
    diffNum(diffs, `${label}.residual`, got.residual, want.residual);
}

interface SolverInputs {
    a: Float64Array;
    n: number;
    C: number[][];
    dE: number[];
    negPhi: number[];
}

/** T1 body: FromA for each rhs shape, then the frozen solver reusing the gradient solve's `fac`. */
function runSolves(inp: SolverInputs, mode: FactorMode) {
    const fromA = {} as Record<RhsKind, SolveRaw>;
    const frozen = {} as Record<RhsKind, SolveRaw>;
    let op: FrozenSaddleOperator | undefined;
    for (const kind of RHS_KINDS) {
        const { top, bottom } = rhsFor(kind, inp.n, inp.dE, inp.negPhi);
        const r = solveSaddleFromA(inp.a, inp.n, inp.C, top, bottom, mode);
        fromA[kind] = { x: r.x, lambda: r.lambda, residual: r.residual };
        if (kind === 'gradient') op = { kind: 'dense', a: inp.a, n: inp.n, C: inp.C, fac: r.fac };
    }
    for (const kind of RHS_KINDS) {
        const { top, bottom } = rhsFor(kind, inp.n, inp.dE, inp.negPhi);
        frozen[kind] = solveSaddleFrozen(op as FrozenSaddleOperator, top, bottom);
    }
    return { fromA, frozen, op: op as FrozenSaddleOperator };
}

interface ProjRaw {
    vertices: number[];
    ok: boolean;
    iterations: number;
    phiNorm: number;
}
/** T2 body: the projection called directly from the perturbed iterate. */
function runProjection(
    route: Route,
    mode: FactorMode,
    fx: {
        vertices: Vec3[];
        perturbed: Vec3[];
        edges: Edge[];
        alpha: number;
        beta: number;
        epsilon: number;
    },
    set: ConstraintSet,
    frozen: FrozenSaddleOperator,
): ProjRaw {
    const dp = calculateDisjointPairs(fx.edges);
    const r = projectOntoConstraintSet(
        fx.perturbed,
        fx.edges,
        dp,
        fx.alpha,
        fx.beta,
        fx.epsilon,
        set,
        route === 'frozen' ? { factorMode: mode, frozen } : { factorMode: mode },
    );
    return {
        vertices: flatten(r.vertices),
        ok: r.ok,
        iterations: r.iterations,
        phiNorm: r.phiNorm,
    };
}

interface StepRaw {
    accepted: boolean;
    converged: boolean;
    energy: number;
    tau: number;
    residual: number;
    gradientL2Norm: number;
    projectionIterations: number | null;
    reason: string | null;
    vertices: number[];
    descentField: number[] | null;
}
/** T3 body: a CHAIN_STEPS-step chain, each step fed the previous step's vertices. */
function runChain(
    c: (typeof STEP_CASES)[number],
    fx: { vertices: Vec3[]; edges: Edge[]; alpha: number; beta: number; epsilon: number },
    withSaddle: boolean,
): StepRaw[] {
    const dp = calculateDisjointPairs(fx.edges);
    const set = buildSet(c.constraints, fx.vertices, fx.edges);
    const out: StepRaw[] = [];
    let v = fx.vertices;
    for (let s = 0; s < CHAIN_STEPS; s++) {
        // Options omitted (not `undefined`) for `default`, so the object is the pre-option shape.
        const r = sobolevStepSet(v, fx.edges, dp, set, {
            mode: 'analytical',
            alpha: fx.alpha,
            beta: fx.beta,
            epsilon: fx.epsilon,
            collectField: true,
            ...(c.proj === 'frozen' ? { projectionMode: 'frozen' as const } : {}),
            ...(c.factor === 'lu' ? { factorMode: 'lu' as const } : {}),
            ...(withSaddle
                ? {
                      saddle: densePrepared(
                          v,
                          fx.edges,
                          dp,
                          fx,
                          c.factor === 'lu' ? 'lu' : undefined,
                      ),
                  }
                : {}),
        });
        out.push({
            accepted: r.accepted,
            converged: r.converged,
            energy: r.energy,
            tau: r.stats.tau,
            residual: r.stats.residual,
            gradientL2Norm: r.stats.gradientL2Norm,
            projectionIterations: r.stats.projectionIterations,
            reason: r.stats.reason ?? null,
            vertices: flatten(r.vertices),
            descentField: r.descentField ? flatten(r.descentField) : null,
        });
        v = r.vertices;
    }
    return out;
}
const encStep = (s: StepRaw): StepEnc => ({
    ...s,
    energy: enc(s.energy),
    tau: enc(s.tau),
    residual: enc(s.residual),
    gradientL2Norm: enc(s.gradientL2Norm),
    vertices: encVec(s.vertices),
    descentField: s.descentField ? encVec(s.descentField) : null,
});
function diffStep(diffs: string[], label: string, got: StepRaw, want: StepEnc): void {
    // Non-float fields are discrete: exact equality is the whole story for them.
    for (const k of ['accepted', 'converged', 'projectionIterations', 'reason'] as const) {
        if (got[k] !== want[k]) diffs.push(`${label}.${k}: got ${got[k]} != pinned ${want[k]}`);
    }
    diffNum(diffs, `${label}.energy`, got.energy, want.energy);
    diffNum(diffs, `${label}.tau`, got.tau, want.tau);
    diffNum(diffs, `${label}.residual`, got.residual, want.residual);
    diffNum(diffs, `${label}.gradientL2Norm`, got.gradientL2Norm, want.gradientL2Norm);
    diffVec(diffs, `${label}.vertices`, got.vertices, want.vertices);
    if (want.descentField === null || got.descentField === null) {
        if (got.descentField !== want.descentField)
            diffs.push(`${label}.descentField: null mismatch`);
    } else {
        diffVec(diffs, `${label}.descentField`, got.descentField, want.descentField);
    }
}

/** A fixture's runtime form: decoded vertices etc. Works from memory so capture never re-reads the file it replaces. */
function runtimeFixtureOf(fixtures: Record<FixtureName, FixtureData>, name: FixtureName) {
    const f = fixtures[name];
    return {
        vertices: f.vertices.map(decVec) as Vec3[],
        perturbed: f.perturbed.map(decVec) as Vec3[],
        edges: f.edges,
        alpha: f.alpha,
        beta: f.beta,
        epsilon: f.epsilon,
        a: Float64Array.from(decVec(f.a)),
    };
}
const runtimeFixture = (name: FixtureName) => runtimeFixtureOf(golden().fixtures, name);

// ---------------------------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------------------------

const REGEN_REASON = process.env.DENSE_SADDLE_GOLDEN_REGENERATE;
const regenerating = REGEN_REASON !== undefined;
const verify = regenerating ? test.skip : test;

verify(
    'fixture integrity: exactly the declared case matrix, provenance from an untouched seam',
    () => {
        const g = golden();
        expect(g.solverCases.map((c) => c.id)).toEqual(SOLVER_CASES.map(solverId));
        expect(g.stepCases.map((c) => c.id)).toEqual(STEP_CASES.map(stepId));
        expect(Object.keys(g.fixtures)).toEqual([...FIXTURE_NAMES]);
        // The initial capture is only valid from an unmodified dense path (spec §11.2, A24: "after
        // task 8's diff lands there is no uncontaminated path left to capture"). A later
        // regeneration from a modified seam must carry a ruling — enforced at capture time.
        expect(g.provenance.regenerations.length).toBeGreaterThanOrEqual(1);
        expect(g.provenance.regenerations[0].ruling).toBeNull();
        expect(g.provenance.capturedCommit).toMatch(/^[0-9a-f]{40}$/);
        if (g.provenance.regenerations.length === 1) expect(g.provenance.seamDiffVsMain).toBe('');
    },
);

verify('fixture coverage: the golden actually reaches every path it claims to pin', () => {
    const g = golden();
    // A gate that cannot be tripped reads exactly like one that passed. Each predicate below names
    // a value that would mean "this surface is not exercised" — if the capture ever produces one,
    // the fixture is INVALID rather than green.
    const ks = new Set(g.solverCases.map((c) => c.C.length));
    for (const k of [0, 3, 4, 6, 9, 15]) expect(ks.has(k)).toBe(true);
    const chains = g.stepCases.map((c, i) => ({ c: STEP_CASES[i], steps: c.steps }));
    const itersOf = (proj: ProjMode) =>
        chains
            .filter((x) => x.c.proj === proj)
            .flatMap((x) => x.steps.map((s) => s.projectionIterations ?? 0));
    // The projection solve only fires when iterations ≥ 1; zero everywhere would pin only the
    // gradient solve and leave the reassemble/frozen projection solvers un-pinned.
    expect(Math.max(...itersOf('default'))).toBeGreaterThanOrEqual(1);
    expect(Math.max(...itersOf('frozen'))).toBeGreaterThanOrEqual(1);
    // Backtracking exercised on both routes (τ < 1 means a rejected trial's projection ran too).
    for (const proj of ['default', 'frozen'] as const) {
        const taus = chains
            .filter((x) => x.c.proj === proj)
            .flatMap((x) => x.steps.map((s) => dec(s.tau)));
        expect(Math.min(...taus)).toBeLessThan(1);
    }
    // Every step in every chain must have actually taken a step or the vertices pin nothing new.
    for (const x of chains) {
        for (const s of x.steps) expect(s.accepted && !s.converged).toBe(true);
    }
    // Both factorizations are pinned on the direct projection surface with ≥2 correction solves
    // somewhere, so a projection iterate beyond the first (fresh Φ against the SAME or a
    // reassembled operator) is bit-pinned, not just the first correction.
    for (const mode of FACTOR_MODES) {
        for (const route of ROUTES) {
            const its = g.solverCases
                .filter((c) => c.C.length > 0)
                .map((c) => c.projection[mode][route].iterations);
            expect(Math.max(...its)).toBeGreaterThanOrEqual(2);
        }
    }
});

for (const sc of SOLVER_CASES) {
    verify(
        `T1/T1b/T2 ${solverId(sc)} — solveSaddleFromA, solveSaddleFrozen, gradient.ts, projectOntoConstraintSet`,
        () => {
            const g = golden();
            const data = g.solverCases.find((c) => c.id === solverId(sc)) as SolverCaseData;
            const fx = runtimeFixture(sc.fixture);
            const n = fx.vertices.length;
            const C = data.C.map(decVec);
            const dE = decVec(data.dE);
            const negPhi = decVec(data.negPhi);
            const inp: SolverInputs = { a: fx.a, n, C, dE, negPhi };
            const aBefore = Array.from(fx.a);
            const CBefore = C.map((r) => r.slice());
            const set = buildSet(sc.constraints, fx.vertices, fx.edges);
            const dp = calculateDisjointPairs(fx.edges);
            const diffs: string[] = [];

            for (const mode of FACTOR_MODES) {
                const { fromA, frozen, op } = runSolves(inp, mode);
                for (const kind of RHS_KINDS) {
                    const pinned = data.solve[mode][kind];
                    diffSolve(diffs, `T1 fromA/${mode}/${kind}`, fromA[kind], pinned);
                    diffSolve(diffs, `T1 frozen/${mode}/${kind}`, frozen[kind], pinned);
                    // The dense-vs-frozen equivalence the row-8 seam change could break while both routes still look
                    // individually plausible: same `fac`, same rhs ⇒ same bits.
                    diffSolve(
                        diffs,
                        `T1 frozen≡fromA/${mode}/${kind}`,
                        frozen[kind],
                        encSolve(fromA[kind]),
                    );
                }

                // T1b — gradient.ts entry points land on the T1 gradient solve's stored bits.
                const gradPinned = data.solve[mode].gradient;
                const dEvec = unflatten(dE);
                const viaFrozenFn = solveConstrainedGradientSetFrozen(
                    fx.vertices,
                    fx.edges,
                    dp,
                    fx.alpha,
                    fx.beta,
                    fx.epsilon,
                    dEvec,
                    set,
                    mode,
                );
                const asSolve = (r: {
                    gTilde: Vec3[];
                    lambda: number[];
                    residual: number;
                }): SolveRaw => ({
                    x: flatten(r.gTilde),
                    lambda: r.lambda,
                    residual: r.residual,
                });
                diffSolve(diffs, `T1b frozenFn/${mode}`, asSolve(viaFrozenFn), gradPinned);
                // Narrowing, not a cast (row 8 made FrozenSaddleOperator a `kind`-tagged union and
                // `a` lives only on the dense arm). It is also an assertion in its own right: this
                // call passes no `saddle`, so a dense frozen operator is the ONLY correct outcome.
                if (viaFrozenFn.frozen.kind !== 'dense')
                    throw new Error(`T1b frozenFn/${mode}: expected a dense frozen operator`);
                // The returned operator is what the frozen projection reuses: pin its `a`/`C`/`n` too.
                diffVec(diffs, `T1b frozenFn/${mode}.op.a`, viaFrozenFn.frozen.a, encVec(fx.a));
                diffVec(
                    diffs,
                    `T1b frozenFn/${mode}.op.C`,
                    viaFrozenFn.frozen.C.flat(),
                    encVec(C.flat()),
                );
                if (viaFrozenFn.frozen.n !== n)
                    diffs.push(`T1b frozenFn/${mode}.op.n: ${viaFrozenFn.frozen.n} != ${n}`);
                const viaPrepared = solveConstrainedGradientSetFrozen(
                    fx.vertices,
                    fx.edges,
                    dp,
                    fx.alpha,
                    fx.beta,
                    fx.epsilon,
                    dEvec,
                    set,
                    mode,
                    densePrepared(fx.vertices, fx.edges, dp, fx, mode),
                );
                diffSolve(diffs, `T1b prepared/${mode}`, asSolve(viaPrepared), gradPinned);
                if (mode === 'ldlt') {
                    // The public non-frozen entry points run at solveSaddleFromA's default mode.
                    diffSolve(
                        diffs,
                        'T1b gradientSet/default',
                        asSolve(
                            solveConstrainedGradientSet(
                                fx.vertices,
                                fx.edges,
                                dp,
                                fx.alpha,
                                fx.beta,
                                fx.epsilon,
                                dEvec,
                                set,
                            ),
                        ),
                        gradPinned,
                    );
                    if (sc.constraints === 'bary') {
                        diffSolve(
                            diffs,
                            'T1b gradient(x0)/default',
                            asSolve(
                                solveConstrainedGradient(
                                    fx.vertices,
                                    fx.edges,
                                    dp,
                                    fx.alpha,
                                    fx.beta,
                                    fx.epsilon,
                                    dEvec,
                                    barycenterTarget(fx.vertices, fx.edges),
                                ),
                            ),
                            gradPinned,
                        );
                    }
                }

                // T2 — the projection, called directly.
                for (const route of ROUTES) {
                    const p = data.projection[mode][route];
                    const got = runProjection(route, mode, fx, set, op);
                    diffVec(diffs, `T2 ${route}/${mode}.vertices`, got.vertices, p.vertices);
                    diffNum(diffs, `T2 ${route}/${mode}.phiNorm`, got.phiNorm, p.phiNorm);
                    if (got.ok !== p.ok) diffs.push(`T2 ${route}/${mode}.ok: ${got.ok} != ${p.ok}`);
                    if (got.iterations !== p.iterations)
                        diffs.push(
                            `T2 ${route}/${mode}.iterations: ${got.iterations} != ${p.iterations}`,
                        );
                }
            }

            // Contract the frozen operator depends on: the solvers never mutate `a` or `C`.
            diffVec(diffs, 'a after solves', fx.a, encVec(aBefore));
            diffVec(diffs, 'C after solves', C.flat(), encVec(CBefore.flat()));
            expectNoDiffs(diffs);
        },
    );
}

for (const [i, sc] of STEP_CASES.entries()) {
    verify(
        `T3 ${stepId(sc)} — sobolevStepSet ×${CHAIN_STEPS}, with and without a dense PreparedSaddle`,
        () => {
            const data = golden().stepCases[i];
            expect(data.id).toBe(stepId(sc));
            expect(data.steps.length).toBe(CHAIN_STEPS);
            const fx = runtimeFixture(sc.fixture);
            const diffs: string[] = [];
            for (const withSaddle of [false, true]) {
                const got = runChain(sc, fx, withSaddle);
                got.forEach((s, k) => {
                    diffStep(
                        diffs,
                        `T3 ${withSaddle ? 'saddle' : 'plain'} step${k}`,
                        s,
                        data.steps[k],
                    );
                });
            }
            expectNoDiffs(diffs);
        },
    );
}

test('codec: -0, subnormals and extremes survive the JSON round trip bit-for-bit', () => {
    const probes = [
        -0,
        0,
        5e-324,
        -5e-324,
        Number.MAX_VALUE,
        Number.MIN_VALUE,
        0.1 + 0.2,
        1 / 3,
        -1e-7,
        2 ** -1074,
        1e21,
    ];
    const back = decVec(JSON.parse(JSON.stringify(encVec(probes))) as Enc[]);
    const diffs: string[] = [];
    diffVec(diffs, 'probe', back, encVec(probes));
    expectNoDiffs(diffs);
    expect(Object.is(back[0], -0)).toBe(true);
    expect(Object.is(back[1], 0)).toBe(true);
    expect(() => enc(Number.NaN)).toThrow(/non-finite/);
});

// ---------------------------------------------------------------------------------------------
// Capture (guarded). See the file header: FORBIDDEN as a way to make a red test green.
// ---------------------------------------------------------------------------------------------

const SEAM_FILES = [
    'src/core/sobolev/linsolve.ts',
    'src/core/sobolev/gradient.ts',
    'src/core/sobolev/lineSearch.ts',
    'src/core/optimizer.ts',
    'src/gpu/saddle.ts',
] as const;
const REPO_ROOT = fileURLToPath(new URL('../../', import.meta.url));
const git = (...args: string[]): string =>
    execFileSync('git', args, { cwd: REPO_ROOT, encoding: 'utf8' }).trim();

/** Peak-to-peak size of the seeded perturbation that the direct-projection surface starts from. */
const PERTURB_AMPLITUDE = 0.2;

/** Seeded PRNG already used by test/sobolev/ldlt.test.ts — integer arithmetic, platform-stable. */
function mulberry32(seed: number): () => number {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function capture(reason: string, ruling: string | null): Golden {
    const seamDiffVsMain = [
        git('diff', '--numstat', 'main...HEAD', '--', ...SEAM_FILES),
        git('diff', '--numstat', 'HEAD', '--', ...SEAM_FILES),
    ]
        .filter((s) => s !== '')
        .join('\n');
    if (seamDiffVsMain !== '' && ruling === null) {
        throw new Error(
            `REFUSING to capture: seam files differ from main:\n${seamDiffVsMain}\n` +
                'The reference would bake in that change and certify the wrong baseline (spec §11.2, A24). ' +
                'If a numerical change is intentional, write the ruling first and pass its path in DENSE_SADDLE_GOLDEN_RULING.',
        );
    }
    const commit = git('rev-parse', 'HEAD');
    const prior = existsSync(fileURLToPath(FIXTURE_URL))
        ? (JSON.parse(readFileSync(FIXTURE_URL, 'utf8')) as Golden).provenance.regenerations
        : [];

    const fixtures = {} as Record<FixtureName, FixtureData>;
    FIXTURE_NAMES.forEach((name, idx) => {
        const src = JSON.parse(
            readFileSync(new URL(`../../oracle/fixtures/${name}.json`, import.meta.url), 'utf8'),
        ) as { vertices: Vec3[]; edges: Edge[]; alpha: number; beta: number; epsilon: number };
        const rng = mulberry32(0xa24 + idx);
        // Amplitude is a MEASURED choice (0.02 → every projection converged in 1 correction solve,
        // which pins only the first iterate; 0.2 → 2–4 corrections on both routes, all converging).
        // The coverage test fails the fixture if it ever stops reaching a second correction.
        const perturbed = src.vertices.map(
            (p) => p.map((c) => c + PERTURB_AMPLITUDE * (rng() - 0.5)) as Vec3,
        );
        const dp = calculateDisjointPairs(src.edges);
        fixtures[name] = {
            vertices: src.vertices.map(encVec),
            edges: src.edges,
            alpha: src.alpha,
            beta: src.beta,
            epsilon: src.epsilon,
            perturbed: perturbed.map(encVec),
            a: encVec(assembleAFlat(src.vertices, src.edges, dp, src.alpha, src.beta, src.epsilon)),
        };
    });

    const solverCases: SolverCaseData[] = SOLVER_CASES.map((sc) => {
        const fx = runtimeFixtureOf(fixtures, sc.fixture);
        const n = fx.vertices.length;
        const set = buildSet(sc.constraints, fx.vertices, fx.edges);
        const dp = calculateDisjointPairs(fx.edges);
        const C = evaluateConstraintSet(set, fx.vertices, fx.edges).C;
        const dE = flatten(
            gradientAnalytical(fx.vertices, fx.edges, dp, fx.alpha, fx.beta, fx.epsilon),
        );
        // `.map((v) => -v)` exactly as projectOntoConstraintSet builds its rhsBottom (−0 included).
        const negPhi = evaluateConstraintSet(set, fx.perturbed, fx.edges).phi.map((v) => -v);
        const inp: SolverInputs = { a: fx.a, n, C, dE, negPhi };
        const solve = {} as SolverCaseData['solve'];
        const projection = {} as SolverCaseData['projection'];
        for (const mode of FACTOR_MODES) {
            const { fromA, op } = runSolves(inp, mode);
            solve[mode] = {} as Record<RhsKind, SolveEnc>;
            for (const kind of RHS_KINDS) solve[mode][kind] = encSolve(fromA[kind]);
            projection[mode] = {} as Record<Route, ProjEnc>;
            for (const route of ROUTES) {
                const p = runProjection(route, mode, fx, set, op);
                projection[mode][route] = {
                    vertices: encVec(p.vertices),
                    ok: p.ok,
                    iterations: p.iterations,
                    phiNorm: enc(p.phiNorm),
                };
            }
        }
        return {
            id: solverId(sc),
            fixture: sc.fixture,
            constraints: sc.constraints,
            C: C.map(encVec),
            dE: encVec(dE),
            negPhi: encVec(negPhi),
            solve,
            projection,
        };
    });

    const stepCases: StepCaseData[] = STEP_CASES.map((c) => ({
        id: stepId(c),
        steps: runChain(c, runtimeFixtureOf(fixtures, c.fixture), false).map(encStep),
    }));

    return {
        provenance: {
            capturedCommit: commit,
            seamDiffVsMain,
            seamFileSha256: Object.fromEntries(
                SEAM_FILES.map((f) => [
                    f,
                    createHash('sha256')
                        .update(readFileSync(REPO_ROOT + f))
                        .digest('hex'),
                ]),
            ),
            runtime: `bun ${Bun.version} ${process.platform}-${process.arch}`,
            regenerations: [...prior, { reason, ruling, commit }],
        },
        fixtures,
        solverCases,
        stepCases,
    };
}

// Registered ONLY when the guard env var is set, so a normal run carries no permanently-skipped test.
if (regenerating)
    test('REGENERATE the dense-saddle bit golden (guarded; always ends RED)', () => {
        const reason = REGEN_REASON as string;
        const ruling = process.env.DENSE_SADDLE_GOLDEN_RULING ?? null;
        if (reason.trim().length < 20) {
            throw new Error(
                'DENSE_SADDLE_GOLDEN_REGENERATE must state WHY, in at least 20 characters.',
            );
        }
        if (ruling !== null && !existsSync(REPO_ROOT + ruling)) {
            throw new Error(
                `DENSE_SADDLE_GOLDEN_RULING '${ruling}' does not exist relative to the repo root.`,
            );
        }
        const next = capture(reason, ruling);
        const path = fileURLToPath(FIXTURE_URL);
        writeFileSync(path, `${JSON.stringify(next, null, 2)}\n`);
        // Keep the committed file biome-stable, then prove formatting changed no value.
        execFileSync(`${REPO_ROOT}node_modules/.bin/biome`, ['format', '--write', path], {
            cwd: REPO_ROOT,
        });
        const reread = JSON.parse(readFileSync(path, 'utf8')) as Golden;
        if (JSON.stringify(reread) !== JSON.stringify(next)) {
            throw new Error('regenerated fixture does not round-trip bit-for-bit after formatting');
        }
        console.error(
            '\n!!! DENSE SADDLE BIT GOLDEN REGENERATED !!!\n' +
                'Regenerating to make a failing test pass is FORBIDDEN (spec §11.2, A24). This run is\n' +
                'DELIBERATELY RED so it cannot be mistaken for a pass. Re-run WITHOUT the env vars; the\n' +
                'fixture diff belongs in the same commit as the written ruling that justifies it.\n',
        );
        throw new Error('golden regenerated — re-run without DENSE_SADDLE_GOLDEN_REGENERATE');
    });
