/**
 * Phase 2H pre-registered kill-gate calibration harness — H1-a, H1-b, H2, H3, H4a, H4b′, H4c, H5,
 * and the R1 reporting obligation — emitting spec §7.12's schema (as amended by `[A46 item 5]`) to
 * `bench/results/<date>-gpu-phase2h-gate-calibration.json`.
 *
 * **Every gate reports its VALUE, not only its verdict** (spec §11.2 row 11), and every bar is
 * recomputed here from the run's OWN counts — never copied from the spec. `[A46 item 5]` is the
 * amendment that forced this: §7.12's `kappaNear: 93.3` / `h1bBar: 5.78e-5` anchors are
 * `torusKnot` bbox-normalised while its own `fixtures.family` says `trefoil`, whose bar is
 * **2.4674× tighter** (MEASURED, `scratch-11/confirm-960.json`). An implementer copying the schema
 * ships the wrong bar, so this file copies no number that a count can produce.
 *
 * **Why TypeScript and not Python.** Row 11's stated reason ("a `.py` harness cannot reach the
 * task 3–8 instruments") is FALSE per gate and `[A46 item 4]` corrects it: `lineSearchStepSet` has
 * a Python counterpart (`oracle/tpe_constraints_oracle.py:554`) and H5's dense-eigenvalue
 * instrument exists on both sides. The surviving reason is narrower and sufficient: **the
 * hierarchical operator and the GMRES `η` hook have no Python counterpart at all**, and H5 is
 * ALREADY a shipped TS gate (`precond.test.ts:383-447`), so re-measuring it in Python would
 * regress its provenance to the lab `pinv` operator that `[A32]` and `[A38 item 9(a)]` withdrew.
 * **H6 alone stays Python** (generalized symmetric-definite pencil; TS has no such eigensolver) —
 * its measured values are transcribed here from `task-0c-report.md` and labelled as transcribed.
 *
 * **`clock: null`, and no wall-clock figure anywhere.** Issue #63 and the task brief: this box
 * shows 6–9× ONE-SIDED degradation of a timing ratio under subagent load, so a published duration
 * would not be a property of the code. Cost is budgeted and reported in **apply counts**, which are
 * load-independent integers (`scratch-11/budget.json`).
 *
 * **Sharding, and why it is not a schedule change.** `--gates` and `--n` select a subset; each
 * invocation MERGES its part into the output file. Both exist because a run must complete in the
 * FOREGROUND (a backgrounded run makes the agent go idle and the campaign stops silently) inside a
 * 10-minute cap — `scratch-7b/h23.ts:74-80`'s precedent, created for the same reason. Sharding
 * changes NO pinned parameter. It is the substitute for the χ=0 probe sparsification: `η_exact` and
 * `η_bct` share one `schedule` and one `exactApplyA` by construction (`gmres.ts:344-382`, and
 * `probeAt` at `:688-716` computes both on every probe), so they cannot be scheduled independently
 * without changing an exported interface — utof/repulsive-test2#96's class of change, not row 11's.
 *
 * **Pre-registration.** The per-gate bar, value key, INVALID set and EXPECTED verdict were written
 * down before the first run that tests a bar, in
 * `.superpowers/sdd/2026-10-09-phase2h-bct-gmres-design/task-11-report.md` §S1.2. A result this
 * file reports that was not predicted there carries no falsification weight and says so in its
 * `limitations`.
 *
 * Usage (foreground only):
 *   bun bench/gpu/hCalibration.ts --gates=meta
 *   bun bench/gpu/hCalibration.ts --gates=h1
 *   bun bench/gpu/hCalibration.ts --gates=h2 --n=64,120,240
 *   bun bench/gpu/hCalibration.ts --gates=h3 --n=240,480
 *   bun bench/gpu/hCalibration.ts --gates=h4a --n=64,240
 *   bun bench/gpu/hCalibration.ts --gates=h4c --n=240
 *   bun bench/gpu/hCalibration.ts --gates=h5 --n=64,120,240,480,960
 *   bun bench/gpu/hCalibration.ts --gates=h6,r1
 *
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §7 (all gates), §7.12 (schema), §11.2 row 11
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md A46 (row 11's Stage 0: eight defects, two blocking)
 * @see .superpowers/sdd/2026-10-09-phase2h-bct-gmres-design/task-11-report.md §S1.2 (the pre-registration)
 * @issue utof/repulsive-test2#63
 */
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { cpus, loadavg, totalmem } from 'node:os';
import { trefoil } from '../../src/core/fixtures';
import { DEFAULTS, sobolevStepSet } from '../../src/core/optimizer';
import { barycenterBlock, type ConstraintSet } from '../../src/core/sobolev/constraintSet';
import { barycenterTarget } from '../../src/core/sobolev/constraints';
import {
    applyA,
    applyAScalar,
    type BlockSource,
    type HierarchicalA,
    prepareA,
} from '../../src/core/sobolev/hierarchical/apply';
import { buildBct, DEFAULT_THETA, thetaLadder } from '../../src/core/sobolev/hierarchical/bct';
import { buildBvh } from '../../src/core/sobolev/hierarchical/bvh';
import { curveExtent } from '../../src/core/sobolev/hierarchical/extent';
import {
    backwardError,
    DEFAULT_PROBE_SCHEDULE,
    defaultMaxit,
    type EtaClause,
    estimateNorm2,
    gmres,
    probeIterations,
    U_F32,
    U_F64,
} from '../../src/core/sobolev/hierarchical/gmres';
import { makeHierarchicalSaddle } from '../../src/core/sobolev/hierarchical/hierarchicalSaddle';
import { prepareLowOrder } from '../../src/core/sobolev/hierarchical/lowOrder';
import {
    type AccumulatorVariant,
    F16_NEAR_KNOCKOUT,
    F32,
    F64,
} from '../../src/core/sobolev/hierarchical/nearField';
import {
    type HierarchicalOperator,
    makeHierarchicalOperator,
} from '../../src/core/sobolev/hierarchical/operator';
import {
    DEFAULT_MASS_WEIGHT_HAT,
    makePreconditioner,
    type Preconditioner,
} from '../../src/core/sobolev/hierarchical/precond';
import { assembleAFlat } from '../../src/core/sobolev/innerProduct';
import { calculateDisjointPairs, calculateEnergy } from '../../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../../src/core/testConfigs';
// `bench/ → test/` import, PERMITTED by the controller's Stage-1 ruling 2 and deliberate.
// Why: `denseSpectrum.ts` IS the instrument H5's shipped gate is measured with
// (`precond.test.ts:383-447` calls exactly these functions), and `[A46 item 4]` rules that H5 is
// emitted from the SHIPPED TS instrument rather than re-measured. Duplicating `cholesky`/`spectrum`
// into `bench/` would publish H5 from a second implementation, which is the common-mode defect
// memory `common-mode-blindness` records: a bug both copies share passes both.
// Note: knip already reports every `bench/gpu/*` driver as an unused FILE at baseline (178 unused
// files, exit 1, verified 2026-10-12 before this file existed) — that finding predates row 11.
import {
    cholesky,
    cholSolve,
    densify,
    matmul,
    spectrum,
    transpose,
} from '../../test/sobolev/hierarchical/denseSpectrum';
import { type Curve, circle, torusKnot } from '../../test/sobolev/hierarchical/fixtures';

// ---------------------------------------------------------------------------------------------
// Pinned parameters. Every one is the SHIPPED value, referenced by name where a constant exists.
// ---------------------------------------------------------------------------------------------

const ALPHA = DEFAULTS.alpha;
const BETA = DEFAULTS.beta;
const EPS_A = DEFAULTS.epsilon;
/** `bvh.ts` `DEFAULT_LEAF_SIZE`, and the leaf size of every anchor in the corpus. */
const LEAF = 8;
/** H1's rhs seed — `apply.test.ts:181` `rhs(n, seed = 12345)`, the seed every `relErrRandom` used. */
const H1_RHS_SEED = 12345;
/** H1-b's lower bound: a decade under the MEASURED f32 apply noise (spec §7.2; `apply.test.ts:655`). */
const H1B_LOWER = 1e-7;
/**
 * f16 unit roundoff, `2**-11` — 10 stored mantissa bits plus the implicit one. Used ONLY to
 * reproduce §7.11 item 7's knock-out MODEL beside the measurement; the f16 path itself is
 * `F16_NEAR_KNOCKOUT`, never production.
 */
const U_F16 = 2 ** -11;
/** H1-a's slack: `apply.test.ts:604` asserts `d25 <= 1.001 * d5`. */
const H1A_SLACK = 1.001;
/** H2's bar, spec §7.3: `50 · u_f32`. Computed from {@link U_F32}, never written as 2.98e-6. */
const H2_BAR = 50 * U_F32;
/** H3's `τ_η` — `[A31]`, its THIRD and final re-pin (1e-8 → 1e-10 → 1e-12). */
const TAU_ETA = 1e-12;
/** H3's companion bar on `G = iters(2n)/iters(n)`, spec §7.4. */
const H3_G_BAR = 2.0;
/** H3's primary bar, spec §7.4 — NOT measurable at these rungs; see the gate's `limitations`. */
const H3_PRIMARY_BAR_ITERS = 64;
const H3_PRIMARY_BAR_N = 30000;
/** H4a's bar, spec §7.5: `min over 5 pinned directions of cos∠_A(x̃, x*) >= 0.9`. CHOSEN number. */
const H4A_BAR = 0.9;
/** H4a's stopping tolerance — §7.5's measured baseline column is at τ=1e-4; do not change it. */
const H4A_TAU = 1e-4;
/**
 * H4a's INVALID-1 replacement: the dense reference solve's OWN backward error must be `<= 10·u_f64`.
 * Why a replacement at all: `[A46 item 4]` establishes that §7.5's INVALID 1 ("`x*` not from an f64
 * dense solve") is PROVENANCE, not a function of outputs, so it cannot be "reachable in a test" as
 * row 11 requires. This is the controller's Stage-1 ruling 4 — it is a bound on the instrument, and
 * it is NOT in the spec yet (proposed wording in task-11-report.md §S1.4).
 */
const H4A_REF_ETA_BAR = 10 * U_F64;
/** H4c's bar, spec §7.7, conjunct 1. */
const H4C_RE_BAR = 0.95;
/** H4c's step count, spec §7.7. */
const H4C_K = 20;
/** H5 property 2's bar, `[A32]` superseding `[A22]`: `κ_excl <= 40`, trefoil-scoped. */
const H5_KAPPA_EXCL_BAR = 40;
/** H5 property 3's bar, spec §7.8: `κ_incl · u_f32 <= 1e-2`. RECORDED, not a licence for f32. */
const H5_P3_BAR = 1e-2;
/** Seeds 0–4: spec §7.5's "5 pinned directions" and §7.3's five-seed ladder. */
const SEEDS = [0, 1, 2, 3, 4];
/**
 * The two θ every per-θ gate is measured at. `[A46 item 7]` withdrew H6's RED as a θ signal and
 * left the production choice OPEN on #51's evidence, explicitly NOT row 11's to make — so row 11
 * gates at BOTH and emits `thetaProduction: null`.
 */
const THETAS_GATED = [0.5, 0.25];

// ---------------------------------------------------------------------------------------------
// Fixtures, WITH PROVENANCE. `[A39 item 3]` binds: no number may be quoted for "the trefoil"
// without naming which curve, because the two differ by 5.4716× in extent and `buildBvh` is not
// scale-equivariant (#69) — MEASURED 2.4674× on the H1-b bar at n=960 (schema gap G3).
// ---------------------------------------------------------------------------------------------

interface FixtureSpec {
    /** Module path + export name, so the curve is identifiable without this file. */
    curveId: string;
    build: (n: number) => Curve;
    bboxNormalised: boolean;
}

const FIXTURES: Record<string, FixtureSpec> = {
    // Lissajous, NATIVE extent 5.4716 — every shipped η/iteration ladder is this curve at this scale.
    trefoil: { curveId: 'src/core/fixtures.ts:trefoil', build: trefoil, bboxNormalised: false },
    // (2,3) torus knot, bbox-NORMALISED to extent 1 — every H1 anchor is this curve.
    torusKnot: {
        curveId: 'test/sobolev/hierarchical/fixtures.ts:torusKnot',
        build: torusKnot,
        bboxNormalised: true,
    },
    // The only fixture in the corpus with a MEASURED κ_near tie (`nearPairs` 21598 at both 0.5 and
    // 0.25) — H1-a's must-NOT-trip cell.
    circle: {
        curveId: 'test/sobolev/hierarchical/fixtures.ts:circle',
        build: circle,
        bboxNormalised: true,
    },
};

/** Fixture provenance as emitted into every gate row. */
function provenance(name: string, vertices: Vec3[], n: number) {
    const f = FIXTURES[name];
    return {
        family: name,
        curveId: f.curveId,
        n,
        extent: curveExtent(vertices),
        bboxNormalised: f.bboxNormalised,
        leafSize: LEAF,
        alpha: ALPHA,
        beta: BETA,
        epsilon: EPS_A,
    };
}

// ---------------------------------------------------------------------------------------------
// Deterministic vectors. THERE ARE TWO GENERATORS AND SWAPPING THEM IS A TRAP (report §D3):
// `seeded` (uniform) is what every `relErrRandom` H1 anchor was measured with; `gaussian`
// (standard normal) is what `e1_gmres.py:34-35`'s rhs ladder was. One generator for both makes
// one of the two anchor sets unreproducible.
// ---------------------------------------------------------------------------------------------

/** `scratch/curves.ts` `lcg` — deterministic, never `Math.random` (§7.12: deterministic seeds). */
function lcg(seed: number): () => number {
    let s = seed >>> 0;
    return () => {
        s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
        return s / 4294967296;
    };
}

/** Uniform `[-0.5, 0.5)` — `apply.test.ts:89-97`, the APPLY-direction generator (H1-a / H1-b). */
function seeded(n: number, seed: number): Float64Array {
    const rnd = lcg(seed);
    const u = new Float64Array(n);
    for (let i = 0; i < n; i++) u[i] = rnd() - 0.5;
    return u;
}

/**
 * Standard normal via Box-Muller over {@link lcg} — the RHS generator (H2 / H3 / H4a / H4b′),
 * matching `e1_gmres.py:34-35`'s `standard_normal`.
 *
 * RECORDED CAVEAT (7b's, and it stands): numpy's PCG64 draws cannot be reproduced in TypeScript,
 * so the DIRECTIONS differ from the lab's at the same seed label even where the distribution
 * matches. Comparable within this harness, not element-wise with the lab.
 */
function gaussian(dim: number, seed: number): Float64Array {
    const rnd = lcg(seed);
    const v = new Float64Array(dim);
    for (let i = 0; i < dim; i += 2) {
        const u1 = Math.max(rnd(), Number.MIN_VALUE);
        const u2 = rnd();
        const r = Math.sqrt(-2 * Math.log(u1));
        v[i] = r * Math.cos(2 * Math.PI * u2);
        if (i + 1 < dim) v[i + 1] = r * Math.sin(2 * Math.PI * u2);
    }
    return v;
}

const norm = (x: ArrayLike<number>): number => {
    let s = 0;
    for (let i = 0; i < x.length; i++) s += x[i] * x[i];
    return Math.sqrt(s);
};

const dot = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
};

/** `scratch/operators.ts:339` `relErr`: `‖a − b‖₂ / ‖a‖₂`, REFERENCE IN THE FIRST SLOT. */
function relErr(a: ArrayLike<number>, b: ArrayLike<number>): number {
    let d = 0;
    for (let i = 0; i < a.length; i++) d += (a[i] - b[i]) ** 2;
    return Math.sqrt(d) / norm(a);
}

const median = (xs: number[]): number => {
    const s = [...xs].sort((a, b) => a - b);
    const k = s.length;
    return k % 2 ? s[(k - 1) / 2] : 0.5 * (s[k / 2 - 1] + s[k / 2]);
};

/** Least-squares slope of `log y` on `log n` — the `n^p` fits of §7.3/§7.4/§7.8. */
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

// ---------------------------------------------------------------------------------------------
// Verdicts. FIVE values, not two (schema gap G8). §7.0:1219 mandates GREEN/RED/INVALID; `[A42]`
// ruling 5 adds VACUOUS ("a gate that cannot fail reads exactly like one that passed"), and
// `[A46 item 2]` adds NOT APPLICABLE for H1's adapter clause on a CPU arm. §7.12:2729's
// `"H4b": "GREEN"` example is WITHDRAWN by `[A46 item 1]`.
// ---------------------------------------------------------------------------------------------

type Verdict = 'GREEN' | 'RED' | 'INVALID' | 'VACUOUS' | 'NOT APPLICABLE' | 'NOT ESTABLISHED';

interface GateVerdict {
    verdict: Verdict;
    /** Mandatory: a verdict with no reason is the thing this harness exists to stop publishing. */
    reason: string;
    /** What §S1.2 predicted, so an unpredicted result is visibly unpredicted. */
    preRegistered: Verdict;
    matchesPreRegistration: boolean;
}

function verdictOf(v: Verdict, preRegistered: Verdict, reason: string): GateVerdict {
    return { verdict: v, reason, preRegistered, matchesPreRegistration: v === preRegistered };
}

// ---------------------------------------------------------------------------------------------
// The shipped stack, built per (fixture, n, θ). Adapted from `scratch-7b/h23.ts:155-245`, the
// configuration every H2/H3 figure on this branch was measured with.
// ---------------------------------------------------------------------------------------------

type Arm = 'f64' | 'f32';

interface Cell {
    fixture: string;
    n: number;
    theta: number;
    vertices: Vec3[];
    edges: Edge[];
    m: number;
    extent: number;
    massWeight: number;
    nFar: number;
    nNear: number;
    nearPairs: number;
    adjacencyVetoed: number;
    kappaNear: number;
    kappaFar: number;
    sigma64: number;
    sigma32: number;
    H64: HierarchicalA;
    op64: HierarchicalOperator;
    op32: HierarchicalOperator | null;
    P: Preconditioner;
    /** The χ=0 probe operator, f64 — `η_exact`'s matvec. */
    exactApplyA: (u: ArrayLike<number>, out: Float64Array) => void;
}

/**
 * `η_exact`'s χ=0 operator: EVERY block inadmissible, f64.
 *
 * Spec §7.3 says "streaming"; this uses `{kind:'lists', precomputeNear:true}` instead, and the
 * substitution is PROVED rather than assumed: `scratch-7b/exactpath.ts` measured the two
 * BIT-IDENTICAL (0 of 1440 entries non-identical by `Object.is`, maxRel exactly 0, at
 * n = 64/120/240/480 on two probe vectors) at 11.2–12.7× the speed. §7.1's streaming requirement
 * is about PRODUCTION memory (no O(N²) array at N=30 000), not about numerics, and this is a
 * measure-only harness. Emitted as `chiZeroSource` + `chiZeroBitIdenticalToStream` (schema gap G5)
 * because §7.12's `chiZeroStreaming: true` is a field every measured run contradicts.
 */
function chiZeroOperator(cell: Omit<Cell, 'exactApplyA'>): HierarchicalA {
    const bvh = buildBvh(cell.vertices, cell.edges, LEAF, EPS_A);
    return prepareA(
        bvh,
        cell.n,
        { kind: 'lists', bct: buildBct(bvh, { theta: 0 }) },
        { alpha: ALPHA, beta: BETA, epsilon: EPS_A, variant: F64, precomputeNear: true },
    );
}

function buildCell(fixture: string, n: number, theta: number, arms: Arm[]): Cell {
    const { vertices, edges } = FIXTURES[fixture].build(n);
    const bvh = buildBvh(vertices, edges, LEAF, EPS_A);
    const source: BlockSource = { kind: 'lists', bct: buildBct(bvh, { theta }) };
    const st = source.bct.stats;
    const H64 = prepareA(bvh, n, source, {
        alpha: ALPHA,
        beta: BETA,
        epsilon: EPS_A,
        variant: F64,
    });
    // Built ONLY when the f32 arm is requested. MEASURED reason (`scratch-7b/h23.ts:190-196`): a
    // second resident `HierarchicalA` slows `applyA` 6× at n=960 while leaving `Ã⁻¹` untouched.
    // Iteration counts and η are unaffected — only the cost, which is why it is still conditional.
    const H32 = arms.includes('f32')
        ? prepareA(bvh, n, source, { alpha: ALPHA, beta: BETA, epsilon: EPS_A, variant: F32 })
        : null;
    const op64 = makeHierarchicalOperator(H64);
    // The f32 arm reuses the f64 `σ` AND the f64 preconditioner, so OPERATOR PRECISION is the
    // single variable — exactly what the lab's `mv_f32(A)` did (it kept `sig` and `Mrs` from the
    // f64 `load_system`, `e1_gmres.py:62-83`). `sigma32` is RECORDED beside it so the substitution
    // is falsifiable rather than assumed.
    const op32: HierarchicalOperator | null = H32
        ? { n, sigma: op64.sigma, applyA: (u, out) => applyA(H32, u, out) }
        : null;
    const Lo = prepareLowOrder(vertices, edges, bvh, source, {
        alpha: ALPHA,
        beta: BETA,
        variant: 'vertex',
    });
    // NO `massWeight` argument: the SHIPPED default `DEFAULT_MASS_WEIGHT_HAT / ext²` (#72, `[A20]`).
    // The constant is referenced by NAME only to RECORD the value, never to reproduce the formula.
    const P = makePreconditioner(vertices, edges, Lo, op64.sigma);
    const ext = curveExtent(vertices);
    const partial = {
        fixture,
        n,
        theta,
        vertices,
        edges,
        m: bvh.order.length,
        extent: ext,
        massWeight: DEFAULT_MASS_WEIGHT_HAT / (ext * ext),
        nFar: st.nFar,
        nNear: st.nNear,
        nearPairs: st.nearPairs,
        adjacencyVetoed: st.adjacencyVetoed,
        kappaNear: st.nearPairs / bvh.order.length,
        kappaFar: st.nFar / bvh.order.length,
        sigma64: op64.sigma,
        sigma32: H32 ? makeHierarchicalOperator(H32).sigma : Number.NaN,
        H64,
        op64,
        op32,
        P,
    };
    const Hx = chiZeroOperator(partial);
    return { ...partial, exactApplyA: (u, out) => applyA(Hx, u, out) };
}

/**
 * The structural subset of {@link Cell} that {@link cellCounts} needs. Declared separately so the
 * apply-level gates (which hold a bare {@link HierarchicalA}, never a solve-ready {@link Cell})
 * can emit the same count block without fabricating a Cell.
 */
interface CellCounts {
    theta: number;
    m: number;
    nFar: number;
    nNear: number;
    nearPairs: number;
    adjacencyVetoed: number;
    kappaNear: number;
    kappaFar: number;
    sigma64: number;
    extent: number;
    massWeight: number;
}

/** The counts and the bar every per-θ gate row carries, computed from THIS run's own stats. */
function cellCounts(cell: CellCounts) {
    return {
        theta: cell.theta,
        m: cell.m,
        nFar: cell.nFar,
        nNear: cell.nNear,
        nearPairs: cell.nearPairs,
        adjacencyVetoed: cell.adjacencyVetoed,
        // κ_far = nFar/m. THIS DEFINITION DID NOT EXIST IN THE SPEC until `[A46 item 5]` added it;
        // §7.2 quoted "3.68" with no formula. Confirmed exactly against the torusKnot anchor.
        kappaNear: cell.kappaNear,
        kappaFar: cell.kappaFar,
        sigma: cell.sigma64,
        extent: cell.extent,
        massWeight: cell.massWeight,
    };
}

/**
 * H1-b's bar from the operator's OWN counts: `10 (κ_near + κ_far) u_f32` (spec §7.2,
 * `apply.test.ts:632-634`).
 *
 * ⚠ **A near-coincidence that must not become load-bearing.** On the trefoil at θ=0.25 the counts
 * are κ_near 93.65 / κ_far 3.1875, within 0.4 % and 13 % of the torusKnot's AT θ=0.5 (93.27 /
 * 3.677) — so an implementer who confused BOTH the fixture and θ lands a bar within **0.06 %** of
 * the right one (5.7720e-5 vs 5.7784e-5) and sees nothing wrong. MEASURED,
 * `scratch-11/confirm-960.json`. This is why the fixture is named in every row.
 */
const h1bBar = (counts: { kappaNear: number; kappaFar: number }): number =>
    10 * (counts.kappaNear + counts.kappaFar) * U_F32;

// ---------------------------------------------------------------------------------------------
// H1-a / H1-b — spec §7.2. Apply-level gates, both measured on `torusKnot` (every anchor is that
// curve) plus the `circle` n=960 must-NOT-trip tie cell.
// ---------------------------------------------------------------------------------------------

/**
 * `δ_compress(θ)` of spec §7.2: `‖A_χ0 u − A_θ u‖ / ‖A_χ0 u‖`, both f64 — `apply.test.ts:216-230`,
 * which is exactly `run-accuracy.ts:119`'s `relErrRandom`. The rhs is `seeded(n, 12345)` and the
 * reference is in the first slot; both are load-bearing for reproducing the archived anchors.
 */
function scalarApplyOf(H: HierarchicalA, u: ArrayLike<number>): Float64Array {
    const out = new Float64Array(H.n);
    applyAScalar(H, u, out);
    return out;
}

function prepareAt(
    fixture: string,
    n: number,
    theta: number,
    variant: AccumulatorVariant = F64,
    extra: { invertAdmissibility?: boolean } = {},
): HierarchicalA {
    const { vertices, edges } = FIXTURES[fixture].build(n);
    const bvh = buildBvh(vertices, edges, LEAF, EPS_A);
    // θ=0 takes the STREAM path, exactly as `apply.test.ts:160-163` does, so the archived
    // `relErrRandom` anchors reproduce through the same code they were measured through.
    const source: BlockSource =
        theta === 0
            ? { kind: 'stream', params: { theta: 0, ...extra } }
            : { kind: 'lists', bct: buildBct(bvh, { theta, ...extra }) };
    return prepareA(bvh, n, source, {
        alpha: ALPHA,
        beta: BETA,
        epsilon: EPS_A,
        variant,
        precomputeNear: true,
        // Knock-outs are opted into HERE, per configuration, never by default (`prepareA`'s guard).
        knockOut: extra.invertAdmissibility === true || variant === F16_NEAR_KNOCKOUT,
    });
}

function gateH1a(ns: number[]) {
    const rows: Record<string, unknown>[] = [];
    const n = ns.includes(960) ? 960 : ns[ns.length - 1];
    const u = seeded(n, H1_RHS_SEED);

    // ---- Decision region: torusKnot, θ descending. κ_near is MEASURED distinct at every rung
    // (93.27 → 185.75 → 227.96 at n=960), so a tie HERE would itself be the defect.
    // ---- Coverage region: circle, the must-NOT-trip tie cell. REPORTED, never folded into the
    // conjunction (`[A38 item 5]`'s decision/coverage split).
    const cells: { fixture: string; thetas: number[]; region: 'decision' | 'coverage' }[] = [
        { fixture: 'torusKnot', thetas: [0.5, 0.25, 0.1], region: 'decision' },
        { fixture: 'circle', thetas: [0.5, 0.25], region: 'coverage' },
    ];

    let monotoneOk = true;
    let tieWithDifferingCounts = false;
    const ties: Record<string, [number, number][]> = {};

    for (const { fixture, thetas, region } of cells) {
        const { vertices } = FIXTURES[fixture].build(n);
        const refStream = scalarApplyOf(prepareAt(fixture, n, 0), u);
        // The χ=0 substitution, RE-MEASURED on this run rather than cited: `scratch-7b/exactpath.ts`
        // proved lists≡stream bit-identical, and every other gate here relies on it, so it is
        // checked here where it costs one extra apply.
        const bvh = buildBvh(vertices, FIXTURES[fixture].build(n).edges, LEAF, EPS_A);
        const refLists = scalarApplyOf(
            prepareA(
                bvh,
                n,
                { kind: 'lists', bct: buildBct(bvh, { theta: 0 }) },
                { alpha: ALPHA, beta: BETA, epsilon: EPS_A, variant: F64, precomputeNear: true },
            ),
            u,
        );
        let chiZeroBitIdentical = refStream.length === refLists.length;
        for (let i = 0; i < refStream.length && chiZeroBitIdentical; i++) {
            if (!Object.is(refStream[i], refLists[i])) chiZeroBitIdentical = false;
        }

        const lad = thetaLadder(bvh, thetas, {});
        ties[fixture] = lad.ties as [number, number][];
        let prev: { theta: number; delta: number; nearPairs: number } | null = null;
        for (let r = 0; r < thetas.length; r++) {
            const theta = thetas[r];
            const H = prepareAt(fixture, n, theta);
            const delta = relErr(refStream, scalarApplyOf(H, u));
            const st = H.stats;
            const tiedWithPrevious = lad.rungs[r].tiedWithPrevious;
            const monotone = prev === null ? true : delta <= H1A_SLACK * prev.delta;
            // INVALID clause 1 of §7.2: a `κ_near` TIE where the counts actually differ means the
            // ladder is not a ladder. The tie is on `nearPairs`, the counts are `nearPairs` — so
            // the clause is "flagged tied but counts differ", which is a contradiction detector on
            // `thetaLadder` itself.
            const tieCountsDiffer =
                tiedWithPrevious && prev !== null && st.nearPairs !== prev.nearPairs;
            if (region === 'decision' && !monotone) monotoneOk = false;
            if (tieCountsDiffer) tieWithDifferingCounts = true;
            rows.push({
                region,
                fixture: provenance(fixture, vertices, n),
                ...cellCounts({
                    theta,
                    m: H.m,
                    nFar: st.nFar,
                    nNear: st.nNear,
                    nearPairs: st.nearPairs,
                    adjacencyVetoed: st.adjacencyVetoed,
                    kappaNear: st.nearPairs / H.m,
                    kappaFar: st.nFar / H.m,
                    sigma64: Number.NaN,
                    extent: curveExtent(vertices),
                    massWeight: DEFAULT_MASS_WEIGHT_HAT / curveExtent(vertices) ** 2,
                }),
                deltaCompress: delta,
                ratioToPrevious: prev === null ? null : delta / prev.delta,
                monotoneWithinSlack: monotone,
                tiedWithPrevious,
                invalidTieWithDifferingCounts: tieCountsDiffer,
                chiZeroSource: 'stream-theta0 (reference) vs lists-theta0-precomputeNear (check)',
                chiZeroBitIdenticalToStream: chiZeroBitIdentical,
            });
            prev = { theta, delta, nearPairs: st.nearPairs };
        }
    }

    // ---- RED-side reachability. `invertAdmissibility` is §7.2 H1-a's knock-out (2): one sign of
    // Eq. 28 flips, so `δ_compress` must RISE as θ falls. A gate whose RED side is never
    // demonstrated reads exactly like one that cannot fire (memory
    // `gates-name-their-impossible-values`), so it is measured here and LABELLED as a knock-out.
    const koFixture = 'torusKnot';
    const koRows: Record<string, unknown>[] = [];
    const koRef = scalarApplyOf(prepareAt(koFixture, n, 0), u);
    let koPrev: number | null = null;
    let koRose = false;
    for (const theta of [0.5, 0.25]) {
        const d = relErr(
            koRef,
            scalarApplyOf(prepareAt(koFixture, n, theta, F64, { invertAdmissibility: true }), u),
        );
        if (koPrev !== null && d > H1A_SLACK * koPrev) koRose = true;
        koRows.push({
            theta,
            deltaCompress: d,
            ratioToPrevious: koPrev === null ? null : d / koPrev,
        });
        koPrev = d;
    }

    const verdict: Verdict = tieWithDifferingCounts ? 'INVALID' : monotoneOk ? 'GREEN' : 'RED';
    return {
        bar: `delta_compress(theta_{k+1}) <= ${H1A_SLACK} * delta_compress(theta_k), theta descending`,
        barSource: 'spec §7.2; apply.test.ts:604',
        valueKey: 'rows[].deltaCompress',
        operator: 'f64 vs f64 (compression only; precision is H1-b’s job)',
        rows,
        ties,
        invalid: tieWithDifferingCounts ? ['tie-with-differing-counts'] : [],
        // INVALID clause 4 of §7.2 is the adapter certification. `[A46 item 2]`: on a CPU harness
        // it CANNOT NOT FIRE (`certifyAdapter`'s fall-through returns `'software'`,
        // `bench/gpu/adapterCert.ts:117-122`), and §7.0:1219 would then forbid publishing the run —
        // a predicate that cannot not fire reads exactly like a broken harness. Scoped to GPU runs.
        invalidClause4AdapterCert: 'NOT APPLICABLE',
        invalidClause4Reason:
            'spec §7.2:1327 requires adapterCert.classification === "hardware"; there is no GPU anywhere in H1-H6 and a CPU Bun harness supplies neither GPUAdapterInfo nor browser HostEvidence, so the clause is scoped to GPU runs per [A46 item 2]',
        knockOut: {
            label: 'KNOCK-OUT, NOT A MEASUREMENT',
            what: 'invertAdmissibility: one sign of Eq. 28 flipped (spec §7.2 knock-out 2)',
            expected: 'delta_compress RISES as theta falls => the gate must read RED',
            rose: koRose,
            demonstratesRedSideReachable: koRose,
            rows: koRows,
        },
        limitations: [
            'Single rung n=960: §7.11 item 10 records that H1-a’s monotonicity is single-geometry, and kappa_near spans 14.67x across geometries.',
            'The decision region is torusKnot only; circle is coverage and is NEVER folded into the conjunction ([A38 item 5]).',
        ],
        verdict: verdictOf(
            verdict,
            'GREEN',
            verdict === 'GREEN'
                ? `delta_compress non-increasing within ${H1A_SLACK} at every descending rung on the decision fixture; the circle tie fired as pre-registered and is coverage-only`
                : verdict === 'RED'
                  ? 'delta_compress ROSE as theta fell on the decision fixture'
                  : 'thetaLadder flagged a tie at rungs whose nearPairs differ',
        ),
    };
}

function gateH1b(ns: number[]) {
    const n = ns.includes(960) ? 960 : ns[ns.length - 1];
    const fixture = 'torusKnot';
    const { vertices } = FIXTURES[fixture].build(n);
    const u = seeded(n, H1_RHS_SEED);
    const rows: Record<string, unknown>[] = [];
    let allInBand = true;

    for (const theta of THETAS_GATED) {
        const Href = prepareAt(fixture, n, theta);
        const kappaNear = Href.stats.nearPairs / Href.m;
        const kappaFar = Href.stats.nFar / Href.m;
        const bar = h1bBar({ kappaNear, kappaFar });
        const ref = scalarApplyOf(Href, u);
        const d32 = relErr(ref, scalarApplyOf(prepareAt(fixture, n, theta, F32), u));
        // The 12-column vector apply, same band — `apply.test.ts:680-689`.
        const U = new Float64Array(3 * n);
        for (let c = 0; c < 3; c++) U.set(seeded(n, 300 + c), c * n);
        const vecOut = (H: HierarchicalA): Float64Array => {
            const out = new Float64Array(3 * n);
            applyA(H, U, out);
            return out;
        };
        const d32v = relErr(
            vecOut(prepareAt(fixture, n, theta)),
            vecOut(prepareAt(fixture, n, theta, F32)),
        );
        const inBand = d32 >= H1B_LOWER && d32 <= bar;
        if (!inBand) allInBand = false;
        // The f16 knock-out: must EXCEED the bar. The MEASURED overshoot is 45.6x; §7.11 item 7's
        // MODEL said 820x (m * u_f16, worst-case linear growth) and is 17.8x high — the real growth
        // is ~sqrt(m) * u_f16. The model is recorded beside the measurement, not instead of it.
        const d16 = relErr(ref, scalarApplyOf(prepareAt(fixture, n, theta, F16_NEAR_KNOCKOUT), u));
        rows.push({
            fixture: provenance(fixture, vertices, n),
            theta,
            m: Href.m,
            nearPairs: Href.stats.nearPairs,
            nFar: Href.stats.nFar,
            kappaNear,
            kappaFar,
            uF32: U_F32,
            bar,
            deltaPrecisionF32: d32,
            deltaPrecisionF32VectorApply: d32v,
            marginBelowBar: bar / d32,
            lowerBound: H1B_LOWER,
            inBand,
            knockOutF16: {
                label: 'KNOCK-OUT, NOT A MEASUREMENT OF PRODUCTION',
                deltaPrecision: d16,
                overBarBy: d16 / bar,
                // §7.11 item 7's model, WITH ITS NOTATION DEFECT RECORDED. The spec writes
                // "m * u_f16" and then substitutes 97 — which is `kappa_near + kappa_far`
                // (96.946 at n=960, torusKnot), NOT `m` (= 960). Only the kappa reading
                // reproduces the spec's own quoted 4.7e-2 and "~820x"; the literal `m` reading is
                // 9.9x higher. Both are emitted so the disagreement is visible rather than
                // silently resolved in the harness's favour.
                modelKappaForm: (kappaNear + kappaFar) * U_F16,
                modelKappaFormOverBarBy: ((kappaNear + kappaFar) * U_F16) / bar,
                modelLiteralMForm: Href.m * U_F16,
                modelLiteralMFormOverBarBy: (Href.m * U_F16) / bar,
                modelNotationDefect:
                    'spec §7.11 item 7 writes "m * u_f16" but substitutes 97 = kappa_near + kappa_far; only the kappa reading reproduces its own quoted 4.7e-2 / ~820x',
                modelOverstatesMeasuredBy: ((kappaNear + kappaFar) * U_F16) / d16,
                exceedsBar: d16 > bar,
            },
        });
    }

    return {
        bar: '10 * (kappa_near + kappa_far) * u_f32, from the run’s OWN counts; two-sided with a 1e-7 floor',
        barSource:
            'spec §7.2; apply.test.ts:632-634, :655-659; [A46 item 5] (computed, never copied)',
        valueKey: 'rows[].deltaPrecisionF32',
        operatorPair: 'f32 vs f64 at the SAME theta (precision only; compression is H1-a’s job)',
        uF32Definition: 'U_F32 = 2**-24 (UNIT ROUNDOFF, not machine epsilon 2**-23) — [A39 item 1]',
        rows,
        invalid: [],
        limitations: [
            'A GREEN H1-b is NOT evidence that the kernel chain is f32. MEASURED (apply.test.ts:660-664): evaluating the whole hi chain in f64 with ONE fround on the return moved delta_precision by -1.2 % and the band still passed. The claim "the chain is rounded" is carried by the ROUNDING-PLACEMENT PINS (nearField.test.ts), not by this band.',
            'The band bounds ACCUMULATION noise only (kappa_near + kappa_far terms per row).',
            'Output accumulation is f32 and NOT CONFIGURABLE — see precision.outputAccumulation.',
        ],
        verdict: verdictOf(
            allInBand ? 'GREEN' : 'RED',
            'GREEN',
            allInBand
                ? 'delta_precision inside [1e-7, bar] at both gated theta, on the scalar AND the 12-column apply, and the f16 knock-out exceeds the bar'
                : 'delta_precision outside the two-sided band at at least one gated theta',
        ),
    };
}

// ---------------------------------------------------------------------------------------------
// H2 / H3 — spec §7.3, §7.4. Both come from GMRES solves on `trefoil` at native extent.
// ---------------------------------------------------------------------------------------------

interface SolveRow {
    seed: number;
    arm: Arm;
    iters: number;
    maxit: number;
    tol: number;
    stop: string;
    censored: boolean;
    degenerate: boolean;
    matvecs: number;
    probeExactMatvecs: number;
    probeBctMatvecs: number;
    precondApplies: number;
    normA2: number;
    normA2Matvecs: number;
    etaBctMin: number;
    etaBctMinIter: number;
    etaExactMin: number;
    etaExactMinIter: number;
    probesPlanned: number;
    probesTaken: number;
    invalid: EtaClause[];
}

/**
 * One solve, with the η probe armed. `tol: 0` plus `etaTarget: undefined` is H2's RUN-TO-FLOOR;
 * `etaTarget: τ_η` is H3's COUNT-TO-τ. Both use `maxit = defaultMaxit(3n)` — spec §7.3:1401 pins
 * it ("exactly E1's budget, so the measured floors transfer literally"), so it is never widened or
 * narrowed to buy budget. Sharding by (n, arm, seed) is the budget lever instead.
 */
function solve(cell: Cell, arm: Arm, seed: number, etaTarget?: number): SolveRow {
    const dim = 3 * cell.n;
    const op = arm === 'f64' ? cell.op64 : (cell.op32 as HierarchicalOperator);
    const b = gaussian(dim, seed);
    const maxit = defaultMaxit(dim);
    const r = gmres(op, cell.P, b, {
        tol: 0,
        maxit,
        side: 'left',
        probe: {
            exactApplyA: cell.exactApplyA,
            // The OPERATOR's own unit roundoff, never u_f32 for an f64 run: #59's unscoped
            // `η < u_f32/10` rules every correct f64 run INVALID (spec §7.3, clause 2).
            uOp: arm === 'f64' ? U_F64 : U_F32,
            schedule: DEFAULT_PROBE_SCHEDULE,
            etaTarget,
        },
    });
    const p = r.probe as NonNullable<typeof r.probe>;
    return {
        seed,
        arm,
        iters: r.iters,
        maxit: r.maxit,
        tol: r.tol,
        stop: r.stop,
        censored: r.censored,
        degenerate: r.degenerate,
        matvecs: r.matvecs,
        probeExactMatvecs: p.exactMatvecs,
        probeBctMatvecs: p.bctMatvecs,
        precondApplies: r.precondApplies,
        normA2: p.normA2,
        normA2Matvecs: p.normA2Matvecs,
        etaBctMin: p.etaBctMin,
        etaBctMinIter: p.etaBctMinIter,
        etaExactMin: p.etaMin,
        etaExactMinIter: p.etaMinIter,
        probesPlanned: p.planned.length,
        probesTaken: p.records.length,
        invalid: p.invalid,
    };
}

/**
 * H2's aggregates and verdict, recomputed from WHATEVER rows the document holds.
 *
 * Separate from {@link gateH2} because the campaign is SHARDED: `--n` and `--arms` split one gate
 * across foreground invocations, and a verdict computed inside the measuring function would be a
 * verdict over one shard. The driver merges rows first and then calls this, so the published
 * verdict is always over the full merged set. Nothing here measures anything.
 */
function finaliseH2(block: Record<string, unknown>): void {
    const rows = (block.rows ?? []) as Record<string, unknown>[];
    const invalid = new Set<string>();
    const gatedFloors: { n: number; v: number }[] = [];
    let worst = Number.NEGATIVE_INFINITY;
    for (const row of rows) {
        const arms = (row.arms ?? {}) as Record<string, Record<string, unknown>>;
        for (const [armName, a] of Object.entries(arms)) {
            const seeds = (a.seeds ?? []) as {
                invalid: string[];
                etaBctMin: number;
                etaExactMin: number;
            }[];
            for (const sr of seeds) for (const c of sr.invalid) invalid.add(c);
            if (seeds.length === 0) continue;
            // RECOMPUTED from the merged seed set, never read from the shard that wrote it.
            const uOp = armName === 'f64' ? U_F64 : U_F32;
            const floors = seeds.map((x) => x.etaBctMin);
            const floorMax = Math.max(...floors);
            a.seedsRun = seeds.map((x) => (x as unknown as { seed: number }).seed);
            a.etaBctFloorMax = floorMax;
            a.etaBctFloorMaxInUOp = floorMax / uOp;
            a.etaBctFloorMedian = median(floors);
            a.marginBelowBar = armName === 'f32' ? H2_BAR / floorMax : null;
            a.etaExactMinMedian = median(seeds.map((x) => x.etaExactMin));
            a.etaExactOverEtaBct = median(seeds.map((x) => x.etaExactMin / x.etaBctMin));
            if (armName === 'f32') {
                gatedFloors.push({ n: row.n as number, v: floorMax });
                worst = Math.max(worst, floorMax);
            }
        }
        // §7.3's run-to-floor EXPECTS censoring at the larger rungs (`gmres.ts`: "Under H2's
        // run-to-floor this is expected true and is not an INVALID trigger"), so it is surfaced as
        // a VALUE rather than left implicit inside the seed rows: at n >= 240 every solve exhausts
        // maxit=600 and the gated floor is a minimum over a CENSORED trajectory. That is by design
        // — §7.3:1401 pins maxit so the measured floors transfer literally — but a reader who does
        // not know it would mistake the floor for a converged limit.
        row.censoredAny = Object.values(arms).some((a) =>
            ((a.seeds ?? []) as { censored: boolean }[]).some((x) => x.censored),
        );
        row.degenerateAny = Object.values(arms).some((a) =>
            ((a.seeds ?? []) as { degenerate: boolean }[]).some((x) => x.degenerate),
        );
    }
    gatedFloors.sort((a, b) => a.n - b.n);
    const measured = Number.isFinite(worst);
    block.worstGatedFloor = measured ? worst : null;
    block.marginBelowBar = measured ? H2_BAR / worst : null;
    block.gatedRungs = gatedFloors.map((g) => g.n);
    block.floorExponentInN =
        gatedFloors.length >= 2
            ? fitExponent(
                  gatedFloors.map((g) => g.n),
                  gatedFloors.map((g) => g.v),
              )
            : null;
    block.invalid = [...invalid];
    const verdict: Verdict = !measured
        ? 'NOT ESTABLISHED'
        : invalid.size > 0
          ? 'INVALID'
          : worst <= H2_BAR
            ? 'GREEN'
            : 'RED';
    block.verdict = verdictOf(
        verdict,
        'GREEN',
        !measured
            ? 'no gated (f32) rung has been measured yet'
            : verdict === 'GREEN'
              ? `worst f32 eta_bct floor ${worst.toExponential(4)} = ${(worst / U_F32).toFixed(4)} u_f32 over rungs ${gatedFloors.map((g) => g.n).join(',')}, ${(H2_BAR / worst).toFixed(1)}x under the bar, with an EMPTY invalid set on every cell`
              : verdict === 'INVALID'
                ? `probe INVALID clauses fired: ${[...invalid].join(', ')}`
                : `worst f32 eta_bct floor ${worst.toExponential(4)} exceeds the bar ${H2_BAR.toExponential(4)}`,
    );
}

function gateH2(ns: number[], arms: Arm[], seeds: number[]) {
    const rows: Record<string, unknown>[] = [];

    for (const n of ns) {
        // θ=0.5 is the rung every A26 floor was measured at. The f32 arm carries the bar and the
        // f64 arm is its recorded companion; `--arms` shards them because at n=960 both arms
        // together exceed the 10-minute foreground cap (MEASURED: 11.4 s at n=64 scales as ~n x the
        // apply-equivalent ladder of scratch-11/budget.json).
        const cell = buildCell('trefoil', n, 0.5, arms);
        const perArm: Record<string, unknown> = {};
        for (const arm of arms) {
            const seedRows = seeds.map((s) => solve(cell, arm, s));
            const floors = seedRows.map((s) => s.etaBctMin);
            const floorMax = Math.max(...floors);
            perArm[arm] = {
                gated: arm === 'f32',
                // WHY only the f32 arm is gated: §7.11 item 4 forbids transferring a bar derived at
                // one precision pair to another, and the f64 arm's floor is 3.3-5.3 u_f64 ([A26]) —
                // five decades under a bar expressed in u_f32. A "GREEN" there would be an artefact
                // of the unit, not a measurement.
                uOp: arm === 'f64' ? U_F64 : U_F32,
                etaBctFloorMax: floorMax,
                etaBctFloorMaxInUOp: floorMax / (arm === 'f64' ? U_F64 : U_F32),
                etaBctFloorMedian: median(floors),
                marginBelowBar: arm === 'f32' ? H2_BAR / floorMax : null,
                // `η_exact` is the δ_compress MONITOR and carries NO bar ([A21], §7.3's amendment
                // box): its floor IS ‖E x̃‖/(‖A_σ‖‖x̃‖+‖b‖) with E = A_χ0 − A_bct. Recorded so the
                // gated and non-gated quantities cannot be confused — the defect the box is about.
                etaExactMinMedian: median(seedRows.map((s) => s.etaExactMin)),
                etaExactOverEtaBct: median(seedRows.map((s) => s.etaExactMin / s.etaBctMin)),
                seeds: seedRows,
            };
        }
        rows.push({
            n,
            fixture: provenance('trefoil', cell.vertices, n),
            ...cellCounts(cell),
            sigmaF32Recorded: cell.sigma32,
            sigmaSubstitution:
                'the f32 arm reuses the f64 sigma AND the f64 preconditioner, so operator precision is the single variable (e1_gmres.py:62-83)',
            chiZeroSource: 'lists-theta0-precomputeNear',
            chiZeroBitIdenticalToStream: true,
            chiZeroBitIdentityProof:
                '.superpowers/sdd/2026-10-09-phase2h-bct-gmres-design/scratch-7b/exactpath.ts (0 of 1440 entries non-identical by Object.is, maxRel exactly 0)',
            etaExactVariant: 'f64',
            arms: perArm,
        });
    }

    const block: Record<string, unknown> = {
        rowKey: 'n',
        bar: `eta_bct <= 50 * u_f32 = ${H2_BAR}`,
        barSource: 'spec §7.3 (computed from U_F32 = 2**-24, never written as a decimal literal)',
        valueKey: 'rows[].arms.f32.etaBctFloorMax',
        gatedArm: 'f32',
        recordedArm: 'f64',
        gatedQuantity:
            'EtaProbeReport.etaBctMin — the GATED value, read precomputed. eta_exact (etaMin) is the delta_compress monitor and carries NO bar ([A21], §7.3)',
        probeSchedule: DEFAULT_PROBE_SCHEDULE,
        probesAtMaxit600: probeIterations(600, DEFAULT_PROBE_SCHEDULE).length,
        seedsRun: seeds,
        rows,
        rungsNotMeasured: [
            {
                n: 1920,
                status: 'NOT MEASURED',
                reason: 'MEASURED cost: one n=960 run-to-floor solve is 138 s wall with both operators resident and 33 s with only f64 (the 4.2x gap independently reproduces scratch-7b/h23.ts:190-196\u2019s "a second resident HierarchicalA slows applyA 6x"). The per-doubling cost ratio MEASURED between n=480 and n=960 is 3.0x against the 1.66x the apply-count budget predicted, because the budget omitted the Arnoldi orthogonalisation: full GMRES at maxit=600 costs O(iters^2 * dim) = 600^2 * 2880 = 1.04e9 at n=960, roughly 8x the near-field apply work, and it is INDEPENDENT of the apply count. n=1920 therefore costs ~69 min of foreground time for a rung whose value the ladder already determines.',
                whyItCostsTheGateNothing:
                    'the gated floor is FLAT-to-FALLING over the five measured rungs (0.4118 -> 0.3112 u_f32, exponent -0.103), the margin GROWS with n (121.4x -> 160.6x), and [A26] already published this quantity at n=1920. H3 covers n=1920 in its own currency.',
            },
        ],
        limitations: [
            'H2 certifies BACKWARD error, not solution or direction quality (spec §7.3:1557). A GREEN H2 says the residual is as small as the precision allows, not that x̃ is close to x*.',
            'eta_exact and eta_bct share ONE schedule and ONE exactApplyA by construction (gmres.ts:344-382; probeAt at :688-716 computes both on every probe), so the chi=0 probe cost cannot be reduced without changing an exported interface. The chi=0 share of one solve is 79 % at n=960 and 89 % at n=1920 (MEASURED, scratch-11/budget.json).',
            'INVALID clause 3 (exact-equals-bct) is scoped by the HARNESS, not by etaViolations: it is an inversion only where nFar > 0. nFar is MEASURED > 0 at every gated cell (1186 at n=960), so no gated row is exempt.',
            'CENSORED BY DESIGN at n >= 240: every solve exhausts maxit=600 and stops with `maxit`, so `censored` is true and the gated floor is a MINIMUM OVER A CENSORED TRAJECTORY, not a converged limit. §7.3:1401 pins maxit precisely so the floors transfer literally, and gmres.ts records that under run-to-floor censoring "is expected true and is not an INVALID trigger" — but it is published per row (`censoredAny`) because a reader who does not know it would read the floor as a limit.',
        ],
    };
    finaliseH2(block);
    return block as typeof block & { verdict: GateVerdict };
}

/**
 * H3's pairs, aggregates and verdict, recomputed from WHATEVER rungs the document holds. Split from
 * {@link gateH3} for the same reason as {@link finaliseH2}, and here it is load-bearing rather than
 * merely convenient: `G(n) = iters(2n)/iters(n)` needs BOTH legs of a doubling, so a sharded run
 * that measured n=240 yesterday and n=480 today can only form the pair after the merge.
 */
function finaliseH3(block: Record<string, unknown>): void {
    const rungs = ((block.rungs ?? []) as Record<string, unknown>[])
        .slice()
        .sort((x, y) => (x.n as number) - (y.n as number));
    block.rungs = rungs;
    const invalid = new Set<string>();
    for (const r of rungs) {
        for (const sr of (r.seeds ?? []) as {
            invalid: string[];
            censored: boolean;
            degenerate: boolean;
        }[]) {
            // §7.4's two INVALID predicates, both read PRECOMPUTED off GmresResult rather than
            // re-derived — `gmres.ts` exposes them for exactly this reason (a harness that derives
            // one and is handed the other reads the wrong one).
            if (sr.censored) invalid.add('censored');
            if (sr.degenerate) invalid.add('degenerate');
            for (const c of sr.invalid) invalid.add(c);
        }
    }
    // `G(n) = iters(2n)/iters(n)` over true DOUBLING pairs only; non-doubling rungs are dropped
    // rather than interpolated.
    const pairs: Record<string, unknown>[] = [];
    let worstGMinusBand = Number.NEGATIVE_INFINITY;
    let bandMax = 0;
    for (let i = 0; i + 1 < rungs.length; i++) {
        const lo = rungs[i];
        const hi = rungs[i + 1];
        if ((hi.n as number) !== 2 * (lo.n as number)) continue;
        const iLo = lo.itersMedian as number;
        const iHi = hi.itersMedian as number;
        const G = iHi / iLo;
        // The quantisation band: iteration counts are INTEGERS, so G carries +/-1 iteration on each
        // side. Without it a G of 1.02 and a G of 1.02 +/- 0.5 read identically.
        const band = G * (1 / iLo + 1 / iHi);
        bandMax = Math.max(bandMax, band);
        worstGMinusBand = Math.max(worstGMinusBand, G - band);
        pairs.push({
            n: lo.n,
            n2: hi.n,
            itersAtN: iLo,
            itersAt2N: iHi,
            G,
            band,
            gMinusBand: G - band,
            red: G - band >= H3_G_BAR,
        });
    }
    const worstG = pairs.length ? Math.max(...pairs.map((p) => p.G as number)) : null;
    const lastRung = rungs[rungs.length - 1];
    block.pairs = pairs;
    block.worstG = worstG;
    block.worstGMinusBand = Number.isFinite(worstGMinusBand) ? worstGMinusBand : null;
    block.invalid = [...invalid];
    // §7.4's quantisation INVALID predicate: `band > (G_bar - 1)/2`. `[A42]` ruling 5 applies — a
    // predicate that cannot fire in the live regime is VACUOUS, not PASS, and the artefact must
    // publish the band's VALUE so the margin is visible. §7.4:2004-2006 argues the same margin as a
    // virtue; the [A42] label is still the correct one.
    const quantThreshold = (H3_G_BAR - 1) / 2;
    const quantVacuous = pairs.length > 0 && bandMax <= quantThreshold;
    block.quantisationPredicate = {
        predicate: `band > (G_bar - 1)/2 = ${quantThreshold}`,
        bandMax: pairs.length ? bandMax : null,
        marginUnderThreshold: bandMax > 0 ? quantThreshold / bandMax : null,
        verdict: pairs.length === 0 ? 'NOT ESTABLISHED' : quantVacuous ? 'VACUOUS' : 'LIVE',
        reason:
            pairs.length === 0
                ? 'no doubling pair has been measured yet'
                : quantVacuous
                  ? `the band is ${bandMax.toFixed(4)}, ${(quantThreshold / bandMax).toFixed(1)}x under the threshold ${quantThreshold}, so the predicate CANNOT fire in this regime — VACUOUS per [A42] ruling 5, not PASS`
                  : 'the band reaches the threshold: the iteration counts are too small for G to be read at this bar',
    };
    block.primaryBar = {
        bar: `iters(${H3_PRIMARY_BAR_N}) <= ${H3_PRIMARY_BAR_ITERS}`,
        status: 'NOT MEASURABLE AT THESE RUNGS',
        reason: `the largest rung here is n=${lastRung?.n ?? 'none'}; n=${H3_PRIMARY_BAR_N} is ${(H3_PRIMARY_BAR_N / ((lastRung?.n as number) || 1)).toFixed(0)}x beyond it. Row 11 does NOT re-extrapolate it`,
        dischargedElsewhere:
            '[A31] discharged it at n=15360 (iters 46, projection 50.1, margin 1.28x) — that cell, not this harness, is the primary bar\u2019s evidence',
        worstGProjection:
            worstG === null || lastRung === undefined
                ? null
                : {
                      fromN: lastRung.n,
                      fromIters: lastRung.itersMedian,
                      doublingsTo30k: Math.log2(H3_PRIMARY_BAR_N / (lastRung.n as number)),
                      projectedIters:
                          (lastRung.itersMedian as number) *
                          worstG ** Math.log2(H3_PRIMARY_BAR_N / (lastRung.n as number)),
                      label: 'PROJECTION from this run\u2019s worst measured G, NOT a measurement',
                  },
    };
    const companionGreen = Number.isFinite(worstGMinusBand) && worstGMinusBand < H3_G_BAR;
    const verdict: Verdict =
        pairs.length === 0
            ? 'NOT ESTABLISHED'
            : invalid.size > 0
              ? 'INVALID'
              : companionGreen
                ? 'GREEN'
                : 'RED';
    block.verdict = verdictOf(
        verdict,
        'GREEN',
        pairs.length === 0
            ? 'no doubling pair has been measured yet'
            : verdict === 'GREEN'
              ? `worst G - band = ${worstGMinusBand.toFixed(4)} against the bar ${H3_G_BAR} over pairs ${pairs.map((p) => `${p.n}->${p.n2}`).join(',')}; the quantisation predicate is VACUOUS with band ${bandMax.toFixed(4)}`
              : verdict === 'INVALID'
                ? `INVALID predicates fired: ${[...invalid].join(', ')}`
                : `worst G - band = ${worstGMinusBand.toFixed(4)} reaches the bar ${H3_G_BAR}`,
    );
}

function gateH3(ns: number[], seeds: number[]) {
    // H3 is f64-OPERATOR-ONLY and that is a measured decision, not an omission: the f32 η floor is
    // ~3e-8, so NO fixed τ_η is both reachable and non-degenerate in f32 — at τ_η=1e-6 the f32
    // counts collapsed to 19 and 12 at n=960/1920 (§10.2's retracted-claims row). H3's scope is the
    // preconditioner's SPECTRAL mesh-independence; precision is H1-b's and H4a's job.
    const rungs: Record<string, unknown>[] = [];
    for (const n of ns) {
        const cell = buildCell('trefoil', n, 0.5, ['f64']);
        const rows = seeds.map((sd) => solve(cell, 'f64', sd, TAU_ETA));
        rungs.push({
            fixture: provenance('trefoil', cell.vertices, n),
            n,
            theta: cell.theta,
            itersMedian: median(rows.map((r) => r.iters)),
            itersPerSeed: rows.map((r) => r.iters),
            maxit: rows[0]?.maxit ?? null,
            censoredAny: rows.some((r) => r.censored),
            degenerateAny: rows.some((r) => r.degenerate),
            seeds: rows,
        });
    }

    const block: Record<string, unknown> = {
        rowKey: 'n',
        rowsField: 'rungs',
        companionBar: `G - band < ${H3_G_BAR}`,
        companionBarSource: 'spec §7.4',
        valueKey: 'pairs[].gMinusBand',
        operator: 'f64 ONLY',
        operatorReason:
            'the f32 eta floor is ~3e-8, so no fixed tau_eta is both reachable and non-degenerate in f32 (MEASURED collapse to 19 and 12 iters at n=960/1920, §10.2)',
        tauEta: TAU_ETA,
        tauEtaSource: '[A31], its THIRD re-pin (1e-8 -> 1e-10 -> 1e-12)',
        stoppingCurrency:
            'eta_bct, NOT eta_exact (EtaProbeOptions.etaTarget): eta_exact floors at the BCT compression error (1.87e-3 at n=64, MEASURED), so every usable tau_eta is unreachable on it and the count would be censored at every n',
        seedsRun: seeds,
        rungs,
        limitations: [
            'G needs true DOUBLING pairs; non-doubling rungs are dropped from `pairs` rather than interpolated.',
            'tau_eta, side and maxit are pinned identically across every rung by construction — parameter drift between the two legs of a pair would make G meaningless.',
            'H3 measures the preconditioner\u2019s SPECTRAL mesh-independence on the f64 operator. It says nothing about precision, which is H1-b\u2019s and H4a\u2019s job.',
        ],
    };
    finaliseH3(block);
    return block as typeof block & { verdict: GateVerdict };
}

// ---------------------------------------------------------------------------------------------
// H4a / H4b′ — spec §7.5, §7.6. Both read the SAME gated iterate, so they share one solve.
// ---------------------------------------------------------------------------------------------

/**
 * The exact dense `A_σ` on one coordinate block: `assembleAFlat`'s all-pairs `A` plus `(σ/n)11ᵀ`.
 *
 * **`σ` is the BCT run's own σ, injected — NOT `tr(A_exact)/n`,** which is why `shiftSobolev`
 * (`src/gpu/saddle.ts:69`) is deliberately not used here: it derives σ from the dense trace, and
 * then the hierarchical and dense systems would differ in the SHIFT as well as in the operator,
 * so `cos∠_A(x̃, x*)` would no longer isolate compression. `tr(A_exact)/n` is RECORDED beside it so
 * the substitution is falsifiable. Same pattern as the f32 arm reusing the f64 σ.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §2.3, §7.5
 */
function denseExactASigma(
    vertices: Vec3[],
    edges: Edge[],
    n: number,
    sigma: number,
): { Asig: Float64Array; traceExactOverN: number } {
    const A = assembleAFlat(vertices, edges, calculateDisjointPairs(edges), ALPHA, BETA, EPS_A);
    let tr = 0;
    for (let i = 0; i < n; i++) tr += A[i * n + i];
    const s = sigma / n;
    for (let i = 0; i < n * n; i++) A[i] += s;
    return { Asig: A, traceExactOverN: tr / n };
}

/** `out = A_σ,exact u` on `3n`: the n×n block applied to each coordinate block. */
function makeDenseMatvec(Asig: Float64Array, n: number) {
    return (u: ArrayLike<number>, out: Float64Array): void => {
        for (let c = 0; c < 3; c++) {
            const o = c * n;
            for (let i = 0; i < n; i++) {
                let row = 0;
                const ri = i * n;
                for (let j = 0; j < n; j++) row += Asig[ri + j] * u[o + j];
                out[o + i] = row;
            }
        }
    };
}

/**
 * `cos∠_A(x, y)` in a GIVEN metric, with `1 − cos` taken from the CHORD rather than from
 * `1 − cos` directly: at `cos ≈ 0.999` the subtraction loses three digits and `1 − cos` is the
 * quantity §7.5's limitation is stated in. `‖x̂ − ŷ‖²_A / 2 = 1 − cos` exactly for unit-A-norm
 * `x̂, ŷ`. Transcribed from `scratch-10/lib.ts:207-221`.
 */
function angleIn(
    mv: (u: ArrayLike<number>, out: Float64Array) => void,
    x: Float64Array,
    y: Float64Array,
) {
    const dim = x.length;
    const Ax = new Float64Array(dim);
    const Ay = new Float64Array(dim);
    mv(x, Ax);
    mv(y, Ay);
    const ax = Math.sqrt(dot(x, Ax));
    const ay = Math.sqrt(dot(y, Ay));
    const cos = dot(x, Ay) / (ax * ay);
    const d = new Float64Array(dim);
    for (let i = 0; i < dim; i++) d[i] = x[i] / ax - y[i] / ay;
    const Ad = new Float64Array(dim);
    mv(d, Ad);
    const chord2 = Math.max(0, dot(d, Ad));
    return {
        cos,
        oneMinusCos: chord2 / 2,
        angleDeg: (2 * Math.asin(Math.min(1, Math.sqrt(chord2) / 2)) * 180) / Math.PI,
        // ε_A = ‖x − y‖_A / ‖y‖_A with y the REFERENCE — §7.5's RECORDED quantity (its bar was
        // retracted: the self-anchoring form #59 itself withdrew, and its δ=0 anchor is
        // non-monotone at n=960 and jumps 7.8x at n=1920).
        epsA: (() => {
            const e = new Float64Array(dim);
            for (let i = 0; i < dim; i++) e[i] = x[i] - y[i];
            const Ae = new Float64Array(dim);
            mv(e, Ae);
            return Math.sqrt(Math.max(0, dot(e, Ae))) / ay;
        })(),
    };
}

/**
 * H4a's aggregates and verdict from WHATEVER rows the document holds — the {@link finaliseH2}
 * contract. Rows that record `status: "NOT MEASURABLE"` (n > 960, §7.5's range limit) are counted
 * as coverage, never folded into the min.
 */
function finaliseH4a(block: Record<string, unknown>): void {
    const rows = ((block.rows ?? []) as Record<string, unknown>[])
        .slice()
        .sort(
            (a, b) => (a.n as number) - (b.n as number) || (a.seed as number) - (b.seed as number),
        );
    block.rows = rows;
    const measured = rows.filter((r) => typeof r.cos === 'number');
    let minCos = Number.POSITIVE_INFINITY;
    let refEtaWorst = 0;
    let bitEqual = false;
    for (const r of measured) {
        minCos = Math.min(minCos, r.cos as number);
        refEtaWorst = Math.max(refEtaWorst, r.referenceBackwardError as number);
        if (Object.is(r.cos, r.cosHatFromH4b)) bitEqual = true;
    }
    const invalid: string[] = [];
    if (refEtaWorst > H4A_REF_ETA_BAR) invalid.push('reference-solve-inaccurate');
    if (bitEqual) invalid.push('cos-bit-equal-to-h4b-surrogate');
    block.rungsMeasured = [...new Set(measured.map((r) => r.n))];
    block.rungsNotMeasurable = [...new Set(rows.filter((r) => r.status).map((r) => r.n))];
    block.minCos = measured.length ? minCos : null;
    block.marginAboveBar = measured.length ? minCos - H4A_BAR : null;
    block.worstReferenceBackwardError = measured.length ? refEtaWorst : null;
    // The INVALID-1 replacement is NEARLY BINDING at the top of H4a's range, so its trend is
    // published rather than left as a pass/fail. This is the FIRST COMPUTABLE statement of the
    // dense reference's own degradation — spec §7.5's n=960 ceiling and [A35]'s "dense LDL^T error
    // crosses 1e-10 between n=480 and 960" are both arrived at by COMPARING solves, whereas this
    // is the instrument reporting on itself. It is consistent with those, NOT a reproduction of
    // them: the fit below crosses the bar well above 960, so it is the weaker bound of the two.
    const perRung = [...new Set(measured.map((r) => r.n as number))].sort((x, y) => x - y);
    const etaByRung = perRung.map((n) =>
        Math.max(
            ...measured.filter((r) => r.n === n).map((r) => r.referenceBackwardError as number),
        ),
    );
    const pEta = perRung.length >= 2 ? fitExponent(perRung, etaByRung) : null;
    block.referenceBackwardErrorTrend =
        pEta === null
            ? null
            : {
                  rungs: perRung,
                  worstPerRung: etaByRung,
                  exponentInN: pEta,
                  marginAtLargestRung: H4A_REF_ETA_BAR / etaByRung[etaByRung.length - 1],
                  projectedCrossingN:
                      perRung[perRung.length - 1] *
                      (H4A_REF_ETA_BAR / etaByRung[etaByRung.length - 1]) ** (1 / pEta),
                  label: 'PROJECTION from this run, NOT a measurement',
              };
    block.invalid = invalid;
    (block.invalidClause1 as Record<string, unknown>).worstMeasured = measured.length
        ? refEtaWorst
        : null;
    (block.invalidClause2 as Record<string, unknown>).fired = bitEqual;
    const verdict: Verdict =
        measured.length === 0
            ? 'NOT ESTABLISHED'
            : invalid.length > 0
              ? 'INVALID'
              : minCos >= H4A_BAR
                ? 'GREEN'
                : 'RED';
    block.verdict = verdictOf(
        verdict,
        'GREEN',
        measured.length === 0
            ? 'no rung at or below n=960 has been measured'
            : verdict === 'GREEN'
              ? `min cos ${minCos.toFixed(6)} >= ${H4A_BAR} over rungs ${[...new Set(measured.map((r) => r.n))].join(',')}, reference backward error worst ${refEtaWorst.toExponential(3)} within ${H4A_REF_ETA_BAR.toExponential(3)}`
              : verdict === 'INVALID'
                ? `the reference instrument failed its own bound (${refEtaWorst.toExponential(3)} vs ${H4A_REF_ETA_BAR.toExponential(3)}), or cos came back bit-equal to H4b-prime's surrogate`
                : `min cos ${minCos.toFixed(6)} below ${H4A_BAR}`,
    );
}

/**
 * H4b-prime's aggregates and verdict. **The VACUOUS label is itself the falsifiable claim here**:
 * it holds only while `q > 0` on every converged direction, and a single `q <= 0` flips this to RED
 * and reopens `[A46 item 1]`. See `verdict.reason`.
 */
function finaliseH4b(block: Record<string, unknown>): void {
    const rows = ((block.rows ?? []) as Record<string, unknown>[])
        .slice()
        .sort(
            (a, b) => (a.n as number) - (b.n as number) || (a.seed as number) - (b.seed as number),
        );
    block.rows = rows;
    const anyQNonPositive = rows.some((r) => !r.qPositive);
    const anyXtbNonPositive = rows.some((r) => !((r.xTb as number) > 0));
    const blind = rows.filter((r) => r.cosBelowBarButQPositive).length;
    block.directions = rows.length;
    block.qPositiveFraction = rows.length
        ? rows.filter((r) => r.qPositive).length / rows.length
        : null;
    block.qChi0PositiveFraction = rows.length
        ? rows.filter((r) => r.qChi0Positive).length / rows.length
        : null;
    block.rowsWithCosBelowBarAndQPositive = blind;
    block.invalid = anyXtbNonPositive ? ['x~^T b <= 0'] : [];
    (block.overstatementCalibration as Record<string, unknown>).values = rows.map((r) => ({
        n: r.n,
        seed: r.seed,
        epsA: r.epsA,
        overstatementRatio: r.overstatementRatio,
    }));
    const verdict: Verdict =
        rows.length === 0 ? 'NOT ESTABLISHED' : anyQNonPositive ? 'RED' : 'VACUOUS';
    block.verdict = verdictOf(
        verdict,
        'VACUOUS',
        rows.length === 0
            ? 'no direction measured yet'
            : anyQNonPositive
              ? 'a converged solve returned q <= 0 — the VACUOUS label of [A46 item 1] is FALSIFIED by this run and item 1 must be reopened'
              : `q > 0 on ${rows.length}/${rows.length} directions, including ${blind} whose cos_A is below ${H4A_BAR}. [A42] ruling 5: a gate that cannot fail is VACUOUS, not GREEN. Spec 7.6:2071-2077 proves there is nothing to re-register — no reference-free lower bound on cos_A from a single iterate exists. Spec 7.12:2729's "H4b": "GREEN" example is WITHDRAWN`,
    );
}

function gateH4aAndH4b(ns: number[], seeds: number[]) {
    const h4aRows: Record<string, unknown>[] = [];
    const h4bRows: Record<string, unknown>[] = [];

    for (const n of ns) {
        if (n > 960) {
            // §7.5's range limit, PUBLISHED as a limit with a named consequence rather than
            // silently skipped: above n≈960 the f64 dense reference degrades measurably (its own
            // error crosses 1e-10 between n=480 and 960, [A35]), so no reference exists.
            h4aRows.push({
                n,
                status: 'NOT MEASURABLE',
                reason: 'spec §7.5: the dense f64 reference’s practical ceiling is n=960; above it H4a is VACUOUS and that is its OUTPUT, not a defect',
            });
            continue;
        }
        const cell = buildCell('trefoil', n, 0.5, ['f64']);
        const { Asig, traceExactOverN } = denseExactExport(cell);
        const denseMv = makeDenseMatvec(Asig, n);
        const R = cholesky(Asig, n);
        // ‖A_σ,exact‖₂ for the reference solve's own backward error (H4a's INVALID-1 replacement).
        const nrm = estimateNorm2(denseMv, 3 * n, 30, 9001);

        for (const seed of seeds) {
            const dim = 3 * n;
            const b = gaussian(dim, seed);
            // x*: the dense f64 reference. One n×n Cholesky serves all three coordinate blocks
            // because A is block-diagonal over the components (`layout.ts` expandBlockDiag) and the
            // shift is PER BLOCK (`gmres.ts` applyAsigma) — a 3n×3n factorisation would be 9x the
            // work for the same answer.
            const xStar = new Float64Array(b);
            for (let c = 0; c < 3; c++) {
                const blk = xStar.subarray(c * n, (c + 1) * n);
                cholSolve(R, n, blk);
            }
            // INVALID-1 replacement: the INSTRUMENT's own backward error. A reference solve that is
            // not itself accurate makes every cos below it meaningless, and unlike §7.5's
            // provenance clause this IS a function of outputs.
            const Ax = new Float64Array(dim);
            denseMv(xStar, Ax);
            const r = new Float64Array(dim);
            for (let i = 0; i < dim; i++) r[i] = b[i] - Ax[i];
            const refEta = backwardError(norm(r), nrm.norm, norm(xStar), norm(b));

            // x̃: the hierarchical solve at §7.5's own stopping tolerance.
            const g = gmres(cell.op64, cell.P, b, { tol: H4A_TAU, maxit: defaultMaxit(dim) });
            const xTilde = g.x;

            // The GATED cos: metric AND reference both the EXACT dense A_σ (controller Stage-1
            // ruling 3). The densified A_bct,σ cos is recorded beside it because it REMOVES the
            // compression term and so shows how much of the gap the compression is.
            const gated = angleIn(denseMv, xTilde, xStar);
            const bctMv = (u: ArrayLike<number>, out: Float64Array): void => {
                cell.op64.applyA(u, out);
                for (let c = 0; c < 3; c++) {
                    const o = c * n;
                    let s = 0;
                    for (let i = 0; i < n; i++) s += u[o + i];
                    const sh = cell.sigma64 * (s / n);
                    for (let i = 0; i < n; i++) out[o + i] += sh;
                }
            };
            const inBctMetric = angleIn(bctMv, xTilde, xStar);

            // ---- H4b′, on the SAME iterate. `q` is evaluated with the PRODUCTION (θ) matvec:
            // §7.6 rules that matvec, because `q` is then the model-energy decrease in the metric
            // the step is actually taken in. `q_χ0` is recorded beside it — it is NOT a repair,
            // task 0c measured that it FALSE-KILLS rhs whose cos_exact is 1.0.
            const AxT = new Float64Array(dim);
            bctMv(xTilde, AxT);
            const a = dot(xTilde, AxT);
            const c = dot(xTilde, b);
            const q = 2 * c - a;
            const AxT0 = new Float64Array(dim);
            cell.exactApplyA(xTilde, AxT0);
            for (let cc = 0; cc < 3; cc++) {
                const o = cc * n;
                let s = 0;
                for (let i = 0; i < n; i++) s += xTilde[o + i];
                const sh = cell.sigma64 * (s / n);
                for (let i = 0; i < n; i++) AxT0[o + i] += sh;
            }
            const qChi0 = 2 * c - dot(xTilde, AxT0);
            const cosHat = q > 0 ? c / Math.sqrt(a * q) : Number.NaN;
            // §7.5's SECOND, irreplaceable job: H4a is the only place H4b′'s blindness can be
            // QUANTIFIED. cosHat/cos_true = 1/sqrt(1 - eps_A²) exactly (DERIVED, §7.6).
            const overstatement = 1 / Math.sqrt(Math.max(Number.MIN_VALUE, 1 - gated.epsA ** 2));
            // H4a's INVALID 2: cos bit-equal to H4b′'s surrogate (the surrogate was used as the
            // reference). NEAR-VACUOUS and the artefact says so: §7.6 proves cosHat² >= 1 always
            // while cos <= 1, so bit-equality needs exactly 1.0 on both.

            h4aRows.push({
                fixture: provenance('trefoil', cell.vertices, n),
                n,
                seed,
                theta: cell.theta,
                tol: H4A_TAU,
                iters: g.iters,
                stop: g.stop,
                sigmaBct: cell.sigma64,
                traceExactOverN,
                sigmaSubstitution:
                    'the dense reference is shifted by the BCT run’s OWN sigma, so the two systems differ ONLY in the operator; tr(A_exact)/n is recorded for contrast',
                gatedMetric: 'exact dense A_sigma',
                cos: gated.cos,
                oneMinusCos: gated.oneMinusCos,
                angleDeg: gated.angleDeg,
                epsA: gated.epsA,
                cosInBctMetricNotGated: inBctMetric.cos,
                oneMinusCosInBctMetricNotGated: inBctMetric.oneMinusCos,
                referenceBackwardError: refEta,
                referenceBackwardErrorBar: H4A_REF_ETA_BAR,
                referenceValid: refEta <= H4A_REF_ETA_BAR,
                referenceNormA2: nrm.norm,
                // Recorded ON THE H4a ROW so INVALID clause 2 (cos bit-equal to H4b-prime's
                // surrogate) is checkable from this gate's own rows after a shard merge, without
                // joining against H4bPrime's rows.
                cosHatFromH4b: cosHat,
            });
            h4bRows.push({
                n,
                seed,
                matvec: 'production (theta) — spec §7.6’s ruling',
                xTb: c,
                xTAx: a,
                q,
                qPositive: q > 0,
                qChi0,
                qChi0Positive: qChi0 > 0,
                cosHat,
                cosHatAtLeastOne: cosHat >= 1,
                cosTrueFromH4a: gated.cos,
                epsA: gated.epsA,
                overstatementRatio: overstatement,
                cosBelowBarButQPositive: gated.cos < H4A_BAR && q > 0,
            });
        }
    }

    const h4a: Record<string, unknown> = {
        rowKey: 'n,seed',
        bar: `min over the pinned directions of cos_A(x~, x*) >= ${H4A_BAR}`,
        barSource: 'spec §7.5 (CHOSEN number; the delta=0 baseline column is MEASURED)',
        valueKey: 'rows[].cos (metric and reference both the EXACT dense A_sigma)',
        tol: H4A_TAU,
        tolReason:
            'spec §7.5’s measured baseline column is at tau=1e-4; the stopping currency is the RECURRENCE residual, which §5.3 measures as under-reporting the true residual by 3.9e3-6.6e3x — so part of 1-cos is STOPPING error, not compression',
        rows: h4aRows,
        invalidClause1: {
            specText: '"x* not from an f64 dense solve"',
            status: 'REPLACED — NOT A COMPUTABLE PREDICATE',
            reason: '[A46 item 4]: provenance is not a function of outputs, so it cannot be "reachable in a test" as spec §11.2 row 11 requires',
            replacement: `the reference solve’s OWN backward error <= 10*u_f64 = ${H4A_REF_ETA_BAR}`,
            specStatus:
                'NOT YET IN THE SPEC — controller Stage-1 ruling 4; proposed wording in task-11-report.md §S1.4',
        },
        invalidClause2: {
            specText: '"H4a’s cos bit-equal to H4b′’s surrogate"',
            status: 'NEAR-VACUOUS',
            reason: '§7.6:2057 proves cosHat² >= 1 ALWAYS while cos <= 1, so bit-equality requires exactly 1.0 on both sides — reachable in principle, unreachable on any real solve',
        },
        limitations: [
            'COMPRESSION-DOMINATED, pre-registered: [A44] §1 measures E_c/E_p = 500-4150 at n=960, so 1-cos is compression error with the f32 contribution five decades down. H4a does not and cannot separate them.',
            'RANGE LIMIT n <= 960, published as a limit with a named consequence ([A46 item 1]): above it NO gate in H1-H6 certifies direction quality, at any n, and none can be written reference-free (§7.6:2071-2077 proves no reference-free lower bound on cos_A exists). It is NOT ESTABLISHED that one is needed — utof/repulsive-test2#95.',
            'eps_A is RECORDED, never gated: #59’s eps_A <= 2*eps_A(delta=0) is the self-anchoring form #59 itself retracted, and its anchor is non-monotone at n=960 (+1.9 %) and jumps 7.8x at n=1920.',
        ],
    };
    finaliseH4a(h4a);

    // H4b′ IS VACUOUS, and the VACUOUS label is itself falsifiable — see `verdict.reason`.
    const h4b: Record<string, unknown> = {
        rowKey: 'n,seed',
        bar: 'min q > 0, q = 2 x~^T b - x~^T A_sigma x~  (spec §7.6’s repair of #59’s H4b)',
        barSource: 'spec §7.6; WITHDRAWN as a live bar by [A46 item 1]',
        valueKey: 'rows[].q',
        matvec: 'production (theta)',
        rows: h4bRows,
        polarity: 'q <= 0 is RED, NOT INVALID (spec §7.6:2117; #59 had it as INVALID)',
        overstatementCalibration: {
            what: 'cosHat/cos_true = 1/sqrt(1 - eps_A²), DERIVED exactly in §7.6',
            why: 'H4a is the ONLY place H4b′’s blindness can be quantified instead of assumed, and §7.5 says the gate MUST report it',
            availableOnlyWhere: 'n <= 960, where H4a’s dense reference exists',
        },
        limitations: [
            'The chi=0 matvec is NOT a repair: task 0c measured that it FALSE-KILLS rhs whose cos_exact is 1.0 (task-0c-report.md:61).',
            'q’s blindness is a STEP-SCALE finding, not a direction defect: 0c measured the direction right and the step length wrong by 1/lambda (2.75x at n=960, 3.46x at n=2000), consistent with §7.7’s "tau measures step scale, not quality". It stays benign only if the line search runs on those modes, which NOBODY has measured — utof/repulsive-test2#95.',
        ],
    };
    finaliseH4b(h4b);

    return {
        h4a: h4a as typeof h4a & { verdict: GateVerdict },
        h4b: h4b as typeof h4b & { verdict: GateVerdict },
    };
}

/** Split out so the dense assembly is one call site and its σ substitution is stated once. */
function denseExactExport(cell: Cell) {
    return denseExactASigma(cell.vertices, cell.edges, cell.n, cell.sigma64);
}

// ---------------------------------------------------------------------------------------------
// H4c — spec §7.7. The end-to-end descent gate: K accepted steps, hierarchical vs dense f64.
// ---------------------------------------------------------------------------------------------

/**
 * Runs `K` steps of {@link sobolevStepSet} and accumulates `ΔE_total`.
 *
 * **f64 OPERATOR ON BOTH ARMS, and it cannot be otherwise.** `makeHierarchicalSaddle` admits no
 * precision variant and calls `prepareA` with none (`hierarchicalSaddle.ts:321` ⇒ the `F64`
 * default), so `[§7.12]`'s `precision.operator: "f32"` is FALSE for H4c (`[A46 item 4]`); widening
 * `HierarchicalSaddleOptions` is an exported-interface change ⇒ utof/repulsive-test2#96's class.
 *
 * **`projectionMode: 'frozen'` on BOTH arms**, not just the hierarchical one: `optimizer.ts:308-327`
 * REFUSES anything else with a pre-armed saddle, so a `'reassemble'` dense arm would be comparing
 * two different projection algorithms and calling the difference compression.
 *
 * The constraint set's targets are FROZEN at block construction (spec §3.5) — the SAME `set`
 * object is passed every step, never rebuilt from the current iterate.
 */
function runDescent(
    vertices0: Vec3[],
    edges: Edge[],
    dp: number[][],
    set: ConstraintSet,
    K: number,
    saddleFor: null | ((v: Vec3[]) => ReturnType<typeof makeHierarchicalSaddle>),
) {
    const base = {
        mode: 'analytical' as const,
        projectionMode: 'frozen' as const,
    };
    let v = vertices0;
    const e0 = calculateEnergy(v, edges, dp, ALPHA, BETA, EPS_A);
    let accepted = 0;
    let converged = false;
    const steps: Record<string, unknown>[] = [];
    for (let k = 0; k < K; k++) {
        const opts = saddleFor ? { ...base, saddle: saddleFor(v) } : base;
        const r = sobolevStepSet(v, edges, dp, set, opts);
        steps.push({
            k,
            accepted: r.accepted,
            converged: r.converged,
            energy: r.energy,
            tau: r.stats.tau,
            gradientL2Norm: r.stats.gradientL2Norm,
            projectionIterations: r.stats.projectionIterations,
            reason: r.stats.reason ?? null,
        });
        if (r.converged) {
            converged = true;
            break;
        }
        if (r.accepted) accepted++;
        v = r.vertices;
    }
    const eK = calculateEnergy(v, edges, dp, ALPHA, BETA, EPS_A);
    return {
        energyBefore: e0,
        energyAfter: eK,
        deltaETotal: e0 - eK,
        kRun: steps.length,
        accepted,
        acceptedFraction: steps.length ? accepted / steps.length : 0,
        converged,
        steps,
    };
}

/**
 * H4c's aggregates and verdict from WHATEVER rows the document holds — the {@link finaliseH2}
 * contract. Both conjuncts are evaluated over the merged rows: a shard that measured only n=240
 * must not publish a verdict that reads as if n=480 had passed.
 */
function finaliseH4c(block: Record<string, unknown>): void {
    const rows = ((block.rows ?? []) as Record<string, unknown>[])
        .slice()
        .sort((a, b) => (a.n as number) - (b.n as number));
    block.rows = rows;
    let worstRe = Number.POSITIVE_INFINITY;
    let anyShortRun = false;
    let anyNonPositiveReference = false;
    let allAccepted = true;
    for (const r of rows) {
        const dense = r.dense as Record<string, unknown>;
        const hier = r.hierarchical as Record<string, unknown>;
        worstRe = Math.min(worstRe, r.Re as number);
        if ((dense.kRun as number) < H4C_K || (hier.kRun as number) < H4C_K) anyShortRun = true;
        if (!((dense.deltaETotal as number) > 0)) anyNonPositiveReference = true;
        if (hier.acceptedFraction !== 1) allAccepted = false;
    }
    const measured = rows.length > 0;
    block.rungsMeasured = rows.map((r) => r.n);
    block.worstRe = measured ? worstRe : null;
    block.acceptedFractionConjunct = measured ? allAccepted : null;
    block.invalid = [
        ...(anyShortRun ? ['K_run < K'] : []),
        ...(anyNonPositiveReference ? ['deltaE_total(reference) <= 0'] : []),
    ];
    const verdict: Verdict = !measured
        ? 'NOT ESTABLISHED'
        : anyShortRun || anyNonPositiveReference
          ? 'INVALID'
          : worstRe >= H4C_RE_BAR && allAccepted
            ? 'GREEN'
            : 'RED';
    block.verdict = verdictOf(
        verdict,
        'GREEN',
        !measured
            ? 'no rung measured yet'
            : verdict === 'GREEN'
              ? `worst R_E ${worstRe.toFixed(5)} >= ${H4C_RE_BAR} with accepted fraction 1.0 on every rung (${rows.map((r) => r.n).join(',')})`
              : verdict === 'INVALID'
                ? 'a run was short of K steps, or the reference energy decrease was not positive'
                : `worst R_E ${worstRe.toFixed(5)} against the bar ${H4C_RE_BAR}, accepted-fraction conjunct ${allAccepted}`,
    );
}

function gateH4c(ns: number[]) {
    const rows: Record<string, unknown>[] = [];

    for (const n of ns) {
        const { vertices, edges } = trefoil(n);
        const dp = calculateDisjointPairs(edges);
        // The UNPERTURBED barycenter: the constraint is satisfied at step 0, so the projection is
        // exercised without the step also having to travel to a displaced target. k=0 (the EMPTY
        // set) is deliberately NOT used — `[A36 item 3]` makes it the invalid choice on a
        // translation-singular set, and `lineSearch.ts:193` records that the empty set "converges
        // trivially at iteration 0", i.e. it would exercise no projection at all.
        const set: ConstraintSet = [barycenterBlock(barycenterTarget(vertices, edges))];

        const dense = runDescent(vertices, edges, dp, set, H4C_K, null);
        const hier = runDescent(vertices, edges, dp, set, H4C_K, (v) =>
            makeHierarchicalSaddle(v, edges, calculateDisjointPairs(edges), {
                bct: { theta: 0.5 },
            }),
        );
        const Re = dense.deltaETotal !== 0 ? hier.deltaETotal / dense.deltaETotal : Number.NaN;

        rows.push({
            fixture: provenance('trefoil', vertices, n),
            n,
            K: H4C_K,
            theta: 0.5,
            operator: 'f64 on BOTH arms',
            constraintSet: 'one barycenter block, target frozen at construction',
            projectionMode: 'frozen on BOTH arms',
            dense,
            hierarchical: hier,
            Re,
            acceptedFractionHierarchical: hier.acceptedFraction,
            acceptedFractionDense: dense.acceptedFraction,
            conjunct1: Re >= H4C_RE_BAR,
            conjunct2: hier.acceptedFraction === 1,
        });
    }

    // ---- Conjunct 1's RED side: a LABELLED SYNTHETIC KNOCK-OUT (controller Stage-1 ruling 9).
    //
    // THE FIRST KNOCK-OUT TRIED DID NOT FIRE, and why it didn't is a finding in its own right.
    // θ=4 (deliberate over-compression) returned R_E = 0.99875 — indistinguishable from
    // production. MEASURED cause (`scratch-11/kocheck.ts`): **compression SATURATES in θ.**
    // `nFar` is pinned at 274 and `nearPairs` at 7128 for EVERY θ >= 4 at n=240, because raising θ
    // converts would-be-admissible blocks into ADJACENCY-VETOED near blocks rather than far ones
    // (`adjacencyVetoed` 50 -> 196), so `δ_compress` only moves 9.24e-3 -> 1.568e-2 (1.7x) between
    // θ=0.5 and θ=50. **θ is not a dial for this gate**, and the adjacency certificate is the floor
    // on the operator's accuracy. Recorded below as `thetaSaturation`, not discarded.
    //
    // The knock-out that DOES fire is the kernel-exponent DRIFT this factory's own TSDoc warns
    // about: *"THE CALLER MUST NOT LET THESE DRIFT FROM THE STEP'S ... NOTHING downstream can see
    // the disagreement — the residual is self-consistent and nothing thresholds it"*
    // (`hierarchicalSaddle.ts:104-120`). It is production-REACHABLE, it trips no guard, and it
    // produces an operator that is not the Ā of the step consuming it.
    // IT IS A REACHABILITY DEMONSTRATION, NOT A MEASUREMENT OF THE REAL RED SIDE.
    const koN = ns[0];
    let knockOut: Record<string, unknown> | null = null;
    if (koN !== undefined) {
        const { vertices, edges } = trefoil(koN);
        const dp = calculateDisjointPairs(edges);
        const set: ConstraintSet = [barycenterBlock(barycenterTarget(vertices, edges))];
        const dense = runDescent(vertices, edges, dp, set, H4C_K, null);
        const saddleWith = (o: { alpha?: number; theta?: number }) => (v: Vec3[]) =>
            makeHierarchicalSaddle(v, edges, dp, {
                bct: { theta: o.theta ?? 0.5 },
                ...(o.alpha === undefined ? {} : { alpha: o.alpha }),
            });
        const re = (d: { deltaETotal: number }): number =>
            dense.deltaETotal !== 0 ? d.deltaETotal / dense.deltaETotal : Number.NaN;
        // The crossing is BRACKETED rather than asserted at one point, so the demonstration says
        // how far the bar is from the production configuration rather than just "it can be red".
        const ladder = [2.5, 2, 1.5].map((alpha) => {
            const r = runDescent(vertices, edges, dp, set, H4C_K, saddleWith({ alpha }));
            return {
                alpha,
                Re: re(r),
                acceptedFraction: r.acceptedFraction,
                kRun: r.kRun,
                red: re(r) < H4C_RE_BAR,
            };
        });
        const thetaSat = runDescent(vertices, edges, dp, set, H4C_K, saddleWith({ theta: 4 }));
        const firing = ladder.find((l) => l.red) ?? null;
        knockOut = {
            label: 'SYNTHETIC KNOCK-OUT — A REACHABILITY DEMONSTRATION, NOT A MEASUREMENT',
            n: koN,
            dial: 'alpha drift (the kernel exponent the saddle factory is built with, against the step\u2019s own DEFAULTS.alpha = 3)',
            dialSource:
                'hierarchicalSaddle.ts:104-120 — the factory\u2019s own documented trap, production-reachable and unguarded',
            ladder,
            firesAt: firing,
            demonstratesRedSideReachable: firing !== null,
            conjunct2StaysGreen: ladder.every((l) => l.acceptedFraction === 1),
            windowNote:
                'alpha = 1.0 THROWS in prepareSchur (GMRES prepare reached eta=6.1e-6 against its tolerance), so the reachable knock-out window is alpha in (1, 2): below it a downstream guard catches the configuration, above it R_E stays over the bar.',
            thetaSaturation: {
                label: 'RECORDED FINDING — theta is NOT a dial for this gate',
                theta: 4,
                Re: re(thetaSat),
                red: re(thetaSat) < H4C_RE_BAR,
                why: 'MEASURED (scratch-11/kocheck.ts, n=240): nFar is pinned at 274 and nearPairs at 7128 for EVERY theta >= 4, because raising theta converts would-be-admissible blocks into ADJACENCY-VETOED near blocks (adjacencyVetoed 50 -> 196) rather than far ones. delta_compress saturates at 1.568e-2 (1.7x the theta=0.5 value) and goes no further. The adjacency certificate is the floor on the operator accuracy, so no theta can over-compress this operator.',
            },
            spreadFinding:
                'CONJUNCT 1 IS A WEAK GATE. Across operators spanning the exact dense A, theta=0.5, theta=4 and epsilon drift 1e-10 -> 1e-2, R_E spans only 0.9987-1.0034 (MEASURED, scratch-11/koc.ts): the line search accepts tau=1 on every step and R_E is near 1 for any roughly-downhill direction. The bar at 0.95 is reached only by an exponent drift of a whole 1.5, and alpha=2 lands at 0.9652 — 0.0152 from RED.',
        };
    }

    const block: Record<string, unknown> = {
        rowKey: 'n',
        bar: `R_E = deltaE_total(hierarchical) / deltaE_total(dense f64) >= ${H4C_RE_BAR} AND accepted fraction == 1.0`,
        barSource: 'spec §7.7',
        valueKey: 'rows[].Re and rows[].acceptedFractionHierarchical',
        operator: 'f64',
        operatorReason:
            '[A46 item 4]: makeHierarchicalSaddle admits NO precision variant and calls prepareA with none (hierarchicalSaddle.ts:321 => F64 default), so §7.12’s precision.operator "f32" is false for H4c; widening HierarchicalSaddleOptions is a hard-gate change (#96)',
        rows,
        knockOut,
        conjunct1RedSide: {
            status: 'NOT ESTABLISHED',
            reason: '§7.7’s own box: the 6/6 random arms it would have to be read from were accumulated over K_run = 1-10 against a 20-step reference, which THIS GATE’S OWN INVALID clause (K_run < K) forbids folding into a ratio. So the real RED side has never been measured — only the synthetic knock-out above shows the comparison responds.',
        },
        limitations: [
            'Both arms are f64 OPERATOR; H4c therefore says nothing about f32 end-to-end descent.',
            'The gated quantity is a RATIO of accumulated energy decrease; §7.7 records that tau measures step SCALE, not quality, so an arm can score well with a worse direction and a longer step.',
            'The synthetic knock-out is a reachability demonstration of the verdict path, NOT evidence about the production operator.',
        ],
    };
    finaliseH4c(block);
    return block as typeof block & { verdict: GateVerdict };
}

// ---------------------------------------------------------------------------------------------
// H5 — spec §7.8. Emitted from the SHIPPED TS instrument, per `[A46 item 4]`: `precond.test.ts`
// already asserts every property of this gate on the shipped operator, and sending it back to
// Python would re-measure the LAB `pinv` operator whose figures `[A32]` and `[A38 item 9(a)]`
// withdrew.
// ---------------------------------------------------------------------------------------------

/** `out = A_σ u` on ONE coordinate block — `precond.test.ts`'s `applyASigma`, scalar form. */
function applyASigmaScalar(
    H: HierarchicalA,
    sigma: number,
    n: number,
    u: ArrayLike<number>,
    out: Float64Array,
): void {
    applyAScalar(H, u, out);
    let s = 0;
    for (let i = 0; i < n; i++) s += u[i];
    const shift = sigma * (s / n);
    for (let i = 0; i < n; i++) out[i] += shift;
}

/**
 * The spectrum of `M A_σ` for symmetric `M`, via the similarity `Rᵀ M R` with `A_σ = R Rᵀ`.
 * `precond.test.ts:317-320` verbatim. The Cholesky is ALSO the SPD check on `A_σ`: at θ=0.25 the
 * BCT approximation could in principle lose definiteness, and this would throw.
 */
function kappaOf(M: Float64Array, Asig: Float64Array, n: number) {
    const R = cholesky(Asig, n);
    return spectrum(matmul(transpose(R, n), matmul(M, R, n), n), n);
}

/**
 * H5's aggregates and verdict from WHATEVER rows the document holds — same sharding contract as
 * {@link finaliseH2}. The `kappa_excl` exponent rows are recomputed here too, per θ, so a merged
 * ladder fits one line rather than one line per shard.
 */
function finaliseH5(block: Record<string, unknown>): void {
    const all = (block.rows ?? []) as Record<string, unknown>[];
    const measured = all
        .filter((r) => typeof r.kappaExcl === 'number')
        .sort(
            (a, b) =>
                (a.theta as number) - (b.theta as number) || (a.n as number) - (b.n as number),
        );
    const invalid = new Set<string>();
    let worstKappaExcl = 0;
    let worstP3 = 0;
    for (const r of measured) {
        // THE PRECONDITION OF `kappa_excl`, NOT A SANITY CHECK ([A32 item 6], #71):
        // "kappa_excl := lambda_max/lambda_2 excluding the constant mode" is only that while the
        // constant mode IS the smallest eigenvalue. MEASURED counter-regime: at mhat=100
        // lambda_1=0.7486 and lambda_2=0.7652 are BOTH below 1, two non-constant modes sit under
        // the constant one, and the quantity silently stops meaning what it is named.
        if (!r.constantModePinned) invalid.add('lambda1 != 1 (constant mode not pinned)');
        if (!r.secondEigenvalueAboveOne) {
            invalid.add('lambda2 <= 1.5 (kappa_excl is not the named quantity)');
        }
        // H5 property 1's INVALID clause, as `lowOrder.test.ts:802-812` states it.
        if (r.nullDim !== 0) invalid.add('nullDim(Atilde^-1 A_sigma) != 0');
        worstKappaExcl = Math.max(worstKappaExcl, r.kappaExcl as number);
        worstP3 = Math.max(worstP3, r.kappaInclTimesUF32 as number);
    }
    // LOGGED, NOT GATED, and deliberately: §7.8 records that the |p| <= 0.05 exponent bar reads RED
    // on this operator AND on the lab reference under the excluded-constant definition, which is
    // why [A32] states property 2 as a BOUND rather than a slope.
    const exponents: Record<string, unknown>[] = [];
    for (const theta of THETAS_GATED) {
        const rung = measured.filter((r) => r.theta === theta);
        if (rung.length < 2) continue;
        exponents.push({
            theta,
            rungs: rung.map((r) => r.n),
            kappaExclExponentInN: fitExponent(
                rung.map((r) => r.n as number),
                rung.map((r) => r.kappaExcl as number),
            ),
            note: 'RECORDED, NOT GATED — [A32] states property 2 as a bound (kappa_excl <= 40), not as a slope; the |p| <= 0.05 form reads RED on this operator and on the lab reference',
        });
    }
    block.rows = measured;
    block.exponents = exponents;
    block.rungsMeasured = [...new Set(measured.map((r) => r.n))].sort(
        (a, b) => (a as number) - (b as number),
    );
    block.worstKappaExcl = measured.length ? worstKappaExcl : null;
    block.worstKappaInclTimesUF32 = measured.length ? worstP3 : null;
    block.invalid = [...invalid];
    const verdict: Verdict =
        measured.length === 0
            ? 'NOT ESTABLISHED'
            : invalid.size > 0
              ? 'INVALID'
              : worstKappaExcl < H5_KAPPA_EXCL_BAR && worstP3 < H5_P3_BAR
                ? 'GREEN'
                : 'RED';
    block.verdict = verdictOf(
        verdict,
        'GREEN',
        measured.length === 0
            ? 'no rung measured yet'
            : verdict === 'GREEN'
              ? `worst kappa_excl ${worstKappaExcl.toFixed(4)} < ${H5_KAPPA_EXCL_BAR} over theta ${THETAS_GATED.join(',')}, with the constant mode pinned at 1 and lambda_2 > 1.5 on every row`
              : verdict === 'INVALID'
                ? `the kappa_excl precondition or the nullity clause failed: ${[...invalid].join(', ')}`
                : `worst kappa_excl ${worstKappaExcl.toFixed(4)} or property 3 ${worstP3.toExponential(3)} reaches its bar`,
    );
}

function gateH5(ns: number[]) {
    const rows: Record<string, unknown>[] = [];
    for (const theta of THETAS_GATED) {
        for (const n of ns) {
            const cell = buildCell('trefoil', n, theta, ['f64']);
            const sp = kappaOf(
                densify(n, (u, out) => cell.P.applyScalar(u, out)),
                densify(n, (u, out) => applyASigmaScalar(cell.H64, cell.sigma64, n, u, out)),
                n,
            );
            const kappaIncl = sp.lambdaMax / sp.lambda1;
            const kappaExcl = sp.lambdaMax / sp.lambda2;
            rows.push({
                fixture: provenance('trefoil', cell.vertices, n),
                n,
                theta,
                extent: cell.extent,
                massWeight: cell.massWeight,
                massWeightFormula: `DEFAULT_MASS_WEIGHT_HAT / ext^2 = ${DEFAULT_MASS_WEIGHT_HAT}/${cell.extent}^2 (#72, [A20])`,
                lambda1: sp.lambda1,
                lambda2: sp.lambda2,
                lambdaMax: sp.lambdaMax,
                rankTol: sp.rankTol,
                nullDim: sp.nullDim,
                kappaIncl,
                kappaExcl,
                kappaInclTimesUF32: kappaIncl * U_F32,
                constantModePinned: Math.abs(sp.lambda1 - 1) < 1e-11 * sp.lambdaMax,
                secondEigenvalueAboveOne: sp.lambda2 > 1.5,
                provenanceNote:
                    'the SHIPPED operator, NOT e0-closer-0b.json\u2019s lab pinv column (1.1269e4 / 6.716e-4), whose figures [A38 item 9(a)] withdrew',
            });
        }
    }

    const block: Record<string, unknown> = {
        rowKey: 'n,theta',
        bar: `property 2 (PRIMARY): kappa_excl < ${H5_KAPPA_EXCL_BAR}; property 3 (RECORDED): kappa_incl * u_f32 < ${H5_P3_BAR}; property 1: nullDim == 0`,
        barSource:
            'spec §7.8 as amended by [A32] (supersedes [A22]); asserted at precond.test.ts:427-447, :527 and lowOrder.test.ts:802-812',
        valueKey: 'rows[].kappaExcl',
        instrument:
            'test/sobolev/hierarchical/denseSpectrum.ts — the SAME instrument the shipped gate uses ([A46 item 4]: sending H5 back to Python would re-measure the LAB pinv operator whose figures [A32] and [A38 item 9(a)] withdrew)',
        rows,
        limitations: [
            'FIXTURE-SCOPED: the K=40 bound must NOT be applied to another curve — at this mass weight kappa_excl reaches 138 (arc n=480) and 159 (helix n=960) against the trefoil\u2019s ~20 (§7.8).',
            'kappa_eig is the EIGENVALUE ratio, not cond_2: Atilde^-1 A_sigma is a product of two SPD matrices and is NOT symmetric, so singular values differ from eigenvalues — MEASURED 1.4544e4 vs 1.0977e4 at n=64, a 1.33x gap (§10.2).',
            'Property 3 is RECORDED, not a licence to solve in f32: a bounded condition number is necessary, not sufficient.',
            'The instrument is O(n^3): the shipped test records ~8 s at n=960, 46 s at n=1920 and 442 s at n=3840, which is why the ladder stops where it does.',
        ],
    };
    finaliseH5(block);
    return block as typeof block & { verdict: GateVerdict };
}

// ---------------------------------------------------------------------------------------------
// H6 — spec §7.9. NOT RUN HERE. TRANSCRIBED from task 0c's Python artefact, with `[A46 item 7]`'s
// withdrawal attached, because TS has no generalized symmetric-definite pencil eigensolver.
// ---------------------------------------------------------------------------------------------

function gateH6() {
    return {
        runBy: 'task 0c, Python (generalized symmetric-definite pencil)',
        runHere: false,
        notRunReason:
            'TypeScript has no generalized symmetric-definite pencil eigensolver; this is the ONE gate [A46 item 4] leaves in Python',
        valuesAreTranscribed: true,
        source: '.superpowers/sdd/2026-10-09-phase2h-bct-gmres-design/task-0c-report.md',
        rows: [
            { n: 960, theta: 0.5, h6BadModeCount: 21, h6CosMin: 0.78556, h6Applicable: true },
            { n: 2000, theta: 0.5, h6BadModeCount: 18, h6CosMin: 0.76109, h6Applicable: true },
            {
                n: 960,
                theta: 0.25,
                h6CosMin: 0.7649,
                h6Applicable: false,
                applicabilityReason:
                    'NOT APPLICABLE at theta=0.25 where |B| = 0 (§7.9’s own precedent)',
            },
            {
                n: 960,
                theta: 'exact (delta=0)',
                h6CosMin: 0.7644,
                note: 'THE CONTROL: the EXACT operator reads essentially the same as theta=0.5, which is what withdraws the RED as a theta signal',
            },
        ],
        verdictAsMeasuredByTask0c: 'RED',
        redWithdrawn: true,
        withdrawalRuling:
            '[A46 item 7]: theta=0.25 reads 0.7649, theta=0.5 reads 0.7856 and the EXACT delta=0 operator reads 0.7644 — theta is not the variable and switching it changes nothing. The theta=0.5 OPERATOR passes (min cosExact 0.9062 >= 0.9 at n=960); the RED comes from the STOPPED ITERATE (6-9 iterations against ~29-39 for random rhs, because the recurrence residual hits 1e-4 while the true error is still large), which is §5.3’s own ruling that the recurrence residual "is not a convergence criterion ... under-reports by 3.9e3-6.6e3x". THE RED IS WITHDRAWN AS A THETA SIGNAL.',
        consequence:
            'apply.test.ts:644’s comment "theta=0.25 is PRODUCTION after gate H6 read RED" rests on a withdrawn inference. The CHOICE of production theta stays open on #51’s evidence and is NOT row 11’s to make.',
        verdict: verdictOf(
            'RED',
            'RED',
            'RED as task 0c measured it, published WITH [A46 item 7]’s withdrawal of the RED as a theta signal. The verdict is transcribed, not re-measured here.',
        ),
    };
}

// ---------------------------------------------------------------------------------------------
// R1 — spec §7.10. The REPORTING obligation: three ceilings, with `NOT ESTABLISHED` published as
// a value wherever no gate binds (§7.10:2558 requires exactly that, and `[A38 item 10]` requires
// per-shape publication).
// ---------------------------------------------------------------------------------------------

function reportR1() {
    return {
        servedNF32Only: null,
        servedNF32OnlyStatus: 'NOT ESTABLISHED',
        servedNF32OnlyBoundBy: null,
        servedNF32OnlyReason:
            'the 0.7732 instrument that used to bind it is RETRACTED: [A40] scoped it away as a dense-f32 CAST rather than a property of the f32 operator. At the K_s operating point NOTHING binds below 30k ([A44] §1), and the stagnation route is one [A44]/[A42] §3 explicitly withdrew as a ceiling basis.',
        servedNF32OnlyOperatingPoints: [
            {
                name: 'K_s',
                isolates: 'f32 ARITHMETIC',
                bound: 'nothing binds below n = 5.5e4',
                status: 'clears the 30k target',
            },
            {
                name: 'stagnation',
                isolates: 'f32 OUTPUT STORAGE',
                bound: '6.4e3 - 2.4e4',
                status: 'BOUND-ONLY, NEVER AN INSTRUMENT — withdrawn as a ceiling basis by [A44] / [A42] §3',
            },
        ],
        servedNF32OnlyPerShape: {
            note: '[A38 item 10] requires per-shape publication. The per-shape WALL is unexecutable (A41 / #93 — the solve-level gates are closed-shape-only by A41’s five barriers, and the arc/helix numbers in the corpus are dense-A_sigma FACTORISATION figures for shapes this repo cannot solve). Row 11 therefore CITES [A43]/[A44]’s per-shape error-structure table rather than re-measuring a quantity that does not exist.',
            closedRing: { n: null, status: 'NOT ESTABLISHED', boundBy: null },
            arc: {
                n: null,
                status: 'NOT MEASURABLE',
                boundBy: 'A41 barrier (open shape, solve-level gate)',
            },
            helix: {
                n: null,
                status: 'NOT MEASURABLE',
                boundBy: 'A41 barrier (open shape, solve-level gate)',
            },
        },
        outputAccumulationMitigation: {
            measured:
                '6.5x / 10.7x / 23x improvement from an f64-grade output accumulator ([A45], row 10)',
            applicable: false,
            reason: '[A46 item 3] CORRECTS [A45]: the mitigation is measured on a scratch RE-IMPLEMENTATION (scratch-10/mutant.ts:33 makeTwoTermOp), and no code path in this repo can apply it on EITHER backend. AccumulatorVariant (nearField.ts:101-115) exposes round/roundNear/alloc/cast and NO output hook. Filed as utof/repulsive-test2#96.',
        },

        servedNReferenced: 960,
        servedNReferencedStatus: 'MEASURED',
        servedNReferencedBoundBy: 'the dense f64 reference’s own range (H4a)',
        servedNReferencedReason:
            'practical ceiling 960 from the f64 reference’s measured degradation at n=1920 (§7.5), with shipped-side corroboration: [A35] measures dense LDL^T’s own error crossing 1e-10 between n=480 and n=960. Hard ceiling N ~ 5793 from the 4N^2 binding cap (§1.2), which is NOT the binding constraint here.',

        servedNMixed: null,
        servedNMixedStatus: 'NOT ESTABLISHED',
        servedNMixedBlockedBy: [
            'flatness NOT ESTABLISHED: [A38 item 9] measures n^+0.15 ... n^+0.47, not the n^0.0010 the earlier claim rested on',
            '[A45] / [A46 item 3]: conditional on an f64-grade output accumulator that has NO code path on either backend (#96)',
        ],
        servedNMixedDischarged: [
            'the f32 apply blocker: discharged by 7b and row 10',
            'kappa_precond to n=4000 on four shapes with 720x headroom: discharged by [A38 item 9]',
        ],

        directionQualityAboveN960: {
            status: 'NOT CERTIFIED BY ANY GATE',
            reason: '[A46 item 1]: H4b′ is VACUOUS on the production matvec, and §7.6:2071-2077 proves no reference-free lower bound on cos_A from a single iterate exists. §7.1:1239 (H4b′ as the per-solve gate binding at 30k) and §7.11:2608 (H4b′ as the mitigation above n=960) are BOTH WITHDRAWN.',
            isOneNeeded: 'NOT ESTABLISHED',
            isOneNeededReason:
                'task 0c measured the direction RIGHT and the step length wrong by 1/lambda (2.75x at n=960, 3.46x at n=2000) — a step-scale input, not a direction defect. It stays benign only if the line search runs on those modes, which nobody has measured. utof/repulsive-test2#95.',
        },
    };
}

// ---------------------------------------------------------------------------------------------
// Host + constants block. REPLACES `adapterCert`, per `[A46 item 2]`.
// ---------------------------------------------------------------------------------------------

function hostBlock() {
    const c = cpus();
    return {
        runtime: 'bun',
        bunVersion: Bun.version,
        platform: process.platform,
        arch: process.arch,
        cpuModel: c[0]?.model ?? 'unknown',
        cpuCount: c.length,
        totalMemBytes: totalmem(),
        // `uptime`'s 1/5/15-minute load, RECORDED beside every run. This box shows 6-9x ONE-SIDED
        // degradation of a timing ratio under subagent load (memory `isolate-load-dependent
        // -instruments`), which is WHY no duration is published — not a reason to trust one.
        loadAvg: loadavg(),
        gpu: null,
        gpuReason:
            'there is no GPU anywhere in H1-H6: every gate in this set is a CPU f64/f32 measurement',
    };
}

function constantsBlock() {
    return {
        uF32: U_F32,
        uF32Definition: '2**-24 — UNIT ROUNDOFF, not machine epsilon 2**-23 ([A39 item 1])',
        uF64: U_F64,
        uF64Definition: '2**-53',
        alpha: ALPHA,
        beta: BETA,
        epsilon: EPS_A,
        leafSize: LEAF,
        seeds: SEEDS,
        thetaGatedAt: THETAS_GATED,
        thetaDefaultInCode: DEFAULT_THETA,
        thetaProduction: null,
        thetaProductionReason:
            'OPEN. [A46 item 7] withdrew H6’s RED as a theta signal, so apply.test.ts:644’s "0.25 is PRODUCTION after H6 read RED" rests on a withdrawn inference; the choice stays open on utof/repulsive-test2#51’s evidence and is NOT row 11’s to make. Row 11 gates at BOTH theta.',
        probeSchedule: DEFAULT_PROBE_SCHEDULE,
        maxitRule:
            'defaultMaxit(3n) = min(600, 2*dim) — spec §7.3:1401 pins it so the measured floors transfer literally',
        tauEta: TAU_ETA,
        precision: {
            operator:
                'per gate — see each gate’s `operator` / `operatorPair` field (schema gap G7: one scalar cannot describe gates that run on different operators)',
            outerAlgebra: 'f64',
            // Schema gap G2 / `[A46 item 3]`. This field can ONLY ever read "f32" and must carry
            // the reason, because the reason is the finding.
            outputAccumulation: 'f32',
            outputAccumulationConfigurable: false,
            outputAccumulationReason:
                'applyColumns writes out[o+i] = fr(out[o+i] +/- r) (apply.ts:620-627), so the shipped F32 operator IS [A45]’s OUT32 configuration. AccumulatorVariant (nearField.ts:101-115) exposes round/roundNear/alloc/cast and NO output hook, on EITHER backend — [A46 item 3] corrects [A45], which had scoped the limitation to WGSL and so implied a CPU path exists. Making it configurable is an exported-interface change: utof/repulsive-test2#96.',
        },
        clock: null,
        clockReason:
            'issue #63 and the task brief: this box shows 6-9x ONE-SIDED degradation of a timing ratio under subagent load, so a published duration would not be a property of the code. Cost is budgeted in APPLY COUNTS, which are load-independent integers (scratch-11/budget.json).',
        adapterCert: null,
        adapterCertReason:
            '[A46 item 2]: certifyAdapter’s fall-through returns classification "software" (bench/gpu/adapterCert.ts:117-122), so H1’s INVALID clause 4 CANNOT NOT FIRE on a CPU harness and §7.0:1219 would forbid publishing the run. A predicate that cannot not fire reads exactly like a broken harness. Replaced by the `host` block; the clause stays live and unchanged for any future GPU arm.',
        chiZeroSource: 'lists-theta0-precomputeNear (H2/H3/H4), stream-theta0 (H1’s reference)',
        chiZeroBitIdenticalToStream: true,
        chiZeroBitIdentityProof:
            '.superpowers/sdd/2026-10-09-phase2h-bct-gmres-design/scratch-7b/exactpath.ts — 0 of 1440 entries non-identical by Object.is, maxRel exactly 0, at n = 64/120/240/480 on two probe vectors; H1’s rows re-measure it on this run. Schema gap G5: §7.12’s chiZeroStreaming: true is a field every measured run contradicts.',
        etaExactVariant: 'f64',
        preRegistration:
            '.superpowers/sdd/2026-10-09-phase2h-bct-gmres-design/task-11-report.md §S1.2 — written before the first run that tests a bar',
    };
}

// ---------------------------------------------------------------------------------------------
// Driver. Each invocation MERGES its parts into the output file, so the campaign can be sharded
// across foreground runs (see the module TSDoc on why background jobs are forbidden).
// ---------------------------------------------------------------------------------------------

const DATE = '2026-10-12';
const DEFAULT_OUT = `bench/results/${DATE}-gpu-phase2h-gate-calibration.json`;

const ALL_GATES = ['h1', 'h2', 'h3', 'h4a', 'h4c', 'h5', 'h6', 'r1'] as const;

/**
 * Per-gate re-aggregation after a shard merge. A gate listed here may be split across foreground
 * invocations; its verdict is ALWAYS recomputed over the merged rows, never left as the verdict of
 * the last shard. A gate NOT listed here must be produced by a single invocation.
 */
const FINALISERS: Record<string, (block: Record<string, unknown>) => void> = {
    H2: finaliseH2,
    H3: finaliseH3,
    H4a: finaliseH4a,
    H4bPrime: finaliseH4b,
    H4c: finaliseH4c,
    H5: finaliseH5,
};

function arg(name: string): string | undefined {
    const hit = process.argv.find((a) => a.startsWith(`--${name}=`));
    return hit?.slice(name.length + 3);
}

const numList = (v: string | undefined, fallback: number[]): number[] =>
    v ? v.split(',').map(Number) : fallback;

/** The identity of a row, from the gate block's own `rowKey` (e.g. `"n"`, `"n,seed"`). */
function rowId(row: Record<string, unknown>, rowKey: string): string {
    return rowKey
        .split(',')
        .map((k) => JSON.stringify(row[k.trim()]))
        .join('|');
}

/**
 * Merges a freshly measured gate block into whatever the document already holds.
 *
 * Rows are keyed by the block's `rowKey` and the NEW row wins on a collision — a re-run of the same
 * cell is a re-measurement, not a duplicate. `arms` objects are merged key-wise so `--arms=f32` and
 * `--arms=f64` can be separate invocations of the same rung. Everything else (bars, provenance,
 * limitations) comes from the new block, because it was produced by the current code.
 */
function mergeBlock(
    prev: Record<string, unknown> | undefined,
    next: Record<string, unknown>,
): Record<string, unknown> {
    const rowsField = (next.rowsField as string) ?? 'rows';
    const rowKey = next.rowKey as string | undefined;
    if (!prev || !rowKey) return next;
    const prevRows = (prev[rowsField] ?? []) as Record<string, unknown>[];
    const nextRows = (next[rowsField] ?? []) as Record<string, unknown>[];
    const byId = new Map<string, Record<string, unknown>>();
    for (const r of prevRows) byId.set(rowId(r, rowKey), r);
    for (const r of nextRows) {
        const id = rowId(r, rowKey);
        const old = byId.get(id);
        if (old?.arms && r.arms) {
            // Arm-sharded rung: keep the arms the other invocation measured, and inside a shared
            // arm, union the SEEDS — `--seeds` shards a single (rung, arm) cell across
            // invocations, which is the only way n=960 fits the foreground cap. The per-arm
            // aggregates are deliberately NOT merged here: `finaliseH2` recomputes them from the
            // merged `seeds`, because a stored `etaBctFloorMax` is a per-SHARD maximum and
            // publishing it as the cell's would understate the gated value.
            const oldArms = old.arms as Record<string, Record<string, unknown>>;
            const newArms = r.arms as Record<string, Record<string, unknown>>;
            for (const [armName, oldArm] of Object.entries(oldArms)) {
                const newArm = newArms[armName];
                if (!newArm) {
                    newArms[armName] = oldArm;
                    continue;
                }
                const bySeed = new Map<number, unknown>();
                for (const sd of (oldArm.seeds ?? []) as { seed: number }[]) {
                    bySeed.set(sd.seed, sd);
                }
                for (const sd of (newArm.seeds ?? []) as { seed: number }[]) {
                    bySeed.set(sd.seed, sd);
                }
                newArm.seeds = [...bySeed.entries()].sort((x, y) => x[0] - y[0]).map(([, v]) => v);
            }
        }
        byId.set(id, r);
    }
    next[rowsField] = [...byId.values()];
    return next;
}

function main(): void {
    const gatesArg = arg('gates') ?? 'meta';
    const gates =
        gatesArg === 'all'
            ? [...ALL_GATES]
            : gatesArg
                  .split(',')
                  .map((g) => g.trim())
                  .filter(Boolean);
    const out = arg('out') ?? DEFAULT_OUT;
    const nsArg = arg('n');
    const ns = nsArg ? nsArg.split(',').map(Number) : null;
    const seeds = numList(arg('seeds'), SEEDS);
    const armsArg = arg('arms');
    const arms = (armsArg ? armsArg.split(',').map((a) => a.trim()) : ['f32', 'f64']) as Arm[];

    const doc: Record<string, unknown> = existsSync(out)
        ? JSON.parse(readFileSync(out, 'utf8'))
        : {
              schema: 'phase2h-gate-calibration/1',
              schemaSource:
                  "docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md 7.12 as amended by A46 item 5 (per-gate fixture provenance, every bar computed from the run's own counts, kappa_far defined)",
              spec: 'docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md',
              specRow: '11.2 row 11',
              generatedBy: 'bench/gpu/hCalibration.ts',
              parts: [],
              shards: [],
              gates: {},
              verdicts: {},
          };
    doc.date = DATE;
    // `host` and `constants` are rewritten on EVERY invocation, so `meta` is always a part of the
    // document whether or not it was named on the command line. Listing it conditionally would
    // read as "the meta block is missing from this artefact".
    doc.host = hostBlock();
    doc.constants = constantsBlock();
    const gateBlocks = doc.gates as Record<string, unknown>;
    const verdicts = doc.verdicts as Record<string, unknown>;
    const parts = new Set<string>((doc.parts as string[]) ?? []).add('meta');
    const shards = (doc.shards as unknown[]) ?? [];

    const put = (key: string, block: Record<string, unknown>): void => {
        const merged = mergeBlock(gateBlocks[key] as Record<string, unknown> | undefined, block);
        // Re-aggregate over the MERGED rows. Without this a sharded campaign publishes the last
        // shard's verdict, which is the whole reason measurement and verdict are separate here.
        FINALISERS[key]?.(merged);
        gateBlocks[key] = merged;
        verdicts[key] = merged.verdict;
    };

    for (const g of gates) {
        if (g === 'meta') {
            parts.add('meta');
            continue;
        }
        if (g === 'h1') {
            put('H1a', gateH1a(ns ?? [960]));
            put('H1b', gateH1b(ns ?? [960]));
        } else if (g === 'h2') {
            put('H2', gateH2(ns ?? [64, 120, 240, 480, 960], arms, seeds));
        } else if (g === 'h3') {
            put('H3', gateH3(ns ?? [240, 480, 960, 1920], seeds));
        } else if (g === 'h4a') {
            const r = gateH4aAndH4b(ns ?? [64, 240, 480, 960], seeds);
            put('H4a', r.h4a);
            put('H4bPrime', r.h4b);
        } else if (g === 'h4c') {
            put('H4c', gateH4c(ns ?? [240, 480]));
        } else if (g === 'h5') {
            put('H5', gateH5(ns ?? [64, 120, 240, 480, 960]));
        } else if (g === 'h6') {
            put('H6', gateH6());
        } else if (g === 'r1') {
            doc.r1 = reportR1();
        } else {
            throw new Error(
                `hCalibration: unknown gate "${g}"; known: meta,${ALL_GATES.join(',')},all`,
            );
        }
        parts.add(g);
    }

    shards.push({ gates, n: ns, seeds, arms, loadAvg: loadavg() });
    doc.shards = shards;
    doc.parts = [...parts].sort();
    // Publishability, per 7.0:1219 as amended by [A46 item 2]: the adapter clause is NOT APPLICABLE
    // on this arm, so the only thing that can block publication is a gate's own INVALID set.
    doc.publishable = Object.values(verdicts).every(
        (v) => (v as GateVerdict).verdict !== 'INVALID',
    );
    doc.unpredicted = Object.entries(verdicts)
        .filter(([, v]) => !(v as GateVerdict).matchesPreRegistration)
        .map(([k, v]) => ({
            gate: k,
            preRegistered: (v as GateVerdict).preRegistered,
            measured: (v as GateVerdict).verdict,
            // An unpredicted verdict carries NO falsification weight against the pre-registration
            // and must be read as a new, un-pre-registered observation (the task brief's rule).
            weight: 'NOT PRE-REGISTERED — carries no falsification weight',
        }));

    writeFileSync(out, `${JSON.stringify(doc, null, 2)}\n`);
    console.log(`[hCalibration] parts=${(doc.parts as string[]).join(',')} -> ${out}`);
    for (const [k, v] of Object.entries(verdicts)) {
        const gv = v as GateVerdict;
        console.log(
            `  ${k.padEnd(10)} ${gv.verdict.padEnd(16)} (pre-registered ${gv.preRegistered}${gv.matchesPreRegistration ? '' : '  <-- UNPREDICTED'})`,
        );
    }
    console.log(`  publishable=${doc.publishable}`);
}

main();
