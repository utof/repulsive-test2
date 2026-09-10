// test/gpu/cr.test.ts — gate CR: the [DESIGN §4.1] reformulation is EXACT against the
// core's own saddle solve, proven on a CPU f64 Cholesky BEFORE any WGSL exists. CR red ⇒
// no GPU work starts ([DESIGN §6]); localise with `oracle/tpe_phase2a_reformulation.py`
// (the numpy twin) before touching `src/gpu/saddle.ts`.
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §5 (CR), §6, §7, §8
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-gate-calibration.md §F, §J.3
import { expect, test } from 'bun:test';
import { execSync } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { trefoil } from '../../src/core/fixtures';
import { DEFAULTS } from '../../src/core/optimizer';
import {
    barycenterBlock,
    evaluateConstraintSet,
    pointBlock,
    totalLengthBlock,
} from '../../src/core/sobolev/constraintSet';
import { assembleAFlat } from '../../src/core/sobolev/innerProduct';
import { flatten } from '../../src/core/sobolev/layout';
import { solveSaddleFromA, solveSaddleFrozen } from '../../src/core/sobolev/linsolve';
import { calculateDisjointPairs, gradientAnalytical } from '../../src/core/tangentPointEnergy';
import { type Edge, testConfigs, type Vec3 } from '../../src/core/testConfigs';
// EVERY symbol `measure` uses, including saddleSetup and schurWoodburySolve. A missing
// name here is not a nit: `bun test` transpiles without typechecking, so it surfaces as a
// ReferenceError inside measure() that fails ALL TWELVE fixture tests at once — which the
// gate's own reporting rule reads as "CR red ⇒ stop Phase 2".
import {
    asExternal,
    choleskyF64,
    type IrTrace,
    kappa1of3,
    makePreparedSaddle,
    saddleSetup,
    schurWoodburySolve,
    shiftSobolev,
    woodburyM,
} from '../../src/gpu/saddle';

const U_F64 = 2 ** -53;
/** Repo-relative, for the `kappaArtifact` provenance key in the emitted JSON. */
const K1CAL = 'bench/results/2026-09-04-gpu-phase2a-k1-calibration.json';
/**
 * The same artifact resolved relative to THIS FILE, for reading. Not the bare
 * repo-relative string: `readFileSync` resolves that against the CWD, so it would only
 * work when `bun test` is invoked from the repo root. Same idiom the repo already uses
 * for oracle fixtures (`test/sobolev/constraintSetFlow.test.ts:61-65`, `loadFixture`).
 */
const K1CAL_URL = new URL(`../../${K1CAL}`, import.meta.url);

let kappaCache: Record<number, number> | null = null;

/**
 * κ₂(K) per N, READ from the committed [K1CAL] artifact — never transcribed.
 * [CAL]'s binding rule is that no gate number lives in prose unless a committed script
 * emitted it into a committed JSON; a literal table here would be that same failure one
 * level down, going stale silently if the calibration is ever re-run.
 *
 * LAZY, and that is load-bearing: read at module scope, a missing or moved artifact
 * throws during IMPORT, which aborts the whole file before any `test()` registers — so
 * the artifact-presence test below would never run and the failure would surface as an
 * unrelated import error. Deferring the read into the tests makes "the JSON path is
 * asserted to exist" actually true.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-gate-calibration.md §F, §A
 */
function kappaByN(): Record<number, number> {
    if (kappaCache) return kappaCache;
    if (!existsSync(K1CAL_URL))
        throw new Error(
            `CR: the [K1CAL] artifact is missing at ${K1CAL_URL.pathname} — without it every ` +
                'bar would silently fall back to the 1e-12 floor and CR would gate at the wrong level',
        );
    const parsed = JSON.parse(readFileSync(K1CAL_URL, 'utf8')) as {
        rows: { n: number; kappaK: number }[];
    };
    const out: Record<number, number> = {};
    for (const r of parsed.rows) out[r.n] = r.kappaK;
    kappaCache = out;
    return out;
}

// Fail loudly if the artifact moved or its schema changed: without this, every bar
// would silently fall back to the 1e-12 floor and CR would gate on the wrong level.
// This test is the FIRST in the file, so a missing artifact reports as this named test
// failing with the message above rather than as twelve unexplained fixture failures.
test('CR: the [K1CAL] κ₂(K) artifact is present and populated', () => {
    const kappa = kappaByN();
    expect(Object.keys(kappa).length).toBeGreaterThanOrEqual(5);
    expect(kappa[960]).toBeGreaterThan(1e6);
});

/**
 * Bar = max(1e-12, 10·κ₂(K,N)·u_f64) ([CAL §F]). N=8 and N=60 have no [K1CAL] row (its
 * fixture list is 64/120/240/480/960 — 64, not 60), so they take the 1e-12 floor, which
 * is what the floor is for. Do NOT "fix" this by remapping 60 onto the n=64 row: those
 * are different systems and the floor is the pre-registered fallback.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-gate-calibration.md §F
 */
function crBar(n: number): number {
    const kappa = kappaByN()[n];
    return kappa === undefined ? 1e-12 : Math.max(1e-12, 10 * kappa * U_F64);
}

/**
 * γ^q — a deterministic displacement off the constraint manifold. Φ is evaluated at the
 * fixture's own x0/L0, so a projection rhs taken AT γ₀ would be roundoff-scale — a
 * vacuous gate. The frozen projection solves exactly this shape: K(γ₀) frozen, Φ
 * evaluated at the MOVED iterate (`src/core/sobolev/lineSearch.ts:257`, `negPhi`, passed
 * at `:267`). Drawn from {@link uniforms}, not numpy's `default_rng(0)` (PCG64 + a
 * ziggurat, not reproducible in JS) and not `Math.sin` (ECMA-262 §21.3.2
 * implementation-approximated). CR must be re-runnable to the bit, and this value feeds
 * a GATED row.
 * @see docs/superpowers/plans/2026-09-04-webgpu-solver-phase2a-1.md (## Decisions, D6)
 */
function displaced(vertices: Vec3[]): Vec3[] {
    // Seed 7777, 3n values, consumed coordinate-major per vertex. NOT Math.sin: ECMA-262
    // §21.3.2 leaves Math.sin implementation-approximated, and this feeds a GATED row.
    const u = uniforms(vertices.length * 3, 7777);
    return vertices.map((v, i) => [
        v[0] + 1e-3 * u[3 * i],
        v[1] + 1e-3 * u[3 * i + 1],
        v[2] + 1e-3 * u[3 * i + 2],
    ]) as Vec3[];
}

/**
 * splitmix32 — the ensemble PRNG. NOT numpy (`np.random.default_rng(0)` is PCG64 + a
 * ziggurat and is not reproducible in TypeScript) and NOT `Math.random` (not seedable, so
 * CR would stop being re-runnable to the bit). Ten lines, fully specified, and portable to
 * Python in about five — `oracle/tpe_phase2a_reformulation.py` does exactly that, which is
 * what lets the numpy twin cross-check these rows rhs-for-rhs.
 * @see docs/superpowers/plans/2026-09-04-webgpu-solver-phase2a-1.md (## Decisions, D5)
 */
function splitmix32(seed: number): () => number {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x9e3779b9) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 16), 0x21f0aaad) >>> 0;
        t = Math.imul(t ^ (t >>> 15), 0x735a2d97) >>> 0;
        return ((t ^ (t >>> 15)) >>> 0) / 4294967296;
    };
}

/**
 * Uniform draws on [−1,1) from splitmix32 — the ensemble's vectors, and (at seed 7777)
 * the displacement {@link displaced} applies.
 * NOT Box–Muller normals: that needs `Math.log` and `Math.cos`, and ECMA-262 §21.3.2
 * leaves `Math.log` / `Math.sin` / `Math.cos` IMPLEMENTATION-APPROXIMATED — not required
 * to agree bit-for-bit across engines or across versions of one engine. The ensemble's
 * pre-registered failure count would then be engine-dependent, and an executor on a
 * different Bun build would escalate a non-issue (measured: the same seeds gave 2
 * failures under Box–Muller vs 1 under uniforms, while every gated row stayed
 * bit-identical). Everything here — `Math.imul`, XOR, shifts, `/ 2**32`, `2*u - 1` — is
 * exactly specified, so these rows are reproducible to the bit anywhere, Python included.
 * The marginal distribution is not load-bearing: the ensemble exists to excite `A`'s
 * badly-conditioned eigendirections, which a uniform vector does as well as a normal one.
 * @see docs/superpowers/plans/2026-09-04-webgpu-solver-phase2a-1.md (## Decisions, D5, D6)
 */
function uniforms(count: number, seed: number): number[] {
    const u = splitmix32(seed);
    const out: number[] = [];
    while (out.length < count) out.push(2 * u() - 1);
    return out;
}

/**
 * One measured rhs of one fixture — the row schema of `bench/results/gpu-phase2a-cr.json`,
 * which is [CAL §J.3]'s owed deliverable and [DESIGN §8]'s owed IR-convergence-curve schema.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §5, §7, §8
 */
interface CrRow {
    name: string;
    n: number;
    k: number;
    rhsShape: 'gradient' | 'projection';
    /** `production` | `nullspace1` | `rand0`…`rand4` — the row's identity within a fixture. */
    rhsTag: string;
    /**
     * Whether the bar is ASSERTED on this row. The three physically-reachable rhs are
     * gated; the ten seeded draws are recorded only, because one of them exceeds the bar
     * at n=960 with a CORRECT f64 factor and gating them would be a false kill.
     */
    gated: boolean;
    rhsSource: string;
    rhsNorm: number;
    /** [DESIGN §5]'s gated quantity: steps 1–3 ONLY (schurWoodburySolve, pre-refinement). */
    relErr: number;
    /** Steps 1–4 (PreparedSaddle.solve, post-refinement). Recorded, NOT gated. */
    relErrWithIr: number;
    bar: number;
    /** bar / relErr — [CAL §J.3]'s committed deliverable, emitted rather than hand-derived. */
    slack: number;
    /** [DESIGN §8]'s IR convergence curve, per N and per rhs: how many corrections step 4 */
    /** actually applied, and the relative-residual curve it walked. 0 ⇒ z₀ already cleared */
    /** SADDLE_IR_TOL, i.e. relErr and relErrWithIr are the same vector. */
    irRefinements: number;
    irRelResiduals: number[];
    kappaSource: string;
    verdict: 'PASS' | 'FAIL';
    diagM: number[];
    kappaM1: number;
    minDiagLSq: number;
}

const rows: CrRow[] = [];

/**
 * All thirteen rhs of one fixture × one constraint set, measured against the core's own
 * `solveSaddleFromA`. ONE f64 Cholesky, ONE `prepared.factor(C)` and ONE reference LDLᵀ
 * serve all thirteen — the reference factor is reused through the already-exported
 * `solveSaddleFrozen` (`src/core/sobolev/linsolve.ts:932-955`), which is what makes the
 * N=960 fixture affordable (its LDLᵀ of the 2884×2884 K dominates the runtime).
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §5 (CR)
 */
function measure(name: string, vertices: Vec3[], edges: Edge[], withPin: boolean): CrRow[] {
    const n = vertices.length;
    const dp = calculateDisjointPairs(edges);
    const { alpha, beta, epsilon } = DEFAULTS;
    const A = assembleAFlat(vertices, edges, dp, alpha, beta, epsilon);
    const x0: Vec3 = [0, 0, 0];
    for (const v of vertices) {
        x0[0] += v[0] / n;
        x0[1] += v[1] / n;
        x0[2] += v[2] / n;
    }
    const L0 = edges.reduce(
        (s, [a, b]) =>
            s +
            Math.hypot(
                vertices[a][0] - vertices[b][0],
                vertices[a][1] - vertices[b][1],
                vertices[a][2] - vertices[b][2],
            ),
        0,
    );
    const set = withPin
        ? [barycenterBlock(x0), totalLengthBlock(L0), pointBlock(0, vertices[0])]
        : [barycenterBlock(x0), totalLengthBlock(L0)];
    const { C } = evaluateConstraintSet(set, vertices, edges);
    const m = 3 * n;
    const k = C.length;

    // The reformulation side: shift, f64 Cholesky, ONE setup and ONE factor for both rhs
    // shapes. `setup` and `M` are the SAME values production computes inside factor()
    // (they are the exported helpers factor() itself calls), used here for two things: the
    // steps-1–3 solve that IS the gate's quantity, and the diag(M) / κ₁(M) margin the
    // JSON records.
    const { Asigma, sigma } = shiftSobolev(A, n);
    const L = choleskyF64(Asigma, n);
    const setup = saddleSetup(L, n, C);
    const M = woodburyM(setup.W, sigma, n);
    // onTrace is what makes [DESIGN §8]'s IR curve readable from the artifact. Collected
    // into an ARRAY, not a `let … | null`: TS does not narrow a `let` assigned inside a
    // callback (microsoft/TypeScript#9998) and every `trace?.x` would be a TS2339 on
    // `never`. One entry is pushed per ext.solve() call, in call order.
    const traces: IrTrace[] = [];
    const ext = asExternal(
        makePreparedSaddle(A, L, sigma, n, (t) => {
            traces.push(t);
        }).factor(C),
    );

    const zeroM = new Array<number>(m).fill(0);
    const zeroK = new Array<number>(k).fill(0);
    // rhs (a): the gradient shape [b; 0] — exactly what the frozen constrained-gradient
    // solve passes (`src/core/sobolev/gradient.ts:133`, `flatten(dE)` as rhsTop).
    const b = flatten(gradientAnalytical(vertices, edges, dp, alpha, beta, epsilon));
    // rhs (b): the projection shape [0; d], d = −Φ(γ^q) — exactly what the frozen
    // projection passes (`src/core/sobolev/lineSearch.ts:267`), with C and A frozen at γ₀
    // and Φ at the displaced iterate.
    const { phi } = evaluateConstraintSet(set, displaced(vertices), edges);
    const d = phi.map((v) => -v);

    // The 13 rhs: 3 gated, 10 recorded. Order is fixed and matters only for readability
    // of the emitted JSON — every row carries its own `gated` flag.
    const shapes: {
        shape: 'gradient' | 'projection';
        tag: string;
        gated: boolean;
        source: string;
        rhs: number[];
    }[] = [
        {
            shape: 'gradient',
            tag: 'production',
            gated: true,
            source: 'flatten(gradientAnalytical(γ₀))',
            rhs: [...b, ...zeroK],
        },
        {
            shape: 'projection',
            tag: 'production',
            gated: true,
            source: '−Φ(γ₀ + δ), δ = 1e-3·uniforms(3n, seed 7777)',
            rhs: [...zeroM, ...d],
        },
        {
            // The adversarial probe. 𝟙 over each coordinate block is exactly
            // null(Ā) = span{P₀,P₁,P₂} — the direction the shift ([DESIGN §4.1] step 1)
            // and the Woodbury correction (step 3) exist to handle, so a bug in the
            // P/U/M path shows HERE and may not show on a physical gradient, whose
            // 𝟙-component is small. NOT normalised: the relative error is scale-free.
            shape: 'gradient',
            tag: 'nullspace1',
            gated: true,
            source: '𝟙 over 3n — in null(Ā) = span{P₀,P₁,P₂}',
            rhs: [...new Array<number>(m).fill(1), ...zeroK],
        },
    ];
    for (let s = 0; s < 5; s++) {
        // Recorded, NOT gated: one of these exceeds the bar at n=960 with a CORRECT f64
        // factor, because a generic rhs excites A's worst-conditioned eigendirections and
        // both sides then sit at their own κ·u floor. Asserting on them would be a false
        // kill. They are emitted so the sensitivity lives in the artifact.
        shapes.push({
            shape: 'gradient',
            tag: `rand${s}`,
            gated: false,
            source: `splitmix32 uniforms on [-1,1), seed ${1000 + s}`,
            rhs: [...uniforms(m, 1000 + s), ...zeroK],
        });
        shapes.push({
            shape: 'projection',
            tag: `rand${s}`,
            gated: false,
            source: `splitmix32 uniforms on [-1,1), seed ${2000 + s}`,
            rhs: [...zeroM, ...uniforms(k, 2000 + s)],
        });
    }

    // The reference side: ONE LDLᵀ of K, reused for the other TWELVE rhs via the already
    // exported frozen operator — same `solveFactored` path. This is what makes the
    // ensemble free: the reference LDLᵀ dominates and the thirteen rhs together are a
    // fraction of it. Re-factorizing per rhs would cost ~13× per fixture and the ensemble
    // would not be affordable.
    const first = solveSaddleFromA(A, n, C, shapes[0].rhs.slice(0, m), shapes[0].rhs.slice(m));
    const frozen = { a: A, n, C, fac: first.fac };

    let minDiagLSq = Number.POSITIVE_INFINITY;
    for (let i = 0; i < n; i++) minDiagLSq = Math.min(minDiagLSq, L[i * n + i] ** 2);

    /** ‖z − zref‖₂ / ‖zref‖₂ — CR's quantity, spelt once for both vectors. */
    const relErrOf = (z: ArrayLike<number>, zref: number[]): number => {
        let num = 0;
        let den = 0;
        for (let i = 0; i < zref.length; i++) {
            num += (z[i] - zref[i]) ** 2;
            den += zref[i] ** 2;
        }
        return Math.sqrt(num / den);
    };

    const out: CrRow[] = [];
    for (let si = 0; si < shapes.length; si++) {
        const s = shapes[si];
        // Row 0's reference is the factorizing solve; every later row reuses its factor.
        const ref = si === 0 ? first : solveSaddleFrozen(frozen, s.rhs.slice(0, m), s.rhs.slice(m));
        const zref = [...ref.x, ...ref.lambda];
        let rhsNorm = 0;
        for (const v of s.rhs) rhsNorm += v * v;
        rhsNorm = Math.sqrt(rhsNorm);
        // THE GATED VECTOR — [DESIGN §4.1] steps 1–3, before step 4's refinement, which
        // is what [DESIGN §5] pins. Do not replace this with ext.solve(): step 4 repairs
        // any steps-1–3 defect that leaves the relative residual below SADDLE_IR_TOL, so
        // CR would stop distinguishing "the reformulation is exact" from "the
        // reformulation is approximate and IR rescued it" — and step 4 is precisely the
        // part that will later be running on a wrong f32 factor. Measured: degrading M to
        // f32 moves this vector by six orders while ext.solve()'s output stays
        // indistinguishable from correct.
        // @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §5 (CR)
        const z13 = schurWoodburySolve(L, n, C, setup, M, s.rhs);
        // The steps 1–4 vector — recorded, never gated. `traces` grows by exactly one
        // entry per solve, so index it at the length captured before the call.
        // `ext.solve` does NOT throw on the ungated rows that exceed the bar: IR's
        // stopping criterion is on the RESIDUAL and the residual converges fine — it is
        // the ERROR that is stuck at the conditioning floor. That is the whole reason
        // those rows are recorded rather than gated, and it is why no try/catch is needed.
        const traceIndex = traces.length;
        const z14 = ext.solve(s.rhs);
        const trace = traces[traceIndex];
        const relErr = relErrOf(z13, zref);
        const bar = crBar(n);
        const row: CrRow = {
            name: `${name}/${s.shape}:${s.tag}`,
            n,
            k,
            rhsShape: s.shape,
            rhsTag: s.tag,
            gated: s.gated,
            rhsSource: s.source,
            rhsNorm,
            relErr,
            relErrWithIr: relErrOf(z14, zref),
            bar,
            // [CAL §J.3]'s deliverable, emitted rather than left to be derived by hand.
            slack: bar / relErr,
            irRefinements: trace.refinements,
            irRelResiduals: trace.relResiduals,
            kappaSource: 'K1CAL:rows[].kappaK (k=4)',
            // Computed for EVERY row, but only the `gated: true` rows' verdicts bind —
            // the fixture tests assert on those three alone. A FAIL on an ungated row at
            // n=960 is the EXPECTED, measured behaviour of a generic rhs against a correct
            // f64 factor, not a defect; see the calibration debt owed to [CAL §F].
            verdict: relErr <= bar ? 'PASS' : 'FAIL',
            diagM: [M[0][0], M[1][1], M[2][2]],
            kappaM1: kappa1of3(M),
            minDiagLSq,
        };
        rows.push(row);
        out.push(row);
    }
    return out;
}

const FIXTURES: {
    name: string;
    make: () => { vertices: Vec3[]; edges: Edge[] };
    timeoutMs: number;
}[] = [
    // `crossing` is a FIXED 8-vertex graph (two disconnected 4-vertex polylines).
    // TestConfig.generate's params bag is OPTIONAL, not absent —
    // `generate: (params?: Record<string, number>) => GraphState`
    // (src/core/testConfigs.ts:9-14, :12) — and `crossing` ignores it, so `c.generate()`
    // is the right call. The cast below is a no-op kept for readability: GraphState is
    // structurally `{ vertices: Vec3[]; edges: Edge[] }` (:4-7). The registry lookup is
    // at :269-274.
    {
        name: 'crossing8',
        make: () => {
            const c = testConfigs.find((t) => t.id === 'crossing');
            if (!c) throw new Error('CR: the `crossing` test config is missing');
            return c.generate() as { vertices: Vec3[]; edges: Edge[] };
        },
        timeoutMs: 30_000,
    },
    { name: 'trefoil60', make: () => trefoil(60), timeoutMs: 30_000 },
    { name: 'trefoil120', make: () => trefoil(120), timeoutMs: 60_000 },
    { name: 'trefoil240', make: () => trefoil(240), timeoutMs: 120_000 },
    { name: 'trefoil480', make: () => trefoil(480), timeoutMs: 300_000 },
    { name: 'trefoil960', make: () => trefoil(960), timeoutMs: 900_000 },
];

for (const f of FIXTURES)
    for (const withPin of [false, true]) {
        const label = `${f.name}${withPin ? '+pin' : ''}`;
        test(
            `CR: ${label} — §4.1 reformulation matches solveSaddleFromA within the κ-relative bar`,
            () => {
                const { vertices, edges } = f.make();
                const measured = measure(label, vertices, edges, withPin);
                expect(measured.length).toBe(13);
                for (const row of measured) {
                    // Non-vacuity: a projection rhs at roundoff would make this fixture
                    // pass while testing nothing. Holds for all 13.
                    expect(row.rhsNorm).toBeGreaterThan(1e-8);
                }
                // THE GATE — the three physically-reachable rhs ONLY. The ten seeded rows
                // are recorded, not asserted: one of them exceeds the bar at n=960 against
                // a CORRECT f64 factor, and IR cannot repair it (it does not even fire),
                // so gating them would be a false kill. Pin the count so a future edit
                // cannot quietly drop a gated row and leave this loop asserting over fewer.
                const gated = measured.filter((r) => r.gated);
                expect(gated.length).toBe(3);
                for (const row of gated) {
                    // `relErr` is the steps-1–3 quantity [DESIGN §5] pins
                    // (schurWoodburySolve), never `relErrWithIr` — asserting the
                    // post-refinement value would let step 4 repair the thing CR exists
                    // to measure.
                    expect(row.relErr).toBeLessThanOrEqual(row.bar);
                }
            },
            f.timeoutMs,
        );
    }

test('CR: write the committed results JSON', () => {
    const sha = execSync('git rev-parse --short HEAD').toString().trim();
    // FIXED filename, deliberately not date-derived. A `${date}-…` name writes a SECOND
    // artifact on any later calendar day instead of refreshing the committed one, and the
    // `git add` would then stage both — leaving [CAL §J.3]'s deliverable with two
    // candidate sources and no rule for which is current. Provenance is `gitShaShort`
    // INSIDE the file, where it can be read without parsing a filename.
    //
    // NO TIMESTAMP, deliberately. Every `bun test` rewrites this file, so a timestamp
    // would make it dirty the worktree on every run and force a diff that carries no
    // information. With none, the output is BYTE-STABLE: re-running changes the file only
    // when the NUMBERS change, which is exactly what a committed gate artifact should do.
    // Do not add `emittedAt`, and do not gate the write behind an env var either — an
    // env var is a flag an executor can forget to set, and then the artifact silently
    // goes stale instead of being regenerated.
    writeFileSync(
        'bench/results/gpu-phase2a-cr.json',
        `${JSON.stringify(
            {
                gate: 'CR',
                gitShaShort: sha,
                uF64: U_F64,
                barFormula: 'max(1e-12, 10*kappaK*uF64)',
                quantity: '[DESIGN §4.1] steps 1-3 (schurWoodburySolve) vs solveSaddleFromA',
                // The verdict rests on `gated: true` rows only. Stated in the artifact so
                // a reader who has never seen the plan cannot mistake an expected ungated
                // FAIL at n=960 for a red gate.
                gatedRhs: ['gradient:production', 'projection:production', 'gradient:nullspace1'],
                ensemblePrng:
                    'splitmix32 uniforms on [-1,1); seeds 1000+s (gradient), 2000+s (projection), s=0..4; displacement seed 7777',
                kappaArtifact: K1CAL,
                fixtures: rows,
            },
            null,
            2,
        )}\n`,
    );
    expect(rows.length).toBe(FIXTURES.length * 2 * 13); // 6 fixtures × 2 sets × 13 rhs = 156
    expect(rows.filter((r) => r.gated).length).toBe(FIXTURES.length * 2 * 3); // 36
});
