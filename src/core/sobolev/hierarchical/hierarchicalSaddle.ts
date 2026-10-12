/**
 * `makeHierarchicalSaddle` — the PRODUCER of the Phase 2H seam (spec §5.1, §11.2 row 8b).
 *
 * Spec §11.2's rows 1–8 widened `PreparedSaddle` into a `kind`-tagged union with a `'hierarchical'`
 * arm and row 9b built the constraint route (`schur.ts`), but **nothing ever WROTE that arm**:
 * `src/gpu/saddle.ts`'s `makePreparedSaddle` is the only `PreparedSaddle` producer and it is dense,
 * so the hierarchical branches in `optimizer.ts`, `gradient.ts`, `lineSearch.ts` and `linsolve.ts`
 * were unreachable code and the dense path stayed bit-identical only because it was still the only
 * path. This module is what makes the phase reachable, by composing the parts that already exist:
 *
 *     buildBvh -> buildBct -> prepareA -> makeHierarchicalOperator -> prepareLowOrder
 *              -> makePreconditioner        (ONCE, in this factory)
 *     prepareSchur(op, P, C) | prepareNullspace(op, P, edgeBlock, otherRows)
 *                                           (ONCE PER STEP, in `prepare(C)`, routed per §8.2)
 *
 * That sequence previously existed only inside test files (`gmres.test.ts:244-267`,
 * `precond.test.ts:133-152`) and in no exported function.
 *
 * THE §8.2 CONSTRAINT FORK LIVES IN `prepare(C)`. §8.2 rules the route PER BLOCK — `O(1)` blocks
 * (`barycenter`, `point`, `totalLength`) to the Schur complement, `edgeLengthsBlock`'s `|E|` rows
 * to the nullspace route — and names exactly two sets that must work: `{barycenter, totalLength,
 * pins}` (all-Schur) and `{barycenter, edgeLengths, pins}` (hybrid). `prepare` receives the
 * EVALUATED `C` and no block metadata, so {@link findEdgeLengthsRows} recognises the block
 * structurally; see its TSDoc for why that recognition is exact and what happens when it fails.
 *
 * WHAT "FROZEN" MEANS HERE. There is no factorization to reuse (arXiv:2006.07859 §6.3, spec §1.1),
 * so the reusable artefact is the BUILD — tree, block lists, near-field precompute, `L^{2-s}` and
 * `Ã⁻¹`, plus the `‖A_σ‖₂` estimate — and it is shared by the gradient solve and by every
 * projection iterate of every τ-trial of the step. A fresh preconditioned GMRES is what each solve
 * costs. This is also why `optimizer.ts` REFUSES this arm unless `projectionMode: 'frozen'`: the
 * other projection branch calls `assembleAFlat` per iterate (O(N²) f64, 7.2 GB at N = 30 000).
 *
 * REACHABILITY IS MAIN-THREAD / INLINE ONLY (spec A33 item 2). `src/worker/solverWorker.ts:51-52`
 * throws `'solverWorker: \`saddle\` cannot cross the worker boundary'` whenever `'saddle' in
 * msg.args`, `store.ts:362` defaults `solverDriver: 'worker'`, and the object this factory returns
 * holds closures, so it is not structured-cloneable by construction. `dispatch.ts:307` forwards
 * `saddle: args.saddle` and no site in `src/` sets it. Passing this saddle through
 * `dispatchDescentStep` under the worker driver therefore THROWS, by design and pinned by a test.
 * Making the worker route work needs the saddle built INSIDE the worker from cloneable inputs —
 * a different seam, deliberately not smuggled in here.
 *
 * TOPOLOGY. The stack only builds on a NATURAL-ORDER CLOSED RING: `assembleCurveLaplacian`
 * (inside `makePreconditioner`) accepts only edges `(i, i+1)` plus the wrap `(n-1, 0)`, and
 * `prepareLowOrder('vertex')` additionally needs `|E| = |V|` with every vertex starting exactly
 * one edge (`laplacian.ts`'s TOPOLOGY LIMIT paragraph points the refusal at this function).
 * Anything else is refused UP FRONT by {@link HierarchicalSaddleUnsupportedError} rather than
 * reaching a `CurveTopologyError` from three frames down with no mention of the fallback.
 *
 * SCALE IS RECORDED, NOT NORMALISED (issue utof/repulsive-test2#69, spec A33's fifth fact).
 * `buildBvh` does not normalise and its header (`bvh.ts:32-38`) says tree quality assumes a bbox
 * max extent ≈ 1; every `out-bct-*.json` oracle was measured on normalised fixtures while every
 * shipped test builds from the unnormalised `trefoil` (extent 5.4716), and `κ_near` moves 1.70×
 * across the two scales. This factory is the first code to take ARBITRARY production vertices, so
 * it reports {@link HierarchicalSaddleBuild.curveExtent} and leaves the geometry alone: rescaling
 * would require proving the operator's scale transform, which belongs to #69.
 *
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.1, §8.2, §11.2 row 8b, A33, A35
 * @see https://arxiv.org/abs/2006.07859 §6.3 (hierarchical apply ⇒ iterate, do not factor)
 * @issue utof/repulsive-test2#62
 * @issue utof/repulsive-test2#69
 */
import { DEFAULTS } from '../../optimizer';
import type { Edge, Vec3 } from '../../testConfigs';
import { blockIndex } from '../layout';
import type { HierarchicalPreparedSaddle } from '../linsolve';
import { type HierarchicalA, prepareA } from './apply';
import { type Bct, type BctParams, buildBct } from './bct';
import { type Bvh, buildBvh } from './bvh';
import { curveExtent } from './extent';
import { applyAsigma, estimateNorm2 } from './gmres';
import { type LowOrderA, prepareLowOrder } from './lowOrder';
import { type NullspaceSolver, prepareNullspace } from './nullspace';
import { type HierarchicalOperator, makeHierarchicalOperator } from './operator';
import { makePreconditioner, type Preconditioner, type PreconditionerOptions } from './precond';
import { prepareSchur, type SchurOptions, type SchurSolver } from './schur';

/**
 * BVH leaf size. Why 8 and not a measured optimum: it is the value EVERY shipped hierarchical test
 * and every `out-bct-*.json` oracle row was built with (`gmres.test.ts:63`, `precond.test.ts:88`,
 * `operator.test.ts`, `lowOrder.test.ts`); `buildBvh` has no default of its own (`bvh.ts:224`).
 * Changing it invalidates the comparison to every one of those numbers, so it is a named constant
 * rather than an inline literal. Not a tuned parameter — no gate in spec §7 varies it.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.5
 */
export const DEFAULT_LEAF_SIZE = 8;

/**
 * The factory refused to build, for a reason the caller can fix by ROUTING ELSEWHERE rather than by
 * passing different options — today only the topology limit, whose fallback is the dense path.
 *
 * Why a typed refusal at the factory instead of letting the deep throw happen: without it an
 * ordinary junction or crossing curve dies inside `makePreconditioner` → `assembleCurveLaplacian`
 * with `CurveTopologyError: edge [3,7] is not (i,i+1) or the wrap (n-1,0)`, which names an edge but
 * not the decision the caller has to make. `laplacian.ts:13-14` asks for exactly this dispatch.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §11.2 row 8b
 * @see src/core/sobolev/hierarchical/laplacian.ts (TOPOLOGY LIMIT)
 */
export class HierarchicalSaddleUnsupportedError extends Error {
    constructor(
        /** `'topology'`: not a natural-order closed ring. Route the curve to the DENSE path. */
        readonly reason: 'topology',
        message: string,
    ) {
        super(`makeHierarchicalSaddle: ${message}`);
        this.name = 'HierarchicalSaddleUnsupportedError';
    }
}

/**
 * Options of {@link makeHierarchicalSaddle}. Every numeric default is INHERITED, never re-chosen
 * here — see each field for where from.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.1, §5.6, §8.2
 */
export interface HierarchicalSaddleOptions {
    /**
     * Tangent-point kernel exponents and the `ε` regularisation. Defaults are the CORE's
     * `DEFAULTS` (`optimizer.ts:23`, `{alpha: 3, beta: 6, epsilon: 1e-10}`), re-exported rather
     * than re-literalised.
     *
     * **THE CALLER MUST NOT LET THESE DRIFT FROM THE STEP'S.** `dispatch.ts` forwards no
     * `alpha`/`beta`/`epsilon` field (census §C: `DispatchDescentStepArgs` has none), so
     * `sobolevStepSet` falls back to the same `DEFAULTS`; a saddle built here with different values
     * gives an operator that is not the Ā of the step that consumes it, and NOTHING downstream can
     * see the disagreement — the residual is self-consistent and nothing thresholds it
     * (`linsolve.ts:410-413`). `src/gpu/driver.ts:147-155` documents the identical trap for the
     * GPU gradient engine.
     */
    alpha?: number;
    beta?: number;
    epsilon?: number;
    /** BVH leaf size. Default {@link DEFAULT_LEAF_SIZE}. */
    leafSize?: number;
    /**
     * Block-cluster-tree parameters; `theta` defaults to `bct.ts`'s `DEFAULT_THETA`, whose shipped
     * value is 0.25. The PRODUCTION choice of θ is OPEN: spec A46 item 7 withdrew gate H6's RED as
     * a θ signal (θ=0.25 reads 0.7649, θ=0.5 reads 0.7856 and the exact δ=0 operator reads 0.7644,
     * so θ is not the variable), and the calibration artefact publishes `thetaProduction: null`
     * accordingly. 0.25 is the shipped default pending #51's own evidence — do not re-cite H6.
     * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md A46 item 7
     * @issue utof/repulsive-test2#51
     */
    bct?: BctParams;
    /** Forwarded verbatim to {@link makePreconditioner}; its `massWeight` default is `30/ext²`. */
    precond?: PreconditionerOptions;
    /**
     * Forwarded to {@link prepareSchur} on every `prepare(C)`. `tol` defaults to
     * `DEFAULT_SCHUR_TOL` (1e-12, spec A35 item 4's measured knee) — do not re-tune it here.
     * `normA2` is supplied by this factory unless you override it (see
     * {@link HierarchicalSaddleBuild.normA2}).
     *
     * ON THE HYBRID ROUTE (§8.2) these same knobs are forwarded to {@link prepareNullspace}, which
     * applies them to the `O(1)` rows it hands to `prepareSchur` — one set of numbers for both
     * routes, so a `tol`/`etaAccept`/`maxK` chosen here cannot mean two different things depending
     * on which constraint set the step happens to carry. `normA2` is the ONE exception and is
     * deliberately NOT forwarded: see the dispatch comment in {@link makeHierarchicalSaddle}.
     */
    schur?: SchurOptions;
}

/**
 * What the one-time build produced and under which parameters — the provenance a measurement needs
 * in order to be comparable with the corpus (memory `figures-carry-their-regime`).
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §11.2 row 8b
 */
export interface HierarchicalSaddleBuild {
    /** Vertex count; equals `op.n`, which `optimizer.ts:314` checks against `vertices.length`. */
    n: number;
    alpha: number;
    beta: number;
    epsilon: number;
    leafSize: number;
    /** As RESOLVED by `resolveBctParams`, not as passed. */
    theta: number;
    thetaT: number;
    /** `σ = tr(A_bct)/n`, the shift of `A_σ`; the preconditioner was built with this same value. */
    sigma: number;
    /**
     * `‖A_σ‖₂`, from `estimateNorm2`'s 30 power iterations, computed ONCE here and handed to every
     * `prepareSchur`. Why this is not a behaviour change: `estimateNorm2` is deterministic (fixed
     * LCG seed and fixed iteration count, `gmres.ts:273-296`) and depends only on the operator,
     * which is frozen, so the cached value is BIT-IDENTICAL to the one each `prepare` would have
     * computed — it saves 30 matvecs per `prepare`, nothing else. A test asserts the bit equality.
     */
    normA2: number;
    /**
     * Maximum bounding-box side of the vertices AS GIVEN. Recorded because `buildBvh` does not
     * normalise and the oracle corpus was measured at extent ≈ 1 while the shipped fixtures sit at
     * 5.4716 — `κ_near` differs 1.70× between the two, so a figure without its extent is not
     * comparable. This factory deliberately does NOT rescale (issue #69).
     * @issue utof/repulsive-test2#69
     */
    curveExtent: number;
}

/**
 * A built hierarchical saddle: a `HierarchicalPreparedSaddle` for `sobolevStepSet`'s `opts.saddle`,
 * plus the build provenance and the composed parts.
 *
 * `op` and `prepare`'s return type are NARROWED against the seam interface (`HierarchicalOperator`
 * carries `sigma`, which `prepareSchur` requires; `SchurSolver` and `NullspaceSolver` carry
 * `report`, which is where `S`'s conditioning, the projector's health and the per-column GMRES
 * records live). Both are supersets, so this is still assignable to `HierarchicalPreparedSaddle`.
 *
 * `prepare` returns a UNION because §8.2's fork picks the route per constraint set: a
 * `SchurSolver` for an all-`O(1)` set, a `NullspaceSolver` when the set carries
 * `edgeLengthsBlock`'s `|E|` rows. `NullspacePrepareReport` is the discriminable member — it is the
 * only one with a `projector` field, and its own `schur` field is the `SchurPrepareReport` of the
 * `O(1)` rows it routed to `prepareSchur`, so a reader that wants the Schur numbers can get them on
 * either route without a type assertion.
 *
 * NOT RE-ENTRANT — `HierarchicalA`, `LowOrderA` and `Preconditioner` each own shared scratch
 * (`apply.ts`, `lowOrder.ts`, `precond.ts` module TSDoc). One saddle serves one step at a time;
 * do not share it across concurrent solves.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.1, §5.2
 */
export interface HierarchicalSaddle extends HierarchicalPreparedSaddle {
    op: HierarchicalOperator;
    prepare(C: number[][]): SchurSolver | NullspaceSolver;
    build: HierarchicalSaddleBuild;
    /**
     * The composed stack, exposed so a gate harness or a differential test can reach the same
     * objects the solve uses (densify `A.applyA`, read `bct.stats`, count `precond` applies)
     * instead of rebuilding a second, possibly-different one. Reading these is safe; MUTATING them
     * invalidates the saddle.
     */
    parts: {
        bvh: Bvh;
        bct: Bct;
        A: HierarchicalA;
        lowOrder: LowOrderA;
        precond: Preconditioner;
    };
}

/**
 * Rejects anything that is not the natural-order CLOSED ring, i.e. the edge multiset
 * `{(i, i+1) : i < n-1} ∪ {(n-1, 0)}` with `n >= 3`.
 *
 * That is exactly the intersection of the two live limits, neither widened nor narrowed:
 * `assembleCurveLaplacian` requires each edge to be `(p, p+1)` or the wrap and accepts an OPEN
 * path, while `prepareLowOrder('vertex')` requires `|E| = |V|` with a slot→vertex bijection, which
 * an open path fails. Edge ORDER within the list is irrelevant to both (they index by slot), so it
 * is not checked here either.
 */
function assertClosedNaturalRing(vertices: Vec3[], edges: Edge[]): void {
    const n = vertices.length;
    if (n < 3) {
        throw new HierarchicalSaddleUnsupportedError(
            'topology',
            `need >= 3 vertices for a closed ring, got ${n}; route this curve to the DENSE path`,
        );
    }
    if (edges.length !== n) {
        throw new HierarchicalSaddleUnsupportedError(
            'topology',
            `the hierarchical stack needs a CLOSED curve (|E| = |V|), got |E| = ${edges.length}, ` +
                `|V| = ${n} — an open chain has no slot→vertex bijection for prepareLowOrder's ` +
                `'vertex' variant. Route this curve to the DENSE path.`,
        );
    }
    const seen = new Uint8Array(n);
    for (const [p, q] of edges) {
        const slot = q === p + 1 && p >= 0 && q < n ? p : p === n - 1 && q === 0 ? n - 1 : -1;
        if (slot < 0) {
            throw new HierarchicalSaddleUnsupportedError(
                'topology',
                `edge [${p},${q}] is not (i,i+1) or the wrap (n-1,0), so the curve Laplacian ` +
                    `inside the preconditioner is not (cyclic-)tridiagonal (laplacian.ts TOPOLOGY ` +
                    `LIMIT). Junctions, crossings and permuted vertex order go to the DENSE path.`,
            );
        }
        if (seen[slot]) {
            throw new HierarchicalSaddleUnsupportedError(
                'topology',
                `edge [${p},${q}] is duplicated; the edge list must be each of (i,i+1) and the ` +
                    `wrap (n-1,0) exactly once. Route this curve to the DENSE path.`,
            );
        }
        seen[slot] = 1;
    }
    // |E| = n and no slot twice ⇒ every slot once, so the wrap is present and the ring is closed.
}

/**
 * Is `row` EXACTLY the row `edgeLengthsBlock` writes for `edge`? Six nonzeros, `+T_I` at `i1` and
 * `−T_I` at `i2` in the coordinate-block layout, zero everywhere else.
 *
 * The arithmetic is `edgeLengthsBlock`'s own spelling, deliberately: RAW geometric `ℓ` with no `+ε`,
 * `inv = 1/ℓ` and then `ex * inv` (NOT `ex / ℓ`, which is a different float), and the same
 * `ℓ < 1e-14` guard that zeroes the row. That is what makes the `!==` comparisons below exact
 * rather than approximate — see {@link findEdgeLengthsRows} for why exactness is the point.
 * @see src/core/sobolev/constraintSet.ts (`edgeLengthsBlock`)
 * @see src/core/sobolev/layout.ts (`blockIndex` — coordinate-block layout, `c·n + vertex`)
 */
function isEdgeLengthsRow(row: number[] | undefined, edge: Edge, vertices: Vec3[]): boolean {
    const n = vertices.length;
    if (row === undefined || row.length !== 3 * n) return false;
    const [i1, i2] = edge;
    // A self-loop would make `edgeLengthsBlock`'s two `+=` cancel into a zero row, so the pattern
    // below would not describe it; `nullspace.ts`'s `assertEdgeCycle` refuses them outright.
    if (!(i1 >= 0 && i1 < n && i2 >= 0 && i2 < n) || i1 === i2) return false;
    const p1 = vertices[i1];
    const p2 = vertices[i2];
    const ex = p2[0] - p1[0];
    const ey = p2[1] - p1[1];
    const ez = p2[2] - p1[2];
    const ell = Math.sqrt(ex * ex + ey * ey + ez * ez);
    const inv = 1 / ell;
    const T: Vec3 = ell < 1e-14 ? [0, 0, 0] : [ex * inv, ey * inv, ez * inv];
    for (let c = 0; c < 3; c++) {
        const base = blockIndex(c, 0, n);
        for (let i = 0; i < n; i++) {
            const want = i === i1 ? T[c] : i === i2 ? -T[c] : 0;
            // `-0 !== 0` is false in JS, so a degenerate zeroed row still matches. A NaN entry
            // (the `ell0`-mismatch backstop in `edgeLengthsBlock`) never matches, which is right.
            if (row[base + i] !== want) return false;
        }
    }
    return true;
}

/**
 * Index of the first `edgeLengthsBlock` row inside the stacked constraint set, or `null` when the
 * set carries none. This is the ONLY input §8.2's dispatch has: `prepare(C)` receives the EVALUATED
 * `C` and no block metadata (`gradient.ts:139` hands over `evaluateConstraintSet(set, …).C`
 * verbatim), while §8.2 routes PER BLOCK and therefore needs to know WHICH rows are the `|E|` rows,
 * not merely how many rows there are.
 *
 * WHY THE RECOGNITION IS EXACT AND NOT TOLERANT. Stripping the wrong `|E|` rows out of `C` would
 * hand them to the projection as if they were `edgeLengths` rows: those constraints would be
 * silently replaced by the re-derived sparse `C_edge` and nothing downstream could see it — both
 * `residual` and `edgeConstraintResidual` are self-consistent against whichever `C_edge` the route
 * built (`nullspace.ts`). So there is no tolerance to get wrong here, and none is needed:
 * `edgeLengthsBlock` and {@link isEdgeLengthsRow} run identical arithmetic on identical inputs,
 * `evaluateConstraintSet` concatenates block rows verbatim with no scaling, and `nullspace.ts`'s
 * `edgeTangents` is a third copy of the same spelling that `nullspace.test.ts` already pins against
 * the dense rows ("the SPARSE C_edge matches the shipped dense edgeLengthsBlock rows").
 *
 * A MISMATCH IS NOT TOLERATED EITHER — it returns `null`, the caller falls through to
 * `prepareSchur`, and an `O(n)` set then meets {@link SchurRouteError}, which names `k`, `maxK` and
 * the route to take. That is also the backstop for a `C` evaluated at DIFFERENT vertices than the
 * saddle was built from (the unenforced "same vertices" rule, issue utof/repulsive-test2#83): the
 * tangents will not match, so the hybrid set refuses LOUDLY instead of projecting onto the subspace
 * of another curve.
 *
 * Cost is `O(|E|·3n)` — the size of the `C` the caller already built, so the dispatch adds no new
 * asymptotic cost to the step; on an all-`O(1)` set it exits immediately on `C.length < |E|`. It
 * does NOT make the dense `C` itself affordable at N = 30 000 (`|E| × 3n` f64 = 21.6 GB,
 * `nullspace.ts` module TSDoc): passing the set's BLOCKS across the seam instead of its evaluated
 * rows is a seam change that belongs to #62's next milestone, not to this dispatch.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §8.2 (the per-block table), §11.2 row 9b
 * @see src/core/sobolev/constraintSet.ts (`edgeLengthsBlock`, `evaluateConstraintSet`)
 * @issue utof/repulsive-test2#62
 */
function findEdgeLengthsRows(C: number[][], vertices: Vec3[], edges: Edge[]): number | null {
    const m = edges.length;
    if (m === 0 || C.length < m) return null;
    // `evaluateConstraintSet` concatenates blocks in array order, so the `|E|` rows are CONTIGUOUS
    // and `rowStart` is just the number of `O(1)` rows that precede them — at most `C.length − |E|`
    // candidates, and `prepareNullspace` re-checks the bound it needs (`rowStart <= kOther`).
    for (let rowStart = 0; rowStart + m <= C.length; rowStart++) {
        let hit = true;
        for (let r = 0; r < m; r++) {
            if (!isEdgeLengthsRow(C[rowStart + r], edges[r], vertices)) {
                hit = false;
                break;
            }
        }
        if (hit) return rowStart;
    }
    return null;
}

/**
 * Builds the Phase 2H hierarchical saddle for `vertices` — the object to hand to
 * `sobolevStepSet`'s `opts.saddle` (which also REQUIRES `projectionMode: 'frozen'`, see the module
 * TSDoc). The build happens once, here; `prepare(C)` is called once per step by `gradient.ts:139`
 * and its solver serves the gradient solve and every projection iterate.
 *
 * `vertices` must be the SAME vertices the step receives — the Phase 1 rule
 * (`src/gpu/driver.ts:164`, `DensePreparedSaddle`'s TSDoc), unchanged here.
 *
 * `disjointPairs` is accepted for call-shape parity with `sobolevStepSet` and is VALIDATED against
 * `edges`, but it is NOT consumed: the hierarchical operator derives its own adjacent-pair
 * exclusion from the BCT's adjacency certificate (`bct.ts`), not from a precomputed list. The
 * validation is the point — a `disjointPairs` that does not belong to these `edges` means the
 * caller is not building the saddle for the step's inputs, and that disagreement is otherwise
 * invisible (the dense energy/dE change, the hierarchical operator does not).
 *
 * @throws {@link HierarchicalSaddleUnsupportedError} `reason: 'topology'` — not a closed natural
 *   ring; route to the dense path.
 * @throws `RangeError` — `disjointPairs` does not match `edges`.
 * @throws `CurveTopologyError` — zero-length edge, or a `precond.massWeight <= 0`.
 * @throws `Error` — a non-usable `curveExtent` / `sigma` (`makePreconditioner`'s guards).
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.1, §11.2 row 8b, A33, A35
 * @issue utof/repulsive-test2#62
 */
export function makeHierarchicalSaddle(
    vertices: Vec3[],
    edges: Edge[],
    disjointPairs: number[][],
    opts: HierarchicalSaddleOptions = {},
): HierarchicalSaddle {
    const n = vertices.length;
    assertClosedNaturalRing(vertices, edges);
    // O(m) consistency check, NOT the O(m²) re-derivation: on a closed ring every edge is adjacent
    // to exactly its two neighbours, so `calculateDisjointPairs` gives m rows of length m − 3
    // (0 at m = 3, 1 at m = 4). A list built for another curve or another n fails this.
    // @see src/core/tangentPointEnergy.ts (calculateDisjointPairs)
    const expectedRow = n - 3;
    if (disjointPairs.length !== n) {
        throw new RangeError(
            `makeHierarchicalSaddle: disjointPairs has ${disjointPairs.length} rows, expected one ` +
                `per edge (${n}); it was not built from these edges (calculateDisjointPairs(edges))`,
        );
    }
    for (let i = 0; i < n; i++) {
        if (disjointPairs[i]?.length !== expectedRow) {
            throw new RangeError(
                `makeHierarchicalSaddle: disjointPairs[${i}] has ${disjointPairs[i]?.length} ` +
                    `entries, expected ${expectedRow} (every edge of a closed ring is adjacent to ` +
                    `exactly 2 others); it was not built from these edges`,
            );
        }
    }

    const alpha = opts.alpha ?? DEFAULTS.alpha;
    const beta = opts.beta ?? DEFAULTS.beta;
    const epsilon = opts.epsilon ?? DEFAULTS.epsilon;
    const leafSize = opts.leafSize ?? DEFAULT_LEAF_SIZE;

    // The build, in the one order the parts support: the SAME `bvh` and the SAME block lists feed
    // `prepareA` and `prepareLowOrder` (spec §5.2 — `L^{2−s}` must be assembled over the same tree
    // as `A`, or the preconditioner is not a preconditioner for this operator).
    // @see test/sobolev/hierarchical/precond.test.ts:133-152 (the same sequence, by hand)
    const bvh = buildBvh(vertices, edges, leafSize, epsilon);
    const bct = buildBct(bvh, opts.bct ?? {});
    const source = { kind: 'lists', bct } as const;
    const A = prepareA(bvh, n, source, { alpha, beta, epsilon });
    const op = makeHierarchicalOperator(A);
    const lowOrder = prepareLowOrder(vertices, edges, bvh, source, {
        alpha,
        beta,
        variant: 'vertex',
    });
    // `op.sigma` and not an exact trace: spec §2.3 forbids the exact trace in production, and
    // `prepareSchur` throws if `precond.sigma !== op.sigma`. `makePreconditioner` computes its own
    // `ext` for the default `massWeight = 30/ext²` (`precond.ts:315`), so NO scale factor is
    // applied here — applying one too would double-count it (issue #69).
    const precond = makePreconditioner(vertices, edges, lowOrder, op.sigma, opts.precond ?? {});

    // `‖A_σ‖₂` once for the whole build, not once per `prepare` — see `build.normA2`.
    const normA2 = estimateNorm2((u, out) => applyAsigma(op, u, out), 3 * n).norm;

    const build: HierarchicalSaddleBuild = {
        n,
        alpha,
        beta,
        epsilon,
        leafSize,
        theta: bct.params.theta,
        thetaT: bct.params.thetaT,
        sigma: op.sigma,
        normA2,
        curveExtent: curveExtent(vertices),
    };

    return {
        kind: 'hierarchical',
        op,
        build,
        parts: { bvh, bct, A, lowOrder, precond },
        // One `prepare` per step, by contract (`gradient.ts:139`); the returned solver is stored in
        // the frozen operator and reused for every projection rhs (`linsolve.ts:1133`). `prepare`
        // costs k GMRES runs and each `solve` costs one (zero when `rhsTop = 0`, which is what the
        // projection passes — `lineSearch.ts:279`), so a step is k + 1 GMRES independent of the
        // number of τ-trials (9b's measured structural cost).
        // A caller-supplied `schur.normA2` still wins; `?? normA2` rather than a spread so that an
        // explicitly-`undefined` field cannot silently discard the cached value.
        prepare: (C: number[][]): SchurSolver | NullspaceSolver => {
            // §8.2's FORK. It routes PER BLOCK — `barycenter`/`point`/`totalLength` (`O(1)` rows)
            // to the Schur complement, `edgeLengthsBlock`'s `|E|` rows to the nullspace route — so
            // the dispatch recognises the BLOCK and not a row count. §11.2 row 9b's verification
            // column is this line ("Dispatch on `k`: Schur for `k=O(1)`, nullspace for
            // `edgeLengths`' `|E|` rows"), and §8.2 names `{barycenter, edgeLengths, pins}` as one
            // of exactly two sets that "must actually work".
            //
            // §8.2 PINS NO NUMERIC `k` THRESHOLD, and this dispatch does not invent one. The only
            // cap in play is `maxK`, forwarded unchanged to BOTH routes from the single
            // `opts.schur.maxK` (default `SCHUR_MAX_K`), so the dispatch and `prepareSchur`'s own
            // `k > maxK` refusal cannot disagree about where the Schur route ends. An all-`O(1)`
            // set above the cap is deliberately NOT rerouted: it has no `O(n)` block for a
            // projection to absorb, so it must still reach `SchurRouteError`, whose message
            // ("raise maxK only if k really is O(1)") is the one that applies to it.
            //
            // `normA2` IS NOT FORWARDED to the nullspace route, on purpose. The cached value is
            // `‖A_σ‖₂`, and that route certifies against `‖P A_σ P‖₂ ≤ ‖A_σ‖₂`; substituting the
            // larger norm shrinks `η` and makes the certificate LOOSER, which is the unsafe
            // direction (`NullspaceOptions.normA2`). It estimates its own, once, on the projected
            // operator.
            const rowStart = findEdgeLengthsRows(C, vertices, edges);
            if (rowStart !== null) {
                return prepareNullspace(
                    op,
                    precond,
                    { vertices, edges, rowStart },
                    C.filter((_, r) => r < rowStart || r >= rowStart + edges.length),
                    {
                        tol: opts.schur?.tol,
                        maxit: opts.schur?.maxit,
                        maxK: opts.schur?.maxK,
                        etaAccept: opts.schur?.etaAccept,
                        // Forwarded rather than dropped: a knock-out configured on this factory
                        // must reach whichever route the set takes, or a defect injected for a
                        // hybrid set would silently run clean.
                        knockOut: opts.schur?.knockOut,
                        schurDefect: opts.schur?.defect,
                    },
                );
            }
            return prepareSchur(op, precond, C, {
                ...opts.schur,
                normA2: opts.schur?.normA2 ?? normA2,
            });
        },
    };
}
