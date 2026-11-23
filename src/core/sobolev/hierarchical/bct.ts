/**
 * Block cluster tree (BCT) for the Phase 2H hierarchical Sobolev operator: the dual traversal
 * of the R^6 BVH that classifies every cluster pair `(A, B)` as a FAR block (rank-1 lump, one
 * kernel evaluation at the barycentres) or a NEAR block (every edge pair summed exactly by the
 * task-4 near field). Nothing here evaluates a kernel; this module decides WHICH pairs are summed
 * exactly, and it is the structure every measured constant of the phase is quoted against.
 *
 * PORT of the spike `.superpowers/sdd/2026-09-11-phase2h-queue/scratch/bct.ts` (`buildBct`), the
 * code that produced every `out-bct-*.json` / `out-accuracy.json` oracle. Keep the admissibility
 * convention, the certificate, the child push order and the split rule EXACTLY: tasks 4-11 pin
 * those oracles (`kappa_near = 108.1956` at n=30 000), and the list ORDER fixes the accumulation
 * order of the apply, so a "cleanup" here silently moves the f32 noise measurements too.
 *
 * ADMISSIBILITY — Yu/Schumacher/Crane Eq. 28, barycentre form:
 *     max(r_x^A, r_x^B) <= theta * |x_A - x_B|      and      max(r_T^A, r_T^B) <= theta_T.
 * `theta_T` defaults to `Infinity` (OFF). That is legitimate for the HIGH-order kernel, whose
 * `k^0_{2s+1} = 1/|x-y|^{2s+1}` does not involve T at all (the tangent enters through the operator
 * D, not through K). The LOW-order kernel `k^2_{2s+5}(x, y, T_I)` does involve T_I and in principle
 * needs the bound; the spike measured both with it off (`THETA_T: null`), and §5.5 records "theta_T
 * off" as the convention the oracles were taken under. It stays a runtime parameter.
 * Barycentre separation is MORE permissive than Repulsor's hull-distance GJK separation
 * (`[RS §5.1.4]`), so every near count here is an optimistic lower bound (spec §5.5, unmeasured).
 *
 * ADJACENCY CERTIFICATE (`adjacencyCertificate`, default on). The discrete energy omits
 * neighbouring edge pairs: increments are made only "for all pairs of disjoint edges I n J = {}"
 * (arXiv:2006.07859 §4; repo: `calculateDisjointPairs`, src/core/tangentPointEnergy.ts:44-58,
 * which drops `j` when the two edges share any vertex). A near block skips them pair by pair; a
 * FAR block cannot — it is a rank-1 lump with no per-pair structure — so a far block must be
 * PROVABLY adjacency-free, else it injects an interaction the energy does not contain.
 * Proof of the certificate: if `I in A` and `J in B` share a vertex `v` then
 * `|x_I - x_J| <= (l_I + l_J)/2` (each midpoint is `l/2` from `v`), hence
 *     |x_A - x_B| <= |x_A - x_I| + |x_I - x_J| + |x_J - x_B|
 *                 <= r_x^A + r_x^B + (maxEll_A + maxEll_B)/2.
 * Contrapositive: `d > r_x^A + r_x^B + (maxEll_A + maxEll_B)/2` certifies that no adjacent pair
 * spans `(A, B)`; it also certifies `I != J`. O(1) per block, using `Bvh.maxEll` (which exists for
 * this and must not be deleted). Blocks that pass Eq. 28 but fail the certificate are counted in
 * `adjacencyVetoed` and split further. MEASURED on `torusKnot` n=960 leaf=8: 248 vetoes at
 * theta=0.5, 0 at theta<=0.4 (`out-accuracy.json`); the certificate is NOT a no-op.
 *
 * ONE-LEAF SPLIT — deviation from App. B.2.1, recorded (spec §5.5): the paper always produces the
 * four children `(A1,B1),(A1,B2),(A2,B1),(A2,B2)` and makes a block a leaf as soon as "A or B
 * contains just a few edges". Here, when exactly ONE of A, B is a BVH leaf the OTHER is split (two
 * children). Without it an 8-edge leaf pairs against a 10 000-edge cluster as a single dense near
 * block and the near-field pair count stops being O(n). This is the standard H-matrix fix.
 *
 * chi=0 MUST STREAM (spec §7.1). The exact reference operator is the all-inadmissible limit
 * `theta = 0` of THIS traversal (§5.6: "chi (or theta) is a runtime parameter whose minimum value
 * is the exact all-inadmissible path, through the same traversal code"). At N=30 000, L=8 its near
 * list is 1.4e7 blocks (~112 MB) and its precomputed values 3.60 GB — the dense storage the phase
 * exists to escape. So the traversal is written once, as {@link traverseBct}, calling a visitor per
 * block and allocating only an O(depth) stack; {@link buildBct} is one consumer of it that pushes
 * into lists. Two traversal implementations is the defect §7.2's first INVALID predicate exists to
 * catch (a theta that was never plumbed).
 *
 * THE chi=0 GUARD. Written as `max(r) <= theta*d`, theta=0 admits any pair whose radii are EXACTLY
 * zero — every pair of single-primitive nodes — so at `leafSize = 1` "theta=0" would silently yield
 * a fully rank-1-lumped operator instead of the exact one. `theta === 0` therefore forces
 * all-inadmissible explicitly, before the inequality is evaluated.
 *
 * @see https://arxiv.org/abs/2006.07859 App. B.2 (BCT), Eq. 28 (admissibility), §4 (disjoint pairs)
 * @see https://github.com/HenrikSchumacher/Repulsor/blob/main/src/BlockClusterTree.hpp (`far_theta2 = 0.25`, i.e. theta = 0.5)
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.5, §5.6, §7.1, §7.2, §11.2 task 3
 * @see src/core/tangentPointEnergy.ts:44 (`calculateDisjointPairs`)
 * @issue utof/repulsive-test2#57
 */
import type { Bvh } from './bvh';

/**
 * Default spatial separation parameter `theta` of Eq. 28.
 *
 * **THE PRODUCTION CHOICE OF `theta` IS OPEN. 0.25 IS THE SHIPPED DEFAULT, NOT A DECIDED VALUE.**
 * `[A46 item 7]` withdrew gate H6's RED as a theta signal, so nothing currently decides this number
 * and the choice stays open on utof/repulsive-test2#51's own evidence — explicitly NOT row 11's to
 * make. The calibration artefact complies already (`thetaProduction: null`,
 * `thetaProductionReason: "OPEN. [A46 item 7] withdrew H6's RED as a theta signal …"`).
 *
 * SUPERSEDED JUSTIFICATION, quoted rather than deleted because the measurement behind it stands
 * and only the inference drawn from it was withdrawn:
 * > "The VALUE is decided by gate H6 (spec §7.9, task 0c — the adversarial-rhs measurement): GREEN
 * >  keeps 0.5 (the authors' own default; Repulsor `far_theta2 = 0.25`), RED flips it to the
 * >  pre-registered fallback 0.25. **H6 read RED (measured 2026-10-09), so this is 0.25.**
 * >  `min cos∠_A` over the sub-0.9 generalized-eigenmode rhs is 0.8585 at n=2000, below the 0.9 bar
 * >  and equal to the analytic Kantorovich bound `2√(c₁c₂)/(c₁+c₂)` to zero difference; at
 * >  theta=0.25 the bad-mode set is EMPTY at every n tested … Both facts are why the decision rests
 * >  on H6's pre-registration and not on a post-hoc reading."
 *
 * WHY IT WAS WITHDRAWN (`[A46 item 7]`, verbatim): *"theta=0.25 reads 0.7649, theta=0.5 reads
 * 0.7856, and the exact δ=0 operator reads 0.7644 — so **theta is not the variable and switching it
 * changes nothing**. … The RED is WITHDRAWN as a theta signal."* The control row is the decisive
 * one: the EXACT, uncompressed operator scores essentially the same as theta=0.25, so the quantity
 * H6 gated was never responding to theta. `[A47 item 5]` reaches the same conclusion from the other
 * end — the adjacency certificate pins `nFar` and `nearPairs` for every theta ≥ 4, so theta cannot
 * over-compress this operator and *"theta is not a knock-out axis for any accuracy gate in §7"*.
 *
 * The COST of 0.25 over 0.5 is unchanged and still the reason the choice matters: near-field work
 * rises 2.09x (nearPairs 3 245 868 → 6 785 670 at n=30 000) and the 12-column apply 1.98x
 * (128.65 → 254.87 ms, MEASURED), which §4.4 extrapolates to a 0.20–1.07 s GPU solve — above the top
 * of the 0.05–0.5 s band claimed in #51. §4.5 separately measures that theta buys no step-quality
 * improvement (end-to-end ΔE spread ≤0.0008 across all theta, non-monotone). With H6 withdrawn,
 * these two facts are the live input to #51's decision and there is no gate standing in for it.
 *
 * Flipping it is this one line by design (CLAUDE.md: prefer swappable strategies over one hardcoded
 * kernel). Nothing else in `src/` or `test/` may hardcode the default: tests that pin theta=0.5
 * oracles pass `theta: 0.5` explicitly, because those are theta=0.5 measurements regardless of
 * what the default becomes.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §4.1, §4.4, §4.5, §7.9, A46 item 7, A47 item 5
 * @issue utof/repulsive-test2#51
 */
export const DEFAULT_THETA = 0.25;

/**
 * Default tangential bound `theta_T` of Eq. 28: `Infinity` = OFF, the convention every oracle was
 * measured under (`THETA_T: null`). See the module TSDoc for why the high-order kernel does not
 * need it and the low-order kernel in principle does.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.5
 */
export const DEFAULT_THETA_T = Number.POSITIVE_INFINITY;

/**
 * Traversal parameters. Every field is optional; see {@link resolveBctParams} for the defaults.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.5, §5.6
 */
export interface BctParams {
    /**
     * Spatial separation parameter of Eq. 28 (the spec's `chi`). Finite, `>= 0`. `0` is the exact
     * all-inadmissible path (module TSDoc, "chi=0 guard"). Default {@link DEFAULT_THETA}.
     */
    theta?: number;
    /** Tangential radius bound of Eq. 28; `> 0`, `Infinity` disables. Default {@link DEFAULT_THETA_T}. */
    thetaT?: number;
    /**
     * Enforce the adjacency certificate (default `true`). `false` produces a WRONG operator —
     * adjacent edge pairs the energy omits can land in a far block — and exists only to measure
     * what the certificate costs in extra near-field pairs and to prove, in tests, that it is
     * load-bearing.
     */
    adjacencyCertificate?: boolean;
    /**
     * KNOCK-OUT INSTRUMENT, test/calibration only — never set in production. Flips Eq. 28 to
     * `max(r_x^A, r_x^B) >= theta * |x_A - x_B|` (one sign), which is spec §7.2 H1-a's knock-out
     * (2): `delta_compress` must then RISE as theta falls, so the gate reads RED. Task 3 asserts the
     * flip changes the classification; task 4 asserts the RED. Do not delete as dead code.
     * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §7.2
     */
    invertAdmissibility?: boolean;
}

/** {@link BctParams} with every default applied. */
export type ResolvedBctParams = Required<BctParams>;

/**
 * Applies the defaults and validates. Throws `RangeError` on a theta that is negative, NaN or
 * infinite, or a thetaT that is NaN or `<= 0`: chi=0 is the MINIMUM of the runtime range, and
 * "anything `<= 0` means exact" would let a NaN from upstream read as the exact path.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.6
 */
export function resolveBctParams(params: BctParams = {}): ResolvedBctParams {
    const theta = params.theta ?? DEFAULT_THETA;
    const thetaT = params.thetaT ?? DEFAULT_THETA_T;
    if (!(Number.isFinite(theta) && theta >= 0)) {
        throw new RangeError(`bct: theta must be finite and >= 0, got ${theta}`);
    }
    if (!(thetaT > 0)) {
        throw new RangeError(`bct: thetaT must be > 0 (Infinity disables), got ${thetaT}`);
    }
    return {
        theta,
        thetaT,
        adjacencyCertificate: params.adjacencyCertificate ?? true,
        invertAdmissibility: params.invertAdmissibility ?? false,
    };
}

/**
 * Counts collected by one traversal. Field names match the spike's `BctStats` so the oracle JSON
 * keys read across one-to-one (`rows[].stats`). `adjacentSkipped` is deliberately ABSENT: the
 * spike's build-only stats hardcoded it to 0 (`scratch/bct.ts:237`); its only increment site is
 * the near-field precompute (`scratch/bct.ts:359`), whose count `run-bct.ts` never copied into the
 * struct rows. It is a near-field quantity and belongs to task 4.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §11.2 task 3
 */
export interface BctStats {
    /** Admissible, certified blocks — one rank-1 kernel evaluation each. */
    nFar: number;
    /** Leaf-leaf blocks summed exactly. */
    nNear: number;
    /** `sum over near blocks of |A|*|B|` — the exact edge-pair count per apply; `kappa_near = nearPairs / m`. */
    nearPairs: number;
    /** Blocks that passed Eq. 28 but failed the adjacency certificate (and were split further). */
    adjacencyVetoed: number;
    /** `true` iff `theta === 0`: the guard forced every block inadmissible. */
    exact: boolean;
    nNodes: number;
    depth: number;
    leafCount: number;
    /** Mean `|A|` over far blocks (the "fan-out" of #57). */
    meanFarRows: number;
    /** Mean `|A|` over near blocks. */
    meanNearRows: number;
    /** Peak stack occupancy in ints — pins the push ORDER, which fixes the list order. */
    maxStackDepth: number;
}

/** Called once per block with the two BVH node ids `(A, B)`; ordered pairs, `A` is the row side. */
export type BlockVisitor = (A: number, B: number) => void;

/**
 * Result of {@link traverseBct}: the counts, the wall time, and the bytes the traversal itself
 * allocated (its stack only — the point of the streaming path).
 */
export interface BctTraversal {
    stats: BctStats;
    traversalMs: number;
    /** Bytes allocated by the traversal: the O(depth) stack. Independent of the block count. */
    bytes: number;
}

/**
 * THE dual traversal from `(root, root)`, streaming: `onFar(A, B)` per admissible certified block,
 * `onNear(A, B)` per leaf-leaf block, in the spike's exact order. Allocates an O(depth) stack and
 * nothing proportional to the number of blocks, so it is the chi=0 path §7.1 requires
 * ({@link buildBct} is a consumer of this function, not a second implementation).
 *
 * Push order (DO NOT reorder — it fixes the far/near list order and therefore the apply's
 * accumulation order): both internal → `(Al,Bl),(Al,Br),(Ar,Bl),(Ar,Br)`; only A a leaf →
 * `(A,Bl),(A,Br)`; only B a leaf → `(Al,B),(Ar,B)`. Pops are LIFO.
 *
 * Stack bound: a pop expands into at most 4 frames (+3 net), and along any DFS path the number of
 * expansions is at most `2 * depth` (each expansion descends at least one of A, B by one level),
 * so the stack never holds more than `1 + 6 * depth` frames; MEASURED peak 74 ints = 37 frames
 * at depth 12. The overflow throw is a guard against a corrupt tree, not an expected path.
 *
 * Why this takes NO accumulator variant, on purpose: admissibility and block structure are decided in
 * f64 from `bvh` geometry alone, so the f32 and f64 hierarchical operators (`prepareA` variants) share
 * ONE block structure -- the same `bvh` and `bct` objects -- and an f32-vs-f64 solve differential
 * isolates ARITHMETIC error (the compression error is common-mode and cancels). Do NOT rebuild or
 * re-test admissibility in f32, or per variant: that silently destroys the differential and no pin
 * would notice. MEASURED: gate G1 (same `bvh`/`bct` objects, equal `stats`) holds on every cell of the
 * f32 ladder and a cross-theta mutant (`buildAt(c, 0.25)`) trips it. The differential bounds
 * arithmetic error on a FIXED structure; it does NOT test an f32 admissibility test (none exists).
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.5 (rules), §7.1 (streaming)
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md A44 §5 (G1 on the ladder), A42 §5 (gate bookkeeping)
 */
export function traverseBct(
    bvh: Bvh,
    params: BctParams,
    onFar: BlockVisitor,
    onNear: BlockVisitor,
): BctTraversal {
    const t0 = performance.now(); // not Bun.nanoseconds: src/ runs in browsers/workers.
    const {
        theta,
        thetaT,
        adjacencyCertificate: certify,
        invertAdmissibility,
    } = resolveBctParams(params);
    const { left, right, begin, end, C, rx, rt, maxEll } = bvh;
    // chi=0 guard (module TSDoc): `0 <= 0 * d` is true for radius-0 nodes, so the inequality
    // alone cannot express "exact". Force it.
    const exact = theta === 0;

    let nFar = 0;
    let nNear = 0;
    let nearPairs = 0;
    let farRows = 0;
    let nearRows = 0;
    let adjacencyVetoed = 0;
    let maxStackDepth = 0;

    const stack = new Int32Array(2 * (6 * bvh.depth + 8));
    let sp = 0;
    stack[sp++] = 0;
    stack[sp++] = 0;

    const push = (x: number, y: number) => {
        if (sp + 2 > stack.length) throw new Error('bct: traversal stack overflow (corrupt tree?)');
        stack[sp++] = x;
        stack[sp++] = y;
        if (sp > maxStackDepth) maxStackDepth = sp;
    };

    while (sp > 0) {
        const B = stack[--sp];
        const A = stack[--sp];

        if (!exact) {
            const dx = C[3 * A] - C[3 * B];
            const dy = C[3 * A + 1] - C[3 * B + 1];
            const dz = C[3 * A + 2] - C[3 * B + 2];
            const d = Math.sqrt(dx * dx + dy * dy + dz * dz);

            const maxR = rx[A] > rx[B] ? rx[A] : rx[B];
            const maxT = rt[A] > rt[B] ? rt[A] : rt[B];
            // Eq. 28 (barycentre form); the inverted branch is §7.2's knock-out (2).
            const sep = invertAdmissibility ? maxR >= theta * d : maxR <= theta * d;
            const tang = maxT <= thetaT;
            if (sep && tang) {
                // Adjacency certificate — proof in the module TSDoc.
                if (!certify || d > rx[A] + rx[B] + 0.5 * (maxEll[A] + maxEll[B])) {
                    nFar++;
                    farRows += end[A] - begin[A];
                    onFar(A, B);
                    continue;
                }
                adjacencyVetoed++;
            }
        }

        const aLeaf = left[A] < 0;
        const bLeaf = left[B] < 0;
        if (aLeaf && bLeaf) {
            nNear++;
            const rows = end[A] - begin[A];
            nearRows += rows;
            nearPairs += rows * (end[B] - begin[B]);
            onNear(A, B);
            continue;
        }
        // One-leaf split (module TSDoc): split the non-leaf side only.
        if (aLeaf) {
            push(A, left[B]);
            push(A, right[B]);
        } else if (bLeaf) {
            push(left[A], B);
            push(right[A], B);
        } else {
            push(left[A], left[B]);
            push(left[A], right[B]);
            push(right[A], left[B]);
            push(right[A], right[B]);
        }
    }

    return {
        stats: {
            nFar,
            nNear,
            nearPairs,
            adjacencyVetoed,
            exact,
            nNodes: bvh.nNodes,
            depth: bvh.depth,
            leafCount: bvh.leafCount,
            meanFarRows: nFar ? farRows / nFar : 0,
            meanNearRows: nNear ? nearRows / nNear : 0,
            maxStackDepth,
        },
        traversalMs: performance.now() - t0,
        bytes: 4 * stack.length,
    };
}

/**
 * Materialised block lists — what the per-iteration apply (task 4) walks when `theta > 0`.
 * Never build this at chi=0 at scale: use {@link traverseBct} (spec §7.1).
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.5, §11.2 task 3
 */
export interface Bct {
    /** Far blocks `(farA[k], farB[k])`, BVH node ids, in traversal order. */
    farA: Int32Array;
    farB: Int32Array;
    /** Near (leaf-leaf) blocks `(nearA[k], nearB[k])`, in traversal order. */
    nearA: Int32Array;
    nearB: Int32Array;
    stats: BctStats;
    buildMs: number;
    /**
     * Bytes retained by the four lists: exactly `8 * (nFar + nNear)`. The lists are trimmed to
     * length (no growth-buffer over-retention), so unlike `Bvh.bytes` (task 2: full capacity)
     * this is the used size — the spike's convention (`scratch/bct.ts:231`), kept because at
     * chi=0 the list IS the memory question and a 2x growth slack would misreport it.
     */
    bytes: number;
    params: ResolvedBctParams;
}

/** Growable Int32 pair list; `take()` trims to length so nothing beyond `n` is retained. */
class PairList {
    a: Int32Array;
    b: Int32Array;
    n = 0;
    constructor(cap: number) {
        this.a = new Int32Array(cap);
        this.b = new Int32Array(cap);
    }
    push(x: number, y: number): void {
        if (this.n === this.a.length) {
            const na = new Int32Array(this.a.length * 2);
            const nb = new Int32Array(this.b.length * 2);
            na.set(this.a);
            nb.set(this.b);
            this.a = na;
            this.b = nb;
        }
        this.a[this.n] = x;
        this.b[this.n] = y;
        this.n++;
    }
    take(): { a: Int32Array; b: Int32Array } {
        return { a: this.a.slice(0, this.n), b: this.b.slice(0, this.n) };
    }
}

/**
 * Builds the block lists by running {@link traverseBct} once with list-pushing visitors. Same
 * traversal, same order; `stats` are the traversal's. Variant-free for the same reason as
 * {@link traverseBct} (Why: one shared block structure across the f32 and f64 operators).
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.5, §7.1
 * @issue utof/repulsive-test2#57
 */
export function buildBct(bvh: Bvh, params: BctParams = {}): Bct {
    const t0 = performance.now();
    const resolved = resolveBctParams(params);
    const far = new PairList(1 << 14);
    const near = new PairList(1 << 14);
    const t = traverseBct(
        bvh,
        resolved,
        (A, B) => far.push(A, B),
        (A, B) => near.push(A, B),
    );
    const f = far.take();
    const n = near.take();
    return {
        farA: f.a,
        farB: f.b,
        nearA: n.a,
        nearB: n.b,
        stats: t.stats,
        buildMs: performance.now() - t0,
        bytes: 8 * (far.n + near.n),
        params: resolved,
    };
}

/** One rung of {@link thetaLadder}. */
export interface ThetaLadderRung {
    theta: number;
    stats: BctStats;
    /** `true` iff `stats.nearPairs` equals the previous rung's — the `kappa_near` tie of §7.2. */
    tiedWithPrevious: boolean;
}

/** Result of {@link thetaLadder}: the tie is a FLAG, never swallowed. */
export interface ThetaLadder {
    rungs: ThetaLadderRung[];
    /** `[theta_k, theta_{k+1}]` for every tied consecutive pair. */
    ties: [number, number][];
    anyTie: boolean;
}

/**
 * Streams the traversal once per theta (counts only, no lists) and flags consecutive rungs whose
 * `kappa_near` (`nearPairs`) is identical. Spec §7.2 H1-a reads `delta_compress` down a theta
 * ladder and has two tie cases that MUST be told apart: knock-out (1), theta never plumbed — every
 * rung returns the same number although `kappa_near` differs (INVALID); and must-NOT-trip (3), a
 * fixture where two rungs genuinely have the same `kappa_near`, which reads GREEN WITH TIE FLAG.
 * The flag is on `kappa_near` ONLY: MEASURED on `circle` n=960, 0.5 -> 0.25 keeps `nearPairs`
 * 21598 and the near lists bit-identical while `nFar` goes 748 -> 904 (vetoed blocks at 0.5 are
 * merely inadmissible at 0.25 and split one level deeper), so the far lumping differs and
 * `delta_compress` need not tie exactly. This function supplies the flag and the per-rung stats
 * that distinguish the two cases; task 4 / the gate harness consume it.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §7.2 (H1-a knock-outs, INVALID table)
 */
export function thetaLadder(
    bvh: Bvh,
    thetas: readonly number[],
    params: Omit<BctParams, 'theta'> = {},
): ThetaLadder {
    const noop: BlockVisitor = () => {};
    const rungs: ThetaLadderRung[] = [];
    const ties: [number, number][] = [];
    for (const theta of thetas) {
        const { stats } = traverseBct(bvh, { ...params, theta }, noop, noop);
        const prev = rungs[rungs.length - 1];
        const tiedWithPrevious = prev !== undefined && prev.stats.nearPairs === stats.nearPairs;
        if (tiedWithPrevious) ties.push([prev.theta, theta]);
        rungs.push({ theta, stats, tiedWithPrevious });
    }
    return { rungs, ties, anyTie: ties.length > 0 };
}
