/**
 * Exact O(N) solves for the curve Laplacian `stiffness*L + mass*M` -- the `(-Delta)^-1` in the
 * Repulsive Surfaces preconditioner `Atilde^-1 = (-Delta)^-1 L_{2-s} (-Delta)^-1`, which GMRES
 * applies TWICE per iteration. On a polyline whose vertex `i` is joined to `i+1` the matrix is
 * tridiagonal (open) or cyclic-tridiagonal (closed), so the solve is O(N), allocation-free
 * (caller-supplied workspace) and needs no matrix beyond the three bands plus two corners.
 *
 * Element matrices of a 1-simplex of length `l`:
 *   `stiffness/l * [[1,-1],[-1,1]] + (mass*l/6) * [[2,1],[1,2]]` (stiffness + consistent 1D mass).
 *
 * TOPOLOGY LIMIT: only curves whose edge list is exactly the natural path `(i,i+1)` (open) or that
 * path plus one `(n-1,0)` wrap (closed) are representable. Anything else (junctions, crossings,
 * permuted vertex order, missing/duplicate edges) throws {@link CurveTopologyError} naming the
 * edge -- assembling a wrong matrix silently is the failure this prevents. Task 8b's dispatch must
 * route such curves to the dense path.
 *
 * SINGULAR CASE: pure stiffness has the constants in its kernel on open AND closed curves, so
 * `massWeight <= 0` THROWS (no pseudo-inverse route). Default `massWeight = 1`: mass-weight ruling,
 * spec §11.2 / §12 q1 (CLOSED at 1). It stays a runtime parameter; only the default is decided.
 *
 * @see https://arxiv.org/abs/2107.01664 §5.2 (the preconditioner)
 * @see Numerical Recipes §2.7 (`cyclic`, Sherman-Morrison cyclic solve)
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §11.2 (mass-weight ruling), §5.6
 * @issue utof/repulsive-test2#57
 */
import type { Edge, Vec3 } from '../../testConfigs';

/**
 * Thrown when the edge list is not the natural open path or closed ring (or the geometry/weights
 * make the system singular).
 * Why: a junction/permuted curve has no tridiagonal structure; see module TSDoc.
 */
export class CurveTopologyError extends Error {
    constructor(message: string) {
        super(message);
        this.name = 'CurveTopologyError';
    }
}

/**
 * Banded system. `a[i]` multiplies `x[i-1]` (`a[0]` unused), `c[i]` multiplies `x[i+1]`
 * (`c[n-1]` unused); corners are the `(0,n-1)` / `(n-1,0)` entries, closed curves only.
 * Why: the minimal data `solveLaplacian` and {@link triResidual} need; consumed by task 6.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.6
 */
export interface TriSystem {
    n: number;
    a: Float64Array;
    b: Float64Array;
    c: Float64Array;
    cornerTR: number;
    cornerBL: number;
    closed: boolean;
}

/**
 * Assembles `stiffnessWeight*L + massWeight*M`; derives `closed` from the edge list.
 * @throws CurveTopologyError on unrepresentable topology, zero-length edges, or `massWeight <= 0`.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §11.2 (default massWeight = 1)
 */
export function assembleCurveLaplacian(
    vertices: Vec3[],
    edges: Edge[],
    stiffnessWeight = 1,
    massWeight = 1,
): TriSystem {
    const n = vertices.length;
    if (n < 2) throw new CurveTopologyError(`assembleCurveLaplacian: need >= 2 vertices, got ${n}`);
    // Singular for ANY topology: constants lie in the kernel of pure stiffness (spec §11.2 / §12 q1).
    if (!(massWeight > 0)) {
        throw new CurveTopologyError(
            `assembleCurveLaplacian: massWeight=${massWeight} makes the system singular (constants in kernel); need > 0`,
        );
    }
    // seen[i] marks path edge (i,i+1); seen[n-1] marks the wrap (n-1,0).
    const seen = new Uint8Array(n);
    let closed = false;
    for (const [p, q] of edges) {
        let slot: number;
        if (q === p + 1 && p >= 0 && q < n) slot = p;
        else if (p === n - 1 && q === 0 && n >= 3) {
            slot = n - 1;
            closed = true;
        } else
            throw new CurveTopologyError(
                `assembleCurveLaplacian: edge [${p},${q}] is not (i,i+1) or the wrap (n-1,0)`,
            );
        if (seen[slot])
            throw new CurveTopologyError(`assembleCurveLaplacian: duplicate edge [${p},${q}]`);
        seen[slot] = 1;
    }
    const last = closed ? n : n - 1;
    for (let i = 0; i < last; i++) {
        if (!seen[i]) {
            throw new CurveTopologyError(
                `assembleCurveLaplacian: missing edge [${i},${(i + 1) % n}]`,
            );
        }
    }

    const a = new Float64Array(n);
    const b = new Float64Array(n);
    const c = new Float64Array(n);
    let cornerTR = 0;
    let cornerBL = 0;
    for (const [i1, i2] of edges) {
        const dx = vertices[i2][0] - vertices[i1][0];
        const dy = vertices[i2][1] - vertices[i1][1];
        const dz = vertices[i2][2] - vertices[i1][2];
        const l = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (!(l > 0))
            throw new CurveTopologyError(`assembleCurveLaplacian: zero-length edge [${i1},${i2}]`);
        const k = stiffnessWeight / l;
        const mm = (massWeight * l) / 6;
        b[i1] += k + 2 * mm;
        b[i2] += k + 2 * mm;
        const off = -k + mm;
        if (i2 === i1 + 1) {
            c[i1] += off;
            a[i2] += off;
        } else {
            // Wrap edge (n-1,0) lands in the corners, not the bands (spec §5.2 / NR §2.7 alpha, beta).
            cornerTR += off; // row 0, col n-1
            cornerBL += off; // row n-1, col 0
        }
    }
    return { n, a, b, c, cornerTR, cornerBL, closed };
}

/**
 * Caller-owned scratch: allocate once, reuse every solve.
 * Why: GMRES calls the solve twice per iteration, so a per-solve allocation is a per-iteration one.
 * `thomasSweeps` counts sweeps of the last solve (2 closed, 1 open) -- structure check in tests.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.2
 */
export interface LaplacianWorkspace {
    bb: Float64Array;
    u: Float64Array;
    z: Float64Array;
    cp: Float64Array;
    dp: Float64Array;
    thomasSweeps: number;
}

/** Allocates a {@link LaplacianWorkspace}. Why: see the interface (no allocation per solve). */
export function makeLaplacianWorkspace(n: number): LaplacianWorkspace {
    return {
        bb: new Float64Array(n),
        u: new Float64Array(n),
        z: new Float64Array(n),
        cp: new Float64Array(n),
        dp: new Float64Array(n),
        thomasSweeps: 0,
    };
}

/**
 * Thomas algorithm, O(N), NO pivoting: diagonally dominant here (mass > 0), and pivoting would
 * break the O(N) band structure. Why: spec §5.2.
 */
function thomas(
    n: number,
    a: Float64Array,
    b: Float64Array,
    c: Float64Array,
    r: Float64Array,
    x: Float64Array,
    cp: Float64Array,
    dp: Float64Array,
): void {
    let beta = b[0];
    if (beta === 0) throw new Error('laplacian thomas: zero pivot at 0');
    cp[0] = c[0] / beta;
    dp[0] = r[0] / beta;
    for (let i = 1; i < n; i++) {
        beta = b[i] - a[i] * cp[i - 1];
        if (beta === 0) throw new Error(`laplacian thomas: zero pivot at ${i}`);
        cp[i] = c[i] / beta;
        dp[i] = (r[i] - a[i] * dp[i - 1]) / beta;
    }
    x[n - 1] = dp[n - 1];
    for (let i = n - 2; i >= 0; i--) x[i] = dp[i] - cp[i] * x[i + 1];
}

/**
 * Solves `A x = r`: one Thomas sweep (open) or Sherman-Morrison with two (closed). `x` must not
 * alias `r`. Allocation-free.
 * @see Numerical Recipes §2.7 (`cyclic`)
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.2
 */
export function solveLaplacian(
    sys: TriSystem,
    r: Float64Array,
    x: Float64Array,
    ws: LaplacianWorkspace,
): void {
    const { n, a, b, c } = sys;
    if (!sys.closed) {
        thomas(n, a, b, c, r, x, ws.cp, ws.dp);
        ws.thomasSweeps = 1;
        return;
    }
    // NR naming: alpha = bottom-left corner, beta = top-right corner.
    const alpha = sys.cornerBL;
    const beta = sys.cornerTR;
    // gamma = -b[0] is NR's choice: it avoids cancellation in the perturbed ends (b[0]-gamma = 2 b[0]).
    const gamma = -b[0];
    ws.bb.set(b);
    ws.bb[0] = b[0] - gamma;
    ws.bb[n - 1] = b[n - 1] - (alpha * beta) / gamma;
    thomas(n, a, ws.bb, c, r, x, ws.cp, ws.dp);
    ws.u.fill(0);
    ws.u[0] = gamma;
    ws.u[n - 1] = alpha;
    thomas(n, a, ws.bb, c, ws.u, ws.z, ws.cp, ws.dp);
    const fact = (x[0] + (beta * x[n - 1]) / gamma) / (1 + ws.z[0] + (beta * ws.z[n - 1]) / gamma);
    for (let i = 0; i < n; i++) x[i] -= fact * ws.z[i];
    ws.thomasSweeps = 2;
}

/**
 * Relative residual `||Ax - r|| / ||r||` against the assembled bands -- the honesty check.
 * Why: a fast solve that does not solve is not a result (spec §11.2 bars are on this quantity).
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §11.2
 */
export function triResidual(sys: TriSystem, x: Float64Array, r: Float64Array): number {
    const { n, a, b, c, cornerTR, cornerBL, closed } = sys;
    let num = 0;
    let den = 0;
    for (let i = 0; i < n; i++) {
        let v = b[i] * x[i];
        if (i > 0) v += a[i] * x[i - 1];
        if (i < n - 1) v += c[i] * x[i + 1];
        if (closed && i === 0) v += cornerTR * x[n - 1];
        if (closed && i === n - 1) v += cornerBL * x[0];
        const d = v - r[i];
        num += d * d;
        den += r[i] * r[i];
    }
    return Math.sqrt(num / den);
}
