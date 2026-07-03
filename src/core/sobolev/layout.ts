/**
 * Coordinate-block vector layout for the fractional Sobolev-gradient machinery.
 *
 * Why: the |V|×|V| scalar matrices (B, B⁰, A) are expanded into 3|V|×3|V|
 * block-diagonal operators and solved against a flattened dE. Every consumer
 * (inner-product assembly, the saddle solve, the barycenter constraint Jacobian)
 * must agree on the SAME flat layout, or indices silently misalign.
 *
 * Layout: `[x0..x(n-1), y0..y(n-1), z0..z(n-1)]` — three contiguous coordinate
 * blocks, not per-vertex-interleaved. This mirrors the oracle's
 * `flatten_vec3_block`/`unflatten_vec3_block`/`block_index` exactly, so TS and
 * Python index the same flat vector the same way.
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §A ("Use coordinate-block flattening")
 * @see oracle/tpe_stage1_oracle.py (flatten_vec3_block / unflatten_vec3_block / block_index)
 */
import type { Vec3 } from '../testConfigs';

/**
 * Flattens `Vec3[]` (length n) into coordinate-block layout (length 3n):
 * `[x0..x(n-1), y0..y(n-1), z0..z(n-1)]`.
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §A
 */
export function flatten(vecs: Vec3[]): number[] {
    const n = vecs.length;
    const out = new Array<number>(3 * n);
    for (let i = 0; i < n; i++) {
        out[i] = vecs[i][0];
        out[n + i] = vecs[i][1];
        out[2 * n + i] = vecs[i][2];
    }
    return out;
}

/**
 * Inverse of {@link flatten}: coordinate-block flat vector (length 3n) back to
 * `Vec3[]` (length n).
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §A
 */
export function unflatten(flat: number[]): Vec3[] {
    if (flat.length % 3 !== 0) {
        throw new Error('flat vector length must be divisible by 3');
    }
    const n = flat.length / 3;
    const out: Vec3[] = new Array(n);
    for (let i = 0; i < n; i++) {
        out[i] = [flat[i], flat[n + i], flat[2 * n + i]];
    }
    return out;
}

/**
 * Index into the coordinate-block flat vector for a given coordinate
 * (0=x, 1=y, 2=z) and vertex, given n = |V|.
 * @see oracle/tpe_stage1_oracle.py (block_index)
 */
export function blockIndex(coord: number, vertex: number, n: number): number {
    return coord * n + vertex;
}

/**
 * Expands the |V|×|V| scalar Sobolev inner-product matrix A into the 3|V|×3|V| vector-valued
 * block-diagonal operator `Ā = diag(A, A, A)`, consistent with the coordinate-block layout of
 * {@link flatten}/{@link unflatten} (x-block, then y-block, then z-block).
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §A ("expanded to vector-valued functions by Ā = ...")
 * @see oracle/tpe_stage1_oracle.py (expand_vector_inner_product)
 */
export function expandBlockDiag(A: number[][]): number[][] {
    const n = A.length;
    const size = 3 * n;
    const out: number[][] = Array.from({ length: size }, () => new Array<number>(size).fill(0));
    // Three coordinate blocks on the diagonal, zero elsewhere — off-diagonal blocks are left as
    // the zero-fill above; only the diagonal blocks are written.
    for (let block = 0; block < 3; block++) {
        const offset = block * n;
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                out[offset + i][offset + j] = A[i][j];
            }
        }
    }
    return out;
}
