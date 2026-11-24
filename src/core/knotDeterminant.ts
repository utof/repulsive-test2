/**
 * The knot determinant of a closed polyline, modulo a prime: an INDEPENDENT check that a flow
 * kept the knot type. It is computed from one projection (crossings → Fox colouring matrix →
 * a minor's determinant) and shares no code with the collision tests it is used to audit.
 *
 * What it can and cannot say: a changed determinant PROVES the knot type changed. An unchanged
 * one does not prove it was kept (the unknot has 1, the trefoil 3, the figure-eight 5, but
 * e.g. the Conway and Kinoshita–Terasaka knots also have 1).
 * Why modulo a prime: the integer determinant of a tangled embedding (hundreds of crossings)
 * overflows a double, and only equality before/after matters. The value is returned as
 * `min(d, p − d)` because the sign of a minor is not an invariant.
 * @see docs/plans/2026-11-24-medial-sphere-method.md
 */
import type { ClosedCurve } from './medialSphere';

/** Largest projection the audit will take on (a 6000² matrix of doubles is 288 MB). */
export const KNOT_DETERMINANT_MAX_CROSSINGS = 6000;

/** Primes below 2^26, so a product of two residues is exact in a double. */
export const KNOT_DETERMINANT_PRIMES = [67108859, 67108837] as const;

/**
 * `|det| mod p` of the curve's colouring matrix, as `min(d, p − d)`; `null` when the chosen
 * projection is degenerate (a crossing too close to a vertex), in which case call again with
 * another `direction`.
 */
export function knotDeterminantMod(
    c: ClosedCurve,
    p: number = KNOT_DETERMINANT_PRIMES[0],
    direction: readonly [number, number, number] = [0.2672612, 0.5345225, 0.8017837],
): number | null {
    const { n, x } = c;
    // Orthonormal frame (u, v, w = direction): project on (u, v), height along w.
    const [wx, wy, wz] = direction;
    let ux = -wy;
    let uy = wx;
    const uz = 0;
    const lu = Math.hypot(ux, uy, uz);
    ux /= lu;
    uy /= lu;
    const vx = wy * uz - wz * uy;
    const vy = wz * ux - wx * uz;
    const vz = wx * uy - wy * ux;
    const px = new Float64Array(n);
    const py = new Float64Array(n);
    const pz = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        px[i] = ux * x[3 * i] + uy * x[3 * i + 1] + uz * x[3 * i + 2];
        py[i] = vx * x[3 * i] + vy * x[3 * i + 1] + vz * x[3 * i + 2];
        pz[i] = wx * x[3 * i] + wy * x[3 * i + 1] + wz * x[3 * i + 2];
    }
    // Crossings: for each, the under edge + parameter along it, and the over edge.
    const underEdge: number[] = [];
    const underT: number[] = [];
    const overEdge: number[] = [];
    const EDGE_TOL = 1e-9;
    const HEIGHT_TOL = 1e-7;
    let zLo = Infinity;
    let zHi = -Infinity;
    for (let i = 0; i < n; i++) {
        zLo = Math.min(zLo, pz[i], px[i], py[i]);
        zHi = Math.max(zHi, pz[i], px[i], py[i]);
    }
    const extent = zHi - zLo || 1;
    for (let i = 0; i < n; i++) {
        const i2 = (i + 1) % n;
        const ax = px[i];
        const ay = py[i];
        const bx = px[i2] - ax;
        const by = py[i2] - ay;
        const minX = Math.min(ax, px[i2]);
        const maxX = Math.max(ax, px[i2]);
        const minY = Math.min(ay, py[i2]);
        const maxY = Math.max(ay, py[i2]);
        for (let j = i + 2; j < n; j++) {
            const j2 = (j + 1) % n;
            if (j2 === i) continue;
            if (Math.max(px[j], px[j2]) < minX || Math.min(px[j], px[j2]) > maxX) continue;
            if (Math.max(py[j], py[j2]) < minY || Math.min(py[j], py[j2]) > maxY) continue;
            const cx = px[j2] - px[j];
            const cy = py[j2] - py[j];
            const den = bx * cy - by * cx;
            if (den === 0) continue;
            const ex = px[j] - ax;
            const ey = py[j] - ay;
            const s = (ex * cy - ey * cx) / den;
            const t = (ex * by - ey * bx) / den;
            if (s < -EDGE_TOL || s > 1 + EDGE_TOL || t < -EDGE_TOL || t > 1 + EDGE_TOL) continue;
            // A crossing at (or within rounding of) a vertex: this projection is not generic.
            if (s < EDGE_TOL || s > 1 - EDGE_TOL || t < EDGE_TOL || t > 1 - EDGE_TOL) return null;
            const zi = pz[i] + s * (pz[i2] - pz[i]);
            const zj = pz[j] + t * (pz[j2] - pz[j]);
            // Which strand is on top must be unambiguous. Two strands pressed together (a
            // stuck flow leaves them a rounding error apart) have no reliable over/under in
            // the projections whose crossing falls on the contact: measured on one dataset
            // trefoil, one direction of five then reported determinant 1 and the other four 3.
            // Such a projection is refused, like a crossing at a vertex.
            if (Math.abs(zi - zj) <= HEIGHT_TOL * extent) return null;
            if (zi < zj) {
                underEdge.push(i);
                underT.push(s);
                overEdge.push(j);
            } else {
                underEdge.push(j);
                underT.push(t);
                overEdge.push(i);
            }
        }
    }
    const k = underEdge.length;
    if (k <= 2) return 1;
    // A dense k×k matrix: beyond this the picture is not a tangle but an explosion (a flow
    // with its collision test switched off produced 10^5 crossings). Refuse rather than
    // allocate gigabytes; the caller reads `null` as "could not audit".
    if (k > KNOT_DETERMINANT_MAX_CROSSINGS) return null;
    // Under-crossings in order along the curve cut it into k arcs: arc a runs from
    // under-crossing a to under-crossing a+1 (cyclically).
    const order = Array.from({ length: k }, (_, q) => q).sort(
        (q, r) => underEdge[q] - underEdge[r] || underT[q] - underT[r],
    );
    const pos = new Float64Array(k); // curve parameter of the a-th under-crossing in order
    for (let a = 0; a < k; a++) pos[a] = underEdge[order[a]] + underT[order[a]];
    // Arc containing curve parameter `u`: the last under-crossing at or before it.
    const arcOf = (u: number): number => {
        let lo = 0;
        let hi = k; // first index with pos > u
        while (lo < hi) {
            const mid = (lo + hi) >> 1;
            if (pos[mid] <= u) lo = mid + 1;
            else hi = mid;
        }
        return (lo - 1 + k) % k;
    };
    // Over-strand parameter of each crossing: recompute from the over edge.
    // Colouring relation at the a-th under-crossing: 2·over − in − out = 0, with in = arc a−1,
    // out = arc a. Drop the last row and column, take the determinant mod p.
    const m = k - 1;
    const M = new Float64Array(m * m);
    const add = (row: number, col: number, val: number): void => {
        if (row < m && col < m) M[row * m + col] = (((M[row * m + col] + val) % p) + p) % p;
    };
    for (let a = 0; a < k; a++) {
        const q = order[a];
        const oe = overEdge[q];
        const ue = underEdge[q];
        // Parameter along the over edge of the same crossing point.
        const o2 = (oe + 1) % n;
        const u2 = (ue + 1) % n;
        const cxp = px[ue] + underT[q] * (px[u2] - px[ue]);
        const cyp = py[ue] + underT[q] * (py[u2] - py[ue]);
        const ox = px[o2] - px[oe];
        const oy = py[o2] - py[oe];
        const ot = ((cxp - px[oe]) * ox + (cyp - py[oe]) * oy) / (ox * ox + oy * oy);
        add(a, arcOf(oe + ot), 2);
        add(a, (a - 1 + k) % k, -1);
        add(a, a, -1);
    }
    // Gaussian elimination mod p.
    const mul = (a: number, b: number): number => {
        // a, b < 2^26: split b so every partial product stays below 2^53.
        const bh = Math.floor(b / 8192);
        const bl = b - bh * 8192;
        return (((a * bh) % p) * 8192 + a * bl) % p;
    };
    const inv = (a: number): number => {
        let r = 1;
        let base = a;
        let e = p - 2;
        while (e > 0) {
            if (e & 1) r = mul(r, base);
            base = mul(base, base);
            e = Math.floor(e / 2);
        }
        return r;
    };
    let det = 1;
    for (let col = 0; col < m; col++) {
        let piv = -1;
        for (let r = col; r < m; r++)
            if (M[r * m + col] !== 0) {
                piv = r;
                break;
            }
        if (piv < 0) return 0;
        if (piv !== col) {
            for (let q = col; q < m; q++) {
                const tmp = M[col * m + q];
                M[col * m + q] = M[piv * m + q];
                M[piv * m + q] = tmp;
            }
            det = (p - det) % p;
        }
        const d = M[col * m + col];
        det = mul(det, d);
        const dInv = inv(d);
        for (let r = col + 1; r < m; r++) {
            const f = M[r * m + col];
            if (f === 0) continue;
            const g = mul(f, dInv);
            for (let q = col; q < m; q++) {
                const sub = mul(g, M[col * m + q]);
                if (sub !== 0) M[r * m + q] = (M[r * m + q] - sub + p) % p;
            }
        }
    }
    return Math.min(det, p - det);
}

/**
 * Fewest crossings seen in a fixed set of 12 projections of the curve — an upper bound on its
 * crossing number, used as "how untangled is this" (a tidy trefoil shows 3, a tidy unknot 0).
 * Why a separate O(n²) loop instead of reusing the audit's: the audit refuses degenerate
 * projections and builds a matrix; this only counts.
 */
export function fewestProjectedCrossings(c: ClosedCurve): number {
    const { n, x } = c;
    let best = Infinity;
    const px = new Float64Array(n);
    const py = new Float64Array(n);
    for (let k = 0; k < 12; k++) {
        // Directions on a Fibonacci spiral over the half sphere.
        const z = (k + 0.5) / 12;
        const rad = Math.sqrt(1 - z * z);
        const phi = k * 2.399963229728653;
        const w = [rad * Math.cos(phi), rad * Math.sin(phi), z];
        const u = Math.abs(w[2]) < 0.9 ? [-w[1], w[0], 0] : [0, -w[2], w[1]];
        const lu = Math.hypot(u[0], u[1], u[2]);
        for (let q = 0; q < 3; q++) u[q] /= lu;
        const v = [w[1] * u[2] - w[2] * u[1], w[2] * u[0] - w[0] * u[2], w[0] * u[1] - w[1] * u[0]];
        for (let i = 0; i < n; i++) {
            px[i] = u[0] * x[3 * i] + u[1] * x[3 * i + 1] + u[2] * x[3 * i + 2];
            py[i] = v[0] * x[3 * i] + v[1] * x[3 * i + 1] + v[2] * x[3 * i + 2];
        }
        let count = 0;
        for (let i = 0; i < n; i++) {
            const i2 = (i + 1) % n;
            const bx = px[i2] - px[i];
            const by = py[i2] - py[i];
            for (let j = i + 2; j < n; j++) {
                const j2 = (j + 1) % n;
                if (j2 === i) continue;
                const o1 = bx * (py[j] - py[i]) - by * (px[j] - px[i]);
                const o2 = bx * (py[j2] - py[i]) - by * (px[j2] - px[i]);
                if (o1 * o2 >= 0) continue;
                const cx = px[j2] - px[j];
                const cy = py[j2] - py[j];
                const o3 = cx * (py[i] - py[j]) - cy * (px[i] - px[j]);
                const o4 = cx * (py[i2] - py[j]) - cy * (px[i2] - px[j]);
                if (o3 * o4 < 0) count++;
            }
        }
        best = Math.min(best, count);
    }
    return best;
}

/**
 * The determinant residues for both primes, trying a few projection directions until one is
 * generic. `null` only if every direction was degenerate.
 */
export function knotSignature(c: ClosedCurve): [number, number] | null {
    const dirs: [number, number, number][] = [
        [0.2672612, 0.5345225, 0.8017837],
        [0.6396021, -0.4264014, 0.6396021],
        [-0.7427814, 0.557086, 0.3713907],
        [0.1, 0.2, 0.9746794],
        [0.9, 0.3, 0.3162278],
    ];

    for (const d of dirs) {
        const a = knotDeterminantMod(c, KNOT_DETERMINANT_PRIMES[0], d);
        const b = a === null ? null : knotDeterminantMod(c, KNOT_DETERMINANT_PRIMES[1], d);
        if (a !== null && b !== null) return [a, b];
    }
    return null;
}
