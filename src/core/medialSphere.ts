/**
 * The medial-sphere flow for a closed curve in 3-D: a second untangling core, next to the
 * tangent-point (repulsive-curves) one, implemented from the paper's formulas.
 *
 * What it does, per step: from every node, in N directions of its normal plane, find the
 * largest sphere that touches the node and contains no other node (capped at `rMax`); pull the
 * node towards the centres of those spheres (more where there is more room) and shorten the
 * curve; precondition that pull with a curve Laplacian that also links nodes sharing a sphere;
 * take the largest step that provably passes no strand through another; re-sample to even
 * spacing. Because the cap is a little under the working radius, the curve tightens: it
 * SHRINKS at even spacing, the node count falls as it goes, and an unknot ends as a point.
 * Every interaction is local (within `2·rMax`), which is the point: the tangent-point energy
 * needs all pairs exactly where strands are packed.
 *
 * @see docs/plans/2026-11-24-medial-sphere-method.md (the plan, the risks, the go/no-go bars)
 * @see local_files/papers/medial-sphere-preconditioning-2025/paper-flow.txt (Noma, Jacobson,
 *      Singh, SIGGRAPH Asia 2025: §4.1 energy, §4.2 preconditioner, §4.3 time stepping,
 *      §4.4 knot untangling)
 * The authors' code is GPL-3 and was read for parameter VALUES only (`r ≈ 3.36/√α`, the
 * start-up rule for `r`); nothing here is ported from it.
 *
 * Differences from the paper, each deliberate:
 *  - the sphere radius is evaluated in closed form (the paper's own `R_il`) over the nodes
 *    within `2·rMax`, not by a binary search of closest-point queries;
 *  - `P⁻¹g` is conjugate gradients preconditioned by the cyclic tridiagonal part (no sparse
 *    Cholesky in this repo);
 *  - the collision test is a conservative bound (`motionIsSafe`), not a swept-surface
 *    intersection test: it never accepts an unsafe step, but may refuse a safe one.
 */
import type { Vec3 } from './testConfigs';

/** Parameters of the flow. Defaults are the paper's untangling settings. */
export interface MedialSphereOptions {
    /** Directions sampled on the normal circle of each node. Paper: 16. */
    nDirections?: number;
    /** Weight of the extra Laplacian links between nodes sharing a sphere. Paper: 0.01. */
    beta?: number;
    /** Mass weight of the preconditioner `P = L* + (λ/r²)M`, dimensionless. Default 4, which
     *  is the paper's λ = 1 at the upper end of the dataset's scale (see the step). */
    lambda?: number;
    /** Step length before the collision bound. Default 1 (the paper uses 10–20; see the step
     *  for why that is unstable here). */
    tau?: number;
    /** `rMax / r`. Below 1 tightens the curve. Paper, untangling: 0.9. */
    capRatio?: number;
    /** Node spacing as a fraction of `r`. Paper: "h < r/5". */
    spacingRatio?: number;
    /** Trial step for this call (≤ `tau`): pass the previous result's `nextTryStep`. */
    tryStep?: number;
}

/** The working scales of one run, fixed at start-up from the input curve. */
export interface MedialSphereScales {
    /** Working radius: the mean radius of the start curve's empty spheres that hit something. */
    r: number;
    rMax: number;
    /** Energy weight, `α = (3.36 / r)²` — the paper's empirical thickness–weight relation. */
    alpha: number;
    /** Target node spacing. */
    h: number;
}

/** A closed polyline as flat xyz; node `i` is joined to `i+1 (mod n)`. */
export interface ClosedCurve {
    n: number;
    x: Float64Array;
}

const sq = (a: number): number => a * a;

/** Flat copy of a closed curve given as `Vec3[]` in loop order. */
export function closedCurveFrom(vertices: readonly Vec3[]): ClosedCurve {
    const n = vertices.length;
    const x = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) {
        x[3 * i] = vertices[i][0];
        x[3 * i + 1] = vertices[i][1];
        x[3 * i + 2] = vertices[i][2];
    }
    return { n, x };
}

/** Total length of a closed curve. */
export function curveLength(c: ClosedCurve): number {
    let s = 0;
    for (let i = 0; i < c.n; i++) {
        const j = (i + 1) % c.n;
        s += Math.hypot(
            c.x[3 * j] - c.x[3 * i],
            c.x[3 * j + 1] - c.x[3 * i + 1],
            c.x[3 * j + 2] - c.x[3 * i + 2],
        );
    }
    return s;
}

/**
 * Uniform hash grid over points: `forNear(i, radius, visit)` calls `visit(j)` for every other
 * point within `radius` (and possibly a few beyond it — callers test the distance they need).
 * Why a grid and not the repo's BVH: every query here has the same small radius, which is the
 * case a grid answers in O(1) per neighbour with no tree to build or walk.
 */
export function makeGrid(x: Float64Array, n: number, cell: number) {
    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    for (let i = 0; i < n; i++) {
        minX = Math.min(minX, x[3 * i]);
        minY = Math.min(minY, x[3 * i + 1]);
        minZ = Math.min(minZ, x[3 * i + 2]);
    }
    const key = (cx: number, cy: number, cz: number): number =>
        ((cx * 73856093) ^ (cy * 19349663) ^ (cz * 83492791)) >>> 0;
    const buckets = new Map<number, number[]>();
    const cellOf = new Int32Array(3 * n);
    for (let i = 0; i < n; i++) {
        const cx = Math.floor((x[3 * i] - minX) / cell);
        const cy = Math.floor((x[3 * i + 1] - minY) / cell);
        const cz = Math.floor((x[3 * i + 2] - minZ) / cell);
        cellOf[3 * i] = cx;
        cellOf[3 * i + 1] = cy;
        cellOf[3 * i + 2] = cz;
        const k = key(cx, cy, cz);
        const b = buckets.get(k);
        if (b) b.push(i);
        else buckets.set(k, [i]);
    }
    return {
        /** Visits every point whose cell is within `ceil(radius / cell)` cells of point `i`'s. */
        forNear(i: number, radius: number, visit: (j: number) => void): void {
            const reach = Math.max(1, Math.ceil(radius / cell));
            const cx = cellOf[3 * i];
            const cy = cellOf[3 * i + 1];
            const cz = cellOf[3 * i + 2];
            for (let dx = -reach; dx <= reach; dx++)
                for (let dy = -reach; dy <= reach; dy++)
                    for (let dz = -reach; dz <= reach; dz++) {
                        const b = buckets.get(key(cx + dx, cy + dy, cz + dz));
                        if (!b) continue;
                        for (let q = 0; q < b.length; q++) {
                            const j = b[q];
                            // A hash collision can return a far cell's points: check the cell.
                            if (
                                j !== i &&
                                cellOf[3 * j] === cx + dx &&
                                cellOf[3 * j + 1] === cy + dy &&
                                cellOf[3 * j + 2] === cz + dz
                            )
                                visit(j);
                        }
                    }
        },
    };
}

/** Result of {@link medialSpheres}. */
export interface MedialSpheres {
    /** `radius[i·N + l]`: the sphere grown from node `i` along direction `l`, capped at `cap`. */
    radius: Float64Array;
    /** `hit[i·N + l]`: the node the sphere touches, or −1 when the cap was reached. */
    hit: Int32Array;
    /** `dir[(i·N + l)·3 ..]`: the unit direction. */
    dir: Float64Array;
}

/**
 * The largest empty sphere through every node in every sampled direction of its normal plane.
 * A sphere of radius R tangent at γ_i with centre γ_i + R·x contains γ_j exactly when
 * `|d|² < 2R (x·d)`, `d = γ_j − γ_i`; so the largest empty one is `min_j |d|² / (2 x·d)` over
 * the nodes on the x side — the paper's `R_il`, in closed form.
 * @see local_files/papers/medial-sphere-preconditioning-2025/paper-flow.txt (§4.2, "R_il")
 */
export function medialSpheres(c: ClosedCurve, cap: number, nDirections: number): MedialSpheres {
    const { n, x } = c;
    const N = nDirections;
    const radius = new Float64Array(n * N).fill(cap);
    const hit = new Int32Array(n * N).fill(-1);
    const dir = new Float64Array(n * N * 3);
    const grid = makeGrid(x, n, 2 * cap);
    const cos = new Float64Array(N);
    const sin = new Float64Array(N);
    for (let l = 0; l < N; l++) {
        cos[l] = Math.cos((2 * Math.PI * l) / N);
        sin[l] = Math.sin((2 * Math.PI * l) / N);
    }
    const reach2 = sq(2 * cap);
    for (let i = 0; i < n; i++) {
        const p = (i + n - 1) % n;
        const q = (i + 1) % n;
        // Tangent: the average of the incoming and outgoing unit vectors (the paper's choice).
        let ax = x[3 * i] - x[3 * p];
        let ay = x[3 * i + 1] - x[3 * p + 1];
        let az = x[3 * i + 2] - x[3 * p + 2];
        let bx = x[3 * q] - x[3 * i];
        let by = x[3 * q + 1] - x[3 * i + 1];
        let bz = x[3 * q + 2] - x[3 * i + 2];
        const la = Math.hypot(ax, ay, az) || 1;
        const lb = Math.hypot(bx, by, bz) || 1;
        ax /= la;
        ay /= la;
        az /= la;
        bx /= lb;
        by /= lb;
        bz /= lb;
        let tx = ax + bx;
        let ty = ay + by;
        let tz = az + bz;
        const lt = Math.hypot(tx, ty, tz);
        if (lt < 1e-12) {
            // A full fold-back: no tangent. Use the incoming edge; the node is re-sampled away.
            tx = ax;
            ty = ay;
            tz = az;
        } else {
            tx /= lt;
            ty /= lt;
            tz /= lt;
        }
        // Any unit u ⟂ t, then v = t × u.
        let ux: number;
        let uy: number;
        let uz: number;
        if (Math.abs(tx) <= Math.abs(ty) && Math.abs(tx) <= Math.abs(tz)) {
            ux = 0;
            uy = -tz;
            uz = ty;
        } else if (Math.abs(ty) <= Math.abs(tz)) {
            ux = -tz;
            uy = 0;
            uz = tx;
        } else {
            ux = -ty;
            uy = tx;
            uz = 0;
        }
        const lu = Math.hypot(ux, uy, uz);
        ux /= lu;
        uy /= lu;
        uz /= lu;
        const vx = ty * uz - tz * uy;
        const vy = tz * ux - tx * uz;
        const vz = tx * uy - ty * ux;
        const base = i * N;
        for (let l = 0; l < N; l++) {
            dir[(base + l) * 3] = cos[l] * ux + sin[l] * vx;
            dir[(base + l) * 3 + 1] = cos[l] * uy + sin[l] * vy;
            dir[(base + l) * 3 + 2] = cos[l] * uz + sin[l] * vz;
        }
        const xi = x[3 * i];
        const yi = x[3 * i + 1];
        const zi = x[3 * i + 2];
        grid.forNear(i, 2 * cap, (j) => {
            const dx = x[3 * j] - xi;
            const dy = x[3 * j + 1] - yi;
            const dz = x[3 * j + 2] - zi;
            const d2 = dx * dx + dy * dy + dz * dz;
            if (d2 >= reach2 || d2 === 0) return;
            for (let l = 0; l < N; l++) {
                const o = (base + l) * 3;
                const along = dir[o] * dx + dir[o + 1] * dy + dir[o + 2] * dz;
                if (along <= 0) continue;
                const R = d2 / (2 * along);
                if (R < radius[base + l]) {
                    radius[base + l] = R;
                    hit[base + l] = j;
                }
            }
        });
    }
    return { radius, hit, dir };
}

/**
 * The working scales, set once from the start curve (the reference implementation's start-up
 * rule): `r` is the mean radius of the spheres that hit another node within `searchCap`.
 */
export function medialSphereScales(
    c: ClosedCurve,
    searchCap: number,
    opts: MedialSphereOptions = {},
): MedialSphereScales {
    const N = opts.nDirections ?? 16;
    const s = medialSpheres(c, searchCap, N);
    let sum = 0;
    let count = 0;
    for (let k = 0; k < s.radius.length; k++) {
        if (s.hit[k] >= 0) {
            sum += s.radius[k];
            count++;
        }
    }
    const r = count > 0 ? sum / count : searchCap;
    return {
        r,
        rMax: (opts.capRatio ?? 0.9) * r,
        alpha: sq(3.36 / r),
        h: (opts.spacingRatio ?? 0.2) * r,
    };
}

/**
 * Does segment (p,q) meet triangle (a,b,c)? Möller–Trumbore, with a small tolerance on the
 * conservative side (a near miss counts as a hit). Coordinates are passed as offsets into one
 * flat array.
 */
function segmentMeetsTriangle(
    x: Float64Array,
    p: number,
    q: number,
    a: number,
    b: number,
    c: number,
): boolean {
    const TOL = 1e-9;
    const e1x = x[3 * b] - x[3 * a];
    const e1y = x[3 * b + 1] - x[3 * a + 1];
    const e1z = x[3 * b + 2] - x[3 * a + 2];
    const e2x = x[3 * c] - x[3 * a];
    const e2y = x[3 * c + 1] - x[3 * a + 1];
    const e2z = x[3 * c + 2] - x[3 * a + 2];
    const dx = x[3 * q] - x[3 * p];
    const dy = x[3 * q + 1] - x[3 * p + 1];
    const dz = x[3 * q + 2] - x[3 * p + 2];
    const hx = dy * e2z - dz * e2y;
    const hy = dz * e2x - dx * e2z;
    const hz = dx * e2y - dy * e2x;
    const det = e1x * hx + e1y * hy + e1z * hz;
    const scale = Math.hypot(e1x, e1y, e1z) * Math.hypot(e2x, e2y, e2z) * Math.hypot(dx, dy, dz);
    // Parallel to the triangle's plane (or a degenerate triangle): no transversal crossing.
    if (Math.abs(det) <= 1e-14 * scale) return false;
    const sx = x[3 * p] - x[3 * a];
    const sy = x[3 * p + 1] - x[3 * a + 1];
    const sz = x[3 * p + 2] - x[3 * a + 2];
    const u = (sx * hx + sy * hy + sz * hz) / det;
    if (u < -TOL || u > 1 + TOL) return false;
    const qx = sy * e1z - sz * e1y;
    const qy = sz * e1x - sx * e1z;
    const qz = sx * e1y - sy * e1x;
    const v = (dx * qx + dy * qy + dz * qz) / det;
    if (v < -TOL || u + v > 1 + TOL) return false;
    const t = (e2x * qx + e2y * qy + e2z * qz) / det;
    return t >= -TOL && t <= 1 + TOL;
}

/**
 * Re-sample a closed curve to spacing in `[h, 2h]`: drop a node when one of its edges is
 * shorter than `h`, the chord that replaces its two edges is at most `2h`, and NO other edge
 * of the curve meets the triangle (previous, node, next); then split every edge longer than
 * `2h` into equal pieces (the shape is unchanged).
 * Why the triangle test: dropping a node moves the curve across that triangle, and a strand
 * piercing it would end up on the other side — a change of knot type that the collision test
 * of the step never sees. The first version of this function used a distance heuristic
 * instead and let a dataset knot collapse to a point.
 */
export function resampleClosedCurve(c: ClosedCurve, h: number): ClosedCurve {
    let { n, x } = c;
    const dist = (a: number, b: number): number =>
        Math.hypot(x[3 * a] - x[3 * b], x[3 * a + 1] - x[3 * b + 1], x[3 * a + 2] - x[3 * b + 2]);
    if (n > 4) {
        let maxEdge = 2 * h;
        for (let i = 0; i < n; i++) maxEdge = Math.max(maxEdge, dist(i, (i + 1) % n));
        const grid = makeGrid(x, n, maxEdge);
        // The CURRENT curve as a linked ring: drops are applied one at a time, and every test
        // is made against the edges as they are at that moment.
        const next = new Int32Array(n);
        const prev = new Int32Array(n);
        const keep = new Uint8Array(n).fill(1);
        for (let i = 0; i < n; i++) {
            next[i] = (i + 1) % n;
            prev[i] = (i + n - 1) % n;
        }
        let kept = n;
        for (let i = 0; i < n && kept > 4; i++) {
            const a = prev[i];
            const b = next[i];
            const d1 = dist(a, i);
            const d2 = dist(i, b);
            if (!((d1 < h || d2 < h) && dist(a, b) <= 2 * h)) continue;
            // An edge meeting the triangle has an endpoint within (longest current edge) of a
            // point of the triangle, which is within max(d1, d2) of node i.
            const reach = Math.max(d1, d2) + maxEdge;
            let blocked = false;
            grid.forNear(i, reach, (j) => {
                if (blocked || !keep[j] || j === a || j === b) return;
                // Both current edges at j, skipping those that share a corner of the triangle.
                const jn = next[j];
                const jp = prev[j];
                if (jn !== a && jn !== b && jn !== i && segmentMeetsTriangle(x, j, jn, a, i, b))
                    blocked = true;
                else if (
                    jp !== a &&
                    jp !== b &&
                    jp !== i &&
                    segmentMeetsTriangle(x, jp, j, a, i, b)
                )
                    blocked = true;
            });
            if (blocked) continue;
            keep[i] = 0;
            next[a] = b;
            prev[b] = a;
            kept--;
        }
        if (kept < n) {
            const y = new Float64Array(3 * kept);
            let k = 0;
            for (let i = 0; i < n; i++) {
                if (!keep[i]) continue;
                y[3 * k] = x[3 * i];
                y[3 * k + 1] = x[3 * i + 1];
                y[3 * k + 2] = x[3 * i + 2];
                k++;
            }
            x = y;
            n = kept;
        }
    }
    // Split pass: each long edge into equal pieces no longer than 2h.
    const out: number[] = [];
    for (let i = 0; i < n; i++) {
        const j = (i + 1) % n;
        const len = Math.hypot(
            x[3 * j] - x[3 * i],
            x[3 * j + 1] - x[3 * i + 1],
            x[3 * j + 2] - x[3 * i + 2],
        );
        const pieces = Math.max(1, Math.ceil(len / (2 * h)));
        for (let s = 0; s < pieces; s++) {
            const u = s / pieces;
            out.push(
                x[3 * i] + (x[3 * j] - x[3 * i]) * u,
                x[3 * i + 1] + (x[3 * j + 1] - x[3 * i + 1]) * u,
                x[3 * i + 2] + (x[3 * j + 2] - x[3 * i + 2]) * u,
            );
        }
    }
    return { n: out.length / 3, x: Float64Array.from(out) };
}

/** A list of segments over shared nodes: segment `k` joins nodes `a[k]` and `b[k]`. */
export interface SegmentList {
    a: Int32Array;
    b: Int32Array;
}

/** The segments of a closed curve of `n` nodes: `i → i+1 (mod n)`. */
export function ringSegments(n: number): SegmentList {
    const a = new Int32Array(n);
    const b = new Int32Array(n);
    for (let i = 0; i < n; i++) {
        a[i] = i;
        b[i] = (i + 1) % n;
    }
    return { a, b };
}

/**
 * The largest step `s ≤ sTry` for which the straight-line motion `x → x − s·d` provably passes
 * no segment through another (segments sharing a node are not tested against each other).
 *
 * A CONSERVATIVE bound, not an exact collision test: for two segments at distance D whose
 * endpoints move, per unit step, with relative velocities of at most ρ, every pair of points
 * on them stays at least `D − sρ` apart (a point's motion is a convex combination of its
 * segment's endpoints'), so any `s < D/ρ` is free of contact. The result is `safety · min D/ρ`
 * over the pairs that could come into contact within `sTry` — those closer than twice the
 * largest displacement, found with a grid. It never returns an unsafe step; it may return a
 * smaller one than an exact test would.
 * Why not the paper's swept-surface test (or the exact quadratic of `planarCollision.ts`):
 * one pass, no root finding, works in 2-D and 3-D alike, and it yields the step directly
 * instead of by halving. Whether its conservatism costs steps is measured by the benchmark.
 * @see docs/plans/2026-11-24-medial-sphere-method.md
 */
export function maxSafeStep(
    x: Float64Array,
    nNodes: number,
    segs: SegmentList,
    d: Float64Array,
    sTry: number,
    safety = 0.9,
): number {
    const m = segs.a.length;
    let maxSpeed = 0;
    for (let i = 0; i < nNodes; i++)
        maxSpeed = Math.max(maxSpeed, Math.hypot(d[3 * i], d[3 * i + 1], d[3 * i + 2]));
    if (maxSpeed === 0 || sTry <= 0) return sTry;
    let maxEdge = 0;
    for (let k = 0; k < m; k++) {
        const a = segs.a[k];
        const b = segs.b[k];
        maxEdge = Math.max(
            maxEdge,
            Math.hypot(
                x[3 * b] - x[3 * a],
                x[3 * b + 1] - x[3 * a + 1],
                x[3 * b + 2] - x[3 * a + 2],
            ),
        );
    }
    const speed = new Float64Array(nNodes);
    for (let i = 0; i < nNodes; i++) speed[i] = Math.hypot(d[3 * i], d[3 * i + 1], d[3 * i + 2]);
    // Segments by first endpoint (CSR).
    const start = new Int32Array(nNodes + 1);
    for (let k = 0; k < m; k++) start[segs.a[k] + 1]++;
    for (let i = 0; i < nNodes; i++) start[i + 1] += start[i];
    const fill = Int32Array.from(start.subarray(0, nNodes));
    const byFirst = new Int32Array(m);
    for (let k = 0; k < m; k++) byFirst[fill[segs.a[k]]++] = k;

    /** `min D/ρ` over the pairs closer than `within` that would touch within step `s`. */
    const pass = (s: number, within: number): number => {
        // Two segments at distance D have their FIRST endpoints within D + both lengths.
        const reach = within + 2 * maxEdge;
        const grid = makeGrid(x, nNodes, reach);
        let best = Infinity;
        for (let e = 0; e < m; e++) {
            const a1 = segs.a[e];
            const b1 = segs.b[e];
            const ve = Math.max(speed[a1], speed[b1]);
            grid.forNear(a1, reach, (j) => {
                for (let q = start[j]; q < start[j + 1]; q++) {
                    const f = byFirst[q];
                    if (f <= e) continue;
                    const a2 = segs.a[f];
                    const b2 = segs.b[f];
                    if (a1 === a2 || a1 === b2 || b1 === a2 || b1 === b2) continue;
                    // Cheap rejection: the two cannot close a gap of at least
                    // |a1 − a2| − 2·maxEdge at a relative speed of at most ve + vf.
                    const gap =
                        Math.hypot(
                            x[3 * a1] - x[3 * a2],
                            x[3 * a1 + 1] - x[3 * a2 + 1],
                            x[3 * a1 + 2] - x[3 * a2 + 2],
                        ) -
                        2 * maxEdge;
                    if (gap >= s * (ve + Math.max(speed[a2], speed[b2]))) continue;
                    const D = segmentDistance(x, a1, b1, a2, b2);
                    if (D >= within) continue;
                    let rho = 0;
                    for (let u = 0; u < 2; u++) {
                        const p = u === 0 ? a1 : b1;
                        for (let v = 0; v < 2; v++) {
                            const w = v === 0 ? a2 : b2;
                            rho = Math.max(
                                rho,
                                Math.hypot(
                                    d[3 * p] - d[3 * w],
                                    d[3 * p + 1] - d[3 * w + 1],
                                    d[3 * p + 2] - d[3 * w + 2],
                                ),
                            );
                        }
                    }
                    if (rho * s >= D) best = Math.min(best, D / rho);
                }
            });
        }
        return best;
    };
    // Widening search. A pass that looks only at pairs closer than `within` bounds the step by
    // the near pairs; the result is final once no farther pair could touch within it, i.e.
    // once twice the largest displacement fits inside `within`. In a tangle the near pairs are
    // the binding ones, so the first, cheap pass usually settles it; searching the full
    // contact range of the trial step at once made every step an all-pairs scan.
    let s = sTry;
    let within = Math.min(2 * s * maxSpeed, 4 * maxEdge);
    for (;;) {
        const best = pass(s, within);
        if (best !== Infinity) s = Math.min(s, safety * best);
        const contact = 2 * s * maxSpeed;
        if (contact <= within) return s;
        within = Math.min(4 * within, contact);
    }
}

/**
 * Per-node limits that make ANY motion within them crossing-free: node `i` may move at most
 * `cap[i]`, where `cap[i] = frac · min(within, D)` over every pair of segments (sharing no
 * node) that one of `i`'s segments belongs to, D being the pair's distance.
 * Why it is safe: for a pair at distance D all four endpoints are limited to `frac·D` (or to
 * `frac·within ≤ frac·D` when D ≥ within), so their relative displacement is under
 * `2·frac·D < D` and — by the same convexity argument as `maxSafeStep` — no two points of the
 * two segments can meet.
 * Why this exists next to `maxSafeStep`: one global step is throttled by the single worst
 * pair. On a graph, a tight spot near one junction shrank the step of EVERY node geometrically
 * to zero (measured: 0.4 → 1e-14 in 50 steps on an 8×8 grid). A per-node limit lets a tight
 * spot slow down only its own nodes.
 * @see docs/plans/2026-11-24-medial-sphere-method.md
 */
export function nodeMotionCaps(
    x: Float64Array,
    nNodes: number,
    segs: SegmentList,
    within: number,
    frac = 0.45,
): Float64Array {
    const m = segs.a.length;
    const cap = new Float64Array(nNodes).fill(frac * within);
    let maxEdge = 0;
    for (let k = 0; k < m; k++) {
        const a = segs.a[k];
        const b = segs.b[k];
        maxEdge = Math.max(
            maxEdge,
            Math.hypot(
                x[3 * b] - x[3 * a],
                x[3 * b + 1] - x[3 * a + 1],
                x[3 * b + 2] - x[3 * a + 2],
            ),
        );
    }
    const reach = within + 2 * maxEdge;
    const grid = makeGrid(x, nNodes, reach);
    const start = new Int32Array(nNodes + 1);
    for (let k = 0; k < m; k++) start[segs.a[k] + 1]++;
    for (let i = 0; i < nNodes; i++) start[i + 1] += start[i];
    const fill = Int32Array.from(start.subarray(0, nNodes));
    const byFirst = new Int32Array(m);
    for (let k = 0; k < m; k++) byFirst[fill[segs.a[k]]++] = k;
    for (let e = 0; e < m; e++) {
        const a1 = segs.a[e];
        const b1 = segs.b[e];
        grid.forNear(a1, reach, (j) => {
            for (let q = start[j]; q < start[j + 1]; q++) {
                const f = byFirst[q];
                if (f <= e) continue;
                const a2 = segs.a[f];
                const b2 = segs.b[f];
                if (a1 === a2 || a1 === b2 || b1 === a2 || b1 === b2) continue;
                // Cheap rejection before the exact distance.
                if (
                    Math.hypot(
                        x[3 * a1] - x[3 * a2],
                        x[3 * a1 + 1] - x[3 * a2 + 1],
                        x[3 * a1 + 2] - x[3 * a2 + 2],
                    ) -
                        2 * maxEdge >=
                    within
                )
                    continue;
                const D = segmentDistance(x, a1, b1, a2, b2);
                if (D >= within) continue;
                const c = frac * D;
                if (c < cap[a1]) cap[a1] = c;
                if (c < cap[b1]) cap[b1] = c;
                if (c < cap[a2]) cap[a2] = c;
                if (c < cap[b2]) cap[b2] = c;
            }
        });
    }
    return cap;
}

/** Is the motion `x → x − step·d` of a closed curve provably free of strand crossings? */
export function motionIsSafe(c: ClosedCurve, d: Float64Array, step: number): boolean {
    return maxSafeStep(c.x, c.n, ringSegments(c.n), d, step, 1) >= step;
}

/** Distance between segments (a,b) and (c,d) of a flat coordinate array. */
function segmentDistance(x: Float64Array, a: number, b: number, c: number, d: number): number {
    const ux = x[3 * b] - x[3 * a];
    const uy = x[3 * b + 1] - x[3 * a + 1];
    const uz = x[3 * b + 2] - x[3 * a + 2];
    const vx = x[3 * d] - x[3 * c];
    const vy = x[3 * d + 1] - x[3 * c + 1];
    const vz = x[3 * d + 2] - x[3 * c + 2];
    const wx = x[3 * a] - x[3 * c];
    const wy = x[3 * a + 1] - x[3 * c + 1];
    const wz = x[3 * a + 2] - x[3 * c + 2];
    const A = ux * ux + uy * uy + uz * uz;
    const B = ux * vx + uy * vy + uz * vz;
    const C = vx * vx + vy * vy + vz * vz;
    const D = ux * wx + uy * wy + uz * wz;
    const E = vx * wx + vy * wy + vz * wz;
    const det = A * C - B * B;
    let s = det > 1e-300 ? Math.max(0, Math.min(1, (B * E - C * D) / det)) : 0;
    let t = C > 0 ? Math.max(0, Math.min(1, (B * s + E) / C)) : 0;
    // Re-clamp s for the clamped t (the standard two-pass closest-point rule).
    s = A > 0 ? Math.max(0, Math.min(1, (B * t - D) / A)) : 0;
    t = C > 0 ? Math.max(0, Math.min(1, (B * s + E) / C)) : 0;
    return Math.hypot(wx + s * ux - t * vx, wy + s * uy - t * vy, wz + s * uz - t * vz);
}

/** What one step did. */
export interface MedialSphereStepResult {
    curve: ClosedCurve;
    /** Mean pointwise gradient norm per node times `r` (dimensionless) — the stopping measure. */
    gradientPerNode: number;
    /** Step length actually taken (0 only when two strands already touch). */
    step: number;
    /** Whether the collision bound cut the trial step short. */
    limited: boolean;
    /** Trial length to pass as `tryStep` to the next step. */
    nextTryStep: number;
    /** Conjugate-gradient iterations of the preconditioner solve (the three coordinates' max). */
    cgIterations: number;
    /** Sphere links added to the Laplacian. */
    links: number;
}

/**
 * One step of the flow: gradient, preconditioned direction, collision-limited step, re-sample.
 * @see docs/plans/2026-11-24-medial-sphere-method.md
 */
export function medialSphereStep(
    c: ClosedCurve,
    scales: MedialSphereScales,
    opts: MedialSphereOptions = {},
): MedialSphereStepResult {
    const { n, x } = c;
    const N = opts.nDirections ?? 16;
    const beta = opts.beta ?? 0.01;
    const lambda = opts.lambda ?? 4;
    // Step length 1, NOT the paper's 10–20. The preconditioner's Laplacian is the length
    // term's own Hessian, so P⁻¹H ≈ 1 on those modes and a plain step is stable only below 2.
    // Measured on dataset knots: at 10 the step is unstable and only the collision bound
    // holds it (cut short in 91–95% of steps; the gradient per node chatters at 0.1–0.4 and
    // the paper's stopping rule never fires); at 1 the bound cuts 15% of steps, the gradient
    // settles under 1e-3 and the stopping rule fires, in fewer steps (about 150 against 210–230).
    // (The reference code's energy carries a factor 2 ours does not, which does not explain
    // the paper's larger value; whatever does is not in the paper's text.)
    const tau = opts.tau ?? 1;
    const { rMax, alpha, r, h } = scales;
    const spheres = medialSpheres(c, rMax, N);

    // Edge lengths, dual node weights, and the nodal (integrated) gradient
    //   G_i = (t_in − t_out) + α w_i (1/N) Σ_l (γ_i − m_il),   γ_i − m_il = −R_il x_l.
    const len = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        const j = (i + 1) % n;
        len[i] =
            Math.hypot(
                x[3 * j] - x[3 * i],
                x[3 * j + 1] - x[3 * i + 1],
                x[3 * j + 2] - x[3 * i + 2],
            ) || 1e-300;
    }
    const w = new Float64Array(n);
    const G = new Float64Array(3 * n);
    let gradSum = 0;
    for (let i = 0; i < n; i++) {
        const p = (i + n - 1) % n;
        const q = (i + 1) % n;
        w[i] = 0.5 * (len[p] + len[i]);
        for (let k = 0; k < 3; k++) {
            const tin = (x[3 * i + k] - x[3 * p + k]) / len[p];
            const tout = (x[3 * q + k] - x[3 * i + k]) / len[i];
            let pull = 0;
            for (let l = 0; l < N; l++)
                pull -= spheres.radius[i * N + l] * spheres.dir[(i * N + l) * 3 + k];
            G[3 * i + k] = tin - tout + (alpha * w[i] * pull) / N;
        }
        gradSum += Math.hypot(G[3 * i], G[3 * i + 1], G[3 * i + 2]) / w[i];
    }
    // Pointwise gradient (a curvature, 1/length) TIMES r: dimensionless, so the stopping rule
    // means the same at any scale. Dividing by r, as a literal reading of "below 5e-3·r"
    // suggests, is not scale-free: on a curve scaled to extent 2 (r ≈ 0.2) it reads 36× larger
    // than on the dataset's own units (r ≈ 1.2) and the rule never fired.
    const gradientPerNode = (gradSum / n) * r;

    // Sphere links (i, j), each once, with weight β / |γ_i − γ_j|.
    const linkA: number[] = [];
    const linkB: number[] = [];
    const linkW: number[] = [];
    const seen = new Set<number>();
    for (let i = 0; i < n; i++)
        for (let l = 0; l < N; l++) {
            const j = spheres.hit[i * N + l];
            if (j < 0) continue;
            const a = Math.min(i, j);
            const b = Math.max(i, j);
            // Curve neighbours are already linked by the Laplacian itself.
            if (b - a === 1 || b - a === n - 1) continue;
            const id = a * n + b;
            if (seen.has(id)) continue;
            seen.add(id);
            const dist = Math.hypot(
                x[3 * a] - x[3 * b],
                x[3 * a + 1] - x[3 * b + 1],
                x[3 * a + 2] - x[3 * b + 2],
            );
            linkA.push(a);
            linkB.push(b);
            linkW.push(beta / Math.max(dist, 1e-300));
        }

    // P = L* + λM, one copy per coordinate. Tridiagonal part: diag_i = 1/len[i−1] + 1/len[i]
    // + λ w_i, off-diagonal −1/len[i] between i and i+1 (cyclic).
    const diag = new Float64Array(n);
    const off = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        const p = (i + n - 1) % n;
        // Mass weight λ/r²: the Laplacian entries are 1/length and w is a length, so a bare
        // λ·w is not scale-free — the paper's λ = 1 means something else on a curve six times
        // smaller. Measured: the same trefoil stopped on the gradient rule after 107 steps in
        // the dataset's units (r ≈ 1.2) and never on it when scaled to extent 2 (r ≈ 0.2),
        // where the mass term was 36× weaker. Dividing by r² makes the flow the same flow at
        // any size. The default 4 is the paper's λ = 1 at r ≈ 2, the top of the dataset's range
        // (r = 1.2–1.9): at 2 the knot with the largest r was under-damped — two strands were
        // driven into contact and the flow stuck there with a zero safe step.
        diag[i] = 1 / len[p] + 1 / len[i] + (lambda / (r * r)) * w[i];
        off[i] = -1 / len[i];
    }
    const applyP = (v: Float64Array, out: Float64Array): void => {
        for (let i = 0; i < n; i++) {
            const p = (i + n - 1) % n;
            const q = (i + 1) % n;
            out[i] = diag[i] * v[i] + off[p] * v[p] + off[i] * v[q];
        }
        for (let k = 0; k < linkA.length; k++) {
            const a = linkA[k];
            const b = linkB[k];
            const t = linkW[k] * (v[a] - v[b]);
            out[a] += t;
            out[b] -= t;
        }
    };
    const tri = factorCyclicTridiagonal(diag, off);
    const d = new Float64Array(3 * n);
    let cgIterations = 0;
    const rhs = new Float64Array(n);
    const sol = new Float64Array(n);
    for (let k = 0; k < 3; k++) {
        for (let i = 0; i < n; i++) rhs[i] = G[3 * i + k];
        cgIterations = Math.max(
            cgIterations,
            conjugateGradients(applyP, tri, rhs, sol, linkA.length > 0),
        );
        for (let i = 0; i < n; i++) d[3 * i + k] = sol[i];
    }

    // The largest provably collision-free step up to the trial length. The trial starts at τ
    // and afterwards at 2× the last accepted step: the search for near pairs reaches as far
    // as the trial would move the fastest node, so trying τ = 10 every time when the curve
    // only ever accepts 1 makes every step an all-pairs scan.
    const tryStep = Math.min(tau, opts.tryStep ?? tau);
    const step = maxSafeStep(x, n, ringSegments(n), d, tryStep);
    const y = new Float64Array(3 * n);
    for (let i = 0; i < 3 * n; i++) y[i] = x[i] - step * d[i];
    return {
        curve: resampleClosedCurve({ n, x: y }, h),
        gradientPerNode,
        step,
        limited: step < tryStep,
        nextTryStep: Math.min(tau, step > 0 ? 2 * step : tau),
        cgIterations,
        links: linkA.length,
    };
}

/** A factored cyclic tridiagonal system (symmetric): `solve(b, out)`. */
interface CyclicTridiagonal {
    solve(b: Float64Array, out: Float64Array): void;
}

/**
 * Factor the symmetric cyclic tridiagonal matrix with diagonal `diag` and couplings
 * `off[i]` between `i` and `i+1 (mod n)`: Thomas elimination on the open chain plus one
 * Sherman–Morrison correction for the wrap-around coupling.
 */
function factorCyclicTridiagonal(diag: Float64Array, off: Float64Array): CyclicTridiagonal {
    const n = diag.length;
    const wrap = off[n - 1];
    // Sherman–Morrison: A = T + u uᵀ·(wrap/γ) form with u = (γ, 0, …, 0, wrap).
    const gamma = -diag[0];
    const dd = Float64Array.from(diag);
    dd[0] -= gamma;
    dd[n - 1] -= (wrap * wrap) / gamma;
    const cp = new Float64Array(n);
    const denom = new Float64Array(n);
    denom[0] = dd[0];
    for (let i = 1; i < n; i++) {
        cp[i - 1] = off[i - 1] / denom[i - 1];
        denom[i] = dd[i] - off[i - 1] * cp[i - 1];
    }
    const thomas = (b: Float64Array, out: Float64Array): void => {
        out[0] = b[0] / denom[0];
        for (let i = 1; i < n; i++) out[i] = (b[i] - off[i - 1] * out[i - 1]) / denom[i];
        for (let i = n - 2; i >= 0; i--) out[i] -= cp[i] * out[i + 1];
    };
    const u = new Float64Array(n);
    u[0] = gamma;
    u[n - 1] = wrap;
    const z = new Float64Array(n);
    thomas(u, z);
    const scale = 1 + z[0] + (wrap * z[n - 1]) / gamma;
    return {
        solve(b, out) {
            thomas(b, out);
            const f = (out[0] + (wrap * out[n - 1]) / gamma) / scale;
            for (let i = 0; i < n; i++) out[i] -= f * z[i];
        },
    };
}

/**
 * Conjugate gradients for `P x = b`, preconditioned by the exact solve of P's tridiagonal
 * part. With no sphere links the preconditioner IS the inverse and one solve is returned.
 * Stops at a relative residual of 1e-8 or 60 iterations; returns the iteration count.
 */
function conjugateGradients(
    applyP: (v: Float64Array, out: Float64Array) => void,
    tri: CyclicTridiagonal,
    b: Float64Array,
    x: Float64Array,
    hasLinks: boolean,
): number {
    const n = b.length;
    tri.solve(b, x);
    if (!hasLinks) return 0;
    const res = new Float64Array(n);
    const z = new Float64Array(n);
    const p = new Float64Array(n);
    const Ap = new Float64Array(n);
    applyP(x, Ap);
    let bNorm = 0;
    for (let i = 0; i < n; i++) {
        res[i] = b[i] - Ap[i];
        bNorm += b[i] * b[i];
    }
    if (bNorm === 0) return 0;
    tri.solve(res, z);
    let rz = 0;
    for (let i = 0; i < n; i++) {
        p[i] = z[i];
        rz += res[i] * z[i];
    }
    for (let it = 1; it <= 60; it++) {
        applyP(p, Ap);
        let pAp = 0;
        for (let i = 0; i < n; i++) pAp += p[i] * Ap[i];
        const a = rz / pAp;
        let rr = 0;
        for (let i = 0; i < n; i++) {
            x[i] += a * p[i];
            res[i] -= a * Ap[i];
            rr += res[i] * res[i];
        }
        if (rr <= 1e-16 * bNorm) return it;
        tri.solve(res, z);
        let rzNew = 0;
        for (let i = 0; i < n; i++) rzNew += res[i] * z[i];
        const bta = rzNew / rz;
        rz = rzNew;
        for (let i = 0; i < n; i++) p[i] = z[i] + bta * p[i];
    }
    return 60;
}
