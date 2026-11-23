/**
 * Conventional force-directed graph layout (node repulsion + edge springs), as an OPTIONAL
 * pre-pass before the tangent-point flow.
 *
 * Why it exists (user request, 2026-11-23): the tangent-point flow never lets two edges pass
 * through each other, so from a badly scrambled graph it can only relax INSIDE the tangle it was
 * given. A node-repulsion layout sees only nodes, so it pulls a scrambled graph apart globally in
 * milliseconds — and by the same token it DOES pass edges through each other. That is fine for
 * graph drawing (a PKM graph view) and wrong for knots: never run this on a curve whose knot type
 * must be kept. The paper says the same of such layouts ("methods based on nodal proximity may
 * cause edges to cross") and starts its own graph examples from a 2-D drawing.
 * @see local_files/repulsive_orig_paper/SelfAvoiding.tex (line 803, "Graph Drawing")
 *
 * The forces, their constants and the cooling schedule are d3-force's, so a layout here behaves
 * like the one people know from d3 / Obsidian's graph view. d3's constants are in pixels (link
 * distance 30, charge −30, minimum distance 1); here they are expressed in units of the graph's
 * own mean edge length, which makes the result independent of the input's scale.
 * @see https://d3js.org/d3-force (simulation, link, many-body and center forces)
 *
 * NOT here: d3's quadtree approximation of the repulsion. The sum is the exact O(n²) one —
 * measured 1 ms a tick at n = 300, 4 ms at 1 000, 36 ms at 3 000, and a layout is 300 ticks —
 * because every graph the Sobolev flow can currently step is far smaller than where that
 * matters (general graphs take the dense route, @issue utof/repulsive-test2#93).
 */
import type { Edge, Vec3 } from './testConfigs';

/** Options of {@link createForceLayout}. Every default is d3-force's, rescaled (see file doc). */
export interface ForceLayoutOptions {
    /** Rest length of every edge spring. Default: the mean edge length of the input. */
    linkDistance?: number;
    /** Repulsion strength, negative = repel. Default `-linkDistance² / 30` (d3: −30 at 30 px). */
    chargeStrength?: number;
    /** Fraction of velocity lost per tick. Default 0.4. */
    velocityDecay?: number;
    /** The layout is settled when alpha falls below this. Default 0.001. */
    alphaMin?: number;
    /** Cooling rate. Default `1 − alphaMin^(1/300)`, i.e. settled after 300 ticks. */
    alphaDecay?: number;
    /** Vertices that must not move (the app's pins). */
    fixed?: ReadonlySet<number>;
}

/**
 * A running layout. `x` and `v` are flat `[x0,y0,z0, x1,…]`; everything is mutated in place by
 * {@link forceLayoutTick}. Why flat typed arrays: the repulsion loop is O(n²) per tick.
 */
export interface ForceLayout {
    readonly n: number;
    readonly edges: readonly Edge[];
    readonly x: Float64Array;
    readonly v: Float64Array;
    alpha: number;
    ticks: number;
    readonly linkDistance: number;
    readonly chargeStrength: number;
    readonly distanceMin2: number;
    readonly velocityDecay: number;
    readonly alphaMin: number;
    readonly alphaDecay: number;
    /** Per edge: spring strength `1 / min(deg a, deg b)` and bias `deg a / (deg a + deg b)`. */
    readonly linkStrength: Float64Array;
    readonly linkBias: Float64Array;
    readonly fixed: Uint8Array;
    /** Vertex mean of the input; the centering force holds the layout there. */
    readonly centre: Vec3;
    /** True when the input lies in the plane z = const: the layout then stays in that plane. */
    readonly planar: boolean;
    /** State of the deterministic generator used to separate coincident vertices. */
    seed: number;
}

/** Sum of the edge lengths of a flat position array. */
function totalLengthFlat(x: Float64Array, edges: readonly Edge[]): number {
    let s = 0;
    for (const [a, b] of edges)
        s += Math.hypot(
            x[3 * a] - x[3 * b],
            x[3 * a + 1] - x[3 * b + 1],
            x[3 * a + 2] - x[3 * b + 2],
        );
    return s;
}

/**
 * Start a layout from the given positions (they are copied; the input is not mutated).
 * Why from the given positions and not from d3's spiral: the pre-pass should move a graph that
 * is already half laid out as little as it has to.
 */
export function createForceLayout(
    vertices: readonly Vec3[],
    edges: readonly Edge[],
    opts: ForceLayoutOptions = {},
): ForceLayout {
    const n = vertices.length;
    const x = new Float64Array(3 * n);
    const centre: Vec3 = [0, 0, 0];
    let planar = true;
    for (let i = 0; i < n; i++) {
        for (let k = 0; k < 3; k++) {
            x[3 * i + k] = vertices[i][k];
            centre[k] += vertices[i][k] / n;
        }
        if (Math.abs(vertices[i][2] - vertices[0][2]) > 1e-12) planar = false;
    }
    const linkDistance =
        opts.linkDistance ?? (edges.length > 0 ? totalLengthFlat(x, edges) / edges.length : 1);
    const degree = new Float64Array(n);
    for (const [a, b] of edges) {
        degree[a]++;
        degree[b]++;
    }
    const linkStrength = new Float64Array(edges.length);
    const linkBias = new Float64Array(edges.length);
    edges.forEach(([a, b], e) => {
        linkStrength[e] = 1 / Math.min(degree[a], degree[b]);
        linkBias[e] = degree[a] / (degree[a] + degree[b]);
    });
    const fixed = new Uint8Array(n);
    if (opts.fixed) for (const i of opts.fixed) if (i >= 0 && i < n) fixed[i] = 1;
    const alphaMin = opts.alphaMin ?? 0.001;
    return {
        n,
        edges,
        x,
        v: new Float64Array(3 * n),
        alpha: 1,
        ticks: 0,
        linkDistance,
        chargeStrength: opts.chargeStrength ?? -(linkDistance * linkDistance) / 30,
        // d3 clamps the repulsion below 1 px at a 30 px link distance.
        distanceMin2: (linkDistance / 30) ** 2,
        velocityDecay: opts.velocityDecay ?? 0.4,
        alphaMin,
        alphaDecay: opts.alphaDecay ?? 1 - alphaMin ** (1 / 300),
        linkStrength,
        linkBias,
        fixed,
        centre,
        planar,
        seed: 1,
    };
}

/**
 * A tiny deterministic offset for two coincident points (d3 calls it "jiggle"), so that they
 * separate in some direction instead of dividing by zero. Why a seeded generator and not
 * `Math.random`: a layout must be reproducible in tests and benchmarks.
 */
function jiggle(s: ForceLayout): number {
    s.seed = (1664525 * s.seed + 1013904223) % 4294967296;
    return (s.seed / 4294967296 - 0.5) * 1e-6 * s.linkDistance;
}

/**
 * One tick, in d3's order: cool, edge springs, repulsion, centering, then integrate with
 * friction. Mutates `s`.
 */
export function forceLayoutTick(s: ForceLayout): void {
    const { n, x, v, edges } = s;
    s.alpha += (0 - s.alpha) * s.alphaDecay;
    const alpha = s.alpha;

    // Edge springs, evaluated at the PREDICTED positions x + v (d3's link force). The more
    // connected endpoint moves less (`linkBias`).
    for (let e = 0; e < edges.length; e++) {
        const a = 3 * edges[e][0];
        const b = 3 * edges[e][1];
        let dx = x[b] + v[b] - x[a] - v[a];
        let dy = x[b + 1] + v[b + 1] - x[a + 1] - v[a + 1];
        let dz = x[b + 2] + v[b + 2] - x[a + 2] - v[a + 2];
        if (dx === 0 && dy === 0 && dz === 0) {
            dx = jiggle(s);
            dy = jiggle(s);
            if (!s.planar) dz = jiggle(s);
        }
        const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const f = ((len - s.linkDistance) / len) * alpha * s.linkStrength[e];
        const bias = s.linkBias[e];
        v[b] -= dx * f * bias;
        v[b + 1] -= dy * f * bias;
        v[b + 2] -= dz * f * bias;
        v[a] += dx * f * (1 - bias);
        v[a + 1] += dy * f * (1 - bias);
        v[a + 2] += dz * f * (1 - bias);
    }

    // Repulsion between every pair of vertices: velocity change `strength·alpha / distance`,
    // softened below `distanceMin` exactly as d3 does (l ← sqrt(distanceMin²·l)).
    const k = s.chargeStrength * alpha;
    for (let i = 0; i < n; i++) {
        const xi = x[3 * i];
        const yi = x[3 * i + 1];
        const zi = x[3 * i + 2];
        let ax = 0;
        let ay = 0;
        let az = 0;
        for (let j = i + 1; j < n; j++) {
            let dx = x[3 * j] - xi;
            let dy = x[3 * j + 1] - yi;
            let dz = x[3 * j + 2] - zi;
            let l = dx * dx + dy * dy + dz * dz;
            if (l === 0) {
                dx = jiggle(s);
                dy = jiggle(s);
                if (!s.planar) dz = jiggle(s);
                l = dx * dx + dy * dy + dz * dz;
            }
            if (l < s.distanceMin2) l = Math.sqrt(s.distanceMin2 * l);
            const w = k / l;
            ax += dx * w;
            ay += dy * w;
            az += dz * w;
            v[3 * j] -= dx * w;
            v[3 * j + 1] -= dy * w;
            v[3 * j + 2] -= dz * w;
        }
        v[3 * i] += ax;
        v[3 * i + 1] += ay;
        v[3 * i + 2] += az;
    }

    // Integrate with friction; pinned vertices stay where they are.
    const keep = 1 - s.velocityDecay;
    const mean = [0, 0, 0];
    let free = 0;
    for (let i = 0; i < n; i++) {
        if (s.fixed[i]) {
            v[3 * i] = v[3 * i + 1] = v[3 * i + 2] = 0;
            continue;
        }
        free++;
        for (let c = 0; c < 3; c++) {
            v[3 * i + c] *= keep;
            x[3 * i + c] += v[3 * i + c];
            mean[c] += x[3 * i + c];
        }
    }
    // Centering (d3's center force): shift, do not pull. Skipped when anything is pinned — the
    // pins already hold the layout in place, and shifting would move the free part off them.
    if (free === n && n > 0)
        for (let i = 0; i < n; i++)
            for (let c = 0; c < 3; c++) x[3 * i + c] += s.centre[c] - mean[c] / n;
    s.ticks++;
}

/** Has the layout cooled below `alphaMin`? d3's own stopping rule. */
export function forceLayoutSettled(s: ForceLayout): boolean {
    return s.alpha < s.alphaMin;
}

/** Copy the current positions into existing `Vec3` tuples (the app's live buffer). */
export function writeForceLayout(s: ForceLayout, into: Vec3[]): void {
    for (let i = 0; i < s.n; i++) {
        into[i][0] = s.x[3 * i];
        into[i][1] = s.x[3 * i + 1];
        into[i][2] = s.x[3 * i + 2];
    }
}

/**
 * Scale the layout about its centre so that the total edge length equals `targetLength`.
 * Why: the tangent-point energy is not scale-invariant (E ∝ 1/size at α = 3, β = 6), so a
 * pre-pass that also changed the SIZE of the graph would change the energy for a reason that
 * has nothing to do with untangling, and would make "with / without pre-pass" incomparable.
 * No-op when any vertex is pinned (scaling would move the free part relative to the pins).
 */
export function rescaleForceLayout(s: ForceLayout, targetLength: number): void {
    if (s.fixed.some((f) => f === 1)) return;
    const len = totalLengthFlat(s.x, s.edges);
    if (!(len > 0) || !(targetLength > 0)) return;
    const r = targetLength / len;
    for (let i = 0; i < s.n; i++)
        for (let c = 0; c < 3; c++)
            s.x[3 * i + c] = s.centre[c] + (s.x[3 * i + c] - s.centre[c]) * r;
}

/**
 * Run a layout until it settles and return the new positions, rescaled to the input's total
 * edge length. The one-call form used by benchmarks and tests; the app ticks it frame by frame
 * instead so the motion is visible.
 */
export function runForceLayout(
    vertices: readonly Vec3[],
    edges: readonly Edge[],
    opts: ForceLayoutOptions = {},
): { vertices: Vec3[]; ticks: number } {
    const s = createForceLayout(vertices, edges, opts);
    const length0 = totalLengthFlat(s.x, edges);
    while (!forceLayoutSettled(s)) forceLayoutTick(s);
    rescaleForceLayout(s, length0);
    const out: Vec3[] = vertices.map(() => [0, 0, 0]);
    writeForceLayout(s, out);
    return { vertices: out, ticks: s.ticks };
}
