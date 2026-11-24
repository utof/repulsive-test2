// src/core/planarSkeleton.ts
// For a graph that cannot be drawn without crossings (a note vault: 180 notes, 883 links,
// 14,803 crossings in a force layout): which links form the drawing that is kept crossing-free,
// and which are left to be drawn on top of it.
//   1. score every link (`linkScores`): how many neighbours its two ends share;
//   2. a spanning tree that prefers well-scored links (`spanningTreeBy`);
//   3. the caller draws the tree (any tree drawing without crossings);
//   4. the other links are taken back in, best first, each only if drawn straight it comes
//      near nothing already kept (`readmitLinks`). What is left is the overlay.
// Why this and not a maximum planar subgraph: that needs a planarity test and a new embedding
// (so the drawing the user has is thrown away), the fast heuristics keep hardly more than a
// tree on graphs of this kind, and no dependable implementation exists for JavaScript. This
// one keeps the drawing, needs only segment tests, and by Euler's bound (3n − 6 links at
// most) no method can keep more than 60 % of such a vault's links anyway.
// @see _local/research/2026-11-24-11-untangle-first-what-to-cut.md (local), and
//      docs/plans/2026-11-24-medial-sphere-method.md ("What to cut")

/** How the spanning tree is chosen (kept side by side: which is best is measured, not known). */
export type TreeChoice = 'bfs' | 'bfs-best' | 'max-weight';

/**
 * A score per link: the number of neighbours its two ends have in common (its
 * "embeddedness": the triangles it closes).
 * Why: a link inside a tight group of notes is structure; a link between two notes that share
 * nothing else is a jump across the map, and those are the ones worth leaving to the overlay.
 * Self-links score −1 and are never kept; of parallel links only the first counts.
 */
export function linkScores(n: number, edges: readonly (readonly [number, number])[]): Float64Array {
    const nbr: Set<number>[] = Array.from({ length: n }, () => new Set<number>());
    for (const [a, b] of edges)
        if (a !== b) {
            nbr[a].add(b);
            nbr[b].add(a);
        }
    const score = new Float64Array(edges.length);
    const seen = new Set<number>();
    for (let e = 0; e < edges.length; e++) {
        const [a, b] = edges[e];
        const key = Math.min(a, b) * n + Math.max(a, b);
        if (a === b || seen.has(key)) {
            score[e] = -1;
            continue;
        }
        seen.add(key);
        const [small, large] = nbr[a].size <= nbr[b].size ? [nbr[a], nbr[b]] : [nbr[b], nbr[a]];
        let common = 0;
        for (const v of small) if (large.has(v)) common++;
        score[e] = common;
    }
    return score;
}

/**
 * A spanning tree of the part of the graph that hangs together with its most linked node
 * (the root). `parent[i]` = −1 for the root, −2 for a node not joined to it; `via[i]` = the
 * link that joins `i` to its parent (−1 for none).
 *  - 'bfs':        breadth-first, first link found (what the vault benchmark always did);
 *  - 'bfs-best':   breadth-first — every node as near the root as it can be — but among the
 *                  links to the level above, the best-scored one;
 *  - 'max-weight': the tree whose links score most in total (Kruskal), whatever its depth.
 */
export function spanningTreeBy(
    n: number,
    edges: readonly (readonly [number, number])[],
    score: Float64Array,
    choice: TreeChoice,
): { parent: number[]; via: number[]; root: number } {
    const adj: [number, number][][] = Array.from({ length: n }, () => []);
    for (let e = 0; e < edges.length; e++) {
        const [a, b] = edges[e];
        if (score[e] < 0) continue;
        adj[a].push([b, e]);
        adj[b].push([a, e]);
    }
    let root = 0;
    for (let i = 1; i < n; i++) if (adj[i].length > adj[root].length) root = i;
    const parent = new Array<number>(n).fill(-2);
    const via = new Array<number>(n).fill(-1);
    parent[root] = -1;
    if (choice === 'max-weight') {
        // Kruskal over the links, best first; then the tree is hung from the root.
        const order = Array.from(edges.keys())
            .filter((e) => score[e] >= 0)
            .sort((p, q) => score[q] - score[p] || p - q);
        const up = Array.from({ length: n }, (_, i) => i);
        const find = (i: number): number => {
            while (up[i] !== i) {
                up[i] = up[up[i]];
                i = up[i];
            }
            return i;
        };
        const tree: [number, number][][] = Array.from({ length: n }, () => []);
        for (const e of order) {
            const a = find(edges[e][0]);
            const b = find(edges[e][1]);
            if (a === b) continue;
            up[a] = b;
            tree[edges[e][0]].push([edges[e][1], e]);
            tree[edges[e][1]].push([edges[e][0], e]);
        }
        const queue = [root];
        for (let q = 0; q < queue.length; q++)
            for (const [w, e] of tree[queue[q]])
                if (parent[w] === -2) {
                    parent[w] = queue[q];
                    via[w] = e;
                    queue.push(w);
                }
        return { parent, via, root };
    }
    const depth = new Int32Array(n).fill(-1);
    depth[root] = 0;
    const queue = [root];
    for (let q = 0; q < queue.length; q++) {
        const u = queue[q];
        for (const [w, e] of adj[u]) {
            if (parent[w] === -2) {
                parent[w] = u;
                via[w] = e;
                depth[w] = depth[u] + 1;
                queue.push(w);
            } else if (
                choice === 'bfs-best' &&
                depth[w] === depth[u] + 1 &&
                score[e] > score[via[w]]
            ) {
                // Same level above, a better link: `w` has no children yet that depend on which.
                parent[w] = u;
                via[w] = e;
            }
        }
    }
    return { parent, via, root };
}

/**
 * Every node's children, ordered so that children that link to each other stand NEXT to each
 * other: per parent, the links between its children are taken best-scored first into chains
 * (a child joins at most two, no ring), and the chains are laid end to end.
 * Why: in a tree drawing, two children of one node can be joined without crossing anything
 * only if nothing stands between them — and a note that indexes others (2,900 children in a
 * real vault) has many children that link to each other. The order costs nothing; without
 * it those links are left to chance (measured: see the plan, "What to cut").
 * `parent` as from `spanningTreeBy` (−1 root, −2 not in the tree).
 */
export function orderSiblings(
    n: number,
    parent: readonly number[],
    edges: readonly (readonly [number, number])[],
    score: Float64Array,
): number[][] {
    const kids: number[][] = Array.from({ length: n }, () => []);
    for (let i = 0; i < n; i++) if (parent[i] >= 0) kids[parent[i]].push(i);
    const between: number[] = [];
    for (let e = 0; e < edges.length; e++) {
        const [a, b] = edges[e];
        if (score[e] >= 0 && a !== b && parent[a] >= 0 && parent[a] === parent[b]) between.push(e);
    }
    between.sort((p, q) => score[q] - score[p] || p - q);
    const next: number[][] = Array.from({ length: n }, () => []); // chain neighbours (≤ 2)
    const up = Array.from({ length: n }, (_, i) => i);
    const find = (i: number): number => {
        while (up[i] !== i) {
            up[i] = up[up[i]];
            i = up[i];
        }
        return i;
    };
    for (const e of between) {
        const [a, b] = edges[e];
        if (next[a].length >= 2 || next[b].length >= 2) continue;
        const ra = find(a);
        const rb = find(b);
        if (ra === rb) continue; // would close a ring
        up[ra] = rb;
        next[a].push(b);
        next[b].push(a);
    }
    const placed = new Uint8Array(n);
    return kids.map((list) => {
        const out: number[] = [];
        // Chains from their ends first (a child in no chain is a chain of one), in index order.
        for (const start of list) {
            if (placed[start] || next[start].length === 2) continue;
            let prev = -1;
            let at = start;
            while (at >= 0 && !placed[at]) {
                placed[at] = 1;
                out.push(at);
                const onward: number = next[at].find((v) => v !== prev && !placed[v]) ?? -1;
                prev = at;
                at = onward;
            }
        }
        return out;
    });
}

/** Distance from point p to segment a–b, in the plane. */
function pointToSegment(
    px: number,
    py: number,
    ax: number,
    ay: number,
    bx: number,
    by: number,
): number {
    const L2 = (bx - ax) ** 2 + (by - ay) ** 2;
    const t =
        L2 > 0 ? Math.max(0, Math.min(1, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / L2)) : 0;
    return Math.hypot(px - ax - t * (bx - ax), py - ay - t * (by - ay));
}

/** Distance between segments a–b and c–d in the plane (0 if they cross). */
function segmentToSegment(
    ax: number,
    ay: number,
    bx: number,
    by: number,
    cx: number,
    cy: number,
    dx: number,
    dy: number,
): number {
    const d1 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    const d2 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax);
    const d3 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx);
    const d4 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx);
    if (d1 * d2 < 0 && d3 * d4 < 0) return 0;
    return Math.min(
        pointToSegment(ax, ay, cx, cy, dx, dy),
        pointToSegment(bx, by, cx, cy, dx, dy),
        pointToSegment(cx, cy, ax, ay, bx, by),
        pointToSegment(dx, dy, ax, ay, bx, by),
    );
}

/**
 * Take links back into a drawing, best-scored first (among equals the shorter), each only if
 * its straight segment stays at least `clearance` away from every segment already kept that
 * it shares no end with — and, where it shares an end, does not run along the other one
 * (the far ends at least `clearance` from each other's segment).
 * `x`: xy per node (3 numbers per node, as everywhere); `kept[e]` = 1 on entry for the links
 * already drawn (the tree), 1 on return also for the ones taken back in. Links with a
 * negative score, or with an end that has no kept link at all (not in the drawing), are
 * left out. Returns the number taken in.
 * Why a clearance and not just "does not cross": a link that only touches a node it does
 * not end at, or lies on top of another link (both happen in a tidy tree, where a whole
 * level of nodes stands on one line), is not a crossing by the usual test but is no drawing
 * either — and the flow that keeps the drawing afterwards needs a gap to work with.
 * Cost: every segment is entered in a grid at points half a cell apart along it; a link is
 * tested against what the 3×3 cells around its own such points hold.
 */
export function readmitLinks(
    x: Float64Array,
    edges: readonly (readonly [number, number])[],
    kept: Uint8Array,
    score: Float64Array,
    clearance: number,
): number {
    const n = x.length / 3;
    const drawn = new Uint8Array(n);
    let total = 0;
    let count = 0;
    for (let e = 0; e < edges.length; e++)
        if (kept[e]) {
            const [a, b] = edges[e];
            drawn[a] = 1;
            drawn[b] = 1;
            total += Math.hypot(x[3 * a] - x[3 * b], x[3 * a + 1] - x[3 * b + 1]);
            count++;
        }
    if (count === 0) return 0;
    // Cell: the mean kept link, but never less than twice the clearance (the 3×3 search
    // reaches one cell: half a cell of slack for each of the two segments' sample spacing).
    const cell = Math.max(total / count, 2 * clearance, 1e-9);
    const grid = new Map<number, number[]>();
    const key = (cx: number, cy: number): number => (cx + 32768) * 65536 + (cy + 32768);
    const samples = (e: number, visit: (cx: number, cy: number) => void): void => {
        const [a, b] = edges[e];
        const ax = x[3 * a];
        const ay = x[3 * a + 1];
        const bx = x[3 * b];
        const by = x[3 * b + 1];
        const steps = Math.max(1, Math.ceil(Math.hypot(bx - ax, by - ay) / (0.5 * cell)));
        let lastX = Number.NaN;
        let lastY = Number.NaN;
        for (let j = 0; j <= steps; j++) {
            const cx = Math.floor((ax + (j / steps) * (bx - ax)) / cell);
            const cy = Math.floor((ay + (j / steps) * (by - ay)) / cell);
            if (cx === lastX && cy === lastY) continue;
            lastX = cx;
            lastY = cy;
            visit(cx, cy);
        }
    };
    const insert = (e: number): void =>
        samples(e, (cx, cy) => {
            const k = key(cx, cy);
            const list = grid.get(k);
            if (list) {
                if (list[list.length - 1] !== e) list.push(e);
            } else grid.set(k, [e]);
        });
    for (let e = 0; e < edges.length; e++) if (kept[e]) insert(e);
    const length = (e: number): number =>
        Math.hypot(
            x[3 * edges[e][0]] - x[3 * edges[e][1]],
            x[3 * edges[e][0] + 1] - x[3 * edges[e][1] + 1],
        );
    const order: number[] = [];
    for (let e = 0; e < edges.length; e++)
        if (!kept[e] && score[e] >= 0 && drawn[edges[e][0]] && drawn[edges[e][1]]) order.push(e);
    order.sort((p, q) => score[q] - score[p] || length(p) - length(q) || p - q);
    const stamp = new Int32Array(edges.length).fill(-1);
    let taken = 0;
    for (const e of order) {
        const [a, b] = edges[e];
        const ax = x[3 * a];
        const ay = x[3 * a + 1];
        const bx = x[3 * b];
        const by = x[3 * b + 1];
        let free = true;
        samples(e, (cx, cy) => {
            if (!free) return;
            for (let v = cy - 1; v <= cy + 1 && free; v++)
                for (let u = cx - 1; u <= cx + 1 && free; u++) {
                    const list = grid.get(key(u, v));
                    if (!list) continue;
                    for (const f of list) {
                        if (stamp[f] === e) continue;
                        stamp[f] = e;
                        const [c, d] = edges[f];
                        const cx2 = x[3 * c];
                        const cy2 = x[3 * c + 1];
                        const dx2 = x[3 * d];
                        const dy2 = x[3 * d + 1];
                        const sharedA = c === a || d === a;
                        const sharedB = c === b || d === b;
                        if (sharedA && sharedB) {
                            free = false; // the same two nodes: a parallel link
                        } else if (sharedA || sharedB) {
                            // One end in common: neither may run along the other.
                            const mineX = sharedA ? bx : ax;
                            const mineY = sharedA ? by : ay;
                            const shared = sharedA ? a : b;
                            const itsX = c === shared ? dx2 : cx2;
                            const itsY = c === shared ? dy2 : cy2;
                            if (
                                pointToSegment(mineX, mineY, cx2, cy2, dx2, dy2) < clearance ||
                                pointToSegment(itsX, itsY, ax, ay, bx, by) < clearance
                            )
                                free = false;
                        } else if (
                            segmentToSegment(ax, ay, bx, by, cx2, cy2, dx2, dy2) < clearance
                        ) {
                            free = false;
                        }
                        if (!free) break;
                    }
                }
        });
        if (!free) continue;
        kept[e] = 1;
        insert(e);
        taken++;
    }
    return taken;
}
