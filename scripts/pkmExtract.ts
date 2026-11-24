// scripts/pkmExtract.ts — bun scripts/pkmExtract.ts <kind> <inputPath> <outputJson> [name]
// Turns a real note collection into a plain graph for the layout benchmarks:
//   vault  <dir>   every .md file is a node; [[wikilinks]] and relative [text](file.md) links
//                  are edges (an Obsidian / digital-garden vault). Unresolved links are dropped.
//   mm     <file>  a FreeMind / Freeplane mind map: the <node> tree, plus <arrowlink> cross links.
// Output: {name, kind, source, nodes, edges: [[a, b], …], parent?: number[]} — structure only,
// no note text and no titles (the collections are other people's writing; the benchmarks need
// only who links to whom). Only the largest connected component is kept, renumbered from 0.
// Why a script and not committed fixtures: none of the collections used so far carries a
// licence that would let the repo redistribute them; the benchmark reads whatever graphs it
// finds in local_files/pkm-graphs/graphs/ and says so when there are none.
// @see bench/app/pkmGraphs.ts, local_files/pkm-graphs/README.md (which collections, from where)
import { readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { basename, dirname, join, normalize, relative } from 'node:path';

const [kind, input, output, nameArg] = process.argv.slice(2);
if (!kind || !input || !output) {
    console.error('usage: bun scripts/pkmExtract.ts vault|mm <inputPath> <outputJson> [name]');
    process.exit(2);
}

function walk(dir: string, out: string[]): string[] {
    for (const entry of readdirSync(dir)) {
        if (entry === '.git' || entry === 'node_modules') continue;
        const p = join(dir, entry);
        const st = statSync(p);
        if (st.isDirectory()) walk(p, out);
        else if (entry.toLowerCase().endsWith('.md')) out.push(p);
    }
    return out;
}

/** Notes and links of a vault. A link resolves by path first, then by file name (Obsidian). */
function vault(dir: string): { n: number; edges: [number, number][] } {
    const files = walk(dir, []);
    const byPath = new Map<string, number>();
    const byName = new Map<string, number>();
    files.forEach((f, i) => {
        const rel = relative(dir, f).replace(/\.md$/i, '').toLowerCase();
        byPath.set(rel, i);
        const b = basename(rel);
        if (!byName.has(b)) byName.set(b, i);
    });
    const edges: [number, number][] = [];
    const wiki = /\[\[([^\]|#^]+)(?:[#^][^\]|]*)?(?:\|[^\]]*)?\]\]/g;
    const md = /\]\(([^)\s#?]+\.md)(?:#[^)]*)?\)/gi;
    files.forEach((f, i) => {
        const text = readFileSync(f, 'utf8');
        const here = dirname(relative(dir, f));
        const add = (target: string): void => {
            const t = target.trim().replace(/\.md$/i, '').toLowerCase();
            if (!t) return;
            const j =
                byPath.get(normalize(join(here, t)).toLowerCase()) ??
                byPath.get(normalize(t)) ??
                byName.get(basename(t));
            if (j !== undefined && j !== i) edges.push([i, j]);
        };
        for (const m of text.matchAll(wiki)) add(m[1]);
        for (const m of text.matchAll(md)) {
            try {
                add(decodeURIComponent(m[1]));
            } catch {
                add(m[1]);
            }
        }
    });
    return { n: files.length, edges };
}

/** The node tree of a .mm file (tags scanned in order; no XML library needed for this). */
function mindMap(file: string): { n: number; edges: [number, number][]; parent: number[] } {
    const text = readFileSync(file, 'utf8');
    const tag = /<(\/?)(node|arrowlink)\b([^>]*?)(\/?)>/g;
    const parent: number[] = [];
    const ids = new Map<string, number>();
    const links: [number, string][] = [];
    const stack: number[] = [];
    for (const m of text.matchAll(tag)) {
        const [, closing, what, attrs, selfClosing] = m;
        if (what === 'node') {
            if (closing) {
                stack.pop();
                continue;
            }
            const i = parent.length;
            parent.push(stack.length > 0 ? stack[stack.length - 1] : -1);
            const id = /\bID="([^"]*)"/.exec(attrs);
            if (id) ids.set(id[1], i);
            if (!selfClosing) stack.push(i);
        } else if (!closing && stack.length > 0) {
            const to = /\bDESTINATION="([^"]*)"/.exec(attrs);
            if (to) links.push([stack[stack.length - 1], to[1]]);
        }
    }
    const edges: [number, number][] = [];
    parent.forEach((p, i) => {
        if (p >= 0) edges.push([p, i]);
    });
    for (const [from, id] of links) {
        const to = ids.get(id);
        if (to !== undefined && to !== from) edges.push([from, to]);
    }
    return { n: parent.length, edges, parent };
}

const raw = kind === 'mm' ? mindMap(input) : vault(input);
// Undirected, no duplicates, largest connected component, renumbered.
const seen = new Set<string>();
const und: [number, number][] = [];
for (const [a, b] of raw.edges) {
    const key = a < b ? `${a},${b}` : `${b},${a}`;
    if (!seen.has(key)) {
        seen.add(key);
        und.push(a < b ? [a, b] : [b, a]);
    }
}
const comp = new Int32Array(raw.n).fill(-1);
const adj: number[][] = Array.from({ length: raw.n }, () => []);
for (const [a, b] of und) {
    adj[a].push(b);
    adj[b].push(a);
}
let best = -1;
let bestSize = 0;
for (let s = 0, c = 0; s < raw.n; s++) {
    if (comp[s] >= 0) continue;
    let size = 0;
    const queue = [s];
    comp[s] = c;
    while (queue.length > 0) {
        const v = queue.pop() as number;
        size++;
        for (const w of adj[v])
            if (comp[w] < 0) {
                comp[w] = c;
                queue.push(w);
            }
    }
    if (size > bestSize) {
        bestSize = size;
        best = c;
    }
    c++;
}
const renumber = new Int32Array(raw.n).fill(-1);
let n = 0;
for (let i = 0; i < raw.n; i++) if (comp[i] === best) renumber[i] = n++;
const edges = und.filter(([a]) => comp[a] === best).map(([a, b]) => [renumber[a], renumber[b]]);
const out: Record<string, unknown> = {
    name: nameArg ?? basename(input).replace(/\.(mm|md)$/i, ''),
    kind,
    nodes: n,
    edgeCount: edges.length,
    notesInSource: raw.n,
    edges,
};
if ('parent' in raw && n === raw.n) out.parent = raw.parent; // a whole tree: keep its shape
writeFileSync(output, JSON.stringify(out));
console.log(`${out.name}: ${n} nodes, ${edges.length} edges (of ${raw.n} notes in the source)`);
