/**
 * Curve fixtures shared by the Phase 2H hierarchical tests (tasks 3-7, 11). TEST-ONLY; never
 * imported by `src/`.
 *
 * Transcribed from `.superpowers/sdd/2026-09-11-phase2h-queue/scratch/curves.ts` with the SAME
 * arithmetic (same constants, same op order), INCLUDING its bbox normalisation (`curves.ts:49-66`,
 * applied by `closedFrom`/`openFrom`): every `out-bct-*.json` / `out-accuracy.json` oracle was
 * measured on curves rescaled to bounding-box max extent 1, centred at the origin. That is a
 * CONDITION of the oracles, not cosmetics: the R^6 split score adds squared diameters of three
 * position and three unit-tangent coordinates in one number (`curves.ts:13-17`; the SCALE
 * PRECONDITION of `bvh.ts`), so a different spatial scale changes the chosen split axes, the
 * tree, and therefore kappa_near. One copy exists so that six transcriptions cannot drift
 * (task-4 brief ruling 3; the omission of this normalisation already cost one Critical finding).
 *
 * Every generator does h-refinement of a FIXED shape: the same geometric object at every n, only
 * the sampling density changes — the scaling model of the 1k-30k target.
 *
 * @see .superpowers/sdd/2026-09-11-phase2h-queue/scratch/curves.ts
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.5, §11.3
 */
import type { Edge, Vec3 } from '../../../src/core/testConfigs';

export interface Curve {
    vertices: Vec3[];
    edges: Edge[];
}

/** `scratch/curves.ts` `lcg`. */
function lcg(seed: number): () => number {
    let s = seed >>> 0;
    return () => {
        s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
        return s / 4294967296;
    };
}

/** `scratch/curves.ts` `normalize`: bbox max extent 1, bbox centred at the origin. */
function normalize(X: Float64Array, n: number): void {
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n; i++) {
        for (let c = 0; c < 3; c++) {
            const v = X[3 * i + c];
            if (v < lo[c]) lo[c] = v;
            if (v > hi[c]) hi[c] = v;
        }
    }
    const ext = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
    const s = ext > 0 ? 1 / ext : 1;
    for (let i = 0; i < n; i++) {
        for (let c = 0; c < 3; c++) {
            X[3 * i + c] = (X[3 * i + c] - 0.5 * (lo[c] + hi[c])) * s;
        }
    }
}

function toCurve(X: Float64Array, n: number, closed: boolean): Curve {
    normalize(X, n);
    const vertices: Vec3[] = [];
    for (let i = 0; i < n; i++) vertices.push([X[3 * i], X[3 * i + 1], X[3 * i + 2]]);
    const edges: Edge[] = [];
    const m = closed ? n : n - 1;
    for (let I = 0; I < m; I++) edges.push([I, closed ? (I + 1) % n : I + 1]);
    return { vertices, edges };
}

/** `scratch/curves.ts` `circle`. */
export function circle(n: number): Curve {
    const X = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) {
        const t = (2 * Math.PI * i) / n;
        X[3 * i] = Math.cos(t);
        X[3 * i + 1] = Math.sin(t);
        X[3 * i + 2] = 0;
    }
    return toCurve(X, n, true);
}

/** `scratch/curves.ts` `torusKnot(n, 2, 3)`: R=2, r=0.8. */
export function torusKnot(n: number): Curve {
    const p = 2;
    const q = 3;
    const X = new Float64Array(3 * n);
    const R = 2;
    const r = 0.8;
    for (let i = 0; i < n; i++) {
        const t = (2 * Math.PI * i) / n;
        const rad = R + r * Math.cos(q * t);
        X[3 * i] = rad * Math.cos(p * t);
        X[3 * i + 1] = rad * Math.sin(p * t);
        X[3 * i + 2] = r * Math.sin(q * t);
    }
    return toCurve(X, n, true);
}

/** `scratch/curves.ts` `tangle(n, 20260911)`. */
export function tangle(n: number): Curve {
    const rnd = lcg(20260911);
    const H = 7;
    const a: number[] = [];
    const ph: number[] = [];
    for (let k = 0; k < 3 * H; k++) {
        a.push(0.35 * (rnd() - 0.5));
        ph.push(2 * Math.PI * rnd());
    }
    const X = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) {
        const t = (2 * Math.PI * i) / n;
        const base = [Math.cos(t), Math.sin(t), 0.3 * Math.sin(3 * t)];
        for (let c = 0; c < 3; c++) {
            let v = base[c];
            for (let k = 0; k < H; k++) {
                const h = k + 2;
                v += a[c * H + k] * Math.sin(h * t + ph[c * H + k]);
            }
            X[3 * i + c] = v;
        }
    }
    return toCurve(X, n, true);
}

/** `scratch/curves.ts` `openHelix(n, 4)` — open chain, `m = n - 1`. */
export function openHelix(n: number): Curve {
    const turns = 4;
    const X = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) {
        const t = (turns * 2 * Math.PI * i) / (n - 1);
        X[3 * i] = Math.cos(t);
        X[3 * i + 1] = Math.sin(t);
        X[3 * i + 2] = (3 * i) / (n - 1);
    }
    return toCurve(X, n, false);
}

export const FIXTURES = { circle, torusKnot, tangle, openHelix } as const;
export type FixtureName = keyof typeof FIXTURES;
