/**
 * JS f64 throughput calibration for the Phase 2a CPU-side algebra.
 *
 * Slice 2a moves ONLY the Cholesky factorization to the GPU; the Schur +
 * Woodbury algebra that consumes the read-back factor stays in JS f64. Three
 * constants therefore carry the whole P2a prediction table and both Amdahl
 * ceilings of the Phase 2 spec:
 *
 *   backsolveMs  one N-backsolve (forward + back substitution on a dense
 *                lower-triangular factor, flat row-major `Float64Array`,
 *                2N^2 flops) — the shape every K-solve and every setup solve
 *                of spec 4.1 runs.
 *   matvecMs     one dense N x N matvec — the shape every IR residual runs
 *                (three of them per residual, one per coordinate block).
 *   asigmaMs     A_sigma build + f32 cast into a PREALLOCATED, reused buffer
 *                (N^2 adds + N^2 casts) — the per-step upload staging cost.
 *
 * Why sessions AND processes: these constants are not stable to better than
 * ~2x on this box. A single median inside one process understates the spread,
 * because a large part of it is JIT/allocator/machine-load state that only
 * re-rolls on a fresh process. This harness therefore spawns `--processes`
 * child Bun processes, each running `--sessions` independent measurement
 * sessions of median-of-`REPS`, and reports min / median / max across every
 * session of every process. The spec quotes the RANGE, and sets every
 * pre-registered bar below the PESSIMISTIC end of the resulting prediction —
 * not below the central estimate.
 *
 * Why TWO allocation regimes, `reuse` and `churn`: independent implementations
 * of these same three loops disagreed by 2-4x, and the whole difference is
 * allocation pressure, not arithmetic. `reuse` keeps every buffer
 * driver-owned across reps — what slice 2a actually commits to (4.3: the f32
 * staging buffer is REUSED, and the readback target is one `ReadbackBuffer`
 * per engine). `churn` allocates the N x N f32 staging buffer inside each
 * timed rep; its GC then lands in the neighbouring backsolve/matvec timings
 * too. Measuring both is what makes the published range honest: `churn` is the
 * regime the earlier hand-rolled scripts were accidentally in, and it is the
 * cost if buffer reuse is ever lost, so it — not `reuse` — supplies the
 * pessimistic end that the bars must clear.
 *
 * CPU-only (Bun, no WebGPU): the hardware-adapter / INVALID rule of
 * `bench/gpu/README.md` does not apply to this file.
 *
 * **RUN IT ALONE.** These are scalar CPU loops and they are load-sensitive in
 * the same way `bench/gpu/README.md` warns the Phase 1 dE gate is. Measured on
 * this box: an unrelated 6-process pytest run elsewhere on the machine moved
 * the N=960 backsolve from 1.77 ms to 5.14 ms — 2.9x, far outside the spread
 * this harness is designed to characterize. A result taken under competing
 * load is not a wider measurement of the same thing; it is a measurement of
 * something else. `ps` %CPU will not tell you: it is a process-lifetime
 * average, so a long-lived desktop process reads high while idle. This harness
 * therefore records `/proc/loadavg` before and after every run and marks the
 * result `loadOk: false` when the one-minute average exceeds `MAX_LOADAVG`;
 * a `loadOk: false` file must not be used to place a bar.
 *
 * Run (repo root):
 *   bun bench/gpu/jsAlgebra.ts
 *   bun bench/gpu/jsAlgebra.ts --processes 5 --sessions 3 --out <path>
 *
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md 5 (P2a prediction table), 2 (ceilings)
 * @see bench/gpu/README.md ("Provenance" — the gitShaShort convention)
 */

import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, normalize } from 'node:path';

/** Fixture sizes: the P2a gate fixtures, plus N=1000 (the parent's G7 size). */
const FIXTURE_N = [480, 960, 1000];

/** Timed repetitions inside one session; the session's value is their median. */
const REPS = 15;

/** Untimed repetitions before each session, so the JIT has tiered up. */
const WARMUP = 3;

/**
 * One-minute loadavg above which a run is not trustworthy for placing a bar.
 * This box has ~12 cores and a persistently busy desktop session; 1.5 leaves
 * room for that baseline while excluding the multi-process competing loads that
 * moved the N=960 backsolve by 2.9x (see the header).
 */
const MAX_LOADAVG = 1.5;

/** One-minute load average, or null where /proc is unavailable. */
function loadavg1(): number | null {
    try {
        return Number(readFileSync('/proc/loadavg', 'utf8').split(' ')[0]);
    } catch {
        return null;
    }
}

/** Allocation regime under test; see the file header for why both are measured. */
type Mode = 'reuse' | 'churn';

/** One measurement session: the three constants at one N, each a median-of-REPS. */
interface Session {
    n: number;
    mode: Mode;
    backsolveMs: number;
    matvecMs: number;
    asigmaMs: number;
}

const median = (a: number[]): number => a.slice().sort((p, q) => p - q)[a.length >> 1] as number;

/**
 * Dense lower-triangular factor with an O(1) diagonal, deterministic (no
 * Math.random, matching `src/core/fixtures.ts`'s determinism rule) so a re-run
 * measures the same memory traffic.
 * Why a hand-rolled LCG: the numbers only have to be reproducible and non-denormal;
 * denormals would make the measurement about the FPU's slow path, not about bandwidth.
 */
function makeL(n: number): Float64Array {
    const L = new Float64Array(n * n);
    let s = 12345;
    const rnd = (): number => {
        s = (s * 1103515245 + 12345) & 0x7fffffff;
        return s / 0x7fffffff;
    };
    for (let i = 0; i < n; i++) {
        for (let j = 0; j <= i; j++) L[i * n + j] = j === i ? 1 + rnd() : (rnd() - 0.5) * 0.01;
    }
    return L;
}

/**
 * Forward + back substitution, 2N^2 flops. The back sweep reads L column-wise
 * from a row-major buffer — that stride is deliberate and load-bearing: it is
 * what `src/gpu/saddle.ts` will do on the read-back factor, and it is the
 * reason the measured GF/s is well under the matvec's.
 */
function backsolve(
    L: Float64Array,
    n: number,
    b: Float64Array,
    y: Float64Array,
    x: Float64Array,
): void {
    for (let i = 0; i < n; i++) {
        let s = b[i] as number;
        const ro = i * n;
        for (let j = 0; j < i; j++) s -= (L[ro + j] as number) * (y[j] as number);
        y[i] = s / (L[ro + i] as number);
    }
    for (let i = n - 1; i >= 0; i--) {
        let s = y[i] as number;
        for (let k = i + 1; k < n; k++) s -= (L[k * n + i] as number) * (x[k] as number);
        x[i] = s / (L[i * n + i] as number);
    }
}

/** Dense N x N matvec, 2N^2 flops, row-major and contiguous. */
function matvec(A: Float64Array, n: number, v: Float64Array, out: Float64Array): void {
    for (let i = 0; i < n; i++) {
        let s = 0;
        const ro = i * n;
        for (let j = 0; j < n; j++) s += (A[ro + j] as number) * (v[j] as number);
        out[i] = s;
    }
}

/** Driver-owned buffers for one N, allocated once and reused across sessions. */
function alloc(n: number) {
    return {
        L: makeL(n),
        b: new Float64Array(n).fill(1),
        y: new Float64Array(n),
        x: new Float64Array(n),
        o: new Float64Array(n),
        f32: new Float32Array(n * n),
    };
}

/** One session at one N. In `churn` mode the f32 staging buffer is re-allocated per timed rep. */
function session(n: number, buf: ReturnType<typeof alloc>, mode: Mode): Session {
    const { L, b, y, x, o } = buf;
    const stage = (): Float32Array => (mode === 'reuse' ? buf.f32 : new Float32Array(n * n));
    for (let w = 0; w < WARMUP; w++) {
        backsolve(L, n, b, y, x);
        matvec(L, n, b, o);
        const f32 = stage();
        for (let i = 0; i < n * n; i++) f32[i] = (L[i] as number) + 1e-3;
    }
    const bs: number[] = [];
    const mv: number[] = [];
    const sh: number[] = [];
    for (let r = 0; r < REPS; r++) {
        let t = performance.now();
        backsolve(L, n, b, y, x);
        bs.push(performance.now() - t);
        t = performance.now();
        matvec(L, n, b, o);
        mv.push(performance.now() - t);
        t = performance.now();
        const f32 = stage();
        for (let i = 0; i < n * n; i++) f32[i] = (L[i] as number) + 1e-3;
        sh.push(performance.now() - t);
    }
    return { n, mode, backsolveMs: median(bs), matvecMs: median(mv), asigmaMs: median(sh) };
}

const argv = Bun.argv.slice(2);
const flag = (name: string, dflt: string): string => {
    const i = argv.indexOf(`--${name}`);
    return i >= 0 && argv[i + 1] !== undefined ? (argv[i + 1] as string) : dflt;
};

if (argv.includes('--child')) {
    // Child mode: run `--sessions` sessions per N and emit them as JSON on stdout.
    const sessions: Session[] = [];
    for (const n of FIXTURE_N) {
        const buf = alloc(n);
        for (const mode of ['reuse', 'churn'] as Mode[]) {
            for (let s = 0; s < Number(flag('sessions', '3')); s++)
                sessions.push(session(n, buf, mode));
        }
    }
    console.log(JSON.stringify(sessions));
} else {
    const processes = Number(flag('processes', '5'));
    const sessions = flag('sessions', '3');
    const loadBefore = loadavg1();
    const all: Session[] = [];
    for (let p = 0; p < processes; p++) {
        const proc = Bun.spawnSync(['bun', import.meta.path, '--child', '--sessions', sessions]);
        if (!proc.success) throw new Error(`child ${p} failed: ${proc.stderr.toString()}`);
        all.push(...(JSON.parse(proc.stdout.toString().trim()) as Session[]));
    }
    const rows = FIXTURE_N.map((n) => {
        const mine = all.filter((s) => s.n === n);
        // Union over BOTH allocation regimes: `reuse` supplies the optimistic end
        // (what 2a is designed to achieve), `churn` the pessimistic end the bars must clear.
        const stat = (pick: (s: Session) => number) => {
            const v = mine.map(pick);
            const lo = Math.min(...v);
            const hi = Math.max(...v);
            const byMode = Object.fromEntries(
                (['reuse', 'churn'] as Mode[]).map((m) => [
                    m,
                    median(mine.filter((s) => s.mode === m).map(pick)),
                ]),
            );
            return { min: lo, median: median(v), max: hi, spread: hi / lo, medianByMode: byMode };
        };
        return {
            n,
            samples: mine.length,
            backsolveMs: stat((s) => s.backsolveMs),
            matvecMs: stat((s) => s.matvecMs),
            asigmaMs: stat((s) => s.asigmaMs),
        };
    });
    for (const r of rows) {
        const f = (q: {
            min: number;
            max: number;
            spread: number;
            medianByMode: Record<string, number>;
        }) =>
            `${q.medianByMode.reuse?.toFixed(3)}/${q.medianByMode.churn?.toFixed(3)} [${q.min.toFixed(3)}-${q.max.toFixed(3)}] (${q.spread.toFixed(2)}x)`;
        console.log(
            `N=${r.n} reuse/churn [min-max]  backsolve ${f(r.backsolveMs)} | matvec ${f(r.matvecMs)} | Asigma+f32 ${f(r.asigmaMs)}`,
        );
    }
    // Provenance: the Phase 2 spec commit that adds this file is amended repeatedly
    // (CLAUDE.md: one commit per spec doc), so run-time HEAD would key the result to a
    // commit about to be rewritten — the failure `bench/gpu/README.md` "Provenance" records.
    // HEAD~1 is the stable anchor; nothing measured here depends on the spec doc.
    const sha = Bun.spawnSync(['git', 'rev-parse', '--short', 'HEAD~1']).stdout.toString().trim();
    const today = new Date().toISOString().slice(0, 10);
    const loadAfter = loadavg1();
    const loadOk =
        loadBefore !== null &&
        loadAfter !== null &&
        loadBefore <= MAX_LOADAVG &&
        loadAfter <= MAX_LOADAVG;
    if (!loadOk) {
        console.warn(
            `WARNING loadavg ${loadBefore} -> ${loadAfter} exceeds ${MAX_LOADAVG}: this run is NOT usable for placing a bar.`,
        );
    }
    const out = {
        gate: 'phase2aJsAlgebra',
        loadavg: { before: loadBefore, after: loadAfter, max: MAX_LOADAVG },
        loadOk,
        gitShaShort: sha,
        gitShaNote:
            'parent of the (amended) Phase 2 spec commit that adds this script; nothing measured here depends on the spec doc',
        date: today,
        tool: `bun ${Bun.version}`,
        method: {
            processes,
            sessionsPerProcess: Number(sessions),
            repsPerSession: REPS,
            warmup: WARMUP,
        },
        rows,
        sessions: all,
    };
    const outPath = flag(
        'out',
        join(dirname(import.meta.path), '..', 'results', `${today}-js-f64-algebra.json`),
    );
    writeFileSync(outPath, `${JSON.stringify(out, null, 2)}\n`);
    console.log('wrote', normalize(outPath));
}
