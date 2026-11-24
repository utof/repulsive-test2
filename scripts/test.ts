#!/usr/bin/env bun
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
/**
 * Test runner: one `bun test <file>` process per test file, `--jobs` (default 4) at a time,
 * longest first, then a per-file time table.
 *
 * Why: bun 1.3.11 runs every file in ONE process and has no file-level parallelism. In that
 * shared process the matvec/GMRES-bound hierarchical tests slow down x4-6 (measured
 * 2026-11-24: `schur.test.ts` 182 s alone vs 878 s in-suite, `operator.test.ts` 9 s vs 55 s),
 * consistent with JIT type-profile pollution of the hot apply kernel by earlier files. One
 * process per file removes that penalty and lets four files run at once (1285 s -> 389 s wall
 * on the unchanged suite, same load). See README.md "Tests" for the tiers.
 *
 * Why the extra checks on top of bun's exit code: a gate that cannot fail reads exactly like
 * one that passed (@issue utof/repulsive-test2#70). A file counts as green only if bun exited
 * 0 AND printed a summary AND that summary shows no fail and no error; and the full tier
 * fails outright if anything was skipped, because a skipped full-tier test is a check that
 * did not run.
 *
 * Usage:
 *   bun scripts/test.ts [--full] [--jobs N] [path-substring ...] [-- extra bun test args]
 *   bun run test         default tier
 *   bun run test:full    full tier (TEST_TIER=full in every child)
 *
 * Longest-first order comes from `.cache/test-times.json` (git-ignored, rewritten after every
 * run, per tier); files with no recorded time go first, largest first, since a new file may be
 * a heavy one.
 */
import { Glob } from 'bun';

type Tier = 'default' | 'full';

interface Args {
    jobs: number;
    tier: Tier;
    filters: string[];
    passThrough: string[];
}

interface Counts {
    pass: number;
    skip: number;
    todo: number;
    fail: number;
    errors: number;
    /** `Ran N tests across 1 file.`; null when bun printed no summary (crash, kill). */
    ran: number | null;
}

interface FileResult extends Counts {
    file: string;
    exitCode: number;
    seconds: number;
    output: string;
    ok: boolean;
    reason: string;
}

const TIMES_PATH = '.cache/test-times.json';
const TEST_GLOB = 'test/**/*.test.ts';

function parseArgs(argv: string[]): Args {
    let jobs = 4;
    // `TEST_TIER=full bun run test` is honoured too, so the env var alone selects the tier
    let tier: Tier = process.env.TEST_TIER === 'full' ? 'full' : 'default';
    const filters: string[] = [];
    const passThrough: string[] = [];
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i] as string;
        if (a === '--') {
            passThrough.push(...argv.slice(i + 1));
            break;
        }
        if (a === '--full') tier = 'full';
        else if (a === '--jobs' || a === '-j') jobs = Number(argv[++i]);
        else if (a.startsWith('--jobs=')) jobs = Number(a.slice('--jobs='.length));
        else if (/^-j\d+$/.test(a)) jobs = Number(a.slice(2));
        else if (a.startsWith('-')) throw new Error(`unknown option ${a}`);
        else filters.push(a);
    }
    if (!Number.isInteger(jobs) || jobs < 1) throw new Error(`--jobs must be a positive integer`);
    return { jobs, tier, filters, passThrough };
}

function readTimes(): Record<Tier, Record<string, number>> {
    const empty = { default: {}, full: {} };
    if (!existsSync(TIMES_PATH)) return empty;
    try {
        return { ...empty, ...JSON.parse(readFileSync(TIMES_PATH, 'utf8')) };
    } catch {
        return empty;
    }
}

/** Unknown files first (largest first), then known ones by recorded seconds, longest first. */
function order(files: string[], times: Record<string, number>): string[] {
    const unknown = files.filter((f) => times[f] === undefined);
    const known = files.filter((f) => times[f] !== undefined);
    unknown.sort((a, b) => statSync(b).size - statSync(a).size);
    known.sort((a, b) => (times[b] as number) - (times[a] as number));
    return [...unknown, ...known];
}

function lastInt(text: string, re: RegExp): number | null {
    let last: number | null = null;
    for (const m of text.matchAll(re)) last = Number(m[1]);
    return last;
}

/**
 * Parse bun's end-of-file summary from STDERR only (bun writes its report there, `console.log`
 * goes to stdout), last match wins, so a test's own log line cannot spoof the counts.
 */
function parseSummary(text: string): Counts {
    return {
        pass: lastInt(text, /^\s*(\d+) pass\s*$/gm) ?? 0,
        skip: lastInt(text, /^\s*(\d+) skip\s*$/gm) ?? 0,
        todo: lastInt(text, /^\s*(\d+) todo\s*$/gm) ?? 0,
        fail: lastInt(text, /^\s*(\d+) fail\s*$/gm) ?? 0,
        errors: lastInt(text, /^\s*(\d+) errors?\s*$/gm) ?? 0,
        ran: lastInt(text, /^Ran (\d+) tests? across \d+ files?\./gm),
    };
}

/** Collect a stream into `own` and also into the shared `merged` buffer, in arrival order. */
async function drain(
    stream: ReadableStream<Uint8Array>,
    own: string[],
    merged: string[],
): Promise<void> {
    const decoder = new TextDecoder();
    const reader = stream.getReader();
    for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        const text = decoder.decode(value, { stream: true });
        own.push(text);
        merged.push(text);
    }
}

async function runFile(file: string, tier: Tier, passThrough: string[]): Promise<FileResult> {
    const t0 = performance.now();
    const proc = Bun.spawn([process.execPath, 'test', `./${file}`, ...passThrough], {
        env: { ...process.env, TEST_TIER: tier },
        stdout: 'pipe',
        stderr: 'pipe',
    });
    const merged: string[] = [];
    const stderr: string[] = [];
    await Promise.all([drain(proc.stdout, [], merged), drain(proc.stderr, stderr, merged)]);
    const exitCode = await proc.exited;
    const seconds = (performance.now() - t0) / 1000;
    const output = merged.join('');
    const c = parseSummary(stderr.join(''));
    let reason = '';
    if (exitCode !== 0) reason = `exit ${exitCode}`;
    else if (c.ran === null) reason = 'no bun summary';
    else if (c.fail > 0) reason = `${c.fail} fail`;
    else if (c.errors > 0) reason = `${c.errors} error`;
    // bun already exits 1 on these; the text check keeps it honest if that ever changes
    else if (/Unhandled error/i.test(stderr.join(''))) reason = 'unhandled error';
    return { file, exitCode, seconds, output, ...c, ok: reason === '', reason };
}

function pad(s: string | number, n: number, left = false): string {
    const t = String(s);
    return left ? t.padEnd(n) : t.padStart(n);
}

async function main(): Promise<number> {
    const args = parseArgs(process.argv.slice(2));
    const all = [...new Glob(TEST_GLOB).scanSync({ dot: false })].sort();
    const files = args.filters.length
        ? all.filter((f) => args.filters.some((p) => f.includes(p.replace(/^\.\//, ''))))
        : all;
    if (files.length === 0) {
        console.error(`no test files match ${args.filters.join(' ')}`);
        return 1;
    }
    const allTimes = readTimes();
    const queue = order(files, allTimes[args.tier]);
    console.log(
        `tier=${args.tier}  files=${queue.length}  jobs=${args.jobs}` +
            (args.tier === 'default' ? '  (full tier: bun run test:full)' : ''),
    );

    const t0 = performance.now();
    const results: FileResult[] = [];
    let next = 0;
    const worker = async (): Promise<void> => {
        while (next < queue.length) {
            const file = queue[next++] as string;
            const r = await runFile(file, args.tier, args.passThrough);
            results.push(r);
            const tag = r.ok ? 'ok  ' : 'FAIL';
            console.log(
                `[${pad(results.length, String(queue.length).length)}/${queue.length}] ${tag} ` +
                    `${pad(r.seconds.toFixed(1), 6)}s  ${r.pass} pass  ${r.skip} skip  ` +
                    `${r.file}${r.ok ? '' : `  (${r.reason})`}`,
            );
        }
    };
    await Promise.all(Array.from({ length: Math.min(args.jobs, queue.length) }, worker));
    const wall = (performance.now() - t0) / 1000;

    const failed = results.filter((r) => !r.ok);
    for (const r of failed) {
        console.log(`\n${'='.repeat(30)} ${r.file} (${r.reason}) ${'='.repeat(30)}`);
        console.log(r.output.trimEnd());
    }

    results.sort((a, b) => b.seconds - a.seconds);
    console.log(
        `\n${pad('s', 8)} ${pad('ran', 5)} ${pad('pass', 5)} ${pad('skip', 5)} ${pad('fail', 5)}  file`,
    );
    for (const r of results) {
        console.log(
            `${pad(r.seconds.toFixed(2), 8)} ${pad(r.ran ?? '?', 5)} ${pad(r.pass, 5)} ` +
                `${pad(r.skip, 5)} ${pad(r.fail, 5)}  ${r.file}${r.ok ? '' : `  FAIL (${r.reason})`}`,
        );
    }
    const sum = (k: keyof Counts): number => results.reduce((s, r) => s + (r[k] ?? 0), 0);
    const cpu = results.reduce((s, r) => s + r.seconds, 0);
    const skip = sum('skip');
    console.log(
        `\ntier=${args.tier}  files=${results.length}  ran=${sum('ran')}  pass=${sum('pass')}  ` +
            `skip=${skip}  todo=${sum('todo')}  fail=${sum('fail')}  errors=${sum('errors')}  ` +
            `failed files=${failed.length}`,
    );
    console.log(
        `wall ${wall.toFixed(1)} s, sum of file times ${cpu.toFixed(1)} s, jobs=${args.jobs}`,
    );

    // record times for the next run's longest-first order (only files that finished green)
    for (const r of results) if (r.ok) allTimes[args.tier][r.file] = Number(r.seconds.toFixed(2));
    mkdirSync(dirname(TIMES_PATH), { recursive: true });
    writeFileSync(TIMES_PATH, `${JSON.stringify(allTimes, null, 2)}\n`);

    let code = failed.length > 0 ? 1 : 0;
    if (args.tier === 'default') {
        console.log(
            `${skip} test(s) skipped as full-tier: they run only under bun run test:full (TEST_TIER=full).`,
        );
    } else if (skip > 0 && args.passThrough.length === 0) {
        // Why: the full tier is the run that is supposed to check everything; a skip here is a
        // check that silently did not run (README.md "Tests")
        console.log(`FULL TIER SKIPPED ${skip} test(s): it must skip none.`);
        code = 1;
    }
    console.log(code === 0 ? 'PASS' : 'FAIL');
    return code;
}

process.exit(await main());
