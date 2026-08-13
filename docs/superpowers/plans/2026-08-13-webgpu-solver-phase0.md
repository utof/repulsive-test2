# WebGPU Solver — Phase 0 (Gates & Baselines) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute every Phase 0 kill gate and baseline from the spec (`docs/superpowers/specs/2026-08-13-webgpu-solver-design.md` §4–§5) so the GPU milestone can be killed or committed on measured evidence before any product code exists.

**Architecture:** A browser GPU harness (`bench/gpu/`) drives spike pages through headless Chrome via raw CDP (project-memory recipe), asserting a hardware adapter first (G0a). Fixture generators move to a shared `src/core` module consumed by bench, harness, and future gates. All gate results are committed JSON in `bench/results/`, keyed by git SHA + adapter info.

**Tech Stack:** Bun (driver scripts + tests), three.js r185 `three/webgpu` + TSL compute (verified surface: `Fn().compute()`, `instancedArray`, `wgslFn`, `renderer.compute([...])`, `getArrayBufferAsync`, `trackTimestamp`/`resolveTimestampsAsync` — see `docs/2026-08-13-ai-research-webgpu-compute.md` [IMPL §1/§6]), Chrome DevTools Protocol over WebSocket, Python/numpy oracle for G6.

**Scope note (spec §5):** this plan is Phase 0 ONLY. Phases 1–3 get their own plans after this phase's gate report — Phase 2's content is decided by the G5 experiment and cannot be honestly planned before G0t/G1 numbers exist.

**Conventions for every task:**
- Biome formats on commit (4-space, single quotes, lineWidth 100); snippet whitespace is a guide, not a gate.
- `verbatimModuleSyntax` on: `import type { … }` for types.
- Typecheck: `bunx tsc --noEmit`. Tests: `bun test <path>`.
- Every new exported symbol carries TSDoc with `@see` this plan or the spec (CLAUDE.md rule).
- **TSL/WebGPU API steps:** before writing each spike, verify the exact API against `node_modules/three/src/**` (the research doc cites file:line for every symbol) — the surface is younger than the render API and this is the repo's verify-or-not FIRE case ("about to write import for a package surface not already used in this repo").
- Commit messages end with the standard `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer.

**File structure (locked in by this plan):**

| Path | Responsibility |
|---|---|
| `src/core/fixtures.ts` | Deterministic parametric fixture generators (trefoil, near-touch pair) shared by bench/harness/tests. Pure, no deps beyond types. |
| `test/core/fixtures.test.ts` | Generator invariants. |
| `bench/gpu/harness.html` | Single spike page: loads three/webgpu, registers named spikes, exposes `window.__runSpike(name)`. |
| `bench/gpu/spikes.ts` | The spike implementations (G0t/G1/G2/G3/G4), each returning a JSON-serializable result object. Bundled into the page by the dev server. |
| `bench/gpu/drive.ts` | Bun CDP driver: launches Chrome, asserts adapter (G0a), runs named spikes, writes `bench/results/*.json`. |
| `bench/gpu/README.md` | Launch recipe (G0a outcome), how to run each gate, INVALID-vs-FAIL semantics. |
| `oracle/check_kappa_peredge.py` | G6: κ(K) power-iteration estimates, perEdge + total modes, N sweep. |
| `bench/results/2026-08-*-gpu-phase0-*.json` | Committed gate results. |

---

### Task ordering note (review F2)

G0a is the FIRST task per spec §4/§5 — the harness task below is Task 1 and
everything browser-dependent follows it. **Environment fact (CLAUDE.md
scarecrow note overrides the stale project memory):** this is a real Linux
machine with an NVIDIA Quadro RTX 3000 (`nvidia-smi` works); the
memory file claiming "VirtualBox VM, no hardware GPU" predates that
correction — trust CLAUDE.md. The stop-branch below still exists because
*Chrome* may refuse the hardware Vulkan adapter even where the driver works.

### Task 1: GPU harness + G0a hardware adapter (FIRST; spec §4 G0a)

**Files:**
- Create: `bench/gpu/harness.html`, `bench/gpu/spikes.ts`, `bench/gpu/drive.ts`, `bench/gpu/README.md`
- Modify: `server.ts` — **verified fact:** it serves only `/`, `/index.html`, and `*.ts`/`*.tsx` (server.ts:13-46); add a static `.html` route (Content-Type `text/html`, `Cache-Control: no-store`) mirroring the index.html handler, nothing more.

- [ ] **Step 1: Write the harness page + spike registry**

```html
<!-- bench/gpu/harness.html -->
<!doctype html>
<html>
<head><meta charset="utf-8" /><title>gpu gate harness</title></head>
<body>
<script type="module" src="./spikes.ts"></script>
</body>
</html>
```

```ts
// bench/gpu/spikes.ts — spike registry; each spike returns JSON-serializable data.
import * as THREE from 'three/webgpu';

type SpikeResult = Record<string, unknown>;
const spikes: Record<string, () => Promise<SpikeResult>> = {};

/** G0a: report the adapter we actually got. INVALID-vs-FAIL decisions happen
 * in the driver, not here. @see spec §4 G0a */
spikes.adapterInfo = async () => {
    const renderer = new THREE.WebGPURenderer();
    await renderer.init();
    // backend.device verified at WebGPUBackend.js:111 [IMPL §2]
    const device = (renderer.backend as { device: GPUDevice }).device;
    const info = device.adapterInfo; // GPUAdapterInfo: vendor/architecture/device/description
    return {
        vendor: info.vendor,
        architecture: info.architecture,
        device: info.device,
        description: info.description,
        limits: { maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX },
    };
};

// Later tasks append spikes here (g0t, g1, g3, g2, g4).

declare global {
    interface Window {
        __runSpike: (name: string) => Promise<string>;
    }
}
window.__runSpike = async (name) => JSON.stringify(await spikes[name]());
```

- [ ] **Step 2: Write the CDP driver with the adapter precondition**

```ts
// bench/gpu/drive.ts — bun bench/gpu/drive.ts <spike> [--out label]
// Launches Chrome headless against the dev server, asserts hardware adapter
// (G0a), runs the named spike, writes bench/results/<date>-gpu-<label>.json.
// INVALID (software adapter) is a distinct exit from FAIL.
// @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §3, §4 G0a
import { spawn } from 'node:child_process';

const CHROME = process.env.CHROME_BIN ?? 'google-chrome';
const PORT = 9223;
// Candidate flag sets, tried in order until adapterInfo reports NVIDIA hardware.
const FLAG_SETS: string[][] = [
    ['--headless=new', '--enable-unsafe-webgpu', '--enable-features=Vulkan',
     '--use-angle=vulkan', '--enable-gpu', '--no-sandbox'],
    ['--headless=new', '--enable-unsafe-webgpu', '--enable-features=Vulkan',
     '--ignore-gpu-blocklist', '--no-sandbox'],
    // headed fallback (spec G0a: allowed after honest headless effort)
    ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--no-sandbox'],
];
```

CDP transport (review F5 — the concrete steps, no library, no new deps):
1. `spawn(CHROME, [...flags, `--remote-debugging-port=${PORT}`, 'http://localhost:3000/bench/gpu/harness.html'])`.
2. Poll `http://localhost:${PORT}/json` (fetch, retry ~200 ms up to 10 s) until
   a target with `url` ending `harness.html` appears; take its
   `webSocketDebuggerUrl`.
3. `new WebSocket(webSocketDebuggerUrl)` (Bun-native WebSocket); implement
   `send(method, params)` with incrementing `id` and a Map of pending
   promises resolved on id-matched messages.
4. `Runtime.enable`, subscribe `Runtime.consoleAPICalled` (grep out the known
   SwiftShader noise lines listed in `bench/gpu/README.md`).
5. `Runtime.evaluate` with `{ expression: `window.__runSpike('<name>')`,
   awaitPromise: true, returnByValue: true }`; parse `result.result.value`.
6. Kill the Chrome process; write the results JSON
   `{ gate, status, adapter, flags, gitShaShort, date, data }`.

```ts
function classifyAdapter(info: { vendor: string; description: string }): 'hardware' | 'software' {
    const d = `${info.vendor} ${info.description}`.toLowerCase();
    return d.includes('swiftshader') || d.includes('llvmpipe') || d.includes('software')
        ? 'software'
        : 'hardware';
}
```

- [ ] **Step 3: Run G0a**

Run: `bun run dev` (in background), then `bun bench/gpu/drive.ts adapterInfo --out g0a`
Expected PASS: adapter JSON with `vendor` containing `nvidia` (Quadro RTX
3000 class), result file `"gate": "G0a", "status": "PASS", "flags": [...]`.
Headed-only success IS a pass — record which flag set.
**STOP-BRANCH (review F2): if NO flag set (headless or headed) yields a
hardware adapter, STOP.** Run only the CPU-side tasks (fixtures, CPU
baselines, G6) and report to the user with options (different Chrome channel,
`chrome://gpu` diagnosis, host-side run). Do NOT execute browser gates on a
software adapter — every such result is INVALID by spec §3 and worthless.

- [ ] **Step 4: Write `bench/gpu/README.md`**

Contents: the winning launch recipe verbatim, INVALID-vs-FAIL semantics (a
software-adapter run of ANY gate is INVALID — rerun with the recipe, never
record it), how to run each spike, the SwiftShader console-noise lines to
ignore (from project memory), and an empty "Phase 0 gate report" table
(filled by the final task — it lives HERE, not in the plan doc, so the
one-commit-per-plan-doc rule stays intact; review F7).

- [ ] **Step 5: Commit**

```bash
git add bench/gpu/ server.ts
git commit -m "feat(bench/gpu): CDP harness + G0a hardware-adapter gate (webgpu phase0 T1)"
```

---

### Task 2: Shared fixture module (spec §3 fixture debt)

**Files:**
- Create: `src/core/fixtures.ts`
- Test: `test/core/fixtures.test.ts`
- Modify: `bench/sobolev.bench.ts` (replace its private `trefoil` with the import; the function body moves verbatim)

- [ ] **Step 1: Write the failing test**

```ts
// test/core/fixtures.test.ts
import { describe, expect, test } from 'bun:test';
import { nearTouchPair, trefoil } from '../../src/core/fixtures';

describe('trefoil', () => {
    test('produces n vertices and n closed-loop edges', () => {
        const { vertices, edges } = trefoil(60);
        expect(vertices.length).toBe(60);
        expect(edges.length).toBe(60);
        expect(edges[59]).toEqual([59, 0]);
    });
    test('is deterministic', () => {
        expect(trefoil(120)).toEqual(trefoil(120));
    });
    test('matches the bench parametrization at i=1, n=4 exactly', () => {
        // p(t) = (sin t + 2 sin 2t, cos t − 2 cos 2t, −sin 3t), t = 2πi/n
        const t = (2 * Math.PI) / 4;
        const [x, y, z] = trefoil(4).vertices[1];
        expect(x).toBe(Math.sin(t) + 2 * Math.sin(2 * t));
        expect(y).toBe(Math.cos(t) - 2 * Math.cos(2 * t));
        expect(z).toBe(-Math.sin(3 * t));
    });
});

describe('nearTouchPair', () => {
    test('two disjoint edges separated by the requested gap', () => {
        const { vertices, edges, gap } = nearTouchPair(1e-6);
        expect(vertices.length).toBe(4);
        expect(edges).toEqual([
            [0, 1],
            [2, 3],
        ]);
        expect(gap).toBe(1e-6);
        // closest approach between the two parallel segments is the gap (f64)
        expect(vertices[2][1] - vertices[0][1]).toBeCloseTo(1e-6, 12);
    });
    test('f32 storage destroys the gap — the cancellation property T1/G2 need (review F1)', () => {
        // The whole point of this fixture: coordinates are O(1) values NOT
        // exactly representable in f32, so rounding the positions to f32
        // corrupts the 1e-6 gap by orders of magnitude. If this assertion
        // fails, the fixture is NOT stressing cancellation and every gate
        // built on it is vacuous. @see PREC Q1 (error law ~ β·u_f·(extent/gap))
        const { vertices } = nearTouchPair(1e-6);
        const d32 = Math.fround(vertices[2][1]) - Math.fround(vertices[0][1]);
        const relErr = Math.abs(d32 - 1e-6) / 1e-6;
        expect(relErr).toBeGreaterThan(1e-3);
    });
});
```

- [ ] **Step 2: Run it, confirm it fails**

Run: `bun test test/core/fixtures.test.ts`
Expected: FAIL — `Cannot find module '../../src/core/fixtures'`.

- [ ] **Step 3: Implement**

```ts
// src/core/fixtures.ts
import type { Edge, Vec3 } from './testConfigs';

/**
 * Parametric closed trefoil at arbitrary N — the canonical perf/verification
 * fixture for the WebGPU milestone's fixture matrix (N=60/120/240/480/960/1000).
 * Body moved VERBATIM from bench/sobolev.bench.ts so existing baseline results
 * stay comparable. Deterministic: no Math.random.
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §3 (fixture debt)
 */
export function trefoil(n: number): { vertices: Vec3[]; edges: Edge[] } {
    const vertices: Vec3[] = [];
    const edges: Edge[] = [];
    for (let i = 0; i < n; i++) {
        const t = (2 * Math.PI * i) / n;
        vertices.push([
            Math.sin(t) + 2 * Math.sin(2 * t),
            Math.cos(t) - 2 * Math.cos(2 * t),
            -Math.sin(3 * t),
        ]);
        edges.push([i, (i + 1) % n]);
    }
    return { vertices, edges };
}

/**
 * Two parallel edges at controlled closest-approach `gap`, positioned at O(1)
 * coordinates that are NOT exactly representable in f32 (0.1/2.1/0.3/4.1) —
 * REQUIRED so that f32 position storage corrupts the gap via catastrophic
 * cancellation, which is the failure mode T1/G2 exist to detect. An
 * axis-aligned layout through the origin (exact f32 coords) would cancel
 * nothing and make every gate built on this fixture vacuous (plan review F1).
 * Layout: A (0.1,2.1,0.3)→(4.1,2.1,0.3); B (0.1,2.1+gap,0.3)→(4.1,2.1+gap,0.3).
 * Disjoint (no shared vertices) → the pair contributes energy.
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §3 (T1), §4 (G2)
 * @see docs/2026-08-13-ai-research-gpu-precision.md Q1 (extent/gap error law)
 */
export function nearTouchPair(gap: number): {
    vertices: Vec3[];
    edges: Edge[];
    gap: number;
} {
    const vertices: Vec3[] = [
        [0.1, 2.1, 0.3],
        [4.1, 2.1, 0.3],
        [0.1, 2.1 + gap, 0.3],
        [4.1, 2.1 + gap, 0.3],
    ];
    return { vertices, edges: [[0, 1], [2, 3]], gap };
}
```

(`Edge` is exported at `src/core/testConfigs.ts:2` and the bench already
imports it — verified; review F9.)

- [ ] **Step 4: Run tests, confirm pass**

Run: `bun test test/core/fixtures.test.ts` → PASS.
Run: `bunx tsc --noEmit` → clean.

- [ ] **Step 5: Point the bench at the shared module**

In `bench/sobolev.bench.ts`: delete the private `trefoil` function (≈line 73)
and add `import { trefoil } from '../src/core/fixtures';`. Run
`bun bench/sobolev.bench.ts --help 2>/dev/null || bun bench/sobolev.bench.ts | head -5`
to confirm it still starts (full runs come in Task 2).

- [ ] **Step 6: Commit**

```bash
git add src/core/fixtures.ts test/core/fixtures.test.ts bench/sobolev.bench.ts
git commit -m "feat(fixtures): shared trefoil + near-touch generators (webgpu phase0 T2)"
```

---

### Task 3: CPU baselines to N=1000 (spec §4 "baselines BEFORE any GPU kernel")

**Files:**
- Modify: `bench/sobolev.bench.ts` (case list + reps guard)
- Create (by running): `bench/results/2026-08-13-cpu-baseline-largeN.json`

- [ ] **Step 1: Add the large-N cases via an explicit `--large` mode (review F4)**

The bench's default path runs a full cross-product sizes × {total,perEdge} ×
{reassemble,frozen} × {lu,ldlt} (`bench/sobolev.bench.ts:236-257`) — adding
480/960/1000 to `sizes` would also run reassemble variants at those N
(re-factors per Armijo trial → plausibly hours) and the isolated
micro-medians run `solveSaddle` 9× unconditionally (lines 191-213). So:
add a `--large` CLI flag that bypasses the cross-product and runs ONLY this
explicit tuple list, skipping the isolated-primitive micros for N≥480:

| n | constraintMode | projectionMode | factorMode | K |
|---|---|---|---|---|
| 240 | total | frozen | ldlt | 5 |
| 240 | perEdge | frozen | ldlt | 5 |
| 480 | total | frozen | ldlt | 5 |
| 480 | perEdge | frozen | ldlt | 5 |
| 960 | total | frozen | ldlt | 3 |
| 1000 | total | frozen | ldlt | 3 |

This needs `runCase` to take reps as a parameter (it hardcodes K=5 at line
168) — thread it through, default unchanged so existing invocations are
untouched. perEdge at N≥960: measure one step first; if it exceeds 30 s,
record `"skipped": "step > 30s"` — a skip IS a baseline datum.

- [ ] **Step 2: Run and save**

Run: `bun bench/sobolev.bench.ts --save cpu-baseline-largeN`
Expected: a markdown table printing per-phase medians for every case; JSON
written under `bench/results/`. Sanity: N=480 full step should extrapolate
from N=120 by roughly (480/120)³ ≈ 64× on the factor phase — if it's off by
more than ~5×, investigate before committing (wrong mode or JIT anomaly).

- [ ] **Step 3: Commit the results**

```bash
git add bench/sobolev.bench.ts bench/results/2026-08-13-cpu-baseline-largeN.json
git commit -m "bench: CPU baselines trefoil N=240..1000 (webgpu phase0 T3, pre-registered)"
```

---
### Task 4: `trackTimestamp` in the Viewer boot path (spec §6)

**Files:**
- Modify: `src/scene/Viewer.tsx:376-381` (gl factory)

- [ ] **Step 1: Make the change**

```tsx
gl={async (props) => {
    // trackTimestamp: required by the WebGPU milestone's G3/G7 GPU-time gates
    // (injects timestamp queries around compute/render passes; experimental
    // API — re-verify on any three upgrade).
    // @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §6
    const renderer = new THREE.WebGPURenderer({ ...(props as any), trackTimestamp: true });
    await renderer.init();
    return renderer;
}}
```

- [ ] **Step 2: Verify it boots**

Run: `bunx tsc --noEmit` → clean. Boot check via the Task 1 harness recipe
(hardware adapter): app renders, sim steps, no new console errors beyond the
noise lines documented in `bench/gpu/README.md`.

- [ ] **Step 3: Commit**

```bash
git add src/scene/Viewer.tsx
git commit -m "feat(viewer): trackTimestamp on WebGPURenderer for GPU timing gates (phase0 T4)"
```

---

### Task 5: G3 — timestamp viability (before G0t, which consumes its methodology)

**Files:**
- Modify: `bench/gpu/spikes.ts` (add `g3` spike)

- [ ] **Step 1: Add the spike**

```ts
/** G3: CV of GPU-timestamp totals over 5 runs of a fixed ≥100-dispatch
 * workload. @see spec §4 G3 */
spikes.g3 = async () => {
    const renderer = new THREE.WebGPURenderer({ trackTimestamp: true });
    await renderer.init();
    // workload: 100 iterations of a trivial 64k-thread FMA kernel, batched
    const { Fn, instancedArray, instanceIndex, Loop } = await import('three/tsl');
    const buf = instancedArray(65536, 'float');
    // Inner-loop the FMA chain so the fixed workload is ≥~10 ms of GPU time
    // total across the 100 dispatches — Dawn quantizes timestamps to 100 µs,
    // so CV<10% needs total ≫ the quantum (review F6). If quantization still
    // dominates, relaunch with --enable-webgpu-developer-features as the
    // PRIMARY headless recipe [PREC Q5], not a fallback.
    const kernel = Fn(() => {
        const e = buf.element(instanceIndex);
        const acc = e.toVar();
        // Loop count tuned during implementation so 100 dispatches ≈ 10-30 ms GPU.
        Loop(1024, () => {
            acc.assign(acc.mul(1.000001).add(0.5));
        });
        e.assign(acc);
    })().compute(65536);
    const totals: number[] = [];
    for (let run = 0; run < 5; run++) {
        renderer.compute(Array.from({ length: 100 }, () => kernel));
        await renderer.resolveTimestampsAsync(THREE.TimestampQuery.COMPUTE);
        totals.push(renderer.info.compute.timestamp);
    }
    const mean = totals.reduce((a, b) => a + b) / totals.length;
    const sd = Math.sqrt(totals.map((t) => (t - mean) ** 2).reduce((a, b) => a + b) / totals.length);
    return { totalsMs: totals, cv: sd / mean, pass: sd / mean < 0.1 };
};
```

(NOTE — same-node batching: if `renderer.compute()` dedupes repeated
identical nodes, build 100 distinct kernel nodes; check against
`Renderer.js:2718` behavior while implementing. If timestamps read 0 with
quantization, relaunch Chrome with `--enable-webgpu-developer-features`
[PREC Q5] and record that in the README.)

- [ ] **Step 2: Run, record**

Run: `bun bench/gpu/drive.ts g3 --out g3`
Expected: `cv < 0.1` → PASS recorded. FAIL → per spec, perf gates fall back
to CPU wall-clock + single terminal sync; record the decision in the results
file and README.

- [ ] **Step 3: Commit** (`git add bench/gpu/ bench/results/*g3*.json`, message `feat(bench/gpu): G3 timestamp-viability gate (phase0 T5)`)

---

### Task 6: G0t — throughput probe (G5's estimator input)

**Files:**
- Modify: `bench/gpu/spikes.ts` (add `g0t` spike)

- [ ] **Step 1: Add the spike** — two microbenchmarks under G3 methodology
(5 runs, medians, GPU timestamps or the G3-fallback wall clock):
  1. **FMA rate:** the Task-5 kernel scaled to 2²² threads × an inner
     `for (let i = 0; i < 64; i++)` unrolled multiply-add chain in WGSL via
     `wgslFn` (so the loop can't be folded); report GFLOP/s.
  2. **Dense matvec:** `wgslFn` kernel computing `y = M·x` for a committed
     random-but-seeded 3072×3072 f32 matrix in a storage buffer (one thread
     per row, row-major dot); report effective GFLOP/s and ms per matvec.

Result object: `{ fmaGflops, matvecMs, matvecGflops, method: 'timestamp'|'wallclock' }`.

- [ ] **Step 2: Run, sanity-check, record**

Run: `bun bench/gpu/drive.ts g0t --out g0t`
Sanity (assumption bands, not sourced from the research docs — review F8):
Quadro RTX 3000 peak ≈5.3 TFLOPs f32, spec memory bandwidth ≈336 GB/s; the
FMA probe should land within 20–70% of peak; the matvec is bandwidth-bound
(3072²×4 B ≈ 38 MB/matvec → floor ≈ 0.11 ms; expect 0.1–0.5 ms). Numbers
wildly outside → suspect the kernel got optimized away; fix before recording.

- [ ] **Step 3: Commit** (`feat(bench/gpu): G0t throughput probe — G5 estimator input (phase0 T6)`)

---

### Task 7: G1 — dispatch batching

**Files:**
- Modify: `bench/gpu/spikes.ts` (add `g1` spike)

- [ ] **Step 1: Add the spike**

250 DISTINCT no-op compute nodes (1 workgroup each). Arm A: one
`renderer.compute(nodes)` call. Arm B: 250 separate `renderer.compute(node)`
calls. Measure CPU wall time of the submission loop (performance.now around
the calls; one `await device.queue.onSubmittedWorkDone()` terminal sync per
arm — the [IMPL §4] single-terminal-sync methodology), 5 runs, medians.
Return `{ batchedMs, unbatchedMs, ratio, pass: batchedMs < 2 && batchedMs < 0.25 * unbatchedMs }`.

- [ ] **Step 2: Run, record**

Run: `bun bench/gpu/drive.ts g1 --out g1`
Expected: batched well under the 2 ms clause (per-node residual after
amortizing submit/encoder/pass is ~3–7 µs × 250 ≈ 0.8–1.7 ms [IMPL §4
breakdown]), unbatched ≈ 250 × ~30 µs ≈ 7.5 ms. PASS both clauses. FAIL →
**milestone kill/reshape decision point** (spec §4 G1): stop, report to the
user before any further task.

- [ ] **Step 3: Commit** (`feat(bench/gpu): G1 dispatch-batching kill gate (phase0 T7)`)

---

### Task 8: G2 — two-float reassociation spike (production-kernel precursor)

**Files:**
- Modify: `bench/gpu/spikes.ts` (add `g2` spike)
- Modify: `src/core/fixtures.ts` if the near-touch fixture needs a hi/lo split helper (add `splitHiLo(v: Vec3[]): { hi: Float32Array; lo: Float32Array }` with TSDoc)

- [ ] **Step 1: Add the spike**

A `wgslFn` kernel that computes, for the `nearTouchPair(1e-6)` fixture stored
as hi/lo f32 buffers, the tangent-point pair kernel numerator/denominator
pieces the production kernel will use — at minimum the full
`k(p,q,T) = |T×(p−q)|^α / (|p−q|+ε…)^β`-shaped evaluation for the 4
endpoint combinations, with ALL coordinate differences via
`(hi_i − hi_j) + (lo_i − lo_j)`. CPU side computes the same quantity in f64
(port the arithmetic from `src/core/tangentPointEnergy.ts`'s kernel for ONE
pair — read the file, copy the op order, ε placement after norms).
Return `{ gpu, cpu64, relErr, pass: relErr < 1e-5 }` — the T1 tolerance.
Also compute a plain-f32 (no lo) variant and return its relErr as
`relErrPlain` — expected ~1e-1 territory at gap 1e-6 [PREC Q1]; if
`relErrPlain` is also < 1e-5 the spike itself is broken (the fixture isn't
stressing cancellation) — that's a spike bug, not a pass.

- [ ] **Step 2: Run, record**

Run: `bun bench/gpu/drive.ts g2 --out g2`
Expected: `relErr < 1e-5` AND `relErrPlain > 1e-3` (separation proves the
two-float path survived this compiler). FAIL of the first clause →
**the near-touch accuracy story dies on this adapter**; spec says `'gpu'`
boots to fallback — report before continuing.
NOTE (spec §4 G2): this spike de-risks; the SHIPPING gate re-runs through the
real production `wgslFn` kernel in Phase 1 — carry the fixture and tolerance
forward unchanged.

- [ ] **Step 3: Commit** (`feat(bench/gpu): G2 two-float reassociation spike (phase0 T8)`)

---

### Task 9: G4 — zero-readback render spike

**Files:**
- Modify: `bench/gpu/spikes.ts` (add `g4` spike; this one needs a visible canvas — the harness page gains a `<canvas>` and the spike renders ~120 frames)

- [ ] **Step 1: Add the spike**

1. CPU-precompute 120 frames' worth of wave positions for a 64-edge polyline
   (`y = sin(x + t)`, f32 arrays) — uploaded, NOT GPU-evaluated (spec §4 G4).
2. Build a `LineSegments2`-style fat line whose geometry uses a
   `StorageBufferAttribute` for the interleaved `instanceStart/instanceEnd`
   data (usage flags verified STORAGE|VERTEX [IMPL §2]).
3. Per frame: upload that frame's per-vertex hi/lo buffers ONCE (this spike's
   stand-in for "solver output lands in a storage buffer"), then a compute
   pass runs the **combine/scatter kernel**: `pos = hi + lo` per vertex, then
   scatter vertex i, i+1 into instance slot i's start/end — the §2.7 kernel.
4. Render. Count readbacks (wrap `getArrayBufferAsync` with a counter): must
   be 0 during the 120-frame loop.
5. After the loop: ONE verification readback of the instance buffer for the
   last frame; compare element-wise `Object.is(fround(expected), actual)` —
   f32-exact per spec.
Return `{ frames: 120, readbacksDuringLoop, mismatches, pass: readbacksDuringLoop === 0 && mismatches === 0 }`.

- [ ] **Step 2: Run, record**

Run: `bun bench/gpu/drive.ts g4 --out g4`
PASS → Phase 3's zero-readback architecture is real. FAIL (can't bind the
storage attribute / Line2NodeMaterial fights it) → record, and per spec §4
Phase 3 downgrades to 1-readback-per-frame permanently; that decision goes in
the gate report, not silently.

- [ ] **Step 3: Commit** (`feat(bench/gpu): G4 zero-readback render spike + combine/scatter kernel (phase0 T9)`)

---

### Task 10: G6 — perEdge conditioning sweep

**Files:**
- Create: `oracle/check_kappa_peredge.py`

- [ ] **Step 1: Write the script**

Using `tpe_stage1_oracle`/`tpe_constraints_oracle` machinery (import, don't
copy): for trefoil N=60/120/240/480/960 (generate the same parametrization in
numpy), assemble K = [[Ā, Cᵀ],[C, 0]] for BOTH constraint sets (barycenter+
total vs barycenter+perEdge), estimate κ₂(K) by power iteration on K and on
K⁻¹ (via factorization) — mirroring [PREC Q2]'s method. Print a table and the
fitted growth exponent; write `bench/results/2026-08-13-gpu-phase0-g6.json`
with `{ n, mode, kappa }` rows and `kappaTimesUf32` (κ × 5.96e-8).
Gate check per spec: flag any mode/N where κ·u_f32 > 0.1.

- [ ] **Step 2: Run**

Run: `uv run --with numpy --with scipy python oracle/check_kappa_peredge.py`
Expected: total-mode κ matches [PREC Q2]'s measured points within ~2×
(3.0e1/1.9e2/1.7e3 at N=32/64/128 — different fixture, same law); perEdge
rows are NEW DATA. If perEdge κ·u_f32 > 0.1 at N≤960 → per spec G6, perEdge
mode falls back to CPU f64 solve and the gate report says so.

- [ ] **Step 3: Commit** (`feat(oracle): G6 perEdge conditioning sweep + committed results (phase0 T10)`)

---

### Task 11: Tolerance-harness skeleton that can FAIL (spec §5 Phase 0)

**Files:**
- Modify: `bench/gpu/spikes.ts` (add `toleranceSkeleton` spike)

- [ ] **Step 1: Add the spike**

Implements the T1/T2/T3 *checking machinery* (rel-err and cosine comparators,
fixture loading for `nearTouchPair` + `trefoil(60)`, JSON result shape with
per-gate pass/fail) wired to a **deliberately wrong kernel** (the plain-f32
G2 variant plus a naive left-to-right sum). Expected result: T1 FAILS on the
gap=1e-6 fixture and T2 likely fails at trefoil N≥512-equivalent — proving
the gates can fail. Return the full gate-result object.

- [ ] **Step 2: Run, confirm it reports failures**

Run: `bun bench/gpu/drive.ts toleranceSkeleton --out t-skeleton`
Expected: `t1.pass === false` (that IS this task's pass condition — a
harness that can't fail certifies nothing; same lesson as issue #7).

- [ ] **Step 3: Commit** (`feat(bench/gpu): T1-T3 tolerance harness skeleton, proven falsifiable (phase0 T11)`)

---

### Task 12: Gate report + go/no-go

**Files:**
- Modify: `bench/gpu/README.md` (fill its "Phase 0 gate report" table — the
  report lives in the README, NOT this plan doc, so the one-commit-per-plan-doc
  rule stays intact; review F7)
- All `bench/results/2026-08-13-gpu-phase0-*.json` committed by prior tasks

- [ ] **Step 1: Fill the report table (in `bench/gpu/README.md`)**

| Gate | Result | Number | Consequence |
|---|---|---|---|
| G0a | | adapter = | |
| G0t | | FMA / matvec = | feeds G5 estimator |
| G1 | | batched / unbatched ms = | |
| G2 (spike) | | relErr / relErrPlain = | |
| G3 | | CV = | |
| G4 | | readbacks / mismatches = | |
| G6 | | perEdge κ@960, κ·u = | |
| Baselines | | N=480/960/1000 full-step ms = | G7 ratio anchor at N=480 |

- [ ] **Step 2: Decision per spec §4**

All feasibility gates green → report to the user: Phase 1 plan can be
written. Any red → the spec names the consequence (kill, reshape to
kernels-only, or mode-narrowing); surface it, do NOT proceed silently.

- [ ] **Step 3: Commit**

```bash
git add bench/gpu/README.md
git commit -m "docs(bench/gpu): phase0 gate report — go/no-go (webgpu phase0 T12)"
```

---

## Deferred to follow-up plans (explicitly NOT here)

- **Phase 1 plan** (dE gather kernel, `'gpu'` driver, T1–T3 shipping gates,
  G2 through the production kernel): written after Task 12 reports green.
- **Phase 2 plan** (G5 experiment + chosen solve variant + GPU line search +
  τ-agreement): CANNOT be written before G0t/G1 numbers exist — G5's
  estimator consumes them (spec §4).
- **Phase 3 plan** (Curve.tsx rewrite, §2.8 authority flip, G7): after
  Phase 2.
