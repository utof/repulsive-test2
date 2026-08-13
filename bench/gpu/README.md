# bench/gpu — GPU gate harness

CDP-driven headless-Chrome harness for the WebGPU solver Phase 0 kill gates.
@see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §3–§5
@see docs/superpowers/plans/2026-08-13-webgpu-solver-phase0.md

## Winning launch recipe (G0a)

Verified 2026-08-13 against the real NVIDIA Quadro RTX 3000 in this machine
(`git a1f6022`). **Headless succeeded on the first flag set** — no headed
fallback was needed:

```
google-chrome \
  --headless=new \
  --enable-unsafe-webgpu \
  --enable-features=Vulkan \
  --use-angle=vulkan \
  --enable-gpu \
  --no-sandbox \
  --remote-debugging-port=9223 \
  --user-data-dir=<fresh temp dir> \
  http://localhost:3000/bench/gpu/harness.html
```

Result: `vendor: "nvidia"`, `architecture: "turing"` (Quadro RTX 3000 is a
Turing-class part) — a genuine hardware adapter, not SwiftShader/llvmpipe.
`bench/gpu/drive.ts` tries `FLAG_SETS` in this order automatically and
records whichever one first classifies as `hardware`; the raw result lives at
`bench/results/2026-08-13-gpu-g0a.json`.

A **fresh `--user-data-dir` per Chrome launch is required** — reusing the
default profile can make `google-chrome` hand off to an already-running
instance and silently ignore the new process's flags (including
`--remote-debugging-port`), which looks like a hang, not a flag failure.

## INVALID vs FAIL semantics

- **FAIL** — a gate ran against a confirmed hardware adapter and the
  measurement itself did not meet its threshold (e.g. G1's batching ratio,
  G3's CV). This is real, recordable evidence.
- **INVALID** — a gate ran against a software adapter (SwiftShader, llvmpipe,
  or anything `classifyAdapter()` in `drive.ts` calls `'software'`). Software
  rasterizers do not exercise real GPU dispatch/timing/precision behavior, so
  *any* number from such a run is worthless for a gate decision — per spec §3
  it must be **rerun with the recipe above**, never recorded as a gate
  result. `drive.ts` marks non-G0a runs `"status": "INVALID"` automatically
  when the adapter it detects is software; it never fabricates a PASS/FAIL
  for those.
- G0a itself is special: it *is* the hardware-adapter check. If no flag set
  (headless or headed) yields hardware, that is a **STOP-BRANCH**
  (`"status": "FAIL"` with full per-flag-set `attempts` diagnostics) — the
  rest of the browser-dependent gates cannot run honestly. See plan Task 1
  Step 3.

## Running a spike

Dev server must be up: `bun run dev` (serves `bench/gpu/harness.html` and
bundles `bench/gpu/spikes.ts` on the fly).

```
bun bench/gpu/drive.ts adapterInfo --out g0a   # G0a — must be run first
bun bench/gpu/drive.ts <spike> --out <label>   # any later spike, once G0a has PASSed
```

- `adapterInfo` (gate `G0a`) tries every entry in `FLAG_SETS` until one
  classifies as hardware, recording every attempt.
- Every other spike reads the flags from the most recent **PASSing**
  `bench/results/*-gpu-g0a*.json` and launches Chrome once with those flags —
  it does not re-search `FLAG_SETS`. If no passing G0a result file exists,
  the driver refuses to run and exits non-zero.
- Results are written to `bench/results/<date>-gpu-<label>.json`:
  `{ gate, status, adapter, flags, attempts, gitShaShort, date, data, consoleLines }`.

## Console noise to ignore

These lines can appear even on a genuine hardware adapter and are not
evidence of software fallback; `drive.ts`'s `CONSOLE_NOISE` filters them out
of `consoleLines` automatically:

- `RangeError: createBuffer failed, size (144) is too large` (~60/s from
  three's fat-lines)
- `Instance dropped in popErrorScope`
- Duplicate-key `0` React warning (`Warning: Encountered two children with
  the same key, \`0\`...`)

## Phase 0 gate report

Filled by Task 12 once every gate below has a committed result.

| Gate | Result | Number | Consequence |
|---|---|---|---|
| G0a | | | |
| G0t | | | |
| G1 | | | |
| G2 (spike) | | | |
| G3 | | | |
| G4 | | | |
| G6 | | | |
| Baselines | | | |
