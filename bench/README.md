# Sobolev solver bench

Per-phase wall-clock harness for the constrained fractional Sobolev descent step
(`sobolevStepSet`). It is the measure-before/after gate for the solver-perf
milestone and the permanent "what improved with each iteration" ledger.

See `docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md` (Task 2) and
`local_files/2026-07-03-next-steps-briefing.md` §1 for context.

## Usage

```sh
bun run bench                                        # print the markdown table
bun bench/sobolev.bench.ts --save baseline           # + write bench/results/<date>-baseline.json
bun bench/sobolev.bench.ts --baseline bench/results/<date>-baseline.json --save e0-reuse
bun bench/sobolev.bench.ts --big                     # add the N=240 cases
```

- `--save <label>` writes `bench/results/<YYYY-MM-DD>-<label>.json`.
- `--baseline <path>` adds a `Δ%` column to every printed number, matched by case
  name (`N<nV>-<constraintMode>`; frozen-projection cases append `-frozen` —
  reassemble names stay unsuffixed so pre-Task-6 baselines keep joining).

## Cases

A parametric closed **trefoil** generated in-file
(`p(t) = (sin t + 2 sin 2t, cos t − 2 cos 2t, −sin 3t)`, `t = 2πi/N`, edges
`[i, (i+1) mod N]`) at **N = 60 and N = 120** (`--big` adds 240), each under two
constraint sets:

- `total` — `[barycenter, totalLength]`
- `perEdge` — `[barycenter, edgeLengths]`

and two projection strategies (`projectionMode`, solver-perf Task 6):
`reassemble` (per-iterate rebuild) and `frozen` (one K(γ₀) LU per step —
reference-impl reuse).

Targets are frozen from the initial geometry, mirroring the store's
frozen-target lifecycle (spec §3.5).

## Methodology

- **Full step:** 2 warmup steps (discarded), then the **median of 5** measured
  `sobolevStepSet` calls, each from the same initial vertices (the step is pure
  and deterministic, so every repeat does identical work). Per-phase medians come
  from the opt-in phase-timing collector (`collectTimings: true`).
- **Isolated primitives:** **median of 7** (after 2 warmups) for
  `calculateEnergy`, `gradientAnalytical`, `assembleA`, `expandBlockDiag`,
  `solveSaddle` on the same geometry.
- Phase keys are the schema of `SobolevPhaseKey` (`src/core/sobolev/phaseTimings.ts`).
  Sub-phases (`bHigh`/`bLow` under `assembleA`; later `factor` under `saddle`)
  **overlap** their parents — sums across keys double-count by design. Compare
  like-to-like across runs; never rename a key (it breaks ledger comparability).

## Noise & ledger convention

This is a noisy VM (VirtualBox): expect **±5%** run-to-run (briefing §1). Treat
`|Δ| < 10%` as noise unless reproduced. `bench/results/*.json` are **committed** —
one file per measured iteration, keyed by the short git SHA it was run at — so the
before/after story of each perf commit is auditable.
