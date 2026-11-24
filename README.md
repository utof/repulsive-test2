# repulsive-test2

To install dependencies:

```bash
bun install
```

To run:

```bash
bun run 
```

This project was created using `bun init` in bun v1.2.19. [Bun](https://bun.com) is a fast all-in-one JavaScript runtime.

## Penalties (soft constraints)

The simulation's main job is pushing the curve **away from itself** so it doesn't
cross or clump (the big `energy:` number in the HUD). **Penalties** are extra *soft
preferences* layered on top: gentle springs that pull the curve toward a shape you
want. They don't force anything — they tip the balance, and the optimizer juggles
"don't self-intersect" against whichever preferences you've switched on. Unlike the
hard length/barycenter constraints, penalties enter the **objective** only; the
fractional Sobolev inner product is unchanged.

Each control's number is a **weight** — how hard that spring pulls. `0` unplugs the
spring; bigger pulls harder. The weights have **no universal scale** (the paper gives
no values — they're free knobs), so tune by eye: start small, watch, adjust.

| Control | What it prefers | Notes |
|---|---|---|
| **Length w** | Make the whole curve **shorter** (shrink total length). | Fights the self-repulsion, which wants to spread out. |
| **Diff w** | Give neighboring edges the **same length** → evenly-spaced points (beads on a string). | Only affects "interior" points with *exactly* 2 edges; endpoints and Y-junctions are ignored. |
| **Field w** | Make edges point **parallel to an arrow** you choose. | Does nothing unless > 0. |
| **Field X** | The arrow's **direction** (x, y, z) for Field w. | Only the direction matters — it's normalized, so `(52,111,0)` ≡ `(0.5,1,0)`. `‖X‖ ≈ 0` disables the field. |
| **Grow L** | *Separate mechanism* — slowly grow/shrink the curve's length **target** over time. | See below. |

**Reading a weight value:** `0` = off. `1` = on, but often *barely visible* — a weight of
1 is a small voice against the large repulsion energy. `10` = strong enough to visibly
reshape the curve. Cranked high: **Length w** contracts/collapses the curve; **Diff w**
snaps points to equal spacing; **Field w** (with an arrow set) rotates edges to line up
along it. Weights **stack** — set two and both springs pull at once.

**Grow L** is a different kind of knob. The weights above are soft *preferences*; Grow L
instead animates the **hard length target** the constraint enforces after each accepted
step. `1.0` = frozen (nothing happens, the default). Below `1` shrinks the target, above
`1` grows it; it's clamped to `[0.9, 1.1]` so a single step is always gentle (though the
factor *compounds* across steps). It only bites when a length constraint is active.

Degenerate edges (length `< 1e-14`) contribute zero to every penalty. All-zero configs
are inert — the descent runs bit-identical to a penalty-free build.

See `docs/superpowers/plans/2026-07-03-sobolev-penalties.md` (§2 derivations, §4 Task 5
store/UI) and `SelfAvoiding.tex` lines 762–767 for the paper's penalty catalog.

## Tests

Two tiers. Run them with `bun run …`, not bare `bun test` (that is the serial debug mode below).

| Command | What runs | How |
|---|---|---|
| `bun run test` | **default tier**: everything except the tests marked full-tier | one `bun test` process per file, 4 at a time (`--jobs N`), longest first |
| `bun run test:full` | **full tier**: every test (`TEST_TIER=full`) | same runner; **fails if any test was skipped** |
| `bun run test:serial` | plain `bun test`: all files in one process, default tier (`TEST_TIER=full bun test` for all) | for debugging; the GMRES-bound hierarchical files run x4-6 slower in a shared process |

The runner (`scripts/test.ts`) fails if any file fails, errors or fails to load, prints the output
of the failing files, then a per-file time table, and in the default tier the number of tests
**skipped as full-tier**. A subset: `bun scripts/test.ts schur nullspace` (path substrings);
`--` passes the rest to every `bun test` (e.g. `-- -t 'n=240'`).

**What only the full tier checks** (each site says so in a comment, with what the default tier keeps):
the n = 960 cells of `schur` and `nullspace` (the 1e-10 bar at the floor of the production range,
against a refined dense truth), the `schur` tol sweep behind `DEFAULT_SCHUR_TOL`, the θ = 0 streaming
cross-check on four of six constraint sets, `hierarchicalSaddleReal`'s row-8b cells at n = 960
(n = 240 runs by default), the `precond` κ ladder's n = 960 rung and its in-band mass-weight sweep,
`apply`'s n = 2000 `delta_compress` row, the CR gate on `trefoil960` (± pin) and the writer of
`bench/results/gpu-phase2a-cr.json`, and three wall-clock instruments (`laplacian` O(N), the
`precond` / `lowOrder` cost logs).

**The full tier is mandatory before merging** any change under `src/core/sobolev/` (the hierarchical
stack, `linsolve`, the constraint and line-search code), `src/gpu/saddle*`, `src/core/dispatch.ts`,
`src/core/optimizer.ts`, or to any tolerance or default constant; and before a release. A full run
rewrites `bench/results/gpu-phase2a-cr.json`: commit it if its numbers changed, otherwise
`git checkout -- bench/results/gpu-phase2a-cr.json` (the rest is formatter churn, issue #47).

**Marking a test full-tier**: `fullTest` / `fullDescribe` from `test/tier.ts`, never by deleting it
or loosening it, with a comment saying what the default tier still checks and what waits. bun runs a
skipped describe's *body*, so heavy work there goes in a `lazy()` getter (also in `test/tier.ts`)
that the tests call.
