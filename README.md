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
