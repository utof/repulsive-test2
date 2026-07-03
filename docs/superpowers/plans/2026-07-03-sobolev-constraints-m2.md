# Sobolev Constraints M2 (per-edge length + point pins) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement milestone M2 of `docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md` — per-edge length constraints (`edgeLengthsBlock`, |E| rows) and point-pin constraints (`pointBlock`, 3 rows/pin) for the constrained Sobolev descent, with the §9a per-constraint toggles: the M1 "Fix length" checkbox becomes a 3-way `none | total | per-edge` Length select, the Barycenter checkbox stays, point constraints ship machinery + tests only (no picking UI).

**Architecture:** The M1 `ConstraintSet` abstraction (`src/core/sobolev/constraintSet.ts`) is already generic end-to-end (gradient solve, projection, line search, `sobolevStepSet`), so M2 core is only two new block builders + one helper in `constraintSet.ts` — `gradient.ts`, `lineSearch.ts`, `optimizer.ts` are untouched. Verification mirrors M1: the M1 constraints oracle (`oracle/tpe_constraints_oracle.py`) grows `edgelengths`/`point` modes emitting 4 new self-checked goldens; TS tests diff against them plus oracle-independent flow properties. Store gains `lengthMode: 'none'|'total'|'perEdge'` (source of truth) with the M1 `lengthConstraint` boolean kept as a write-through mirror so **zero existing test files are edited** (§4.5 gate via §5.5), plus a frozen `sobolevEll0` vector with the exact `sobolevX0`/`sobolevL0` lifecycle (§3.5).

**Tech Stack:** TypeScript (strict) + bun test; Python (numpy/scipy via `uv run`) for the oracle; zustand store; React 19 + R3F for UI. Formatting via biome (lefthook pre-commit reformats staged files, incl. `.json` goldens).

**Branch/commit discipline (overrides the per-task commit steps of the generic skill):** stay on `feat/sobolev-stage1`. ONE feat commit for the whole milestone, message starting `feat(sobolev): per-edge length + point constraints (constraints M2)`, iterated with `--amend` (CLAUDE.md one-commit-per-plan rule). The plan doc itself gets its own `docs:` commit. Do NOT push. No task commits before Task 6.

**Executor map (user directive 2026-07-03):** math tasks (Tasks 1–3: constraint rows, Jacobians, oracle, golden/flow tests) — orchestrator implements inline, no subagent, no review pass. Code tasks (Task 4 store, Task 5 UI) — Opus subagent(s), followed by an Opus spec+code review. Task 6 (verification, boot check, commit, report) — orchestrator, no subagent.

---

## Ground-truth math (derived from spec §2 — verify against these, not vibes)

With ℓ_I = ‖γ_{i2} − γ_{i1}‖ RAW (no +ε), T_I = e_I/ℓ_I (T_I = 0 if ℓ_I < 1e-14), columns via `blockIndex(coord, vertex, n)`:

- **Edge length, row I:** Φ_I = ℓ⁰_I − ℓ_I. Since dℓ_I = T_I·(dγ_{i2} − dγ_{i1}) and dΦ_I = −dℓ_I: row I has `+T_I` in i1's columns, `−T_I` in i2's columns, **that edge only**. The totalLength row is EXACTLY the sum of these rows (same accumulation order per column ⇒ bit-identical) — the §3.4 rank rule; Task 2 tests this identity directly.
- **Point, vertex i:** Φ = γ_i − x_i (paper-verbatim sign: CURRENT − target, opposite to the lengths' target−current — keep it, spec §2 "keep the paper's signs"). C rows: `C[r][blockIndex(r, i, n)] = 1`, r ∈ {0,1,2}; no length terms; Jacobian independent of target.
- **Scales (§3.3, OUR invention — flag in TSDoc):** edgeLengths → `max(1, L)` (L = Σℓ raw); point → `max(1, R)`, R = max distance from ANY vertex to the pin target.
- **Rank (§3.4):** totalLength+edgeLengths in one set throws at construction (already implemented in `assertValidConstraintSet` via `kind`; M2 tests it with REAL blocks). Pin + per-edge on its own edges: generically independent, must solve.
- **Never-throw backstops (frame loop):** mismatched `ell0` length ⇒ NaN Φ rows (`ell0[r] ?? NaN`); out-of-range pin index ⇒ NaN Φ + zero C rows. NaN Φ ⇒ projection never converges ⇒ existing `projection_failed` rejection. Same philosophy as the M1 dispatch NaN backstop in `store.ts`.

## File structure

| File | Change |
|---|---|
| `oracle/tpe_constraints_oracle.py` | +`edge_lengths`, `edge_lengths_phi_and_C`, `point_phi_and_C`, `edge_lengths_block`, `point_block`; `main()` gains `edgelengths`/`point` modes. Length-mode output must stay **semantically identical** (verified). |
| `oracle/golden/{crossing,junction-y}-edgelengths.json`, `oracle/golden/{crossing,linked-rings}-point.json` | NEW goldens (self-checked at generation). Existing goldens untouched. |
| `oracle/README.md` | file-list + regen commands + status for M2. |
| `src/core/sobolev/constraintSet.ts` | +`edgeLengths()`, `edgeLengthsBlock()`, `pointBlock()`. Nothing else in core changes. |
| `src/store.ts` | +`LengthMode` type, `lengthMode` field (+setter), `sobolevEll0` frozen vector, `lengthConstraint` → write-through mirror; `dispatchDescentStep` gains `lengthMode?`/`sobolevEll0?`. |
| `src/scene/Viewer.tsx` | dispatch passes `lengthMode` + `sobolevEll0` instead of `lengthConstraint`. |
| `src/ui/ControlPanel.tsx` | "Fix length" checkbox → 3-way Length select; Barycenter checkbox stays. |
| `test/sobolev/constraintSetM2.test.ts` | NEW: FD Jacobians, exact Φ/C structure, sum-identity, degenerate guard, rank rule, NaN backstops. |
| `test/sobolev/constraintSetM2Flow.test.ts` | NEW: golden diffs (4 goldens), per-edge flow, pin flow, pin+edges rank-feasibility flow. |
| `test/store-constraints-m2.test.ts` | NEW: lengthMode defaults/mirror, ℓ⁰ lifecycle, dispatch combos. |

Impl-file budget check (spec §7, M2 ≤ ~6): constraintSet.ts, store.ts, Viewer.tsx, ControlPanel.tsx = 4. ✓

**Existing exported signatures must not change** (spec §3.2). `dispatchDescentStep`'s args object gains optional fields only (same prescribed-extension pattern as M1). **Zero edits to existing test files** (§4.5 via §5.5) — the `lengthConstraint` mirror exists precisely to honor this; `test/store-constraints.test.ts` and all M1/stage-1 tests must pass unmodified.

**Anchor-comment rule (CLAUDE.md + project memory):** every `@see` cites a resolvable path + section (e.g. `docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §3.3`), never "Task N".

---

### Task 1: Oracle extension + 4 goldens + README [MATH — orchestrator inline]

**Files:**
- Modify: `oracle/tpe_constraints_oracle.py`
- Create: `oracle/golden/crossing-edgelengths.json`, `oracle/golden/junction-y-edgelengths.json`, `oracle/golden/crossing-point.json`, `oracle/golden/linked-rings-point.json`
- Modify: `oracle/README.md`

- [x] **Step 1.1: Extend the oracle script.** Update the module docstring (M2 modes; usage line `mode: "length" (default) | "edgelengths" | "point"`). Add after `total_length_block`:

```python
def edge_lengths(vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Raw per-edge lengths ell_I = ||e_I|| in edge order, NO +eps.

    Constraints are geometric — same raw-length rule as total_length /
    barycenter_phi_and_C (spec §2)."""
    return np.array(
        [safe_norm(vertices[int(i2)] - vertices[int(i1)]) for i1, i2 in edges],
        dtype=float,
    )


def edge_lengths_phi_and_C(
    vertices: np.ndarray, edges: np.ndarray, ell0: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Phi_I = ell0_I - ell_I (paper sign, target minus current), |E| rows.

    Row I: +T_I at i1's columns, -T_I at i2's, THAT EDGE ONLY (spec §2:
    dPhi_I = -dell_I, dell_I = T_I . (dgamma_i2 - dgamma_i1)). The totalLength
    row is exactly the SUM of these rows — the §3.4 rank rule. Degenerate
    edges contribute T = 0 (safe_unit's 1e-14 guard)."""
    n = vertices.shape[0]
    m = edges.shape[0]
    phi = np.zeros(m, dtype=float)
    C = np.zeros((m, 3 * n), dtype=float)
    for r, (i1, i2) in enumerate(edges):
        i1 = int(i1)
        i2 = int(i2)
        ev = vertices[i2] - vertices[i1]
        phi[r] = float(ell0[r]) - safe_norm(ev)
        T = safe_unit(ev)
        for c in range(3):
            C[r, block_index(c, i1, n)] += T[c]
            C[r, block_index(c, i2, n)] += -T[c]
    return phi, C


def point_phi_and_C(
    vertices: np.ndarray, vertex_index: int, target: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Phi = gamma_i - x_i (paper-verbatim sign: CURRENT minus target, spec §2),
    3 rows; C is the identity block C[r, block_index(r, i, n)] = 1 — no
    length terms."""
    n = vertices.shape[0]
    C = np.zeros((3, 3 * n), dtype=float)
    for r in range(3):
        C[r, block_index(r, vertex_index, n)] = 1.0
    return vertices[vertex_index] - target, C


def edge_lengths_block(ell0: np.ndarray) -> Block:
    # scale max(1, L) is OUR tunable choice, not paper-sourced (spec §3.3;
    # formula-audit item 9).
    return {
        "kind": "edgeLengths",
        "evaluate": lambda V, E: edge_lengths_phi_and_C(V, E, ell0),
        "scale": lambda V, E: max(1.0, total_length(V, E)),
    }


def point_block(vertex_index: int, target: np.ndarray) -> Block:
    # scale max(1, R), R = max distance from ANY vertex to the pin target —
    # OUR tunable choice, not paper-sourced (spec §3.3).
    t = np.asarray(target, dtype=float)
    return {
        "kind": "point",
        "evaluate": lambda V, E: point_phi_and_C(V, vertex_index, t),
        "scale": lambda V, E: max(1.0, float(max(safe_norm(v - t) for v in V))),
    }
```

Rework `main()`: replace the mode gate + fixed `blocks` with (barycenter FIRST in every mode — spec §3.2; targets frozen from the INITIAL geometry — spec §3.5):

```python
    mode = argv[3] if len(argv) == 4 else "length"
    if mode not in ("length", "edgelengths", "point"):
        print(
            f"unknown mode {mode!r}; expected length | edgelengths | point",
            file=sys.stderr,
        )
        return 2
```

and after `x0 = ...; L0 = total_length(vertices, edges)`:

```python
    extra_targets: Dict[str, Any] = {}
    if mode == "length":
        blocks = [barycenter_block(x0), total_length_block(L0)]
        row_order = "barycenter rows 0..2, totalLength row 3"
        extra_targets["L0_total_length_target"] = float(L0)
    elif mode == "edgelengths":
        ell0 = edge_lengths(vertices, edges)
        blocks = [barycenter_block(x0), edge_lengths_block(ell0)]
        row_order = "barycenter rows 0..2, edgeLengths rows 3..2+|E| (edge order)"
        extra_targets["ell0_edge_length_targets"] = [float(x) for x in ell0]
    else:  # point
        pin_index = 0
        pin_target = vertices[pin_index].copy()
        blocks = [barycenter_block(x0), point_block(pin_index, pin_target)]
        row_order = "barycenter rows 0..2, point rows 3..5 (pinned vertex x,y,z)"
        extra_targets["pin_vertex_index"] = pin_index
        extra_targets["pin_target"] = [float(x) for x in pin_target]
```

In the property checks, make the post-step drift check mode-specific (keep the length one verbatim):

```python
        if mode == "length":
            L1 = total_length(newV, edges)
            drift = abs(L1 - L0) / max(1.0, L0)
            check("length drift |L-L0|/max(1,L0) <= 1e-8 after step", drift <= 1.0e-8, f"drift = {drift:.3e}")
        elif mode == "edgelengths":
            ell1 = edge_lengths(newV, edges)
            ell0_arr = np.asarray(extra_targets["ell0_edge_length_targets"], dtype=float)
            worst = float(np.max(np.abs(ell1 - ell0_arr) / ell0_arr))
            check(
                "per-edge drift max_I |l_I - l0_I|/l0_I <= 1e-8 after step",
                worst <= 1.0e-8,
                f"max drift = {worst:.3e}",
            )
        else:  # point
            pin = newV[int(extra_targets["pin_vertex_index"])]
            dist = safe_norm(pin - np.asarray(extra_targets["pin_target"], dtype=float))
            check(
                "pin distance ||gamma_pin - target|| <= 1e-8 after step",
                dist <= 1.0e-8,
                f"dist = {dist:.3e}",
            )
```

In `out`: `"constraint_set": [b["kind"] for b in blocks]` and `"row_order": row_order` in the conventions dict (derived values equal the current hardcoded strings for length mode); replace the top-level `"L0_total_length_target": float(L0),` with `**extra_targets,` at the end of the dict literal. Everything else stays byte-for-byte.

- [x] **Step 1.2: Prove length-mode output is unchanged.** Regenerate the crossing length golden to the scratchpad and semantically compare with the committed one:

```bash
uv run --with numpy --with scipy python oracle/tpe_constraints_oracle.py \
  oracle/fixtures/crossing.json <scratchpad>/crossing-length-check.json
python3 -c "import json,sys; a=json.load(open('oracle/golden/crossing-length.json')); b=json.load(open('<scratchpad>/crossing-length-check.json')); sys.exit(0 if a==b else 1)" \
  && echo LENGTH_MODE_IDENTICAL
```

Expected: all property checks PASS, `LENGTH_MODE_IDENTICAL`. (Byte diff is expected to differ — committed goldens are biome-reformatted at commit time; semantic equality is the gate.)

- [x] **Step 1.3: Generate the 4 M2 goldens.**

```bash
for f in crossing junction-y; do
  uv run --with numpy --with scipy python oracle/tpe_constraints_oracle.py \
    oracle/fixtures/$f.json oracle/golden/$f-edgelengths.json edgelengths
done
for f in crossing linked-rings; do
  uv run --with numpy --with scipy python oracle/tpe_constraints_oracle.py \
    oracle/fixtures/$f.json oracle/golden/$f-point.json point
done
```

Expected per run: `all property checks passed` (exit 0) — FD Jacobian ≤1e-6 rel, residual ≤1e-10, descent positivity, C·g̃ compatibility, line search accepted, energy decrease, mode drift ≤1e-8, per-block Φ tolerance. If a fixture fails "line search accepted" (possible on junction-y), swap that fixture for `helix` and note it in the report — do NOT loosen gates.

- [x] **Step 1.4: README.** In `oracle/README.md`: extend the `tpe_constraints_oracle.py` bullet with the M2 modes + new golden names; add the two regen loops above to the "Constraints goldens" block; add a status line `Status (2026-07-03, constraints M2): ...` naming the 4 goldens and that all embedded checks passed. Existing text untouched.

### Task 2: TS constraint blocks + unit/FD tests (TDD) [MATH — orchestrator inline]

**Files:**
- Test: `test/sobolev/constraintSetM2.test.ts` (new)
- Modify: `src/core/sobolev/constraintSet.ts`

- [x] **Step 2.1: Write the failing test file** `test/sobolev/constraintSetM2.test.ts`:

```ts
import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import {
    assertValidConstraintSet,
    barycenterBlock,
    type ConstraintSet,
    edgeLengths,
    edgeLengthsBlock,
    evaluateConstraintSet,
    pointBlock,
    totalLength,
    totalLengthBlock,
} from '../../src/core/sobolev/constraintSet';
import { barycenterTarget } from '../../src/core/sobolev/constraints';
import { blockIndex, unflatten } from '../../src/core/sobolev/layout';
import type { Edge, Vec3 } from '../../src/core/testConfigs';

interface Fixture {
    name: string;
    vertices: Vec3[];
    edges: Edge[];
    alpha: number;
    beta: number;
    epsilon: number;
}

// Load at runtime (avoids needing resolveJsonModule in tsconfig; test/** is
// typechecked), mirroring test/sobolev/constraintSet.test.ts.
function loadFixture(name: string): Fixture {
    return JSON.parse(
        readFileSync(new URL(`../../oracle/fixtures/${name}.json`, import.meta.url), 'utf8'),
    ) as Fixture;
}

function euclideanNorm(a: number[]): number {
    let sumSq = 0;
    for (const x of a) sumSq += x * x;
    return Math.sqrt(sumSq);
}

function euclideanDiff(a: number[], b: number[]): number {
    let sumSq = 0;
    for (let i = 0; i < a.length; i++) {
        const d = a[i] - b[i];
        sumSq += d * d;
    }
    return Math.sqrt(sumSq);
}

// Central-difference FD check of a stacked set's Jacobian on a deterministic
// direction — the §4.4.1/§5.4.1 pattern of constraintSet.test.ts; gate 1e-6
// rel (see oracle/README.md "Known tolerance caveats" for why not tighter).
function fdJacobianRel(set: ConstraintSet, vertices: Vec3[], edges: Edge[]): number {
    const n = vertices.length;
    const eta = 1e-6;
    const h = Array.from({ length: 3 * n }, (_, k) => Math.sin(0.11 + 0.37 * k));
    const offsetsPlus = unflatten(h.map((x) => eta * x));
    const offsetsMinus = unflatten(h.map((x) => -eta * x));
    const plus: Vec3[] = vertices.map((v, i) => [
        v[0] + offsetsPlus[i][0],
        v[1] + offsetsPlus[i][1],
        v[2] + offsetsPlus[i][2],
    ]);
    const minus: Vec3[] = vertices.map((v, i) => [
        v[0] + offsetsMinus[i][0],
        v[1] + offsetsMinus[i][1],
        v[2] + offsetsMinus[i][2],
    ]);
    const { C } = evaluateConstraintSet(set, vertices, edges);
    const { phi: phiPlus } = evaluateConstraintSet(set, plus, edges);
    const { phi: phiMinus } = evaluateConstraintSet(set, minus, edges);
    const fd = phiPlus.map((p, r) => (p - phiMinus[r]) / (2 * eta));
    const Ch = C.map((row) => row.reduce((s, x, k) => s + x * h[k], 0));
    return euclideanDiff(fd, Ch) / Math.max(1, euclideanNorm(Ch));
}

// Edge-length rows on a junction fixture too (spec §5.4.1) — junction-y
// exercises shared-vertex accumulation across a degree-3 vertex.
for (const name of ['crossing', 'junction-y'] as const) {
    test(`edgeLengthsBlock: stacked [barycenter, edgeLengths] FD Jacobian on ${name} ≤ 1e-6 rel (spec §5.4.1)`, () => {
        const { vertices, edges } = loadFixture(name);
        const x0 = barycenterTarget(vertices, edges);
        const ell0 = edgeLengths(vertices, edges);
        const set: ConstraintSet = [barycenterBlock(x0), edgeLengthsBlock(ell0)];
        const { phi, C } = evaluateConstraintSet(set, vertices, edges);
        expect(phi.length).toBe(3 + edges.length);
        expect(C.length).toBe(3 + edges.length);
        const rel = fdJacobianRel(set, vertices, edges);
        console.log(
            `[constraintSetM2] ${name}: [barycenter, edgeLengths] FD-Jacobian rel = ${rel.toExponential(3)}`,
        );
        expect(rel).toBeLessThanOrEqual(1e-6);
    });
}

test('pointBlock: [barycenter, point] FD Jacobian on crossing ≤ 1e-6 rel; rows are the exact identity block (spec §2)', () => {
    const { vertices, edges } = loadFixture('crossing');
    const x0 = barycenterTarget(vertices, edges);
    // Off-vertex target → non-trivial Φ; the point Jacobian is target-independent.
    const target: Vec3 = [vertices[0][0] + 0.1, vertices[0][1] - 0.2, vertices[0][2] + 0.3];
    const set: ConstraintSet = [barycenterBlock(x0), pointBlock(0, target)];
    const rel = fdJacobianRel(set, vertices, edges);
    console.log(`[constraintSetM2] crossing: [barycenter, point] FD-Jacobian rel = ${rel.toExponential(3)}`);
    expect(rel).toBeLessThanOrEqual(1e-6);

    // Exact structure on a different vertex: Φ = γ_i − x_i (paper sign:
    // CURRENT minus target), C[r] has a single 1 at blockIndex(r, i, n).
    const n = vertices.length;
    const { phi, C } = pointBlock(2, target).evaluate(vertices, edges);
    expect(phi).toEqual([
        vertices[2][0] - target[0],
        vertices[2][1] - target[1],
        vertices[2][2] - target[2],
    ]);
    expect(C.length).toBe(3);
    for (let r = 0; r < 3; r++) {
        for (let k = 0; k < 3 * n; k++) {
            expect(C[r][k]).toBe(k === blockIndex(r, 2, n) ? 1 : 0);
        }
    }
});

test('edgeLengthsBlock: Φ is exactly ℓ⁰ − ℓ per edge (zero at the anchored targets)', () => {
    const { vertices, edges } = loadFixture('crossing');
    const ell0 = edgeLengths(vertices, edges);
    const atAnchor = edgeLengthsBlock(ell0).evaluate(vertices, edges);
    expect(atAnchor.phi.length).toBe(edges.length);
    for (const v of atAnchor.phi) expect(v).toBe(0);

    const shifted = edgeLengthsBlock(ell0.map((l) => l + 0.25)).evaluate(vertices, edges);
    const now = edgeLengths(vertices, edges);
    for (let r = 0; r < edges.length; r++) {
        expect(shifted.phi[r]).toBe(ell0[r] + 0.25 - now[r]);
    }
});

// The §3.4 rank-dependence rule made falsifiable: the totalLength row is the
// SUM of the edgeLengths rows — bit-identically, since per column both sides
// perform the same additions in the same (edge) order.
test('edgeLengthsBlock: rows sum EXACTLY to the totalLength row (spec §3.4 rank-dependence)', () => {
    for (const name of ['crossing', 'junction-y'] as const) {
        const { vertices, edges } = loadFixture(name);
        const ell0 = edgeLengths(vertices, edges);
        const L0 = totalLength(vertices, edges);
        const edgeEval = edgeLengthsBlock(ell0).evaluate(vertices, edges);
        const totalEval = totalLengthBlock(L0).evaluate(vertices, edges);
        const n3 = 3 * vertices.length;
        for (let k = 0; k < n3; k++) {
            let s = 0;
            for (const row of edgeEval.C) s += row[k];
            expect(s).toBe(totalEval.C[0][k]);
        }
        // Φ sums too, but association order differs → approximate gate.
        const phiSum = edgeEval.phi.reduce((a, b) => a + b, 0);
        expect(Math.abs(phiSum - totalEval.phi[0])).toBeLessThanOrEqual(1e-12);
    }
});

test('edgeLengthsBlock: zero-length edge (degenerate guard) yields a finite all-zero row, no NaN', () => {
    // Edge (0,1) is zero-length → T = [0,0,0] (1e-14 guard); edge (1,2) has
    // length 1 along +x → T = [1,0,0]. Same setup as the M1 degenerate test.
    const degenerateVertices: Vec3[] = [
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
    ];
    const degenerateEdges: Edge[] = [
        [0, 1],
        [1, 2],
    ];
    const ell0 = edgeLengths(degenerateVertices, degenerateEdges);
    const { phi, C } = edgeLengthsBlock(ell0).evaluate(degenerateVertices, degenerateEdges);
    expect(phi[0]).toBe(0);
    expect(phi[1]).toBe(0);
    const n = degenerateVertices.length;
    for (const row of C) {
        for (const v of row) expect(Number.isFinite(v)).toBe(true);
    }
    for (const v of C[0]) expect(v).toBe(0);
    expect(C[1][blockIndex(0, 1, n)]).toBe(1);
    expect(C[1][blockIndex(0, 2, n)]).toBe(-1);
});

test('rank rule (spec §3.4): totalLength + edgeLengths (REAL blocks) throws at construction; valid M2 compositions pass', () => {
    const { vertices, edges } = loadFixture('crossing');
    const x0 = barycenterTarget(vertices, edges);
    const L0 = totalLength(vertices, edges);
    const ell0 = edgeLengths(vertices, edges);
    const pinTarget: Vec3 = [vertices[0][0], vertices[0][1], vertices[0][2]];
    expect(() =>
        assertValidConstraintSet([totalLengthBlock(L0), edgeLengthsBlock(ell0)]),
    ).toThrow(/§3\.4/);
    expect(() =>
        assertValidConstraintSet([
            barycenterBlock(x0),
            edgeLengthsBlock(ell0),
            pointBlock(0, pinTarget),
        ]),
    ).not.toThrow();
    expect(() =>
        assertValidConstraintSet([edgeLengthsBlock(ell0), pointBlock(0, pinTarget)]),
    ).not.toThrow();
});

// Never-throw backstops: malformed inputs must surface as NaN Φ (→ the
// existing projection_failed rejection), never as a frame-loop throw.
test('edgeLengthsBlock: mismatched ℓ⁰ length yields NaN Φ rows, never throws', () => {
    const { vertices, edges } = loadFixture('crossing');
    const { phi, C } = edgeLengthsBlock([1]).evaluate(vertices, edges);
    expect(phi.length).toBe(edges.length);
    expect(Number.isNaN(phi[0])).toBe(false);
    expect(Number.isNaN(phi[1])).toBe(true);
    for (const row of C) {
        for (const v of row) expect(Number.isFinite(v)).toBe(true);
    }
});

test('pointBlock: out-of-range vertexIndex yields NaN Φ and zero C rows, never throws', () => {
    const { vertices, edges } = loadFixture('crossing');
    const { phi, C } = pointBlock(999, [0, 0, 0]).evaluate(vertices, edges);
    expect(phi.length).toBe(3);
    expect(phi.every((v) => Number.isNaN(v))).toBe(true);
    for (const row of C) {
        for (const v of row) expect(v).toBe(0);
    }
});
```

- [x] **Step 2.2: Run to verify RED.** `bun test test/sobolev/constraintSetM2.test.ts` — expected failure: module load error, `edgeLengths`/`edgeLengthsBlock`/`pointBlock` not exported from `constraintSet.ts`.

- [x] **Step 2.3: Implement in `src/core/sobolev/constraintSet.ts`.** Update the module TSDoc (M2 ships all four builders). Add after `totalLengthBlock`:

```ts
/**
 * Raw per-edge geometric lengths ℓ_I = ‖γ_{i2} − γ_{i1}‖ in edge order — NO +ε
 * (constraints are geometric; same raw-length rule as {@link totalLength}, do
 * NOT "unify" with the ℓ^ε of innerProduct.ts). Shared by
 * {@link edgeLengthsBlock}'s targets and the store's frozen-ℓ⁰ lifecycle.
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("Use raw geometric lengths ... not ℓ^ε")
 * @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §2, §3.5
 */
export function edgeLengths(vertices: Vec3[], edges: Edge[]): number[] {
    return edges.map(([i1, i2]) => {
        const p1 = vertices[i1];
        const p2 = vertices[i2];
        const ex = p2[0] - p1[0];
        const ey = p2[1] - p1[1];
        const ez = p2[2] - p1[2];
        return Math.sqrt(ex * ex + ey * ey + ez * ez);
    });
}

/**
 * Per-edge length constraint block: Φ_{len,I}(γ) = ℓ⁰_I − ℓ_I ∈ R, one row per
 * edge (|E| rows, edge order), paper sign convention (target minus current).
 * Row I touches ONLY edge I's endpoints: `+T_I` at i1's columns, `−T_I` at
 * i2's (dΦ_I = −dℓ_I, dℓ_I = T_I·(dγ_{i2} − dγ_{i1})). The total-length row is
 * exactly the SUM of these rows — hence the §3.4 mutual exclusion with
 * `totalLengthBlock`, enforced at construction by {@link assertValidConstraintSet}.
 *
 * Degenerate guard: T_I = [0,0,0] when ℓ_I < 1e-14 — same guard, same constant
 * as `totalLengthBlock` / `barycenterPhiAndC` (constraints.ts). A degenerate
 * edge zeroes its row → singular saddle → the existing `singular_system`
 * rejection path is the backstop (spec §2 — never crash the frame loop).
 *
 * A mismatched `ell0` (length ≠ |E|) yields NaN Φ rows instead of throwing:
 * projection then never converges and the step is REJECTED
 * ('projection_failed') — the same never-throw backstop as the dispatch's
 * missing-L⁰ NaN in store.ts.
 *
 * Projection-tolerance scale: max(1, L), L = Σℓ_I raw — OUR tunable choice,
 * NOT paper-sourced (same flagging convention as `totalLengthBlock`).
 * @see local_files/2026-07-02-sobolev-formula-audit.md (Item 9 — "Unstated inventions")
 * @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §2, §3.3, §3.4, §5.1
 */
export function edgeLengthsBlock(ell0: number[]): ConstraintBlock {
    return {
        kind: 'edgeLengths',
        evaluate(vertices, edges) {
            const n = vertices.length;
            const m = edges.length;
            const phi = new Array<number>(m).fill(0);
            const C: number[][] = Array.from({ length: m }, () =>
                new Array<number>(3 * n).fill(0),
            );
            for (let r = 0; r < m; r++) {
                const [i1, i2] = edges[r];
                const p1 = vertices[i1];
                const p2 = vertices[i2];
                const ex = p2[0] - p1[0];
                const ey = p2[1] - p1[1];
                const ez = p2[2] - p1[2];
                // RAW geometric length, no +ε — see the edgeLengths/totalLength anchors.
                const ell = Math.sqrt(ex * ex + ey * ey + ez * ez);
                // NaN backstop for a mismatched ell0 — see the TSDoc above.
                phi[r] = (ell0[r] ?? Number.NaN) - ell;
                // Degenerate guard: T_I = 0 when ‖e_I‖ < 1e-14 (same constant as
                // barycenterPhiAndC's guard, constraints.ts).
                // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B
                let T: Vec3;
                if (ell < 1e-14) {
                    T = [0, 0, 0];
                } else {
                    const inv = 1 / ell;
                    T = [ex * inv, ey * inv, ez * inv];
                }
                for (let c = 0; c < 3; c++) {
                    C[r][blockIndex(c, i1, n)] += T[c];
                    C[r][blockIndex(c, i2, n)] += -T[c];
                }
            }
            return { phi, C };
        },
        scale(vertices, edges) {
            // OUR tunable choice, NOT paper-sourced — same flagging convention as
            // lineSearch.ts's barycenterScale.
            // @see local_files/2026-07-02-sobolev-formula-audit.md (Item 9 — "Unstated inventions")
            // @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §3.3
            return Math.max(1, totalLength(vertices, edges));
        },
    };
}

/**
 * Point (pin) constraint block: Φ_pt,i(γ) = γ_i − x_i ∈ R³ (3 rows) —
 * paper-verbatim sign (CURRENT minus target, unlike the length constraints'
 * target-minus-current; keep as-is so future audits can diff against the
 * excerpts 1:1, spec §2). Jacobian: identity block,
 * C[r][blockIndex(r, i, n)] = 1 — no length terms.
 *
 * M2 ships MACHINERY + tests only (no picking UI, spec §5.3); knip may flag
 * this export as unused from src/ — expected and non-blocking.
 *
 * An out-of-range `vertexIndex` yields NaN Φ rows (and zero C rows) instead of
 * throwing — the same never-throw projection_failed backstop as
 * `edgeLengthsBlock`'s mismatched ℓ⁰.
 *
 * Projection-tolerance scale: max(1, R), R = max distance from ANY vertex to
 * the pin target — OUR tunable choice, NOT paper-sourced.
 * @see local_files/2026-07-02-sobolev-formula-audit.md (Item 9 — "Unstated inventions")
 * @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §2, §3.3, §5.1, §5.3
 */
export function pointBlock(vertexIndex: number, target: Vec3): ConstraintBlock {
    return {
        kind: 'point',
        evaluate(vertices, _edges) {
            const n = vertices.length;
            const C: number[][] = Array.from({ length: 3 }, () =>
                new Array<number>(3 * n).fill(0),
            );
            const p = vertices[vertexIndex];
            if (p === undefined) {
                // NaN backstop (out-of-range pin) — see the TSDoc above.
                return { phi: [Number.NaN, Number.NaN, Number.NaN], C };
            }
            for (let r = 0; r < 3; r++) {
                C[r][blockIndex(r, vertexIndex, n)] = 1;
            }
            return { phi: [p[0] - target[0], p[1] - target[1], p[2] - target[2]], C };
        },
        scale(vertices, _edges) {
            // OUR tunable choice, NOT paper-sourced.
            // @see local_files/2026-07-02-sobolev-formula-audit.md (Item 9 — "Unstated inventions")
            // @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §3.3
            let R = 0;
            for (const v of vertices) {
                const dx = v[0] - target[0];
                const dy = v[1] - target[1];
                const dz = v[2] - target[2];
                R = Math.max(R, Math.sqrt(dx * dx + dy * dy + dz * dz));
            }
            return Math.max(1, R);
        },
    };
}
```

- [x] **Step 2.4: Run to verify GREEN.** `bun test test/sobolev/constraintSetM2.test.ts` → all pass. Then full `bun test` (all 142 existing + new pass) and `bunx tsc --noEmit` (clean).

### Task 3: Golden-diff + flow + rank-feasibility tests [MATH — orchestrator inline]

**Files:**
- Test: `test/sobolev/constraintSetM2Flow.test.ts` (new)
- Depends on: Task 1 goldens, Task 2 blocks.

- [x] **Step 3.1: Write** `test/sobolev/constraintSetM2Flow.test.ts`. These are verification tests against already-generated goldens (the goldens are the falsifiability, not a red phase):

```ts
import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { sobolevStepSet } from '../../src/core/optimizer';
import {
    assertValidConstraintSet,
    barycenterBlock,
    type ConstraintSet,
    edgeLengths,
    edgeLengthsBlock,
    evaluateConstraintSet,
    pointBlock,
} from '../../src/core/sobolev/constraintSet';
import { barycenterTarget } from '../../src/core/sobolev/constraints';
import { solveConstrainedGradientSet } from '../../src/core/sobolev/gradient';
import { flatten } from '../../src/core/sobolev/layout';
import { lineSearchStepSet } from '../../src/core/sobolev/lineSearch';
import { calculateDisjointPairs, calculateEnergy } from '../../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../../src/core/testConfigs';

// The four constraints-M2 golden pairs (spec §5.4.2): per-edge lengths on
// crossing + junction-y (the junction exercises shared-vertex accumulation),
// point pin on crossing + linked-rings. Generated by
// oracle/tpe_constraints_oracle.py modes 'edgelengths'/'point' — the stage-1
// and M1 goldens are never touched.
// @see oracle/README.md ("Constraints goldens")
// @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §5.2, §5.4

// Mirrored line-search tunable, used only to re-verify Armijo from RETURNED
// numbers. OUR constant, not the paper's.
// @see local_files/2026-07-02-sobolev-formula-audit.md (Item 9)
const ARMIJO_C1 = 1e-4;

interface Fixture {
    name: string;
    vertices: Vec3[];
    edges: Edge[];
    alpha: number;
    beta: number;
    epsilon: number;
}

interface GoldenStep {
    accepted: boolean;
    tau: number;
    energy_before: number;
    energy_after: number;
    slope: number;
    gradient_l2_norm: number;
    vertices: Vec3[];
    projection_iterations: number;
    projection_phi_norm: number;
}

interface GoldenM2Base {
    dE: Vec3[];
    x0_barycenter_target: Vec3;
    g_tilde: Vec3[];
    g_tilde_flat: number[];
    lambda: number[];
    line_search_step: GoldenStep;
}

interface GoldenEdgeLengths extends GoldenM2Base {
    ell0_edge_length_targets: number[];
}

interface GoldenPoint extends GoldenM2Base {
    pin_vertex_index: number;
    pin_target: Vec3;
}

// Load at runtime (avoids needing resolveJsonModule in tsconfig; test/** is
// typechecked), mirroring test/sobolev/constraintSetFlow.test.ts.
function loadFixture(name: string): Fixture {
    return JSON.parse(
        readFileSync(new URL(`../../oracle/fixtures/${name}.json`, import.meta.url), 'utf8'),
    ) as Fixture;
}

function loadGolden<T>(basename: string): T {
    return JSON.parse(
        readFileSync(new URL(`../../oracle/golden/${basename}.json`, import.meta.url), 'utf8'),
    ) as T;
}

function euclideanNorm(a: number[]): number {
    let sumSq = 0;
    for (const x of a) sumSq += x * x;
    return Math.sqrt(sumSq);
}

function euclideanDiff(a: number[], b: number[]): number {
    let sumSq = 0;
    for (let i = 0; i < a.length; i++) {
        const d = a[i] - b[i];
        sumSq += d * d;
    }
    return Math.sqrt(sumSq);
}

function matVec(M: number[][], v: number[]): number[] {
    return M.map((row) => row.reduce((s, x, j) => s + x * v[j], 0));
}

function dot(a: number[], b: number[]): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
}

// DESIGN DECISION (preserve): inputs are the oracle's own outputs (golden.dE,
// golden targets), NOT TS-side recomputations — decouples the solve comparison
// from cross-language FD noise. Same rationale as constraintSetFlow.test.ts.
// @see oracle/README.md ("Known tolerance caveats")
function goldenSuite(
    label: string,
    name: string,
    fixture: Fixture,
    golden: GoldenM2Base,
    set: ConstraintSet,
    expectedRows: number,
): void {
    test(`solveConstrainedGradientSet[${label}]: ${name} — matches oracle g̃/λ to 1e-9, residual ≤ 1e-10, descent, C·g̃ ≈ 0`, () => {
        const { vertices, edges, alpha, beta, epsilon } = fixture;
        const disjointPairs = calculateDisjointPairs(edges);

        const { gTilde, lambda, residual } = solveConstrainedGradientSet(
            vertices,
            edges,
            disjointPairs,
            alpha,
            beta,
            epsilon,
            golden.dE,
            set,
        );
        const gFlat = flatten(gTilde);

        const gRelDiff =
            euclideanDiff(gFlat, golden.g_tilde_flat) / euclideanNorm(golden.g_tilde_flat);
        const lambdaRelDiff =
            euclideanDiff(lambda, golden.lambda) / Math.max(1, euclideanNorm(golden.lambda));
        const descentDot = dot(flatten(golden.dE), gFlat);
        const { C } = evaluateConstraintSet(set, vertices, edges);
        const constraintRel = euclideanNorm(matVec(C, gFlat)) / Math.max(1, euclideanNorm(gFlat));

        console.log(
            `[gradientM2:${label}] ${name} (|V| = ${vertices.length}): g̃ rel diff = ${gRelDiff.toExponential(3)}, ` +
                `λ rel diff = ${lambdaRelDiff.toExponential(3)} (k = ${lambda.length}), ` +
                `residual = ${residual.toExponential(3)}, ‖C·g̃‖/max(1,‖g̃‖) = ${constraintRel.toExponential(3)}`,
        );

        expect(lambda.length).toBe(expectedRows);
        expect(gRelDiff).toBeLessThanOrEqual(1e-9);
        expect(lambdaRelDiff).toBeLessThanOrEqual(1e-9);
        expect(residual).toBeLessThanOrEqual(1e-10);
        expect(descentDot).toBeGreaterThan(0);
        expect(constraintRel).toBeLessThanOrEqual(1e-10);
    });

    test(`lineSearchStepSet[${label}]: ${name} — matches oracle acceptance, τ, iterations, energy, vertices`, () => {
        const { vertices, edges, alpha, beta, epsilon } = fixture;
        const disjointPairs = calculateDisjointPairs(edges);
        const gold = golden.line_search_step;

        const result = lineSearchStepSet(
            vertices,
            edges,
            disjointPairs,
            alpha,
            beta,
            epsilon,
            golden.dE,
            golden.g_tilde,
            set,
        );

        const energyRelDiff =
            Math.abs(result.energyAfter - gold.energy_after) / Math.abs(gold.energy_after);
        const vertexRelDiff =
            euclideanDiff(flatten(result.vertices), flatten(gold.vertices)) /
            euclideanNorm(flatten(gold.vertices));

        console.log(
            `[lineSearchM2:${label}] ${name}: accepted = ${result.accepted}, τ = ${result.tau}, ` +
                `projection iterations = ${result.projectionIterations}, ` +
                `E ${result.energyBefore.toExponential(6)} → ${result.energyAfter.toExponential(6)} ` +
                `(rel diff vs oracle = ${energyRelDiff.toExponential(3)}), ` +
                `vertices rel diff = ${vertexRelDiff.toExponential(3)}`,
        );

        expect(gold.accepted).toBe(true);
        expect(result.accepted).toBe(gold.accepted);
        // τ EXACT (powers of two — bit-identical across languages; drift means
        // the accept/reject LOGIC diverged). Same gate as the M1 flow tests.
        expect(result.tau).toBe(gold.tau);
        expect(result.projectionIterations).toBe(gold.projection_iterations);
        expect(energyRelDiff).toBeLessThanOrEqual(1e-12);
        expect(vertexRelDiff).toBeLessThanOrEqual(1e-9);

        // Armijo re-verified from RETURNED numbers — spec §C step 9.
        expect(result.slope).toBeDefined();
        expect(result.energyAfter).toBeLessThanOrEqual(
            result.energyBefore - ARMIJO_C1 * result.tau * (result.slope as number),
        );
    });
}

for (const name of ['crossing', 'junction-y'] as const) {
    const fixture = loadFixture(name);
    const golden = loadGolden<GoldenEdgeLengths>(`${name}-edgelengths`);
    goldenSuite(
        'edgeLengths',
        name,
        fixture,
        golden,
        // Barycenter FIRST — the row-order rule of spec §3.2.
        [
            barycenterBlock(golden.x0_barycenter_target),
            edgeLengthsBlock(golden.ell0_edge_length_targets),
        ],
        3 + fixture.edges.length,
    );
}

for (const name of ['crossing', 'linked-rings'] as const) {
    const fixture = loadFixture(name);
    const golden = loadGolden<GoldenPoint>(`${name}-point`);
    goldenSuite(
        'point',
        name,
        fixture,
        golden,
        [
            barycenterBlock(golden.x0_barycenter_target),
            pointBlock(golden.pin_vertex_index, golden.pin_target),
        ],
        6,
    );
}

function maxEdgeDrift(current: Vec3[], edges: Edge[], ell0: number[]): number {
    const now = edgeLengths(current, edges);
    let worst = 0;
    for (let i = 0; i < now.length; i++) {
        worst = Math.max(worst, Math.abs(now[i] - ell0[i]) / ell0[i]);
    }
    return worst;
}

function distTo(p: Vec3, q: Vec3): number {
    return Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
}

// Flow property (spec §5.4.3, oracle-independent): per-edge mode on crossing —
// "isometric untangling": every edge's |ℓ_I − ℓ⁰_I|/ℓ⁰_I ≤ 1e-8 after EVERY
// step, energy strictly decreases, all steps accepted.
test('flow: 5 steps on crossing with [barycenter, edgeLengths] — accepted, energy ↓, every-edge drift ≤ 1e-8 each step', () => {
    const { vertices, edges, alpha, beta, epsilon } = loadFixture('crossing');
    const disjointPairs = calculateDisjointPairs(edges);
    // Frozen targets: computed ONCE from the initial state (spec §3.5).
    const x0 = barycenterTarget(vertices, edges);
    const ell0 = edgeLengths(vertices, edges);
    const set: ConstraintSet = [barycenterBlock(x0), edgeLengthsBlock(ell0)];
    const opts = { mode: 'analytical' as const, alpha, beta, epsilon };

    let current = vertices;
    let previousEnergy = calculateEnergy(current, edges, disjointPairs, alpha, beta, epsilon);
    for (let step = 0; step < 5; step++) {
        const r = sobolevStepSet(current, edges, disjointPairs, set, opts);
        expect(r.accepted).toBe(true);
        expect(r.energy).toBeLessThan(previousEnergy);
        current = r.vertices;
        previousEnergy = r.energy;
        const worst = maxEdgeDrift(current, edges, ell0);
        console.log(
            `[flowM2:perEdge] step ${step + 1}: τ = ${r.stats.tau}, E = ${r.energy.toExponential(6)}, ` +
                `‖g̃‖ = ${r.stats.gradientL2Norm.toExponential(3)}, max per-edge drift = ${worst.toExponential(3)}`,
        );
        expect(worst).toBeLessThanOrEqual(1e-8);
    }
});

// Point flow (spec §5.4.5): pin vertex 0 on crossing, 3 steps — γ₀ within 1e-8
// of the target after every step while energy decreases. The pin-only set also
// exercises a barycenter-less ConstraintSet (spec §9a).
test('flow: 3 steps on crossing with [point(0)] — accepted, energy ↓, ‖γ₀ − target‖ ≤ 1e-8 each step', () => {
    const { vertices, edges, alpha, beta, epsilon } = loadFixture('crossing');
    const disjointPairs = calculateDisjointPairs(edges);
    const target: Vec3 = [vertices[0][0], vertices[0][1], vertices[0][2]];
    const set: ConstraintSet = [pointBlock(0, target)];
    const opts = { mode: 'analytical' as const, alpha, beta, epsilon };

    let current = vertices;
    let previousEnergy = calculateEnergy(current, edges, disjointPairs, alpha, beta, epsilon);
    for (let step = 0; step < 3; step++) {
        const r = sobolevStepSet(current, edges, disjointPairs, set, opts);
        expect(r.accepted).toBe(true);
        expect(r.energy).toBeLessThan(previousEnergy);
        current = r.vertices;
        previousEnergy = r.energy;
        const pinDist = distTo(current[0], target);
        console.log(
            `[flowM2:point] step ${step + 1}: τ = ${r.stats.tau}, E = ${r.energy.toExponential(6)}, ` +
                `‖γ₀ − target‖ = ${pinDist.toExponential(3)}`,
        );
        expect(pinDist).toBeLessThanOrEqual(1e-8);
    }
});

// Rank rule, feasible side (spec §5.4.4): pinning a vertex WHOSE OWN edges are
// per-edge length-constrained is generically independent — the composed set
// must still yield an accepted, energy-decreasing step honoring both blocks.
test('rank rule: [edgeLengths, point(0)] on crossing — construction passes, accepted step honors pin and every ℓ_I', () => {
    const { vertices, edges, alpha, beta, epsilon } = loadFixture('crossing');
    const disjointPairs = calculateDisjointPairs(edges);
    const ell0 = edgeLengths(vertices, edges);
    const target: Vec3 = [vertices[0][0], vertices[0][1], vertices[0][2]];
    const set: ConstraintSet = [edgeLengthsBlock(ell0), pointBlock(0, target)];
    assertValidConstraintSet(set);

    const e0 = calculateEnergy(vertices, edges, disjointPairs, alpha, beta, epsilon);
    const r = sobolevStepSet(vertices, edges, disjointPairs, set, {
        mode: 'analytical',
        alpha,
        beta,
        epsilon,
    });
    console.log(
        `[flowM2:pin+edges] τ = ${r.stats.tau}, E ${e0.toExponential(6)} → ${r.energy.toExponential(6)}, ` +
            `pin dist = ${distTo(r.vertices[0], target).toExponential(3)}, ` +
            `max per-edge drift = ${maxEdgeDrift(r.vertices, edges, ell0).toExponential(3)}`,
    );
    expect(r.accepted).toBe(true);
    expect(r.energy).toBeLessThan(e0);
    expect(distTo(r.vertices[0], target)).toBeLessThanOrEqual(1e-8);
    expect(maxEdgeDrift(r.vertices, edges, ell0)).toBeLessThanOrEqual(1e-8);
});
```

- [x] **Step 3.2: Run.** `bun test test/sobolev/constraintSetM2Flow.test.ts` → all pass (golden gates: g̃ ≤1e-9 rel, λ ≤1e-9 rel, residual ≤1e-10, τ exact, energy ≤1e-12 rel, vertices ≤1e-9 rel). Then full `bun test` + `bunx tsc --noEmit`. If a flow test fails to accept a step, that is a REAL finding — investigate (systematic-debugging), do not loosen gates.

### Task 4: Store — lengthMode + frozen ℓ⁰ + dispatch (TDD) [CODE — Opus subagent + Opus spec/code review]

**Files:**
- Test: `test/store-constraints-m2.test.ts` (new)
- Modify: `src/store.ts`

**Contract (binding):** existing exported signatures unchanged; `dispatchDescentStep` args gain OPTIONAL `lengthMode?: LengthMode` and `sobolevEll0?: number[]` only. `test/store-constraints.test.ts` and every other existing test must pass WITHOUT edits — the `lengthConstraint` boolean stays as a write-through mirror of `lengthMode`.

- [x] **Step 4.1: Write the failing test file** `test/store-constraints-m2.test.ts`:

```ts
import { expect, test } from 'bun:test';
import { sobolevStep, sobolevStepSet } from '../src/core/optimizer';
import {
    barycenterBlock,
    edgeLengths,
    edgeLengthsBlock,
    totalLength,
} from '../src/core/sobolev/constraintSet';
import { barycenterTarget } from '../src/core/sobolev/constraints';
import { flatten } from '../src/core/sobolev/layout';
import { dispatchDescentStep, useSimStore } from '../src/store';

// Constraints-M2 store/dispatch coverage: the 3-way lengthMode (spec §5.3)
// with its legacy-boolean write-through mirror, the frozen-ℓ⁰ lifecycle
// (mirrors the sobolevL0 tests in test/store-constraints.test.ts — same three
// re-anchor points), and the lengthMode-driven ConstraintSet dispatch.
// NOTE: the store is a module-level singleton shared across test files —
// every test here restores the defaults it changes (lengthMode 'total'),
// because test/store-constraints.test.ts asserts those defaults.
// @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §3.5, §5.3, §9a

test('store: lengthMode defaults to total; setLengthMode syncs the legacy boolean mirror and clears converged', () => {
    useSimStore.getState().setPreset('crossing');
    expect(useSimStore.getState().lengthMode).toBe('total');
    expect(useSimStore.getState().lengthConstraint).toBe(true);

    useSimStore.setState({ sobolevConverged: true });
    useSimStore.getState().setLengthMode('perEdge');
    expect(useSimStore.getState().lengthMode).toBe('perEdge');
    // perEdge is still "a length constraint" for the legacy mirror.
    expect(useSimStore.getState().lengthConstraint).toBe(true);
    // A converged verdict is per-constraint-set: toggling invalidates it.
    expect(useSimStore.getState().sobolevConverged).toBe(false);

    useSimStore.getState().setLengthMode('none');
    expect(useSimStore.getState().lengthConstraint).toBe(false);

    // The legacy M1 setter writes THROUGH lengthMode (mirror can't diverge).
    useSimStore.getState().setLengthConstraint(true);
    expect(useSimStore.getState().lengthMode).toBe('total');
    useSimStore.getState().setLengthConstraint(false);
    expect(useSimStore.getState().lengthMode).toBe('none');

    useSimStore.getState().setLengthMode('total');
});

test('store: ℓ⁰ re-anchors on play and on commit from live positions, never mid-run (frozen-targets lifecycle)', () => {
    useSimStore.getState().setPreset('crossing');
    const st = useSimStore.getState();
    // Rebuild (setPreset) is re-anchor point 1: ℓ⁰ = current graph edge lengths.
    expect(st.sobolevEll0).toEqual(edgeLengths(st.graph.vertices, st.graph.edges));

    // Perturb the live buffer, then play: ℓ⁰ must re-anchor to the CURRENT
    // geometry, not the preset.
    st.live[0][0] += 0.123;
    useSimStore.getState().setRunning(true);
    const ell0AtPlay = useSimStore.getState().sobolevEll0;
    expect(ell0AtPlay).toEqual(
        edgeLengths(useSimStore.getState().live, useSimStore.getState().graph.edges),
    );

    // Mid-run motion of the live buffer must NOT move the frozen target
    // (same array identity — nothing recomputed it).
    useSimStore.getState().live[1][1] += 0.5;
    expect(useSimStore.getState().sobolevEll0).toBe(ell0AtPlay);

    // Pause = vertex commit → re-anchor from the committed positions.
    useSimStore.getState().setRunning(false);
    const ell0AtCommit = useSimStore.getState().sobolevEll0;
    expect(ell0AtCommit).toEqual(
        edgeLengths(useSimStore.getState().live, useSimStore.getState().graph.edges),
    );
    expect(ell0AtCommit).not.toEqual(ell0AtPlay);
});

test('dispatchDescentStep: lengthMode builds the ConstraintSet — perEdge matches the explicit set, mode wins over the legacy boolean, legacy shape unchanged', () => {
    useSimStore.getState().setPreset('crossing');
    const st = useSimStore.getState();
    const vertices = st.graph.vertices;
    const edges = st.graph.edges;
    const disjointPairs = st.disjointPairs;
    const x0 = barycenterTarget(vertices, edges);
    const L0 = totalLength(vertices, edges);
    const ell0 = edgeLengths(vertices, edges);
    const base = {
        descentMode: 'sobolev' as const,
        vertices,
        edges,
        disjointPairs,
        mode: 'analytical' as const,
        stepSize: 0.001,
        x0,
        sobolevL0: L0,
        sobolevEll0: ell0,
    };
    const opts = { mode: 'analytical' as const };

    // perEdge: [barycenter, edgeLengths] — barycenter first (spec §3.2 row order).
    const perEdge = dispatchDescentStep({
        ...base,
        barycenterConstraint: true,
        lengthMode: 'perEdge',
    });
    const perEdgeRef = sobolevStepSet(
        vertices,
        edges,
        disjointPairs,
        [barycenterBlock(x0), edgeLengthsBlock(ell0)],
        opts,
    );
    expect(perEdge.energy).toBe(perEdgeRef.energy);
    expect(flatten(perEdge.vertices)).toEqual(flatten(perEdgeRef.vertices));

    // lengthMode 'total' is bit-identical to the M1 boolean path.
    const totalViaMode = dispatchDescentStep({ ...base, lengthMode: 'total' });
    const totalViaBool = dispatchDescentStep({ ...base, lengthConstraint: true });
    expect(totalViaMode.energy).toBe(totalViaBool.energy);
    expect(flatten(totalViaMode.vertices)).toEqual(flatten(totalViaBool.vertices));

    // lengthMode WINS over a contradictory legacy boolean.
    const modeWins = dispatchDescentStep({
        ...base,
        barycenterConstraint: true,
        lengthConstraint: true,
        lengthMode: 'none',
    });
    const noneRef = sobolevStepSet(vertices, edges, disjointPairs, [barycenterBlock(x0)], opts);
    expect(modeWins.energy).toBe(noneRef.energy);
    expect(flatten(modeWins.vertices)).toEqual(flatten(noneRef.vertices));

    // All toggle fields ABSENT (pre-M1 call shape): still bit-identical to the
    // legacy sobolevStep(x0) path (extra sobolevEll0 data alone must not
    // change the dispatch decision).
    const legacy = dispatchDescentStep(base);
    const legacyRef = sobolevStep(vertices, edges, disjointPairs, x0, opts);
    expect(legacy.energy).toBe(legacyRef.energy);
    expect(flatten(legacy.vertices)).toEqual(flatten(legacyRef.vertices));
});
```

- [x] **Step 4.2: Run to verify RED.** `bun test test/store-constraints-m2.test.ts` — expected failures: `lengthMode` undefined, `setLengthMode` is not a function, `sobolevEll0` undefined, dispatch ignoring `lengthMode` (energy mismatch vs the perEdge reference). `test/store-constraints.test.ts` must still be green at this point.

- [x] **Step 4.3: Implement in `src/store.ts`.**

(a) Extend the constraintSet import with `edgeLengths` and `edgeLengthsBlock`.

(b) Add below `DescentMode`:

```ts
/**
 * 3-way length-constraint mode for the sobolev ConstraintSet (spec §5.3):
 * 'none' | 'total' (M1 total-length row) | 'perEdge' (M2, |E| rows). The
 * §3.4 totalLength/edgeLengths mutual exclusion is enforced BY CONSTRUCTION —
 * one select, one value. 'total' is the default (preserves the M1 default
 * lengthConstraint = true).
 * @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §5.3, §3.4
 */
export type LengthMode = 'none' | 'total' | 'perEdge';
```

(c) `dispatchDescentStep`: add `lengthMode?: LengthMode;` and `sobolevEll0?: number[];` to the args type (after `lengthConstraint?`). Extend the legacy-shape guard and the set building:

```ts
        if (
            args.barycenterConstraint === undefined &&
            args.lengthConstraint === undefined &&
            args.lengthMode === undefined
        ) {
            // Pre-M1 call shape: legacy barycenter-only path, bit-identical to
            // sobolevStep(x0) (spec §4.2 back-compat).
            ...unchanged body...
        }
        const set: ConstraintSet = [];
        if (args.barycenterConstraint ?? true) set.push(barycenterBlock(args.x0));
        // M2 (spec §5.3): the 3-way lengthMode supersedes the M1 boolean; when
        // absent it degrades to the M1 semantics (lengthConstraint ?? true →
        // 'total') so M1 call sites stay bit-identical.
        const lengthMode: LengthMode =
            args.lengthMode ?? ((args.lengthConstraint ?? true) ? 'total' : 'none');
        if (lengthMode === 'total') {
            // An enabled length constraint requires its frozen L⁰ (spec §3.5).
            // NaN backstop if a caller omits it: Φ becomes NaN, projection can
            // never converge, and the step is REJECTED ('projection_failed')
            // instead of silently drifting or throwing in the frame loop.
            set.push(totalLengthBlock(args.sobolevL0 ?? Number.NaN));
        } else if (lengthMode === 'perEdge') {
            // Same NaN backstop for a missing frozen ℓ⁰ vector: NaN Φ rows →
            // projection can't converge → 'projection_failed', never a throw.
            set.push(edgeLengthsBlock(args.sobolevEll0 ?? args.edges.map(() => Number.NaN)));
        }
```

(the `assertValidConstraintSet(set)` call and the rest stay). Update `dispatchDescentStep`'s TSDoc to mention the mode precedence and cite spec §5.3 alongside §4.2/§9a.

(d) `SimStore`: extend the frozen-targets comment block to cover ℓ⁰, then:

```ts
    sobolevX0: Vec3;
    sobolevL0: number;
    sobolevEll0: number[];
    // 3-way length mode (spec §5.3) — the SOURCE OF TRUTH for the length
    // constraint. lengthConstraint below is a WRITE-THROUGH MIRROR
    // (true ⟺ lengthMode !== 'none'; its setter delegates to setLengthMode),
    // kept ONLY so the M1 store contract and test/store-constraints.test.ts
    // stay valid unmodified (§4.5 acceptance gate, applied to M2 via §5.5).
    // New code reads/writes lengthMode.
    // @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §5.3, §4.5
    lengthMode: LengthMode;
    barycenterConstraint: boolean;
    lengthConstraint: boolean;
```

and `setLengthMode(m: LengthMode): void;` next to `setLengthConstraint`.

(e) Initial state: add `lengthMode: 'total',` next to `lengthConstraint: true,` and `sobolevEll0: edgeLengths(built.graph.vertices, built.graph.edges),` next to `sobolevL0`.

(f) `rebuild`: add `sobolevEll0: edgeLengths(b.graph.vertices, b.graph.edges),` next to the `sobolevL0` re-anchor.

(g) `setRunning(true)` branch: add `sobolevEll0: edgeLengths(s.live, s.graph.edges),`; `setRunning(false)` commit branch: same. (Both next to the existing `sobolevL0` lines — the frozen-targets lifecycle anchor already covers them; extend its text to name ℓ⁰.)

(h) Setters:

```ts
        // Toggling a constraint invalidates ONLY the converged verdict (it is
        // per-constraint-set); targets re-anchor at the next run start, and
        // sobolevStats stay — they describe the last step actually taken.
        // @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §4.2, §5.3, §9a
        setLengthMode: (m) =>
            set({ lengthMode: m, lengthConstraint: m !== 'none', sobolevConverged: false }),
        // Legacy M1 setter: writes THROUGH the 3-way mode so the boolean mirror
        // and lengthMode can never diverge (see the lengthMode field anchor).
        setLengthConstraint: (b) => get().setLengthMode(b ? 'total' : 'none'),
```

- [x] **Step 4.4: Run to verify GREEN.** `bun test test/store-constraints-m2.test.ts` passes; `bun test` fully green (in particular `test/store-constraints.test.ts` UNMODIFIED and green); `bunx tsc --noEmit` clean.

### Task 5: UI — Length select + Viewer wiring [CODE — Opus subagent + Opus spec/code review; may be batched with Task 4]

**Files:**
- Modify: `src/ui/ControlPanel.tsx`
- Modify: `src/scene/Viewer.tsx`

- [x] **Step 5.1: ControlPanel.** Change the store import line to `import { type DescentMode, type LengthMode, type Mode, useSimStore } from '../store';`. Replace the selector pair

```ts
    const lengthConstraint = useSimStore((s) => s.lengthConstraint);
    const setLengthConstraint = useSimStore((s) => s.setLengthConstraint);
```

with

```ts
    const lengthMode = useSimStore((s) => s.lengthMode);
    const setLengthMode = useSimStore((s) => s.setLengthMode);
```

and replace the entire "Fix length" checkbox `<label>` (keep the Barycenter checkbox `<label>` above it untouched — §9a) with:

```tsx
                {/* 3-way Length select (M2, spec §5.3) replacing the M1 "Fix length"
                    checkbox: none | total | per-edge. The §3.4 totalLength/edgeLengths
                    mutual exclusion is enforced BY CONSTRUCTION — one select, one
                    value. Sobolev-only: disabled (not hidden) in raw mode, same as
                    the Barycenter checkbox.
                    @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §5.3, §3.4 */}
                <label
                    style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 8,
                        opacity: descentMode === 'sobolev' ? 1 : 0.4,
                    }}
                >
                    <span>Length:</span>
                    <select
                        value={lengthMode}
                        disabled={descentMode !== 'sobolev'}
                        onChange={(e) => setLengthMode(e.target.value as LengthMode)}
                        style={{ padding: 8, fontSize: 14 }}
                    >
                        <option value="none">None</option>
                        <option value="total">Total</option>
                        <option value="perEdge">Per-edge</option>
                    </select>
                </label>
```

- [x] **Step 5.2: Viewer.** In the `dispatchDescentStep` call inside `Simulation`, replace `lengthConstraint: st.lengthConstraint,` with:

```ts
                lengthMode: st.lengthMode,
                sobolevEll0: st.sobolevEll0,
```

(keep `x0`, `sobolevL0`, `barycenterConstraint`; extend the adjacent frozen-targets comment to mention ℓ⁰ and cite spec §5.3 alongside §4.2/§9a).

- [x] **Step 5.3: Verify.** `bunx tsc --noEmit` clean; full `bun test` green (no UI unit tests — the Task 6 boot check is the functional gate). Run `bunx biome check --write --linter-enabled=false` on the two touched files.

### Task 6: Acceptance gates, boot check, commit, report [orchestrator — no subagent]

- [x] **Step 6.1: Full gates.** `bun test` (all green; zero edits to existing test files — verify with `git status`/`git diff --stat` that no pre-existing test file is modified); `bunx tsc --noEmit` clean; re-run all four M2 oracle regen commands → property checks PASS; re-run the Step 1.2 length-mode semantic-identity check.
- [x] **Step 6.2: knip note.** `bunx knip --no-exit-code` — `pointBlock` (and possibly `edgeLengths`) may be flagged unused from src/: expected (spec §5.3), non-blocking, note in the report.
- [x] **Step 6.3: Boot check** (headless recipe in project memory `headless-webgpu-boot-check`; grep out the known SwiftShader spam). Confirm: Descent=Sobolev shows the Barycenter checkbox + Length select (default Total), both disabled in raw mode; select Per-edge, run the crossing preset: energy decreases, L drift stays ≤ ~1e-8, no rejections; quote a few frames (E, τ, ‖g̃‖, L drift). Also sanity-run `total` mode to confirm no M1 regression.
- [x] **Step 6.4: Commits.** First (if not yet committed) `docs: implementation plan for Sobolev constraints M2` with only the plan doc. Then stage everything else and commit `feat(sobolev): per-edge length + point constraints (constraints M2)` (body: summary, spec ref, verification results, Co-Authored-By/session trailers). The lefthook pre-commit will biome-reformat staged JSON goldens — expected; JSON.parse-based tests are whitespace-insensitive. Iterate with `--amend` only. DO NOT PUSH.
- [x] **Step 6.5: §9 report** to the user: files + line counts, exports changed (must be: none — optional-args extension only), tests edited (must be: none existing), verbatim `bun test` tail, flow numbers, oracle regen output, boot-check frames, ambiguities resolved, commit hash.

---

## Self-review notes (spec coverage)

- §5.1 core: Task 2 (blocks; §3.4 throw already live in M1 `assertValidConstraintSet` — Task 2 tests it with real blocks). Saddle size 3|V|+3+|E|(+3/pin) needs no solver change — nothing touched in `linsolve.ts`.
- §5.2 oracle: Task 1 (both modes, 4 goldens, existing goldens untouched).
- §5.3 store/UI: Tasks 4–5 (3-way select, Barycenter checkbox stays per §9a; point = machinery+tests only; knip note in Task 6).
- §5.4 tests: 1→Task 2 (FD incl. junction-y; point exact identity), 2→Task 3 (4 goldens), 3→Task 3 (per-edge flow), 4→Tasks 2+3 (throw + pin+edges accepted step), 5→Task 3 (pin flow).
- §5.5 gates: Task 6 (incl. per-edge in-app check).
- §9a: barycenter toggle already shipped in M1; length toggles subsumed by `lengthMode`; empty set already supported.
- Known spec tension resolved: §5.3 "replace the checkbox" vs §4.5 "zero edits to existing test files" → `lengthMode` is the source of truth, `lengthConstraint` stays as a write-through mirror so `test/store-constraints.test.ts` passes unmodified. Report as a resolved ambiguity.
