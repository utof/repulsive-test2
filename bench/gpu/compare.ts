// bench/gpu/compare.ts — T1/T2/T3 comparators + seeded PRNG, shared by the Phase 0
// spikes (spikes.ts) and the Phase 1 gates (phase1.ts). Bodies moved VERBATIM out of
// spikes.ts (Phase 0 T11 / T6) so both files check against one definition.
// @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §3 (T1–T3)
// @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md — "Task 3: `GpuTangentPoint` + reduction + T1/T2/T3 through the production kernel"

/**
 * Deterministic xorshift32 PRNG — no `Math.random`, so seeded data (the
 * matvec matrix in particular) is reproducible across runs/machines.
 * @see docs/superpowers/plans/2026-08-13-webgpu-solver-phase0.md Task 6
 */
export function xorshift32(seed: number): () => number {
    let s = seed >>> 0 || 1;
    return () => {
        s ^= s << 13;
        s >>>= 0;
        s ^= s >>> 17;
        s ^= s << 5;
        s >>>= 0;
        return s / 4294967296;
    };
}

/**
 * Run `run` with `Math.random` replaced by `xorshift32(seed)` so fixture
 * generators that call `Math.random` internally (`testConfigs[*].generate`)
 * produce the same geometry on every gate run. Restored in `finally` so a
 * throwing fixture can never leak the patched PRNG into later spikes.
 * @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md — "Task 3: `GpuTangentPoint` + reduction + T1/T2/T3 through the production kernel" (T2/T3 fixtures "with Math.random seeded")
 */
export function withSeed<T>(seed: number, run: () => T): T {
    const orig = Math.random;
    Math.random = xorshift32(seed);
    try {
        return run();
    } finally {
        Math.random = orig;
    }
}

/**
 * Generic rel-err comparator for T1 (per-pair kernel value) / T2 (total
 * energy): `|gpu - cpuRef| / |cpuRef|`, matching the spec §3 tolerance
 * definitions verbatim ("rel err").
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §3 (T1, T2)
 */
export function relErrComparator(gpu: number, cpuRef: number): number {
    return Math.abs(gpu - cpuRef) / Math.abs(cpuRef);
}

/**
 * Generic cosine comparator for T3 (gradient direction agreement):
 * `dot(a,b) / (|a| * |b|)`. Spec §3 T3 threshold is `cosine > 1 - 1e-6`.
 * Zero-safe (plan Task 3, comparators): both vectors zero ⇒ 1 (identical
 * direction-less gradients, e.g. a fully symmetric fixture); exactly one
 * zero ⇒ −1 (GPU and CPU disagree on whether there is any gradient at all —
 * must FAIL the T3 gate, not produce NaN that `>` silently passes over).
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §3 (T3)
 * @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md — "Task 3: `GpuTangentPoint` + reduction + T1/T2/T3 through the production kernel" (comparators)
 */
export function cosineComparator(a: number[], b: number[]): number {
    if (a.length !== b.length) throw new Error('cosineComparator: length mismatch');
    let dot = 0;
    let na = 0;
    let nb = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na === 0 && nb === 0) return 1;
    if (na === 0 || nb === 0) return -1;
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
}
