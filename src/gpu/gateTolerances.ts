/**
 * Spec-pinned GPU accuracy-gate tolerances, shared by the runtime boot gate
 * (`selfTest.ts`) and the bench kill gates (`bench/gpu/*`) so retuning one
 * cannot silently desync the others. (`nearTouchPair` gap arguments are
 * fixture geometry, NOT tolerances, and intentionally do not live here.)
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §3
 */

/**
 * T1: per-pair kernel value vs f64 oracle, pass iff `relErr < T1_REL_ERR`.
 * Why: spec §3 T1 (rel err < 1e-5).
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §3
 */
export const T1_REL_ERR = 1e-5;

/**
 * T2: total energy vs f64, pass iff `relErr < T2_REL_ERR`.
 * Why: spec §3 T2 (rel err < 1e-6).
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §3
 */
export const T2_REL_ERR = 1e-6;

/**
 * T3: gradient direction vs f64, pass iff `cosine > T3_COSINE_MIN`.
 * Why: spec §3 T3 (cosine > 1 − 1e-6). Kept as the expression `1 - 1e-6`,
 * not a pre-evaluated decimal literal, so the float is identical to the
 * original inline spelling.
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §3
 */
export const T3_COSINE_MIN = 1 - 1e-6;
