// src/gpu/wgsl.ts — pure WGSL source generation for the tangent-point kernels.
// No three.js import: unit-testable in Bun; the single home of the
// precision-critical op order. Positions buffer `p` is interleaved
// [hi.x,hi.y,hi.z,lo.x,lo.y,lo.z] per vertex (6 floats).
// @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (D1–D3)
// @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.2, §2.3

/** f32 unit roundoff. @see plan D3 */
export const F32_U = 2 ** -24;
/** Guard multiplier over the two-float noise floor. @see plan D3 */
export const GUARD_FACTOR = 32;

/** α, β, ε of the tangent-point energy, baked into the WGSL at build time. @see src/core/tangentPointEnergy.ts:48-56 */
export interface KernelParams {
    alpha: number;
    beta: number;
    epsilon: number;
}

/**
 * `x^n` in WGSL: repeated multiplication for integer n (spec §2.3), `pow()`
 * only for non-integer n. Left-assoc `((x*x)*x)` on purpose — plan D2.
 * @see docs/2026-08-13-ai-research-gpu-precision.md Q1 ("WGSL's weaker pow")
 */
export function powExpr(x: string, n: number): string {
    if (!Number.isInteger(n)) return `pow(${x}, ${n})`;
    if (n === 0) return '1.0';
    const k = Math.abs(n);
    // k=1 → "(x)"; k=3 → "((x * x) * x)" — these exact strings are the test contract (test/gpu/wgsl.test.ts).
    let e = x;
    for (let i = 1; i < k; i++) e = `(${e} * ${x})`;
    if (k === 1) e = `(${x})`;
    return n < 0 ? `(1.0 / (${e}))` : e;
}

const f32 = (v: number): string => (/[.e]/i.test(String(v)) ? String(v) : `${v}.0`);

/**
 * Shared WGSL block — module-scope struct + helper fns ONLY, no entry point.
 * Consumed as a `wgsl(src)` INCLUDE of each entry `wgslFn` (three r185
 * `FunctionNode.js:168` wgslFn(code, includes); `CodeNode.js:99-113` builds
 * includes first). It must NOT be concatenated into the entry source:
 * `WGSLNodeFunction.js:4` parses `^fn name(...)` and throws
 * "Function is not a WGSL code." on anything else (plan review r1 #1).
 * Contents: two-float difference, manual cross (CPU component order,
 * tangentPointEnergy.ts:93-95), D3 degeneracy guards, and `kernelDerivs`
 * (CPU closure at tangentPointEnergy.ts:164-248) in the reciprocal-power
 * form of plan D2. `+ eps` after every norm is KEPT (plan D2).
 */
export function pairKernelWgsl(p: KernelParams): string {
    const { alpha, beta, epsilon } = p;
    return /* wgsl */ `
        struct KD { f: f32, df_dd: vec3<f32>, df_de: vec3<f32> }

        // (hi_i - hi_j) + (lo_i - lo_j) BEFORE anything else — the reassociation G2 gates. @see spec §2.3
        fn diff2(p: ptr<storage, array<f32>, read>, i: u32, j: u32) -> vec3<f32> {
            let x = (p[6u * i + 0u] - p[6u * j + 0u]) + (p[6u * i + 3u] - p[6u * j + 3u]);
            let y = (p[6u * i + 1u] - p[6u * j + 1u]) + (p[6u * i + 4u] - p[6u * j + 4u]);
            let z = (p[6u * i + 2u] - p[6u * j + 2u]) + (p[6u * i + 5u] - p[6u * j + 5u]);
            return vec3<f32>(x, y, z);
        }
        // D3: two-float noise floor of a difference between vertices i and j.
        fn diffGuard(p: ptr<storage, array<f32>, read>, i: u32, j: u32) -> f32 {
            let k = ${f32(GUARD_FACTOR)} * ${f32(F32_U * F32_U)};
            let gx = k * max(abs(p[6u * i + 0u]), abs(p[6u * j + 0u]));
            let gy = k * max(abs(p[6u * i + 1u]), abs(p[6u * j + 1u]));
            let gz = k * max(abs(p[6u * i + 2u]), abs(p[6u * j + 2u]));
            return sqrt(gx * gx + gy * gy + gz * gz);
        }
        fn cross3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
            return vec3<f32>(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
        }
        // D3: noise floor of a cross product from its per-component product scales.
        fn crossGuard(e: vec3<f32>, d: vec3<f32>) -> f32 {
            let sx = abs(e.y * d.z) + abs(e.z * d.y);
            let sy = abs(e.z * d.x) + abs(e.x * d.z);
            let sz = abs(e.x * d.y) + abs(e.y * d.x);
            return ${f32(GUARD_FACTOR)} * ${f32(F32_U)} * sqrt(sx * sx + sy * sy + sz * sz);
        }
        fn norm3(v: vec3<f32>) -> f32 { return sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }

        // dGuard: precomputed diffGuard for d (caller passes it; avoids re-reading p).
        fn kernelDerivs(e: vec3<f32>, d: vec3<f32>, dGuard: f32) -> KD {
            let alpha = ${f32(alpha)};
            let beta = ${f32(beta)};
            let eps = ${f32(epsilon)};
            let rd = norm3(d);
            var dHat = vec3<f32>(0.0);
            if (rd > dGuard) { dHat = d * (1.0 / rd); }           // CPU: !(rd < 1e-14) → D3; STRICT: guard may be exactly 0 (collinear/coincident) and 0>=0 would give 1/0
            let d_eps = rd + eps;                                  // ε after norm — tangentPointEnergy.ts:61-65, plan D2
            let c = cross3(e, d);
            let rc = norm3(c);
            let c_eps = rc + eps;
            let invD = 1.0 / d_eps;
            let invDb = ${powExpr('invD', beta)};                  // d_eps^-β — D2
            let cPowA = ${powExpr('c_eps', alpha)};
            let f = cPowA * invDb;
            var dc_dd = vec3<f32>(0.0);
            var dc_de = vec3<f32>(0.0);
            if (rc > crossGuard(e, d)) {                           // CPU: rc >= 1e-14 → D3; STRICT (see rd above; the 'crossing' preset has guard == rc == 0)
                let invRc = 1.0 / rc;
                dc_dd = cross3(c, e) * invRc;
                dc_de = cross3(d, c) * invRc;
            }
            let coeff_c = alpha * ${powExpr('c_eps', alpha - 1)} * invDb;
            let coeff_d = -beta * f * invD;                        // = -β c^α / d^(β+1) without forming d^(β+1) — D2
            var out: KD;
            out.f = f;
            out.df_dd = coeff_c * dc_dd + coeff_d * dHat;
            out.df_de = coeff_c * dc_de;
            return out;
        }
    `;
}

/**
 * Gradient gather — one invocation per edge I; both ordered roles (plan D1);
 * writes ONLY I's two endpoint slots, already ×0.5 (the CPU's final loop,
 * tangentPointEnergy.ts:380-390). Buffers: p (6V f32), edges (2E u32),
 * pairOffsets (E+1), pairIndices, slots (6E f32).
 * Slot contract: endpoint i1 of edge I lives at slots[6I+0..2] and i2 at
 * slots[6I+3..5], i.e. slot index s = 2·I + endpoint(0|1) with 3 floats each —
 * the vertex→slot CSR consumed by `tpVertexGather` MUST use that same s.
 * ENTRY FUNCTION ONLY — `pairKernelWgsl` is attached as a wgslFn include
 * (see its TSDoc). Every `ptr<storage,…,MODE>` here must equal the bound
 * node's access: `p` and topology are `.toReadOnly()` nodes ⇒ `read`;
 * `slots` is a read_write node ⇒ `read_write` (WGSLNodeBuilder.js:2155
 * emits the NODE's access; a mode mismatch is a shader compile error).
 * @see src/core/tangentPointEnergy.ts:250-378 (the loop being gathered)
 */
export function gradientKernelWgsl(p: KernelParams): string {
    const { alpha, epsilon } = p;
    return /* wgsl */ `
        fn tpGradient(
            p: ptr<storage, array<f32>, read>,
            edges: ptr<storage, array<u32>, read>,
            pairOffsets: ptr<storage, array<u32>, read>,
            pairIndices: ptr<storage, array<u32>, read>,
            slots: ptr<storage, array<f32>, read_write>,
            edgeCount: u32,
            I: u32
        ) -> void {
            if (I >= edgeCount) { return; }
            let alpha = ${f32(alpha)};
            let eps = ${f32(epsilon)};
            let i1 = edges[2u * I + 0u];
            let i2 = edges[2u * I + 1u];
            let eI = diff2(p, i2, i1);
            let reI = norm3(eI);
            var eIhat = vec3<f32>(0.0);
            if (reI > diffGuard(p, i2, i1)) { eIhat = eI * (1.0 / reI); }   // STRICT — D3
            let ellI = reI + eps;
            let ellIpow = ${powExpr('ellI', 1 - alpha)};
            let dPowCoeff = (1.0 - alpha) * ${powExpr('ellI', -alpha)};
            var g1 = vec3<f32>(0.0);
            var g2 = vec3<f32>(0.0);
            for (var k = pairOffsets[I]; k < pairOffsets[I + 1u]; k = k + 1u) {
                let J = pairIndices[k];
                let j1 = edges[2u * J + 0u];
                let j2 = edges[2u * J + 1u];
                let eJ = diff2(p, j2, j1);
                let ellJ = norm3(eJ) + eps;
                let ellJpow = ${powExpr('ellJ', 1 - alpha)};

                // ---- Role A: ordered pair (I,J), e = eI, d = v_i - v_j (plan D1)
                var sumF = 0.0;
                let baseA = 0.25 * ellIpow * ellJ;
                for (var ii = 0u; ii < 2u; ii = ii + 1u) {
                    for (var jj = 0u; jj < 2u; jj = jj + 1u) {
                        let i = select(i2, i1, ii == 0u);
                        let j = select(j2, j1, jj == 0u);
                        let kd = kernelDerivs(eI, diff2(p, i, j), diffGuard(p, i, j));
                        sumF = sumF + kd.f;
                        if (ii == 0u) { g1 = g1 + baseA * kd.df_dd; } else { g2 = g2 + baseA * kd.df_dd; }
                        g1 = g1 - baseA * kd.df_de;
                        g2 = g2 + baseA * kd.df_de;
                    }
                }
                let sI = 0.25 * ellJ * sumF;
                g1 = g1 + sI * dPowCoeff * (-eIhat);
                g2 = g2 + sI * dPowCoeff * eIhat;

                // ---- Role B: ordered pair (J,I), e = eJ, d = v_a - v_b, a in J, b in I (plan D1)
                var sumFb = 0.0;
                let baseB = 0.25 * ellJpow * ellI;
                for (var aa = 0u; aa < 2u; aa = aa + 1u) {
                    for (var bb = 0u; bb < 2u; bb = bb + 1u) {
                        let a = select(j2, j1, aa == 0u);
                        let b = select(i2, i1, bb == 0u);
                        let kd = kernelDerivs(eJ, diff2(p, a, b), diffGuard(p, a, b));
                        sumFb = sumFb + kd.f;
                        if (bb == 0u) { g1 = g1 - baseB * kd.df_dd; } else { g2 = g2 - baseB * kd.df_dd; }
                    }
                }
                let sJb = 0.25 * ellJpow * sumFb;   // dE via ell_J of (J,I): that "J" is our I
                g1 = g1 + sJb * (-eIhat);
                g2 = g2 + sJb * eIhat;
            }
            // ×0.5: disjointPairs lists both (I,J) and (J,I) — gradientAnalytical's final loop. @see plan D1
            let h1 = 0.5 * g1;
            let h2 = 0.5 * g2;
            slots[6u * I + 0u] = h1.x; slots[6u * I + 1u] = h1.y; slots[6u * I + 2u] = h1.z;
            slots[6u * I + 3u] = h2.x; slots[6u * I + 4u] = h2.y; slots[6u * I + 5u] = h2.z;
        }
    `;
}

/**
 * Vertex gather — one invocation per vertex v; sums its incident endpoint
 * slots (vertex→slot CSR, stored order) into grad[3v..3v+2]. No atomics.
 * `slots` is declared read_write here although only read: it is the SAME
 * read_write node the gradient kernel writes, and the pointer mode must
 * match the node's access (review r1 #2).
 * Slot contract: `slots[3s..3s+2]` with s = 2·I + endpoint(0|1), exactly as
 * `tpGradient` writes them (i1 → 6I+0..2, i2 → 6I+3..5); the CSR builder must
 * emit s under that rule.
 * @see plan D1
 */
export function vertexGatherWgsl(): string {
    return /* wgsl */ `
        fn tpVertexGather(
            slots: ptr<storage, array<f32>, read_write>,
            incidentOffsets: ptr<storage, array<u32>, read>,
            incidentSlots: ptr<storage, array<u32>, read>,
            grad: ptr<storage, array<f32>, read_write>,
            vertexCount: u32,
            v: u32
        ) -> void {
            if (v >= vertexCount) { return; }
            var g = vec3<f32>(0.0);
            for (var k = incidentOffsets[v]; k < incidentOffsets[v + 1u]; k = k + 1u) {
                let s = incidentSlots[k];
                g = g + vec3<f32>(slots[3u * s + 0u], slots[3u * s + 1u], slots[3u * s + 2u]);
            }
            grad[3u * v + 0u] = g.x; grad[3u * v + 1u] = g.y; grad[3u * v + 2u] = g.z;
        }
    `;
}

/**
 * Per-edge partial energy, role A only (ordered pairs summed over ALL I
 * cover both orders), ×0.5 = calculateEnergy's `totalEnergy / 2`.
 * @see src/core/tangentPointEnergy.ts:48-113
 */
export function energyKernelWgsl(p: KernelParams): string {
    const { alpha, epsilon } = p;
    // ENTRY FUNCTION ONLY — pairKernelWgsl is attached as a wgslFn include (review r1 #1).
    return /* wgsl */ `
        fn tpEnergy(
            p: ptr<storage, array<f32>, read>,
            edges: ptr<storage, array<u32>, read>,
            pairOffsets: ptr<storage, array<u32>, read>,
            pairIndices: ptr<storage, array<u32>, read>,
            partials: ptr<storage, array<f32>, read_write>,
            edgeCount: u32,
            I: u32
        ) -> void {
            if (I >= edgeCount) { return; }
            let eps = ${f32(epsilon)};
            let i1 = edges[2u * I + 0u];
            let i2 = edges[2u * I + 1u];
            let eI = diff2(p, i2, i1);
            let ellI = norm3(eI) + eps;
            let ellIpow = ${powExpr('ellI', 1 - alpha)};
            var acc = 0.0;
            for (var k = pairOffsets[I]; k < pairOffsets[I + 1u]; k = k + 1u) {
                let J = pairIndices[k];
                let j1 = edges[2u * J + 0u];
                let j2 = edges[2u * J + 1u];
                let ellJ = norm3(diff2(p, j2, j1)) + eps;
                var sumK = 0.0;
                for (var ii = 0u; ii < 2u; ii = ii + 1u) {
                    for (var jj = 0u; jj < 2u; jj = jj + 1u) {
                        let i = select(i2, i1, ii == 0u);
                        let j = select(j2, j1, jj == 0u);
                        sumK = sumK + kernelDerivs(eI, diff2(p, i, j), diffGuard(p, i, j)).f;
                    }
                }
                acc = acc + 0.25 * ellIpow * ellJ * sumK;
            }
            // /2 — calculateEnergy's totalEnergy / 2 (both pair orders listed). @see tangentPointEnergy.ts:106-112
            partials[I] = 0.5 * acc;
        }
    `;
}
