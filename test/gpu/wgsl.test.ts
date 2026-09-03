// test/gpu/wgsl.test.ts
import { expect, test } from 'bun:test';
import {
    energyKernelWgsl,
    gradientKernelWgsl,
    pairKernelWgsl,
    powExpr,
    vertexGatherWgsl,
} from '../../src/gpu/wgsl';

const P = { alpha: 3, beta: 6, epsilon: 1e-10 };

test('powExpr: integer exponents expand to repeated multiplication, never pow()', () => {
    expect(powExpr('x', 1)).toBe('(x)');
    expect(powExpr('x', 3)).toBe('((x * x) * x)');
    expect(powExpr('x', 0)).toBe('1.0');
    expect(powExpr('x', -2)).toBe('(1.0 / ((x * x)))');
    expect(powExpr('x', 2.5)).toBe('pow(x, 2.5)');
});

test('pair kernel (shared include): no pow() for integer α/β; two-float differences; eps kept; reciprocal-power form; guards strict', () => {
    const src = pairKernelWgsl(P);
    expect(src).not.toMatch(/\bpow\(/);
    expect(src).toContain('(p[6u * i + 0u] - p[6u * j + 0u]) + (p[6u * i + 3u] - p[6u * j + 3u])');
    expect(src).toContain('let eps = 1e-10;');
    expect(src).toContain('let d_eps = rd + eps;'); // D2: ε after norm, like the CPU
    expect(src).toContain('let invD = 1.0 / d_eps;');
    expect(src).toContain('let coeff_d = -beta * f * invD;'); // D2: never forms d^(β+1)
    expect(src).not.toMatch(/\*\s*rd\b/); // no d6*rd product anywhere
    expect(src).toContain('if (rd > dGuard)'); // D3: degenerate iff ≤ guard (guard may be exactly 0)
    expect(src).toContain('if (rc > crossGuard(e, d))');
    expect(src).not.toMatch(/>=\s*(dGuard|crossGuard|diffGuard)/);
});

test('pair kernel: non-integer α uses pow() for that exponent only', () => {
    const src = pairKernelWgsl({ alpha: 2.5, beta: 6, epsilon: 1e-10 });
    expect(src).toContain('pow(c_eps, 2.5)');
    expect(src).not.toContain('pow(invD');
});

test('entry sources are ONE bare fn each (three WGSLNodeFunction parser: ^fn name(...)); shared block lives in the include', () => {
    for (const src of [gradientKernelWgsl(P), vertexGatherWgsl(), energyKernelWgsl(P)]) {
        expect(src.trim().startsWith('fn ')).toBe(true);
        expect(src).not.toContain('struct KD');
        expect((src.match(/\bfn\s+\w+\s*\(/g) ?? []).length).toBe(1);
    }
    expect(pairKernelWgsl(P)).not.toContain('fn tpGradient(');
});

test('entry points + symmetry factor location', () => {
    expect(gradientKernelWgsl(P)).toContain('fn tpGradient(');
    expect(gradientKernelWgsl(P)).toContain('0.5 * g1'); // D1: ×0.5 in the edge kernel
    expect(gradientKernelWgsl(P)).toContain('if (reI > diffGuard(p, i2, i1))');
    expect(vertexGatherWgsl()).toContain('fn tpVertexGather(');
    expect(vertexGatherWgsl()).not.toContain('0.5');
    expect(energyKernelWgsl(P)).toContain('fn tpEnergy(');
    expect(energyKernelWgsl(P)).toContain('partials[I] = 0.5 * acc;');
});
