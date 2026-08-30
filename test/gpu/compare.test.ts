// test/gpu/compare.test.ts — comparators + seeded Math.random helper.
// @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (Task 3, comparators)
import { expect, test } from 'bun:test';
import { cosineComparator, relErrComparator, withSeed } from '../../bench/gpu/compare';

test('comparators + withSeed', () => {
    expect(relErrComparator(1.00001, 1)).toBeCloseTo(1e-5, 10);
    expect(cosineComparator([1, 0, 0], [1, 0, 0])).toBe(1);
    expect(cosineComparator([0, 0], [0, 0])).toBe(1);
    expect(cosineComparator([1, 0], [0, 0])).toBe(-1);
    expect(() => cosineComparator([1], [1, 2])).toThrow();
    const orig = Math.random;
    const a = withSeed(7, () => [Math.random(), Math.random()]);
    const b = withSeed(7, () => [Math.random(), Math.random()]);
    expect(a).toEqual(b);
    expect(Math.random).toBe(orig); // restored
});

test('withSeed restores Math.random even when the callback throws', () => {
    const orig = Math.random;
    expect(() =>
        withSeed(1, () => {
            throw new Error('boom');
        }),
    ).toThrow('boom');
    expect(Math.random).toBe(orig);
});
