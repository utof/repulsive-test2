// test/stats.test.ts — bench median helpers: two deliberate even-length semantics.
// @see bench/stats.ts
import { expect, test } from 'bun:test';
import { median, upperMedian } from '../bench/stats';

test('odd length: both variants return the middle element', () => {
    expect(median([5, 1, 3])).toBe(3);
    expect(upperMedian([5, 1, 3])).toBe(3);
});

test('even length: median averages the middle two, upperMedian takes the upper', () => {
    expect(median([4, 1, 3, 2])).toBe(2.5);
    expect(upperMedian([4, 1, 3, 2])).toBe(3);
});

test('single element', () => {
    expect(median([7])).toBe(7);
    expect(upperMedian([7])).toBe(7);
});

test('numeric (not lexicographic) sort', () => {
    expect(median([10, 9, 100])).toBe(10);
    expect(upperMedian([10, 9, 100, 2])).toBe(10);
});

test('input array is not mutated', () => {
    const xs = [3, 1, 2, 4];
    median(xs);
    upperMedian(xs);
    expect(xs).toEqual([3, 1, 2, 4]);
});
