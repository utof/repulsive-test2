/**
 * The routing rules of `src/core/sobolevRoute.ts`. These are decisions, not numerics: each case
 * below is one sentence of the module's TSDoc turned into an assertion.
 */
import { describe, expect, test } from 'bun:test';
import {
    HIERARCHICAL_AUTO_MIN_N,
    isClosedNaturalRing,
    PAIR_LIST_MAX_N,
    pairListFits,
    resolveSobolevRoute,
} from '../../src/core/sobolevRoute';
import type { Edge } from '../../src/core/testConfigs';

const ring = (n: number): Edge[] => Array.from({ length: n }, (_, i) => [i, (i + 1) % n] as Edge);
const chain = (n: number): Edge[] => Array.from({ length: n - 1 }, (_, i) => [i, i + 1] as Edge);

describe('isClosedNaturalRing', () => {
    test('accepts a ring in natural order', () => {
        expect(isClosedNaturalRing(5, ring(5))).toBe(true);
    });
    test('accepts the same ring with its edges listed in another order', () => {
        expect(isClosedNaturalRing(5, [...ring(5)].reverse())).toBe(true);
    });
    test('rejects an open chain, a duplicated edge, a reversed edge and a chord', () => {
        expect(isClosedNaturalRing(5, chain(5))).toBe(false);
        const dup = ring(5);
        dup[1] = [0, 1];
        expect(isClosedNaturalRing(5, dup)).toBe(false);
        const rev = ring(5);
        rev[2] = [3, 2];
        expect(isClosedNaturalRing(5, rev)).toBe(false);
        const chord = ring(5);
        chord[4] = [4, 2];
        expect(isClosedNaturalRing(5, chord)).toBe(false);
    });
    test('rejects fewer than three vertices', () => {
        expect(
            isClosedNaturalRing(2, [
                [0, 1],
                [1, 0],
            ] as Edge[]),
        ).toBe(false);
    });
});

describe('pairListFits', () => {
    test('switches exactly at PAIR_LIST_MAX_N', () => {
        expect(pairListFits(PAIR_LIST_MAX_N)).toBe(true);
        expect(pairListFits(PAIR_LIST_MAX_N + 1)).toBe(false);
    });
});

describe('resolveSobolevRoute', () => {
    const small = HIERARCHICAL_AUTO_MIN_N - 1;
    const mid = HIERARCHICAL_AUTO_MIN_N;
    const big = PAIR_LIST_MAX_N + 1;

    test("'auto' switches a closed ring to hierarchical at HIERARCHICAL_AUTO_MIN_N", () => {
        expect(resolveSobolevRoute('auto', small, ring(small), true)).toBe('dense');
        expect(resolveSobolevRoute('auto', mid, ring(mid), true)).toBe('hierarchical');
    });
    test('an explicit preference is honoured when that route can run', () => {
        expect(resolveSobolevRoute('dense', mid, ring(mid), true)).toBe('dense');
        expect(resolveSobolevRoute('hierarchical', small, ring(small), true)).toBe('hierarchical');
    });
    test('hierarchical falls back to dense for an open chain and for finite differences', () => {
        expect(resolveSobolevRoute('hierarchical', mid, chain(mid), true)).toBe('dense');
        expect(resolveSobolevRoute('hierarchical', mid, ring(mid), false)).toBe('dense');
        expect(resolveSobolevRoute('auto', mid, chain(mid), true)).toBe('dense');
    });
    test('above the pair-list limit a ring is hierarchical whatever the preference', () => {
        expect(resolveSobolevRoute('dense', big, ring(big), true)).toBe('hierarchical');
        expect(resolveSobolevRoute('auto', big, ring(big), true)).toBe('hierarchical');
    });
    test('above the pair-list limit anything else has no route', () => {
        expect(resolveSobolevRoute('auto', big, chain(big), true)).toBeNull();
        expect(resolveSobolevRoute('hierarchical', big, ring(big), false)).toBeNull();
    });
});
