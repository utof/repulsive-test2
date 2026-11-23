/**
 * Sample-median helpers shared by the bench harnesses.
 *
 * Two variants exist DELIBERATELY: the harnesses were written independently
 * and their stored results under `bench/results/` are keyed to the variant each
 * one used. They differ only for an EVEN sample count. Unifying them is a
 * re-baselining decision, not a cleanup — do not collapse these to one
 * function without reading the issue.
 * @issue utof/repulsive-test2#65
 * @see .superpowers/sdd/2026-09-11-phase2h-queue/reports/dry-audit.md (finding #1)
 */

/**
 * True median: odd length -> the middle element; even length -> the mean of
 * the two middle elements. Sorts a COPY (the caller's array is never mutated).
 * Empty input returns `NaN` (`(undefined + undefined) / 2`), as the original
 * `bench/sobolev.bench.ts` / `bench/gpu/spikes.ts` copies did.
 *
 * Why: `/ 2` is bit-identical to the original `0.5 *` spelling (exact power-of-two scaling in binary64).
 * Kept separate from {@link upperMedian} so stored bench results stay comparable.
 */
export function median(xs: number[]): number {
    const s = [...xs].sort((a, b) => a - b);
    const mid = Math.floor(s.length / 2);
    return s.length % 2 === 1 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

/**
 * UPPER median: element `floor(n / 2)` of the sorted sample, i.e. for an even
 * length the LARGER of the two middle elements (no averaging). Sorts a COPY.
 * Empty input returns `undefined` (cast to number), as the originals did.
 *
 * Why: `bench/gpu/phase1.ts` and `bench/gpu/jsAlgebra.ts` produced their stored
 * results with this variant; "fixing" it to {@link median} would shift
 * even-count values and break comparability with committed results JSON.
 */
export function upperMedian(xs: number[]): number {
    return [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
}
