/**
 * Known SwiftShader/software-fallback console noise lines to filter out of
 * captured console output (they fire even on a hardware adapter and are not
 * evidence of anything).
 *
 * Why its own module: both hosts (`hosts/chrome.ts`, `hosts/firefox.ts`) need
 * the same filter, and a per-host copy is how the `classifyAdapter` duplicate
 * in #42/#43 happened — one copy got a fix the other never saw.
 *
 * @see bench/gpu/README.md — "Console noise to ignore"
 * @issue utof/repulsive-test2#43
 */
const CONSOLE_NOISE = [
    /RangeError: createBuffer failed, size \(144\) is too large/,
    /Instance dropped in popErrorScope/,
    /Warning: Encountered two children with the same key, `0`/,
];

/** True when `line` is known-irrelevant console noise. */
export function isNoise(line: string): boolean {
    return CONSOLE_NOISE.some((re) => re.test(line));
}
