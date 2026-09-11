/**
 * The single software-rasterizer predicate for the GPU gate harness.
 *
 * Kept dependency-free and in its own module so it can be unit-tested:
 * `drive.ts` calls `main()` at import (`drive.ts:435`), so nothing inside it
 * is reachable from a test without launching the whole driver. That is the
 * dedupe #43 asks for; `feat/webgpu-phase2` solved the same testability
 * problem differently (export from `drive.ts` behind `import.meta.main`), so
 * expect a conflict when these branches meet — the four-field predicate below
 * is the part that must survive either way.
 *
 * @issue utof/repulsive-test2#43
 */

/**
 * The `GPUAdapterInfo` fields the classification reads. All optional: Firefox
 * (naga/wgpu) reports a blank `adapterInfo`, and the record is built from an
 * untyped `evalSpike` result, so absent fields are reachable at runtime.
 */
export type AdapterIdentity = {
    vendor?: string;
    architecture?: string;
    device?: string;
    description?: string;
};

/**
 * PASS/FAIL classification of an adapter, per spec §3: any result obtained
 * against a software rasterizer is INVALID, never a passing (or failing) gate
 * measurement.
 *
 * **Matches on all four fields, not vendor+description.** Chrome's SwiftShader
 * adapter leaves `description` empty and puts 'swiftshader' in `architecture`
 * only, so a vendor+description concat labels it `hardware` and hands spec §4
 * G0a a PASS on a software rasterizer. That is a probed observation, not a
 * precaution — see `src/gpu/selfTest.ts` `adapterIsSoftware` on
 * `feat/webgpu-phase2` for the fixture table.
 *
 * **Anything that does not positively identify itself is `'software'`.** An
 * adapter we cannot name is one we cannot certify as hardware, and a gate
 * measured against an uncertified adapter has no provenance. Firefox reports a
 * blank `adapterInfo`, and blank strings match none of the markers — so a pure
 * substring test falls through to `'hardware'` and silently records a
 * software-fallback run as a valid hardware gate. This matches the ruling
 * `drive.ts` already applies to the adapter it could not determine at all
 * ("genuinely unknown, treated as 'software' so it never masquerades as a
 * passing hardware run").
 *
 * Still substring-matches rather than reading
 * `GPUAdapterInfo.isFallbackAdapter`; adopting that boolean would change the
 * G0a results payload and stays deferred.
 *
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §3
 * @issue utof/repulsive-test2#24
 * @issue utof/repulsive-test2#42
 * @issue utof/repulsive-test2#53
 */
export function classifyAdapter(info: AdapterIdentity): 'hardware' | 'software' {
    const d =
        `${info?.vendor ?? ''} ${info?.architecture ?? ''} ${info?.device ?? ''} ${info?.description ?? ''}`.toLowerCase();
    if (d.trim() === '') return 'software';
    return d.includes('swiftshader') || d.includes('llvmpipe') || d.includes('software')
        ? 'software'
        : 'hardware';
}
