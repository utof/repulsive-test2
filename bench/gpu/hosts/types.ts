/**
 * The browser-host seam for the GPU gate harness.
 *
 * Why: `drive.ts` hardcoded Chrome-over-CDP, and there is no Chrome on the dev
 * box — every GPU kill gate was unrunnable. Rather than swap one hardcoded
 * browser for another, the host is a variant behind this interface so
 * `GPU_HOST=firefox|chrome` picks, per CLAUDE.md ("Prefer swappable strategies
 * over one hardcoded kernel").
 *
 * @issue utof/repulsive-test2#54
 * @see bench/gpu/README.md — "Hosts: Chrome/Dawn vs Firefox/naga"
 */

/**
 * A launch recipe: the argv, prefs and environment that one attempt uses.
 *
 * Chrome carries everything in `argv` (its `--enable-unsafe-webgpu` family);
 * Firefox needs `prefs` (written to the throwaway profile's `user.js`) and
 * `env` (the Vulkan ICD pin that decides *which* physical GPU is used). Both
 * are persisted into the result JSON so a recipe can be replayed exactly.
 *
 * Why `env` is load-bearing and not cosmetic: measured 2026-09-11, Firefox 155
 * on this box picks the Intel UHD 630 iGPU by default and never touches the
 * NVIDIA Quadro RTX 3000 (`nvidia-smi --query-compute-apps` lists no firefox
 * process without the pin, and lists one with it). Since Firefox blanks
 * `adapterInfo`, nothing in the payload would otherwise reveal which GPU ran.
 *
 * @issue utof/repulsive-test2#54
 */
export type LaunchVariant = {
    /** Human-readable name, recorded in the result JSON. */
    label: string;
    /** Command-line arguments, excluding the remote-control port and profile. */
    argv: string[];
    /** Browser preferences (Firefox `user.js`); empty for Chrome. */
    prefs: Record<string, boolean | number | string>;
    /** Extra environment for the browser process (e.g. the Vulkan ICD pin). */
    env: Record<string, string>;
};

/**
 * A live browser session pointed at `harness.html`, with the page's
 * `window.__runSpike` already resolved.
 *
 * Why this is the whole surface: `drive.ts` only ever needed three things from
 * CDP — evaluate an async expression and get JSON back, collect console lines,
 * and tear down. Everything else in the old driver (`/json` target discovery,
 * `Runtime.enable`, the id-keyed pending map) is transport detail that differs
 * per host and belongs below this line.
 *
 * @issue utof/repulsive-test2#54
 */
export type HostSession = {
    /** Evaluate `window.__runSpike(name)` in the page and parse its JSON result. */
    evalSpike(name: string): Promise<Record<string, unknown>>;
    /** Console lines captured during the session, noise-filtered by the host. */
    readonly consoleLines: string[];
    /** Anything the host learned out-of-band about the adapter (see `AdapterCertification`). */
    hostEvidence(): Promise<HostEvidence>;
    /** Close the transport and kill the browser process. */
    close(): void;
};

/**
 * Host-supplied evidence about *which* adapter the run actually used, beyond
 * the four `GPUAdapterInfo` identity strings.
 *
 * Why this exists: `classifyAdapter` rules a blank `GPUAdapterInfo` `'software'`
 * on purpose (`adapterClass.ts:39-47` — an adapter we cannot name is one we
 * cannot certify). Firefox blanks all four fields, so under that predicate every
 * Firefox gate is INVALID. The fix is not to relax the predicate but to give the
 * host a *separate, positively-evidenced* channel, recorded so a reader can see
 * exactly how hardware was certified.
 *
 * @issue utof/repulsive-test2#53
 * @issue utof/repulsive-test2#54
 */
export type HostEvidence = {
    /**
     * `GPUAdapterInfo.isFallbackAdapter` as reported by the page.
     * Measured 2026-09-11 on Firefox 155: `false` under the NVIDIA and Intel
     * ICDs, `true` under llvmpipe — i.e. it does discriminate CPU adapters, and
     * is the knock-out lever the Firefox certification path relies on.
     * `undefined` when the host could not read it.
     */
    isFallbackAdapter?: boolean;
    /**
     * Out-of-band identification of the GPU the browser was pinned to, read
     * from `vulkaninfo` under the same ICD environment the browser got.
     * `null` when no pin was applied (so the adapter is whatever the loader
     * picked, and is therefore un-identified).
     */
    devicePin: DevicePin | null;
};

/**
 * The identity of a Vulkan physical device, read out-of-band under a pinned
 * ICD. `deviceType` is verbatim from `vulkaninfo --summary`.
 * @see bench/gpu/vulkanPin.ts
 */
export type DevicePin = {
    /** Absolute path of the ICD manifest the browser was pinned to. */
    icdPath: string;
    /** `deviceName` from `vulkaninfo --summary`, e.g. "Quadro RTX 3000". */
    deviceName: string;
    /** `deviceType`, e.g. `PHYSICAL_DEVICE_TYPE_DISCRETE_GPU` / `..._CPU`. */
    deviceType: string;
};

/**
 * One selectable browser host.
 *
 * @issue utof/repulsive-test2#54
 */
export type BrowserHost = {
    readonly id: 'chrome' | 'firefox';
    /**
     * The WGSL compiler backend.
     *
     * Why recorded: WGSL permits reassociation and fusion with no portable
     * opt-out (gpuweb/gpuweb#2076, open since 2021), and naga is *measured* to
     * delete Dekker compensation terms that Dawn preserves. A numerics gate
     * measured on one backend is not evidence about the other, so every result
     * record carries this field.
     * @see https://github.com/gpuweb/gpuweb/issues/2076
     * @issue utof/repulsive-test2#51 trap 3
     */
    readonly shaderBackend: 'dawn' | 'naga';
    /** The browser binary this host will spawn. */
    readonly binary: string;
    /** `<name> <version>` for the result record, or `'unknown'` if unreadable. */
    version(): string;
    /** Launch recipes tried in order by the G0a adapter gate. */
    readonly launchVariants: LaunchVariant[];
    /**
     * Apply host-level environment overrides to a recipe before it is launched
     * *and* before it is recorded.
     *
     * Why the hook exists: a non-G0a gate replays the launch recipe from the
     * last passing G0a run, so an operator override such as `GPU_VK_ICD` would
     * otherwise apply to the G0a gate and be silently dropped by every gate
     * after it. Resolving here means the recipe written into the result JSON is
     * the recipe that actually ran — a stale `launch.env` would be false
     * provenance of exactly the kind this whole seam exists to prevent.
     * @issue utof/repulsive-test2#54
     */
    resolveVariant(variant: LaunchVariant): LaunchVariant;
    /** Launch the browser on `url` and return a ready session. */
    launch(variant: LaunchVariant, url: string, port: number): Promise<HostSession>;
};
