/**
 * The `GPU_HOST` variant switch.
 *
 * Why a switch rather than a replacement: CLAUDE.md — "Prefer swappable
 * strategies over one hardcoded kernel … it keeps the losing branch cheap to
 * delete." Chrome is not the losing branch here; Dawn and naga disagree on
 * exactly the compensated arithmetic Phase 2H's precision argument rests on
 * (gpuweb/gpuweb#2076), so both hosts must stay runnable and every result must
 * record which one produced it.
 *
 * @issue utof/repulsive-test2#54
 */
import { chromeHost } from './chrome';
import { firefoxHost } from './firefox';
import type { BrowserHost } from './types';

/** Every selectable host, keyed by its `GPU_HOST` value. */
export const HOSTS: Record<BrowserHost['id'], BrowserHost> = {
    chrome: chromeHost,
    firefox: firefoxHost,
};

/**
 * Resolve a host id to its implementation.
 *
 * Defaults to `firefox` because that is the browser that exists on the dev box
 * — `bench/gpu/drive.ts` defaulting to `google-chrome` is precisely why every
 * GPU kill gate was unrunnable (#54). Set `GPU_HOST=chrome` on a box that has
 * Chrome. An unknown id throws rather than silently falling back, so a typo can
 * never quietly move a numerics measurement to the other shader backend.
 *
 * @issue utof/repulsive-test2#54
 */
export function selectHost(id: string | undefined): BrowserHost {
    const key = (id ?? 'firefox') as BrowserHost['id'];
    const host = HOSTS[key];
    if (!host) {
        throw new Error(`Unknown GPU_HOST '${id}'. Known hosts: ${Object.keys(HOSTS).join(', ')}`);
    }
    return host;
}
