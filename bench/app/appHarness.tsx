/**
 * The REAL app (`src/App.tsx`, its store, its Viewer, its worker) plus a `window.__runSpike` hook,
 * so a headless browser can drive it the way `bench/gpu/harness.html` drives the GPU spikes.
 *
 * Why one entry point for both: the dev server bundles each requested file separately, so a
 * driver module loaded next to `src/index.tsx` would get its OWN copy of the store. Rendering the
 * app from here keeps the store this hook writes to the one the app reads.
 *
 * Spike name: `run:<vertices>:<steps>[:<driver>]`, e.g. `run:30000:3:worker`. It selects the large
 * trefoil preset, switches to Sobolev descent, presses play, and reports what the frame loop did.
 * @see bench/app/driveApp.ts
 */
import { createRoot } from 'react-dom/client';
import { App } from '../../src/App';
import { resolveSobolevRoute } from '../../src/core/sobolevRoute';
import { type SolverDriver, useSimStore } from '../../src/store';

const root = document.getElementById('root');
if (!root) throw new Error('appHarness: #root is missing');
createRoot(root).render(<App />);

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function run(vertices: number, steps: number, driver: SolverDriver | undefined) {
    const store = useSimStore;
    const tBuild0 = performance.now();
    store.getState().setPreset('knot-large');
    store.getState().setParam('vertices', vertices);
    store.getState().regenerate();
    const buildMs = performance.now() - tBuild0;
    store.getState().setDescentMode('sobolev');
    if (driver) store.getState().setSolverDriver(driver);
    // Let React and the Viewer's effects (worker construction, renderer init) settle.
    await sleep(1500);

    const st0 = store.getState();
    const route = resolveSobolevRoute(
        st0.sobolevSolver,
        st0.graph.vertices.length,
        st0.graph.edges,
        st0.mode === 'analytical',
    );
    const energyAtLoad = st0.energy;
    const pairListRows = st0.disjointPairs.length;

    // Frame counter: with the step on the worker the UI thread should keep rendering.
    let frames = 0;
    let counting = true;
    const tick = () => {
        frames++;
        if (counting) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);

    const stepLog: { step: number; atMs: number; energy: number }[] = [];
    let lastStep = st0.step;
    const t0 = performance.now();
    const unsub = store.subscribe((s) => {
        if (s.step !== lastStep) {
            lastStep = s.step;
            stepLog.push({ step: s.step, atMs: performance.now() - t0, energy: s.energy });
        }
    });
    store.getState().setRunning(true);
    const deadline = t0 + 600_000;
    while (performance.now() < deadline) {
        const s = store.getState();
        if (s.step - st0.step >= steps || !s.running) break;
        await sleep(50);
    }
    const elapsedMs = performance.now() - t0;
    counting = false;
    unsub();
    const end = store.getState();
    const result = {
        vertices: end.graph.vertices.length,
        requestedSteps: steps,
        stepsTaken: end.step - st0.step,
        stillRunning: end.running,
        route,
        solverDriverAtEnd: end.solverDriver,
        pairListRows,
        buildMs: Math.round(buildMs),
        energyAtLoad,
        stepLog,
        elapsedMs: Math.round(elapsedMs),
        framesDuringRun: frames,
        fpsDuringRun: Math.round((frames / elapsedMs) * 1000),
        sobolevStats: end.sobolevStats,
        sobolevTimings: end.sobolevTimings,
        sobolevConverged: end.sobolevConverged,
    };
    end.setRunning(false);
    return result;
}

declare global {
    interface Window {
        __runSpike: (name: string) => Promise<string>;
    }
}
window.__runSpike = async (name) => {
    const [kind, n, steps, driver] = name.split(':');
    if (kind !== 'run') throw new Error(`appHarness: unknown spike '${name}'`);
    return JSON.stringify(await run(Number(n), Number(steps), driver as SolverDriver | undefined));
};
