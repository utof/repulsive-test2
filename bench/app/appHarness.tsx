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
 * Spike name: `prepass:<nodes>:<edges>:<steps>` — see `prePass` below.
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

/**
 * Spike `prepass:<nodes>:<edges>:<steps>`: the random-graph preset, Sobolev descent, played
 * twice from the SAME scrambled start — once plainly, once with the force-layout pre-pass — so
 * the frame loop's handling of the pre-pass phase is exercised in a real browser.
 * @see src/core/forceLayout.ts
 */
async function prePass(nodes: number, edges: number, steps: number) {
    const store = useSimStore;
    store.getState().setPreset('random');
    store.getState().setParam('vertices', nodes);
    store.getState().setParam('edges', edges);
    store.getState().regenerate();
    store.getState().setDescentMode('sobolev');
    await sleep(1500);
    const start = store.getState().live.map((p) => [...p] as [number, number, number]);
    const energyAtLoad = store.getState().energy;

    const arm = async (withPrePass: boolean) => {
        store.getState().setRunning(false);
        const live = store.getState().live;
        start.forEach((p, i) => {
            live[i][0] = p[0];
            live[i][1] = p[1];
            live[i][2] = p[2];
        });
        store.getState().setForcePrePass(withPrePass);
        store.getState().setRunning(false); // commit the restored start
        await sleep(100);
        const step0 = store.getState().step;
        let prePassMs: number | null = null;
        let prePassFrames = 0;
        let energyAfterPrePass: number | null = null;
        let counting = true;
        const tick = () => {
            if (store.getState().prePass) prePassFrames++;
            if (counting) requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
        const t0 = performance.now();
        store.getState().setRunning(true);
        const sawPrePass = store.getState().prePass !== null;
        while (performance.now() - t0 < 120_000) {
            const s = store.getState();
            if (sawPrePass && prePassMs === null && s.prePassDone) {
                prePassMs = performance.now() - t0;
                energyAfterPrePass = s.energy;
            }
            if (s.step - step0 >= steps || !s.running) break;
            await sleep(10);
        }
        counting = false;
        const end = store.getState();
        const out = {
            withPrePass,
            sawPrePass,
            prePassMs: prePassMs === null ? null : Math.round(prePassMs),
            prePassFrames,
            energyAfterPrePass,
            stepsTaken: end.step - step0,
            energyAtEnd: end.energy,
            elapsedMs: Math.round(performance.now() - t0),
            stillRunning: end.running,
            reason: end.sobolevStats?.reason ?? null,
            lengthDrift: Math.abs(
                end.live.length > 0 && end.sobolevL0 > 0
                    ? end.graph.edges.reduce(
                          (sum, [a, b]) =>
                              sum +
                              Math.hypot(
                                  end.live[a][0] - end.live[b][0],
                                  end.live[a][1] - end.live[b][1],
                                  end.live[a][2] - end.live[b][2],
                              ),
                          0,
                      ) /
                          end.sobolevL0 -
                          1
                    : 0,
            ),
        };
        end.setRunning(false);
        return out;
    };
    const plain = await arm(false);
    const withForce = await arm(true);
    store.getState().setForcePrePass(false);
    return { nodes, edges, energyAtLoad, plain, withForce };
}

declare global {
    interface Window {
        __runSpike: (name: string) => Promise<string>;
    }
}
window.__runSpike = async (name) => {
    const [kind, n, steps, driver] = name.split(':');
    if (kind === 'prepass')
        return JSON.stringify(await prePass(Number(n), Number(steps), Number(driver)));
    if (kind !== 'run') throw new Error(`appHarness: unknown spike '${name}'`);
    return JSON.stringify(await run(Number(n), Number(steps), driver as SolverDriver | undefined));
};
