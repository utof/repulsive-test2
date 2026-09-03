import { OrbitControls } from '@react-three/drei';
import {
    Canvas,
    extend,
    type ThreeElement,
    type ThreeToJSXElements,
    useFrame,
    useThree,
} from '@react-three/fiber';
import { useCallback, useEffect, useRef } from 'react';
// Task 10 (fat WebGPU edges): these live in three's addons tree, not the `three/webgpu`
// namespace, so `extend(THREE as any)` below doesn't register them — hence the separate
// `extend({...})` call. WebGPU-safe variants only (webgpu/LineSegments2, not the WebGL-only
// examples/jsm/lines/LineSegments2). @see docs/superpowers/specs/2026-07-02-react-three-webgpu-switch-design.md Task 10.
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineSegments2 } from 'three/addons/lines/webgpu/LineSegments2.js';
import * as THREE from 'three/webgpu';
import { GpuDriver, gpuStepSupported } from '../gpu/driver';
import { runGpuSelfTest } from '../gpu/selfTest';
import {
    buildStepArgs,
    type DescentStepOutcome,
    dispatchDescentStep,
    type SolverWorkerRequest,
    type SolverWorkerResponse,
    useSimStore,
} from '../store';
import { Curve } from './Curve';
import { GradientArrows } from './GradientArrows';
import { PinControls } from './PinControls';

declare module '@react-three/fiber' {
    // Task 10: LineSegments2/LineSegmentsGeometry live in three/addons, so `ThreeToJSXElements<typeof
    // THREE>` (which only walks the three/webgpu namespace) doesn't cover the <lineSegments2>/
    // <lineSegmentsGeometry>/<line2NodeMaterial> intrinsics registered by extend({...}) below —
    // declare them by hand, mirroring how ThreeToJSXElements derives element props.
    interface ThreeElements extends ThreeToJSXElements<typeof THREE> {
        lineSegments2: ThreeElement<typeof LineSegments2>;
        lineSegmentsGeometry: ThreeElement<typeof LineSegmentsGeometry>;
        line2NodeMaterial: ThreeElement<typeof THREE.Line2NodeMaterial>;
    }
}
// `as any`: registering the whole three/webgpu namespace as R3F intrinsics; the cast is the
// spec-sanctioned bridge for the NodeMaterial/WebGPU classes under strict. @see spec §6.
extend(THREE as any);
// Task 10: register the fat-line classes as R3F intrinsics (<lineSegments2>, <lineSegmentsGeometry>,
// <line2NodeMaterial>). Line2NodeMaterial ships in the three/webgpu namespace (already imported
// as THREE.*), unlike LineSegments2/LineSegmentsGeometry which live in three/addons.
extend({ LineSegments2, LineSegmentsGeometry, Line2NodeMaterial: THREE.Line2NodeMaterial });

const CAMERA_POS: [number, number, number] = [3, 3, 3];
// zoom is reported as a ratio normalized to the starting camera distance (zoom=1 at spawn);
// camera.position.length() assumes the orbit target is the origin (true for default OrbitControls).
// Moving CAMERA_POS or setting a non-origin target changes the zoom baseline. @see spec §8 (zoom stat).
const BASE_DISTANCE = Math.hypot(...CAMERA_POS);

// The optimization loop. Lives inside <Canvas> so it can use useFrame. Mutates the
// live buffer in place (no React render); publishes stats/zoom throttled (~10Hz / 5Hz).
// @see docs/superpowers/specs/2026-07-02-react-three-webgpu-switch-design.md §4.1 (Simulation)
function Simulation() {
    const statAcc = useRef(0);
    const camAcc = useRef(0);
    const iters = useRef(0);
    // E₀ reuse across steps (Task 4): the previous ACCEPTED step's returned
    // energy, which is exactly calculateEnergy(the vertices that become this
    // step's input), so reusing it as the next step's Armijo baseline is
    // bit-identical within a continuous run. Nulled at every !running boundary
    // below (run start, user pause, auto-pause, preset rebuild — the store
    // re-anchors targets at those same boundaries), making a stale E₀ — which
    // would corrupt the Armijo gate — structurally impossible.
    // @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 4)
    const lastEnergy = useRef<number | null>(null);
    // The store's penaltyEpoch as of the last frame. A MID-RUN penalty-config
    // change (slider/X move) bumps the store nonce but crosses no !running
    // boundary, so the boundary-null below never fires — the cached E₀ would
    // stay under the OLD objective and corrupt the Armijo gate (silent wrong
    // step). When the epoch changes we drop the cache, forcing a fresh E₀ under
    // the NEW config. @see docs/superpowers/plans/2026-07-03-sobolev-penalties.md §2.4
    const lastPenaltyEpoch = useRef(0);
    // Worker-driver state (§D1/§D4/§D5). workerRef: the live worker (null in main
    // mode / before construction). inFlight: the §D1 single-flight guard — a TOKEN
    // rather than a boolean, held between a step SEND and its result, so at most one
    // step is ever outstanding. `null` means idle, so every `!inFlight.current` idle
    // check reads exactly as the old `=== false` did; the token additionally records
    // WHICH backend owns the outstanding step and the dispatchGeneration it was sent
    // under, so a landing result can be matched against the epoch that produced it.
    // sentTopologyVersion: the graphVersion whose topology the worker already has
    // cached (§D4); -1 forces a resend on a fresh worker. lastResultTime: wall
    // clock of the last applied worker result — the worker path has no frame delta
    // so it accumulates elapsed time for the same ~10 Hz stat throttle.
    // @see docs/superpowers/plans/2026-07-04-worker-solver.md §D1, §D4, §D5
    // @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md — "Task 7: Viewer + ControlPanel integration"
    const workerRef = useRef<Worker | null>(null);
    const inFlight = useRef<null | { backend: 'gpu' | 'worker'; generation: number }>(null);
    const sentTopologyVersion = useRef(-1);
    const lastResultTime = useRef(0);
    // Monotonic dispatch epoch. Bumped by the solver-lifecycle effect below BEFORE
    // any early return, so every driver flip / worker rebuild opens a new epoch; a
    // result carrying an older generation is dropped instead of applied and may not
    // clear the current token. This is what makes an in-flight step safe to abandon
    // without cancelling it (a posted Worker message can't be recalled, and a GPU
    // readback's `mapAsync` can't either).
    // @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md — "Task 7: Viewer + ControlPanel integration"
    const dispatchGeneration = useRef(0);

    // Subscribed (not getState) so the worker-lifecycle effect re-runs when the
    // driver flips — the per-frame branch below reads the fresh getState value.
    const solverDriver = useSimStore((s) => s.solverDriver);
    // 'gpu' driver state. gl: the Viewer's ONE WebGPURenderer — the device the solver
    // shares with rendering, never a second adapter (R3F types state.gl as a
    // WebGLRenderer, hence the cast; the gl factory below is what constructs it).
    // gpuDriverRef: the live GpuDriver, non-null only while 'gpu' is selected AND the
    // boot self-test passed. gpuAvailable is SUBSCRIBED so the GPU lifecycle effect
    // re-runs when the boot self-test lands, or when device loss / an uncaptured
    // error clears it mid-run (fallback gpu → worker).
    // @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md — "2.1 Device, thread, and library surface"
    // @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md — "2.6 Driver integration"
    const gl = useThree((s) => s.gl) as unknown as THREE.WebGPURenderer;
    const gpuDriverRef = useRef<GpuDriver | null>(null);
    const gpuAvailable = useSimStore((s) => s.gpuAvailable);

    // §D8: the SINGLE result-application path shared by both drivers — in-place
    // `live` mutation, E₀ caching, target-length schedule advance on accepted
    // steps, the throttled ~10 Hz stats publish (via elapsed-time accumulation),
    // and auto-pause on a rejected/converged step. The main driver calls it inline
    // with the frame delta; the worker driver calls it from onmessage with the
    // wall-clock elapsed since the last result. Reads the store fresh so it always
    // targets the CURRENT live buffer. Behavior mirrors the pre-worker Simulation
    // branch-for-branch. Deps [] — it only touches stable refs + the store
    // singleton, so its identity is stable (keeps the worker effect from churning).
    // @see docs/superpowers/plans/2026-07-04-worker-solver.md §D8
    const applyStepOutcome = useCallback((result: DescentStepOutcome, elapsed: number) => {
        const st = useSimStore.getState();
        if (result.accepted) {
            // Copy-then-discard: the stepper returns a fresh Vec3[]; we mutate the
            // existing live tuples in place (preserving their identity for Curve's
            // non-reactive buffer) and drop `vertices`. @see spec §5.
            for (let i = 0; i < st.live.length; i++) {
                const v = result.vertices[i];
                const l = st.live[i];
                l[0] = v[0];
                l[1] = v[1];
                l[2] = v[2];
            }
            // Cache E₀ for the next step: result.energy is calculateEnergy at
            // exactly the vertices just copied into `live` (this step's output =
            // next step's input), so next step's reuse is bit-identical.
            // @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 4)
            lastEnergy.current = result.energy;
            // Target-length animation (paper tex line 760; plan §4 Task 5): advance
            // the frozen length schedule ONLY on ACCEPTED steps and ONLY when growth
            // is enabled — rate 1.0 leaves the store untouched so default runs are
            // bit-identical. @see docs/superpowers/plans/2026-07-03-sobolev-penalties.md §4 Task 5
            if (st.lengthGrowthRate !== 1) st.advanceLengthSchedule();
            iters.current += 1;
            statAcc.current += elapsed;
            if (statAcc.current > 0.1) {
                statAcc.current = 0;
                useSimStore.setState({
                    step: iters.current,
                    energy: result.energy,
                    sobolevStats: result.stats,
                    sobolevTimings: result.timings,
                    // §D14: publish the field this accepted step used so GradientArrows
                    // renders it directly (issue #9); null when collectField was off or
                    // the saddle was singular. @see …worker-solver.md §D14
                    arrowField: result.descentField,
                });
            }
        } else {
            // Sobolev converged (spec §C step 5) or rejected the step (spec §C step
            // 10: leave vertices unchanged, report — never throw): auto-pause instead
            // of spinning. Publish diagnostics un-throttled first; setRunning(false)
            // commits live positions and deliberately preserves them (see store).
            // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §C (steps 5, 10)
            useSimStore.setState({
                step: iters.current,
                sobolevStats: result.stats,
                sobolevConverged: result.converged,
                sobolevTimings: result.timings,
                // §D14: publish the field even on a rejected/converged step (a rejected
                // line search still solved g̃; singular → null hides the arrows). This
                // is the unthrottled boundary block. @see …worker-solver.md §D14
                arrowField: result.descentField,
            });
            st.setRunning(false);
        }
    }, []);

    // Worker→main protocol handler (§D5/§D6). An `error` clears the §D1 single-flight
    // guard unconditionally; a `result` clears it ONLY when it matches the outstanding
    // token of the current dispatch epoch (see `owned` below) — an unowned result must
    // leave the token alone, so do not "simplify" this back to an unconditional clear.
    // @see docs/superpowers/plans/2026-07-04-worker-solver.md §D5, §D6
    const handleWorkerMessage = useCallback(
        (resp: SolverWorkerResponse) => {
            if (resp.type === 'error') {
                // §D6 auto-fallback: a worker-side throw drops us to the synchronous
                // main driver so the solver keeps running (today's jank, but correct).
                console.error(
                    'solverDriver: worker posted an error; falling back to main:',
                    resp.message,
                );
                inFlight.current = null;
                useSimStore.getState().setSolverDriver('main');
                return;
            }
            // The step worker only ever posts `result` (arrows `fieldResult` is a
            // SEPARATE worker instance in GradientArrows, §D13-c); narrow it out so
            // the union's field variant can't reach the step-application path.
            if (resp.type !== 'result') return;
            // Token match: this response may clear the single-flight guard — and be
            // applied — only if it IS the outstanding worker step of the CURRENT
            // dispatch epoch. After a driver flip / worker rebuild the epoch is bumped,
            // so a response still in flight from the old epoch must neither clear the
            // new epoch's token nor reach `live`. During a steady worker run `owned`
            // is always true, so the pre-token behavior is unchanged.
            // @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md — "Task 7: Viewer + ControlPanel integration"
            const token = inFlight.current;
            const owned =
                token?.backend === 'worker' && token.generation === dispatchGeneration.current;
            if (owned) inFlight.current = null;
            const st = useSimStore.getState();
            // §D5: DROP a result whose topology (graphVersion) no longer matches the
            // store, or that landed after a pause — applying it would mutate committed
            // or foreign buffers. Safe because the E₀ cache is nulled at those same
            // boundaries, so no stale energy survives the drop. The generation is part
            // of the same drop rule (and of the message) so a dropped result says which
            // epoch it came from. The DRIVER is part of the rule too: a result landing
            // after a programmatic flip to 'main' (e.g. from inside `onerror`, whose
            // passive-effect teardown is not flushed synchronously) but before the
            // effect cleanup nulls `onmessage` would otherwise pass gv+running while
            // the main driver is already stepping `live`. Mirrors the GPU branch's
            // `solverDriver !== 'gpu'` check; 'gpu' is NOT dropped here because the
            // worker legitimately serves per-step fallbacks under it (spec §1).
            // @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md — "Task 7: Viewer + ControlPanel integration"
            if (
                !owned ||
                resp.graphVersion !== st.graphVersion ||
                !st.running ||
                st.solverDriver === 'main'
            ) {
                const reason = !owned
                    ? 'epoch'
                    : st.solverDriver === 'main'
                      ? 'driver'
                      : 'gv/running';
                console.warn(
                    `solverDriver: dropping stale worker result (reason=${reason}, gv ${resp.graphVersion} vs ${st.graphVersion}, running=${st.running}, driver=${st.solverDriver}, token ${token?.backend ?? 'none'}/${token?.generation ?? 'none'} vs gen ${dispatchGeneration.current})`,
                );
                return;
            }
            // No frame delta off-thread: accumulate wall-clock between results for
            // the same ~10 Hz throttle the main path gets from `delta`.
            const now = performance.now();
            const elapsed = (now - lastResultTime.current) / 1000;
            lastResultTime.current = now;
            applyStepOutcome(resp.result, elapsed);
        },
        [applyStepOutcome],
    );

    // Worker lifecycle (§D3/§D6): construct in worker mode, tear down on driver
    // flip / unmount. Auto-fallback to 'main' on construction failure or an
    // uncaught worker error. @see docs/superpowers/plans/2026-07-04-worker-solver.md §D3, §D6
    useEffect(() => {
        // FIRST two statements, BEFORE any early return: the bump is what invalidates
        // results still in flight from the previous epoch. It must precede the
        // `needsWorker` return, because a flip to 'main' returns early and would
        // otherwise leave an outstanding worker result free to land on the buffer the
        // main driver is already stepping. Clearing the token also re-opens the
        // single-flight gate for the new epoch.
        // @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md — "Task 7: Viewer + ControlPanel integration"
        dispatchGeneration.current++;
        inFlight.current = null;
        // The worker is kept alive under 'gpu' as well: 'gpu' serves sobolev +
        // analytical + penalties-off only, and any other config forces a PER-STEP
        // fallback to 'worker' without changing the selected driver; only 'main'
        // tears the worker down. (The driver-level chain gpu → worker → main is
        // spec §2.6; the per-step fallback that keeps the worker alive is spec §1.)
        // @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md — "1. Scope"
        // @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md — "2.6 Driver integration"
        const needsWorker = solverDriver !== 'main';
        if (!needsWorker) return;
        let worker: Worker;
        try {
            // §D3: dev-server PATH STRING, NOT new URL(import.meta.url) — Bun.build
            // emits no worker chunk for browser targets, so the plain string is
            // load-bearing; the dev server bundles any requested .ts on the fly
            // (server.ts, the `path.endsWith('.ts')` branch).
            // @see docs/superpowers/plans/2026-07-04-worker-solver.md §D3
            worker = new Worker('/src/worker/solverWorker.ts', { type: 'module' });
        } catch (err) {
            console.error('solverDriver: worker construction failed; falling back to main:', err);
            useSimStore.getState().setSolverDriver('main');
            return;
        }
        worker.onmessage = (event: MessageEvent<SolverWorkerResponse>) =>
            handleWorkerMessage(event.data);
        worker.onerror = (event: ErrorEvent) => {
            console.error('solverDriver: worker error; falling back to main:', event.message);
            inFlight.current = null;
            useSimStore.getState().setSolverDriver('main');
        };
        workerRef.current = worker;
        // Fresh worker: no in-flight step, no topology cached yet (force a resend
        // next frame), and seed the worker-path stat clock.
        inFlight.current = null;
        sentTopologyVersion.current = -1;
        lastResultTime.current = performance.now();
        return () => {
            // Null onmessage BEFORE terminate: a result already queued on this
            // worker must not fire after teardown — with the driver now flipped it
            // could still pass the §D5 gv+running gate while the main driver has
            // already stepped, double-applying to `live`.
            // @see docs/superpowers/plans/2026-07-04-worker-solver.md §D5 (drop-rule boundary)
            worker.onmessage = null;
            worker.terminate();
            workerRef.current = null;
            inFlight.current = null;
        };
    }, [solverDriver, handleWorkerMessage]);

    // GPU driver lifecycle (fallback chain gpu → worker → main): construct the
    // GpuDriver on the Viewer's renderer when 'gpu' is selected and the boot
    // self-test passed; if 'gpu' is selected while unavailable (self-test failed,
    // device lost, uncaptured error — all of which clear gpuAvailable) fall back to
    // 'worker' — the UI keeps the option disabled, this is the belt-and-braces path.
    // Cleanup disposes the driver (its buffers), nulls the ref and clears the
    // single-flight token. A step still awaiting readback then resolves or REJECTS
    // against a dead engine (GpuDriver.step contract); both the .then and the .catch
    // in useFrame recognise it as superseded via the token/epoch checks plus
    // `gpuDriverRef.current !== drv`, so a dispose we initiated never flips the
    // driver or clears gpuAvailable. The epoch itself is bumped by the worker
    // effect above on every solverDriver change (the gpuAvailable→false path flips
    // the driver here, so it ends up bumped too).
    // @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md — "2.6 Driver integration"
    // @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md — "6. Risks not covered by gates" (mid-run device loss)
    // @see src/gpu/driver.ts — GpuDriver.step TSDoc (single-flight / dispose-mid-flight contract)
    useEffect(() => {
        if (solverDriver !== 'gpu') return;
        if (!gpuAvailable) {
            useSimStore.getState().setSolverDriver('worker');
            return;
        }
        const drv = new GpuDriver(gl);
        gpuDriverRef.current = drv;
        return () => {
            drv.dispose();
            gpuDriverRef.current = null;
            inFlight.current = null;
        };
    }, [solverDriver, gl, gpuAvailable]);

    useFrame((state, delta) => {
        const st = useSimStore.getState();

        if (st.running) {
            // Penalty-config change since last frame ⇒ invalidate the reused E₀
            // (see the lastPenaltyEpoch anchor). Checked before each SEND on BOTH
            // drivers (§2). Constraint-target animation does NOT bump the epoch
            // (targets aren't in the objective, plan §2.4).
            if (st.penaltyEpoch !== lastPenaltyEpoch.current) {
                lastPenaltyEpoch.current = st.penaltyEpoch;
                lastEnergy.current = null;
            }
            // Per-step routing (spec §1): 'gpu' serves sobolev + analytical +
            // penalties-off only; any other config sends THIS step to the worker
            // without changing the selected driver (the driver re-checks the same
            // predicate as its last line of defense). Note `useWorker` is decided from
            // the config at SEND time, so an unowned/stale token from the other
            // backend is dropped by that backend's own epoch/gv rule.
            // @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md — "1. Scope"
            const useWorker =
                st.solverDriver === 'worker' ||
                (st.solverDriver === 'gpu' && !gpuStepSupported(st));
            if (st.solverDriver === 'gpu' && !useWorker) {
                const drv = gpuDriverRef.current;
                // Same single-flight latch as the worker path (§D1): `drv` may be null
                // for a frame while the lifecycle effect (re)constructs it.
                if (drv && !inFlight.current) {
                    const token = {
                        backend: 'gpu' as const,
                        generation: dispatchGeneration.current,
                    };
                    inFlight.current = token;
                    const gv = st.graphVersion;
                    drv.step(st, lastEnergy.current ?? undefined)
                        .then(({ outcome, usedGpu }) => {
                            if (inFlight.current === token) inFlight.current = null;
                            const now = useSimStore.getState();
                            // Same §D5 drop rule as the worker path (gv + running) plus
                            // the epoch, the driver still being 'gpu', and the driver
                            // INSTANCE still being ours (effect cleanup disposed it ⇒ a
                            // result that resolved against the dead engine must not
                            // reach `live`). @see the GPU lifecycle effect above.
                            if (
                                token.generation !== dispatchGeneration.current ||
                                gpuDriverRef.current !== drv ||
                                gv !== now.graphVersion ||
                                !now.running ||
                                now.solverDriver !== 'gpu'
                            )
                                return;
                            // usedGpu=false ⇒ the driver served this step with CPU dE:
                            // non-finite GPU readback (plan D2) or a topology the kernel
                            // can't build (no disjoint pairs / too many edges, spec §1).
                            // The step is still correct; only the acceleration is lost.
                            if (!usedGpu)
                                console.warn(
                                    'solverDriver: gpu step ran with CPU dE (non-finite GPU dE or unsupported topology)',
                                );
                            // No frame delta across the await: wall-clock elapsed since
                            // the last applied result, same ~10 Hz throttle as the worker.
                            const t = performance.now();
                            const elapsed = (t - lastResultTime.current) / 1000;
                            lastResultTime.current = t;
                            applyStepOutcome(outcome, elapsed);
                        })
                        .catch((err) => {
                            // A rejection from a driver WE disposed / superseded (effect
                            // cleanup mid-flight: unmount, driver switch, gpuAvailable
                            // cleared) is not a device failure — the GpuDriver.step
                            // contract says swallow it. Detect it by epoch / instance
                            // identity; only a rejection from the CURRENT driver in the
                            // CURRENT epoch flips gpu → worker (spec §2.6) and marks the
                            // GPU unavailable for the rest of the session.
                            // @see src/gpu/driver.ts — GpuDriver.step TSDoc
                            const superseded =
                                token.generation !== dispatchGeneration.current ||
                                gpuDriverRef.current !== drv;
                            if (inFlight.current === token) inFlight.current = null;
                            if (superseded) return;
                            console.error(
                                'solverDriver: gpu step failed; falling back to worker:',
                                err,
                            );
                            useSimStore.getState().setGpuAvailable(false);
                            useSimStore.getState().setSolverDriver('worker');
                        });
                }
            } else if (useWorker) {
                // §D1 single-flight: send only when the worker exists and no step is
                // outstanding — the result's onmessage clears inFlight. One step per
                // round-trip preserves today's step-serialized semantics (schedule
                // advance, E₀ chaining, pin/penalty pickup) with zero drift; a sobolev
                // step is O(|V|³), sized for |V| ≤ ~300, so stacking would only starve
                // the render loop with no visual benefit — which is the whole point of
                // moving it off the main thread.
                // @see docs/superpowers/plans/2026-07-04-worker-solver.md §D1
                // @see local_files/sobolev-gradient-handoff.md ("|V| ≤ ~300, interactive rates")
                const worker = workerRef.current;
                if (worker && !inFlight.current) {
                    // §D4: (re)send topology once per graphVersion BEFORE the step —
                    // the worker recomputes + caches disjointPairs, so per-step
                    // messages never carry the O(E²) pairs.
                    if (st.graphVersion !== sentTopologyVersion.current) {
                        sentTopologyVersion.current = st.graphVersion;
                        const topo: SolverWorkerRequest = {
                            type: 'topology',
                            graphVersion: st.graphVersion,
                            edges: st.graph.edges,
                        };
                        worker.postMessage(topo);
                    }
                    // Strip the topology fields the worker restores from its cache
                    // (§D4): the per-step payload must never carry disjointPairs.
                    // buildStepArgs (§D7) is the SAME assembly the main driver uses,
                    // so the two paths can't drift; collectTimings is hardcoded here
                    // exactly as at the main call site.
                    const {
                        edges: _edges,
                        disjointPairs: _pairs,
                        ...args
                    } = buildStepArgs(st, lastEnergy.current ?? undefined);
                    const stepReq: SolverWorkerRequest = {
                        type: 'step',
                        graphVersion: st.graphVersion,
                        // §D14: collect the step's field ONLY when the arrows are
                        // visible, so the payload grows only when it's consumed
                        // (issue #9). @see …worker-solver.md §D14
                        args: { ...args, collectTimings: true, collectField: st.showArrows },
                    };
                    worker.postMessage(stepReq);
                    inFlight.current = {
                        backend: 'worker',
                        generation: dispatchGeneration.current,
                    };
                }
            } else {
                // Main driver: today's exact synchronous path — the SAME buildStepArgs
                // (§D7) + the SAME applyStepOutcome (§D8), with the frame delta as the
                // stat-throttle clock. This is the fallback / A/B baseline (§D6) and
                // the only path the store tests exercise.
                const result = dispatchDescentStep({
                    ...buildStepArgs(st, lastEnergy.current ?? undefined),
                    collectTimings: true,
                    // §D14: collect the field only when arrows are visible (issue #9).
                    collectField: st.showArrows,
                });
                applyStepOutcome(result, delta);
            }
        } else {
            // keep the iteration counter in sync with a rebuilt/committed state
            iters.current = st.step;
            // Any !running boundary (run start after pause, user pause, auto-pause,
            // preset rebuild, vertex commit) invalidates the cached E₀: the next run
            // may start from re-anchored targets / different vertices. Null forces a
            // fresh E₀ on the run's first step. A worker step still in flight is
            // dropped by §D5 (running=false) when it lands, so a stale value can never
            // reach the Armijo gate. @see the lastEnergy ref anchor / solver-perf Task 4.
            lastEnergy.current = null;
        }

        camAcc.current += delta;
        if (camAcc.current > 0.2) {
            camAcc.current = 0;
            const dist = state.camera.position.length();
            st.setZoom(dist > 0 ? BASE_DISTANCE / dist : 1);
        }
    });

    return null;
}

// Why: the only file that touches WebGPU wiring — <Canvas flat> (no tone-mapping so the viz
// palette stays exact) + the async WebGPURenderer gl factory + OrbitControls. @see spec §4.1/§6.
export function Viewer() {
    // Hooks first — unconditional/top-level (rules-of-hooks; spec §3). The WebGPU guard
    // below must NOT precede them.
    // Why: <any> — drei's OrbitControls instance type isn't exported for a typed ref here; we
    // only call `.reset?.()` on it. @see spec §6 (sanctioned casts).
    const controls = useRef<any>(null);
    const viewResetNonce = useSimStore((s) => s.viewResetNonce);
    const graphVersion = useSimStore((s) => s.graphVersion);

    // Reset re-centres the camera (old Reset zeroed zoom). @see spec §4.1.
    // viewResetNonce is an intentional TRIGGER dep — the effect fires controls.reset() when
    // Reset bumps the nonce; it deliberately doesn't read the value. Removing it breaks Reset.
    // biome-ignore lint/correctness/useExhaustiveDependencies: intentional nonce trigger (see above)
    useEffect(() => {
        controls.current?.reset?.();
    }, [viewResetNonce]);

    // Guard: WebGPU unavailable (old browser / non-secure context) — surface, don't blank.
    // @ts-expect-error - `navigator.gpu` isn't in TS 5.9's bundled lib.dom.d.ts yet (WebGPU
    // types are still bleeding-edge); runtime check is the real gate. @see spec §6.
    if (typeof navigator !== 'undefined' && !navigator.gpu) {
        return (
            <div style={{ padding: 20, color: '#ffaa00' }}>
                WebGPU is unavailable in this browser/context. Use a WebGPU-capable browser over
                http://localhost.
            </div>
        );
    }

    return (
        <Canvas
            flat
            camera={{ position: CAMERA_POS, fov: 50 }}
            gl={async (props) => {
                // `as any`: R3F's gl-factory props aren't typed for WebGPURenderer's ctor; the
                // awaited init() is what actually enables the WebGPU backend. @see spec §6.
                // trackTimestamp: required by the WebGPU milestone's G3/G7 GPU-time gates
                // (injects timestamp queries around compute/render passes; experimental
                // API — re-verify on any three upgrade).
                // @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §6
                const renderer = new THREE.WebGPURenderer({
                    ...(props as any),
                    trackTimestamp: true,
                });
                await renderer.init();
                // Phase 1 boot gate: G2 through the production kernel, once per
                // adapter (this renderer). `then`-ed, NOT awaited — first paint must
                // not wait on a compute round-trip; the store default gpuAvailable=false
                // keeps the UI option disabled until the verdict lands. runGpuSelfTest
                // never throws (a throw is a failed gate).
                // @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md — "2.6 Driver integration"
                runGpuSelfTest(renderer).then((r) => {
                    useSimStore.getState().setGpuAvailable(r.ok);
                    if (!r.ok)
                        console.warn('gpu: boot self-test failed; GPU driver unavailable', r);
                });
                // Device loss / uncaptured error ⇒ GPU unavailable + fall back gpu → worker
                // (spec §6 "Mid-run device loss"; the fallback chain is §2.6). Both hooks
                // are instance properties initialised to UNBOUND prototype methods that use
                // `this` (three r185 Renderer.js:595/606 → _onDeviceLost :1225 sets
                // this._isDeviceLost; _onError :1247), and WebGPUBackend.js:273/283 looks
                // them up on the renderer at fire time — so an own-property override here
                // is what the backend calls, and the previous implementation must be chained
                // with `.call(renderer, …)`. r185 passes an info OBJECT to onError although
                // @types/three Renderer.d.ts:264 types it as a string; bridged without `any`.
                // One-way: nothing re-arms gpuAvailable for the rest of the session.
                // @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md — "6. Risks not covered by gates"
                // @see node_modules/three/src/renderers/webgpu/WebGPUBackend.js (device.lost / onuncapturederror)
                const prevLost = renderer.onDeviceLost;
                renderer.onDeviceLost = (info) => {
                    useSimStore.getState().setGpuAvailable(false);
                    if (useSimStore.getState().solverDriver === 'gpu')
                        useSimStore.getState().setSolverDriver('worker');
                    prevLost.call(renderer, info);
                };
                const errHooks = renderer as unknown as {
                    onError: (info: string | { message?: string }) => void;
                };
                const prevErr = errHooks.onError;
                errHooks.onError = (info) => {
                    useSimStore.getState().setGpuAvailable(false);
                    if (useSimStore.getState().solverDriver === 'gpu')
                        useSimStore.getState().setSolverDriver('worker');
                    prevErr.call(renderer, info);
                };
                return renderer;
            }}
        >
            <ambientLight intensity={0.8} />
            <directionalLight position={[5, 5, 5]} intensity={0.6} />
            {/* makeDefault registers this instance as R3F state.controls so
                PinControls can toggle `.enabled` to suspend orbiting during a
                vertex drag (Decision D7). @see docs/superpowers/plans/2026-07-03-pin-drag-ui.md */}
            <OrbitControls makeDefault ref={controls} minPolarAngle={0} maxPolarAngle={Math.PI} />
            {/* Distinct key PREFIXES: both siblings previously shared the bare
                graphVersion key, and duplicate keys among siblings are undefined
                reconciler behavior in React ("children may be duplicated and/or
                omitted") — stale meshes could linger across preset regenerations. */}
            <Curve key={`curve-${graphVersion}`} />
            <GradientArrows key={`arrows-${graphVersion}`} />
            {/* Pin picking/drag overlay — keyed on graphVersion for the same
                remount-on-topology-change contract as Curve/GradientArrows. */}
            <PinControls key={`pins-${graphVersion}`} />
            <Simulation />
        </Canvas>
    );
}
