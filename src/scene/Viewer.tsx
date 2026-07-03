import { OrbitControls } from '@react-three/drei';
import {
    Canvas,
    extend,
    type ThreeElement,
    type ThreeToJSXElements,
    useFrame,
} from '@react-three/fiber';
import { useEffect, useRef } from 'react';
// Task 10 (fat WebGPU edges): these live in three's addons tree, not the `three/webgpu`
// namespace, so `extend(THREE as any)` below doesn't register them — hence the separate
// `extend({...})` call. WebGPU-safe variants only (webgpu/LineSegments2, not the WebGL-only
// examples/jsm/lines/LineSegments2). @see docs/superpowers/specs/2026-07-02-react-three-webgpu-switch-design.md Task 10.
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineSegments2 } from 'three/addons/lines/webgpu/LineSegments2.js';
import * as THREE from 'three/webgpu';
import { dispatchDescentStep, useSimStore } from '../store';
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

    useFrame((state, delta) => {
        const st = useSimStore.getState();

        if (st.running) {
            // Penalty-config change since last frame ⇒ invalidate the reused E₀
            // (see the lastPenaltyEpoch anchor). Constraint-target animation does
            // NOT bump the epoch (targets aren't in the objective, plan §2.4).
            if (st.penaltyEpoch !== lastPenaltyEpoch.current) {
                lastPenaltyEpoch.current = st.penaltyEpoch;
                lastEnergy.current = null;
            }
            // ONE descent step per frame at most — a sobolev step is dense assembly +
            // O(|V|³) saddle solves (several per line-search trial), sized for the
            // stage-1 budget of |V| ≤ ~300; stacking multiple per frame would stall
            // the render loop with no visual benefit.
            // @see local_files/sobolev-gradient-handoff.md ("our targets are |V| ≤ ~300, interactive rates")
            const result = dispatchDescentStep({
                descentMode: st.descentMode,
                vertices: st.live,
                edges: st.graph.edges,
                disjointPairs: st.disjointPairs,
                mode: st.mode,
                stepSize: st.stepSize,
                // Frozen targets + per-block toggles: dispatch builds the
                // ConstraintSet (barycenter first) from these. The frame loop only
                // READS the frozen x₀/L⁰/ℓ⁰ — re-anchoring happens in the store (see
                // the frozen-targets lifecycle anchor there).
                // @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §4.2, §5.3, §9a
                x0: st.sobolevX0,
                sobolevL0: st.sobolevL0,
                barycenterConstraint: st.barycenterConstraint,
                lengthMode: st.lengthMode,
                sobolevEll0: st.sobolevEll0,
                // Interactive pins → pointBlocks (briefing §5B). Frozen targets,
                // READ-ONLY here — the frame loop never mutates them (same
                // contract as sobolevEll0 above); re-anchoring/drag happen in the
                // store / PinControls. @see docs/superpowers/plans/2026-07-03-pin-drag-ui.md
                pins: st.pins,
                // Projection strategy A/B (solver-perf Task 6): store default
                // 'frozen' (reference-impl reuse), 'reassemble' selectable.
                projectionMode: st.projectionMode,
                // Soft-constraint penalties (5C). Default all-off ⇒ the core no-ops
                // via penaltiesActive, bit-identical to the penalty-free build; a
                // config change bumps penaltyEpoch (handled above).
                // @see docs/superpowers/plans/2026-07-03-sobolev-penalties.md §2.4
                penalties: st.penalties,
                // Always collect per-phase timings for the Stats.tsx readout; the
                // raw path ignores this and the collector is inert when off.
                // @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 3)
                collectTimings: true,
                // Reuse the previous accepted step's E₀ (null on the first frame
                // of any run → fresh recompute). @see the lastEnergy ref anchor
                // above / plan Task 4.
                energyBefore: lastEnergy.current ?? undefined,
            });
            if (result.accepted) {
                // Copy-then-discard: both steppers are pure by design and return a fresh
                // Vec3[] each frame; we mutate the existing live tuples in place (preserving their
                // identity for Curve's non-reactive buffer) and drop `vertices`. Revisit only if N
                // grows enough that the per-frame alloc matters (scratch-buffer optimizer). @see spec §5.
                for (let i = 0; i < st.live.length; i++) {
                    const v = result.vertices[i];
                    const l = st.live[i];
                    l[0] = v[0];
                    l[1] = v[1];
                    l[2] = v[2];
                }
                // Cache E₀ for the next step: result.energy is calculateEnergy at
                // exactly the vertices just copied into `live` (this step's output =
                // next step's input), so next frame's reuse is bit-identical.
                // @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 4)
                lastEnergy.current = result.energy;
                // Target-length animation (paper tex line 760; plan §4 Task 5):
                // advance the frozen length schedule ONLY on ACCEPTED steps and
                // ONLY when growth is enabled — rate 1.0 leaves the store untouched
                // so default runs are bit-identical. The scaled target enters the
                // NEXT step's projection. Does NOT invalidate the E₀ cache (targets
                // aren't in the objective, plan §2.4).
                // @see docs/superpowers/plans/2026-07-03-sobolev-penalties.md §4 Task 5
                if (st.lengthGrowthRate !== 1) st.advanceLengthSchedule();
                iters.current += 1;
                statAcc.current += delta;
                if (statAcc.current > 0.1) {
                    statAcc.current = 0;
                    useSimStore.setState({
                        step: iters.current,
                        energy: result.energy,
                        sobolevStats: result.stats,
                        sobolevTimings: result.timings,
                    });
                }
            } else {
                // Sobolev converged (spec §C step 5) or rejected the step (spec §C
                // step 10: leave vertices unchanged, report — never throw): auto-pause
                // instead of spinning on rejected steps. Publish the diagnostics
                // un-throttled first; setRunning(false) commits live positions and
                // deliberately preserves them (see store).
                // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §C (steps 5, 10)
                useSimStore.setState({
                    step: iters.current,
                    sobolevStats: result.stats,
                    sobolevConverged: result.converged,
                    sobolevTimings: result.timings,
                });
                st.setRunning(false);
            }
        } else {
            // keep the iteration counter in sync with a rebuilt/committed state
            iters.current = st.step;
            // Any !running boundary (run start after pause, user pause, auto-pause,
            // preset rebuild, vertex commit) invalidates the cached E₀: the next
            // run may start from re-anchored targets / different vertices. Null
            // forces a fresh E₀ on the run's first step, so a stale value can never
            // reach the Armijo gate. @see the lastEnergy ref anchor / plan Task 4.
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
                const renderer = new THREE.WebGPURenderer(props as any);
                await renderer.init();
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
