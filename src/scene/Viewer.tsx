import { OrbitControls } from '@react-three/drei';
import { Canvas, extend, type ThreeToJSXElements, useFrame } from '@react-three/fiber';
import { useEffect, useRef } from 'react';
import * as THREE from 'three/webgpu';
import { step as descentStep } from '../core/optimizer';
import { useSimStore } from '../store';
import { Curve } from './Curve';
import { GradientArrows } from './GradientArrows';

declare module '@react-three/fiber' {
    interface ThreeElements extends ThreeToJSXElements<typeof THREE> {}
}
// `as any`: registering the whole three/webgpu namespace as R3F intrinsics; the cast is the
// spec-sanctioned bridge for the NodeMaterial/WebGPU classes under strict. @see spec §6.
extend(THREE as any);

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

    useFrame((state, delta) => {
        const st = useSimStore.getState();

        if (st.running) {
            const { vertices, energy } = descentStep(st.live, st.graph.edges, st.disjointPairs, {
                mode: st.mode,
                stepSize: st.stepSize,
            });
            // Copy-then-discard: optimizer.step() is pure by design and returns a fresh
            // Vec3[] each frame; we mutate the existing live tuples in place (preserving their
            // identity for Curve's non-reactive buffer) and drop `vertices`. Revisit only if N
            // grows enough that the per-frame alloc matters (scratch-buffer optimizer). @see spec §5.
            for (let i = 0; i < st.live.length; i++) {
                const v = vertices[i];
                const l = st.live[i];
                l[0] = v[0];
                l[1] = v[1];
                l[2] = v[2];
            }
            iters.current += 1;
            statAcc.current += delta;
            if (statAcc.current > 0.1) {
                statAcc.current = 0;
                useSimStore.setState({ step: iters.current, energy });
            }
        } else {
            // keep the iteration counter in sync with a rebuilt/committed state
            iters.current = st.step;
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
            <OrbitControls ref={controls} minPolarAngle={0} maxPolarAngle={Math.PI} />
            <Curve key={graphVersion} />
            <GradientArrows key={graphVersion} />
            <Simulation />
        </Canvas>
    );
}
