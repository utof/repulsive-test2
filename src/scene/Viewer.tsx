import { OrbitControls } from '@react-three/drei';
import { Canvas, extend, type ThreeToJSXElements } from '@react-three/fiber';
import { useEffect, useRef } from 'react';
import * as THREE from 'three/webgpu';
import { useSimStore } from '../store';
import { Curve } from './Curve';

declare module '@react-three/fiber' {
    interface ThreeElements extends ThreeToJSXElements<typeof THREE> {}
}
// `as any`: registering the whole three/webgpu namespace as R3F intrinsics; the cast is the
// spec-sanctioned bridge for the NodeMaterial/WebGPU classes under strict. @see spec §6.
extend(THREE as any);

const CAMERA_POS: [number, number, number] = [3, 3, 3];

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
        </Canvas>
    );
}
