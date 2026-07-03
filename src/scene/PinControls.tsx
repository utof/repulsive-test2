import { type ThreeEvent, useFrame, useThree } from '@react-three/fiber';
import { useEffect, useRef } from 'react';
import * as THREE from 'three/webgpu';
import { useSimStore } from '../store';

// Pick spheres are larger than Curve.tsx's VERTEX_RADIUS (0.06) so a vertex is
// easy to grab; markers are a touch larger still so a pinned vertex reads as
// clearly pinned. Gold = enabled pin, gray = disabled. Interaction-design
// choices are the implementer's per the milestone brief.
// @see docs/superpowers/plans/2026-07-03-pin-drag-ui.md (Decision D8)
const PICK_RADIUS = 0.12;
const MARKER_RADIUS = 0.09;
const PIN_COLOR = '#ffd700';
const PIN_DISABLED_COLOR = '#6a6a6a';

// Module-scope scratch objects (Curve.tsx / GradientArrows.tsx pattern) — reused
// every frame / event so picking and dragging allocate nothing on the hot path.
const tmpObj = new THREE.Object3D();
const tmpPoint = new THREE.Vector3();
const tmpNormal = new THREE.Vector3();
// Dedicated raycaster for the end-of-drag hover test (Decision D10) — never
// mutate R3F's live state.raycaster from event handlers.
const pickRaycaster = new THREE.Raycaster();

/**
 * R3F v9 replaces the event's `target`/`currentTarget` at RUNTIME with an object
 * exposing the pointer-capture methods (so `e.target.setPointerCapture(id)` is the
 * documented capture API), but the STATIC type of `target` is the DOM
 * `EventTarget`, which lacks them — hence this cast. @see plan Decision D7 and
 * node_modules/@react-three/fiber .../core/events (setPointerCapture on target).
 */
type PointerCapturer = {
    setPointerCapture(id: number): void;
    releasePointerCapture(id: number): void;
};

/**
 * Interactive point-constraint picking + drag (pin-drag milestone, briefing §5B).
 * Renders an INVISIBLE raycast overlay (an InstancedMesh of pick spheres tracking
 * the live vertices) plus a visible gold marker per pin. Grabbing a vertex creates
 * a pin at its current position (store `addPin`) and drags its frozen target along
 * a camera-facing plane (store `setPinTarget`); the sobolev descent then holds the
 * pinned vertex at the target and relaxes the rest of the curve around it. Lives
 * inside <Canvas> so it can use R3F pointer events / useThree.
 *
 * Drag/interaction model, drag plane, and OrbitControls suppression are Decisions
 * D3/D4/D7; abnormal-termination cleanup (pointercancel / lostpointercapture /
 * mid-drag remount) and the hover-accurate end-of-drag cursor are Decision D10;
 * the R3F pointer-event + drei `state.controls` APIs were verified via context7
 * and against the installed R3F events source (see the plan's "API verification").
 * @see docs/superpowers/plans/2026-07-03-pin-drag-ui.md (Decisions D3, D4, D7, D8, D10)
 * @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §5.3
 * @see src/scene/Curve.tsx (live-buffer per-frame instance-matrix pattern)
 */
export function PinControls() {
    const count = useSimStore((s) => s.graph.vertices.length);
    const pins = useSimStore((s) => s.pins);

    // The default controls instance (OrbitControls has `makeDefault` in Viewer);
    // toggling `.enabled` is the documented way to gate orbiting during a drag.
    // `as any`: R3F types state.controls loosely — same sanctioned-cast convention
    // as Viewer.tsx's OrbitControls ref. @see plan Decision D7.
    // biome-ignore lint/suspicious/noExplicitAny: R3F state.controls is loosely typed
    const controls = useThree((s) => s.controls) as any;
    const gl = useThree((s) => s.gl);

    const pickRef = useRef<THREE.InstancedMesh>(null);
    // Which vertex is being dragged (null = idle) and the frozen camera-facing
    // drag plane captured at grab time (Decision D4). Refs, not state: the drag
    // runs entirely in pointer-event handlers + the live buffer, off React's
    // render path (same rationale as the live buffer never being subscribed).
    const dragIndex = useRef<number | null>(null);
    const dragPlane = useRef(new THREE.Plane());

    // Unmount cleanup (Decision D10): Viewer keys this component on graphVersion,
    // so a preset change/regenerate REMOUNTS it — mid-drag that would strand the
    // shared makeDefault OrbitControls disabled and the cursor on 'grabbing'
    // forever. Restore both unconditionally; nothing else in the app disables the
    // default controls, so the blanket re-enable can't stomp another consumer.
    useEffect(() => {
        return () => {
            if (controls) controls.enabled = true;
            gl.domElement.style.cursor = '';
        };
    }, [controls, gl]);

    // Track the live vertices with the invisible pick spheres every frame, and
    // refresh the mesh-level bounding sphere so raycasting the MOVED instances
    // stays correct (three caches boundingSphere until asked to recompute).
    useFrame(() => {
        const mesh = pickRef.current;
        if (!mesh) return;
        const live = useSimStore.getState().live;
        for (let i = 0; i < count; i++) {
            tmpObj.position.set(live[i][0], live[i][1], live[i][2]);
            tmpObj.updateMatrix();
            mesh.setMatrixAt(i, tmpObj.matrix);
        }
        mesh.instanceMatrix.needsUpdate = true;
        mesh.computeBoundingSphere();
    });

    const setCursor = (c: string) => {
        gl.domElement.style.cursor = c;
    };

    const onPointerOver = () => {
        if (dragIndex.current === null) setCursor('grab');
    };
    // Doubles as the ABNORMAL drag-termination hook (Decision D10). This R3F
    // version never dispatches the object-level onPointerCancel /
    // onLostPointerCapture props: its canvas handlers for both events only call
    // cancelPointer([]), which fires onPointerOut on every hovered object
    // (@react-three/fiber dist events: handlePointer's cancelation switch +
    // cancelPointer) — so out-while-dragging is where cancellation surfaces.
    // The instanceId filter matters: during a capture the DRAGGED instance is
    // pinned into the hover set via the captured intersection (it can only go
    // out through a cancel), while OTHER instances the cursor merely crosses
    // mid-drag produce ordinary outs that must not end the drag. Capture is NOT
    // released here — on these paths it is already gone, and releasing again
    // can throw.
    const onPointerOut = (e: ThreeEvent<PointerEvent>) => {
        if (dragIndex.current === null) {
            setCursor('');
            return;
        }
        if (e.instanceId === dragIndex.current) {
            dragIndex.current = null;
            if (controls) controls.enabled = true;
            setCursor('');
        }
    };

    const onPointerDown = (e: ThreeEvent<PointerEvent>) => {
        const i = e.instanceId;
        if (i === undefined) return;
        // Beat sibling handlers and keep receiving move/up off the sphere; disable
        // orbit for the duration of the drag (Decision D7).
        e.stopPropagation();
        (e.target as unknown as PointerCapturer).setPointerCapture(e.pointerId);
        if (controls) controls.enabled = false;

        const store = useSimStore.getState();
        // Create the pin if new — snapshots the vertex's current position as the
        // frozen target so it initially just holds the vertex where it is (D3).
        if (!store.pins.some((p) => p.vertexIndex === i)) store.addPin(i);

        // Camera-facing drag plane through the grabbed vertex (Decision D4).
        const live = store.live;
        tmpPoint.set(live[i][0], live[i][1], live[i][2]);
        e.camera.getWorldDirection(tmpNormal);
        dragPlane.current.setFromNormalAndCoplanarPoint(tmpNormal, tmpPoint);
        dragIndex.current = i;
        setCursor('grabbing');
    };

    const onPointerMove = (e: ThreeEvent<PointerEvent>) => {
        const i = dragIndex.current;
        if (i === null) return;
        // Intersect the current mouse ray with the frozen plane; camera-facing ⇒
        // never parallel ⇒ always a hit, but guard anyway (Decision D4).
        const hit = e.ray.intersectPlane(dragPlane.current, tmpPoint);
        if (!hit) return;
        // Move BOTH the frozen target and the live vertex (Decision D3): the pin
        // constraint pulls the vertex to `target` under descent; writing `live`
        // too makes the grabbed vertex track the cursor even while paused, and
        // keeps target == live so the play/commit re-anchor is a no-op.
        useSimStore.getState().setPinTarget(i, [hit.x, hit.y, hit.z]);
        const live = useSimStore.getState().live;
        if (live[i]) {
            live[i][0] = hit.x;
            live[i][1] = hit.y;
            live[i][2] = hit.z;
        }
    };

    const endDrag = (e: ThreeEvent<PointerEvent>) => {
        if (dragIndex.current === null) return;
        // Tolerant release (Decision D10): capture may already be gone (browser
        // auto-release around pointerup, or a lostpointercapture race) and the
        // underlying DOM releasePointerCapture throws on a dead pointerId —
        // never let that escape a frame-loop-adjacent handler.
        try {
            (e.target as unknown as PointerCapturer).releasePointerCapture(e.pointerId);
        } catch {}
        dragIndex.current = null;
        if (controls) controls.enabled = true;
        // Cursor reflects ACTUAL hover at release (Decision D10): R3F's hover
        // bookkeeping is unusable here — the captured intersection pins the
        // dragged instance into the hover set even when the pointer is nowhere
        // near it — so raycast the pick mesh with the release ray instead. In
        // the normal flow the dragged vertex sits ON that ray (it tracked
        // ray∩plane all drag), so this lands on 'grab'.
        const mesh = pickRef.current;
        let overPick = false;
        if (mesh) {
            pickRaycaster.ray.copy(e.ray);
            overPick = pickRaycaster.intersectObject(mesh, false).length > 0;
        }
        setCursor(overPick ? 'grab' : '');
    };

    return (
        <group>
            {/* Invisible raycast overlay: opacity 0 + depthWrite false keeps
                `object.visible` TRUE (so the raycaster still tests it) while
                drawing nothing and never occluding the real geometry. Larger than
                the visual vertex for an easier grab. @see plan Decision D8. */}
            <instancedMesh
                ref={pickRef}
                args={[undefined, undefined, count]}
                onPointerOver={onPointerOver}
                onPointerOut={onPointerOut}
                onPointerDown={onPointerDown}
                onPointerMove={onPointerMove}
                onPointerUp={endDrag}
            >
                <sphereGeometry args={[PICK_RADIUS, 12, 12]} />
                <meshBasicNodeMaterial transparent opacity={0} depthWrite={false} />
            </instancedMesh>
            {/* Visible pin markers at each pin's frozen target (gold = enabled,
                gray = disabled). Unlit basic material so a pin reads clearly
                regardless of scene lighting. */}
            {pins.map((pin) => (
                <mesh key={pin.vertexIndex} position={pin.target}>
                    <sphereGeometry args={[MARKER_RADIUS, 16, 16]} />
                    <meshBasicNodeMaterial color={pin.enabled ? PIN_COLOR : PIN_DISABLED_COLOR} />
                </mesh>
            ))}
        </group>
    );
}
