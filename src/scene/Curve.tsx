import { useFrame } from '@react-three/fiber';
import { useMemo, useRef } from 'react';
// Task 10 (fat WebGPU edges): fat-line geometry lives in three/addons, not the `three/webgpu`
// namespace. @see docs/superpowers/specs/2026-07-02-react-three-webgpu-switch-design.md Task 10.
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import * as THREE from 'three/webgpu';
import { useSimStore } from '../store';

const VERTEX_RADIUS = 0.06;
const tmp = new THREE.Object3D();

// Why: reads graph topology from the store (re-renders only on rebuild — Viewer keys us on
// graphVersion), and pulls live positions every frame from the non-reactive buffer.
// @see spec §4.1 (scene/Curve) / §5 (transient-update pattern).
export function Curve() {
    const edges = useSimStore((s) => s.graph.edges);
    const count = useSimStore((s) => s.graph.vertices.length);

    // Task 10: LineSegmentsGeometry (fat lines) instead of plain BufferGeometry — positions are
    // written via setPositions() below rather than a raw 'position' BufferAttribute.
    const lineGeom = useMemo(() => new LineSegmentsGeometry(), [edges]);

    const meshRef = useRef<THREE.InstancedMesh>(null);

    useFrame(() => {
        // INVARIANT: `live` is index-aligned with this render's `edges`/`count`. Safe because
        // every store mutation that changes topology (setPreset/regenerate/reset) also bumps
        // graphVersion, and Viewer keys <Curve> on it — so R3F synchronously unmounts this
        // instance (via layout-effect) before a stale frame could index a shrunk `live`. Keep
        // that keying if you add async/debounced rebuilds, else these reads go out of bounds.
        const live = useSimStore.getState().live;

        // Task 10: LineSegmentsGeometry.setPositions() wants a fresh flat array (it re-derives
        // instanceStart/instanceEnd interleaved attributes from it internally) — this per-frame
        // allocation is a deliberate, plan-accepted cost of the fat-line path, not a Task 7-style
        // regression of the old in-place BufferAttribute write.
        const flat = new Float32Array(edges.length * 6);
        for (let e = 0; e < edges.length; e++) {
            const [a, b] = edges[e];
            const va = live[a];
            const vb = live[b];
            flat[e * 6 + 0] = va[0];
            flat[e * 6 + 1] = va[1];
            flat[e * 6 + 2] = va[2];
            flat[e * 6 + 3] = vb[0];
            flat[e * 6 + 4] = vb[1];
            flat[e * 6 + 5] = vb[2];
        }
        lineGeom.setPositions(flat);

        const mesh = meshRef.current;
        if (mesh) {
            for (let i = 0; i < count; i++) {
                tmp.position.set(live[i][0], live[i][1], live[i][2]);
                tmp.updateMatrix();
                mesh.setMatrixAt(i, tmp.matrix);
            }
            mesh.instanceMatrix.needsUpdate = true;
        }
    });

    return (
        <group>
            {/* Task 10: fat WebGPU line (parity with the old lineWidth:3) — WebGPU-safe
                LineSegments2/Line2NodeMaterial (three/addons/lines/webgpu/*), not the WebGL-only
                examples/jsm/lines/LineSegments2 or drei's <Line>. */}
            <lineSegments2 args={[lineGeom]}>
                <line2NodeMaterial color="#4a9eff" linewidth={3} worldUnits={false} />
            </lineSegments2>
            <instancedMesh ref={meshRef} args={[undefined, undefined, count]}>
                <sphereGeometry args={[VERTEX_RADIUS, 16, 16]} />
                <meshStandardNodeMaterial color="#ff6b6b" />
            </instancedMesh>
        </group>
    );
}
