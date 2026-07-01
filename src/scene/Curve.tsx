import { useFrame } from '@react-three/fiber';
import { useMemo, useRef } from 'react';
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

    const lineGeom = useMemo(() => {
        const g = new THREE.BufferGeometry();
        g.setAttribute(
            'position',
            new THREE.BufferAttribute(new Float32Array(edges.length * 2 * 3), 3),
        );
        return g;
    }, [edges]);

    const meshRef = useRef<THREE.InstancedMesh>(null);

    useFrame(() => {
        // INVARIANT: `live` is index-aligned with this render's `edges`/`count`. Safe because
        // every store mutation that changes topology (setPreset/regenerate/reset) also bumps
        // graphVersion, and Viewer keys <Curve> on it — so R3F synchronously unmounts this
        // instance (via layout-effect) before a stale frame could index a shrunk `live`. Keep
        // that keying if you add async/debounced rebuilds, else these reads go out of bounds.
        const live = useSimStore.getState().live;

        const pos = lineGeom.getAttribute('position') as THREE.BufferAttribute;
        const arr = pos.array as Float32Array;
        for (let e = 0; e < edges.length; e++) {
            const [a, b] = edges[e];
            const va = live[a];
            const vb = live[b];
            arr[e * 6 + 0] = va[0];
            arr[e * 6 + 1] = va[1];
            arr[e * 6 + 2] = va[2];
            arr[e * 6 + 3] = vb[0];
            arr[e * 6 + 4] = vb[1];
            arr[e * 6 + 5] = vb[2];
        }
        pos.needsUpdate = true;

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
            <lineSegments geometry={lineGeom}>
                <lineBasicNodeMaterial color="#4a9eff" />
            </lineSegments>
            <instancedMesh ref={meshRef} args={[undefined, undefined, count]}>
                <sphereGeometry args={[VERTEX_RADIUS, 16, 16]} />
                <meshStandardNodeMaterial color="#ff6b6b" />
            </instancedMesh>
        </group>
    );
}
