import { useLayoutEffect, useMemo, useRef } from 'react';
import * as THREE from 'three/webgpu';
import { DEFAULTS } from '../core/optimizer';
import { gradientAnalytical, gradientFiniteDiff, norm } from '../core/tangentPointEnergy';
import type { Vec3 } from '../core/testConfigs';
import { useSimStore } from '../store';

// World-space analogue of the old MIN/MAX pixel clamps. MIN keeps near-convergence tiny
// gradients visible; MAX is a world-space length cap on huge gradients (still an ACTIVE clamp —
// don't delete). What real 3D geometry dissolves is the old screen-space vanishing-point whip,
// not the length cap. @see spec §4.1 (GradientArrows) / §6.
const ARROW_SCALE = 0.2;
const MIN_WORLD = 0.08;
const MAX_WORLD = 1.2;
const CONE_RADIUS = 0.03;
// Mirror of Curve.tsx's VERTEX_RADIUS: the cone is offset outward by this + half its length so
// its base sits on the sphere surface and it points away from the vertex (instead of the cone
// centre coinciding with the vertex and half the cone penetrating the sphere). Keep in sync.
const VERTEX_RADIUS = 0.06;

const UP = new THREE.Vector3(0, 1, 0);
const tmp = new THREE.Object3D();
const dirVec = new THREE.Vector3();

interface Arrow {
    pos: Vec3;
    dir: Vec3; // unit negative-gradient
    len: number; // world units
}

// Why: instanced cones pointing along the −gradient (descent direction) at each vertex, shown
// only when paused. Real 3D geometry replaces the old projectArrow hack, so the camera does
// foreshortening for free (dissolves the vanishing-point whip). @see spec §4.1 / §6.
export function GradientArrows() {
    const running = useSimStore((s) => s.running);
    const mode = useSimStore((s) => s.mode);
    const graph = useSimStore((s) => s.graph);
    const disjointPairs = useSimStore((s) => s.disjointPairs);

    // Recompute only when paused config changes. graph.vertices is current here because
    // <Simulation/>'s stop path committed the live buffer (spec §6 stale-arrow fix).
    const arrows = useMemo<Arrow[]>(() => {
        if (running) return [];
        const grad =
            mode === 'analytical'
                ? gradientAnalytical(
                      graph.vertices,
                      graph.edges,
                      disjointPairs,
                      DEFAULTS.alpha,
                      DEFAULTS.beta,
                      DEFAULTS.epsilon,
                  )
                : gradientFiniteDiff(
                      graph.vertices,
                      graph.edges,
                      disjointPairs,
                      DEFAULTS.alpha,
                      DEFAULTS.beta,
                      DEFAULTS.epsilon,
                      DEFAULTS.h,
                  );
        const out: Arrow[] = [];
        for (let i = 0; i < graph.vertices.length; i++) {
            const g = grad[i];
            const gn = norm(g);
            if (gn <= 1e-6) continue; // parity with old index.tsx skip
            out.push({
                pos: graph.vertices[i],
                dir: [-g[0] / gn, -g[1] / gn, -g[2] / gn],
                len: Math.max(MIN_WORLD, Math.min(MAX_WORLD, Math.log(1 + gn) * ARROW_SCALE)),
            });
        }
        return out;
    }, [running, mode, graph, disjointPairs]);

    const meshRef = useRef<THREE.InstancedMesh>(null);
    useLayoutEffect(() => {
        const mesh = meshRef.current;
        if (!mesh) return;
        for (let i = 0; i < arrows.length; i++) {
            const a = arrows[i];
            dirVec.set(a.dir[0], a.dir[1], a.dir[2]);
            // coneGeometry is centred on its origin (height along local Y ∈ [-len/2, len/2]);
            // push it out by sphere radius + half length so the base rests on the sphere surface
            // and the cone extends outward along −gradient rather than straddling the vertex.
            const off = VERTEX_RADIUS + a.len / 2;
            tmp.position.set(
                a.pos[0] + a.dir[0] * off,
                a.pos[1] + a.dir[1] * off,
                a.pos[2] + a.dir[2] * off,
            );
            tmp.quaternion.setFromUnitVectors(UP, dirVec);
            tmp.scale.set(1, a.len, 1);
            tmp.updateMatrix();
            mesh.setMatrixAt(i, tmp.matrix);
        }
        mesh.count = arrows.length;
        mesh.instanceMatrix.needsUpdate = true;
    }, [arrows]);

    if (running) return null;

    const color = mode === 'analytical' ? '#00ff88' : '#ffaa00';
    const maxCount = graph.vertices.length;
    return (
        <instancedMesh ref={meshRef} args={[undefined, undefined, maxCount]}>
            <coneGeometry args={[CONE_RADIUS, 1, 8]} />
            <meshBasicNodeMaterial color={color} />
        </instancedMesh>
    );
}
