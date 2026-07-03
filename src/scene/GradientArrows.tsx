import { useFrame } from '@react-three/fiber';
import { useRef } from 'react';
import * as THREE from 'three/webgpu';
import { DEFAULTS } from '../core/optimizer';
import { solveConstrainedGradient } from '../core/sobolev/gradient';
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
// Recompute cadence while running: the descent already computes this field once per step, so
// the arrows recompute (raw: O(E²) gradient; sobolev: a full dense saddle solve) is bounded to
// ~5 Hz instead of per-frame — diagnostics don't need more, and in sobolev mode an unthrottled
// recompute would double the dominant O(|V|³) cost of every frame.
// @see local_files/sobolev-gradient-handoff.md ("our targets are |V| ≤ ~300, interactive rates")
const RECOMPUTE_INTERVAL = 0.2;

const UP = new THREE.Vector3(0, 1, 0);
const tmp = new THREE.Object3D();
const dirVec = new THREE.Vector3();

// Why: instanced cones pointing along the DESCENT direction at each vertex, live in both
// paused and running states (reads the non-reactive `live` buffer like Curve.tsx), gated by
// the store's showArrows toggle. The field matches the ACTIVE descent mode: raw → −dE (the
// L² gradient), sobolev → −g̃ (the constrained Sobolev gradient from the saddle solve) — the
// whole point of the A/B toggle is seeing that these directions differ.
// @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("Gradient saddle system"), §C
export function GradientArrows() {
    const showArrows = useSimStore((s) => s.showArrows);
    const mode = useSimStore((s) => s.mode);
    const count = useSimStore((s) => s.graph.vertices.length);

    const meshRef = useRef<THREE.InstancedMesh>(null);
    const acc = useRef(Number.POSITIVE_INFINITY); // ∞ → first frame always computes
    const lastKey = useRef('');

    useFrame((_, delta) => {
        const mesh = meshRef.current;
        if (!mesh) {
            // Toggle off (null render) unmounts the mesh; clear the memo key so the
            // remounted mesh recomputes immediately instead of showing its
            // uninitialized instance matrices behind an unchanged key.
            lastKey.current = '';
            return;
        }
        const st = useSimStore.getState();
        acc.current += delta;

        // Skip when nothing the field depends on changed (paused + same config), and
        // throttle while running (st.step is published ~10 Hz by Simulation; the
        // interval above caps us at ~5 Hz). A fresh mesh (empty key) bypasses the
        // throttle so re-enabling the toggle never flashes stale instances.
        const fresh = lastKey.current === '';
        const key = `${st.step}|${st.mode}|${st.descentMode}|${st.graphVersion}|${st.running}`;
        if (key === lastKey.current) return;
        if (!fresh && st.running && acc.current < RECOMPUTE_INTERVAL) return;
        acc.current = 0;
        lastKey.current = key;

        // INVARIANT: `live` is index-aligned with this render's `count` — same
        // graphVersion-keyed remount contract as Curve.tsx (see its useFrame note).
        const live = st.live;
        const edges = st.graph.edges;
        const disjointPairs = st.disjointPairs;

        const dE =
            st.mode === 'analytical'
                ? gradientAnalytical(
                      live,
                      edges,
                      disjointPairs,
                      DEFAULTS.alpha,
                      DEFAULTS.beta,
                      DEFAULTS.epsilon,
                  )
                : gradientFiniteDiff(
                      live,
                      edges,
                      disjointPairs,
                      DEFAULTS.alpha,
                      DEFAULTS.beta,
                      DEFAULTS.epsilon,
                      DEFAULTS.h,
                  );
        let field: Vec3[] = dE;
        if (st.descentMode === 'sobolev') {
            try {
                field = solveConstrainedGradient(
                    live,
                    edges,
                    disjointPairs,
                    DEFAULTS.alpha,
                    DEFAULTS.beta,
                    DEFAULTS.epsilon,
                    dE,
                    st.sobolevX0,
                ).gTilde;
            } catch {
                // Singular saddle system (see sobolevStep's 'singular_system' contract):
                // no defined Sobolev direction — hide the arrows rather than show −dE
                // mislabelled as the sobolev field.
                mesh.count = 0;
                mesh.instanceMatrix.needsUpdate = true;
                return;
            }
        }

        let n = 0;
        for (let i = 0; i < count; i++) {
            const g = field[i];
            const gn = norm(g);
            if (gn <= 1e-6) continue; // parity with old index.tsx skip
            const dir: Vec3 = [-g[0] / gn, -g[1] / gn, -g[2] / gn];
            const len = Math.max(MIN_WORLD, Math.min(MAX_WORLD, Math.log(1 + gn) * ARROW_SCALE));
            dirVec.set(dir[0], dir[1], dir[2]);
            // coneGeometry is centred on its origin (height along local Y ∈ [-len/2, len/2]);
            // push it out by sphere radius + half length so the base rests on the sphere surface
            // and the cone extends outward along the descent direction rather than straddling
            // the vertex.
            const off = VERTEX_RADIUS + len / 2;
            tmp.position.set(
                live[i][0] + dir[0] * off,
                live[i][1] + dir[1] * off,
                live[i][2] + dir[2] * off,
            );
            tmp.quaternion.setFromUnitVectors(UP, dirVec);
            tmp.scale.set(1, len, 1);
            tmp.updateMatrix();
            mesh.setMatrixAt(n, tmp.matrix);
            n++;
        }
        mesh.count = n;
        mesh.instanceMatrix.needsUpdate = true;
    });

    if (!showArrows) return null;

    // Why: these hexes must stay in sync with the Stats.tsx legend (analytical=green,
    // finiteDiff=orange); editing one without the other silently desyncs the label
    // from the visual. Color keys on the dE source (mode), not the descent mode.
    const color = mode === 'analytical' ? '#00ff88' : '#ffaa00';
    return (
        <instancedMesh ref={meshRef} args={[undefined, undefined, count]}>
            <coneGeometry args={[CONE_RADIUS, 1, 8]} />
            <meshBasicNodeMaterial color={color} />
        </instancedMesh>
    );
}
