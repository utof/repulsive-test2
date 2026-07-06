import { useFrame } from '@react-three/fiber';
import { useEffect, useRef } from 'react';
import * as THREE from 'three/webgpu';
import { computeArrowField } from '../core/arrowField';
import { norm } from '../core/tangentPointEnergy';
import type { Vec3 } from '../core/testConfigs';
import { type SolverWorkerRequest, type SolverWorkerResponse, useSimStore } from '../store';

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

/**
 * Orient the instanced cones along the NEGATED descent field at each vertex.
 * Shared by the worker-result path (with the received field + current live
 * positions) and the synchronous fallback (§D13-c). `field` is the RAW field
 * (dE or g̃); this negates it, mirroring the old inline loop verbatim. Cone
 * bases sit on the live spheres; a ≤1-step direction staleness is acceptable
 * for a ≤5 Hz diagnostic (the field may be one step older than `live`).
 * @see docs/superpowers/plans/2026-07-04-worker-solver.md §D13
 */
function applyConeMatrices(mesh: THREE.InstancedMesh, field: Vec3[], live: Vec3[], count: number) {
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
}

// Why: instanced cones pointing along the DESCENT direction at each vertex, live in both
// paused and running states (reads the non-reactive `live` buffer like Curve.tsx), gated by
// the store's showArrows toggle. The field matches the ACTIVE descent mode: raw → −dE (the
// L² gradient), sobolev → −g̃ (the constrained Sobolev gradient from the saddle solve) — the
// whole point of the A/B toggle is seeing that these directions differ.
//
// §D13: the field compute (raw O(E²) gradient; sobolev a full dense saddle solve) runs OFF the
// main thread on a DEDICATED arrows worker — separate from Simulation's step worker so a
// ~500 ms field compute at N=200 never serializes behind (or blocks) descent steps. useFrame
// only POSTs a field request on key change (single-flight) and applies the received field; the
// synchronous computeArrowField is the permanent fallback if the worker fails. This is
// independent of the store's solverDriver toggle (that governs the STEP path only).
// @see docs/superpowers/plans/2026-07-04-worker-solver.md §D13
// @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("Gradient saddle system"), §C
export function GradientArrows() {
    const showArrows = useSimStore((s) => s.showArrows);
    const mode = useSimStore((s) => s.mode);
    const count = useSimStore((s) => s.graph.vertices.length);

    const meshRef = useRef<THREE.InstancedMesh>(null);
    const acc = useRef(Number.POSITIVE_INFINITY); // ∞ → first frame always computes
    const lastKey = useRef('');
    // Arrows-worker state (§D13-c), mirroring Viewer's step-worker refs. workerRef: the
    // dedicated arrows worker (null until the effect constructs it / after teardown).
    // inFlight: single-flight guard — true between a field SEND and its result (like §D1),
    // so at most one field compute is ever outstanding. sentTopologyVersion: the graphVersion
    // whose topology the worker has cached (§D4); -1 forces a resend on a fresh worker.
    // fallbackToSync: PERMANENT flag — once the worker fails, useFrame computes inline forever.
    // @see docs/superpowers/plans/2026-07-04-worker-solver.md §D13
    const workerRef = useRef<Worker | null>(null);
    const inFlight = useRef(false);
    const sentTopologyVersion = useRef(-1);
    const fallbackToSync = useRef(false);

    // Arrows-worker lifecycle (§D13-c): construct when showArrows is on, tear down on toggle-off
    // / unmount (the component is keyed on graphVersion in Viewer, so a topology change also
    // remounts and rebuilds the worker). Auto-fall-back PERMANENTLY to the synchronous inline
    // compute on construction failure or any worker error (posted or uncaught) — the fallback
    // reuses the SAME computeArrowField so the two paths cannot drift.
    // @see docs/superpowers/plans/2026-07-04-worker-solver.md §D13
    useEffect(() => {
        if (!showArrows) return;
        if (fallbackToSync.current) return; // already fell back permanently — never rebuild
        let worker: Worker;
        try {
            // §D3: dev-server PATH STRING, NOT new URL(import.meta.url) — Bun.build emits no
            // worker chunk for browser targets, so the plain string is load-bearing; the dev
            // server bundles any requested .ts on the fly (server.ts, the `.ts` branch). Same
            // load-bearing reason as Viewer.tsx's step worker.
            // @see docs/superpowers/plans/2026-07-04-worker-solver.md §D3
            worker = new Worker('/src/worker/solverWorker.ts', { type: 'module' });
        } catch (err) {
            console.error('GradientArrows: worker construction failed; falling back to sync:', err);
            fallbackToSync.current = true;
            return;
        }
        worker.onmessage = (event: MessageEvent<SolverWorkerResponse>) => {
            const resp = event.data;
            if (resp.type === 'error') {
                // A worker-side throw (e.g. field-before-topology) → permanent §D13-c fallback
                // so the arrows keep updating (synchronously) instead of freezing.
                console.error(
                    'GradientArrows: worker posted an error; falling back to sync:',
                    resp.message,
                );
                fallbackToSync.current = true;
                inFlight.current = false;
                lastKey.current = ''; // force an immediate sync recompute next tick
                return;
            }
            if (resp.type !== 'fieldResult') return; // this worker only handles field requests
            inFlight.current = false;
            const mesh = meshRef.current;
            if (!mesh) return;
            const st = useSimStore.getState();
            // §D13-c drop rule: a field whose graphVersion no longer matches the store landed
            // after a topology change — its index mapping is stale, so discard it and let the
            // next useFrame tick re-request. (In practice the graphVersion key remounts this
            // component on such a change; this is the defensive mid-flight boundary.)
            if (resp.graphVersion !== st.graphVersion) return;
            if (resp.field === null) {
                // Singular saddle system: no defined Sobolev direction — hide the arrows
                // (today's behavior). @see §D13 / GradientArrows old try/catch.
                mesh.count = 0;
                mesh.instanceMatrix.needsUpdate = true;
                return;
            }
            // Apply the RECEIVED field against the CURRENT live positions (cone bases track the
            // live spheres; ≤1-step direction staleness is fine for a ≤5 Hz diagnostic).
            applyConeMatrices(mesh, resp.field, st.live, st.graph.vertices.length);
        };
        worker.onerror = (event: ErrorEvent) => {
            // A worker that fails to LOAD fires onerror without a ctor throw and can never post
            // a protocol error — without this the arrows would freeze on a stuck inFlight.
            console.error('GradientArrows: worker error; falling back to sync:', event.message);
            fallbackToSync.current = true;
            inFlight.current = false;
            lastKey.current = '';
        };
        workerRef.current = worker;
        // Fresh worker: no in-flight request, no topology cached yet (force a resend), and
        // clear the memo key so the first tick re-requests against the new worker.
        inFlight.current = false;
        sentTopologyVersion.current = -1;
        lastKey.current = '';
        return () => {
            // Null onmessage BEFORE terminate: a fieldResult already queued on this worker must
            // not fire after teardown (same reason as Viewer.tsx's step-worker cleanup).
            worker.onmessage = null;
            worker.terminate();
            workerRef.current = null;
            inFlight.current = false;
        };
    }, [showArrows]);

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

        // INVARIANT: `live` is index-aligned with this render's `count` — same
        // graphVersion-keyed remount contract as Curve.tsx (see its useFrame note).
        const live = st.live;

        if (fallbackToSync.current) {
            // §D13-c permanent fallback: today's exact synchronous path, reusing the SAME
            // helper the worker runs so the two can't drift.
            acc.current = 0;
            lastKey.current = key;
            const field = computeArrowField(
                live,
                st.graph.edges,
                st.disjointPairs,
                st.mode,
                st.descentMode,
                st.sobolevX0,
            );
            if (field === null) {
                mesh.count = 0;
                mesh.instanceMatrix.needsUpdate = true;
                return;
            }
            applyConeMatrices(mesh, field, live, st.graph.vertices.length);
            return;
        }

        // Worker path (§D13-c): POST a field request instead of computing inline. Single-flight
        // (like §D1) — while a request is outstanding we neither re-post nor advance lastKey, so
        // the next tick re-evaluates once the result clears inFlight (no onmessage loop).
        const worker = workerRef.current;
        if (!worker) return; // effect not yet run / torn down this frame
        if (inFlight.current) return;
        // §D4: (re)send topology once per graphVersion BEFORE the field request so the worker
        // caches disjointPairs and per-request messages never carry the O(E²) pairs.
        if (st.graphVersion !== sentTopologyVersion.current) {
            sentTopologyVersion.current = st.graphVersion;
            const topo: SolverWorkerRequest = {
                type: 'topology',
                graphVersion: st.graphVersion,
                edges: st.graph.edges,
            };
            worker.postMessage(topo);
        }
        const req: SolverWorkerRequest = {
            type: 'field',
            graphVersion: st.graphVersion,
            args: { vertices: live, mode: st.mode, descentMode: st.descentMode, x0: st.sobolevX0 },
        };
        worker.postMessage(req);
        inFlight.current = true;
        acc.current = 0;
        lastKey.current = key;
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
