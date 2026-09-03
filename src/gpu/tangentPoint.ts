// src/gpu/tangentPoint.ts — renderer-bound tangent-point kernels. The ONLY src/gpu file importing three/tsl.
// TSL plumbing + wgslFn body (spec §2.1); one device; one upload + one readback per call (spec §5).
// @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md — "Task 3: `GpuTangentPoint` + reduction + T1/T2/T3 through the production kernel" / D1–D4
// @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.2, §2.5, §5 Phase 1
import { attributeArray, instancedArray, instanceIndex, uint, wgsl, wgslFn } from 'three/tsl';
import type { StorageBufferNode } from 'three/webgpu';
import * as THREE from 'three/webgpu';
import { DEFAULTS } from '../core/optimizer';
import type { Vec3 } from '../core/testConfigs';
import { makeReduceSum, REDUCE_LANES, REDUCE_MAX_EDGES } from './reduce';
import { type PackedTopology, writeHiLoInterleaved } from './topology';
import {
    energyKernelWgsl,
    gradientKernelWgsl,
    type KernelParams,
    pairKernelWgsl,
    vertexGatherWgsl,
} from './wgsl';

export { REDUCE_MAX_EDGES };

/**
 * Storage node of one f32 array. Typed explicitly (not `ReturnType<typeof
 * attributeArray>`) for the reason documented on `reduce.ts`'s `Arr`: the
 * @types/three 0.185 `ArrayFunction` overload set resolves `ReturnType` to
 * its LAST overload (`'uvec4'`). Class fields are annotated with these so a
 * reader of the class does not depend on TS 4.0 constructor inference.
 * @see src/gpu/reduce.ts (Arr)
 */
type FloatArr = StorageBufferNode<'float'>;
/** Storage node of one u32 array (the topology CSR buffers). Same reason as `FloatArr`. */
type UintArr = StorageBufferNode<'uint'>;

/**
 * The r185.1 backend surface `dispose()` relies on. @types/three 0.185's `Backend`
 * (`node_modules/@types/three/src/renderers/common/Backend.d.ts`) declares none of these
 * although all three are public methods in the JS — the same typing-gap class as GitHub
 * issue #22 (`as never` on `renderer.compute`).
 * Why: `has`/`get` — three/src/renderers/common/Backend.js:757-783 (`get` INSERTS an empty
 * record for an unknown object, so `has` must be asked first); `destroyAttribute` —
 * three/src/renderers/webgpu/WebGPUBackend.js:2595 → utils/WebGPUAttributeUtils.js:361-370
 * (`data.buffer.destroy()` then `backend.delete(attribute)`).
 */
type AttributeBackend = {
    has(object: object): boolean;
    get(object: object): { buffer?: unknown };
    destroyAttribute(attribute: THREE.BufferAttribute): void;
};

/**
 * Owns the storage buffers + compute nodes for one topology. `wallMs` covers
 * pack→upload→compute→readback→Vec3[] (plan D4; the ≥5× gate counts readback
 * inside the GPU dE phase — spec §5, review-3 F9). Energy is for gates and
 * the boot self-test only — never fed to descent (spec §2.3 never-mix).
 * New topology (graphVersion) ⇒ new instance; new (α,β,ε) ⇒ new instance
 * (α/β/ε are baked into the WGSL source at build time, `src/gpu/wgsl.ts`).
 * @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md — "Task 3: `GpuTangentPoint` + reduction + T1/T2/T3 through the production kernel" / D1–D4
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.2, §2.3, §2.5, §5
 */
export class GpuTangentPoint {
    readonly vertexCount: number;
    private readonly positions: FloatArr;
    private readonly slots: FloatArr;
    private readonly grad: FloatArr;
    private readonly partials: FloatArr;
    private readonly energyOut: FloatArr;
    /** The five topology CSR nodes, held so `dispose()` can release their GPU buffers (see `dispose`). */
    private readonly topology: readonly UintArr[];
    // `unknown[]` (not ComputeNode[]): mirrors `makeReduceSum`'s return type so the two node
    // lists concatenate without a cast at the call site; the `as never` at `renderer.compute` is
    // the known @types/three gap tracked in GitHub issue #22.
    private readonly gradNodes: unknown[];
    private readonly energyNodes: unknown[];
    private disposed = false;

    constructor(
        private readonly renderer: THREE.WebGPURenderer,
        topology: PackedTopology,
        params: KernelParams = DEFAULTS,
    ) {
        // Domain checks FIRST — before any renderer/TSL object is created (test/gpu/tangentPoint.test.ts
        // proves the renderer is never touched). Upper bound: pass 2 of the reduction folds ≤128 partials
        // (reduce.ts). Lower bounds: a 0-length storage attribute becomes a 0-byte GPUBuffer
        // (WebGPUAttributeUtils.js:92-96 sizes it from `array.byteLength`) and Dawn REJECTS it at bind-group
        // creation — verified on the Quadro RTX 3000, three r185.1: "Binding size for [Buffer] is zero …
        // While calling [Device].CreateBindGroup" — dropping the dispatch, so gradient()/energy() would
        // return finite all-zero readbacks with no exception. Full hardware evidence (the complete Dawn
        // message, the all-zero readback table, the onuncapturederror path):
        // @issue utof/repulsive-test2#25
        // @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md — "Task 2: Topology packing, interleaved hi/lo writer, guard mirrors (pure)"
        // `edgeCount === 0` zero-sizes `slots`/`partials`/`edges`; `pairIndices.length === 0` (every edge
        // touches every other: one edge, a 2-edge path, a triangle …) zero-sizes `pairIndices` alone. The
        // driver (plan "Task 6: Store, boot self-test, driver module") treats a constructor throw as
        // "route this step to the CPU worker".
        if (topology.edgeCount > REDUCE_MAX_EDGES)
            throw new Error('GpuTangentPoint: edge count exceeds two-pass reduction limit');
        if (topology.edgeCount === 0) throw new Error('GpuTangentPoint: topology has no edges');
        if (topology.pairIndices.length === 0)
            throw new Error('GpuTangentPoint: topology has no disjoint pairs');
        this.vertexCount = topology.vertexCount;
        // One dynamic positions buffer: [hi.xyz, lo.xyz] per vertex. setUsage verified: three/src/core/BufferAttribute.js:167.
        // .toReadOnly(): the kernels declare `p: ptr<storage, array<f32>, read>` and the emitted access comes from the NODE
        // (WGSLNodeBuilder.js:2155 `getStorageAccess(bufferNode)`) — a read_write node bound to a `read` pointer param is a
        // shader compile error (plan "Task 1: WGSL source generators (pure)" — pointer access mode must match
        // the node's). CPU upload via needsUpdate needs no shader write access.
        this.positions = attributeArray(
            new Float32Array(6 * topology.vertexCount),
            'float',
        ).toReadOnly();
        this.positions.value.setUsage(THREE.DynamicDrawUsage);
        // Topology CSR: uploaded once per instance (spec §2.2), every one a `read` pointer in the WGSL ⇒ .toReadOnly().
        const edges = attributeArray(topology.edges, 'uint').toReadOnly();
        const pairOffsets = attributeArray(topology.pairOffsets, 'uint').toReadOnly();
        const pairIndices = attributeArray(topology.pairIndices, 'uint').toReadOnly();
        const incidentOffsets = attributeArray(topology.incidentOffsets, 'uint').toReadOnly();
        const incidentSlots = attributeArray(topology.incidentSlots, 'uint').toReadOnly();
        this.topology = [edges, pairOffsets, pairIndices, incidentOffsets, incidentSlots];
        // GPU-written buffers stay read_write (plain instancedArray) — their WGSL params are `read_write`.
        // `slots` is bound read_write in BOTH tpGradient (writer) and tpVertexGather (reader): one node, one access
        // mode (wgsl.ts vertexGatherWgsl TSDoc; plan "Task 1: WGSL source generators (pure)").
        this.slots = instancedArray(6 * topology.edgeCount, 'float');
        this.grad = instancedArray(3 * topology.vertexCount, 'float');
        this.partials = instancedArray(topology.edgeCount, 'float');
        // Object keys MUST equal the WGSL fn parameter names — FunctionCallNode.js:150 looks each input up by name and
        // substitutes `float(0)` (with a console error, not a throw) for a missing one. `uint(n)` is a ConstNode emitted
        // as the literal `<n>u` (NodeBuilder.js:1433); `instanceIndex` is the g0t-spike pattern (bench/gpu/spikes.ts:161).
        const common = {
            p: this.positions,
            edges,
            pairOffsets,
            pairIndices,
            edgeCount: uint(topology.edgeCount),
            I: instanceIndex,
        };
        // Shared struct+helpers go in as an INCLUDE (CodeNode.js:170 wgsl(src, includes); FunctionNode.js:168
        // wgslFn(code, includes)) — three's WGSL parser accepts exactly one bare `fn` per wgslFn
        // (WGSLNodeFunction.js:4 `^fn name(...)` regex; plan "Task 1: WGSL source generators (pure)").
        const shared = wgsl(pairKernelWgsl(params));
        const gradFn = wgslFn(gradientKernelWgsl(params), [shared]);
        const gatherFn = wgslFn(vertexGatherWgsl());
        const energyFn = wgslFn(energyKernelWgsl(params), [shared]);
        // Workgroup size = REDUCE_LANES everywhere (reduce.ts TSDoc: one shape for every solver dispatch). The thread
        // count is rounded UP to whole workgroups (WebGPUBackend.js:1666 `Math.ceil(count / size)`); the kernels'
        // own `if (I >= edgeCount) { return; }` / `if (v >= vertexCount)` guards absorb the tail (wgsl.ts) — do not
        // rely solely on the early return ComputeNode.js:215 injects, which is conditional on `allowEarlyReturns`.
        this.gradNodes = [
            gradFn({ ...common, slots: this.slots }).compute(topology.edgeCount, [REDUCE_LANES]),
            gatherFn({
                slots: this.slots,
                incidentOffsets,
                incidentSlots,
                grad: this.grad,
                vertexCount: uint(topology.vertexCount),
                v: instanceIndex,
            }).compute(topology.vertexCount, [REDUCE_LANES]),
        ];
        const red = makeReduceSum(this.partials, topology.edgeCount);
        this.energyOut = red.out;
        this.energyNodes = [
            energyFn({ ...common, partials: this.partials }).compute(topology.edgeCount, [
                REDUCE_LANES,
            ]),
            ...red.nodes,
        ];
    }

    /**
     * The ONE positions upload per call (spec §5 Phase 1): in-place hi/lo
     * rewrite of the attribute's own array + `needsUpdate` (G4-spike pattern,
     * bench/gpu/spikes.ts:645-649). `.value` is the `StorageBufferAttribute`
     * (Arrays.js `attributeArray` → `storage(buffer, …)`; Phase 0 readback
     * used the same `buf.value`, spikes.ts:481).
     */
    private upload(vertices: Vec3[]): void {
        if (this.disposed) throw new Error('GpuTangentPoint: disposed');
        if (vertices.length !== this.vertexCount)
            throw new Error('GpuTangentPoint: vertex count mismatch');
        writeHiLoInterleaved(vertices, this.positions.value.array as Float32Array);
        this.positions.value.needsUpdate = true; // the one upload (G4-spike pattern, spikes.ts:645-649)
    }

    /**
     * dE at `vertices` (Vec3[] of length `vertexCount`), the D1 gather in one
     * submit: edge kernel (both roles, ×0.5, writes I's two slots) then the
     * vertex gather over the incidence CSR. `wallMs` = upload → compute →
     * readback → Vec3[] (plan D4 timing method — readback INCLUDED).
     * @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (D1, D4)
     * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.2, §2.5
     */
    async gradient(vertices: Vec3[]): Promise<{ dE: Vec3[]; wallMs: number }> {
        const t0 = performance.now();
        this.upload(vertices);
        this.renderer.compute(this.gradNodes as never); // edge gather + vertex gather, ONE submit (spec §2.5)
        // ONE readback per call (spec §5). `as never` on compute: @types/three gap, GitHub issue #22.
        const flat = new Float32Array(await this.renderer.getArrayBufferAsync(this.grad.value));
        const dE: Vec3[] = Array.from({ length: this.vertexCount }, (_, i) => [
            flat[3 * i],
            flat[3 * i + 1],
            flat[3 * i + 2],
        ]);
        return { dE, wallMs: performance.now() - t0 };
    }

    /**
     * Total energy at `vertices`: per-edge partials (role A only, ×0.5) then
     * the two-pass 64-lane reduction, one submit. Gate/boot-self-test use ONLY
     * — never fed to descent (spec §2.3 never-mix).
     * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.2, §2.3
     * @see src/gpu/reduce.ts (makeReduceSum)
     */
    async energy(vertices: Vec3[]): Promise<{ energy: number; wallMs: number }> {
        const t0 = performance.now();
        this.upload(vertices);
        this.renderer.compute(this.energyNodes as never); // kernel + 2 reduce passes, ONE submit
        const out = new Float32Array(await this.renderer.getArrayBufferAsync(this.energyOut.value));
        return { energy: out[0], wallMs: performance.now() - t0 };
    }

    /**
     * Release pipelines/bindings and the GPU buffers of every attribute this
     * instance owns (`positions`, the five topology CSR buffers, `slots`,
     * `grad`, `partials`, `energyOut`); the instance is unusable afterwards
     * (`upload` throws). NOT released: `makeReduceSum`'s internal pass-1
     * partials buffer (`src/gpu/reduce.ts:75`, ≤128 f32 = ≤512 B per
     * instance) — it is not returned by that exported function, so it is only
     * reclaimed by GC of the node. Tracked in GitHub issue #22 (comment).
     * @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md — "Task 3: `GpuTangentPoint` + reduction + T1/T2/T3 through the production kernel"
     * @issue utof/repulsive-test2#22
     */
    dispose(): void {
        this.disposed = true;
        // 1. Compute nodes: ComputeNode.dispose (ComputeNode.js:116) dispatches 'dispose'; the renderer's listener
        //    (Renderer.js:2774-2780, registered on first compute) drops the pipeline, bind groups and node cache.
        //    That releases ONLY uniform buffers and samplers (Bindings.js:_destroyBindings, 248-283) — storage
        //    buffers are skipped, so the attributes below must be destroyed explicitly. Nodes first, so no bind
        //    group still references a destroyed buffer.
        for (const n of [...this.gradNodes, ...this.energyNodes])
            (n as { dispose: () => void }).dispose();
        // 2. Storage attributes. In r185.1 `BufferAttribute.dispose()` (node_modules/three/src/core/
        //    BufferAttribute.js:683-687) ONLY dispatches a 'dispose' EVENT — it frees no GPU memory — and NO
        //    renderer/backend listener is registered on a compute-only storage attribute (the sole
        //    attribute-side 'dispose' listener, WebGPUAttributeUtils.js:443, targets a ReadbackBuffer, which this
        //    class never uses — `getArrayBufferAsync(attr)` with target=null destroys its staging buffer inline,
        //    WebGPUAttributeUtils.js:503). `Attributes.delete` (Attributes.js:46-56) is the only path that frees
        //    the GPUBuffer and it is reached solely from geometry disposal (Geometries.js:194-208). So call what
        //    it calls: `backend.destroyAttribute` (node_modules/three/src/renderers/webgpu/utils/
        //    WebGPUAttributeUtils.js:361-370 — `data.buffer.destroy()` + `backend.delete(attribute)`) and
        //    `info.destroyAttribute` (node_modules/three/src/renderers/common/Info.js:324-334 — removes the STRONG
        //    `memoryMap` entry that would otherwise pin the JS attribute and its CPU array for the renderer's
        //    lifetime, and keeps `renderer.info.memory` honest). Guarded with `has` + `buffer`: an attribute never
        //    uploaded (e.g. `partials` when `energy()` was never called) has no backend record, and
        //    `destroyAttribute` would throw on `undefined.destroy()`.
        //    Verified before/after on the Quadro RTX 3000 / three r0.185.1: `queue.writeBuffer` into each buffer
        //    after dispose ⇒ "[Buffer (unlabeled)] used in submit while destroyed. - While calling
        //    [Queue].WriteBuffer(...)"; `backend.has` false for all ten; `renderer.info.memory.storageAttributes`
        //    11 → 1 (1504 B → 4 B, the one surviving buffer being the reduce.ts partials of issue #22).
        const backend = this.renderer.backend as THREE.Backend & AttributeBackend;
        const owned = [
            this.positions,
            ...this.topology,
            this.slots,
            this.grad,
            this.partials,
            this.energyOut,
        ];
        for (const b of owned) {
            const attr = b.value;
            if (backend.has(attr) && backend.get(attr).buffer !== undefined)
                backend.destroyAttribute(attr);
            this.renderer.info.destroyAttribute(attr);
            attr.dispose(); // end-of-life event for any listener a caller attached; inert inside three r185.1 (above)
        }
    }
}
