// src/gpu/reduce.ts — fixed 64-lane two-pass workgroup-memory tree sum (spec §2.2).
// No subgroupAdd (vendor-dependent tree shape), no Kahan/two-sum (WGSL §15.7.5 lets
// the compiler reassociate) — PREC Q3/Q5. Deterministic tree ⇒ reproducible sums.
// @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.2
// @see docs/2026-08-13-ai-research-gpu-precision.md Q3, Q5
// @see https://github.com/mrdoob/three.js/blob/r185/examples/webgpu_compute_reduce.html (REDUCE 2)
import {
    Fn,
    float,
    If,
    instancedArray,
    invocationLocalIndex,
    uint,
    workgroupArray,
    workgroupBarrier,
    workgroupId,
} from 'three/tsl';
import type { StorageBufferNode } from 'three/webgpu';

/**
 * Lanes per workgroup. Every compute node in the GPU solver is INTENDED to
 * dispatch with this workgroup size, so that the reduction and the kernels
 * share one shape: the two passes below already do, and the tangent-point
 * kernels (plan Task 3, `GpuTangentPoint`) must too.
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.2
 */
export const REDUCE_LANES = 64;

/**
 * Two 64-lane passes (each lane loads 2 values ⇒ 128 inputs per group)
 * cover ≤ 128·128 = 16,384 inputs. Pass 2 is a single group, so it can only
 * fold 128 partials — hence the hard cap. `GpuTangentPoint` rejects larger
 * edge counts before touching the renderer (D4 / Task 3).
 * @see docs/superpowers/plans/2026-08-29-webgpu-solver-phase1.md (Task 3, D4)
 */
export const REDUCE_MAX_EDGES = 128 * 128;

/**
 * `instancedArray(n, 'float')` / `attributeArray(Float32Array, 'float')` node.
 * Not `ReturnType<typeof instancedArray>`: `ArrayFunction` is overloaded
 * (@types/three `accessors/Arrays.d.ts`) and `ReturnType` picks the LAST
 * overload (`"uvec4"`), which would reject every real float buffer.
 */
type Arr = StorageBufferNode<'float'>;

/**
 * `workgroupArray(...).element(i)` exists at runtime
 * (`three/src/nodes/gpgpu/WorkgroupInfoNode.js:204`, returns a
 * `WorkgroupInfoElementNode` ⊂ `ArrayElementNode`) but @types/three 0.185
 * omits it from `WorkgroupInfoNode.d.ts`. Type it locally through the float
 * storage element (same assign/addAssign/add surface) rather than `any`.
 */
type SharedFloatArray = { element: (index: unknown) => ReturnType<Arr['element']> };

/**
 * Build the two compute nodes that sum `input[0..n)` into `out[0]`.
 *
 * Pass 1: ceil(n/128) groups; lane l loads `base+l` and `base+l+64` (both
 * guarded against `count`, so a partial tail group contributes zeros), tree
 * strides 32..1 with a workgroup barrier after each level; lane 0 writes the
 * group's partial. Pass 2: ONE group over the ≤128 partials, same tree,
 * writes `out[0]`. The tree shape is fixed by construction (no subgroup ops),
 * so the same input always reduces to the same f32 bits.
 *
 * Both nodes must be submitted in order in one `renderer.compute([...])`
 * (spec §2.5). Throws above `REDUCE_MAX_EDGES`.
 * @see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §2.2, §2.5
 * @see https://github.com/mrdoob/three.js/blob/r185/examples/webgpu_compute_reduce.html (REDUCE 2)
 */
export function makeReduceSum(input: Arr, n: number): { nodes: unknown[]; out: Arr } {
    if (n > REDUCE_MAX_EDGES) {
        throw new Error('makeReduceSum: input exceeds two-pass reduction limit');
    }
    const groups = Math.max(1, Math.ceil(n / (2 * REDUCE_LANES)));
    const partials = instancedArray(groups, 'float');
    const out = instancedArray(1, 'float');
    const pass = (src: Arr, count: number, dst: Arr, dispatchGroups: number) =>
        Fn(() => {
            const shared = workgroupArray('float', REDUCE_LANES) as unknown as SharedFloatArray;
            const lane = invocationLocalIndex;
            const base = workgroupId.x.mul(uint(2 * REDUCE_LANES));
            const a = base.add(lane);
            const b = a.add(uint(REDUCE_LANES));
            const va = float(0).toVar('va');
            const vb = float(0).toVar('vb');
            // Guards: the last group of pass 1 (and pass 2 whenever
            // groups < 128) reads past `count` without them. An out-of-bounds
            // storage read is a dynamic error with an implementation-chosen
            // outcome (any value from the bound GPUBuffer, the zero value for
            // the store type, or a trap — which then makes the invocation miss
            // `workgroupBarrier` below), so the tail must be guarded
            // explicitly (spec §2.2). WGSL §6.5.7 "Out-of-Bounds Access":
            // @see https://www.w3.org/TR/WGSL/#out-of-bounds-access-sec
            If(a.lessThan(uint(count)), () => {
                va.assign(src.element(a));
            });
            If(b.lessThan(uint(count)), () => {
                vb.assign(src.element(b));
            });
            shared.element(lane).assign(va.add(vb));
            workgroupBarrier();
            // Fixed stride tree 32→1: level s adds lane+s into lane for
            // lanes < s. Barrier after EVERY level — a missing one is a race
            // that shows up as a wrong integer in phase1Reduce.
            for (const s of [32, 16, 8, 4, 2, 1]) {
                If(lane.lessThan(uint(s)), () => {
                    shared.element(lane).addAssign(shared.element(lane.add(uint(s))));
                });
                workgroupBarrier();
            }
            If(lane.equal(uint(0)), () => {
                dst.element(workgroupId.x).assign(shared.element(uint(0)));
            });
        })().compute(dispatchGroups * REDUCE_LANES, [REDUCE_LANES]);
    return { nodes: [pass(input, n, partials, groups), pass(partials, groups, out, 1)], out };
}
