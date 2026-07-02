/**
 * Serializes the deterministic test configurations to JSON fixtures for the
 * Python Stage-1 oracle (`oracle/tpe_stage1_oracle.py`).
 *
 * Why: the Sobolev-gradient TS implementation (Stage 1) is verified by diffing
 * against an independent Python reference on these exact inputs. Only
 * deterministic configs are usable as fixtures — `stress`/`random`/`chain` use
 * unseeded Math.random and are excluded on purpose.
 *
 * @see local_files/sobolev-gradient-handoff.md §5 (verification protocol)
 *
 * Run: bun oracle/gen_fixtures.ts
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import type { Edge, GraphState, Vec3 } from '../src/core/testConfigs';
import { testConfigs } from '../src/core/testConfigs';

// Deterministic subset of the app's presets (no Math.random in their generators).
const DETERMINISTIC_IDS = ['crossing', 'helix', 'linked-rings', 'knot'] as const;

/**
 * Y-junction + crossing bar: the one topology class missing from the app's
 * deterministic presets. Central vertex of degree 3 (junction), three open
 * arms (endpoint vertices), plus a disjoint 3-edge polyline passing 0.35
 * above the junction plane so near-approach disjoint pairs exist.
 *
 * Why: the handoff requires the assembly to be exercised on graphs with
 * degree-≥3 junctions with NO special-casing — this fixture is that test.
 * @see local_files/sobolev-gradient-handoff.md §2 ("junction vertices of degree ≥ 3")
 */
function createJunctionY(): GraphState {
    const vertices: Vec3[] = [[0, 0, 0]];
    const edges: Edge[] = [];
    const dirs: Vec3[] = [
        [1, 0, 0],
        [-0.5, Math.sqrt(3) / 2, 0],
        [-0.5, -Math.sqrt(3) / 2, 0],
    ];
    for (let arm = 0; arm < 3; arm++) {
        let prev = 0;
        for (let k = 1; k <= 3; k++) {
            const t = 0.5 * k;
            // deterministic out-of-plane bend so the graph is genuinely 3D
            vertices.push([dirs[arm][0] * t, dirs[arm][1] * t, 0.15 * Math.sin(t * (arm + 1))]);
            const idx = vertices.length - 1;
            edges.push([prev, idx]);
            prev = idx;
        }
    }
    const bar: Vec3[] = [
        [-1.2, -0.4, 0.35],
        [-0.4, -0.13, 0.35],
        [0.4, 0.13, 0.35],
        [1.2, 0.4, 0.35],
    ];
    const base = vertices.length;
    for (const v of bar) vertices.push(v);
    for (let k = 0; k < 3; k++) edges.push([base + k, base + k + 1]);
    return { vertices, edges };
}

const outDir = join(import.meta.dir, 'fixtures');
mkdirSync(outDir, { recursive: true });

const cases: { id: string; state: GraphState }[] = [
    ...DETERMINISTIC_IDS.map((id) => {
        const cfg = testConfigs.find((c) => c.id === id);
        if (!cfg) throw new Error(`missing config ${id}`);
        return { id, state: cfg.generate() };
    }),
    { id: 'junction-y', state: createJunctionY() },
];

for (const { id, state } of cases) {
    const fixture = {
        name: id,
        vertices: state.vertices,
        edges: state.edges,
        alpha: 3,
        beta: 6,
        epsilon: 1e-10,
    };
    const path = join(outDir, `${id}.json`);
    writeFileSync(path, JSON.stringify(fixture));
    console.log(`${path}: |V|=${state.vertices.length} |E|=${state.edges.length}`);
}
