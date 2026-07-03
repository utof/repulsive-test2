import {
    dispatchDescentStep,
    type SolverWorkerRequest,
    type SolverWorkerResponse,
} from '../core/dispatch';
import { calculateDisjointPairs } from '../core/tangentPointEnergy';
import type { Edge } from '../core/testConfigs';

/**
 * Stateless compute-server solver worker (§2, §D11). Holds NO authoritative
 * state beyond a topology cache — the store (main thread) stays the single
 * source of truth. Each `step` runs the SAME pure {@link dispatchDescentStep}
 * on the SAME inputs the main thread would pass, so per-step results are
 * bit-identical to the synchronous path (§2; gate: test/worker-solver.test.ts).
 * Imports ONLY from src/core/** so the worker bundle never pulls React/zustand
 * (§D2/§D11).
 * @see docs/superpowers/plans/2026-07-04-worker-solver.md §2, §D4, §D5, §D11
 */

// Bun/browser worker-side typing: shadow the global `self` with the Worker type
// so `self.onmessage` / `self.postMessage` are the worker-scope members (not
// Window's). Module-scoped `declare` — shadows locally, no global conflict.
// @see https://bun.sh/docs/api/workers ("declare var self: Worker")
declare var self: Worker;

// Topology cache (§D4): the O(E²) disjointPairs is recomputed ONCE per topology
// via the same deterministic calculateDisjointPairs the store uses (⇒ identical
// arrays), so per-step messages carry only vertices + dynamic config, never the
// disjointPairs. Refreshed on init and on every graphVersion change.
// @see docs/superpowers/plans/2026-07-04-worker-solver.md §D4
let topology: { graphVersion: number; edges: Edge[]; disjointPairs: number[][] } | null = null;

self.onmessage = (event: MessageEvent<SolverWorkerRequest>) => {
    const msg = event.data;
    try {
        if (msg.type === 'topology') {
            topology = {
                graphVersion: msg.graphVersion,
                edges: msg.edges,
                disjointPairs: calculateDisjointPairs(msg.edges),
            };
            return;
        }
        // msg.type === 'step': restore the topology fields from the cache (§D4)
        // and run the SAME pure function the main thread would (§2 bit-identity).
        if (topology === null) throw new Error('solverWorker: step received before topology');
        const result = dispatchDescentStep({
            ...msg.args,
            edges: topology.edges,
            disjointPairs: topology.disjointPairs,
        });
        // Echo the request's graphVersion UNTOUCHED so the main thread can run
        // the §D5 mismatch-drop test against its current store value.
        // @see docs/superpowers/plans/2026-07-04-worker-solver.md §D5
        const response: SolverWorkerResponse = {
            type: 'result',
            graphVersion: msg.graphVersion,
            result,
        };
        self.postMessage(response);
    } catch (err) {
        // Any throw becomes a protocol error; the main thread auto-falls back to
        // the synchronous 'main' driver on receipt (§D6). Never let the worker
        // die silently — the main driver would stall waiting for a result.
        // @see docs/superpowers/plans/2026-07-04-worker-solver.md §D6, §D11
        const response: SolverWorkerResponse = {
            type: 'error',
            message: err instanceof Error ? err.message : String(err),
        };
        self.postMessage(response);
    }
};
