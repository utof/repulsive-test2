import { testConfigs } from '../core/testConfigs';
import {
    type DescentMode,
    type LengthMode,
    type Mode,
    type ProjectionMode,
    useSimStore,
} from '../store';

const btn = (bg: string) => ({
    padding: '10px 20px',
    fontSize: 16,
    cursor: 'pointer',
    background: bg,
    color: '#fff',
    border: 'none',
    borderRadius: 5,
});

export function ControlPanel() {
    const selectedTestId = useSimStore((s) => s.selectedTestId);
    const testParams = useSimStore((s) => s.testParams);
    const running = useSimStore((s) => s.running);
    const mode = useSimStore((s) => s.mode);
    const descentMode = useSimStore((s) => s.descentMode);
    const stepSize = useSimStore((s) => s.stepSize);
    const setPreset = useSimStore((s) => s.setPreset);
    const setParam = useSimStore((s) => s.setParam);
    const regenerate = useSimStore((s) => s.regenerate);
    const reset = useSimStore((s) => s.reset);
    const setMode = useSimStore((s) => s.setMode);
    const setDescentMode = useSimStore((s) => s.setDescentMode);
    const setStepSize = useSimStore((s) => s.setStepSize);
    const barycenterConstraint = useSimStore((s) => s.barycenterConstraint);
    const lengthMode = useSimStore((s) => s.lengthMode);
    const setBarycenterConstraint = useSimStore((s) => s.setBarycenterConstraint);
    const setLengthMode = useSimStore((s) => s.setLengthMode);
    const projectionMode = useSimStore((s) => s.projectionMode);
    const setProjectionMode = useSimStore((s) => s.setProjectionMode);
    const showArrows = useSimStore((s) => s.showArrows);
    const setShowArrows = useSimStore((s) => s.setShowArrows);
    const setRunning = useSimStore((s) => s.setRunning);
    // Interactive point pins (briefing §5B): the list is populated by clicking a
    // vertex in the 3D view (PinControls); here the user enables/removes each.
    const pins = useSimStore((s) => s.pins);
    const setPinEnabled = useSimStore((s) => s.setPinEnabled);
    const removePin = useSimStore((s) => s.removePin);
    // Soft-constraint penalty catalog (5C) + target-length animation (plan §4
    // Task 5). Weight sliders (0 = off), an X-vector input for the field penalty,
    // and the growth-rate control.
    const penalties = useSimStore((s) => s.penalties);
    const setPenaltyTotalLength = useSimStore((s) => s.setPenaltyTotalLength);
    const setPenaltyLengthDiff = useSimStore((s) => s.setPenaltyLengthDiff);
    const setPenaltyFieldWeight = useSimStore((s) => s.setPenaltyFieldWeight);
    const setPenaltyFieldX = useSimStore((s) => s.setPenaltyFieldX);
    const lengthGrowthRate = useSimStore((s) => s.lengthGrowthRate);
    const setLengthGrowthRate = useSimStore((s) => s.setLengthGrowthRate);

    const selectedTest = testConfigs.find((t) => t.id === selectedTestId) ?? testConfigs[0];
    // Field-penalty X vector (store default [1,0,0]); read defensively since
    // PenaltyConfig types `field` optional (the store keeps it defined).
    const fieldWeight = penalties.field?.weight ?? 0;
    const fieldX = penalties.field?.X ?? [1, 0, 0];

    return (
        <>
            {/* Test selection + params */}
            <div
                style={{
                    marginBottom: 10,
                    display: 'flex',
                    gap: 15,
                    flexWrap: 'wrap',
                    alignItems: 'center',
                }}
            >
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span>Test:</span>
                    <select
                        value={selectedTestId}
                        onChange={(e) => setPreset(e.target.value)}
                        style={{ padding: 8, fontSize: 14, minWidth: 200 }}
                    >
                        {testConfigs.map((t) => (
                            <option key={t.id} value={t.id}>
                                {t.name}
                            </option>
                        ))}
                    </select>
                </label>

                {selectedTest.params?.map((p) => (
                    <label key={p.name} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span>{p.name}:</span>
                        <input
                            type="range"
                            min={p.min}
                            max={p.max}
                            value={testParams[p.name] ?? p.default}
                            onChange={(e) => setParam(p.name, parseInt(e.target.value))}
                            style={{ width: 80 }}
                        />
                        <span style={{ fontFamily: 'monospace', width: 40 }}>
                            {testParams[p.name] ?? p.default}
                        </span>
                    </label>
                ))}

                {selectedTest.params && (
                    <button type="button" onClick={regenerate} style={btn('#5577cc')}>
                        Regenerate
                    </button>
                )}
            </div>

            {/* Run controls */}
            <div
                style={{
                    marginBottom: 15,
                    display: 'flex',
                    gap: 15,
                    flexWrap: 'wrap',
                    alignItems: 'center',
                }}
            >
                <button
                    type="button"
                    onClick={() => setRunning(!running)}
                    style={btn(running ? '#ff4444' : '#44aa44')}
                >
                    {running ? 'Stop' : 'Start'}
                </button>
                <button type="button" onClick={reset} style={btn('#666')}>
                    Reset
                </button>
                {/* A/B toggle between the untouched raw descent and the constrained Sobolev
                    flow — the point is comparing raw τ≈1e-5 vs Sobolev τ≈1 side by side.
                    @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §C */}
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span>Descent:</span>
                    <select
                        value={descentMode}
                        onChange={(e) => setDescentMode(e.target.value as DescentMode)}
                        style={{ padding: 8, fontSize: 14 }}
                    >
                        <option value="raw">Raw L²</option>
                        <option value="sobolev">Sobolev</option>
                    </select>
                </label>
                {/* Per-constraint-block toggles for the sobolev ConstraintSet (one
                    checkbox per block). Sobolev-only: disabled (not hidden) in raw
                    mode — the raw path takes no constraints and stays byte-identical.
                    @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §4.3, §9a */}
                <label
                    style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 8,
                        opacity: descentMode === 'sobolev' ? 1 : 0.4,
                    }}
                >
                    <input
                        type="checkbox"
                        checked={barycenterConstraint}
                        disabled={descentMode !== 'sobolev'}
                        onChange={(e) => setBarycenterConstraint(e.target.checked)}
                    />
                    <span>Barycenter</span>
                </label>
                {/* 3-way Length select (M2, spec §5.3) replacing the M1 "Fix length"
                    checkbox: none | total | per-edge. The §3.4 totalLength/edgeLengths
                    mutual exclusion is enforced BY CONSTRUCTION — one select, one
                    value. Sobolev-only: disabled (not hidden) in raw mode, same as
                    the Barycenter checkbox.
                    @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §5.3, §3.4 */}
                <label
                    style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 8,
                        opacity: descentMode === 'sobolev' ? 1 : 0.4,
                    }}
                >
                    <span>Length:</span>
                    <select
                        value={lengthMode}
                        disabled={descentMode !== 'sobolev'}
                        onChange={(e) => setLengthMode(e.target.value as LengthMode)}
                        style={{ padding: 8, fontSize: 14 }}
                    >
                        <option value="none">None</option>
                        <option value="total">Total</option>
                        <option value="perEdge">Per-edge</option>
                    </select>
                </label>
                {/* Projection strategy A/B (solver-perf Task 6): frozen = the
                    reference implementation's one-LU-per-step reuse (default),
                    reassemble = per-iterate rebuild (stricter step quality on
                    junction fixtures — see oracle/README.md measurement table).
                    Sobolev-only, same disabled treatment as the Length select. */}
                <label
                    style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 8,
                        opacity: descentMode === 'sobolev' ? 1 : 0.4,
                    }}
                >
                    <span>Projection:</span>
                    <select
                        value={projectionMode}
                        disabled={descentMode !== 'sobolev'}
                        onChange={(e) => setProjectionMode(e.target.value as ProjectionMode)}
                        style={{ padding: 8, fontSize: 14 }}
                    >
                        <option value="frozen">Frozen (reuse)</option>
                        <option value="reassemble">Reassemble</option>
                    </select>
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span>Mode:</span>
                    <select
                        value={mode}
                        onChange={(e) => setMode(e.target.value as Mode)}
                        style={{ padding: 8, fontSize: 14 }}
                    >
                        <option value="analytical">Analytical</option>
                        <option value="finiteDiff">Finite Diff</option>
                    </select>
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span>Step Size:</span>
                    <input
                        type="range"
                        min="-5"
                        max="-2"
                        step="0.1"
                        value={Math.log10(stepSize)}
                        onChange={(e) => setStepSize(10 ** parseFloat(e.target.value))}
                        style={{ width: 100 }}
                    />
                    <span style={{ fontFamily: 'monospace', width: 60 }}>
                        {stepSize.toExponential(0)}
                    </span>
                </label>
                {/* Descent-direction arrows toggle — visible in BOTH paused and running
                    states (see GradientArrows), not just on pause. */}
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <input
                        type="checkbox"
                        checked={showArrows}
                        onChange={(e) => setShowArrows(e.target.checked)}
                    />
                    <span>Arrows</span>
                </label>
            </div>

            {/* Interactive point-pin list (briefing §5B). Pins are created by
                clicking a vertex in the 3D view (PinControls); each becomes a
                point constraint in the sobolev ConstraintSet, so like the
                constraint toggles the "Pins:" label dims in raw mode (picking
                still works — the marker shows — but only sobolev descent honors
                the constraint, Decision D9).
                @see docs/superpowers/plans/2026-07-03-pin-drag-ui.md (Decision D9) */}
            <div
                style={{
                    marginBottom: 15,
                    display: 'flex',
                    gap: 10,
                    flexWrap: 'wrap',
                    alignItems: 'center',
                }}
            >
                <span style={{ opacity: descentMode === 'sobolev' ? 1 : 0.4 }}>Pins:</span>
                {pins.length === 0 ? (
                    <span style={{ color: '#888', fontStyle: 'italic' }}>
                        click a vertex in the view to pin it (drag to move)
                    </span>
                ) : (
                    pins.map((pin) => (
                        <span
                            key={pin.vertexIndex}
                            style={{
                                display: 'inline-flex',
                                alignItems: 'center',
                                gap: 6,
                                padding: '4px 8px',
                                border: '1px solid #555',
                                borderRadius: 5,
                            }}
                        >
                            <label style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                                <input
                                    type="checkbox"
                                    checked={pin.enabled}
                                    onChange={(e) =>
                                        setPinEnabled(pin.vertexIndex, e.target.checked)
                                    }
                                />
                                <span>Pin v{pin.vertexIndex}</span>
                            </label>
                            <button
                                type="button"
                                onClick={() => removePin(pin.vertexIndex)}
                                title="remove pin"
                                style={{
                                    cursor: 'pointer',
                                    background: 'transparent',
                                    color: '#ff6666',
                                    border: 'none',
                                    fontSize: 18,
                                    lineHeight: 1,
                                    padding: 0,
                                }}
                            >
                                ×
                            </button>
                        </span>
                    ))
                )}
            </div>

            {/* Soft-constraint penalties (5C) + target-length animation (plan §4
                Task 5). These enter the sobolev OBJECTIVE (energy + differential),
                never the raw L² path — so, like the constraint selects, the inputs
                disable and the block dims in raw mode. Weight 0 = off; the field X
                is a constant vector (normalized in the core). "Grow L" is a per-
                accepted-step multiplicative factor for the frozen length targets
                (1.0 = off, clamped [0.9, 1.1]).
                @see docs/superpowers/plans/2026-07-03-sobolev-penalties.md §4 Task 5 */}
            <div
                style={{
                    marginBottom: 15,
                    display: 'flex',
                    gap: 15,
                    flexWrap: 'wrap',
                    alignItems: 'center',
                    opacity: descentMode === 'sobolev' ? 1 : 0.4,
                }}
            >
                <span>Penalties:</span>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span>Length w:</span>
                    <input
                        type="number"
                        step="0.1"
                        min="0"
                        value={penalties.totalLength ?? 0}
                        disabled={descentMode !== 'sobolev'}
                        onChange={(e) => setPenaltyTotalLength(parseFloat(e.target.value) || 0)}
                        style={{ width: 70, padding: 4, fontSize: 14 }}
                    />
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span>Diff w:</span>
                    <input
                        type="number"
                        step="0.1"
                        min="0"
                        value={penalties.lengthDiff ?? 0}
                        disabled={descentMode !== 'sobolev'}
                        onChange={(e) => setPenaltyLengthDiff(parseFloat(e.target.value) || 0)}
                        style={{ width: 70, padding: 4, fontSize: 14 }}
                    />
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span>Field w:</span>
                    <input
                        type="number"
                        step="0.1"
                        min="0"
                        value={fieldWeight}
                        disabled={descentMode !== 'sobolev'}
                        onChange={(e) => setPenaltyFieldWeight(parseFloat(e.target.value) || 0)}
                        style={{ width: 70, padding: 4, fontSize: 14 }}
                    />
                </label>
                {/* Not a <label>: this wraps THREE independent axis inputs, so a
                    single label/control association would be wrong (a11y). */}
                <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span>Field X:</span>
                    {[0, 1, 2].map((axis) => (
                        <input
                            key={axis}
                            type="number"
                            step="0.1"
                            value={fieldX[axis]}
                            disabled={descentMode !== 'sobolev'}
                            onChange={(e) => {
                                const next: [number, number, number] = [
                                    fieldX[0],
                                    fieldX[1],
                                    fieldX[2],
                                ];
                                next[axis] = parseFloat(e.target.value) || 0;
                                setPenaltyFieldX(next);
                            }}
                            style={{ width: 50, padding: 4, fontSize: 14 }}
                        />
                    ))}
                </span>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span>Grow L:</span>
                    <input
                        type="range"
                        min="0.9"
                        max="1.1"
                        step="0.005"
                        value={lengthGrowthRate}
                        disabled={descentMode !== 'sobolev'}
                        onChange={(e) => setLengthGrowthRate(parseFloat(e.target.value))}
                        style={{ width: 90 }}
                    />
                    <span style={{ fontFamily: 'monospace', width: 44 }}>
                        {lengthGrowthRate.toFixed(3)}
                    </span>
                </label>
            </div>
        </>
    );
}
