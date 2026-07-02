import { testConfigs } from '../core/testConfigs';
import { type DescentMode, type Mode, useSimStore } from '../store';

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
    const lengthConstraint = useSimStore((s) => s.lengthConstraint);
    const setBarycenterConstraint = useSimStore((s) => s.setBarycenterConstraint);
    const setLengthConstraint = useSimStore((s) => s.setLengthConstraint);
    const showArrows = useSimStore((s) => s.showArrows);
    const setShowArrows = useSimStore((s) => s.setShowArrows);
    const setRunning = useSimStore((s) => s.setRunning);

    const selectedTest = testConfigs.find((t) => t.id === selectedTestId) ?? testConfigs[0];

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
                        checked={lengthConstraint}
                        disabled={descentMode !== 'sobolev'}
                        onChange={(e) => setLengthConstraint(e.target.checked)}
                    />
                    <span>Fix length</span>
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
        </>
    );
}
