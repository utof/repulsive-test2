import { testConfigs } from '../core/testConfigs';
import { type Mode, useSimStore } from '../store';

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
    const stepSize = useSimStore((s) => s.stepSize);
    const setPreset = useSimStore((s) => s.setPreset);
    const setParam = useSimStore((s) => s.setParam);
    const regenerate = useSimStore((s) => s.regenerate);
    const reset = useSimStore((s) => s.reset);
    const setMode = useSimStore((s) => s.setMode);
    const setStepSize = useSimStore((s) => s.setStepSize);
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
            </div>
        </>
    );
}
