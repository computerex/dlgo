interface SettingsPanelProps {
  temperature: number;
  topP: number;
  topK: number;
  maxTokens: number;
  systemPrompt: string;
  reasoningEffort: 'low' | 'medium' | 'high';
  useGPU: boolean;
  onTemperatureChange: (v: number) => void;
  onTopPChange: (v: number) => void;
  onTopKChange: (v: number) => void;
  onMaxTokensChange: (v: number) => void;
  onSystemPromptChange: (v: string) => void;
  onReasoningEffortChange: (v: 'low' | 'medium' | 'high') => void;
  onGPUChange: (v: boolean) => void;
}

export function SettingsPanel({
  temperature, topP, topK, maxTokens, systemPrompt, reasoningEffort, useGPU,
  onTemperatureChange, onTopPChange, onTopKChange, onMaxTokensChange, onSystemPromptChange, onReasoningEffortChange, onGPUChange,
}: SettingsPanelProps) {
  return (
    <div className="space-y-5">
      <div className="text-xs font-medium uppercase tracking-wide text-[var(--text-secondary)]">Settings</div>

      {/* Backend Toggle */}
      <div className="space-y-1.5">
        <div className="text-xs text-[var(--text-secondary)]">Backend</div>
        <div className="flex rounded-lg overflow-hidden border border-[var(--border)]">
          <button
            onClick={() => onGPUChange(false)}
            className={`flex-1 px-3 py-2 text-xs font-medium transition-colors ${
              !useGPU
                ? 'bg-[var(--accent)] text-white'
                : 'bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
            }`}
          >
            CPU
          </button>
          <button
            onClick={() => onGPUChange(true)}
            className={`flex-1 px-3 py-2 text-xs font-medium transition-colors ${
              useGPU
                ? 'bg-[var(--accent)] text-white'
                : 'bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
            }`}
          >
            GPU (Vulkan)
          </button>
        </div>
        <p className="text-[10px] text-[var(--text-secondary)]">
          {useGPU
            ? 'Using Vulkan GPU acceleration'
            : 'Using CPU inference'}
        </p>
      </div>

      {/* Reasoning Effort */}
      <div className="space-y-1.5">
        <div className="text-xs text-[var(--text-secondary)]">Reasoning Effort</div>
        <div className="flex rounded-lg overflow-hidden border border-[var(--border)]">
          {(['low', 'medium', 'high'] as const).map(level => (
            <button
              key={level}
              onClick={() => onReasoningEffortChange(level)}
              className={`flex-1 px-3 py-2 text-xs font-medium transition-colors capitalize ${
                reasoningEffort === level
                  ? 'bg-[var(--accent)] text-white'
                  : 'bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
            >
              {level}
            </button>
          ))}
        </div>
        <p className="text-[10px] text-[var(--text-secondary)]">
          Controls how much the model reasons before responding
        </p>
      </div>

      <SliderSetting
        label="Temperature"
        value={temperature}
        min={0}
        max={2}
        step={0.05}
        onChange={onTemperatureChange}
      />

      <SliderSetting
        label="Top P"
        value={topP}
        min={0}
        max={1}
        step={0.05}
        onChange={onTopPChange}
      />

      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <label className="text-xs text-[var(--text-secondary)]">Top K</label>
          <span className="text-xs font-mono text-[var(--text-primary)]">{topK}</span>
        </div>
        <input
          type="number"
          value={topK}
          onChange={e => onTopKChange(parseInt(e.target.value) || 0)}
          min={0}
          max={200}
          className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded px-3 py-1.5 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent)]"
        />
      </div>

      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <label className="text-xs text-[var(--text-secondary)]">Max Tokens</label>
          <span className="text-xs font-mono text-[var(--text-primary)]">{maxTokens}</span>
        </div>
        <input
          type="number"
          value={maxTokens}
          onChange={e => onMaxTokensChange(parseInt(e.target.value) || 256)}
          min={1}
          max={8192}
          className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded px-3 py-1.5 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent)]"
        />
      </div>

      <div className="space-y-1.5">
        <label className="text-xs text-[var(--text-secondary)]">System Prompt</label>
        <textarea
          value={systemPrompt}
          onChange={e => onSystemPromptChange(e.target.value)}
          placeholder="You are a helpful assistant."
          rows={3}
          className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded px-3 py-2 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-secondary)] focus:outline-none focus:border-[var(--accent)] resize-none"
        />
      </div>
    </div>
  );
}

function SliderSetting({
  label, value, min, max, step, onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <label className="text-xs text-[var(--text-secondary)]">{label}</label>
        <span className="text-xs font-mono text-[var(--text-primary)]">{value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={e => onChange(parseFloat(e.target.value))}
        className="w-full accent-[var(--accent)] h-1.5"
      />
    </div>
  );
}
