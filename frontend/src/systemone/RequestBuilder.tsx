import { useMemo } from 'react';
import { MAX_READOUT_OPTIONS } from './models';
import { S1_PRESETS, type S1Preset } from './presets';
import type { S1Question, S1QuestionType, S1RequestState } from './types';

interface RequestBuilderProps {
  req: S1RequestState;
  onChange: (patch: Partial<S1RequestState>) => void;
  onRun: () => void;
  onCancel: () => void;
  running: boolean;
  canRun: boolean;
  runBlockedReason: string;
  jsonError: string | null;
  validationErrors: string[];
  activeModelLabel: string;
  serverOnline: boolean;
}

function newQuestion(n: number): S1Question {
  return {
    id: `q${n}`,
    type: 'choice',
    instructions: '',
    criteria: [
      { key: '', label: '' },
      { key: '', label: '' },
    ],
    levels: ['Level 0', 'Level 1'],
  };
}

const TYPE_LABEL: Record<S1QuestionType, string> = {
  choice: 'choice',
  noul: 'noul (yes/no)',
  score: 'score',
};

export function RequestBuilder({
  req,
  onChange,
  onRun,
  onCancel,
  running,
  canRun,
  runBlockedReason,
  jsonError,
  validationErrors,
  activeModelLabel,
  serverOnline,
}: RequestBuilderProps) {
  const prettyState = useMemo(() => {
    if (!req.stateIsJson) return null;
    try {
      return JSON.stringify(JSON.parse(req.stateText), null, 2);
    } catch {
      return null;
    }
  }, [req.stateIsJson, req.stateText]);

  const updateQuestion = (idx: number, patch: Partial<S1Question>) => {
    const questions = req.questions.map((qq, i) => (i === idx ? { ...qq, ...patch } : qq));
    onChange({ questions });
  };

  const removeQuestion = (idx: number) => {
    onChange({ questions: req.questions.filter((_, i) => i !== idx) });
  };

  const addQuestion = () => {
    let n = req.questions.length + 1;
    let id = `q${n}`;
    const ids = new Set(req.questions.map(x => x.id));
    while (ids.has(id)) {
      n += 1;
      id = `q${n}`;
    }
    onChange({ questions: [...req.questions, newQuestion(n)] });
  };

  const applyPreset = (p: S1Preset) => {
    onChange({ stateText: p.stateText, stateIsJson: p.stateIsJson, questions: p.questions.map(x => ({ ...x, criteria: x.criteria.map(c => ({ ...c })), levels: [...x.levels] })) });
  };

  return (
    <div className="flex-1 overflow-y-auto">
      {/* Presets row */}
      <div className="px-4 py-2.5 border-b border-[var(--border)] flex items-center gap-2 overflow-x-auto bg-[var(--bg-secondary)]">
        <span className="text-[10px] uppercase tracking-wider text-[var(--text-secondary)] shrink-0">Presets</span>
        {S1_PRESETS.map(p => (
          <button
            key={p.name}
            onClick={() => applyPreset(p)}
            disabled={running}
            title={p.blurb}
            className="text-[11px] px-2.5 py-1 rounded-full border border-[var(--border)] text-[var(--text-secondary)] hover:border-[var(--accent)] hover:text-[var(--text-primary)] transition-colors whitespace-nowrap disabled:opacity-40"
          >
            {p.name}
          </button>
        ))}
        <div className="ml-auto shrink-0 flex items-center gap-2">
          {running ? (
            <button
              onClick={onCancel}
              className="text-xs px-3.5 py-1.5 rounded-lg border border-[var(--danger)] text-[var(--danger)] hover:bg-[var(--danger)] hover:text-white transition-colors font-medium"
            >
              Cancel
            </button>
          ) : (
            <button
              onClick={onRun}
              disabled={!canRun}
              title={canRun ? 'POST /v1/systemone' : runBlockedReason}
              className="text-xs px-4 py-1.5 rounded-lg font-semibold transition-colors disabled:opacity-40"
              style={{ background: 'var(--accent)', color: 'white' }}
            >
              Decide ▸
            </button>
          )}
        </div>
      </div>

      <div className="p-4 space-y-4">
        {/* State */}
        <section>
          <div className="flex items-center justify-between mb-1.5">
            <label className="text-xs font-semibold uppercase tracking-wider text-[var(--text-secondary)]">State</label>
            <div className="flex rounded-lg overflow-hidden border border-[var(--border)]">
              {(['text', 'json'] as const).map(mode => (
                <button
                  key={mode}
                  onClick={() => onChange({ stateIsJson: mode === 'json' })}
                  className={`text-[10px] px-2.5 py-1 transition-colors ${
                    (mode === 'json') === req.stateIsJson
                      ? 'bg-[var(--accent)] text-white'
                      : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                  }`}
                >
                  {mode === 'json' ? 'JSON' : 'text'}
                </button>
              ))}
            </div>
          </div>
          <textarea
            value={req.stateText}
            onChange={e => onChange({ stateText: e.target.value })}
            spellCheck={false}
            className={`w-full h-32 bg-[var(--bg-secondary)] border rounded-lg px-3 py-2 text-[13px] font-mono leading-relaxed focus:outline-none focus:border-[var(--accent)] resize-y ${
              jsonError ? 'border-[var(--danger)]' : 'border-[var(--border)]'
            }`}
            placeholder={req.stateIsJson ? '{ "key": "value", … } — rendered to text by the server' : 'The state / document the questions are evaluated against…'}
          />
          <div className="flex items-center justify-between mt-1">
            <div className="text-[10px] text-[var(--danger)]">{jsonError ?? ''}</div>
            <div className="flex items-center gap-2">
              {prettyState !== null && (
                <button
                  onClick={() => onChange({ stateText: prettyState })}
                  className="text-[10px] text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                >
                  pretty-print
                </button>
              )}
              <span className="text-[10px] text-[var(--text-secondary)] font-mono">{req.stateText.length.toLocaleString()} chars</span>
            </div>
          </div>
        </section>

        {/* Questions */}
        <section>
          <div className="flex items-center justify-between mb-1.5">
            <label className="text-xs font-semibold uppercase tracking-wider text-[var(--text-secondary)]">
              Questions <span className="normal-case font-normal">({req.questions.length} · evaluated in one batched prefill)</span>
            </label>
            <button
              onClick={addQuestion}
              disabled={running}
              className="text-[11px] px-2.5 py-1 rounded-lg border border-[var(--border)] text-[var(--text-secondary)] hover:border-[var(--accent)] hover:text-[var(--text-primary)] transition-colors disabled:opacity-40"
            >
              + question
            </button>
          </div>

          <div className="space-y-3">
            {req.questions.map((qq, idx) => {
              const overCapacity =
                (qq.type === 'choice' && qq.criteria.filter(c => c.key.trim()).length > MAX_READOUT_OPTIONS) ||
                (qq.type === 'score' && qq.levels.filter(l => l.trim()).length > MAX_READOUT_OPTIONS);
              return (
                <div key={idx} className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-3">
                  <div className="flex items-center gap-2 mb-2">
                    <input
                      value={qq.id}
                      onChange={e => updateQuestion(idx, { id: e.target.value })}
                      className="w-24 bg-[var(--bg-tertiary)] border border-[var(--border)] rounded px-2 py-1 text-xs font-mono focus:outline-none focus:border-[var(--accent)]"
                      placeholder="id"
                    />
                    <select
                      value={qq.type}
                      onChange={e => updateQuestion(idx, { type: e.target.value as S1QuestionType })}
                      className="bg-[var(--bg-tertiary)] border border-[var(--border)] rounded px-2 py-1 text-xs focus:outline-none focus:border-[var(--accent)]"
                    >
                      {(Object.keys(TYPE_LABEL) as S1QuestionType[]).map(t => (
                        <option key={t} value={t}>
                          {TYPE_LABEL[t]}
                        </option>
                      ))}
                    </select>
                    <button
                      onClick={() => removeQuestion(idx)}
                      disabled={running || req.questions.length <= 1}
                      className="ml-auto text-[var(--text-secondary)] hover:text-[var(--danger)] text-sm px-1 disabled:opacity-30"
                      title="Remove question"
                    >
                      ✕
                    </button>
                  </div>

                  <textarea
                    value={qq.instructions}
                    onChange={e => updateQuestion(idx, { instructions: e.target.value })}
                    placeholder="Instructions — what should be decided about the state?"
                    rows={2}
                    className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded px-2.5 py-1.5 text-[13px] leading-snug focus:outline-none focus:border-[var(--accent)] resize-y"
                  />

                  {qq.type === 'choice' && (
                    <div className="mt-2 space-y-1.5">
                      {qq.criteria.map((c, ci) => (
                        <div key={ci} className="flex gap-1.5 items-center">
                          <input
                            value={c.key}
                            onChange={e => {
                              const criteria = qq.criteria.map((cc, i) => (i === ci ? { ...cc, key: e.target.value } : cc));
                              updateQuestion(idx, { criteria });
                            }}
                            placeholder="option"
                            className="w-32 bg-[var(--bg-tertiary)] border border-[var(--border)] rounded px-2 py-1 text-xs font-mono focus:outline-none focus:border-[var(--accent)]"
                          />
                          <input
                            value={c.label}
                            onChange={e => {
                              const criteria = qq.criteria.map((cc, i) => (i === ci ? { ...cc, label: e.target.value } : cc));
                              updateQuestion(idx, { criteria });
                            }}
                            placeholder="description"
                            className="flex-1 bg-[var(--bg-tertiary)] border border-[var(--border)] rounded px-2 py-1 text-xs focus:outline-none focus:border-[var(--accent)]"
                          />
                          <button
                            onClick={() => updateQuestion(idx, { criteria: qq.criteria.filter((_, i) => i !== ci) })}
                            disabled={qq.criteria.length <= 2}
                            className="text-[var(--text-secondary)] hover:text-[var(--danger)] text-xs px-1 disabled:opacity-30"
                          >
                            ✕
                          </button>
                        </div>
                      ))}
                      <button
                        onClick={() => updateQuestion(idx, { criteria: [...qq.criteria, { key: '', label: '' }] })}
                        className="text-[10px] text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                      >
                        + option
                      </button>
                    </div>
                  )}

                  {qq.type === 'noul' && (
                    <div className="mt-2 space-y-1.5">
                      <p className="text-[10px] text-[var(--text-secondary)] leading-snug">
                        Optional semantic criteria. Supplying both makes the local noul prompt equivalent to Jev&apos;s true/false decision form.
                      </p>
                      {['true', 'false'].map(key => {
                        const existing = qq.criteria.find(c => c.key === key);
                        return (
                          <div key={key} className="flex gap-1.5 items-center">
                            <span className="w-12 text-[10px] font-mono text-[var(--text-secondary)]">{key}</span>
                            <input
                              value={existing?.label ?? ''}
                              onChange={e => {
                                const other = qq.criteria.filter(c => c.key !== key);
                                updateQuestion(idx, { criteria: [...other, { key, label: e.target.value }] });
                              }}
                              placeholder={key === 'true' ? 'Condition holds when…' : 'Condition does not hold when…'}
                              className="flex-1 bg-[var(--bg-tertiary)] border border-[var(--border)] rounded px-2 py-1 text-xs focus:outline-none focus:border-[var(--accent)]"
                            />
                          </div>
                        );
                      })}
                    </div>
                  )}

                  {qq.type === 'score' && (
                    <div className="mt-2 space-y-1.5">
                      {qq.levels.map((lv, li) => (
                        <div key={li} className="flex gap-1.5 items-center">
                          <span className="text-[10px] font-mono text-[var(--text-secondary)] w-4 text-right">{li}</span>
                          <input
                            value={lv}
                            onChange={e => {
                              const levels = qq.levels.map((ll, i) => (i === li ? e.target.value : ll));
                              updateQuestion(idx, { levels });
                            }}
                            placeholder={`level ${li} description`}
                            className="flex-1 bg-[var(--bg-tertiary)] border border-[var(--border)] rounded px-2 py-1 text-xs focus:outline-none focus:border-[var(--accent)]"
                          />
                          <div className="flex flex-col">
                            <button
                              onClick={() => {
                                if (li === 0) return;
                                const levels = [...qq.levels];
                                [levels[li - 1], levels[li]] = [levels[li], levels[li - 1]];
                                updateQuestion(idx, { levels });
                              }}
                              disabled={li === 0}
                              className="text-[8px] leading-none text-[var(--text-secondary)] hover:text-[var(--text-primary)] disabled:opacity-30"
                            >
                              ▲
                            </button>
                            <button
                              onClick={() => {
                                if (li === qq.levels.length - 1) return;
                                const levels = [...qq.levels];
                                [levels[li + 1], levels[li]] = [levels[li], levels[li + 1]];
                                updateQuestion(idx, { levels });
                              }}
                              disabled={li === qq.levels.length - 1}
                              className="text-[8px] leading-none text-[var(--text-secondary)] hover:text-[var(--text-primary)] disabled:opacity-30"
                            >
                              ▼
                            </button>
                          </div>
                          <button
                            onClick={() => updateQuestion(idx, { levels: qq.levels.filter((_, i) => i !== li) })}
                            disabled={qq.levels.length <= 2}
                            className="text-[var(--text-secondary)] hover:text-[var(--danger)] text-xs px-1 disabled:opacity-30"
                          >
                            ✕
                          </button>
                        </div>
                      ))}
                      <button
                        onClick={() => updateQuestion(idx, { levels: [...qq.levels, `Level ${qq.levels.length}`] })}
                        disabled={qq.levels.length >= 10}
                        className="text-[10px] text-[var(--text-secondary)] hover:text-[var(--text-primary)] disabled:opacity-40"
                      >
                        + level
                      </button>
                    </div>
                  )}

                  {overCapacity && (
                    <div className="mt-2 text-[10px] text-[var(--danger)]">
                      ⚠ More than {MAX_READOUT_OPTIONS} options — the readout head degrades this question to uniform
                      probabilities (abstained). Trim to {MAX_READOUT_OPTIONS} or fewer.
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </section>

        {/* Run options */}
        <section className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-3 space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-xs font-semibold uppercase tracking-wider text-[var(--text-secondary)]">min_confidence</label>
            <span className="text-xs font-mono text-[var(--text-primary)]">{req.minConfidence === 0 ? 'off' : req.minConfidence.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min={0}
            max={0.9}
            step={0.05}
            value={req.minConfidence}
            onChange={e => onChange({ minConfidence: parseFloat(e.target.value) })}
            className="w-full accent-[var(--accent)]"
          />
          <p className="text-[10px] text-[var(--text-secondary)] leading-snug">
            Jev confidence measures how concentrated the distribution is — clamp((n·maxp−1)/(n−1)) — not the top
            probability. Abstains answers below this threshold.
          </p>
        </section>

        {validationErrors.length > 0 && (
          <div className="rounded-lg border border-[var(--danger)]/50 bg-[var(--danger)]/5 p-3">
            <div className="text-[10px] uppercase tracking-wider text-[var(--danger)] mb-1">Fix before running</div>
            <ul className="list-disc list-inside text-[11px] text-[var(--text-primary)] space-y-0.5">
              {validationErrors.map((v, i) => (
                <li key={i}>{v}</li>
              ))}
            </ul>
          </div>
        )}

        {!serverOnline && (
          <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-tertiary)] p-3 text-[11px] text-[var(--text-secondary)] leading-relaxed">
            <span className="text-[var(--danger)]">● server offline.</span> Start the GPU decision server with the
            playground frontend:
            <pre className="mt-1.5 p-2 bg-[var(--bg-primary)] rounded overflow-x-auto text-[10px] leading-relaxed">{`$env:DLGO_SSM_BATCH = '1'
.\\bench\\systemone\\server-9b.exe \\\\
  -host 127.0.0.1 -port 8080 -gpu -ctx 8192 \\\\
  -frontend frontend\\dist`}</pre>
            Or for HMR during development: <code className="text-[var(--text-primary)]">npm run dev</code> in{' '}
            <code className="text-[var(--text-primary)]">frontend/</code> (proxies to :8080).
          </div>
        )}

        <div className="text-[10px] text-[var(--text-secondary)] pb-2">
          Decisions run against <span className="text-[var(--text-primary)] font-mono">{activeModelLabel}</span> —
          switch backbones in the left panel (one resident model at a time).
        </div>
      </div>
    </div>
  );
}
