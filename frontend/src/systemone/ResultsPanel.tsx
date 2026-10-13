import { useMemo, useState } from 'react';
import type { S1Answer, S1Question, S1RunResponse, TimingInfo } from './types';

// ---------------------------------------------------------------------------
// Shared bits

function ProbBar({ value, max, color }: { value: number; max: number; color: string }) {
  const pct = max > 0 ? Math.max(1.5, (value / max) * 100) : 0;
  return (
    <div className="flex-1 h-2 rounded-full bg-[var(--bg-tertiary)] overflow-hidden">
      <div className="h-full rounded-full transition-all duration-300" style={{ width: `${pct}%`, background: color }} />
    </div>
  );
}

function Chip({ tone, children, title }: { tone: 'ok' | 'warn' | 'bad' | 'dim' | 'accent'; children: React.ReactNode; title?: string }) {
  const map: Record<string, string> = {
    ok: 'bg-[var(--success)]/15 text-[var(--success)]',
    warn: 'bg-[var(--accent)]/15 text-[var(--accent-hover)]',
    bad: 'bg-[var(--danger)]/15 text-[var(--danger)]',
    dim: 'bg-[var(--text-secondary)]/10 text-[var(--text-secondary)]',
    accent: 'bg-[var(--accent)]/15 text-[var(--accent-hover)]',
  };
  return (
    <span title={title} className={`px-1.5 py-0.5 rounded text-[9px] font-bold uppercase tracking-wide ${map[tone]}`}>
      {children}
    </span>
  );
}

function fmtMs(ms: number | null): string {
  if (ms === null) return '—';
  if (ms >= 10000) return `${(ms / 1000).toFixed(1)} s`;
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
  return `${Math.round(ms)} ms`;
}

// ---------------------------------------------------------------------------
// Answer card

interface AnswerCardProps {
  answer: S1Answer;
  question?: S1Question;
}

function AnswerCard({ answer, question }: AnswerCardProps) {
  const labelOf = useCallback2(answer, question);
  const probs = useMemo(() => {
    const entries = Object.entries(answer.probabilities ?? {});
    entries.sort((a, b) => b[1] - a[1]);
    return entries;
  }, [answer.probabilities]);
  const maxP = probs.length > 0 ? probs[0][1] : 0;
  const colors = ['#6366f1', '#818cf8', '#34d399', '#f59e0b', '#f87171', '#a78bfa'];

  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-3">
      <div className="flex items-center gap-1.5 mb-2 flex-wrap">
        <span className="text-xs font-semibold font-mono text-[var(--text-primary)]">{answer.id}</span>
        <Chip tone="dim">{answer.type}</Chip>
        {answer.degraded && (
          <Chip tone="bad" title="Exceeded the readout's 5-option capacity — uniform probabilities, no signal">
            degraded
          </Chip>
        )}
        {answer.abstained && !answer.degraded && <Chip tone="warn">abstained</Chip>}
        {answer.type !== 'noul' && answer.confidence !== undefined && !answer.degraded && (
          <span
            className="ml-auto text-[10px] font-mono text-[var(--text-secondary)]"
            title="Jev confidence = clamp((n·maxp−1)/(n−1)) — distribution concentration, not top probability"
          >
            conf {answer.confidence.toFixed(3)}
          </span>
        )}
      </div>

      {answer.type === 'choice' && (
        <>
          <div className="mb-2 text-[13px]">
            <span className="text-[var(--text-secondary)]">→ </span>
            <span className="font-semibold" style={{ color: 'var(--accent-hover)' }}>
              {answer.choice ?? '—'}
            </span>
            {answer.choice && labelOf(answer.choice) && (
              <span className="text-[var(--text-secondary)]"> — {labelOf(answer.choice)}</span>
            )}
          </div>
          <div className="space-y-1.5">
            {probs.map(([key, p], i) => (
              <div key={key} className="flex items-center gap-2">
                <span className={`w-28 shrink-0 truncate text-[11px] font-mono ${key === answer.choice ? 'text-[var(--text-primary)] font-semibold' : 'text-[var(--text-secondary)]'}`} title={key}>
                  {key}
                </span>
                <ProbBar value={p} max={maxP} color={key === answer.choice ? colors[0] : colors[Math.min(i, colors.length - 1)]} />
                <span className="w-12 text-right text-[10px] font-mono text-[var(--text-secondary)]">{(p * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </>
      )}

      {answer.type === 'noul' && (
        <div className="flex items-center gap-3">
          <div
            className="text-2xl font-bold font-mono tabular-nums"
            style={{ color: (answer.noul ?? 0) >= 0.5 ? 'var(--success)' : 'var(--danger)' }}
            title="Probability of yes"
          >
            {((answer.noul ?? 0) * 100).toFixed(1)}%
          </div>
          <div className="flex-1">
            <div className="flex h-3 rounded-full overflow-hidden bg-[var(--bg-tertiary)]">
              <div className="h-full transition-all duration-300" style={{ width: `${(answer.noul ?? 0) * 100}%`, background: 'var(--success)' }} />
              <div className="h-full flex-1" style={{ background: 'var(--danger)' }} />
            </div>
            <div className="flex justify-between mt-1 text-[10px] font-mono text-[var(--text-secondary)]">
              <span>yes {((answer.noul ?? 0) * 100).toFixed(1)}%</span>
              <span>no {((1 - (answer.noul ?? 0)) * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
      )}

      {answer.type === 'score' && (
        <>
          <div className="flex items-baseline gap-2 mb-2">
            <span className="text-2xl font-bold font-mono tabular-nums" style={{ color: 'var(--accent-hover)' }}>
              {(answer.score ?? 0).toFixed(2)}
            </span>
            <span className="text-[10px] text-[var(--text-secondary)]">probability-weighted mean level</span>
          </div>
          <div className="space-y-1.5">
            {probs.map(([key, p], i) => {
              const idx = parseInt(key, 10);
              const levelText = question?.levels[idx] ?? answer.legend?.[key] ?? `level ${key}`;
              return (
                <div key={key} className="flex items-center gap-2">
                  <span className="w-4 shrink-0 text-[10px] font-mono text-[var(--text-secondary)] text-right">{key}</span>
                  <span className="flex-1 min-w-0 truncate text-[11px]" title={levelText}>
                    {levelText}
                  </span>
                  <ProbBar value={p} max={maxP} color={colors[Math.min(i, colors.length - 1)]} />
                  <span className="w-12 text-right text-[10px] font-mono text-[var(--text-secondary)]">{(p * 100).toFixed(1)}%</span>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

// small helper kept outside the component to avoid re-creation churn
function useCallback2(answer: S1Answer, question?: S1Question) {
  return (key: string) => {
    const c = question?.criteria.find(x => x.key === key);
    if (c && c.label) return c.label;
    return answer.legend?.[key] ?? '';
  };
}

// ---------------------------------------------------------------------------
// Timing strip

function TimingStrip({ timing }: { timing: TimingInfo }) {
  const total = timing.wallMs;
  const q = timing.queueMs ?? 0;
  const d = timing.decisionMs ?? 0;
  const other = Math.max(0, total - q - d);
  const seg = (v: number) => `${Math.min(100, (v / Math.max(total, 0.001)) * 100)}%`;
  return (
    <div>
      <div className="flex h-2 rounded-full overflow-hidden bg-[var(--bg-tertiary)]">
        <div style={{ width: seg(q), background: 'var(--accent)' }} title={`queue ${fmtMs(timing.queueMs)}`} />
        <div style={{ width: seg(d), background: 'var(--success)' }} title={`decision ${fmtMs(timing.decisionMs)}`} />
        <div style={{ width: seg(other), background: 'var(--border)' }} title="HTTP overhead" />
      </div>
      <div className="flex gap-3 mt-1.5 text-[10px] font-mono text-[var(--text-secondary)]">
        <span>
          <span className="inline-block w-2 h-2 rounded-sm align-middle mr-1" style={{ background: 'var(--accent)' }} />
          queue {fmtMs(timing.queueMs)}
        </span>
        <span>
          <span className="inline-block w-2 h-2 rounded-sm align-middle mr-1" style={{ background: 'var(--success)' }} />
          decision {fmtMs(timing.decisionMs)}
        </span>
        <span className="ml-auto text-[var(--text-primary)] font-semibold">total {fmtMs(total)}</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Panel

interface ResultsPanelProps {
  running: boolean;
  runSeconds: number;
  error: string | null;
  response: S1RunResponse | null;
  timing: TimingInfo | null;
  presetName: string | null;
  cold: boolean;
  questions: S1Question[];
  wireBody: unknown;
}

export function ResultsPanel({ running, runSeconds, error, response, timing, presetName, cold, questions, wireBody }: ResultsPanelProps) {
  const [tab, setTab] = useState<'answers' | 'request' | 'response'>('answers');

  const qById = useMemo(() => {
    const m = new Map<string, S1Question>();
    for (const q of questions) m.set(q.id, q);
    return m;
  }, [questions]);

  const answers = useMemo(() => {
    if (!response?.answers) return [];
    return Object.keys(response.answers)
      .sort()
      .map(id => {
        const a = response.answers[id] ?? {};
        return {
          id,
          type: (a.type ?? 'choice') as 'choice' | 'noul' | 'score',
          choice: a.choice as string | undefined,
          score: a.score as number | undefined,
          noul: a.noul as number | undefined,
          probabilities: a.probabilities as Record<string, number> | undefined,
          confidence: a.confidence as number | undefined,
          legend: a.legend as Record<string, string> | undefined,
          abstained: !!a.abstained,
          degraded: !!a.degraded,
        };
      });
  }, [response]);

  const engine = response?.engine ?? (Array.isArray(response?.results) ? response.results.find(r => r.engine)?.engine : null) ?? null;
  const inputTokens = response?.usage?.input_tokens ?? response?.prompt_tokens ?? null;

  const curl = useMemo(() => {
    const body = JSON.stringify(wireBody);
    return `curl -s http://localhost:8080/v1/systemone -H "Content-Type: application/json" -d '${body.replace(/'/g, "'\\''")}'`;
  }, [wireBody]);

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-[var(--border)] flex items-center gap-1">
        {(['answers', 'request', 'response'] as const).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`text-[11px] px-2.5 py-1 rounded-lg transition-colors ${
              tab === t ? 'bg-[var(--accent)]/15 text-[var(--accent-hover)] font-semibold' : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
            }`}
          >
            {t}
          </button>
        ))}
        {response && engine && (
          <span className="ml-auto flex items-center gap-1.5">
            <Chip tone={engine === 'gpu' ? 'ok' : 'dim'} title="Which engine served the decisions">{engine}</Chip>
            {cold && <Chip tone="warn" title="First decision after load — includes prefix-cache warm-up">cold</Chip>}
          </span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {running && (
          <div className="rounded-xl border border-[var(--accent)]/40 bg-[var(--accent)]/5 p-4 text-center">
            <div className="text-sm text-[var(--accent-hover)] animate-pulse">deciding… {runSeconds > 0 && `${runSeconds}s`}</div>
            <div className="text-[10px] text-[var(--text-secondary)] mt-1">prefix prefill → batched question evaluation</div>
          </div>
        )}

        {!running && error && (
          <div className="rounded-xl border border-[var(--danger)]/50 bg-[var(--danger)]/5 p-3">
            <div className="text-[10px] uppercase tracking-wider text-[var(--danger)] mb-1">Request failed</div>
            <div className="text-xs text-[var(--text-primary)] leading-relaxed break-words">{error}</div>
          </div>
        )}

        {!running && !error && !response && (
          <div className="text-center py-16 text-[var(--text-secondary)]">
            <div className="text-3xl mb-2 opacity-40">⚖</div>
            <div className="text-xs">Run a decision to see calibrated answers.</div>
            <div className="text-[10px] mt-1 opacity-70">Load a backbone on the left, pick a preset, hit Decide.</div>
          </div>
        )}

        {tab === 'answers' && response && !running && (
          <>
            {timing && (
              <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] uppercase tracking-wider text-[var(--text-secondary)]">Latency</span>
                  {presetName && <span className="text-[10px] text-[var(--text-secondary)]">{presetName}</span>}
                  {inputTokens !== null && (
                    <span className="text-[10px] font-mono text-[var(--text-secondary)]" title="Longest question prompt (state ingested once)">
                      {inputTokens.toLocaleString()} tok
                    </span>
                  )}
                </div>
                <TimingStrip timing={timing} />
              </div>
            )}
            {answers.map(a => (
              <AnswerCard key={a.id} answer={a} question={qById.get(a.id)} />
            ))}
          </>
        )}

        {tab === 'request' && (
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[10px] uppercase tracking-wider text-[var(--text-secondary)]">POST /v1/systemone</span>
              <button
                onClick={() => navigator.clipboard?.writeText(curl)}
                className="text-[10px] px-2 py-0.5 rounded border border-[var(--border)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-[var(--accent)]"
              >
                copy curl
              </button>
            </div>
            <pre className="p-2.5 bg-[var(--bg-primary)] border border-[var(--border)] rounded-lg text-[10px] leading-relaxed overflow-x-auto whitespace-pre-wrap break-all font-mono">
              {JSON.stringify(wireBody, null, 2)}
            </pre>
          </div>
        )}

        {tab === 'response' && response && (
          <pre className="p-2.5 bg-[var(--bg-primary)] border border-[var(--border)] rounded-lg text-[10px] leading-relaxed overflow-x-auto whitespace-pre-wrap break-all font-mono">
            {JSON.stringify(response, null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
}
