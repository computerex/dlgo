import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ModelPanel } from './ModelPanel';
import { RequestBuilder } from './RequestBuilder';
import { ResultsPanel } from './ResultsPanel';
import { HistoryPanel } from './HistoryPanel';
import { S1_MODELS } from './models';
import { S1_PRESETS } from './presets';
import { collectEngine, decide, parseAnswers } from './api';
import type { S1Answer, S1RequestState, S1Run, S1RunResponse, TimingInfo } from './types';

const HISTORY_KEY = 'dlgo-s1-history';
const REQUEST_KEY = 'dlgo-s1-request';

function initialRequest(): S1RequestState {
  const p = S1_PRESETS[0];
  return {
    stateText: p.stateText,
    stateIsJson: p.stateIsJson,
    questions: p.questions.map(x => ({ ...x, criteria: x.criteria.map(c => ({ ...c })), levels: [...x.levels] })),
    minConfidence: 0,
    readoutPath: S1_MODELS[0].readoutPath,
  };
}

function loadStoredRequest(): S1RequestState | null {
  try {
    const raw = localStorage.getItem(REQUEST_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as S1RequestState;
    if (!parsed || typeof parsed.stateText !== 'string' || !Array.isArray(parsed.questions)) return null;
    // The persisted pre-semantic-noul Pokemon preset had no true/false labels.
    // Discard only that stale shape so first load receives the corrected preset.
    const charizard = parsed.questions.find(q => q.id === 'charizard_water_weak' && q.type === 'noul');
    if (charizard && !charizard.criteria.some(c => c.key === 'true' || c.key === 'false')) return null;
    return parsed;
  } catch {
    return null;
  }
}

function loadStoredHistory(): S1Run[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

/** Build the exact wire body from editor state. Returns null when invalid JSON. */
function buildWire(req: S1RequestState, modelId: string): { body: Record<string, unknown> | null; jsonError: string | null } {
  let state: unknown = req.stateText;
  let jsonError: string | null = null;
  if (req.stateIsJson) {
    try {
      state = JSON.parse(req.stateText);
    } catch (e) {
      jsonError = `Invalid JSON state: ${(e as Error).message}`;
      return { body: null, jsonError };
    }
  }
  const questions: Record<string, unknown> = {};
  for (const q of req.questions) {
    if (q.type === 'choice') {
      const criteria: Record<string, string> = {};
      for (const c of q.criteria) {
        if (c.key.trim()) criteria[c.key.trim()] = c.label;
      }
      questions[q.id] = { type: 'choice', instructions: q.instructions, criteria };
    } else if (q.type === 'score') {
      questions[q.id] = { type: 'score', instructions: q.instructions, criteria: q.levels };
    } else {
      const criteria: Record<string, string> = {};
      for (const c of q.criteria) {
        if ((c.key === 'true' || c.key === 'false') && c.label.trim()) criteria[c.key] = c.label;
      }
      questions[q.id] = {
        type: 'noul',
        instructions: q.instructions,
        ...(Object.keys(criteria).length > 0 ? { criteria } : {}),
      };
    }
  }
  const body: Record<string, unknown> = {
    model: modelId,
    readout: req.readoutPath,
    state,
    questions,
  };
  if (req.minConfidence > 0) body.min_confidence = req.minConfidence;
  return { body, jsonError };
}

function validate(req: S1RequestState): string[] {
  const errs: string[] = [];
  if (!req.stateText.trim()) errs.push('State is empty.');
  if (req.questions.length === 0) errs.push('At least one question is required.');
  const ids = new Set<string>();
  req.questions.forEach((q, i) => {
    const tag = `Question ${q.id || `#${i + 1}`}`;
    if (!q.id.trim()) errs.push(`${tag}: id is empty.`);
    else if (ids.has(q.id)) errs.push(`${tag}: duplicate id.`);
    ids.add(q.id);
    if (!q.instructions.trim()) errs.push(`${tag}: instructions are empty.`);
    if (q.type === 'choice') {
      const keys = q.criteria.map(c => c.key.trim()).filter(Boolean);
      if (keys.length < 2) errs.push(`${tag}: choice needs at least 2 options.`);
      if (new Set(keys).size !== keys.length) errs.push(`${tag}: duplicate option keys.`);
    }
    if (q.type === 'score' && q.levels.map(l => l.trim()).filter(Boolean).length < 2) {
      errs.push(`${tag}: score needs at least 2 levels.`);
    }
  });
  return errs;
}

interface PlaygroundProps {
  onBackToChat: () => void;
}

export function SystemOnePlayground({ onBackToChat }: PlaygroundProps) {
  const [req, setReq] = useState<S1RequestState>(() => loadStoredRequest() ?? initialRequest());
  const [activeId, setActiveId] = useState('');
  const [refreshSignal, setRefreshSignal] = useState(0);
  void setRefreshSignal; // wired to load-completion events when the panel exposes them
  const [modelError, setModelError] = useState('');
  const [running, setRunning] = useState(false);
  const [runSeconds, setRunSeconds] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<S1RunResponse | null>(null);
  const [timing, setTiming] = useState<TimingInfo | null>(null);
  const [presetName, setPresetName] = useState<string | null>(null);
  const [cold, setCold] = useState(true);
  const [history, setHistory] = useState<S1Run[]>(() => loadStoredHistory());
  const abortRef = useRef<AbortController | null>(null);
  const timerRef = useRef<number | null>(null);

  // Persist editor + history.
  useEffect(() => {
    localStorage.setItem(REQUEST_KEY, JSON.stringify(req));
  }, [req]);
  useEffect(() => {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 30)));
  }, [history]);

  // Elapsed seconds ticker while running.
  useEffect(() => {
    if (!running) {
      setRunSeconds(0);
      if (timerRef.current) window.clearInterval(timerRef.current);
      return;
    }
    timerRef.current = window.setInterval(() => setRunSeconds(s => s + 1), 1000);
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current);
    };
  }, [running]);

  // Keep the readout path in sync with the active model unless the user
  // manually overrode it.
  const activeSpec = useMemo(() => S1_MODELS.find(m => m.id === activeId) ?? null, [activeId]);
  const readoutIsFromSpec = useMemo(
    () => S1_MODELS.some(m => m.readoutPath === req.readoutPath),
    [req.readoutPath],
  );
  useEffect(() => {
    if (activeSpec && readoutIsFromSpec && req.readoutPath !== activeSpec.readoutPath) {
      setReq(r => ({ ...r, readoutPath: activeSpec.readoutPath }));
    }
  }, [activeSpec, readoutIsFromSpec, req.readoutPath]);

  const jsonError = useMemo(() => {
    if (!req.stateIsJson) return null;
    try {
      JSON.parse(req.stateText);
      return null;
    } catch (e) {
      return (e as Error).message;
    }
  }, [req.stateIsJson, req.stateText]);

  const validationErrors = useMemo(() => validate(req), [req]);

  const wire = useMemo(() => (activeId ? buildWire(req, activeId) : { body: null, jsonError: null }), [req, activeId]);

  const canRun = !!activeId && !!wire.body && validationErrors.length === 0 && !running;
  const runBlockedReason = !activeId
    ? 'Load a backbone first'
    : validationErrors.length > 0
      ? validationErrors[0]
      : jsonError
        ? 'Invalid JSON state'
        : running
          ? 'A decision is already running'
          : '';

  const onRun = useCallback(async () => {
    if (!wire.body || !activeId) return;
    const controller = new AbortController();
    abortRef.current = controller;
    setRunning(true);
    setError(null);
    setResponse(null);
    setTiming(null);
    try {
      const { data, timing: t } = await decide(wire.body, controller.signal);
      setResponse(data);
      setTiming(t);
      const answers: S1Answer[] = parseAnswers(data.answers);
      const spec = S1_MODELS.find(m => m.id === activeId);
      const run: S1Run = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        ts: Date.now(),
        modelId: activeId,
        label: presetName ?? 'custom',
        totalMs: t.wallMs,
        timing: t,
        engine: collectEngine(data),
        inputTokens: data.usage?.input_tokens ?? null,
        answers,
        request: { ...req, questions: req.questions.map(q => ({ ...q, criteria: q.criteria.map(c => ({ ...c })), levels: [...q.levels] })) },
      };
      setHistory(h => [run, ...h].slice(0, 30));
      setCold(false);
      void spec;
    } catch (e: unknown) {
      if ((e as Error)?.name === 'AbortError') {
        setError('Canceled.');
      } else {
        setError((e as Error).message || String(e));
      }
    } finally {
      setRunning(false);
      abortRef.current = null;
    }
  }, [wire.body, activeId, presetName, req]);

  const onCancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const restoreRun = useCallback((run: S1Run) => {
    setReq({ ...run.request, questions: run.request.questions.map(q => ({ ...q, criteria: q.criteria.map(c => ({ ...c })), levels: [...q.levels] })) });
    setPresetName(null);
  }, []);

  const activeSpecForLabel = activeSpec ?? null;

  return (
    <div className="h-screen flex bg-[var(--bg-primary)] text-[var(--text-primary)]">
      {/* Left: backbones */}
      <aside className="w-[290px] shrink-0 border-r border-[var(--border)] bg-[var(--bg-secondary)] flex flex-col">
        <div className="px-4 py-3 border-b border-[var(--border)] flex items-center gap-2">
          <h1 className="text-sm font-bold tracking-tight">
            <span style={{ color: 'var(--accent)' }}>dlgo</span>
            <span className="text-[var(--text-secondary)] font-normal ml-1.5 text-xs">system one</span>
          </h1>
        </div>
        <div className="flex-1 min-h-0">
          <ModelPanel activeId={activeId} onSelectActive={setActiveId} refreshSignal={refreshSignal} onLoadError={setModelError} />
        </div>
        <div className="border-t border-[var(--border)] px-3 py-2 flex items-center justify-between">
          <button onClick={onBackToChat} className="text-[11px] text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors">
            ← chat playground
          </button>
          <a
            href="https://github.com/computerex/dlgo"
            onClick={e => e.preventDefault()}
            className="text-[10px] text-[var(--text-secondary)] opacity-60"
            title="POST /v1/systemone — Jev wire-compatible"
          >
            Jev-compatible API
          </a>
        </div>
      </aside>

      {/* Center: request builder */}
      <main className="flex-1 flex flex-col min-w-0">
        <div className="px-4 py-2.5 border-b border-[var(--border)] bg-[var(--bg-secondary)] flex items-center gap-3">
          <span className="text-sm font-semibold">Decision playground</span>
          {modelError ? (
            <span className="text-[11px] text-[var(--danger)] truncate">{modelError}</span>
          ) : activeSpecForLabel ? (
            <span className="text-[11px] text-[var(--text-secondary)] truncate">
              {activeSpecForLabel.label} · head {activeSpecForLabel.readoutName} · device-any
            </span>
          ) : (
            <span className="text-[11px] text-[var(--text-secondary)]">no backbone selected</span>
          )}
        </div>
        <RequestBuilder
          req={req}
          onChange={patch => setReq(r => ({ ...r, ...patch }))}
          onRun={onRun}
          onCancel={onCancel}
          running={running}
          canRun={canRun}
          runBlockedReason={runBlockedReason}
          jsonError={jsonError}
          validationErrors={validationErrors}
          activeModelLabel={activeSpecForLabel ? `${activeSpecForLabel.label} (${activeSpecForLabel.id})` : 'no model loaded'}
          serverOnline={!!activeId || modelError === ''}
        />
      </main>

      {/* Right: results + history */}
      <aside className="w-[400px] shrink-0 border-l border-[var(--border)] bg-[var(--bg-secondary)] flex flex-col">
        <div className="flex-1 min-h-0">
          <ResultsPanel
            running={running}
            runSeconds={runSeconds}
            error={error}
            response={response}
            timing={timing}
            presetName={presetName}
            cold={cold}
            questions={req.questions}
            wireBody={wire.body ?? {}}
          />
        </div>
        <HistoryPanel runs={history} onRestore={restoreRun} onClear={() => setHistory([])} />
      </aside>
    </div>
  );
}
