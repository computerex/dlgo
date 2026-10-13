import { useCallback, useEffect, useState } from 'react';
import { S1_MODELS, type S1ModelSpec } from './models';
import { fetchHealth, fetchModelList, type HealthInfo, type ModelListResponse } from './api';
import type { LoadedModelInfo } from './types';

interface ModelPanelProps {
  activeId: string;
  onSelectActive: (id: string) => void;
  /** Bumped by the parent when a load/unload finishes, to trigger a refresh. */
  refreshSignal: number;
  onLoadError: (msg: string) => void;
}

function fmtMB(v: number): string {
  if (v >= 1024) return `${(v / 1024).toFixed(1)} GB`;
  return `${Math.round(v)} MB`;
}

export function ModelPanel({ activeId, onSelectActive, refreshSignal, onLoadError }: ModelPanelProps) {
  const [loaded, setLoaded] = useState<LoadedModelInfo[]>([]);
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [busy, setBusy] = useState<string | null>(null); // spec id being loaded/unloaded
  const [stage, setStage] = useState('');

  const refresh = useCallback(async () => {
    try {
      const [list, h] = await Promise.all([fetchModelList(), fetchHealth()]);
      setLoaded(list.data as LoadedModelInfo[]);
      setHealth(h);
      // Auto-select the first loaded playground backbone (e.g. after page load
      // against a server that preloaded one via -model).
      const loadedIds = new Set(list.data.map(m => m.id));
      const currentOk = activeId !== '' && loadedIds.has(activeId);
      if (!currentOk) {
        const first = list.data.find(m => S1_MODELS.some(s => s.id === m.id));
        if (first) onSelectActive(first.id);
      }
    } catch {
      setLoaded([]);
      setHealth(null);
    }
  }, [activeId, onSelectActive]);

  useEffect(() => {
    refresh();
  }, [refresh, refreshSignal]);

  useEffect(() => {
    const iv = setInterval(refresh, 5000);
    return () => clearInterval(iv);
  }, [refresh]);

  const loadModel = useCallback(
    async (spec: S1ModelSpec) => {
      setBusy(spec.id);
      setStage('unloading current model…');
      onLoadError('');
      try {
        const list: ModelListResponse = await fetchModelList();
        for (const m of list.data) {
          try {
            await fetch('/v1/models', {
              method: 'DELETE',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ id: m.id }),
            });
          } catch {
            /* keep going — best effort */
          }
        }
        setStage(`loading ${spec.params} backbone into VRAM… (up to a minute)`);
        const res = await fetch('/v1/models', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ id: spec.id, path: spec.ggufPath, gpu: true, context: spec.ctx }),
        });
        if (!res.ok) {
          let msg = `HTTP ${res.status}`;
          try {
            const err = await res.json();
            if (err?.error?.message) msg = err.error.message;
          } catch {
            /* ignore */
          }
          onLoadError(`Load failed: ${msg}`);
        }
        await refresh();
        onSelectActive(spec.id);
      } catch {
        onLoadError('Server unreachable — is dlgo-server running on :8080?');
      } finally {
        setBusy(null);
        setStage('');
      }
    },
    [onLoadError, onSelectActive, refresh],
  );

  const unload = useCallback(
    async (id: string) => {
      setBusy(id);
      setStage(`unloading ${id}…`);
      onLoadError('');
      try {
        await fetch('/v1/models', {
          method: 'DELETE',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ id }),
        });
        await refresh();
      } catch {
        onLoadError('Server unreachable — is dlgo-server running on :8080?');
      } finally {
        setBusy(null);
        setStage('');
      }
    },
    [onLoadError, refresh],
  );

  const vram = health?.vram;
  const vramPct = vram && vram.total_mb > 0 ? Math.min(100, (vram.used_mb / vram.total_mb) * 100) : 0;
  const loadedIds = new Set(loaded.map(m => m.id));
  const knownLoaded = loaded.filter(m => S1_MODELS.some(s => s.id === m.id));
  const unknownLoaded = loaded.filter(m => !S1_MODELS.some(s => s.id === m.id));

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-[var(--border)]">
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-secondary)]">Backbone</h2>
          <span className="text-[10px] text-[var(--text-secondary)]" title="One model at a time — VRAM hygiene">
            one at a time
          </span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {S1_MODELS.map(spec => {
          const isLoaded = loadedIds.has(spec.id);
          const isActive = activeId === spec.id && isLoaded;
          const isBusy = busy === spec.id;
          const otherBusy = busy !== null && !isBusy;
          return (
            <div
              key={spec.id}
              className={`rounded-xl border p-3 transition-colors ${
                isActive
                  ? 'border-[var(--accent)] bg-[var(--accent)]/10'
                  : 'border-[var(--border)] bg-[var(--bg-tertiary)] hover:border-[var(--text-secondary)]/40'
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <button
                  className={`text-left flex-1 min-w-0 ${isLoaded ? 'cursor-pointer' : 'cursor-default'}`}
                  onClick={() => isLoaded && onSelectActive(spec.id)}
                  title={isLoaded ? `Decide with ${spec.label}` : 'Load this model first'}
                >
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full shrink-0" style={{ background: isLoaded ? spec.accent : 'var(--border)' }} />
                    <span className="font-semibold text-sm text-[var(--text-primary)] truncate">{spec.label}</span>
                    <span className="text-[10px] text-[var(--text-secondary)] shrink-0">{spec.quant}</span>
                  </div>
                  <div className="mt-1.5 flex items-center gap-1.5 text-[10px]">
                    <span className="px-1.5 py-0.5 rounded bg-[var(--bg-primary)] text-[var(--success)] font-mono" title="splits-gp2 test, GPU engine">
                      {spec.testGpu.toFixed(1)}% GPU
                    </span>
                    <span className="px-1.5 py-0.5 rounded bg-[var(--bg-primary)] text-[var(--text-secondary)] font-mono" title="splits-gp2 test, CPU engine">
                      {spec.testCpu.toFixed(1)}% CPU
                    </span>
                  </div>
                  <div className="mt-1 text-[10px] text-[var(--text-secondary)] truncate" title={`${spec.readoutName} · device-any`}>
                    head {spec.readoutName} · ECE {spec.eceGpu.toFixed(3)}
                  </div>
                </button>
                {isBusy ? (
                  <span className="text-[10px] text-[var(--accent)] animate-pulse whitespace-nowrap pt-1">···</span>
                ) : isLoaded ? (
                  <button
                    onClick={() => unload(spec.id)}
                    disabled={otherBusy}
                    className="text-[10px] px-2 py-1 rounded border border-[var(--border)] text-[var(--text-secondary)] hover:border-[var(--danger)] hover:text-[var(--danger)] transition-colors disabled:opacity-40 shrink-0"
                    title="Unload (frees VRAM)"
                  >
                    unload
                  </button>
                ) : (
                  <button
                    onClick={() => loadModel(spec)}
                    disabled={otherBusy}
                    className="text-[10px] px-2 py-1 rounded font-medium transition-colors disabled:opacity-40 shrink-0"
                    style={{ background: spec.accent, color: '#0a0a0f' }}
                    title={`Load ${spec.ggufPath}`}
                  >
                    load
                  </button>
                )}
              </div>
              {isActive && <div className="mt-2 text-[10px] leading-snug text-[var(--text-secondary)]">{spec.note}</div>}
              {!isLoaded && !isBusy && <div className="mt-2 text-[10px] leading-snug text-[var(--text-secondary)] opacity-60">{spec.note}</div>}
            </div>
          );
        })}

        {unknownLoaded.map(m => (
          <div key={m.id} className="rounded-xl border border-[var(--border)] bg-[var(--bg-tertiary)] p-3">
            <div className="flex items-center justify-between gap-2">
              <div className="min-w-0">
                <div className="text-sm font-medium truncate">{m.id}</div>
                <div className="text-[10px] text-[var(--text-secondary)]">
                  {m.gpu ? 'GPU' : 'CPU'} · non-playground model
                </div>
              </div>
              <button
                onClick={() => unload(m.id)}
                disabled={busy !== null}
                className="text-[10px] px-2 py-1 rounded border border-[var(--border)] text-[var(--text-secondary)] hover:border-[var(--danger)] hover:text-[var(--danger)] disabled:opacity-40 shrink-0"
              >
                unload
              </button>
            </div>
          </div>
        ))}
      </div>

      <div className="border-t border-[var(--border)] p-3 space-y-2">
        <div className="flex items-center justify-between text-[10px] text-[var(--text-secondary)]">
          <span>VRAM</span>
          {vram ? (
            <span className="font-mono">
              {fmtMB(vram.used_mb)} / {fmtMB(vram.total_mb)}
            </span>
          ) : (
            <span>{health ? 'no GPU' : 'server offline'}</span>
          )}
        </div>
        <div className="h-1.5 rounded-full bg-[var(--bg-tertiary)] overflow-hidden">
          <div
            className="h-full rounded-full transition-all"
            style={{
              width: `${vramPct}%`,
              background: vramPct > 85 ? 'var(--danger)' : vramPct > 60 ? 'var(--accent)' : 'var(--success)',
            }}
          />
        </div>
        {busy && stage && <div className="text-[10px] text-[var(--accent)] animate-pulse">{stage}</div>}
        {knownLoaded.length > 0 && !busy && (
          <div className="text-[10px] text-[var(--success)]">
            ● {knownLoaded.find(m => m.id === activeId)?.id ?? knownLoaded[0].id} ready ({knownLoaded[0].gpu ? 'GPU' : 'CPU'})
          </div>
        )}
      </div>
    </div>
  );
}
