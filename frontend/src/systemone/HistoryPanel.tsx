import type { S1Run } from './types';

interface HistoryPanelProps {
  runs: S1Run[];
  onRestore: (run: S1Run) => void;
  onClear: () => void;
}

function topSummary(run: S1Run): string {
  const parts: string[] = [];
  for (const a of run.answers) {
    if (a.type === 'noul') {
      parts.push(`${a.id}=${a.noul !== undefined ? (a.noul * 100).toFixed(0) + '%' : '?'}`);
    } else if (a.type === 'score') {
      parts.push(`${a.id}=${a.score?.toFixed(2) ?? '?'}`);
    } else {
      parts.push(`${a.id}=${a.choice ?? '?'}`);
    }
  }
  return parts.join('  ');
}

export function HistoryPanel({ runs, onRestore, onClear }: HistoryPanelProps) {
  return (
    <div className="border-t border-[var(--border)]">
      <div className="flex items-center justify-between px-3 py-2">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-secondary)]">
          Runs {runs.length > 0 && <span className="font-mono normal-case">({runs.length})</span>}
        </span>
        {runs.length > 0 && (
          <button onClick={onClear} className="text-[10px] text-[var(--text-secondary)] hover:text-[var(--danger)]">
            clear
          </button>
        )}
      </div>
      {runs.length === 0 ? (
        <div className="px-3 pb-3 text-[10px] text-[var(--text-secondary)]">No runs yet — decisions appear here with latency and answers.</div>
      ) : (
        <div className="max-h-56 overflow-y-auto px-2 pb-2 space-y-1">
          {runs.map(r => (
            <button
              key={r.id}
              onClick={() => onRestore(r)}
              title="Restore this request into the editor"
              className="w-full text-left rounded-lg border border-[var(--border)] bg-[var(--bg-tertiary)] hover:border-[var(--accent)] transition-colors p-2"
            >
              <div className="flex items-center gap-2 text-[10px]">
                <span className="font-semibold truncate flex-1">{r.label}</span>
                <span className="font-mono text-[var(--accent-hover)]">{r.totalMs >= 1000 ? `${(r.totalMs / 1000).toFixed(2)}s` : `${Math.round(r.totalMs)}ms`}</span>
                <span className="text-[var(--text-secondary)] font-mono">{new Date(r.ts).toLocaleTimeString()}</span>
              </div>
              <div className="mt-1 text-[10px] text-[var(--text-secondary)] font-mono truncate">{topSummary(r)}</div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
