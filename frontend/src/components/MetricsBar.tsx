interface MetricsBarProps {
  tokPerSec: number;
  ttft: number;
  totalMs: number;
}

export function MetricsBar({ tokPerSec, ttft, totalMs }: MetricsBarProps) {
  return (
    <div className="flex items-center justify-center gap-6 px-4 py-2 text-xs text-[var(--text-secondary)] border-t border-[var(--border)] bg-[var(--bg-secondary)]">
      <div className="flex items-center gap-1.5">
        <span className="text-[var(--success)] font-mono font-bold">{tokPerSec.toFixed(1)}</span>
        <span>tok/s</span>
      </div>
      <div className="flex items-center gap-1.5">
        <span className="font-mono">{ttft.toFixed(0)}</span>
        <span>ms TTFT</span>
      </div>
      <div className="flex items-center gap-1.5">
        <span className="font-mono">{(totalMs / 1000).toFixed(1)}</span>
        <span>s total</span>
      </div>
    </div>
  );
}
