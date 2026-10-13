// System One playground API: decision runs, timing parsing, health polling.
// Model load/unload reuse the shared helpers in ../api.

import type { S1RunResponse, TimingInfo } from './types';

export interface DecideResult {
  data: S1RunResponse;
  timing: TimingInfo;
}

export class S1ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

function parseServerTiming(header: string | null): { queueMs: number | null; decisionMs: number | null } {
  if (!header) return { queueMs: null, decisionMs: null };
  let queueMs: number | null = null;
  let decisionMs: number | null = null;
  const re = /([a-zA-Z_]+)\s*;\s*dur=([\d.]+)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(header)) !== null) {
    const name = m[1].toLowerCase();
    const v = parseFloat(m[2]);
    if (Number.isNaN(v)) continue;
    if (name === 'queue') queueMs = v;
    if (name === 'decision') decisionMs = v;
  }
  return { queueMs, decisionMs };
}

/** POST /v1/systemone. Measures wall time and parses the Server-Timing header. */
export async function decide(body: unknown, signal?: AbortSignal): Promise<DecideResult> {
  const t0 = performance.now();
  let res: Response;
  try {
    res = await fetch('/v1/systemone', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    });
  } catch (e: unknown) {
    if ((e as Error)?.name === 'AbortError') throw e;
    throw new S1ApiError(0, 'Server unreachable — is dlgo-server running on :8080?');
  }
  const wallMs = performance.now() - t0;

  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const err = await res.json();
      if (err?.error?.message) msg = err.error.message;
    } catch {
      /* keep status text */
    }
    throw new S1ApiError(res.status, msg);
  }

  const data: S1RunResponse = await res.json();
  const { queueMs, decisionMs } = parseServerTiming(res.headers.get('Server-Timing'));
  return { data, timing: { queueMs, decisionMs, wallMs } };
}

export interface HealthInfo {
  status: string;
  vram?: { total_mb: number; free_mb: number; used_mb: number } | null;
}

export async function fetchHealth(): Promise<HealthInfo> {
  const res = await fetch('/health');
  if (!res.ok) throw new S1ApiError(res.status, res.statusText);
  return res.json();
}

export interface ModelListResponse {
  object: string;
  data: Array<{ id: string; object: string; created: number; owned_by: string; architecture?: string; gpu: boolean; path?: string }>;
  available: Array<{ id: string; path: string }>;
}

export async function fetchModelList(): Promise<ModelListResponse> {
  const res = await fetch('/v1/models');
  if (!res.ok) throw new S1ApiError(res.status, res.statusText);
  return res.json();
}

/** Normalize the answers map into a sorted array of parsed answers. */
export function parseAnswers(raw: Record<string, any> | undefined): import('./types').S1Answer[] {
  if (!raw) return [];
  return Object.keys(raw)
    .sort()
    .map(id => {
      const a = raw[id] ?? {};
      return {
        id,
        type: (a.type ?? 'choice') as 'choice' | 'noul' | 'score',
        choice: a.choice,
        score: a.score,
        noul: a.noul,
        probabilities: a.probabilities,
        confidence: a.confidence,
        legend: a.legend,
        abstained: !!a.abstained,
        degraded: !!a.degraded,
      };
    });
}

export function collectEngine(data: S1RunResponse): string | null {
  if (data.engine) return data.engine;
  if (Array.isArray(data.results)) {
    for (const r of data.results) {
      if (r.engine) return r.engine;
    }
  }
  return null;
}
