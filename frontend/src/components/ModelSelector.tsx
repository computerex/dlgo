import { useState, useEffect, useCallback } from 'react';
import type { ModelObject } from '../api';
import { listModels, loadModel, unloadModel } from '../api';

interface ModelSelectorProps {
  selectedModel: string;
  onSelectModel: (id: string) => void;
}

export function ModelSelector({ selectedModel, onSelectModel }: ModelSelectorProps) {
  const [models, setModels] = useState<ModelObject[]>([]);
  const [showLoader, setShowLoader] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const refreshModels = useCallback(async () => {
    try {
      const m = await listModels();
      setModels(m);
      if (m.length > 0 && !selectedModel) {
        onSelectModel(m[0].id);
      }
    } catch {
      // server not running yet
    }
  }, [selectedModel, onSelectModel]);

  useEffect(() => {
    refreshModels();
    const interval = setInterval(refreshModels, 5000);
    return () => clearInterval(interval);
  }, [refreshModels]);

  const handleLoad = async (path: string, gpu: boolean, ctx: number) => {
    setLoading(true);
    setError('');
    try {
      await loadModel({ path, gpu, context: ctx });
      await refreshModels();
      setShowLoader(false);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load model');
    } finally {
      setLoading(false);
    }
  };

  const handleUnload = async (id: string) => {
    try {
      await unloadModel(id);
      if (selectedModel === id) onSelectModel('');
      await refreshModels();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to unload model');
    }
  };

  return (
    <div className="space-y-3">
      <label className="text-xs font-medium uppercase tracking-wide text-[var(--text-secondary)]">Model</label>

      <select
        value={selectedModel}
        onChange={e => onSelectModel(e.target.value)}
        className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent)]"
      >
        <option value="">Select a model...</option>
        {models.map(m => (
          <option key={m.id} value={m.id}>
            {m.id} {m.architecture ? `(${m.architecture})` : ''}
          </option>
        ))}
      </select>

      <div className="flex gap-2">
        <button
          onClick={() => setShowLoader(true)}
          className="flex-1 px-3 py-2 bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white rounded-lg text-xs font-medium transition-colors"
        >
          Load Model
        </button>
        {selectedModel && (
          <button
            onClick={() => handleUnload(selectedModel)}
            className="px-3 py-2 border border-[var(--danger)] text-[var(--danger)] hover:bg-[var(--danger)] hover:text-white rounded-lg text-xs font-medium transition-colors"
          >
            Unload
          </button>
        )}
      </div>

      {error && <p className="text-xs text-[var(--danger)]">{error}</p>}

      {showLoader && (
        <ModelLoaderDialog
          onLoad={handleLoad}
          onClose={() => setShowLoader(false)}
          loading={loading}
        />
      )}
    </div>
  );
}

function ModelLoaderDialog({
  onLoad,
  onClose,
  loading,
}: {
  onLoad: (path: string, gpu: boolean, ctx: number) => void;
  onClose: () => void;
  loading: boolean;
}) {
  const [path, setPath] = useState('');
  const [gpu, setGpu] = useState(true);
  const [ctx, setCtx] = useState(2048);

  return (
    <div className="border border-[var(--border)] rounded-lg p-4 bg-[var(--bg-tertiary)] space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium">Load GGUF Model</span>
        <button onClick={onClose} className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] text-sm">✕</button>
      </div>
      <input
        type="text"
        value={path}
        onChange={e => setPath(e.target.value)}
        placeholder="/path/to/model.gguf"
        className="w-full bg-[var(--bg-primary)] border border-[var(--border)] rounded px-3 py-2 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-secondary)] focus:outline-none focus:border-[var(--accent)]"
      />
      <div className="flex items-center gap-4">
        <label className="flex items-center gap-2 text-sm cursor-pointer">
          <input type="checkbox" checked={gpu} onChange={e => setGpu(e.target.checked)} className="accent-[var(--accent)]" />
          GPU (Vulkan)
        </label>
        <label className="flex items-center gap-2 text-sm">
          Context:
          <input
            type="number"
            value={ctx}
            onChange={e => setCtx(parseInt(e.target.value) || 2048)}
            className="w-20 bg-[var(--bg-primary)] border border-[var(--border)] rounded px-2 py-1 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent)]"
          />
        </label>
      </div>
      <button
        onClick={() => onLoad(path, gpu, ctx)}
        disabled={!path || loading}
        className="w-full px-3 py-2 bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-40"
      >
        {loading ? 'Loading...' : 'Load'}
      </button>
    </div>
  );
}
