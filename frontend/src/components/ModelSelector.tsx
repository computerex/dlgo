import { useState, useEffect, useCallback } from 'react';
import type { ModelObject } from '../api';
import { listModels, loadModel, unloadModel } from '../api';

interface ModelSelectorProps {
  selectedModel: string;
  useGPU: boolean;
  onSelectModel: (id: string) => void;
  onGPUStatusChange: (gpu: boolean) => void;
}

export function ModelSelector({ selectedModel, useGPU, onSelectModel, onGPUStatusChange }: ModelSelectorProps) {
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
        onGPUStatusChange(m[0].gpu);
      }
      if (selectedModel) {
        const current = m.find(x => x.id === selectedModel);
        if (current) {
          onGPUStatusChange(current.gpu);
        }
      }
    } catch {
      // server not running yet
    }
  }, [selectedModel, onSelectModel, onGPUStatusChange]);

  useEffect(() => {
    refreshModels();
    const interval = setInterval(refreshModels, 5000);
    return () => clearInterval(interval);
  }, [refreshModels]);

  const handleSelectModel = (id: string) => {
    onSelectModel(id);
    const model = models.find(m => m.id === id);
    if (model) {
      onGPUStatusChange(model.gpu);
    }
  };

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

  const currentModel = models.find(m => m.id === selectedModel);

  return (
    <div className="space-y-3">
      <label className="text-xs font-medium uppercase tracking-wide text-[var(--text-secondary)]">Model</label>

      <select
        value={selectedModel}
        onChange={e => handleSelectModel(e.target.value)}
        className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent)]"
      >
        <option value="">Select a model...</option>
        {models.map(m => (
          <option key={m.id} value={m.id}>
            {m.id} {m.architecture ? `(${m.architecture})` : ''}
          </option>
        ))}
      </select>

      {currentModel && (
        <div className="flex items-center gap-2 text-xs">
          <span className={`px-2 py-0.5 rounded-full font-medium ${
            currentModel.gpu
              ? 'bg-[var(--success)]/20 text-[var(--success)]'
              : 'bg-[var(--text-secondary)]/20 text-[var(--text-secondary)]'
          }`}>
            {currentModel.gpu ? 'GPU' : 'CPU'}
          </span>
          {currentModel.architecture && (
            <span className="text-[var(--text-secondary)]">{currentModel.architecture}</span>
          )}
        </div>
      )}

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
          defaultGPU={useGPU}
        />
      )}
    </div>
  );
}

function ModelLoaderDialog({
  onLoad,
  onClose,
  loading,
  defaultGPU,
}: {
  onLoad: (path: string, gpu: boolean, ctx: number) => void;
  onClose: () => void;
  loading: boolean;
  defaultGPU: boolean;
}) {
  const [path, setPath] = useState('');
  const [gpu, setGpu] = useState(defaultGPU);
  const [ctx, setCtx] = useState(2048);

  return (
    <div className="border border-[var(--border)] rounded-lg p-4 bg-[var(--bg-tertiary)] space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium">Load GGUF Model</span>
        <button onClick={onClose} className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] text-sm">&#10005;</button>
      </div>
      <input
        type="text"
        value={path}
        onChange={e => setPath(e.target.value)}
        placeholder="/path/to/model.gguf"
        className="w-full bg-[var(--bg-primary)] border border-[var(--border)] rounded px-3 py-2 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-secondary)] focus:outline-none focus:border-[var(--accent)]"
      />
      <div className="flex items-center gap-4">
        <div className="flex rounded-lg overflow-hidden border border-[var(--border)]">
          <button
            onClick={() => setGpu(false)}
            className={`px-3 py-1.5 text-xs font-medium transition-colors ${
              !gpu
                ? 'bg-[var(--accent)] text-white'
                : 'bg-[var(--bg-primary)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
            }`}
          >
            CPU
          </button>
          <button
            onClick={() => setGpu(true)}
            className={`px-3 py-1.5 text-xs font-medium transition-colors ${
              gpu
                ? 'bg-[var(--accent)] text-white'
                : 'bg-[var(--bg-primary)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
            }`}
          >
            GPU
          </button>
        </div>
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
