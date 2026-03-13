import { useState, useEffect, useCallback } from 'react';
import type { ModelObject, AvailableModel } from '../api';
import { listModels, listAvailableModels, loadModel, unloadModel } from '../api';

interface ModelSelectorProps {
  selectedModel: string;
  useGPU: boolean;
  onSelectModel: (id: string) => void;
  onGPUStatusChange: (gpu: boolean) => void;
}

export function ModelSelector({ selectedModel, useGPU, onSelectModel, onGPUStatusChange }: ModelSelectorProps) {
  const [loadedModels, setLoadedModels] = useState<ModelObject[]>([]);
  const [availableModels, setAvailableModels] = useState<AvailableModel[]>([]);
  const [loadingModelId, setLoadingModelId] = useState<string | null>(null);
  const [error, setError] = useState('');

  const refreshModels = useCallback(async () => {
    try {
      const [loaded, available] = await Promise.all([
        listModels(),
        listAvailableModels()
      ]);
      setLoadedModels(loaded);
      setAvailableModels(available);
      
      // Auto-select first loaded model if none selected
      if (loaded.length > 0 && !selectedModel) {
        const firstModel = loaded[0];
        onSelectModel(firstModel.id);
        onGPUStatusChange(firstModel.gpu);
      }
      
      // Update GPU status for currently selected model
      if (selectedModel) {
        const current = loaded.find(m => m.id === selectedModel);
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

  const handleLoad = async (model: AvailableModel, gpu: boolean, ctx: number) => {
    setLoadingModelId(model.id);
    setError('');
    try {
      await loadModel({ id: model.id, path: model.path, gpu, context: ctx });
      await refreshModels();
      onSelectModel(model.id);
      onGPUStatusChange(gpu);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load model');
    } finally {
      setLoadingModelId(null);
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

  const currentLoadedModel = loadedModels.find(m => m.id === selectedModel);

  return (
    <div className="space-y-3">
      <label className="text-xs font-medium uppercase tracking-wide text-[var(--text-secondary)]">Model</label>

      {/* Loaded Models Section */}
      {loadedModels.length > 0 && (
        <>
          <select
            value={selectedModel}
            onChange={e => onSelectModel(e.target.value)}
            className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent)]"
          >
            <option value="">Select a loaded model...</option>
            {loadedModels.map(m => (
              <option key={m.id} value={m.id}>
                {m.id} {m.architecture ? `(${m.architecture})` : ''} • {m.gpu ? 'GPU' : 'CPU'}
              </option>
            ))}
          </select>

          {currentLoadedModel && (
            <div className="flex items-center gap-2 text-xs">
              <span className={`px-2 py-0.5 rounded-full font-medium ${
                currentLoadedModel.gpu
                  ? 'bg-[var(--success)]/20 text-[var(--success)]'
                  : 'bg-[var(--text-secondary)]/20 text-[var(--text-secondary)]'
              }`}>
                {currentLoadedModel.gpu ? 'GPU' : 'CPU'}
              </span>
              {currentLoadedModel.architecture && (
                <span className="text-[var(--text-secondary)]">{currentLoadedModel.architecture}</span>
              )}
            </div>
          )}

          {selectedModel && (
            <button
              onClick={() => handleUnload(selectedModel)}
              className="w-full px-3 py-2 border border-[var(--danger)] text-[var(--danger)] hover:bg-[var(--danger)] hover:text-white rounded-lg text-xs font-medium transition-colors"
            >
              Unload {selectedModel}
            </button>
          )}
        </>
      )}

      {/* Available Models Section */}
      {availableModels.length > 0 && (
        <div className="border border-[var(--border)] rounded-lg p-3 bg-[var(--bg-tertiary)]">
          <div className="text-xs font-medium text-[var(--text-secondary)] mb-2 uppercase tracking-wide">
            Available Models ({availableModels.length})
          </div>
          <div className="max-h-48 overflow-y-auto space-y-1">
            {availableModels.map(m => {
              const isLoaded = loadedModels.some(lm => lm.id === m.id);
              const isSelected = selectedModel === m.id;
              const isLoading = loadingModelId === m.id;
              return (
                <div
                  key={m.id}
                  className={`flex items-center justify-between gap-2 px-2 py-1.5 rounded text-sm ${
                    isSelected 
                      ? 'bg-[var(--accent)]/20 border border-[var(--accent)]' 
                      : 'hover:bg-[var(--bg-primary)]'
                  }`}
                >
                  <span className="flex-1 min-w-0 text-[var(--text-primary)]" title={m.id}>
                    <span className="block truncate">{m.id}</span>
                  </span>
                  {isLoading ? (
                    <span className="ml-2 px-2 py-0.5 text-[var(--accent)] text-xs animate-pulse">
                      Loading...
                    </span>
                  ) : !isLoaded ? (
                    <button
                      onClick={() => handleLoad(m, useGPU, 0)}
                      disabled={loadingModelId !== null}
                      className="ml-2 px-2 py-0.5 bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white rounded text-xs transition-colors disabled:opacity-40"
                    >
                      Load
                    </button>
                  ) : (
                    <span className="ml-2 px-2 py-0.5 bg-[var(--success)]/20 text-[var(--success)] rounded text-xs">
                      Loaded
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {error && <p className="text-xs text-[var(--danger)]">{error}</p>}

      {loadedModels.length === 0 && availableModels.length === 0 && (
        <p className="text-xs text-[var(--text-secondary)]">No models found. Use --models-dir flag to specify model directories.</p>
      )}
    </div>
  );
}
