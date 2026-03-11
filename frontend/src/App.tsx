import { useState, useCallback } from 'react';
import { ChatPanel } from './components/ChatPanel';
import { ModelSelector } from './components/ModelSelector';
import { SettingsPanel } from './components/SettingsPanel';
import { loadModel, unloadModel, listModels } from './api';

function App() {
  const [model, setModel] = useState('');
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [topK, setTopK] = useState(40);
  const [maxTokens, setMaxTokens] = useState(512);
  const [systemPrompt, setSystemPrompt] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [useGPU, setUseGPU] = useState(false);
  const [reloading, setReloading] = useState(false);

  const handleGPUToggle = useCallback(async (gpu: boolean) => {
    if (gpu === useGPU) return;
    setUseGPU(gpu);

    // If a model is loaded, reload it with the new backend
    if (model) {
      setReloading(true);
      try {
        const models = await listModels();
        const current = models.find(m => m.id === model);
        if (current?.path) {
          await unloadModel(model);
          await loadModel({ id: model, path: current.path, gpu, context: 2048 });
        }
      } catch (e) {
        console.error('Failed to toggle backend:', e);
      } finally {
        setReloading(false);
      }
    }
  }, [useGPU, model]);

  return (
    <div className="h-screen flex">
      {/* Sidebar */}
      <div
        className={`
          ${sidebarOpen ? 'w-72' : 'w-0'}
          transition-all duration-200 overflow-hidden
          bg-[var(--bg-secondary)] border-r border-[var(--border)]
          flex flex-col
        `}
      >
        <div className="p-4 border-b border-[var(--border)]">
          <h1 className="text-lg font-bold tracking-tight">
            <span className="text-[var(--accent)]">dlgo</span>
            <span className="text-[var(--text-secondary)] font-normal ml-1 text-sm">server</span>
          </h1>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          <ModelSelector
            selectedModel={model}
            useGPU={useGPU}
            onSelectModel={setModel}
            onGPUStatusChange={setUseGPU}
          />
          <div className="border-t border-[var(--border)] pt-4">
            <SettingsPanel
              temperature={temperature}
              topP={topP}
              topK={topK}
              maxTokens={maxTokens}
              systemPrompt={systemPrompt}
              useGPU={useGPU}
              onTemperatureChange={setTemperature}
              onTopPChange={setTopP}
              onTopKChange={setTopK}
              onMaxTokensChange={setMaxTokens}
              onSystemPromptChange={setSystemPrompt}
              onGPUChange={handleGPUToggle}
            />
          </div>
        </div>

        <div className="p-4 border-t border-[var(--border)] text-xs text-[var(--text-secondary)]">
          Built with dlgo inference engine
        </div>
      </div>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top bar */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-[var(--border)] bg-[var(--bg-secondary)]">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors text-lg"
            title={sidebarOpen ? 'Close sidebar' : 'Open sidebar'}
          >
            {sidebarOpen ? '\u25C0' : '\u25B6'}
          </button>
          <div className="flex items-center gap-2 text-sm">
            {model ? (
              <>
                <span className="text-[var(--text-primary)] font-medium">{model}</span>
                <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold uppercase ${
                  useGPU
                    ? 'bg-[var(--success)]/20 text-[var(--success)]'
                    : 'bg-[var(--text-secondary)]/20 text-[var(--text-secondary)]'
                }`}>
                  {useGPU ? 'GPU' : 'CPU'}
                </span>
                {reloading && (
                  <span className="text-[var(--text-secondary)] text-xs animate-pulse">switching...</span>
                )}
              </>
            ) : (
              <span className="text-[var(--text-secondary)]">No model selected</span>
            )}
          </div>
        </div>

        {/* Chat */}
        <ChatPanel
          model={model}
          temperature={temperature}
          topP={topP}
          topK={topK}
          maxTokens={maxTokens}
          systemPrompt={systemPrompt}
        />
      </div>
    </div>
  );
}

export default App;
