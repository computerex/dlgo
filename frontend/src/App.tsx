import { useState, useCallback } from 'react';
import { ChatPanel } from './components/ChatPanel';
import { ChatList } from './components/ChatList';
import { ModelSelector } from './components/ModelSelector';
import { SettingsPanel } from './components/SettingsPanel';
import { loadModel, unloadModel, listModels } from './api';
import type { ChatSession } from './api';

function App() {
  const [model, setModel] = useState('');
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [topK, setTopK] = useState(40);
  const [maxTokens, setMaxTokens] = useState(512);
  const [systemPrompt, setSystemPrompt] = useState('');
  const [reasoningEffort, setReasoningEffort] = useState<'low' | 'medium' | 'high'>('medium');
  const [leftSidebarOpen, setLeftSidebarOpen] = useState(true);
  const [rightSidebarOpen, setRightSidebarOpen] = useState(true);
  const [useGPU, setUseGPU] = useState(false);
  const [reloading, setReloading] = useState(false);
  const [currentChat, setCurrentChat] = useState<ChatSession | null>(null);

  const handleGPUToggle = useCallback(async (gpu: boolean) => {
    if (gpu === useGPU) return;
    setUseGPU(gpu);

    if (model) {
      setReloading(true);
      try {
        const models = await listModels();
        const current = models.find(m => m.id === model);
        if (current?.path) {
          await unloadModel(model);
          await loadModel({ id: model, path: current.path, gpu, context: 0 });
        }
      } catch (e) {
        console.error('Failed to toggle backend:', e);
      } finally {
        setReloading(false);
      }
    }
  }, [useGPU, model]);

  const handleChatSelect = useCallback((chat: ChatSession | null) => {
    setCurrentChat(chat);
    if (chat?.model) {
      setModel(chat.model);
    }
  }, []);

  return (
    <div className="h-screen flex">
      {/* Left Sidebar - Chat List */}
      <div
        className={`
          ${leftSidebarOpen ? 'w-64' : 'w-0'}
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

        <ChatList
          currentChat={currentChat}
          onSelectChat={handleChatSelect}
          currentModel={model}
        />
      </div>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top bar */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border)] bg-[var(--bg-secondary)]">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setLeftSidebarOpen(!leftSidebarOpen)}
              className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors text-lg"
              title={leftSidebarOpen ? 'Close chat list' : 'Open chat list'}
            >
              {leftSidebarOpen ? '\u25C0' : '\u25B6'}
            </button>
            <div className="flex items-center gap-2 text-sm">
              {currentChat ? (
                <>
                  <span className="text-[var(--text-primary)] font-medium truncate max-w-[200px]">{currentChat.title}</span>
                  <span className="text-[var(--text-secondary)]">•</span>
                  <span className="text-[var(--text-secondary)] text-xs">{model || 'No model'}</span>
                </>
              ) : (
                <span className="text-[var(--text-secondary)]">No chat selected</span>
              )}
              {model && (
                <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold uppercase ${
                  useGPU
                    ? 'bg-[var(--success)]/20 text-[var(--success)]'
                    : 'bg-[var(--text-secondary)]/20 text-[var(--text-secondary)]'
                }`}>
                  {useGPU ? 'GPU' : 'CPU'}
                </span>
              )}
              {reloading && (
                <span className="text-[var(--text-secondary)] text-xs animate-pulse">switching...</span>
              )}
            </div>
          </div>
          <button
            onClick={() => setRightSidebarOpen(!rightSidebarOpen)}
            className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors text-lg"
            title={rightSidebarOpen ? 'Close settings' : 'Open settings'}
          >
            {rightSidebarOpen ? '\u25B6' : '\u25C0'}
          </button>
        </div>

        {/* Chat */}
        <ChatPanel
          chat={currentChat}
          model={model}
          temperature={temperature}
          topP={topP}
          topK={topK}
          maxTokens={maxTokens}
          systemPrompt={systemPrompt}
          reasoningEffort={reasoningEffort}
        />
      </div>

      {/* Right Sidebar - Settings */}
      <div
        className={`
          ${rightSidebarOpen ? 'w-72' : 'w-0'}
          transition-all duration-200 overflow-hidden
          bg-[var(--bg-secondary)] border-l border-[var(--border)]
          flex flex-col
        `}
      >
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
              reasoningEffort={reasoningEffort}
              useGPU={useGPU}
              onTemperatureChange={setTemperature}
              onTopPChange={setTopP}
              onTopKChange={setTopK}
              onMaxTokensChange={setMaxTokens}
              onSystemPromptChange={setSystemPrompt}
              onReasoningEffortChange={setReasoningEffort}
              onGPUChange={handleGPUToggle}
            />
          </div>
        </div>

        <div className="flex items-center justify-center px-4 py-2 text-xs text-[var(--text-secondary)] border-t border-[var(--border)] bg-[var(--bg-secondary)] h-[33px]">
          <span>Built with dlgo inference engine</span>
        </div>
      </div>
    </div>
  );
}

export default App;