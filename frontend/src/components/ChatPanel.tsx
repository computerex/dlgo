import { useState, useRef, useEffect, useCallback } from 'react';
import type { Message, ChatSession, ChatMessage as ApiChatMessage } from '../api';
import { chatCompletion } from '../api';
import { MessageBubble } from './MessageBubble';
import { MetricsBar } from './MetricsBar';

interface ChatPanelProps {
  chat: ChatSession | null;
  model: string;
  temperature: number;
  topP: number;
  topK: number;
  maxTokens: number;
  systemPrompt: string;
  reasoningEffort: 'low' | 'medium' | 'high';
}

interface LocalChatMessage extends Message {
  id: string;
  streaming?: boolean;
  metrics?: { tokPerSec: number; ttft: number; totalMs: number };
}

export function ChatPanel({ chat, model, temperature, topP, topK, maxTokens, systemPrompt, reasoningEffort }: ChatPanelProps) {
  const [messages, setMessages] = useState<LocalChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [generating, setGenerating] = useState(false);
  const [metrics, setMetrics] = useState<{ tokPerSec: number; ttft: number; totalMs: number } | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Load messages when chat changes
  useEffect(() => {
    if (chat) {
      const loadedMessages: LocalChatMessage[] = chat.messages.map((m: ApiChatMessage) => ({
        id: m.id,
        role: m.role as 'system' | 'user' | 'assistant',
        content: m.content,
      }));
      setMessages(loadedMessages);
    } else {
      setMessages([]);
    }
    setMetrics(null);
  }, [chat?.id]);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(scrollToBottom, [messages, scrollToBottom]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || generating || !model || !chat) return;

    const userMsg: LocalChatMessage = { id: crypto.randomUUID(), role: 'user', content: text };
    const assistantMsg: LocalChatMessage = { id: crypto.randomUUID(), role: 'assistant', content: '', streaming: true };

    setMessages(prev => [...prev, userMsg, assistantMsg]);
    setInput('');
    setGenerating(true);
    setMetrics(null);

    const apiMessages: Message[] = [];
    if (systemPrompt) {
      apiMessages.push({ role: 'system', content: systemPrompt });
    }
    for (const m of [...messages, userMsg]) {
      apiMessages.push({ role: m.role, content: m.content });
    }

    const abort = new AbortController();
    abortRef.current = abort;
    let tokenCount = 0;
    const startTime = performance.now();
    let firstTokenTime = 0;

    try {
      await chatCompletion(
        { model, messages: apiMessages, temperature, top_p: topP, top_k: topK, max_tokens: maxTokens, reasoning_effort: reasoningEffort },
        {
          onToken: (token) => {
            tokenCount++;
            if (tokenCount === 1) firstTokenTime = performance.now();
            setMessages(prev =>
              prev.map(m =>
                m.id === assistantMsg.id ? { ...m, content: m.content + token } : m
              )
            );
          },
          onDone: () => {
            const totalMs = performance.now() - startTime;
            const ttft = firstTokenTime ? firstTokenTime - startTime : totalMs;
            const genMs = totalMs - ttft;
            const tokPerSec = genMs > 0 ? (tokenCount / genMs) * 1000 : 0;
            const m = { tokPerSec, ttft, totalMs };
            setMetrics(m);
            setMessages(prev =>
              prev.map(msg =>
                msg.id === assistantMsg.id ? { ...msg, streaming: false, metrics: m } : msg
              )
            );
          },
          onError: (error) => {
            setMessages(prev =>
              prev.map(m =>
                m.id === assistantMsg.id ? { ...m, content: `Error: ${error}`, streaming: false } : m
              )
            );
          },
        },
        abort.signal,
      );
    } catch (e: unknown) {
      if (e instanceof Error && e.name !== 'AbortError') {
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantMsg.id ? { ...m, content: `Error: ${e instanceof Error ? e.message : 'Unknown error'}`, streaming: false } : m
          )
        );
      }
    } finally {
      setGenerating(false);
      abortRef.current = null;
    }
  };

  const handleStop = () => {
    abortRef.current?.abort();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleClear = () => {
    setMessages([]);
    setMetrics(null);
  };

  if (!chat) {
    return (
      <div className="flex-1 flex items-center justify-center text-[var(--text-secondary)] text-sm">
        <div className="text-center">
          <div className="text-4xl mb-4 opacity-30">💬</div>
          <p>Select a chat or create a new one</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full text-[var(--text-secondary)] text-sm">
            <div className="text-center">
              <div className="text-4xl mb-4 opacity-30">⚡</div>
              <p>Send a message to start chatting</p>
              {!model && <p className="text-xs mt-2 text-[var(--danger)]">No model loaded</p>}
            </div>
          </div>
        )}
        {messages.map(msg => (
          <MessageBubble key={msg.id} message={msg} streaming={msg.streaming} />
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Metrics */}
      {metrics && <MetricsBar {...metrics} />}

      {/* Input area */}
      <div className="border-t border-[var(--border)] p-4">
        <div className="flex gap-2 items-end max-w-3xl mx-auto">
          <textarea
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={model ? 'Type a message...' : 'Load a model first'}
            disabled={!model}
            rows={1}
            className="flex-1 resize-none bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-4 py-3 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-secondary)] focus:outline-none focus:border-[var(--accent)] disabled:opacity-50 min-h-[44px] max-h-[200px]"
            style={{ height: 'auto', overflow: 'hidden' }}
            onInput={e => {
              const t = e.currentTarget;
              t.style.height = 'auto';
              t.style.height = Math.min(t.scrollHeight, 200) + 'px';
            }}
          />
          {generating ? (
            <button
              onClick={handleStop}
              className="px-4 py-3 bg-[var(--danger)] hover:opacity-90 text-white rounded-lg text-sm font-medium transition-opacity"
            >
              Stop
            </button>
          ) : (
            <button
              onClick={handleSend}
              disabled={!input.trim() || !model}
              className="px-4 py-3 bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              Send
            </button>
          )}
          <button
            onClick={handleClear}
            className="px-3 py-3 text-[var(--text-secondary)] hover:text-[var(--text-primary)] text-sm transition-colors"
            title="Clear chat"
          >
            ✕
          </button>
        </div>
      </div>
    </div>
  );
}
