const API_BASE = '/v1';

export interface Message {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface ChatCompletionRequest {
  model: string;
  messages: Message[];
  temperature?: number;
  top_p?: number;
  top_k?: number;
  max_tokens?: number;
  stream?: boolean;
  reasoning_effort?: 'low' | 'medium' | 'high';
}

export interface ModelObject {
  id: string;
  object: string;
  created: number;
  owned_by: string;
  architecture?: string;
  quantization?: string;
  parameters?: string;
  gpu: boolean;
  path?: string;
}

export interface AvailableModel {
  id: string;
  path: string;
}

export interface LoadModelRequest {
  id?: string;
  path?: string;
  gpu?: boolean;
  context?: number;
}

export async function listModels(): Promise<ModelObject[]> {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error(`Failed to list models: ${res.statusText}`);
  const data = await res.json();
  return data.data || [];
}

export async function listAvailableModels(): Promise<AvailableModel[]> {
  const res = await fetch(`${API_BASE}/models/available`);
  if (!res.ok) throw new Error(`Failed to list available models: ${res.statusText}`);
  const data = await res.json();
  return data.data || [];
}

// Chat management
export interface ChatMessage {
  id: string;
  role: string;
  content: string;
  created_at: string;
}

export interface ChatSession {
  id: string;
  title: string;
  model: string;
  messages: ChatMessage[];
  created_at: string;
  updated_at: string;
}

export async function listChats(): Promise<ChatSession[]> {
  const res = await fetch(`${API_BASE}/chats`);
  if (!res.ok) throw new Error(`Failed to list chats: ${res.statusText}`);
  const data = await res.json();
  return data.data || [];
}

export async function createChat(title: string, model: string): Promise<ChatSession> {
  const res = await fetch(`${API_BASE}/chats`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title, model }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: { message: res.statusText } }));
    throw new Error(err.error?.message || 'Failed to create chat');
  }
  return res.json();
}

export async function getChat(id: string): Promise<ChatSession> {
  const res = await fetch(`${API_BASE}/chats/${id}`);
  if (!res.ok) throw new Error(`Failed to get chat: ${res.statusText}`);
  return res.json();
}

export async function deleteChat(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/chats/${id}`, {
    method: 'DELETE',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: { message: res.statusText } }));
    throw new Error(err.error?.message || 'Failed to delete chat');
  }
}

export async function loadModel(req: LoadModelRequest): Promise<void> {
  const res = await fetch(`${API_BASE}/models`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: { message: res.statusText } }));
    throw new Error(err.error?.message || 'Failed to load model');
  }
}

export async function unloadModel(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/models`, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: { message: res.statusText } }));
    throw new Error(err.error?.message || 'Failed to unload model');
  }
}

export interface StreamCallbacks {
  onToken: (token: string) => void;
  onDone: (finishReason: string) => void;
  onError: (error: string) => void;
}

export async function chatCompletion(
  req: ChatCompletionRequest,
  callbacks: StreamCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_BASE}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...req, stream: true }),
    signal,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: { message: res.statusText } }));
    callbacks.onError(err.error?.message || `HTTP ${res.status}`);
    return;
  }

  const reader = res.body?.getReader();
  if (!reader) {
    callbacks.onError('No response body');
    return;
  }

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6).trim();
      if (data === '[DONE]') {
        callbacks.onDone('stop');
        return;
      }
      try {
        const chunk = JSON.parse(data);
        const choice = chunk.choices?.[0];
        if (choice?.delta?.content) {
          callbacks.onToken(choice.delta.content);
        }
        if (choice?.finish_reason) {
          callbacks.onDone(choice.finish_reason);
          return;
        }
      } catch {
        // skip malformed chunks
      }
    }
  }
  callbacks.onDone('stop');
}
