import { useState, useEffect, useCallback } from 'react';
import type { ChatSession } from '../api';
import { listChats, createChat, deleteChat } from '../api';

interface ChatListProps {
  currentChat: ChatSession | null;
  onSelectChat: (chat: ChatSession | null) => void;
  currentModel: string;
}

export function ChatList({ currentChat, onSelectChat, currentModel }: ChatListProps) {
  const [chats, setChats] = useState<ChatSession[]>([]);
  const [isCreating, setIsCreating] = useState(false);
  const [newChatTitle, setNewChatTitle] = useState('');

  const refreshChats = useCallback(async () => {
    try {
      const chatList = await listChats();
      setChats(chatList);
    } catch (e) {
      console.error('Failed to load chats:', e);
    }
  }, []);

  useEffect(() => {
    refreshChats();
    const interval = setInterval(refreshChats, 5000);
    return () => clearInterval(interval);
  }, [refreshChats]);

  const handleCreateChat = async () => {
    if (!currentModel) {
      alert('Please load a model first');
      return;
    }
    
    const title = newChatTitle.trim() || 'New Chat';
    try {
      const chat = await createChat(title, currentModel);
      setNewChatTitle('');
      setIsCreating(false);
      await refreshChats();
      onSelectChat(chat);
    } catch (e) {
      console.error('Failed to create chat:', e);
      alert('Failed to create chat');
    }
  };

  const handleDeleteChat = async (e: React.MouseEvent, chatId: string) => {
    e.stopPropagation();
    if (!confirm('Delete this chat?')) return;
    
    try {
      await deleteChat(chatId);
      if (currentChat?.id === chatId) {
        onSelectChat(null);
      }
      await refreshChats();
    } catch (e) {
      console.error('Failed to delete chat:', e);
      alert('Failed to delete chat');
    }
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    
    if (days === 0) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } else if (days === 1) {
      return 'Yesterday';
    } else if (days < 7) {
      return date.toLocaleDateString([], { weekday: 'short' });
    } else {
      return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    }
  };

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <div className="p-3 border-b border-[var(--border)]">
        <button
          onClick={() => setIsCreating(true)}
          disabled={!currentModel}
          className="w-full px-3 py-2 bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          <span>+</span>
          <span>New Chat</span>
        </button>
        {!currentModel && (
          <p className="text-xs text-[var(--text-secondary)] mt-2 text-center">
            Load a model to create chats
          </p>
        )}
      </div>

      {isCreating && (
        <div className="p-3 border-b border-[var(--border)] bg-[var(--bg-tertiary)]">
          <input
            type="text"
            value={newChatTitle}
            onChange={(e) => setNewChatTitle(e.target.value)}
            placeholder="Chat title..."
            className="w-full bg-[var(--bg-primary)] border border-[var(--border)] rounded px-3 py-2 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-secondary)] focus:outline-none focus:border-[var(--accent)] mb-2"
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleCreateChat();
              if (e.key === 'Escape') {
                setIsCreating(false);
                setNewChatTitle('');
              }
            }}
            autoFocus
          />
          <div className="flex gap-2">
            <button
              onClick={handleCreateChat}
              className="flex-1 px-3 py-1.5 bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white rounded text-xs font-medium transition-colors"
            >
              Create
            </button>
            <button
              onClick={() => {
                setIsCreating(false);
                setNewChatTitle('');
              }}
              className="flex-1 px-3 py-1.5 border border-[var(--border)] hover:bg-[var(--bg-primary)] text-[var(--text-secondary)] rounded text-xs font-medium transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      <div className="flex-1 overflow-y-auto">
        {chats.length === 0 ? (
          <div className="p-4 text-center text-[var(--text-secondary)] text-sm">
            <p>No chats yet</p>
            <p className="text-xs mt-1">Create a new chat to start</p>
          </div>
        ) : (
          <div className="divide-y divide-[var(--border)]">
            {chats.map((chat) => (
              <div
                key={chat.id}
                onClick={() => onSelectChat(chat)}
                className={`group px-3 py-3 cursor-pointer transition-colors ${
                  currentChat?.id === chat.id
                    ? 'bg-[var(--accent)]/10 border-l-2 border-l-[var(--accent)]'
                    : 'hover:bg-[var(--bg-primary)] border-l-2 border-l-transparent'
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <p className={`text-sm font-medium truncate ${
                      currentChat?.id === chat.id
                        ? 'text-[var(--accent)]'
                        : 'text-[var(--text-primary)]'
                    }`}>
                      {chat.title}
                    </p>
                    <p className="text-xs text-[var(--text-secondary)] mt-0.5">
                      {formatDate(chat.updated_at)} • {chat.messages.length} messages
                    </p>
                  </div>
                  <button
                    onClick={(e) => handleDeleteChat(e, chat.id)}
                    className="opacity-0 group-hover:opacity-100 text-[var(--text-secondary)] hover:text-[var(--danger)] transition-all p-1"
                    title="Delete chat"
                  >
                    &#x2715;
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
