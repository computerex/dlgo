import type { Message } from '../api';

interface MessageBubbleProps {
  message: Message & { id: string };
  streaming?: boolean;
}

export function MessageBubble({ message, streaming }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';

  return (
    <div className={`max-w-3xl mx-auto ${isUser ? 'flex justify-end' : ''}`}>
      <div
        className={`
          rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap
          ${isUser
            ? 'bg-[var(--accent)] text-white max-w-[80%]'
            : isSystem
            ? 'bg-[var(--bg-tertiary)] text-[var(--text-secondary)] border border-[var(--border)] max-w-[90%]'
            : 'bg-[var(--bg-secondary)] text-[var(--text-primary)] max-w-[90%]'
          }
        `}
      >
        {!isUser && (
          <div className="text-xs text-[var(--text-secondary)] mb-1 font-medium uppercase tracking-wide">
            {isSystem ? 'System' : 'Assistant'}
          </div>
        )}
        <div>
          {message.content}
          {streaming && (
            <span className="inline-block w-2 h-4 bg-[var(--accent)] ml-0.5 animate-pulse rounded-sm" />
          )}
        </div>
      </div>
    </div>
  );
}
