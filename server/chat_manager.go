package server

import (
	"sync"
	"time"
)

// ChatMessage represents a single message in a chat
type ChatMessage struct {
	ID        string    `json:"id"`
	Role      string    `json:"role"`
	Content   string    `json:"content"`
	CreatedAt time.Time `json:"created_at"`
}

// ChatSession represents a complete chat conversation
type ChatSession struct {
	ID        string        `json:"id"`
	Title     string        `json:"title"`
	Model     string        `json:"model"`
	Messages  []ChatMessage `json:"messages"`
	CreatedAt time.Time     `json:"created_at"`
	UpdatedAt time.Time     `json:"updated_at"`
}

// ChatManager manages multiple chat sessions
type ChatManager struct {
	mu    sync.RWMutex
	chats map[string]*ChatSession
}

// NewChatManager creates a new chat manager
func NewChatManager() *ChatManager {
	return &ChatManager{
		chats: make(map[string]*ChatSession),
	}
}

// CreateChat creates a new chat session
func (cm *ChatManager) CreateChat(title, model string) *ChatSession {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	now := time.Now()
	chat := &ChatSession{
		ID:        generateChatID(),
		Title:     title,
		Model:     model,
		Messages:  []ChatMessage{},
		CreatedAt: now,
		UpdatedAt: now,
	}
	cm.chats[chat.ID] = chat
	return chat
}

// GetChat retrieves a chat by ID
func (cm *ChatManager) GetChat(id string) *ChatSession {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	return cm.chats[id]
}

// ListChats returns all chats sorted by updated time (most recent first)
func (cm *ChatManager) ListChats() []*ChatSession {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	chats := make([]*ChatSession, 0, len(cm.chats))
	for _, chat := range cm.chats {
		chats = append(chats, chat)
	}

	// Sort by updated time descending
	for i := 0; i < len(chats)-1; i++ {
		for j := i + 1; j < len(chats); j++ {
			if chats[i].UpdatedAt.Before(chats[j].UpdatedAt) {
				chats[i], chats[j] = chats[j], chats[i]
			}
		}
	}

	return chats
}

// DeleteChat removes a chat session
func (cm *ChatManager) DeleteChat(id string) bool {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	if _, exists := cm.chats[id]; exists {
		delete(cm.chats, id)
		return true
	}
	return false
}

// AddMessage adds a message to a chat
func (cm *ChatManager) AddMessage(chatID string, role, content string) *ChatMessage {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	chat, exists := cm.chats[chatID]
	if !exists {
		return nil
	}

	msg := ChatMessage{
		ID:        generateMessageID(),
		Role:      role,
		Content:   content,
		CreatedAt: time.Now(),
	}

	chat.Messages = append(chat.Messages, msg)
	chat.UpdatedAt = time.Now()

	// Update title if first user message
	if role == "user" && len(chat.Messages) <= 2 && chat.Title == "New Chat" {
		// Truncate first message to use as title
		title := content
		if len(title) > 30 {
			title = title[:27] + "..."
		}
		chat.Title = title
	}

	return &msg
}

// UpdateChatTitle updates a chat's title
func (cm *ChatManager) UpdateChatTitle(id, title string) bool {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	chat, exists := cm.chats[id]
	if !exists {
		return false
	}

	chat.Title = title
	chat.UpdatedAt = time.Now()
	return true
}

// generateChatID generates a unique chat ID
func generateChatID() string {
	return "chat-" + randomHex(12)
}

// generateMessageID generates a unique message ID
func generateMessageID() string {
	return "msg-" + randomHex(12)
}
