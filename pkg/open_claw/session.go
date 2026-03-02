package open_claw

import (
	"sync"
	"time"

	"github.com/google/uuid"
)

// Session represents a conversation session (mirrors OpenClaw sessions)
type Session struct {
	Key       string            `json:"key"`
	Label     string            `json:"label,omitempty"`
	AgentID   string            `json:"agentId,omitempty"`
	CreatedAt time.Time         `json:"createdAt"`
	UpdatedAt time.Time         `json:"updatedAt"`
	Model     string            `json:"model,omitempty"`
	Messages  []*Message        `json:"messages,omitempty"`
	Meta      map[string]string `json:"meta,omitempty"`
	mu        sync.RWMutex
}

type Message struct {
	Role    string `json:"role"` // "human" | "assistant" | "system"
	Content string `json:"content"`
}

// NewSession creates a new session
func NewSession(label, agentID string) *Session {
	now := time.Now()
	return &Session{
		Key:       uuid.New().String(),
		Label:     label,
		AgentID:   agentID,
		CreatedAt: now,
		UpdatedAt: now,
		Messages:  []*Message{},
		Meta:      map[string]string{},
	}
}

func (s *Session) AddMessage(role, content string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Messages = append(s.Messages, &Message{Role: role, Content: content})
	s.UpdatedAt = time.Now()
}

func (s *Session) GetMessages() []*Message {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]*Message, len(s.Messages))
	copy(out, s.Messages)
	return out
}

// SessionStore manages sessions
type SessionStore struct {
	sessions map[string]*Session
	mu       sync.RWMutex
}

func NewSessionStore() *SessionStore {
	return &SessionStore{sessions: map[string]*Session{}}
}

func (ss *SessionStore) Get(key string) *Session {
	ss.mu.RLock()
	defer ss.mu.RUnlock()
	return ss.sessions[key]
}

func (ss *SessionStore) Put(s *Session) {
	ss.mu.Lock()
	defer ss.mu.Unlock()
	ss.sessions[s.Key] = s
}

func (ss *SessionStore) List(limit int) []*Session {
	ss.mu.RLock()
	defer ss.mu.RUnlock()
	var out []*Session
	for _, s := range ss.sessions {
		out = append(out, s)
	}
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	return out
}

func (ss *SessionStore) Delete(key string) {
	ss.mu.Lock()
	defer ss.mu.Unlock()
	delete(ss.sessions, key)
}
