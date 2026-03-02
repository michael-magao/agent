package open_claw

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true // allow all origins for local dev
	},
}

// Gateway implements the OpenClaw WebSocket control plane
type Gateway struct {
	Config      *Config
	Store       *SessionStore
	Agent       Agent
	startTime   time.Time
	clients     map[*websocket.Conn]bool
	clientsMu   sync.RWMutex
}

func NewGateway(cfg *Config, store *SessionStore, agent Agent) *Gateway {
	return &Gateway{
		Config:    cfg,
		Store:     store,
		Agent:     agent,
		startTime: time.Now(),
		clients:   map[*websocket.Conn]bool{},
	}
}

func (g *Gateway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("[gateway] upgrade: %v", err)
		return
	}
	g.clientsMu.Lock()
	g.clients[conn] = true
	g.clientsMu.Unlock()
	defer func() {
		g.clientsMu.Lock()
		delete(g.clients, conn)
		g.clientsMu.Unlock()
		conn.Close()
	}()

	for {
		_, data, err := conn.ReadMessage()
		if err != nil {
			break
		}
		var req Request
		if err := json.Unmarshal(data, &req); err != nil {
			g.sendError(conn, "", "invalid_json", err.Error())
			continue
		}
		if req.Type != FrameRequest {
			continue
		}
		g.handleRequest(conn, &req)
	}
}

func (g *Gateway) handleRequest(conn *websocket.Conn, req *Request) {
	id := req.ID
	if id == "" {
		id = uuid.New().String()
	}
	switch req.Method {
	case "connect":
		g.sendHelloOk(conn, id)
	case "sessions.list":
		g.handleSessionsList(conn, id, req.Params)
	case "sessions.resolve":
		g.handleSessionsResolve(conn, id, req.Params)
	case "agent":
		g.handleAgent(conn, id, req.Params)
	case "send":
		g.handleSend(conn, id, req.Params)
	case "health", "status":
		g.handleHealth(conn, id)
	default:
		g.sendError(conn, id, "method_not_found", "method "+req.Method+" not implemented")
	}
}

func (g *Gateway) sendHelloOk(conn *websocket.Conn, reqID string) {
	uptime := time.Since(g.startTime).Milliseconds()
	snap := &HelloOk{
		Type:     "hello-ok",
		Protocol: ProtocolVersion,
		Server: map[string]any{
			"version": "openclaw-go/0.1",
			"connId":  uuid.New().String(),
		},
		Snapshot: &Snapshot{
			Presence:     []any{},
			Health:       map[string]any{"status": "ok"},
			StateVersion: map[string]int{"presence": 1, "health": 1},
			UptimeMs:     uptime,
		},
	}
	g.sendResponse(conn, reqID, true, snap)
}

func (g *Gateway) handleSessionsList(conn *websocket.Conn, reqID string, params any) {
	sessions := g.Store.List(50)
	out := make([]map[string]any, 0, len(sessions))
	for _, s := range sessions {
		out = append(out, map[string]any{
			"key":       s.Key,
			"label":     s.Label,
			"agentId":   s.AgentID,
			"createdAt": s.CreatedAt,
			"updatedAt": s.UpdatedAt,
		})
	}
	g.sendResponse(conn, reqID, true, map[string]any{"sessions": out})
}

func (g *Gateway) handleSessionsResolve(conn *websocket.Conn, reqID string, params any) {
	key := ""
	if m, ok := params.(map[string]any); ok {
		if k, ok := m["key"].(string); ok {
			key = k
		}
	}
	if key == "" {
		s := g.Store.List(1)
		if len(s) > 0 {
			key = s[0].Key
		}
	}
	if key == "" {
		sess := NewSession("main", "default")
		g.Store.Put(sess)
		key = sess.Key
	}
	sess := g.Store.Get(key)
	if sess == nil {
		sess = NewSession("main", "default")
		g.Store.Put(sess)
	}
	g.sendResponse(conn, reqID, true, map[string]any{
		"session": map[string]any{
			"key":     sess.Key,
			"label":   sess.Label,
			"agentId": sess.AgentID,
		},
	})
}

func (g *Gateway) handleAgent(conn *websocket.Conn, reqID string, params any) {
	message := ""
	sessionKey := ""
	if m, ok := params.(map[string]any); ok {
		if msg, ok := m["message"].(string); ok {
			message = msg
		}
		if key, ok := m["sessionKey"].(string); ok {
			sessionKey = key
		}
	}
	if message == "" {
		g.sendError(conn, reqID, "bad_request", "message required")
		return
	}
	sess := g.Store.Get(sessionKey)
	if sess == nil {
		sess = NewSession("main", "default")
		g.Store.Put(sess)
		sessionKey = sess.Key
	}
	sess.AddMessage("human", message)
	reply, err := g.Agent.Run(sessionKey, message)
	if err != nil {
		g.sendError(conn, reqID, "agent_error", err.Error())
		return
	}
	sess.AddMessage("assistant", reply)
	g.sendResponse(conn, reqID, true, map[string]any{
		"reply":      reply,
		"sessionKey": sessionKey,
	})
}

func (g *Gateway) handleSend(conn *websocket.Conn, reqID string, params any) {
	// send message to a channel - simplified, just acknowledge
	g.sendResponse(conn, reqID, true, map[string]any{"sent": true})
}

func (g *Gateway) handleHealth(conn *websocket.Conn, reqID string) {
	g.sendResponse(conn, reqID, true, map[string]any{
		"status": "ok",
		"uptime": time.Since(g.startTime).Seconds(),
	})
}

func (g *Gateway) sendResponse(conn *websocket.Conn, id string, ok bool, payload any) {
	res := Response{Type: FrameResponse, ID: id, OK: ok, Payload: payload}
	data, _ := json.Marshal(res)
	if err := conn.WriteMessage(websocket.TextMessage, data); err != nil {
		log.Printf("[gateway] write: %v", err)
	}
}

func (g *Gateway) sendError(conn *websocket.Conn, id, code, msg string) {
	res := Response{
		Type: FrameResponse,
		ID:   id,
		OK:   false,
		Error: &Error{Code: code, Message: msg},
	}
	data, _ := json.Marshal(res)
	conn.WriteMessage(websocket.TextMessage, data)
}

// Run starts the HTTP + WebSocket server
func (g *Gateway) Run() error {
	port := 18789
	if g.Config != nil && g.Config.Gateway != nil && g.Config.Gateway.Port > 0 {
		port = g.Config.Gateway.Port
	}
	addr := fmt.Sprintf("127.0.0.1:%d", port)
	mux := http.NewServeMux()
	mux.Handle("/", g)
	log.Printf("[gateway] listening on ws://%s", addr)
	return http.ListenAndServe(addr, mux)
}
