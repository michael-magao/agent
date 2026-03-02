package open_claw

// Frame types per OpenClaw protocol
const (
	FrameRequest  = "req"
	FrameResponse = "res"
	FrameEvent    = "event"
)

// Protocol version
const ProtocolVersion = 3

// Request frame (client -> gateway)
type Request struct {
	Type   string `json:"type"` // "req"
	ID     string `json:"id"`
	Method string `json:"method"`
	Params any    `json:"params,omitempty"`
}

// Response frame (gateway -> client)
type Response struct {
	Type    string `json:"type"` // "res"
	ID      string `json:"id"`
	OK      bool   `json:"ok"`
	Payload any    `json:"payload,omitempty"`
	Error   *Error `json:"error,omitempty"`
}

type Error struct {
	Code       string `json:"code"`
	Message    string `json:"message"`
	Details    any    `json:"details,omitempty"`
	Retryable  bool   `json:"retryable,omitempty"`
	RetryAfter int    `json:"retryAfterMs,omitempty"`
}

// Event frame (gateway -> client)
type Event struct {
	Type        string `json:"type"` // "event"
	Event       string `json:"event"`
	Payload     any    `json:"payload,omitempty"`
	Seq         int64  `json:"seq,omitempty"`
	StateVersion any   `json:"stateVersion,omitempty"`
}

// HelloOk snapshot sent after connect
type HelloOk struct {
	Type     string   `json:"type"` // "hello-ok"
	Protocol int      `json:"protocol"`
	Server   any      `json:"server"`
	Snapshot *Snapshot `json:"snapshot"`
	Policy   any      `json:"policy,omitempty"`
	Auth     any      `json:"auth,omitempty"`
}

type Snapshot struct {
	Presence   []any `json:"presence"`
	Health     any   `json:"health"`
	StateVersion any `json:"stateVersion"`
	UptimeMs   int64 `json:"uptimeMs"`
}

// Connect params (first request)
type ConnectParams struct {
	MinProtocol int    `json:"minProtocol"`
	MaxProtocol int    `json:"maxProtocol"`
	Client      *ClientInfo `json:"client"`
	Role        string `json:"role"`
	Scopes      []string `json:"scopes,omitempty"`
}

type ClientInfo struct {
	ID       string `json:"id"`
	Version  string `json:"version"`
	Platform string `json:"platform"`
	Mode     string `json:"mode"`
}
