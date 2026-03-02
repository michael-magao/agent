package open_claw

// Channel represents a messaging channel (WhatsApp, Telegram, Slack, WebChat, etc.)
// Mirrors OpenClaw channel interface for future extensions
type Channel interface {
	// Name returns the channel identifier
	Name() string
	// Start connects and begins receiving messages
	Start(gw *Gateway) error
	// Stop disconnects the channel
	Stop() error
}

// InboundMessage represents a message from a channel
type InboundMessage struct {
	Channel   string
	Sender    string
	Content   string
	SessionKey string
}
