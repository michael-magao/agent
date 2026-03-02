package open_claw

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// Config mirrors OpenClaw's openclaw.json structure
type Config struct {
	Agent     *AgentConfig     `json:"agent,omitempty"`
	Gateway   *GatewayConfig   `json:"gateway,omitempty"`
	Channels  map[string]any   `json:"channels,omitempty"`
	Workspace string           `json:"workspace,omitempty"`
}

type AgentConfig struct {
	Model string `json:"model,omitempty"`
}

type GatewayConfig struct {
	Port int    `json:"port,omitempty"`
	Bind string `json:"bind,omitempty"`
}

// DefaultConfig returns default OpenClaw config
func DefaultConfig() *Config {
	return &Config{
		Agent: &AgentConfig{
			Model: "anthropic/claude-opus-4-6",
		},
		Gateway: &GatewayConfig{
			Port: 18789,
			Bind: "127.0.0.1",
		},
		Workspace: "~/.openclaw/workspace",
	}
}

// OpenClawHome returns ~/.openclaw or OPENCLAW_STATE_DIR
func OpenClawHome() string {
	if dir := os.Getenv("OPENCLAW_STATE_DIR"); dir != "" {
		return dir
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".openclaw")
}

// ConfigPath returns path to openclaw.json
func ConfigPath() string {
	if p := os.Getenv("OPENCLAW_CONFIG_PATH"); p != "" {
		return p
	}
	return filepath.Join(OpenClawHome(), "openclaw.json")
}

// LoadConfig loads config from ~/.openclaw/openclaw.json
func LoadConfig() (*Config, error) {
	cfg := DefaultConfig()
	path := ConfigPath()
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return cfg, nil
		}
		return nil, err
	}
	if err := json.Unmarshal(data, cfg); err != nil {
		return nil, err
	}
	if cfg.Agent == nil {
		cfg.Agent = DefaultConfig().Agent
	}
	if cfg.Gateway == nil {
		cfg.Gateway = DefaultConfig().Gateway
	}
	return cfg, nil
}
