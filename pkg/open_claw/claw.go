// Package open_claw provides a Go implementation of OpenClaw — a personal AI
// assistant framework. It mirrors the core functionality of the open-source
// OpenClaw project (https://github.com/openclaw/openclaw):
//
//   - Gateway: WebSocket control plane (ws://127.0.0.1:18789)
//   - Agent: LLM-backed agent with session management
//   - CLI: gateway, agent, onboard, doctor, message send
//   - Config: ~/.openclaw/openclaw.json
//
// Run: go run ./cmd/openclaw gateway
//      go run ./cmd/openclaw agent --message "Hello"
package open_claw
