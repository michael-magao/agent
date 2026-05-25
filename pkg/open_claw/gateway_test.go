package open_claw

import (
	"net/http/httptest"
	"testing"
)

func TestGatewayCheckOriginAllowsLoopback(t *testing.T) {
	gateway := NewGateway(DefaultConfig(), NewSessionStore(), nil)
	req := httptest.NewRequest("GET", "http://127.0.0.1:18789", nil)
	req.Header.Set("Origin", "http://localhost:3000")

	if !gateway.checkOrigin(req) {
		t.Fatal("expected loopback origin to be allowed")
	}
}

func TestGatewayCheckOriginRejectsRemoteOrigin(t *testing.T) {
	gateway := NewGateway(DefaultConfig(), NewSessionStore(), nil)
	req := httptest.NewRequest("GET", "http://127.0.0.1:18789", nil)
	req.Header.Set("Origin", "https://example.com")

	if gateway.checkOrigin(req) {
		t.Fatal("expected remote origin to be rejected")
	}
}
