package open_claw

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestLLMAgentReturnsHTTPStatusErrors(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "bad upstream", http.StatusBadGateway)
	}))
	defer server.Close()

	agent := &LLMAgent{
		Model:   "test-model",
		APIKey:  "test-key",
		BaseURL: server.URL,
		Client:  server.Client(),
		Timeout: time.Second,
	}

	_, err := agent.runOpenAICompatible("hello")
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "502 Bad Gateway") {
		t.Fatalf("expected status in error, got %v", err)
	}
}
