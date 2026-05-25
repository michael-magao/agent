package open_claw

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// Agent processes messages and returns responses
type Agent interface {
	Run(sessionKey, message string) (string, error)
}

// LLMAgent calls external LLM API (OpenAI/Anthropic/Ollama compatible)
type LLMAgent struct {
	Model     string
	APIKey    string
	BaseURL   string // e.g. https://api.openai.com/v1 or http://localhost:11434
	UseOllama bool   // use /api/generate for Ollama
	Client    *http.Client
	Timeout   time.Duration
}

func NewLLMAgent(cfg *Config) *LLMAgent {
	model := "gpt-4"
	if cfg != nil && cfg.Agent != nil && cfg.Agent.Model != "" {
		model = cfg.Agent.Model
	}
	return &LLMAgent{
		Model:     model,
		UseOllama: strings.Contains(model, "ollama") || strings.HasPrefix(model, "ollama/"),
		Timeout:   60 * time.Second,
	}
}

func (a *LLMAgent) Run(sessionKey, message string) (string, error) {
	base := a.BaseURL
	if base == "" {
		base = os.Getenv("OPENAI_BASE_URL")
	}
	if a.UseOllama || (base != "" && strings.Contains(base, "11434")) {
		return a.runOllama(message)
	}
	return a.runOpenAICompatible(message)
}

func (a *LLMAgent) runOllama(message string) (string, error) {
	base := a.BaseURL
	if base == "" {
		base = "http://localhost:11434"
	}
	model := a.Model
	if strings.HasPrefix(model, "ollama/") {
		model = strings.TrimPrefix(model, "ollama/")
	}
	if model == "" {
		model = "llama2"
	}
	body, err := json.Marshal(map[string]any{
		"model":  model,
		"prompt": message,
		"stream": false,
	})
	if err != nil {
		return "", err
	}
	ctx, cancel := context.WithTimeout(context.Background(), a.timeout())
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/api/generate", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := a.httpClient().Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("ollama api returned %s: %s", resp.Status, readBodySnippet(resp.Body))
	}
	var out struct {
		Response string `json:"response"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}
	return strings.TrimSpace(out.Response), nil
}

func (a *LLMAgent) runOpenAICompatible(message string) (string, error) {
	base := a.BaseURL
	if base == "" {
		base = os.Getenv("OPENAI_BASE_URL")
	}
	if base == "" {
		base = "https://api.openai.com/v1"
	}
	model := a.Model
	if model == "" {
		model = "gpt-4"
	}
	apiKey := a.APIKey
	if apiKey == "" {
		apiKey = getEnv("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY")
	}
	if apiKey == "" {
		return "", fmt.Errorf("no API key set (OPENAI_API_KEY or ANTHROPIC_API_KEY)")
	}
	reqBody := map[string]any{
		"model": model,
		"messages": []map[string]string{
			{"role": "user", "content": message},
		},
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}
	ctx, cancel := context.WithTimeout(context.Background(), a.timeout())
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)
	resp, err := a.httpClient().Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("llm api returned %s: %s", resp.Status, readBodySnippet(resp.Body))
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}
	if len(out.Choices) == 0 {
		return "", fmt.Errorf("no response from model")
	}
	return strings.TrimSpace(out.Choices[0].Message.Content), nil
}

func (a *LLMAgent) httpClient() *http.Client {
	if a.Client != nil {
		return a.Client
	}
	return &http.Client{Timeout: a.timeout()}
}

func (a *LLMAgent) timeout() time.Duration {
	if a.Timeout > 0 {
		return a.Timeout
	}
	return 60 * time.Second
}

func readBodySnippet(r io.Reader) string {
	data, err := io.ReadAll(io.LimitReader(r, 4096))
	if err != nil {
		return err.Error()
	}
	return strings.TrimSpace(string(data))
}

func getEnv(keys ...string) string {
	for _, k := range keys {
		if v := getEnvOne(k); v != "" {
			return v
		}
	}
	return ""
}

func getEnvOne(k string) string {
	return os.Getenv(k)
}
