package open_claw

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
)

// Agent processes messages and returns responses
type Agent interface {
	Run(sessionKey, message string) (string, error)
}


// LLMAgent calls external LLM API (OpenAI/Anthropic/Ollama compatible)
type LLMAgent struct {
	Model       string
	APIKey      string
	BaseURL     string // e.g. https://api.openai.com/v1 or http://localhost:11434
	UseOllama   bool   // use /api/generate for Ollama
}

func NewLLMAgent(cfg *Config) *LLMAgent {
	model := "gpt-4"
	if cfg != nil && cfg.Agent != nil && cfg.Agent.Model != "" {
		model = cfg.Agent.Model
	}
	return &LLMAgent{
		Model:     model,
		UseOllama: strings.Contains(model, "ollama") || strings.HasPrefix(model, "ollama/"),
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
	body, _ := json.Marshal(map[string]any{
		"model":  model,
		"prompt": message,
		"stream": false,
	})
	resp, err := http.Post(base+"/api/generate", "application/json", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
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
	body, _ := json.Marshal(reqBody)
	req, _ := http.NewRequest("POST", base+"/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
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
