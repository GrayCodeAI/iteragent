package iteragent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// OpenAICompatConfig configures an OpenAI-compatible provider.
type OpenAICompatConfig struct {
	BaseURL string
	Model   string
	APIKey  string
}

type openaiCompatProvider struct {
	cfg    OpenAICompatConfig
	client *http.Client
}

// NewOpenAICompat returns an OpenAI-compatible provider.
func NewOpenAICompat(cfg OpenAICompatConfig) Provider {
	return &openaiCompatProvider{
		cfg:    cfg,
		client: &http.Client{Timeout: 120 * time.Second},
	}
}

func (p *openaiCompatProvider) Name() string {
	return fmt.Sprintf("openai-compat(%s)", p.cfg.Model)
}

type openaiResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// openaiReasoningEffort maps ThinkingLevel to reasoning_effort string for OpenAI.
func openaiReasoningEffort(level ThinkingLevel) string {
	switch level {
	case ThinkingLevelMinimal, ThinkingLevelLow:
		return "low"
	case ThinkingLevelMedium:
		return "medium"
	case ThinkingLevelHigh:
		return "high"
	default:
		return ""
	}
}

// supportsReasoningEffort returns true if the base URL is an OpenAI endpoint.
func (p *openaiCompatProvider) supportsReasoningEffort() bool {
	return strings.Contains(p.cfg.BaseURL, "openai.com")
}

func (p *openaiCompatProvider) Complete(ctx context.Context, messages []Message, opts ...CompletionOptions) (string, error) {
	var opt CompletionOptions
	if len(opts) > 0 {
		opt = opts[0]
	}

	reqMap := map[string]interface{}{
		"model":    p.cfg.Model,
		"messages": messages,
		"stream":   false,
	}

	if opt.MaxTokens > 0 {
		reqMap["max_tokens"] = opt.MaxTokens
	}
	if opt.Temperature > 0 {
		reqMap["temperature"] = opt.Temperature
	}

	// Add reasoning_effort only for OpenAI endpoints that support it.
	if p.supportsReasoningEffort() && opt.ThinkingLevel != ThinkingLevelOff && opt.ThinkingLevel != "" {
		effort := openaiReasoningEffort(opt.ThinkingLevel)
		if effort != "" {
			reqMap["reasoning_effort"] = effort
		}
	}

	body, err := json.Marshal(reqMap)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", p.cfg.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.cfg.APIKey)

	resp, err := p.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read response: %w", err)
	}

	var result openaiResponse
	if err := json.Unmarshal(raw, &result); err != nil {
		return "", fmt.Errorf("unmarshal response: %w", err)
	}
	if result.Error != nil {
		return "", fmt.Errorf("openai error: %s", result.Error.Message)
	}
	if len(result.Choices) == 0 {
		return "", fmt.Errorf("empty response from openai")
	}

	return result.Choices[0].Message.Content, nil
}
