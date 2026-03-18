package iteragent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// AnthropicConfig configures the Anthropic provider.
type AnthropicConfig struct {
	Model  string
	APIKey string
}

type anthropicProvider struct {
	cfg    AnthropicConfig
	client *http.Client
}

// NewAnthropic returns a native Anthropic provider.
func NewAnthropic(cfg AnthropicConfig) Provider {
	return &anthropicProvider{
		cfg:    cfg,
		client: &http.Client{Timeout: 120 * time.Second},
	}
}

func (p *anthropicProvider) Name() string {
	return fmt.Sprintf("anthropic(%s)", p.cfg.Model)
}

type anthropicResponse struct {
	Content []struct {
		Text string `json:"text"`
	} `json:"content"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// thinkingBudget returns the token budget for the given thinking level (Anthropic).
func thinkingBudget(level ThinkingLevel) int {
	switch level {
	case ThinkingLevelMinimal:
		return 1024
	case ThinkingLevelLow:
		return 4096
	case ThinkingLevelMedium:
		return 8192
	case ThinkingLevelHigh:
		return 16000
	default:
		return 0
	}
}

func (p *anthropicProvider) Complete(ctx context.Context, messages []Message, opts ...CompletionOptions) (string, error) {
	var opt CompletionOptions
	if len(opts) > 0 {
		opt = opts[0]
	}

	var system string
	var filtered []Message
	for _, m := range messages {
		if m.Role == "system" {
			system = m.Content
		} else {
			filtered = append(filtered, m)
		}
	}

	maxTokens := 4096
	if opt.MaxTokens > 0 {
		maxTokens = opt.MaxTokens
	}

	reqMap := map[string]interface{}{
		"model":      p.cfg.Model,
		"max_tokens": maxTokens,
		"messages":   filtered,
	}

	// System prompt: use cache_control breakpoint when caching is enabled.
	if system != "" {
		cacheEnabled := opt.CacheConfig != nil && opt.CacheConfig.Enabled && opt.CacheConfig.CacheSystem
		if cacheEnabled {
			reqMap["system"] = []map[string]interface{}{
				{
					"type": "text",
					"text": system,
					"cache_control": map[string]string{
						"type": "ephemeral",
					},
				},
			}
		} else {
			reqMap["system"] = system
		}
	}

	// Add thinking config if enabled.
	if opt.ThinkingLevel != ThinkingLevelOff && opt.ThinkingLevel != "" {
		budget := thinkingBudget(opt.ThinkingLevel)
		reqMap["thinking"] = map[string]interface{}{
			"type":          "enabled",
			"budget_tokens": budget,
		}
	}

	body, err := json.Marshal(reqMap)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.anthropic.com/v1/messages", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", p.cfg.APIKey)
	req.Header.Set("anthropic-version", "2023-06-01")
	// Enable prompt caching beta feature.
	if opt.CacheConfig != nil && opt.CacheConfig.Enabled {
		req.Header.Set("anthropic-beta", "prompt-caching-2024-07-31")
	}

	resp, err := p.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read response: %w", err)
	}

	var result anthropicResponse
	if err := json.Unmarshal(raw, &result); err != nil {
		return "", fmt.Errorf("unmarshal response: %w", err)
	}
	if result.Error != nil {
		return "", fmt.Errorf("anthropic error: %s", result.Error.Message)
	}
	if len(result.Content) == 0 {
		return "", fmt.Errorf("empty response from anthropic")
	}

	return result.Content[0].Text, nil
}
