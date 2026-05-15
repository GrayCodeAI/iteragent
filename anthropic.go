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
// The returned provider implements Provider with streaming support.
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

// buildAnthropicBody constructs the JSON request body for Anthropic completions.
func (p *anthropicProvider) buildAnthropicBody(messages []Message, opt CompletionOptions, stream bool) ([]byte, error) {
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
	if stream {
		reqMap["stream"] = true
	}
	cacheEnabled := opt.CacheConfig != nil && opt.CacheConfig.Enabled && opt.CacheConfig.CacheSystem
	if system != "" {
		if cacheEnabled {
			reqMap["system"] = []map[string]interface{}{
				{"type": "text", "text": system, "cache_control": map[string]string{"type": "ephemeral"}},
			}
		} else {
			reqMap["system"] = system
		}
	}
	if opt.ThinkingLevel != ThinkingLevelOff && opt.ThinkingLevel != "" {
		budget := thinkingBudget(opt.ThinkingLevel)
		reqMap["thinking"] = map[string]interface{}{
			"type": "enabled", "budget_tokens": budget,
		}
	}
	return json.Marshal(reqMap)
}

// CompleteStream implements Provider. It uses SSE to deliver tokens
// incrementally via onToken as they arrive from the Anthropic API.
func (p *anthropicProvider) CompleteStream(ctx context.Context, messages []Message, opt CompletionOptions, onToken func(string)) (string, error) {
	body, err := p.buildAnthropicBody(messages, opt, true)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	headers := map[string]string{
		"x-api-key":         p.cfg.APIKey,
		"anthropic-version": "2023-06-01",
	}
	if opt.CacheConfig != nil && opt.CacheConfig.Enabled {
		headers["anthropic-beta"] = "prompt-caching-2024-07-31"
	}

	var full strings.Builder
	sseClient := NewSSEClient()
	err = sseClient.Stream(ctx, "https://api.anthropic.com/v1/messages", headers, body, func(e SSEEvent) {
		if token, ok := ParseAnthropicSSE(e.Data); ok && token != "" {
			full.WriteString(token)
			if onToken != nil {
				onToken(token)
			}
		}
	})
	if err != nil {
		return "", fmt.Errorf("anthropic stream: %w", err)
	}
	result := full.String()
	if result == "" {
		return "", fmt.Errorf("empty streaming response from anthropic")
	}
	return result, nil
}

func (p *anthropicProvider) Complete(ctx context.Context, messages []Message, opts ...CompletionOptions) (string, error) {
	var opt CompletionOptions
	if len(opts) > 0 {
		opt = opts[0]
	}

	body, err := p.buildAnthropicBody(messages, opt, false)
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
