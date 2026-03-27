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
// The returned provider implements both Provider and TokenStreamer.
func NewOpenAICompat(cfg OpenAICompatConfig) Provider {
	return &openaiCompatProvider{
		cfg:    cfg,
		client: &http.Client{Timeout: 120 * time.Second},
	}
}

func (p *openaiCompatProvider) Name() string {
	return fmt.Sprintf("openai-compat(%s)", p.cfg.Model)
}

// ContextWindow returns the context window for the configured model.
func (p *openaiCompatProvider) ContextWindow() int {
	return openaiCompatContextWindow(p.cfg.Model)
}

func openaiCompatContextWindow(model string) int {
	switch {
	case strings.HasPrefix(model, "gpt-3.5-turbo-instruct"):
		return 4_096
	case strings.HasPrefix(model, "gpt-3.5"):
		return 16_385
	case model == "gpt-4" || model == "gpt-4-0314":
		return 8_192
	case strings.HasPrefix(model, "gpt-4-32k"):
		return 32_768
	case strings.HasPrefix(model, "gpt-4-turbo"), strings.HasPrefix(model, "gpt-4o"),
		strings.HasPrefix(model, "o1"), strings.HasPrefix(model, "o3"):
		return 128_000
	case strings.HasPrefix(model, "llama-3"):
		return 128_000
	case strings.HasPrefix(model, "llama3"), strings.HasPrefix(model, "llama2"):
		return 8_192
	case strings.HasPrefix(model, "mistral"), strings.HasPrefix(model, "mixtral"):
		return 32_768
	case strings.HasPrefix(model, "deepseek"):
		return 128_000
	default:
		return 128_000
	}
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

// buildOpenAIBody constructs the JSON request body for OpenAI-compat completions.
func (p *openaiCompatProvider) buildOpenAIBody(messages []Message, opt CompletionOptions, stream bool) ([]byte, error) {
	model := p.cfg.Model
	if opt.Model != "" {
		model = opt.Model
	}
	reqMap := map[string]interface{}{
		"model":    model,
		"messages": messages,
		"stream":   stream,
	}
	if opt.MaxTokens > 0 {
		reqMap["max_tokens"] = opt.MaxTokens
	}
	if opt.Temperature > 0 {
		reqMap["temperature"] = opt.Temperature
	}
	if p.supportsReasoningEffort() && opt.ThinkingLevel != ThinkingLevelOff && opt.ThinkingLevel != "" {
		if effort := openaiReasoningEffort(opt.ThinkingLevel); effort != "" {
			reqMap["reasoning_effort"] = effort
		}
	}
	return json.Marshal(reqMap)
}

// CompleteStream implements TokenStreamer using OpenAI SSE format.
func (p *openaiCompatProvider) CompleteStream(ctx context.Context, messages []Message, opt CompletionOptions, onToken func(string)) (string, error) {
	body, err := p.buildOpenAIBody(messages, opt, true)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	headers := map[string]string{
		"Authorization": "Bearer " + p.cfg.APIKey,
	}

	var full strings.Builder
	sseClient := NewSSEClient()
	err = sseClient.Stream(ctx, p.cfg.BaseURL+"/chat/completions", headers, body, func(e SSEEvent) {
		if e.Data == "[DONE]" {
			return
		}
		if token, ok := ParseOpenAISSE(e.Data); ok && token != "" {
			full.WriteString(token)
			if onToken != nil {
				onToken(token)
			}
		}
	})
	if err != nil {
		return "", fmt.Errorf("openai stream: %w", err)
	}
	result := full.String()
	if result == "" {
		return "", fmt.Errorf("empty streaming response from openai-compat")
	}
	return result, nil
}

func (p *openaiCompatProvider) Complete(ctx context.Context, messages []Message, opts ...CompletionOptions) (string, error) {
	var opt CompletionOptions
	if len(opts) > 0 {
		opt = opts[0]
	}

	body, err := p.buildOpenAIBody(messages, opt, false)
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
