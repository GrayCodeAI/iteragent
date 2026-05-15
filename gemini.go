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

// GeminiConfig configures the Gemini provider.
type GeminiConfig struct {
	Model  string
	APIKey string
}

type geminiProvider struct {
	cfg    GeminiConfig
	client *http.Client
}

// NewGemini returns a Google Gemini provider.
// The returned provider implements both Provider and TokenStreamer.
func NewGemini(cfg GeminiConfig) Provider {
	return &geminiProvider{
		cfg:    cfg,
		client: &http.Client{Timeout: 120 * time.Second},
	}
}

func (p *geminiProvider) Name() string {
	return fmt.Sprintf("gemini(%s)", p.cfg.Model)
}

type geminiResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// geminiThinkingBudget returns the thinking budget for the given level (Gemini).
func geminiThinkingBudget(level ThinkingLevel) int {
	switch level {
	case ThinkingLevelMinimal:
		return 512
	case ThinkingLevelLow:
		return 2048
	case ThinkingLevelMedium:
		return 4096
	case ThinkingLevelHigh:
		return 8192
	default:
		return 0
	}
}

// buildGeminiBody constructs the JSON request body for Gemini completions.
func (p *geminiProvider) buildGeminiBody(messages []Message, opt CompletionOptions) ([]byte, error) {
	var system string
	var contents []string
	for _, m := range messages {
		if m.Role == "system" {
			system = m.Content
		} else {
			contents = append(contents, m.Content)
		}
	}

	reqMap := map[string]interface{}{}
	if system != "" {
		reqMap["systemInstruction"] = map[string]interface{}{
			"parts": []map[string]string{{"text": system}},
		}
	}
	var contentsSlice []map[string]interface{}
	for _, c := range contents {
		contentsSlice = append(contentsSlice, map[string]interface{}{
			"parts": []map[string]string{{"text": c}},
		})
	}
	reqMap["contents"] = contentsSlice

	genConfig := map[string]interface{}{}
	if opt.MaxTokens > 0 {
		genConfig["maxOutputTokens"] = opt.MaxTokens
	}
	if opt.Temperature > 0 {
		genConfig["temperature"] = opt.Temperature
	}
	if opt.ThinkingLevel != ThinkingLevelOff && opt.ThinkingLevel != "" {
		budget := geminiThinkingBudget(opt.ThinkingLevel)
		genConfig["thinkingConfig"] = map[string]interface{}{
			"thinkingBudget": budget,
		}
	}
	if len(genConfig) > 0 {
		reqMap["generationConfig"] = genConfig
	}
	return json.Marshal(reqMap)
}

// CompleteStream implements TokenStreamer using the Gemini streaming endpoint.
func (p *geminiProvider) CompleteStream(ctx context.Context, messages []Message, opt CompletionOptions, onToken func(string)) (string, error) {
	body, err := p.buildGeminiBody(messages, opt)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	streamURL := fmt.Sprintf(
		"https://generativelanguage.googleapis.com/v1beta/models/%s:streamGenerateContent?alt=sse",
		p.cfg.Model)

	var full strings.Builder
	sseClient := NewSSEClient()
	headers := map[string]string{"X-Goog-Api-Key": p.cfg.APIKey}
	err = sseClient.Stream(ctx, streamURL, headers, body, func(e SSEEvent) {
		if token, ok := ParseGeminiSSE(e.Data); ok && token != "" {
			full.WriteString(token)
			if onToken != nil {
				onToken(token)
			}
		}
	})
	if err != nil {
		return "", fmt.Errorf("gemini stream: %w", err)
	}
	result := full.String()
	if result == "" {
		return "", fmt.Errorf("empty streaming response from gemini")
	}
	return result, nil
}

func (p *geminiProvider) Complete(ctx context.Context, messages []Message, opts ...CompletionOptions) (string, error) {
	var opt CompletionOptions
	if len(opts) > 0 {
		opt = opts[0]
	}

	body, err := p.buildGeminiBody(messages, opt)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent", p.cfg.Model)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Goog-Api-Key", p.cfg.APIKey)

	resp, err := p.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read response: %w", err)
	}

	var result geminiResponse
	if err := json.Unmarshal(raw, &result); err != nil {
		return "", fmt.Errorf("unmarshal response: %w", err)
	}
	if result.Error != nil {
		return "", fmt.Errorf("gemini error: %s", result.Error.Message)
	}
	if len(result.Candidates) == 0 {
		return "", fmt.Errorf("empty response from gemini")
	}

	var sb strings.Builder
	for _, part := range result.Candidates[0].Content.Parts {
		sb.WriteString(part.Text)
	}
	return sb.String(), nil
}
