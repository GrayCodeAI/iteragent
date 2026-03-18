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

func (p *geminiProvider) Complete(ctx context.Context, messages []Message, opts ...CompletionOptions) (string, error) {
	var opt CompletionOptions
	if len(opts) > 0 {
		opt = opts[0]
	}

	var system string
	var contents []string
	for _, m := range messages {
		if m.Role == "system" {
			system = m.Content
		} else {
			contents = append(contents, m.Content)
		}
	}

	// Build request as a generic map to support optional fields.
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

	// Add generationConfig including thinkingConfig if enabled.
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

	body, err := json.Marshal(reqMap)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent?key=%s", p.cfg.Model, p.cfg.APIKey)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

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
