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

type geminiRequest struct {
	Contents []struct {
		Parts []struct {
			Text string `json:"text"`
		} `json:"parts"`
	} `json:"contents"`
	SystemInstruction *struct {
		Parts []struct {
			Text string `json:"text"`
		} `json:"parts"`
	} `json:"systemInstruction,omitempty"`
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

func (p *geminiProvider) Complete(ctx context.Context, messages []Message) (string, error) {
	var system string
	var contents []string
	for _, m := range messages {
		if m.Role == "system" {
			system = m.Content
		} else {
			contents = append(contents, m.Content)
		}
	}

	reqBody := geminiRequest{}
	if system != "" {
		reqBody.SystemInstruction = &struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		}{}
		reqBody.SystemInstruction.Parts = []struct {
			Text string `json:"text"`
		}{{Text: system}}
	}
	for _, c := range contents {
		reqBody.Contents = append(reqBody.Contents, struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		}{Parts: []struct {
			Text string `json:"text"`
		}{{Text: c}}})
	}

	body, err := json.Marshal(reqBody)
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
