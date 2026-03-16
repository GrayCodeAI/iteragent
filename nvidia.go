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

type nvidiaProvider struct {
	cfg    OpenAICompatConfig
	client *http.Client
}

func NewNvidia(cfg OpenAICompatConfig) Provider {
	return &nvidiaProvider{
		cfg:    cfg,
		client: &http.Client{Timeout: 120 * time.Second},
	}
}

func (p *nvidiaProvider) Name() string {
	return fmt.Sprintf("nvidia(%s)", p.cfg.Model)
}

func (p *nvidiaProvider) Complete(ctx context.Context, messages []Message) (string, error) {
	url := p.cfg.BaseURL + "/chat/completions"
	if p.cfg.BaseURL == "" {
		url = "https://integrate.api.nvidia.com/v1/chat/completions"
	}

	reqBody := openaiRequest{
		Model:    p.cfg.Model,
		Messages: messages,
		Stream:   false,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.cfg.APIKey)

	resp, err := p.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("do request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != 200 {
		return "", fmt.Errorf("nvidia API error (%d): %s", resp.StatusCode, string(respBody))
	}

	var parsed openaiResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return "", fmt.Errorf("unmarshal response: %w", err)
	}

	if len(parsed.Choices) == 0 {
		return "", fmt.Errorf("no response from nvidia")
	}

	return parsed.Choices[0].Message.Content, nil
}
