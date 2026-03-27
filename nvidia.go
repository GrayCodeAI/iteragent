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

func (p *nvidiaProvider) Complete(ctx context.Context, messages []Message, opts ...CompletionOptions) (string, error) {
	url := p.cfg.BaseURL
	if url == "" {
		url = "https://integrate.api.nvidia.com/v1/chat/completions"
	} else {
		url = url + "/chat/completions"
	}

	var nvidiaOpt CompletionOptions
	if len(opts) > 0 {
		nvidiaOpt = opts[0]
	}
	nvidiaModel := p.cfg.Model
	if nvidiaOpt.Model != "" {
		nvidiaModel = nvidiaOpt.Model
	}
	reqBody := map[string]interface{}{
		"model":    nvidiaModel,
		"messages": messages,
		"stream":   false,
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

// CompleteStream implements TokenStreamer for Nvidia using the OpenAI-compatible SSE endpoint.
func (p *nvidiaProvider) CompleteStream(ctx context.Context, messages []Message, opt CompletionOptions, onToken func(string)) (string, error) {
	url := p.cfg.BaseURL
	if url == "" {
		url = "https://integrate.api.nvidia.com/v1/chat/completions"
	} else {
		url = url + "/chat/completions"
	}

	nvidiaStreamModel := p.cfg.Model
	if opt.Model != "" {
		nvidiaStreamModel = opt.Model
	}
	reqBody := map[string]interface{}{
		"model":    nvidiaStreamModel,
		"messages": messages,
		"stream":   true,
	}
	if opt.MaxTokens > 0 {
		reqBody["max_tokens"] = opt.MaxTokens
	}
	if opt.Temperature > 0 {
		reqBody["temperature"] = opt.Temperature
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	var full strings.Builder
	sseClient := NewSSEClient()
	err = sseClient.Stream(ctx, url, map[string]string{"Authorization": "Bearer " + p.cfg.APIKey}, body, func(e SSEEvent) {
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
		return "", fmt.Errorf("nvidia stream: %w", err)
	}
	return full.String(), nil
}
