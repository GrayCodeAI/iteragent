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

type AzureOpenAIConfig struct {
	APIKey        string
	Endpoint      string
	Deployment    string
	APIVersion    string
	MaxTokens     int
	Temperature   float32
	ThinkingLevel ThinkingLevel
}

type AzureOpenAIProvider struct {
	config AzureOpenAIConfig
	client *http.Client
}

func NewAzureOpenAI(config AzureOpenAIConfig) *AzureOpenAIProvider {
	return &AzureOpenAIProvider{
		config: config,
		client: &http.Client{Timeout: 120 * time.Second},
	}
}

func (p *AzureOpenAIProvider) Name() string {
	return "azure_openai"
}

// TODO: Add ThinkingLevel support for Azure OpenAI when provider supports it.
func (p *AzureOpenAIProvider) Complete(ctx context.Context, messages []Message, opts ...CompletionOptions) (string, error) {
	apiVersion := p.config.APIVersion
	if apiVersion == "" {
		apiVersion = "2024-02-15-preview"
	}

	url := fmt.Sprintf("%s/openai/deployments/%s/chat/completions?api-version=%s",
		p.config.Endpoint, p.config.Deployment, apiVersion)

	body := map[string]interface{}{
		"messages": messagesToAzureFormat(messages),
		"stream":   false,
	}

	if p.config.MaxTokens > 0 {
		body["max_tokens"] = p.config.MaxTokens
	}
	if p.config.Temperature > 0 {
		body["temperature"] = p.config.Temperature
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonBody))
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("api-key", p.config.APIKey)

	resp, err := p.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("Azure OpenAI error (%d): %s", resp.StatusCode, string(respBody))
	}

	var response struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.Unmarshal(respBody, &response); err != nil {
		return "", fmt.Errorf("parse response: %w", err)
	}

	if len(response.Choices) == 0 {
		return "", fmt.Errorf("no response choices")
	}

	return response.Choices[0].Message.Content, nil
}

// CompleteStream implements Provider for Azure OpenAI using the SSE streaming endpoint.
func (p *AzureOpenAIProvider) CompleteStream(ctx context.Context, messages []Message, opt CompletionOptions, onToken func(string)) (string, error) {
	apiVersion := p.config.APIVersion
	if apiVersion == "" {
		apiVersion = "2024-02-15-preview"
	}
	url := fmt.Sprintf("%s/openai/deployments/%s/chat/completions?api-version=%s",
		p.config.Endpoint, p.config.Deployment, apiVersion)

	body := map[string]interface{}{
		"messages": messagesToAzureFormat(messages),
		"stream":   true,
	}
	if opt.MaxTokens > 0 {
		body["max_tokens"] = opt.MaxTokens
	}
	if opt.Temperature > 0 {
		body["temperature"] = opt.Temperature
	}
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	var full strings.Builder
	sseClient := NewSSEClient()
	err = sseClient.Stream(ctx, url, map[string]string{"api-key": p.config.APIKey}, jsonBody, func(e SSEEvent) {
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
		return "", fmt.Errorf("azure stream: %w", err)
	}
	return full.String(), nil
}

func messagesToAzureFormat(messages []Message) []map[string]interface{} {
	result := make([]map[string]interface{}, len(messages))
	for i, msg := range messages {
		m := map[string]interface{}{
			"role":    msg.Role,
			"content": msg.Content,
		}
		result[i] = m
	}
	return result
}

