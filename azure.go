package iteragent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
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
		client: &http.Client{},
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

// CompleteStream implements TokenStreamer for Azure OpenAI using the SSE streaming endpoint.
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

func (p *AzureOpenAIProvider) Stream(ctx context.Context, config StreamConfig, messages []Message, onEvent func(StreamEvent)) (Message, error) {
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

	if config.MaxTokens > 0 {
		body["max_tokens"] = config.MaxTokens
	}
	if config.Temperature > 0 {
		body["temperature"] = config.Temperature
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return Message{}, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonBody))
	if err != nil {
		return Message{}, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("api-key", p.config.APIKey)

	resp, err := p.client.Do(req)
	if err != nil {
		return Message{}, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body) // best-effort for error message
		return Message{}, fmt.Errorf("Azure OpenAI error (%d): %s", resp.StatusCode, string(respBody))
	}

	var content strings.Builder
	decoder := NewSSEDecoder(resp.Body)

	for {
		event, err := decoder.Decode()
		if err == io.EOF {
			break
		}
		if err != nil {
			break
		}

		if event.Type == "content" || event.Type == "content_block" {
			content.WriteString(event.Content)
			onEvent(event)
		}
	}

	return Message{
		Role:    "assistant",
		Content: content.String(),
	}, nil
}

type SSEDecoder struct {
	reader io.Reader
}

func NewSSEDecoder(reader io.Reader) *SSEDecoder {
	return &SSEDecoder{reader: reader}
}

func (d *SSEDecoder) Decode() (StreamEvent, error) {
	var line string
	for {
		buf := make([]byte, 1024)
		n, err := d.reader.Read(buf)
		if n == 0 || err != nil {
			return StreamEvent{}, err
		}
		line = string(buf[:n])
		if strings.HasPrefix(line, "data:") {
			break
		}
	}

	line = strings.TrimSpace(strings.TrimPrefix(line, "data:"))
	if line == "[DONE]" {
		return StreamEvent{}, io.EOF
	}

	var delta struct {
		Choices []struct {
			Delta struct {
				Content string `json:"content"`
			} `json:"delta"`
		} `json:"choices"`
	}

	if err := json.Unmarshal([]byte(line), &delta); err != nil {
		return StreamEvent{}, err
	}

	content := ""
	if len(delta.Choices) > 0 {
		content = delta.Choices[0].Delta.Content
	}

	return StreamEvent{
		Type:    StreamEventContent,
		Content: content,
	}, nil
}
