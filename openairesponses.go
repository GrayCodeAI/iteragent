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

type OpenAIResponsesConfig struct {
	APIKey        string
	BaseURL       string
	Model         string
	MaxTokens     int
	Temperature   float32
	ThinkingLevel ThinkingLevel
}

type OpenAIResponsesProvider struct {
	config OpenAIResponsesConfig
	client *http.Client
}

func NewOpenAIResponses(config OpenAIResponsesConfig) Provider {
	return &OpenAIResponsesProvider{
		config: config,
		client: &http.Client{},
	}
}

func (p *OpenAIResponsesProvider) Name() string {
	return fmt.Sprintf("openai_responses(%s)", p.config.Model)
}

type responsesRequest struct {
	Model             string                   `json:"model"`
	Input             []map[string]interface{} `json:"input"`
	MaxTokens         int                      `json:"max_tokens,omitempty"`
	Temperature       float32                  `json:"temperature,omitempty"`
	Store             bool                     `json:"store,omitempty"`
	ResponseFormat    string                   `json:"response_format,omitempty"`
	Tools             []json.RawMessage        `json:"tools,omitempty"`
	ParallelToolCalls bool                     `json:"parallel_tool_calls,omitempty"`
}

type responsesMessage struct {
	Role    string `json:"role"`
	Content string `json:"content,omitempty"`
	Type    string `json:"type,omitempty"`
}

func messagesToResponsesFormat(messages []Message) []map[string]interface{} {
	result := make([]map[string]interface{}, 0, len(messages))
	for _, msg := range messages {
		if msg.Role == "system" {
			result = append(result, map[string]interface{}{
				"role":    "system",
				"type":    "message",
				"content": msg.Content,
			})
		} else if msg.Role == "user" {
			result = append(result, map[string]interface{}{
				"role":    "user",
				"type":    "message",
				"content": msg.Content,
			})
		} else if msg.Role == "assistant" {
			result = append(result, map[string]interface{}{
				"role":    "assistant",
				"type":    "message",
				"content": msg.Content,
			})
		}
	}
	return result
}

func (p *OpenAIResponsesProvider) Complete(ctx context.Context, messages []Message, opts ...CompletionOptions) (string, error) {
	baseURL := p.config.BaseURL
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}

	url := baseURL + "/responses"

	body := responsesRequest{
		Model:       p.config.Model,
		Input:       messagesToResponsesFormat(messages),
		MaxTokens:   p.config.MaxTokens,
		Temperature: p.config.Temperature,
		Store:       false,
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
	req.Header.Set("Authorization", "Bearer "+p.config.APIKey)

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
		return "", fmt.Errorf("OpenAI Responses API error (%d): %s", resp.StatusCode, string(respBody))
	}

	var response struct {
		Output []struct {
			Type     string `json:"type"`
			Content  string `json:"content,omitempty"`
			Thoughts string `json:"thoughts,omitempty"`
		} `json:"output"`
	}

	if err := json.Unmarshal(respBody, &response); err != nil {
		return "", fmt.Errorf("parse response: %w", err)
	}

	var result strings.Builder
	for _, output := range response.Output {
		if output.Type == "message" {
			result.WriteString(output.Content)
		}
	}

	return result.String(), nil
}

// CompleteStream implements TokenStreamer for the OpenAI Responses API.
// It uses the response.content_part.delta SSE event to deliver text tokens incrementally.
func (p *OpenAIResponsesProvider) CompleteStream(ctx context.Context, messages []Message, opt CompletionOptions, onToken func(string)) (string, error) {
	baseURL := p.config.BaseURL
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}
	url := baseURL + "/responses"

	maxTokens := p.config.MaxTokens
	if opt.MaxTokens > 0 {
		maxTokens = opt.MaxTokens
	}
	temperature := p.config.Temperature
	if opt.Temperature > 0 {
		temperature = opt.Temperature
	}
	body := responsesRequest{
		Model:       p.config.Model,
		Input:       messagesToResponsesFormat(messages),
		MaxTokens:   maxTokens,
		Temperature: temperature,
		Store:       false,
	}
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	var full strings.Builder
	sseClient := NewSSEClient()
	err = sseClient.Stream(ctx, url, map[string]string{"Authorization": "Bearer " + p.config.APIKey}, jsonBody, func(e SSEEvent) {
		if token, ok := ParseOpenAIResponsesSSE(e); ok {
			full.WriteString(token)
			if onToken != nil {
				onToken(token)
			}
		}
	})
	if err != nil {
		return "", fmt.Errorf("openai responses stream: %w", err)
	}
	return full.String(), nil
}

func (p *OpenAIResponsesProvider) Stream(ctx context.Context, config StreamConfig, messages []Message, onEvent func(StreamEvent)) (Message, error) {
	baseURL := p.config.BaseURL
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}

	url := baseURL + "/responses"

	body := responsesRequest{
		Model:       p.config.Model,
		Input:       messagesToResponsesFormat(messages),
		MaxTokens:   config.MaxTokens,
		Temperature: config.Temperature,
		Store:       false,
	}

	jsonBody, _ := json.Marshal(body)

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonBody))
	if err != nil {
		return Message{}, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.config.APIKey)
	req.Header.Set("Accept", "text/event-stream")

	resp, err := p.client.Do(req)
	if err != nil {
		return Message{}, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return Message{}, fmt.Errorf("OpenAI Responses API error (%d): %s", resp.StatusCode, string(respBody))
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
			onEvent(StreamEvent{
				Type:    StreamEventContent,
				Content: event.Content,
			})
		}
	}

	return Message{
		Role:    "assistant",
		Content: content.String(),
	}, nil
}
