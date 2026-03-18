package iteragent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

type VertexConfig struct {
	ProjectID   string
	Location    string
	Model       string
	Credentials string
	MaxTokens   int
	Temperature float32
}

type VertexProvider struct {
	config VertexConfig
	client *http.Client
}

func NewVertex(config VertexConfig) *VertexProvider {
	return &VertexProvider{
		config: config,
		client: &http.Client{},
	}
}

func (p *VertexProvider) Name() string {
	return fmt.Sprintf("vertex(%s)", p.config.Model)
}

func (p *VertexProvider) getAccessToken(ctx context.Context) (string, error) {
	credFile := p.config.Credentials
	if credFile == "" {
		credFile = os.Getenv("GOOGLE_APPLICATION_CREDENTIALS")
	}

	if credFile != "" {
		data, err := os.ReadFile(credFile)
		if err != nil {
			return "", err
		}
		var creds struct {
			ClientEmail string `json:"client_email"`
			PrivateKey  string `json:"private_key"`
		}
		if err := json.Unmarshal(data, &creds); err != nil {
			return "", err
		}
		return "dummy_token_from_service_account", nil
	}

	tokenSrc := os.Getenv("GOOGLE_ACCESS_TOKEN")
	if tokenSrc != "" {
		return tokenSrc, nil
	}

	return "", fmt.Errorf("no credentials found for Vertex AI")
}

func (p *VertexProvider) Complete(ctx context.Context, messages []Message, opts ...CompletionOptions) (string, error) {
	location := p.config.Location
	if location == "" {
		location = "us-central1"
	}

	url := fmt.Sprintf("https://%s-aiplatform.googleapis.com/v1/projects/%s/locations/%s/publishers/google/models/%s:generateContent",
		location, p.config.ProjectID, location, p.config.Model)

	token, err := p.getAccessToken(ctx)
	if err != nil {
		return "", err
	}

	var system string
	var contents []map[string]interface{}
	for _, m := range messages {
		if m.Role == "system" {
			system = m.Content
		} else {
			role := "user"
			if m.Role == "assistant" {
				role = "model"
			}
			contents = append(contents, map[string]interface{}{
				"role": role,
				"parts": []map[string]string{
					{"text": m.Content},
				},
			})
		}
	}

	body := map[string]interface{}{
		"contents": contents,
	}
	if system != "" {
		body["systemInstruction"] = map[string]interface{}{
			"parts": []map[string]string{
				{"text": system},
			},
		}
	}
	if p.config.MaxTokens > 0 {
		body["maxOutputTokens"] = p.config.MaxTokens
	}
	if p.config.Temperature > 0 {
		body["temperature"] = p.config.Temperature
	}

	jsonBody, _ := json.Marshal(body)

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonBody))
	if err != nil {
		return "", err
	}

	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("Vertex AI error (%d): %s", resp.StatusCode, string(respBody))
	}

	var response struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
	}

	if err := json.Unmarshal(respBody, &response); err != nil {
		return "", fmt.Errorf("parse response: %w", err)
	}

	if len(response.Candidates) == 0 {
		return "", fmt.Errorf("no response candidates")
	}

	texts := []string{}
	for _, part := range response.Candidates[0].Content.Parts {
		texts = append(texts, part.Text)
	}

	return strings.Join(texts, ""), nil
}

func (p *VertexProvider) Stream(ctx context.Context, config StreamConfig, messages []Message, onEvent func(StreamEvent)) (Message, error) {
	location := p.config.Location
	if location == "" {
		location = "us-central1"
	}

	url := fmt.Sprintf("https://%s-aiplatform.googleapis.com/v1/projects/%s/locations/%s/publishers/google/models/%s:streamGenerateContent",
		location, p.config.ProjectID, location, p.config.Model)

	token, err := p.getAccessToken(ctx)
	if err != nil {
		return Message{}, err
	}

	var contents []map[string]interface{}
	for _, m := range messages {
		role := "user"
		if m.Role == "assistant" {
			role = "model"
		}
		contents = append(contents, map[string]interface{}{
			"role": role,
			"parts": []map[string]string{
				{"text": m.Content},
			},
		})
	}

	body := map[string]interface{}{
		"contents": contents,
	}
	if config.MaxTokens > 0 {
		body["maxOutputTokens"] = config.MaxTokens
	}
	if config.Temperature > 0 {
		body["temperature"] = config.Temperature
	}

	jsonBody, _ := json.Marshal(body)

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonBody))
	if err != nil {
		return Message{}, err
	}

	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return Message{}, err
	}
	defer resp.Body.Close()

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

		if event.Type == "content" {
			content.WriteString(event.Content)
			onEvent(event)
		}
	}

	return Message{
		Role:    "assistant",
		Content: content.String(),
	}, nil
}
