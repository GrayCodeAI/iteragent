package iteragent

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
)

type SSEEvent struct {
	Event string
	Data  string
}

type SSEClient struct {
	client *http.Client
}

func NewSSEClient() *SSEClient {
	return &SSEClient{
		client: &http.Client{},
	}
}

func (c *SSEClient) Stream(ctx context.Context, url string, headers map[string]string, body []byte, onEvent func(SSEEvent)) error {
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	for k, v := range headers {
		req.Header.Set(k, v)
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("SSE request failed (%d): %s", resp.StatusCode, string(body))
	}

	reader := bufio.NewReader(resp.Body)
	var event string
	var data strings.Builder

	for {
		line, err := reader.ReadString('\n')
		if err == io.EOF {
			break
		}
		if err != nil {
			break
		}

		line = strings.TrimRight(line, "\r\n")

		if strings.HasPrefix(line, "event:") {
			event = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			continue
		}

		if strings.HasPrefix(line, "data:") {
			data.WriteString(strings.TrimSpace(strings.TrimPrefix(line, "data:")))
			data.WriteString("\n")
			continue
		}

		if line == "" && data.Len() > 0 {
			onEvent(SSEEvent{
				Event: event,
				Data:  strings.TrimSpace(data.String()),
			})
			event = ""
			data.Reset()
		}
	}

	return nil
}

type SSEResponse struct {
	mu       sync.Mutex
	content  strings.Builder
	messages []Message
	stopped  bool
}

func NewSSEResponse() *SSEResponse {
	return &SSEResponse{
		messages: []Message{},
	}
}

func (r *SSEResponse) AddContent(content string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.content.WriteString(content)
}

func (r *SSEResponse) GetContent() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.content.String()
}

func (r *SSEResponse) AddMessage(msg Message) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.messages = append(r.messages, msg)
}

func (r *SSEResponse) GetMessages() []Message {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.messages
}

func (r *SSEResponse) Stop() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.stopped = true
}

func (r *SSEResponse) IsStopped() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.stopped
}

func ParseAnthropicSSE(data string) (string, bool) {
	var event struct {
		Type  string `json:"type"`
		Delta string `json:"delta"`
		Text  string `json:"text"`
		Index int    `json:"index"`
	}

	if err := json.Unmarshal([]byte(data), &event); err != nil {
		return "", false
	}

	switch event.Type {
	case "content_block_delta":
		if event.Delta != "" {
			return event.Delta, true
		}
	case "message_delta":
		return event.Text, true
	}

	return "", false
}

func ParseOpenAISSE(data string) (string, bool) {
	var event struct {
		Choices []struct {
			Delta struct {
				Content string `json:"content"`
			} `json:"delta"`
		} `json:"choices"`
	}

	if err := json.Unmarshal([]byte(data), &event); err != nil {
		return "", false
	}

	if len(event.Choices) > 0 && event.Choices[0].Delta.Content != "" {
		return event.Choices[0].Delta.Content, true
	}

	return "", false
}

func ParseGeminiSSE(data string) (string, bool) {
	var event struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
	}

	if err := json.Unmarshal([]byte(data), &event); err != nil {
		return "", false
	}

	if len(event.Candidates) > 0 && len(event.Candidates[0].Content.Parts) > 0 {
		return event.Candidates[0].Content.Parts[0].Text, true
	}

	return "", false
}
