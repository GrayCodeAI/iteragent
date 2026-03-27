package iteragent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// anthropicToolUseRequest is the body sent to /v1/messages when tools are present.
type anthropicToolUseRequest struct {
	Model     string                    `json:"model"`
	MaxTokens int                       `json:"max_tokens"`
	System    interface{}               `json:"system,omitempty"`
	Messages  []anthropicNativeMessage  `json:"messages"`
	Tools     []anthropicToolDef        `json:"tools,omitempty"`
	Stream    bool                      `json:"stream,omitempty"`
	Thinking  map[string]interface{}    `json:"thinking,omitempty"`
}

type anthropicNativeMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"` // string or []anthropicContentBlock
}

type anthropicContentBlock struct {
	Type  string          `json:"type"`
	Text  string          `json:"text,omitempty"`
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
	// For tool_result:
	ToolUseID string      `json:"tool_use_id,omitempty"`
	Content   interface{} `json:"content,omitempty"`
}

type anthropicToolDef struct {
	Name        string     `json:"name"`
	Description string     `json:"description"`
	InputSchema ToolSchema `json:"input_schema"`
}

type anthropicNativeResponse struct {
	Content []anthropicContentBlock `json:"content"`
	Error   *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
	StopReason string `json:"stop_reason"`
}

// NativeToolResult carries a single tool execution result back to Anthropic.
type NativeToolResult struct {
	ToolUseID string
	Content   string
	IsError   bool
}

// NativeToolCaller marks anthropicProvider as supporting native tool-calling.
func (p *anthropicProvider) SupportsNativeTools() bool { return true }

// CompleteWithNativeTools sends a completion request with native tool definitions.
// It executes tool_use blocks using executeFn and returns the final text response
// after all tool calls are resolved (up to maxRounds).
func (p *anthropicProvider) CompleteWithNativeTools(
	ctx context.Context,
	messages []Message,
	tools []Tool,
	opts CompletionOptions,
	executeFn func(name string, input json.RawMessage) (string, error),
	maxRounds int,
) (string, error) {
	if maxRounds <= 0 {
		maxRounds = 10
	}

	nativeDefs := buildAnthropicToolDefs(tools)

	// Convert iteragent messages to native format.
	var system string
	nativeMsgs := make([]anthropicNativeMessage, 0, len(messages))
	for _, m := range messages {
		if m.Role == "system" {
			system = m.Content
			continue
		}
		nativeMsgs = append(nativeMsgs, anthropicNativeMessage{
			Role:    m.Role,
			Content: m.Content,
		})
	}

	for round := 0; round < maxRounds; round++ {
		maxTokens := 4096
		if opts.MaxTokens > 0 {
			maxTokens = opts.MaxTokens
		}

		model := p.cfg.Model
		if opts.Model != "" {
			model = opts.Model
		}
		reqBody := anthropicToolUseRequest{
			Model:     model,
			MaxTokens: maxTokens,
			Messages:  nativeMsgs,
			Tools:     nativeDefs,
		}
		if system != "" {
			reqBody.System = system
		}
		if opts.ThinkingLevel != ThinkingLevelOff && opts.ThinkingLevel != "" {
			budget := thinkingBudget(opts.ThinkingLevel)
			reqBody.Thinking = map[string]interface{}{
				"type": "enabled", "budget_tokens": budget,
			}
		}

		body, err := json.Marshal(reqBody)
		if err != nil {
			return "", fmt.Errorf("marshal native tools request: %w", err)
		}

		req, err := http.NewRequestWithContext(ctx, "POST", "https://api.anthropic.com/v1/messages", bytes.NewReader(body))
		if err != nil {
			return "", fmt.Errorf("create request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("x-api-key", p.cfg.APIKey)
		req.Header.Set("anthropic-version", "2023-06-01")
		if opts.CacheConfig != nil && opts.CacheConfig.Enabled {
			req.Header.Set("anthropic-beta", "prompt-caching-2024-07-31")
		}

		resp, err := p.client.Do(req)
		if err != nil {
			return "", fmt.Errorf("http request: %w", err)
		}
		raw, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return "", fmt.Errorf("read response: %w", err)
		}

		var result anthropicNativeResponse
		if err := json.Unmarshal(raw, &result); err != nil {
			return "", fmt.Errorf("unmarshal response: %w", err)
		}
		if result.Error != nil {
			return "", fmt.Errorf("anthropic error: %s", result.Error.Message)
		}

		// Collect text and tool_use blocks.
		var textOut string
		var toolUseCalls []anthropicContentBlock
		for _, block := range result.Content {
			switch block.Type {
			case "text":
				textOut += block.Text
			case "thinking":
				// Silently consume thinking blocks.
			case "tool_use":
				toolUseCalls = append(toolUseCalls, block)
			}
		}

		// If no tool calls, we're done.
		if result.StopReason == "end_turn" || len(toolUseCalls) == 0 {
			return textOut, nil
		}

		// Append the assistant message (with tool_use blocks) to conversation.
		nativeMsgs = append(nativeMsgs, anthropicNativeMessage{
			Role:    "assistant",
			Content: result.Content,
		})

		// Execute each tool call and build tool_result blocks.
		toolResults := make([]anthropicContentBlock, 0, len(toolUseCalls))
		for _, call := range toolUseCalls {
			var resultContent string
			var isErr bool
			if executeFn != nil {
				res, execErr := executeFn(call.Name, call.Input)
				if execErr != nil {
					resultContent = fmt.Sprintf("Error: %s", execErr)
					isErr = true
				} else {
					resultContent = res
				}
			} else {
				resultContent = fmt.Sprintf("Tool %s is not available", call.Name)
				isErr = true
			}
			_ = isErr
			toolResults = append(toolResults, anthropicContentBlock{
				Type:      "tool_result",
				ToolUseID: call.ID,
				Content:   resultContent,
			})
		}

		// Append user message with tool results.
		nativeMsgs = append(nativeMsgs, anthropicNativeMessage{
			Role:    "user",
			Content: toolResults,
		})
	}

	return "", fmt.Errorf("anthropic native tools: exceeded max rounds (%d)", maxRounds)
}

func buildAnthropicToolDefs(tools []Tool) []anthropicToolDef {
	defs := make([]anthropicToolDef, len(tools))
	for i, t := range tools {
		schema, desc := parseToolDescription(t.Description)
		defs[i] = anthropicToolDef{
			Name:        t.Name,
			Description: desc,
			InputSchema: schema,
		}
	}
	return defs
}
