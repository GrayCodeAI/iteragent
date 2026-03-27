package iteragent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// openaiFunction is the schema for a single function in the OpenAI tools array.
type openaiFunction struct {
	Name        string     `json:"name"`
	Description string     `json:"description,omitempty"`
	Parameters  ToolSchema `json:"parameters"`
}

type openaiTool struct {
	Type     string         `json:"type"` // always "function"
	Function openaiFunction `json:"function"`
}

type openaiToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON string
}

type openaiToolCall struct {
	ID       string                 `json:"id"`
	Type     string                 `json:"type"`
	Function openaiToolCallFunction `json:"function"`
}

type openaiNativeMessage struct {
	Role       string           `json:"role"`
	Content    interface{}      `json:"content,omitempty"` // string or nil
	ToolCalls  []openaiToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
	Name       string           `json:"name,omitempty"`
}

type openaiNativeResponse struct {
	Choices []struct {
		Message      openaiNativeMessage `json:"message"`
		FinishReason string              `json:"finish_reason"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// SupportsNativeTools marks openaiCompatProvider as supporting function calling.
func (p *openaiCompatProvider) SupportsNativeTools() bool { return true }

// CompleteWithNativeFunctions sends a completion request with native OpenAI
// function_calling / tools. It runs up to maxRounds of tool calls.
func (p *openaiCompatProvider) CompleteWithNativeFunctions(
	ctx context.Context,
	messages []Message,
	tools []Tool,
	opts CompletionOptions,
	executeFn func(name string, argsJSON string) (string, error),
	maxRounds int,
) (string, error) {
	if maxRounds <= 0 {
		maxRounds = 10
	}

	openaiTools := buildOpenAITools(tools)

	// Convert iteragent messages to OpenAI native format.
	nativeMsgs := make([]openaiNativeMessage, 0, len(messages))
	for _, m := range messages {
		nativeMsgs = append(nativeMsgs, openaiNativeMessage{
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
		reqBody := map[string]interface{}{
			"model":      model,
			"max_tokens": maxTokens,
			"messages":   nativeMsgs,
			"tools":      openaiTools,
		}
		if opts.Temperature > 0 {
			reqBody["temperature"] = opts.Temperature
		}

		body, err := json.Marshal(reqBody)
		if err != nil {
			return "", fmt.Errorf("marshal openai native tools request: %w", err)
		}

		req, err := http.NewRequestWithContext(ctx, "POST", p.cfg.BaseURL+"/chat/completions", bytes.NewReader(body))
		if err != nil {
			return "", fmt.Errorf("create request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")
		if p.cfg.APIKey != "" {
			req.Header.Set("Authorization", "Bearer "+p.cfg.APIKey)
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

		var result openaiNativeResponse
		if err := json.Unmarshal(raw, &result); err != nil {
			return "", fmt.Errorf("unmarshal response: %w", err)
		}
		if result.Error != nil {
			return "", fmt.Errorf("openai error: %s", result.Error.Message)
		}
		if len(result.Choices) == 0 {
			return "", fmt.Errorf("no choices in openai response")
		}

		choice := result.Choices[0]

		// No tool calls — done.
		if choice.FinishReason == "stop" || len(choice.Message.ToolCalls) == 0 {
			if s, ok := choice.Message.Content.(string); ok {
				return s, nil
			}
			return "", nil
		}

		// Append assistant message with tool_calls.
		nativeMsgs = append(nativeMsgs, choice.Message)

		// Execute each tool call and append tool results.
		for _, tc := range choice.Message.ToolCalls {
			var resultContent string
			if executeFn != nil {
				res, execErr := executeFn(tc.Function.Name, tc.Function.Arguments)
				if execErr != nil {
					resultContent = fmt.Sprintf("Error: %s", execErr)
				} else {
					resultContent = res
				}
			} else {
				resultContent = fmt.Sprintf("Tool %s is not available", tc.Function.Name)
			}
			nativeMsgs = append(nativeMsgs, openaiNativeMessage{
				Role:       "tool",
				Content:    resultContent,
				ToolCallID: tc.ID,
			})
		}
	}

	return "", fmt.Errorf("openai native tools: exceeded max rounds (%d)", maxRounds)
}

func buildOpenAITools(tools []Tool) []openaiTool {
	result := make([]openaiTool, len(tools))
	for i, t := range tools {
		schema, desc := parseToolDescription(t.Description)
		result[i] = openaiTool{
			Type: "function",
			Function: openaiFunction{
				Name:        t.Name,
				Description: desc,
				Parameters:  schema,
			},
		}
	}
	return result
}
