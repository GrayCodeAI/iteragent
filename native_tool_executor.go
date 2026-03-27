package iteragent

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
)

// NativeToolCallRequest is a single tool call from a native API response.
type NativeToolCallRequest struct {
	ID        string          // Unique call ID (used for tool_result correlation)
	Name      string          // Tool name
	ArgsJSON  json.RawMessage // Raw JSON arguments
	ArgsStr   string          // Raw string arguments (OpenAI format)
}

// NativeToolCallResult holds the outcome of one tool execution.
type NativeToolCallResult struct {
	ID      string // Matches NativeToolCallRequest.ID
	Name    string
	Content string
	IsError bool
}

// NativeToolExecutor executes multiple native tool calls, optionally in
// parallel, using the agent's registered tool set.
type NativeToolExecutor struct {
	tools   map[string]Tool
	parallel bool
}

// NewNativeToolExecutor creates an executor backed by the given tool map.
// When parallel is true, all calls in a batch are run concurrently.
func NewNativeToolExecutor(tools map[string]Tool, parallel bool) *NativeToolExecutor {
	return &NativeToolExecutor{tools: tools, parallel: parallel}
}

// Execute runs all tool calls and returns results in the same order.
func (e *NativeToolExecutor) Execute(ctx context.Context, calls []NativeToolCallRequest) []NativeToolCallResult {
	results := make([]NativeToolCallResult, len(calls))
	if e.parallel {
		var wg sync.WaitGroup
		for i, call := range calls {
			wg.Add(1)
			go func(idx int, c NativeToolCallRequest) {
				defer wg.Done()
				results[idx] = e.execOne(ctx, c)
			}(i, call)
		}
		wg.Wait()
	} else {
		for i, call := range calls {
			results[i] = e.execOne(ctx, call)
		}
	}
	return results
}

func (e *NativeToolExecutor) execOne(ctx context.Context, call NativeToolCallRequest) NativeToolCallResult {
	tool, ok := e.tools[call.Name]
	if !ok {
		return NativeToolCallResult{
			ID:      call.ID,
			Name:    call.Name,
			Content: fmt.Sprintf("unknown tool: %s", call.Name),
			IsError: true,
		}
	}

	// Parse the JSON arguments into map[string]string for Tool.Execute.
	args, err := parseNativeToolArgs(call.ArgsJSON, call.ArgsStr)
	if err != nil {
		return NativeToolCallResult{
			ID:      call.ID,
			Name:    call.Name,
			Content: fmt.Sprintf("invalid arguments: %s", err),
			IsError: true,
		}
	}

	result, execErr := tool.Execute(ctx, args)
	if execErr != nil {
		return NativeToolCallResult{
			ID:      call.ID,
			Name:    call.Name,
			Content: fmt.Sprintf("ERROR: %s\nOutput: %s", execErr, result),
			IsError: true,
		}
	}
	return NativeToolCallResult{
		ID:      call.ID,
		Name:    call.Name,
		Content: result,
	}
}

// parseNativeToolArgs converts native API tool arguments to map[string]interface{}.
// Prefers argsJSON when non-nil; falls back to argsStr (OpenAI JSON string).
func parseNativeToolArgs(argsJSON json.RawMessage, argsStr string) (map[string]interface{}, error) {
	src := argsStr
	if len(argsJSON) > 0 {
		src = string(argsJSON)
	}
	if src == "" || src == "null" {
		return map[string]interface{}{}, nil
	}

	var result map[string]interface{}
	if err := json.Unmarshal([]byte(src), &result); err != nil {
		return nil, fmt.Errorf("cannot parse tool args %q: %w", src, err)
	}
	return result, nil
}
