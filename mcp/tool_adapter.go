package mcp

import (
	"context"
	"encoding/json"
	"fmt"
)

// ExecutableTool is an MCP tool ready to be called from iteragent.
type ExecutableTool struct {
	Name        string
	Description string
	InputSchema json.RawMessage
	Execute     func(ctx context.Context, args map[string]string) (string, error)
}

// toolCaller is satisfied by both *Client and *McpClient.
type toolCaller interface {
	ListTools(ctx context.Context) ([]Tool, error)
	CallTool(ctx context.Context, name string, args map[string]interface{}) (*CallToolResult, error)
}

// ToolAdapter converts MCP tools into ExecutableTools for the iteragent runtime.
type ToolAdapter struct {
	client toolCaller
}

// NewToolAdapter creates a ToolAdapter backed by a legacy HTTP Client.
func NewToolAdapter(client *Client) *ToolAdapter {
	return &ToolAdapter{client: client}
}

// NewMcpToolAdapter creates a ToolAdapter backed by a full McpClient (stdio or HTTP).
func NewMcpToolAdapter(client *McpClient) *ToolAdapter {
	return &ToolAdapter{client: client}
}

// GetTools fetches the tool list from the MCP server and wraps each tool for use
// as an iteragent ExecutableTool.
func (a *ToolAdapter) GetTools(ctx context.Context) ([]ExecutableTool, error) {
	mcpTools, err := a.client.ListTools(ctx)
	if err != nil {
		return nil, err
	}

	tools := make([]ExecutableTool, len(mcpTools))
	for i, t := range mcpTools {
		mt := t
		tools[i] = ExecutableTool{
			Name:        mt.Name,
			Description: mt.Description,
			InputSchema: mt.InputSchema,
			Execute: func(ctx context.Context, args map[string]string) (string, error) {
				// Convert string args to interface{} with JSON-aware typing.
				mapArgs := make(map[string]interface{}, len(args))
				for k, v := range args {
					var val interface{}
					if err := json.Unmarshal([]byte(v), &val); err == nil {
						mapArgs[k] = val
					} else {
						mapArgs[k] = v
					}
				}

				result, err := a.client.CallTool(ctx, mt.Name, mapArgs)
				if err != nil {
					return "", err
				}

				if result.IsError {
					var errMsg string
					for _, c := range result.Content {
						errMsg += c.Text + "\n"
					}
					return "", fmt.Errorf("mcp tool error: %s", errMsg)
				}

				var output string
				for _, c := range result.Content {
					if c.Text != "" {
						output += c.Text + "\n"
					}
				}
				return output, nil
			},
		}
	}
	return tools, nil
}
