package mcp

import (
	"context"
	"encoding/json"
	"fmt"

	iteragent "github.com/GrayCodeAI/iteragent"
)

type ToolAdapter struct {
	client *Client
}

func NewToolAdapter(client *Client) *ToolAdapter {
	return &ToolAdapter{client: client}
}

func (a *ToolAdapter) GetTools(ctx context.Context) ([]iteragent.Tool, error) {
	mcpTools, err := a.client.ListTools(ctx)
	if err != nil {
		return nil, err
	}

	tools := make([]iteragent.Tool, len(mcpTools))
	for i, t := range mcpTools {
		mt := t
		tools[i] = iteragent.Tool{
			Name:        mt.Name,
			Description: mt.Description,
			Execute: func(ctx context.Context, args map[string]string) (string, error) {
				mapArgs := make(map[string]interface{})
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
					output += c.Text + "\n"
				}
				return output, nil
			},
		}
	}

	return tools, nil
}
