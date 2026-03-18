package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// Tool describes an MCP tool.
type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"inputSchema"`
}

// ContentBlock is a piece of content in a tool result.
type ContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

// CallToolResult holds the result of a tools/call invocation.
type CallToolResult struct {
	Content []ContentBlock `json:"content"`
	IsError bool           `json:"isError,omitempty"`
}

// ListToolsResult holds the result of a tools/list call.
type ListToolsResult struct {
	Tools []Tool `json:"tools"`
}

// Client is a legacy HTTP-only MCP client retained for backward compatibility.
// For new code, prefer McpClient which supports both stdio and HTTP transports.
type Client struct {
	URL    string
	client *http.Client
}

// NewClient creates a legacy HTTP-only MCP client.
func NewClient(url string) *Client {
	return &Client{
		URL:    url,
		client: &http.Client{Timeout: 30 * time.Second},
	}
}

func (c *Client) post(ctx context.Context, body []byte) (json.RawMessage, error) {
	req, err := http.NewRequestWithContext(ctx, "POST", c.URL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()

	var result struct {
		Result json.RawMessage `json:"result"`
		Error  *struct {
			Message string `json:"message"`
			Code    int    `json:"code"`
		} `json:"error,omitempty"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	if result.Error != nil {
		return nil, fmt.Errorf("mcp error %d: %s", result.Error.Code, result.Error.Message)
	}
	return result.Result, nil
}

// ListTools lists the tools available on the MCP server.
func (c *Client) ListTools(ctx context.Context) ([]Tool, error) {
	body, _ := json.Marshal(map[string]interface{}{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "tools/list",
	})
	raw, err := c.post(ctx, body)
	if err != nil {
		return nil, err
	}
	var result ListToolsResult
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, fmt.Errorf("decode tools: %w", err)
	}
	return result.Tools, nil
}

// CallTool invokes a named tool with the given arguments.
func (c *Client) CallTool(ctx context.Context, name string, args map[string]interface{}) (*CallToolResult, error) {
	body, _ := json.Marshal(map[string]interface{}{
		"jsonrpc": "2.0",
		"id":      2,
		"method":  "tools/call",
		"params": map[string]interface{}{
			"name":      name,
			"arguments": args,
		},
	})
	raw, err := c.post(ctx, body)
	if err != nil {
		return nil, err
	}
	var result CallToolResult
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, fmt.Errorf("decode call result: %w", err)
	}
	return &result, nil
}

// Ping checks if the server responds.
func (c *Client) Ping(ctx context.Context) error {
	body, _ := json.Marshal(map[string]interface{}{
		"jsonrpc": "2.0",
		"id":      0,
		"method":  "ping",
	})
	_, err := c.post(ctx, body)
	return err
}
