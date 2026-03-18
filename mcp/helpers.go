package mcp

import (
	"context"
	"encoding/json"
	"fmt"
)

// McpClient is a high-level MCP client backed by a Transport.
type McpClient struct {
	Transport Transport
}

// NewMcpClient wraps an existing transport without performing the initialize handshake.
func NewMcpClient(t Transport) *McpClient {
	return &McpClient{Transport: t}
}

// ConnectStdio spawns a child process and returns an initialized McpClient.
func ConnectStdio(ctx context.Context, command string, args []string, env map[string]string) (*McpClient, error) {
	transport, err := NewStdioTransport(command, args, env)
	if err != nil {
		return nil, err
	}
	c := &McpClient{Transport: transport}
	if err := c.initialize(ctx); err != nil {
		_ = transport.Close()
		return nil, fmt.Errorf("mcp initialize: %w", err)
	}
	return c, nil
}

// ConnectHTTP creates an HTTP-based McpClient and performs the initialize handshake.
func ConnectHTTP(ctx context.Context, url string, headers map[string]string) (*McpClient, error) {
	c := &McpClient{Transport: NewHTTPTransport(url, headers)}
	if err := c.initialize(ctx); err != nil {
		return nil, fmt.Errorf("mcp initialize: %w", err)
	}
	return c, nil
}

// initialize performs the MCP initialize handshake.
func (c *McpClient) initialize(ctx context.Context) error {
	params, _ := json.Marshal(map[string]interface{}{
		"protocolVersion": "2024-11-05",
		"capabilities":    map[string]interface{}{},
		"clientInfo": map[string]interface{}{
			"name":    "iteragent",
			"version": "0.1.0",
		},
	})
	req := JsonRpcRequest{
		JSONRPC: "2.0",
		Method:  "initialize",
		Params:  params,
	}
	_, err := c.Transport.Send(ctx, req)
	return err
}

// ListTools returns the list of tools advertised by the MCP server.
func (c *McpClient) ListTools(ctx context.Context) ([]Tool, error) {
	req := JsonRpcRequest{
		JSONRPC: "2.0",
		Method:  "tools/list",
	}
	resp, err := c.Transport.Send(ctx, req)
	if err != nil {
		return nil, err
	}

	var result struct {
		Tools []Tool `json:"tools"`
	}
	if resp.Result != nil {
		if err := json.Unmarshal(resp.Result, &result); err != nil {
			return nil, fmt.Errorf("decode tools: %w", err)
		}
	}
	return result.Tools, nil
}

// CallTool invokes a tool by name with the given arguments.
func (c *McpClient) CallTool(ctx context.Context, name string, args map[string]interface{}) (*CallToolResult, error) {
	paramsRaw, _ := json.Marshal(map[string]interface{}{
		"name":      name,
		"arguments": args,
	})
	req := JsonRpcRequest{
		JSONRPC: "2.0",
		Method:  "tools/call",
		Params:  paramsRaw,
	}
	resp, err := c.Transport.Send(ctx, req)
	if err != nil {
		return nil, err
	}
	if resp.Error != nil {
		return nil, fmt.Errorf("MCP error: %s", resp.Error.Message)
	}

	var result CallToolResult
	if resp.Result != nil {
		if err := json.Unmarshal(resp.Result, &result); err != nil {
			return nil, fmt.Errorf("decode call result: %w", err)
		}
	}
	return &result, nil
}

// Close closes the underlying transport.
func (c *McpClient) Close() error {
	return c.Transport.Close()
}

// RunMCPCommand connects to a stdio MCP server, lists its tools, and returns them.
func RunMCPCommand(ctx context.Context, cmd string, args []string) ([]Tool, error) {
	client, err := ConnectStdio(ctx, cmd, args, nil)
	if err != nil {
		return nil, err
	}
	defer client.Close()
	return client.ListTools(ctx)
}
