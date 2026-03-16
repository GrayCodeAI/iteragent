package mcp

import (
	"context"
	"encoding/json"
	"fmt"
)

type McpClient struct {
	Transport Transport
}

func NewMcpClient(t Transport) *McpClient {
	return &McpClient{Transport: t}
}

func ConnectStdio(command string, args []string, env map[string]string) (*McpClient, error) {
	transport, err := NewStdioTransport(command, args, env)
	if err != nil {
		return nil, err
	}
	return &McpClient{Transport: transport}, nil
}

func ConnectHTTP(url string, headers map[string]string) *McpClient {
	transport := NewHTTPTransport(url, headers)
	return &McpClient{Transport: transport}
}

func (c *McpClient) ListTools(ctx context.Context) ([]Tool, error) {
	req := JsonRpcRequest{
		JSONRPC: "2.0",
		Method:  "tools/list",
		Params:  nil,
		ID:      1,
	}

	resp, err := c.Transport.Send(req)
	if err != nil {
		return nil, err
	}

	var result struct {
		Tools []Tool `json:"tools"`
	}
	if resp.Result != nil {
		if err := json.Unmarshal(resp.Result, &result); err != nil {
			return nil, err
		}
	}
	return result.Tools, nil
}

func (c *McpClient) CallTool(ctx context.Context, name string, args map[string]string) (string, error) {
	paramsRaw, _ := json.Marshal(map[string]interface{}{
		"name":      name,
		"arguments": args,
	})

	req := JsonRpcRequest{
		JSONRPC: "2.0",
		Method:  "tools/call",
		Params:  paramsRaw,
		ID:      2,
	}

	resp, err := c.Transport.Send(req)
	if err != nil {
		return "", err
	}

	if resp.Error != nil {
		return "", fmt.Errorf("MCP error: %s", resp.Error.Message)
	}

	var result struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
	}
	if resp.Result != nil {
		if err := json.Unmarshal(resp.Result, &result); err != nil {
			return "", err
		}
	}

	text := ""
	for _, c := range result.Content {
		text += c.Text
	}
	return text, nil
}

func RunMCPCommand(ctx context.Context, cmd string, args []string) ([]Tool, error) {
	parts := append([]string{cmd}, args...)
	command := parts[0]
	commandArgs := parts[1:]

	transport, err := NewStdioTransport(command, commandArgs, nil)
	if err != nil {
		return nil, err
	}

	client := &McpClient{Transport: transport}
	return client.ListTools(ctx)
}
