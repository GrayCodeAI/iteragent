package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"sync"
)

type Transport interface {
	Send(request JsonRpcRequest) (JsonRpcResponse, error)
	Close() error
}

type JsonRpcRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
	ID      interface{}     `json:"id,omitempty"`
}

type JsonRpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *JsonRpcError   `json:"error,omitempty"`
	ID      interface{}     `json:"id,omitempty"`
}

type JsonRpcError struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}

type McpError struct {
	Code    int
	Message string
}

func (e *McpError) Error() string {
	return e.Message
}

type StdioTransport struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader
	mu     sync.Mutex
	id     int
}

func NewStdioTransport(command string, args []string, env map[string]string) (*StdioTransport, error) {
	cmd := exec.Command(command, args...)

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to get stdin pipe: %w", err)
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to get stdout pipe: %w", err)
	}

	if env != nil {
		for k, v := range env {
			cmd.Env = append(cmd.Env, k+"="+v)
		}
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start %s: %w", command, err)
	}

	return &StdioTransport{
		cmd:    cmd,
		stdin:  stdin,
		stdout: bufio.NewReader(stdout),
		id:     1,
	}, nil
}

func (t *StdioTransport) Send(request JsonRpcRequest) (JsonRpcResponse, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	request.ID = t.id
	t.id++

	data, err := json.Marshal(request)
	if err != nil {
		return JsonRpcResponse{}, fmt.Errorf("marshal request: %w", err)
	}
	data = append(data, '\n')

	if _, err := t.stdin.Write(data); err != nil {
		return JsonRpcResponse{}, fmt.Errorf("write request: %w", err)
	}

	line, err := t.stdout.ReadBytes('\n')
	if err != nil {
		return JsonRpcResponse{}, fmt.Errorf("read response: %w", err)
	}

	var response JsonRpcResponse
	if err := json.Unmarshal(line, &response); err != nil {
		return JsonRpcResponse{}, fmt.Errorf("parse response: %w", err)
	}

	if response.Error != nil {
		return response, fmt.Errorf("RPC error %d: %s", response.Error.Code, response.Error.Message)
	}

	return response, nil
}

func (t *StdioTransport) Close() error {
	if t.cmd != nil && t.cmd.Process != nil {
		return t.cmd.Process.Kill()
	}
	return nil
}

type HTTPTransport struct {
	url     string
	headers map[string]string
	client  *HTTPClient
	mu      sync.Mutex
	id      int
}

type HTTPClient struct {
	baseURL string
	headers map[string]string
}

func NewHTTPTransport(url string, headers map[string]string) *HTTPTransport {
	return &HTTPTransport{
		url:     url,
		headers: headers,
		client:  &HTTPClient{baseURL: url, headers: headers},
		id:      1,
	}
}

func (t *HTTPTransport) Send(request JsonRpcRequest) (JsonRpcResponse, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	request.ID = t.id
	t.id++

	data, _ := json.Marshal(request)

	resp, err := t.client.Post(context.Background(), "/", data)
	if err != nil {
		return JsonRpcResponse{}, err
	}

	var response JsonRpcResponse
	if err := json.Unmarshal(resp, &response); err != nil {
		return JsonRpcResponse{}, fmt.Errorf("parse response: %w", err)
	}

	if response.Error != nil {
		return response, fmt.Errorf("RPC error %d: %s", response.Error.Code, response.Error.Message)
	}

	return response, nil
}

func (t *HTTPTransport) Close() error {
	return nil
}

func (c *HTTPClient) Post(ctx context.Context, path string, body []byte) ([]byte, error) {
	url := c.baseURL + path
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	for k, v := range c.headers {
		req.Header.Set(k, v)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	return io.ReadAll(resp.Body)
}
