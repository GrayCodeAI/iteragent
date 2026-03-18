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
	"sync/atomic"
)

// Transport is the low-level JSON-RPC 2.0 MCP transport.
type Transport interface {
	Send(ctx context.Context, request JsonRpcRequest) (JsonRpcResponse, error)
	Close() error
}

// JsonRpcRequest is a JSON-RPC 2.0 request.
type JsonRpcRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
	ID      interface{}     `json:"id,omitempty"`
}

// JsonRpcResponse is a JSON-RPC 2.0 response.
type JsonRpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *JsonRpcError   `json:"error,omitempty"`
	ID      interface{}     `json:"id,omitempty"`
}

// JsonRpcError is a JSON-RPC 2.0 error object.
type JsonRpcError struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}

// McpError wraps an MCP protocol error.
type McpError struct {
	Code    int
	Message string
}

func (e *McpError) Error() string {
	return e.Message
}

// ---- Stdio Transport ----

// StdioTransport communicates with an MCP server over a child process's stdio.
type StdioTransport struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader
	mu     sync.Mutex
	nextID atomic.Int64
}

// NewStdioTransport spawns a child process and returns a stdio transport.
// env is a map of additional environment variables (may be nil).
func NewStdioTransport(command string, args []string, env map[string]string) (*StdioTransport, error) {
	cmd := exec.Command(command, args...)

	if env != nil {
		for k, v := range env {
			cmd.Env = append(cmd.Env, k+"="+v)
		}
	}

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to get stdin pipe: %w", err)
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to get stdout pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start %s: %w", command, err)
	}

	return &StdioTransport{
		cmd:    cmd,
		stdin:  stdin,
		stdout: bufio.NewReader(stdout),
	}, nil
}

func (t *StdioTransport) Send(ctx context.Context, request JsonRpcRequest) (JsonRpcResponse, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if request.ID == nil {
		request.ID = t.nextID.Add(1)
	}

	data, err := json.Marshal(request)
	if err != nil {
		return JsonRpcResponse{}, fmt.Errorf("marshal request: %w", err)
	}
	data = append(data, '\n')

	if _, err := t.stdin.Write(data); err != nil {
		return JsonRpcResponse{}, fmt.Errorf("write request: %w", err)
	}

	// Read response line with context cancellation support.
	type readResult struct {
		resp JsonRpcResponse
		err  error
	}
	ch := make(chan readResult, 1)
	go func() {
		line, err := t.stdout.ReadBytes('\n')
		if err != nil {
			ch <- readResult{err: fmt.Errorf("read response: %w", err)}
			return
		}
		var resp JsonRpcResponse
		if err := json.Unmarshal(bytes.TrimSpace(line), &resp); err != nil {
			ch <- readResult{err: fmt.Errorf("parse response: %w", err)}
			return
		}
		ch <- readResult{resp: resp}
	}()

	select {
	case <-ctx.Done():
		return JsonRpcResponse{}, ctx.Err()
	case r := <-ch:
		if r.err != nil {
			return JsonRpcResponse{}, r.err
		}
		if r.resp.Error != nil {
			return r.resp, fmt.Errorf("RPC error %d: %s", r.resp.Error.Code, r.resp.Error.Message)
		}
		return r.resp, nil
	}
}

func (t *StdioTransport) Close() error {
	_ = t.stdin.Close()
	if t.cmd != nil && t.cmd.Process != nil {
		return t.cmd.Process.Kill()
	}
	return nil
}

// ---- HTTP Transport ----

// HTTPTransport communicates with an MCP server over HTTP POST.
type HTTPTransport struct {
	url     string
	headers map[string]string
	client  *http.Client
	mu      sync.Mutex
	nextID  atomic.Int64
}

// NewHTTPTransport creates an HTTP-based MCP transport.
func NewHTTPTransport(url string, headers map[string]string) *HTTPTransport {
	return &HTTPTransport{
		url:     url,
		headers: headers,
		client:  &http.Client{},
	}
}

func (t *HTTPTransport) Send(ctx context.Context, request JsonRpcRequest) (JsonRpcResponse, error) {
	t.mu.Lock()
	if request.ID == nil {
		request.ID = t.nextID.Add(1)
	}
	t.mu.Unlock()

	data, err := json.Marshal(request)
	if err != nil {
		return JsonRpcResponse{}, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", t.url, bytes.NewReader(data))
	if err != nil {
		return JsonRpcResponse{}, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range t.headers {
		req.Header.Set(k, v)
	}

	resp, err := t.client.Do(req)
	if err != nil {
		return JsonRpcResponse{}, fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()

	var response JsonRpcResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return JsonRpcResponse{}, fmt.Errorf("parse response: %w", err)
	}

	if response.Error != nil {
		return response, fmt.Errorf("RPC error %d: %s", response.Error.Code, response.Error.Message)
	}

	return response, nil
}

func (t *HTTPTransport) Close() error { return nil }
