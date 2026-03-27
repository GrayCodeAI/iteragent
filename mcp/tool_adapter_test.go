package mcp_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/GrayCodeAI/iteragent/mcp"
)

// ---------------------------------------------------------------------------
// mock transport for McpClient
// ---------------------------------------------------------------------------

type mockTransport struct {
	sendFn func(ctx context.Context, req mcp.JsonRpcRequest) (mcp.JsonRpcResponse, error)
}

func (m *mockTransport) Send(ctx context.Context, req mcp.JsonRpcRequest) (mcp.JsonRpcResponse, error) {
	return m.sendFn(ctx, req)
}
func (m *mockTransport) Close() error { return nil }

// ---------------------------------------------------------------------------
// NewMcpClient / NewToolAdapter / NewMcpToolAdapter
// ---------------------------------------------------------------------------

func TestNewMcpClient_NonNil(t *testing.T) {
	tr := &mockTransport{sendFn: func(_ context.Context, _ mcp.JsonRpcRequest) (mcp.JsonRpcResponse, error) {
		return mcp.JsonRpcResponse{}, nil
	}}
	c := mcp.NewMcpClient(tr)
	if c == nil {
		t.Fatal("expected non-nil McpClient")
	}
}

func TestNewToolAdapter_FromLegacyClient(t *testing.T) {
	c := mcp.NewClient("http://localhost:9999")
	ta := mcp.NewToolAdapter(c)
	if ta == nil {
		t.Fatal("expected non-nil ToolAdapter")
	}
}

func TestNewMcpToolAdapter_NonNil(t *testing.T) {
	tr := &mockTransport{sendFn: func(_ context.Context, _ mcp.JsonRpcRequest) (mcp.JsonRpcResponse, error) {
		return mcp.JsonRpcResponse{}, nil
	}}
	c := mcp.NewMcpClient(tr)
	ta := mcp.NewMcpToolAdapter(c)
	if ta == nil {
		t.Fatal("expected non-nil ToolAdapter")
	}
}

func TestToolAdapter_Client_ReturnsNilForLegacy(t *testing.T) {
	c := mcp.NewClient("http://localhost:9999")
	ta := mcp.NewToolAdapter(c)
	if ta.Client() != nil {
		t.Error("expected Client() to return nil for legacy Client")
	}
}

func TestToolAdapter_Client_ReturnsMcpClient(t *testing.T) {
	tr := &mockTransport{sendFn: func(_ context.Context, _ mcp.JsonRpcRequest) (mcp.JsonRpcResponse, error) {
		return mcp.JsonRpcResponse{}, nil
	}}
	mc := mcp.NewMcpClient(tr)
	ta := mcp.NewMcpToolAdapter(mc)
	if ta.Client() == nil {
		t.Error("expected non-nil Client() for McpClient-backed adapter")
	}
}

// ---------------------------------------------------------------------------
// ToolAdapter.GetTools — via McpClient with mock transport
// ---------------------------------------------------------------------------

func toolsListResponse(tools []mcp.Tool) mcp.JsonRpcResponse {
	result, _ := json.Marshal(map[string]interface{}{"tools": tools})
	return mcp.JsonRpcResponse{JSONRPC: "2.0", Result: result}
}

func callToolResponse(text string, isError bool) mcp.JsonRpcResponse {
	result, _ := json.Marshal(mcp.CallToolResult{
		Content: []mcp.ContentBlock{{Type: "text", Text: text}},
		IsError: isError,
	})
	return mcp.JsonRpcResponse{JSONRPC: "2.0", Result: result}
}

func TestGetTools_EmptyList(t *testing.T) {
	tr := &mockTransport{sendFn: func(_ context.Context, _ mcp.JsonRpcRequest) (mcp.JsonRpcResponse, error) {
		return toolsListResponse(nil), nil
	}}
	ta := mcp.NewMcpToolAdapter(mcp.NewMcpClient(tr))
	tools, err := ta.GetTools(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(tools) != 0 {
		t.Errorf("expected 0 tools, got %d", len(tools))
	}
}

func TestGetTools_SingleTool(t *testing.T) {
	mcpTools := []mcp.Tool{
		{Name: "echo", Description: "Echoes input", InputSchema: json.RawMessage(`{"type":"object"}`)},
	}
	tr := &mockTransport{sendFn: func(_ context.Context, _ mcp.JsonRpcRequest) (mcp.JsonRpcResponse, error) {
		return toolsListResponse(mcpTools), nil
	}}
	ta := mcp.NewMcpToolAdapter(mcp.NewMcpClient(tr))
	tools, err := ta.GetTools(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}
	if tools[0].Name != "echo" {
		t.Errorf("expected tool name 'echo', got %q", tools[0].Name)
	}
	if tools[0].Description != "Echoes input" {
		t.Errorf("expected description 'Echoes input', got %q", tools[0].Description)
	}
	if tools[0].Execute == nil {
		t.Error("expected non-nil Execute function")
	}
}

func TestGetTools_MultipleTools(t *testing.T) {
	mcpTools := []mcp.Tool{
		{Name: "tool_a", Description: "A"},
		{Name: "tool_b", Description: "B"},
		{Name: "tool_c", Description: "C"},
	}
	tr := &mockTransport{sendFn: func(_ context.Context, _ mcp.JsonRpcRequest) (mcp.JsonRpcResponse, error) {
		return toolsListResponse(mcpTools), nil
	}}
	ta := mcp.NewMcpToolAdapter(mcp.NewMcpClient(tr))
	tools, err := ta.GetTools(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(tools) != 3 {
		t.Fatalf("expected 3 tools, got %d", len(tools))
	}
}

func TestGetTools_Execute_Success(t *testing.T) {
	mcpTools := []mcp.Tool{{Name: "greeter", Description: "Greets"}}
	callCount := 0
	tr := &mockTransport{sendFn: func(_ context.Context, req mcp.JsonRpcRequest) (mcp.JsonRpcResponse, error) {
		if req.Method == "tools/list" {
			return toolsListResponse(mcpTools), nil
		}
		callCount++
		return callToolResponse("hello, world", false), nil
	}}
	ta := mcp.NewMcpToolAdapter(mcp.NewMcpClient(tr))
	tools, _ := ta.GetTools(context.Background())

	result, err := tools[0].Execute(context.Background(), map[string]interface{}{"name": "world"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result, "hello, world") {
		t.Errorf("expected 'hello, world' in result, got %q", result)
	}
	if callCount != 1 {
		t.Errorf("expected 1 call to CallTool, got %d", callCount)
	}
}

func TestGetTools_Execute_ToolError(t *testing.T) {
	mcpTools := []mcp.Tool{{Name: "bad_tool", Description: "Always fails"}}
	tr := &mockTransport{sendFn: func(_ context.Context, req mcp.JsonRpcRequest) (mcp.JsonRpcResponse, error) {
		if req.Method == "tools/list" {
			return toolsListResponse(mcpTools), nil
		}
		return callToolResponse("something went wrong", true), nil
	}}
	ta := mcp.NewMcpToolAdapter(mcp.NewMcpClient(tr))
	tools, _ := ta.GetTools(context.Background())

	_, err := tools[0].Execute(context.Background(), nil)
	if err == nil {
		t.Fatal("expected error for tool error response")
	}
	if !strings.Contains(err.Error(), "mcp tool error") {
		t.Errorf("expected 'mcp tool error' in error, got %q", err.Error())
	}
}

func TestGetTools_Execute_TypedArgs(t *testing.T) {
	// Typed args should be passed directly to the MCP server.
	mcpTools := []mcp.Tool{{Name: "typed", Description: ""}}
	var capturedArgs map[string]interface{}
	tr := &mockTransport{sendFn: func(_ context.Context, req mcp.JsonRpcRequest) (mcp.JsonRpcResponse, error) {
		if req.Method == "tools/list" {
			return toolsListResponse(mcpTools), nil
		}
		// Extract the arguments from params.
		var params struct {
			Arguments map[string]interface{} `json:"arguments"`
		}
		json.Unmarshal(req.Params, &params)
		capturedArgs = params.Arguments
		return callToolResponse("ok", false), nil
	}}
	ta := mcp.NewMcpToolAdapter(mcp.NewMcpClient(tr))
	tools, _ := ta.GetTools(context.Background())

	tools[0].Execute(context.Background(), map[string]interface{}{
		"count": float64(42),
		"name":  "alice",
		"flag":  true,
	})

	if capturedArgs["count"] != float64(42) {
		t.Errorf("expected count=42 (float64), got %v (%T)", capturedArgs["count"], capturedArgs["count"])
	}
	if capturedArgs["flag"] != true {
		t.Errorf("expected flag=true, got %v", capturedArgs["flag"])
	}
}

func TestGetTools_Execute_StringArgPassedThrough(t *testing.T) {
	// String args should be passed through as plain strings.
	mcpTools := []mcp.Tool{{Name: "plain", Description: ""}}
	var capturedArgs map[string]interface{}
	tr := &mockTransport{sendFn: func(_ context.Context, req mcp.JsonRpcRequest) (mcp.JsonRpcResponse, error) {
		if req.Method == "tools/list" {
			return toolsListResponse(mcpTools), nil
		}
		var params struct {
			Arguments map[string]interface{} `json:"arguments"`
		}
		json.Unmarshal(req.Params, &params)
		capturedArgs = params.Arguments
		return callToolResponse("ok", false), nil
	}}
	ta := mcp.NewMcpToolAdapter(mcp.NewMcpClient(tr))
	tools, _ := ta.GetTools(context.Background())

	tools[0].Execute(context.Background(), map[string]interface{}{"path": "/some/path"})
	if capturedArgs["path"] != "/some/path" {
		t.Errorf("expected '/some/path', got %v", capturedArgs["path"])
	}
}

func TestGetTools_ListError_Propagated(t *testing.T) {
	tr := &mockTransport{sendFn: func(_ context.Context, _ mcp.JsonRpcRequest) (mcp.JsonRpcResponse, error) {
		return mcp.JsonRpcResponse{}, &mcp.McpError{Code: -32603, Message: "internal error"}
	}}
	ta := mcp.NewMcpToolAdapter(mcp.NewMcpClient(tr))
	_, err := ta.GetTools(context.Background())
	if err == nil {
		t.Fatal("expected error propagated from ListTools failure")
	}
}

// ---------------------------------------------------------------------------
// McpClient.ListTools / CallTool via mock transport
// ---------------------------------------------------------------------------

func TestMcpClient_ListTools_Success(t *testing.T) {
	tools := []mcp.Tool{
		{Name: "search", Description: "Search the web"},
	}
	tr := &mockTransport{sendFn: func(_ context.Context, _ mcp.JsonRpcRequest) (mcp.JsonRpcResponse, error) {
		return toolsListResponse(tools), nil
	}}
	c := mcp.NewMcpClient(tr)
	result, err := c.ListTools(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result) != 1 || result[0].Name != "search" {
		t.Errorf("unexpected tools: %+v", result)
	}
}

func TestMcpClient_ListTools_Empty(t *testing.T) {
	tr := &mockTransport{sendFn: func(_ context.Context, _ mcp.JsonRpcRequest) (mcp.JsonRpcResponse, error) {
		return toolsListResponse(nil), nil
	}}
	c := mcp.NewMcpClient(tr)
	result, err := c.ListTools(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result) != 0 {
		t.Errorf("expected empty list, got %d tools", len(result))
	}
}

func TestMcpClient_CallTool_Success(t *testing.T) {
	tr := &mockTransport{sendFn: func(_ context.Context, _ mcp.JsonRpcRequest) (mcp.JsonRpcResponse, error) {
		return callToolResponse("result text", false), nil
	}}
	c := mcp.NewMcpClient(tr)
	result, err := c.CallTool(context.Background(), "echo", map[string]interface{}{"input": "hi"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Error("expected IsError=false")
	}
	if len(result.Content) == 0 || result.Content[0].Text != "result text" {
		t.Errorf("unexpected content: %+v", result.Content)
	}
}

func TestMcpClient_Close(t *testing.T) {
	closed := false
	tr := &mockTransport{
		sendFn: func(_ context.Context, _ mcp.JsonRpcRequest) (mcp.JsonRpcResponse, error) {
			return mcp.JsonRpcResponse{}, nil
		},
	}
	_ = closed
	c := mcp.NewMcpClient(tr)
	if err := c.Close(); err != nil {
		t.Errorf("unexpected Close() error: %v", err)
	}
}

// ---------------------------------------------------------------------------
// legacy Client via httptest
// ---------------------------------------------------------------------------

func legacyMCPServer(t *testing.T, tools []mcp.Tool) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Method string `json:"method"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		switch req.Method {
		case "tools/list":
			result, _ := json.Marshal(mcp.ListToolsResult{Tools: tools})
			json.NewEncoder(w).Encode(map[string]interface{}{"result": json.RawMessage(result)})
		case "tools/call":
			result, _ := json.Marshal(mcp.CallToolResult{
				Content: []mcp.ContentBlock{{Type: "text", Text: "called"}},
			})
			json.NewEncoder(w).Encode(map[string]interface{}{"result": json.RawMessage(result)})
		case "ping":
			json.NewEncoder(w).Encode(map[string]interface{}{"result": json.RawMessage(`{}`)})
		default:
			json.NewEncoder(w).Encode(map[string]interface{}{"error": map[string]interface{}{"code": -32601, "message": "method not found"}})
		}
	}))
}

func TestLegacyClient_ListTools(t *testing.T) {
	tools := []mcp.Tool{{Name: "calc", Description: "Calculator"}}
	srv := legacyMCPServer(t, tools)
	defer srv.Close()

	c := mcp.NewClient(srv.URL)
	result, err := c.ListTools(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result) != 1 || result[0].Name != "calc" {
		t.Errorf("unexpected tools: %+v", result)
	}
}

func TestLegacyClient_CallTool(t *testing.T) {
	srv := legacyMCPServer(t, nil)
	defer srv.Close()

	c := mcp.NewClient(srv.URL)
	result, err := c.CallTool(context.Background(), "calc", map[string]interface{}{"expr": "1+1"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Content) == 0 || result.Content[0].Text != "called" {
		t.Errorf("unexpected content: %+v", result.Content)
	}
}

func TestLegacyClient_Ping(t *testing.T) {
	srv := legacyMCPServer(t, nil)
	defer srv.Close()

	c := mcp.NewClient(srv.URL)
	if err := c.Ping(context.Background()); err != nil {
		t.Errorf("unexpected Ping error: %v", err)
	}
}

func TestLegacyClient_Error(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"error": map[string]interface{}{"code": -32600, "message": "invalid request"},
		})
	}))
	defer srv.Close()

	c := mcp.NewClient(srv.URL)
	_, err := c.ListTools(context.Background())
	if err == nil {
		t.Fatal("expected error for server error response")
	}
	if !strings.Contains(err.Error(), "invalid request") {
		t.Errorf("expected 'invalid request' in error, got %q", err.Error())
	}
}

// ---------------------------------------------------------------------------
// ConnectHTTP — initialize handshake via httptest
// ---------------------------------------------------------------------------

func TestConnectHTTP_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Accept the initialize handshake.
		resp := mcp.JsonRpcResponse{JSONRPC: "2.0", Result: json.RawMessage(`{"protocolVersion":"2024-11-05"}`)}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	c, err := mcp.ConnectHTTP(context.Background(), srv.URL, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if c == nil {
		t.Fatal("expected non-nil McpClient")
	}
	c.Close()
}

func TestConnectHTTP_InitializeError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := mcp.JsonRpcResponse{
			JSONRPC: "2.0",
			Error:   &mcp.JsonRpcError{Code: -32600, Message: "not supported"},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	_, err := mcp.ConnectHTTP(context.Background(), srv.URL, nil)
	if err == nil {
		t.Fatal("expected error when initialize fails")
	}
	if !strings.Contains(err.Error(), "mcp initialize") {
		t.Errorf("expected 'mcp initialize' in error, got %q", err.Error())
	}
}
