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
// JsonRpcRequest / JsonRpcResponse / JsonRpcError / McpError
// ---------------------------------------------------------------------------

func TestJsonRpcRequest_Serialization(t *testing.T) {
	req := mcp.JsonRpcRequest{
		JSONRPC: "2.0",
		Method:  "tools/list",
		ID:      1,
	}
	b, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal error: %v", err)
	}
	var m map[string]interface{}
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if m["jsonrpc"] != "2.0" {
		t.Errorf("expected jsonrpc='2.0', got %v", m["jsonrpc"])
	}
	if m["method"] != "tools/list" {
		t.Errorf("expected method='tools/list', got %v", m["method"])
	}
}

func TestJsonRpcRequest_WithParams(t *testing.T) {
	params, _ := json.Marshal(map[string]string{"name": "echo"})
	req := mcp.JsonRpcRequest{
		JSONRPC: "2.0",
		Method:  "tools/call",
		Params:  params,
		ID:      42,
	}
	b, _ := json.Marshal(req)
	var m map[string]interface{}
	json.Unmarshal(b, &m)
	if m["params"] == nil {
		t.Error("expected params in serialized request")
	}
}

func TestJsonRpcResponse_NoError(t *testing.T) {
	raw := `{"jsonrpc":"2.0","result":{"ok":true},"id":1}`
	var resp mcp.JsonRpcResponse
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if resp.Error != nil {
		t.Errorf("expected nil error, got %+v", resp.Error)
	}
	if resp.JSONRPC != "2.0" {
		t.Errorf("expected jsonrpc='2.0', got %q", resp.JSONRPC)
	}
}

func TestJsonRpcResponse_WithError(t *testing.T) {
	raw := `{"jsonrpc":"2.0","error":{"code":-32600,"message":"invalid request"},"id":1}`
	var resp mcp.JsonRpcResponse
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if resp.Error == nil {
		t.Fatal("expected non-nil error")
	}
	if resp.Error.Code != -32600 {
		t.Errorf("expected code -32600, got %d", resp.Error.Code)
	}
	if resp.Error.Message != "invalid request" {
		t.Errorf("expected message 'invalid request', got %q", resp.Error.Message)
	}
}

func TestMcpError_Error(t *testing.T) {
	err := &mcp.McpError{Code: -32601, Message: "method not found"}
	if err.Error() != "method not found" {
		t.Errorf("expected 'method not found', got %q", err.Error())
	}
}

// ---------------------------------------------------------------------------
// HTTPTransport
// ---------------------------------------------------------------------------

// jsonRPCHandler returns an httptest.Server that echoes a JSON-RPC response.
func jsonRPCHandler(t *testing.T, result interface{}) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req mcp.JsonRpcRequest
		json.NewDecoder(r.Body).Decode(&req)
		resultBytes, _ := json.Marshal(result)
		resp := mcp.JsonRpcResponse{
			JSONRPC: "2.0",
			Result:  resultBytes,
			ID:      req.ID,
		}
		json.NewEncoder(w).Encode(resp)
	}))
}

// jsonRPCErrorHandler returns a server that always returns an RPC error.
func jsonRPCErrorHandler(t *testing.T, code int, msg string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]interface{}{
			"jsonrpc": "2.0",
			"id":      1,
			"error": map[string]interface{}{
				"code":    code,
				"message": msg,
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
}

func TestHTTPTransport_Send_Success(t *testing.T) {
	srv := jsonRPCHandler(t, map[string]string{"status": "ok"})
	defer srv.Close()

	tr := mcp.NewHTTPTransport(srv.URL, nil)
	resp, err := tr.Send(context.Background(), mcp.JsonRpcRequest{
		JSONRPC: "2.0",
		Method:  "ping",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Result == nil {
		t.Error("expected non-nil result")
	}
}

func TestHTTPTransport_Send_RPCError(t *testing.T) {
	srv := jsonRPCErrorHandler(t, -32601, "method not found")
	defer srv.Close()

	tr := mcp.NewHTTPTransport(srv.URL, nil)
	_, err := tr.Send(context.Background(), mcp.JsonRpcRequest{
		JSONRPC: "2.0",
		Method:  "unknown/method",
	})
	if err == nil {
		t.Fatal("expected error for RPC error response")
	}
	if !strings.Contains(err.Error(), "method not found") {
		t.Errorf("expected 'method not found' in error, got %q", err.Error())
	}
}

func TestHTTPTransport_Send_InvalidJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("not json {{{{"))
	}))
	defer srv.Close()

	tr := mcp.NewHTTPTransport(srv.URL, nil)
	_, err := tr.Send(context.Background(), mcp.JsonRpcRequest{
		JSONRPC: "2.0",
		Method:  "ping",
	})
	if err == nil {
		t.Fatal("expected error for invalid JSON response")
	}
}

func TestHTTPTransport_Send_CustomHeaders(t *testing.T) {
	var capturedAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedAuth = r.Header.Get("Authorization")
		resp := mcp.JsonRpcResponse{JSONRPC: "2.0", Result: json.RawMessage(`{}`)}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	tr := mcp.NewHTTPTransport(srv.URL, map[string]string{
		"Authorization": "Bearer test-token",
	})
	_, _ = tr.Send(context.Background(), mcp.JsonRpcRequest{JSONRPC: "2.0", Method: "ping"})

	if capturedAuth != "Bearer test-token" {
		t.Errorf("expected 'Bearer test-token', got %q", capturedAuth)
	}
}

func TestHTTPTransport_Send_ContentTypeHeader(t *testing.T) {
	var capturedCT string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedCT = r.Header.Get("Content-Type")
		resp := mcp.JsonRpcResponse{JSONRPC: "2.0", Result: json.RawMessage(`{}`)}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	tr := mcp.NewHTTPTransport(srv.URL, nil)
	_, _ = tr.Send(context.Background(), mcp.JsonRpcRequest{JSONRPC: "2.0", Method: "ping"})

	if capturedCT != "application/json" {
		t.Errorf("expected Content-Type 'application/json', got %q", capturedCT)
	}
}

func TestHTTPTransport_Send_IDAutoAssigned(t *testing.T) {
	var capturedID interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req mcp.JsonRpcRequest
		json.NewDecoder(r.Body).Decode(&req)
		capturedID = req.ID
		resp := mcp.JsonRpcResponse{JSONRPC: "2.0", Result: json.RawMessage(`{}`), ID: req.ID}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	tr := mcp.NewHTTPTransport(srv.URL, nil)
	// Send without ID — should be auto-assigned
	_, _ = tr.Send(context.Background(), mcp.JsonRpcRequest{JSONRPC: "2.0", Method: "ping"})

	if capturedID == nil {
		t.Error("expected auto-assigned ID, got nil")
	}
}

func TestHTTPTransport_Send_IncrementalIDs(t *testing.T) {
	var ids []interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req mcp.JsonRpcRequest
		json.NewDecoder(r.Body).Decode(&req)
		ids = append(ids, req.ID)
		resp := mcp.JsonRpcResponse{JSONRPC: "2.0", Result: json.RawMessage(`{}`), ID: req.ID}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	tr := mcp.NewHTTPTransport(srv.URL, nil)
	for i := 0; i < 3; i++ {
		_, _ = tr.Send(context.Background(), mcp.JsonRpcRequest{JSONRPC: "2.0", Method: "ping"})
	}

	if len(ids) != 3 {
		t.Fatalf("expected 3 requests, got %d", len(ids))
	}
	// IDs should be increasing numbers (float64 after JSON decode)
	id0, ok0 := ids[0].(float64)
	id1, ok1 := ids[1].(float64)
	id2, ok2 := ids[2].(float64)
	if !ok0 || !ok1 || !ok2 {
		t.Fatalf("expected numeric IDs, got %v %v %v", ids[0], ids[1], ids[2])
	}
	if !(id0 < id1 && id1 < id2) {
		t.Errorf("expected increasing IDs, got %.0f %.0f %.0f", id0, id1, id2)
	}
}

func TestHTTPTransport_Send_ContextCancelled(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	tr := mcp.NewHTTPTransport(srv.URL, nil)
	_, err := tr.Send(ctx, mcp.JsonRpcRequest{JSONRPC: "2.0", Method: "ping"})
	if err == nil {
		t.Fatal("expected error from cancelled context")
	}
}

func TestHTTPTransport_Close(t *testing.T) {
	tr := mcp.NewHTTPTransport("http://localhost:9999", nil)
	if err := tr.Close(); err != nil {
		t.Errorf("expected Close() to return nil, got %v", err)
	}
}

func TestHTTPTransport_Send_MethodInBody(t *testing.T) {
	var capturedMethod string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req mcp.JsonRpcRequest
		json.NewDecoder(r.Body).Decode(&req)
		capturedMethod = req.Method
		resp := mcp.JsonRpcResponse{JSONRPC: "2.0", Result: json.RawMessage(`{}`)}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	tr := mcp.NewHTTPTransport(srv.URL, nil)
	_, _ = tr.Send(context.Background(), mcp.JsonRpcRequest{
		JSONRPC: "2.0",
		Method:  "tools/call",
	})

	if capturedMethod != "tools/call" {
		t.Errorf("expected method 'tools/call', got %q", capturedMethod)
	}
}

func TestNewHTTPTransport_NilHeaders(t *testing.T) {
	srv := jsonRPCHandler(t, map[string]string{"ok": "true"})
	defer srv.Close()

	tr := mcp.NewHTTPTransport(srv.URL, nil)
	_, err := tr.Send(context.Background(), mcp.JsonRpcRequest{JSONRPC: "2.0", Method: "ping"})
	if err != nil {
		t.Errorf("unexpected error with nil headers: %v", err)
	}
}
