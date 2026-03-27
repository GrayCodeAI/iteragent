package iteragent

import (
	"context"
	"encoding/json"
	"testing"
)

// ── Schema generator ──────────────────────────────────────────────────────────

func TestToNativeDefinition_NoParams(t *testing.T) {
	tool := Tool{
		Name:        "list_files",
		Description: "Lists files in a directory.",
		Execute:     nil,
	}
	def := tool.ToNativeDefinition()
	if def.Name != "list_files" {
		t.Errorf("Name = %q, want %q", def.Name, "list_files")
	}
	if def.InputSchema.Type != "object" {
		t.Errorf("InputSchema.Type = %q, want %q", def.InputSchema.Type, "object")
	}
	if len(def.InputSchema.Properties) != 0 {
		t.Errorf("expected 0 properties for tool with no Params block")
	}
}

func TestToNativeDefinition_WithParams(t *testing.T) {
	tool := Tool{
		Name: "read_file",
		Description: `Read the contents of a file.

Params:
  path (string, required): the path to read
  encoding (string): text encoding (default utf-8)`,
		Execute: nil,
	}
	def := tool.ToNativeDefinition()

	if len(def.InputSchema.Properties) != 2 {
		t.Fatalf("expected 2 properties, got %d", len(def.InputSchema.Properties))
	}
	pathProp, ok := def.InputSchema.Properties["path"]
	if !ok {
		t.Fatal("expected 'path' property")
	}
	if pathProp.Type != "string" {
		t.Errorf("path.Type = %q, want %q", pathProp.Type, "string")
	}
	if len(def.InputSchema.Required) != 1 || def.InputSchema.Required[0] != "path" {
		t.Errorf("Required = %v, want [path]", def.InputSchema.Required)
	}
}

func TestToolsToNativeDefinitions_JSON(t *testing.T) {
	tools := []Tool{
		{Name: "ping", Description: "Send a ping.", Execute: nil},
		{Name: "echo", Description: "Echo text.\nParams:\n  text (string, required): text to echo", Execute: nil},
	}
	data, err := NativeDefinitionsJSON(tools)
	if err != nil {
		t.Fatalf("NativeDefinitionsJSON: %v", err)
	}
	var defs []NativeToolDefinition
	if err := json.Unmarshal(data, &defs); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(defs) != 2 {
		t.Errorf("expected 2 defs, got %d", len(defs))
	}
}

// ── parseNativeToolArgs ───────────────────────────────────────────────────────

func TestParseNativeToolArgs_JSON(t *testing.T) {
	raw := json.RawMessage(`{"path": "/tmp/foo", "count": 3}`)
	args, err := parseNativeToolArgs(raw, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if args["path"] != "/tmp/foo" {
		t.Errorf("path = %v, want /tmp/foo", args["path"])
	}
	if args["count"] != float64(3) {
		t.Errorf("count = %v (%T), want 3.0", args["count"], args["count"])
	}
}

func TestParseNativeToolArgs_String(t *testing.T) {
	args, err := parseNativeToolArgs(nil, `{"name": "alice", "active": true}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if args["name"] != "alice" {
		t.Errorf("name = %v, want alice", args["name"])
	}
	if args["active"] != true {
		t.Errorf("active = %v, want true", args["active"])
	}
}

func TestParseNativeToolArgs_Empty(t *testing.T) {
	args, err := parseNativeToolArgs(nil, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(args) != 0 {
		t.Errorf("expected empty map for empty input, got %v", args)
	}
}

func TestParseNativeToolArgs_Invalid(t *testing.T) {
	_, err := parseNativeToolArgs(nil, "not-json")
	if err == nil {
		t.Error("expected error for invalid JSON, got nil")
	}
}

// ── NativeToolExecutor ────────────────────────────────────────────────────────

func TestNativeToolExecutor_SingleCall(t *testing.T) {
	tools := map[string]Tool{
		"echo": {
			Name:        "echo",
			Description: "echo",
			Execute: func(_ context.Context, args map[string]interface{}) (string, error) {
				return ArgStr(args, "text"), nil
			},
		},
	}
	exec := NewNativeToolExecutor(tools, false)
	results := exec.Execute(context.Background(), []NativeToolCallRequest{
		{ID: "c1", Name: "echo", ArgsJSON: json.RawMessage(`{"text":"hello"}`)},
	})
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].Content != "hello" {
		t.Errorf("content = %q, want %q", results[0].Content, "hello")
	}
	if results[0].IsError {
		t.Error("unexpected error flag")
	}
}

func TestNativeToolExecutor_UnknownTool(t *testing.T) {
	exec := NewNativeToolExecutor(map[string]Tool{}, false)
	results := exec.Execute(context.Background(), []NativeToolCallRequest{
		{ID: "x", Name: "no_such_tool", ArgsJSON: json.RawMessage(`{}`)},
	})
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if !results[0].IsError {
		t.Error("expected IsError=true for unknown tool")
	}
}

func TestNativeToolExecutor_Parallel(t *testing.T) {
	var calls int
	var mu = make(chan struct{}, 1)
	mu <- struct{}{}
	tools := map[string]Tool{
		"count": {
			Name: "count",
			Execute: func(_ context.Context, _ map[string]interface{}) (string, error) {
				<-mu
				calls++
				mu <- struct{}{}
				return "ok", nil
			},
		},
	}
	exec := NewNativeToolExecutor(tools, true)
	batch := make([]NativeToolCallRequest, 5)
	for i := range batch {
		batch[i] = NativeToolCallRequest{Name: "count", ArgsJSON: json.RawMessage(`{}`)}
	}
	results := exec.Execute(context.Background(), batch)
	if len(results) != 5 {
		t.Fatalf("expected 5 results, got %d", len(results))
	}
	if calls != 5 {
		t.Errorf("expected 5 calls, got %d", calls)
	}
}

// ── ArgStr helper ─────────────────────────────────────────────────────────────

func TestArgStr_String(t *testing.T) {
	args := map[string]interface{}{"key": "value"}
	if got := ArgStr(args, "key"); got != "value" {
		t.Errorf("ArgStr = %q, want %q", got, "value")
	}
}

func TestArgStr_Number(t *testing.T) {
	args := map[string]interface{}{"n": float64(42)}
	if got := ArgStr(args, "n"); got != "42" {
		t.Errorf("ArgStr = %q, want %q", got, "42")
	}
}

func TestArgStr_Missing(t *testing.T) {
	args := map[string]interface{}{}
	if got := ArgStr(args, "missing"); got != "" {
		t.Errorf("ArgStr = %q, want empty for missing key", got)
	}
}

func TestArgStr_Bool(t *testing.T) {
	args := map[string]interface{}{"flag": true}
	if got := ArgStr(args, "flag"); got != "true" {
		t.Errorf("ArgStr = %q, want %q", got, "true")
	}
}
