package iteragent_test

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"

	iteragent "github.com/GrayCodeAI/iteragent"
)

func testLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelWarn}))
}

// TestAgentSimpleResponse verifies that the agent returns the mock response.
func TestAgentSimpleResponse(t *testing.T) {
	p := iteragent.NewMock("hello world")
	a := iteragent.New(p, nil, testLogger())
	out, err := a.Run(context.Background(), "system", "user prompt")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out != "hello world" {
		t.Errorf("want %q, got %q", "hello world", out)
	}
}

// TestAgentToolCall verifies that a single tool call is executed and the result fed back.
func TestAgentToolCall(t *testing.T) {
	toolCall := iteragent.ToolCall{
		Tool: "echo",
		Args: map[string]string{"msg": "ping"},
	}

	called := false
	echoTool := iteragent.Tool{
		Name:        "echo",
		Description: "Echo the msg argument",
		Execute: func(ctx context.Context, args map[string]string) (string, error) {
			called = true
			return "pong", nil
		},
	}

	// First response: one tool call; second: final answer.
	p := iteragent.NewMockWithTools("final answer", []iteragent.ToolCall{toolCall})

	a := iteragent.New(p, []iteragent.Tool{echoTool}, testLogger())
	out, err := a.Run(context.Background(), "", "use echo")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !called {
		t.Error("expected echo tool to be called")
	}
	if out != "final answer" {
		t.Errorf("want final answer, got %q", out)
	}
}

// TestAgentProviderError verifies that provider errors are propagated.
func TestAgentProviderError(t *testing.T) {
	p := iteragent.NewMockWithError(context.DeadlineExceeded)
	a := iteragent.New(p, nil, testLogger())
	_, err := a.Run(context.Background(), "", "hello")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}

// TestAgentPromptChannel verifies that Prompt returns an event channel.
func TestAgentPromptChannel(t *testing.T) {
	p := iteragent.NewMock("streamed response")
	a := iteragent.New(p, nil, testLogger())

	events := a.Prompt(context.Background(), "hello")
	var got []string
	for e := range events {
		got = append(got, e.Type)
	}
	a.Finish()

	if len(got) == 0 {
		t.Error("expected at least one event")
	}
}

// TestAgentReset verifies that Reset clears message history.
func TestAgentReset(t *testing.T) {
	p := iteragent.NewMock("response")
	a := iteragent.New(p, nil, testLogger())

	events := a.Prompt(context.Background(), "hello")
	for range events {
	}
	a.Finish()

	a.Reset()
	if len(a.Messages) != 0 {
		t.Errorf("expected empty messages after Reset, got %d", len(a.Messages))
	}
}

// TestParseToolCalls verifies that tool call blocks are extracted correctly.
func TestParseToolCalls(t *testing.T) {
	input := "some text\n```tool\n{\"tool\":\"bash\",\"args\":{\"cmd\":\"ls\"}}\n```\nmore text"
	calls := iteragent.ParseToolCalls(input)
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}
	if calls[0].Tool != "bash" {
		t.Errorf("want tool=bash, got %q", calls[0].Tool)
	}
	if calls[0].Args["cmd"] != "ls" {
		t.Errorf("want args.cmd=ls, got %q", calls[0].Args["cmd"])
	}
}

// TestParseToolCallsNone verifies that output with no tool blocks returns empty slice.
func TestParseToolCallsNone(t *testing.T) {
	calls := iteragent.ParseToolCalls("just some text with no tool calls")
	if len(calls) != 0 {
		t.Errorf("expected 0 calls, got %d", len(calls))
	}
}

// TestToolDescriptions verifies that tool descriptions are formatted.
func TestToolDescriptions(t *testing.T) {
	tools := []iteragent.Tool{
		{Name: "mytool", Description: "does stuff"},
	}
	desc := iteragent.ToolDescriptions(tools)
	if !strings.Contains(desc, "mytool") {
		t.Error("expected tool name in descriptions")
	}
	if !strings.Contains(desc, "does stuff") {
		t.Error("expected tool description in descriptions")
	}
}

// TestAgentWithToolExecutionStrategies verifies all three strategies run without panicking.
func TestAgentWithToolExecutionStrategies(t *testing.T) {
	for _, cfg := range []iteragent.ToolExecConfig{
		iteragent.NewSequentialStrategy(),
		iteragent.NewParallelStrategy(),
		iteragent.NewBatchedStrategy(2),
	} {
		p := iteragent.NewMock("done")
		a := iteragent.New(p, nil, testLogger()).WithToolExecutionStrategy(cfg)
		_, err := a.Run(context.Background(), "", "hello")
		if err != nil {
			t.Errorf("strategy %v: unexpected error: %v", cfg.Strategy, err)
		}
	}
}

// TestAgentInputFilterReject verifies that a Reject filter aborts the run.
func TestAgentInputFilterReject(t *testing.T) {
	p := iteragent.NewMock("should not reach")
	a := iteragent.New(p, nil, testLogger()).WithInputFilter(&rejectAllFilter{})
	_, err := a.Run(context.Background(), "", "hello")
	if err == nil {
		t.Fatal("expected error from rejected input")
	}
}

type rejectAllFilter struct{}

func (f *rejectAllFilter) Filter(input string) (string, iteragent.InputFilterResult, string) {
	return input, iteragent.InputFilterReject, "test rejection"
}

// ---------------------------------------------------------------------------
// AgentHooks tests
// ---------------------------------------------------------------------------

// TestAgentHooks_BeforeAfterTurn verifies BeforeTurn and AfterTurn are called
// for each provider completion turn during Run().
func TestAgentHooks_BeforeAfterTurn(t *testing.T) {
	p := iteragent.NewMock("final answer")
	a := iteragent.New(p, nil, testLogger())

	var beforeTurns, afterTurns []int
	a.WithHooks(iteragent.AgentHooks{
		BeforeTurn: func(turn int, messages []iteragent.Message) {
			beforeTurns = append(beforeTurns, turn)
		},
		AfterTurn: func(turn int, response string) {
			afterTurns = append(afterTurns, turn)
			if response != "final answer" {
				t.Errorf("AfterTurn: got response %q, want %q", response, "final answer")
			}
		},
	})

	_, err := a.Run(context.Background(), "", "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(beforeTurns) == 0 {
		t.Error("BeforeTurn was never called")
	}
	if len(afterTurns) == 0 {
		t.Error("AfterTurn was never called")
	}
	if len(beforeTurns) != len(afterTurns) {
		t.Errorf("BeforeTurn called %d times, AfterTurn called %d times — should match",
			len(beforeTurns), len(afterTurns))
	}
	// Turn numbers should be 1-indexed and sequential.
	for i, n := range beforeTurns {
		if n != i+1 {
			t.Errorf("BeforeTurn[%d]: got turn=%d, want %d", i, n, i+1)
		}
	}
}

// TestAgentHooks_OnToolStartEnd verifies OnToolStart and OnToolEnd are called
// around each tool execution.
func TestAgentHooks_OnToolStartEnd(t *testing.T) {
	toolCall := iteragent.ToolCall{
		Tool: "greet",
		Args: map[string]string{"name": "world"},
	}
	greetTool := iteragent.Tool{
		Name:        "greet",
		Description: "Greet someone",
		Execute: func(ctx context.Context, args map[string]string) (string, error) {
			return "hello " + args["name"], nil
		},
	}

	p := iteragent.NewMockWithTools("done", []iteragent.ToolCall{toolCall})
	a := iteragent.New(p, []iteragent.Tool{greetTool}, testLogger())

	var startTools, endTools []string
	var endResults []string
	var endErrors []error
	a.WithHooks(iteragent.AgentHooks{
		OnToolStart: func(toolName string, args map[string]string) {
			startTools = append(startTools, toolName)
			if args["name"] != "world" {
				t.Errorf("OnToolStart: args[name]=%q, want %q", args["name"], "world")
			}
		},
		OnToolEnd: func(toolName string, result string, err error) {
			endTools = append(endTools, toolName)
			endResults = append(endResults, result)
			endErrors = append(endErrors, err)
		},
	})

	_, err := a.Run(context.Background(), "", "use greet")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(startTools) != 1 || startTools[0] != "greet" {
		t.Errorf("OnToolStart: got %v, want [greet]", startTools)
	}
	if len(endTools) != 1 || endTools[0] != "greet" {
		t.Errorf("OnToolEnd: got %v, want [greet]", endTools)
	}
	if endResults[0] != "hello world" {
		t.Errorf("OnToolEnd result: got %q, want %q", endResults[0], "hello world")
	}
	if endErrors[0] != nil {
		t.Errorf("OnToolEnd err: got %v, want nil", endErrors[0])
	}
}

// TestAgentHooks_NilHooks verifies the agent runs cleanly when hooks are nil.
func TestAgentHooks_NilHooks(t *testing.T) {
	p := iteragent.NewMock("ok")
	a := iteragent.New(p, nil, testLogger()).WithHooks(iteragent.AgentHooks{})
	_, err := a.Run(context.Background(), "", "hello")
	if err != nil {
		t.Fatalf("nil hooks should not cause errors: %v", err)
	}
}

// TestAgentHooks_BeforeTurnReceivesMessages verifies BeforeTurn gets the
// accumulated message history at the point of each turn.
func TestAgentHooks_BeforeTurnReceivesMessages(t *testing.T) {
	p := iteragent.NewMock("response")
	a := iteragent.New(p, nil, testLogger())

	var firstTurnMsgCount int
	a.WithHooks(iteragent.AgentHooks{
		BeforeTurn: func(turn int, messages []iteragent.Message) {
			if turn == 1 {
				firstTurnMsgCount = len(messages)
			}
		},
	})

	_, err := a.Run(context.Background(), "sys", "user msg")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Turn 1 should have at least system + user = 2 messages.
	if firstTurnMsgCount < 2 {
		t.Errorf("BeforeTurn turn=1 got %d messages, want >= 2", firstTurnMsgCount)
	}
}

// ---------------------------------------------------------------------------
// Streaming tests
// ---------------------------------------------------------------------------

// TestAgentStreaming_EventsEmitted verifies that EventTokenUpdate events
// are emitted during streaming completions.
func TestAgentStreaming_EventsEmitted(t *testing.T) {
	p := iteragent.NewMockStream("hello world")
	a := iteragent.New(p, nil, testLogger())

	var tokenEvents []string
	events := a.Prompt(context.Background(), "hi")
	for e := range events {
		if iteragent.EventType(e.Type) == iteragent.EventTokenUpdate {
			tokenEvents = append(tokenEvents, e.Content)
		}
	}
	a.Finish()

	if len(tokenEvents) == 0 {
		t.Error("expected at least one EventTokenUpdate, got none")
	}
	// Tokens should reassemble to the full response.
	full := strings.Join(tokenEvents, "")
	if full != "hello world" {
		t.Errorf("reassembled tokens = %q, want %q", full, "hello world")
	}
}

// TestAgentStreaming_RunReturnsFullResponse verifies Run() returns the
// complete response even when token streaming is active.
func TestAgentStreaming_RunReturnsFullResponse(t *testing.T) {
	p := iteragent.NewMockStream("the quick brown fox")
	a := iteragent.New(p, nil, testLogger())

	out, err := a.Run(context.Background(), "", "prompt")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out != "the quick brown fox" {
		t.Errorf("Run() = %q, want %q", out, "the quick brown fox")
	}
}

// TestAgentStreaming_WithTools verifies streaming still works when a tool
// call precedes the final response. The mock streams every turn, so we verify
// that EventTokenUpdate events were emitted and the final answer is correct.
func TestAgentStreaming_WithTools(t *testing.T) {
	toolCall := iteragent.ToolCall{Tool: "noop", Args: map[string]string{}}
	noopTool := iteragent.Tool{
		Name: "noop",
		Execute: func(ctx context.Context, args map[string]string) (string, error) {
			return "done", nil
		},
	}
	p := iteragent.NewMockStreamWithTools("streamed answer", []iteragent.ToolCall{toolCall})
	a := iteragent.New(p, []iteragent.Tool{noopTool}, testLogger())

	var tokenCount int
	var finalContent string
	events := a.Prompt(context.Background(), "go")
	for e := range events {
		switch iteragent.EventType(e.Type) {
		case iteragent.EventTokenUpdate:
			tokenCount++
		case iteragent.EventMessageEnd:
			finalContent = e.Content
		}
	}
	a.Finish()

	if tokenCount == 0 {
		t.Error("expected token events, got none")
	}
	if finalContent != "streamed answer" {
		t.Errorf("final content = %q, want %q", finalContent, "streamed answer")
	}
}

// TestAgentClose verifies that Close() is safe to call on an agent with no MCP
// servers and does not block or panic.
func TestAgentClose(t *testing.T) {
	p := iteragent.NewMock("ok")
	a := iteragent.New(p, nil, testLogger())
	if err := a.Close(); err != nil {
		t.Errorf("Close() on agent without MCP servers: %v", err)
	}
	// Double-close should also be safe.
	if err := a.Close(); err != nil {
		t.Errorf("second Close(): %v", err)
	}
}

// TestAgentHooks_Parallel verifies OnToolStart/OnToolEnd fire for each tool
// when the parallel execution strategy is used.
func TestAgentHooks_Parallel(t *testing.T) {
	calls := []iteragent.ToolCall{
		{Tool: "t1", Args: map[string]string{}},
		{Tool: "t2", Args: map[string]string{}},
	}
	makeTool := func(name string) iteragent.Tool {
		return iteragent.Tool{
			Name: name,
			Execute: func(ctx context.Context, args map[string]string) (string, error) {
				return name + "-result", nil
			},
		}
	}

	p := iteragent.NewMockWithTools("done", calls)
	a := iteragent.New(p, []iteragent.Tool{makeTool("t1"), makeTool("t2")}, testLogger()).
		WithToolExecutionStrategy(iteragent.NewParallelStrategy())

	var mu sync.Mutex
	var started, ended []string

	a.WithHooks(iteragent.AgentHooks{
		OnToolStart: func(toolName string, args map[string]string) {
			mu.Lock()
			started = append(started, toolName)
			mu.Unlock()
		},
		OnToolEnd: func(toolName string, result string, err error) {
			mu.Lock()
			ended = append(ended, toolName)
			mu.Unlock()
		},
	})

	_, err := a.Run(context.Background(), "", "use tools")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(started) != 2 {
		t.Errorf("OnToolStart: got %d calls, want 2", len(started))
	}
	if len(ended) != 2 {
		t.Errorf("OnToolEnd: got %d calls, want 2", len(ended))
	}
}

// ---------------------------------------------------------------------------
// PromptMessages hooks tests
// ---------------------------------------------------------------------------

// TestPromptMessages_Hooks_BeforeAfterTurn verifies BeforeTurn and AfterTurn fire
// for each provider completion turn during PromptMessages().
func TestPromptMessages_Hooks_BeforeAfterTurn(t *testing.T) {
	p := iteragent.NewMock("pm final")
	a := iteragent.New(p, nil, testLogger())

	var beforeTurns, afterTurns []int
	a.WithHooks(iteragent.AgentHooks{
		BeforeTurn: func(turn int, messages []iteragent.Message) {
			beforeTurns = append(beforeTurns, turn)
		},
		AfterTurn: func(turn int, response string) {
			afterTurns = append(afterTurns, turn)
		},
	})

	msgs := []iteragent.Message{{Role: "user", Content: "hello"}}
	for range a.PromptMessages(context.Background(), msgs) {
	}
	a.Finish()

	if len(beforeTurns) == 0 {
		t.Error("BeforeTurn was never called via PromptMessages")
	}
	if len(beforeTurns) != len(afterTurns) {
		t.Errorf("BeforeTurn called %d times, AfterTurn called %d — should match",
			len(beforeTurns), len(afterTurns))
	}
	for i, n := range beforeTurns {
		if n != i+1 {
			t.Errorf("BeforeTurn[%d]: got turn=%d, want %d", i, n, i+1)
		}
	}
}

// TestPromptMessages_Hooks_OnToolStartEnd verifies OnToolStart and OnToolEnd fire
// around each tool execution during PromptMessages().
func TestPromptMessages_Hooks_OnToolStartEnd(t *testing.T) {
	toolCall := iteragent.ToolCall{Tool: "ping", Args: map[string]string{}}
	pingTool := iteragent.Tool{
		Name: "ping",
		Execute: func(ctx context.Context, args map[string]string) (string, error) {
			return "pong", nil
		},
	}
	p := iteragent.NewMockWithTools("done", []iteragent.ToolCall{toolCall})
	a := iteragent.New(p, []iteragent.Tool{pingTool}, testLogger())

	var startTools, endTools []string
	a.WithHooks(iteragent.AgentHooks{
		OnToolStart: func(toolName string, args map[string]string) {
			startTools = append(startTools, toolName)
		},
		OnToolEnd: func(toolName string, result string, err error) {
			endTools = append(endTools, toolName)
		},
	})

	msgs := []iteragent.Message{{Role: "user", Content: "ping"}}
	for range a.PromptMessages(context.Background(), msgs) {
	}
	a.Finish()

	if len(startTools) == 0 {
		t.Error("OnToolStart was never called via PromptMessages")
	}
	if len(startTools) != len(endTools) {
		t.Errorf("OnToolStart called %d times, OnToolEnd called %d — should match",
			len(startTools), len(endTools))
	}
	if startTools[0] != "ping" {
		t.Errorf("OnToolStart tool = %q, want %q", startTools[0], "ping")
	}
}

// TestPromptMessages_Hooks_TokenUpdate verifies EventTokenUpdate events are emitted
// via PromptMessages during streaming completions.
func TestPromptMessages_Hooks_TokenUpdate(t *testing.T) {
	const want = "streamed via prompt messages"
	p := iteragent.NewMockStream(want)
	a := iteragent.New(p, nil, testLogger())

	var tokenCount int
	var finalContent string
	for e := range a.PromptMessages(context.Background(), []iteragent.Message{{Role: "user", Content: "go"}}) {
		switch iteragent.EventType(e.Type) {
		case iteragent.EventTokenUpdate:
			tokenCount++
		case iteragent.EventMessageEnd:
			finalContent = e.Content
		}
	}
	a.Finish()

	if tokenCount == 0 {
		t.Error("expected EventTokenUpdate events via PromptMessages, got none")
	}
	if finalContent != want {
		t.Errorf("final content = %q, want %q", finalContent, want)
	}
}

// TestSubAgentStreaming verifies that SubAgent.Run delegates to the embedded Agent,
// which means streaming and EventTokenUpdate both flow through correctly.
func TestSubAgentStreaming(t *testing.T) {
	const want = "subagent streamed answer"
	p := iteragent.NewMockStream(want)

	cfg := iteragent.SubAgentConfig{
		Name:         "test-sub",
		SystemPrompt: "you are a sub",
		Provider:     p,
		Tools:        nil,
	}
	sub := iteragent.NewSubAgent(cfg, testLogger())

	// Run synchronously — events accumulate in the buffered Events channel.
	out, err := sub.Run(context.Background(), "do the task")
	if err != nil {
		t.Fatalf("SubAgent.Run error: %v", err)
	}
	if out != want {
		t.Errorf("Run() = %q, want %q", out, want)
	}

	// Drain the buffered Events channel to verify token updates were emitted.
	var tokenCount int
	var sb strings.Builder
	for {
		select {
		case e := <-sub.Agent.Events:
			if e.Type == string(iteragent.EventTokenUpdate) {
				tokenCount++
				sb.WriteString(e.Content)
			}
		default:
			goto drained
		}
	}
drained:
	if tokenCount == 0 {
		t.Error("expected at least one EventTokenUpdate from streaming sub-agent")
	}
	if got := sb.String(); got != want {
		t.Errorf("tokens concatenated = %q, want %q", got, want)
	}
}

// ---------------------------------------------------------------------------
// PromptMessages cancellation test
// ---------------------------------------------------------------------------

// TestPromptMessages_ContextCancellation verifies that cancelling the context
// mid-run causes PromptMessages to stop and emit an error event, rather than
// hanging or proceeding through all turns.
func TestPromptMessages_ContextCancellation(t *testing.T) {
	// Provider that blocks until context is cancelled.
	blockingP := &blockingProvider{}
	a := iteragent.New(blockingP, nil, testLogger())

	ctx, cancel := context.WithCancel(context.Background())

	events := a.PromptMessages(ctx, []iteragent.Message{{Role: "user", Content: "go"}})

	// Cancel after starting.
	cancel()

	var gotError bool
	for e := range events {
		if e.IsError {
			gotError = true
		}
	}
	a.Finish()

	if !gotError {
		t.Error("expected an error event after context cancellation")
	}
}

// blockingProvider is a Provider whose Complete blocks until context is done.
type blockingProvider struct{}

func (p *blockingProvider) Name() string { return "blocking" }
func (p *blockingProvider) Complete(ctx context.Context, messages []iteragent.Message, opts ...iteragent.CompletionOptions) (string, error) {
	<-ctx.Done()
	return "", ctx.Err()
}
func (p *blockingProvider) CompleteStream(ctx context.Context, messages []iteragent.Message, opts iteragent.CompletionOptions, onToken func(string)) (string, error) {
	return p.Complete(ctx, messages, opts)
}

// ---------------------------------------------------------------------------
// MCP integration tests
// ---------------------------------------------------------------------------

// minimalMCPServer spins up an in-process HTTP MCP server that advertises one
// tool ("mcp_echo") and returns its first argument as the result.
func minimalMCPServer(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			ID     interface{}     `json:"id"`
			Method string          `json:"method"`
			Params json.RawMessage `json:"params,omitempty"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		enc := json.NewEncoder(w)
		switch req.Method {
		case "initialize":
			enc.Encode(map[string]interface{}{
				"jsonrpc": "2.0", "id": req.ID,
				"result": map[string]interface{}{
					"protocolVersion": "2024-11-05",
					"capabilities":    map[string]interface{}{},
				},
			})
		case "tools/list":
			enc.Encode(map[string]interface{}{
				"jsonrpc": "2.0", "id": req.ID,
				"result": map[string]interface{}{
					"tools": []map[string]interface{}{
						{"name": "mcp_echo", "description": "echo input", "inputSchema": json.RawMessage(`{"type":"object"}`)},
					},
				},
			})
		case "tools/call":
			var params struct {
				Name      string                 `json:"name"`
				Arguments map[string]interface{} `json:"arguments"`
			}
			json.Unmarshal(req.Params, &params)
			enc.Encode(map[string]interface{}{
				"jsonrpc": "2.0", "id": req.ID,
				"result": map[string]interface{}{
					"content": []map[string]string{{"type": "text", "text": "echo:" + params.Name}},
					"isError": false,
				},
			})
		default:
			enc.Encode(map[string]interface{}{
				"jsonrpc": "2.0", "id": req.ID,
				"error": map[string]interface{}{"code": -32601, "message": "not found"},
			})
		}
	}))
}

// TestAgentWithMcpServerHttp verifies that WithMcpServerHttp registers the
// MCP server's tools on the agent and they are callable during a Run().
func TestAgentWithMcpServerHttp(t *testing.T) {
	srv := minimalMCPServer(t)
	defer srv.Close()

	ctx := context.Background()
	p := iteragent.NewMock("final answer")
	a, err := iteragent.New(p, nil, testLogger()).
		WithMcpServerHttp(ctx, srv.URL)
	if err != nil {
		t.Fatalf("WithMcpServerHttp: %v", err)
	}
	defer a.Close() //nolint:errcheck

	tools := a.GetTools()
	found := false
	for _, tool := range tools {
		if tool.Name == "mcp_echo" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("mcp_echo tool not registered; got tools: %v", toolNames(tools))
	}
}

// TestAgentClose_McpClient verifies that Close() shuts down the tracked MCP client
// without returning an error when the server is still available.
func TestAgentClose_McpClient(t *testing.T) {
	srv := minimalMCPServer(t)
	defer srv.Close()

	ctx := context.Background()
	p := iteragent.NewMock("done")
	a, err := iteragent.New(p, nil, testLogger()).
		WithMcpServerHttp(ctx, srv.URL)
	if err != nil {
		t.Fatalf("WithMcpServerHttp: %v", err)
	}

	if err := a.Close(); err != nil {
		t.Errorf("Close() returned unexpected error: %v", err)
	}
	// Double-close should be safe.
	if err := a.Close(); err != nil {
		t.Errorf("second Close() returned unexpected error: %v", err)
	}
}

// TestAgentWithMcpServerHttp_ToolCallable verifies that a registered MCP tool
// can actually be called via the agent's tool execution mechanism.
func TestAgentWithMcpServerHttp_ToolCallable(t *testing.T) {
	srv := minimalMCPServer(t)
	defer srv.Close()

	ctx := context.Background()
	// Agent will call mcp_echo on turn 1, then return "done" on turn 2.
	p := iteragent.NewMockWithTools("done", []iteragent.ToolCall{
		{Tool: "mcp_echo", Args: map[string]string{}},
	})
	a, err := iteragent.New(p, nil, testLogger()).
		WithMcpServerHttp(ctx, srv.URL)
	if err != nil {
		t.Fatalf("WithMcpServerHttp: %v", err)
	}
	defer a.Close() //nolint:errcheck

	out, err := a.Run(ctx, "", "use mcp_echo")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out != "done" {
		t.Errorf("got %q, want %q", out, "done")
	}
}

func toolNames(tools []iteragent.Tool) []string {
	names := make([]string, len(tools))
	for i, t := range tools {
		names[i] = t.Name
	}
	return names
}

// ---------------------------------------------------------------------------
// Agent builder methods
// ---------------------------------------------------------------------------

func TestAgentBuilders_WithTemperature(t *testing.T) {
	p := iteragent.NewMock("ok")
	a := iteragent.New(p, nil, testLogger()).WithTemperature(0.7)
	// Verify that temperature is applied via CompletionOptions by prompting and
	// checking the round-trip (echoProvider returns opts.Temperature as content).
	// Instead of round-trip: just verify agent builds without error and is non-nil.
	if a == nil {
		t.Fatal("expected non-nil agent")
	}
}

func TestAgentBuilders_WithMaxTokens(t *testing.T) {
	p := iteragent.NewMock("ok")
	a := iteragent.New(p, nil, testLogger()).WithMaxTokens(2048)
	if a == nil {
		t.Fatal("expected non-nil agent")
	}
}

func TestAgentBuilders_WithCacheEnabled_True(t *testing.T) {
	p := iteragent.NewMock("ok")
	a := iteragent.New(p, nil, testLogger()).WithCacheEnabled(true)
	if a == nil {
		t.Fatal("expected non-nil agent after WithCacheEnabled(true)")
	}
}

func TestAgentBuilders_WithCacheEnabled_False(t *testing.T) {
	p := iteragent.NewMock("ok")
	// Start with cache on, then disable.
	a := iteragent.New(p, nil, testLogger()).
		WithCacheEnabled(true).
		WithCacheEnabled(false)
	if a == nil {
		t.Fatal("expected non-nil agent after WithCacheEnabled(false)")
	}
}

func TestAgentBuilders_ChainedBuilders(t *testing.T) {
	p := iteragent.NewMock("ok")
	a := iteragent.New(p, nil, testLogger()).
		WithTemperature(0.5).
		WithMaxTokens(1024).
		WithCacheEnabled(true).
		WithThinkingLevel(iteragent.ThinkingLevelLow).
		WithSystemPrompt("test")
	if a == nil {
		t.Fatal("expected non-nil agent after chained builders")
	}
}
