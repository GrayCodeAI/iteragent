package iteragent_test

import (
	"context"
	"log/slog"
	"os"
	"strings"
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
