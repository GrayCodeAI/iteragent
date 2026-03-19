package iteragent_test

import (
	"context"
	"testing"

	iteragent "github.com/GrayCodeAI/iteragent"
)

// ---------------------------------------------------------------------------
// NewSubAgent
// ---------------------------------------------------------------------------

func TestNewSubAgent_BasicFields(t *testing.T) {
	p := iteragent.NewMock("hello")
	sa := iteragent.NewSubAgent(iteragent.SubAgentConfig{
		Name:         "coder",
		SystemPrompt: "You are a coder.",
		Provider:     p,
		MaxTurns:     5,
	}, testLogger())

	if sa.Name != "coder" {
		t.Errorf("expected Name 'coder', got %q", sa.Name)
	}
	if sa.SystemPrompt != "You are a coder." {
		t.Errorf("expected SystemPrompt, got %q", sa.SystemPrompt)
	}
	if sa.MaxTurns != 5 {
		t.Errorf("expected MaxTurns 5, got %d", sa.MaxTurns)
	}
	if sa.Agent == nil {
		t.Error("expected non-nil embedded Agent")
	}
}

func TestNewSubAgent_ZeroMaxTurns(t *testing.T) {
	p := iteragent.NewMock("hi")
	sa := iteragent.NewSubAgent(iteragent.SubAgentConfig{
		Name:     "helper",
		Provider: p,
	}, testLogger())
	if sa.MaxTurns != 0 {
		t.Errorf("expected 0 MaxTurns, got %d", sa.MaxTurns)
	}
}

// ---------------------------------------------------------------------------
// WithMaxTurns
// ---------------------------------------------------------------------------

func TestSubAgent_WithMaxTurns(t *testing.T) {
	p := iteragent.NewMock("ok")
	sa := iteragent.NewSubAgent(iteragent.SubAgentConfig{
		Name:     "sa",
		Provider: p,
	}, testLogger()).WithMaxTurns(10)

	if sa.MaxTurns != 10 {
		t.Errorf("expected MaxTurns 10 after WithMaxTurns(10), got %d", sa.MaxTurns)
	}
}

func TestSubAgent_WithMaxTurns_Zero(t *testing.T) {
	p := iteragent.NewMock("ok")
	sa := iteragent.NewSubAgent(iteragent.SubAgentConfig{
		Name:     "sa",
		Provider: p,
		MaxTurns: 5,
	}, testLogger()).WithMaxTurns(0)

	if sa.MaxTurns != 0 {
		t.Errorf("expected MaxTurns reset to 0, got %d", sa.MaxTurns)
	}
}

func TestSubAgent_WithMaxTurns_Chained(t *testing.T) {
	p := iteragent.NewMock("ok")
	sa := iteragent.NewSubAgent(iteragent.SubAgentConfig{
		Name:     "sa",
		Provider: p,
	}, testLogger()).WithMaxTurns(3).WithMaxTurns(7)

	if sa.MaxTurns != 7 {
		t.Errorf("expected last WithMaxTurns to win (7), got %d", sa.MaxTurns)
	}
}

// ---------------------------------------------------------------------------
// SubAgent.Run
// ---------------------------------------------------------------------------

func TestSubAgent_Run_ReturnsResponse(t *testing.T) {
	p := iteragent.NewMock("sub-agent response")
	sa := iteragent.NewSubAgent(iteragent.SubAgentConfig{
		Name:         "helper",
		SystemPrompt: "Be helpful.",
		Provider:     p,
	}, testLogger())

	result, err := sa.Run(context.Background(), "do something")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "sub-agent response" {
		t.Errorf("expected 'sub-agent response', got %q", result)
	}
}

func TestSubAgent_Run_UsesSystemPrompt(t *testing.T) {
	// The system prompt is forwarded to Agent.Run; we verify the call succeeds
	// with a non-empty system prompt.
	p := iteragent.NewMock("done")
	sa := iteragent.NewSubAgent(iteragent.SubAgentConfig{
		Name:         "auditor",
		SystemPrompt: "Audit everything carefully.",
		Provider:     p,
	}, testLogger())

	_, err := sa.Run(context.Background(), "audit this code")
	if err != nil {
		t.Fatalf("unexpected error with system prompt: %v", err)
	}
}

// ---------------------------------------------------------------------------
// SubAgentPool
// ---------------------------------------------------------------------------

func TestSubAgentPool_RegisterAndGet(t *testing.T) {
	pool := iteragent.NewSubAgentPool(testLogger())
	pool.Register(iteragent.SubAgentConfig{
		Name:     "coder",
		Provider: iteragent.NewMock("code"),
	})

	if !pool.Has("coder") {
		t.Error("expected 'coder' to be registered")
	}
	sa := pool.Get("coder")
	if sa == nil {
		t.Fatal("Get returned nil for registered agent")
	}
	if sa.Name != "coder" {
		t.Errorf("expected name 'coder', got %q", sa.Name)
	}
}

func TestSubAgentPool_Has_Missing(t *testing.T) {
	pool := iteragent.NewSubAgentPool(testLogger())
	if pool.Has("nonexistent") {
		t.Error("Has should return false for unregistered agent")
	}
}

func TestSubAgentPool_Get_Missing(t *testing.T) {
	pool := iteragent.NewSubAgentPool(testLogger())
	if pool.Get("ghost") != nil {
		t.Error("Get should return nil for unregistered agent")
	}
}

func TestSubAgentPool_List(t *testing.T) {
	pool := iteragent.NewSubAgentPool(testLogger())
	pool.Register(iteragent.SubAgentConfig{Name: "a", Provider: iteragent.NewMock("a")})
	pool.Register(iteragent.SubAgentConfig{Name: "b", Provider: iteragent.NewMock("b")})

	names := pool.List()
	if len(names) != 2 {
		t.Errorf("expected 2 agents, got %d", len(names))
	}
	nameSet := map[string]bool{}
	for _, n := range names {
		nameSet[n] = true
	}
	if !nameSet["a"] || !nameSet["b"] {
		t.Errorf("expected both 'a' and 'b' in list, got %v", names)
	}
}

func TestSubAgentPool_Run_Success(t *testing.T) {
	pool := iteragent.NewSubAgentPool(testLogger())
	pool.Register(iteragent.SubAgentConfig{
		Name:     "writer",
		Provider: iteragent.NewMock("draft complete"),
	})

	result, err := pool.Run(context.Background(), "writer", "write a blog post")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "draft complete" {
		t.Errorf("expected 'draft complete', got %q", result)
	}
}

func TestSubAgentPool_Run_Missing(t *testing.T) {
	pool := iteragent.NewSubAgentPool(testLogger())
	_, err := pool.Run(context.Background(), "nobody", "task")
	if err == nil {
		t.Error("expected error for missing agent")
	}
}
