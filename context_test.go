package iteragent_test

import (
	"fmt"
	"strings"
	"sync"
	"testing"

	iteragent "github.com/GrayCodeAI/iteragent"
)

// ---------------------------------------------------------------------------
// Token estimation
// ---------------------------------------------------------------------------

func TestEstimateTokens(t *testing.T) {
	// 4 chars per token
	if got := iteragent.EstimateTokens("abcd"); got != 1 {
		t.Errorf("EstimateTokens(4 chars) = %d, want 1", got)
	}
	if got := iteragent.EstimateTokens(""); got != 0 {
		t.Errorf("EstimateTokens(\"\") = %d, want 0", got)
	}
	if got := iteragent.EstimateTokens("abcdefgh"); got != 2 {
		t.Errorf("EstimateTokens(8 chars) = %d, want 2", got)
	}
}

func TestEstimateTotalTokens(t *testing.T) {
	msgs := []iteragent.Message{
		{Role: "user", Content: "abcd"},     // 1 token + 4 overhead = 5
		{Role: "assistant", Content: "abcd"}, // 1 token + 4 overhead = 5
	}
	got := iteragent.EstimateTotalTokens(msgs)
	if got != 10 {
		t.Errorf("EstimateTotalTokens = %d, want 10", got)
	}
}

// ---------------------------------------------------------------------------
// CompactMessagesTiered
// ---------------------------------------------------------------------------

// makeMessages builds a slice of alternating user/assistant messages,
// each with contentLen chars of content.
func makeMessages(n, contentLen int) []iteragent.Message {
	msgs := make([]iteragent.Message, n)
	for i := range msgs {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		msgs[i] = iteragent.Message{Role: role, Content: strings.Repeat("x", contentLen)}
	}
	return msgs
}

// TestCompactMessagesTiered_BelowLimit verifies that messages under the limit
// are returned unchanged.
func TestCompactMessagesTiered_BelowLimit(t *testing.T) {
	msgs := makeMessages(4, 10)
	cfg := iteragent.ContextConfig{
		MaxTokens:          100000,
		KeepRecent:         10,
		KeepFirst:          2,
		ToolOutputMaxLines: 50,
		WarningThreshold:   0.8,
		Strategy:           &iteragent.DefaultCompactionStrategy{},
	}
	got := iteragent.CompactMessagesTiered(msgs, cfg)
	if len(got) != len(msgs) {
		t.Errorf("expected %d messages unchanged, got %d", len(msgs), len(got))
	}
}

// TestCompactMessagesTiered_Level1_TruncatesToolOutputs verifies that long tool
// result messages are truncated at level 1.
func TestCompactMessagesTiered_Level1_TruncatesToolOutputs(t *testing.T) {
	// Build a tool result message with many lines.
	manyLines := strings.Join(make([]string, 200), "\n") // 200 empty lines
	longToolResult := iteragent.Message{
		Role:    "user",
		Content: "Tool foo result:\n" + manyLines,
	}

	msgs := []iteragent.Message{
		{Role: "system", Content: "sys"},
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "calling tool"},
		longToolResult,
		{Role: "assistant", Content: "done"},
	}

	// Set MaxTokens very low so level 1 is triggered.
	cfg := iteragent.ContextConfig{
		MaxTokens:          50,
		KeepRecent:         2,
		KeepFirst:          1,
		ToolOutputMaxLines: 10, // should truncate the 200-line tool result
		WarningThreshold:   0.8,
		Strategy:           &iteragent.DefaultCompactionStrategy{},
	}

	got := iteragent.CompactMessagesTiered(msgs, cfg)

	// Find the tool result in the compacted output and verify it's shorter.
	for _, m := range got {
		if strings.Contains(m.Content, "Tool foo result:") {
			lineCount := strings.Count(m.Content, "\n")
			if lineCount >= 200 {
				t.Errorf("tool output not truncated: still has %d lines", lineCount)
			}
			return
		}
	}
	// It's also acceptable for the tool result to be fully dropped (level 2/3 may remove it).
}

// TestCompactMessagesTiered_Level3_KeepsFirstAndLast verifies that at level 3
// the head (keepFirst) and tail (keepRecent) messages are preserved and the
// middle is dropped.
func TestCompactMessagesTiered_Level3_KeepsFirstAndLast(t *testing.T) {
	// Build many messages so level 3 is definitely triggered.
	msgs := make([]iteragent.Message, 20)
	for i := range msgs {
		msgs[i] = iteragent.Message{
			Role:    "user",
			Content: fmt.Sprintf("%s-msg-%d", strings.Repeat("x", 500), i),
		}
	}

	cfg := iteragent.ContextConfig{
		MaxTokens:          10,  // extremely low → all levels triggered
		KeepRecent:         3,
		KeepFirst:          2,
		ToolOutputMaxLines: 5,
		WarningThreshold:   0.8,
		Strategy:           &iteragent.DefaultCompactionStrategy{},
	}

	got := iteragent.CompactMessagesTiered(msgs, cfg)

	if len(got) > cfg.KeepFirst+cfg.KeepRecent {
		t.Errorf("after level-3 compaction: got %d messages, want at most %d",
			len(got), cfg.KeepFirst+cfg.KeepRecent)
	}

	// First message should be the original first.
	if !strings.Contains(got[0].Content, "msg-0") {
		t.Errorf("first message not preserved: %q", got[0].Content)
	}
	// Last message should be the original last.
	last := got[len(got)-1]
	if !strings.Contains(last.Content, fmt.Sprintf("msg-%d", len(msgs)-1)) {
		t.Errorf("last message not preserved: %q", last.Content)
	}
}

// TestCompactMessagesTiered_SingleMessage verifies a single-message slice is
// returned unchanged (edge case guard in the function).
func TestCompactMessagesTiered_SingleMessage(t *testing.T) {
	msgs := []iteragent.Message{{Role: "user", Content: "hi"}}
	cfg := iteragent.ContextConfig{MaxTokens: 1, KeepRecent: 1, KeepFirst: 1, Strategy: &iteragent.DefaultCompactionStrategy{}}
	got := iteragent.CompactMessagesTiered(msgs, cfg)
	if len(got) != 1 {
		t.Errorf("single message should pass through unchanged, got %d", len(got))
	}
}

// ---------------------------------------------------------------------------
// ContextTracker
// ---------------------------------------------------------------------------

func TestContextTracker_InitialState(t *testing.T) {
	ct := iteragent.NewContextTracker()
	if ct.TotalTokens() != 0 {
		t.Errorf("expected 0 total tokens initially, got %d", ct.TotalTokens())
	}
	if ct.InputTokens() != 0 {
		t.Errorf("expected 0 input tokens initially, got %d", ct.InputTokens())
	}
	if ct.OutputTokens() != 0 {
		t.Errorf("expected 0 output tokens initially, got %d", ct.OutputTokens())
	}
	if ct.CacheHitRate() != 0.0 {
		t.Errorf("expected 0.0 cache hit rate initially, got %f", ct.CacheHitRate())
	}
}

func TestContextTracker_UpdateWithRealUsage(t *testing.T) {
	ct := iteragent.NewContextTracker()
	usage := &iteragent.Usage{
		InputTokens:  100,
		OutputTokens: 50,
		TotalTokens:  150,
	}
	ct.UpdateWithRealUsage(usage)

	if ct.InputTokens() != 100 {
		t.Errorf("expected 100 input tokens, got %d", ct.InputTokens())
	}
	if ct.OutputTokens() != 50 {
		t.Errorf("expected 50 output tokens, got %d", ct.OutputTokens())
	}
	if ct.TotalTokens() != 150 {
		t.Errorf("expected 150 total tokens, got %d", ct.TotalTokens())
	}
}

func TestContextTracker_UpdateWithRealUsage_ClearsEstimated(t *testing.T) {
	ct := iteragent.NewContextTracker()
	ct.AddEstimatedTokens(500)

	usage := &iteragent.Usage{
		InputTokens:  100,
		OutputTokens: 50,
		TotalTokens:  150,
	}
	ct.UpdateWithRealUsage(usage)

	// estimated should be cleared; total should equal TotalTokens only
	if ct.TotalTokens() != 150 {
		t.Errorf("expected 150 after clearing estimates, got %d", ct.TotalTokens())
	}
}

func TestContextTracker_AddEstimatedTokens(t *testing.T) {
	ct := iteragent.NewContextTracker()
	ct.AddEstimatedTokens(100)
	ct.AddEstimatedTokens(200)

	if ct.TotalTokens() != 300 {
		t.Errorf("expected 300 estimated tokens, got %d", ct.TotalTokens())
	}
}

func TestContextTracker_TotalTokens_RealPlusEstimated(t *testing.T) {
	ct := iteragent.NewContextTracker()
	usage := &iteragent.Usage{
		InputTokens:  80,
		OutputTokens: 20,
		TotalTokens:  100,
	}
	ct.UpdateWithRealUsage(usage)
	ct.AddEstimatedTokens(50)

	if ct.TotalTokens() != 150 {
		t.Errorf("expected real(100) + estimated(50) = 150, got %d", ct.TotalTokens())
	}
}

func TestContextTracker_CacheHitRate(t *testing.T) {
	ct := iteragent.NewContextTracker()
	usage := &iteragent.Usage{
		InputTokens:  60,
		OutputTokens: 10,
		TotalTokens:  70,
		CacheRead:    40,
	}
	ct.UpdateWithRealUsage(usage)

	// CacheHitRate = CacheRead / (InputTokens + CacheRead + CacheWrite) = 40/100 = 0.4
	got := ct.CacheHitRate()
	if got < 0.39 || got > 0.41 {
		t.Errorf("expected ~0.4 cache hit rate, got %f", got)
	}
}

func TestContextTracker_CacheHitRate_ZeroWithNoUsage(t *testing.T) {
	ct := iteragent.NewContextTracker()
	if ct.CacheHitRate() != 0.0 {
		t.Errorf("expected 0.0 cache hit rate with no usage, got %f", ct.CacheHitRate())
	}
}

func TestContextTracker_ConcurrentAccess(t *testing.T) {
	ct := iteragent.NewContextTracker()
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(2)
		go func(n int) {
			defer wg.Done()
			ct.AddEstimatedTokens(10)
		}(i)
		go func(n int) {
			defer wg.Done()
			_ = ct.TotalTokens()
		}(i)
	}
	wg.Wait()
}

// ---------------------------------------------------------------------------
// ExecutionTracker
// ---------------------------------------------------------------------------

func TestExecutionTracker_InitialState(t *testing.T) {
	limits := iteragent.DefaultExecutionLimits()
	et := iteragent.NewExecutionTracker(limits)

	if et.TurnCount() != 0 {
		t.Errorf("expected 0 turns, got %d", et.TurnCount())
	}
	if et.TotalTokens() != 0 {
		t.Errorf("expected 0 tokens, got %d", et.TotalTokens())
	}
	if et.AtTurnLimit() {
		t.Error("should not be at turn limit initially")
	}
	if et.AtTokenLimit() {
		t.Error("should not be at token limit initially")
	}
	if !et.ShouldContinue() {
		t.Error("ShouldContinue should be true initially")
	}
}

func TestExecutionTracker_IncrementTurn(t *testing.T) {
	et := iteragent.NewExecutionTracker(iteragent.DefaultExecutionLimits())
	et.IncrementTurn(100)
	et.IncrementTurn(200)

	if et.TurnCount() != 2 {
		t.Errorf("expected 2 turns, got %d", et.TurnCount())
	}
	if et.TotalTokens() != 300 {
		t.Errorf("expected 300 tokens, got %d", et.TotalTokens())
	}
}

func TestExecutionTracker_AtTurnLimit(t *testing.T) {
	limits := iteragent.ExecutionLimits{MaxTurns: 2, MaxTokens: 1000000}
	et := iteragent.NewExecutionTracker(limits)
	et.IncrementTurn(0)
	et.IncrementTurn(0)

	if !et.AtTurnLimit() {
		t.Error("should be at turn limit after 2 turns")
	}
	if et.ShouldContinue() {
		t.Error("ShouldContinue should be false at turn limit")
	}
}

func TestExecutionTracker_AtTokenLimit(t *testing.T) {
	limits := iteragent.ExecutionLimits{MaxTurns: 50, MaxTokens: 100}
	et := iteragent.NewExecutionTracker(limits)
	et.IncrementTurn(100)

	if !et.AtTokenLimit() {
		t.Error("should be at token limit")
	}
}

// ---------------------------------------------------------------------------
// TestCompactMessagesTiered_AssistantSummary verifies that old assistant messages
// are replaced with summaries at level 2.
func TestCompactMessagesTiered_AssistantSummary(t *testing.T) {
	longAssistant := iteragent.Message{
		Role:    "assistant",
		Content: strings.Repeat("a", 1000), // definitely gets summarized to first 200 chars
	}
	msgs := []iteragent.Message{
		{Role: "system", Content: strings.Repeat("s", 10)},
		{Role: "user", Content: strings.Repeat("u", 10)},
		longAssistant,
		{Role: "user", Content: strings.Repeat("u2", 10)},
		{Role: "assistant", Content: strings.Repeat("a2", 10)}, // recent — kept intact
	}

	cfg := iteragent.ContextConfig{
		MaxTokens:          80,  // enough to trigger level 2 but not drop everything
		KeepRecent:         2,
		KeepFirst:          1,
		ToolOutputMaxLines: 50,
		WarningThreshold:   0.8,
		Strategy:           &iteragent.DefaultCompactionStrategy{},
	}

	got := iteragent.CompactMessagesTiered(msgs, cfg)

	// The long assistant message should have been shortened.
	for _, m := range got {
		if m.Role == "assistant" && len(m.Content) == 1000 {
			t.Error("long assistant message should have been summarized but was preserved as-is")
		}
	}
}
