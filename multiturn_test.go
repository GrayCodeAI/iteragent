package iteragent_test

import (
	"context"
	"strings"
	"testing"

	iteragent "github.com/GrayCodeAI/iteragent"
)

// sequentialProvider serves responses from a slice in order.
// It also records each messages slice it receives so we can inspect history.
type sequentialProvider struct {
	responses []string
	callIdx   int
	received  [][]iteragent.Message
}

func (p *sequentialProvider) Name() string { return "sequential-mock" }

func (p *sequentialProvider) Complete(ctx context.Context, messages []iteragent.Message, opts ...iteragent.CompletionOptions) (string, error) {
	p.received = append(p.received, append([]iteragent.Message{}, messages...))
	if p.callIdx >= len(p.responses) {
		return "done", nil
	}
	resp := p.responses[p.callIdx]
	p.callIdx++
	return resp, nil
}

// TestMultiTurnContextRetained verifies that after each Prompt() call the prior
// conversation is included in the messages sent to the provider on the next call.
func TestMultiTurnContextRetained(t *testing.T) {
	p := &sequentialProvider{
		responses: []string{
			"turn1 response",
			"turn2 response",
			"turn3 response",
			"turn4 response",
			"turn5 response",
		},
	}
	a := iteragent.New(p, nil, testLogger())

	prompts := []string{"msg1", "msg2", "msg3", "msg4", "msg5"}
	for i, prompt := range prompts {
		events := a.Prompt(context.Background(), prompt)
		for range events {
		}
		a.Finish()

		// After turn i+1, a.Messages should contain 2*(i+1) messages
		// (alternating user/assistant, no system message).
		want := (i + 1) * 2
		if len(a.Messages) != want {
			t.Errorf("after turn %d: want %d messages in a.Messages, got %d", i+1, want, len(a.Messages))
		}
	}

	// On turn 5, the provider should have received all prior messages.
	// The messages slice for call 4 (0-indexed) should start with system message
	// followed by 8 prior messages (4 user + 4 assistant) then the new user message.
	if len(p.received) < 5 {
		t.Fatalf("expected 5 provider calls, got %d", len(p.received))
	}

	lastCall := p.received[4] // 5th call, 0-indexed
	// lastCall[0] = system message, then alternating user/assistant from prior turns
	// + new user message for turn 5.
	// We expect: system + (user+assistant)*4 + user = 1 + 8 + 1 = 10 messages.
	if len(lastCall) != 10 {
		t.Errorf("turn 5 provider call: want 10 messages (system + 8 history + new user), got %d", len(lastCall))
	}

	// Check that early user messages are present in the last call.
	var userMsgs []string
	for _, m := range lastCall {
		if m.Role == "user" {
			userMsgs = append(userMsgs, m.Content)
		}
	}
	for _, prompt := range prompts {
		found := false
		for _, um := range userMsgs {
			if strings.Contains(um, prompt) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("turn 5 provider call missing user message %q", prompt)
		}
	}
}

// TestMultiTurnMessagesAfterClear verifies that after Reset(), a.Messages is empty
// and the next Prompt() starts fresh.
func TestMultiTurnMessagesAfterClear(t *testing.T) {
	p := &sequentialProvider{
		responses: []string{"r1", "r2"},
	}
	a := iteragent.New(p, nil, testLogger())

	// First turn.
	events := a.Prompt(context.Background(), "hello")
	for range events {
	}
	a.Finish()
	if len(a.Messages) != 2 {
		t.Fatalf("after turn 1: want 2 messages, got %d", len(a.Messages))
	}

	// Reset clears history.
	a.Reset()
	if len(a.Messages) != 0 {
		t.Fatalf("after Reset: want 0 messages, got %d", len(a.Messages))
	}

	// Second turn should only have the new user message (+ system), not the old history.
	events = a.Prompt(context.Background(), "fresh start")
	for range events {
	}
	a.Finish()

	// Provider's second call should only have system + new user = 2 messages.
	if len(p.received) < 2 {
		t.Fatalf("expected 2 provider calls, got %d", len(p.received))
	}
	secondCall := p.received[1]
	if len(secondCall) != 2 {
		t.Errorf("after Reset, second call: want 2 messages (system+user), got %d", len(secondCall))
	}
}
