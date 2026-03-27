package iteragent_test

import (
	"context"
	"strings"
	"testing"

	iteragent "github.com/GrayCodeAI/iteragent"
)

// loopingProvider always returns the same tool call, simulating a stuck loop.
type loopingProvider struct {
	toolName  string
	callCount int
}

func (p *loopingProvider) Name() string { return "looping-mock" }

func (p *loopingProvider) Complete(ctx context.Context, messages []iteragent.Message, opts ...iteragent.CompletionOptions) (string, error) {
	p.callCount++
	// Always return the same tool call.
	return "```tool\n{\"tool\":\"" + p.toolName + "\",\"args\":{}}\n```", nil
}

// TestStuckLoopDetection verifies that the agent aborts after 3 identical tool calls.
func TestStuckLoopDetection(t *testing.T) {
	noop := iteragent.Tool{
		Name:        "noop",
		Description: "does nothing",
		Execute: func(ctx context.Context, args map[string]interface{}) (string, error) {
			return "ok", nil
		},
	}

	p := &loopingProvider{toolName: "noop"}
	a := iteragent.New(p, []iteragent.Tool{noop}, testLogger())

	var gotError bool
	var errMsg string
	events := a.Prompt(context.Background(), "trigger loop")
	for e := range events {
		if iteragent.EventType(e.Type) == iteragent.EventError {
			gotError = true
			errMsg = e.Content
		}
	}
	a.Finish()

	if !gotError {
		t.Error("expected EventError for stuck loop, got none")
	}
	if !strings.Contains(errMsg, "stuck loop") {
		t.Errorf("expected error message to mention 'stuck loop', got: %q", errMsg)
	}
	// Provider should be called at most ~3 times before detection (3 identical calls).
	if p.callCount > 5 {
		t.Errorf("stuck loop not detected early enough: provider called %d times", p.callCount)
	}
}
