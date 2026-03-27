// basic — minimal iteragent example demonstrating streaming tokens and lifecycle hooks.
//
// Usage:
//
//	ANTHROPIC_API_KEY=sk-... go run ./examples/basic
//	GEMINI_API_KEY=...       go run ./examples/basic
package main

import (
	"context"
	"fmt"
	"log"
	"log/slog"
	"os"
	"time"

	"github.com/GrayCodeAI/iteragent"
)

func main() {
	// Auto-detect provider from environment variables.
	providerName := os.Getenv("ITERATE_PROVIDER")
	if providerName == "" {
		if os.Getenv("ANTHROPIC_API_KEY") != "" {
			providerName = "anthropic"
		} else {
			providerName = "gemini"
		}
	}

	p, err := iteragent.NewProvider(providerName)
	if err != nil {
		log.Fatalf("provider: %v\n\nSet ANTHROPIC_API_KEY or GEMINI_API_KEY to run this example.", err)
	}

	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelWarn}))

	// Lifecycle hooks: log turn timing and tool calls.
	var turnStart time.Time
	hooks := iteragent.AgentHooks{
		BeforeTurn: func(turn int, messages []iteragent.Message) {
			turnStart = time.Now()
			fmt.Printf("[hook] turn %d start (%d messages)\n", turn, len(messages))
		},
		AfterTurn: func(turn int, response string) {
			fmt.Printf("[hook] turn %d done in %s\n", turn, time.Since(turnStart).Round(time.Millisecond))
		},
		OnToolStart: func(toolName string, args map[string]interface{}) {
			fmt.Printf("[hook] → tool %s\n", toolName)
		},
		OnToolEnd: func(toolName string, result string, err error) {
			if err != nil {
				fmt.Printf("[hook] ← tool %s error: %v\n", toolName, err)
			} else {
				fmt.Printf("[hook] ← tool %s ok (%d chars)\n", toolName, len(result))
			}
		},
	}

	a := iteragent.New(p, iteragent.DefaultTools("."), logger).
		WithSystemPrompt("You are a helpful coding assistant. Answer concisely.").
		WithHooks(hooks)

	// Close shuts down any MCP server connections when we're done.
	defer func() {
		if err := a.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "close: %v\n", err)
		}
	}()

	fmt.Printf("Provider: %s\n\n", p.Name())

	ctx := context.Background()
	question := "What is 2 + 2? Reply with just the number."
	fmt.Printf("Q: %s\nA: ", question)

	// Stream tokens live as they arrive via EventTokenUpdate.
	var tokenCount int
	events := a.Prompt(ctx, question)
	for e := range events {
		switch iteragent.EventType(e.Type) {
		case iteragent.EventTokenUpdate:
			fmt.Print(e.Content)
			tokenCount++
		case iteragent.EventError:
			fmt.Fprintf(os.Stderr, "\nerror: %s\n", e.Content)
		}
	}
	a.Finish()

	fmt.Printf("\n\n(%d token chunks streamed)\n", tokenCount)
}
