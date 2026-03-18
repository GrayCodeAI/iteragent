// basic — minimal iteragent example.
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
	"os"

	"github.com/GrayCodeAI/iteragent"
	"log/slog"
)

func main() {
	// Auto-detect provider from environment variables.
	providerName := os.Getenv("ITERATE_PROVIDER")
	if providerName == "" {
		// Default: try Anthropic first, then Gemini.
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

	a := iteragent.New(p, iteragent.DefaultTools("."), logger).
		WithSystemPrompt("You are a helpful coding assistant. Answer concisely.")

	fmt.Printf("Provider: %s\n\n", p.Name())

	ctx := context.Background()
	question := "What is 2 + 2? Reply with just the number."

	fmt.Printf("Q: %s\n", question)

	out, err := a.Run(ctx, "", question)
	if err != nil {
		log.Fatalf("run: %v", err)
	}

	fmt.Printf("A: %s\n", out)
}
