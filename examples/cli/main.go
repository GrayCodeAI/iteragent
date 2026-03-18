// cli — interactive iteragent REPL example.
//
// Usage:
//
//	ANTHROPIC_API_KEY=sk-... go run ./examples/cli
//	ANTHROPIC_API_KEY=sk-... go run ./examples/cli --model claude-opus-4-6
//	ANTHROPIC_API_KEY=sk-... go run ./examples/cli --thinking medium
//
// Commands:
//
//	/help       — show available commands
//	/clear      — reset conversation history
//	/tools      — list available tools
//	/thinking   — cycle thinking level
//	/quit       — exit
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"strings"

	iteragent "github.com/GrayCodeAI/iteragent"
)

func main() {
	providerName := flag.String("provider", "", "Provider (anthropic, gemini, openai, groq…)")
	model := flag.String("model", "", "Override model name")
	thinkingFlag := flag.String("thinking", "off", "Thinking level: off, minimal, low, medium, high")
	flag.Parse()

	if *model != "" {
		os.Setenv("ITERATE_MODEL", *model)
	}

	p, err := iteragent.NewProvider(*providerName)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n\nSet ANTHROPIC_API_KEY or GEMINI_API_KEY.\n", err)
		os.Exit(1)
	}

	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelWarn}))
	thinking := iteragent.ThinkingLevel(*thinkingFlag)

	a := iteragent.New(p, iteragent.DefaultTools("."), logger).
		WithSystemPrompt("You are a helpful coding assistant with access to tools.").
		WithThinkingLevel(thinking)

	fmt.Printf("iteragent cli — provider: %s, thinking: %s\n", p.Name(), thinking)
	fmt.Println("Type a message, or /help for commands.")
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)
	ctx := context.Background()

	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		// Slash commands.
		if strings.HasPrefix(line, "/") {
			switch strings.ToLower(strings.Fields(line)[0]) {
			case "/quit", "/exit", "/q":
				fmt.Println("bye")
				return
			case "/clear":
				a.Reset()
				fmt.Println("Conversation cleared.")
				continue
			case "/tools":
				tools := a.GetTools()
				fmt.Printf("%d tools available:\n", len(tools))
				for _, t := range tools {
					firstLine := strings.SplitN(t.Description, "\n", 2)[0]
					fmt.Printf("  %-20s %s\n", t.Name, firstLine)
				}
				continue
			case "/thinking":
				parts := strings.Fields(line)
				if len(parts) == 2 {
					a = a.WithThinkingLevel(iteragent.ThinkingLevel(parts[1]))
					fmt.Printf("Thinking set to %s.\n", parts[1])
				} else {
					fmt.Println("Usage: /thinking off|minimal|low|medium|high")
				}
				continue
			case "/help":
				fmt.Println("Commands:")
				fmt.Println("  /clear           — reset conversation")
				fmt.Println("  /tools           — list available tools")
				fmt.Println("  /thinking <lvl>  — set thinking level (off|minimal|low|medium|high)")
				fmt.Println("  /quit            — exit")
				continue
			default:
				fmt.Printf("Unknown command: %s (try /help)\n", line)
				continue
			}
		}

		// Stream the response.
		events := a.Prompt(ctx, line)
		var lastOutput string
		for e := range events {
			switch iteragent.EventType(e.Type) {
			case iteragent.EventMessageUpdate:
				// Print incrementally — overwrite previous partial line.
				fmt.Print("\r\033[K") // clear line
				preview := e.Content
				if len(preview) > 120 {
					preview = preview[:120] + "…"
				}
				fmt.Print(preview)
			case iteragent.EventToolExecutionStart:
				fmt.Printf("\n[tool] %s\n", e.ToolName)
			case iteragent.EventMessageEnd:
				lastOutput = e.Content
			case iteragent.EventError:
				fmt.Printf("\nError: %s\n", e.Content)
			}
		}
		a.Finish()

		fmt.Print("\r\033[K") // clear partial line
		fmt.Printf("%s\n\n", lastOutput)
	}
}
