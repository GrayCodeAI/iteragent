package iteragent

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
)

// ToolCall represents a tool invocation.
type ToolCall struct {
	Tool string            `json:"tool"`
	Args map[string]string `json:"args"`
}

// Tool represents a capability the agent can invoke.
type Tool struct {
	Name        string
	Description string
	Execute     func(ctx context.Context, args map[string]string) (string, error)
}

// Event represents a step in the agent's reasoning.
type Event struct {
	Type    string
	Content string
}

// Agent is the core reasoning loop.
type Agent struct {
	provider Provider
	tools    map[string]Tool
	logger   *slog.Logger
	Events   chan Event
}

// New creates a new Agent.
func New(p Provider, tools []Tool, logger *slog.Logger) *Agent {
	toolMap := make(map[string]Tool, len(tools))
	for _, t := range tools {
		toolMap[t.Name] = t
	}
	return &Agent{
		provider: p,
		tools:    toolMap,
		logger:   logger,
		Events:   make(chan Event, 64),
	}
}

// Run executes the agent loop with the given system prompt and user message.
func (a *Agent) Run(ctx context.Context, systemPrompt, userMessage string) (string, error) {
	allTools := make([]Tool, 0, len(a.tools))
	for _, t := range a.tools {
		allTools = append(allTools, t)
	}

	messages := []Message{
		{Role: "system", Content: systemPrompt + "\n\n" + ToolDescriptions(allTools)},
		{Role: "user", Content: userMessage},
	}

	const maxIterations = 20
	for i := 0; i < maxIterations; i++ {
		a.logger.Info("agent iteration", "step", i+1)

		response, err := a.provider.Complete(ctx, messages)
		if err != nil {
			a.emit(Event{Type: "error", Content: err.Error()})
			return "", fmt.Errorf("provider error at step %d: %w", i+1, err)
		}

		a.emit(Event{Type: "thought", Content: response})

		messages = append(messages, Message{
			Role:    "assistant",
			Content: response,
		})

		calls := ParseToolCalls(response)
		if len(calls) == 0 {
			a.emit(Event{Type: "done", Content: response})
			return response, nil
		}

		var toolResults strings.Builder
		for _, call := range calls {
			tool, ok := a.tools[call.Tool]
			if !ok {
				result := fmt.Sprintf("unknown tool: %s", call.Tool)
				toolResults.WriteString(fmt.Sprintf("Tool %s: %s\n", call.Tool, result))
				a.emit(Event{Type: "tool_result", Content: result})
				continue
			}

			a.emit(Event{Type: "tool_call", Content: fmt.Sprintf("%s(%v)", call.Tool, call.Args)})
			a.logger.Info("executing tool", "tool", call.Tool, "args", call.Args)

			result, err := tool.Execute(ctx, call.Args)
			if err != nil {
				result = fmt.Sprintf("ERROR: %s\nOutput: %s", err.Error(), result)
			}

			a.emit(Event{Type: "tool_result", Content: result})
			toolResults.WriteString(fmt.Sprintf("Tool %s result:\n%s\n\n", call.Tool, result))
		}

		messages = append(messages, Message{
			Role:    "user",
			Content: toolResults.String(),
		})
	}

	return "", fmt.Errorf("agent exceeded max iterations (%d)", maxIterations)
}

func (a *Agent) emit(e Event) {
	select {
	case a.Events <- e:
	default:
	}
}

// ParseToolCalls extracts tool calls from LLM output.
func ParseToolCalls(output string) []ToolCall {
	var calls []ToolCall
	lines := strings.Split(output, "\n")
	inBlock := false
	var block strings.Builder

	for _, line := range lines {
		if strings.HasPrefix(line, "```tool") {
			inBlock = true
			block.Reset()
			continue
		}
		if inBlock && line == "```" {
			var call ToolCall
			if err := json.Unmarshal([]byte(block.String()), &call); err == nil {
				calls = append(calls, call)
			}
			inBlock = false
			continue
		}
		if inBlock {
			block.WriteString(line + "\n")
		}
	}
	return calls
}

// ToolMap converts a slice of tools to a name-indexed map.
func ToolMap(tools []Tool) map[string]Tool {
	m := make(map[string]Tool, len(tools))
	for _, t := range tools {
		m[t.Name] = t
	}
	return m
}

// ToolDescriptions returns a formatted string of all tool descriptions.
func ToolDescriptions(tools []Tool) string {
	var sb strings.Builder
	sb.WriteString("## Available tools\n\n")
	sb.WriteString("Call tools using:\n```tool\n{\"tool\":\"name\",\"args\":{\"key\":\"value\"}}\n```\n\n")
	for _, t := range tools {
		sb.WriteString(fmt.Sprintf("### %s\n%s\n\n", t.Name, t.Description))
	}
	return sb.String()
}
