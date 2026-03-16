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
	Type       string
	Content    string
	ToolCallID string
	ToolName   string
	Args       map[string]interface{}
	Result     string
	IsError    bool
	Thinking   string
}

// Agent is the core reasoning loop.
type Agent struct {
	provider      Provider
	tools         map[string]Tool
	logger        *slog.Logger
	Events        chan Event
	SystemPrompt  string
	Model         string
	ThinkingLevel ThinkingLevel
	MaxTokens     int
	Temperature   float32
	Skills        []Skill
	Messages      []Message
}

// New creates a new Agent.
func New(p Provider, tools []Tool, logger *slog.Logger) *Agent {
	toolMap := make(map[string]Tool, len(tools))
	for _, t := range tools {
		toolMap[t.Name] = t
	}
	return &Agent{
		provider:      p,
		tools:         toolMap,
		logger:        logger,
		Events:        make(chan Event, 64),
		ThinkingLevel: ThinkingLevelOff,
		MaxTokens:     4096,
		Temperature:   0.7,
		Messages:      []Message{},
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
		a.emit(Event{Type: string(EventTurnStart), Content: fmt.Sprintf("turn %d", i+1)})

		response, err := a.provider.Complete(ctx, messages)
		if err != nil {
			a.emit(Event{Type: string(EventError), Content: err.Error(), IsError: true})
			return "", fmt.Errorf("provider error at step %d: %w", i+1, err)
		}

		a.emit(Event{Type: string(EventMessageUpdate), Content: response})

		messages = append(messages, Message{
			Role:    "assistant",
			Content: response,
		})

		calls := ParseToolCalls(response)
		if len(calls) == 0 {
			a.emit(Event{Type: string(EventMessageEnd), Content: response})
			a.emit(Event{Type: string(EventTurnEnd), Content: ""})
			return response, nil
		}

		var toolResults strings.Builder
		for _, call := range calls {
			tool, ok := a.tools[call.Tool]
			if !ok {
				result := fmt.Sprintf("unknown tool: %s", call.Tool)
				toolResults.WriteString(fmt.Sprintf("Tool %s: %s\n", call.Tool, result))
				a.emit(Event{Type: string(EventToolExecutionEnd), ToolName: call.Tool, Result: result, IsError: true})
				continue
			}

			argsMap := make(map[string]interface{})
			for k, v := range call.Args {
				argsMap[k] = v
			}
			a.emit(Event{
				Type:       string(EventToolExecutionStart),
				ToolName:   call.Tool,
				ToolCallID: call.Tool + "-" + fmt.Sprintf("%d", i),
				Args:       argsMap,
			})
			a.logger.Info("executing tool", "tool", call.Tool, "args", call.Args)

			result, err := tool.Execute(ctx, call.Args)
			isError := false
			if err != nil {
				result = fmt.Sprintf("ERROR: %s\nOutput: %s", err.Error(), result)
				isError = true
			}

			a.emit(Event{
				Type:     string(EventToolExecutionEnd),
				ToolName: call.Tool,
				Result:   result,
				IsError:  isError,
			})
			toolResults.WriteString(fmt.Sprintf("Tool %s result:\n%s\n\n", call.Tool, result))
		}

		messages = append(messages, Message{
			Role:    "user",
			Content: toolResults.String(),
		})
		a.emit(Event{Type: string(EventTurnEnd), Content: ""})
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

func (a *Agent) WithSystemPrompt(prompt string) *Agent {
	a.SystemPrompt = prompt
	return a
}

func (a *Agent) WithModel(model string) *Agent {
	a.Model = model
	return a
}

func (a *Agent) WithThinkingLevel(level ThinkingLevel) *Agent {
	a.ThinkingLevel = level
	return a
}

func (a *Agent) WithMaxTokens(maxTokens int) *Agent {
	a.MaxTokens = maxTokens
	return a
}

func (a *Agent) WithTemperature(temp float32) *Agent {
	a.Temperature = temp
	return a
}

func (a *Agent) WithSkills(skills []Skill) *Agent {
	a.Skills = skills
	return a
}

func (a *Agent) WithSkillSet(skillSet *SkillSet) *Agent {
	if skillSet != nil {
		a.Skills = skillSet.Skills
	}
	return a
}

func (a *Agent) WithTools(tools []Tool) *Agent {
	toolMap := make(map[string]Tool, len(tools))
	for _, t := range tools {
		toolMap[t.Name] = t
	}
	a.tools = toolMap
	return a
}

func (a *Agent) GetTools() []Tool {
	tools := make([]Tool, 0, len(a.tools))
	for _, t := range a.tools {
		tools = append(tools, t)
	}
	return tools
}

func (a *Agent) Prompt(ctx context.Context, text string) chan Event {
	events := make(chan Event, 64)
	go func() {
		defer close(events)
		events <- Event{Type: "message_start", Content: text}
		output, err := a.Run(ctx, a.SystemPrompt, text)
		if err != nil {
			events <- Event{Type: "error", Content: err.Error()}
		} else {
			events <- Event{Type: "message_end", Content: output}
		}
	}()
	return events
}

func (a *Agent) PromptMessages(ctx context.Context, messages []Message) chan Event {
	events := make(chan Event, 64)
	go func() {
		defer close(events)
		allTools := a.GetTools()
		systemContent := a.SystemPrompt
		if systemContent != "" {
			systemContent += "\n\n"
		}
		systemContent += ToolDescriptions(allTools)

		fullMessages := []Message{{Role: "system", Content: systemContent}}
		fullMessages = append(fullMessages, messages...)

		events <- Event{Type: "message_start", Content: ""}

		for i := 0; i < 20; i++ {
			events <- Event{Type: "turn_start", Content: fmt.Sprintf("turn %d", i+1)}

			response, err := a.provider.Complete(ctx, fullMessages)
			if err != nil {
				events <- Event{Type: "error", Content: err.Error()}
				break
			}

			events <- Event{Type: "content", Content: response}

			fullMessages = append(fullMessages, Message{Role: "assistant", Content: response})

			calls := ParseToolCalls(response)
			if len(calls) == 0 {
				events <- Event{Type: "message_end", Content: response}
				break
			}

			var toolResults strings.Builder
			for _, call := range calls {
				tool, ok := a.tools[call.Tool]
				if !ok {
					result := fmt.Sprintf("unknown tool: %s", call.Tool)
					toolResults.WriteString(fmt.Sprintf("Tool %s: %s\n", call.Tool, result))
					events <- Event{Type: "tool_result", Content: result}
					continue
				}

				events <- Event{Type: "tool_execution_start", Content: call.Tool}

				result, err := tool.Execute(ctx, call.Args)
				if err != nil {
					result = fmt.Sprintf("ERROR: %s\nOutput: %s", err.Error(), result)
				}

				events <- Event{Type: "tool_execution_end", Content: result}
				toolResults.WriteString(fmt.Sprintf("Tool %s result:\n%s\n\n", call.Tool, result))
			}

			fullMessages = append(fullMessages, Message{Role: "user", Content: toolResults.String()})
			events <- Event{Type: "turn_end", Content: ""}
		}
	}()
	return events
}
