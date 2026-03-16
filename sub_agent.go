package iteragent

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
)

type SubAgent struct {
	Name         string
	SystemPrompt string
	Provider     Provider
	Tools        map[string]Tool
	Agent        *Agent
	MaxTurns     int
}

type SubAgentConfig struct {
	Name         string
	SystemPrompt string
	Provider     Provider
	Tools        []Tool
	MaxTurns     int
}

func NewSubAgent(cfg SubAgentConfig, logger *slog.Logger) *SubAgent {
	toolMap := make(map[string]Tool, len(cfg.Tools))
	for _, t := range cfg.Tools {
		toolMap[t.Name] = t
	}

	agent := &Agent{
		provider: cfg.Provider,
		tools:    toolMap,
		logger:   logger,
		Events:   make(chan Event, 64),
	}

	return &SubAgent{
		Name:         cfg.Name,
		SystemPrompt: cfg.SystemPrompt,
		Agent:        agent,
	}
}

func (s *SubAgent) Run(ctx context.Context, task string) (string, error) {
	messages := []Message{
		{Role: "system", Content: s.SystemPrompt},
		{Role: "user", Content: task},
	}

	var result strings.Builder
	turns := 0
	maxTurns := 20

	for turns < maxTurns {
		turns++

		response, err := s.Provider.Complete(ctx, messages)
		if err != nil {
			return "", fmt.Errorf("sub-agent error at turn %d: %w", turns, err)
		}

		result.WriteString(response + "\n")

		calls := ParseToolCalls(response)
		if len(calls) == 0 {
			return response, nil
		}

		for _, call := range calls {
			tool, ok := s.Tools[call.Tool]
			if !ok {
				result.WriteString(fmt.Sprintf("unknown tool: %s\n", call.Tool))
				continue
			}

			toolResult, err := tool.Execute(ctx, call.Args)
			if err != nil {
				toolResult = fmt.Sprintf("ERROR: %s", err.Error())
			}

			result.WriteString(fmt.Sprintf("[%s result]: %s\n", call.Tool, toolResult))

			messages = append(messages, Message{
				Role:    "assistant",
				Content: response,
			})
			messages = append(messages, Message{
				Role:    "user",
				Content: toolResult,
			})
		}
	}

	return result.String(), fmt.Errorf("sub-agent exceeded max turns (%d)", maxTurns)
}

type SubAgentPool struct {
	agents map[string]*SubAgent
	logger *slog.Logger
}

func NewSubAgentPool(logger *slog.Logger) *SubAgentPool {
	return &SubAgentPool{
		agents: make(map[string]*SubAgent),
		logger: logger,
	}
}

func (p *SubAgentPool) Register(cfg SubAgentConfig) {
	p.agents[cfg.Name] = NewSubAgent(cfg, p.logger)
}

func (p *SubAgentPool) Get(name string) *SubAgent {
	return p.agents[name]
}

func (p *SubAgentPool) Has(name string) bool {
	_, ok := p.agents[name]
	return ok
}

func (p *SubAgentPool) List() []string {
	names := make([]string, 0, len(p.agents))
	for name := range p.agents {
		names = append(names, name)
	}
	return names
}

func (p *SubAgentPool) Run(ctx context.Context, name, task string) (string, error) {
	agent, ok := p.agents[name]
	if !ok {
		return "", fmt.Errorf("sub-agent %q not found", name)
	}
	return agent.Run(ctx, task)
}
