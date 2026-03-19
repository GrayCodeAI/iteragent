package iteragent

import (
	"context"
	"fmt"
	"log/slog"
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
		MaxTurns:     cfg.MaxTurns,
		Agent:        agent,
	}
}

// WithMaxTurns sets the maximum number of tool-call iterations before the
// sub-agent stops. A value of 0 means use the default (20 iterations).
// The limit is stored on the SubAgent and consulted at Run time.
func (s *SubAgent) WithMaxTurns(n int) *SubAgent {
	s.MaxTurns = n
	return s
}

// Run executes the sub-agent on the given task. It delegates to the embedded
// Agent so that TokenStreamer, hooks, and context compaction all apply.
func (s *SubAgent) Run(ctx context.Context, task string) (string, error) {
	return s.Agent.Run(ctx, s.SystemPrompt, task)
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
