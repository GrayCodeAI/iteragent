package iteragent

import (
	"fmt"
	"strings"
	"time"
)

type ExecutionLimits struct {
	MaxTokens     int
	MaxIterations int
	Timeout       time.Duration
}

type ContextConfig struct {
	MaxTokens   int
	MaxMessages int
	CompactAt   float64
}

type Context struct {
	Messages    []Message
	TotalTokens int
	Config      ContextConfig
}

func NewContext(cfg ContextConfig) *Context {
	return &Context{
		Messages: make([]Message, 0),
		Config:   cfg,
	}
}

func (c *Context) AddMessage(msg Message) {
	c.Messages = append(c.Messages, msg)
	c.TotalTokens += estimateTokens(msg.Content)
	c.compactIfNeeded()
}

func (c *Context) compactIfNeeded() {
	if c.Config.CompactAt == 0 {
		c.Config.CompactAt = 0.8
	}

	if c.TotalTokens > c.Config.MaxTokens && c.Config.MaxTokens > 0 {
		c.Compact()
	}
}

func (c *Context) Compact() {
	if len(c.Messages) <= 2 {
		return
	}

	keep := 2
	compactMsg := Message{
		Role:    "system",
		Content: fmt.Sprintf("[%d previous messages condensed]", len(c.Messages)-keep),
	}

	c.Messages = append([]Message{compactMsg}, c.Messages[keep:]...)
	c.TotalTokens = 0
	for _, m := range c.Messages {
		c.TotalTokens += estimateTokens(m.Content)
	}
}

func (c *Context) Reset() {
	c.Messages = make([]Message, 0)
	c.TotalTokens = 0
}

func estimateTokens(text string) int {
	return len(strings.Fields(text)) + len(text)/4
}

func TotalTokens(messages []Message) int {
	total := 0
	for _, m := range messages {
		total += estimateTokens(m.Content)
	}
	return total
}

func CompactMessages(messages []Message, maxTokens int) []Message {
	if TotalTokens(messages) <= maxTokens {
		return messages
	}

	if len(messages) <= 2 {
		return messages
	}

	keep := 2
	compactMsg := Message{
		Role:    "system",
		Content: fmt.Sprintf("[%d previous messages condensed]", len(messages)-keep),
	}

	result := append([]Message{compactMsg}, messages[keep:]...)

	if TotalTokens(result) > maxTokens && len(result) > 3 {
		return CompactMessages(result[1:], maxTokens)
	}

	return result
}
