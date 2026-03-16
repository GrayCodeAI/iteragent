package iteragent

import (
	"fmt"
	"math"
	"sync"
	"time"
)

const charsPerToken = 4

func EstimateTokens(text string) int {
	return len(text) / charsPerToken
}

func EstimateMessageTokens(msg Message) int {
	if msg.Content == "" {
		return 0
	}
	tokens := EstimateTokens(msg.Content) + 4
	return tokens
}

func EstimateTotalTokens(messages []Message) int {
	total := 0
	for _, msg := range messages {
		total += EstimateMessageTokens(msg)
	}
	return total
}

type ContextTracker struct {
	mu              sync.Mutex
	inputTokens     int
	outputTokens    int
	estimatedTokens int
	lastRealUsage   *Usage
	lastUpdate      time.Time
}

func NewContextTracker() *ContextTracker {
	return &ContextTracker{
		lastUpdate: time.Now(),
	}
}

func (c *ContextTracker) UpdateWithRealUsage(usage *Usage) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.inputTokens = usage.InputTokens
	c.outputTokens = usage.OutputTokens
	c.lastRealUsage = usage
	c.lastUpdate = time.Now()
	c.estimatedTokens = 0
}

func (c *ContextTracker) AddEstimatedTokens(tokens int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.estimatedTokens += tokens
}

func (c *ContextTracker) TotalTokens() int {
	c.mu.Lock()
	defer c.mu.Unlock()

	realTotal := 0
	if c.lastRealUsage != nil {
		realTotal = c.lastRealUsage.TotalTokens
	}

	return realTotal + c.estimatedTokens
}

func (c *ContextTracker) InputTokens() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.lastRealUsage != nil {
		return c.lastRealUsage.InputTokens
	}
	return 0
}

func (c *ContextTracker) OutputTokens() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.lastRealUsage != nil {
		return c.lastRealUsage.OutputTokens
	}
	return 0
}

func (c *ContextTracker) CacheHitRate() float64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.lastRealUsage != nil {
		return c.lastRealUsage.CacheHitRate()
	}
	return 0.0
}

type ContextConfig struct {
	MaxTokens        int
	WarningThreshold float64
	Strategy         CompactionStrategy
}

func DefaultContextConfig() ContextConfig {
	return ContextConfig{
		MaxTokens:        100000,
		WarningThreshold: 0.8,
		Strategy:         &DefaultCompactionStrategy{},
	}
}

func (c *ContextConfig) WarningTokens() int {
	return int(float64(c.MaxTokens) * c.WarningThreshold)
}

type ExecutionLimits struct {
	MaxTurns    int
	MaxTokens   int
	MaxDuration time.Duration
}

func DefaultExecutionLimits() ExecutionLimits {
	return ExecutionLimits{
		MaxTurns:    50,
		MaxTokens:   1000000,
		MaxDuration: 10 * time.Minute,
	}
}

type ExecutionTracker struct {
	mu          sync.Mutex
	turnCount   int
	totalTokens int
	startTime   time.Time
	limits      ExecutionLimits
}

func NewExecutionTracker(limits ExecutionLimits) *ExecutionTracker {
	return &ExecutionTracker{
		startTime: time.Now(),
		limits:    limits,
	}
}

func (e *ExecutionTracker) IncrementTurn(tokens int) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.turnCount++
	e.totalTokens += tokens
}

func (e *ExecutionTracker) TurnCount() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.turnCount
}

func (e *ExecutionTracker) TotalTokens() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.totalTokens
}

func (e *ExecutionTracker) Elapsed() time.Duration {
	e.mu.Lock()
	defer e.mu.Unlock()
	return time.Since(e.startTime)
}

func (e *ExecutionTracker) AtTurnLimit() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.turnCount >= e.limits.MaxTurns
}

func (e *ExecutionTracker) AtTokenLimit() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.totalTokens >= e.limits.MaxTokens
}

func (e *ExecutionTracker) AtDurationLimit() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return time.Since(e.startTime) >= e.limits.MaxDuration
}

func (e *ExecutionTracker) ShouldContinue() bool {
	return !e.AtTurnLimit() && !e.AtTokenLimit() && !e.AtDurationLimit()
}

func CompactMessagesTiered(messages []Message, maxTokens int) []Message {
	if len(messages) <= 1 {
		return messages
	}

	currentTokens := EstimateTotalTokens(messages)
	if currentTokens <= maxTokens {
		return messages
	}

	result := make([]Message, len(messages))
	copy(result, messages)

	for currentTokens > maxTokens && len(result) > 1 {
		removed := false
		for i := 1; i < len(result)-1; i++ {
			if result[i].Role == "tool" || result[i].Role == "user" {
				toolResult := result[i]
				truncated := toolResult.Content
				if len(truncated) > 500 {
					truncated = truncated[:500] + "... [truncated]"
				}
				result[i].Content = truncated
				currentTokens = EstimateTotalTokens(result)
				removed = true
				break
			}
		}
		if !removed {
			break
		}
	}

	for currentTokens > maxTokens && len(result) > 2 {
		result = result[1:]
		currentTokens = EstimateTotalTokens(result)
	}

	return result
}

type MessageSummary struct {
	Content    string
	TokenCount int
}

func SummarizeMessages(messages []Message) []Message {
	if len(messages) <= 10 {
		return messages
	}

	var summary []Message
	summary = append(summary, messages[0])

	midStart := len(messages) / 3
	midEnd := 2 * len(messages) / 3

	if midStart < len(messages) && midEnd > midStart {
		midSection := messages[midStart:midEnd]
		summary = append(summary, Message{
			Role:    "system",
			Content: fmt.Sprintf("[%d messages omitted]", len(midSection)),
		})
	}

	summary = append(summary, messages[len(messages)-1:]...)

	return summary
}

func TruncateToolOutput(content string, maxTokens int) string {
	maxChars := maxTokens * charsPerToken
	if len(content) <= maxChars {
		return content
	}
	return content[:maxChars] + "\n\n[Output truncated]"
}

func CalculateTokenBuffer(currentTokens, maxTokens int) int {
	return maxTokens - currentTokens
}

func EstimateResponseTokens(availableTokens int) int {
	return int(math.Min(float64(availableTokens*2/3), 4096))
}
