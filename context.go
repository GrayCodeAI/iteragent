package iteragent

import (
	"fmt"
	"math"
	"strings"
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

// ContextConfig controls context compaction behaviour.
type ContextConfig struct {
	MaxTokens          int
	KeepRecent         int
	KeepFirst          int
	ToolOutputMaxLines int
	WarningThreshold   float64
	Strategy           CompactionStrategy
}

// DefaultContextConfig returns sensible defaults for context management.
func DefaultContextConfig() ContextConfig {
	return ContextConfig{
		MaxTokens:          100000,
		KeepRecent:         10,
		KeepFirst:          2,
		ToolOutputMaxLines: 50,
		WarningThreshold:   0.8,
		Strategy:           &DefaultCompactionStrategy{},
	}
}

func (c *ContextConfig) WarningTokens() int {
	threshold := c.WarningThreshold
	if threshold == 0 {
		threshold = 0.8
	}
	return int(float64(c.MaxTokens) * threshold)
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

// isToolResultMessage returns true if the message is a tool result (role "tool" or
// user message containing "Tool ... result:").
func isToolResultMessage(msg Message) bool {
	if msg.Role == "tool" {
		return true
	}
	if msg.Role == "user" && strings.Contains(msg.Content, " result:") {
		return true
	}
	return false
}

// truncateLines keeps the first headN and last tailN lines of content, joining
// them with a truncation notice. Returns original if total lines <= headN+tailN.
func truncateLines(content string, headN, tailN int) string {
	lines := strings.Split(content, "\n")
	total := len(lines)
	keep := headN + tailN
	if total <= keep {
		return content
	}
	dropped := total - keep
	head := lines[:headN]
	tail := lines[total-tailN:]
	return strings.Join(head, "\n") +
		fmt.Sprintf("\n\n[... %d lines truncated ...]\n\n", dropped) +
		strings.Join(tail, "\n")
}

// CompactMessagesTiered applies a 3-tier compaction strategy to messages.
//
// Level 1 — Truncate tool outputs (head+tail):
//
//	Find messages with Role == "tool" or user messages that look like tool results.
//	Truncate content to ToolOutputMaxLines lines (head half + tail half).
//
// Level 2 — Summarize old turns:
//
//	Keep the last KeepRecent messages intact.
//	Replace older assistant messages with a summary; drop their tool results.
//
// Level 3 — Drop middle:
//
//	Keep first KeepFirst and last KeepRecent messages; drop everything in between.
func CompactMessagesTiered(messages []Message, cfg ContextConfig) []Message {
	if len(messages) <= 1 {
		return messages
	}

	maxTokens := cfg.MaxTokens
	keepRecent := cfg.KeepRecent
	keepFirst := cfg.KeepFirst
	maxLines := cfg.ToolOutputMaxLines

	if keepRecent <= 0 {
		keepRecent = 10
	}
	if keepFirst <= 0 {
		keepFirst = 2
	}
	if maxLines <= 0 {
		maxLines = 50
	}

	currentTokens := EstimateTotalTokens(messages)
	if currentTokens <= maxTokens {
		return messages
	}

	// ── Level 1: Truncate tool outputs ────────────────────────────────────────
	result := make([]Message, len(messages))
	copy(result, messages)

	headN := maxLines / 2
	tailN := maxLines - headN

	for i, msg := range result {
		if isToolResultMessage(msg) {
			result[i].Content = truncateLines(msg.Content, headN, tailN)
		}
	}

	currentTokens = EstimateTotalTokens(result)
	if currentTokens <= maxTokens {
		return result
	}

	// ── Level 2: Summarize old turns ──────────────────────────────────────────
	// Determine the boundary: keep the last keepRecent messages intact.
	cutoff := len(result) - keepRecent
	if cutoff < 0 {
		cutoff = 0
	}

	var compacted []Message
	i := 0
	for i < cutoff {
		msg := result[i]
		if msg.Role == "assistant" {
			// Count tool calls embedded in response.
			calls := ParseToolCalls(msg.Content)
			var summary string
			if len(calls) > 0 {
				summary = fmt.Sprintf("[Assistant used %d tool(s)]", len(calls))
			} else {
				if len(msg.Content) > 200 {
					summary = msg.Content[:200]
				} else {
					summary = msg.Content
				}
			}
			compacted = append(compacted, Message{Role: "assistant", Content: summary})
			// Skip adjacent tool result messages.
			i++
			for i < cutoff && isToolResultMessage(result[i]) {
				i++
			}
		} else {
			compacted = append(compacted, msg)
			i++
		}
	}
	// Append the recent tail intact.
	compacted = append(compacted, result[cutoff:]...)

	currentTokens = EstimateTotalTokens(compacted)
	if currentTokens <= maxTokens {
		return compacted
	}

	// ── Level 3: Drop middle ───────────────────────────────────────────────────
	if len(compacted) <= keepFirst+keepRecent {
		return compacted
	}

	head := compacted[:keepFirst]
	tail := compacted[len(compacted)-keepRecent:]
	return append(head, tail...)
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
