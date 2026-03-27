package iteragent

import (
	"context"
	"fmt"
	"strings"
)

// LLMCompactionStrategy uses the provider itself to generate a prose summary
// of old conversation turns instead of silently dropping them.
//
// When the context exceeds the threshold:
//  1. The last KeepRecent messages are kept verbatim.
//  2. Everything between the system message and the KeepRecent tail is sent to
//     the provider with a summarisation prompt.
//  3. The old turns are replaced by a single "system" summary message.
//
// Falls back to DefaultCompactionStrategy on any provider error.
type LLMCompactionStrategy struct {
	// Provider used to generate the summary. Must not be nil.
	Provider Provider
	// KeepRecent is the number of recent messages to preserve intact.
	// Defaults to 8 if zero.
	KeepRecent int
	// SummaryPrompt overrides the default summarisation instruction.
	SummaryPrompt string
}

const defaultSummaryPrompt = `You are a conversation summariser. Summarise the following conversation history concisely (≤ 300 words). Preserve: decisions made, files changed, key findings, open questions. Output plain prose only — no markdown, no bullet symbols, no headings.`

// Compact implements CompactionStrategy.
func (s *LLMCompactionStrategy) Compact(messages []Message, maxTokens int) []Message {
	keepRecent := s.KeepRecent
	if keepRecent <= 0 {
		keepRecent = 8
	}

	// Nothing to compact if the message list is tiny.
	if len(messages) <= keepRecent+1 {
		return messages
	}

	// Identify system message (always messages[0] by convention).
	system := messages[0]

	// Slice to summarise: everything after system, before the tail.
	tail := messages[len(messages)-keepRecent:]
	toSummarise := messages[1 : len(messages)-keepRecent]

	if len(toSummarise) == 0 {
		return messages
	}

	// Build a transcript from the messages to summarise.
	var transcript strings.Builder
	for _, m := range toSummarise {
		role := m.Role
		if role == "" {
			role = "unknown"
		}
		content := m.Content
		if len(content) > 2000 {
			content = content[:2000] + "\n[... truncated ...]"
		}
		transcript.WriteString(fmt.Sprintf("[%s]: %s\n\n", role, content))
	}

	summaryPrompt := s.SummaryPrompt
	if summaryPrompt == "" {
		summaryPrompt = defaultSummaryPrompt
	}

	summaryMessages := []Message{
		{Role: "system", Content: summaryPrompt},
		{Role: "user", Content: transcript.String()},
	}

	summary, err := s.Provider.Complete(context.Background(), summaryMessages, CompletionOptions{MaxTokens: 512})
	if err != nil {
		// Graceful fallback: use tiered compaction instead.
		cfg := DefaultContextConfig()
		cfg.MaxTokens = maxTokens
		cfg.KeepRecent = keepRecent
		return CompactMessagesTiered(messages, cfg)
	}

	summaryMsg := Message{
		Role:    "system",
		Content: fmt.Sprintf("[Conversation summary — %d messages compacted]\n\n%s", len(toSummarise), summary),
	}

	result := make([]Message, 0, 2+len(tail))
	result = append(result, system, summaryMsg)
	result = append(result, tail...)
	return result
}

// WithLLMCompaction configures the agent to use LLM-assisted context
// compaction. The agent's own provider is used for summarisation.
func (a *Agent) WithLLMCompaction(keepRecent int) *Agent {
	a.contextConfig.Strategy = &LLMCompactionStrategy{
		Provider:   a.provider,
		KeepRecent: keepRecent,
	}
	return a
}
