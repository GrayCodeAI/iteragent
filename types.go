package iteragent

import (
	"encoding/json"
	"time"
)

// Version is the current version of the iteragent library.
const Version = "0.3.0"

type ContentType string

const (
	ContentTypeText     ContentType = "text"
	ContentTypeImage    ContentType = "image"
	ContentTypeThinking ContentType = "thinking"
	ContentTypeToolCall ContentType = "toolCall"
)

type Content struct {
	Type     ContentType      `json:"type"`
	Text     string           `json:"text,omitempty"`
	Thinking string           `json:"thinking,omitempty"`
	Image    *ContentImage    `json:"image,omitempty"`
	ToolCall *ContentToolCall `json:"toolCall,omitempty"`
}

type ContentImage struct {
	Data     string `json:"data"`
	MimeType string `json:"mimeType"`
}

type ContentToolCall struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// Message represents a single message in a conversation with an LLM.
type Message struct {
	// Role is the message role, such as "system", "user", "assistant", or "tool".
	Role string `json:"role"`
	// Content is the message text content.
	Content string `json:"content"`
	// Timestamp is the Unix timestamp in milliseconds when the message was created.
	Timestamp int64 `json:"timestamp,omitempty"`
	// Usage tracks token usage for this message if provided by the LLM provider.
	Usage *Usage `json:"usage,omitempty"`
	// Error holds any error message associated with this message.
	Error string `json:"error,omitempty"`
	// StopReason indicates why the LLM stopped generating.
	StopReason string `json:"stopReason,omitempty"`
	// Model is the name of the LLM model that generated this message.
	Model string `json:"model,omitempty"`
	// Provider identifies the LLM provider used.
	Provider string `json:"provider,omitempty"`
}

func NewUserMessage(content string) Message {
	return Message{
		Role:      "user",
		Content:   content,
		Timestamp: time.Now().UnixMilli(),
	}
}

func NewAssistantMessage(content string) Message {
	return Message{
		Role:      "assistant",
		Content:   content,
		Timestamp: time.Now().UnixMilli(),
	}
}

func NewSystemMessage(content string) Message {
	return Message{
		Role:      "system",
		Content:   content,
		Timestamp: time.Now().UnixMilli(),
	}
}

type ExtensionMessage struct {
	Role string                 `json:"role"`
	Kind string                 `json:"kind"`
	Data map[string]interface{} `json:"data"`
}

type StopReason string

const (
	StopReasonStop    StopReason = "stop"
	StopReasonLength  StopReason = "length"
	StopReasonToolUse StopReason = "tool_use"
	StopReasonError   StopReason = "error"
	StopReasonAborted StopReason = "aborted"
)

type Usage struct {
	InputTokens  int `json:"inputTokens"`
	OutputTokens int `json:"outputTokens"`
	TotalTokens  int `json:"totalTokens"`
	CacheRead    int `json:"cacheRead,omitempty"`
	CacheWrite   int `json:"cacheWrite,omitempty"`
}

func (u *Usage) CacheHitRate() float64 {
	if u.InputTokens == 0 {
		return 0.0
	}
	return float64(u.CacheRead) / float64(u.InputTokens)
}

type ThinkingLevel string

const (
	ThinkingLevelOff     ThinkingLevel = "off"
	ThinkingLevelMinimal ThinkingLevel = "minimal"
	ThinkingLevelLow     ThinkingLevel = "low"
	ThinkingLevelMedium  ThinkingLevel = "medium"
	ThinkingLevelHigh    ThinkingLevel = "high"
)

type CacheStrategy string

const (
	CacheStrategyAuto     CacheStrategy = "auto"
	CacheStrategyDisabled CacheStrategy = "disabled"
	CacheStrategyManual   CacheStrategy = "manual"
)

type CacheConfig struct {
	Enabled       bool
	Strategy      CacheStrategy
	CacheSystem   bool
	CacheTools    bool
	CacheMessages bool
}

func DefaultCacheConfig() CacheConfig {
	return CacheConfig{
		Enabled:       true,
		Strategy:      CacheStrategyAuto,
		CacheSystem:   true,
		CacheTools:    true,
		CacheMessages: true,
	}
}

// ToolExecutionStrategy selects how multiple concurrent tool calls are executed.
type ToolExecutionStrategy int

const (
	ToolExecSequential ToolExecutionStrategy = iota
	ToolExecParallel
	ToolExecBatched
)

// ToolExecConfig holds the tool execution strategy and its parameters.
type ToolExecConfig struct {
	Strategy  ToolExecutionStrategy
	BatchSize int // only used for ToolExecBatched
}

// DefaultToolExecConfig returns the default tool execution configuration (parallel).
func DefaultToolExecConfig() ToolExecConfig {
	return ToolExecConfig{Strategy: ToolExecParallel}
}

func NewParallelStrategy() ToolExecConfig {
	return ToolExecConfig{Strategy: ToolExecParallel}
}

func NewSequentialStrategy() ToolExecConfig {
	return ToolExecConfig{Strategy: ToolExecSequential}
}

func NewBatchedStrategy(size int) ToolExecConfig {
	return ToolExecConfig{Strategy: ToolExecBatched, BatchSize: size}
}

type ToolContext struct {
	ToolCallID string
	ToolName   string
	Cancel     <-chan struct{}
	OnUpdate   func(ToolResult)
	OnProgress func(string)
}

type ToolResult struct {
	Content []Content
	Details map[string]interface{}
	IsError bool
}

func NewToolResult(content string) ToolResult {
	return ToolResult{
		Content: []Content{{Type: ContentTypeText, Text: content}},
		IsError: false,
	}
}

func NewErrorResult(err string) ToolResult {
	return ToolResult{
		Content: []Content{{Type: ContentTypeText, Text: err}},
		IsError: true,
	}
}

type ToolError struct {
	Failed      string
	NotFound    string
	InvalidArgs string
	Cancelled   bool
}

func (e *ToolError) Error() string {
	if e.Failed != "" {
		return e.Failed
	}
	if e.NotFound != "" {
		return "Tool not found: " + e.NotFound
	}
	if e.InvalidArgs != "" {
		return "Invalid arguments: " + e.InvalidArgs
	}
	if e.Cancelled {
		return "Tool execution cancelled"
	}
	return "Unknown tool error"
}

type EventType string

const (
	EventAgentStart          EventType = "agent_start"
	EventAgentEnd            EventType = "agent_end"
	EventTurnStart           EventType = "turn_start"
	EventTurnEnd             EventType = "turn_end"
	EventMessageStart        EventType = "message_start"
	EventMessageUpdate       EventType = "message_update"
	EventMessageEnd          EventType = "message_end"
	EventToolExecutionStart  EventType = "tool_execution_start"
	EventToolExecutionUpdate EventType = "tool_execution_update"
	EventToolExecutionEnd    EventType = "tool_execution_end"
	EventProgressMessage     EventType = "progress_message"
	EventInputRejected       EventType = "input_rejected"
	EventInputWarned         EventType = "input_warned"
	EventContextCompacted    EventType = "context_compacted"
	EventTokenUpdate         EventType = "token_update" // incremental token from streaming provider
	EventError               EventType = "error"
)

type AgentEvent struct {
	Type    EventType
	Message string
	Data    map[string]interface{}
}

func NewAgentEvent(eventType EventType, message string) AgentEvent {
	return AgentEvent{
		Type:    eventType,
		Message: message,
	}
}

type InputFilterResult int

const (
	InputFilterAllow InputFilterResult = iota
	InputFilterReject
	InputFilterWarn
)

type InputFilter interface {
	Filter(input string) (filtered string, result InputFilterResult, reason string)
}

type CompactionStrategy interface {
	Compact(messages []Message, maxTokens int) []Message
}

type DefaultCompactionStrategy struct{}

func (c *DefaultCompactionStrategy) Compact(messages []Message, maxTokens int) []Message {
	cfg := DefaultContextConfig()
	cfg.MaxTokens = maxTokens
	return CompactMessagesTiered(messages, cfg)
}

// ToolDefinition describes a tool that an LLM can invoke.
type ToolDefinition struct {
	// Name is the unique identifier for the tool.
	Name string
	// Label is a human-readable label shown to the LLM.
	Label string
	// Description explains what the tool does, shown to the LLM for tool selection.
	Description string
	// Parameters is a map of parameter names to their schema/metadata.
	Parameters map[string]interface{}
	// Schema is the raw JSON Schema defining the tool's input format.
	Schema json.RawMessage
}

type ProviderError struct {
	Code    string
	Message string
	Retry   bool
}

func (e *ProviderError) Error() string {
	return e.Message
}

