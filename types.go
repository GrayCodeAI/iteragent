package iteragent

import (
	"context"
	"encoding/json"
	"time"
)

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

type Message struct {
	Role       string `json:"role"`
	Content    string `json:"content"`
	Timestamp  int64  `json:"timestamp,omitempty"`
	Usage      *Usage `json:"usage,omitempty"`
	Error      string `json:"error,omitempty"`
	StopReason string `json:"stopReason,omitempty"`
	Model      string `json:"model,omitempty"`
	Provider   string `json:"provider,omitempty"`
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
	totalInput := u.InputTokens + u.CacheRead + u.CacheWrite
	if totalInput == 0 {
		return 0.0
	}
	return float64(u.CacheRead) / float64(totalInput)
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
	Strategy       ToolExecutionStrategy
	BatchSize      int // only used for ToolExecBatched
	MaxOutputLines int // cap tool result to this many lines (head+tail); 0 = 200
}

// DefaultToolExecConfig returns the default tool execution configuration (parallel).
func DefaultToolExecConfig() ToolExecConfig {
	return ToolExecConfig{Strategy: ToolExecParallel, MaxOutputLines: 200}
}

// Legacy string-based constants kept for compatibility.
const (
	ToolExecTypeParallel   = "parallel"
	ToolExecTypeSequential = "sequential"
	ToolExecTypeBatched    = "batched"
)

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
	EventTokenUpdate         EventType = "token_update"    // incremental token from streaming provider
	EventThinkingUpdate      EventType = "thinking_update" // incremental thinking token (extended thinking)
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

type ToolDefinition struct {
	Name        string
	Label       string
	Description string
	Parameters  map[string]interface{}
	Schema      json.RawMessage
}

type StreamConfig struct {
	Model         string
	APIKey        string
	BaseURL       string
	Headers       map[string]string
	MaxTokens     int
	Temperature   float32
	ThinkingLevel ThinkingLevel
	Stream        bool
	APIVersion    string
}

type ModelConfig struct {
	Model                string
	BaseURL              string
	APIKey               string
	Headers              map[string]string
	MaxTokens            int
	Temperature          float32
	ThinkingLevel        ThinkingLevel
	ExtraBody            map[string]interface{}
	AuthStyle            string
	SupportsThinking     bool
	SupportsCacheControl bool
	ReasoningFormat      string
}

func DefaultModelConfig() ModelConfig {
	return ModelConfig{
		MaxTokens:     4096,
		Temperature:   0.7,
		ThinkingLevel: ThinkingLevelOff,
		AuthStyle:     "basic",
	}
}

type ApiProtocol string

const (
	ProtocolAnthropic       ApiProtocol = "anthropic"
	ProtocolOpenAI          ApiProtocol = "openai"
	ProtocolOpenAICompat    ApiProtocol = "openai_compat"
	ProtocolOpenAIResponses ApiProtocol = "openai_responses"
	ProtocolAzureOpenAI     ApiProtocol = "azure_openai"
	ProtocolGoogle          ApiProtocol = "google"
	ProtocolGoogleVertex    ApiProtocol = "google_vertex"
	ProtocolBedrock         ApiProtocol = "bedrock"
)

type StreamEventType string

const (
	StreamEventContent      StreamEventType = "content"
	StreamEventContentBlock StreamEventType = "content_block"
	StreamEventContentStart StreamEventType = "content_block_start"
	StreamEventContentStop  StreamEventType = "content_block_stop"
	StreamEventMessageStart StreamEventType = "message_start"
	StreamEventMessageDelta StreamEventType = "message_delta"
	StreamEventMessageStop  StreamEventType = "message_stop"
	StreamEventError        StreamEventType = "error"
	StreamEventPing         StreamEventType = "ping"
)

type StreamEvent struct {
	Type    StreamEventType
	Index   int
	Content string
	Error   string
}

type ProviderError struct {
	Code    string
	Message string
	Retry   bool
}

func (e *ProviderError) Error() string {
	return e.Message
}

type ProviderRegistry struct {
	providers map[ApiProtocol]StreamProvider
}

func NewProviderRegistry() *ProviderRegistry {
	return &ProviderRegistry{
		providers: make(map[ApiProtocol]StreamProvider),
	}
}

func (r *ProviderRegistry) Register(protocol ApiProtocol, provider StreamProvider) {
	r.providers[protocol] = provider
}

func (r *ProviderRegistry) Get(protocol ApiProtocol) (StreamProvider, bool) {
	p, ok := r.providers[protocol]
	return p, ok
}

func (r *ProviderRegistry) Has(protocol ApiProtocol) bool {
	_, ok := r.providers[protocol]
	return ok
}

func (r *ProviderRegistry) Protocols() []ApiProtocol {
	protocols := make([]ApiProtocol, 0, len(r.providers))
	for p := range r.providers {
		protocols = append(protocols, p)
	}
	return protocols
}

type StreamProvider interface {
	Stream(ctx context.Context, config StreamConfig, messages []Message, onEvent func(StreamEvent)) (Message, error)
}
