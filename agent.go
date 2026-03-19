package iteragent

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"sync"

	"github.com/GrayCodeAI/iteragent/mcp"
	"github.com/GrayCodeAI/iteragent/openapi"
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

// AgentHooks provides optional callbacks for lifecycle events in the agent loop.
// All fields are optional; nil functions are silently skipped.
type AgentHooks struct {
	// BeforeTurn is called before each provider completion turn. turn is 1-indexed.
	BeforeTurn func(turn int, messages []Message)
	// AfterTurn is called after each provider completion turn with the response.
	AfterTurn func(turn int, response string)
	// OnToolStart is called before each tool execution.
	OnToolStart func(toolName string, args map[string]string)
	// OnToolEnd is called after each tool execution with the result and any error.
	OnToolEnd func(toolName string, result string, err error)
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

	// Context compaction config.
	contextConfig ContextConfig

	// Tool execution config.
	toolExecConfig ToolExecConfig

	// Input filters.
	inputFilters []InputFilter

	// Lifecycle hooks.
	hooks AgentHooks

	// Concurrency state — tracks whether a streaming operation is in progress.
	mu          sync.Mutex
	isStreaming bool
	pendingWg   sync.WaitGroup

	// activeEvents is set during streaming so emit() can route events there.
	activeEvents chan Event

	// cancelFn cancels the running goroutine when Reset() is called.
	cancelFn context.CancelFunc

	// cacheConfig controls prompt caching behaviour.
	cacheConfig CacheConfig

	// mcpClients holds MCP server connections owned by this agent.
	// They are shut down when Close() is called.
	mcpClients []*mcp.McpClient
}

// New creates a new Agent.
func New(p Provider, tools []Tool, logger *slog.Logger) *Agent {
	toolMap := make(map[string]Tool, len(tools))
	for _, t := range tools {
		toolMap[t.Name] = t
	}
	return &Agent{
		provider:       p,
		tools:          toolMap,
		logger:         logger,
		Events:         make(chan Event, 64),
		ThinkingLevel:  ThinkingLevelOff,
		MaxTokens:      4096,
		Temperature:    0.7,
		Messages:       []Message{},
		contextConfig:  DefaultContextConfig(),
		toolExecConfig: DefaultToolExecConfig(),
	}
}

// completionOpts builds a CompletionOptions from the Agent's current settings.
func (a *Agent) completionOpts() CompletionOptions {
	var cache *CacheConfig
	if a.cacheConfig.Enabled {
		c := a.cacheConfig
		cache = &c
	}
	return CompletionOptions{
		ThinkingLevel: a.ThinkingLevel,
		MaxTokens:     a.MaxTokens,
		Temperature:   a.Temperature,
		CacheConfig:   cache,
	}
}

// maybeCompact checks if messages exceed the warning threshold and compacts them.
// Returns the (possibly compacted) messages and a bool indicating compaction happened.
func (a *Agent) maybeCompact(messages []Message, emitFn func(Event)) []Message {
	cfg := a.contextConfig
	warningTokens := cfg.WarningTokens()
	if warningTokens == 0 {
		warningTokens = int(float64(cfg.MaxTokens) * 0.8)
	}
	current := EstimateTotalTokens(messages)
	if current < warningTokens {
		return messages
	}
	compacted := CompactMessagesTiered(messages, cfg)
	emitFn(Event{Type: string(EventContextCompacted), Content: "context compacted"})
	return compacted
}

// runFilters applies all registered input filters to the user content.
// Returns the (possibly modified) content, and a bool indicating whether to abort.
func (a *Agent) runFilters(content string, emitFn func(Event)) (string, bool) {
	for _, f := range a.inputFilters {
		filtered, result, reason := f.Filter(content)
		switch result {
		case InputFilterReject:
			emitFn(Event{Type: string(EventInputRejected), Content: reason, IsError: true})
			return content, true
		case InputFilterWarn:
			emitFn(Event{Type: string(EventInputWarned), Content: reason})
			content = filtered
		default:
			content = filtered
		}
	}
	return content, false
}

// executeTools runs tool calls using the configured strategy and returns the combined results string.
func (a *Agent) executeTools(ctx context.Context, calls []ToolCall, iteration int, emitFn func(Event)) string {
	switch a.toolExecConfig.Strategy {
	case ToolExecParallel:
		return a.executeToolsParallel(ctx, calls, iteration, emitFn)
	case ToolExecBatched:
		return a.executeToolsBatched(ctx, calls, iteration, emitFn)
	default: // ToolExecSequential
		return a.executeToolsSequential(ctx, calls, iteration, emitFn)
	}
}

func (a *Agent) executeToolsSequential(ctx context.Context, calls []ToolCall, iteration int, emitFn func(Event)) string {
	var toolResults strings.Builder
	for _, call := range calls {
		result, isError := a.executeSingleTool(ctx, call, iteration, emitFn)
		if isError && result == fmt.Sprintf("unknown tool: %s", call.Tool) {
			toolResults.WriteString(fmt.Sprintf("Tool %s: %s\n", call.Tool, result))
		} else {
			toolResults.WriteString(fmt.Sprintf("Tool %s result:\n%s\n\n", call.Tool, result))
		}
	}
	return toolResults.String()
}

func (a *Agent) executeToolsParallel(ctx context.Context, calls []ToolCall, iteration int, emitFn func(Event)) string {
	type indexedResult struct {
		call    ToolCall
		result  string
		isError bool
		unknown bool
	}
	results := make([]indexedResult, len(calls))

	var wg sync.WaitGroup
	for idx, call := range calls {
		wg.Add(1)
		go func(i int, c ToolCall) {
			defer wg.Done()
			emitFn(Event{
				Type:       string(EventToolExecutionStart),
				ToolName:   c.Tool,
				ToolCallID: fmt.Sprintf("%s-%d-%d", c.Tool, iteration, i),
			})
			tool, ok := a.tools[c.Tool]
			if !ok {
				res := fmt.Sprintf("unknown tool: %s", c.Tool)
				if a.hooks.OnToolEnd != nil {
					a.hooks.OnToolEnd(c.Tool, res, fmt.Errorf("unknown tool: %s", c.Tool))
				}
				results[i] = indexedResult{call: c, result: res, isError: true, unknown: true}
				emitFn(Event{Type: string(EventToolExecutionEnd), ToolName: c.Tool, Result: results[i].result, IsError: true})
				return
			}
			if a.hooks.OnToolStart != nil {
				a.hooks.OnToolStart(c.Tool, c.Args)
			}
			res, err := tool.Execute(ctx, c.Args)
			isErr := false
			if err != nil {
				res = fmt.Sprintf("ERROR: %s\nOutput: %s", err.Error(), res)
				isErr = true
			}
			if a.hooks.OnToolEnd != nil {
				a.hooks.OnToolEnd(c.Tool, res, err)
			}
			results[i] = indexedResult{call: c, result: res, isError: isErr}
			emitFn(Event{Type: string(EventToolExecutionEnd), ToolName: c.Tool, Result: res, IsError: isErr})
		}(idx, call)
	}
	wg.Wait()

	var sb strings.Builder
	for _, r := range results {
		if r.unknown {
			sb.WriteString(fmt.Sprintf("Tool %s: %s\n", r.call.Tool, r.result))
		} else {
			sb.WriteString(fmt.Sprintf("Tool %s result:\n%s\n\n", r.call.Tool, r.result))
		}
	}
	return sb.String()
}

func (a *Agent) executeToolsBatched(ctx context.Context, calls []ToolCall, iteration int, emitFn func(Event)) string {
	batchSize := a.toolExecConfig.BatchSize
	if batchSize <= 0 {
		batchSize = 4
	}
	var sb strings.Builder
	for start := 0; start < len(calls); start += batchSize {
		end := start + batchSize
		if end > len(calls) {
			end = len(calls)
		}
		batch := calls[start:end]
		sb.WriteString(a.executeToolsParallel(ctx, batch, iteration, emitFn))
	}
	return sb.String()
}

// executeSingleTool runs one tool call and emits start/end events.
// Returns (result string, isError bool).
func (a *Agent) executeSingleTool(ctx context.Context, call ToolCall, iteration int, emitFn func(Event)) (string, bool) {
	tool, ok := a.tools[call.Tool]
	if !ok {
		result := fmt.Sprintf("unknown tool: %s", call.Tool)
		emitFn(Event{Type: string(EventToolExecutionEnd), ToolName: call.Tool, Result: result, IsError: true})
		return result, true
	}

	argsMap := make(map[string]interface{})
	for k, v := range call.Args {
		argsMap[k] = v
	}
	emitFn(Event{
		Type:       string(EventToolExecutionStart),
		ToolName:   call.Tool,
		ToolCallID: fmt.Sprintf("%s-%d", call.Tool, iteration),
		Args:       argsMap,
	})
	a.logger.Info("executing tool", "tool", call.Tool, "args", call.Args)

	if a.hooks.OnToolStart != nil {
		a.hooks.OnToolStart(call.Tool, call.Args)
	}

	result, err := tool.Execute(ctx, call.Args)
	isError := false
	if err != nil {
		result = fmt.Sprintf("ERROR: %s\nOutput: %s", err.Error(), result)
		isError = true
	}

	if a.hooks.OnToolEnd != nil {
		a.hooks.OnToolEnd(call.Tool, result, err)
	}

	emitFn(Event{
		Type:     string(EventToolExecutionEnd),
		ToolName: call.Tool,
		Result:   result,
		IsError:  isError,
	})
	return result, isError
}

// Run executes the agent loop with the given system prompt and user message.
// An optional emitFn can be provided to route events to a specific channel.
func (a *Agent) Run(ctx context.Context, systemPrompt, userMessage string, emitFn ...func(Event)) (string, error) {
	emit := a.emit
	if len(emitFn) > 0 && emitFn[0] != nil {
		emit = emitFn[0]
	}

	// Apply input filters to the user message.
	filtered, reject := a.runFilters(userMessage, emit)
	if reject {
		return "", fmt.Errorf("input rejected by filter")
	}
	userMessage = filtered

	allTools := make([]Tool, 0, len(a.tools))
	for _, t := range a.tools {
		allTools = append(allTools, t)
	}

	messages := []Message{
		{Role: "system", Content: systemPrompt + "\n\n" + ToolDescriptions(allTools)},
		{Role: "user", Content: userMessage},
	}

	opts := a.completionOpts()

	const maxIterations = 20
	for i := 0; i < maxIterations; i++ {
		a.logger.Info("agent iteration", "step", i+1)
		emit(Event{Type: string(EventTurnStart), Content: fmt.Sprintf("turn %d", i+1)})

		// Context compaction check before provider call.
		messages = a.maybeCompact(messages, emit)

		if a.hooks.BeforeTurn != nil {
			a.hooks.BeforeTurn(i+1, messages)
		}

		var (
			response string
			err      error
		)
		if ts, ok := a.provider.(TokenStreamer); ok {
			response, err = RetryWithResult(ctx, DefaultRetryConfig, func() (string, error) {
				return ts.CompleteStream(ctx, messages, opts, func(token string) {
					emit(Event{Type: string(EventTokenUpdate), Content: token})
				})
			})
		} else {
			response, err = RetryWithResult(ctx, DefaultRetryConfig, func() (string, error) {
				return a.provider.Complete(ctx, messages, opts)
			})
		}
		if err != nil {
			emit(Event{Type: string(EventError), Content: err.Error(), IsError: true})
			return "", fmt.Errorf("provider error at step %d: %w", i+1, err)
		}

		if a.hooks.AfterTurn != nil {
			a.hooks.AfterTurn(i+1, response)
		}

		emit(Event{Type: string(EventMessageUpdate), Content: response})

		messages = append(messages, Message{
			Role:    "assistant",
			Content: response,
		})

		calls := ParseToolCalls(response)
		if len(calls) == 0 {
			emit(Event{Type: string(EventMessageEnd), Content: response})
			emit(Event{Type: string(EventTurnEnd), Content: ""})
			return response, nil
		}

		toolResultStr := a.executeTools(ctx, calls, i, emit)

		messages = append(messages, Message{
			Role:    "user",
			Content: toolResultStr,
		})
		emit(Event{Type: string(EventTurnEnd), Content: ""})
	}

	return "", fmt.Errorf("agent exceeded max iterations (%d)", maxIterations)
}

func (a *Agent) emit(e Event) {
	a.mu.Lock()
	active := a.activeEvents
	a.mu.Unlock()
	if active != nil {
		select {
		case active <- e:
		default:
		}
		return
	}
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
	sb.WriteString("IMPORTANT - How to call tools:\n")
	sb.WriteString("When you need to use a tool, you MUST use this EXACT format:\n\n")
	sb.WriteString("```tool\n")
	sb.WriteString("{\"tool\":\"TOOL_NAME\",\"args\":{\"arg1\":\"value1\"}}\n")
	sb.WriteString("```\n\n")
	sb.WriteString("Replace TOOL_NAME with the tool name and provide arguments as JSON.\n\n")
	sb.WriteString("Available tools:\n\n")
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

func (a *Agent) AddTool(tool Tool) *Agent {
	a.tools[tool.Name] = tool
	return a
}

// WithContextConfig sets the context compaction configuration.
func (a *Agent) WithContextConfig(cfg ContextConfig) *Agent {
	a.contextConfig = cfg
	return a
}

// WithToolExecutionStrategy sets the tool execution strategy.
func (a *Agent) WithToolExecutionStrategy(cfg ToolExecConfig) *Agent {
	a.toolExecConfig = cfg
	return a
}

// WithInputFilter adds an input filter to the agent.
func (a *Agent) WithInputFilter(f InputFilter) *Agent {
	a.inputFilters = append(a.inputFilters, f)
	return a
}

// WithHooks sets lifecycle hook callbacks on the agent.
func (a *Agent) WithHooks(h AgentHooks) *Agent {
	a.hooks = h
	return a
}

// WithCacheConfig enables prompt caching with the given configuration.
func (a *Agent) WithCacheConfig(cfg CacheConfig) *Agent {
	a.cacheConfig = cfg
	return a
}

// WithCacheEnabled is a convenience builder that enables or disables prompt
// caching using the DefaultCacheConfig when enabled is true.
func (a *Agent) WithCacheEnabled(enabled bool) *Agent {
	if enabled {
		a.cacheConfig = DefaultCacheConfig()
	} else {
		a.cacheConfig = CacheConfig{}
	}
	return a
}

// WithMcpServerStdio connects to an MCP server via stdio (spawns a child process),
// performs the initialize handshake, and registers all advertised tools.
// Returns an error if the server fails to start or initialize.
func (a *Agent) WithMcpServerStdio(ctx context.Context, command string, args ...string) (*Agent, error) {
	client, err := mcp.ConnectStdio(ctx, command, args, nil)
	if err != nil {
		return nil, fmt.Errorf("mcp stdio connect: %w", err)
	}
	return a.registerMcpTools(ctx, mcp.NewMcpToolAdapter(client))
}

// WithMcpServerHttp connects to an MCP server via HTTP, performs the initialize
// handshake, and registers all advertised tools.
func (a *Agent) WithMcpServerHttp(ctx context.Context, url string) (*Agent, error) {
	client, err := mcp.ConnectHTTP(ctx, url, nil)
	if err != nil {
		return nil, fmt.Errorf("mcp http connect: %w", err)
	}
	return a.registerMcpTools(ctx, mcp.NewMcpToolAdapter(client))
}

func (a *Agent) registerMcpTools(ctx context.Context, adapter *mcp.ToolAdapter) (*Agent, error) {
	tools, err := adapter.GetTools(ctx)
	if err != nil {
		return nil, fmt.Errorf("list mcp tools: %w", err)
	}
	for _, t := range tools {
		execute := t.Execute
		a.tools[t.Name] = Tool{
			Name:        t.Name,
			Description: t.Description,
			Execute:     execute,
		}
	}
	// Track the client so Close() can shut it down.
	if client := adapter.Client(); client != nil {
		a.mu.Lock()
		a.mcpClients = append(a.mcpClients, client)
		a.mu.Unlock()
	}
	return a, nil
}

// Close shuts down any MCP server connections owned by this agent,
// cancels any running operation, and waits for it to finish.
// Safe to call multiple times.
func (a *Agent) Close() error {
	a.Reset() // cancel + drain pending work

	a.mu.Lock()
	clients := a.mcpClients
	a.mcpClients = nil
	a.mu.Unlock()

	var errs []error
	for _, c := range clients {
		if err := c.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) > 0 {
		msgs := make([]string, len(errs))
		for i, e := range errs {
			msgs[i] = e.Error()
		}
		return fmt.Errorf("mcp close errors: %s", strings.Join(msgs, "; "))
	}
	return nil
}

// WithOpenApiFile loads an OpenAPI spec from a JSON file and registers its operations as tools.
func (a *Agent) WithOpenApiFile(path string, cfg openapi.Config) (*Agent, error) {
	adapter, err := openapi.FromFile(path, cfg)
	if err != nil {
		return nil, fmt.Errorf("openapi from file: %w", err)
	}
	return a.registerOpenApiTools(adapter)
}

// WithOpenApiUrl fetches an OpenAPI spec from a URL and registers its operations as tools.
func (a *Agent) WithOpenApiUrl(ctx context.Context, url string, cfg openapi.Config) (*Agent, error) {
	adapter, err := openapi.FromURL(ctx, url, cfg)
	if err != nil {
		return nil, fmt.Errorf("openapi from url: %w", err)
	}
	return a.registerOpenApiTools(adapter)
}

func (a *Agent) registerOpenApiTools(adapter *openapi.Adapter) (*Agent, error) {
	tools, err := adapter.GetTools()
	if err != nil {
		return nil, fmt.Errorf("list openapi tools: %w", err)
	}
	for _, t := range tools {
		execute := t.Execute
		a.tools[t.Name] = Tool{
			Name:        t.Name,
			Description: t.Description,
			Execute:     execute,
		}
	}
	return a, nil
}

// Finish blocks until the current streaming operation completes.
// Safe to call when no operation is running (no-op). Call this after
// draining the events channel returned by Prompt or PromptMessages
// when you need to access Messages right away.
func (a *Agent) Finish() {
	a.pendingWg.Wait()
}

// Reset cancels any running streaming operation, waits for it to finish,
// then clears the message history. Safe to call at any time.
func (a *Agent) Reset() {
	a.mu.Lock()
	if a.cancelFn != nil {
		a.cancelFn()
		a.cancelFn = nil
	}
	a.mu.Unlock()
	// Wait for the goroutine to exit before clearing state.
	a.pendingWg.Wait()
	a.mu.Lock()
	a.Messages = nil
	a.isStreaming = false
	a.mu.Unlock()
}

// Prompt sends a text prompt and returns an event channel immediately.
// The agent loop runs concurrently; call Finish() after draining events
// to ensure state is fully updated. Calls Finish() first to resolve
// any previous in-progress operation.
func (a *Agent) Prompt(ctx context.Context, text string) chan Event {
	a.Finish()

	loopCtx, cancel := context.WithCancel(ctx)
	events := make(chan Event, 64)

	a.mu.Lock()
	a.cancelFn = cancel
	a.isStreaming = true
	a.activeEvents = events
	a.mu.Unlock()

	// emitFn routes events to the per-call channel.
	emitFn := func(e Event) {
		select {
		case events <- e:
		default:
		}
	}

	a.pendingWg.Add(1)
	go func() {
		defer func() {
			a.pendingWg.Done()
			a.mu.Lock()
			a.isStreaming = false
			a.activeEvents = nil
			a.mu.Unlock()
			cancel()
			close(events)
		}()
		emitFn(Event{Type: string(EventMessageStart), Content: text})
		output, err := a.Run(loopCtx, a.SystemPrompt, text, emitFn)
		if err != nil {
			emitFn(Event{Type: string(EventError), Content: err.Error(), IsError: true})
		} else {
			emitFn(Event{Type: string(EventMessageEnd), Content: output})
		}
	}()
	return events
}

// PromptMessages sends a set of messages and returns an event channel immediately.
// The agent loop runs concurrently; call Finish() after draining events.
// Calls Finish() first to resolve any previous in-progress operation.
func (a *Agent) PromptMessages(ctx context.Context, messages []Message) chan Event {
	a.Finish()

	loopCtx, cancel := context.WithCancel(ctx)
	events := make(chan Event, 64)

	a.mu.Lock()
	a.cancelFn = cancel
	a.isStreaming = true
	a.activeEvents = events
	a.mu.Unlock()

	// emitFn routes events to the per-call channel.
	emitFn := func(e Event) {
		select {
		case events <- e:
		default:
		}
	}

	a.pendingWg.Add(1)
	go func() {
		defer func() {
			a.pendingWg.Done()
			a.mu.Lock()
			a.isStreaming = false
			a.activeEvents = nil
			a.mu.Unlock()
			cancel()
			close(events)
		}()

		allTools := a.GetTools()
		systemContent := a.SystemPrompt
		if systemContent != "" {
			systemContent += "\n\n"
		}
		systemContent += ToolDescriptions(allTools)

		fullMessages := []Message{{Role: "system", Content: systemContent}}
		fullMessages = append(fullMessages, messages...)

		// Apply input filters to the last user message if present.
		for i := len(fullMessages) - 1; i >= 0; i-- {
			if fullMessages[i].Role == "user" {
				filtered, reject := a.runFilters(fullMessages[i].Content, emitFn)
				if reject {
					emitFn(Event{Type: string(EventError), Content: "input rejected by filter", IsError: true})
					return
				}
				fullMessages[i].Content = filtered
				break
			}
		}

		emitFn(Event{Type: string(EventMessageStart), Content: ""})

		opts := a.completionOpts()

		for i := 0; i < 20; i++ {
			// Check for cancellation before each turn.
			select {
			case <-loopCtx.Done():
				emitFn(Event{Type: string(EventError), Content: loopCtx.Err().Error(), IsError: true})
				return
			default:
			}

			emitFn(Event{Type: string(EventTurnStart), Content: fmt.Sprintf("turn %d", i+1)})

			// Context compaction check before provider call.
			fullMessages = a.maybeCompact(fullMessages, emitFn)

			if a.hooks.BeforeTurn != nil {
				a.hooks.BeforeTurn(i+1, fullMessages)
			}

			var (
				response string
				turnErr  error
			)
			if ts, ok := a.provider.(TokenStreamer); ok {
				response, turnErr = RetryWithResult(loopCtx, DefaultRetryConfig, func() (string, error) {
					return ts.CompleteStream(loopCtx, fullMessages, opts, func(token string) {
						emitFn(Event{Type: string(EventTokenUpdate), Content: token})
					})
				})
			} else {
				response, turnErr = RetryWithResult(loopCtx, DefaultRetryConfig, func() (string, error) {
					return a.provider.Complete(loopCtx, fullMessages, opts)
				})
			}
			if turnErr != nil {
				emitFn(Event{Type: string(EventError), Content: turnErr.Error(), IsError: true})
				break
			}

			if a.hooks.AfterTurn != nil {
				a.hooks.AfterTurn(i+1, response)
			}

			emitFn(Event{Type: string(EventMessageUpdate), Content: response})

			fullMessages = append(fullMessages, Message{Role: "assistant", Content: response})

			calls := ParseToolCalls(response)
			if len(calls) == 0 {
				emitFn(Event{Type: string(EventMessageEnd), Content: response})
				break
			}

			toolResultStr := a.executeTools(loopCtx, calls, i, emitFn)

			fullMessages = append(fullMessages, Message{Role: "user", Content: toolResultStr})
			emitFn(Event{Type: string(EventTurnEnd), Content: ""})
		}
	}()
	return events
}
