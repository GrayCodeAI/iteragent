package iteragent

import (
	"context"
	"fmt"
	"strings"
)

type MockProvider struct {
	model         string
	response      string
	toolCalls     []ToolCall
	toolCallIndex int
	error         error
}

func NewMock(response string) *MockProvider {
	return &MockProvider{
		model:    "mock",
		response: response,
	}
}

func NewMockWithTools(response string, toolCalls []ToolCall) *MockProvider {
	return &MockProvider{
		model:     "mock",
		response:  response,
		toolCalls: toolCalls,
	}
}

func NewMockWithError(err error) *MockProvider {
	return &MockProvider{
		model: "mock",
		error: err,
	}
}

func (p *MockProvider) Name() string {
	return fmt.Sprintf("mock(%s)", p.model)
}

func (p *MockProvider) Complete(ctx context.Context, messages []Message) (string, error) {
	if p.error != nil {
		return "", p.error
	}

	if p.toolCallIndex < len(p.toolCalls) {
		call := p.toolCalls[p.toolCallIndex]
		p.toolCallIndex++
		return fmt.Sprintf("```tool\n%s\n```", mustJson(call)), nil
	}

	return p.response, nil
}

func (p *MockProvider) Stream(ctx context.Context, config StreamConfig, messages []Message, onEvent func(StreamEvent)) (Message, error) {
	if p.error != nil {
		return Message{}, p.error
	}

	for _, char := range strings.Split(p.response, "") {
		select {
		case <-ctx.Done():
			return Message{}, ctx.Err()
		default:
		}
		onEvent(StreamEvent{
			Type:    StreamEventContent,
			Content: char,
		})
	}

	return Message{
		Role:    "assistant",
		Content: p.response,
	}, nil
}

func mustJson(v interface{}) string {
	switch v := v.(type) {
	case string:
		return v
	default:
		return fmt.Sprintf("%+v", v)
	}
}

type MockProviderBuilder struct {
	mock *MockProvider
}

func Mock() *MockProviderBuilder {
	return &MockProviderBuilder{
		mock: &MockProvider{
			model: "mock",
		},
	}
}

func (b *MockProviderBuilder) Text(text string) *MockProviderBuilder {
	b.mock.response = text
	return b
}

func (b *MockProviderBuilder) Model(model string) *MockProviderBuilder {
	b.mock.model = model
	return b
}

func (b *MockProviderBuilder) WithTools(toolCalls ...ToolCall) *MockProviderBuilder {
	b.mock.toolCalls = toolCalls
	return b
}

func (b *MockProviderBuilder) WithError(err error) *MockProviderBuilder {
	b.mock.error = err
	return b
}

func (b *MockProviderBuilder) Build() Provider {
	return b.mock
}

type MockStreamProvider struct {
	events []StreamEvent
	index  int
}

func NewMockStreamProvider(events []StreamEvent) *MockStreamProvider {
	return &MockStreamProvider{
		events: events,
	}
}

func (p *MockStreamProvider) Stream(ctx context.Context, config StreamConfig, messages []Message, onEvent func(StreamEvent)) (Message, error) {
	var content strings.Builder

	for _, event := range p.events {
		select {
		case <-ctx.Done():
			return Message{}, ctx.Err()
		default:
		}
		onEvent(event)
		if event.Content != "" {
			content.WriteString(event.Content)
		}
	}

	return Message{
		Role:    "assistant",
		Content: content.String(),
	}, nil
}
