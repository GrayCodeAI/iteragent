package iteragent_test

import (
	"testing"

	iteragent "github.com/GrayCodeAI/iteragent"
)

func TestProviderContextWindow_Fallback(t *testing.T) {
	// A provider that does not implement ContextWindower should return 128k default.
	p := &sequentialProvider{}
	got := iteragent.ProviderContextWindow(p)
	if got != 128_000 {
		t.Errorf("want 128_000 fallback, got %d", got)
	}
}

func TestAnthropicContextWindow(t *testing.T) {
	tests := []struct {
		model string
		want  int
	}{
		{"claude-sonnet-4-6", 200_000},
		{"claude-3-5-sonnet-20241022", 200_000},
		{"claude-3-opus-20240229", 200_000},
		{"claude-3-haiku-20240307", 200_000},
		{"claude-instant-1.2", 100_000},
		{"claude-2.1", 100_000},
	}
	for _, tt := range tests {
		p := iteragent.NewAnthropic(iteragent.AnthropicConfig{Model: tt.model, APIKey: "key"})
		got := iteragent.ProviderContextWindow(p)
		if got != tt.want {
			t.Errorf("Anthropic model %q: want %d, got %d", tt.model, tt.want, got)
		}
	}
}

func TestOpenAICompatContextWindow(t *testing.T) {
	tests := []struct {
		model string
		want  int
	}{
		{"gpt-4o", 128_000},
		{"gpt-4o-mini", 128_000},
		{"gpt-4-turbo", 128_000},
		{"gpt-4", 8_192},
		{"gpt-4-32k", 32_768},
		{"gpt-3.5-turbo", 16_385},
		{"gpt-3.5-turbo-instruct", 4_096},
		{"o1", 128_000},
		{"o3-mini", 128_000},
		{"llama-3.3-70b-versatile", 128_000},
		{"llama3", 8_192},
		{"mistral-large", 32_768},
		{"deepseek-chat", 128_000},
	}
	for _, tt := range tests {
		p := iteragent.NewOpenAICompat(iteragent.OpenAICompatConfig{Model: tt.model, APIKey: "key", BaseURL: "http://localhost"})
		got := iteragent.ProviderContextWindow(p)
		if got != tt.want {
			t.Errorf("OpenAI-compat model %q: want %d, got %d", tt.model, tt.want, got)
		}
	}
}

func TestGeminiContextWindow(t *testing.T) {
	tests := []struct {
		model string
		want  int
	}{
		{"gemini-2.5-pro", 1_048_576},
		{"gemini-2.0-flash", 1_000_000},
		{"gemini-1.5-pro", 1_000_000},
		{"gemini-1.5-flash", 1_000_000},
		{"gemini-1.0-pro", 32_760},
		{"gemini-pro", 32_760},
	}
	for _, tt := range tests {
		p := iteragent.NewGemini(iteragent.GeminiConfig{Model: tt.model, APIKey: "key"})
		got := iteragent.ProviderContextWindow(p)
		if got != tt.want {
			t.Errorf("Gemini model %q: want %d, got %d", tt.model, tt.want, got)
		}
	}
}
