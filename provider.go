package iteragent

import (
	"context"
	"fmt"
	"os"
)

// Provider is the unified LLM interface.
type Provider interface {
	Complete(ctx context.Context, messages []Message) (string, error)
	Name() string
}

// NewProvider returns the provider selected by ITERATE_PROVIDER.
// Supported values: ollama, openai, anthropic, groq, gemini (default: gemini)
func NewProvider(providerName string) (Provider, error) {
	if providerName == "" {
		providerName = os.Getenv("ITERATE_PROVIDER")
	}
	if providerName == "" {
		providerName = "gemini"
	}

	switch providerName {
	case "ollama":
		return NewOpenAICompat(OpenAICompatConfig{
			BaseURL: getEnvOr("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
			Model:   getEnvOr("ITERATE_MODEL", "llama3"),
			APIKey:  "ollama",
		}), nil

	case "openai":
		key := os.Getenv("OPENAI_API_KEY")
		if key == "" {
			return nil, fmt.Errorf("OPENAI_API_KEY is required for openai provider")
		}
		return NewOpenAICompat(OpenAICompatConfig{
			BaseURL: "https://api.openai.com/v1",
			Model:   getEnvOr("ITERATE_MODEL", "gpt-4o"),
			APIKey:  key,
		}), nil

	case "anthropic":
		key := os.Getenv("ANTHROPIC_API_KEY")
		if key == "" {
			return nil, fmt.Errorf("ANTHROPIC_API_KEY is required for anthropic provider")
		}
		return NewAnthropic(AnthropicConfig{
			Model:  getEnvOr("ITERATE_MODEL", "claude-sonnet-4-6"),
			APIKey: key,
		}), nil

	case "groq":
		key := os.Getenv("GROQ_API_KEY")
		if key == "" {
			return nil, fmt.Errorf("GROQ_API_KEY is required for groq provider")
		}
		return NewOpenAICompat(OpenAICompatConfig{
			BaseURL: "https://api.groq.com/openai/v1",
			Model:   getEnvOr("ITERATE_MODEL", "llama-3.3-70b-versatile"),
			APIKey:  key,
		}), nil

	case "gemini":
		key := os.Getenv("GEMINI_API_KEY")
		if key == "" {
			return nil, fmt.Errorf("GEMINI_API_KEY is required for gemini provider")
		}
		return NewGemini(GeminiConfig{
			Model:  getEnvOr("ITERATE_MODEL", "gemini-2.0-flash"),
			APIKey: key,
		}), nil

	default:
		baseURL := os.Getenv("ITERATE_BASE_URL")
		if baseURL == "" {
			return nil, fmt.Errorf("unknown provider %q — set ITERATE_BASE_URL for custom endpoints", providerName)
		}
		return NewOpenAICompat(OpenAICompatConfig{
			BaseURL: baseURL,
			Model:   getEnvOr("ITERATE_MODEL", "default"),
			APIKey:  getEnvOr("ITERATE_API_KEY", "none"),
		}), nil
	}
}

func getEnvOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
