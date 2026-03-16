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
// Supported values: ollama, openai, anthropic, groq, gemini, nvidia (default: gemini)
// If apiKey is provided, it takes priority over environment variables.
func NewProvider(providerName string, apiKey ...string) (Provider, error) {
	providedKey := ""
	if len(apiKey) > 0 {
		providedKey = apiKey[0]
	}

	if providerName == "" {
		providerName = os.Getenv("ITERATE_PROVIDER")
	}
	if providerName == "" {
		providerName = "gemini"
	}

	switch providerName {
	case "ollama":
		key := providedKey
		if key == "" {
			key = os.Getenv("OPENAI_API_KEY")
		}
		return NewOpenAICompat(OpenAICompatConfig{
			BaseURL: getEnvOr("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
			Model:   getEnvOr("ITERATE_MODEL", "llama3"),
			APIKey:  key,
		}), nil

	case "openai":
		key := providedKey
		if key == "" {
			key = os.Getenv("OPENAI_API_KEY")
		}
		if key == "" {
			return nil, fmt.Errorf("OPENAI_API_KEY is required for openai provider (or use --api-key)")
		}
		return NewOpenAICompat(OpenAICompatConfig{
			BaseURL: "https://api.openai.com/v1",
			Model:   getEnvOr("ITERATE_MODEL", "gpt-4o"),
			APIKey:  key,
		}), nil

	case "anthropic":
		key := providedKey
		if key == "" {
			key = os.Getenv("ANTHROPIC_API_KEY")
		}
		if key == "" {
			return nil, fmt.Errorf("ANTHROPIC_API_KEY is required for anthropic provider (or use --api-key)")
		}
		return NewAnthropic(AnthropicConfig{
			Model:  getEnvOr("ITERATE_MODEL", "claude-sonnet-4-6"),
			APIKey: key,
		}), nil

	case "groq":
		key := providedKey
		if key == "" {
			key = os.Getenv("GROQ_API_KEY")
		}
		if key == "" {
			return nil, fmt.Errorf("GROQ_API_KEY is required for groq provider (or use --api-key)")
		}
		return NewOpenAICompat(OpenAICompatConfig{
			BaseURL: "https://api.groq.com/openai/v1",
			Model:   getEnvOr("ITERATE_MODEL", "llama-3.3-70b-versatile"),
			APIKey:  key,
		}), nil

	case "gemini":
		key := providedKey
		if key == "" {
			key = os.Getenv("GEMINI_API_KEY")
		}
		if key == "" {
			return nil, fmt.Errorf("GEMINI_API_KEY is required for gemini provider (or use --api-key)")
		}
		return NewGemini(GeminiConfig{
			Model:  getEnvOr("ITERATE_MODEL", "gemini-2.0-flash"),
			APIKey: key,
		}), nil

	case "nvidia":
		key := providedKey
		if key == "" {
			key = os.Getenv("NVIDIA_API_KEY")
		}
		if key == "" {
			return nil, fmt.Errorf("NVIDIA_API_KEY is required for nvidia provider (or use --api-key)")
		}
		return NewNvidia(OpenAICompatConfig{
			BaseURL: getEnvOr("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
			Model:   getEnvOr("ITERATE_MODEL", "nvidia/llama-3.3-nemotron-70b-instruct"),
			APIKey:  key,
		}), nil

	default:
		baseURL := os.Getenv("ITERATE_BASE_URL")
		if baseURL == "" {
			return nil, fmt.Errorf("unknown provider %q — set ITERATE_BASE_URL for custom endpoints", providerName)
		}
		key := providedKey
		if key == "" {
			key = os.Getenv("ITERATE_API_KEY")
		}
		return NewOpenAICompat(OpenAICompatConfig{
			BaseURL: baseURL,
			Model:   getEnvOr("ITERATE_MODEL", "default"),
			APIKey:  key,
		}), nil
	}
}

func getEnvOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
