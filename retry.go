// Package iteragent is a lightweight, zero-dependency Go agent framework
// for building LLM-powered applications with multiple provider support,
// tool use, streaming, and context compaction.
package iteragent

import (
	"context"
	"strings"
	"time"
)

// RetryConfig defines the parameters for retry behavior.
type RetryConfig struct {
	MaxAttempts  int           // Maximum number of retry attempts (default: 3)
	InitialDelay time.Duration // Initial delay between retries (default: 1s)
	MaxDelay     time.Duration // Maximum delay between retries (default: 30s)
	Multiplier   float64       // Multiplier applied to delay after each attempt (default: 2.0)
}

var DefaultRetryConfig = RetryConfig{
	MaxAttempts:  3,
	InitialDelay: time.Second,
	MaxDelay:     30 * time.Second,
	Multiplier:   2.0,
}

type RetryableError struct {
	Err error
}

func (e *RetryableError) Error() string {
	return e.Err.Error()
}

func IsRetryable(err error) bool {
	if err == nil {
		return false
	}

	// Explicit RetryableError wrapper always retries.
	if _, ok := err.(*RetryableError); ok {
		return true
	}

	// Context cancellation / deadline should NOT be retried.
	if err == context.Canceled || err == context.DeadlineExceeded {
		return false
	}

	errStr := err.Error()

	// HTTP status codes that indicate transient server-side problems.
	retryableStatusCodes := []string{
		"429",  // Too Many Requests
		"500",  // Internal Server Error
		"502",  // Bad Gateway
		"503",  // Service Unavailable
		"504",  // Gateway Timeout
		"529",  // Anthropic overloaded
	}
	for _, code := range retryableStatusCodes {
		// Match "API error 429:", "status 429", "error (429)", "HTTP 429", etc.
		if containsIgnoreCase(errStr, " "+code) ||
			containsIgnoreCase(errStr, "("+code+")") ||
			containsIgnoreCase(errStr, "error "+code) {
			return true
		}
	}

	// Transient network and infrastructure phrases (all providers).
	retryablePhrases := []string{
		"rate limit",
		"rate_limit",
		"ratelimit",
		"too many requests",
		"quota exceeded",
		"timeout",
		"timed out",
		"deadline exceeded",
		"temporary",
		"temporarily",
		"connection reset",
		"connection refused",
		"connection error",
		"no such host",
		"network",
		"eof",
		"broken pipe",
		"overloaded",
		"server error",
		"internal error",
		"service unavailable",
		"bad gateway",
		// Azure-specific
		"content filter",
		"azure openai error (429)",
		"azure openai error (500)",
		"azure openai error (503)",
		// Vertex-specific
		"vertex error (429)",
		"vertex error (500)",
		"vertex error (503)",
		// Bedrock-specific
		"throttlingexception",
		"serviceunavailableexception",
		"internalservererror",
	}

	for _, phrase := range retryablePhrases {
		if containsIgnoreCase(errStr, phrase) {
			return true
		}
	}

	return false
}

func containsIgnoreCase(s, substr string) bool {
	return strings.Contains(strings.ToLower(s), strings.ToLower(substr))
}

func Retry(ctx context.Context, cfg RetryConfig, fn func() error) error {
	var lastErr error
	delay := cfg.InitialDelay

	for attempt := 1; attempt <= cfg.MaxAttempts; attempt++ {
		err := fn()
		if err == nil {
			return nil
		}

		lastErr = err

		if !IsRetryable(err) || attempt == cfg.MaxAttempts {
			return err
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(delay):
		}

		delay = time.Duration(float64(delay) * cfg.Multiplier)
		if delay > cfg.MaxDelay {
			delay = cfg.MaxDelay
		}
	}

	return lastErr
}

func RetryWithResult[T any](ctx context.Context, cfg RetryConfig, fn func() (T, error)) (T, error) {
	var zero T

	var lastErr error
	delay := cfg.InitialDelay

	for attempt := 1; attempt <= cfg.MaxAttempts; attempt++ {
		result, err := fn()
		if err == nil {
			return result, nil
		}

		lastErr = err

		if !IsRetryable(err) || attempt == cfg.MaxAttempts {
			return zero, err
		}

		select {
		case <-ctx.Done():
			return zero, ctx.Err()
		case <-time.After(delay):
		}

		delay = time.Duration(float64(delay) * cfg.Multiplier)
		if delay > cfg.MaxDelay {
			delay = cfg.MaxDelay
		}
	}

	return zero, lastErr
}
