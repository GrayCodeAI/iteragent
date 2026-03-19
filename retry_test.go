package iteragent_test

import (
	"context"
	"errors"
	"testing"
	"time"

	iteragent "github.com/GrayCodeAI/iteragent"
)

// TestIsRetryable verifies which error strings are considered retryable.
func TestIsRetryable(t *testing.T) {
	cases := []struct {
		err  error
		want bool
	}{
		// nil
		{nil, false},
		// explicit context errors — must NOT retry
		{context.Canceled, false},
		{context.DeadlineExceeded, false},
		// HTTP status codes
		{errors.New("API error 429: rate limited"), true},
		{errors.New("error (429): too many requests"), true},
		{errors.New("error 500: internal server error"), true},
		{errors.New("error 502: bad gateway"), true},
		{errors.New("error 503: service unavailable"), true},
		{errors.New("error 504: gateway timeout"), true},
		{errors.New("error 529: overloaded"), true},
		// Generic phrases
		{errors.New("rate limit exceeded"), true},
		{errors.New("rate_limit hit"), true},
		{errors.New("ratelimit"), true},
		{errors.New("too many requests"), true},
		{errors.New("quota exceeded"), true},
		{errors.New("connection refused"), true},
		{errors.New("connection reset"), true},
		{errors.New("connection error"), true},
		{errors.New("network error"), true},
		{errors.New("no such host"), true},
		{errors.New("request timeout"), true},
		{errors.New("timed out"), true},
		{errors.New("deadline exceeded"), true},
		{errors.New("temporary failure"), true},
		{errors.New("temporarily unavailable"), true},
		{errors.New("overloaded"), true},
		{errors.New("server error"), true},
		{errors.New("internal error"), true},
		{errors.New("service unavailable"), true},
		{errors.New("bad gateway"), true},
		{errors.New("broken pipe"), true},
		{errors.New("unexpected eof"), true},
		{errors.New("RATE LIMIT"), true}, // case-insensitive
		// Azure-specific
		{errors.New("Azure OpenAI error (429): quota"), true},
		{errors.New("Azure OpenAI error (500): server"), true},
		{errors.New("Azure OpenAI error (503): unavailable"), true},
		// Vertex-specific
		{errors.New("Vertex error (429): quota"), true},
		{errors.New("Vertex error (503): unavailable"), true},
		// Bedrock-specific
		{errors.New("ThrottlingException: slow down"), true},
		{errors.New("ServiceUnavailableException"), true},
		{errors.New("InternalServerError"), true},
		// Non-retryable
		{errors.New("invalid API key"), false},
		{errors.New("not found"), false},
		{errors.New("bad request"), false},
		{errors.New("401 unauthorized"), false},
		{errors.New("403 forbidden"), false},
		{errors.New("404 not found"), false},
	}

	for _, c := range cases {
		got := iteragent.IsRetryable(c.err)
		if got != c.want {
			t.Errorf("IsRetryable(%v) = %v, want %v", c.err, got, c.want)
		}
	}
}

// TestRetryableError verifies RetryableError is treated as retryable.
func TestRetryableError(t *testing.T) {
	err := &iteragent.RetryableError{Err: errors.New("custom retryable")}
	if !iteragent.IsRetryable(err) {
		t.Error("RetryableError should be retryable")
	}
	if err.Error() != "custom retryable" {
		t.Errorf("RetryableError.Error() = %q, want %q", err.Error(), "custom retryable")
	}
}

// TestRetryWithResult_Success verifies immediate success returns on first attempt.
func TestRetryWithResult_Success(t *testing.T) {
	calls := 0
	cfg := iteragent.RetryConfig{MaxAttempts: 3, InitialDelay: time.Millisecond, MaxDelay: time.Millisecond, Multiplier: 1}
	result, err := iteragent.RetryWithResult(context.Background(), cfg, func() (string, error) {
		calls++
		return "ok", nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "ok" {
		t.Errorf("got %q, want %q", result, "ok")
	}
	if calls != 1 {
		t.Errorf("expected 1 call, got %d", calls)
	}
}

// TestRetryWithResult_RetriesOnRetryable verifies retryable errors are retried up to MaxAttempts.
func TestRetryWithResult_RetriesOnRetryable(t *testing.T) {
	calls := 0
	cfg := iteragent.RetryConfig{MaxAttempts: 3, InitialDelay: time.Millisecond, MaxDelay: time.Millisecond, Multiplier: 1}
	_, err := iteragent.RetryWithResult(context.Background(), cfg, func() (string, error) {
		calls++
		return "", errors.New("rate limit hit")
	})
	if err == nil {
		t.Fatal("expected error after all retries")
	}
	if calls != 3 {
		t.Errorf("expected 3 calls, got %d", calls)
	}
}

// TestRetryWithResult_NoRetryOnNonRetryable verifies non-retryable errors fail immediately.
func TestRetryWithResult_NoRetryOnNonRetryable(t *testing.T) {
	calls := 0
	cfg := iteragent.RetryConfig{MaxAttempts: 3, InitialDelay: time.Millisecond, MaxDelay: time.Millisecond, Multiplier: 1}
	_, err := iteragent.RetryWithResult(context.Background(), cfg, func() (string, error) {
		calls++
		return "", errors.New("invalid api key")
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if calls != 1 {
		t.Errorf("expected 1 call (no retry), got %d", calls)
	}
}

// TestRetryWithResult_SucceedsOnSecondAttempt verifies recovery on second attempt.
func TestRetryWithResult_SucceedsOnSecondAttempt(t *testing.T) {
	calls := 0
	cfg := iteragent.RetryConfig{MaxAttempts: 3, InitialDelay: time.Millisecond, MaxDelay: time.Millisecond, Multiplier: 1}
	result, err := iteragent.RetryWithResult(context.Background(), cfg, func() (string, error) {
		calls++
		if calls < 2 {
			return "", errors.New("429 rate limited")
		}
		return "recovered", nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "recovered" {
		t.Errorf("got %q, want %q", result, "recovered")
	}
	if calls != 2 {
		t.Errorf("expected 2 calls, got %d", calls)
	}
}

// TestRetryWithResult_ContextCancellation verifies cancellation stops retries.
func TestRetryWithResult_ContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	calls := 0
	cfg := iteragent.RetryConfig{MaxAttempts: 5, InitialDelay: time.Second, MaxDelay: time.Second, Multiplier: 1}
	_, err := iteragent.RetryWithResult(ctx, cfg, func() (string, error) {
		calls++
		return "", errors.New("rate limit")
	})
	if err == nil {
		t.Fatal("expected error from cancelled context")
	}
	// Should have run at most once (first attempt) before context check.
	if calls > 1 {
		t.Errorf("expected at most 1 call with cancelled context, got %d", calls)
	}
}

// TestRetry_Success verifies Retry returns nil on immediate success.
func TestRetry_Success(t *testing.T) {
	cfg := iteragent.RetryConfig{MaxAttempts: 3, InitialDelay: time.Millisecond, MaxDelay: time.Millisecond, Multiplier: 1}
	err := iteragent.Retry(context.Background(), cfg, func() error {
		return nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// TestRetry_ExhaustsRetries verifies Retry returns the last error after MaxAttempts.
func TestRetry_ExhaustsRetries(t *testing.T) {
	calls := 0
	cfg := iteragent.RetryConfig{MaxAttempts: 2, InitialDelay: time.Millisecond, MaxDelay: time.Millisecond, Multiplier: 1}
	err := iteragent.Retry(context.Background(), cfg, func() error {
		calls++
		return errors.New("error 503: service unavailable")
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if calls != 2 {
		t.Errorf("expected 2 calls, got %d", calls)
	}
}
