package iteragent_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	iteragent "github.com/GrayCodeAI/iteragent"
)

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// openaiJSONResponse builds a valid OpenAI chat completion response body.
func openaiJSONResponse(content string) []byte {
	resp := map[string]interface{}{
		"choices": []map[string]interface{}{
			{"message": map[string]string{"content": content, "role": "assistant"}},
		},
	}
	b, _ := json.Marshal(resp)
	return b
}

// openaiErrorResponse builds an OpenAI error response body.
func openaiErrorResponse(msg string) []byte {
	resp := map[string]interface{}{
		"error": map[string]string{"message": msg},
	}
	b, _ := json.Marshal(resp)
	return b
}

// openaiCompatProvider returns a provider pointed at the given test server URL.
func newTestOpenAICompat(serverURL, model string) iteragent.Provider {
	return iteragent.NewOpenAICompat(iteragent.OpenAICompatConfig{
		BaseURL: serverURL,
		Model:   model,
		APIKey:  "test-key",
	})
}

// ---------------------------------------------------------------------------
// NewOpenAICompat
// ---------------------------------------------------------------------------

func TestNewOpenAICompat_NonNil(t *testing.T) {
	p := iteragent.NewOpenAICompat(iteragent.OpenAICompatConfig{
		BaseURL: "https://api.openai.com/v1",
		Model:   "gpt-4o",
		APIKey:  "sk-test",
	})
	if p == nil {
		t.Fatal("expected non-nil provider")
	}
}

func TestNewOpenAICompat_Name(t *testing.T) {
	p := iteragent.NewOpenAICompat(iteragent.OpenAICompatConfig{
		BaseURL: "https://api.openai.com/v1",
		Model:   "gpt-4o-mini",
		APIKey:  "sk-test",
	})
	name := p.Name()
	if !strings.Contains(name, "gpt-4o-mini") {
		t.Errorf("expected model in Name(), got %q", name)
	}
	if !strings.Contains(name, "openai-compat") {
		t.Errorf("expected 'openai-compat' in Name(), got %q", name)
	}
}

func TestNewOpenAICompat_DifferentModels(t *testing.T) {
	models := []string{"gpt-4.1", "gpt-4o", "o4-mini", "llama-4-scout"}
	for _, model := range models {
		p := iteragent.NewOpenAICompat(iteragent.OpenAICompatConfig{Model: model})
		if !strings.Contains(p.Name(), model) {
			t.Errorf("expected model %q in Name(), got %q", model, p.Name())
		}
	}
}

// ---------------------------------------------------------------------------
// Complete — via httptest
// ---------------------------------------------------------------------------

func TestOpenAICompatComplete_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write(openaiJSONResponse("hello from openai"))
	}))
	defer srv.Close()

	p := newTestOpenAICompat(srv.URL, "gpt-4o")
	result, err := p.Complete(context.Background(), []iteragent.Message{
		{Role: "user", Content: "hi"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "hello from openai" {
		t.Errorf("expected 'hello from openai', got %q", result)
	}
}

func TestOpenAICompatComplete_APIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write(openaiErrorResponse("rate limit exceeded"))
	}))
	defer srv.Close()

	p := newTestOpenAICompat(srv.URL, "gpt-4o")
	_, err := p.Complete(context.Background(), []iteragent.Message{
		{Role: "user", Content: "hi"},
	})
	if err == nil {
		t.Fatal("expected error for API error response")
	}
	if !strings.Contains(err.Error(), "rate limit exceeded") {
		t.Errorf("expected error message to contain 'rate limit exceeded', got %q", err.Error())
	}
}

func TestOpenAICompatComplete_EmptyChoices(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		resp := map[string]interface{}{"choices": []interface{}{}}
		b, _ := json.Marshal(resp)
		w.Write(b)
	}))
	defer srv.Close()

	p := newTestOpenAICompat(srv.URL, "gpt-4o")
	_, err := p.Complete(context.Background(), []iteragent.Message{
		{Role: "user", Content: "hi"},
	})
	if err == nil {
		t.Fatal("expected error for empty choices")
	}
	if !strings.Contains(err.Error(), "empty response") {
		t.Errorf("expected 'empty response' in error, got %q", err.Error())
	}
}

func TestOpenAICompatComplete_Non200Status(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Return valid-looking JSON but with a 400 — the provider reads body anyway
		w.WriteHeader(http.StatusBadRequest)
		w.Write(openaiErrorResponse("bad request"))
	}))
	defer srv.Close()

	p := newTestOpenAICompat(srv.URL, "gpt-4o")
	_, err := p.Complete(context.Background(), []iteragent.Message{
		{Role: "user", Content: "hi"},
	})
	// Either gets an error from the JSON error field or an HTTP error
	if err == nil {
		t.Fatal("expected error for 400 status")
	}
}

func TestOpenAICompatComplete_InvalidJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte("not valid json {{{{"))
	}))
	defer srv.Close()

	p := newTestOpenAICompat(srv.URL, "gpt-4o")
	_, err := p.Complete(context.Background(), []iteragent.Message{
		{Role: "user", Content: "hi"},
	})
	if err == nil {
		t.Fatal("expected error for invalid JSON response")
	}
}

func TestOpenAICompatComplete_AuthHeader(t *testing.T) {
	var capturedAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		w.Write(openaiJSONResponse("ok"))
	}))
	defer srv.Close()

	p := iteragent.NewOpenAICompat(iteragent.OpenAICompatConfig{
		BaseURL: srv.URL,
		Model:   "gpt-4o",
		APIKey:  "my-secret-key",
	})
	_, _ = p.Complete(context.Background(), []iteragent.Message{{Role: "user", Content: "hi"}})

	if capturedAuth != "Bearer my-secret-key" {
		t.Errorf("expected 'Bearer my-secret-key', got %q", capturedAuth)
	}
}

func TestOpenAICompatComplete_ContentTypeHeader(t *testing.T) {
	var capturedCT string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedCT = r.Header.Get("Content-Type")
		w.Header().Set("Content-Type", "application/json")
		w.Write(openaiJSONResponse("ok"))
	}))
	defer srv.Close()

	p := newTestOpenAICompat(srv.URL, "gpt-4o")
	_, _ = p.Complete(context.Background(), []iteragent.Message{{Role: "user", Content: "hi"}})

	if capturedCT != "application/json" {
		t.Errorf("expected Content-Type 'application/json', got %q", capturedCT)
	}
}

func TestOpenAICompatComplete_RequestBody_Model(t *testing.T) {
	var body map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/json")
		w.Write(openaiJSONResponse("ok"))
	}))
	defer srv.Close()

	p := newTestOpenAICompat(srv.URL, "gpt-4o-mini")
	_, _ = p.Complete(context.Background(), []iteragent.Message{{Role: "user", Content: "hi"}})

	if body["model"] != "gpt-4o-mini" {
		t.Errorf("expected model 'gpt-4o-mini' in request body, got %v", body["model"])
	}
}

func TestOpenAICompatComplete_RequestBody_Messages(t *testing.T) {
	var body map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/json")
		w.Write(openaiJSONResponse("ok"))
	}))
	defer srv.Close()

	p := newTestOpenAICompat(srv.URL, "gpt-4o")
	msgs := []iteragent.Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "What is 2+2?"},
	}
	_, _ = p.Complete(context.Background(), msgs)

	if body["messages"] == nil {
		t.Error("expected messages in request body")
	}
}

func TestOpenAICompatComplete_ContextCancelled(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done() // hang until client cancels
	}))
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	p := newTestOpenAICompat(srv.URL, "gpt-4o")
	_, err := p.Complete(ctx, []iteragent.Message{{Role: "user", Content: "hi"}})
	if err == nil {
		t.Fatal("expected error from cancelled context")
	}
}

// ---------------------------------------------------------------------------
// CompleteStream — via httptest SSE server
// ---------------------------------------------------------------------------

func openaiSSEServer(t *testing.T, tokens []string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, tok := range tokens {
			payload := fmt.Sprintf(`{"choices":[{"delta":{"content":%q}}]}`, tok)
			fmt.Fprintf(w, "data: %s\n\n", payload)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
}

func TestOpenAICompatSSEParsing_MultipleTokens(t *testing.T) {
	srv := openaiSSEServer(t, []string{"Hello", " ", "world"})
	defer srv.Close()

	client := iteragent.NewSSEClient()
	var collected strings.Builder
	err := client.Stream(context.Background(), srv.URL, nil, []byte(`{}`), func(e iteragent.SSEEvent) {
		if e.Data == "[DONE]" {
			return
		}
		if tok, ok := iteragent.ParseOpenAISSE(e.Data); ok {
			collected.WriteString(tok)
		}
	})
	if err != nil {
		t.Fatalf("stream error: %v", err)
	}
	if got := collected.String(); got != "Hello world" {
		t.Errorf("expected 'Hello world', got %q", got)
	}
}

func TestOpenAICompatSSEParsing_DoneIgnored(t *testing.T) {
	srv := openaiSSEServer(t, []string{"only"})
	defer srv.Close()

	client := iteragent.NewSSEClient()
	var tokens []string
	_ = client.Stream(context.Background(), srv.URL, nil, []byte(`{}`), func(e iteragent.SSEEvent) {
		if e.Data == "[DONE]" {
			return
		}
		if tok, ok := iteragent.ParseOpenAISSE(e.Data); ok {
			tokens = append(tokens, tok)
		}
	})
	if len(tokens) != 1 || tokens[0] != "only" {
		t.Errorf("expected ['only'], got %v", tokens)
	}
}

// ---------------------------------------------------------------------------
// openaiReasoningEffort mapping (via ParseOpenAISSE which is already tested;
// here we verify the mapping behaviour at the request-body level)
// ---------------------------------------------------------------------------

func TestOpenAICompatComplete_ReasoningEffort_OpenAIURL(t *testing.T) {
	// The reasoning_effort field is only added for openai.com base URLs.
	// We can't intercept the request body to openai.com, so we verify the
	// non-openai path (Groq, etc.) does NOT add reasoning_effort.
	var body map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/json")
		w.Write(openaiJSONResponse("ok"))
	}))
	defer srv.Close()

	// Non-openai.com URL — no reasoning_effort expected
	p := iteragent.NewOpenAICompat(iteragent.OpenAICompatConfig{
		BaseURL: srv.URL,
		Model:   "llama-3",
		APIKey:  "k",
	})
	_, _ = p.Complete(context.Background(), []iteragent.Message{{Role: "user", Content: "hi"}})

	if _, exists := body["reasoning_effort"]; exists {
		t.Error("expected no reasoning_effort for non-openai.com provider")
	}
}

func TestOpenAICompatComplete_MaxTokens_Forwarded(t *testing.T) {
	var body map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/json")
		w.Write(openaiJSONResponse("ok"))
	}))
	defer srv.Close()

	p := newTestOpenAICompat(srv.URL, "gpt-4o")
	opts := iteragent.CompletionOptions{MaxTokens: 512}
	_, _ = p.Complete(context.Background(), []iteragent.Message{{Role: "user", Content: "hi"}}, opts)

	if v, ok := body["max_tokens"].(float64); !ok || int(v) != 512 {
		t.Errorf("expected max_tokens=512, got %v", body["max_tokens"])
	}
}

func TestOpenAICompatComplete_Temperature_Forwarded(t *testing.T) {
	var body map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/json")
		w.Write(openaiJSONResponse("ok"))
	}))
	defer srv.Close()

	p := newTestOpenAICompat(srv.URL, "gpt-4o")
	opts := iteragent.CompletionOptions{Temperature: 0.7}
	_, _ = p.Complete(context.Background(), []iteragent.Message{{Role: "user", Content: "hi"}}, opts)

	if v, ok := body["temperature"].(float64); !ok || v != 0.7 {
		t.Errorf("expected temperature=0.7, got %v", body["temperature"])
	}
}

func TestOpenAICompatComplete_ZeroTemperature_NotForwarded(t *testing.T) {
	var body map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/json")
		w.Write(openaiJSONResponse("ok"))
	}))
	defer srv.Close()

	p := newTestOpenAICompat(srv.URL, "gpt-4o")
	// No temperature in opts — should not be forwarded
	_, _ = p.Complete(context.Background(), []iteragent.Message{{Role: "user", Content: "hi"}})

	if _, exists := body["temperature"]; exists {
		t.Error("expected temperature not forwarded when unset")
	}
}
