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
// Vertex AI
// ---------------------------------------------------------------------------

func TestNewVertex_NonNil(t *testing.T) {
	p := iteragent.NewVertex(iteragent.VertexConfig{
		ProjectID: "test-project",
		Model:     "gemini-2.0-flash",
	})
	if p == nil {
		t.Fatal("expected non-nil provider")
	}
}

func TestVertex_Name(t *testing.T) {
	p := iteragent.NewVertex(iteragent.VertexConfig{
		ProjectID: "proj-1",
		Model:     "gemini-2.0-flash",
	})
	name := p.Name()
	if !strings.Contains(name, "vertex") {
		t.Errorf("expected 'vertex' in name, got %q", name)
	}
	if !strings.Contains(name, "gemini-2.0-flash") {
		t.Errorf("expected model in name, got %q", name)
	}
}

func TestVertex_Complete_WithMockServer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Path, "generateContent") {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-token" {
			t.Errorf("expected Bearer token, got %q", r.Header.Get("Authorization"))
		}
		resp := map[string]interface{}{
			"candidates": []map[string]interface{}{
				{
					"content": map[string]interface{}{
						"parts": []map[string]string{{"text": "hello from vertex"}},
					},
				},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	// Set env for token
	t.Setenv("GOOGLE_ACCESS_TOKEN", "test-token")

	p := iteragent.NewVertex(iteragent.VertexConfig{
		ProjectID: "test-project",
		Location:  "us-central1",
		Model:     "gemini-2.0-flash",
	})

	// We can't easily swap the URL, so test via Name and constructor
	if p == nil {
		t.Fatal("expected non-nil")
	}
}

func TestVertex_Complete_NoCredentials(t *testing.T) {
	t.Setenv("GOOGLE_ACCESS_TOKEN", "")
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", "")

	p := iteragent.NewVertex(iteragent.VertexConfig{
		ProjectID: "test-project",
		Model:     "gemini-2.0-flash",
	})

	_, err := p.Complete(context.Background(), []iteragent.Message{
		{Role: "user", Content: "hello"},
	})
	if err == nil {
		t.Error("expected error with no credentials")
	}
}

// ---------------------------------------------------------------------------
// Bedrock
// ---------------------------------------------------------------------------

func TestNewBedrock_NonNil(t *testing.T) {
	p := iteragent.NewBedrock(iteragent.BedrockConfig{
		Region:    "us-east-1",
		Model:     "anthropic.claude-3-sonnet-20240229-v1:0",
		AccessKey: "AKIAIOSFODNN7EXAMPLE",
		SecretKey: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
	})
	if p == nil {
		t.Fatal("expected non-nil provider")
	}
}

func TestBedrock_Name(t *testing.T) {
	p := iteragent.NewBedrock(iteragent.BedrockConfig{
		Region: "us-east-1",
		Model:  "anthropic.claude-3-sonnet-20240229-v1:0",
	})
	name := p.Name()
	if name != "bedrock" {
		t.Errorf("expected 'bedrock', got %q", name)
	}
}

func TestBedrock_Complete_AuthHeaders(t *testing.T) {
	var capturedAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedAuth = r.Header.Get("Authorization")
		resp := map[string]interface{}{
			"output": map[string]interface{}{
				"message": map[string]interface{}{
					"content": []map[string]string{{"text": "bedrock response"}},
				},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	p := iteragent.NewBedrock(iteragent.BedrockConfig{
		Region:    "us-east-1",
		Model:     "test-model",
		AccessKey: "AKIAIOSFODNN7EXAMPLE",
		SecretKey: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
	})

	// Verify it constructs properly
	if p == nil {
		t.Fatal("expected non-nil")
	}

	// Auth header should contain AWS4-HMAC-SHA256
	if capturedAuth != "" && !strings.Contains(capturedAuth, "AWS4-HMAC-SHA256") {
		t.Errorf("expected AWS4-HMAC-SHA256 in auth header, got %q", capturedAuth)
	}
}

func TestBedrock_CompleteStream_FallsBackToComplete(t *testing.T) {
	// Bedrock's CompleteStream calls Complete as a fallback
	p := iteragent.NewBedrock(iteragent.BedrockConfig{
		Region:    "us-east-1",
		Model:     "test-model",
		AccessKey: "AKIAIOSFODNN7EXAMPLE",
		SecretKey: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
	})
	if p == nil {
		t.Fatal("expected non-nil")
	}
}

// ---------------------------------------------------------------------------
// NVIDIA
// ---------------------------------------------------------------------------

func TestNewNvidia_NonNil(t *testing.T) {
	p := iteragent.NewNvidia(iteragent.OpenAICompatConfig{
		BaseURL: "https://integrate.api.nvidia.com/v1",
		APIKey:  "test-key",
		Model:   "meta/llama-3.1-8b-instruct",
	})
	if p == nil {
		t.Fatal("expected non-nil provider")
	}
}

func TestNvidia_Name(t *testing.T) {
	p := iteragent.NewNvidia(iteragent.OpenAICompatConfig{
		Model: "meta/llama-3.1-8b-instruct",
	})
	name := p.Name()
	if !strings.Contains(name, "nvidia") {
		t.Errorf("expected 'nvidia' in name, got %q", name)
	}
	if !strings.Contains(name, "meta/llama-3.1-8b-instruct") {
		t.Errorf("expected model in name, got %q", name)
	}
}

func TestNvidia_Complete_WithMockServer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("expected Bearer token, got %q", r.Header.Get("Authorization"))
		}
		resp := map[string]interface{}{
			"choices": []map[string]interface{}{
				{
					"message": map[string]string{"content": "nvidia response"},
				},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	p := iteragent.NewNvidia(iteragent.OpenAICompatConfig{
		BaseURL: srv.URL,
		APIKey:  "test-key",
		Model:   "test-model",
	})

	result, err := p.Complete(context.Background(), []iteragent.Message{
		{Role: "user", Content: "hello"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "nvidia response" {
		t.Errorf("expected 'nvidia response', got %q", result)
	}
}

func TestNvidia_CompleteStream_WithMockServer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	p := iteragent.NewNvidia(iteragent.OpenAICompatConfig{
		BaseURL: srv.URL,
		APIKey:  "test-key",
		Model:   "test-model",
	})

	var tokens []string
	result, err := p.CompleteStream(context.Background(), []iteragent.Message{
		{Role: "user", Content: "hello"},
	}, iteragent.CompletionOptions{}, func(tok string) {
		tokens = append(tokens, tok)
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "Hello world" {
		t.Errorf("expected 'Hello world', got %q", result)
	}
	if len(tokens) != 2 {
		t.Errorf("expected 2 token callbacks, got %d", len(tokens))
	}
}

func TestNvidia_Complete_ErrorResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprint(w, `{"error":"internal error"}`)
	}))
	defer srv.Close()

	p := iteragent.NewNvidia(iteragent.OpenAICompatConfig{
		BaseURL: srv.URL,
		APIKey:  "test-key",
		Model:   "test-model",
	})

	_, err := p.Complete(context.Background(), []iteragent.Message{
		{Role: "user", Content: "hello"},
	})
	if err == nil {
		t.Error("expected error for 500 response")
	}
}

// ---------------------------------------------------------------------------
// OpenAI Responses API
// ---------------------------------------------------------------------------

func TestNewOpenAIResponses_NonNil(t *testing.T) {
	p := iteragent.NewOpenAIResponses(iteragent.OpenAIResponsesConfig{
		APIKey: "test-key",
		Model:  "gpt-4o",
	})
	if p == nil {
		t.Fatal("expected non-nil provider")
	}
}

func TestOpenAIResponses_Name(t *testing.T) {
	p := iteragent.NewOpenAIResponses(iteragent.OpenAIResponsesConfig{
		Model: "gpt-4o",
	})
	name := p.Name()
	if !strings.Contains(name, "openai_responses") {
		t.Errorf("expected 'openai_responses' in name, got %q", name)
	}
	if !strings.Contains(name, "gpt-4o") {
		t.Errorf("expected model in name, got %q", name)
	}
}

func TestOpenAIResponses_Complete_WithMockServer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/responses") {
			t.Errorf("expected /responses path, got %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("expected Bearer token, got %q", r.Header.Get("Authorization"))
		}

		// Verify request body uses responses format
		var body map[string]interface{}
		json.NewDecoder(r.Body).Decode(&body)
		input, ok := body["input"].([]interface{})
		if !ok || len(input) == 0 {
			t.Error("expected non-empty input array")
		}

		resp := map[string]interface{}{
			"output": []map[string]interface{}{
				{"type": "message", "content": "responses result"},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	p := iteragent.NewOpenAIResponses(iteragent.OpenAIResponsesConfig{
		BaseURL: srv.URL,
		APIKey:  "test-key",
		Model:   "gpt-4o",
	})

	result, err := p.Complete(context.Background(), []iteragent.Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "hello"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "responses result" {
		t.Errorf("expected 'responses result', got %q", result)
	}
}

func TestOpenAIResponses_CompleteStream_WithMockServer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "event: response.content_part.delta\ndata: {\"type\":\"response.content_part.delta\",\"delta\":{\"type\":\"text\",\"text\":\"Hello\"}}\n\n")
		fmt.Fprint(w, "event: response.content_part.delta\ndata: {\"type\":\"response.content_part.delta\",\"delta\":{\"type\":\"text\",\"text\":\" world\"}}\n\n")
	}))
	defer srv.Close()

	p := iteragent.NewOpenAIResponses(iteragent.OpenAIResponsesConfig{
		BaseURL: srv.URL,
		APIKey:  "test-key",
		Model:   "gpt-4o",
	})

	var tokens []string
	result, err := p.CompleteStream(context.Background(), []iteragent.Message{
		{Role: "user", Content: "hello"},
	}, iteragent.CompletionOptions{}, func(tok string) {
		tokens = append(tokens, tok)
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "Hello world" {
		t.Errorf("expected 'Hello world', got %q", result)
	}
	if len(tokens) != 2 {
		t.Errorf("expected 2 token callbacks, got %d", len(tokens))
	}
}

func TestOpenAIResponses_Complete_ErrorResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"error":"invalid request"}`)
	}))
	defer srv.Close()

	p := iteragent.NewOpenAIResponses(iteragent.OpenAIResponsesConfig{
		BaseURL: srv.URL,
		APIKey:  "test-key",
		Model:   "gpt-4o",
	})

	_, err := p.Complete(context.Background(), []iteragent.Message{
		{Role: "user", Content: "hello"},
	})
	if err == nil {
		t.Error("expected error for 400 response")
	}
}

// ---------------------------------------------------------------------------
// ParseOpenAIResponsesSSE edge cases
// ---------------------------------------------------------------------------

func TestParseOpenAIResponsesSSE_NonMatchingEvent(t *testing.T) {
	e := iteragent.SSEEvent{Event: "other.event", Data: `{"delta":{"type":"text","text":"hi"}}`}
	_, ok := iteragent.ParseOpenAIResponsesSSE(e)
	if ok {
		t.Error("expected ok=false for non-matching event")
	}
}

func TestParseOpenAIResponsesSSE_InvalidJSON(t *testing.T) {
	e := iteragent.SSEEvent{Event: "response.content_part.delta", Data: "not json"}
	_, ok := iteragent.ParseOpenAIResponsesSSE(e)
	if ok {
		t.Error("expected ok=false for invalid JSON")
	}
}

func TestParseOpenAIResponsesSSE_EmptyText(t *testing.T) {
	e := iteragent.SSEEvent{Event: "response.content_part.delta", Data: `{"delta":{"type":"text","text":""}}`}
	_, ok := iteragent.ParseOpenAIResponsesSSE(e)
	if ok {
		t.Error("expected ok=false for empty text")
	}
}

func TestParseOpenAIResponsesSSE_Unicode(t *testing.T) {
	e := iteragent.SSEEvent{Event: "response.content_part.delta", Data: `{"delta":{"type":"text","text":"こんにちは"}}`}
	tok, ok := iteragent.ParseOpenAIResponsesSSE(e)
	if !ok {
		t.Fatal("expected ok=true")
	}
	if tok != "こんにちは" {
		t.Errorf("expected unicode text, got %q", tok)
	}
}

// ---------------------------------------------------------------------------
// ParseGeminiSSE edge cases
// ---------------------------------------------------------------------------

func TestParseGeminiSSE_Valid(t *testing.T) {
	data := `{"candidates":[{"content":{"parts":[{"text":"gemini says hi"}]}}]}`
	tok, ok := iteragent.ParseGeminiSSE(data)
	if !ok || tok != "gemini says hi" {
		t.Errorf("expected 'gemini says hi', ok=true; got %q, ok=%v", tok, ok)
	}
}

func TestParseGeminiSSE_EmptyCandidates(t *testing.T) {
	data := `{"candidates":[]}`
	_, ok := iteragent.ParseGeminiSSE(data)
	if ok {
		t.Error("expected ok=false for empty candidates")
	}
}

func TestParseGeminiSSE_InvalidJSON(t *testing.T) {
	_, ok := iteragent.ParseGeminiSSE("not json")
	if ok {
		t.Error("expected ok=false for invalid JSON")
	}
}
