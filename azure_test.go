package iteragent_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	iteragent "github.com/GrayCodeAI/iteragent"
)

// ---------------------------------------------------------------------------
// NewAzureOpenAI
// ---------------------------------------------------------------------------

func TestNewAzureOpenAI_NonNil(t *testing.T) {
	p := iteragent.NewAzureOpenAI(iteragent.AzureOpenAIConfig{
		Endpoint:   "https://my.openai.azure.com",
		Deployment: "gpt-4o",
		APIKey:     "test-key",
	})
	if p == nil {
		t.Fatal("expected non-nil provider")
	}
}

func TestNewAzureOpenAI_Name(t *testing.T) {
	p := iteragent.NewAzureOpenAI(iteragent.AzureOpenAIConfig{
		Endpoint:   "https://my.openai.azure.com",
		Deployment: "gpt-4o",
		APIKey:     "test-key",
	})
	name := p.Name()
	if !strings.Contains(name, "azure") {
		t.Errorf("expected 'azure' in Name(), got %q", name)
	}
}

// ---------------------------------------------------------------------------
// Complete — via httptest
// ---------------------------------------------------------------------------

// azureResponse builds a valid Azure OpenAI chat completion response.
func azureResponse(content string) []byte {
	resp := map[string]interface{}{
		"choices": []map[string]interface{}{
			{"message": map[string]string{"content": content, "role": "assistant"}},
		},
	}
	b, _ := json.Marshal(resp)
	return b
}

func newTestAzureProvider(serverURL string) *iteragent.AzureOpenAIProvider {
	return iteragent.NewAzureOpenAI(iteragent.AzureOpenAIConfig{
		Endpoint:   serverURL,
		Deployment: "gpt-4o",
		APIKey:     "test-azure-key",
		APIVersion: "2024-02-15-preview",
	})
}

func TestAzureComplete_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write(azureResponse("azure says hello"))
	}))
	defer srv.Close()

	p := newTestAzureProvider(srv.URL)
	result, err := p.Complete(context.Background(), []iteragent.Message{
		{Role: "user", Content: "hi"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "azure says hello" {
		t.Errorf("expected 'azure says hello', got %q", result)
	}
}

func TestAzureComplete_Non200Error(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "rate limited", http.StatusTooManyRequests)
	}))
	defer srv.Close()

	p := newTestAzureProvider(srv.URL)
	_, err := p.Complete(context.Background(), []iteragent.Message{
		{Role: "user", Content: "hi"},
	})
	if err == nil {
		t.Fatal("expected error for non-200 response")
	}
	if !strings.Contains(err.Error(), "Azure OpenAI error") {
		t.Errorf("expected 'Azure OpenAI error' in error, got %q", err.Error())
	}
}

func TestAzureComplete_EmptyChoices(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		resp := map[string]interface{}{"choices": []interface{}{}}
		b, _ := json.Marshal(resp)
		w.Write(b)
	}))
	defer srv.Close()

	p := newTestAzureProvider(srv.URL)
	_, err := p.Complete(context.Background(), []iteragent.Message{
		{Role: "user", Content: "hi"},
	})
	if err == nil {
		t.Fatal("expected error for empty choices")
	}
	if !strings.Contains(err.Error(), "no response choices") {
		t.Errorf("expected 'no response choices', got %q", err.Error())
	}
}

func TestAzureComplete_InvalidJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("not valid json"))
	}))
	defer srv.Close()

	p := newTestAzureProvider(srv.URL)
	_, err := p.Complete(context.Background(), []iteragent.Message{
		{Role: "user", Content: "hi"},
	})
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestAzureComplete_APIKeyHeader(t *testing.T) {
	var capturedKey string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedKey = r.Header.Get("api-key")
		w.Header().Set("Content-Type", "application/json")
		w.Write(azureResponse("ok"))
	}))
	defer srv.Close()

	p := iteragent.NewAzureOpenAI(iteragent.AzureOpenAIConfig{
		Endpoint:   srv.URL,
		Deployment: "gpt-4o",
		APIKey:     "my-azure-key",
		APIVersion: "2024-02-15-preview",
	})
	_, _ = p.Complete(context.Background(), []iteragent.Message{{Role: "user", Content: "hi"}})

	if capturedKey != "my-azure-key" {
		t.Errorf("expected api-key 'my-azure-key', got %q", capturedKey)
	}
}

func TestAzureComplete_URLContainsDeployment(t *testing.T) {
	var capturedPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		w.Write(azureResponse("ok"))
	}))
	defer srv.Close()

	p := iteragent.NewAzureOpenAI(iteragent.AzureOpenAIConfig{
		Endpoint:   srv.URL,
		Deployment: "my-deployment",
		APIKey:     "k",
		APIVersion: "2024-02-15-preview",
	})
	_, _ = p.Complete(context.Background(), []iteragent.Message{{Role: "user", Content: "hi"}})

	if !strings.Contains(capturedPath, "my-deployment") {
		t.Errorf("expected deployment name in URL path, got %q", capturedPath)
	}
}

func TestAzureComplete_URLContainsAPIVersion(t *testing.T) {
	var capturedQuery string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedQuery = r.URL.RawQuery
		w.Header().Set("Content-Type", "application/json")
		w.Write(azureResponse("ok"))
	}))
	defer srv.Close()

	p := iteragent.NewAzureOpenAI(iteragent.AzureOpenAIConfig{
		Endpoint:   srv.URL,
		Deployment: "gpt-4o",
		APIKey:     "k",
		APIVersion: "2025-01-01",
	})
	_, _ = p.Complete(context.Background(), []iteragent.Message{{Role: "user", Content: "hi"}})

	if !strings.Contains(capturedQuery, "2025-01-01") {
		t.Errorf("expected api-version in query, got %q", capturedQuery)
	}
}

func TestAzureComplete_DefaultAPIVersion(t *testing.T) {
	var capturedQuery string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedQuery = r.URL.RawQuery
		w.Header().Set("Content-Type", "application/json")
		w.Write(azureResponse("ok"))
	}))
	defer srv.Close()

	// No APIVersion set — should default to 2024-02-15-preview
	p := iteragent.NewAzureOpenAI(iteragent.AzureOpenAIConfig{
		Endpoint:   srv.URL,
		Deployment: "gpt-4o",
		APIKey:     "k",
	})
	_, _ = p.Complete(context.Background(), []iteragent.Message{{Role: "user", Content: "hi"}})

	if !strings.Contains(capturedQuery, "2024-02-15-preview") {
		t.Errorf("expected default api-version in query, got %q", capturedQuery)
	}
}

func TestAzureComplete_RequestBody_MessagesForwarded(t *testing.T) {
	var body map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/json")
		w.Write(azureResponse("ok"))
	}))
	defer srv.Close()

	p := newTestAzureProvider(srv.URL)
	_, _ = p.Complete(context.Background(), []iteragent.Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "Hello"},
	})

	if body["messages"] == nil {
		t.Error("expected messages in request body")
	}
	msgs, ok := body["messages"].([]interface{})
	if !ok || len(msgs) != 2 {
		t.Errorf("expected 2 messages, got %v", body["messages"])
	}
}

func TestAzureComplete_MaxTokens_Forwarded(t *testing.T) {
	var body map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/json")
		w.Write(azureResponse("ok"))
	}))
	defer srv.Close()

	p := iteragent.NewAzureOpenAI(iteragent.AzureOpenAIConfig{
		Endpoint:   srv.URL,
		Deployment: "gpt-4o",
		APIKey:     "k",
		APIVersion: "2024-02-15-preview",
		MaxTokens:  1024,
	})
	_, _ = p.Complete(context.Background(), []iteragent.Message{{Role: "user", Content: "hi"}})

	if v, ok := body["max_tokens"].(float64); !ok || int(v) != 1024 {
		t.Errorf("expected max_tokens=1024, got %v", body["max_tokens"])
	}
}

func TestAzureComplete_ContextCancelled(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	p := newTestAzureProvider(srv.URL)
	_, err := p.Complete(ctx, []iteragent.Message{{Role: "user", Content: "hi"}})
	if err == nil {
		t.Fatal("expected error from cancelled context")
	}
}

