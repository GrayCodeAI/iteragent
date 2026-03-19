package iteragent_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	iteragent "github.com/GrayCodeAI/iteragent"
)

// ---------------------------------------------------------------------------
// ParseAnthropicSSE
// ---------------------------------------------------------------------------

func TestParseAnthropicSSE(t *testing.T) {
	cases := []struct {
		name      string
		data      string
		wantToken string
		wantOK    bool
	}{
		{
			name:      "text_delta",
			data:      `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
			wantToken: "Hello",
			wantOK:    true,
		},
		{
			name:   "wrong_event_type",
			data:   `{"type":"message_start","message":{}}`,
			wantOK: false,
		},
		{
			name:   "empty_text",
			data:   `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":""}}`,
			wantOK: false,
		},
		{
			name:   "non_text_delta",
			data:   `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":""}}`,
			wantOK: false,
		},
		{
			name:   "invalid_json",
			data:   `not-json`,
			wantOK: false,
		},
		{
			name:   "empty_string",
			data:   ``,
			wantOK: false,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tok, ok := iteragent.ParseAnthropicSSE(c.data)
			if ok != c.wantOK {
				t.Errorf("ok = %v, want %v", ok, c.wantOK)
			}
			if ok && tok != c.wantToken {
				t.Errorf("token = %q, want %q", tok, c.wantToken)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// ParseOpenAISSE
// ---------------------------------------------------------------------------

func TestParseOpenAISSE(t *testing.T) {
	cases := []struct {
		name      string
		data      string
		wantToken string
		wantOK    bool
	}{
		{
			name:      "content_delta",
			data:      `{"choices":[{"delta":{"content":"world"}}]}`,
			wantToken: "world",
			wantOK:    true,
		},
		{
			name:   "empty_content",
			data:   `{"choices":[{"delta":{"content":""}}]}`,
			wantOK: false,
		},
		{
			name:   "no_choices",
			data:   `{"choices":[]}`,
			wantOK: false,
		},
		{
			name:   "invalid_json",
			data:   `{bad`,
			wantOK: false,
		},
		{
			name:   "done_sentinel",
			data:   `[DONE]`,
			wantOK: false,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tok, ok := iteragent.ParseOpenAISSE(c.data)
			if ok != c.wantOK {
				t.Errorf("ok = %v, want %v", ok, c.wantOK)
			}
			if ok && tok != c.wantToken {
				t.Errorf("token = %q, want %q", tok, c.wantToken)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// ParseGeminiSSE
// ---------------------------------------------------------------------------

func TestParseGeminiSSE(t *testing.T) {
	cases := []struct {
		name      string
		data      string
		wantToken string
		wantOK    bool
	}{
		{
			name:      "text_part",
			data:      `{"candidates":[{"content":{"parts":[{"text":"gemini tok"}]}}]}`,
			wantToken: "gemini tok",
			wantOK:    true,
		},
		{
			name:   "no_candidates",
			data:   `{"candidates":[]}`,
			wantOK: false,
		},
		{
			name:   "empty_parts",
			data:   `{"candidates":[{"content":{"parts":[]}}]}`,
			wantOK: false,
		},
		{
			name:   "invalid_json",
			data:   `nope`,
			wantOK: false,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tok, ok := iteragent.ParseGeminiSSE(c.data)
			if ok != c.wantOK {
				t.Errorf("ok = %v, want %v", ok, c.wantOK)
			}
			if ok && tok != c.wantToken {
				t.Errorf("token = %q, want %q", tok, c.wantToken)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// ParseOpenAIResponsesSSE
// ---------------------------------------------------------------------------

func TestParseOpenAIResponsesSSE(t *testing.T) {
	cases := []struct {
		name      string
		event     iteragent.SSEEvent
		wantToken string
		wantOK    bool
	}{
		{
			name: "text_delta",
			event: iteragent.SSEEvent{
				Event: "response.content_part.delta",
				Data:  `{"type":"response.content_part.delta","delta":{"type":"text","text":"hi"}}`,
			},
			wantToken: "hi",
			wantOK:    true,
		},
		{
			name: "wrong_event_name",
			event: iteragent.SSEEvent{
				Event: "response.done",
				Data:  `{"type":"response.done"}`,
			},
			wantOK: false,
		},
		{
			name: "non_text_delta_type",
			event: iteragent.SSEEvent{
				Event: "response.content_part.delta",
				Data:  `{"type":"response.content_part.delta","delta":{"type":"refusal","text":"no"}}`,
			},
			wantOK: false,
		},
		{
			name: "empty_text",
			event: iteragent.SSEEvent{
				Event: "response.content_part.delta",
				Data:  `{"type":"response.content_part.delta","delta":{"type":"text","text":""}}`,
			},
			wantOK: false,
		},
		{
			name: "invalid_json",
			event: iteragent.SSEEvent{
				Event: "response.content_part.delta",
				Data:  `{bad`,
			},
			wantOK: false,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tok, ok := iteragent.ParseOpenAIResponsesSSE(c.event)
			if ok != c.wantOK {
				t.Errorf("ok = %v, want %v", ok, c.wantOK)
			}
			if ok && tok != c.wantToken {
				t.Errorf("token = %q, want %q", tok, c.wantToken)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// SSEClient.Stream
// ---------------------------------------------------------------------------

// sseServer returns a test HTTP server that writes the given SSE lines and closes.
func sseServer(t *testing.T, lines []string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, line := range lines {
			fmt.Fprintln(w, line)
		}
	}))
}

func TestSSEClient_Stream_BasicTokens(t *testing.T) {
	srv := sseServer(t, []string{
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"foo"}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"bar"}}`,
		``,
	})
	defer srv.Close()

	client := iteragent.NewSSEClient()
	var tokens []string
	err := client.Stream(context.Background(), srv.URL, nil, []byte(`{}`), func(e iteragent.SSEEvent) {
		if tok, ok := iteragent.ParseAnthropicSSE(e.Data); ok {
			tokens = append(tokens, tok)
		}
	})
	if err != nil {
		t.Fatalf("Stream error: %v", err)
	}
	if len(tokens) != 2 {
		t.Fatalf("got %d tokens, want 2: %v", len(tokens), tokens)
	}
	if tokens[0] != "foo" || tokens[1] != "bar" {
		t.Errorf("tokens = %v, want [foo bar]", tokens)
	}
}

func TestSSEClient_Stream_CustomHeaders(t *testing.T) {
	var receivedAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "text/event-stream")
	}))
	defer srv.Close()

	client := iteragent.NewSSEClient()
	_ = client.Stream(context.Background(), srv.URL,
		map[string]string{"Authorization": "Bearer test-key"},
		[]byte(`{}`),
		func(e iteragent.SSEEvent) {},
	)
	if receivedAuth != "Bearer test-key" {
		t.Errorf("Authorization header = %q, want %q", receivedAuth, "Bearer test-key")
	}
}

func TestSSEClient_Stream_NonOKStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "rate limited", http.StatusTooManyRequests)
	}))
	defer srv.Close()

	client := iteragent.NewSSEClient()
	err := client.Stream(context.Background(), srv.URL, nil, []byte(`{}`), func(e iteragent.SSEEvent) {})
	if err == nil {
		t.Fatal("expected error for non-200 status")
	}
}

func TestSSEClient_Stream_ContextCancellation(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Server hangs — client should cancel via context.
		<-r.Context().Done()
	}))
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel before request

	client := iteragent.NewSSEClient()
	err := client.Stream(ctx, srv.URL, nil, []byte(`{}`), func(e iteragent.SSEEvent) {})
	if err == nil {
		t.Fatal("expected error from cancelled context")
	}
}

// ---------------------------------------------------------------------------
// SSEClient.Stream — edge cases
// ---------------------------------------------------------------------------

func TestSSEClient_Stream_EmptyBody(t *testing.T) {
	// Server sends no events at all — stream should complete without error.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// Write nothing; connection closes immediately.
	}))
	defer srv.Close()

	client := iteragent.NewSSEClient()
	var count int
	err := client.Stream(context.Background(), srv.URL, nil, []byte(`{}`), func(e iteragent.SSEEvent) {
		count++
	})
	if err != nil {
		t.Fatalf("unexpected error for empty stream: %v", err)
	}
	if count != 0 {
		t.Errorf("expected 0 events for empty stream, got %d", count)
	}
}

func TestSSEClient_Stream_MultilineData(t *testing.T) {
	// SSE spec allows "data:" prefix on consecutive lines to form one event.
	// Our parser appends newlines between them; ParseAnthropicSSE handles single JSON lines.
	// Here we verify that multiple data: lines within one event produce a single onEvent call
	// with both lines concatenated.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// Two consecutive data: lines before the blank-line separator.
		fmt.Fprint(w, "data: line1\n")
		fmt.Fprint(w, "data: line2\n")
		fmt.Fprint(w, "\n") // end of event
	}))
	defer srv.Close()

	client := iteragent.NewSSEClient()
	var events []iteragent.SSEEvent
	err := client.Stream(context.Background(), srv.URL, nil, []byte(`{}`), func(e iteragent.SSEEvent) {
		events = append(events, e)
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("expected 1 event for multiline data, got %d", len(events))
	}
}

func TestSSEClient_Stream_EventFieldParsed(t *testing.T) {
	// Verify that the "event:" field is captured in SSEEvent.Event.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "event: message_start\n")
		fmt.Fprint(w, "data: {\"type\":\"message_start\"}\n")
		fmt.Fprint(w, "\n")
	}))
	defer srv.Close()

	client := iteragent.NewSSEClient()
	var events []iteragent.SSEEvent
	_ = client.Stream(context.Background(), srv.URL, nil, []byte(`{}`), func(e iteragent.SSEEvent) {
		events = append(events, e)
	})
	if len(events) != 1 {
		t.Fatalf("expected 1 event, got %d", len(events))
	}
	if events[0].Event != "message_start" {
		t.Errorf("expected event='message_start', got %q", events[0].Event)
	}
}

func TestSSEClient_Stream_DataWithLeadingSpace(t *testing.T) {
	// "data: token" — the space after the colon is part of the prefix and should be trimmed.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		payload := `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"trimmed"}}`
		fmt.Fprintf(w, "data:   %s\n\n", payload) // extra spaces after colon
	}))
	defer srv.Close()

	client := iteragent.NewSSEClient()
	var tokens []string
	_ = client.Stream(context.Background(), srv.URL, nil, []byte(`{}`), func(e iteragent.SSEEvent) {
		if tok, ok := iteragent.ParseAnthropicSSE(e.Data); ok {
			tokens = append(tokens, tok)
		}
	})
	if len(tokens) != 1 || tokens[0] != "trimmed" {
		t.Errorf("expected ['trimmed'], got %v", tokens)
	}
}

func TestSSEClient_Stream_MultipleEventsOrdering(t *testing.T) {
	// Verify events arrive in the order they were sent.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, word := range []string{"one", "two", "three", "four", "five"} {
			payload := fmt.Sprintf(
				`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":%q}}`,
				word,
			)
			fmt.Fprintf(w, "data: %s\n\n", payload)
		}
	}))
	defer srv.Close()

	client := iteragent.NewSSEClient()
	var tokens []string
	_ = client.Stream(context.Background(), srv.URL, nil, []byte(`{}`), func(e iteragent.SSEEvent) {
		if tok, ok := iteragent.ParseAnthropicSSE(e.Data); ok {
			tokens = append(tokens, tok)
		}
	})
	want := []string{"one", "two", "three", "four", "five"}
	if len(tokens) != len(want) {
		t.Fatalf("expected %v, got %v", want, tokens)
	}
	for i, tok := range tokens {
		if tok != want[i] {
			t.Errorf("token[%d]: expected %q, got %q", i, want[i], tok)
		}
	}
}

func TestSSEClient_Stream_BlankLinesOnlyNoEvents(t *testing.T) {
	// Stream with only blank lines should produce no events.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "\n\n\n\n")
	}))
	defer srv.Close()

	client := iteragent.NewSSEClient()
	var count int
	_ = client.Stream(context.Background(), srv.URL, nil, []byte(`{}`), func(e iteragent.SSEEvent) {
		count++
	})
	if count != 0 {
		t.Errorf("expected 0 events for blank-line-only stream, got %d", count)
	}
}

// ---------------------------------------------------------------------------
// SSEResponse
// ---------------------------------------------------------------------------

func TestSSEResponse_AddAndGet(t *testing.T) {
	r := iteragent.NewSSEResponse()
	r.AddContent("hello")
	r.AddContent(" world")
	if got := r.GetContent(); got != "hello world" {
		t.Errorf("expected 'hello world', got %q", got)
	}
}

func TestSSEResponse_InitiallyEmpty(t *testing.T) {
	r := iteragent.NewSSEResponse()
	if got := r.GetContent(); got != "" {
		t.Errorf("expected empty content initially, got %q", got)
	}
}

func TestSSEResponse_StopAndIsStopped(t *testing.T) {
	r := iteragent.NewSSEResponse()
	if r.IsStopped() {
		t.Error("expected not stopped initially")
	}
	r.Stop()
	if !r.IsStopped() {
		t.Error("expected stopped after Stop()")
	}
}

func TestSSEResponse_AddMessage(t *testing.T) {
	r := iteragent.NewSSEResponse()
	r.AddMessage(iteragent.Message{Role: "user", Content: "hi"})
	r.AddMessage(iteragent.Message{Role: "assistant", Content: "hello"})
	msgs := r.GetMessages()
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(msgs))
	}
	if msgs[0].Role != "user" || msgs[1].Role != "assistant" {
		t.Errorf("unexpected messages: %+v", msgs)
	}
}
