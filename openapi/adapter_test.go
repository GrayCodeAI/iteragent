package openapi

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func minimalSpec(t *testing.T) []byte {
	t.Helper()
	return []byte(`{
		"openapi": "3.0.0",
		"info": {"title": "Test API", "version": "1.0.0"},
		"servers": [{"url": "https://api.example.com"}],
		"paths": {
			"/pets": {
				"get": {
					"operationId": "listPets",
					"summary": "List all pets",
					"tags": ["pets"],
					"parameters": [
						{"name": "limit", "in": "query", "required": false, "schema": {"type": "integer"}}
					]
				},
				"post": {
					"operationId": "createPet",
					"description": "Create a new pet",
					"tags": ["pets"],
					"requestBody": {
						"required": true,
						"content": {
							"application/json": {
								"schema": {
									"type": "object",
									"properties": {
										"name": {"type": "string"},
										"age":  {"type": "integer"}
									},
									"required": ["name"]
								}
							}
						}
					}
				}
			},
			"/pets/{id}": {
				"get": {
					"operationId": "getPet",
					"summary": "Get a pet by ID",
					"tags": ["pets"],
					"parameters": [
						{"name": "id", "in": "path", "required": true, "schema": {"type": "string"}}
					]
				},
				"delete": {
					"operationId": "deletePet",
					"summary": "Delete a pet",
					"tags": ["admin"],
					"parameters": [
						{"name": "id", "in": "path", "required": true, "schema": {"type": "string"}}
					]
				}
			},
			"/health": {
				"get": {
					"operationId": "healthCheck",
					"summary": "Health check",
					"tags": ["system"]
				}
			}
		}
	}`)
}

// ---------------------------------------------------------------------------
// ParseSpec
// ---------------------------------------------------------------------------

func TestParseSpec_Valid(t *testing.T) {
	spec, err := ParseSpec(minimalSpec(t))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spec.Info.Title != "Test API" {
		t.Errorf("expected title 'Test API', got %q", spec.Info.Title)
	}
	if len(spec.Servers) != 1 || spec.Servers[0].URL != "https://api.example.com" {
		t.Errorf("unexpected servers: %+v", spec.Servers)
	}
	if len(spec.Paths) != 3 {
		t.Errorf("expected 3 paths, got %d", len(spec.Paths))
	}
}

func TestParseSpec_InvalidJSON(t *testing.T) {
	_, err := ParseSpec([]byte(`not json`))
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestParseSpec_Empty(t *testing.T) {
	spec, err := ParseSpec([]byte(`{}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spec.Info.Title != "" {
		t.Errorf("expected empty title, got %q", spec.Info.Title)
	}
}

// ---------------------------------------------------------------------------
// LoadSpec (alias)
// ---------------------------------------------------------------------------

func TestLoadSpec_Alias(t *testing.T) {
	spec, err := LoadSpec(minimalSpec(t))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spec.Info.Version != "1.0.0" {
		t.Errorf("expected version 1.0.0, got %q", spec.Info.Version)
	}
}

// ---------------------------------------------------------------------------
// FromFile
// ---------------------------------------------------------------------------

func TestFromFile_Valid(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "spec.json")
	if err := os.WriteFile(path, minimalSpec(t), 0o600); err != nil {
		t.Fatal(err)
	}
	adapter, err := FromFile(path, Config{Filter: AllOperations()})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if adapter == nil {
		t.Fatal("expected non-nil adapter")
	}
	if adapter.config.BaseURL != "https://api.example.com" {
		t.Errorf("expected base URL from spec, got %q", adapter.config.BaseURL)
	}
}

func TestFromFile_Missing(t *testing.T) {
	_, err := FromFile("/nonexistent/path/spec.json", Config{})
	if err == nil {
		t.Error("expected error for missing file")
	}
}

func TestFromFile_InvalidJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(path, []byte(`{bad json}`), 0o600); err != nil {
		t.Fatal(err)
	}
	_, err := FromFile(path, Config{})
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

// ---------------------------------------------------------------------------
// NewAdapter
// ---------------------------------------------------------------------------

func TestNewAdapter_BaseURLFromSpec(t *testing.T) {
	spec, _ := ParseSpec(minimalSpec(t))
	a := NewAdapter(spec, Config{Filter: AllOperations()})
	if a.config.BaseURL != "https://api.example.com" {
		t.Errorf("expected base URL from spec servers, got %q", a.config.BaseURL)
	}
}

func TestNewAdapter_BaseURLOverride(t *testing.T) {
	spec, _ := ParseSpec(minimalSpec(t))
	a := NewAdapter(spec, Config{BaseURL: "https://override.example.com", Filter: AllOperations()})
	if a.config.BaseURL != "https://override.example.com" {
		t.Errorf("expected override base URL, got %q", a.config.BaseURL)
	}
}

func TestNewAdapter_NoServers(t *testing.T) {
	spec := &Spec{Paths: map[string]PathItem{}}
	a := NewAdapter(spec, Config{BaseURL: "https://manual.example.com"})
	if a.config.BaseURL != "https://manual.example.com" {
		t.Errorf("expected manual base URL, got %q", a.config.BaseURL)
	}
}

// ---------------------------------------------------------------------------
// OperationFilter.matches
// ---------------------------------------------------------------------------

func TestOperationFilter_All(t *testing.T) {
	f := AllOperations()
	op := &Operation{OperationID: "anything", Tags: []string{"foo"}}
	if !f.matches("/any/path", op) {
		t.Error("AllOperations should match everything")
	}
}

func TestOperationFilter_ByOperationID_Match(t *testing.T) {
	f := ByOperationID("listPets", "createPet")
	if !f.matches("/pets", &Operation{OperationID: "listPets"}) {
		t.Error("expected match for listPets")
	}
	if !f.matches("/pets", &Operation{OperationID: "createPet"}) {
		t.Error("expected match for createPet")
	}
}

func TestOperationFilter_ByOperationID_NoMatch(t *testing.T) {
	f := ByOperationID("listPets")
	if f.matches("/pets", &Operation{OperationID: "deletePet"}) {
		t.Error("expected no match for deletePet")
	}
}

func TestOperationFilter_ByTag_Match(t *testing.T) {
	f := ByTag("pets")
	op := &Operation{Tags: []string{"pets", "animals"}}
	if !f.matches("/pets", op) {
		t.Error("expected match for pets tag")
	}
}

func TestOperationFilter_ByTag_NoMatch(t *testing.T) {
	f := ByTag("admin")
	op := &Operation{Tags: []string{"pets"}}
	if f.matches("/pets", op) {
		t.Error("expected no match")
	}
}

func TestOperationFilter_ByPathPrefix_Match(t *testing.T) {
	f := ByPathPrefix("/pets")
	if !f.matches("/pets", &Operation{}) {
		t.Error("expected match for /pets")
	}
	if !f.matches("/pets/123", &Operation{}) {
		t.Error("expected match for /pets/123")
	}
}

func TestOperationFilter_ByPathPrefix_NoMatch(t *testing.T) {
	f := ByPathPrefix("/pets")
	if f.matches("/health", &Operation{}) {
		t.Error("expected no match for /health")
	}
}

// ---------------------------------------------------------------------------
// GetTools
// ---------------------------------------------------------------------------

func TestGetTools_AllOperations(t *testing.T) {
	spec, _ := ParseSpec(minimalSpec(t))
	a := NewAdapter(spec, Config{Filter: AllOperations()})
	tools, err := a.GetTools()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// spec has: GET /pets, POST /pets, GET /pets/{id}, DELETE /pets/{id}, GET /health = 5 ops
	if len(tools) != 5 {
		t.Errorf("expected 5 tools, got %d", len(tools))
	}
}

func TestGetTools_FilteredByTag(t *testing.T) {
	spec, _ := ParseSpec(minimalSpec(t))
	a := NewAdapter(spec, Config{Filter: ByTag("admin")})
	tools, err := a.GetTools()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(tools) != 1 {
		t.Errorf("expected 1 tool, got %d", len(tools))
	}
	if tools[0].Name != "deletePet" {
		t.Errorf("expected deletePet, got %q", tools[0].Name)
	}
}

func TestGetTools_FilteredByOperationID(t *testing.T) {
	spec, _ := ParseSpec(minimalSpec(t))
	a := NewAdapter(spec, Config{Filter: ByOperationID("healthCheck", "getPet")})
	tools, err := a.GetTools()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(tools) != 2 {
		t.Errorf("expected 2 tools, got %d", len(tools))
	}
	names := map[string]bool{}
	for _, t2 := range tools {
		names[t2.Name] = true
	}
	if !names["healthCheck"] || !names["getPet"] {
		t.Errorf("unexpected tool names: %v", names)
	}
}

func TestGetTools_FilteredByPathPrefix(t *testing.T) {
	spec, _ := ParseSpec(minimalSpec(t))
	a := NewAdapter(spec, Config{Filter: ByPathPrefix("/health")})
	tools, err := a.GetTools()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(tools) != 1 {
		t.Errorf("expected 1 tool, got %d", len(tools))
	}
	if tools[0].Name != "healthCheck" {
		t.Errorf("expected healthCheck, got %q", tools[0].Name)
	}
}

func TestGetTools_ToolNameFromOperationID(t *testing.T) {
	spec, _ := ParseSpec(minimalSpec(t))
	a := NewAdapter(spec, Config{Filter: ByOperationID("listPets")})
	tools, err := a.GetTools()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(tools) != 1 || tools[0].Name != "listPets" {
		t.Errorf("expected tool name 'listPets', got %v", tools)
	}
}

func TestGetTools_FallbackName(t *testing.T) {
	// Operation without operationId — name should be "<method>_<cleanPath>"
	spec := &Spec{
		Paths: map[string]PathItem{
			"/foo/bar": {
				Get: &Operation{Summary: "Foo bar"},
			},
		},
	}
	a := NewAdapter(spec, Config{Filter: AllOperations()})
	tools, err := a.GetTools()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}
	if tools[0].Name != "get_foo_bar" {
		t.Errorf("expected fallback name 'get_foo_bar', got %q", tools[0].Name)
	}
}

func TestGetTools_DescriptionFallback(t *testing.T) {
	spec := &Spec{
		Paths: map[string]PathItem{
			"/x": {
				Get: &Operation{OperationID: "getX"},
			},
		},
	}
	a := NewAdapter(spec, Config{Filter: AllOperations()})
	tools, err := a.GetTools()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if tools[0].Description != "GET /x" {
		t.Errorf("expected fallback description 'GET /x', got %q", tools[0].Description)
	}
}

// ---------------------------------------------------------------------------
// cleanPath
// ---------------------------------------------------------------------------

func TestCleanPath_Basic(t *testing.T) {
	cases := []struct{ in, want string }{
		{"/pets", "pets"},
		{"/pets/{id}", "pets_id"},
		{"/a/b/c", "a_b_c"},
		{"pets", "pets"},
		{"/users/{userId}/posts/{postId}", "users_userId_posts_postId"},
	}
	for _, c := range cases {
		got := cleanPath(c.in)
		if got != c.want {
			t.Errorf("cleanPath(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// ---------------------------------------------------------------------------
// buildSchema
// ---------------------------------------------------------------------------

func TestBuildSchema_Parameters(t *testing.T) {
	op := &Operation{
		Parameters: []Parameter{
			{Name: "id", In: "path", Required: true, Schema: &Schema{Type: "string"}},
			{Name: "limit", In: "query", Required: false, Schema: &Schema{Type: "integer"}},
		},
	}
	raw, err := buildSchema(op)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var s map[string]interface{}
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatalf("invalid JSON schema: %v", err)
	}
	props := s["properties"].(map[string]interface{})
	if _, ok := props["id"]; !ok {
		t.Error("expected 'id' property")
	}
	if _, ok := props["limit"]; !ok {
		t.Error("expected 'limit' property")
	}
	required := s["required"].([]interface{})
	if len(required) != 1 || required[0] != "id" {
		t.Errorf("expected required=[id], got %v", required)
	}
}

func TestBuildSchema_RequestBodyMerge(t *testing.T) {
	op := &Operation{
		Parameters: []Parameter{
			{Name: "id", In: "path", Required: true, Schema: &Schema{Type: "string"}},
		},
		RequestBody: &RequestBody{
			Required: true,
			Content: map[string]Media{
				"application/json": {
					Schema: &Schema{
						Properties: map[string]*Schema{
							"name": {Type: "string"},
							"age":  {Type: "integer"},
						},
						Required: []string{"name"},
					},
				},
			},
		},
	}
	raw, err := buildSchema(op)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var s map[string]interface{}
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatalf("invalid JSON schema: %v", err)
	}
	props := s["properties"].(map[string]interface{})
	if _, ok := props["name"]; !ok {
		t.Error("expected 'name' from requestBody")
	}
	if _, ok := props["age"]; !ok {
		t.Error("expected 'age' from requestBody")
	}
	if _, ok := props["id"]; !ok {
		t.Error("expected 'id' from parameter")
	}
}

func TestBuildSchema_NoParameters(t *testing.T) {
	op := &Operation{}
	raw, err := buildSchema(op)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var s map[string]interface{}
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatalf("invalid JSON schema: %v", err)
	}
	if s["type"] != "object" {
		t.Errorf("expected type=object, got %v", s["type"])
	}
	if _, hasRequired := s["required"]; hasRequired {
		t.Error("expected no 'required' field for empty parameters")
	}
}

// ---------------------------------------------------------------------------
// callOperation via httptest server
// ---------------------------------------------------------------------------

func TestCallOperation_QueryParams(t *testing.T) {
	var capturedURL string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedURL = r.URL.String()
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"ok":true}`))
	}))
	defer srv.Close()

	spec := &Spec{
		Paths: map[string]PathItem{
			"/pets": {
				Get: &Operation{
					OperationID: "listPets",
					Parameters: []Parameter{
						{Name: "limit", In: "query", Schema: &Schema{Type: "integer"}},
					},
				},
			},
		},
	}
	a := NewAdapter(spec, Config{BaseURL: srv.URL, Filter: AllOperations()})
	tools, _ := a.GetTools()
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}

	result, err := tools[0].Execute(context.Background(), map[string]string{"limit": "10"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != `{"ok":true}` {
		t.Errorf("unexpected result: %q", result)
	}
	if !strings.Contains(capturedURL, "limit=10") {
		t.Errorf("expected limit=10 in URL, got %q", capturedURL)
	}
}

func TestCallOperation_PathParams(t *testing.T) {
	var capturedPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"id":"42"}`))
	}))
	defer srv.Close()

	spec := &Spec{
		Paths: map[string]PathItem{
			"/pets/{id}": {
				Get: &Operation{
					OperationID: "getPet",
					Parameters: []Parameter{
						{Name: "id", In: "path", Required: true, Schema: &Schema{Type: "string"}},
					},
				},
			},
		},
	}
	a := NewAdapter(spec, Config{BaseURL: srv.URL, Filter: AllOperations()})
	tools, _ := a.GetTools()

	_, err := tools[0].Execute(context.Background(), map[string]string{"id": "42"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if capturedPath != "/pets/42" {
		t.Errorf("expected path /pets/42, got %q", capturedPath)
	}
}

func TestCallOperation_RequestBody(t *testing.T) {
	var capturedBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedBody, _ = readAll(r.Body)
		w.WriteHeader(http.StatusCreated)
		w.Write([]byte(`{"created":true}`))
	}))
	defer srv.Close()

	spec := &Spec{
		Paths: map[string]PathItem{
			"/pets": {
				Post: &Operation{
					OperationID: "createPet",
					RequestBody: &RequestBody{
						Required: true,
						Content: map[string]Media{
							"application/json": {
								Schema: &Schema{
									Properties: map[string]*Schema{
										"name": {Type: "string"},
									},
								},
							},
						},
					},
				},
			},
		},
	}
	a := NewAdapter(spec, Config{BaseURL: srv.URL, Filter: AllOperations()})
	tools, _ := a.GetTools()

	_, err := tools[0].Execute(context.Background(), map[string]string{"name": "Fluffy"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var body map[string]string
	if err := json.Unmarshal(capturedBody, &body); err != nil {
		t.Fatalf("invalid body JSON: %v", err)
	}
	if body["name"] != "Fluffy" {
		t.Errorf("expected name=Fluffy in body, got %v", body)
	}
}

func TestCallOperation_AuthHeader(t *testing.T) {
	var capturedAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedAuth = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	spec := &Spec{
		Paths: map[string]PathItem{
			"/secure": {
				Get: &Operation{OperationID: "secureOp"},
			},
		},
	}
	a := NewAdapter(spec, Config{BaseURL: srv.URL, APIKey: "my-secret-key", Filter: AllOperations()})
	tools, _ := a.GetTools()

	_, err := tools[0].Execute(context.Background(), map[string]string{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if capturedAuth != "Bearer my-secret-key" {
		t.Errorf("expected Bearer auth header, got %q", capturedAuth)
	}
}

func TestCallOperation_CustomHeaders(t *testing.T) {
	var capturedHeader string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedHeader = r.Header.Get("X-Custom")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	spec := &Spec{
		Paths: map[string]PathItem{
			"/x": {
				Get: &Operation{OperationID: "xOp"},
			},
		},
	}
	a := NewAdapter(spec, Config{
		BaseURL: srv.URL,
		Headers: map[string]string{"X-Custom": "custom-value"},
		Filter:  AllOperations(),
	})
	tools, _ := a.GetTools()

	_, err := tools[0].Execute(context.Background(), map[string]string{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if capturedHeader != "custom-value" {
		t.Errorf("expected X-Custom: custom-value, got %q", capturedHeader)
	}
}

func TestCallOperation_4xxError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		w.Write([]byte(`not found`))
	}))
	defer srv.Close()

	spec := &Spec{
		Paths: map[string]PathItem{
			"/missing": {
				Get: &Operation{OperationID: "getMissing"},
			},
		},
	}
	a := NewAdapter(spec, Config{BaseURL: srv.URL, Filter: AllOperations()})
	tools, _ := a.GetTools()

	_, err := tools[0].Execute(context.Background(), map[string]string{})
	if err == nil {
		t.Fatal("expected error for 4xx response")
	}
	if !strings.Contains(err.Error(), "404") {
		t.Errorf("expected 404 in error, got %q", err.Error())
	}
}

func TestCallOperation_5xxError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`server error`))
	}))
	defer srv.Close()

	spec := &Spec{
		Paths: map[string]PathItem{
			"/broken": {
				Get: &Operation{OperationID: "getBroken"},
			},
		},
	}
	a := NewAdapter(spec, Config{BaseURL: srv.URL, Filter: AllOperations()})
	tools, _ := a.GetTools()

	_, err := tools[0].Execute(context.Background(), map[string]string{})
	if err == nil {
		t.Fatal("expected error for 5xx response")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("expected 500 in error, got %q", err.Error())
	}
}

// ---------------------------------------------------------------------------
// FromURL
// ---------------------------------------------------------------------------

func TestFromURL_Valid(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write(minimalSpec(t))
	}))
	defer srv.Close()

	a, err := FromURL(context.Background(), srv.URL+"/spec.json", Config{Filter: AllOperations()})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	tools, err := a.GetTools()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(tools) == 0 {
		t.Error("expected tools from fetched spec")
	}
}

func TestFromURL_InvalidScheme(t *testing.T) {
	_, err := FromURL(context.Background(), "ftp://example.com/spec.json", Config{})
	if err == nil {
		t.Error("expected error for ftp scheme")
	}
}

func TestFromURL_InvalidURL(t *testing.T) {
	_, err := FromURL(context.Background(), "://bad url", Config{})
	if err == nil {
		t.Error("expected error for invalid URL")
	}
}

// ---------------------------------------------------------------------------
// helpers (private, only used in tests)
// ---------------------------------------------------------------------------

func readAll(r interface{ Read([]byte) (int, error) }) ([]byte, error) {
	var buf []byte
	tmp := make([]byte, 512)
	for {
		n, err := r.Read(tmp)
		if n > 0 {
			buf = append(buf, tmp[:n]...)
		}
		if err != nil {
			break
		}
	}
	return buf, nil
}
