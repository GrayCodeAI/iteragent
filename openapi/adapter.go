package openapi

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"
)

// OperationFilterKind selects which operations become tools.
type OperationFilterKind int

const (
	// OperationFilterAll includes every operation (default).
	OperationFilterAll OperationFilterKind = iota
	// OperationFilterByOperationID includes only operations whose operationId is in the list.
	OperationFilterByOperationID
	// OperationFilterByTag includes only operations that have at least one of the given tags.
	OperationFilterByTag
	// OperationFilterByPathPrefix includes only operations whose path starts with the prefix.
	OperationFilterByPathPrefix
)

// OperationFilter controls which operations are exposed as agent tools.
type OperationFilter struct {
	Kind   OperationFilterKind
	Values []string
}

// AllOperations returns a filter that includes every operation.
func AllOperations() OperationFilter {
	return OperationFilter{Kind: OperationFilterAll}
}

// ByOperationID returns a filter that includes only the named operation IDs.
func ByOperationID(ids ...string) OperationFilter {
	return OperationFilter{Kind: OperationFilterByOperationID, Values: ids}
}

// ByTag returns a filter that includes only operations with at least one of the given tags.
func ByTag(tags ...string) OperationFilter {
	return OperationFilter{Kind: OperationFilterByTag, Values: tags}
}

// ByPathPrefix returns a filter that includes only operations under the given path prefix.
func ByPathPrefix(prefix string) OperationFilter {
	return OperationFilter{Kind: OperationFilterByPathPrefix, Values: []string{prefix}}
}

// Spec is an OpenAPI 3.0 specification.
type Spec struct {
	OpenAPI string              `json:"openapi"`
	Info    Info                `json:"info"`
	Servers []Server            `json:"servers"`
	Paths   map[string]PathItem `json:"paths"`
}

type Info struct {
	Title   string `json:"title"`
	Version string `json:"version"`
}

type Server struct {
	URL string `json:"url"`
}

type PathItem struct {
	Get    *Operation `json:"get"`
	Post   *Operation `json:"post"`
	Put    *Operation `json:"put"`
	Delete *Operation `json:"delete"`
	Patch  *Operation `json:"patch"`
}

type Operation struct {
	OperationID string              `json:"operationId"`
	Summary     string              `json:"summary"`
	Description string              `json:"description"`
	Tags        []string            `json:"tags"`
	Parameters  []Parameter         `json:"parameters"`
	RequestBody *RequestBody        `json:"requestBody"`
	Responses   map[string]Response `json:"responses"`
}

type Parameter struct {
	Name        string  `json:"name"`
	In          string  `json:"in"`
	Description string  `json:"description"`
	Required    bool    `json:"required"`
	Schema      *Schema `json:"schema"`
}

type RequestBody struct {
	Required bool             `json:"required"`
	Content  map[string]Media `json:"content"`
}

type Media struct {
	Schema *Schema `json:"schema"`
}

// Schema is a JSON Schema fragment.
type Schema struct {
	Type        string             `json:"type,omitempty"`
	Format      string             `json:"format,omitempty"`
	Description string             `json:"description,omitempty"`
	Properties  map[string]*Schema `json:"properties,omitempty"`
	Required    []string           `json:"required,omitempty"`
	Items       *Schema            `json:"items,omitempty"`
	Enum        []interface{}      `json:"enum,omitempty"`
}

type Response struct {
	Description string `json:"description"`
}

// Config holds per-adapter HTTP configuration.
type Config struct {
	BaseURL string
	APIKey  string
	Headers map[string]string
	Filter  OperationFilter
}

// Adapter exposes OpenAPI operations as agent tools.
type Adapter struct {
	spec   *Spec
	config Config
	client *http.Client
}

// ParseSpec parses a raw JSON OpenAPI spec.
func ParseSpec(data []byte) (*Spec, error) {
	var spec Spec
	if err := json.Unmarshal(data, &spec); err != nil {
		return nil, fmt.Errorf("parse OpenAPI spec: %w", err)
	}
	return &spec, nil
}

// LoadSpec is an alias for ParseSpec.
func LoadSpec(data []byte) (*Spec, error) {
	return ParseSpec(data)
}

// FromFile reads a JSON OpenAPI spec from a file path.
func FromFile(path string, cfg Config) (*Adapter, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read spec file %s: %w", path, err)
	}
	spec, err := ParseSpec(data)
	if err != nil {
		return nil, err
	}
	return NewAdapter(spec, cfg), nil
}

// FromURL fetches a JSON OpenAPI spec from a URL.
func FromURL(ctx context.Context, rawURL string, cfg Config) (*Adapter, error) {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return nil, fmt.Errorf("invalid url: %w", err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return nil, fmt.Errorf("url must be http or https, got %s", parsed.Scheme)
	}

	client := &http.Client{Timeout: 30 * time.Second}
	req, err := http.NewRequestWithContext(ctx, "GET", rawURL, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetch spec: %w", err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read spec body: %w", err)
	}
	spec, err := ParseSpec(data)
	if err != nil {
		return nil, err
	}
	return NewAdapter(spec, cfg), nil
}

// NewAdapter creates an Adapter from a parsed spec and config.
func NewAdapter(spec *Spec, config Config) *Adapter {
	if config.BaseURL == "" && len(spec.Servers) > 0 {
		config.BaseURL = spec.Servers[0].URL
	}
	return &Adapter{
		spec:   spec,
		config: config,
		client: &http.Client{Timeout: 30 * time.Second},
	}
}

// Tool is a callable OpenAPI operation.
type Tool struct {
	Name        string
	Description string
	Schema      json.RawMessage
	Execute     func(ctx context.Context, args map[string]string) (string, error)
}

// GetTools returns all operations that pass the filter as callable Tools.
func (a *Adapter) GetTools() ([]Tool, error) {
	var tools []Tool

	methods := []struct {
		name string
		op   func(*PathItem) *Operation
	}{
		{"get", func(p *PathItem) *Operation { return p.Get }},
		{"post", func(p *PathItem) *Operation { return p.Post }},
		{"put", func(p *PathItem) *Operation { return p.Put }},
		{"delete", func(p *PathItem) *Operation { return p.Delete }},
		{"patch", func(p *PathItem) *Operation { return p.Patch }},
	}

	for path, item := range a.spec.Paths {
		for _, m := range methods {
			op := m.op(&item)
			if op == nil {
				continue
			}
			if !a.config.Filter.matches(path, op) {
				continue
			}

			operation := op
			pathStr := path
			methodStr := m.name

			toolName := operation.OperationID
			if toolName == "" {
				toolName = fmt.Sprintf("%s_%s", methodStr, cleanPath(pathStr))
			}

			desc := operation.Description
			if desc == "" {
				desc = operation.Summary
			}
			if desc == "" {
				desc = fmt.Sprintf("%s %s", strings.ToUpper(methodStr), pathStr)
			}

			schema, _ := buildSchema(operation)

			tool := Tool{
				Name:        toolName,
				Description: desc,
				Schema:      schema,
				Execute: func(ctx context.Context, args map[string]string) (string, error) {
					return a.callOperation(ctx, pathStr, methodStr, operation, args)
				},
			}
			tools = append(tools, tool)
		}
	}
	return tools, nil
}

// matches checks whether an operation passes through this filter.
func (f *OperationFilter) matches(path string, op *Operation) bool {
	switch f.Kind {
	case OperationFilterByOperationID:
		for _, id := range f.Values {
			if op.OperationID == id {
				return true
			}
		}
		return false
	case OperationFilterByTag:
		for _, tag := range op.Tags {
			for _, want := range f.Values {
				if tag == want {
					return true
				}
			}
		}
		return false
	case OperationFilterByPathPrefix:
		for _, prefix := range f.Values {
			if strings.HasPrefix(path, prefix) {
				return true
			}
		}
		return false
	default: // OperationFilterAll
		return true
	}
}

// buildSchema creates a JSON Schema for an operation's parameters.
func buildSchema(op *Operation) (json.RawMessage, error) {
	properties := map[string]*Schema{}
	required := []string{}

	for _, param := range op.Parameters {
		s := &Schema{Description: param.Description}
		if param.Schema != nil {
			s.Type = param.Schema.Type
			s.Format = param.Schema.Format
			s.Enum = param.Schema.Enum
		} else {
			s.Type = "string"
		}
		properties[param.Name] = s
		if param.Required {
			required = append(required, param.Name)
		}
	}

	// Merge request body properties if present.
	if op.RequestBody != nil {
		if media, ok := op.RequestBody.Content["application/json"]; ok && media.Schema != nil {
			for name, prop := range media.Schema.Properties {
				properties[name] = prop
			}
			for _, r := range media.Schema.Required {
				required = append(required, r)
			}
		}
	}

	topSchema := map[string]interface{}{
		"type":       "object",
		"properties": properties,
	}
	if len(required) > 0 {
		topSchema["required"] = required
	}

	data, err := json.Marshal(topSchema)
	if err != nil {
		return nil, err
	}
	return data, nil
}

func cleanPath(path string) string {
	path = strings.TrimPrefix(path, "/")
	path = strings.ReplaceAll(path, "/", "_")
	path = strings.ReplaceAll(path, "{", "")
	path = strings.ReplaceAll(path, "}", "")
	return path
}

func (a *Adapter) callOperation(ctx context.Context, path, method string, op *Operation, args map[string]string) (string, error) {
	u := a.config.BaseURL + path

	var queryParams []string

	for _, param := range op.Parameters {
		val, ok := args[param.Name]
		if !ok {
			continue
		}
		switch param.In {
		case "query":
			queryParams = append(queryParams, fmt.Sprintf("%s=%s", url.QueryEscape(param.Name), url.QueryEscape(val)))
		case "path":
			u = strings.ReplaceAll(u, "{"+param.Name+"}", val)
		}
	}

	if len(queryParams) > 0 {
		u += "?" + strings.Join(queryParams, "&")
	}

	var bodyReader io.Reader
	if op.RequestBody != nil {
		bodyData, _ := json.Marshal(args)
		bodyReader = strings.NewReader(string(bodyData))
	} else {
		bodyReader = strings.NewReader("")
	}

	req, err := http.NewRequestWithContext(ctx, strings.ToUpper(method), u, bodyReader)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	if a.config.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+a.config.APIKey)
	}
	for k, v := range a.config.Headers {
		req.Header.Set(k, v)
	}

	resp, err := a.client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	if resp.StatusCode >= 400 {
		return "", fmt.Errorf("API error %d: %s", resp.StatusCode, string(respBody))
	}
	return string(respBody), nil
}
