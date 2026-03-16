package openapi

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	iteragent "github.com/GrayCodeAI/iteragent"
)

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
}

type Operation struct {
	Summary     string              `json:"summary"`
	Description string              `json:"description"`
	Parameters  []Parameter         `json:"parameters"`
	RequestBody *RequestBody        `json:"requestBody"`
	Responses   map[string]Response `json:"responses"`
}

type Parameter struct {
	Name        string `json:"name"`
	In          string `json:"in"`
	Description string `json:"description"`
	Required    bool   `json:"required"`
	Schema      Schema `json:"schema"`
}

type RequestBody struct {
	Required bool             `json:"required"`
	Content  map[string]Media `json:"content"`
}

type Media struct {
	Schema Schema `json:"schema"`
}

type Schema struct {
	Type string `json:"type"`
}

type Response struct {
	Description string `json:"description"`
}

type Config struct {
	BaseURL string
	APIKey  string
	Headers map[string]string
}

type Adapter struct {
	spec   *Spec
	config Config
	client *http.Client
}

func LoadSpec(data []byte) (*Spec, error) {
	var spec Spec
	if err := json.Unmarshal(data, &spec); err != nil {
		return nil, fmt.Errorf("parse OpenAPI spec: %w", err)
	}
	return &spec, nil
}

func NewAdapter(spec *Spec, config Config) *Adapter {
	baseURL := config.BaseURL
	if baseURL == "" && len(spec.Servers) > 0 {
		baseURL = spec.Servers[0].URL
	}

	return &Adapter{
		spec:   spec,
		config: config,
		client: &http.Client{Timeout: 30},
	}
}

func (a *Adapter) GetTools() ([]iteragent.Tool, error) {
	var tools []iteragent.Tool

	for path, item := range a.spec.Paths {
		for method, op := range map[string]*Operation{"get": item.Get, "post": item.Post, "put": item.Put, "delete": item.Delete} {
			if op == nil {
				continue
			}

			operation := op
			pathStr := path
			methodStr := method

			tool := iteragent.Tool{
				Name:        fmt.Sprintf("%s_%s", methodStr, cleanPath(pathStr)),
				Description: operation.Description,
				Execute: func(ctx context.Context, args map[string]string) (string, error) {
					return a.callOperation(ctx, pathStr, methodStr, operation, args)
				},
			}
			tools = append(tools, tool)
		}
	}

	return tools, nil
}

func cleanPath(path string) string {
	path = strings.TrimPrefix(path, "/")
	path = strings.ReplaceAll(path, "/", "_")
	path = strings.ReplaceAll(path, "{", "")
	path = strings.ReplaceAll(path, "}", "")
	return path
}

func (a *Adapter) callOperation(ctx context.Context, path, method string, op *Operation, args map[string]string) (string, error) {
	url := a.config.BaseURL + path

	var queryParams []string
	var body string

	for _, param := range op.Parameters {
		if val, ok := args[param.Name]; ok {
			switch param.In {
			case "query":
				queryParams = append(queryParams, fmt.Sprintf("%s=%s", param.Name, val))
			case "path":
				url = strings.ReplaceAll(url, "{"+param.Name+"}", val)
			}
		}
	}

	if len(queryParams) > 0 {
		url += "?" + strings.Join(queryParams, "&")
	}

	if op.RequestBody != nil && len(args) > 0 {
		bodyBytes, _ := json.Marshal(args)
		body = string(bodyBytes)
	}

	req, err := http.NewRequestWithContext(ctx, strings.ToUpper(method), url, strings.NewReader(body))
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
