package iteragent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// geminiFunctionDeclaration maps to Gemini's FunctionDeclaration proto.
type geminiFunctionDeclaration struct {
	Name        string     `json:"name"`
	Description string     `json:"description,omitempty"`
	Parameters  ToolSchema `json:"parameters"`
}

type geminiToolDef struct {
	FunctionDeclarations []geminiFunctionDeclaration `json:"function_declarations"`
}

type geminiFunctionCall struct {
	Name string          `json:"name"`
	Args json.RawMessage `json:"args"`
}

type geminiFunctionResponse struct {
	Name     string          `json:"name"`
	Response json.RawMessage `json:"response"`
}

type geminiNativePart struct {
	Text             string                  `json:"text,omitempty"`
	FunctionCall     *geminiFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *geminiFunctionResponse `json:"functionResponse,omitempty"`
}

type geminiNativeContent struct {
	Role  string             `json:"role"`
	Parts []geminiNativePart `json:"parts"`
}

type geminiNativeRequest struct {
	Contents         []geminiNativeContent `json:"contents"`
	Tools            []geminiToolDef       `json:"tools,omitempty"`
	SystemInstruction *geminiNativeContent `json:"system_instruction,omitempty"`
	GenerationConfig  map[string]interface{} `json:"generationConfig,omitempty"`
}

type geminiNativeResponse struct {
	Candidates []struct {
		Content       geminiNativeContent `json:"content"`
		FinishReason  string              `json:"finishReason"`
	} `json:"candidates"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// SupportsNativeTools marks geminiProvider as supporting function declarations.
func (p *geminiProvider) SupportsNativeTools() bool { return true }

// CompleteWithNativeFunctions sends a completion with Gemini function declarations
// and runs up to maxRounds of function_call / function_response turns.
func (p *geminiProvider) CompleteWithNativeFunctions(
	ctx context.Context,
	messages []Message,
	tools []Tool,
	opts CompletionOptions,
	executeFn func(name string, args json.RawMessage) (string, error),
	maxRounds int,
) (string, error) {
	if maxRounds <= 0 {
		maxRounds = 10
	}

	geminiTools := buildGeminiToolDefs(tools)

	// Convert iteragent messages to Gemini native format.
	var systemInstruction *geminiNativeContent
	nativeContents := make([]geminiNativeContent, 0, len(messages))
	for _, m := range messages {
		if m.Role == "system" {
			systemInstruction = &geminiNativeContent{
				Parts: []geminiNativePart{{Text: m.Content}},
			}
			continue
		}
		role := m.Role
		if role == "assistant" {
			role = "model"
		}
		nativeContents = append(nativeContents, geminiNativeContent{
			Role:  role,
			Parts: []geminiNativePart{{Text: m.Content}},
		})
	}

	geminiNativeModel := p.cfg.Model
	if opts.Model != "" {
		geminiNativeModel = opts.Model
	}
	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent?key=%s",
		geminiNativeModel, p.cfg.APIKey)

	for round := 0; round < maxRounds; round++ {
		reqBody := geminiNativeRequest{
			Contents:          nativeContents,
			Tools:             geminiTools,
			SystemInstruction: systemInstruction,
		}
		if opts.MaxTokens > 0 {
			reqBody.GenerationConfig = map[string]interface{}{"maxOutputTokens": opts.MaxTokens}
		}

		body, err := json.Marshal(reqBody)
		if err != nil {
			return "", fmt.Errorf("marshal gemini native tools request: %w", err)
		}

		req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
		if err != nil {
			return "", fmt.Errorf("create request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := p.client.Do(req)
		if err != nil {
			return "", fmt.Errorf("http request: %w", err)
		}
		raw, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return "", fmt.Errorf("read response: %w", err)
		}

		var result geminiNativeResponse
		if err := json.Unmarshal(raw, &result); err != nil {
			return "", fmt.Errorf("unmarshal response: %w", err)
		}
		if result.Error != nil {
			return "", fmt.Errorf("gemini error: %s", result.Error.Message)
		}
		if len(result.Candidates) == 0 {
			return "", fmt.Errorf("no candidates in gemini response")
		}

		candidate := result.Candidates[0]

		// Collect text and function_call parts.
		var textOut string
		var functionCalls []geminiFunctionCall
		for _, part := range candidate.Content.Parts {
			if part.Text != "" {
				textOut += part.Text
			}
			if part.FunctionCall != nil {
				functionCalls = append(functionCalls, *part.FunctionCall)
			}
		}

		// No function calls — done.
		if len(functionCalls) == 0 {
			return textOut, nil
		}

		// Append model turn with function calls.
		nativeContents = append(nativeContents, geminiNativeContent{
			Role:  "model",
			Parts: candidate.Content.Parts,
		})

		// Execute each function call and build function_response parts.
		responseParts := make([]geminiNativePart, 0, len(functionCalls))
		for _, fc := range functionCalls {
			var resultContent string
			if executeFn != nil {
				res, execErr := executeFn(fc.Name, fc.Args)
				if execErr != nil {
					resultContent = fmt.Sprintf(`{"error": %q}`, execErr.Error())
				} else {
					resultContent = res
				}
			} else {
				resultContent = fmt.Sprintf(`{"error": "Tool %s is not available"}`, fc.Name)
			}

			responseJSON := json.RawMessage(fmt.Sprintf(`{"result": %s}`, jsonStringOrObject(resultContent)))
			responseParts = append(responseParts, geminiNativePart{
				FunctionResponse: &geminiFunctionResponse{
					Name:     fc.Name,
					Response: responseJSON,
				},
			})
		}

		nativeContents = append(nativeContents, geminiNativeContent{
			Role:  "user",
			Parts: responseParts,
		})
	}

	return "", fmt.Errorf("gemini native tools: exceeded max rounds (%d)", maxRounds)
}

func buildGeminiToolDefs(tools []Tool) []geminiToolDef {
	decls := make([]geminiFunctionDeclaration, len(tools))
	for i, t := range tools {
		schema, desc := parseToolDescription(t.Description)
		decls[i] = geminiFunctionDeclaration{
			Name:        t.Name,
			Description: desc,
			Parameters:  schema,
		}
	}
	return []geminiToolDef{{FunctionDeclarations: decls}}
}

// jsonStringOrObject wraps s in JSON quotes if it's not already valid JSON.
func jsonStringOrObject(s string) string {
	if len(s) > 0 && (s[0] == '{' || s[0] == '[' || s[0] == '"') {
		if json.Valid([]byte(s)) {
			return s
		}
	}
	b, _ := json.Marshal(s)
	return string(b)
}
