package iteragent

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ToolSchema is an OpenAPI-compatible JSON Schema fragment used by native
// function-calling APIs (Anthropic tool_use, OpenAI function_calling, Gemini
// function declarations).
type ToolSchema struct {
	Type        string                 `json:"type"`
	Description string                 `json:"description,omitempty"`
	Properties  map[string]*ToolSchema `json:"properties,omitempty"`
	Items       *ToolSchema            `json:"items,omitempty"`
	Required    []string               `json:"required,omitempty"`
	Enum        []interface{}          `json:"enum,omitempty"`
}

// NativeToolDefinition is the schema payload used by native tool-calling APIs.
type NativeToolDefinition struct {
	Name        string     `json:"name"`
	Description string     `json:"description"`
	InputSchema ToolSchema `json:"input_schema"`
}

// ToNativeDefinition converts a Tool to a NativeToolDefinition by parsing the
// Tool description for parameter annotations.
//
// Annotation format (anywhere in Description):
//
//	Params:
//	  name (type, required): description
//	  name (type): description
//
// Supported types: string, number, integer, boolean, array, object.
// If no Params block is found, a schema with no required parameters is generated.
func (t Tool) ToNativeDefinition() NativeToolDefinition {
	schema, description := parseToolDescription(t.Description)
	return NativeToolDefinition{
		Name:        t.Name,
		Description: description,
		InputSchema: schema,
	}
}

// ToolsToNativeDefinitions converts a slice of Tools to their native API schemas.
func ToolsToNativeDefinitions(tools []Tool) []NativeToolDefinition {
	defs := make([]NativeToolDefinition, len(tools))
	for i, t := range tools {
		defs[i] = t.ToNativeDefinition()
	}
	return defs
}

// NativeDefinitionsJSON serialises a slice of native definitions to JSON.
func NativeDefinitionsJSON(tools []Tool) ([]byte, error) {
	defs := ToolsToNativeDefinitions(tools)
	return json.Marshal(defs)
}

// parseToolDescription extracts the base description and a ToolSchema from a
// freeform Tool.Description string.
func parseToolDescription(raw string) (ToolSchema, string) {
	schema := ToolSchema{
		Type:       "object",
		Properties: map[string]*ToolSchema{},
	}

	paramsIdx := strings.Index(raw, "Params:")
	if paramsIdx == -1 {
		paramsIdx = strings.Index(raw, "Parameters:")
	}

	baseDesc := strings.TrimSpace(raw)
	if paramsIdx == -1 {
		return schema, baseDesc
	}

	baseDesc = strings.TrimSpace(raw[:paramsIdx])
	paramBlock := raw[paramsIdx:]

	// Strip the header line ("Params:" or "Parameters:")
	lines := strings.Split(paramBlock, "\n")
	for _, line := range lines[1:] {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		name, ps, ok := parseParamLine(line)
		if !ok {
			continue
		}
		schema.Properties[name] = ps
		if ps.Description == "required" || strings.Contains(line, ", required)") || strings.Contains(line, ",required)") {
			schema.Required = append(schema.Required, name)
		}
	}

	return schema, baseDesc
}

// parseParamLine parses one parameter annotation line.
// Format: name (type[, required]): description
// or:     name (type): description
func parseParamLine(line string) (name string, schema *ToolSchema, ok bool) {
	// Must have an opening paren.
	parenOpen := strings.Index(line, "(")
	if parenOpen < 0 {
		return "", nil, false
	}
	parenClose := strings.Index(line, ")")
	if parenClose < parenOpen {
		return "", nil, false
	}

	name = strings.TrimSpace(line[:parenOpen])
	if name == "" {
		return "", nil, false
	}

	typeAndFlags := strings.TrimSpace(line[parenOpen+1 : parenClose])
	parts := strings.SplitN(typeAndFlags, ",", 2)
	typeName := strings.TrimSpace(parts[0])

	var desc string
	rest := strings.TrimSpace(line[parenClose+1:])
	if strings.HasPrefix(rest, ":") {
		desc = strings.TrimSpace(rest[1:])
	}

	ps := &ToolSchema{
		Type:        normaliseTypeName(typeName),
		Description: desc,
	}
	return name, ps, true
}

func normaliseTypeName(t string) string {
	switch strings.ToLower(strings.TrimSpace(t)) {
	case "string", "str":
		return "string"
	case "number", "float", "float32", "float64":
		return "number"
	case "integer", "int", "int32", "int64":
		return "integer"
	case "bool", "boolean":
		return "boolean"
	case "array", "[]string", "[]int", "[]interface{}":
		return "array"
	case "object", "map":
		return "object"
	default:
		return "string" // safe default
	}
}

// FormatSchemaAsString returns a compact one-line representation of a schema
// for debugging / display purposes.
func FormatSchemaAsString(s ToolSchema) string {
	if len(s.Properties) == 0 {
		return fmt.Sprintf("{type:%s}", s.Type)
	}
	props := make([]string, 0, len(s.Properties))
	for k, v := range s.Properties {
		props = append(props, fmt.Sprintf("%s:%s", k, v.Type))
	}
	return fmt.Sprintf("{type:%s, props:[%s], required:%v}", s.Type, strings.Join(props, ","), s.Required)
}
