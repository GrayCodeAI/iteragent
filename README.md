# iteragent

`iteragent` is a lightweight, zero-dependency Go agent framework for building LLM-powered applications. It provides a unified interface to multiple LLM providers, tool-use capabilities, streaming support, context compaction, and an extensible plugin system via MCP and OpenAPI.

## Features

- **Unified Provider Interface** — Switch between Anthropic, OpenAI, Azure OpenAI, Google Gemini, AWS Bedrock, NVIDIA, OpenAI-compatible APIs, and OpenAI Responses API with a single interface
- **Streaming Support** — Real-time token-by-token streaming via SSE for all supported providers
- **Tool Use** — Built-in tools (shell, file I/O, search, git operations, test execution) plus MCP and OpenAPI tool discovery
- **Context Compaction** — Automatic 3-tier context management to handle long conversations within token limits
- **Retry Logic** — Smart retry with exponential backoff for rate limits and transient errors
- **Zero Dependencies** — Uses only the Go standard library — no supply-chain risk from third-party packages
- **Concurrent Execution** — Safe parallel tool execution with configurable strategies (sequential, parallel, batched)
- **Sub-Agents** — Decompose tasks into specialized sub-agents with isolated tool sets

## Installation

```bash
go get github.com/GrayCodeAI/iteragent
```

## Quick Start

```go
package main

import (
	"fmt"
	"github.com/GrayCodeAI/iteragent"
)

func main() {
	agent := iteragent.New(
		iteragent.NewGemini(iteragent.GeminiConfig{
			Model:  "gemini-2.0-flash",
			APIKey: "YOUR_API_KEY",
		}),
		nil,
		nil,
	)

	response, err := agent.Run(
		context.Background(),
		"You are a helpful assistant.",
		"What is the capital of France?",
	)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	fmt.Println(response)
}
```

## Providers

| Provider | Function | Environment Variable for API Key |
|----------|----------|----------------------------------|
| Anthropic | `NewAnthropic(AntropicConfig{...})` | `ANTHROPIC_API_KEY` |
| OpenAI | `NewOpenAICompat(OpenAICompatConfig{...})` | `OPENAI_API_KEY` |
| Azure OpenAI | `NewAzureOpenAI(AzureOpenAIConfig{...})` | — |
| Google Gemini | `NewGemini(GeminiConfig{...})` | `GEMINI_API_KEY` |
| AWS Bedrock | `NewBedrock(BedrockConfig{...})` | — |
| NVIDIA | `NewOpenAICompat(OpenAICompatConfig{...})` | `NVIDIA_API_KEY` |
| OpenAI Responses | `NewOpenAIResponses(OpenAIResponsesConfig{...})` | — |

Alternatively, use `iteragent.NewProvider(name, apiKey)` to select via environment variable:

```bash
ITERATE_PROVIDER=gemini GEMINI_API_KEY=xxx go run main.go
```

## Architecture

- **Agent** — Core reasoning loop that manages message history, tool execution, and context compaction
- **Provider** — LLM backend interface with `Complete` and optional `CompleteStream` (TokenStreamer) methods
- **Tools** — Callable functions the agent can invoke (shell, file read/write, git, search, etc.)
- **Context Compaction** — Automatically compresses conversation history when token limits are approached
- **MCP** — Connect to external tool servers via Model Context Protocol
- **OpenAPI** — Auto-generate tools from OpenAPI specifications

## Building and Testing

```bash
go build ./...
go test ./...
go test -race ./...
```

## License

MIT — see [LICENSE](LICENSE)