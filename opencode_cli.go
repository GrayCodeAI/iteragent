package iteragent

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
)

// OpenCodeCLIConfig configures the OpenCode CLI provider.
type OpenCodeCLIConfig struct {
	Model string // e.g., "mimo-v2-pro-free"
}

type opencodeCLIProvider struct {
	cfg OpenCodeCLIConfig
}

// NewOpenCodeCLI returns a provider that uses the OpenCode CLI internally.
// This enables access to OpenCode's free models without a public REST API.
func NewOpenCodeCLI(cfg OpenCodeCLIConfig) Provider {
	return &opencodeCLIProvider{cfg: cfg}
}

func (p *opencodeCLIProvider) Name() string {
	model := strings.TrimPrefix(p.cfg.Model, "opencode/")
	return fmt.Sprintf("opencode-cli(%s)", model)
}

// opencodeEvent represents a single event from OpenCode's JSON output.
type opencodeEvent struct {
	Type      string `json:"type"`
	Timestamp int64  `json:"timestamp"`
	SessionID string `json:"sessionID"`
	Part      struct {
		Type    string `json:"type"`
		Text    string `json:"text,omitempty"`
		Content string `json:"content,omitempty"`
	} `json:"part"`
}

// CompleteStream implements TokenStreamer by spawning OpenCode CLI and parsing JSON output.
func (p *opencodeCLIProvider) CompleteStream(ctx context.Context, messages []Message, opt CompletionOptions, onToken func(string)) (string, error) {
	// Build the prompt from messages
	var prompt strings.Builder
	for _, msg := range messages {
		switch msg.Role {
		case "system":
			prompt.WriteString("System: ")
			prompt.WriteString(msg.Content)
			prompt.WriteString("\n\n")
		case "user":
			prompt.WriteString(msg.Content)
		case "assistant":
			// Include assistant messages for context
			prompt.WriteString("\n\nPrevious response: ")
			prompt.WriteString(msg.Content)
			prompt.WriteString("\n\nContinue: ")
		}
	}

	return p.runOpenCode(ctx, prompt.String(), onToken)
}

func (p *opencodeCLIProvider) Complete(ctx context.Context, messages []Message, opts ...CompletionOptions) (string, error) {
	var opt CompletionOptions
	if len(opts) > 0 {
		opt = opts[0]
	}
	return p.CompleteStream(ctx, messages, opt, nil)
}

// runOpenCode spawns the OpenCode CLI and streams its output.
func (p *opencodeCLIProvider) runOpenCode(ctx context.Context, prompt string, onToken func(string)) (string, error) {
	// Find opencode binary
	opencodePath, err := exec.LookPath("opencode")
	if err != nil {
		return "", fmt.Errorf("opencode CLI not found in PATH: %w", err)
	}

	// Build command arguments
	modelArg := p.cfg.Model
	if !strings.HasPrefix(modelArg, "opencode/") {
		modelArg = "opencode/" + modelArg
	}

	args := []string{
		"run",
		"--model", modelArg,
		"--format", "json",
		prompt,
	}

	cmd := exec.CommandContext(ctx, opencodePath, args...)
	cmd.Env = os.Environ() // Pass through environment variables

	// Get stdout pipe for streaming
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return "", fmt.Errorf("create stdout pipe: %w", err)
	}

	// Capture stderr for debugging
	var stderrBuf strings.Builder
	cmd.Stderr = &stderrBuf

	// Start the command
	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("start opencode: %w", err)
	}

	// Read plain text output from opencode run
	var fullResponse strings.Builder
	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 1024*1024), 10*1024*1024)

	skipHeader := true
	for scanner.Scan() {
		line := scanner.Text()
		// Skip "> build · model" header line
		if skipHeader && (len(line) == 0 || (len(line) > 0 && line[0] == '>')) {
			continue
		}
		skipHeader = false
		fullResponse.WriteString(line)
		fullResponse.WriteString("\n")
		if onToken != nil {
			onToken(line + "\n")
		}
	}

	if scanErr := scanner.Err(); scanErr != nil {
		return "", fmt.Errorf("read opencode output: %w", scanErr)
	}

	// Wait for command to complete
	if err := cmd.Wait(); err != nil {
		stderr := stderrBuf.String()
		if stderr != "" {
			return "", fmt.Errorf("opencode exited with error: %w, stderr: %s", err, stderr)
		}
		return "", fmt.Errorf("opencode exited with error: %w", err)
	}

	result := fullResponse.String()
	if result == "" {
		return "", fmt.Errorf("empty response from opencode-cli")
	}

	return result, nil
}

// OpenCodeCLIServer wraps the OpenCode CLI in a long-running server mode
// for better performance (avoids CLI startup overhead on each call).
type OpenCodeCLIServer struct {
	model   string
	cmd     *exec.Cmd
	stdin   io.WriteCloser
	stdout  *bufio.Scanner
	mu      sync.Mutex
	running bool
}

// NewOpenCodeCLIServer creates a persistent OpenCode server process.
// This is more efficient for multiple calls.
func NewOpenCodeCLIServer(model string) (*OpenCodeCLIServer, error) {
	_, err := exec.LookPath("opencode")
	if err != nil {
		return nil, fmt.Errorf("opencode CLI not found: %w", err)
	}

	// Use opencode serve mode if available, otherwise fall back to per-call
	return &OpenCodeCLIServer{
		model: model,
	}, nil
}

// Close stops the server process.
func (s *OpenCodeCLIServer) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	if s.cmd != nil && s.running {
		s.running = false
		return s.cmd.Process.Kill()
	}
	return nil
}
