package iteragent

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

// OpenCodeCLIConfig configures the OpenCode CLI provider.
type OpenCodeCLIConfig struct {
	Model string // e.g., "mimo-v2-pro-free"
}

// opencodeCLIProvider runs a persistent `opencode serve` process and sends
// each message via `opencode run --attach <url> --continue`, avoiding the
// cold-start overhead of spawning a new process per message.
type opencodeCLIProvider struct {
	cfg       OpenCodeCLIConfig
	mu        sync.Mutex
	serverURL string
	serverCmd *exec.Cmd
	isFirst   bool // true until the first message is sent (no session to continue yet)
}

// NewOpenCodeCLI returns a provider that uses the OpenCode CLI internally.
// This enables access to OpenCode's free models without a public REST API.
func NewOpenCodeCLI(cfg OpenCodeCLIConfig) Provider {
	return &opencodeCLIProvider{cfg: cfg, isFirst: true}
}

func (p *opencodeCLIProvider) Name() string {
	model := strings.TrimPrefix(p.cfg.Model, "opencode/")
	return fmt.Sprintf("opencode-cli(%s)", model)
}

// Close shuts down the background opencode server process.
func (p *opencodeCLIProvider) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.serverCmd != nil && p.serverCmd.Process != nil {
		_ = p.serverCmd.Process.Kill()
		p.serverCmd = nil
		p.serverURL = ""
		p.isFirst = true
	}
	return nil
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

// CompleteStream implements TokenStreamer by routing through the persistent server.
func (p *opencodeCLIProvider) CompleteStream(ctx context.Context, messages []Message, opt CompletionOptions, onToken func(string)) (string, error) {
	// Only send the last user message — the server holds the full session history.
	prompt := lastUserMessage(messages)
	if prompt == "" {
		return "", fmt.Errorf("no user message to send")
	}

	if err := p.ensureServer(ctx); err != nil {
		// Fall back to direct (per-call) mode if the server can't start.
		return p.runDirect(ctx, messages, onToken)
	}

	return p.runViaServer(ctx, prompt, onToken)
}

func (p *opencodeCLIProvider) Complete(ctx context.Context, messages []Message, opts ...CompletionOptions) (string, error) {
	var opt CompletionOptions
	if len(opts) > 0 {
		opt = opts[0]
	}
	return p.CompleteStream(ctx, messages, opt, nil)
}

// ensureServer starts `opencode serve` on a free port if not already running.
func (p *opencodeCLIProvider) ensureServer(ctx context.Context) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.serverURL != "" {
		return nil
	}

	opencodePath, err := exec.LookPath("opencode")
	if err != nil {
		return fmt.Errorf("opencode CLI not found in PATH: %w", err)
	}

	port, err := findFreePort()
	if err != nil {
		return fmt.Errorf("find free port: %w", err)
	}

	cmd := exec.Command(opencodePath, "serve", "--port", strconv.Itoa(port))
	cmd.Env = os.Environ()
	cmd.Stdout = nil
	cmd.Stderr = nil

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start opencode server: %w", err)
	}

	url := fmt.Sprintf("http://127.0.0.1:%d", port)

	// Poll until the server is accepting connections (up to 15 s).
	client := &http.Client{Timeout: 500 * time.Millisecond}
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		resp, err := client.Get(url)
		if err == nil {
			resp.Body.Close()
			p.serverURL = url
			p.serverCmd = cmd
			return nil
		}
		select {
		case <-ctx.Done():
			_ = cmd.Process.Kill()
			return ctx.Err()
		case <-time.After(250 * time.Millisecond):
		}
	}

	_ = cmd.Process.Kill()
	return fmt.Errorf("opencode server did not become ready within 15s")
}

// runViaServer sends a message to the running server and streams the response.
func (p *opencodeCLIProvider) runViaServer(ctx context.Context, prompt string, onToken func(string)) (string, error) {
	opencodePath, _ := exec.LookPath("opencode")

	modelArg := p.cfg.Model
	if !strings.HasPrefix(modelArg, "opencode/") {
		modelArg = "opencode/" + modelArg
	}

	args := []string{
		"run",
		"--attach", p.serverURL,
		"--model", modelArg,
		"--format", "json",
	}

	p.mu.Lock()
	if !p.isFirst {
		args = append(args, "--continue")
	}
	p.isFirst = false
	p.mu.Unlock()

	args = append(args, prompt)
	return p.execAndStream(ctx, opencodePath, args, onToken)
}

// runDirect spawns a one-shot `opencode run` without a server (fallback).
func (p *opencodeCLIProvider) runDirect(ctx context.Context, messages []Message, onToken func(string)) (string, error) {
	opencodePath, err := exec.LookPath("opencode")
	if err != nil {
		return "", fmt.Errorf("opencode CLI not found in PATH: %w", err)
	}

	modelArg := p.cfg.Model
	if !strings.HasPrefix(modelArg, "opencode/") {
		modelArg = "opencode/" + modelArg
	}

	// Flatten messages into a single prompt for direct mode.
	var buf strings.Builder
	for _, msg := range messages {
		switch msg.Role {
		case "system":
			buf.WriteString(msg.Content)
			buf.WriteString("\n\n")
		case "user":
			buf.WriteString(msg.Content)
		case "assistant":
			buf.WriteString(msg.Content)
			buf.WriteString("\n\n")
		}
	}

	args := []string{"run", "--model", modelArg, "--format", "json", buf.String()}
	return p.execAndStream(ctx, opencodePath, args, onToken)
}

// execAndStream runs an opencode command and streams JSON events to onToken.
func (p *opencodeCLIProvider) execAndStream(ctx context.Context, bin string, args []string, onToken func(string)) (string, error) {
	cmd := exec.CommandContext(ctx, bin, args...)
	cmd.Env = os.Environ()

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return "", fmt.Errorf("create stdout pipe: %w", err)
	}
	var stderrBuf strings.Builder
	cmd.Stderr = &stderrBuf

	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("start opencode: %w", err)
	}

	var fullResponse strings.Builder
	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 1024*1024), 10*1024*1024)

	skipHeader := true
	for scanner.Scan() {
		line := scanner.Text()
		if skipHeader && (len(line) == 0 || line[0] == '>') {
			continue
		}
		skipHeader = false

		var event opencodeEvent
		if err := json.Unmarshal([]byte(line), &event); err == nil && event.Part.Text != "" {
			fullResponse.WriteString(event.Part.Text)
			if onToken != nil {
				onToken(event.Part.Text)
			}
		}
	}

	if scanErr := scanner.Err(); scanErr != nil {
		return "", fmt.Errorf("read opencode output: %w", scanErr)
	}

	if err := cmd.Wait(); err != nil {
		if stderr := stderrBuf.String(); stderr != "" {
			return "", fmt.Errorf("opencode error: %w — %s", err, stderr)
		}
		return "", fmt.Errorf("opencode error: %w", err)
	}

	result := fullResponse.String()
	if result == "" {
		return "", fmt.Errorf("empty response from opencode-cli")
	}
	return result, nil
}

// lastUserMessage returns the content of the last user message in the slice.
func lastUserMessage(messages []Message) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			return messages[i].Content
		}
	}
	return ""
}

// findFreePort returns a free TCP port on localhost.
func findFreePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}
