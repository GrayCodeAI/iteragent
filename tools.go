package iteragent

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

var (
	protectedPaths   []string
	protectedPathsMu sync.RWMutex

	dangerousPatterns []*regexp.Regexp
	dangerousPatternsMu sync.RWMutex
)

func init() {
	patterns := []string{
		// Recursive force remove
		`\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*|-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*)\b`,
		// Dangerous chmod
		`\bchmod\s+-R\s+777\b`,
		// Write to system paths
		`>\s*/etc/`,
		// Pipe to shell from network
		`\bcurl\b.*\|\s*(ba)?sh\b`,
		`\bwget\b.*\|\s*(ba)?sh\b`,
		// Fork bombs and process abuse
		`:(){ :\|:& };:`,
		`\bmkfs\b`,
		`\bdd\b.*of=/dev/`,
		// Destructive redirects
		`>\s*/dev/sd`,
		`>\s*/dev/nvme`,
		// Privilege escalation
		`\bsudo\s+rm\b`,
		`\bsudo\s+chmod\b.*777`,
		// Network exfiltration
		`\bnc\s+-[el]`,
		`\bncat\b.*-e`,
		`\bsocat\b.*exec`,
		// Environment variable leakage to network
		`\benv\b.*\|\s*(nc|curl|wget)`,
		`\bprintenv\b.*\|\s*(nc|curl|wget)`,
		// Dangerous git operations
		`\bgit\s+push\s+.*--force\b.*\bmain\b`,
		`\bgit\s+push\s+.*--force\b.*\bmaster\b`,
		`\bgit\s+clean\s+-[a-zA-Z]*f[a-zA-Z]*d\b`,
		// Shellshock-style
		`\(\)\s*\{.*\bexport\b`,
	}
	for _, p := range patterns {
		re, err := regexp.Compile(p)
		if err == nil {
			dangerousPatterns = append(dangerousPatterns, re)
		}
	}
}

func SetProtectedPaths(paths []string) {
	protectedPathsMu.Lock()
	defer protectedPathsMu.Unlock()
	protectedPaths = paths
}

func GetProtectedPaths() []string {
	protectedPathsMu.RLock()
	defer protectedPathsMu.RUnlock()
	return protectedPaths
}

func isPathProtected(path string) bool {
	protectedPathsMu.RLock()
	defer protectedPathsMu.RUnlock()
	for _, p := range protectedPaths {
		if strings.HasPrefix(path, p) {
			return true
		}
	}
	return false
}

func isCommandDangerous(cmd string) bool {
	dangerousPatternsMu.RLock()
	defer dangerousPatternsMu.RUnlock()
	for _, re := range dangerousPatterns {
		if re.MatchString(cmd) {
			return true
		}
	}
	return false
}

// safeJoin joins repoPath with the user-supplied relative path and ensures the
// result stays within repoPath, preventing path traversal attacks.
func safeJoin(repoPath, rel string) (string, error) {
	absRepo, err := filepath.Abs(repoPath)
	if err != nil {
		return "", fmt.Errorf("invalid repo path: %w", err)
	}
	absRepo = filepath.Clean(absRepo) + string(filepath.Separator)

	joined := filepath.Join(repoPath, rel)
	absJoined, err := filepath.Abs(joined)
	if err != nil {
		return "", fmt.Errorf("invalid path: %w", err)
	}
	absJoined = filepath.Clean(absJoined)
	if !strings.HasPrefix(absJoined, absRepo) {
		return "", fmt.Errorf("path %q is outside the repository", rel)
	}
	// Resolve symlinks to prevent symlink-based path traversal.
	resolved, err := filepath.EvalSymlinks(absJoined)
	if err != nil {
		// File may not exist yet (for write operations) — that's OK.
		if os.IsNotExist(err) {
			return absJoined, nil
		}
		return "", fmt.Errorf("resolve symlink: %w", err)
	}
	if !strings.HasPrefix(resolved, absRepo) {
		return "", fmt.Errorf("path %q resolves outside the repository via symlink", rel)
	}
	return resolved, nil
}

// DefaultTools returns all built-in tools available to the agent.
func DefaultTools(repoPath string) []Tool {
	return []Tool{
		BashTool(repoPath),
		ReadFileTool(repoPath),
		WriteFileTool(repoPath),
		EditFileTool(repoPath),
		ListFilesTool(repoPath),
		SearchTool(repoPath),
		GitDiffTool(repoPath),
		GitCommitTool(repoPath),
		GitRevertTool(repoPath),
		RunTestsTool(repoPath),
	}
}

func BashTool(repoPath string) Tool {
	return Tool{
		Name:        "bash",
		Description: "Run a shell command in the repo directory.\nArgs: {\"cmd\": \"go build ./...\"}",
		Execute: func(ctx context.Context, args map[string]string) (string, error) {
			cmd := args["cmd"]
			if cmd == "" {
				return "", fmt.Errorf("cmd is required")
			}
			if isCommandDangerous(cmd) {
				return "", fmt.Errorf("command contains dangerous pattern: %s", cmd)
			}
			ctx, cancel := context.WithTimeout(ctx, 60*time.Second)
			defer cancel()

			c := exec.CommandContext(ctx, "bash", "-c", cmd)
			c.Dir = repoPath
			var out bytes.Buffer
			c.Stdout = &out
			c.Stderr = &out
			err := c.Run()
			return out.String(), err
		},
	}
}

func ReadFileTool(repoPath string) Tool {
	return Tool{
		Name:        "read_file",
		Description: "Read a file from the repo.\nArgs: {\"path\": \"internal/agent/agent.go\"}",
		Execute: func(ctx context.Context, args map[string]string) (string, error) {
			path, err := safeJoin(repoPath, args["path"])
			if err != nil {
				return "", err
			}
			data, err := os.ReadFile(path)
			if err != nil {
				return "", fmt.Errorf("read %s: %w", args["path"], err)
			}
			return string(data), nil
		},
	}
}

func WriteFileTool(repoPath string) Tool {
	return Tool{
		Name:        "write_file",
		Description: "Write or overwrite a file in the repo.\nArgs: {\"path\": \"internal/agent/agent.go\", \"content\": \"...\"}",
		Execute: func(ctx context.Context, args map[string]string) (string, error) {
			path, err := safeJoin(repoPath, args["path"])
			if err != nil {
				return "", err
			}
			if isPathProtected(path) {
				return "", fmt.Errorf("write to %s is protected", args["path"])
			}
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				return "", err
			}
			if err := os.WriteFile(path, []byte(args["content"]), 0o644); err != nil {
				return "", fmt.Errorf("write %s: %w", args["path"], err)
			}
			return fmt.Sprintf("wrote %s (%d bytes)", args["path"], len(args["content"])), nil
		},
	}
}

func EditFileTool(repoPath string) Tool {
	return Tool{
		Name:        "edit_file",
		Description: "Edit a file by replacing oldString with newString.\nArgs: {\"path\": \"file.go\", \"oldString\": \"old\", \"newString\": \"new\"}",
		Execute: func(ctx context.Context, args map[string]string) (string, error) {
			path, err := safeJoin(repoPath, args["path"])
			if err != nil {
				return "", err
			}
			if isPathProtected(path) {
				return "", fmt.Errorf("edit %s is protected", args["path"])
			}
			oldStr := args["oldString"]
			newStr := args["newString"]
			if oldStr == "" {
				return "", fmt.Errorf("oldString is required")
			}

			data, err := os.ReadFile(path)
			if err != nil {
				return "", fmt.Errorf("read %s: %w", args["path"], err)
			}

			content := string(data)
			if !strings.Contains(content, oldStr) {
				return "", fmt.Errorf("oldString not found in file")
			}

			newContent := strings.Replace(content, oldStr, newStr, 1)
			if err := os.WriteFile(path, []byte(newContent), 0o644); err != nil {
				return "", fmt.Errorf("write %s: %w", args["path"], err)
			}
			return fmt.Sprintf("edited %s", args["path"]), nil
		},
	}
}

	func ListFilesTool(repoPath string) Tool {
	return Tool{
		Name:        "list_files",
		Description: "List all files in the repo.\nArgs: {}",
		Execute: func(ctx context.Context, args map[string]string) (string, error) {
			const maxFiles = 1000
			var files []string
			var walkErr error
			err := filepath.Walk(repoPath, func(path string, info os.FileInfo, err error) error {
				if err != nil {
					walkErr = err
					return nil
				}
				if info.IsDir() && (info.Name() == ".git" || info.Name() == "vendor" || info.Name() == "node_modules") {
					return filepath.SkipDir
				}
				rel, _ := filepath.Rel(repoPath, path)
				files = append(files, rel)
				if len(files) >= maxFiles {
					return filepath.SkipAll
				}
				return nil
			})
			if err != nil && err != filepath.SkipAll {
				return strings.Join(files, "\n"), err
			}
			result := strings.Join(files, "\n")
			if len(files) >= maxFiles {
				result += fmt.Sprintf("\n\n[truncated: showing first %d files]", maxFiles)
			}
			return result, walkErr
		},
	}
}

func SearchTool(repoPath string) Tool {
	return Tool{
		Name:        "search",
		Description: "Search for text in files.\nArgs: {\"pattern\": \"TODO\", \"path\": \".\"}",
		Execute: func(ctx context.Context, args map[string]string) (string, error) {
			pattern := args["pattern"]
			if pattern == "" {
				return "", fmt.Errorf("pattern is required")
			}
			path := repoPath
			if args["path"] != "" {
				var err error
				path, err = safeJoin(repoPath, args["path"])
				if err != nil {
					return "", err
				}
			}

			ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
			defer cancel()

			c := exec.CommandContext(ctx, "grep", "-r", "-n", "--", pattern, path)
			c.Dir = repoPath
			out, err := c.CombinedOutput()
			return string(out), err
		},
	}
}

func GitDiffTool(repoPath string) Tool {
	return Tool{
		Name:        "git_diff",
		Description: "Show current unstaged changes.\nArgs: {}",
		Execute: func(ctx context.Context, args map[string]string) (string, error) {
			c := exec.CommandContext(ctx, "git", "diff")
			c.Dir = repoPath
			out, err := c.Output()
			return string(out), err
		},
	}
}

	func GitCommitTool(repoPath string) Tool {
	return Tool{
		Name:        "git_commit",
		Description: "Stage all changes and commit.\nArgs: {\"message\": \"feat: improve error handling\"}",
		Execute: func(ctx context.Context, args map[string]string) (string, error) {
			// Check if there are any changes to commit.
			status := exec.CommandContext(ctx, "git", "status", "--porcelain")
			status.Dir = repoPath
			statusOut, err := status.CombinedOutput()
			if err != nil {
				return string(statusOut), fmt.Errorf("git status: %w", err)
			}
			if len(bytes.TrimSpace(statusOut)) == 0 {
				return "nothing to commit", nil
			}

			msg := args["message"]
			if msg == "" {
				msg = fmt.Sprintf("iterate: auto-improvement session %s", time.Now().Format("2006-01-02"))
			}

			// Stage only tracked/changed files, not everything.
			diffFiles := exec.CommandContext(ctx, "git", "diff", "--name-only", "--diff-filter=ACMR")
			diffFiles.Dir = repoPath
			diffOut, _ := diffFiles.CombinedOutput()

			untracked := exec.CommandContext(ctx, "git", "ls-files", "--others", "--exclude-standard")
			untracked.Dir = repoPath
			untrackedOut, _ := untracked.CombinedOutput()

			allFiles := strings.TrimSpace(string(diffOut) + "\n" + string(untrackedOut))
			if allFiles == "" {
				return "nothing to commit", nil
			}

			fileList := strings.Split(allFiles, "\n")
			addArgs := append([]string{"add", "--"}, fileList...)
			add := exec.CommandContext(ctx, "git", addArgs...)
			add.Dir = repoPath
			if out, err := add.CombinedOutput(); err != nil {
				return string(out), fmt.Errorf("git add: %w", err)
			}

			commit := exec.CommandContext(ctx, "git", "commit", "-m", msg)
			commit.Dir = repoPath
			commit.Env = append(os.Environ(),
				"GIT_AUTHOR_NAME=iterate[bot]",
				"GIT_AUTHOR_EMAIL=iterate@users.noreply.github.com",
				"GIT_COMMITTER_NAME=iterate[bot]",
				"GIT_COMMITTER_EMAIL=iterate@users.noreply.github.com",
			)
			out, err := commit.CombinedOutput()
			return string(out), err
		},
	}
}

func GitRevertTool(repoPath string) Tool {
	return Tool{
		Name:        "git_revert",
		Description: "Discard all unstaged changes.\nArgs: {}",
		Execute: func(ctx context.Context, args map[string]string) (string, error) {
			c := exec.CommandContext(ctx, "git", "checkout", "--", ".")
			c.Dir = repoPath
			out, err := c.CombinedOutput()
			return string(out), err
		},
	}
}

func RunTestsTool(repoPath string) Tool {
	return Tool{
		Name:        "run_tests",
		Description: "Run go build and go test.\nArgs: {}",
		Execute: func(ctx context.Context, args map[string]string) (string, error) {
			ctx, cancel := context.WithTimeout(ctx, 120*time.Second)
			defer cancel()

			var results strings.Builder

			build := exec.CommandContext(ctx, "go", "build", "./...")
			build.Dir = repoPath
			out, err := build.CombinedOutput()
			results.WriteString("=== go build ===\n")
			results.Write(out)
			if err != nil {
				results.WriteString("\nBUILD FAILED\n")
				return results.String(), fmt.Errorf("build failed")
			}
			results.WriteString("BUILD OK\n\n")

			test := exec.CommandContext(ctx, "go", "test", "./...")
			test.Dir = repoPath
			out, err = test.CombinedOutput()
			results.WriteString("=== go test ===\n")
			results.Write(out)
			if err != nil {
				results.WriteString("\nTESTS FAILED\n")
				return results.String(), fmt.Errorf("tests failed")
			}
			results.WriteString("ALL TESTS PASSED\n")

			return results.String(), nil
		},
	}
}
