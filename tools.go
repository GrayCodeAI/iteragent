package iteragent

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

var protectedPaths []string

var dangerousPatterns = []string{
	"rm -rf",
	"git push --force",
	"git push -f",
	"--force-with-lease",
	"chmod -R 777",
	"> /etc/",
	"curl .* | sh",
	"wget .* | sh",
}

func SetProtectedPaths(paths []string) {
	protectedPaths = paths
}

func GetProtectedPaths() []string {
	return protectedPaths
}

func isPathProtected(path string) bool {
	for _, p := range protectedPaths {
		if strings.HasPrefix(path, p) {
			return true
		}
	}
	return false
}

func isCommandDangerous(cmd string) bool {
	for _, pattern := range dangerousPatterns {
		if strings.Contains(cmd, pattern) {
			return true
		}
	}
	return false
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
			_ = c.Run()
			return out.String(), nil
		},
	}
}

func ReadFileTool(repoPath string) Tool {
	return Tool{
		Name:        "read_file",
		Description: "Read a file from the repo.\nArgs: {\"path\": \"internal/agent/agent.go\"}",
		Execute: func(ctx context.Context, args map[string]string) (string, error) {
			path := filepath.Join(repoPath, args["path"])
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
			path := filepath.Join(repoPath, args["path"])
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
			path := filepath.Join(repoPath, args["path"])
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
			var files []string
			err := filepath.Walk(repoPath, func(path string, info os.FileInfo, err error) error {
				if err != nil {
					return nil
				}
				if info.IsDir() && (info.Name() == ".git" || info.Name() == "vendor" || info.Name() == "node_modules") {
					return filepath.SkipDir
				}
				rel, _ := filepath.Rel(repoPath, path)
				files = append(files, rel)
				return nil
			})
			return strings.Join(files, "\n"), err
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
				path = filepath.Join(repoPath, args["path"])
			}

			c := exec.CommandContext(ctx, "grep", "-r", "-n", pattern, path)
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
			msg := args["message"]
			if msg == "" {
				msg = fmt.Sprintf("iterate: auto-improvement session %s", time.Now().Format("2006-01-02"))
			}

			add := exec.CommandContext(ctx, "git", "add", "-A")
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
