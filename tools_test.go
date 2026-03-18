package iteragent_test

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	iteragent "github.com/GrayCodeAI/iteragent"
)

// TestDefaultToolsExist verifies that default tools are registered.
func TestDefaultToolsExist(t *testing.T) {
	tools := iteragent.DefaultTools(".")
	names := make(map[string]bool)
	for _, t := range tools {
		names[t.Name] = true
	}
	required := []string{"bash", "read_file", "write_file", "list_files", "search"}
	for _, name := range required {
		if !names[name] {
			t.Errorf("missing required tool: %s", name)
		}
	}
}

// TestReadFileTool verifies the read_file tool reads a file using a repo-relative path.
func TestReadFileTool(t *testing.T) {
	dir := t.TempDir()
	_ = os.WriteFile(filepath.Join(dir, "test.txt"), []byte("hello from test"), 0o644)

	tools := iteragent.ToolMap(iteragent.DefaultTools(dir))
	tool, ok := tools["read_file"]
	if !ok {
		t.Fatal("read_file tool not found")
	}

	// Path is relative to repoPath (dir).
	out, err := tool.Execute(context.Background(), map[string]string{"path": "test.txt"})
	if err != nil {
		t.Fatalf("read_file error: %v", err)
	}
	if !strings.Contains(out, "hello from test") {
		t.Errorf("expected file content in output, got: %s", out)
	}
}

// TestWriteFileTool verifies the write_file tool creates a file using a repo-relative path.
func TestWriteFileTool(t *testing.T) {
	dir := t.TempDir()

	tools := iteragent.ToolMap(iteragent.DefaultTools(dir))
	tool, ok := tools["write_file"]
	if !ok {
		t.Fatal("write_file tool not found")
	}

	// Path is relative to repoPath (dir).
	_, err := tool.Execute(context.Background(), map[string]string{
		"path":    "out.txt",
		"content": "written by test",
	})
	if err != nil {
		t.Fatalf("write_file error: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(dir, "out.txt"))
	if err != nil {
		t.Fatalf("read back error: %v", err)
	}
	if string(data) != "written by test" {
		t.Errorf("want %q, got %q", "written by test", string(data))
	}
}

// TestListFilesTool verifies the list_files tool returns directory contents.
func TestListFilesTool(t *testing.T) {
	dir := t.TempDir()
	_ = os.WriteFile(filepath.Join(dir, "a.txt"), []byte("a"), 0o644)
	_ = os.WriteFile(filepath.Join(dir, "b.go"), []byte("b"), 0o644)

	tools := iteragent.ToolMap(iteragent.DefaultTools(dir))
	tool, ok := tools["list_files"]
	if !ok {
		t.Fatal("list_files tool not found")
	}

	out, err := tool.Execute(context.Background(), map[string]string{"path": dir})
	if err != nil {
		t.Fatalf("list_files error: %v", err)
	}
	if !strings.Contains(out, "a.txt") || !strings.Contains(out, "b.go") {
		t.Errorf("expected file names in output, got: %s", out)
	}
}

// TestBashTool verifies the bash tool runs a command.
func TestBashTool(t *testing.T) {
	tools := iteragent.ToolMap(iteragent.DefaultTools("."))
	tool, ok := tools["bash"]
	if !ok {
		t.Fatal("bash tool not found")
	}

	out, err := tool.Execute(context.Background(), map[string]string{"cmd": "echo hello_bash"})
	if err != nil {
		t.Fatalf("bash error: %v", err)
	}
	if !strings.Contains(out, "hello_bash") {
		t.Errorf("expected hello_bash in output, got: %s", out)
	}
}

// TestContextCompaction verifies that CompactMessagesTiered reduces message count
// when the token budget is exceeded.
func TestContextCompaction(t *testing.T) {
	// Build 40 messages each with ~500 chars (~125 tokens). Total ~5000 tokens.
	var msgs []iteragent.Message
	for i := 0; i < 40; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		msgs = append(msgs, iteragent.Message{Role: role, Content: strings.Repeat("x", 500)})
	}

	// Set a small budget so compaction triggers (threshold = 80% of 2000 = 1600 tokens).
	cfg := iteragent.DefaultContextConfig()
	cfg.MaxTokens = 2000
	cfg.KeepRecent = 5
	cfg.KeepFirst = 2
	compacted := iteragent.CompactMessagesTiered(msgs, cfg)

	if len(compacted) >= len(msgs) {
		t.Errorf("expected compaction to reduce messages from %d, got %d", len(msgs), len(compacted))
	}
}

// TestSkillsLoadSKILLmd verifies that SKILL.md is preferred over skill.md.
func TestSkillsLoadSKILLmd(t *testing.T) {
	dir := t.TempDir()
	skillDir := filepath.Join(dir, "myskill")
	_ = os.MkdirAll(skillDir, 0o755)

	content := "---\nname: myskill\ndescription: does things\n---\n\nFull body here."
	_ = os.WriteFile(filepath.Join(skillDir, "SKILL.md"), []byte(content), 0o644)

	set, err := iteragent.LoadSkills([]string{dir})
	if err != nil {
		t.Fatalf("LoadSkills: %v", err)
	}
	if len(set.Skills) != 1 {
		t.Fatalf("expected 1 skill, got %d", len(set.Skills))
	}
	if set.Skills[0].Name != "myskill" {
		t.Errorf("want name=myskill, got %q", set.Skills[0].Name)
	}
	if set.Skills[0].Content != "Full body here." {
		t.Errorf("unexpected body: %q", set.Skills[0].Content)
	}
}

// TestSkillsFormatForPrompt verifies the XML format.
func TestSkillsFormatForPrompt(t *testing.T) {
	dir := t.TempDir()
	skillDir := filepath.Join(dir, "mys")
	_ = os.MkdirAll(skillDir, 0o755)
	_ = os.WriteFile(filepath.Join(skillDir, "SKILL.md"), []byte("---\nname: mys\ndescription: my skill\n---\nbody"), 0o644)

	set, _ := iteragent.LoadSkills([]string{dir})
	prompt := set.FormatForPrompt()

	if !strings.Contains(prompt, "<available_skills>") {
		t.Error("expected <available_skills> XML tag")
	}
	if !strings.Contains(prompt, `name="mys"`) {
		t.Error("expected skill name in XML")
	}
	if strings.Contains(prompt, "body") {
		t.Error("full skill body should NOT appear in prompt (progressive disclosure)")
	}
}
