package iteragent_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	iteragent "github.com/GrayCodeAI/iteragent"
)

// ---------------------------------------------------------------------------
// parseSkillFile (via LoadSkills + FormatForPrompt as integration)
// ---------------------------------------------------------------------------

func writeSkillFile(t *testing.T, dir, skillName, content string) {
	t.Helper()
	skillDir := filepath.Join(dir, skillName)
	if err := os.MkdirAll(skillDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(skillDir, "SKILL.md"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestLoadSkills_Empty(t *testing.T) {
	ss, err := iteragent.LoadSkills([]string{t.TempDir()})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(ss.Skills) != 0 {
		t.Errorf("expected 0 skills, got %d", len(ss.Skills))
	}
}

func TestLoadSkills_MissingDir(t *testing.T) {
	ss, err := iteragent.LoadSkills([]string{"/nonexistent/path"})
	if err != nil {
		t.Fatalf("expected graceful skip for missing dir, got error: %v", err)
	}
	if len(ss.Skills) != 0 {
		t.Errorf("expected 0 skills, got %d", len(ss.Skills))
	}
}

func TestLoadSkills_BasicSkill(t *testing.T) {
	dir := t.TempDir()
	writeSkillFile(t, dir, "evolve", `---
name: evolve
description: Safely modify your own source code
---

# Evolve skill

Do stuff here.
`)

	ss, err := iteragent.LoadSkills([]string{dir})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(ss.Skills) != 1 {
		t.Fatalf("expected 1 skill, got %d", len(ss.Skills))
	}
	s := ss.Skills[0]
	if s.Name != "evolve" {
		t.Errorf("expected name 'evolve', got %q", s.Name)
	}
	if s.Description != "Safely modify your own source code" {
		t.Errorf("unexpected description: %q", s.Description)
	}
	if !strings.Contains(s.Content, "Do stuff here") {
		t.Errorf("expected body in Content, got %q", s.Content)
	}
}

func TestLoadSkills_FallbackSkillMd(t *testing.T) {
	dir := t.TempDir()
	skillDir := filepath.Join(dir, "myskill")
	os.MkdirAll(skillDir, 0o755)
	// Use skill.md (lowercase) instead of SKILL.md
	os.WriteFile(filepath.Join(skillDir, "skill.md"), []byte(`---
name: myskill
description: a test skill
---
body text`), 0o644)

	ss, _ := iteragent.LoadSkills([]string{dir})
	if len(ss.Skills) != 1 || ss.Skills[0].Name != "myskill" {
		t.Errorf("expected skill from skill.md fallback, got %+v", ss.Skills)
	}
}

func TestLoadSkills_FallbackNameFromDir(t *testing.T) {
	dir := t.TempDir()
	// Skill file with no frontmatter name — should fall back to directory name.
	writeSkillFile(t, dir, "autoname", "No frontmatter here, just body text.")

	ss, _ := iteragent.LoadSkills([]string{dir})
	if len(ss.Skills) != 1 || ss.Skills[0].Name != "autoname" {
		t.Errorf("expected skill named 'autoname' from dir, got %+v", ss.Skills)
	}
}

func TestLoadSkills_Deduplication_LastWins(t *testing.T) {
	dir1 := t.TempDir()
	dir2 := t.TempDir()

	// Same skill name in two dirs — dir2 (later) should win.
	writeSkillFile(t, dir1, "myskill", `---
name: myskill
description: version from dir1
---
dir1 body`)
	writeSkillFile(t, dir2, "myskill", `---
name: myskill
description: version from dir2
---
dir2 body`)

	ss, _ := iteragent.LoadSkills([]string{dir1, dir2})

	if len(ss.Skills) != 1 {
		t.Fatalf("expected 1 skill after dedup, got %d", len(ss.Skills))
	}
	if ss.Skills[0].Description != "version from dir2" {
		t.Errorf("expected dir2 to win dedup, got description %q", ss.Skills[0].Description)
	}
}

func TestLoadSkills_Deduplication_PreservesOrder(t *testing.T) {
	dir := t.TempDir()
	writeSkillFile(t, dir, "aardvark", `---
name: aardvark
description: first
---`)
	writeSkillFile(t, dir, "zebra", `---
name: zebra
description: last
---`)

	ss, _ := iteragent.LoadSkills([]string{dir})
	if len(ss.Skills) != 2 {
		t.Fatalf("expected 2 skills, got %d", len(ss.Skills))
	}
	// Both should be present.
	names := map[string]bool{}
	for _, s := range ss.Skills {
		names[s.Name] = true
	}
	if !names["aardvark"] || !names["zebra"] {
		t.Errorf("missing skills: %v", names)
	}
}

func TestLoadSkills_SkillGet(t *testing.T) {
	dir := t.TempDir()
	writeSkillFile(t, dir, "research", `---
name: research
description: Research skill
---
body`)

	ss, _ := iteragent.LoadSkills([]string{dir})
	s := ss.Get("research")
	if s == nil {
		t.Fatal("Get('research') returned nil")
	}
	if s.Name != "research" {
		t.Errorf("expected name 'research', got %q", s.Name)
	}
}

func TestLoadSkills_SkillGetMissing(t *testing.T) {
	ss := iteragent.SkillSetEmpty()
	if ss.Get("nonexistent") != nil {
		t.Error("Get on empty set should return nil")
	}
}

// ---------------------------------------------------------------------------
// FormatForPrompt
// ---------------------------------------------------------------------------

func TestFormatForPrompt_Empty(t *testing.T) {
	ss := iteragent.SkillSetEmpty()
	if ss.FormatForPrompt() != "" {
		t.Error("empty skill set should return empty prompt")
	}
}

func TestFormatForPrompt_ContainsSkillNames(t *testing.T) {
	dir := t.TempDir()
	writeSkillFile(t, dir, "evolve", `---
name: evolve
description: Evolution skill
---`)
	writeSkillFile(t, dir, "research", `---
name: research
description: Research skill
---`)

	ss, _ := iteragent.LoadSkills([]string{dir})
	prompt := ss.FormatForPrompt()

	if !strings.Contains(prompt, "evolve") {
		t.Error("prompt should contain 'evolve'")
	}
	if !strings.Contains(prompt, "research") {
		t.Error("prompt should contain 'research'")
	}
	if !strings.Contains(prompt, "<available_skills>") {
		t.Error("prompt should have <available_skills> tag")
	}
	if !strings.Contains(prompt, "read_file") {
		t.Error("prompt should mention read_file tool")
	}
}

// ---------------------------------------------------------------------------
// parseSkillFile edge cases
// directly tested via LoadSkills by reading files we write
// ---------------------------------------------------------------------------

func TestLoadSkills_QuotedFrontmatterValues(t *testing.T) {
	dir := t.TempDir()
	writeSkillFile(t, dir, "quoted", `---
name: "quoted skill"
description: 'single quoted desc'
---
body`)

	ss, _ := iteragent.LoadSkills([]string{dir})
	if len(ss.Skills) != 1 {
		t.Fatalf("expected 1 skill, got %d", len(ss.Skills))
	}
	if ss.Skills[0].Name != "quoted skill" {
		t.Errorf("expected name 'quoted skill', got %q", ss.Skills[0].Name)
	}
	if ss.Skills[0].Description != "single quoted desc" {
		t.Errorf("expected desc 'single quoted desc', got %q", ss.Skills[0].Description)
	}
}

func TestLoadSkills_NoFrontmatterBody(t *testing.T) {
	dir := t.TempDir()
	writeSkillFile(t, dir, "plain", "# Just a plain markdown file\n\nNo frontmatter.")

	ss, _ := iteragent.LoadSkills([]string{dir})
	if len(ss.Skills) != 1 {
		t.Fatalf("expected 1 skill, got %d", len(ss.Skills))
	}
	if !strings.Contains(ss.Skills[0].Content, "plain markdown") {
		t.Errorf("full content should be body when no frontmatter, got %q", ss.Skills[0].Content)
	}
}
