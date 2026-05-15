package iteragent

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Skill represents a loaded agent skill.
type Skill struct {
	Name        string
	Description string
	// Content holds the full skill body (everything after the frontmatter).
	// It is NOT injected into the system prompt by default — only metadata is.
	Content string
	// RawFrontmatter is the raw YAML frontmatter block.
	RawFrontmatter string
	Path           string
}

// SkillSet is a collection of loaded skills.
type SkillSet struct {
	Skills []Skill
}

// LoadSkills scans dirs for skill subdirectories. For each subdirectory it looks
// for SKILL.md (AgentSkills standard) first, then falls back to skill.md.
// Skills are deduplicated by name; the last directory in the list wins,
// so callers can override built-in skills by appending their own dirs.
func LoadSkills(dirs []string) (*SkillSet, error) {
	// Use an ordered map: name → skill, preserving insertion order for stable output.
	seen := map[string]int{} // name → index in skills slice
	var skills []Skill

	for _, dir := range dirs {
		entries, err := os.ReadDir(dir)
		if err != nil {
			continue
		}

		for _, entry := range entries {
			if !entry.IsDir() {
				continue
			}

			// Try SKILL.md first (AgentSkills standard), then skill.md.
			var skillPath string
			candidates := []string{
				filepath.Join(dir, entry.Name(), "SKILL.md"),
				filepath.Join(dir, entry.Name(), "skill.md"),
			}
			for _, c := range candidates {
				if _, err := os.Stat(c); err == nil {
					skillPath = c
					break
				}
			}
			if skillPath == "" {
				continue
			}

			data, err := os.ReadFile(skillPath)
			if err != nil {
				continue
			}

			raw := string(data)
			name, desc, frontmatter, body := parseSkillFile(raw)
			if name == "" {
				name = entry.Name()
			}

			skill := Skill{
				Name:           name,
				Description:    desc,
				Content:        body,
				RawFrontmatter: frontmatter,
				Path:           skillPath,
			}

			// Deduplicate: later dirs override earlier ones (last-writer-wins).
			if idx, exists := seen[name]; exists {
				skills[idx] = skill
			} else {
				seen[name] = len(skills)
				skills = append(skills, skill)
			}
		}
	}

	return &SkillSet{Skills: skills}, nil
}

// parseSkillFile parses a skill file and returns (name, description, frontmatter, body).
func parseSkillFile(content string) (name, desc, frontmatter, body string) {
	lines := strings.Split(content, "\n")
	inFrontmatter := false
	frontmatterEnd := -1
	var fmLines []string

	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "---" {
			if !inFrontmatter && i == 0 {
				inFrontmatter = true
				continue
			}
			if inFrontmatter {
				frontmatterEnd = i
				inFrontmatter = false
				continue
			}
		}
		if inFrontmatter {
			fmLines = append(fmLines, line)
			if strings.HasPrefix(trimmed, "name:") {
				name = strings.TrimSpace(strings.TrimPrefix(trimmed, "name:"))
				// Remove surrounding quotes if present.
				name = strings.Trim(name, `"'`)
			}
			if strings.HasPrefix(trimmed, "description:") {
				desc = strings.TrimSpace(strings.TrimPrefix(trimmed, "description:"))
				desc = strings.Trim(desc, `"'`)
			}
		}
	}

	frontmatter = strings.Join(fmLines, "\n")

	if frontmatterEnd >= 0 && frontmatterEnd+1 < len(lines) {
		body = strings.TrimSpace(strings.Join(lines[frontmatterEnd+1:], "\n"))
	} else if frontmatterEnd < 0 {
		// No frontmatter — entire file is the body.
		body = strings.TrimSpace(content)
	}

	return
}

// FormatForPrompt returns an XML <available_skills> block listing skill metadata.
// Only name and description are included — the full content is NOT injected.
// The LLM can request full skill content via the read_file tool.
func (s *SkillSet) FormatForPrompt() string {
	if len(s.Skills) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("\n\n<available_skills>\n")
	for _, skill := range s.Skills {
		sb.WriteString(fmt.Sprintf(
			"  <skill name=%q description=%q path=%q />\n",
			skill.Name, skill.Description, skill.Path,
		))
	}
	sb.WriteString("</available_skills>\n")
	sb.WriteString("\nTo use a skill, read its full content with the read_file tool using the path above.\n")
	return sb.String()
}

// Get returns the skill with the given name, or nil if not found.
func (s *SkillSet) Get(name string) *Skill {
	for i := range s.Skills {
		if s.Skills[i].Name == name {
			return &s.Skills[i]
		}
	}
	return nil
}

// SkillSetEmpty returns an empty SkillSet.
func SkillSetEmpty() *SkillSet {
	return &SkillSet{Skills: []Skill{}}
}
