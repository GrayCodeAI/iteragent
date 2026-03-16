package iteragent

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type Skill struct {
	Name        string
	Description string
	Content     string
	Path        string
}

type SkillSet struct {
	Skills []Skill
}

func LoadSkills(dirs []string) (*SkillSet, error) {
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

			skillPath := filepath.Join(dir, entry.Name(), "skill.md")
			data, err := os.ReadFile(skillPath)
			if err != nil {
				continue
			}

			content := string(data)
			name, desc := parseSkillFrontmatter(content)

			skills = append(skills, Skill{
				Name:        name,
				Description: desc,
				Content:     content,
				Path:        skillPath,
			})
		}
	}

	return &SkillSet{Skills: skills}, nil
}

func parseSkillFrontmatter(content string) (name, desc string) {
	lines := strings.Split(content, "\n")
	inFrontmatter := false

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "---" {
			if inFrontmatter {
				inFrontmatter = false
				continue
			}
			inFrontmatter = true
			continue
		}

		if inFrontmatter {
			if strings.HasPrefix(line, "name:") {
				name = strings.TrimSpace(strings.TrimPrefix(line, "name:"))
			}
			if strings.HasPrefix(line, "description:") {
				desc = strings.TrimSpace(strings.TrimPrefix(line, "description:"))
			}
		}
	}

	return name, desc
}

func (s *SkillSet) FormatForPrompt() string {
	if len(s.Skills) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("\n\n## Available Skills\n\n")
	sb.WriteString("You have access to skills that can help you:\n\n")

	for _, skill := range s.Skills {
		sb.WriteString(fmt.Sprintf("### %s\n", skill.Name))
		sb.WriteString(fmt.Sprintf("%s\n\n", skill.Description))
	}

	return sb.String()
}

func SkillSetEmpty() *SkillSet {
	return &SkillSet{Skills: []Skill{}}
}
