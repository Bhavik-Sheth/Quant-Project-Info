---
name: repo-documenter
description: Comprehensive documentation generator for repositories and sub-repositories. Use this skill when the user wants to understand a codebase, generate README files, create architecture documentation, or document specific modules. It provides tools to map the repository structure and templates for standard documentation files.
---

# Repo Documenter

This skill helps you create thorough documentation for any repository.

## When to Use
Use this skill when:
- The user asks to "document this repo" or "explain this codebase".
- You need to create a `README.md`, `ARCHITECTURE.md`, or `CONTRIBUTING.md`.
- You are onboarding into a new project and need to understand the structure.

## Workflow

### 1. Map the Territory
First, get a high-level view of the repository structure to understand what you are working with.

Run the `generate_structure.py` script:
\`\`\`bash
python SKILLS/repo-documenter/scripts/generate_structure.py [path_to_repo]
\`\`\`
*Note: If no path is provided, it defaults to the current directory.*

### 2. Analyze Key Files
Based on the structure, identify and read key configuration and entry point files to understand the project's purpose and dependencies.

- **Node.js**: `package.json`
- **Python**: `setup.py`, `requirements.txt`, `pyproject.toml`
- **General**: `docker-compose.yml`, `Makefile`

### 3. Plan the Documentation
Propose a documentation plan to the user. Common documents include:
- **README.md**: High-level overview, installation, usage.
- **ARCHITECTURE.md**: System design, data flow, component breakdown.
- **CONTRIBUTING.md**: Developer guidelines.
- **API Reference**: Detailed function/class documentation (if applicable).

### 4. drafted Content
Use the templates provided in `references/templates.md` to structure your documentation.

**To View Templates:**
\`\`\`
view_file SKILLS/repo-documenter/references/templates.md
\`\`\`

**Drafting Process:**
1.  **Overview**: Summarize the project's goal.
2.  **Installation/Usage**: precise commands based on your analysis of config files.
3.  **Structure**: Use the output from `generate_structure.py` to explain the folder layout.
4.  **Components**: Describe key modules found during analysis.

## Tips
- **Be Concise**: Users want actionable info, not fluff.
- **Verify Commands**: Ensure installation/run commands are actually valid based on the codebase.
- **Update Existing**: If a `README.md` already exists, offer to improve it rather than overwrite it, unless asked.
