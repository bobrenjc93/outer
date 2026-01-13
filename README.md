# Outer

An iterative AI-driven development workflow tool that automates software development using AI providers (Claude, Codex, or Gemini).

## Overview

Outer orchestrates AI-powered development by:

1. Reading requirements from a `requirements.md` file
2. Generating a structured `plan.md` with TODO items
3. Iteratively executing tasks one at a time
4. Validating completed work against original requirements
5. Adding new tasks if gaps are found
6. Continuing until all requirements are met

## Installation

Ensure you have Python 3 and at least one supported AI CLI installed:

- [Claude CLI](https://github.com/anthropics/claude-code)
- [Codex CLI](https://github.com/openai/codex)
- [Gemini CLI](https://geminicli.com/)

## Usage

```bash
python outer.py -d /path/to/project -p claude --yolo -m 100
```

### Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--provider` | `-p` | `claude` | AI provider: `claude`, `codex`, or `gemini` |
| `--dir` | `-d` | `.` | Target project directory |
| `--requirements` | `-r` | `requirements.md` | Requirements filename |
| `--plan` | | `plan.md` | Plan filename |
| `--max-iterations` | `-m` | `50` | Maximum task iterations |
| `--dry-run` | | | Show what would be done without executing |
| `--yolo` | | | Skip permission prompts (auto-approve mode) |

## Workflow

### 1. Create Requirements

Create a `requirements.md` file in your project directory describing what you want to build.

### 2. Run Outer

```bash
python outer.py -d ./my-project -p claude
```

### 3. Generated Plan Format

Outer generates TODO items in this format:

```markdown
- [ ] TODO: <brief description>
  - Status: PENDING | IN_PROGRESS | DONE | VERIFIED
  - Priority: HIGH | MEDIUM | LOW
  - Verification: <how to verify this task is complete>
  - Notes: <any additional context>
```

### 4. Iteration Loop

- The AI executes one TODO at a time
- Each task is verified before marking complete
- After all TODOs are done, requirements are validated
- New TODOs are added if gaps are found
- Process continues until requirements are fully met

## Example

```bash
# Create a project with requirements
mkdir my-app && cd my-app
echo "Build a REST API with user authentication" > requirements.md

# Run outer with Claude
python outer.py -p claude --yolo

# Monitor progress as tasks are completed
# Final plan.md shows all completed work
```

## Supported Providers

| Provider | CLI Command | YOLO Flag |
|----------|-------------|-----------|
| Claude | `claude -p` | `--dangerously-skip-permissions` |
| Codex | `codex -q` | `--full-auto` |
| Gemini | `gemini` | `--auto-approve` |

## License

MIT
