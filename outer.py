#!/usr/bin/env python3
"""
Outer - An iterative AI-driven development workflow tool.

This script:
1. Reads requirements.md from the current directory
2. Asks an AI (Claude/Codex/Gemini) to generate a plan.md with TODOs
3. Iteratively executes tasks, updating plan.md
4. Validates completed work against requirements
5. Continues until all requirements are met

Usage:
python outer.py -d /path/to/project -p claude --yolo -m 100
"""

import argparse
import os
import re
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Optional


class AIProvider(Enum):
    CLAUDE = "claude"
    CODEX = "codex"
    GEMINI = "gemini"


# TODO format specification
TODO_FORMAT = """
## TODO Format Specification

Each TODO item must follow this exact format:

```
- [ ] TODO: <brief description>
  - Status: PENDING | IN_PROGRESS | DONE | VERIFIED
  - Priority: HIGH | MEDIUM | LOW
  - Verification: <how to verify this task is complete>
  - Notes: <any additional context>
```

Example:
- [ ] TODO: Implement user authentication
  - Status: PENDING
  - Priority: HIGH
  - Verification: Run `pytest tests/test_auth.py` - all tests pass
  - Notes: Use JWT tokens for session management
"""


def get_ai_command(provider: AIProvider, yolo: bool = False) -> list[str]:
    """Get the command to invoke the specified AI provider."""
    commands = {
        AIProvider.CLAUDE: ["claude", "-p"],
        AIProvider.CODEX: ["codex", "exec", "--skip-git-repo-check"],
        AIProvider.GEMINI: ["gemini"],  # Adjust based on actual CLI
    }

    # YOLO flags for each provider (skip permission prompts)
    yolo_flags = {
        AIProvider.CLAUDE: ["--dangerously-skip-permissions"],
        AIProvider.CODEX: ["--full-auto"],
        AIProvider.GEMINI: ["--auto-approve"],  # Adjust based on actual CLI
    }

    cmd = commands[provider].copy()
    if yolo:
        cmd.extend(yolo_flags[provider])

    return cmd


def call_ai(provider: AIProvider, prompt: str, working_dir: Path, yolo: bool = False, timeout: int = 300, max_retries: int = 3) -> str:
    """Call the AI provider with the given prompt and return the response.

    Uses exponential backoff for retries on failure.
    """
    cmd = get_ai_command(provider, yolo=yolo)

    last_exception = None
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                cmd + [prompt],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(working_dir),
            )

            if result.returncode != 0:
                print(f"Warning: AI command returned non-zero exit code: {result.returncode}")
                print(f"Stderr: {result.stderr}")
                # Treat non-zero exit code as a failure that can be retried
                raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

            return result.stdout.strip()
        except subprocess.TimeoutExpired as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = 10 * (2 ** attempt)  # Exponential backoff: 10s, 20s, 40s, ...
                print(f"AI call timed out (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Error: AI call timed out after {max_retries} attempts")
        except subprocess.CalledProcessError as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = 10 * (2 ** attempt)  # Exponential backoff: 10s, 20s, 40s, ...
                print(f"AI call failed (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Error: AI call failed after {max_retries} attempts")
        except FileNotFoundError:
            print(f"Error: {provider.value} CLI not found. Make sure it's installed and in PATH.")
            sys.exit(1)

    # All retries exhausted
    print(f"Error: AI invocation failed after {max_retries} attempts. Stopping.")
    sys.exit(1)


def read_file(filepath: str) -> Optional[str]:
    """Read and return file contents, or None if file doesn't exist."""
    path = Path(filepath)
    if path.exists():
        return path.read_text()
    return None


def write_file(filepath: str, content: str) -> None:
    """Write content to file."""
    Path(filepath).write_text(content)


def count_pending_todos(plan_content: str) -> int:
    """Count the number of pending TODO items in the plan."""
    # Match unchecked checkboxes with TODO
    pattern = r"- \[ \] TODO:"
    return len(re.findall(pattern, plan_content))


def count_total_todos(plan_content: str) -> int:
    """Count total TODO items (pending and completed)."""
    pattern = r"- \[[ x]\] TODO:"
    return len(re.findall(pattern, plan_content))


def is_plan_complete(plan_content: str) -> bool:
    """Check if all TODOs in the plan are marked as done."""
    pending = count_pending_todos(plan_content)
    return pending == 0


def generate_initial_plan(provider: AIProvider, requirements: str, working_dir: Path, yolo: bool = False) -> str:
    """Generate the initial plan.md from requirements."""
    prompt = f"""You are a software architect. Read the following requirements and create a detailed implementation plan.

{TODO_FORMAT}

Requirements:
---
{requirements}
---

Create a plan.md file with:
1. An overview section summarizing the project
2. A list of TODO items following the exact format specified above
3. Each TODO should be atomic and independently verifiable
4. Include verification steps (tests, manual checks, etc.) for each TODO
5. Order TODOs by dependency (foundational items first)

Output ONLY the contents of plan.md, no additional commentary."""

    return call_ai(provider, prompt, working_dir, yolo=yolo)


def execute_single_todo(provider: AIProvider, plan_content: str, requirements: str, working_dir: Path, yolo: bool = False) -> str:
    """Ask the AI to execute one TODO item and update the plan."""
    prompt = f"""You are a software developer. Here is the current project plan:

---
{plan_content}
---

And the original requirements:
---
{requirements}
---

Your task:
1. Find the FIRST uncompleted TODO item (marked with `- [ ]`)
2. Implement that task completely
3. Add verification (write a test, add a check, etc.)
4. Run the verification to confirm the task is complete
5. Update the plan.md to mark the TODO as done (change `- [ ]` to `- [x]`) and update Status to DONE or VERIFIED

Important:
- Only work on ONE TODO at a time
- Make sure to actually create/modify files as needed
- Include verification in your implementation
- Update the Status field in the TODO item

After completing the task, output the UPDATED plan.md content.
Start your response with "UPDATED_PLAN:" followed by the complete updated plan.md content."""

    response = call_ai(provider, prompt, working_dir, yolo=yolo, timeout=600)

    # Extract the updated plan from the response
    if "UPDATED_PLAN:" in response:
        updated_plan = response.split("UPDATED_PLAN:", 1)[1].strip()
        return updated_plan

    # If format not followed, return original plan
    print("Warning: AI did not follow expected output format. Plan may not be updated.")
    return plan_content


def validate_against_requirements(provider: AIProvider, plan_content: str, requirements: str, working_dir: Path, yolo: bool = False) -> tuple[bool, str]:
    """Check if the completed plan meets all requirements. Returns (is_complete, updated_plan)."""
    prompt = f"""You are a software architect reviewing completed work.

Original Requirements:
---
{requirements}
---

Current Plan (with completed TODOs):
---
{plan_content}
---

Your task:
1. Carefully compare the completed work against ALL original requirements
2. Identify any requirements that are NOT fully addressed
3. If there are gaps, add NEW TODO items to address them
4. Use the same TODO format as the existing items

{TODO_FORMAT}

Respond in this format:
ANALYSIS: <brief analysis of coverage>
GAPS_FOUND: YES or NO
UPDATED_PLAN:
<complete updated plan.md content, with any new TODOs added at the end>"""

    response = call_ai(provider, prompt, working_dir, yolo=yolo, timeout=300)

    # Parse the response
    gaps_found = "GAPS_FOUND: YES" in response.upper()

    if "UPDATED_PLAN:" in response:
        updated_plan = response.split("UPDATED_PLAN:", 1)[1].strip()
        return (not gaps_found, updated_plan)

    return (not gaps_found, plan_content)


def main():
    parser = argparse.ArgumentParser(
        description="AI Orchestrator - Iterative AI-driven development workflow"
    )
    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        choices=["claude", "codex", "gemini"],
        default="claude",
        help="AI provider to use (default: claude)",
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=str,
        default=".",
        help="Target project directory (default: current directory)",
    )
    parser.add_argument(
        "--requirements",
        "-r",
        type=str,
        default="requirements.md",
        help="Requirements filename (default: requirements.md)",
    )
    parser.add_argument(
        "--plan",
        type=str,
        default="plan.md",
        help="Plan filename (default: plan.md)",
    )
    parser.add_argument(
        "--max-iterations",
        "-m",
        type=int,
        default=50,
        help="Maximum number of task iterations (default: 50)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Skip permission prompts (maps to provider-specific flags)",
    )

    args = parser.parse_args()
    provider = AIProvider(args.provider)

    # Resolve and validate target directory
    target_dir = Path(args.dir).resolve()
    if not target_dir.exists():
        print(f"Error: Directory '{target_dir}' does not exist.")
        sys.exit(1)
    if not target_dir.is_dir():
        print(f"Error: '{target_dir}' is not a directory.")
        sys.exit(1)

    # Build full paths for requirements and plan files
    requirements_path = target_dir / args.requirements
    plan_path = target_dir / args.plan

    print(f"AI Orchestrator - Using {provider.value}")
    print(f"Target directory: {target_dir}")
    if args.yolo:
        print("YOLO mode: Permission prompts disabled")
    print("=" * 50)

    # Step 1: Read requirements
    requirements = read_file(requirements_path)
    if not requirements:
        print(f"Error: {requirements_path} not found.")
        print("Please create a requirements.md file with your project requirements.")
        sys.exit(1)

    print(f"✓ Loaded requirements from {args.requirements}")

    # Step 2: Generate or load plan
    plan_content = read_file(plan_path)
    if plan_content:
        print(f"✓ Found existing {args.plan}")
    else:
        print(f"Generating initial plan...")
        if args.dry_run:
            print("[DRY RUN] Would generate plan.md")
            return
        plan_content = generate_initial_plan(provider, requirements, target_dir, yolo=args.yolo)
        write_file(plan_path, plan_content)
        print(f"✓ Generated {args.plan}")

    # Main loop
    iteration = 0
    validation_cycles = 0
    max_validation_cycles = 3  # Prevent infinite validation loops

    while iteration < args.max_iterations:
        iteration += 1
        print(f"\n{'='*50}")
        print(f"Iteration {iteration}")
        print(f"{'='*50}")

        # Read current plan
        plan_content = read_file(plan_path)
        pending_todos = count_pending_todos(plan_content)
        total_todos = count_total_todos(plan_content)

        print(f"TODOs: {total_todos - pending_todos}/{total_todos} completed")

        # Step 3-4: Check if plan has pending TODOs
        if pending_todos > 0:
            print(f"\nExecuting next TODO...")
            if args.dry_run:
                print("[DRY RUN] Would execute next TODO")
                break

            # Step 3: Execute one TODO
            updated_plan = execute_single_todo(provider, plan_content, requirements, target_dir, yolo=args.yolo)
            write_file(plan_path, updated_plan)
            print(f"✓ Task completed and plan updated")

            # Reset validation cycles when doing tasks
            validation_cycles = 0
            continue

        # Step 5: All TODOs done - validate against requirements
        print("\nAll TODOs completed. Validating against requirements...")

        if args.dry_run:
            print("[DRY RUN] Would validate against requirements")
            break

        todos_before = count_total_todos(plan_content)
        is_complete, updated_plan = validate_against_requirements(provider, plan_content, requirements, target_dir, yolo=args.yolo)
        todos_after = count_total_todos(updated_plan)

        write_file(plan_path, updated_plan)

        new_todos_added = todos_after > todos_before

        if new_todos_added:
            print(f"✓ Added {todos_after - todos_before} new TODO(s) to address gaps")
            validation_cycles += 1

            if validation_cycles >= max_validation_cycles:
                print(f"\nWarning: Reached maximum validation cycles ({max_validation_cycles}).")
                print("There may be requirements that cannot be automatically addressed.")
                break

            # Step 6: Go back to step 3
            continue

        # Step 7: No new TODOs added - we're done!
        print("\n" + "=" * 50)
        print("✓ ALL REQUIREMENTS MET!")
        print("=" * 50)
        print(f"\nCompleted {total_todos} tasks in {iteration} iterations.")
        print(f"Final plan saved to: {plan_path}")
        break

    else:
        print(f"\nWarning: Reached maximum iterations ({args.max_iterations}).")
        print("Some tasks may still be pending.")

    # Final summary
    plan_content = read_file(plan_path)
    pending = count_pending_todos(plan_content)
    total = count_total_todos(plan_content)

    print(f"\nFinal Status: {total - pending}/{total} TODOs completed")
    if pending > 0:
        print(f"Remaining: {pending} pending TODO(s)")


if __name__ == "__main__":
    main()
