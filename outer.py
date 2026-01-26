#!/usr/bin/env python3
"""
Outer - An iterative AI-driven development workflow tool.

This script:
1. Reads requirements.md from the current directory
2. Asks an AI (Claude/Codex/Gemini/Opencode) to generate a plan.md with TODOs
3. Iteratively executes tasks, updating plan.md
4. Validates completed work against requirements
5. Continues until all requirements are met

Usage:
python outer.py -d /path/to/project -p claude --yolo -m 100
"""

import argparse
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# Session state for tracking prompts
_session_dir: Optional[Path] = None
_step_counter: int = 0
_iteration_start_time: Optional[float] = None
_session_start_time: Optional[float] = None


def timestamp() -> str:
    """Return current timestamp in HH:MM:SS format."""
    return datetime.now().strftime("%H:%M:%S")


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


class ProgressIndicator:
    """Show elapsed time while a task is running."""

    SPINNER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, description: str = "Running"):
        self.description = description
        self.start_time = None
        self._stop_event = threading.Event()
        self._thread = None
        self._tqdm_bar = None

    def _spinner_loop(self):
        """Background thread that updates the spinner."""
        spinner_idx = 0
        while not self._stop_event.is_set():
            elapsed = time.time() - self.start_time
            spinner = self.SPINNER_CHARS[spinner_idx % len(self.SPINNER_CHARS)]
            # Clear line and print status
            print(f"\r{spinner} {self.description} [{format_duration(elapsed)}]", end="", flush=True)
            spinner_idx += 1
            self._stop_event.wait(0.1)  # Update every 100ms

    def start(self):
        """Start the progress indicator."""
        self.start_time = time.time()

        if TQDM_AVAILABLE:
            # Use tqdm with a custom format showing elapsed time
            self._tqdm_bar = tqdm(
                total=None,
                desc=self.description,
                bar_format="{desc}: {elapsed}",
                leave=False,
            )
        else:
            # Fall back to simple spinner
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._spinner_loop, daemon=True)
            self._thread.start()

    def stop(self) -> float:
        """Stop the progress indicator and return elapsed time."""
        elapsed = time.time() - self.start_time if self.start_time else 0

        if TQDM_AVAILABLE and self._tqdm_bar:
            self._tqdm_bar.close()
            self._tqdm_bar = None
        else:
            self._stop_event.set()
            if self._thread:
                self._thread.join(timeout=0.5)
            # Clear the spinner line
            print("\r" + " " * 80 + "\r", end="", flush=True)

        return elapsed


def init_session() -> Path:
    """Initialize a new session directory for storing prompts."""
    global _session_dir, _step_counter, _session_start_time
    _session_dir = Path(f"/tmp/outer/{uuid.uuid4()}")
    _session_dir.mkdir(parents=True, exist_ok=True)
    _step_counter = 0
    _session_start_time = time.time()
    print(f"[{timestamp()}] Session directory: {_session_dir}")
    return _session_dir


def get_next_step_number() -> int:
    """Get the next step number and increment the counter."""
    global _step_counter
    _step_counter += 1
    return _step_counter


def get_step_prompt_file(step: int) -> Path:
    """Get the path for a step's prompt file."""
    return _session_dir / f"step_{step}.txt"


def get_step_output_files(step: int, attempt: int = 0) -> tuple[Path, Path, Path]:
    """Get the paths for a step's stdout, stderr, and exit code files.

    If attempt > 0, includes the attempt number in the filename to preserve
    output from retry attempts.
    """
    suffix = f"_attempt{attempt}" if attempt > 0 else ""
    return (
        _session_dir / f"step_{step}{suffix}.stdout",
        _session_dir / f"step_{step}{suffix}.stderr",
        _session_dir / f"step_{step}{suffix}.exitcode",
    )


def save_step_result(step: int, stdout: str, stderr: str, exit_code: int) -> None:
    """Save the stdout, stderr, and exit code for a step."""
    stdout_file = _session_dir / f"step_{step}.stdout"
    stderr_file = _session_dir / f"step_{step}.stderr"
    exitcode_file = _session_dir / f"step_{step}.exitcode"
    stdout_file.write_text(stdout)
    stderr_file.write_text(stderr)
    exitcode_file.write_text(str(exit_code))
    print(f"[{timestamp()}] Step {step} results saved to: {_session_dir}/step_{step}.{{stdout,stderr,exitcode}}")


class AIProvider(Enum):
    CLAUDE = "claude"
    CODEX = "codex"
    GEMINI = "gemini"
    OPENCODE = "opencode"


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


def get_ai_command(provider: AIProvider, prompt_file: str, yolo: bool = False, prefix: str = "") -> str:
    """Get a display-friendly command string that references the prompt file."""
    base_commands = {
        AIProvider.CLAUDE: "claude",
        AIProvider.CODEX: "codex",
        AIProvider.GEMINI: "gemini",
        AIProvider.OPENCODE: "opencode",
    }

    yolo_flags = {
        AIProvider.CLAUDE: "--dangerously-skip-permissions",
        AIProvider.CODEX: "--yolo",
        AIProvider.GEMINI: "--yolo",
        AIProvider.OPENCODE: "",
    }

    # Command templates with prompt file substitution
    prompt_templates = {
        AIProvider.CLAUDE: '-p "$(cat {prompt_file})"',
        AIProvider.CODEX: 'exec --json --skip-git-repo-check "$(cat {prompt_file})"',
        AIProvider.GEMINI: '-p "$(cat {prompt_file})"',
        AIProvider.OPENCODE: 'run "$(cat {prompt_file})"',
    }

    parts = [base_commands[provider]]
    if prefix:
        parts.append(prefix)
    if yolo:
        parts.append(yolo_flags[provider])
    parts.append(prompt_templates[provider].format(prompt_file=shlex.quote(prompt_file)))

    return " ".join(parts)


def _get_ai_command_with_prompt(provider: AIProvider, prompt: str, yolo: bool = False, prefix: str = "") -> list[str]:
    """Get the command with the actual prompt embedded (for subprocess execution)."""
    base_commands = {
        AIProvider.CLAUDE: ["claude"],
        AIProvider.CODEX: ["codex"],
        AIProvider.GEMINI: ["gemini"],
        AIProvider.OPENCODE: ["opencode"],
    }

    yolo_flags = {
        AIProvider.CLAUDE: ["--dangerously-skip-permissions"],
        AIProvider.CODEX: ["--yolo"],
        AIProvider.GEMINI: ["--yolo"],
        AIProvider.OPENCODE: [],
    }

    prompt_flags = {
        AIProvider.CLAUDE: ["-p", prompt],
        AIProvider.CODEX: ["exec", "--skip-git-repo-check", prompt],
        AIProvider.GEMINI: ["-p", prompt],
        AIProvider.OPENCODE: ["run", prompt],
    }

    cmd = base_commands[provider].copy()
    if prefix:
        # Split the prefix string into individual arguments
        cmd.extend(shlex.split(prefix))
    if yolo:
        cmd.extend(yolo_flags[provider])
    cmd.extend(prompt_flags[provider])

    return cmd


def call_ai(providers: list[AIProvider], prompt: str, working_dir: Path, yolo: bool = False, timeout: int | None = None, max_retries: int = 3, current_todo: str | None = None, prefix: str = "") -> str:
    """Call the AI provider with the given prompt and return the response.

    Uses exponential backoff for retries on failure. If multiple providers are given,
    fails over to the next provider after exhausting retries for the current one.
    Output is written to files in real-time so they can be tailed.
    """
    # Write prompt to sequential file for reproducibility
    step = get_next_step_number()
    prompt_file = get_step_prompt_file(step)
    prompt_file.write_text(prompt)

    for provider_idx, provider in enumerate(providers):
        # Build the actual command with prompt inline (for subprocess)
        actual_cmd = _get_ai_command_with_prompt(provider, prompt, yolo=yolo, prefix=prefix)
        # Build display command referencing the file (for logging)
        display_cmd = get_ai_command(provider, str(prompt_file), yolo=yolo, prefix=prefix)
        is_last_provider = provider_idx == len(providers) - 1

        last_exception = None
        for attempt in range(max_retries):
            # Get output file paths with attempt number to preserve output from retries
            stdout_file, stderr_file, exitcode_file = get_step_output_files(step, attempt)

            try:
                # Print command for reproducibility (references prompt file)
                print(f"\n[{timestamp()}] [CMD] cd {shlex.quote(str(working_dir))} && {display_cmd}")

                # Print output file paths BEFORE starting so user can tail them
                print(f"[{timestamp()}] Output files (tail -f to monitor):")
                print(f"  stdout:   {stdout_file}")
                print(f"  stderr:   {stderr_file}")
                print(f"  exitcode: {exitcode_file}\n")

                # Start progress indicator
                progress = ProgressIndicator(f"Step {step}: {provider.value}")
                progress.start()

                # Use Popen with file redirection for real-time output
                with open(stdout_file, "w") as stdout_fh, open(stderr_file, "w") as stderr_fh:
                    process = subprocess.Popen(
                        actual_cmd,
                        stdout=stdout_fh,
                        stderr=stderr_fh,
                        text=True,
                        cwd=str(working_dir),
                    )

                    # Wait for process with optional timeout
                    try:
                        returncode = process.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                        # Stop progress and raise
                        progress.stop()
                        raise

                # Stop progress indicator
                elapsed = progress.stop()

                # Write exit code
                exitcode_file.write_text(str(returncode))

                # Read output back
                stdout_content = stdout_file.read_text()
                stderr_content = stderr_file.read_text()

                print(f"[{timestamp()}] Step {step} completed in {format_duration(elapsed)} (exit code: {returncode})")

                if returncode != 0:
                    print(f"Warning: AI command returned non-zero exit code: {returncode}")
                    print(f"Stderr: {stderr_content}")
                    # Treat non-zero exit code as a failure that can be retried
                    raise subprocess.CalledProcessError(returncode, actual_cmd, stdout_content, stderr_content)

                return stdout_content.strip()
            except subprocess.TimeoutExpired as e:
                last_exception = e
                # Write exit code as timeout indicator
                exitcode_file.write_text("timeout")
                if attempt < max_retries - 1:
                    wait_time = 10 * (2 ** attempt)  # Exponential backoff: 10s, 20s, 40s, ...
                    print(f"AI call timed out with {provider.value} (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    if current_todo:
                        print(f"\n[{timestamp()}] Retrying TODO: {current_todo}")
                else:
                    print(f"Error: AI call timed out after {max_retries} attempts with {provider.value}")
            except subprocess.CalledProcessError as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 10 * (2 ** attempt)  # Exponential backoff: 10s, 20s, 40s, ...
                    print(f"AI call failed with {provider.value} (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    if current_todo:
                        print(f"\n[{timestamp()}] Retrying TODO: {current_todo}")
                else:
                    print(f"Error: AI call failed after {max_retries} attempts with {provider.value}")
            except FileNotFoundError:
                print(f"Error: {provider.value} CLI not found. Make sure it's installed and in PATH.")
                if is_last_provider:
                    sys.exit(1)
                break  # Move to next provider

        # All retries exhausted for this provider
        if not is_last_provider:
            print(f"Failing over to next provider: {providers[provider_idx + 1].value}")
            if current_todo:
                print(f"\n[{timestamp()}] Retrying TODO: {current_todo}")
        else:
            print(f"Error: AI invocation failed after trying all providers. Stopping.")
            sys.exit(1)

    # Should not reach here, but just in case
    print("Error: No providers available.")
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


def is_valid_plan_format(plan_content: str) -> bool:
    """Check if the plan follows the expected TODO format.

    A valid plan has at least one TODO item with the expected structure.
    """
    # Check for at least one TODO item (completed or pending)
    todo_pattern = r"- \[[ x]\] TODO:"
    if not re.search(todo_pattern, plan_content):
        return False

    # Check for Status field (required in TODO format)
    status_pattern = r"- Status:\s*(PENDING|IN_PROGRESS|DONE|VERIFIED)"
    if not re.search(status_pattern, plan_content):
        return False

    return True


def generate_initial_plan(providers: list[AIProvider], working_dir: Path, plan_filename: str, requirements_filename: str, yolo: bool = False, prefix: str = "") -> None:
    """Generate the initial plan.md from requirements."""
    prompt = f"""You are a software architect. Read {requirements_filename} for the project requirements and create a detailed implementation plan.

{TODO_FORMAT}

Write a {plan_filename} file with:
1. An overview section summarizing the project
2. A list of TODO items following the exact format specified above
3. Each TODO should be atomic and independently verifiable
4. Include verification steps (tests, manual checks, etc.) for each TODO
5. Order TODOs by dependency (foundational items first)

Write the plan directly to {plan_filename}. Do not output the contents to stdout."""

    call_ai(providers, prompt, working_dir, yolo=yolo, prefix=prefix)


def get_first_pending_todo(plan_content: str) -> Optional[str]:
    """Extract the description of the first pending TODO item."""
    match = re.search(r"- \[ \] TODO:\s*(.+?)(?:\n|$)", plan_content)
    if match:
        return match.group(1).strip()
    return None


def execute_single_todo(providers: list[AIProvider], working_dir: Path, plan_filename: str, requirements_filename: str, current_todo: str | None = None, yolo: bool = False, prefix: str = "") -> None:
    """Ask the AI to execute one TODO item and update the plan directly on disk."""
    prompt = f"""You are a software developer. Read {plan_filename} for the project plan and {requirements_filename} for the original requirements.

Your task:
1. Read {plan_filename} to find the FIRST uncompleted TODO item (marked with `- [ ]`)
2. Implement that task completely
3. Run verification to confirm the task is complete
4. IMPORTANT: Update {plan_filename} directly to mark the TODO as done (change `- [ ]` to `- [x]`) and update Status to DONE or VERIFIED

CRITICAL REQUIREMENTS:
- You MUST complete at least one TODO before finishing. Do not exit without marking at least one TODO as done in {plan_filename}.
- If a TODO is too large or complex to complete in one session, you MUST:
  1. Edit {plan_filename} to break it down into smaller, more manageable sub-TODOs
  2. Complete at least ONE of those smaller TODOs
  3. Mark that smaller TODO as done in {plan_filename}
- Never end your session with zero TODOs completed. This is a hard requirement.
- You MUST edit {plan_filename} to mark the completed TODO. This is mandatory.

Important:
- Only work on ONE TODO at a time
- Make sure to actually create/modify files as needed
- Update the Status field in the TODO item
- Do NOT output the plan content - just update the file directly"""

    call_ai(providers, prompt, working_dir, yolo=yolo, current_todo=current_todo, prefix=prefix)


def force_breakdown_todo(providers: list[AIProvider], stuck_todo: str, working_dir: Path, plan_filename: str, yolo: bool = False, prefix: str = "") -> None:
    """Force the AI to break down a stuck TODO into smaller pieces."""
    prompt = f"""You are a software architect. The following TODO has been stuck and not making progress:

STUCK TODO: {stuck_todo}

Read {plan_filename} for the current plan.

Your task:
1. This TODO is too large. You MUST break it down into 2-4 smaller, atomic sub-TODOs.
2. Edit {plan_filename} directly to:
   - Remove or comment out the original stuck TODO
   - Add the new smaller TODOs in its place, following the same format
   - Each sub-TODO must be small enough to complete in one session
3. Then complete the FIRST of the new smaller TODOs
4. Mark that completed TODO as done in {plan_filename}

CRITICAL: You MUST edit {plan_filename} and mark at least one TODO as completed before finishing.
Do NOT output the plan content - just update the file directly."""

    call_ai(providers, prompt, working_dir, yolo=yolo, current_todo=stuck_todo, prefix=prefix)


def validate_against_requirements(providers: list[AIProvider], plan_content: str, working_dir: Path, plan_filename: str, requirements_filename: str, yolo: bool = False, prefix: str = "") -> tuple[bool, str]:
    """Check if the completed plan meets all requirements. Returns (is_complete, updated_plan)."""
    prompt = f"""You are a software architect reviewing completed work.

Read {requirements_filename} for the original requirements and {plan_filename} for the current plan with completed TODOs.

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

    response = call_ai(providers, prompt, working_dir, yolo=yolo, prefix=prefix)

    # Parse the response
    gaps_found = "GAPS_FOUND: YES" in response.upper()

    if "UPDATED_PLAN:" in response:
        updated_plan = response.split("UPDATED_PLAN:", 1)[1].strip()
        return (not gaps_found, updated_plan)

    return (not gaps_found, plan_content)


def main():
    parser = argparse.ArgumentParser(
        description="Outer - Iterative AI-driven development workflow"
    )
    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        default="claude",
        help="AI provider(s) to use, comma-separated for failover (e.g., 'claude,codex,gemini'). Default: claude",
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
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Custom prefix to add after the provider command (e.g., '--model opus')",
    )

    args = parser.parse_args()

    # Parse and validate providers
    provider_names = [p.strip().lower() for p in args.provider.split(",")]
    valid_providers = {"claude", "codex", "gemini", "opencode"}
    providers = []
    for name in provider_names:
        if name not in valid_providers:
            print(f"Error: Invalid provider '{name}'. Valid options: claude, codex, gemini, opencode")
            sys.exit(1)
        providers.append(AIProvider(name))

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

    provider_list = ", ".join(p.value for p in providers)
    print(f"Outer - Using provider(s): {provider_list}")
    print(f"Target directory: {target_dir}")
    if args.yolo:
        print("YOLO mode: Permission prompts disabled")
    print("=" * 50)

    # Initialize session for prompt logging
    init_session()

    # Step 1: Check requirements file exists
    if not requirements_path.exists():
        print(f"Error: {requirements_path} not found.")
        print("Please create a requirements.md file with your project requirements.")
        sys.exit(1)

    print(f"✓ Found requirements at {args.requirements}")

    # Step 2: Generate or load plan
    plan_content = read_file(plan_path)
    if plan_content:
        if is_valid_plan_format(plan_content):
            pending_todos = count_pending_todos(plan_content)
            total_todos = count_total_todos(plan_content)
            completed_todos = total_todos - pending_todos
            print(f"✓ Found existing {args.plan} in valid TODO format")
            print(f"  Resuming: {completed_todos}/{total_todos} TODOs completed, {pending_todos} remaining")
        else:
            print(f"Warning: Existing {args.plan} is not in expected TODO format.")
            print("Regenerating plan...")
            if args.dry_run:
                print("[DRY RUN] Would regenerate plan.md")
                return
            generate_initial_plan(providers, target_dir, args.plan, args.requirements, yolo=args.yolo, prefix=args.prefix)
            print(f"✓ Regenerated {args.plan}")
    else:
        print(f"Generating initial plan...")
        if args.dry_run:
            print("[DRY RUN] Would generate plan.md")
            return
        generate_initial_plan(providers, target_dir, args.plan, args.requirements, yolo=args.yolo, prefix=args.prefix)
        print(f"✓ Generated {args.plan}")

    # Main loop
    iteration = 0
    validation_cycles = 0
    max_validation_cycles = 3  # Prevent infinite validation loops
    stall_count = 0  # Track consecutive iterations with no progress
    max_stall_count = 3  # After this many stalls, force task breakdown
    last_pending_count = -1  # Track pending count to detect stalls

    while iteration < args.max_iterations:
        iteration += 1
        iteration_start = time.time()
        print(f"\n{'='*50}")
        print(f"[{timestamp()}] Iteration {iteration}")
        print(f"{'='*50}")

        # Read current plan from disk
        plan_content = read_file(plan_path)

        pending_todos = count_pending_todos(plan_content)
        total_todos = count_total_todos(plan_content)

        print(f"TODOs: {total_todos - pending_todos}/{total_todos} completed")

        # Detect stalls (same pending count as last iteration)
        if pending_todos > 0 and pending_todos == last_pending_count:
            stall_count += 1
            print(f"[{timestamp()}] ⚠ No progress detected (stall count: {stall_count}/{max_stall_count})")
        else:
            stall_count = 0  # Reset on progress
        last_pending_count = pending_todos

        # Step 3-4: Check if plan has pending TODOs
        if pending_todos > 0:
            if args.dry_run:
                print("[DRY RUN] Would execute next TODO")
                break

            # If we've stalled too many times, force a breakdown
            if stall_count >= max_stall_count:
                stuck_todo = get_first_pending_todo(plan_content)
                print(f"\n[{timestamp()}] ⚠ Task appears stuck. Forcing breakdown of: {stuck_todo}")
                force_breakdown_todo(providers, stuck_todo, target_dir, args.plan, yolo=args.yolo, prefix=args.prefix)
                stall_count = 0  # Reset after breakdown attempt
                elapsed = time.time() - iteration_start
                print(f"[{timestamp()}] ✓ Task breakdown attempted (iteration took {format_duration(elapsed)})")
            else:
                next_todo = get_first_pending_todo(plan_content)
                print(f"\n[{timestamp()}] Executing TODO: {next_todo}")
                execute_single_todo(providers, target_dir, args.plan, args.requirements, current_todo=next_todo, yolo=args.yolo, prefix=args.prefix)
                elapsed = time.time() - iteration_start
                print(f"[{timestamp()}] ✓ Task execution completed (iteration took {format_duration(elapsed)})")

            # Read the plan from disk to check for updates (AI should have modified it)
            new_plan_content = read_file(plan_path)
            new_pending = count_pending_todos(new_plan_content)

            if new_pending < pending_todos:
                print(f"[{timestamp()}] ✓ Progress: {pending_todos - new_pending} TODO(s) completed")
            elif new_pending > pending_todos:
                print(f"[{timestamp()}] ✓ Task broken down: {new_pending - pending_todos} new TODO(s) added")

            # Reset validation cycles when doing tasks
            validation_cycles = 0
            continue

        # Step 5: All TODOs done - validate against requirements
        print(f"\n[{timestamp()}] All TODOs completed. Validating against requirements...")

        if args.dry_run:
            print("[DRY RUN] Would validate against requirements")
            break

        todos_before = count_total_todos(plan_content)
        is_complete, updated_plan = validate_against_requirements(providers, plan_content, target_dir, args.plan, args.requirements, yolo=args.yolo, prefix=args.prefix)
        todos_after = count_total_todos(updated_plan)

        write_file(plan_path, updated_plan)

        new_todos_added = todos_after > todos_before

        if new_todos_added:
            elapsed = time.time() - iteration_start
            print(f"[{timestamp()}] ✓ Added {todos_after - todos_before} new TODO(s) to address gaps (iteration took {format_duration(elapsed)})")
            validation_cycles += 1

            if validation_cycles >= max_validation_cycles:
                print(f"\nWarning: Reached maximum validation cycles ({max_validation_cycles}).")
                print("There may be requirements that cannot be automatically addressed.")
                break

            # Step 6: Go back to step 3
            continue

        # Step 7: No new TODOs added - we're done!
        elapsed = time.time() - iteration_start
        print(f"\n[{timestamp()}] " + "=" * 50)
        print(f"[{timestamp()}] ✓ ALL REQUIREMENTS MET!")
        print("=" * 50)
        print(f"\nCompleted {total_todos} tasks in {iteration} iterations.")
        print(f"Final iteration took {format_duration(elapsed)}")
        print(f"Final plan saved to: {plan_path}")
        break

    else:
        print(f"\nWarning: Reached maximum iterations ({args.max_iterations}).")
        print("Some tasks may still be pending.")

    # Final summary
    plan_content = read_file(plan_path)
    pending = count_pending_todos(plan_content)
    total = count_total_todos(plan_content)

    total_elapsed = time.time() - _session_start_time if _session_start_time else 0
    print(f"\n[{timestamp()}] Final Status: {total - pending}/{total} TODOs completed")
    print(f"[{timestamp()}] Total session time: {format_duration(total_elapsed)}")
    if pending > 0:
        print(f"Remaining: {pending} pending TODO(s)")


if __name__ == "__main__":
    main()
