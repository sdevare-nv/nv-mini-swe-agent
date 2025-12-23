"""Terminal Bench agent with JSON keystrokes parsing (terminus_2 style)."""

import json
import re
import subprocess
from dataclasses import dataclass

from minisweagent.agents.default import (
    AgentConfig,
    DefaultAgent,
    ExecutionTimeoutError,
    FormatError,
    Submitted,
)


@dataclass
class TerminalBenchAgentConfig(AgentConfig):
    completion_confirmation_template: str = (
        "Current terminal state:\n{{terminal_state}}\n\n"
        "Are you sure you want to mark the task as complete? "
        'If so, include "task_complete": true in your JSON response again.'
    )


@dataclass
class ParsedCommand:
    keystrokes: str
    duration: float


@dataclass
class ParseResult:
    commands: list[ParsedCommand]
    is_task_complete: bool
    error: str | None
    warning: str | None


class TerminalBenchAgent(DefaultAgent):
    """Agent for Terminal Bench with JSON keystrokes parsing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, config_class=TerminalBenchAgentConfig, **kwargs)
        self._pending_completion = False

    def parse_action(self, response: dict) -> dict:
        content = response["content"]
        result = self._parse_json_response(content)

        if result.error:
            raise FormatError(self.render_template(self.config.format_error_template, error_message=result.error))

        if result.warning:
            print(f"Parser warning: {result.warning}")

        if result.is_task_complete:
            if self._pending_completion:
                raise Submitted("Task marked as complete by agent.")
            self._pending_completion = True  # Set to true to avoid submitting multiple times.
        else:
            self._pending_completion = False

        # Note: For collapse detection, use combined keystrokes
        combined_action = "".join(cmd.keystrokes for cmd in result.commands)
        self.check_collapse(combined_action)

        return {
            "action": combined_action,
            "commands": result.commands,
            "is_confirmation": result.is_task_complete,
            **response,
        }

    def _parse_json_response(self, content: str) -> ParseResult:
        """Parse a JSON response containing commands with keystrokes."""
        content = content.strip()
        warnings = []

        # Try to extract JSON from markdown code block if present
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
        if json_match:
            content = json_match.group(1).strip()
            warnings.append("JSON was wrapped in markdown code block")

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            return ParseResult(commands=[], is_task_complete=False, error=f"Invalid JSON: {e}", warning=None)

        if not isinstance(data, dict):
            return ParseResult(
                commands=[],
                is_task_complete=False,
                error="Response must be a JSON object",
                warning=None,
            )

        commands = []
        raw_commands = data.get("commands", [])

        if not isinstance(raw_commands, list):
            return ParseResult(
                commands=[],
                is_task_complete=False,
                error='"commands" must be an array',
                warning=None,
            )

        # Note: the duration value is not used for the execution of the commands.
        for i, cmd in enumerate(raw_commands):
            if not isinstance(cmd, dict):
                warnings.append(f"Command {i} is not an object, skipping")
                continue

            keystrokes = cmd.get("keystrokes", "")
            if not isinstance(keystrokes, str):
                warnings.append(f"Command {i} keystrokes is not a string, skipping")
                continue

            duration = cmd.get("duration", 1.0)
            if not isinstance(duration, (int, float)):
                duration = 1.0
                warnings.append(f"Command {i} duration is not a number, using default 1.0")

            commands.append(ParsedCommand(keystrokes=keystrokes, duration=min(float(duration), 60.0)))

        is_task_complete = data.get("task_complete", False)
        if not isinstance(is_task_complete, bool):
            is_task_complete = str(is_task_complete).lower() == "true"

        return ParseResult(
            commands=commands,
            is_task_complete=is_task_complete,
            error=None,
            warning="; ".join(warnings) if warnings else None,
        )

    def execute_action(self, action: dict) -> dict:
        """Execute multiple commands sequentially."""
        commands: list[ParsedCommand] = action.get("commands", [])
        all_outputs = []

        for cmd in commands:
            command_str = cmd.keystrokes.rstrip("\n")

            if not command_str:
                continue

            try:
                output = self.env.execute(command_str)
                if output.get("output"):
                    all_outputs.append(output["output"])
            except subprocess.TimeoutExpired as e:
                output_text = e.output.decode("utf-8", errors="replace") if e.output else ""
                raise ExecutionTimeoutError(
                    # TODO: This uses the default timeout set for per step. We should use the duration of the command instead.
                    self.render_template(
                        self.config.timeout_template,
                        action={"action": command_str},
                        output=output_text,
                        timeout_sec=cmd.duration,
                    )
                )

        combined_output = "\n".join(all_outputs) if all_outputs else ""

        if action.get("is_confirmation"):
            confirmation_msg = self.render_template(
                self.config.completion_confirmation_template,
                terminal_state=combined_output,
            )
            combined_output = confirmation_msg

        return {"output": combined_output}
