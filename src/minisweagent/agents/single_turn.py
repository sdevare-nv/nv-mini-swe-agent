"""Single-turn agent for diff generation without execution loop."""

import os
import platform
import re
from dataclasses import asdict, dataclass
from typing import Any

from jinja2 import Template

from minisweagent import Model


@dataclass
class SingleTurnAgentConfig:
    system_template: str = "You are a helpful assistant that generates code patches."
    instance_template: str = (
        "Generate a patch to solve the following issue:\n\n{{task}}\n\n"
        "Please provide your patch in a code block using ```diff format."
    )
    cost_limit: float = 3.0


class LimitsExceeded(Exception):
    """Raised when the agent has reached its cost limit."""


class SingleTurnAgent:
    """Agent that performs single-turn inference to generate a diff patch.
    
    Unlike the default agent, this agent does not execute commands in a loop.
    It simply queries the model once to generate a patch, which is then
    evaluated separately by applying it to a container and running tests.
    """

    def __init__(
        self,
        model: Model,
        env=None,
        responses_create_params: dict[str, Any] | None = None,
        *,
        config_class: type = SingleTurnAgentConfig,
        **kwargs,
    ):
        self.config = config_class(**kwargs)
        self.model = model
        self.env = env  # Only used during eval, not during generation
        self.responses_create_params = responses_create_params or {}
        self.messages: list[dict] = []
        self.responses: list[dict] = []

    def render_template(self, template: str, **kwargs) -> str:
        cs = asdict(self.config) | asdict(self.model.config) | platform.uname()._asdict()
        return Template(template).render(**kwargs, **cs, **os.environ)

    def run(self, task: str) -> tuple[str, str]:
        """Run single-turn inference. Return exit status & generated diff."""
        self.messages = []
        self.responses = []

        # Check if custom messages are provided
        if "input" in self.responses_create_params and len(self.responses_create_params["input"]) > 0:
            for message in self.responses_create_params["input"]:
                self.messages.append({"role": message["role"], "content": message["content"]})
        else:
            # Standard system + user message
            self.messages.append({"role": "system", "content": self.render_template(self.config.system_template)})
            self.messages.append({"role": "user", "content": self.render_template(self.config.instance_template, task=task)})

        # Check cost limits before querying
        if 0 < self.config.cost_limit <= self.model.cost:
            return "LimitsExceeded", ""

        # Query model once
        kwargs = {
            key: self.responses_create_params[key]
            for key in ["temperature", "top_p"]
            if key in self.responses_create_params
        }
        
        response = self.model.query(self.messages, self.responses, **kwargs)

        if not response["content"]:
            return "LimitsExceeded", ""

        if "response_obj" in response:
            self.responses.append(response["response_obj"])
        diff = self._extract_diff(response["content"])

        return "Submitted", diff

    def _extract_diff(self, content: str) -> str:
        """Extract diff/patch from model output.
        
        Supports multiple formats:
        1. <solution> tags with SEARCH/REPLACE edits
        2. diff code blocks
        3. Plain code blocks
        4. Raw content
        """
        # Try to find content in <solution> tags first
        solution_match = re.search(r"<solution>(.*?)</solution>", content, re.DOTALL)
        if solution_match:
            return solution_match.group(1).strip()
        
        # Try to find diff in code blocks (```diff, ```python, or just ```)
        diffs = re.findall(r"```(?:diff|python)?\s*\n(.*?)```", content, re.DOTALL)
        if diffs:
            # If multiple blocks, join them
            return "\n\n".join(d.strip() for d in diffs)
        
        # Otherwise return the entire content
        return content.strip()

