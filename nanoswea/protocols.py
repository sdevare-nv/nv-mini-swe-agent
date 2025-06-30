from typing import Protocol


class Model(Protocol):
    """Protocol for language models."""

    cost: float

    def query(self, messages: list[dict[str, str]]) -> str:
        """Query the model with a list of messages and return the response."""
        ...


class Environment(Protocol):
    """Protocol for execution environments."""

    def execute(self, command: str, cwd: str = "/testbed") -> str:
        """Execute a command in the environment and return the raw output."""
        ...
