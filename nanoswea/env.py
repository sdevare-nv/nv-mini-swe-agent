import asyncio
import subprocess
from typing import Protocol

from swerex.deployment.docker import DockerDeployment
from swerex.runtime.abstract import Command as RexCommand


class Environment(Protocol):
    def execute(self, command: str, cwd: str = "/testbed") -> str:
        """Execute a command in the environment and return the raw output."""
        ...


class DockerEnvironment:
    def __init__(self, image: str):
        """This class executes bash commands in a Docker container for sandboxing."""
        self.deployment = DockerDeployment(image=image)
        asyncio.run(self.deployment.start())

    def execute(self, command: str, cwd: str = "/testbed") -> str:
        """Execute a command in the environment and return the raw output."""
        output = asyncio.run(
            self.deployment.runtime.execute(RexCommand(command=command, shell=True, check=False, cwd=cwd))
        )
        return f"stdout: {output.stdout}\nstderr: {output.stderr}\nexit_code: {output.exit_code}"


class LocalEnvironment:
    def __init__(self):
        """This class executes bash commands directly on the local machine."""
        pass

    def execute(self, command: str, cwd: str = "/testbed") -> str:
        """Execute a command in the local environment and return the raw output."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=30,
            )
            return f"stdout: {result.stdout}\nstderr: {result.stderr}\nexit_code: {result.returncode}"
        except subprocess.TimeoutExpired as e:
            raise TimeoutError from e
