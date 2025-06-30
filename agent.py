import asyncio
import re
from typing import Any

import litellm
from jinja2 import Template
from pydantic import BaseModel
from swerex.deployment.docker import DockerDeployment
from swerex.exceptions import CommandTimeoutError
from swerex.runtime.abstract import Command as RexCommand
from tenacity import retry, stop_after_attempt, wait_exponential


class AgentConfig(BaseModel):
    system_template: str
    instance_template: str
    step_limit: int = 0
    cost_limit: float = 3.0
    model_name: str
    model_kwargs: dict[str, Any] = {}


class Model:
    def __init__(self, model_name: str, model_kwargs: dict[str, Any]):
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.cost = 0.0

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
    def query(self, messages: list[dict[str, str]]) -> str:
        response: litellm.types.utils.ModelResponse = litellm.completion(  # type: ignore
            model=self.model_name, messages=messages, **self.model_kwargs
        )
        self.cost += litellm.cost_calculator.completion_cost(response)
        return response.choices[0].message.content  # type: ignore


class Environment:
    def __init__(self, image: str):
        """This class executes bash commands in a Docker container for sandboxing."""
        self.deployment = DockerDeployment(image=image)
        asyncio.run(self.deployment.start())

    def execute(self, command: str, cwd: str = "/testbed") -> str:
        """Execute a command in the environment.
        No persistent shell is used, so the command must be self-contained.
        No exceptions will be raised, the output is always a string.
        """
        try:
            output = asyncio.run(
                self.deployment.runtime.execute(RexCommand(command=command, shell=True, check=False, cwd=cwd))
            )
            return f"stdout: {output.stdout}\nstderr: {output.stderr}\nexit_code: {output.exit_code}"
        except CommandTimeoutError:
            return "The command timed out. Please change your command and make sure it doesn't require input."
        except Exception as e:
            return f"Error executing action: {e}"


class Agent:
    def __init__(self, config: AgentConfig, env: Environment, problem_statement: str):
        self.config = config
        self.problem_statement = problem_statement
        instance_message = Template(config.instance_template).render(problem_statement=problem_statement)
        print(config.system_template)
        print(instance_message)
        self.history = [
            {"role": "system", "content": config.system_template},
            {"role": "user", "content": instance_message},
        ]
        self.model = Model(config.model_name, config.model_kwargs)
        self.env = env

    @property
    def n_steps(self) -> int:
        return len(self.history) // 2

    def run(self) -> str:
        """Run the agent and return the final observation.
        Essentially just calls `step` until the agent is finished.
        """
        is_finished = False
        while not is_finished:
            is_finished, _, observation = self.step()
        return observation

    def step(self) -> tuple[bool, str, str]:
        """Query the LM and execute the action

        Returns:
            is_finished: whether the agent has finished its task
            response: Model reponse (if any)
            observation: The observation or error message
        """
        if self.n_steps >= self.config.step_limit:
            return True, "", "step_limit_exceeded"
        if self.model.cost >= self.config.cost_limit:
            return True, "", "cost_limit_exceeded"
        message = self.model.query(self.history)
        assert isinstance(message, str)
        print(message)
        self.history.append({"role": "assistant", "content": message})
        action = self.parse_action(message)
        is_done = False
        if action:
            observation = self.execute_action(action)
            if action == "submit":
                is_done = True
        else:
            observation = "Please always provide exactly one action in triple backticks."
        self.history.append({"role": "user", "content": observation})
        print(observation)
        return is_done, message, observation

    @staticmethod
    def parse_action(message: str) -> str:
        """Parse the action from the message
        (assumes action is the first triple backticks block)
        """
        match = re.search(r"```[a-zA-Z]*\n(.*?)(?=\n```|```)", message, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def execute_action(self, action: str) -> str:
        """Execute the action and return the observation"""
        if action == "submit":
            return self.env.execute("git add -A && git diff --cached")
        return self.env.execute(action).strip() or "The command returned no output."
