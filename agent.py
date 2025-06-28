import asyncio
import re
from typing import Any

import litellm
from jinja2 import Template
from pydantic import BaseModel, ConfigDict
from swerex.deployment.docker import DockerDeployment
from swerex.runtime.abstract import Command as RexCommand


class AgentConfig(BaseModel):
    system_template: str
    instance_template: str
    step_limit: int = 0
    cost_limit: float = 3.0
    model_name: str
    model_kwargs: dict[str, Any] = {}
    model_config = ConfigDict(extra="forbid")


class InstanceConfig(BaseModel):
    image: str
    problem_statement: str
    model_config = ConfigDict(extra="forbid")


class Model:
    def __init__(self, model_name: str, model_kwargs: dict[str, Any]):
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.cost = 0.0

    def query(self, messages: list[dict[str, str]]) -> str:
        response: litellm.types.utils.ModelResponse = litellm.completion(  # type: ignore
            model=self.model_name, messages=messages, **self.model_kwargs
        )
        self.cost += litellm.cost_calculator.completion_cost(response)
        return response.choices[0].message.content  # type: ignore


class Environment:
    def __init__(self, image: str):
        self.deployment = DockerDeployment(image=image)
        asyncio.run(self.deployment.start())

    def execute(self, command: str, cwd: str = "/testbed") -> str:
        return asyncio.run(
            self.deployment.runtime.execute(RexCommand(command=command, shell=True, check=False, cwd=cwd))
        ).stdout

    def close(self):
        asyncio.run(self.deployment.stop())


class Agent:
    def __init__(self, config: AgentConfig, instance_config: InstanceConfig):
        self.config = config
        self.instance_config = instance_config
        instance_message = Template(config.instance_template).render(**instance_config.model_dump())
        self.history = [
            {"role": "system", "content": config.system_template},
            {"role": "user", "content": instance_message},
        ]
        self.model = Model(config.model_name, config.model_kwargs)
        self.env = Environment(instance_config.image)

    @property
    def n_steps(self) -> int:
        return len(self.history) // 2

    def run(self) -> str:
        """Run the agent and return the final observation"""
        is_finished = False
        while not is_finished:
            is_finished, _, observation = self.step()
        return observation

    def step(self) -> tuple[bool, str, str]:
        """

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
        self.history.append({"role": "assistant", "content": message})
        action = self.parse_action(message)
        is_done = False
        if action:
            observation = self.execute_action(action)
            self.history.append({"role": "user", "content": observation})
            if action.strip() == "submit":
                is_done = True
        else:
            self.history.append({"role": "assistant", "content": "Please always provide an action."})
        return is_done, message, observation

    @staticmethod
    def parse_action(message: str) -> str:
        """Parse the action from the message
        (assumes action is the first triple backticks block)
        """
        match = re.search(r"```(.*)```", message, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def execute_action(self, action: str) -> str:
        """Execute the action and return the observation"""
        if action.strip() == "submit":
            return self.env.execute("git add -A && git diff --cached")
        observation = self.env.execute(action)
        if not observation:
            return "The command returned no output."
        return observation
