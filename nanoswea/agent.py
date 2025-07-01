import re
import subprocess

from jinja2 import Template
from pydantic import BaseModel
from rich.console import Console

from nanoswea import Environment, Model


class AgentConfig(BaseModel):
    system_template: str
    instance_template: str
    step_limit: int = 0
    cost_limit: float = 3.0

console = Console(highlight=False)  # Print with colors


class Agent:
    def __init__(self, config: AgentConfig, model: Model, env: Environment, problem_statement: str):
        self.config = config
        instance_message = Template(config.instance_template).render(problem_statement=problem_statement)
        console.print(f"[bold green]System template:[/bold green]\n{config.system_template}", highlight=False)
        console.print(f"[bold green]Instance message:[/bold green]\n{instance_message}", highlight=False)
        self.history = [
            {"role": "system", "content": config.system_template},
            {"role": "user", "content": instance_message},
        ]
        self.model = model
        self.env = env

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
        if self.model.n_calls >= self.config.step_limit:
            return True, "", "step_limit_exceeded"
        if self.model.cost >= self.config.cost_limit:
            return True, "", "cost_limit_exceeded"
        message = self.model.query(self.history)
        assert isinstance(message, str)
        console.print(f"[bold red]Assistant (step {self.model.n_calls}):[/bold red]\n{message}")
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
        console.print(f"[bold green]User (step {self.model.n_calls}):[/bold green]\n{observation}")
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
            action = "git add -A && git diff --cached"

        try:
            return self.env.execute(action).strip() or "The command returned no output."
        except (TimeoutError, subprocess.TimeoutExpired):
            return "The command timed out. Please change your command and make sure it doesn't require input."
        except Exception as e:
            return f"Error executing action: {e}"
