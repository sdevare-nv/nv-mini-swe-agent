import re
import subprocess
from dataclasses import dataclass

from jinja2 import Template
from rich.console import Console

from nanoswea import Environment, Model


@dataclass
class AgentConfig:
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
        if 0 < self.config.step_limit <= self.model.n_calls:
            return True, "", "step_limit_exceeded"
        if 0 < self.config.cost_limit <= self.model.cost:
            return True, "", "cost_limit_exceeded"
        message = self.model.query(self.history)
        assert isinstance(message, str)
        console.print(f"[bold red]Assistant (step {self.model.n_calls}):[/bold red]\n{message}")
        self.history.append({"role": "assistant", "content": message})
        action = self.parse_action(message)
        is_finished, observation = self.execute_action(action)
        self.history.append({"role": "user", "content": observation})
        console.print(f"[bold green]Observation (step {self.model.n_calls}):[/bold green]\n{observation}")
        return is_finished, message, observation

    @staticmethod
    def parse_action(message: str) -> str:
        """Parse the action from the message
        (assumes action is the first triple backticks block)
        """
        match = re.search(r"```[a-zA-Z]*\n(.*?)(?=\n```|```)", message, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def execute_action(self, action: str) -> tuple[bool, str]:
        """Execute the action and return the observation.
        If the first line of the stdout of the action is "NANO_SWE_AGENT_FINAL_OUTPUT", assume the agent has finished its task.

        Returns:
            is_finished: whether the agent has finished its task
            observation: The observation or error message
        """
        if not action:
            return False, "Please always provide exactly one action in triple backticks."
        try:
            output = self.env.execute(action)
        except (TimeoutError, subprocess.TimeoutExpired):
            return False, "The command timed out. Please change your command and make sure it doesn't require input."
        except Exception as e:
            return False, f"Error executing action: {e}"
        if output.get("stdout") and output["stdout"].splitlines()[0] == "NANO_SWE_AGENT_FINAL_OUTPUT":
            return True, "\n".join(output["stdout"].splitlines()[1:])
        return (False, "\n".join([f"<{key}>\n{value}\n</{key}>" for key, value in output.items()]))