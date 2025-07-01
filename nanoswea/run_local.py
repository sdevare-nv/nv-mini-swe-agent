#!/usr/bin/env python3
import json
from pathlib import Path

import typer
import yaml
from rich.console import Console

from nanoswea import package_dir
from nanoswea.agent import Agent, AgentConfig
from nanoswea.environment import LocalEnvironment, LocalEnvironmentConfig
from nanoswea.model import LitellmModel, ModelConfig

console = Console(highlight=False)
app = typer.Typer()


def get_multiline_problem_statement() -> str:
    """Get a multiline problem statement from the user."""
    console.print(
        "[bold yellow]What do you want to do?[/bold yellow] (press [bold red]Ctrl+D[/bold red] in a [red]new line[/red] to finish)"
    )
    lines = []
    while True:
        try:
            lines.append(console.input("[bold green]>[/bold green] "))
        except EOFError:
            break

    return "\n".join(lines).strip()


def run_local(problem_statement: str, config_path: str, model_name: str | None = None, yolo: bool = False) -> Agent:
    """Run the agent locally in the current directory."""
    config = yaml.safe_load(Path(config_path).read_text())
    agent_config = AgentConfig(**(config["agent"] | {"confirm_actions": not yolo}))

    if model_name:
        config["model"]["model_name"] = model_name
    model_config = ModelConfig(**config["model"])

    model = LitellmModel(model_config)
    env = LocalEnvironment(LocalEnvironmentConfig())
    agent = Agent(agent_config, model, env, problem_statement)

    try:
        result = agent.run()
    except KeyboardInterrupt:
        console.print("\n[bold red]KeyboardInterrupt -- goodbye[/bold red]")
    else:
        Path("patch.txt").write_text(result)
    finally:
        Path("traj.json").write_text(
            json.dumps(agent.history, indent=2),
        )
        print(f"\nFinal result: {result}")
        print(f"Total cost: ${agent.model.cost:.4f}")
        print(f"Total steps: {agent.model.n_calls}")
    return agent


@app.command()
def main(
    config: str = typer.Option(str(package_dir / "config" / "local.yaml"), "--config", help="Path to config file"),
    model: str | None = typer.Option(None, "--model", help="Model to use", show_default=False),
    problem: str | None = typer.Option(None, "--problem", help="Problem statement", show_default=False),
    yolo: bool = typer.Option(False, "--yolo", help="Run without confirmation"),
) -> Agent:
    """Run nano-SWE-agent right here, right now."""
    problem_statement = problem
    if not problem_statement:
        problem_statement = get_multiline_problem_statement()

    return run_local(problem_statement, config, model, yolo)


if __name__ == "__main__":
    app()
