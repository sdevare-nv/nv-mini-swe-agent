#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import yaml
from rich.console import Console

from nanoswea import package_dir
from nanoswea.agent import Agent, AgentConfig
from nanoswea.environment import LocalEnvironment, LocalEnvironmentConfig
from nanoswea.model import LitellmModel, ModelConfig

console = Console(highlight=False)


def get_multiline_problem_statement() -> str:
    """Get a multiline problem statement from the user."""
    console.print("\nEnter your problem statement (press Ctrl+D to finish):")
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

    result = agent.run()
    print(f"\nFinal result: {result}")
    print(f"Total cost: ${agent.model.cost:.4f}")
    print(f"Total steps: {agent.model.n_calls}")

    print("Saving output files")
    Path("patch.txt").write_text(result)
    Path("traj.json").write_text(
        json.dumps(agent.history, indent=2),
    )
    return agent


def run_from_cli(cli_args: list[str] | None = None) -> Agent:
    parser = argparse.ArgumentParser(description="Run nano-SWE-agent locally")
    parser.add_argument("--config", help="Path to config file", default=package_dir / "config" / "local.yaml")
    parser.add_argument("--model", help="Model to use (overrides config)")
    parser.add_argument("--problem", help="Problem statement")
    parser.add_argument("--yolo", help="Run without confirmation", action="store_true")

    args = parser.parse_args(args=cli_args)

    problem_statement = args.problem
    if not problem_statement:
        problem_statement = get_multiline_problem_statement()

    return run_local(problem_statement, args.config, args.model, args.yolo)


if __name__ == "__main__":
    run_from_cli()
