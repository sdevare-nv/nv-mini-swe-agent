#!/usr/bin/env python3
import json
import os
from pathlib import Path

import requests
import typer
import yaml

from nanoswea import package_dir
from nanoswea.agent import Agent, AgentConfig
from nanoswea.environment import DockerEnvironment, DockerEnvironmentConfig
from nanoswea.model import LitellmModel, ModelConfig

app = typer.Typer()


def fetch_github_issue(issue_url: str) -> str:
    """Fetch GitHub issue text from the URL."""
    # Convert GitHub issue URL to API URL
    api_url = issue_url.replace("github.com", "api.github.com/repos").replace("/issues/", "/issues/")

    headers = {}
    if github_token := os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = f"token {github_token}"

    response = requests.get(api_url, headers=headers)
    issue_data = response.json()

    title = issue_data["title"]
    body = issue_data["body"] or ""

    return f"GitHub Issue: {title}\n\n{body}"


def run_github_issue(issue_url: str, repo_url: str, model_name: str | None = None) -> Agent:
    problem_statement = fetch_github_issue(issue_url)

    config = yaml.safe_load((package_dir / "config" / "github_issue.yaml").read_text())
    agent_config = AgentConfig(**config["agent"])
    if model_name:
        config["model"]["model_name"] = model_name
    model_config = ModelConfig(**config["model"])

    model = LitellmModel(model_config)
    env = DockerEnvironment(DockerEnvironmentConfig(**config.get("environment", {})))
    agent = Agent(agent_config, model, env, problem_statement)

    if github_token := os.getenv("GITHUB_TOKEN"):
        repo_url = repo_url.replace("https://github.com/", f"https://{github_token}@github.com/") + ".git"

    agent.env.execute(f"git clone {repo_url} /testbed", cwd="/")

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


@app.command()
def main(
    issue_url: str = typer.Option(prompt="Enter GitHub issue URL", help="GitHub issue URL"),
    model: str | None = typer.Option(None, "--model", help="Model to use"),
) -> Agent:
    """Run nano-SWE-agent on a GitHub issue"""
    repo_url = issue_url.split("/issues/")[0]
    return run_github_issue(issue_url, repo_url, model)


if __name__ == "__main__":
    app()
