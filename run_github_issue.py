#!/usr/bin/env python3
import argparse
from pathlib import Path

import requests
import yaml

from agent import Agent, AgentConfig, InstanceConfig


def fetch_github_issue(issue_url: str) -> str:
    """Fetch GitHub issue text from the URL."""
    # Convert GitHub issue URL to API URL
    api_url = issue_url.replace("github.com", "api.github.com/repos").replace("/issues/", "/issues/")

    response = requests.get(api_url)
    issue_data = response.json()

    title = issue_data["title"]
    body = issue_data["body"] or ""

    return f"GitHub Issue: {title}\n\n{body}"


def main():
    parser = argparse.ArgumentParser(description="Run nano-SWE-agent on a GitHub issue")
    parser.add_argument("issue_url", help="GitHub issue URL")

    args = parser.parse_args()

    repo_url = args.issue_url.split("/issues/")[0]
    problem_statement = fetch_github_issue(args.issue_url)

    config = yaml.safe_load((Path(__file__).parent / "config" / "github_issue.yaml").read_text())
    agent_config = AgentConfig(**config)

    instance_config = InstanceConfig(
        image=config["image"],
        problem_statement=problem_statement,
    )

    agent = Agent(agent_config, instance_config)

    print(f"Cloning {repo_url} to /testbed...")
    agent.env.execute(f"git clone {repo_url} /testbed", cwd="/")

    result = agent.run()
    print(f"\nFinal result: {result}")
    print(f"Total cost: ${agent.model.cost:.4f}")
    print(f"Total steps: {agent.n_steps}")

    print("Saving patch to patch.txt...")
    Path("patch.txt").write_text(result)


if __name__ == "__main__":
    main()
