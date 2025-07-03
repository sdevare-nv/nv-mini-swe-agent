import os

import typer

from microswea.agents.default import DefaultAgent
from microswea.environments.local import LocalEnvironment
from microswea.models.litellm_model import LitellmModel

app = typer.Typer()


@app.command()
def main(
    problem: str = typer.Option(..., "--problem", help="Problem statement", show_default=False, prompt=True),
) -> DefaultAgent:
    """Run a simple hello world agent."""
    agent = DefaultAgent(
        LitellmModel(model_name=os.getenv("MSWEA_MODEL_NAME")),
        LocalEnvironment(),
        problem,
    )
    agent.run()
    return agent


if __name__ == "__main__":
    app()
