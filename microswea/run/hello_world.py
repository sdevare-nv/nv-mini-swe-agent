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
    model_name = os.getenv("MSWEA_MODEL_NAME")
    if not model_name:
        model_name = input(
            "What model do you want to use? (assuming that you've set the API key as the environment variable already)"
        )
    agent = DefaultAgent(
        LitellmModel(model_name=model_name),
        LocalEnvironment(),
        problem,
    )
    agent.run()
    return agent


if __name__ == "__main__":
    app()
