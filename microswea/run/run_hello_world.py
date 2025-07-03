import os

from microswea.agents.default import DefaultAgent
from microswea.environments.local import LocalEnvironment
from microswea.models.litellm_model import LitellmModel

if __name__ == "__main__":
    agent = DefaultAgent(
        LitellmModel(model_name=os.getenv("MSWEA_MODEL_NAME")),
        LocalEnvironment(),
        input("What do you want to do? "),
    )
    agent.run()
