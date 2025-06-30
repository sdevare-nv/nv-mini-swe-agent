from typing import Any

import litellm
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential


class ModelConfig(BaseModel):
    model_name: str
    model_kwargs: dict[str, Any] = {}


class LitellmModel:
    def __init__(self, config: ModelConfig):
        self.model_name = config.model_name
        self.model_kwargs = config.model_kwargs
        self.cost = 0.0

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
    def query(self, messages: list[dict[str, str]]) -> str:
        response: litellm.types.utils.ModelResponse = litellm.completion(  # type: ignore
            model=self.model_name, messages=messages, **self.model_kwargs
        )
        self.cost += litellm.cost_calculator.completion_cost(response)
        return response.choices[0].message.content  # type: ignore
