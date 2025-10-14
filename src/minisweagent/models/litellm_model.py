import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import litellm
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS

logger = logging.getLogger("litellm_model")
litellm.return_response_headers = True
# SET_COOKIE_ID = "set-cookie"
SET_COOKIE_ID = "set-cookie"  # Use standard HTTP header name for receiving cookies


@dataclass
class LitellmModelConfig:
    model_name: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    litellm_model_registry: Path | None = None


class LitellmModel:
    def __init__(self, **kwargs):
        self.config = LitellmModelConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        self.response_headers = None

        if self.config.litellm_model_registry is not None:
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))

    def _add_tokens_ids_to_messages(self, messages: list[dict[str, str]], responses: list[dict[str, str]]):
        processed_messages = []
        responses_idx = 0
        for message in messages:
            if message["role"] in ["system", "user"]:
                processed_messages.append(message)
            elif message["role"] == "assistant":
                assistant_message = message.copy()
                response = responses[responses_idx]
                if response.get("provider_specific_fields", {}):
                    provider_specific_fields = response["provider_specific_fields"]
                    for key in ["prompt_token_ids", "generation_token_ids", "generation_log_probs"]:
                        if key in provider_specific_fields:
                            assistant_message[key] = provider_specific_fields[key]
                responses_idx += 1
                processed_messages.append(assistant_message)

        return processed_messages

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=5, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                litellm.exceptions.UnsupportedParamsError,
                litellm.exceptions.NotFoundError,
                litellm.exceptions.PermissionDeniedError,
                litellm.exceptions.ContextWindowExceededError,
                litellm.exceptions.APIError,
                litellm.exceptions.AuthenticationError,
                KeyboardInterrupt,
            )
        ),
    )
    def _query(self, messages: list[dict[str, str]], responses: list[dict[str, str]], **kwargs):
        try:
            raw_cookie = (
                self.response_headers.get(SET_COOKIE_ID)
                if self.response_headers and SET_COOKIE_ID in self.response_headers
                else None
            )
            extra_headers = kwargs.get("extra_headers", {}).copy()
            if raw_cookie:
                cookie_value = raw_cookie.split(";")[0].strip()
                extra_headers["Cookie"] = cookie_value
                print("DEBUG:gym-parsed-cookie", cookie_value)

            response = litellm.completion(
                model=self.config.model_name,
                messages=self._add_tokens_ids_to_messages(messages, responses),
                timeout=7200,  # 2 hours,
                extra_headers=extra_headers,
                **(self.config.model_kwargs | kwargs),
            )
            if not self.response_headers:
                self.response_headers = response._response_headers
            return response
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def query(self, messages: list[dict[str, str]], responses: list[dict[str, str]], **kwargs) -> dict:
        response = self._query(messages, responses, **kwargs)
        if hasattr(response.choices[0].message, "provider_specific_fields"):
            provider_specific_fields = response.choices[0].message.provider_specific_fields
        else:
            provider_specific_fields = {}
        try:
            cost = litellm.cost_calculator.completion_cost(response)
            self.n_calls += 1
            self.cost += cost
            GLOBAL_MODEL_STATS.add(cost)
        except Exception:
            self.n_calls += 1
            self.cost += 0
            GLOBAL_MODEL_STATS.add(0)

        return {
            "content": response.choices[0].message.content or "",  # type: ignore
            "response_obj": response.model_dump() | {"provider_specific_fields": provider_specific_fields},
        }
