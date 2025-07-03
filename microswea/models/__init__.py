"""Model implementations for micro-SWE-agent.
You can ignore this file if you explicitly set your model in your run script.
"""


def get_model_class(model_name: str) -> type:
    if any(s in model_name for s in ["anthropic", "sonnet", "opus"]):
        from microswea.models.anthropic import AnthropicModel

        return AnthropicModel
    else:
        from microswea.models.litellm_model import LitellmModel

        return LitellmModel