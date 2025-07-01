#!/usr/bin/env python3
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

# Add the parent directory to the path so we can import from nanoswea
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanoswea.agent import Agent
from nanoswea.extra.model.test_models import DeterministicModel
from nanoswea.run_github_issue import run_from_cli


@pytest.fixture
def test_data():
    """Load test fixtures with the expected model responses from YAML file"""
    yaml_path = Path(__file__).parent / "test_data" / "github_issue_test_data.yaml"
    with yaml_path.open() as f:
        return yaml.safe_load(f)


def test_github_issue_end_to_end(test_data):
    """Test the complete flow from CLI to final result using real environment but deterministic model"""

    model_responses = test_data["model_responses"]
    expected_observations = test_data["expected_observations"]

    # Create deterministic model with the test responses
    deterministic_model = DeterministicModel(model_responses)

    # Track the agent instance
    captured_agent = None

    # Store the original Agent __init__ method
    original_agent_init = Agent.__init__

    def capture_agent_init(self, *args, **kwargs):
        nonlocal captured_agent
        # Call the original __init__
        original_agent_init(self, *args, **kwargs)
        # Capture this agent instance
        captured_agent = self

    # Patch the Agent constructor and the model
    with patch.object(Agent, "__init__", capture_agent_init):
        with patch("nanoswea.run_github_issue.LitellmModel") as mock_model_class:
            mock_model_class.return_value = deterministic_model

            # Run the CLI with the GitHub issue URL
            github_url = "https://github.com/SWE-agent/test-repo/issues/1"
            run_from_cli([github_url])

    # Verify that we captured the agent
    assert captured_agent is not None, "Agent should have been captured"

    # Check that the agent history matches expected flow
    # The history should contain: system message, user message, then alternating assistant/user messages
    history = captured_agent.history

    # Verify we have the right number of messages
    # Should be: system + user (initial) + (assistant + user) * number_of_steps
    expected_total_messages = 2 + (len(model_responses) * 2)
    assert len(history) == expected_total_messages, f"Expected {expected_total_messages} messages, got {len(history)}"

    # Check that model responses match what we provided
    for i, expected_model_response in enumerate(model_responses):
        # Assistant messages are at indices 2, 4, 6, etc.
        assistant_message_index = 2 + (i * 2)
        assert history[assistant_message_index]["role"] == "assistant"
        assert history[assistant_message_index]["content"] == expected_model_response

    # Check that environment observations match expected
    for i, expected_observation in enumerate(expected_observations):
        # User messages (observations) are at indices 3, 5, 7, etc.
        user_message_index = 3 + (i * 2)
        assert history[user_message_index]["role"] == "user"
        # The observation should match the expected output
        actual_observation = history[user_message_index]["content"]

        # Normalize whitespace for more robust comparison
        def normalize_whitespace(s):
            # Strip leading/trailing whitespace and normalize internal whitespace
            return "\n".join(line.rstrip() for line in s.strip().split("\n"))

        normalized_actual = normalize_whitespace(actual_observation)
        normalized_expected = normalize_whitespace(expected_observation)

        assert normalized_actual == normalized_expected, (
            f"Step {i + 1} observation mismatch:\nExpected: {repr(normalized_expected)}\nActual: {repr(normalized_actual)}"
        )

    # Verify that the agent completed all steps
    assert captured_agent.model.n_calls == len(model_responses), (
        f"Expected {len(model_responses)} steps, got {captured_agent.model.n_calls}"
    )

    # Check that the final step was a submit (agent finished)
    final_model_response = model_responses[-1]
    assert "submit" in final_model_response, "Final model response should contain submit command"

    captured_agent.env.cleanup()
