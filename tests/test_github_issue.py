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


def normalize_whitespace(s: str) -> str:
    """Strip leading/trailing whitespace and normalize internal whitespace"""
    return "\n".join(line.rstrip() for line in s.strip().split("\n"))


def assert_observations_match(expected_observations: list[str], history: list[dict]) -> None:
    """Compare expected observations with actual observations from agent history

    Args:
        expected_observations: List of expected observation strings
        history: Agent conversation history (list of message dicts with 'role' and 'content')
    """
    # Extract actual observations from agent history
    # User messages (observations) are at indices 3, 5, 7, etc.
    actual_observations = []
    for i in range(len(expected_observations)):
        user_message_index = 3 + (i * 2)
        assert history[user_message_index]["role"] == "user"
        actual_observations.append(history[user_message_index]["content"])

    assert len(actual_observations) == len(expected_observations), (
        f"Expected {len(expected_observations)} observations, got {len(actual_observations)}"
    )

    for i, (expected_observation, actual_observation) in enumerate(zip(expected_observations, actual_observations)):
        normalized_actual = normalize_whitespace(actual_observation)
        normalized_expected = normalize_whitespace(expected_observation)

        assert normalized_actual == normalized_expected, (
            f"Step {i + 1} observation mismatch:\nExpected: {repr(normalized_expected)}\nActual: {repr(normalized_actual)}"
        )


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

    assert captured_agent is not None, "Agent should have been captured"
    history = captured_agent.history

    # Verify we have the right number of messages
    # Should be: system + user (initial) + (assistant + user) * number_of_steps
    expected_total_messages = 2 + (len(model_responses) * 2)
    assert len(history) == expected_total_messages, f"Expected {expected_total_messages} messages, got {len(history)}"

    # Check that environment observations match expected
    assert_observations_match(expected_observations, history)

    # Verify that the agent completed all steps
    assert captured_agent.model.n_calls == len(model_responses), (
        f"Expected {len(model_responses)} steps, got {captured_agent.model.n_calls}"
    )

    captured_agent.env.cleanup()
