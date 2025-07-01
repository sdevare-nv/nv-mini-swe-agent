from unittest.mock import patch

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


def test_github_issue_end_to_end(test_data):
    """Test the complete flow from CLI to final result using real environment but deterministic model"""

    model_responses = test_data["model_responses"]
    expected_observations = test_data["expected_observations"]

    # Patch the model and run the CLI
    with patch("nanoswea.run_github_issue.LitellmModel") as mock_model_class:
        mock_model_class.return_value = DeterministicModel(model_responses)

        # Run the CLI with the GitHub issue URL
        github_url = "https://github.com/SWE-agent/test-repo/issues/1"
        agent = run_from_cli([github_url])

    history = agent.history

    # Verify we have the right number of messages
    # Should be: system + user (initial) + (assistant + user) * number_of_steps
    expected_total_messages = 2 + (len(model_responses) * 2)
    assert len(history) == expected_total_messages, f"Expected {expected_total_messages} messages, got {len(history)}"

    # Check that environment observations match expected
    assert_observations_match(expected_observations, history)

    # Verify that the agent completed all steps
    assert agent.model.n_calls == len(model_responses), (
        f"Expected {len(model_responses)} steps, got {agent.model.n_calls}"
    )