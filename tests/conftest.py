from pathlib import Path
import json

import pytest


@pytest.fixture
def test_data():
    """Load test fixtures from the trajectory JSON file"""
    json_path = Path(__file__).parent / "test_data" / "github_issue.traj.json"
    with json_path.open() as f:
        trajectory = json.load(f)

    # Extract model responses (assistant messages, starting from index 2)
    model_responses = []
    # Extract expected observations (user messages, starting from index 3)
    expected_observations = []

    for i, message in enumerate(trajectory):
        if i < 2:  # Skip system message (0) and initial user message (1)
            continue

        if message["role"] == "assistant":
            model_responses.append(message["content"])
        elif message["role"] == "user":
            expected_observations.append(message["content"])

    return {"model_responses": model_responses, "expected_observations": expected_observations}
