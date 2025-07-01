from pathlib import Path

import pytest
import yaml


@pytest.fixture
def test_data():
    """Load test fixtures with the expected model responses from YAML file"""
    yaml_path = Path(__file__).parent / "test_data" / "github_issue_test_data.yaml"
    with yaml_path.open() as f:
        return yaml.safe_load(f)
