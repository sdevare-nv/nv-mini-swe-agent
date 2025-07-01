from unittest.mock import patch

from nanoswea import package_dir
from nanoswea.extra.model.test_models import DeterministicModel
from nanoswea.extra.run.run_swebench import run_from_cli


def test_swebench_end_to_end(test_data):
    """Test the complete SWEBench flow using the _test subset with deterministic model"""

    model_responses = test_data["model_responses"]

    with patch("nanoswea.extra.run.run_swebench.LitellmModel") as mock_model_class:
        mock_model_class.return_value = DeterministicModel(model_responses)

        run_from_cli(["--subset", "_test", "--split", "test", "--slice", "0:1", "--output", "results.json"])

    produced_output_path = package_dir.parent / "results.json"
    expected_output_path = package_dir.parent / "tests" / "test_data" / "results.json"
    assert expected_output_path.read_text() == produced_output_path.read_text()
