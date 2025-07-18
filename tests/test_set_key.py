import sys
from unittest.mock import patch

import pytest

from microsweagent.run.micro_extra import main


def test_set_key_insufficient_args():
    """Test that ValueError is raised when insufficient arguments are provided."""
    test_args = ["micro-extra", "set-key", "OPENAI_API_KEY"]  # Missing value

    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit, match="2"):
            main()


def test_set_key_no_value_arg():
    """Test that ValueError is raised when no arguments are provided to set-key."""
    test_args = ["micro-extra", "set-key"]  # Missing key and value

    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit, match="2"):
            main()


def test_set_key_too_many_args():
    """Test that ValueError is raised when too many arguments are provided."""
    test_args = ["micro-extra", "set-key", "OPENAI_API_KEY", "sk-test123", "extra"]

    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit, match="2"):
            main()


def test_set_key_preserves_other_commands():
    """Test that other commands still work when set-key is not called."""
    test_args = ["micro-extra", "gh", "issue-url"]

    with (
        patch.object(sys, "argv", test_args),
        patch("microsweagent.run.github_issue.app") as mock_gh_app,
    ):
        main()
        mock_gh_app.assert_called_once()


def test_set_key_does_not_interfere_with_inspector():
    """Test that inspector command still works when set-key is available."""
    test_args = ["micro-extra", "i", "trajectory.json"]

    with (
        patch.object(sys, "argv", test_args),
        patch("microsweagent.run.inspector.app") as mock_inspect_app,
    ):
        main()
        mock_inspect_app.assert_called_once()
