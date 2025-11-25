import pytest

from minisweagent.agents.single_turn import SingleTurnAgent
from minisweagent.models.test_models import DeterministicModel


def test_successful_diff_generation():
    """Test agent generates a diff successfully."""
    agent = SingleTurnAgent(
        model=DeterministicModel(
            outputs=[
                """Here's the fix for the issue:
                
```diff
diff --git a/example.py b/example.py
index abc123..def456 100644
--- a/example.py
+++ b/example.py
@@ -1,3 +1,3 @@
 def hello():
-    print("Hello")
+    print("Hello, World!")
     return
```
"""
            ]
        ),
    )

    exit_status, result = agent.run("Fix the greeting function")
    assert exit_status == "Submitted"
    assert "diff --git" in result
    assert "Hello, World!" in result
    assert agent.model.n_calls == 1


def test_cost_limit_enforcement():
    """Test agent stops when cost limit is reached before querying."""
    model = DeterministicModel(outputs=["```diff\n--- a/file.py\n+++ b/file.py\n```"])
    model.cost = 5.0

    agent = SingleTurnAgent(
        model=model,
        cost_limit=3.0,
    )

    exit_status, result = agent.run("Generate a patch")
    assert exit_status == "LimitsExceeded"
    assert result == ""
    assert agent.model.n_calls == 0


def test_extract_diff_from_code_block():
    """Test diff extraction from code blocks."""
    agent = SingleTurnAgent(
        model=DeterministicModel(
            outputs=[
                """
```diff
diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1 +1 @@
-old line
+new line
```
"""
            ]
        ),
    )

    exit_status, result = agent.run("Test diff extraction")
    assert exit_status == "Submitted"
    assert result.startswith("diff --git")
    assert "old line" in result
    assert "new line" in result


def test_extract_diff_without_language_tag():
    """Test diff extraction from plain code blocks."""
    agent = SingleTurnAgent(
        model=DeterministicModel(
            outputs=[
                """
```
diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1 +1 @@
-old
+new
```
"""
            ]
        ),
    )

    exit_status, result = agent.run("Test extraction")
    assert exit_status == "Submitted"
    assert "diff --git" in result


def test_extract_diff_without_code_block():
    """Test diff extraction when no code blocks present."""
    agent = SingleTurnAgent(
        model=DeterministicModel(
            outputs=[
                """
diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1 +1 @@
-old
+new
"""
            ]
        ),
    )

    exit_status, result = agent.run("Test plain diff")
    assert exit_status == "Submitted"
    assert "diff --git" in result


def test_extract_from_solution_tags():
    """Test extraction from <solution> tags."""
    agent = SingleTurnAgent(
        model=DeterministicModel(
            outputs=[
                """
<think>
This is my reasoning about the bug.
</think>

<solution>
```python
### test.py
<<<<<<< SEARCH
def old_function():
    pass
=======
def new_function():
    return True
>>>>>>> REPLACE
```
</solution>
"""
            ]
        ),
    )

    exit_status, result = agent.run("Test solution tag extraction")
    assert exit_status == "Submitted"
    assert "### test.py" in result
    assert "<<<<<<< SEARCH" in result
    assert ">>>>>>> REPLACE" in result


def test_extract_multiple_code_blocks():
    """Test extraction of multiple code blocks."""
    agent = SingleTurnAgent(
        model=DeterministicModel(
            outputs=[
                """
First edit:
```python
### file1.py
<<<<<<< SEARCH
old code 1
=======
new code 1
>>>>>>> REPLACE
```

Second edit:
```python
### file2.py
<<<<<<< SEARCH
old code 2
=======
new code 2
>>>>>>> REPLACE
```
"""
            ]
        ),
    )

    exit_status, result = agent.run("Test multiple blocks")
    assert exit_status == "Submitted"
    assert "### file1.py" in result
    assert "### file2.py" in result


def test_custom_config():
    """Test agent works with custom configuration."""
    agent = SingleTurnAgent(
        model=DeterministicModel(outputs=["```diff\npatch content\n```"]),
        system_template="You are a patch generator.",
        instance_template="Generate patch for: {{task}}",
        cost_limit=5.0,
    )

    exit_status, result = agent.run("Custom task")
    assert exit_status == "Submitted"
    assert result == "patch content"
    assert agent.messages[0]["content"] == "You are a patch generator."
    assert "Custom task" in agent.messages[1]["content"]


def test_empty_response():
    """Test agent handles empty model response."""
    agent = SingleTurnAgent(
        model=DeterministicModel(outputs=[""]),
    )

    exit_status, result = agent.run("Generate patch")
    assert exit_status == "LimitsExceeded"
    assert result == ""


def test_custom_input_messages():
    """Test agent can work with custom input messages."""
    agent = SingleTurnAgent(
        model=DeterministicModel(outputs=["```diff\ncustom patch\n```"]),
        responses_create_params={
            "input": [
                {"role": "system", "content": "Custom system message"},
                {"role": "user", "content": "Custom user message"},
            ]
        },
    )

    exit_status, result = agent.run("This task will be ignored")
    assert exit_status == "Submitted"
    assert result == "custom patch"
    assert agent.messages[0]["content"] == "Custom system message"
    assert agent.messages[1]["content"] == "Custom user message"


def test_message_tracking():
    """Test that messages are properly tracked."""
    agent = SingleTurnAgent(
        model=DeterministicModel(outputs=["```diff\ntest patch\n```"]),
    )

    exit_status, result = agent.run("Test message tracking")
    assert exit_status == "Submitted"
    
    # Should have system and user messages only (no execution loop)
    assert len(agent.messages) == 2
    assert [msg["role"] for msg in agent.messages] == ["system", "user"]
    
    # DeterministicModel doesn't return response_obj, so responses list stays empty
    # (Real models like LiteLLM will populate this)
    assert len(agent.responses) == 0


def test_response_params():
    """Test that temperature and top_p are passed through."""
    agent = SingleTurnAgent(
        model=DeterministicModel(outputs=["```diff\npatch\n```"]),
        responses_create_params={
            "temperature": 0.8,
            "top_p": 0.95,
        },
    )

    exit_status, result = agent.run("Test params")
    assert exit_status == "Submitted"
    # DeterministicModel will ignore these params but they should not cause errors

