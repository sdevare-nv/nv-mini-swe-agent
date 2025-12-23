#!/usr/bin/env python3

"""Run mini-SWE-agent on Terminal Bench instances for evaluation."""

import re
from pathlib import Path

from minisweagent.agents.terminal_bench import TerminalBenchAgent
from minisweagent.config import builtin_config_dir
from minisweagent.environments import DockerEnvironment, SingularityEnvironment
from minisweagent.run.extra.base_runner import SWEGymRunner, make_runner_command
from minisweagent.run.extra.evaluators import Evaluator
from minisweagent.run.extra.runner_config import ProcessInstanceConfig
from minisweagent.run.extra.utils.pytest_parser import PytestParser, UnitTestStatus

_HELP_TEXT = """Run mini-SWE-agent on Terminal Bench instances for full test evaluation."""


class TerminalBenchEvaluator(Evaluator):
    """Evaluator for Terminal Bench with custom test harness."""

    def evaluate(
        self,
        cfg: ProcessInstanceConfig,
        trajectory_data: dict,
        env: SingularityEnvironment | DockerEnvironment,
        model_patch: str,
        instance_dir: Path,
    ) -> dict:
        """Evaluate by running Terminal Bench test harness."""
        instance = cfg.instance
        instance_id = instance["instance_id"]
        test_script = instance["test_script"]
        test_files = instance["test_files"]

        instance_dir.mkdir(parents=True, exist_ok=True)

        # Parse output the uv installation from test_script
        # NB: This is a hack to get the test script to work in the container.
        # We don't have to install uv in the container again.
        test_script = test_script.replace("apt-get update", "")
        test_script = test_script.replace("apt-get install -y curl", "")
        test_script = re.sub(r"curl\s+-LsSf\s+https://astral\.sh/uv/[^/]+/install\.sh\s*\|\s*sh", "", test_script)
        test_script = test_script.replace("source $HOME/.local/bin/env", "")

        # NB: uv is mounted in the container at /tmp/uv, so we need to use that path.
        test_script = test_script.replace("uv", "/tmp/uv")

        # copy the test_script to the container
        env.execute(command=f"cat > run-tests.sh <<'EOF'\n{test_script}\n\nEOF")

        # make a /tests directory and write the test files to it
        env.execute(command="mkdir -p /tests")
        for test_file in test_files:
            filename = test_file["filename"]
            content = test_file["content"]
            env.execute(command=f"cat > /tests/{filename} <<'EOF'\n{content}\n\nEOF")

        # run the test script
        result = env.execute(command="TEST_DIR=/tests bash run-tests.sh")
        output = result["output"]

        print(f"Test output: {output}")

        # use the pytest parser to parse the output
        # TODO: We only support pytest for now. We need to add support for other parsers.
        
        parser = PytestParser()
        try:
            parsed_output = parser.parse(output)
            test_status = [status for status in parsed_output.values()]
            resolved = all(status == UnitTestStatus.PASSED for status in test_status)

            for test_name, status in parsed_output.items():
                parsed_output[test_name] = status.value

            return {
                "instance_id": instance_id,
                "eval_report": {
                    "resolved": resolved,
                    "parsed_output": parsed_output,
                },
            }
        except Exception as e:
            return {
                "instance_id": instance_id,
                "eval_report": {
                    "resolved": False,
                    "parsed_output": str(e),
                },
            }


class TestRunner(SWEGymRunner):
    """Runner that evaluates using Terminal Bench evaluator."""

    def get_evaluator(self, subset: str) -> Evaluator:
        """Get the Terminal Bench evaluator."""
        return TerminalBenchEvaluator()

    def create_agent(self, cfg, model, env, agent_config):
        """Create TerminalBenchAgent instead of default agent."""
        return TerminalBenchAgent(
            model=model,
            env=env,
            responses_create_params=cfg.responses_create_params,
            **agent_config,
        )


app = make_runner_command(
    TestRunner,
    _HELP_TEXT,
    config=builtin_config_dir / "extra" / "terminal_bench.yaml",
)

if __name__ == "__main__":
    app()
