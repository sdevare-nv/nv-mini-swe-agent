#!/usr/bin/env python3

"""Run mini-SWE-agent on SWE-GYM instances for full test evaluation."""
# Read this first: https://mini-swe-agent.com/latest/usage/swebench/  (usage docs)

import json
from pathlib import Path

from swegym.harness.constants import SWEbenchInstance
from swegym.harness.docker_build import setup_logger
from swegym.harness.grading import get_eval_report
from swegym.harness.test_spec import make_test_spec

from minisweagent.config import builtin_config_dir
from minisweagent.environments import DockerEnvironment, SingularityEnvironment
from minisweagent.run.extra.base_runner import SWEGymRunner, make_runner_command

_HELP_TEXT = """Run mini-SWE-agent on SWEGym instances for full test evaluation.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/swebench/[/bold green]
[/not dim]
"""


class TestRunner(SWEGymRunner):
    """Runner that evaluates by running actual tests using SWEGym harness."""

    def run_eval(
        self,
        trajectory_data: dict,
        instance: SWEbenchInstance | dict,
        env: SingularityEnvironment | DockerEnvironment,
        model_patch: str,
        instance_dir: Path,
        run_id: str,
        is_golden: bool = False,
    ) -> dict:
        """Evaluate by running actual tests."""
        test_spec = make_test_spec(instance)
        pred = {"instance_id": test_spec.instance_id, "model_patch": model_patch}
        instance_id = test_spec.instance_id

        instance_dir.mkdir(parents=True, exist_ok=True)
        log_file = instance_dir / f"run_instance_{run_id}.log"
        report_path = instance_dir / f"report_{run_id}.json"
        patch_file = instance_dir / f"patch_{run_id}.diff"
        patch_file.write_text(model_patch)

        logger = setup_logger(instance_id, log_file)
        logger.info(f"DEBUG test_spec {test_spec}")
        logger.info(f"DEBUG eval_script {test_spec.eval_script}")

        if is_golden:
            env.execute(command=f"cat > patch.diff <<'EOF'\n{model_patch}\n\nEOF")
            env.execute(command="git status --porcelain")
            env.execute(command="git apply --check patch.diff")
            env.execute(command="git apply patch.diff")

        eval_script = test_spec.eval_script.replace("#!/bin/bash", "")
        res = env.execute(command=eval_script, is_eval=True)

        test_output, returncode = res["output"], res["returncode"]
        print(f"[EVAL]{instance_id} returncode: {returncode}")
        test_output_path = instance_dir / f"test_output_{run_id}.txt"
        test_output_path.write_text(test_output)
        print(f"[EVAL]{instance_id} Test output written to {test_output_path}")

        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            log_path=test_output_path,
            include_tests_status=True,
        )
        print(f"[EVAL]{instance_id} Result: resolved: {report[instance_id]['resolved']}")

        report_path.write_text(json.dumps(report, indent=4))

        return {
            "instance_id": instance_id,
            "model_patch": model_patch,
            "eval_report": report,
        }


app, main = make_runner_command(
    TestRunner,
    _HELP_TEXT,
    config=builtin_config_dir / "extra" / "swebench.yaml",
)

if __name__ == "__main__":
    app()
