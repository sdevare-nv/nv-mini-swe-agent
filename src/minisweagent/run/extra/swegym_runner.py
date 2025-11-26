#!/usr/bin/env python3

"""Run mini-SWE-agent on SWE-GYM instances for full test evaluation."""
# Read this first: https://mini-swe-agent.com/latest/usage/swebench/  (usage docs)

import json
import re
from pathlib import Path

from swegym.harness.docker_build import setup_logger
from swegym.harness.grading import get_eval_report
from swegym.harness.test_spec import make_test_spec

from minisweagent.config import builtin_config_dir
from minisweagent.environments import DockerEnvironment, SingularityEnvironment
from minisweagent.run.extra.base_runner import SWEGymRunner, make_runner_command
from minisweagent.run.extra.evaluators import Evaluator
from minisweagent.run.extra.runner_config import ProcessInstanceConfig

_HELP_TEXT = """Run mini-SWE-agent on SWEGym instances for full test evaluation.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/swebench/[/bold green]
[/not dim]
"""


class ExternalTestEvaluator(Evaluator):
    """Evaluator for external benchmarks using SWEGym harness."""

    def evaluate(
        self,
        cfg: ProcessInstanceConfig,
        trajectory_data: dict,
        env: SingularityEnvironment | DockerEnvironment,
        model_patch: str,
        instance_dir: Path,
    ) -> dict:
        """Evaluate by running actual tests using SWEGym harness."""
        instance = cfg.instance
        test_spec = make_test_spec(instance)
        pred = {"instance_id": test_spec.instance_id, "model_patch": model_patch}
        instance_id = test_spec.instance_id

        instance_dir.mkdir(parents=True, exist_ok=True)
        log_file = instance_dir / f"run_instance_{cfg.run_id}.log"
        report_path = instance_dir / f"report_{cfg.run_id}.json"
        patch_file = instance_dir / f"patch_{cfg.run_id}.diff"
        patch_file.write_text(model_patch)

        logger = setup_logger(instance_id, log_file)
        logger.info(f"DEBUG test_spec {test_spec}")
        logger.info(f"DEBUG eval_script {test_spec.eval_script}")

        if cfg.run_golden:
            env.execute(command=f"cat > patch.diff <<'EOF'\n{model_patch}\n\nEOF")
            env.execute(command="git status --porcelain")
            env.execute(command="git apply --check patch.diff")
            env.execute(command="git apply patch.diff")

        eval_script = test_spec.eval_script.replace("#!/bin/bash", "")
        res = env.execute(command=eval_script, is_eval=True)

        test_output, returncode = res["output"], res["returncode"]
        print(f"[EVAL]{instance_id} returncode: {returncode}")
        test_output_path = instance_dir / f"test_output_{cfg.run_id}.txt"
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


def _check_tests_passed(output: dict, instance: dict) -> dict:
    """Check which fail_to_pass (f2p) and pass_to_pass (p2p) tests are passed and failed."""
    if not output:
        return {
            "fail_to_pass_passed": [],
            "fail_to_pass_failed": [],
            "pass_to_pass_passed": [],
            "pass_to_pass_failed": [],
        }

    passed_tests = {test["name"] for test in output.get("tests", []) if test.get("status") == "PASSED"}

    f2p_str = instance.get("fail_to_pass_select", instance.get("fail_to_pass", "[]"))
    p2p_str = instance.get("pass_to_pass_select", instance.get("pass_to_pass", "[]"))

    f2p = set(eval(f2p_str)) if isinstance(f2p_str, str) else set(f2p_str)
    p2p = set(eval(p2p_str)) if isinstance(p2p_str, str) else set(p2p_str)

    fail_to_pass_passed = sorted([test for test in f2p if test in passed_tests])
    fail_to_pass_failed = sorted([test for test in f2p if test not in passed_tests])
    pass_to_pass_passed = sorted([test for test in p2p if test in passed_tests])
    pass_to_pass_failed = sorted([test for test in p2p if test not in passed_tests])

    return {
        "fail_to_pass_passed": fail_to_pass_passed,
        "fail_to_pass_failed": fail_to_pass_failed,
        "pass_to_pass_passed": pass_to_pass_passed,
        "pass_to_pass_failed": pass_to_pass_failed,
    }


class InternalTestEvaluator(Evaluator):
    """Evaluator for internal benchmarks with custom test harness."""

    def evaluate(
        self,
        cfg: ProcessInstanceConfig,
        trajectory_data: dict,
        env: SingularityEnvironment | DockerEnvironment,
        model_patch: str,
        instance_dir: Path,
    ) -> dict:
        """Evaluate by running internal test harness."""
        instance = cfg.instance
        instance_id = instance["instance_id"]
        base_dockerfile = instance.get("base_dockerfile", "")
        instance_dockerfile = instance.get("instance_dockerfile", "")

        env_lines = []
        for line in (base_dockerfile + "\n" + instance_dockerfile).split("\n"):
            line = line.strip()
            if line.startswith("ENV "):
                export_line = line.replace("ENV ", "export ", 1)
                if "=" in export_line:
                    export_line = re.sub(r"\s*=\s*", "=", export_line)
                else:
                    parts = export_line.split(None, 2)
                    if len(parts) >= 3:
                        key = parts[1]
                        value = parts[2]
                        export_line = f'export {key}="{value}"'
                env_lines.append(export_line)

        env_exports = "\n".join(env_lines)
        repo_cmd = instance.get("before_repo_set_cmd", "").strip()
        if repo_cmd:
            repo_cmd = repo_cmd.split("\n")[-1]

        test_files_str = instance.get("selected_test_files_to_run", "[]")
        if isinstance(test_files_str, str):
            test_files = ",".join(eval(test_files_str))
        else:
            test_files = ",".join(test_files_str)

        env.execute(command="mkdir -p /workspace")
        env.execute(command=f"cat > /workspace/run_script.sh <<'EOF'\n{instance.get('run_script.sh')}\n\nEOF")
        env.execute(command=f"cat > /workspace/parsing_script.py <<'EOF'\n{instance.get('parsing_script.py')}\n\nEOF")

        script = f"""#!/bin/bash
set -e

{env_exports}

# Apply patch
cd /app

# Note: the patch is already applied in the container, so we don't need to apply it again.

# Setup repository
{repo_cmd}

# Run tests
bash /workspace/run_script.sh {test_files} > /workspace/stdout.log 2> /workspace/stderr.log || true

# Parse results
python /workspace/parsing_script.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json

cat /workspace/output.json
"""

        res = env.execute(command=script, is_eval=True)
        test_output, returncode = res["output"], res["returncode"]
        print(f"WIP: {test_output}, {returncode}")

        try:
            test_output = json.loads(test_output)
        except Exception as e:
            print(f"Error parsing test output: {e}")
            test_output = {}

        test_results = _check_tests_passed(test_output, instance)

        eval_report = {
            instance_id: {
                "patch_is_None": False if model_patch else True,
                "patch_exists": True if model_patch else False,
                "patch_successfully_applied": True if model_patch else False,
                "resolved": True
                if test_results["fail_to_pass_failed"] == []
                and test_results["pass_to_pass_failed"] == []
                and len(test_results["fail_to_pass_passed"])
                and len(test_results["pass_to_pass_passed"])
                else False,
                "test_status": {
                    "FAIL_TO_PASS": {
                        "success": test_results["fail_to_pass_passed"],
                        "failure": test_results["fail_to_pass_failed"],
                    },
                    "PASS_TO_PASS": {
                        "success": test_results["pass_to_pass_passed"],
                        "failure": test_results["pass_to_pass_failed"],
                    },
                },
            },
        }

        return {
            "instance_id": instance_id,
            "model_patch": model_patch,
            "eval_report": eval_report,
        }


class TestRunner(SWEGymRunner):
    """Runner that evaluates using appropriate evaluator based on subset."""

    def get_evaluator(self, subset: str) -> Evaluator:
        """Get the right evaluator based on subset name."""
        internal_benchmarks = {"nv-internal-1"}
        if subset in internal_benchmarks:
            return InternalTestEvaluator()
        return ExternalTestEvaluator()


app = make_runner_command(
    TestRunner,
    _HELP_TEXT,
    config=builtin_config_dir / "extra" / "swebench.yaml",
)

if __name__ == "__main__":
    app()
