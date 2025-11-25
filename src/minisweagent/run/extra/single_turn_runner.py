#!/usr/bin/env python3

"""Run mini-SWE-agent with single-turn diff generation for SWEGym instances."""

import json
from pathlib import Path

from swegym.harness.constants import SWEbenchInstance
from swegym.harness.docker_build import setup_logger
from swegym.harness.grading import get_eval_report
from swegym.harness.test_spec import make_test_spec

from minisweagent.agents.single_turn import SingleTurnAgent
from minisweagent.config import builtin_config_dir
from minisweagent.environments import DockerEnvironment, SingularityEnvironment
from minisweagent.run.extra.base_runner import SWEGymRunner, make_runner_command
from minisweagent.run.extra.runner_config import ProcessInstanceConfig
from minisweagent.run.extra.utils.search_replace_parser import extract_and_convert_to_diff

_HELP_TEXT = """Run mini-SWE-agent with single-turn diff generation.

[not dim]
The agent generates a diff patch in a single model query without execution loop.
The patch is then evaluated by applying it to the container and running tests.
[/not dim]
"""


class SingleTurnRunner(SWEGymRunner):
    """Runner that uses single-turn agent to generate patches and evaluates with tests."""

    def create_agent(self, cfg: ProcessInstanceConfig, model, env, agent_config: dict):
        """Create single-turn agent for the instance."""
        # Filter out step_limit and collapse_limit - not applicable for single-turn agent
        filtered_config = {k: v for k, v in agent_config.items() if k not in ["step_limit", "collapse_limit"]}
        return SingleTurnAgent(
            model,
            env,
            cfg.responses_create_params,
            **filtered_config,
        )

    def run_eval(
        self,
        cfg: ProcessInstanceConfig,
        trajectory_data: dict,
        env: SingularityEnvironment | DockerEnvironment,
        model_patch: str,
        instance_dir: Path,
    ) -> dict:
        """Evaluate by applying the patch and running tests."""
        instance = cfg.instance
        test_spec = make_test_spec(instance)
        instance_id = test_spec.instance_id

        instance_dir.mkdir(parents=True, exist_ok=True)
        log_file = instance_dir / f"run_instance_{cfg.run_id}.log"
        report_path = instance_dir / f"report_{cfg.run_id}.json"
        
        logger = setup_logger(instance_id, log_file)
        logger.info(f"DEBUG test_spec {test_spec}")
        logger.info(f"DEBUG eval_script {test_spec.eval_script}")

        # Convert SEARCH/REPLACE format to unified diff if needed
        try:
            unified_diff = extract_and_convert_to_diff(model_patch)
            logger.info("Converted SEARCH/REPLACE format to unified diff")
        except Exception as e:
            logger.warning(f"Failed to convert SEARCH/REPLACE format: {e}")
            # Fall back to using the patch as-is
            unified_diff = model_patch
        
        # Save the original model output and the unified diff
        raw_patch_file = instance_dir / f"patch_raw_{cfg.run_id}.txt"
        raw_patch_file.write_text(model_patch)
        
        patch_file = instance_dir / f"patch_{cfg.run_id}.diff"
        patch_file.write_text(unified_diff)
        
        pred = {"instance_id": test_spec.instance_id, "model_patch": unified_diff}

        # Apply the patch (works for both golden patches and agent-generated patches)
        env.execute(command=f"cat > patch.diff <<'EOF'\n{unified_diff}\n\nEOF")
        env.execute(command="git status --porcelain")
        env.execute(command="git apply --check patch.diff")
        env.execute(command="git apply patch.diff")

        # Run the evaluation script
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
            "model_patch": unified_diff,
            "model_patch_raw": model_patch,
            "eval_report": report,
        }


app = make_runner_command(
    SingleTurnRunner,
    _HELP_TEXT,
    config=builtin_config_dir / "extra" / "single_turn.yaml",
)

if __name__ == "__main__":
    app()

