import json
from pathlib import Path

from minisweagent.config import builtin_config_dir
from minisweagent.environments import DockerEnvironment, SingularityEnvironment
from minisweagent.run.extra.base_runner import SWEGymRunner, make_runner_command
from minisweagent.run.extra.evaluators import Evaluator
from minisweagent.run.extra.runner_config import ProcessInstanceConfig
from minisweagent.run.extra.utils.parsing import get_changed_files_from_diff

_HELP_TEXT = """Run mini-SWE-agent on SWEGym instances for file localization evaluation.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/swebench/[/bold green]
[/not dim]
"""


class LocalizationEvaluator(Evaluator):
    """Evaluator for file localization by comparing predicted files with ground truth."""

    def evaluate(
        self,
        cfg: ProcessInstanceConfig,
        trajectory_data: dict,
        env: SingularityEnvironment | DockerEnvironment,
        model_patch: str,
        instance_dir: Path,
    ) -> dict:
        """Evaluate by computing overlap between predicted and ground truth files."""
        instance = cfg.instance
        instance_id = instance["instance_id"]
        gold_patch = instance["patch"]
        gt_files = get_changed_files_from_diff(gold_patch)

        try:
            predicted_files = json.loads(trajectory_data["messages"][-1]["content"])["files"]
            predicted_files = [file.replace(f"{cfg.testbed_path}/", "") for file in predicted_files]
        except Exception as e:
            print(f"[EVAL]{instance_id} Error parsing predicted files: {e}")
            predicted_files = []

        gt_set = set(gt_files)
        pred_set = set(predicted_files)
        overlap_score = len(gt_set & pred_set) / len(gt_set) if gt_set else 0
        exact_match = 1.0 if gt_set == pred_set else 0.0
        false_positives = list(pred_set - gt_set)
        false_negatives = list(gt_set - pred_set)
        eval_report = {
            "overlap_score": overlap_score,
            "exact_match": exact_match,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }
        print(f"[EVAL]{instance_id} Result: {eval_report}")

        report = {
            "instance_id": instance_id,
            "eval_report": eval_report,
            "gt_files": gt_files,
            "predicted_files": predicted_files,
        }
        with open(instance_dir / f"report_{cfg.run_id}.json", "w") as f:
            json.dump(report, f)

        return report


class LocalizationRunner(SWEGymRunner):
    """Runner that evaluates file localization by comparing predicted files with ground truth."""

    def get_evaluator(self, subset: str) -> Evaluator:
        """Get the localization evaluator."""
        return LocalizationEvaluator()


app = make_runner_command(
    LocalizationRunner,
    _HELP_TEXT,
    config=builtin_config_dir / "extra" / "localization.yaml",
)

if __name__ == "__main__":
    app()
