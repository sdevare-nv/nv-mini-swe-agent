#!/usr/bin/env python3

"""Run mini-SWE-agent on SWE-bench instances in batch mode."""
# Read this first: https://mini-swe-agent.com/latest/usage/swebench/  (usage docs)

import concurrent.futures
import json
import random
import re
import threading
import time
import traceback
from pathlib import Path
from typing import cast

import typer
import yaml
from datasets import load_dataset
from rich.live import Live

from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments import ENV_MAP, DockerEnvironment, SingularityEnvironment
from minisweagent.models import get_model
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.utils.save import save_traj
from swegym.harness.test_spec import make_test_spec, TestSpec
from swegym.harness.constants import SWEbenchInstance, RUN_EVALUATION_LOG_DIR
from swegym.harness.docker_build import setup_logger
from swegym.harness.grading import get_eval_report

_HELP_TEXT = """Run mini-SWE-agent on SWEBench instances.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/swebench/[/bold green]
[/not dim]
"""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
}


_OUTPUT_FILE_LOCK = threading.Lock()


class ProgressTrackingAgent(DefaultAgent):
    """Simple wrapper around DefaultAgent that provides progress updates."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager: RunBatchProgressManager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        """Override step to provide progress updates."""
        self.progress_manager.update_instance_status(
            self.instance_id, f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})"
        )
        return super().step()


def get_swebench_docker_image_name(instance: dict) -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
        # image_name = f"ghcr.io/epoch-research/swe-bench.eval.x86_64.{iid}:latest"
    return image_name


def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str):
    """Update the output JSON file with results from a single instance."""
    with _OUTPUT_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_preds_file(output_path: Path, instance_id: str):
    """Remove an instance from the predictions file."""
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if instance_id in output_data:
            del output_data[instance_id]
            output_path.write_text(json.dumps(output_data, indent=2))


def run_eval(
    instance: SWEbenchInstance,
    env: SingularityEnvironment | DockerEnvironment,
    model_patch: str,
    instance_dir: str,
):
    # TODO: sugam - run evaluation for the sample
    instances = [cast(SWEbenchInstance, instance)]
    test_spec = list(map(make_test_spec, instances))[0]

    pred = {"instance_id": test_spec.instance_id, "model_patch": model_patch}

    instance_id = test_spec.instance_id

    log_dir = instance_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run_instance.log"
    report_path = log_dir / "report.json"
    patch_file = log_dir / "patch.diff"
    with open(patch_file, "w") as f:
        f.write(model_patch)

    logger = setup_logger(instance_id, log_file)
    logger.info(f"DEBUG test_spec {test_spec}")
    logger.info(f"DEBUG eval_script {test_spec.eval_script}")
    eval_script = test_spec.eval_script.replace("#!/bin/bash", "")
    res = env.execute(command=eval_script)

    test_output, returncode = res["output"], res["returncode"]
    logger.info(f"DEBUG eval output: {test_output}")
    logger.info(f"DEBUG returncode: {returncode}")
    test_output_path = log_dir / "test_output.txt"
    with open(test_output_path, "w") as f:
        f.write(test_output)
        logger.info(f"Test output for {instance_id} written to {test_output_path}")

    report = get_eval_report(
        test_spec=test_spec,
        prediction=pred,
        log_path=test_output_path,
        include_tests_status=True,
    )
    logger.info(f"report: {report}\nResult for {instance_id}: resolved: {report[instance_id]['resolved']}")

    with open(report_path, "w") as f:
        f.write(json.dumps(report, indent=4))


def process_instance(
    instance: dict,
    output_dir: Path,
    model_name: str | None,
    config_path: str | Path,
    progress_manager: RunBatchProgressManager,
    convert_to_sif: bool,
    api_key: str,
    base_url: str,
    env_cls: SingularityEnvironment | DockerEnvironment,
) -> None:
    """Process a single SWEBench instance."""
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    # avoid inconsistent state if something here fails and there's leftover previous files
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)

    image_name = get_swebench_docker_image_name(instance)
    config = yaml.safe_load(get_config_path(config_path).read_text())
    model_kwargs = config.setdefault("model", {}).setdefault("model_kwargs", {})
    if api_key:
        model_kwargs["api_key"] = api_key
    if base_url:
        model_kwargs["base_url"] = base_url

    model = get_model(model_name, config=config.get("model", {}))
    task = instance["problem_statement"]

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pulling/starting docker")

    agent = None
    extra_info = None

    try:
        env = env_cls(**(config.get("environment", {}) | {"image": image_name}))
        if convert_to_sif:
            progress_manager.on_instance_end(instance_id, "Image Converted to SIF")
            return
        agent = ProgressTrackingAgent(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=instance_id,
            **config.get("agent", {}),
        )
        exit_status, result = agent.run(task)
        print(f"DEBUG: Running eval for {instance_id}")
        run_eval(instance=instance, env=env, model_patch=result, instance_dir=instance_dir)
        print(f"DEBUG: Eval completed for {instance_id}")

    except Exception as e:
        if convert_to_sif:
            progress_manager.on_instance_end(instance_id, "Error pulling image")
        print(f"Error processing instance {instance_id}: {e}\n{traceback.format_exc()}")
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        if convert_to_sif:
            return
        save_traj(
            agent,
            instance_dir / f"{instance_id}.traj.json",
            exit_status=exit_status,
            result=result,
            extra_info=extra_info,
            instance_id=instance_id,
        )
        update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, result)
        progress_manager.on_instance_end(instance_id, exit_status)


def filter_instances(
    instances: list[dict], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False
) -> list[dict]:
    """Filter and slice a list of SWEBench instances."""
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    instances = [instance for instance in instances if re.match(filter_spec, instance["instance_id"])]
    if (after_filter := len(instances)) != before_filter:
        print(f"Instance filter: {before_filter} -> {after_filter} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        if (after_slice := len(instances)) != before_filter:
            print(f"Instance slice: {before_filter} -> {after_slice} instances")
    return instances


@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset"),
    split: str = typer.Option("dev", "--split", help="Dataset split"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances"),
    output: str = typer.Option("", "-o", "--output", help="Output directory"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances"),
    config: Path = typer.Option(
        builtin_config_dir / "extra" / "swebench.yaml", "-c", "--config", help="Path to a config file"
    ),
    convert_to_sif: bool = typer.Option(False, "--convert_to_sif", help="Convert docker images to SIF"),
    api_key: str | None = typer.Option(None, "--api_key", help="API Key for model endpoint"),
    base_url: str | None = typer.Option(None, "--base_url", help="Base URL for model endpoint"),
    env: str = typer.Option("singularity", "--env", help="Environment to use"),
) -> None:
    env_cls = ENV_MAP[env]
    dataset_path = DATASET_MAPPING.get(subset, subset)
    print(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))

    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)
    output_path = Path(output)
    if not redo_existing and (output_path / "preds.json").exists():
        existing_instances = list(json.loads((output_path / "preds.json").read_text()).keys())
        print(f"Skipping {len(existing_instances)} existing instances")
        instances = [instance for instance in instances if instance["instance_id"] not in existing_instances]

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Running on {len(instances)} instances...")
    print(f"Results will be saved to {output_path}")

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                instance_id = futures[future]
                print(f"Error in future for instance {instance_id}: {e}")
                traceback.print_exc()
                progress_manager.on_uncaught_exception(instance_id, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_instance,
                    instance,
                    output_path,
                    model,
                    config,
                    progress_manager,
                    convert_to_sif,
                    api_key,
                    base_url,
                    env_cls,
                ): instance["instance_id"]
                for instance in instances
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                print("Cancelling all pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)


if __name__ == "__main__":
    app()
