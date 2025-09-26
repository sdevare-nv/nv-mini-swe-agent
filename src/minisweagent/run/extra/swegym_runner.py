#!/usr/bin/env python3

"""Run mini-SWE-agent on SWE-GYM instances in batch mode."""
# Read this first: https://mini-swe-agent.com/latest/usage/swebench/  (usage docs)

import concurrent.futures
import json
import random
import re
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import cast, Dict, Any, List

import typer
import yaml
from datasets import load_dataset
from rich.live import Live
from swegym.harness.constants import SWEbenchInstance
from swegym.harness.docker_build import setup_logger
from swegym.harness.grading import get_eval_report
from swegym.harness.test_spec import make_test_spec

from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments import ENV_MAP, DockerEnvironment, SingularityEnvironment
from minisweagent.models import get_model
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.utils.save import save_traj

_HELP_TEXT = """Run mini-SWE-agent on SWEGym instances.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/swebench/[/bold green]
[/not dim]
"""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

DATASET_MAPPING = {
    "gym": "SWE-Gym/SWE-Gym",
    "verified": "princeton-nlp/SWE-Bench_Verified",
}


_OUTPUT_FILE_LOCK = threading.Lock()


class ProgressTrackingAgent(DefaultAgent):
    """Simple wrapper around DefaultAgent that provides progress updates."""

    def __init__(
        self,
        *args,
        progress_manager: RunBatchProgressManager,
        instance_id: str = "",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.progress_manager: RunBatchProgressManager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        """Override step to provide progress updates."""
        self.progress_manager.update_instance_status(
            self.instance_id, f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})"
        )
        return super().step()


def get_swegym_docker_image_name(instance: dict, subset: str) -> str:
    """Get the image name for a SWEGym instance."""
    if subset == "gym":
        image_name = instance.get("image_name", None)
        if image_name is None:
            iid = instance["instance_id"]
            id_docker_compatible = iid.replace("__", "_s_")
            image_name = f"xingyaoww/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    if subset == "verified":
        image_name = instance.get("image_name", None)
        if image_name is None:
            iid = instance["instance_id"]
            id_docker_compatible = iid.replace("__", "_1776_")
            image_name = f"swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name


def run_eval(
    instance: SWEbenchInstance,
    env: SingularityEnvironment | DockerEnvironment,
    model_patch: str,
    instance_dir: str,
    run_id: str,
    is_golden: bool = False,
):
    instances = [cast(SWEbenchInstance, instance)]
    test_spec = list(map(make_test_spec, instances))[0]

    pred = {"instance_id": test_spec.instance_id, "model_patch": model_patch}

    instance_id = test_spec.instance_id

    log_dir = instance_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_instance_{run_id}.log"
    report_path = log_dir / f"report_{run_id}.json"
    patch_file = log_dir / f"patch_{run_id}.diff"
    with open(patch_file, "w") as f:
        f.write(model_patch)

    logger = setup_logger(instance_id, log_file)
    logger.info(f"DEBUG test_spec {test_spec}")
    logger.info(f"DEBUG eval_script {test_spec.eval_script}")

    if is_golden:
        res = env.execute(command=f"cat > patch.diff <<'EOF'\n{model_patch}\n\nEOF")
        res = env.execute(command="git status --porcelain")
        res = env.execute(command="git apply --check patch.diff")
        res = env.execute(command="git apply patch.diff")
        # print(f"DEBUG git apply output: {res['output']}")

    eval_script = test_spec.eval_script.replace("#!/bin/bash", "")
    res = env.execute(command=eval_script, is_eval=True)

    test_output, returncode = res["output"], res["returncode"]
    logger.info(f"DEBUG eval output: {test_output}")
    logger.info(f"DEBUG returncode: {returncode}")
    test_output_path = log_dir / f"test_output_{run_id}.txt"
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

    return {
        "instance_id": instance_id,
        "model_patch": model_patch,
        "eval_report": report,
    }


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
    responses_create_params: Dict[str, Any],
    cache_dir_template: str | None,
    run_id: str,
    subset: str,
    run_golden: bool,
    step_timeout: int,
    eval_timeout: int,
) -> None:
    """Process a single SWEGym instance."""
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id

    image_name = get_swegym_docker_image_name(instance, subset)
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
    env = None
    eval_report = None
    extra_info = None
    try:
        env = env_cls(
            cache_dir_template=cache_dir_template,
            **(
                config.get("environment", {})
                | {
                    "image": image_name,
                    "step_timeout": step_timeout,
                    "eval_timeout": eval_timeout,
                    "instance_id": instance_id,
                }
            ),
        )
        if convert_to_sif:
            progress_manager.on_instance_end(instance_id, "Image Converted to SIF")
            env.cleanup()
            return None, None

        agent = ProgressTrackingAgent(
            model,
            env,
            responses_create_params,
            progress_manager=progress_manager,
            instance_id=instance_id,
            **config.get("agent", {}),
        )

        if not run_golden:
            exit_status, result = agent.run(task)
        else:
            exit_status, result = "Gold Patch Applied", instance.get("patch", "")

        # print(f"DEBUG: result: {result}", instance_id)
        # print(f"DEBUG: Running eval for {instance_id}")
        eval_report = run_eval(
            instance=instance,
            env=env,
            model_patch=result,
            instance_dir=instance_dir,
            run_id=run_id,
            is_golden=run_golden,
        )
        print(f"DEBUG: Eval completed for {instance_id}")
        data = save_traj(
            agent,
            instance_dir / f"{instance_id}_{run_id}.traj.json",
            exit_status=exit_status,
            result=result,
            extra_info=extra_info,
            instance_id=instance_id,
        )
        env.cleanup()
        progress_manager.on_instance_end(instance_id, exit_status)
        return data, eval_report

    except Exception as e:
        if env:
            env.cleanup()

        if convert_to_sif:
            progress_manager.on_instance_end(instance_id, "Error pulling image")
            return None, None

        print(f"Error processing instance {instance_id}: {e}\n{traceback.format_exc()}")
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
        data = save_traj(
            agent,
            instance_dir / f"{instance_id}_{run_id}.traj.json",
            exit_status=exit_status,
            result=result,
            extra_info=extra_info,
            instance_id=instance_id,
        )
        progress_manager.on_instance_end(instance_id, exit_status)
        return data, eval_report


def filter_instances(
    instances: list[dict], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False
) -> list[dict]:
    """Filter and slice a list of SWEGym instances."""
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


def _main(
    subset: str = "gym",
    split: str = "train",
    slice_spec: str = "",
    filter_spec: str = "",
    shuffle: bool = False,
    output: str = "",
    workers: int = 1,
    model: str | None = None,
    redo_existing: bool = False,
    config: Path = builtin_config_dir / "extra" / "swebench.yaml",
    convert_to_sif: bool = False,
    api_key: str | None = None,
    base_url: str | None = None,
    env: str = "singularity",
    instance_id: str = "",
    instance_dict: dict = None,
    responses_create_params: str = "",
    cache_dir_template: str | None = None,
    run_golden: bool = False,
    step_timeout: int = 600,
    eval_timeout: int = 600,
):
    if responses_create_params:
        responses_create_params = json.loads(responses_create_params)

    run_id = f"{int(time.time())}_{str(uuid.uuid4())}"
    env_cls = ENV_MAP[env]
    dataset_path = DATASET_MAPPING.get(subset, subset)
    print(f"Loading dataset {dataset_path}, split {split}...")

    instances = instance_dict if [instance_dict] else list(load_dataset(dataset_path, split=split))

    if instance_id:
        instance_id = instance_id.lower()
        instances = [instance for instance in instances if instance["instance_id"].lower() == instance_id]

    for instance in instances:
        instance["instance_id"] = instance["instance_id"].lower()

    assert len(instances) != 0, "No valid instances found!"

    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Running on {len(instances)} instances...")
    print(f"Results will be saved to {output_path}")

    progress_manager = RunBatchProgressManager(
        len(instances), output_path / f"exit_statuses_{time.time()}_{run_id}.yaml"
    )
    results = {}

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                data, eval_report = future.result()
                if data is None:
                    continue
                results[data["instance_id"]] = data
                results[data["instance_id"]]["eval_report"] = eval_report
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
                    responses_create_params,
                    cache_dir_template,
                    run_id,
                    subset,
                    run_golden,
                    step_timeout,
                    eval_timeout,
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

    return results


@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEGym subset to use or path to a dataset"),
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
    instance_id: str = typer.Option("", "--instance_id", help="Instance ID to run"),
    instance_dict: dict = typer.Option(None, "--instance_dict", help="Instance dictionary to run"),
    responses_create_params: str = typer.Option(
        "", "--responses_create_params", help="Input messages to override the initial system and user message"
    ),
    cache_dir_template: str | None = typer.Option(
        None,
        "--cache_dir_template",
        help="The path to the singularity cache dir. This is where the images will be converted and stored. This is a template string that will be formatted with the instance ID.",
    ),
    run_golden: bool = typer.Option(False, "--run_golden", help="Run golden patch"),
    step_timeout: int = typer.Option(600, "--step_timeout", help="Timeout for each turn of the agent"),
    eval_timeout: int = typer.Option(600, "--eval_timeout", help="Timeout for the eval"),
) -> None:
    _main(
        subset=subset,
        split=split,
        slice_spec=slice_spec,
        filter_spec=filter_spec,
        shuffle=shuffle,
        output=output,
        workers=workers,
        model=model,
        redo_existing=redo_existing,
        config=config,
        convert_to_sif=convert_to_sif,
        api_key=api_key,
        base_url=base_url,
        env=env,
        instance_id=instance_id,
        instance_dict=instance_dict,
        responses_create_params=responses_create_params,
        cache_dir_template=cache_dir_template,
        run_golden=run_golden,
        step_timeout=step_timeout,
        eval_timeout=eval_timeout,
    )


if __name__ == "__main__":
    app()
