#!/usr/bin/env python3

"""Base class for SWEGym runners with common functionality."""

import concurrent.futures
import json
import random
import re
import time
import traceback
import uuid
from abc import abstractmethod
from pathlib import Path
from typing import Any

import typer
import yaml
from datasets import load_dataset

from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments import ENV_MAP, DockerEnvironment, SingularityEnvironment
from minisweagent.models import get_model
from minisweagent.run.extra.evaluators import Evaluator
from minisweagent.run.extra.runner_config import ProcessInstanceConfig, RunnerConfig
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.utils.save import save_traj


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


class SWEGymRunner:
    """Base class for running SWEGym instances with different evaluation strategies."""

    INTERNAL_DATASET_MAPPING = {
        "nv-internal-1": "",
    }

    EXTERNAL_DATASET_MAPPING = {
        "gym": "SWE-Gym/SWE-Gym",
        "verified": "princeton-nlp/SWE-Bench_Verified",
    }

    SUBSET_TO_CONDA_ENV = {
        "nv-internal-1": None,
        "gym": "testbed",
        "verified": "testbed",
    }

    @abstractmethod
    def get_evaluator(self, subset: str) -> Evaluator:
        """Get the appropriate evaluator for the given subset. Override in subclass."""
        pass

    @staticmethod
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
        if subset == "nv-internal-1":
            image_name = ""
        return image_name

    @staticmethod
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
            print(f"Instance filter: {before_filter} -> {after_filter} instances", flush=True)
        if slice_spec:
            values = [int(x) if x else None for x in slice_spec.split(":")]
            instances = instances[slice(*values)]
            if (after_slice := len(instances)) != before_filter:
                print(f"Instance slice: {before_filter} -> {after_slice} instances", flush=True)
        return instances

    def create_agent(self, cfg: ProcessInstanceConfig, model, env, agent_config: dict) -> ProgressTrackingAgent:
        """Create agent for the instance. Override in subclasses for custom agent creation."""
        return ProgressTrackingAgent(
            model,
            env,
            cfg.responses_create_params,
            progress_manager=cfg.progress_manager,
            instance_id=cfg.instance["instance_id"],
            **agent_config,
        )

    def process_instance(self, cfg: ProcessInstanceConfig) -> tuple[dict | None, dict | None]:
        """Process a single SWEGym instance."""
        instance_id = cfg.instance["instance_id"]
        instance_dir = cfg.output_dir / instance_id

        image_name = self.get_swegym_docker_image_name(cfg.instance, cfg.subset)
        # TODO: use a better way to replace the testbed_path in the config
        config_text = get_config_path(cfg.config_path).read_text()
        config_text = config_text.replace("{{testbed_path}}", cfg.testbed_path)
        config = yaml.safe_load(config_text)

        model_kwargs = config.setdefault("model", {}).setdefault("model_kwargs", {})

        if cfg.api_key:
            model_kwargs["api_key"] = cfg.api_key
        if cfg.base_url:
            model_kwargs["base_url"] = cfg.base_url

        model = get_model(cfg.model_name, config=config.get("model", {}))

        task = cfg.instance["problem_statement"]

        cfg.progress_manager.on_instance_start(instance_id)
        cfg.progress_manager.update_instance_status(instance_id, "Pulling/starting docker")

        agent = None
        env = None
        eval_report = None
        extra_info = None
        try:
            print(f"[EVAL]{instance_id} Creating environment...", flush=True)
            env = cfg.env_cls(
                cache_dir_template=cfg.cache_dir_template,
                **(
                    config.get("environment", {})
                    | {
                        "image": image_name,
                        "step_timeout": cfg.step_timeout,
                        "eval_timeout": cfg.eval_timeout,
                        "instance_id": instance_id,
                        "cwd": cfg.testbed_path,
                        "conda_env": self.SUBSET_TO_CONDA_ENV[cfg.subset],
                    }
                ),
            )
            print(f"[EVAL]{instance_id} Environment created", flush=True)

            if cfg.convert_to_sif:
                cfg.progress_manager.on_instance_end(instance_id, "Image Converted to SIF")
                env.cleanup()
                return None, None

            agent_config = config.get("agent", {})
            agent_config["step_limit"] = cfg.step_limit
            agent_config["collapse_limit"] = cfg.collapse_limit
            agent = self.create_agent(cfg, model, env, agent_config)

            print(f"[EVAL]{instance_id} Running agent...", flush=True)
            if not cfg.run_golden:
                exit_status, result = agent.run(task)
            else:
                exit_status, result = "Gold Patch Applied", cfg.instance.get("patch", "")

            print(f"[EVAL]{instance_id} Running eval", flush=True)

            data = save_traj(
                agent,
                instance_dir / f"{instance_id}_{cfg.run_id}.traj.json",
                exit_status=exit_status,
                result=result,
                extra_info=extra_info,
                instance_id=instance_id,
            )

            evaluator = self.get_evaluator(cfg.subset)
            eval_report = evaluator.evaluate(
                cfg=cfg,
                trajectory_data=data,
                env=env,
                model_patch=result,
                instance_dir=instance_dir,
            )
            print(f"[EVAL]{instance_id} Eval completed", flush=True)

            env.cleanup()
            cfg.progress_manager.on_instance_end(instance_id, exit_status)
            return data, eval_report

        except Exception as e:
            if env:
                env.cleanup()

            if cfg.convert_to_sif:
                cfg.progress_manager.on_instance_end(instance_id, "Error pulling image")
                return None, None

            print(f"[MINI-SWE-AGENT]{instance_id} Error processing instance: {e}\n{traceback.format_exc()}")
            exit_status, result = type(e).__name__, str(e)
            extra_info = {"traceback": traceback.format_exc()}
            data = save_traj(
                agent,
                instance_dir / f"{instance_id}_{cfg.run_id}.traj.json",
                exit_status=exit_status,
                result=result,
                extra_info=extra_info,
                instance_id=instance_id,
            )
            cfg.progress_manager.on_instance_end(instance_id, exit_status)
            return data, eval_report

    def run(self, cfg: RunnerConfig) -> dict:
        """Run the SWEGym runner on specified instances."""
        responses_create_params = json.loads(cfg.responses_create_params) if cfg.responses_create_params else {}

        run_id = f"{int(time.time())}_{str(uuid.uuid4())}"

        if cfg.subset in self.INTERNAL_DATASET_MAPPING:
            dataset_path = self.INTERNAL_DATASET_MAPPING[cfg.subset]
        else:
            dataset_path = self.EXTERNAL_DATASET_MAPPING[cfg.subset]

        env_cls = ENV_MAP[cfg.env]

        assert dataset_path == "" and cfg.instance_dict, "No instance dict provided for internal dataset"

        instance_dict = json.loads(cfg.instance_dict) if cfg.instance_dict else None
        instances = [instance_dict] if instance_dict else list(load_dataset(dataset_path, split=cfg.split))

        if cfg.instance_id:
            instance_id = cfg.instance_id.lower()
            instances = [instance for instance in instances if instance["instance_id"].lower() == instance_id]

        for instance in instances:
            instance["instance_id"] = instance["instance_id"].lower()

        assert len(instances) != 0, "No valid instances found!"

        instances = self.filter_instances(instances, filter_spec=cfg.filter, slice_spec=cfg.slice, shuffle=cfg.shuffle)
        output_path = Path(cfg.output)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Running on {len(instances)} instances...", flush=True)
        print(f"Results will be saved to {output_path}", flush=True)

        progress_manager = RunBatchProgressManager(
            len(instances), output_path / f"exit_statuses_{time.time()}_{run_id}.yaml"
        )
        results = {}

        def process_futures(futures: dict[concurrent.futures.Future, str]):
            completed = 0
            total = len(futures)
            for future in concurrent.futures.as_completed(futures):
                try:
                    data, eval_report = future.result()
                    completed += 1
                    print(f"Progress: {completed}/{total} instances completed", flush=True)
                    if data is None:
                        continue
                    results[data["instance_id"]] = data
                    results[data["instance_id"]]["eval_report"] = eval_report
                except concurrent.futures.CancelledError:
                    pass
                except Exception as e:
                    instance_id = futures[future]
                    print(f"Error in future for instance {instance_id}: {e}", flush=True)
                    traceback.print_exc()
                    progress_manager.on_uncaught_exception(instance_id, e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.workers) as executor:
            futures = {
                executor.submit(
                    self.process_instance,
                    ProcessInstanceConfig(
                        **cfg.model_dump(exclude={"responses_create_params"}),
                        instance=instance,
                        output_dir=output_path,
                        progress_manager=progress_manager,
                        env_cls=env_cls,
                        responses_create_params=responses_create_params,
                        run_id=run_id,
                    ),
                ): instance["instance_id"]
                for instance in instances
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                print("Cancelling all pending jobs. Press ^C again to exit immediately.", flush=True)
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)

        return results


def create_typer_options_from_config(cfg_model: type[RunnerConfig], overrides: dict[str, Any]):
    """Create typer options from a Pydantic model with overrides."""
    options = {}
    for field_name, field_info in cfg_model.model_fields.items():
        default_value = overrides.get(field_name, field_info.default)
        description = field_info.description or ""

        short_flags = {
            "output": "-o",
            "workers": "-w",
            "model": "-m",
            "config": "-c",
        }

        flags = [f"--{field_name}"]
        if field_name in short_flags:
            flags.insert(0, short_flags[field_name])

        options[field_name] = typer.Option(default_value, *flags, help=description)

    return options


def make_runner_command(runner_cls: type[SWEGymRunner], help_text: str, **default_overrides):
    """Factory to create a complete typer command for a runner class."""
    app = typer.Typer(rich_markup_mode="rich", add_completion=False)
    options = create_typer_options_from_config(RunnerConfig, default_overrides)

    @app.command(help=help_text)
    def main(
        subset: str = options["subset"],
        split: str = options["split"],
        slice: str = options["slice"],
        filter: str = options["filter"],
        shuffle: bool = options["shuffle"],
        output: str = options["output"],
        workers: int = options["workers"],
        model: str | None = options["model"],
        redo_existing: bool = options["redo_existing"],
        config: Path = options["config"],
        convert_to_sif: bool = options["convert_to_sif"],
        api_key: str | None = options["api_key"],
        base_url: str | None = options["base_url"],
        env: str = options["env"],
        instance_id: str = options["instance_id"],
        instance_dict: str | None = options["instance_dict"],
        responses_create_params: str = options["responses_create_params"],
        cache_dir_template: str | None = options["cache_dir_template"],
        run_golden: bool = options["run_golden"],
        step_timeout: int = options["step_timeout"],
        eval_timeout: int = options["eval_timeout"],
        step_limit: int = options["step_limit"],
        collapse_limit: int = options["collapse_limit"],
        testbed_path: str = options["testbed_path"],
    ) -> None:
        runner_cfg = RunnerConfig(
            subset=subset,
            split=split,
            slice=slice,
            filter=filter,
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
            step_limit=step_limit,
            collapse_limit=collapse_limit,
            testbed_path=testbed_path,
        )
        runner_cls().run(runner_cfg)

    return app
