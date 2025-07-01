#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import yaml
from datasets import load_dataset

from nanoswea import package_dir
from nanoswea.agent import Agent, AgentConfig
from nanoswea.environment import DockerEnvironment
from nanoswea.model import LitellmModel, ModelConfig

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "_test": "klieret/swe-bench-dummy-test-dataset",
}


def get_image_name(instance: dict) -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name


def update_output_file(output_path: Path, instance_id: str, model_config: ModelConfig, result: str):
    """Update the output JSON file with results from a single instance."""
    output_data = {}
    if output_path.exists():
        output_data = json.loads(output_path.read_text())

    output_data[instance_id] = {
        "model_name_or_path": model_config.model_name,
        "instance_id": instance_id,
        "model_patch": result,
    }

    output_path.write_text(json.dumps(output_data, indent=2))


def process_instance(instance: dict, agent_config: AgentConfig, model_config: ModelConfig, output_path: Path) -> dict:
    """Process a single SWEBench instance."""
    instance_id = instance["instance_id"]
    problem_statement = instance["problem_statement"]
    image_name = get_image_name(instance)

    model = LitellmModel(model_config)
    env = DockerEnvironment(image_name)
    agent = Agent(agent_config, model, env, problem_statement)
    result = agent.run()

    print(f"Instance {instance_id} completed")
    print(f"Cost: ${model.cost:.4f}")
    print(f"Steps: {model.n_calls}")

    update_output_file(output_path, instance_id, model_config, result)

    return {
        "instance_id": instance_id,
        "result": result,
        "cost": model.cost,
        "steps": model.n_calls,
    }


def process_instances(agent_config: AgentConfig, model_config: ModelConfig, instances: list[dict], output_path: Path):
    """Process a list of SWEBench instances."""
    results = []
    running_cost = 0.0

    for i, instance in enumerate(instances):
        instance_id = instance["instance_id"]
        image_name = get_image_name(instance)

        print(f"\n{'=' * 60}")
        print(f"Running instance {i + 1}/{len(instances)}: {instance_id}")
        print(f"Image: {image_name}")
        print(f"{'=' * 60}")

        result = process_instance(instance, agent_config, model_config, output_path)
        results.append(result)
        running_cost += result["cost"]

        # Print summary after each instance
        print(f"\nRunning total - Instances: {i + 1}/{len(instances)}, Cost: ${running_cost:.4f}")


def run_swebench(subset: str = "lite", split: str = "dev", slice_spec: str = "", output: str = "results.json"):
    """Run nano-SWE-agent on SWEBench instances.

    Args:
        subset: SWEBench subset to use ("lite", "verified", "full", "multimodal", "multilingual")
        split: Dataset split ("dev" or "test")
        slice_spec: Slice specification (e.g., "0:5" for first 5 instances)
        output: Output JSON file path
    """
    # Load dataset
    dataset_path = DATASET_MAPPING[subset]
    print(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))

    # Apply slice if specified
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]

    # Load agent configuration
    config = yaml.safe_load((package_dir / "config" / "github_issue.yaml").read_text())
    agent_config = AgentConfig(**config["agent"])
    model_config = ModelConfig(**config["model"])

    output_path = Path(output)
    print(f"Running on {len(instances)} instances...")
    print(f"Results will be saved to {output_path}")

    process_instances(agent_config, model_config, instances, output_path)


def run_from_cli(cli_args: list[str] | None = None):
    """Run from command line interface with optional argument override."""
    parser = argparse.ArgumentParser(description="Run nano-SWE-agent on SWEBench instances")
    parser.add_argument(
        "--subset",
        choices=list(DATASET_MAPPING.keys()),
        default="lite",
        help="SWEBench subset to use",
    )
    parser.add_argument("--split", choices=["dev", "test"], default="dev", help="Dataset split")
    parser.add_argument("--slice", default="", help="Slice specification (e.g., '0:5' for first 5 instances)")
    parser.add_argument("-o", "--output", default="results.json", help="Output JSON file path")

    args = parser.parse_args(args=cli_args)
    run_swebench(subset=args.subset, split=args.split, slice_spec=args.slice, output=args.output)


if __name__ == "__main__":
    run_from_cli()
