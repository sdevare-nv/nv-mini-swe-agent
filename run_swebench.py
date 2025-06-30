#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import yaml
from datasets import load_dataset

from agent import Agent, AgentConfig, InstanceConfig

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
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


def update_output_file(output_path: Path, instance_id: str, agent_config: AgentConfig, result: str):
    """Update the output JSON file with results from a single instance."""
    output_data = {}
    if output_path.exists():
        output_data = json.loads(output_path.read_text())

    output_data[instance_id] = {
        "model_name_or_path": agent_config.model_name,
        "instance_id": instance_id,
        "model_patch": result,
    }

    output_path.write_text(json.dumps(output_data, indent=2))


def process_instance(instance: dict, agent_config: AgentConfig, output_path: Path) -> dict:
    """Process a single SWEBench instance."""
    instance_id = instance["instance_id"]
    problem_statement = instance["problem_statement"]
    image_name = get_image_name(instance)

    instance_config = InstanceConfig(image=image_name, problem_statement=problem_statement)
    agent = Agent(agent_config, instance_config)
    result = agent.run()

    # Capture cost and steps before resetting
    instance_cost = agent.model.cost
    instance_steps = agent.n_steps

    # Reset model cost for next instance
    agent.model.cost = 0.0

    print(f"Instance {instance_id} completed")
    print(f"Cost: ${instance_cost:.4f}")
    print(f"Steps: {instance_steps}")

    update_output_file(output_path, instance_id, agent_config, result)

    return {
        "instance_id": instance_id,
        "result": result,
        "cost": instance_cost,
        "steps": instance_steps,
    }


def process_instances(agent_config: AgentConfig, instances: list[dict], output_path: Path):
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

        result = process_instance(instance, agent_config, output_path)
        results.append(result)
        running_cost += result["cost"]

        # Print summary after each instance
        print(f"\nRunning total - Instances: {i + 1}/{len(instances)}, Cost: ${running_cost:.4f}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run nano-SWE-agent on SWEBench instances")
    parser.add_argument(
        "--subset",
        choices=["lite", "verified", "full", "multimodal", "multilingual"],
        default="lite",
        help="SWEBench subset to use",
    )
    parser.add_argument("--split", choices=["dev", "test"], default="dev", help="Dataset split")
    parser.add_argument("--slice", default="", help="Slice specification (e.g., '0:5' for first 5 instances)")
    parser.add_argument("-o", "--output", default="results.json", help="Output JSON file path")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load dataset
    dataset_path = DATASET_MAPPING[args.subset]
    print(f"Loading dataset {dataset_path}, split {args.split}...")
    instances = list(load_dataset(dataset_path, split=args.split))

    # Apply slice if specified
    if args.slice:
        values = [int(x) if x else None for x in args.slice.split(":")]
        instances = instances[slice(*values)]

    # Load agent configuration
    config = yaml.safe_load((Path(__file__).parent / "config" / "github_issue.yaml").read_text())
    agent_config = AgentConfig(**config)

    output_path = Path(args.output)
    print(f"Running on {len(instances)} instances...")
    print(f"Results will be saved to {output_path}")

    process_instances(agent_config, instances, output_path)


if __name__ == "__main__":
    main()
