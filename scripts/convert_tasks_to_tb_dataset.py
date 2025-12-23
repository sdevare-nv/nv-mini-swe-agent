import argparse
import json
from pathlib import Path

import yaml


def convert_tasks_to_tb_dataset(input_path: Path, output_path: Path):
    samples = []
    for task_path in input_path.iterdir():
        if task_path.is_dir():
            task_name = task_path.name

            sample = dict(
                instance_id=f"{task_name}",
                problem_statement="",
                test_script="",
                test_files=[],
                subset="terminal-bench",
                split="train",
                responses_create_params={"input": []},
                agent_ref={"type": "responses_api_agents", "name": "terminal_bench_simple_agent_train"}, # change to terminal_bench_simple_agent_val for validation set
            )

        with open(task_path / "task.yaml", "r") as f:
            task_yaml = yaml.safe_load(f)

        sample["problem_statement"] = task_yaml["instruction"]
        sample = sample | {**task_yaml}

        with open(task_path / "run-tests.sh", "r") as f:
            sample["test_script"] = f.read()

        for f in (task_path / "tests").iterdir():
            if f.is_file():
                sample["test_files"].append({"filename": f.name, "content": f.read_text()})

        samples.append(sample)

    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    """
    """
    args = argparse.ArgumentParser()
    args.add_argument("--input_path", type=Path, required=True)
    args.add_argument("--output_path", type=Path, required=True)
    args = args.parse_args()

    convert_tasks_to_tb_dataset(args.input_path, args.output_path)
