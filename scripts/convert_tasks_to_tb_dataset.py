import json
from pathlib import Path
import yaml

base_path = Path("")

samples = []
for task_path in base_path.iterdir():
    if task_path.is_dir():
        task_name = task_path.name

        sample = dict(
            instance_id=f"{task_name}",
            problem_statement = "",
            test_script = "",
            test_files = []
        )

        with open(task_path / "task.yaml", "r") as f:
            task_yaml = yaml.safe_load(f)

        sample["problem_statement"] = task_yaml["instruction"]

        with open(task_path / "run-tests.sh", "r") as f:
            sample["test_script"] = f.read()

        for f in (task_path / "tests").iterdir():
            if f.is_file():
                sample["test_files"].append({"filename": f.name, "content": f.read_text()})

        samples.append(sample)

with open("terminal_bench_samples.jsonl", "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")