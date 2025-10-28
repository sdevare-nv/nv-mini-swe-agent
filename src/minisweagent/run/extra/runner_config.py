from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from minisweagent.config import builtin_config_dir
from minisweagent.environments import DockerEnvironment, SingularityEnvironment


class RunnerConfig(BaseModel):
    subset: str = Field(default="lite", description="SWEGym subset to use or path to a dataset")
    split: str = Field(default="dev", description="Dataset split")
    slice: str = Field(default="", description="Slice specification (e.g., '0:5' for first 5 instances)")
    filter: str = Field(default="", description="Filter instance IDs by regex")
    shuffle: bool = Field(default=False, description="Shuffle instances")
    output: str = Field(default="", description="Output directory")
    workers: int = Field(default=1, description="Number of worker threads for parallel processing")
    model: str | None = Field(default=None, description="Model to use")
    redo_existing: bool = Field(default=False, description="Redo existing instances")
    config: Path = Field(
        default=builtin_config_dir / "extra" / "swebench.yaml", description="Path to a config file"
    )
    convert_to_sif: bool = Field(default=False, description="Convert docker images to SIF")
    api_key: str | None = Field(default=None, description="API Key for model endpoint")
    base_url: str | None = Field(default=None, description="Base URL for model endpoint")
    env: str = Field(default="singularity", description="Environment to use")
    instance_id: str = Field(default="", description="Instance ID to run")
    instance_dict: str | None = Field(default=None, description="Instance dictionary to run")
    responses_create_params: str = Field(
        default="", description="Input messages to override the initial system and user message"
    )
    cache_dir_template: str | None = Field(
        default=None,
        description="The path to the singularity cache dir. This is where the images will be converted and stored. This is a template string that will be formatted with the instance ID.",
    )
    run_golden: bool = Field(default=False, description="Run golden patch")
    step_timeout: int = Field(default=600, description="Timeout for each turn of the agent")
    eval_timeout: int = Field(default=600, description="Timeout for the eval")
    step_limit: int = Field(default=250, description="Limit the number of steps the agent takes")
    collapse_limit: int = Field(
        default=0, description="Terminate agent if it generates the same output this many times (0 to disable)"
    )
    testbed_path: str = Field(default="/testbed", description="Path to the testbed directory")


class ProcessInstanceConfig(RunnerConfig):
    instance: dict[str, Any]
    output_dir: Path
    progress_manager: Any
    env_cls: type[SingularityEnvironment | DockerEnvironment]
    responses_create_params: dict[str, Any]
    run_id: str

    @property
    def model_name(self) -> str | None:
        return self.model

    @property
    def config_path(self) -> str | Path:
        return self.config

