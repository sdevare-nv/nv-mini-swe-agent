"""Environment implementations for mini-SWE-agent."""

from minisweagent.environments.docker import DockerEnvironment
from minisweagent.environments.enroot import EnrootEnvironment
from minisweagent.environments.singularity import SingularityEnvironment

ENV_MAP = {
    "singularity": SingularityEnvironment,
    "docker": DockerEnvironment,
    "enroot": EnrootEnvironment,
}