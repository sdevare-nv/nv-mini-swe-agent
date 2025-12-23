#!/usr/bin/env python3

"""Evaluator system for running different benchmark types.

Simple structure for selecting evaluators based on benchmark subset.

To add a new evaluator:
1. Create a class that inherits from Evaluator
2. Implement the evaluate() method
3. Import it in swegym_runner.py and add to get_evaluator()
"""

from abc import ABC, abstractmethod
from pathlib import Path

from minisweagent.environments import DockerEnvironment, SingularityEnvironment
from minisweagent.run.extra.runner_config import ProcessInstanceConfig


class Evaluator(ABC):
    """Base class for evaluators that run tests on patches."""

    @abstractmethod
    def evaluate(
        self,
        cfg: ProcessInstanceConfig,
        trajectory_data: dict,
        env: SingularityEnvironment | DockerEnvironment,
        model_patch: str,
        instance_dir: Path,
    ) -> dict:
        """Run evaluation and return report."""
        pass
