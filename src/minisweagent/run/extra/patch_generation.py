from minisweagent.config import builtin_config_dir
from minisweagent.run.extra.base_runner import ProgressTrackingAgent, make_runner_command
from minisweagent.run.extra.runner_config import ProcessInstanceConfig
from minisweagent.run.extra.swegym_runner import TestRunner
from minisweagent.run.extra.utils.parsing import get_changed_files_from_diff

_HELP_TEXT = """Run mini-SWE-agent on SWEGym instances for patch generation with oracle file information.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/swebench/[/bold green]
[/not dim]
"""


class PatchGenerationAgent(ProgressTrackingAgent):
    """Agent that provides oracle file information to the prompt."""

    def __init__(self, *args, oracle_files: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.oracle_files = oracle_files

    def render_template(self, template: str, **kwargs) -> str:
        """Override to include oracle_files in template context."""
        return super().render_template(template, oracle_files=self.oracle_files, **kwargs)


class PatchGenerationRunner(TestRunner):
    """Runner that generates patches when given oracle file information and evaluates with tests."""

    def create_agent(self, cfg: ProcessInstanceConfig, model, env, agent_config: dict) -> ProgressTrackingAgent:
        """Create agent with oracle file information."""
        oracle_files = get_changed_files_from_diff(cfg.instance["patch"])
        oracle_files_str = "\n".join([f"- {cfg.testbed_path}/{f}" for f in oracle_files])
        return PatchGenerationAgent(
            model,
            env,
            cfg.responses_create_params,
            progress_manager=cfg.progress_manager,
            instance_id=cfg.instance["instance_id"],
            oracle_files=oracle_files_str,
            **agent_config,
        )


app = make_runner_command(
    PatchGenerationRunner,
    _HELP_TEXT,
    config=builtin_config_dir / "extra" / "patch_generation.yaml",
)

if __name__ == "__main__":
    app()

