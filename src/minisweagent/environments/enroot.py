import os
import shlex
import subprocess
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnrootEnvironmentConfig:
    image: str
    """Image to use for the container, e.g., 'docker://ubuntu:22.04'"""
    cwd: str = "/"
    """Working directory in which to execute commands."""
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables to set in the container. These will override host variables."""
    timeout: int = 30
    """Timeout for executing commands in the container."""
    executable: str = "enroot"
    """Path to the enroot executable."""
    start_args: list[str] = field(default_factory=list)
    """Additional arguments to pass to the `enroot start` command."""


class EnrootEnvironment:
    def __init__(self, *, config_class: type = EnrootEnvironmentConfig, **kwargs):
        """This class executes bash commands in an Enroot container.
        See `EnrootEnvironmentConfig` for keyword arguments.

        Note: Unlike the original Docker version, this class forwards all host environment
        variables by default, which is the standard behavior for Enroot.
        The `env` config option can still be used to set/override specific variables.
        """
        self.container_name: str | None = None
        self.config = config_class(**kwargs)
        self._setup_container()

    def _setup_container(self):
        """Imports the enroot image and creates the container filesystem."""
        # Step 1: Import the image. Enroot handles caching, so this is fast if already present.
        
        container_dir = os.environ["ENROOT_CACHE_PATH"]
        container_output_path = os.path.join(container_dir, f"{self.config.image}.sqsh".replace("/", "_"))
        
        #TODO(sdd): add a flag to override this behavior
        if not os.path.exists(container_output_path):
            import_cmd = [self.config.executable, "import", "-o", container_output_path, f"docker://{self.config.image}"]
            print(f"Importing image with command: {shlex.join(import_cmd)}")
            try:
                subprocess.run(
                    import_cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # Importing a large image can take time
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"Enroot import failed.\nStderr: {e.stderr}\nStdout: {e.stdout}")
                raise e
            print(f"Successfully imported image '{self.config.image}'")
        else:
            print(f"Image already present '{self.config.image}'")

        # Step 2: Create the container filesystem with a unique name
        self.container_name = f"minisweagent-{uuid.uuid4().hex[:8]}"
        create_cmd = [
            self.config.executable,
            "create",
            "--name",
            self.container_name,
            container_output_path,
        ]
        print(f"Creating container with command: {shlex.join(create_cmd)}")
        try:
            subprocess.run(
                create_cmd,
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Enroot create failed.\nStderr: {e.stderr}\nStdout: {e.stdout}")
            raise e
        print(f"Created container '{self.container_name}'")

    def execute(self, command: str, cwd: str = "") -> dict[str, Any]:
        """Execute a command in the Enroot container and return the result as a dict."""
        cwd = cwd or self.config.cwd
        assert self.container_name, "Container not created"

        cmd = [
            self.config.executable,
            "start",
            "--rw",  # Make container writable
            *self.config.start_args,
        ]
        for key, value in self.config.env.items():
            cmd.extend(["--env", f"{key}={value}"])
        cmd.extend([self.container_name])
        cmd.extend(["bash", "-lc", command])
        print(f"Executing cmd in enroot environment with command: {shlex.join(cmd)}")

        result = subprocess.run(
            cmd,
            text=True,
            timeout=self.config.timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        output = result.stdout
        output = output.replace("/root/.bashrc: line 3: cd: /lustre/fsw/portfolios/llmservice/users/root: No such file or directory", "")
        return {"output": output, "returncode": result.returncode}

    def cleanup(self):
        """Removes the Enroot container and its filesystem."""
        if getattr(self, "container_name", None) is not None:
            print(f"Removing container {self.container_name}")
            # Force remove, run in background, and ignore all output
            cmd = f"{self.config.executable} remove --force {self.container_name} >/dev/null 2>&1 &"
            subprocess.Popen(cmd, shell=True)

    def __del__(self):
        """Cleanup container when object is destroyed."""
        self.cleanup()
