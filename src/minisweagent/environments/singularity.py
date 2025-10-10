import glob
import logging
import os
import shlex
import signal
import socket
import subprocess
import tempfile
import time
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger("singularity_environment")

END_TRAJECTORY_COMMAND = "echo MINI_SWE_AGENT_FINAL_OUTPUT && git add -A && git diff --cached"


def find_free_port():
    """Finds and returns an available TCP port on the host."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


@dataclass
class SingularityEnvironmentConfig:
    """Configuration for the Singularity environment."""

    image: str
    """Image to use for the container, e.g., 'ubuntu:22.04'"""
    cwd: str = "/"
    """Default working directory in which to execute commands."""
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables to set in the container. These will override host variables."""
    step_timeout: int = 600
    """Timeout for executing a single command via the API."""
    eval_timeout: int = 600
    """Timeout for executing the eval via the API."""
    executable: str = "singularity"
    """Path to the singularity executable."""
    start_args: list[str] = field(default_factory=list)
    """Additional arguments to pass to the `singularity run` command (e.g., '--nv' for GPU)."""
    cache_dir_template: str | None = None
    """Directory to cache singularity SIF files. This is a template string that will be formatted with the instance ID."""
    instance_id: str | None = None
    """Instance ID to use for the cache directory."""


class SingularityEnvironment:
    """
    Manages a long-running Singularity container with a FastAPI server.

    This class starts a FastAPI server within the container to execute commands,
    avoiding the overhead of `singularity exec` for each command. It automatically
    installs server dependencies (FastAPI, Uvicorn) into the container's base
    environment on startup.
    """

    def __init__(self, *, config_class: type = SingularityEnvironmentConfig, **kwargs):
        """
        Initializes the environment, pulls the image, and starts the container server.
        See `SingularityEnvironmentConfig` for keyword arguments.
        """
        self.sif_path: str | None = None
        self.server_script_path: str | None = None
        self.server_process: subprocess.Popen | None = None
        self.port: int | None = None
        self.config = config_class(**kwargs)
        self._is_cleaned_up = False
        self._fallback_mode = False
        self.pwd = "testbed"

        assert self.config.cache_dir_template is not None, (
            "cache_dir_template cannot be None for Singularity environment"
        )

        try:
            self._setup_sif()
            self._create_server_script()
            self.port = find_free_port()
            self._install_dependencies()
            self._health_check()
        except KeyboardInterrupt:
            print("Initialization interrupted by user")
            self.cleanup()
            raise
        except Exception as e:
            print(f"An error occurred during initialization: {e}")
            print("Enabling fallback mode due to initialization failure.")
            self._fallback_mode = True

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=20, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=(
            retry_if_not_exception_type(
                (
                    FileNotFoundError,
                    PermissionError,
                    KeyboardInterrupt,
                )
            )
        ),
    )
    def _install_dependencies(self):
        server_path_in_container = f"/tmp/{os.path.basename(self.server_script_path)}"

        cmd = [
            self.config.executable,
            "run",
            "--writable-tmpfs",
            "--containall",
            "--no-mount",
            "home,tmp,bind-paths",
            "--bind",
            f"{self.server_script_path}:{server_path_in_container}:ro",
            "--pwd",
            self.pwd,
            *self.config.start_args,
        ]
        for key, value in self.config.env.items():
            cmd.extend(["--env", f"{key}={value}"])

        cmd.append(self.sif_path)

        pip_timeout = self.config.step_timeout + 60

        # The installation directory
        uv_install_dir = "/tmp/singularity_server/uv"
        venv_path = "/tmp/singularity_server/.venv"

        install_and_run_cmd = f"""echo '127.0.0.1 localhost' > /etc/hosts & mkdir -p {uv_install_dir} && cd /tmp/singularity_server &&
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="{uv_install_dir}" sh && source {uv_install_dir}/env &&
uv venv {venv_path} --python 3.12 &&
timeout {pip_timeout} uv pip install --no-cache-dir --python {venv_path}/bin/python "fastapi[standard]==0.117.1" &&
{venv_path}/bin/python {server_path_in_container} --port {self.port}"""

        cmd.extend(["/bin/bash", "-c", install_and_run_cmd])

        print(f"Starting container with command: {shlex.join(cmd)}")
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )

        print(f"Container server started successfully on port {self.port}.")

    def _health_check(self):
        """Waits for the container's API server to become responsive."""
        max_wait = self.config.step_timeout
        start_time = time.time()

        while time.time() - start_time < max_wait:
            if self.server_process and self.server_process.poll() is not None:
                break

            try:
                response = requests.get(f"http://localhost:{self.port}/health", timeout=10)
                if response.status_code == 200:
                    response = requests.post(
                        f"http://localhost:{self.port}/run_command",
                        json={"command": "echo 'hello'"},
                        timeout=self.config.step_timeout,
                    )
                    response.raise_for_status()
                    return  # Server is up
            except requests.exceptions.RequestException:
                time.sleep(1)

        # If loop finishes or breaks, the server failed to start
        elapsed = time.time() - start_time
        print(f"Failed to start the Singularity server within {elapsed:.1f}s (timeout: {max_wait}s).")
        if self.server_process and self.server_process.stdout:
            logs = self.server_process.stdout.read()
            print(f"--- Container Server Logs ---\n{logs}")

        # Set fallback mode flag
        self._fallback_mode = True
        print("Singularity server failed to start. Enabling fallback mode.")
        return

    def _find_container(self) -> str:
        """Find the container file using multiple strategies.

        Tries in order:
        1. Exact match with "__" replaced by "_1776_"
        2. Exact match with "__" replaced by "_s_"
        3. Fuzzy search in container directory for files with either replacement

        Returns:
            str: Path to the container file (may not exist if all strategies fail)
        """
        instance_id = self.config.instance_id
        container_formatter = self.config.cache_dir_template

        # Strategy 1: Try _1776_ replacement (original case and lowercase)
        container_name = container_formatter.format(instance_id=instance_id.replace("__", "_1776_"))
        if os.path.exists(container_name):
            return container_name

        # Try lowercase version
        container_name_lower = container_formatter.format(instance_id=instance_id.replace("__", "_1776_").lower())
        if os.path.exists(container_name_lower):
            return container_name_lower

        # Strategy 2: Try _s_ replacement (original case and lowercase)
        container_name_s = container_formatter.format(instance_id=instance_id.replace("__", "_s_"))
        if os.path.exists(container_name_s):
            return container_name_s

        # Try lowercase version
        container_name_s_lower = container_formatter.format(instance_id=instance_id.replace("__", "_s_").lower())
        if os.path.exists(container_name_s_lower):
            return container_name_s_lower

        # Strategy 3: Fuzzy search in container directory
        container_dir = os.path.dirname(container_name)
        if os.path.exists(container_dir):
            # Build search patterns for both replacements
            replaced_id_1776 = instance_id.replace("__", "_1776_")
            replaced_id_s = instance_id.replace("__", "_s_")

            # Search for .sif files with either replacement pattern (case-insensitive)
            # Include both original case and lowercase versions
            patterns = [
                os.path.join(container_dir, f"*{replaced_id_1776}*.sif"),
                os.path.join(container_dir, f"*{replaced_id_s}*.sif"),
                os.path.join(container_dir, f"*{replaced_id_1776.lower()}*.sif"),
                os.path.join(container_dir, f"*{replaced_id_s.lower()}*.sif"),
            ]

            matching_files = []
            for pattern in patterns:
                matching_files.extend(glob.glob(pattern))

            if matching_files:
                # Use the first matching file found
                container_path = matching_files[0]
                print(f"Using fuzzy match: {container_path}")
                return container_path
            else:
                print(
                    f"No container found with instance_id replacements "
                    f"'{replaced_id_1776}' or '{replaced_id_s}' in {container_dir}"
                )
        else:
            print(f"Container directory {container_dir} does not exist")

        # Return the original name as fallback (even though it doesn't exist)
        print(f"Using non-existent container path: {container_name}")
        os.makedirs(container_dir, exist_ok=True)
        return container_name

    def _setup_sif(self):
        """Pulls the singularity image"""
        # Set up SIF image cache

        self.sif_path = self._find_container()

        # Pull the image if it doesn't exist
        if not os.path.exists(self.sif_path):
            pull_cmd = [self.config.executable, "pull", "--name", self.sif_path, f"docker://{self.config.image}"]
            print(f"Pulling image with command: {shlex.join(pull_cmd)}")
            try:
                subprocess.run(pull_cmd, check=True, capture_output=True, text=True, timeout=self.config.step_timeout)
                print(f"Successfully pulled image to '{self.sif_path}'")
            except subprocess.CalledProcessError as e:
                print(f"Singularity pull failed.\nStderr: {e.stderr}\nStdout: {e.stdout}")
                raise e
        else:
            print(f"Image already present at '{self.sif_path}'")

    def _create_server_script(self):
        """Reads the FastAPI server script from a file and saves it to a temporary file."""
        # Read the server script from the separate file
        server_script_path = Path(__file__).parent / "singularity_server.py"
        server_script_content = server_script_path.read_text()

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py", prefix="singularity_server_") as f:
            f.write(server_script_content)
            self.server_script_path = f.name

    def execute(self, command: str, cwd: str = "", is_eval: bool = False) -> dict[str, Any]:
        """
        Executes a command by calling the API endpoint in the container.

        Returns:
            A dictionary with 'output' (str) and 'returncode' (int).
        """
        if self._fallback_mode:
            if command.strip() == END_TRAJECTORY_COMMAND:
                return {"output": "MINI_SWE_AGENT_FINAL_OUTPUT", "returncode": 0}
            else:
                prompt_message = (
                    "The environment container failed to start. "
                    "Please run the following command to end the trajectory: "
                    f"{END_TRAJECTORY_COMMAND}"
                )
                return {"output": prompt_message, "returncode": 1}

        if self._is_cleaned_up or not self.server_process or not self.port:
            raise RuntimeError("Cannot execute command: The environment has been cleaned up or initialization failed.")

        # target_cwd = cwd or self.config.cwd
        subprocess_timeout = self.config.eval_timeout if is_eval else self.config.step_timeout
        http_timeout = subprocess_timeout + 30

        try:
            response = requests.post(
                f"http://localhost:{self.port}/run_command",
                json={"command": command, "timeout": subprocess_timeout},
                timeout=http_timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to execute command via API: {e}")
            # If the server process died, show the logs
            if self.server_process.poll() is not None:
                logs = self.server_process.stdout.read() if self.server_process.stdout else "[no logs available]"
                print(f"--- Container Server Logs ---\n{logs}")
            return {"output": f"HTTP Request Error: {e}", "returncode": -1}

    def cleanup(self):
        """Stops the server and cleans up all temporary resources."""
        if self._is_cleaned_up:
            return

        print(f"\nCleaning up Singularity environment (port {self.port})...")

        # Try graceful shutdown first
        if self.port and self.server_process and self.server_process.poll() is None:
            try:
                print(f"Requesting graceful shutdown for port {self.port}...")
                requests.post(f"http://localhost:{self.port}/shutdown", timeout=3)
                time.sleep(5)
            except Exception:
                pass  # Graceful shutdown failed, proceed to force termination

        # Force process termination if still running
        if self.server_process and self.server_process.poll() is None:
            print(f"Terminating server process {self.server_process.pid}...")
            try:
                # Try to kill the process group first (this should kill the singularity container too)
                pgid = os.getpgid(self.server_process.pid)
                os.killpg(pgid, signal.SIGTERM)
                self.server_process.wait(timeout=10)
                print("Server process terminated")
            except (ProcessLookupError, PermissionError, OSError, subprocess.TimeoutExpired):
                # Force kill if graceful termination fails
                try:
                    pgid = os.getpgid(self.server_process.pid)
                    os.killpg(pgid, signal.SIGKILL)
                    self.server_process.wait(timeout=3)
                    print("Server process force killed")
                except Exception:
                    print(f"WARNING: Could not terminate server process {self.server_process.pid}")
                    if self.port:
                        try:
                            subprocess.run(["pkill", "-9", "-f", f"--port {self.port}"], timeout=5, capture_output=True)
                            print(f"Attempted emergency kill of processes listening on port {self.port}")
                        except Exception as e:
                            print(f"Emergency kill failed: {e}")

            self.server_process = None

        # Clean up temporary files
        if self.server_script_path and os.path.exists(self.server_script_path):
            os.remove(self.server_script_path)
            self.server_script_path = None

        self._is_cleaned_up = True
        print("Cleanup complete.")

    def __enter__(self):
        """Enter the runtime context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context and perform cleanup."""
        self.cleanup()

    def __del__(self):
        """Calls the cleanup method when the object is destroyed."""
        self.cleanup()
