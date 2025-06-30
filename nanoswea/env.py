import subprocess
import uuid


class LocalEnvironment:
    def __init__(self):
        """This class executes bash commands directly on the local machine."""
        pass

    def execute(self, command: str, cwd: str = "/testbed") -> str:
        """Execute a command in the local environment and return the raw output."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=30,
            )
            return f"stdout: {result.stdout}\nstderr: {result.stderr}\nexit_code: {result.returncode}"
        except subprocess.TimeoutExpired as e:
            raise TimeoutError from e


class DockerEnvironment:
    def __init__(self, image: str):
        """This class executes bash commands in a Docker container using direct docker commands."""
        self.image = image
        self.container_name = f"nanoswea-{uuid.uuid4().hex[:8]}"
        self.container_id = None
        self._start_container()

    def _start_container(self):
        """Start the Docker container."""
        try:
            # Start container in detached mode with /testbed as working directory
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    self.container_name,
                    "-w",
                    "/testbed",
                    self.image,
                    "sleep",
                    "infinity",  # Keep container running
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                msg = f"Failed to start container: {result.stderr}"
                raise RuntimeError(msg)
            self.container_id = result.stdout.strip()
        except subprocess.TimeoutExpired as e:
            msg = "Failed to start Docker container"
            raise TimeoutError(msg) from e

    def execute(self, command: str, cwd: str = "/testbed") -> str:
        """Execute a command in the Docker container and return the raw output."""
        if not self.container_id:
            msg = "Container not started"
            raise RuntimeError(msg)

        try:
            result = subprocess.run(
                ["docker", "exec", "-w", cwd, self.container_id, "bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return f"stdout: {result.stdout}\nstderr: {result.stderr}\nexit_code: {result.returncode}"
        except subprocess.TimeoutExpired as e:
            raise TimeoutError from e

    def cleanup(self):
        """Stop and remove the Docker container."""
        if self.container_id:
            try:
                subprocess.run(["docker", "stop", self.container_id], capture_output=True)
                subprocess.run(["docker", "rm", self.container_id], capture_output=True)
            except Exception:
                pass  # Best effort cleanup

    def __del__(self):
        """Cleanup container when object is destroyed."""
        self.cleanup()
