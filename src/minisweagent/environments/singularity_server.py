import uvicorn
import subprocess
import os
import argparse
import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import textwrap

CONDA_ENV = None
shutdown_event = asyncio.Event()


class CommandRequest(BaseModel):
    command: str
    timeout: float | None = None


class CommandResult(BaseModel):
    output: str
    returncode: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    yield
    # Shutdown logic
    print("FastAPI server shutting down...")
    # Give time for any pending requests to complete
    await asyncio.sleep(0.1)


app = FastAPI(lifespan=lifespan)


# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    print(f"Received signal {signum}, shutting down gracefully...")
    shutdown_event.set()


@app.post("/run_command", response_model=CommandResult)
def run_command(req: CommandRequest):
    activation_cmd = ""
    if CONDA_ENV:
        # TODO(sugam): /testbed is hardcoded here.
        activation_cmd = (
            f"cd /testbed && source $(conda info --base)/etc/profile.d/conda.sh && conda activate {CONDA_ENV} && "
        )

    clean_command = textwrap.dedent(req.command)
    full_command = f"{activation_cmd}{clean_command}"

    try:
        result = subprocess.run(
            full_command,
            shell=True,
            executable="/bin/bash",
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=req.timeout,
        )

        full_output = result.stdout
        actual_output = full_output.strip()

        return CommandResult(output=actual_output, returncode=result.returncode)
    except subprocess.TimeoutExpired as e:
        timeout_output = f"Command timed out after {req.timeout} seconds"
        if e.stdout:
            timeout_output += f"\nPartial output:\n{e.stdout.decode('utf-8', errors='replace').strip()}"
        return CommandResult(output=timeout_output, returncode=124)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/shutdown")
async def shutdown():
    """Endpoint to trigger server shutdown"""
    print("Shutdown requested via API")
    shutdown_event.set()

    # Schedule shutdown after a brief delay to allow response to be sent
    async def delayed_shutdown():
        await asyncio.sleep(0.1)
        # Signal uvicorn to shutdown gracefully
        os.kill(os.getpid(), signal.SIGTERM)

    asyncio.create_task(delayed_shutdown())
    return {"message": "Shutdown initiated"}


if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True, help="Port to run the server on")
    parser.add_argument(
        "--conda_env", type=str, default="testbed", help="Name of the conda environment to run commands in"
    )
    args = parser.parse_args()

    CONDA_ENV = args.conda_env
    print(f"Commands will run inside the '{CONDA_ENV}' Conda environment.")

    # Run with explicit shutdown settings
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        timeout_graceful_shutdown=5,
        timeout_keep_alive=2,
        access_log=False,
        server_header=False,
    )
