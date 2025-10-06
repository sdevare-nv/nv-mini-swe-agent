import argparse
import asyncio
import os
import signal
import socket
import subprocess
import textwrap
import time
from contextlib import asynccontextmanager
from random import uniform

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

CONDA_ENV = None
shutdown_event = asyncio.Event()


class CommandRequest(BaseModel):
    command: str
    timeout: float | None = None


class CommandResult(BaseModel):
    output: str
    returncode: int


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def is_port_in_use(host: str, port: int) -> bool:
    print(f"Checking if port {port} is in use...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


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


def start_server(app: FastAPI, host: str, initial_port: int):
    """
    Starts the Uvicorn server on the given port. If the port is in use,
    it finds a new free port and attempts to start the server there.
    """
    port = initial_port
    while True:
        # Check if port is available before attempting to start server
        if is_port_in_use(host, port):
            print(f"⚠️ Port {port} is already in use.")
            port = find_free_port()
            time.sleep(uniform(1, 8))
            continue

        print(f"Attempting to start server on http://{host}:{port}")
        uvicorn.run(
            app,
            host=host,
            port=port,
            timeout_graceful_shutdown=5,
            timeout_keep_alive=2,
            access_log=False,
            server_header=False,
        )
        break


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
    start_server(app, host="0.0.0.0", initial_port=args.port)
