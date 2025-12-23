import argparse
import asyncio
import os
import signal
import textwrap
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

shutdown_event = asyncio.Event()


class CommandRequest(BaseModel):
    command: str
    timeout: float | None = None
    cwd: str = "/testbed"
    conda_env: str | None = None


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
async def run_command(req: CommandRequest):
    activation_cmd = (
        f"cd {req.cwd} && source $(conda info --base)/etc/profile.d/conda.sh && conda activate {req.conda_env} && "
        if req.conda_env
        else f"cd {req.cwd} && "
    )

    clean_command = textwrap.dedent(req.command)
    full_command = f"{activation_cmd}{clean_command}"

    try:
        process = await asyncio.create_subprocess_shell(
            full_command,
            executable="/bin/bash",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=req.timeout)
        actual_output = stdout.decode("utf-8", errors="replace").strip()

        return CommandResult(output=actual_output, returncode=process.returncode)
    except asyncio.TimeoutError:
        # Kill the process if it's still running
        if process.returncode is None:
            process.kill()
            await process.wait()

        timeout_output = f"Command timed out after {req.timeout} seconds"
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


def start_server(app: FastAPI, host: str, port: int):
    """
    Starts the Uvicorn server on the given port.
    Port conflicts are now handled by the parent process.
    """
    print(f"Starting server on http://{host}:{port}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        timeout_graceful_shutdown=5,
        timeout_keep_alive=2,
        access_log=False,
        server_header=False,
    )


if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True, help="Port to run the server on")
    args = parser.parse_args()

    # Run with explicit shutdown settings
    start_server(app, host="0.0.0.0", port=args.port)
