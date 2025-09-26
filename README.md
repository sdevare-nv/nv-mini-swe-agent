# NV-Mini-SWE-Agent


## Setup
```bash
uv sync
source activate .venv/bin/activate
uv pip install "swegym @ git+https://github.com/sdevare-nv/nv-SWE-Bench-Package.git@31e1cb8f0241da1707d00faa633c3d6ce1a8ba3b"
```

## SBATCH SWE-Gym (Singularity Containers)
```bash
sbatch scripts/run_swegym.sh
```