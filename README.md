<div align="center">

<img src="docs/assets/nano-swe-agent-banner.svg" alt="nano-swe-agent banner" style="height: 12em"/>
<h1>The 100 line AI agent that solves GitHub issues</h1>

</div>

nano-SWE-agent is an AI agent implemented in [100 lines of python](https://github.com/SWE-agent/nano-swe-agent/blob/main/nanoswea/agent.py)!
Okay, maybe add another 100 lines for a [minimal sandboxed environment](https://github.com/SWE-agent/nano-swe-agent/blob/main/nanoswea/environment.py) 
and [model config](https://github.com/SWE-agent/nano-swe-agent/blob/main/nanoswea/model.py) -- but that's it, and no fancy packages used!

That's enough to solve XX% of the GitHub issues in SWE-bench verified benchmark, one of the most popular tests for agentic capabilities.

nano-SWE-agent simple, readable, and an excellent starting point for building any type of AI agent project.

The project builds on our experience building SWE-agent, one of the earliest and most successful software engineering agents for research.
However, there are a couple of design differences.
Chiefly, this project will grow by offering a family of extremely simple implementations of the agent, environment, and model classes.
If you have a particular idea in mind, you might have to write a few lines of code -- but this will be easier than with any other project.

On research side, the simplicity of this agent is perfect for training your own LM with fine tuning or reinforcement learning.

## 🔥 Let's fire it up!

Just try it without installing

```bash
pip install pipx && pipx run nano-swe-agent
```

Install it for experimentation

```bash
git clone https://github.com/SWE-agent/nano-swe-agent.git
cd nano-swe-agent
pip install -e .
```

This will expose `nswea` (local agent) and `nswea-gh` (run on github issues with docker as sandbox).
But you can also just run the executables as

```bash
python nanoswea/run_local.py
```

etc.

You can put your LM API keys in a `.env` at the repository root or make sure they're set in your shell.
You can also change default behaviors by setting the following variables in `.env` or in the environment:

```bash
NSWEA_MODEL_NAME="claude-sonnet-4-20250514"  # your favorite model
NSWEA_LOCAL_CONFIG_PATH="/path/to/your/own/config"  # override the default config for nswea 
```
