<div align="center">

<img src="docs/assets/nano-swe-agent-banner.svg" alt="nano-swe-agent banner" style="height: 12em"/>
<h1>The 100 line AI agent that solves GitHub issues</h1>

</div>

`nano-SWE-agent` offers an AI agent implemented in [100 lines of python](https://github.com/SWE-agent/nano-swe-agent/blob/main/nanoswea/agents/micro.py)!
Okay, maybe add another 100 lines total for a [minimal environment](https://github.com/SWE-agent/nano-swe-agent/blob/main/nanoswea/environments/local.py) 
and [model config](https://github.com/SWE-agent/nano-swe-agent/blob/main/nanoswea/models/litellm_model.py) -- but that's it, and no fancy packages used!
Of course, there's still [power-ups](#powering-up) for you to mix in: Each of these three components has variants that allow you to achieve anything!

- 🔥 Run instantly without installation: `pip install pipx && pipx run nano-swe-agent`
- ⚙️ Take full control & quickly prototype your own agent ideas
- ✅ Solves XX % of GitHub issues in SWE-Bench verified
- 🏋 Lean assumptions-free baseline system made for reinforcement learning and finetuning

The project builds on our experience building [SWE-agent](https://swe-agent.com), one of the earliest and most successful software engineering agents for research.

## 🔥 Try it without permanent installation <a target="fire"/>

```bash
pip install pipx && pipx run nano-swe-agent
```

Just make sure that you have your LM API key set as an environment variable.
You can also change default behaviors by setting the following variables in `.env` or in the environment:

```bash
NSWEA_MODEL_NAME="claude-sonnet-4-20250514"  # your favorite model
NSWEA_LOCAL_CONFIG_PATH="/path/to/your/own/config"  # override the default config for nswea 
```

## Install & experiment

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

## Powering up <a target="powerup"/>

Everything in this package follows the following simple recepe:

1. Pick an [agent class](https://github.com/SWE-agent/nano-swe-agent/blob/main/nanoswea/agents) (what's the control flow you need?)
2. Pick an [environment class](https://github.com/SWE-agent/nano-swe-agent/blob/main/nanoswea/environments) (how should actions be executed?)
3. Pick a [model class](https://github.com/SWE-agent/nano-swe-agent/blob/main/nanoswea/models) (how is the LM queried?)
4. Bind them all together in a [run script](https://github.com/SWE-agent/nano-swe-agent/blob/main/nanoswea/run) (how to invoke the agent?)

We aim to keep all of these components very simple, but offer lots of choice between them -- enough to cover a broad range of
things that you might want to do.