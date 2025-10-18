# Project Agent MCP

## Prerequisites
- Python 3.13+
- Set `OPENAI_API_KEY` in your environment before running the agent

## Installation
```bash
pip install -e .[dev]
```

## Running the CLI
```bash
export OPENAI_API_KEY=your_key
python main.py --prompt "What is 2 plus 5?"
```


## Running with uv
If you use [uv](https://github.com/astral-sh/uv):
```bash
uv sync
OPENAI_API_KEY=your_key uv run python main.py --prompt "What is 2 plus 5?"
```
For interactive mode:
```bash
OPENAI_API_KEY=your_key uv run python main.py
```

## Interactive Mode
```bash
python main.py
```
Type `exit` to leave the session.

## Notebooks
1. `notebooks/base_agent_demo.ipynb`: Live OpenAI demo showing BaseAgent + MCP tool integration.
2. `notebooks/context_manager_demo.ipynb`: Explore ContextManager message formatting.
3. `notebooks/mcp_client_demo.ipynb`: Register and execute MCP tools locally.
4. `notebooks/telemetry_manager_demo.ipynb`: Capture spans/events and emit dashboards.

## Tests
```bash
python -m pytest
```
