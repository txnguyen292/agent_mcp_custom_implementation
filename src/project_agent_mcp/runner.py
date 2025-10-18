"""
Utilities for running the BaseAgent from the command line or other applications.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

from loguru import logger

from .base_agent import BaseAgent
from .context_manager import ContextManager
from .mcp_client import MCPClient
from .openai_client import OpenAIClient
from .telemetry import TelemetryManager

ToolHandler = Callable[..., Any]


@dataclass
class AgentRunner:
    """Convenience wrapper that exposes one-shot and interactive agent execution."""

    agent: BaseAgent
    telemetry: TelemetryManager

    def run_once(self, prompt: str) -> str:
        logger.info(f"Processing prompt: {prompt}")
        response = self.agent.process_message(prompt)
        logger.info(f"Agent response: {response}")
        return response

    def interactive(self) -> None:
        """Simple REPL-style loop for manual testing."""
        print("Interactive BaseAgent session. Type 'exit' to quit.")
        try:
            while True:
                prompt = input("You: ").strip()
                if prompt.lower() in {"exit", "quit"}:
                    print("Exiting agent session.")
                    break
                if not prompt:
                    continue
                response = self.run_once(prompt)
                print(f"Agent: {response}")
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting agent session.")


def build_runner(
    *,
    model: str = "gpt-4o-mini",
    tool_factories: Optional[Iterable[Callable[[MCPClient], None]]] = None,
    dashboard_dir: Path | str = Path("dashboards") / "runner",
    dashboard_filename: str = "index.html",
    mcp_client: Optional[MCPClient] = None,
) -> AgentRunner:
    """
    Create an AgentRunner configured with the given model and optional tools.

    Args:
        model: OpenAI model identifier.
        tool_factories: Callables that accept an MCPClient and register tools.
        dashboard_dir: Destination for telemetry dashboard artifacts.
        dashboard_filename: HTML file name for dashboard output.
        mcp_client: Optional pre-configured MCP client instance.
    """
    client = mcp_client or MCPClient()

    if tool_factories:
        for factory in tool_factories:
            factory(client)

    telemetry = TelemetryManager(dashboard_dir=dashboard_dir, dashboard_filename=dashboard_filename)
    agent = BaseAgent(
        mcp_client=client,
        llm_client=OpenAIClient(model=model),
        context_manager=ContextManager(),
        telemetry=telemetry,
    )
    return AgentRunner(agent=agent, telemetry=telemetry)


def register_basic_math_tools(mcp_client: MCPClient) -> None:
    """Register basic arithmetic tools (add, subtract, multiply, divide)."""

    def add(a: float, b: float) -> float:
        return a + b

    def subtract(a: float, b: float) -> float:
        return a - b

    def multiply(a: float, b: float) -> float:
        return a * b

    def divide(a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Division by zero is not allowed.")
        return a / b

    def register(name: str, handler: Callable[..., Any], description: str) -> None:
        mcp_client.register_tool(
            name=name,
            handler=handler,
            description=description,
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        )

    register("add_numbers", add, "Add two numbers together.")
    register("subtract_numbers", subtract, "Subtract the second number from the first.")
    register("multiply_numbers", multiply, "Multiply two numbers.")
    register("divide_numbers", divide, "Divide the first number by the second.")


def register_add_numbers_tool(mcp_client: MCPClient) -> None:
    """
    Backwards compatible helper that now registers the full set of basic math tools.
    """

    logger.warning(
        "register_add_numbers_tool is deprecated; use register_basic_math_tools instead."
    )
    register_basic_math_tools(mcp_client)
