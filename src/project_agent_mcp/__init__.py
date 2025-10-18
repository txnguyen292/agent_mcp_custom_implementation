"""
Project Agent MCP package exposing the BaseAgent entry point.
"""

from .base_agent import BaseAgent
from .context_manager import ContextManager
from .mcp_client import MCPClient, ToolDefinition
from .openai_client import LLMResponse, LLMToolCall, OpenAIClient
from .telemetry import TelemetryManager
from .dashboard_app import main as telemetry_dashboard_app
from .runner import (
    AgentRunner,
    DEFAULT_SYSTEM_MESSAGE,
    build_runner,
    register_basic_math_tools,
    register_add_numbers_tool,
)

__all__ = [
    "BaseAgent",
    "ContextManager",
    "MCPClient",
    "ToolDefinition",
    "LLMResponse",
    "LLMToolCall",
    "OpenAIClient",
    "TelemetryManager",
    "AgentRunner",
    "DEFAULT_SYSTEM_MESSAGE",
    "build_runner",
    "register_basic_math_tools",
    "register_add_numbers_tool",
    "telemetry_dashboard_app",
]
