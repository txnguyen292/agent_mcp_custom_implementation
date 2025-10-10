"""
Project Agent MCP package exposing the BaseAgent entry point.
"""

from .base_agent import BaseAgent
from .context_manager import ContextManager
from .mcp_client import MCPClient, ToolDefinition
from .openai_client import LLMResponse, LLMToolCall, OpenAIClient
from .telemetry import TelemetryManager
from .runner import AgentRunner, build_runner, register_add_numbers_tool

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
    "build_runner",
    "register_add_numbers_tool",
]
