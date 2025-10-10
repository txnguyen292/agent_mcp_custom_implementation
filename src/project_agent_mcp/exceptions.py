"""
Custom exceptions raised by the BaseAgent stack.
"""


class AgentError(Exception):
    """Base exception for agent-related failures."""


class MCPClientError(AgentError):
    """Raised when the MCP client cannot complete an operation."""


class ToolExecutionError(MCPClientError):
    """Raised when executing a tool fails."""


class LLMResponseError(AgentError):
    """Raised when the LLM response cannot be parsed or is invalid."""
