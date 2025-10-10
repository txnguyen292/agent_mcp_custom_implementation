"""
MCP client wrapper responsible for tool discovery and execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from loguru import logger

from .exceptions import MCPClientError, ToolExecutionError

try:
    # The official MCP package exposes several client helpers (SSE, stdio, etc.).
    # We import lazily so that unit tests can run without an active server.
    from mcp.client.session import Session as MCPBaseSession  # type: ignore
except Exception:  # pragma: no cover - fallback when MCP is unavailable
    MCPBaseSession = None  # type: ignore


@dataclass
class ToolDefinition:
    """Lightweight representation of a tool exposed by an MCP server."""

    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)

    def to_openai_schema(self) -> Dict[str, Any]:
        """Return the tool definition in OpenAI function-call format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema or {"type": "object", "properties": {}},
            },
        }


class MCPClient:
    """
    Thin wrapper over the MCP python client.

    It supports two modes:
    1. Connected mode, delegating to an MCP session object.
    2. Local registration mode, useful for tests or when running without a server.
    """

    def __init__(self, session: Optional[Any] = None) -> None:
        self._session = session
        self._logger = logger.bind(component="MCPClient")
        self._definitions: Dict[str, ToolDefinition] = {}
        self._handlers: Dict[str, Callable[..., Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def connect(self, session: Any) -> None:
        """
        Attach an MCP session implementation returned by the official `mcp` package.
        """
        if MCPBaseSession is None:
            raise MCPClientError(
                "The mcp package is not available. Install it and provide a valid session."
            )
        if not isinstance(session, MCPBaseSession):
            self._logger.warning(
                "Session does not inherit from MCPBaseSession; continuing but execution may fail."
            )
        self._session = session

    def register_tool(
        self,
        name: str,
        handler: Callable[..., Any],
        description: str = "",
        input_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a local tool handler. Useful for tests or offline usage.
        """
        definition = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema or {"type": "object", "properties": {}},
        )
        self._definitions[name] = definition
        self._handlers[name] = handler
        self._logger.debug(f"Registered local MCP tool '{name}'")

    def list_tools(self) -> List[ToolDefinition]:
        """Discover available tools, either from the session or the local registry."""
        if self._session is not None:
            return list(self._list_remote_tools())
        return list(self._definitions.values())

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool via MCP."""
        if self._session is not None:
            return self._execute_remote_tool(name, arguments)
        if name not in self._handlers:
            raise ToolExecutionError(f"Unknown tool '{name}' in local MCP registry.")
        handler = self._handlers[name]
        try:
            return handler(**arguments)
        except TypeError:
            # Fall back to passing the raw argument dictionary.
            return handler(arguments)
        except Exception as exc:  # pragma: no cover - bubbling exception
            self._logger.exception(f"Tool '{name}' failed: {exc}")
            raise ToolExecutionError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _list_remote_tools(self) -> Iterable[ToolDefinition]:
        try:
            tool_infos = self._session.list_tools()  # type: ignore[attr-defined]
        except AttributeError as exc:  # pragma: no cover - depends on concrete session
            raise MCPClientError("Connected MCP session does not expose list_tools().") from exc
        except Exception as exc:  # pragma: no cover
            raise MCPClientError(f"Failed to list tools via MCP: {exc}") from exc

        definitions: List[ToolDefinition] = []
        for tool in tool_infos or []:
            # The official client returns pydantic models with attributes; we support both dicts
            # and objects for flexibility.
            if hasattr(tool, "name"):
                name = tool.name
                description = getattr(tool, "description", "")
                schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)
            else:
                name = tool.get("name", "")
                description = tool.get("description", "")
                schema = tool.get("input_schema") or tool.get("inputSchema")
            definitions.append(
                ToolDefinition(
                    name=name,
                    description=description,
                    input_schema=schema or {"type": "object", "properties": {}},
                )
            )
        return definitions

    def _execute_remote_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        try:
            executor = getattr(self._session, "call_tool", None)
            if executor is None:
                executor = getattr(self._session, "execute_tool", None)
            if executor is None:
                raise MCPClientError("MCP session does not provide a tool execution method.")
            return executor(name=name, arguments=arguments)
        except Exception as exc:  # pragma: no cover - relies on actual server
            self._logger.exception(f"Remote tool '{name}' execution failed: {exc}")
            raise ToolExecutionError(str(exc)) from exc
