"""
BaseAgent implementation orchestrating OpenAI interactions and MCP tool usage.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from loguru import logger

from .context_manager import ContextManager
from .exceptions import AgentError, MCPClientError, ToolExecutionError
from .mcp_client import MCPClient, ToolDefinition
from .openai_client import LLMResponse, LLMToolCall, OpenAIClient
from .telemetry import TelemetryManager


class BaseAgent:
    """Co-ordinates LLM reasoning with MCP tool execution."""

    def __init__(
        self,
        mcp_client: MCPClient,
        llm_client: OpenAIClient,
        context_manager: Optional[ContextManager] = None,
        telemetry: Optional[TelemetryManager] = None,
        max_tool_iterations: int = 5,
    ) -> None:
        self.mcp_client = mcp_client
        self.llm_client = llm_client
        self.context_manager = context_manager or ContextManager()
        self.telemetry = telemetry or TelemetryManager()
        self.max_tool_iterations = max_tool_iterations
        self._logger = logger.bind(component="BaseAgent")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_message(self, user_message: str) -> str:
        """
        Entry point for handling a user query end-to-end.
        """
        self.context_manager.reset()
        self.context_manager.add_user_message(user_message)
        self.telemetry.record_event("user_message", {"content": user_message})
        self._logger.info("Processing user message")

        tool_definitions = self._discover_tools()
        tool_schemas = [tool.to_openai_schema() for tool in tool_definitions]
        final_response: Optional[str] = None

        with self.telemetry.span(
            "agent.process_message",
            {"tool_count": len(tool_schemas), "max_iterations": self.max_tool_iterations},
        ):
            for iteration in range(self.max_tool_iterations):
                self._logger.debug(f"Agent iteration {iteration + 1}")
                messages = self.context_manager.as_openai_messages()

                llm_response = self.llm_client.chat_completion(messages, tool_schemas)
                self._record_llm_response(llm_response, iteration)

                # Store assistant response including tool calls (if any) for the next round.
                tool_call_payload = self._format_tool_calls(llm_response.tool_calls)
                self.context_manager.add_model_message(
                    llm_response.content,
                    metadata={"iteration": iteration},
                    tool_calls=tool_call_payload if tool_call_payload else None,
                )

                if llm_response.tool_calls:
                    self._logger.debug(f"LLM requested {len(llm_response.tool_calls)} tool call(s)")
                    self._handle_tool_calls(llm_response.tool_calls)
                    continue

                if llm_response.content is not None:
                    final_response = llm_response.content
                    break

            if final_response is None:
                raise AgentError(
                    f"Agent stopped after {self.max_tool_iterations} iterations without a final answer."
                )

        self.telemetry.record_event("final_response", {"content": final_response})
        self.telemetry.write_dashboard()
        return final_response

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _discover_tools(self) -> List[ToolDefinition]:
        try:
            tools = self.mcp_client.list_tools()
            self._logger.debug(f"Discovered {len(tools)} tool(s)")
            return tools
        except MCPClientError as exc:
            self._logger.warning(f"Failed to list tools via MCP: {exc}")
            return []

    def _handle_tool_calls(self, tool_calls: List[LLMToolCall]) -> None:
        for index, call in enumerate(tool_calls):
            call_id = self._resolve_call_id(call, index)
            self.telemetry.record_event(
                "tool_call",
                {"tool": call.name, "arguments": call.arguments, "call_id": call_id},
            )
            try:
                result = self.mcp_client.execute_tool(call.name, call.arguments)
                self.telemetry.record_event(
                    "tool_result",
                    {"tool": call.name, "result": result},
                )
                self.context_manager.add_tool_message(
                    tool_name=call.name,
                    content={"result": result},
                    tool_call_id=call_id,
                )
            except ToolExecutionError as exc:
                self.telemetry.record_event(
                    "tool_error",
                    {"tool": call.name, "error": str(exc), "call_id": call_id},
                )
                self.context_manager.add_tool_message(
                    tool_name=call.name,
                    content={"error": str(exc)},
                    tool_call_id=call_id,
                )
        # After gathering tool responses we expect the LLM to be called again, so nothing else here.

    def _record_llm_response(self, response: LLMResponse, iteration: int) -> None:
        event_payload = {
            "iteration": iteration,
            "content": response.content,
            "reasoning": response.reasoning,
            "tool_calls": [
                {"name": call.name, "arguments": call.arguments, "call_id": call.call_id}
                for call in response.tool_calls
            ],
        }
        self.telemetry.record_event("llm_response", event_payload)

    def _format_tool_calls(self, tool_calls: List[LLMToolCall]) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        for idx, call in enumerate(tool_calls):
            call_id = self._resolve_call_id(call, idx)
            formatted.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
            )
        return formatted

    @staticmethod
    def _resolve_call_id(call: LLMToolCall, index: int) -> str:
        return call.call_id or f"call_{index}"
