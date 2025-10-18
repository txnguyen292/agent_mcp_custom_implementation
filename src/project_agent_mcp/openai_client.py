"""
OpenAI client abstraction that returns structured responses to the BaseAgent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from .exceptions import LLMResponseError


@dataclass
class LLMToolCall:
    """Represents a single tool call requested by the LLM."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    call_id: Optional[str] = None


@dataclass
class LLMResponse:
    """Structured wrapper for OpenAI chat completions."""

    content: Optional[str]
    tool_calls: List[LLMToolCall] = field(default_factory=list)
    reasoning: Optional[str] = None
    answer: Optional[str] = None
    raw: Any = None

    @property
    def is_final(self) -> bool:
        """True when the response contains final user-facing content."""
        return self.content is not None and not self.tool_calls


class OpenAIClient:
    """
    Wrapper around the OpenAI SDK with optional instrumentation hooks.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        client: Optional[Any] = None,
        runner: Optional[Callable[..., Any]] = None,
        auto_instrument: bool = True,
    ) -> None:
        self.model = model
        self._client = client
        self._runner = runner
        self._logger = logger.bind(component="OpenAIClient")
        self._instrumentation_source: Optional[str] = None

        if auto_instrument:
            self._instrumentation_source = self._attempt_instrumentation()

        if self._client is None and self._runner is None:
            self._client = self._build_default_client()

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: float = 0.0,
    ) -> LLMResponse:
        """
        Execute a chat completion call using the provided history and tools.
        """
        if self._runner is not None:
            result = self._runner(messages=messages, tools=tools, model=self.model)
            return self._ensure_response(result)

        if self._client is None:
            raise LLMResponseError(
                "OpenAI client is not configured. Provide an SDK client or a custom runner."
            )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools or None,
                temperature=temperature,
            )
        except Exception as exc:  # pragma: no cover - depends on network/API
            self._logger.exception(f"OpenAI chat completion failed: {exc}")
            raise LLMResponseError(str(exc)) from exc

        return self._parse_chat_response(response)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_default_client(self) -> Optional[Any]:
        try:
            openai_module = import_module("openai")
        except ModuleNotFoundError:
            self._logger.warning("OpenAI SDK not installed; falling back to custom runner mode.")
            return None

        # The 1.x SDK exposes the OpenAI class as the main entry point.
        client_factory = getattr(openai_module, "OpenAI", None)
        if client_factory is None:
            raise LLMResponseError("OpenAI SDK is installed but does not expose OpenAI() factory.")

        client = client_factory()
        self._logger.debug(f"Created default OpenAI client for model '{self.model}'")
        return client

    def _ensure_response(self, response: Any) -> LLMResponse:
        if isinstance(response, LLMResponse):
            return response
        if not isinstance(response, dict):
            raise LLMResponseError("Custom runner must return LLMResponse or dict payload.")
        content = response.get("content")
        tool_calls = [
            LLMToolCall(
                name=call["name"],
                arguments=call.get("arguments", {}),
                call_id=call.get("call_id"),
            )
            for call in response.get("tool_calls", [])
        ]
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            reasoning=response.get("reasoning"),
            answer=response.get("answer"),
            raw=response,
        )

    def _parse_chat_response(self, response: Any) -> LLMResponse:
        choice = getattr(response, "choices", None)
        if not choice:
            raise LLMResponseError("OpenAI response did not include choices.")

        message = choice[0].message
        content, reasoning, answer = self._extract_message_content_and_reasoning(message)
        tool_calls_payload = getattr(message, "tool_calls", None) or []

        tool_calls: List[LLMToolCall] = []
        for call in tool_calls_payload:
            function = getattr(call, "function", None) or call.get("function", {})
            name = getattr(function, "name", None) or function.get("name")
            arguments = getattr(function, "arguments", None) or function.get("arguments", "{}")
            call_id = getattr(call, "id", None) or call.get("id")

            if isinstance(arguments, str):
                try:
                    parsed_arguments = json.loads(arguments) if arguments else {}
                except json.JSONDecodeError:
                    parsed_arguments = {"raw": arguments}
            else:
                parsed_arguments = arguments or {}
            tool_calls.append(
                LLMToolCall(name=name, arguments=parsed_arguments, call_id=call_id)
            )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            reasoning=reasoning,
            answer=answer,
            raw=response,
        )

    def _attempt_instrumentation(self) -> Optional[str]:
        """
        Try to instrument the OpenAI SDK using Traceloop/OpenLLMetry if available.
        """
        instrumentation_targets = [
            "traceloop_sdk.integrations.openai:instrument_openai",
            "openllmetry.instrumentation.openai:instrument_openai",
        ]

        for path in instrumentation_targets:
            module_path, function_name = path.split(":")
            try:
                module = import_module(module_path)
                instrument_fn = getattr(module, function_name)
                instrument_fn()  # type: ignore[call-arg]
                self._logger.debug(f"Instrumented OpenAI client via {path}")
                return path
            except ModuleNotFoundError:
                continue
            except Exception as exc:
                self._logger.debug(f"Failed to instrument OpenAI with {path}: {exc}")
        return None

    def _extract_message_content_and_reasoning(self, message: Any) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Extract textual content and reasoning thoughts from an OpenAI response message.
        Supports both string and content-part formats.
        """
        content = getattr(message, "content", None)
        reasoning: Optional[str] = None

        if isinstance(content, list):
            text_parts: List[str] = []
            reasoning_parts: List[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
                elif part.get("type") == "reasoning":
                    text = part.get("text")
                    if isinstance(text, str):
                        reasoning_parts.append(text)
            content = "\n".join(text_parts) if text_parts else None
            reasoning = "\n".join(reasoning_parts) if reasoning_parts else None
        elif hasattr(content, "text"):
            # Some SDKs wrap parts in objects with text attribute.
            try:
                text = getattr(content, "text")
                if isinstance(text, list):
                    text_parts = [t.value for t in text if hasattr(t, "value")]
                    content = "\n".join(text_parts) if text_parts else None
                elif isinstance(text, str):
                    content = text
            except Exception:  # pragma: no cover
                pass

        # Some models expose explicit reasoning fields.
        if reasoning is None:
            reasoning_candidate = getattr(message, "reasoning", None)
            if isinstance(reasoning_candidate, str):
                reasoning = reasoning_candidate
            elif isinstance(reasoning_candidate, list):
                reasoning = "\n".join(
                    item.get("text", "")
                    for item in reasoning_candidate
                    if isinstance(item, dict)
                ).strip() or None

        extracted_answer: Optional[str] = None

        if isinstance(content, str):
            reason_text = None
            answer_text = None

            if "Reasoning:" in content:
                after_reason = content.split("Reasoning:", 1)[1]
                if "Answer:" in after_reason:
                    reason_text, after_reason = after_reason.split("Answer:", 1)
                    answer_text = after_reason.strip()
                else:
                    reason_text = after_reason.strip()
                if reason_text:
                    reasoning = reasoning or reason_text.strip()
            if "Answer:" in content:
                answer_text = content.split("Answer:", 1)[1].strip()

            if answer_text:
                extracted_answer = answer_text

        return content, reasoning, extracted_answer
