"""
Conversation context management for the BaseAgent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class ConversationMessage:
    """Represents a single message exchanged during an agent run."""

    role: str
    content: Any
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    def to_openai_message(self) -> Dict[str, Any]:
        """
        Convert the message to the structure expected by the OpenAI chat API.
        Tool messages need to be serialized to strings.
        """
        message: Dict[str, Any] = {"role": self.role}

        if self.role == "tool":
            if self.tool_call_id:
                message["tool_call_id"] = self.tool_call_id
            message["name"] = self.name or self.metadata.get("tool_name", "")
            if isinstance(self.content, (dict, list)):
                message["content"] = json.dumps(self.content)
            else:
                message["content"] = str(self.content)
        else:
            if self.name:
                message["name"] = self.name
            message["content"] = "" if self.content is None else str(self.content)
            if self.tool_calls:
                message["tool_calls"] = self.tool_calls
        return message


class ContextManager:
    """Maintains the ordered list of conversation turns for an agent run."""

    def __init__(self) -> None:
        self._messages: List[ConversationMessage] = []

    def reset(self) -> None:
        """Clear the conversation history."""
        self._messages.clear()

    @property
    def messages(self) -> Iterable[ConversationMessage]:
        """Expose the message stream."""
        return tuple(self._messages)

    def as_openai_messages(self) -> List[Dict[str, Any]]:
        """Return the history as OpenAI-compatible messages list."""
        return [message.to_openai_message() for message in self._messages]

    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._messages.append(
            ConversationMessage(role="user", content=content, metadata=metadata or {})
        )

    def add_model_message(
        self,
        content: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._messages.append(
            ConversationMessage(
                role="assistant",
                content=content,
                metadata=metadata or {},
                tool_calls=tool_calls,
            )
        )

    def add_tool_message(
        self,
        tool_name: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        tool_call_id: Optional[str] = None,
    ) -> None:
        payload = metadata or {}
        payload.setdefault("tool_name", tool_name)
        self._messages.append(
            ConversationMessage(
                role="tool",
                name=tool_name,
                content=content,
                metadata=payload,
                tool_call_id=tool_call_id,
            )
        )
