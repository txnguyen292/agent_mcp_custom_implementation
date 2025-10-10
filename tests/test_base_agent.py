import json
import sys
from pathlib import Path

from project_agent_mcp import (  # noqa: E402
    BaseAgent,
    ContextManager,
    LLMResponse,
    LLMToolCall,
    MCPClient,
    OpenAIClient,
    TelemetryManager,
)


class FakeLLMRunner:
    """Deterministic runner used to simulate OpenAI responses."""

    def __init__(self) -> None:
        self.invocations = 0

    def __call__(self, messages, tools, model):
        self.invocations += 1
        if self.invocations == 1:
            return LLMResponse(
                content=None,
                tool_calls=[
                    LLMToolCall(
                        name="add_numbers",
                        arguments={"a": 1, "b": 2},
                        call_id="call_add_numbers",
                    )
                ],
            )
        return LLMResponse(content="The sum is 3.", tool_calls=[])


class FinalOnlyRunner:
    def __init__(self) -> None:
        self.invocations = 0

    def __call__(self, messages, tools, model):
        self.invocations += 1
        return LLMResponse(content="Final message.", tool_calls=[])


def test_base_agent_runs_tool_and_writes_dashboard(tmp_path):
    runner = FakeLLMRunner()
    llm_client = OpenAIClient(runner=runner)

    mcp_client = MCPClient()
    mcp_client.register_tool(
        "add_numbers",
        lambda a, b: a + b,
        description="Adds two numbers together.",
        input_schema={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    )

    telemetry = TelemetryManager(dashboard_dir=tmp_path, dashboard_filename="agent.html")
    context_manager = ContextManager()

    agent = BaseAgent(
        mcp_client=mcp_client,
        llm_client=llm_client,
        context_manager=context_manager,
        telemetry=telemetry,
    )

    result = agent.process_message("Add 1 and 2.")

    assert result == "The sum is 3."
    dashboard_path = Path(tmp_path) / "agent.html"
    assert dashboard_path.exists()
    html_content = dashboard_path.read_text()
    assert "Agent Run Dashboard" in html_content
    assert "tool_result" in html_content

    messages = context_manager.as_openai_messages()
    assert any(message["role"] == "tool" for message in messages)


def test_base_agent_final_response_without_tools(tmp_path):
    runner = FinalOnlyRunner()
    llm_client = OpenAIClient(runner=runner)
    mcp_client = MCPClient()
    telemetry = TelemetryManager(dashboard_dir=tmp_path, dashboard_filename="simple.html")
    context_manager = ContextManager()
    agent = BaseAgent(
        mcp_client=mcp_client,
        llm_client=llm_client,
        context_manager=context_manager,
        telemetry=telemetry,
    )

    result = agent.process_message("Hello there!")

    assert result == "Final message."
    dashboard_path = Path(tmp_path) / "simple.html"
    assert dashboard_path.exists()
    html = dashboard_path.read_text()
    script_segment = html.split("const data = ", 1)[1]
    if ";\n" in script_segment:
        json_blob = script_segment.split(";\n", 1)[0]
    else:
        json_blob = script_segment.split(";", 1)[0]
    dashboard_json = json.loads(json_blob)
    assert dashboard_json["events"][-1]["type"] == "final_response"
