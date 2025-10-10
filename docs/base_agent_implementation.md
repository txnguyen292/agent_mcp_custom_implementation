# BaseAgent Implementation

## Overview
The BaseAgent is a single-agent system designed to plan, reason, and execute tasks using MCP servers and OpenAI API calls. It follows a ReAct-style architecture, focusing on tool use and step-by-step reasoning.

## Architecture
- **Agent Core**: Orchestrates planning, reasoning, and execution.
- **MCP Client**: Connects to MCP servers, discovers tools, and executes tool calls.
- **LLM Interface**: Communicates with the language model, sends queries, and receives tool call decisions.
- **Tool Manager**: Converts MCP tool definitions to LLM-compatible schemas and manages tool execution.
- **Context Manager**: Maintains conversation history and intermediate results.
- **Error Handler**: Handles exceptions and retries during tool execution.

## Workflow
1. **Receive User Input**: The agent receives a user query.
2. **Discover Tools**: The MCP client connects to the relevant MCP server and lists available tools.
3. **Tool Schema Conversion**: Tool definitions are converted to LLM-compatible function schemas.
4. **LLM Reasoning**: The agent sends the user query and tool schemas to the LLM, which decides which tool(s) to use and with what arguments.
5. **Tool Execution**: The MCP client executes the selected tool(s) with the provided arguments.
6. **Result Integration**: Tool results are sent back to the LLM for further reasoning or final response generation.
7. **Respond to User**: The agent returns the final answer to the user.

## Example Class Structure
```python
class BaseAgent:
    def __init__(self, mcp_client: MCPClient, llm_client: LLMClient):
        self.mcp_client = mcp_client
        self.llm_client = llm_client
        self.context = []

    def process_message(self, user_message: str) -> str:
        """
        Processes a user message, plans tool use, executes tools, and returns a response.
        Args:
            user_message (str): The user's input message.
        Returns:
            str: The agent's response.
        """
        # Discover tools
        tools = self.mcp_client.list_tools()
        # Convert to LLM function schemas
        function_schemas = self._convert_tools(tools)
        # Send to LLM
        llm_response = self.llm_client.chat_completion(
            user_message, function_schemas, self.context
        )
        # Execute tool calls if present
        if llm_response.tool_calls:
            for call in llm_response.tool_calls:
                result = self.mcp_client.execute_tool(call.name, call.arguments)
                self.context.append(result)
        # Generate final response
        return llm_response.final_answer
```

## Design Principles
- **Modularity**: Each component has a clear responsibility.
- **Extensibility**: Easy to add new tools or MCP servers.
- **Reliability**: Robust error handling and context management.
- **Maintainability**: Clean interfaces and comprehensive documentation.

## Operational Requirements
- **Logging**: Use Loguru to capture structured logs across agent workflows.
- **Tracing**: Integrate Traceloop SDK for end-to-end traceability and metrics.
- **MCP Scripts**: Leverage the off-the-shelf `mcp` package to manage MCP server connections and tooling.

## Next Steps
- Implement the BaseAgent class in code.
- Develop MCP client and LLM client modules.
- Add unit tests for all components.
- Document usage examples and edge cases.
