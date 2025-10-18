import argparse
import builtins
import os
from pathlib import Path
from rich import print
from project_agent_mcp import AgentRunner, build_runner, register_basic_math_tools

builtins.print = print  # Override built-in print with rich print

def build_cli_runner() -> AgentRunner:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY environment variable must be set before running the agent."
        )
    return build_runner(
        model="gpt-4o-mini",
        tool_factories=[register_basic_math_tools],
        dashboard_dir=Path("dashboards") / "cli",
        dashboard_filename="run.html",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BaseAgent against the OpenAI API.")
    parser.add_argument(
        "--prompt",
        type=str,
        help="Optional single prompt to run. If omitted, the agent starts an interactive session.",
    )
    args = parser.parse_args()

    runner = build_cli_runner()

    if args.prompt:
        response = runner.run_once(args.prompt)
        dashboard_path = runner.telemetry.dashboard_dir / runner.telemetry.dashboard_filename
        print("Agent response:", response)
        print("Dashboard written to:", dashboard_path.resolve())
    else:
        runner.interactive()


if __name__ == "__main__":
    main()
