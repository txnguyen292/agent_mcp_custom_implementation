"""
Streamlit application for exploring agent telemetry runs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

DEFAULT_DASHBOARD_ROOT = Path("dashboards")


def list_run_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(root.glob("**/*.json"), key=lambda path: path.stat().st_mtime, reverse=True)


def load_run(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    data["_source_path"] = path
    return data


def format_run_label(path: Path, root: Path) -> str:
    timestamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    try:
        relative = path.relative_to(root)
    except ValueError:
        try:
            relative = path.relative_to(DEFAULT_DASHBOARD_ROOT)
        except ValueError:
            relative = path.name
    return f"{relative} · {timestamp}"


def main(default_root: Path | str = DEFAULT_DASHBOARD_ROOT) -> None:
    st.set_page_config(page_title="Agent Telemetry Dashboard", layout="wide")

    root_input = st.sidebar.text_input(
        "Telemetry directory", str(default_root), help="Folder containing *.json telemetry files."
    )
    root = Path(root_input).expanduser().resolve()

    refresh = st.sidebar.button("Refresh")
    if refresh:
        st.rerun()

    run_files = list_run_files(root)
    if not run_files:
        st.info(f"No telemetry JSON files found under {root}. Run the agent to generate artifacts.")
        return

    labels = [format_run_label(path, root) for path in run_files]
    selected_label = st.sidebar.selectbox("Select run", labels, index=0)
    selected_path = run_files[labels.index(selected_label)]
    data = load_run(selected_path)

    events = data.get("events", [])
    spans = data.get("spans", [])

    st.title("Agent Telemetry Dashboard")
    st.caption(f"Viewing: `{selected_path}`")

    st.sidebar.metric("Events", len(events))
    st.sidebar.metric("Spans", len(spans))

    tabs = st.tabs(["Summary", "Conversation", "Timeline & Details"])

    with tabs[0]:
        show_summary(events, spans)

    with tabs[1]:
        show_conversation_view(events)

    with tabs[2]:
        show_detailed_view(events, spans, data)


def render_value(value: Any) -> None:
    if isinstance(value, (dict, list)):
        st.json(value)
    else:
        st.write(value)


def render_content_html(content: Any) -> str:
    if isinstance(content, (dict, list)):
        return "<pre style='background:#0f172a;color:#f8fafc;padding:0.75rem;border-radius:8px;overflow-x:auto;'>" + json.dumps(content, indent=2) + "</pre>"
    if content is None:
        return "<em>None</em>"
    return f"<div style='white-space:pre-wrap;'>{content}</div>"


def show_summary(events: List[Dict[str, Any]], spans: List[Dict[str, Any]]) -> None:
    """Render a high-level overview with two tables."""
    conversation_rows: List[Dict[str, Any]] = []
    for idx, event in enumerate(events, start=1):
        event_type = event.get("type", "unknown")
        payload = event.get("payload") or {}
        timestamp = event.get("timestamp", "")

        if event_type == "system_message":
            conversation_rows.append(
                {
                    "Turn": idx,
                    "Role": "system",
                    "Content": payload.get("content", ""),
                    "Timestamp": timestamp,
                }
            )
        elif event_type == "user_message":
            conversation_rows.append(
                {
                    "Turn": idx,
                    "Role": "user",
                    "Content": payload.get("content", ""),
                    "Timestamp": timestamp,
                }
            )
        elif event_type == "llm_response":
            if payload.get("reasoning"):
                conversation_rows.append(
                    {
                        "Turn": idx,
                        "Role": "assistant (reasoning)",
                        "Content": payload.get("reasoning", ""),
                        "Timestamp": timestamp,
                    }
                )
            response_text = payload.get("answer") or payload.get("content")
            if response_text:
                conversation_rows.append(
                    {
                        "Turn": idx,
                        "Role": "assistant",
                        "Content": response_text,
                        "Timestamp": timestamp,
                    }
                )

    if conversation_rows:
        st.markdown("### Conversation Overview")
        st.dataframe(conversation_rows, use_container_width=True)
    else:
        st.info("No conversation turns recorded.")

    if spans:
        st.markdown("### Span Summary")
        span_rows = [
            {
                "Name": span.get("name"),
                "Duration (ms)": span.get("duration_ms"),
                "Start": span.get("start_time"),
                "End": span.get("end_time"),
            }
            for span in spans
        ]
        st.dataframe(span_rows, use_container_width=True)
    else:
        st.info("No spans recorded.")


def show_conversation_view(events: List[Dict[str, Any]]) -> None:
    """Render a simplified turn-by-turn conversation view."""
    st.markdown("### Conversation Play-by-Play")
    turns = build_conversation_turns(events)

    for idx, turn in enumerate(turns, start=1):
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        timestamp = turn.get("timestamp", "")
        metadata = turn.get("metadata")

        with st.container():
            st.markdown(
                f"<div style='background:#f8fafc;border-radius:12px;padding:1rem;margin-bottom:0.75rem;'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                f"<span style='font-weight:600;text-transform:capitalize;'>{idx}. {role}</span>"
                f"<span style='font-size:0.8rem;color:#64748b;'>{timestamp}</span>"
                f"</div>"
                f"<div style='margin-top:0.5rem;padding-left:1rem;border-left:3px solid #cbd5f5;'>"
                f"{render_content_html(content)}"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            if metadata:
                with st.expander("Metadata"):
                    st.json(metadata)


def show_detailed_view(
    events: List[Dict[str, Any]],
    spans: List[Dict[str, Any]],
    data: Dict[str, Any],
) -> None:
    """Render the detailed timeline and raw data view."""
    event_types = sorted({event.get("type", "unknown") for event in events})
    selected_types = st.multiselect(
        "Filter event types", event_types, default=event_types
    )

    st.markdown("### Timeline")
    for idx, event in enumerate(events, start=1):
        event_type = event.get("type", "unknown")
        if event_type not in selected_types:
            continue

        timestamp = event.get("timestamp", "unknown time")
        heading = f"{idx:02d}. {event_type} · {timestamp}"
        expanded = event_type in {"final_response", "tool_error"}

        with st.expander(heading, expanded=expanded):
            payload = event.get("payload") or {}
            render_event_payload(event_type, payload)

    st.markdown("### Spans")
    if spans:
        span_rows = [
            {
                "Name": span.get("name"),
                "Duration (ms)": span.get("duration_ms"),
                "Start": span.get("start_time"),
                "End": span.get("end_time"),
                "Attributes": span.get("attributes"),
            }
            for span in spans
        ]
        st.dataframe(span_rows, use_container_width=True)
    else:
        st.write("No spans recorded.")

    st.markdown("### Raw Data")
    st.json(data)


def build_conversation_turns(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    for event in events:
        event_type = event.get("type", "unknown")
        payload = event.get("payload") or {}
        timestamp = event.get("timestamp", "")

        if event_type == "system_message":
            turns.append(
                {"role": "system", "content": payload.get("content", ""), "timestamp": timestamp}
            )
        elif event_type == "user_message":
            turns.append(
                {"role": "user", "content": payload.get("content", ""), "timestamp": timestamp}
            )
        elif event_type == "llm_response":
            if payload.get("reasoning"):
                turns.append(
                    {
                        "role": "assistant (reasoning)",
                        "content": payload["reasoning"],
                        "timestamp": timestamp,
                    }
                )
            response_text = payload.get("answer") or payload.get("content")
            if response_text:
                turns.append(
                    {"role": "assistant", "content": response_text, "timestamp": timestamp}
                )
        elif event_type == "tool_call":
            turns.append(
                {
                    "role": f"tool:{payload.get('tool', 'unknown')}",
                    "content": payload.get("arguments"),
                    "timestamp": timestamp,
                    "metadata": {"type": "tool_call"},
                }
            )
        elif event_type == "tool_result":
            turns.append(
                {
                    "role": f"tool:{payload.get('tool', 'unknown')}:result",
                    "content": payload.get("result"),
                    "timestamp": timestamp,
                    "metadata": {"type": "tool_result"},
                }
            )
    return turns


def render_event_payload(event_type: str, payload: Dict[str, Any]) -> None:
    if event_type == "user_message":
        st.markdown("**User Message**")
        st.write(payload.get("content", ""))
    elif event_type == "system_message":
        st.markdown("**System Instruction**")
        st.info(payload.get("content", ""))
    elif event_type == "llm_response":
        if payload.get("answer"):
            st.markdown("**Answer**")
            st.success(payload["answer"])
        if payload.get("reasoning"):
            st.markdown("**Reasoning**")
            st.info(payload["reasoning"])
        if payload.get("content") and not payload.get("answer"):
            st.markdown("**Assistant Reply**")
            st.write(payload.get("content"))
        if payload.get("tool_calls"):
            st.markdown("**Planned Tool Calls**")
            st.json(payload.get("tool_calls"))
    elif event_type in {"tool_call", "tool_result", "tool_error"}:
        st.markdown(f"**Tool:** `{payload.get('tool', 'unknown')}`")
        if event_type == "tool_call":
            st.markdown("**Arguments**")
            render_value(payload.get("arguments", {}))
        elif event_type == "tool_result":
            st.markdown("**Result**")
            render_value(payload.get("result"))
        else:
            st.markdown("**Error**")
            st.error(payload.get("error", ""))
    elif event_type == "final_response":
        if payload.get("answer"):
            st.markdown("**Final Answer**")
            st.success(payload["answer"])
        if payload.get("reasoning"):
            st.markdown("**Reasoning**")
            st.info(payload["reasoning"])
        if payload.get("content") and not payload.get("answer"):
            st.markdown("**Assistant Reply**")
            st.write(payload.get("content"))
    else:
        st.json(payload)


if __name__ == "__main__":
    main()
