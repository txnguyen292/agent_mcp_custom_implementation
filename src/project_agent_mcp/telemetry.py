"""
Telemetry and dashboard utilities for the BaseAgent.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class SpanRecord:
    name: str
    start_time: float
    end_time: Optional[float] = None
    attributes: Optional[Dict[str, Any]] = None

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return round((self.end_time - self.start_time) * 1000, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat()
            if self.end_time
            else None,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes or {},
        }


class TelemetryManager:
    """
    Collects spans and events to power a lightweight local dashboard.
    """

    def __init__(
        self,
        dashboard_dir: Path | str = "dashboards",
        dashboard_filename: str = "index.html",
        enable_dashboard: bool = True,
    ) -> None:
        self.dashboard_dir = Path(dashboard_dir)
        self.dashboard_filename = dashboard_filename
        self.enable_dashboard = enable_dashboard
        self.events: List[Dict[str, Any]] = []
        self.spans: List[SpanRecord] = []
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logger.bind(component="TelemetryManager")

    @contextmanager
    def span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Context manager that records timing information for a block."""
        record = SpanRecord(name=name, start_time=time.time(), attributes=attributes or {})
        try:
            yield record
            record.end_time = time.time()
            self.spans.append(record)
        except Exception:
            record.end_time = time.time()
            record.attributes["status"] = "error"
            self.spans.append(record)
            raise
        finally:
            duration = record.duration_ms if record.duration_ms is not None else "N/A"
            self._logger.debug(f"Span '{name}' completed in {duration}ms")

    def record_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Store an event that will appear in the dashboard."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            "payload": payload,
        }
        self.events.append(event)
        self._logger.debug(f"Recorded telemetry event '{event_type}'")

    def write_dashboard(self) -> Optional[Path]:
        """Render the collected telemetry into a standalone HTML dashboard."""
        if not self.enable_dashboard:
            self._logger.debug("Dashboard generation disabled; skipping write.")
            return None

        dashboard_path = self.dashboard_dir / self.dashboard_filename
        data = {
            "events": self.events,
            "spans": [span.to_dict() for span in self.spans],
            "generated_at": datetime.utcnow().isoformat(),
        }
        html = self._render_dashboard_html(data)
        dashboard_path.write_text(html, encoding="utf-8")
        self._logger.info(f"Dashboard written to {dashboard_path}")
        return dashboard_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _render_dashboard_html(self, data: Dict[str, Any]) -> str:
        serialized = json.dumps(data, indent=2)
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <title>Agent Run Dashboard</title>
    <style>
        :root {{
            color-scheme: light dark;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
            margin: 2rem;
            background: #f8f9fb;
            color: #1f2933;
        }}
        h1 {{
            margin-bottom: 0.5rem;
        }}
        .meta {{
            margin-bottom: 2rem;
            color: #52606d;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1.5rem;
        }}
        .card {{
            background: #fff;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
        }}
        .timeline {{
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }}
        .timeline-entry {{
            border-radius: 10px;
            padding: 1rem;
            border: 1px solid #e2e8f0;
            background: #f8fafc;
        }}
        .timeline-entry.user {{ border-left: 4px solid #2563eb; }}
        .timeline-entry.assistant {{ border-left: 4px solid #16a34a; }}
        .timeline-entry.tool {{ border-left: 4px solid #d97706; }}
        .timeline-entry.system {{ border-left: 4px solid #9333ea; }}
        .timeline-entry h3 {{
            margin: 0 0 0.35rem;
            font-size: 1rem;
        }}
        .timeline-entry .timestamp {{
            display: block;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 0.5rem;
        }}
        .timeline-entry pre {{
            background: #0f172a;
            color: #f8fafc;
            padding: 0.75rem;
            border-radius: 8px;
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        th, td {{
            padding: 0.75rem;
            border-bottom: 1px solid #e2e8f0;
            text-align: left;
            vertical-align: top;
        }}
        th {{
            background: #f1f5f9;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.08em;
            color: #475569;
        }}
        pre {{
            background: #0f172a;
            color: #f8fafc;
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
        }}
        .tag {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            background: #e0f2fe;
            color: #0369a1;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .reasoning {{
            background: rgba(59, 130, 246, 0.08);
            border-left: 3px solid #3b82f6;
            padding: 0.75rem;
            border-radius: 8px;
            margin-top: 0.75rem;
        }}
    </style>
</head>
<body>
    <h1>Agent Run Dashboard</h1>
    <p class="meta">Generated at {data['generated_at']}</p>

    <div class="grid">
        <div class="card">
            <h2>Conversation Timeline</h2>
            <div id="timeline" class="timeline"></div>
        </div>
        <div class="card">
            <h2>Spans</h2>
            <table>
                <thead>
                    <tr><th>Name</th><th>Duration (ms)</th><th>Attributes</th></tr>
                </thead>
                <tbody id="span-table-body"></tbody>
            </table>
        </div>
        <div class="card">
            <h2>Event Log</h2>
            <table>
                <thead>
                    <tr><th>Timestamp</th><th>Type</th><th>Payload</th></tr>
                </thead>
                <tbody id="event-table-body"></tbody>
            </table>
        </div>
        <div class="card">
            <h2>Raw Data</h2>
            <pre id="raw-data"></pre>
        </div>
    </div>

    <script>
        const data = {serialized};
        const formatJSON = (payload) => {{
            if (payload === undefined) return "undefined";
            if (payload === null) return "null";
            if (typeof payload === "string") return payload;
            if (typeof payload === "number" || typeof payload === "boolean") return String(payload);
            return JSON.stringify(payload, null, 2);
        }};
        const timelineContainer = document.getElementById("timeline");

        const TIMELINE_TYPES = {{
            user_message: {{ role: "user", title: "User Message" }},
            llm_response: {{ role: "assistant", title: "LLM Response" }},
            tool_call: {{ role: "tool", title: "Tool Call" }},
            tool_result: {{ role: "tool", title: "Tool Result" }},
            tool_error: {{ role: "tool", title: "Tool Error" }},
            final_response: {{ role: "assistant", title: "Final Response" }},
        }};

        const renderTimelineEntry = (event) => {{
            const info = TIMELINE_TYPES[event.type];
            if (!info) return;

            const entry = document.createElement("div");
            entry.className = "timeline-entry " + info.role;

            const header = document.createElement("h3");
            header.textContent = info.title;
            entry.appendChild(header);

            const timestamp = document.createElement("span");
            timestamp.className = "timestamp";
            timestamp.textContent = event.timestamp;
            entry.appendChild(timestamp);

            const payload = event.payload || {{}};

            if (event.type === "user_message") {{
                entry.appendChild(createTextParagraph(payload.content ?? ""));
            }} else if (event.type === "llm_response") {{
                if (payload.content) {{
                    entry.appendChild(createSection("Assistant Reply", payload.content));
                }}
                if (payload.reasoning) {{
                    entry.appendChild(createReasoning(payload.reasoning));
                }}
                if (payload.tool_calls && payload.tool_calls.length) {{
                    const list = document.createElement("ul");
                    payload.tool_calls.forEach((call) => {{
                        const item = document.createElement("li");
                        item.innerHTML = "<strong>" + call.name + "</strong> " + formatJSON(call.arguments);
                        list.appendChild(item);
                    }});
                    const wrapper = document.createElement("div");
                    wrapper.innerHTML = "<strong>Planned Tool Calls</strong>";
                    wrapper.appendChild(list);
                    entry.appendChild(wrapper);
                }}
            }} else if (event.type === "tool_call") {{
                entry.appendChild(createTextParagraph("Tool: " + (payload.tool ?? "unknown")));
                if (payload.call_id) {{
                    entry.appendChild(createTag("call id: " + payload.call_id));
                }}
                entry.appendChild(createCodeBlock(payload.arguments));
            }} else if (event.type === "tool_result") {{
                entry.appendChild(createTextParagraph("Tool: " + (payload.tool ?? "unknown")));
                entry.appendChild(createCodeBlock(payload.result));
            }} else if (event.type === "tool_error") {{
                entry.appendChild(createTextParagraph("Tool: " + (payload.tool ?? "unknown")));
                const errorTag = document.createElement("span");
                errorTag.className = "tag";
                errorTag.textContent = "Error";
                entry.appendChild(errorTag);
                entry.appendChild(createTextParagraph(payload.error ?? ""));
            }} else if (event.type === "final_response") {{
                entry.appendChild(createSection("Assistant Reply", payload.content ?? ""));
            }}

            timelineContainer.appendChild(entry);
        }};

        const createTextParagraph = (text) => {{
            const p = document.createElement("p");
            p.textContent = text;
            return p;
        }};

        const createSection = (title, text) => {{
            const wrapper = document.createElement("div");
            const heading = document.createElement("strong");
            heading.textContent = title;
            wrapper.appendChild(heading);
            const body = document.createElement("p");
            body.textContent = text;
            wrapper.appendChild(body);
            return wrapper;
        }};

        const createReasoning = (text) => {{
            const block = document.createElement("div");
            block.className = "reasoning";
            const title = document.createElement("strong");
            title.textContent = "Model Reasoning";
            block.appendChild(title);
            const body = document.createElement("p");
            body.textContent = text;
            block.appendChild(body);
            return block;
        }};

        const createTag = (label) => {{
            const span = document.createElement("span");
            span.className = "tag";
            span.textContent = label;
            return span;
        }};

        const createCodeBlock = (value) => {{
            const pre = document.createElement("pre");
            pre.textContent = formatJSON(value);
            return pre;
        }};

        data.events.forEach(renderTimelineEntry);

        const spanBody = document.getElementById("span-table-body");
        data.spans.forEach((span) => {{
            const row = document.createElement("tr");
            row.innerHTML = `
                <td>${{span.name}}</td>
                <td>${{span.duration_ms ?? "N/A"}}</td>
                <td><pre>${{formatJSON(span.attributes)}}</pre></td>
            `;
            spanBody.appendChild(row);
        }});

        const eventBody = document.getElementById("event-table-body");
        data.events.forEach((event) => {{
            const row = document.createElement("tr");
            row.innerHTML = `
                <td>${{event.timestamp}}</td>
                <td>${{event.type}}</td>
                <td><pre>${{formatJSON(event.payload)}}</pre></td>
            `;
            eventBody.appendChild(row);
        }});

        document.getElementById("raw-data").textContent = formatJSON(data);
    </script>
</body>
</html>
"""
