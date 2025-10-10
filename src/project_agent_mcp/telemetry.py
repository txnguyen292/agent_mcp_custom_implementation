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
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; margin: 2rem; background: #f8f9fb; color: #1f2933; }}
        h1 {{ margin-bottom: 0.5rem; }}
        .meta {{ margin-bottom: 2rem; color: #52606d; }}
        .card {{ background: #fff; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 10px 30px rgba(15,23,42,0.08); }}
        pre {{ background: #0f172a; color: #f8fafc; padding: 1rem; border-radius: 6px; overflow-x: auto; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
        th, td {{ padding: 0.75rem; border-bottom: 1px solid #e2e8f0; text-align: left; }}
        th {{ background: #f1f5f9; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.08em; color: #475569; }}
        .badge {{ display: inline-block; padding: 0.2rem 0.5rem; border-radius: 999px; background: #e0f2fe; color: #0284c7; font-size: 0.75rem; }}
    </style>
</head>
<body>
    <h1>Agent Run Dashboard</h1>
    <p class="meta">Generated at {data['generated_at']}</p>

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
        <h2>Events</h2>
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

    <script>
        const data = {serialized};
        const formatJSON = (payload) => JSON.stringify(payload, null, 2);

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
                <td><span class="badge">${{event.timestamp}}</span></td>
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
