from __future__ import annotations

import queue
import threading
import tkinter as tk
from datetime import datetime
from tkinter import ttk
from typing import Any


ROUTES = (
    "1", "2", "3", "4", "5", "6", "7", "S", "L", "SIR",
    "N", "Q", "R", "W", "J", "Z", "G", "B", "D", "F",
    "M", "SK", "A", "C", "E", "SR",
)

ROUTE_COLORS = {
    "1": "#ee352e", "2": "#ee352e", "3": "#ee352e",
    "4": "#00933c", "5": "#00933c", "6": "#00933c",
    "7": "#b933ad", "S": "#6d6e71",
    "A": "#0039a6", "C": "#0039a6", "E": "#0039a6",
    "B": "#ff6319", "D": "#ff6319", "F": "#ff6319",
    "M": "#ff6319", "G": "#6cbe45",
    "J": "#996633", "Z": "#996633", "L": "#a7a9ac",
    "N": "#fccc0a", "Q": "#fccc0a", "R": "#fccc0a",
    "W": "#fccc0a", "SIR": "#0039a6",
    "SK": "#6d6e71", "SR": "#6d6e71",
}

BACKGROUND = "#f3f6fa"
CARD = "#ffffff"
INK = "#172338"
MUTED = "#718096"
ACCENT = "#1769aa"
NA = "N/A"


def _format_value(value: Any) -> str:
    if value is None or value == "":
        return NA
    return str(value)


def _format_direction(record: dict[str, Any]) -> str:
    direction = record.get("direction", record.get("direction_name"))
    if direction not in (None, ""):
        return str(direction)
    direction_id = record.get("direction_id")
    if direction_id not in (None, ""):
        return f"Dir {direction_id}"
    return NA


def _format_incident(record: dict[str, Any]) -> str:
    incident = record.get("incident", record.get("predicted_incident"))
    if incident not in (None, ""):
        if isinstance(incident, bool):
            return "Incident predicted" if incident else "No incident predicted"
        return str(incident)
    has_incident = record.get("has_incident")
    if isinstance(has_incident, bool):
        return "Incident predicted" if has_incident else "No incident predicted"
    return NA


def _format_delay(record: dict[str, Any]) -> str:
    delay = record.get(
        "delay_minutes",
        record.get("predicted_delay_minutes", record.get("delay")),
    )
    if delay is None or delay == "":
        return NA
    if isinstance(delay, (int, float)):
        return f"{delay:g} min"
    return str(delay)


def _format_confidence(record: dict[str, Any]) -> str:
    confidence = record.get("confidence", record.get("confidence_score"))
    if confidence is None or confidence == "":
        return NA
    if isinstance(confidence, (int, float)):
        if 0 <= confidence <= 1:
            confidence *= 100
        return f"{confidence:g}%"
    return str(confidence)


def normalize_predictions(data: Any) -> dict[str, list[dict[str, str]]]:
    """Map predictor records to dashboard rows, retaining N/A rows for missing routes."""
    if isinstance(data, dict) and "predictions" in data:
        data = data["predictions"]

    if isinstance(data, dict):
        if any(key in data for key in ("route_id", "line", "route")):
            records = [data]
        else:
            records = []
            for route, value in data.items():
                if isinstance(value, dict):
                    records.append({"route_id": route, **value})
                elif isinstance(value, list):
                    records.extend(
                        {"route_id": route, **item}
                        for item in value
                        if isinstance(item, dict)
                    )
    elif isinstance(data, list):
        records = [item for item in data if isinstance(item, dict)]
    else:
        records = []

    grouped: dict[str, list[dict[str, str]]] = {route: [] for route in ROUTES}
    for record in records:
        route = record.get("route_id", record.get("line", record.get("route")))
        if route is None:
            continue
        route = str(route).upper()
        if route not in grouped:
            continue
        grouped[route].append({
            "direction": _format_direction(record),
            "incident": _format_incident(record),
            "delay": _format_delay(record),
            "confidence": _format_confidence(record),
        })

    for route, rows in grouped.items():
        if not rows:
            grouped[route] = [{
                "direction": NA,
                "incident": NA,
                "delay": NA,
                "confidence": NA,
            }]
    return grouped


class MtaDashboard:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Predict MTA | Subway outlook")
        self.root.geometry("1120x760")
        self.root.minsize(820, 560)
        self.root.configure(bg=BACKGROUND)

        self.result_queue: queue.Queue[tuple[Any, str]] = queue.Queue()
        self.loading = False
        self.route_rows: dict[str, list[str]] = {}
        self.status_var = tk.StringVar(value="Ready")
        self.updated_var = tk.StringVar(value="Waiting for prediction data")
        self._configure_styles()
        self._build_layout()
        self._render(normalize_predictions(None))
        self.refresh()
        self.root.after(100, self._check_results)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "Routes.Treeview",
            background=CARD,
            fieldbackground=CARD,
            foreground=INK,
            rowheight=48,
            borderwidth=0,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Routes.Treeview.Heading",
            background="#edf2f7",
            foreground=MUTED,
            font=("Segoe UI", 9, "bold"),
            padding=(12, 12),
            relief="flat",
        )
        style.map(
            "Routes.Treeview",
            background=[("selected", "#dceeff")],
            foreground=[("selected", INK)],
        )
        style.configure(
            "Refresh.TButton",
            background=ACCENT,
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            padding=(16, 10),
            borderwidth=0,
        )
        style.map("Refresh.TButton", background=[("active", "#12578c")])

    def _build_layout(self) -> None:
        page = tk.Frame(self.root, bg=BACKGROUND, padx=30, pady=24)
        page.pack(fill="both", expand=True)

        header = tk.Frame(page, bg=BACKGROUND)
        header.pack(fill="x", pady=(0, 20))
        title_block = tk.Frame(header, bg=BACKGROUND)
        title_block.pack(side="left", fill="x", expand=True)
        tk.Label(
            title_block,
            text="Predict MTA",
            bg=BACKGROUND,
            fg=INK,
            font=("Segoe UI", 25, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title_block,
            text="NYC subway · next-hour incident outlook",
            bg=BACKGROUND,
            fg=MUTED,
            font=("Segoe UI", 11),
        ).pack(anchor="w", pady=(3, 0))

        self.refresh_button = ttk.Button(
            header,
            text="Refresh predictions",
            style="Refresh.TButton",
            command=self.refresh,
        )
        self.refresh_button.pack(side="right", anchor="center")

        stats = tk.Frame(page, bg=BACKGROUND)
        stats.pack(fill="x", pady=(0, 18))
        self._stat_card(stats, "SUBWAY SERVICES", str(len(ROUTES)), "Routes tracked")
        self.incident_value = self._stat_card(
            stats, "PREDICTED INCIDENTS", NA, "Next hour"
        )
        self.confidence_value = self._stat_card(
            stats, "AVG. CONFIDENCE", NA, "Available predictions"
        )

        table_card = tk.Frame(page, bg=CARD, highlightthickness=1,
                              highlightbackground="#e2e8f0")
        table_card.pack(fill="both", expand=True)
        table_heading = tk.Frame(table_card, bg=CARD, padx=18, pady=15)
        table_heading.pack(fill="x")
        tk.Label(
            table_heading,
            text="Subway line status",
            bg=CARD,
            fg=INK,
            font=("Segoe UI", 13, "bold"),
        ).pack(side="left")
        tk.Label(
            table_heading,
            text="PREDICTIONS",
            bg=CARD,
            fg=MUTED,
            font=("Segoe UI", 8, "bold"),
        ).pack(side="right")

        columns = ("line", "direction", "incident", "delay", "confidence")
        self.tree = ttk.Treeview(
            table_card,
            columns=columns,
            show="headings",
            style="Routes.Treeview",
            selectmode="browse",
        )
        for column, heading, width in (
            ("line", "LINE", 110),
            ("direction", "DIRECTION", 170),
            ("incident", "PREDICTED INCIDENT", 300),
            ("delay", "PREDICTED DELAY", 190),
            ("confidence", "CONFIDENCE", 150),
        ):
            self.tree.heading(column, text=heading, anchor="w")
            self.tree.column(column, width=width, minwidth=80, anchor="w")
        scrollbar = ttk.Scrollbar(
            table_card, orient="vertical", command=self.tree.yview
        )
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=(0, 10))
        scrollbar.pack(side="right", fill="y", padx=(0, 10), pady=(0, 10))

        footer = tk.Frame(page, bg=BACKGROUND)
        footer.pack(fill="x", pady=(14, 0))
        tk.Label(
            footer,
            textvariable=self.status_var,
            bg=BACKGROUND,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(side="left")
        tk.Label(
            footer,
            textvariable=self.updated_var,
            bg=BACKGROUND,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(side="right")

    @staticmethod
    def _stat_card(parent: tk.Frame, title: str, value: str, note: str) -> tk.Label:
        card = tk.Frame(
            parent,
            bg=CARD,
            padx=18,
            pady=15,
            highlightthickness=1,
            highlightbackground="#e2e8f0",
        )
        card.pack(side="left", fill="both", expand=True, padx=(0, 12))
        tk.Label(
            card, text=title, bg=CARD, fg=MUTED,
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w")
        value_label = tk.Label(
            card, text=value, bg=CARD, fg=INK,
            font=("Segoe UI", 20, "bold"),
        )
        value_label.pack(anchor="w", pady=(6, 1))
        tk.Label(
            card, text=note, bg=CARD, fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(anchor="w")
        return value_label

    def _render(self, predictions: dict[str, list[dict[str, str]]]) -> None:
        self.tree.delete(*self.tree.get_children())
        self.route_rows.clear()
        confidence_values: list[float] = []
        incidents = 0

        for route in ROUTES:
            for prediction in predictions[route]:
                incident = prediction["incident"]
                incident_text = incident.lower()
                if incident_text not in (NA.lower(), "no", "none", "false") \
                        and not incident_text.startswith("no "):
                    incidents += 1
                confidence = prediction["confidence"]
                if confidence.endswith("%"):
                    try:
                        confidence_values.append(float(confidence[:-1]))
                    except ValueError:
                        pass
                item_id = self.tree.insert(
                    "",
                    "end",
                    values=(
                        f"●  {route}",
                        prediction["direction"],
                        incident,
                        prediction["delay"],
                        confidence,
                    ),
                    tags=(route,),
                )
                self.route_rows.setdefault(route, []).append(item_id)

        for route, color in ROUTE_COLORS.items():
            self.tree.tag_configure(route, foreground=color)
        self.incident_value.configure(text=str(incidents))
        average = (
            f"{sum(confidence_values) / len(confidence_values):.0f}%"
            if confidence_values else NA
        )
        self.confidence_value.configure(text=average)

    def refresh(self) -> None:
        if self.loading:
            return
        self.loading = True
        self.refresh_button.configure(state="disabled")
        self.status_var.set("Loading prediction data from predict.py…")

        def load() -> None:
            try:
                from predict import get_predictions

                result = get_predictions()
                self.result_queue.put((result, ""))
            except Exception as error:
                self.result_queue.put((None, f"{type(error).__name__}: {error}"))

        threading.Thread(target=load, daemon=True).start()

    def _check_results(self) -> None:
        try:
            result, error = self.result_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._check_results)
            return

        self.loading = False
        self.refresh_button.configure(state="normal")
        self._render(normalize_predictions(result))
        self.updated_var.set(f"Updated {datetime.now().strftime('%I:%M:%S %p')}")
        if error:
            self.status_var.set(f"Prediction unavailable — showing N/A defaults ({error})")
        elif result is None:
            self.status_var.set("Prediction module is not ready — showing N/A defaults")
        else:
            self.status_var.set("Showing results from predict.py")
        self.root.after(100, self._check_results)


def main() -> None:
    root = tk.Tk()
    MtaDashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()