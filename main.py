import tkinter as tk
from tkinter import ttk
import threading
import queue
import math
import argparse
from pathlib import Path
import pandas as pd
from datetime import date, datetime
from zoneinfo import ZoneInfo

from predict import get_predictions
from train.train import TZ


MODERATE_PROBABILITY_THRESHOLD = 0.25
HIGH_PROBABILITY_THRESHOLD = 0.50


DIRECTION_TERMINALS = {
    "1": ("Van Cortlandt Park–242 St", "South Ferry"),
    "2": ("Wakefield–241 St", "Flatbush Av–Brooklyn College"),
    "3": ("Harlem–148 St", "New Lots Av"),
    "4": ("Woodlawn", "Crown Hts–Utica Av / New Lots Av"),
    "5": ("Eastchester–Dyre Av / Nereid Av", "Flatbush Av / Bowling Green"),
    "6": ("Pelham Bay Park / Parkchester", "Brooklyn Bridge–City Hall"),
    "7": ("Flushing–Main St", "34 St–Hudson Yards"),
    "A": ("Inwood–207 St", "Ozone Park–Lefferts Blvd / Far Rockaway / Rockaway Park"),
    "B": ("Bedford Park Blvd / 145 St", "Brighton Beach"),
    "C": ("168 St", "Euclid Av"),
    "D": ("Norwood–205 St", "Coney Island–Stillwell Av"),
    "E": ("Jamaica Center–Parsons/Archer", "World Trade Center"),
    "F": ("Jamaica–179 St", "Coney Island–Stillwell Av"),
    "G": ("Court Sq", "Church Av"),
    "J": ("Jamaica Center–Parsons/Archer", "Broad St"),
    "L": ("8 Av", "Canarsie–Rockaway Pkwy"),
    "M": ("Forest Hills–71 Av", "Middle Village–Metropolitan Av"),
    "N": ("Astoria–Ditmars Blvd", "Coney Island–Stillwell Av"),
    "Q": ("96 St", "Coney Island–Stillwell Av"),
    "R": ("Forest Hills–71 Av", "Bay Ridge–95 St"),
    "W": ("Astoria–Ditmars Blvd", "Whitehall St–South Ferry"),
    "Z": ("Jamaica Center–Parsons/Archer", "Broad St"),
    "SI": ("St. George", "Tottenville"),
    "SIR": ("St. George", "Tottenville"),
}

DIRECTION_NAMES = {
    "7": ("Uptown / Eastbound", "Downtown / Westbound"),
    "L": ("Uptown / Westbound", "Downtown / Eastbound"),
    "SI": ("Northbound", "Southbound"),
    "SIR": ("Northbound", "Southbound"),
}


def format_line_direction(line, direction_id):
    line = str(line).upper()
    display_line = "SIR" if line == "SI" else line
    terminal_line = line[:-1] if line in ("6X", "7X") else line
    direction_id = int(direction_id)
    if direction_id not in (0, 1):
        direction = "Unknown direction"
    elif terminal_line in DIRECTION_TERMINALS:
        direction = DIRECTION_NAMES.get(
            terminal_line, ("Uptown / Northbound", "Downtown / Southbound")
        )[direction_id]
        direction = f"{direction} ({DIRECTION_TERMINALS[terminal_line][direction_id]})"
    else:
        direction = ("Uptown / Northbound", "Downtown / Southbound")[direction_id]
    return f"{display_line} - {direction}"


def probability_band(probability):
    if probability >= HIGH_PROBABILITY_THRESHOLD:
        return "HIGH"
    if probability >= MODERATE_PROBABILITY_THRESHOLD:
        return "MODERATE"
    return "LOW"


class RoundedPanel(tk.Frame):
    def __init__(self, parent, background, surface, border, radius=18, inset=9, **kwargs):
        super().__init__(parent, bg=background, bd=0, highlightthickness=0, **kwargs)
        self._background = background
        self._surface = surface
        self._border = border
        self._radius = radius
        self._inset = inset
        self.canvas = tk.Canvas(self, bg=background, bd=0, highlightthickness=0)
        self.canvas.place(relwidth=1, relheight=1)
        self.content = tk.Frame(self, bg=surface, bd=0, highlightthickness=0)
        self.content._theme_surface = True
        self.content.place(
            x=inset,
            y=inset,
            relwidth=1,
            relheight=1,
            width=-2 * inset,
            height=-2 * inset
        )
        self.bind("<Configure>", self._draw)

    def set_theme(self, background, surface, border):
        self._background = background
        self._surface = surface
        self._border = border
        self.configure(bg=background)
        self.canvas.configure(bg=background)
        self.content.configure(bg=surface)
        self._draw()

    def _draw(self, _event=None):
        width = self.winfo_width()
        height = self.winfo_height()
        if width < 2 or height < 2:
            return
        radius = min(self._radius, width // 2, height // 2)
        points = []
        for cx, cy, start in (
            (width - radius, radius, -90),
            (width - radius, height - radius, 0),
            (radius, height - radius, 90),
            (radius, radius, 180)
        ):
            for step in range(7):
                angle = (start + step * 90 / 6) * 3.141592653589793 / 180
                points.extend((
                    cx + radius * math.cos(angle),
                    cy + radius * math.sin(angle)
                ))
        self.canvas.delete("panel")
        self.canvas.create_polygon(
            points,
            smooth=True,
            splinesteps=12,
            fill=self._surface,
            outline=self._border,
            width=1,
            tags="panel"
        )


class MtaDashboard:
    def __init__(self, root, target_date=None, target_hour=None):
        self.root = root
        self.target_date = target_date
        self.target_hour = target_hour
        self.root.title("PredictMTA - Subway Disruption Dashboard")
        self.root.geometry("1400x800")
        self.root.minsize(900, 650)
        self.root.configure(bg="#f4f4f4")
        self.logo_image = tk.PhotoImage(
            file=str(Path(__file__).resolve().with_name("logo.png"))
        )
        image_scale = max(
            1,
            (max(self.logo_image.width(), self.logo_image.height()) + 79) // 80
        )
        if image_scale > 1:
            self.logo_image = self.logo_image.subsample(image_scale, image_scale)
        self.root.iconphoto(True, self.logo_image)
        self.prediction_queue = queue.Queue()
        self.sort_reverse = {}
        self.predictions = pd.DataFrame()
        self.all_page_size = 200
        self.all_page_index = 0
        self.all_total_pages = 0
        self.all_sort_column = None
        self.all_sort_reverse = False
        self.follow_next_hour = True
        self.current_next_hour = (
            target_hour if target_hour is not None
            else 0 if target_date is not None
            else (datetime.now(ZoneInfo(TZ)).hour + 1) % 24
        )
        self.dark_mode = False
        self.themes = {
            "light": {
                "background": "#f3f6fb",
                "surface": "#ffffff",
                "text": "#172033",
                "muted": "#718096",
                "accent": "#2563eb",
                "accent_active": "#1d4ed8",
                "border": "#dfe6ef",
                "input": "#ffffff",
                "heading": "#edf2f8",
                "selected": "#dbeafe",
            },
            "dark": {
                "background": "#101722",
                "surface": "#192331",
                "text": "#e6edf7",
                "muted": "#9ba9bc",
                "accent": "#3b82f6",
                "accent_active": "#2563eb",
                "border": "#2b394b",
                "input": "#202c3b",
                "heading": "#202c3b",
                "selected": "#294a75",
            },
        }

        self.create_widgets()
        self.apply_theme()
        self.load_predictions()
        self.check_queue()
        self.update_clock()
        self.root.after_idle(self.root.focus_set)

    def create_widgets(self):
        header = tk.Frame(self.root, bg="#f4f4f4")
        header.pack(fill="x", padx=28, pady=(20, 6))
        tk.Label(
            header,
            image=self.logo_image,
            bg="#f4f4f4",
            bd=0
        ).pack(side="left", padx=(0, 12))
        title = tk.Label(
            header,
            text="NYC Subway Disruption Predictions",
            font=("Segoe UI", 22, "bold"),
            bg="#f4f4f4"
        )
        title.pack(side="left", anchor="w")
        self.theme_button = ttk.Button(
            header,
            text="Dark mode",
            command=self.toggle_theme,
            style="Action.TButton"
        )
        self.theme_button.pack(side="right", anchor="center")

        self.clock_label = tk.Label(
            self.root,
            text="Loading date and time...",
            font=("Segoe UI", 11),
            bg="#f4f4f4"
        )
        self.clock_label.pack(anchor="w", padx=30, pady=(0, 10))
        if self.target_date is not None:
            self.target_date_label = tk.Label(
                self.root,
                text=(
                    f"Target date: {self.target_date.isoformat()} · "
                    f"Target time: {self.format_hour(self.current_next_hour)}"
                ),
                font=("Segoe UI", 10, "bold"),
                bg="#f4f4f4"
            )
            self.target_date_label.pack(anchor="w", padx=30, pady=(0, 8))

        self.pages = ttk.Notebook(self.root)
        self.pages.pack(fill="both", expand=True, padx=20, pady=10)

        home_page = tk.Frame(self.pages, bg="#f4f4f4")
        self.pages.add(home_page, text="Next-hour outlook")

        # Summary statistics
        stats_frame = tk.Frame(home_page, bg="#f4f4f4")
        stats_frame.pack(fill="x", padx=30, pady=10)

        self.total_label = self.create_stat_card(
            stats_frame, "High-probability disruptions next hour", "Loading..."
        )

        self.average_label = self.create_stat_card(
            stats_frame, "Average probability", "Loading..."
        )

        self.highest_label = self.create_stat_card(
            stats_frame, "Highest probability", "Loading..."
        )
        self.no_incidents_label = tk.Label(
            home_page,
            text="No predicted incidents in the next hour",
            font=("Segoe UI", 13, "bold"),
            bg="#f4f4f4",
            fg="#16845b"
        )

        # Predictions table
        table_frame = RoundedPanel(
            home_page, "#f4f4f4", "#ffffff", "#dfe6ef"
        )
        table_frame.pack(
            fill="both",
            expand=True,
            padx=30,
            pady=20
        )
        table_content = table_frame.content

        columns = ("line", "stop", "hour", "probability")

        self.tree = ttk.Treeview(
            table_content,
            columns=columns,
            show="headings"
        )

        self.tree.heading(
            "line",
            text="Subway Line / Direction",
            command=lambda: self.sort_table("line")
        )

        self.tree.heading(
            "stop",
            text="Stop",
            command=lambda: self.sort_table("stop")
        )

        self.tree.heading(
            "hour",
            text="Prediction Hour",
            command=lambda: self.sort_table("hour")
        )

        self.tree.heading(
            "probability",
            text="Disruption Probability",
            command=lambda: self.sort_table("probability")
        )

        self.tree.column("line", anchor="w", width=560)
        self.tree.column("stop", anchor="w", width=280)
        self.tree.column("hour", anchor="center", width=140)
        self.tree.column("probability", anchor="center", width=160)

        scrollbar = ttk.Scrollbar(
            table_content,
            orient="vertical",
            command=self.tree.yview
        )

        self.tree.configure(
            yscrollcommand=scrollbar.set
        )
        self.tree.tag_configure("HIGH", foreground="#dc2626")
        self.tree.tag_configure("MODERATE", foreground="#b45309")
        self.tree.tag_configure("LOW", foreground="#15803d")

        self.tree.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        scrollbar.pack(side="right", fill="y", padx=(0, 8), pady=8)

        all_page = tk.Frame(self.pages, bg="#f4f4f4")
        self.pages.add(all_page, text="Explore all predictions")

        filters = RoundedPanel(
            all_page, "#f4f4f4", "#ffffff", "#dfe6ef", radius=14, inset=8,
            height=94
        )
        filters.pack(fill="x", padx=30, pady=15)
        filters.pack_propagate(False)
        filter_content = filters.content
        self.line_filter = tk.StringVar(value="All lines")
        self.direction_filter = tk.StringVar(value="All directions")
        self.stop_filter = tk.StringVar(value="All stops")
        self.hour_filter = tk.StringVar(value=self.format_hour(self.current_next_hour))
        self.line_combo = self._add_filter(
            filter_content, "Subway line", self.line_filter, ["All lines"]
        )
        self.direction_combo = self._add_filter(
            filter_content, "Direction", self.direction_filter, ["All directions"]
        )
        self.stop_combo = self._add_filter(
            filter_content, "Stop", self.stop_filter, ["All stops"]
        )
        self.hour_combo = self._add_filter(
            filter_content,
            "Prediction time",
            self.hour_filter,
            ["All hours"] + [self.format_hour(hour) for hour in range(24)]
        )
        self.line_combo.bind("<<ComboboxSelected>>", self.line_filter_changed)
        self.direction_combo.bind("<<ComboboxSelected>>", self.direction_filter_changed)
        self.stop_combo.bind("<<ComboboxSelected>>", self.filter_changed)
        self.hour_combo.bind("<<ComboboxSelected>>", self.hour_filter_changed)

        all_table_frame = RoundedPanel(
            all_page, "#f4f4f4", "#ffffff", "#dfe6ef"
        )
        all_table_frame.pack(fill="both", expand=True, padx=30, pady=(0, 20))
        all_table_content = all_table_frame.content
        self.all_tree = ttk.Treeview(
            all_table_content,
            columns=("line", "stop", "hour", "probability"),
            show="headings"
        )
        for column, heading, width in (
            ("line", "Subway Line / Direction", 420),
            ("stop", "Stop", 310),
            ("hour", "Prediction Hour", 160),
            ("probability", "Disruption Probability", 190)
        ):
            self.all_tree.heading(
                column,
                text=heading,
                command=lambda selected=column: self.sort_table(
                    selected, self.all_tree
                )
            )
            self.all_tree.column(column, anchor="w" if column == "line" else "center", width=width)
        all_scrollbar = ttk.Scrollbar(
            all_table_content, orient="vertical", command=self.all_tree.yview
        )
        self.all_tree.configure(yscrollcommand=all_scrollbar.set)
        self.all_tree.tag_configure("HIGH", foreground="#dc2626")
        self.all_tree.tag_configure("MODERATE", foreground="#b45309")
        self.all_tree.tag_configure("LOW", foreground="#15803d")
        self.all_tree.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        all_scrollbar.pack(side="right", fill="y", padx=(0, 8), pady=8)
        page_controls = tk.Frame(all_page, bg="#f4f4f4")
        page_controls.pack(fill="x", padx=30, pady=(0, 12))
        self.page_label = tk.Label(
            page_controls,
            text="Page 0 of 0",
            font=("Segoe UI", 9),
            bg="#f4f4f4",
            fg="gray"
        )
        page_controls.grid_columnconfigure(0, weight=1)
        page_controls.grid_columnconfigure(1, weight=1)
        page_controls.grid_columnconfigure(2, weight=1)
        self.page_label.grid(row=0, column=0, sticky="w")
        go_controls = tk.Frame(page_controls, bg="#f4f4f4")
        go_controls.grid(row=0, column=1)
        tk.Label(
            go_controls,
            text="Go to page:",
            font=("Segoe UI", 9),
            bg="#f4f4f4",
            fg="gray"
        ).pack(side="left", padx=(0, 5))
        self.page_number = tk.StringVar(value="1")
        self.page_entry = ttk.Entry(
            go_controls,
            textvariable=self.page_number,
            width=7
        )
        self.page_entry.pack(side="left")
        self.page_entry.bind("<Return>", self.go_to_page)
        self.page_entry.bind("<FocusIn>", lambda _event: self.page_entry.selection_clear())
        ttk.Button(
            go_controls,
            text="Go",
            command=self.go_to_page,
            style="Action.TButton"
        ).pack(side="left", padx=(5, 0))
        navigation_controls = tk.Frame(page_controls, bg="#f4f4f4")
        navigation_controls.grid(row=0, column=2, sticky="e")
        self.previous_page_button = ttk.Button(
            navigation_controls,
            text="Previous",
            command=self.previous_prediction_page,
            style="Action.TButton"
        )
        self.previous_page_button.pack(side="left", padx=(0, 8))
        self.next_page_button = ttk.Button(
            navigation_controls,
            text="Next",
            command=self.next_prediction_page,
            style="Action.TButton"
        )
        self.next_page_button.pack(side="left")
        self.tree.configure(style="Modern.Treeview")
        self.all_tree.configure(style="Modern.Treeview")

        # Status message
        self.status_label = tk.Label(
            self.root,
            text="Loading predictions...",
            font=("Segoe UI", 10),
            bg="#f4f4f4",
            fg="gray"
        )
        self.status_label.pack(pady=(0, 15))

        # Refresh button
        refresh_button = ttk.Button(
            self.root,
            text="Refresh Predictions",
            command=self.load_predictions,
            style="Action.TButton"
        )
        refresh_button.pack(pady=(0, 20))

    @staticmethod
    def format_hour(hour):
        return datetime.strptime(f"{hour:02d}", "%H").strftime("%I:00 %p")

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.apply_theme()

    def apply_theme(self):
        colors = self.themes["dark" if self.dark_mode else "light"]
        self.root.configure(bg=colors["background"])
        self.theme_button.configure(
            text="Light mode" if self.dark_mode else "Dark mode"
        )

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "TNotebook",
            background=colors["background"],
            borderwidth=0,
            tabmargins=(0, 0, 0, 0)
        )
        style.configure(
            "TNotebook.Tab",
            background=colors["heading"],
            foreground=colors["muted"],
            padding=(16, 9),
            font=("Segoe UI", 10, "bold"),
            borderwidth=0
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", colors["surface"])],
            foreground=[("selected", colors["accent"])],
            padding=[("selected", (18, 10))],
            font=[("selected", ("Segoe UI", 10, "bold"))]
        )
        style.configure(
            "Modern.Treeview",
            background=colors["surface"],
            fieldbackground=colors["surface"],
            foreground=colors["text"],
            rowheight=42,
            borderwidth=0,
            font=("Segoe UI", 10)
        )
        self.tree.tag_configure(
            "HIGH", foreground="#f87171" if self.dark_mode else "#dc2626"
        )
        self.tree.tag_configure(
            "MODERATE", foreground="#fbbf24" if self.dark_mode else "#b45309"
        )
        self.tree.tag_configure(
            "LOW", foreground="#4ade80" if self.dark_mode else "#15803d"
        )
        self.all_tree.tag_configure(
            "HIGH", foreground="#f87171" if self.dark_mode else "#dc2626"
        )
        self.all_tree.tag_configure(
            "MODERATE", foreground="#fbbf24" if self.dark_mode else "#b45309"
        )
        self.all_tree.tag_configure(
            "LOW", foreground="#4ade80" if self.dark_mode else "#15803d"
        )
        style.configure(
            "Modern.Treeview.Heading",
            background=colors["heading"],
            foreground=colors["muted"],
            font=("Segoe UI", 9, "bold"),
            padding=(12, 12),
            relief="flat"
        )
        style.map(
            "Modern.Treeview",
            background=[("selected", colors["selected"])],
            foreground=[("selected", colors["text"])]
        )
        style.configure(
            "TCombobox",
            fieldbackground=colors["input"],
            background=colors["input"],
            foreground=colors["text"],
            arrowcolor=colors["muted"],
            padding=8
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", colors["input"])],
            foreground=[("readonly", colors["text"])]
        )
        style.configure(
            "Action.TButton",
            background=colors["accent"],
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            padding=(16, 10),
            borderwidth=0
        )
        style.map(
            "Action.TButton",
            background=[("active", colors["accent_active"]), ("pressed", colors["accent_active"])],
            foreground=[("disabled", colors["muted"])]
        )
        style.configure(
            "TScrollbar",
            background=colors["heading"],
            troughcolor=colors["background"],
            bordercolor=colors["background"],
            arrowcolor=colors["muted"]
        )

        def recolor(widget):
            if isinstance(widget, RoundedPanel):
                widget.set_theme(
                    colors["background"], colors["surface"], colors["border"]
                )
            elif isinstance(widget, tk.Frame):
                if not hasattr(widget, "_theme_surface"):
                    widget._theme_surface = widget.cget("bg").lower() in ("white", "#ffffff")
                new_bg = colors["surface"] if widget._theme_surface else colors["background"]
                widget.configure(bg=new_bg)
                try:
                    widget.configure(highlightbackground=colors["border"])
                except tk.TclError:
                    pass
            elif isinstance(widget, tk.Label):
                if not hasattr(widget, "_theme_surface"):
                    widget._theme_surface = widget.cget("bg").lower() in ("white", "#ffffff")
                    widget._theme_muted = widget.cget("fg").lower() in ("gray", "grey")
                new_bg = colors["surface"] if widget._theme_surface else colors["background"]
                new_fg = colors["muted"] if widget._theme_muted else colors["text"]
                widget.configure(bg=new_bg, fg=new_fg)
            for child in widget.winfo_children():
                recolor(child)

        for child in self.root.winfo_children():
            recolor(child)

    def _add_filter(self, parent, label, variable, values):
        group = tk.Frame(parent, bg="#f4f4f4")
        group.pack(side="left", padx=(0, 20))
        tk.Label(group, text=label, bg="#f4f4f4").pack(anchor="w")
        combo = ttk.Combobox(
            group, textvariable=variable, values=values, state="readonly",
            width=28, takefocus=False, exportselection=False
        )
        combo.pack(anchor="w", pady=(4, 0))
        return combo

    def update_clock(self):
        now = datetime.now(ZoneInfo(TZ))
        self.clock_label.config(
            text=now.strftime("%A, %B %d, %Y  ·  %I:%M:%S %p")
        )
        if self.target_date is not None:
            self.root.after(1000, self.update_clock)
            return
        next_hour = (now.hour + 1) % 24
        if next_hour != self.current_next_hour:
            self.current_next_hour = next_hour
            self.render_next_hour()
            if self.follow_next_hour:
                self.hour_filter.set(self.format_hour(next_hour))
                self.display_all_predictions()
        self.root.after(1000, self.update_clock)

    def create_stat_card(self, parent, title, value):
        card = RoundedPanel(
            parent,
            "#f4f4f4",
            "#ffffff",
            "#dfe6ef",
            radius=16,
            inset=12
        )
        card.configure(height=118)
        card.pack_propagate(False)

        card.pack(
            side="left",
            expand=True,
            fill="both",
            padx=8
        )

        content = card.content
        tk.Label(
            content,
            text=title,
            font=("Segoe UI", 10, "bold"),
            bg="white",
            fg="gray"
        ).pack(pady=(5, 0))

        value_label = tk.Label(
            content,
            text=value,
            font=("Segoe UI", 22, "bold"),
            bg="white"
        )

        value_label.pack(pady=(5, 0))

        return value_label

    def load_predictions(self):
        self.status_label.config(
            text="Loading predictions..."
        )

        for item in self.tree.get_children():
            self.tree.delete(item)
        for item in self.all_tree.get_children():
            self.all_tree.delete(item)

        self.sort_reverse = {}

        thread = threading.Thread(
            target=self.fetch_predictions,
            daemon=True
        )

        thread.start()

    def fetch_predictions(self):
        try:
            predictions = get_predictions(self.target_date)

            self.prediction_queue.put(
                ("success", predictions)
            )

        except Exception as error:
            self.prediction_queue.put(
                ("error", str(error))
            )

    def check_queue(self):
        try:
            status, result = self.prediction_queue.get_nowait()

            if status == "success":
                self.display_predictions(result)

            else:
                self.status_label.config(
                    text=f"Error loading predictions: {result}"
                )

        except queue.Empty:
            pass

        self.root.after(100, self.check_queue)

    def display_predictions(self, predictions):
        if predictions is None or predictions.empty:
            self.status_label.config(
                text="No predictions available."
            )

            self.total_label.config(text="0")
            self.average_label.config(text="N/A")
            self.highest_label.config(text="N/A")
            self.predictions = pd.DataFrame(
                columns=["line", "direction_id", "stop_id", "stop_name", "hour", "probability"]
            )
            self.update_line_filter()
            self.no_incidents_label.pack(pady=(0, 10))
            self.populate_tree(self.tree, self.predictions)
            self.display_all_predictions()

            return

        required_columns = {
            "line",
            "direction_id",
            "stop_id",
            "stop_name",
            "hour",
            "probability",
            "data_day"
        }

        if not required_columns.issubset(predictions.columns):
            self.status_label.config(
                text="Error: Prediction DataFrame is missing required columns."
            )
            return

        predictions = predictions[predictions["line"].astype(str) != "SS"].copy()
        if predictions.empty:
            self.status_label.config(text="No predictions available.")
            self.total_label.config(text="0")
            self.average_label.config(text="N/A")
            self.highest_label.config(text="N/A")
            self.predictions = predictions
            self.update_line_filter()
            self.no_incidents_label.pack(pady=(0, 10))
            self.display_all_predictions()
            return

        self.predictions = predictions
        self.update_line_filter()
        self.follow_next_hour = self.target_date is None
        self.current_next_hour = (
            (datetime.now(ZoneInfo(TZ)).hour + 1) % 24
            if self.target_date is None
            else self.target_hour if self.target_hour is not None
            else 0
        )
        self.hour_filter.set(self.format_hour(self.current_next_hour))
        self.render_next_hour()
        self.display_all_predictions()

        self.status_label.config(
            text=(
                f"No stops with high predicted probability for {self.format_hour(self.current_next_hour)}."
                if not self.tree.get_children()
                else f"Loaded high-probability disruptions for {self.format_hour(self.current_next_hour)}."
            )
        )

    def populate_tree(self, tree, predictions):
        tree.delete(*tree.get_children())
        for _, row in predictions.iterrows():
            level = probability_band(row["probability"])
            stop_name = row["stop_name"] if pd.notna(row["stop_name"]) else row["stop_id"]
            tree.insert(
                "",
                "end",
                values=(
                    format_line_direction(row["line"], row["direction_id"]),
                    stop_name,
                    self.format_hour(int(row["hour"])),
                    f"{level} · {row['probability'] * 100:.1f}%"
                ),
                tags=(level,)
            )

    def update_line_filter(self):
        lines = sorted({
            "SIR" if str(line).upper() == "SI" else str(line).upper()
            for line in self.predictions.get("line", [])
        })
        current = self.line_filter.get()
        self.line_combo["values"] = ["All lines"] + lines
        if current not in self.line_combo["values"]:
            self.line_filter.set("All lines")
        self.update_direction_filter()

    def update_direction_filter(self):
        if self.predictions.empty:
            self.direction_combo["values"] = ["All directions"]
            self.direction_filter.set("All directions")
            self.update_stop_filter()
            return
        rows = self.predictions
        selected_line = self.line_filter.get()
        if selected_line != "All lines":
            rows = rows[rows["line"].map(
                lambda line: "SIR" if str(line).upper() == "SI" else str(line).upper()
            ) == selected_line]
        directions = sorted({
            format_line_direction(row.line, row.direction_id)
            for row in rows[["line", "direction_id"]].itertuples(index=False)
        })
        current = self.direction_filter.get()
        self.direction_combo["values"] = ["All directions"] + directions
        if current not in self.direction_combo["values"]:
            self.direction_filter.set("All directions")
        self.update_stop_filter()

    def update_stop_filter(self):
        if self.predictions.empty:
            self.stop_option_ids = {}
            self.stop_combo["values"] = ["All stops"]
            self.stop_filter.set("All stops")
            return
        rows = self.filtered_by_line_direction()
        stops = rows[["stop_id", "stop_name"]].drop_duplicates()
        labels = {}
        for row in stops.itertuples(index=False):
            name = row.stop_name or row.stop_id
            labels[f"{name} [{row.stop_id}]"] = str(row.stop_id)
        self.stop_option_ids = labels
        values = ["All stops"] + sorted(labels)
        self.stop_combo["values"] = values
        if self.stop_filter.get() not in values:
            self.stop_filter.set("All stops")

    def filtered_by_line_direction(self):
        rows = self.predictions
        selected_line = self.line_filter.get()
        if selected_line != "All lines":
            rows = rows[rows["line"].map(
                lambda line: "SIR" if str(line).upper() == "SI" else str(line).upper()
            ) == selected_line]
        selected_direction = self.direction_filter.get()
        if selected_direction != "All directions":
            rows = rows[rows.apply(
                lambda row: format_line_direction(row["line"], row["direction_id"])
                == selected_direction,
                axis=1
            )]
        return rows

    def line_filter_changed(self, _event=None):
        self.direction_filter.set("All directions")
        self.stop_filter.set("All stops")
        self.all_page_index = 0
        self.update_direction_filter()
        self.display_all_predictions()

    def direction_filter_changed(self, _event=None):
        self.stop_filter.set("All stops")
        self.all_page_index = 0
        self.update_stop_filter()
        self.display_all_predictions()

    def filter_changed(self, _event=None):
        self.all_page_index = 0
        self.display_all_predictions()

    def hour_filter_changed(self, _event=None):
        self.follow_next_hour = False
        self.all_page_index = 0
        self.display_all_predictions()

    def render_next_hour(self):
        if self.predictions.empty:
            high_probability_predictions = self.predictions
        else:
            next_predictions = self.predictions[
                self.predictions["hour"] == self.current_next_hour
            ]
            high_probability_predictions = next_predictions[
                next_predictions["probability"] >= HIGH_PROBABILITY_THRESHOLD
            ]
        total = len(high_probability_predictions)
        self.total_label.config(text=str(total))
        self.average_label.config(
            text=f"{high_probability_predictions['probability'].mean() * 100:.1f}%" if total else "N/A"
        )
        self.highest_label.config(
            text=f"{high_probability_predictions['probability'].max() * 100:.1f}%" if total else "N/A"
        )
        if total:
            self.no_incidents_label.pack_forget()
        else:
            self.no_incidents_label.pack(pady=(0, 10))
        self.populate_tree(self.tree, high_probability_predictions)

    def display_all_predictions(self):
        if not hasattr(self, "all_tree"):
            return
        rows = self.filtered_by_line_direction()
        selected_stop = self.stop_filter.get()
        if selected_stop != "All stops":
            rows = rows[rows["stop_id"].astype(str) == self.stop_option_ids[selected_stop]]
        selected_hour = self.hour_filter.get()
        if selected_hour != "All hours":
            hour = datetime.strptime(selected_hour, "%I:00 %p").hour
            rows = rows[rows["hour"] == hour]

        if self.all_sort_column == "line":
            rows = rows.sort_values(
                ["line", "direction_id"],
                ascending=not self.all_sort_reverse
            )
        elif self.all_sort_column == "stop":
            rows = rows.sort_values("stop_name", key=lambda values: values.fillna("").str.lower(),
                                    ascending=not self.all_sort_reverse)
        elif self.all_sort_column == "hour":
            rows = rows.sort_values("hour", ascending=not self.all_sort_reverse)
        elif self.all_sort_column == "probability":
            rows = rows.sort_values("probability", ascending=not self.all_sort_reverse)

        total_pages = (len(rows) + self.all_page_size - 1) // self.all_page_size
        self.all_total_pages = total_pages
        if total_pages == 0:
            self.all_page_index = 0
        else:
            self.all_page_index = min(self.all_page_index, total_pages - 1)
        start = self.all_page_index * self.all_page_size
        self.populate_tree(self.all_tree, rows.iloc[start:start + self.all_page_size])
        self.page_number.set(str(self.all_page_index + 1 if total_pages else 0))
        self.page_label.config(
            text=f"Page {self.all_page_index + 1} of {total_pages} · {len(rows):,} predictions"
            if total_pages else "No matching stops"
        )
        self.previous_page_button.configure(
            state="normal" if self.all_page_index > 0 else "disabled"
        )
        self.next_page_button.configure(
            state="normal" if self.all_page_index + 1 < total_pages else "disabled"
        )

    def go_to_page(self, _event=None):
        try:
            requested_page = int(self.page_number.get())
        except ValueError:
            self.status_label.config(text="Enter a whole page number.")
            return
        total_pages = self.all_total_pages
        if not 1 <= requested_page <= total_pages:
            self.status_label.config(
                text=f"Page must be between 1 and {total_pages}."
            )
            return
        self.all_page_index = requested_page - 1
        self.display_all_predictions()

    def previous_prediction_page(self):
        if self.all_page_index > 0:
            self.all_page_index -= 1
            self.display_all_predictions()

    def next_prediction_page(self):
        self.all_page_index += 1
        self.display_all_predictions()

    def sort_table(self, column, tree=None):
        tree = tree or self.tree
        if tree is self.all_tree:
            if self.all_sort_column == column:
                self.all_sort_reverse = not self.all_sort_reverse
            else:
                self.all_sort_column = column
                self.all_sort_reverse = False
            self.all_page_index = 0
            self.display_all_predictions()
            return

        items = []

        for item in tree.get_children(""):
            values = tree.item(item, "values")

            if column == "hour":
                sort_value = datetime.strptime(values[2], "%I:%M %p").hour

            elif column == "probability":
                sort_value = float(values[3].split()[-1].replace("%", ""))

            elif column == "stop":
                sort_value = values[1]

            else:
                sort_value = values[0]

            items.append((sort_value, item))

        reverse = self.sort_reverse.get(column, False)
        self.sort_reverse[column] = not reverse

        items.sort(
            key=lambda item: item[0],
            reverse=reverse
        )

        for index, (_, item) in enumerate(items):
            tree.move(item, "", index)


def parse_target_date(value):
    try:
        parsed_date = date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "target date must use YYYY-MM-DD format"
        ) from error
    if parsed_date.isoformat() != value:
        raise argparse.ArgumentTypeError(
            "target date must use YYYY-MM-DD format"
        )
    return parsed_date


def parse_target_hour(value):
    try:
        hour = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "target hour must be an integer from 0 to 23"
        ) from error
    if not 0 <= hour <= 23:
        raise argparse.ArgumentTypeError(
            "target hour must be an integer from 0 to 23"
        )
    return hour


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the PredictMTA dashboard.")
    parser.add_argument(
        "target_date",
        nargs="?",
        type=parse_target_date,
        help="prediction target date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "target_hour",
        nargs="?",
        type=parse_target_hour,
        help="target hour from 0 (12 AM) to 23 (11 PM); requires a target date"
    )
    args = parser.parse_args()
    if args.target_hour is not None and args.target_date is None:
        parser.error("target hour requires a target date")
    root = tk.Tk()
    app = MtaDashboard(root, args.target_date, args.target_hour)
    root.mainloop()