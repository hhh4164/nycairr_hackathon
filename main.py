
import tkinter as tk
from tkinter import ttk
import threading
import queue
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from predict import get_predictions


class MtaDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("PredictMTA - Subway Disruption Dashboard")
        self.root.geometry("900x650")
        self.root.configure(bg="#f4f4f4")

        self.prediction_queue = queue.Queue()
        self.sort_reverse = {}

        self.create_widgets()
        self.load_predictions()
        self.check_queue()

    def create_widgets(self):
        title = tk.Label(
            self.root,
            text="NYC Subway Disruption Predictions",
            font=("Arial", 22, "bold"),
            bg="#f4f4f4"
        )
        title.pack(pady=(20, 5))

        self.prediction_date_label = tk.Label(
            self.root,
            text="Predicting for: Loading...",
            font=("Arial", 14, "bold"),
            bg="#f4f4f4"
        )
        self.prediction_date_label.pack(pady=(5, 15))

        # Summary statistics
        stats_frame = tk.Frame(self.root, bg="#f4f4f4")
        stats_frame.pack(fill="x", padx=30, pady=10)

        self.total_label = self.create_stat_card(
            stats_frame, "Total Predictions", "Loading..."
        )
        self.average_label = self.create_stat_card(
            stats_frame, "Average Risk", "Loading..."
        )
        self.highest_label = self.create_stat_card(
            stats_frame, "Highest Risk", "Loading..."
        )

        # Predictions table
        table_frame = tk.Frame(self.root, bg="#f4f4f4")
        table_frame.pack(fill="both", expand=True, padx=30, pady=20)

        columns = ("line", "hour", "risk")

        self.tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings"
        )

        self.tree.heading(
            "line",
            text="Subway Line",
            command=lambda: self.sort_table("line")
        )
        self.tree.heading(
            "hour",
            text="Prediction Hour",
            command=lambda: self.sort_table("hour")
        )
        self.tree.heading(
            "risk",
            text="Disruption Risk",
            command=lambda: self.sort_table("risk")
        )

        self.tree.column("line", anchor="center", width=200)
        self.tree.column("hour", anchor="center", width=250)
        self.tree.column("risk", anchor="center", width=250)

        scrollbar = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.tree.yview
        )
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Status message
        self.status_label = tk.Label(
            self.root,
            text="Loading predictions...",
            font=("Arial", 10),
            bg="#f4f4f4",
            fg="gray"
        )
        self.status_label.pack(pady=(0, 15))

        # Refresh button
        refresh_button = tk.Button(
            self.root,
            text="Refresh Predictions",
            command=self.load_predictions,
            font=("Arial", 11),
            padx=15,
            pady=5
        )
        refresh_button.pack(pady=(0, 20))

    def create_stat_card(self, parent, title, value):
        card = tk.Frame(
            parent,
            bg="white",
            bd=1,
            relief="solid",
            padx=20,
            pady=15
        )
        card.pack(side="left", expand=True, fill="both", padx=8)

        tk.Label(
            card,
            text=title,
            font=("Arial", 11),
            bg="white",
            fg="gray"
        ).pack()

        value_label = tk.Label(
            card,
            text=value,
            font=("Arial", 20, "bold"),
            bg="white"
        )
        value_label.pack(pady=(5, 0))

        return value_label

    def load_predictions(self):
        self.status_label.config(text="Loading predictions...")

        for item in self.tree.get_children():
            self.tree.delete(item)

        thread = threading.Thread(
            target=self.fetch_predictions,
            daemon=True
        )
        thread.start()

    def fetch_predictions(self):
        try:
            predictions = get_predictions()
            self.prediction_queue.put(("success", predictions))
        except Exception as error:
            self.prediction_queue.put(("error", str(error)))

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
            self.status_label.config(text="No predictions available.")
            self.prediction_date_label.config(text="Predicting for: N/A")
            self.total_label.config(text="0")
            self.average_label.config(text="N/A")
            self.highest_label.config(text="N/A")
            return

        required_columns = {"line", "hour", "risk", "data_day"}

        if not required_columns.issubset(predictions.columns):
            self.status_label.config(
                text="Error: Prediction DataFrame is missing required columns."
            )
            return

        # predict.py uses the previous day's data to predict the next day.
        data_day = pd.to_datetime(predictions["data_day"].iloc[0])
        prediction_date = data_day + timedelta(days=1)

        self.prediction_date_label.config(
            text=f"Predicting for: {prediction_date.strftime('%B %d, %Y')}"
        )

        # Summary statistics
        total = len(predictions)
        average_risk = predictions["risk"].mean() * 100
        highest_risk = predictions["risk"].max() * 100

        self.total_label.config(text=str(total))
        self.average_label.config(text=f"{average_risk:.1f}%")
        self.highest_label.config(text=f"{highest_risk:.1f}%")

        # Populate table
        for _, row in predictions.iterrows():
            hour = int(row["hour"])
            hour_label = f"{hour:02d}:00"

            risk_percentage = row["risk"] * 100

            self.tree.insert(
                "",
                "end",
                values=(
                    row["line"],
                    hour_label,
                    f"{risk_percentage:.1f}%"
                )
            )

        self.status_label.config(
            text=f"Loaded {total} predictions successfully."
        )

    def sort_table(self, column):
        items = [
            (self.tree.set(item, column), item)
            for item in self.tree.get_children("")
        ]

        reverse = self.sort_reverse.get(column, False)
        self.sort_reverse[column] = not reverse

        if column == "hour":
            items.sort(
                key=lambda item: int(item[0].split(":")[0]),
                reverse=reverse
                )

        elif column == "risk":
            items.sort(
                key=lambda item: float(item[0].replace("%", "")),
                reverse=reverse
            )

        else:
            items.sort(
                key=lambda item: item[0],
                reverse=reverse
            )

        for index, (_, item) in enumerate(items):
            self.tree.move(item, "", index)


if __name__ == "__main__":
    root = tk.Tk()
    app = MtaDashboard(root)
    root.mainloop()
