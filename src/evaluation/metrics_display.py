import tkinter as tk
from tkinter import ttk
import numpy as np


class MetricsDisplay:
    def __init__(self, metrics_data):
        self.metrics_data = metrics_data
        self.window = tk.Tk()
        self.window.title("Metrics Comparison Matrix")
        self.window.geometry("1200x600")
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.tree = None

    def calculate_overall_metrics(self, model_data):
        languages = model_data.values()
        overall_metrics = {
            metric: sum(lang[metric] for lang in languages) / len(languages)
            for metric in ["adjusted_rand_index", "normalized_mutual_info", "adjusted_mutual_info",
                           "v_measure", "pairwise_precision", "pairwise_recall", "f1_score"]
        }
        return overall_metrics

    def sort_column(self, col, reverse):
        if col == "rank":
            return

        data = [(self.tree.set(k, col), k) for k in self.tree.get_children("") if self.tree.parent(k) == ""]
        try:
            data.sort(key=lambda t: float(t[0]), reverse=reverse)
        except ValueError:
            data.sort(reverse=reverse)

        for index, (_, k) in enumerate(data):
            self.tree.move(k, "", index)

        self.recalculate_ranking()
        self.tree.heading(col, command=lambda: self.sort_column(col, not reverse))

    def recalculate_ranking(self):
        for rank, item in enumerate(self.tree.get_children(""), start=1):
            if self.tree.parent(item) == "":  # Only update top-level items
                self.tree.set(item, "rank", str(rank))

    def display_metrics(self):
        self.tree = ttk.Treeview(self.window, show="headings")
        self.tree["columns"] = ("rank", "model_name", "adjusted_rand_index", "nmi", "ami", "v_measure",
                                "precision", "recall", "f1_score")

        columns_config = [
            ("rank", "Rank"),
            ("model_name", "Model"),
            ("adjusted_rand_index", "Adjusted Rand Index"),
            ("nmi", "NMI"),
            ("ami", "AMI"),
            ("v_measure", "V-Measure"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1_score", "F1-Score")
        ]

        for col_id, col_name in columns_config:
            self.tree.column(col_id, anchor=tk.CENTER, width=120)
            if col_id == "rank":
                self.tree.heading(col_id, text=col_name, anchor=tk.CENTER)
            else:
                self.tree.heading(
                    col_id,
                    text=col_name,
                    anchor=tk.CENTER,
                    command=lambda c=col_id: self.sort_column(c, False)
                )

        overall_metrics = []
        for model_name, model_data in self.metrics_data.items():
            metrics = self.calculate_overall_metrics(model_data)
            overall_metrics.append((model_name, metrics))

        overall_metrics.sort(key=lambda x: x[1]["f1_score"], reverse=True)

        for rank, (model_name, model_metrics) in enumerate(overall_metrics, start=1):
            values = [
                f'{rank}',
                model_name,
                f'{model_metrics["adjusted_rand_index"]:.3f}',
                f'{model_metrics["normalized_mutual_info"]:.3f}',
                f'{model_metrics["adjusted_mutual_info"]:.3f}',
                f'{model_metrics["v_measure"]:.3f}',
                f'{model_metrics["pairwise_precision"]:.3f}',
                f'{model_metrics["pairwise_recall"]:.3f}',
                f'{model_metrics["f1_score"]:.3f}'
            ]
            item = self.tree.insert("", tk.END, values=values)

            for language, lang_data in self.metrics_data[model_name].items():
                lang_values = [
                    "-",
                    f'{language}',
                    f'{lang_data["adjusted_rand_index"]:.3f}',
                    f'{lang_data["normalized_mutual_info"]:.3f}',
                    f'{lang_data["adjusted_mutual_info"]:.3f}',
                    f'{lang_data["v_measure"]:.3f}',
                    f'{lang_data["pairwise_precision"]:.3f}',
                    f'{lang_data["pairwise_recall"]:.3f}',
                    f'{lang_data["f1_score"]:.3f}'
                ]
                self.tree.insert(item, tk.END, values=lang_values)

        self.tree.tag_configure('odd', background='#f0f0f0')
        self.tree.tag_configure('even', background='#ffffff')
        self.apply_row_colors()

        self.tree.pack(fill=tk.BOTH, expand=True)
        self.window.mainloop()

    def apply_row_colors(self):
        for i, item in enumerate(self.tree.get_children("")):
            if self.tree.parent(item) == "":
                self.tree.item(item, tags=('odd' if i % 2 else 'even',))