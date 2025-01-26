import tkinter as tk
from tkinter import ttk
from tabulate import tabulate

class MetricsDisplay:
    def __init__(self, overall_metrics_list):
        self.overall_metrics_list = overall_metrics_list
        self.window = tk.Tk()
        self.window.title("Overall Metrics")

    def display_metrics(self):
        tree = ttk.Treeview(self.window)

        tree["columns"] = ("model_name", "adjusted_rand_index", "normalized_mutual_info", "adjusted_mutual_info", "v_measure", "pairwise_precision", "pairwise_recall", "f1_score")

        tree.column("#0", width=0, stretch=tk.NO)
        tree.column("model_name", anchor=tk.W, width=100)
        tree.column("adjusted_rand_index", anchor=tk.W, width=150)
        tree.column("normalized_mutual_info", anchor=tk.W, width=150)
        tree.column("adjusted_mutual_info", anchor=tk.W, width=150)
        tree.column("v_measure", anchor=tk.W, width=100)
        tree.column("pairwise_precision", anchor=tk.W, width=150)
        tree.column("pairwise_recall", anchor=tk.W, width=150)
        tree.column("f1_score", anchor=tk.W, width=100)

        tree.heading("#0", text="", anchor=tk.W)
        tree.heading("model_name", text="Model", anchor=tk.W)
        tree.heading("adjusted_rand_index", text="Adjusted Rand Index", anchor=tk.W)
        tree.heading("normalized_mutual_info", text="Normalized Mutual Info", anchor=tk.W)
        tree.heading("adjusted_mutual_info", text="Adjusted Mutual Info", anchor=tk.W)
        tree.heading("v_measure", text="V Measure", anchor=tk.W)
        tree.heading("pairwise_precision", text="Pairwise Precision", anchor=tk.W)
        tree.heading("pairwise_recall", text="Pairwise Recall", anchor=tk.W)
        tree.heading("f1_score", text="F1 Score", anchor=tk.W)

        for metric in self.overall_metrics_list:
            tree.insert("", tk.END, values=(metric["model_name"], metric["adjusted_rand_index"], metric["normalized_mutual_info"], metric["adjusted_mutual_info"], metric["v_measure"], metric["pairwise_precision"], metric["pairwise_recall"], metric["f1_score"]))

        tree.pack()

        highest_f1_score = max(metric["f1_score"] for metric in self.overall_metrics_list)

        for item in tree.get_children():
            values = tree.item(item, "values")
            if float(values[7]) == highest_f1_score:
                tree.item(item, tags=("highlight",))
                tree.tag_configure("highlight", background="green")

        self.window.mainloop()