import tkinter as tk
from tkinter import ttk
from loguru import logger
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving images
import matplotlib.pyplot as plt
import seaborn as sns
import os

class MetricsDisplay:
    def __init__(self, metrics_data):
        self.metrics_data = metrics_data
        self.output_dir = ".output"
        os.makedirs(self.output_dir, exist_ok=True)
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

        logger.info(f"Metrics data: {self.metrics_data}")

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

        # --- Generate and save graphs after leaderboard is built ---
        self.save_all_graphs()

        self.tree.pack(fill=tk.BOTH, expand=True)
        self.window.mainloop()

    def apply_row_colors(self):
        for i, item in enumerate(self.tree.get_children("")):
            if self.tree.parent(item) == "":
                self.tree.item(item, tags=('odd' if i % 2 else 'even',))

    def save_all_graphs(self):
        df = self._metrics_to_dataframe()
        self.save_heatmap(df, "f1_score")
        self.save_heatmap(df, "adjusted_rand_index")
        self.save_bar_chart(df, "f1_score")
        self.save_bar_chart(df, "adjusted_rand_index")
        self.save_group_count_chart(df)
        # Add the new academic table graph
        self.save_metrics_histogram()

    def _metrics_to_dataframe(self):
        rows = []
        for model, lang_data in self.metrics_data.items():
            for lang, metrics in lang_data.items():
                row = {
                    'Model': model,
                    'Language': lang,
                    **metrics
                }
                rows.append(row)
        return pd.DataFrame(rows)

    def save_heatmap(self, df, metric):
        pivot = df.pivot(index="Model", columns="Language", values=metric)
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".2f")
        plt.title(f"{metric.replace('_', ' ').title()} Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{metric}_heatmap.png"))
        plt.close()

    def save_bar_chart(self, df, metric):
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Model", y=metric, hue="Language")
        plt.title(f"{metric.replace('_', ' ').title()} by Model and Language")
        plt.ylabel(metric.replace('_', ' ').title())
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{metric}_barplot.png"))
        plt.close()

    def save_group_count_chart(self, df):
        if "predicted_groups" in df.columns:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x="Model", y="predicted_groups", hue="Language")
            plt.title("Number of Predicted Groups by Model and Language")
            plt.ylabel("Number of Groups")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "group_count_barplot.png"))
            plt.close()

    def save_metrics_histogram(self):
        """
        Creates and saves histograms of key metrics across all models.
        This visualization is suitable for academic papers to show the distribution of metrics.
        """
        logger.info("Generating metrics histograms...")
        df = self._metrics_to_dataframe()

        # Key metrics to visualize
        metrics_to_analyze = [
            "adjusted_rand_index", "normalized_mutual_info", "adjusted_mutual_info",
            "v_measure", "pairwise_precision", "pairwise_recall", "f1_score"
        ]

        # Create a figure with subplots - 2 rows, 4 columns (7 metrics total)
        fig, axes = plt.subplots(2, 4, figsize=(16, 10))
        axes = axes.flatten()

        # Create histograms for each metric
        for i, metric in enumerate(metrics_to_analyze):
            if i < len(axes):  # Make sure we don't exceed the number of subplots
                ax = axes[i]

                # Create grouped histogram by model
                for model_name in df['Model'].unique():
                    model_data = df[df['Model'] == model_name]
                    ax.hist(model_data[metric], alpha=0.7, label=model_name, bins=10)

                # Customize the subplot
                ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12)
                ax.set_xlabel('Score', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.7)

                # Add mean lines for each model
                for model_name in df['Model'].unique():
                    model_data = df[df['Model'] == model_name]
                    mean_value = model_data[metric].mean()
                    ax.axvline(x=mean_value, linestyle='--',
                               label=f'{model_name} mean: {mean_value:.3f}')

        # Hide the unused subplot if we have an odd number of metrics
        if len(metrics_to_analyze) < len(axes):
            axes[-1].axis('off')

        # Add overall title
        fig.suptitle('Distribution of Evaluation Metrics Across Models', fontsize=16, fontweight='bold')

        # Add a single legend for all subplots at the bottom
        handles, labels = [], []
        for ax in axes:
            if ax.get_legend_handles_labels()[0]:  # Check if the axis has legend entries
                h, l = ax.get_legend_handles_labels()
                handles.extend(h)
                labels.extend(l)

        # Remove duplicate entries in the legend
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(),
                   loc='lower center', ncol=min(5, len(by_label)),
                   bbox_to_anchor=(0.5, 0.02), fontsize=10)

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, top=0.9)  # Make room for the legend and title

        # Save the figure
        histogram_path = os.path.join(self.output_dir, "metrics_histogram.png")
        plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Metrics histograms saved to {histogram_path}")

        # Create an additional figure showing side-by-side comparison of F1 scores
        plt.figure(figsize=(12, 8))

        # Create a grouped bar chart for F1 scores by model and language
        ax = sns.barplot(data=df, x='Model', y='f1_score', hue='Language')

        # Customize the plot
        plt.title('F1 Score Comparison by Model and Language', fontsize=14, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        # Add a horizontal line for the overall average F1 score
        avg_f1 = df['f1_score'].mean()
        plt.axhline(y=avg_f1, color='red', linestyle='--',
                    label=f'Overall Average: {avg_f1:.3f}')

        # Enhance the legend
        plt.legend(title='Language', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        # Save the figure
        f1_path = os.path.join(self.output_dir, "f1_score_comparison.png")
        plt.savefig(f1_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"F1 score comparison saved to {f1_path}")