"""Visualization functions for chunking strategy evaluation results."""

import os

import matplotlib.pyplot as plt


def create_heatmap_visualizations(results_df, results_dir):
    """Create heatmap visualizations for the grid search results."""
    metrics = ["precision", "recall", "iou_score"]

    for metric in metrics:
        plt.figure(figsize=(10, 8))
        pivot_data = results_df.pivot(
            index="chunk_size", columns="num_retrieved_chunks", values=metric
        )
        heatmap = plt.imshow(pivot_data, cmap="viridis")
        plt.colorbar(heatmap, label=f"{metric.capitalize()} (token-wise)")
        plt.xlabel("number of retrieved chunks")
        plt.ylabel("chunk size")
        plt.title(f"effect of parameters on {metric.capitalize()}")
        plt.xticks(range(len(pivot_data.columns)), pivot_data.columns)
        plt.yticks(range(len(pivot_data.index)), pivot_data.index)

        # add numbers in heatmap cells
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                plt.text(
                    j,
                    i,
                    f"{pivot_data.iloc[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="white",
                )

        plt.tight_layout()
        plt.savefig(
            os.path.join(results_dir, f"heatmap_token_wise_{metric}.png")
        )
        plt.close()


def create_execution_time_plot(results_df, results_dir):
    """Create execution time plots showing performance across parameters."""
    plt.figure(figsize=(12, 6))

    grouped_time = (
        results_df.groupby(["chunk_size", "num_retrieved_chunks"])[
            "execution_time"
        ]
        .mean()
        .reset_index()
    )

    for chunk_size in results_df["chunk_size"].unique():
        subset = grouped_time[grouped_time["chunk_size"] == chunk_size]
        plt.plot(
            subset["num_retrieved_chunks"],
            subset["execution_time"],
            marker="o",
            label=f"chunk size {chunk_size}",
        )

    plt.xlabel("number of retrieved chunks")
    plt.ylabel("execution time (seconds)")
    plt.title("execution time vs. parameters (token-wise evaluation)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "execution_time_token_wise.png"))
    plt.close()
