"""Main entry point for chunking strategy benchmarking."""

import time

import pandas as pd
from tqdm import tqdm

from config import DEFAULT_CONFIG, EvaluationConfig
from evaluator import RAGRetrievalEvaluator
from plots import create_execution_time_plot, create_heatmap_visualizations
from utils import (
    create_timestamped_results_dir,
    load_corpus,
    load_queries_and_ground_truth,
    save_results_to_csv,
)


def run_hyperparameter_grid_search(
    config: EvaluationConfig = None,
) -> pd.DataFrame:
    """
    Run hyperparameter grid search for chunking strategies.

    Args:
        config: Evaluation configuration object

    Returns:
        DataFrame with evaluation results
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Load data
    corpus = load_corpus(config.corpus_path)
    queries, ground_truth = load_queries_and_ground_truth(
        config.questions_path, config.corpus_id
    )

    # Create results directory
    results_dir = create_timestamped_results_dir(config.results_base_dir)

    results = []
    total_combinations = (
        len(config.chunk_sizes)
        * len(config.num_retrieved_chunks)
        * len(config.overlap_sizes)
    )

    print(
        f"Running token-wise grid search with {total_combinations} "
        f"parameter combinations..."
    )
    pbar = tqdm(total=total_combinations)

    for overlap_size in config.overlap_sizes:
        for chunk_size in config.chunk_sizes:
            evaluator = RAGRetrievalEvaluator(
                chunk_size=chunk_size,
                overlap_size=overlap_size,
                embedding_model=config.embedding_model,
            )

            evaluator.build_vector_base(corpus)

            for num_chunks in config.num_retrieved_chunks:
                start_time = time.time()

                metrics = evaluator.evaluate_retrieval(
                    queries=queries,
                    ground_truth=ground_truth,
                    num_retrieved_chunks=num_chunks,
                )

                end_time = time.time()
                execution_time = end_time - start_time

                result_row = {
                    "chunk_size": chunk_size,
                    "overlap_size": overlap_size,
                    "num_retrieved_chunks": num_chunks,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "iou_score": metrics["iou_score"],
                    "execution_time": execution_time,
                    "num_chunks": len(evaluator.chunks),
                }
                results.append(result_row)

                pbar.update(1)
                pbar.set_description(
                    f"Testing chunk_size={chunk_size}, "
                    f"overlap={overlap_size}, "
                    f"num_chunks={num_chunks}, "
                )

    pbar.close()

    # Save results to CSV
    results_df = pd.DataFrame(results)
    csv_path = save_results_to_csv(results, results_dir)
    print(f"Token-wise evaluation results saved to {csv_path}")

    # Create visualizations
    create_heatmap_visualizations(results_df, results_dir)
    create_execution_time_plot(results_df, results_dir)

    return results_df


def main():
    """Main execution function."""
    print("\nRunning token-wise hyperparameter grid search...")
    results_df = run_hyperparameter_grid_search()

    print("\nTop 5 parameter combinations by token-wise IoU score:")
    top_5 = results_df.sort_values(by="iou_score", ascending=False).head(5)
    print(
        top_5[
            [
                "chunk_size",
                "overlap_size",
                "num_retrieved_chunks",
                "precision",
                "recall",
                "iou_score",
            ]
        ]
    )


if __name__ == "__main__":
    main()
