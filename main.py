import pandas as pd
from typing import List, Dict
import os
import time
from datetime import datetime
from tqdm import tqdm
from evaluator import RAGRetrievalEvaluator
from plots import create_heatmap_visualizations, create_execution_time_plot


PARAMETER_GRID = {
    'chunk_sizes': [25, 50, 100, 200, 300, 400, 800],
    'num_retrived_chunks': [1, 3, 5, 7, 10],
    'overlap_sizes': [0]
}


def load_corpus(corpus_path: str = "data/chatlogs.md") -> List[Dict]:
    with open(corpus_path, "r") as f:
        data = f.read()

    if "chatlogs" in corpus_path:
        # The chatlogs corpus doesn't have ending brackets
        data += ".'}]"

    docs = data.split("[")[1:]
    docs = ['[' + doc for doc in docs]
    docs = [eval(doc) for doc in docs]
    return docs


def run_hyperparameter_grid_search(search_grid: dict[str, list[int]]):
    chunk_sizes = search_grid["chunk_sizes"]
    retrieval_counts = search_grid["num_retrived_chunks"]
    overlap_size = search_grid["overlap_sizes"][0]

    # load data
    corpus = load_corpus()
    df_queries = pd.read_csv("data/questions_df.csv")
    df_queries = df_queries[df_queries['corpus_id'] == 'chatlogs']
    queries = df_queries["question"].tolist()
    ground_truth = df_queries["references"].tolist()
    ground_truth = [eval(gt) for gt in ground_truth]

    timestamp = datetime.now().strftime("%y%m%d_%h%m%s")
    results_dir = f"./results/results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    results = []

    total_combinations = len(chunk_sizes) * len(retrieval_counts)
    print(f"running token-wise grid search with {total_combinations} parameter combinations...")
    pbar = tqdm(total=total_combinations)

    for chunk_size in chunk_sizes:
        evaluator = RAGRetrievalEvaluator(chunk_size=chunk_size, overlap_size=overlap_size)

        evaluator.build_vector_base(corpus)

        for num_chunks in retrieval_counts:
            start_time = time.time()

            metrics = evaluator.evaluate_retrieval(
                queries=queries,
                ground_truth=ground_truth,
                num_retrieved_chunks=num_chunks
            )

            end_time = time.time()
            execution_time = end_time - start_time

            result_row = {
                "chunk_size": chunk_size,
                "num_retrieved_chunks": num_chunks,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "iou_score": metrics["iou_score"],
                "execution_time": execution_time,
                "num_chunks": len(evaluator.chunks)
            }
            results.append(result_row)

            pbar.update(1)
            pbar.set_description(f"testing chunk_size={chunk_size}, num_chunks={num_chunks}")

    pbar.close()

    # save results to csv
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(results_dir, f"hyperparameter_results_token_wise.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"token-wise evaluation results saved to {csv_path}")

    # save visualizations
    create_heatmap_visualizations(results_df, results_dir)
    create_execution_time_plot(results_df, results_dir)

    return results_df


def main():
    print("\nrunning token-wise hyperparameter grid search...")
    results_df = run_hyperparameter_grid_search(search_grid=PARAMETER_GRID)
    print("\ntop 5 parameter combinations by token-wise iou score:")
    top_5 = results_df.sort_values(by='iou_score', ascending=False).head(5)
    print(top_5)


if __name__ == "__main__":
    main()
