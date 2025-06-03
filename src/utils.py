"""Utility functions for the chunking strategy benchmarking."""

import os
from datetime import datetime
from typing import Dict, List

import pandas as pd


def load_corpus(corpus_path: str = "data/chatlogs.md") -> List[Dict]:
    """
    Load corpus from file and parse it into document format.

    Args:
        corpus_path: Path to the corpus file

    Returns:
        List of parsed documents
    """
    with open(corpus_path, "r") as f:
        data = f.read()

    if "chatlogs" in corpus_path:
        # The chatlogs corpus doesn't have ending brackets
        data += ".'}]"

    docs = data.split("[")[1:]
    docs = ["[" + doc for doc in docs]
    docs = [eval(doc) for doc in docs]
    return docs


def load_queries_and_ground_truth(
    questions_path: str = "data/questions_df.csv", corpus_id: str = "chatlogs"
) -> tuple[List[str], List[List]]:
    """
    Load queries and ground truth from CSV file.

    Args:
        questions_path: Path to the questions CSV file
        corpus_id: Corpus ID to filter by

    Returns:
        Tuple of (queries, ground_truth)
    """
    df_queries = pd.read_csv(questions_path)
    df_queries = df_queries[df_queries["corpus_id"] == corpus_id]

    queries = df_queries["question"].tolist()
    ground_truth = df_queries["references"].tolist()
    ground_truth = [eval(gt) for gt in ground_truth]

    return queries, ground_truth


def create_timestamped_results_dir(base_dir: str = "./results") -> str:
    """
    Create a timestamped results directory.

    Args:
        base_dir: Base directory for results

    Returns:
        Path to the created timestamped directory
    """
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    results_dir = os.path.join(base_dir, f"results_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def save_results_to_csv(results: List[Dict], results_dir: str) -> str:
    """
    Save evaluation results to CSV file.

    Args:
        results: List of result dictionaries
        results_dir: Directory to save results in

    Returns:
        Path to the saved CSV file
    """
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(
        results_dir, "hyperparameter_results_token_wise.csv"
    )
    results_df.to_csv(csv_path, index=False)
    return csv_path
