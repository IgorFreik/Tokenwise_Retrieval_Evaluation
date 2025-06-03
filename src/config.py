"""Configuration settings for the chunking strategy benchmarking."""

from dataclasses import dataclass
from typing import List


@dataclass
class EvaluationConfig:
    """Configuration for chunking strategy evaluation."""

    # Chunking parameters
    chunk_sizes: List[int] = None
    num_retrieved_chunks: List[int] = None
    overlap_sizes: List[int] = None

    # Model parameters
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Data paths
    corpus_path: str = "data/chatlogs.md"
    questions_path: str = "data/questions_df.csv"
    results_base_dir: str = "./results"

    # Evaluation parameters
    corpus_id: str = "chatlogs"

    def __post_init__(self):
        """Set default values after initialization."""
        if self.chunk_sizes is None:
            self.chunk_sizes = [25, 50, 100, 200, 300, 400, 800]

        if self.num_retrieved_chunks is None:
            self.num_retrieved_chunks = [1, 3, 5, 7, 10]

        if self.overlap_sizes is None:
            self.overlap_sizes = [0]


# Default configuration instance
DEFAULT_CONFIG = EvaluationConfig()
