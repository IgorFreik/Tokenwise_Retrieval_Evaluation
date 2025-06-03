"""RAG Retrieval Evaluator for benchmarking chunking strategies."""

import re
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
from chunking_evaluation.chunking import FixedTokenChunker
from transformers import AutoModel, AutoTokenizer


class RAGRetrievalEvaluator:
    """
    Evaluates RAG retrieval performance across different chunking
    strategies.
    """

    def __init__(
        self,
        chunk_size: int = 200,
        overlap_size: int = 0,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize the RAG Retrieval Evaluator.

        Args:
            chunk_size (int): Number of tokens per chunk
            overlap_size (int): Overlap size between chunks
            embedding_model (str): Hugging Face model for embeddings
        """
        # Load tokenizer and embedding model
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.embedding_model = AutoModel.from_pretrained(embedding_model)
        self.chunker = None
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

        # Store processed corpus information
        self.chunks = None  # List of chunks
        self.chunk_embeddings = None  # Embeddings for chunks
        self.original_corpus = None  # Original corpus text
        self.processed_docs = None  # List of extracted content from corpus
        self.chunk_to_doc_map = None  # Maps chunks to document and position
        # Maps characters to token indices for each document
        self.doc_token_maps = None

    def process_corpus(self, corpus: List[List[Dict[str, Any]]]) -> List[str]:
        """
        Extract content from corpus documents.

        Args:
            corpus: List of documents in the format
                   [{'content': '...', 'role': '...'}]

        Returns:
            List of extracted content strings
        """
        docs = []
        for doc in corpus:
            if isinstance(doc, list):
                doc_content = ""
                for sub_doc in doc:
                    doc_content += sub_doc["content"]
                docs.append(doc_content)
            else:
                raise ValueError(
                    "Invalid document format. "
                    "Please provide a list of documents."
                )

        self.processed_docs = docs
        self.original_corpus = corpus
        return docs

    def build_vector_base(self, corpus: List[Any]):
        """
        Build chunks from corpus and generate embeddings.

        Args:
            corpus: List of documents (either dicts or strings)
        """
        # Process corpus to extract content
        processed_docs = self.process_corpus(corpus)
        self.chunker = FixedTokenChunker(
            chunk_size=self.chunk_size, chunk_overlap=self.overlap_size
        )

        all_chunks = []
        chunk_to_doc_map = []

        # Create token maps for each document
        self.doc_token_maps = []
        for doc in processed_docs:
            self.doc_token_maps.append(self._create_char_to_token_map(doc))

        # Track original document index and position within the document
        for doc_idx, doc in enumerate(processed_docs):
            chunks = self.chunker.split_text(doc)

            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_pos = doc.find(chunk)
                chunk_to_doc_map.append(
                    (doc_idx, chunk_pos, chunk_pos + len(chunk))
                )

        self.chunks = all_chunks
        self.chunk_to_doc_map = chunk_to_doc_map
        self.chunk_embeddings = self._generate_embeddings(all_chunks)

        return all_chunks

    def _create_char_to_token_map(self, text: str) -> Dict[int, int]:
        """
        Create a mapping from character positions to token indices.

        Args:
            text (str): Text to tokenize

        Returns:
            Dict mapping character positions to token indices
        """
        tokenized = self.tokenizer(
            text, return_offsets_mapping=True, add_special_tokens=False
        )
        offsets = tokenized.offset_mapping

        # Create mapping from character positions to token indices
        char_to_token = {}
        for token_idx, (start, end) in enumerate(offsets):
            for char_pos in range(start, end):
                char_to_token[char_pos] = token_idx

        return char_to_token

    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts (List[str]): List of text chunks

        Returns:
            numpy array of embeddings
        """
        # Tokenize texts
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # Generate embeddings
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()

        return embeddings

    def retrieve_chunks(
        self, query: str, num_retrieved_chunks: int
    ) -> List[Tuple[str, Tuple[int, int, int]]]:
        """
        Retrieve most relevant chunks for a given query.

        Args:
            query (str): Query to retrieve chunks for
            num_retrieved_chunks (int): Number of chunks to retrieve

        Returns:
            List of retrieved chunks with their doc index and positions
        """
        query_embedding = self._generate_embeddings([query])

        # Normalize embeddings for cosine similarity
        query_norm = query_embedding / np.linalg.norm(
            query_embedding, axis=1, keepdims=True
        )
        chunk_norms = self.chunk_embeddings / np.linalg.norm(
            self.chunk_embeddings, axis=1, keepdims=True
        )

        # Compute cosine similarity
        similarities = np.dot(query_norm, chunk_norms.T).flatten()

        # Get indices of top k chunks
        top_k_indices = np.argsort(similarities)[-num_retrieved_chunks:][::-1]
        return [
            (self.chunks[idx], self.chunk_to_doc_map[idx])
            for idx in top_k_indices
        ]

    def _remap_ground_truth(
        self, ground_truth: List[List[Dict]]
    ) -> List[List[Tuple[int, int, int]]]:
        """
        Remap ground truth indices to document indices and positions.

        Args:
            ground_truth: List of dictionaries with start_index, end_index,
                         and content

        Returns:
            List of lists of tuples (doc_idx, start_pos, end_pos)
        """
        remapped = []
        for gt_item in ground_truth:
            contents = []
            remapped_question = []

            for reference in gt_item:
                contents.append(reference["content"])

            for content in contents:
                escaped_content = re.escape(content)

                # Look for this content in each document
                for doc_idx, doc in enumerate(self.processed_docs):
                    match = re.search(escaped_content, doc)
                    if match:
                        start_pos = match.start()
                        end_pos = match.end()
                        remapped_question.append((doc_idx, start_pos, end_pos))
                        break
            remapped.append(remapped_question)

        return remapped

    def _chars_to_token_set(
        self, doc_idx: int, start_pos: int, end_pos: int
    ) -> Set[int]:
        """
        Convert a character range to a set of token indices.

        Args:
            doc_idx (int): Document index
            start_pos (int): Starting character position
            end_pos (int): Ending character position

        Returns:
            Set of token indices
        """
        token_set = set()
        char_to_token = self.doc_token_maps[doc_idx]

        for char_pos in range(start_pos, min(end_pos, len(char_to_token))):
            if char_pos in char_to_token:
                token_set.add(char_to_token[char_pos])

        return token_set

    def evaluate_retrieval(
        self,
        queries: List[str],
        ground_truth: List[Any],
        corpus: List[Any] = None,
        num_retrieved_chunks: int = 5,
    ) -> Dict[str, float]:
        """
        Evaluate the retrieval performance using token-wise precision,
        recall, and IoU.

        Args:
            queries (List[str]): List of query strings
            ground_truth (List[Any]): List of ground truth ranges
                                      for each query
            corpus (List[Any], optional): The corpus to evaluate against.
                                    If None, use previously built vector base.
            num_retrieved_chunks (int): Number of chunks to retrieve

        Returns:
            Dict with precision, recall, and IoU metrics
        """
        # Build vector base if corpus is provided
        if corpus is not None:
            self.build_vector_base(corpus)
        elif self.chunks is None:
            raise ValueError(
                "Vector base not built. Call build_vector_base first."
            )

        # Parse ground truth if needed
        if isinstance(ground_truth[0], str):
            parsed_ground_truth = [eval(gt) for gt in ground_truth]
        else:
            parsed_ground_truth = ground_truth

        remapped_ground_truth = self._remap_ground_truth(parsed_ground_truth)

        total_precision = 0.0
        total_recall = 0.0
        total_iou = 0.0

        for query_idx, (query, gt_items) in enumerate(
            zip(queries, remapped_ground_truth)
        ):
            # Skip queries without ground truth
            if not gt_items:
                continue

            # Retrieve chunks
            retrieved_chunks = self.retrieve_chunks(
                query, num_retrieved_chunks=num_retrieved_chunks
            )

            # Convert character ranges to token sets for both retrieved
            # and ground truth
            retrieved_tokens = set()
            for chunk, (doc_idx, start_pos, end_pos) in retrieved_chunks:
                retrieved_tokens.update(
                    self._chars_to_token_set(doc_idx, start_pos, end_pos)
                )

            ground_truth_tokens = set()
            for doc_idx, start_pos, end_pos in gt_items:
                ground_truth_tokens.update(
                    self._chars_to_token_set(doc_idx, start_pos, end_pos)
                )

            # Calculate metrics
            intersection = retrieved_tokens.intersection(ground_truth_tokens)
            union = retrieved_tokens.union(ground_truth_tokens)

            precision = (
                len(intersection) / len(retrieved_tokens)
                if retrieved_tokens
                else 0.0
            )
            recall = (
                len(intersection) / len(ground_truth_tokens)
                if ground_truth_tokens
                else 0.0
            )
            iou = len(intersection) / len(union) if union else 0.0

            total_precision += precision
            total_recall += recall
            total_iou += iou

        # Calculate average metrics
        num_queries = len(queries)
        avg_precision = total_precision / num_queries
        avg_recall = total_recall / num_queries
        avg_iou = total_iou / num_queries

        return {
            "precision": avg_precision,
            "recall": avg_recall,
            "iou_score": avg_iou,
        }
