import pytest
import numpy as np
from src.evaluator import RAGRetrievalEvaluator

@pytest.fixture
def sample_corpus():
    return [
        [{'content': 'This is the first document. It contains some text about machine learning.', 'role': 'user'}],
        [{'content': 'This is the second document. It discusses natural language processing.', 'role': 'user'}]
    ]

@pytest.fixture
def sample_queries():
    return [
        "What is machine learning?",
        "Tell me about NLP"
    ]

@pytest.fixture
def sample_ground_truth():
    return [
        [{'content': 'machine learning', 'start_index': 0, 'end_index': 0}],
        [{'content': 'natural language processing', 'start_index': 0, 'end_index': 0}]
    ]

def test_evaluator_initialization():
    evaluator = RAGRetrievalEvaluator(chunk_size=100, overlap_size=0)
    assert evaluator.chunk_size == 100
    assert evaluator.overlap_size == 0
    assert evaluator.chunks is None
    assert evaluator.chunk_embeddings is None

def test_process_corpus(sample_corpus):
    evaluator = RAGRetrievalEvaluator()
    processed_docs = evaluator.process_corpus(sample_corpus)
    assert len(processed_docs) == 2
    assert 'machine learning' in processed_docs[0]
    assert 'natural language processing' in processed_docs[1]

def test_build_vector_base(sample_corpus):
    evaluator = RAGRetrievalEvaluator(chunk_size=50)
    chunks = evaluator.build_vector_base(sample_corpus)
    assert len(chunks) > 0
    assert evaluator.chunk_embeddings is not None
    assert len(evaluator.chunk_embeddings) == len(chunks)
    assert evaluator.chunk_to_doc_map is not None
    assert len(evaluator.chunk_to_doc_map) == len(chunks)

def test_retrieve_chunks(sample_corpus):
    evaluator = RAGRetrievalEvaluator(chunk_size=50)
    evaluator.build_vector_base(sample_corpus)
    query = "What is machine learning?"
    retrieved_chunks = evaluator.retrieve_chunks(query, num_retrieved_chunks=2)
    assert len(retrieved_chunks) == 2
    assert isinstance(retrieved_chunks[0], tuple)
    assert len(retrieved_chunks[0]) == 2

def test_evaluate_retrieval(sample_corpus, sample_queries, sample_ground_truth):
    evaluator = RAGRetrievalEvaluator(chunk_size=50)
    metrics = evaluator.evaluate_retrieval(
        queries=sample_queries,
        ground_truth=sample_ground_truth,
        corpus=sample_corpus,
        num_retrieved_chunks=2
    )
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'iou_score' in metrics
    assert all(0 <= score <= 1 for score in metrics.values())

def test_invalid_corpus_format():
    evaluator = RAGRetrievalEvaluator()
    invalid_corpus = [{'content': 'Invalid format'}]
    with pytest.raises(ValueError):
        evaluator.process_corpus(invalid_corpus)

def test_empty_corpus():
    evaluator = RAGRetrievalEvaluator()
    empty_corpus = []
    processed_docs = evaluator.process_corpus(empty_corpus)
    assert len(processed_docs) == 0 