import numpy as np
import pytest

from qc_copilot.config import get_settings
from qc_copilot.retrieval.embeddings import OnnxEmbedder, SentenceTransformerEmbedder

MODEL = get_settings().embedding_model
TEXTS = [
    "Dairy & Eggs: report within 2 hours of delivery.",
    "Frostbite Green Peas Frozen 1000 g is a frozen product.",
    "How long do card refunds take?",
]


@pytest.fixture(scope="module")
def onnx():
    return OnnxEmbedder(MODEL)


def test_onnx_embedder_returns_unit_vectors_of_the_model_dimension(onnx):
    vectors = np.array(onnx.encode_documents(TEXTS))
    assert vectors.shape == (3, onnx.dimension) == (3, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert onnx.encode_documents([]) == []


def test_onnx_query_matches_document_encoding(onnx):
    query = np.array(onnx.encode_query(TEXTS[0]))
    doc = np.array(onnx.encode_documents(TEXTS[:1])[0])
    assert np.allclose(query, doc, atol=1e-6)


def test_onnx_batches_are_independent_of_batch_size(onnx):
    one = np.array(onnx.encode_documents(TEXTS, batch_size=1))
    many = np.array(onnx.encode_documents(TEXTS, batch_size=32))
    assert np.allclose(one, many, atol=1e-5)


def test_onnx_and_sentence_transformers_agree(onnx):
    pytest.importorskip("sentence_transformers")
    torch_vectors = np.array(SentenceTransformerEmbedder(MODEL).encode_documents(TEXTS))
    onnx_vectors = np.array(onnx.encode_documents(TEXTS))
    cosine = (torch_vectors * onnx_vectors).sum(axis=1)
    assert np.all(cosine > 0.999)
