"""Local embedding backends.

Two implementations produce the same vectors for the same model. The sentence-transformers
backend is the development and CI default. The ONNX backend runs the identical model
through onnxruntime with no torch dependency, which is what lets the deployed container
fit in 512 MB of memory. Both normalise their output so cosine similarity in Qdrant is
a plain dot product.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Protocol

import numpy as np

from qc_copilot.config import Settings

Backend = Literal["torch", "onnx"]


class Embedder(Protocol):
    model_name: str

    @property
    def dimension(self) -> int: ...

    def encode_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]: ...

    def encode_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)

    @property
    def dimension(self) -> int:
        return int(self._model.get_embedding_dimension())

    def encode_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [vector.tolist() for vector in vectors]

    def encode_query(self, text: str) -> list[float]:
        return self.encode_documents([text])[0]


class OnnxEmbedder:
    """MiniLM-style encoder via onnxruntime: mean pooling over the mask, then L2 norm."""

    def __init__(
        self, model_name: str, onnx_file: str = "onnx/model.onnx", max_length: int = 256
    ) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self.model_name = model_name
        self._tokenizer = Tokenizer.from_file(hf_hub_download(model_name, "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=max_length)
        self._tokenizer.enable_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            hf_hub_download(model_name, onnx_file),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._dimension = int(self._session.get_outputs()[0].shape[-1])

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            vectors.extend(self._encode_batch(texts[start : start + batch_size]))
        return vectors

    def encode_query(self, text: str) -> list[float]:
        return self._encode_batch([text])[0]

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        encoded = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        feed = {"input_ids": input_ids, "attention_mask": attention}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)
        hidden = self._session.run(None, feed)[0]
        mask = attention[..., None].astype(np.float32)
        pooled = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        normalised = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        return normalised.astype(np.float32).tolist()


@lru_cache(maxsize=2)
def get_embedder(model_name: str, backend: Backend = "torch", device: str = "cpu") -> Embedder:
    """Cache the model per name so the weights load once per process."""
    if backend == "onnx":
        return OnnxEmbedder(model_name)
    return SentenceTransformerEmbedder(model_name, device)


def build_embedder(settings: Settings) -> Embedder:
    return get_embedder(
        settings.embedding_model, settings.embedding_backend, settings.embedding_device
    )
