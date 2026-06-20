"""Embeddings for semantic recall — optional and torch-free.

The default :class:`Model2VecEmbedder` wraps `model2vec <https://github.com/MinishLab/model2vec>`_
static embeddings: ~10 MB models, CPU-only, no PyTorch. Bring your own embedder by passing
any object that satisfies the :class:`Embedder` protocol to ``Memory(embedder=...)``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .errors import EmbeddingsUnavailable

DEFAULT_MODEL = "minishlab/potion-base-8M"


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn texts into fixed-length vectors.

    Implementations must be deterministic for a given input and return one vector per
    input text, each of length :attr:`dim`.
    """

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class Model2VecEmbedder:
    """Static-embedding embedder (no torch, no GPU).

    The model is loaded lazily on the first :meth:`embed` call so that constructing a
    :class:`~agentrecall.memory.Memory` in ``embeddings="auto"`` mode is cheap. Raises
    :class:`~agentrecall.errors.EmbeddingsUnavailable` if ``model2vec`` is not installed.
    """

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self._model_name = model
        self._model = None
        self._dim: int | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from model2vec import StaticModel
        except ImportError as exc:  # pragma: no cover - exercised via guarded tests
            raise EmbeddingsUnavailable(
                "Semantic search needs the optional 'semantic' extra. "
                "Install it with:  pip install 'agentrecall[semantic]'"
            ) from exc
        self._model = StaticModel.from_pretrained(self._model_name)
        # model2vec exposes the embedding dimension on the loaded model.
        self._dim = int(self._model.dim)

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        assert self._dim is not None
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        assert self._model is not None
        vectors = self._model.encode(list(texts))
        # model2vec returns a numpy array; normalise to plain Python floats.
        return [[float(x) for x in row] for row in vectors]


def get_default_embedder() -> Embedder | None:
    """Return a :class:`Model2VecEmbedder` if ``model2vec`` is importable, else ``None``.

    Never raises — this is the probe used by ``embeddings="auto"``.
    """
    import importlib.util

    if importlib.util.find_spec("model2vec") is None:
        return None
    return Model2VecEmbedder()


__all__ = ["Embedder", "Model2VecEmbedder", "get_default_embedder", "DEFAULT_MODEL"]
