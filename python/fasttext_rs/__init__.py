from .fasttext_rs import (  # noqa: F401
    FastTextModel,
    load_model,
    tokenize,
    train_supervised,
    train_unsupervised,
)

__all__ = [
    "train_supervised",
    "train_unsupervised",
    "load_model",
    "tokenize",
    "FastTextModel",
]
