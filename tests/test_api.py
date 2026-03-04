"""
Tests for full API parity with the official fasttext Python module.
https://fasttext.cc/docs/en/python-module.html
"""

import os
import pytest
import fasttext_rs

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TRAIN_FILE = os.path.join(DATA_DIR, "cooking.train")
VALID_FILE = os.path.join(DATA_DIR, "cooking.valid")


@pytest.fixture(scope="module")
def model():
    """Train a supervised model for testing."""
    return fasttext_rs.train_supervised(
        input=TRAIN_FILE,
        lr=1.0,
        epoch=5,
        dim=50,
        ws=5,
        word_ngrams=2,
        min_count=1,
        bucket=2000000,
        loss="softmax",
        thread=1,
        verbose=0,
    )


class TestProperties:
    """Test all training arg properties."""

    def test_dim(self, model):
        assert model.dim == 50

    def test_lr(self, model):
        assert model.lr == 1.0

    def test_ws(self, model):
        assert model.ws == 5

    def test_epoch(self, model):
        assert model.epoch == 5

    def test_min_count(self, model):
        assert model.min_count == 1

    def test_min_count_label(self, model):
        assert model.min_count_label == 0

    def test_minn(self, model):
        assert model.minn == 0

    def test_maxn(self, model):
        assert model.maxn == 0

    def test_neg(self, model):
        assert model.neg == 5

    def test_word_ngrams(self, model):
        assert model.word_ngrams == 2

    def test_loss(self, model):
        assert model.loss == "softmax"

    def test_bucket(self, model):
        assert model.bucket == 2000000

    def test_thread(self, model):
        assert model.thread == 1

    def test_lr_update_rate(self, model):
        assert model.lr_update_rate == 100

    def test_t(self, model):
        assert model.t == 1e-4

    def test_label(self, model):
        assert model.label == "__label__"

    def test_verbose(self, model):
        assert model.verbose == 0

    def test_is_quantized(self, model):
        assert model.is_quantized is False

    def test_get_dimension(self, model):
        assert model.get_dimension() == 50
        assert model.get_dimension() == model.dim


class TestDunderMethods:
    """Test __getitem__, __contains__, __repr__."""

    def test_getitem(self, model):
        vec = model["pizza"]
        assert isinstance(vec, list)
        assert len(vec) == model.dim
        assert any(v != 0.0 for v in vec)

    def test_getitem_same_as_get_word_vector(self, model):
        assert model["pizza"] == model.get_word_vector("pizza")

    def test_contains_in_vocab(self, model):
        assert "pizza" in model
        assert "the" in model

    def test_contains_not_in_vocab(self, model):
        assert "xyznonexistent123" not in model

    def test_repr(self, model):
        r = repr(model)
        assert "FastTextModel" in r
        assert "dim=50" in r


class TestDictionary:
    """Test dictionary methods."""

    def test_get_word_id_known(self, model):
        wid = model.get_word_id("pizza")
        assert wid >= 0

    def test_get_word_id_unknown(self, model):
        wid = model.get_word_id("xyznonexistent")
        assert wid == -1

    def test_get_subword_id(self, model):
        sid = model.get_subword_id("<pi")
        assert isinstance(sid, int)
        assert sid >= 0

    def test_get_subwords(self, model):
        subwords, indices = model.get_subwords("pizza")
        assert isinstance(subwords, list)
        assert isinstance(indices, list)
        assert len(indices) >= 1  # at least the word itself
        assert "pizza" in subwords

    def test_get_line(self, model):
        words, labels = model.get_line("__label__baking How to bake a cake?")
        assert "How" in words
        assert "bake" in words
        assert "__label__baking" in labels

    def test_get_line_no_labels(self, model):
        words, labels = model.get_line("How to bake a cake?")
        assert len(words) > 0
        assert len(labels) == 0


class TestPrediction:
    """Test prediction including batch mode."""

    def test_predict_single(self, model):
        labels, probs = model.predict("How to bake bread?")
        assert len(labels) == 1
        assert len(probs) == 1
        assert labels[0].startswith("__label__")
        assert 0.0 <= probs[0] <= 1.0

    def test_predict_top_k(self, model):
        labels, probs = model.predict("How to bake bread?", k=5)
        assert len(labels) == 5
        assert len(probs) == 5
        # Probabilities should be descending
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1]

    def test_predict_batch(self, model):
        results = model.predict(["How to bake bread?", "Best knife?"], k=2)
        assert isinstance(results, list)
        assert len(results) == 2
        for labels, probs in results:
            assert len(labels) == 2
            assert len(probs) == 2

    def test_predict_batch_empty(self, model):
        results = model.predict([], k=1)
        assert results == []


class TestEvaluation:
    """Test test() and test_label()."""

    def test_test(self, model):
        n, p, r = model.test(VALID_FILE, k=1)
        assert n == 3000
        assert 0.0 <= p <= 1.0
        assert 0.0 <= r <= 1.0
        assert p > 0.1  # should be at least somewhat accurate

    def test_test_k5(self, model):
        n, p, r = model.test(VALID_FILE, k=5)
        assert n == 3000
        # R@5 should be higher than R@1
        _, _, r1 = model.test(VALID_FILE, k=1)
        assert r >= r1

    def test_test_label(self, model):
        results = model.test_label(VALID_FILE, k=1)
        assert isinstance(results, dict)
        assert len(results) > 0
        # Check structure
        for label, stats in results.items():
            assert label.startswith("__label__")
            assert "precision" in stats
            assert "recall" in stats
            assert "count" in stats
            assert 0.0 <= stats["precision"] <= 1.0
            assert 0.0 <= stats["recall"] <= 1.0
            assert stats["count"] >= 0


class TestMatrices:
    """Test matrix access methods."""

    def test_get_input_vector(self, model):
        vec = model.get_input_vector(0)
        assert len(vec) == model.dim
        assert any(v != 0.0 for v in vec)

    def test_get_input_vector_out_of_range(self, model):
        vec = model.get_input_vector(999999999)
        assert len(vec) == model.dim
        assert all(v == 0.0 for v in vec)

    def test_get_input_matrix(self, model):
        matrix = model.get_input_matrix()
        assert isinstance(matrix, list)
        assert len(matrix) > 0
        assert len(matrix[0]) == model.dim

    def test_get_output_matrix(self, model):
        matrix = model.get_output_matrix()
        assert isinstance(matrix, list)
        assert len(matrix) == len(model.labels)
        assert len(matrix[0]) == model.dim


class TestTokenize:
    """Test module-level tokenize function."""

    def test_tokenize_basic(self):
        tokens = fasttext_rs.tokenize("Hello world foo bar")
        assert tokens == ["Hello", "world", "foo", "bar"]

    def test_tokenize_empty(self):
        tokens = fasttext_rs.tokenize("")
        assert tokens == []

    def test_tokenize_extra_whitespace(self):
        tokens = fasttext_rs.tokenize("  hello   world  ")
        assert tokens == ["hello", "world"]
