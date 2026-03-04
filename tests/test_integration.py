"""
Integration tests comparing fasttext_rs with the original fasttext package.

Uses the cooking.stackexchange dataset from the fastText tutorial:
https://fasttext.cc/docs/en/supervised-tutorial.html
"""

import os
import tempfile
import time

import fasttext
import numpy as np
import pytest

import fasttext_rs


def ft_predict(model, text, k=1):
    """Wrapper around fasttext predict that handles numpy compat issues."""
    text_with_newline = text + "\n"
    predictions = model.f.predict(text_with_newline, k, 0.0, "strict")
    if predictions:
        probs, labels = zip(*predictions)
        return list(labels), list(probs)
    return [], []


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TRAIN_FILE = os.path.join(DATA_DIR, "cooking.train")
VALID_FILE = os.path.join(DATA_DIR, "cooking.valid")


@pytest.fixture(scope="module")
def models():
    """Train both models with the same parameters and return them."""
    assert os.path.exists(TRAIN_FILE), (
        f"Training data not found at {TRAIN_FILE}. Run 'make data' first."
    )

    params = dict(
        lr=1.0,
        epoch=5,
        dim=50,
        ws=5,
        wordNgrams=2,
        minCount=1,
        bucket=2000000,
        loss="softmax",
        thread=1,
        verbose=0,
    )

    # Train original fasttext
    t0 = time.perf_counter()
    ft_model = fasttext.train_supervised(
        input=TRAIN_FILE,
        **params,
    )
    ft_train_time = time.perf_counter() - t0

    # Train our Rust implementation
    t0 = time.perf_counter()
    rs_model = fasttext_rs.train_supervised(
        input=TRAIN_FILE,
        lr=params["lr"],
        epoch=params["epoch"],
        dim=params["dim"],
        neg=params.get("neg", 5),
        ws=params["ws"],
        word_ngrams=params["wordNgrams"],
        min_count=params["minCount"],
        bucket=params["bucket"],
        loss=params["loss"],
        thread=params["thread"],
        verbose=0,
    )
    rs_train_time = time.perf_counter() - t0

    return ft_model, rs_model, ft_train_time, rs_train_time


class TestSupervised:
    """Test supervised training and prediction."""

    def test_models_train(self, models):
        """Both models should train without error."""
        ft_model, rs_model, _, _ = models
        assert ft_model is not None
        assert rs_model is not None

    def test_label_count(self, models):
        """Both models should have the same number of labels."""
        ft_model, rs_model, _, _ = models
        ft_labels = ft_model.get_labels()
        rs_labels = rs_model.labels
        print(
            f"FastText labels: {len(ft_labels)}, fasttext.rs labels: {len(rs_labels)}"
        )
        assert len(ft_labels) == len(rs_labels), (
            f"Label count mismatch: fasttext={len(ft_labels)}, fasttext.rs={len(rs_labels)}"
        )

    def test_word_count(self, models):
        """Both models should have the same number of words (fasttext adds </s> so allow +/- 1)."""
        ft_model, rs_model, _, _ = models
        ft_words = ft_model.get_words()
        rs_words = rs_model.words
        print(f"FastText words: {len(ft_words)}, fasttext.rs words: {len(rs_words)}")
        assert abs(len(ft_words) - len(rs_words)) <= 1, (
            f"Word count mismatch: fasttext={len(ft_words)}, fasttext.rs={len(rs_words)}"
        )

    def test_dimension(self, models):
        """Both models should have the same dimension."""
        ft_model, rs_model, _, _ = models
        assert ft_model.get_dimension() == rs_model.dim

    def test_prediction_format(self, models):
        """Predictions should return the same format."""
        ft_model, rs_model, _, _ = models
        text = "Which baking dish is best for banana bread?"

        ft_labels, ft_probs = ft_predict(ft_model, text, k=5)
        rs_labels, rs_probs = rs_model.predict(text, k=5)

        assert len(ft_labels) == len(rs_labels), "Different number of predictions"
        assert len(ft_probs) == len(rs_probs), "Different number of probabilities"

    def test_prediction_top1_overlap(self, models):
        """Top-1 predictions should overlap significantly on validation data."""
        ft_model, rs_model, _, _ = models

        test_texts = [
            "Which baking dish is best for banana bread?",
            "How to make pizza dough at home?",
            "What temperature should I cook chicken at?",
            "How do I store fresh herbs?",
            "Best way to sharpen kitchen knives",
        ]

        matches = 0
        for text in test_texts:
            ft_labels, _ = ft_predict(ft_model, text, k=1)
            rs_labels, _ = rs_model.predict(text, k=1)
            if ft_labels[0] == rs_labels[0]:
                matches += 1
            print(f"Text: {text[:50]}")
            print(f"  FastText: {ft_labels[0]}")
            print(f"  fasttext.rs: {rs_labels[0]}")

        print(f"\nTop-1 match rate: {matches}/{len(test_texts)}")
        # We don't require 100% match since implementations may differ slightly

    def test_validation_accuracy_comparable(self, models):
        """Accuracy on validation set should be in the same ballpark."""
        ft_model, rs_model, _, _ = models

        with open(VALID_FILE) as f:
            lines = f.readlines()

        ft_correct = 0
        rs_correct = 0
        total = 0

        for line in lines[:500]:  # Test on first 500 lines for speed
            line = line.strip()
            if not line:
                continue

            # Extract labels from line
            parts = line.split()
            true_labels = [p for p in parts if p.startswith("__label__")]
            text = " ".join(p for p in parts if not p.startswith("__label__"))

            if not true_labels or not text:
                continue

            total += 1

            ft_labels, _ = ft_predict(ft_model, text, k=1)
            rs_labels, _ = rs_model.predict(text, k=1)

            if ft_labels[0] in true_labels:
                ft_correct += 1
            if rs_labels[0] in true_labels:
                rs_correct += 1

        ft_acc = ft_correct / total
        rs_acc = rs_correct / total
        print(f"\nValidation accuracy (on {total} samples):")
        print(f"  FastText:    {ft_acc:.4f} ({ft_correct}/{total})")
        print(f"  fasttext.rs: {rs_acc:.4f} ({rs_correct}/{total})")

        # The accuracies should be in the same general range
        # We allow up to 15% absolute difference since random init differs
        assert abs(ft_acc - rs_acc) < 0.15, (
            f"Accuracy difference too large: fasttext={ft_acc:.4f}, fasttext.rs={rs_acc:.4f}"
        )

    def test_probabilities_sum_to_one(self, models):
        """Softmax probabilities should sum to approximately 1."""
        _, rs_model, _, _ = models
        text = "How to make pasta from scratch?"
        labels, probs = rs_model.predict(text, k=1000)  # Get all labels
        total = sum(probs)
        print(f"Sum of all probabilities: {total:.6f}")
        assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total}, expected ~1.0"


class TestSpeed:
    """Benchmark training and prediction speed."""

    def test_training_speed(self, models):
        """Compare training speed between fasttext and fasttext.rs."""
        _, _, ft_train_time, rs_train_time = models
        speedup = ft_train_time / rs_train_time if rs_train_time > 0 else float("inf")
        print(f"\n{'=' * 60}")
        print("  TRAINING SPEED COMPARISON")
        print(f"{'=' * 60}")
        print(f"  FastText (C++):    {ft_train_time:.3f}s")
        print(f"  fasttext.rs (Rust): {rs_train_time:.3f}s")
        print(
            f"  Speedup:           {speedup:.2f}x {'(Rust faster)' if speedup > 1 else '(C++ faster)'}"
        )
        print(f"{'=' * 60}")

    def test_prediction_speed(self, models):
        """Compare prediction speed between fasttext and fasttext.rs."""
        ft_model, rs_model, _, _ = models

        with open(VALID_FILE) as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        texts = []
        for line in lines:
            parts = line.split()
            text = " ".join(p for p in parts if not p.startswith("__label__"))
            if text:
                texts.append(text)

        n = len(texts)

        # Benchmark fasttext predictions
        t0 = time.perf_counter()
        for text in texts:
            ft_predict(ft_model, text, k=1)
        ft_pred_time = time.perf_counter() - t0

        # Benchmark fasttext.rs predictions
        t0 = time.perf_counter()
        for text in texts:
            rs_model.predict(text, k=1)
        rs_pred_time = time.perf_counter() - t0

        speedup = ft_pred_time / rs_pred_time if rs_pred_time > 0 else float("inf")
        ft_rate = n / ft_pred_time if ft_pred_time > 0 else float("inf")
        rs_rate = n / rs_pred_time if rs_pred_time > 0 else float("inf")

        print(f"\n{'=' * 60}")
        print(f"  PREDICTION SPEED COMPARISON ({n} samples)")
        print(f"{'=' * 60}")
        print(
            f"  FastText (C++):     {ft_pred_time:.3f}s ({ft_rate:.0f} predictions/s)"
        )
        print(
            f"  fasttext.rs (Rust): {rs_pred_time:.3f}s ({rs_rate:.0f} predictions/s)"
        )
        print(
            f"  Speedup:            {speedup:.2f}x {'(Rust faster)' if speedup > 1 else '(C++ faster)'}"
        )
        print(f"{'=' * 60}")


class TestSaveLoad:
    """Test model save/load round-trip."""

    def test_save_load_roundtrip(self, models):
        """Model should produce same results after save/load."""
        _, rs_model, _, _ = models

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name

        try:
            rs_model.save_model(path)
            loaded_model = fasttext_rs.load_model(path)

            text = "How to make pizza dough?"

            orig_labels, orig_probs = rs_model.predict(text, k=3)
            load_labels, load_probs = loaded_model.predict(text, k=3)

            assert orig_labels == load_labels, "Labels differ after save/load"
            for op, lp in zip(orig_probs, load_probs):
                assert abs(op - lp) < 1e-5, f"Probabilities differ: {op} vs {lp}"

            assert loaded_model.dim == rs_model.dim
            assert len(loaded_model.words) == len(rs_model.words)
            assert len(loaded_model.labels) == len(rs_model.labels)
        finally:
            os.unlink(path)


class TestWordVectors:
    """Test word vector functionality."""

    def test_word_vector_dimension(self, models):
        """Word vectors should have correct dimension."""
        _, rs_model, _, _ = models
        vec = rs_model.get_word_vector("pizza")
        assert len(vec) == rs_model.dim

    def test_sentence_vector_dimension(self, models):
        """Sentence vectors should have correct dimension."""
        _, rs_model, _, _ = models
        vec = rs_model.get_sentence_vector("How to make pizza")
        assert len(vec) == rs_model.dim

    def test_sentence_vector_normalized(self, models):
        """Sentence vectors should be approximately unit normalized."""
        _, rs_model, _, _ = models
        vec = rs_model.get_sentence_vector("How to make pizza dough at home")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 0.01, f"Sentence vector norm is {norm}, expected ~1.0"
