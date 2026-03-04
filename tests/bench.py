"""
Performance benchmark: fasttext.rs (Rust) vs fasttext (C++).

Usage:
    uv run python tests/bench.py
"""

import os
import statistics
import time

import fasttext

import fasttext_rs

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TRAIN_FILE = os.path.join(DATA_DIR, "cooking.train")
VALID_FILE = os.path.join(DATA_DIR, "cooking.valid")

TRAIN_PARAMS = dict(
    lr=1.0,
    epoch=5,
    dim=50,
    ws=5,
    wordNgrams=2,
    minCount=1,
    bucket=2000000,
    loss="softmax",
    thread=1,
)
WARMUP = 1
ROUNDS = 5


def ft_predict(model, text, k=1):
    predictions = model.f.predict(text + "\n", k, 0.0, "strict")
    if predictions:
        probs, labels = zip(*predictions)
        return list(labels), list(probs)
    return [], []


def load_texts(path):
    texts = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            text = " ".join(p for p in parts if not p.startswith("__label__"))
            if text:
                texts.append(text)
    return texts


def bench_training():
    ft_times = []
    rs_times = []

    for i in range(WARMUP + ROUNDS):
        t0 = time.perf_counter()
        fasttext.train_supervised(input=TRAIN_FILE, **TRAIN_PARAMS, verbose=0)
        elapsed = time.perf_counter() - t0
        if i >= WARMUP:
            ft_times.append(elapsed)

        t0 = time.perf_counter()
        fasttext_rs.train_supervised(
            input=TRAIN_FILE,
            lr=TRAIN_PARAMS["lr"],
            epoch=TRAIN_PARAMS["epoch"],
            dim=TRAIN_PARAMS["dim"],
            ws=TRAIN_PARAMS["ws"],
            word_ngrams=TRAIN_PARAMS["wordNgrams"],
            min_count=TRAIN_PARAMS["minCount"],
            bucket=TRAIN_PARAMS["bucket"],
            loss=TRAIN_PARAMS["loss"],
            thread=TRAIN_PARAMS["thread"],
            verbose=0,
        )
        elapsed = time.perf_counter() - t0
        if i >= WARMUP:
            rs_times.append(elapsed)

    return ft_times, rs_times


def bench_prediction(ft_model, rs_model, texts, k=1):
    ft_times = []
    rs_times = []

    for i in range(WARMUP + ROUNDS):
        t0 = time.perf_counter()
        for text in texts:
            ft_predict(ft_model, text, k=k)
        elapsed = time.perf_counter() - t0
        if i >= WARMUP:
            ft_times.append(elapsed)

        t0 = time.perf_counter()
        for text in texts:
            rs_model.predict(text, k=k)
        elapsed = time.perf_counter() - t0
        if i >= WARMUP:
            rs_times.append(elapsed)

    return ft_times, rs_times


def bench_prediction_batch(rs_model, texts, k=1):
    rs_times = []
    for i in range(WARMUP + ROUNDS):
        t0 = time.perf_counter()
        rs_model.predict(texts, k=k)
        elapsed = time.perf_counter() - t0
        if i >= WARMUP:
            rs_times.append(elapsed)
    return rs_times


def fmt_stats(times):
    med = statistics.median(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return med, std


def print_comparison(label, ft_times, rs_times, unit="s"):
    ft_med, ft_std = fmt_stats(ft_times)
    rs_med, rs_std = fmt_stats(rs_times)
    speedup = ft_med / rs_med if rs_med > 0 else float("inf")

    print(f"\n  {label}")
    print(f"  {'─' * 50}")
    print(f"  fasttext (C++)  : {ft_med:.4f}{unit} ± {ft_std:.4f}{unit}")
    print(f"  fasttext.rs     : {rs_med:.4f}{unit} ± {rs_std:.4f}{unit}")
    print(f"  speedup         : {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")


def main():
    assert os.path.exists(TRAIN_FILE), f"Run 'make data' first ({TRAIN_FILE} not found)"

    texts = load_texts(VALID_FILE)
    n = len(texts)

    print(f"{'═' * 56}")
    print("  fasttext.rs benchmark")
    print(f"  {ROUNDS} rounds, {WARMUP} warmup, {n} validation samples")
    print(f"{'═' * 56}")

    # Training
    print("\n  ▸ Training...")
    ft_train, rs_train = bench_training()
    print_comparison("Training (5 epochs, cooking.stackexchange)", ft_train, rs_train)

    # Train models for prediction benchmarks
    ft_model = fasttext.train_supervised(input=TRAIN_FILE, **TRAIN_PARAMS, verbose=0)
    rs_model = fasttext_rs.train_supervised(
        input=TRAIN_FILE,
        lr=TRAIN_PARAMS["lr"],
        epoch=TRAIN_PARAMS["epoch"],
        dim=TRAIN_PARAMS["dim"],
        ws=TRAIN_PARAMS["ws"],
        word_ngrams=TRAIN_PARAMS["wordNgrams"],
        min_count=TRAIN_PARAMS["minCount"],
        bucket=TRAIN_PARAMS["bucket"],
        loss=TRAIN_PARAMS["loss"],
        thread=TRAIN_PARAMS["thread"],
        verbose=0,
    )

    # Prediction k=1
    print("\n  ▸ Prediction (k=1)...")
    ft_pred1, rs_pred1 = bench_prediction(ft_model, rs_model, texts, k=1)
    print_comparison(f"Prediction k=1 ({n} samples)", ft_pred1, rs_pred1)

    # Prediction k=5
    print("\n  ▸ Prediction (k=5)...")
    ft_pred5, rs_pred5 = bench_prediction(ft_model, rs_model, texts, k=5)
    print_comparison(f"Prediction k=5 ({n} samples)", ft_pred5, rs_pred5)

    # Batch prediction
    print("\n  ▸ Batch prediction (k=1)...")
    rs_batch = bench_prediction_batch(rs_model, texts, k=1)
    rs_batch_med, rs_batch_std = fmt_stats(rs_batch)
    rs_single_med, _ = fmt_stats(rs_pred1)
    batch_speedup = rs_single_med / rs_batch_med if rs_batch_med > 0 else float("inf")

    print(f"\n  Batch prediction k=1 ({n} samples)")
    print(f"  {'─' * 50}")
    print(f"  single loop     : {rs_single_med:.4f}s")
    print(f"  batch           : {rs_batch_med:.4f}s ± {rs_batch_std:.4f}s")
    print(f"  batch speedup   : {batch_speedup:.2f}x")

    # Accuracy comparison
    print(f"\n  {'─' * 50}")
    print("  Accuracy (validation set, k=1)")
    print(f"  {'─' * 50}")

    ft_correct = rs_correct = total = 0
    with open(VALID_FILE) as f:
        for line in f:
            parts = line.strip().split()
            true_labels = [p for p in parts if p.startswith("__label__")]
            text = " ".join(p for p in parts if not p.startswith("__label__"))
            if not true_labels or not text:
                continue
            total += 1
            ft_labels, _ = ft_predict(ft_model, text)
            rs_labels, _ = rs_model.predict(text, k=1)
            if ft_labels and ft_labels[0] in true_labels:
                ft_correct += 1
            if rs_labels and rs_labels[0] in true_labels:
                rs_correct += 1

    print(f"  fasttext (C++)  : {ft_correct / total:.4f} ({ft_correct}/{total})")
    print(f"  fasttext.rs     : {rs_correct / total:.4f} ({rs_correct}/{total})")

    # Summary table
    ft_t_med, _ = fmt_stats(ft_train)
    rs_t_med, _ = fmt_stats(rs_train)
    ft_p1_med, _ = fmt_stats(ft_pred1)
    rs_p1_med, _ = fmt_stats(rs_pred1)
    ft_p5_med, _ = fmt_stats(ft_pred5)
    rs_p5_med, _ = fmt_stats(rs_pred5)

    print(f"\n{'═' * 56}")
    print("  Summary")
    print(f"{'═' * 56}")
    print(f"  {'Task':<28} {'C++ (s)':>8} {'Rust (s)':>9} {'Speedup':>9}")
    print(f"  {'─' * 50}")
    print(
        f"  {'Training':<28} {ft_t_med:>8.3f} {rs_t_med:>9.3f} {ft_t_med / rs_t_med:>8.2f}x"
    )
    print(
        f"  {'Prediction k=1':<28} {ft_p1_med:>8.3f} {rs_p1_med:>9.3f} {ft_p1_med / rs_p1_med:>8.2f}x"
    )
    print(
        f"  {'Prediction k=5':<28} {ft_p5_med:>8.3f} {rs_p5_med:>9.3f} {ft_p5_med / rs_p5_med:>8.2f}x"
    )
    print(f"  {'Accuracy':<28} {ft_correct / total:>8.4f} {rs_correct / total:>9.4f}")
    print(f"{'═' * 56}")


if __name__ == "__main__":
    main()
