"""
Performance benchmark: fasttext.rs (Rust) vs fasttext (C++).

Follows the fastText supervised tutorial with default parameters:
https://fasttext.cc/docs/en/supervised-tutorial.html

Both implementations pinned to thread=1 for a fair single-threaded comparison.

Usage:
    make bench
"""

import os
import statistics
import time

import fasttext

import fasttext_rs

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TRAIN_FILE = os.path.join(DATA_DIR, "cooking.train")
VALID_FILE = os.path.join(DATA_DIR, "cooking.valid")

WARMUP = 1
ROUNDS = 5


def main():
    assert os.path.exists(TRAIN_FILE), f"Run 'make data' first ({TRAIN_FILE} not found)"

    print(f"{'═' * 56}")
    print("  fasttext.rs benchmark (single-threaded)")
    print(f"  {ROUNDS} rounds, {WARMUP} warmup")
    print(f"  train: {TRAIN_FILE}")
    print(f"  valid: {VALID_FILE}")
    print(f"{'═' * 56}")

    ft_train_times = []
    rs_train_times = []

    for i in range(WARMUP + ROUNDS):
        t0 = time.perf_counter()
        ft_model = fasttext.train_supervised(input=TRAIN_FILE, thread=1, verbose=0)
        ft_elapsed = time.perf_counter() - t0

        t0 = time.perf_counter()
        rs_model = fasttext_rs.train_supervised(input=TRAIN_FILE, verbose=0)
        rs_elapsed = time.perf_counter() - t0

        if i >= WARMUP:
            ft_train_times.append(ft_elapsed)
            rs_train_times.append(rs_elapsed)

    ft_n, ft_p, ft_r = ft_model.test(VALID_FILE, k=1)
    rs_n, rs_p, rs_r = rs_model.test(VALID_FILE, k=1)

    ft_train_med = statistics.median(ft_train_times)
    rs_train_med = statistics.median(rs_train_times)
    train_speedup = ft_train_med / rs_train_med

    print()
    print(f"  | {'Task':<30} | {'C++':>10} | {'Rust':>10} | {'Speedup':>8} |")
    print(f"  |{'-' * 32}|{'-' * 12}|{'-' * 12}|{'-' * 10}|")
    print(
        f"  | {'Training (5 epochs)':<30} | {ft_train_med:>9.3f}s |"
        f" {rs_train_med:>9.3f}s | {train_speedup:>6.2f}x  |"
    )
    print(f"  | {'Precision@1':<30} | {ft_p:>10.4f} | {rs_p:>10.4f} | {'—':>8} |")
    print(f"  | {'Recall@1':<30} | {ft_r:>10.4f} | {rs_r:>10.4f} | {'—':>8} |")
    print(f"  | {'Samples (N)':<30} | {ft_n:>10} | {rs_n:>10} | {'—':>8} |")

    print(f"\n{'═' * 56}")


if __name__ == "__main__":
    main()
