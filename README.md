<p align="center">
  <img src="assets/fasttext-rs-logo.png" alt="fasttext.rs logo" width="500">
</p>

> [!WARNING]
> This was completely AI-generated. Running on a ralph wiggum loop to see what was possible. I HAVE NOT checked the code.

# fasttext.rs

A Rust-backed Python implementation of Facebook's [fastText](https://github.com/facebookresearch/fastText).

## Performance

Benchmarked on the [cooking.stackexchange](https://fasttext.cc/docs/en/supervised-tutorial.html) dataset (12k training, 3k validation) with default parameters, single-threaded. Median of 5 runs, Apple M3 Pro.

| Task | fasttext (C++) | fasttext.rs (Rust) | Speedup |
|---|--:|--:|--:|
| Training (5 epochs) | 12.633s | 6.406s | **1.97x** |
| Inference (3000 samples) | 0.443s | 0.218s | **2.03x** |
| Precision@1 | 0.1363 | 0.1543 | — |
| Recall@1 | 0.0590 | 0.0703 | — |

Run `make bench` to reproduce.

## Installation

```bash
uv pip install -e .
```

## Quick start

```python
import fasttext_rs

# Train
model = fasttext_rs.train_supervised(input="data/cooking.train", epoch=25, lr=1.0, word_ngrams=2)

# Predict
labels, probs = model.predict("Which baking dish is best?")

# Vectors
vec = model.get_word_vector("hello")

# Save / Load
model.save_model("model.bin")
model = fasttext_rs.load_model("model.bin")
```

## Development

```bash
make           # Show all commands
```
