> [!WARNING]
> This was completely AI-generated. Running on a ralph wiggum loop to see what was possible. I HAVE NOT checked the code.

# fasttext.rs

A Rust-backed Python implementation of Facebook's [fastText](https://github.com/facebookresearch/fastText).

## Performance

Benchmarked on the [cooking.stackexchange](https://fasttext.cc/docs/en/supervised-tutorial.html) dataset (12k training, 3k validation). Median of 5 runs, Apple M3 Pro.

| Task | fasttext (C++) | fasttext.rs (Rust) | Speedup |
|---|--:|--:|--:|
| Training (5 epochs) | 7.856s | 6.442s | **1.22x** |
| Precision@1 | 0.4647 | 0.4793 | — |
| Recall@1 | 0.2010 | 0.2367 | — |

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
