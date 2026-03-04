> [!WARNING]
> This was completely AI-generated. Running on a ralph wiggum loop to see what was possible. I HAVE NOT checked the code.

# fasttext.rs

A Rust-backed Python implementation of Facebook's [fastText](https://github.com/facebookresearch/fastText).

## Performance

Benchmarked on the [cooking.stackexchange](https://fasttext.cc/docs/en/supervised-tutorial.html) dataset (12k training samples, 3k validation). Median over 5 runs on Apple M3 Pro.

| Task | fasttext (C++) | fasttext.rs (Rust) | Speedup |
|---|--:|--:|--:|
| Training (5 epochs) | 7.698s | 6.899s | **1.12x** |
| Prediction k=1 (3000 samples) | 0.247s | 0.202s | **1.22x** |
| Prediction k=5 (3000 samples) | 0.362s | 0.185s | **1.96x** |
| Accuracy (k=1) | 46.5% | 47.9% | — |

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
