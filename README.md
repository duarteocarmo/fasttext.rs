> [!WARNING]
> This was completely AI-generated. Running on a ralph wiggum loop to see what was possible. I HAVE NOT checked the code.

# fasttext.rs

A Rust-backed Python implementation of Facebook's [fastText](https://github.com/facebookresearch/fastText).

## Installation

```bash
uv pip install -e .
```

## Quick Start

```python
import fasttext_rs

# Train a supervised model
model = fasttext_rs.train_supervised(input="data/cooking.train", epoch=25, lr=1.0, wordNgrams=2)

# Predict
labels, probs = model.predict("Which baking dish is best?")
print(labels, probs)

# Get word vector
vec = model.get_word_vector("hello")

# Save / Load
model.save_model("model.bin")
model = fasttext_rs.load_model("model.bin")
```

## Development

```bash
make dev      # Build in dev mode
make build    # Build in release mode
make test     # Run tests
make data     # Download test data
make check    # Cargo check + clippy
```
