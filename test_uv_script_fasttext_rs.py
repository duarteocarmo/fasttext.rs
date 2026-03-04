# /// script
# requires-python = ">=3.9"
# dependencies = ["fasttext-rs"]
# ///

"""
Quick smoke test for fasttext-rs.

Usage:
    uv run test_uv_script_fasttext_rs.py
"""

import tempfile
import os

import fasttext_rs

TRAIN_DATA = """\
__label__positive I love this movie it was great
__label__positive fantastic film really enjoyed it
__label__positive wonderful acting and beautiful story
__label__positive best movie I have seen in years
__label__positive amazing performances all around
__label__negative terrible movie waste of time
__label__negative awful film would not recommend
__label__negative boring and poorly written script
__label__negative worst movie ever do not watch
__label__negative disappointing and dull experience
"""

TEST_DATA = """\
__label__positive really great film loved every minute
__label__negative horrible movie total waste
"""

with tempfile.TemporaryDirectory() as tmp:
    train_path = os.path.join(tmp, "train.txt")
    test_path = os.path.join(tmp, "test.txt")

    with open(train_path, "w") as f:
        f.write(TRAIN_DATA)
    with open(test_path, "w") as f:
        f.write(TEST_DATA)

    model = fasttext_rs.train_supervised(input=train_path, epoch=25, lr=1.0, dim=10, verbose=0)

    print(f"words: {len(model.words)}, labels: {len(model.labels)}, dim: {model.dim}")

    labels, probs = model.predict("this film was absolutely wonderful")
    print(f"predict: {labels[0]} ({probs[0]:.4f})")

    n, p, r = model.test(test_path, k=1)
    print(f"test: n={n}, precision@1={p:.4f}, recall@1={r:.4f}")

    vec = model.get_word_vector("movie")
    print(f"vector('movie'): [{vec[0]:.4f}, {vec[1]:.4f}, ...]")

    model_path = os.path.join(tmp, "model.bin")
    model.save_model(model_path)
    loaded = fasttext_rs.load_model(model_path)
    labels2, _ = loaded.predict("this film was absolutely wonderful")
    assert labels == labels2, "save/load roundtrip failed"
    print("save/load: ok")

print("\n✓ all good")
