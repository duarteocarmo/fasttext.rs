default: help

ENV_FLAGS = PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

.PHONY: help
help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done

.PHONY: build
build: # Build the Rust extension in release mode
	$(ENV_FLAGS) uv run maturin develop --release

.PHONY: dev
dev: # Build the Rust extension in debug mode
	$(ENV_FLAGS) uv run maturin develop

.PHONY: test
test: build # Run all tests
	uv run pytest tests/ -v

.PHONY: check
check: # Run cargo check and clippy
	cargo check
	cargo clippy

.PHONY: format
format: # Format Rust and Python code
	cargo fmt
	uv run ruff format .
	uv run ruff check . --fix

.PHONY: lint
lint: # Lint Rust and Python code
	cargo clippy -- -D warnings
	uv run ruff format --check .
	uv run ruff check .

.PHONY: data
data: # Download test data (cooking.stackexchange)
	mkdir -p data
	cd data && curl -L -O https://dl.fbaipublicfiles.com/fasttext/data/cooking.stackexchange.tar.gz
	cd data && tar -xzf cooking.stackexchange.tar.gz
	head -n 12404 data/cooking.stackexchange.txt > data/cooking.train
	tail -n 3000 data/cooking.stackexchange.txt > data/cooking.valid

.PHONY: bench
bench: build # Run performance benchmarks
	uv run python tests/bench.py

.PHONY: clean
clean: # Clean build artifacts
	cargo clean
	@rm -rf target/ dist/ *.egg-info
	@rm -rf .pytest_cache **/.pytest_cache
	@rm -rf __pycache__ **/__pycache__
