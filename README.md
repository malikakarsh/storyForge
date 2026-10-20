# StoryForge

StoryForge is a Python project for developing small decoder-only transformer
language models from scratch on TinyStories, with a focus on configurable
experiments, text generation, and GPU performance optimization.

## Implemented

- Installable Python package with a `src/` layout.
- TinyStories streaming and train/validation story selection.
- Whitespace-aware duplicate detection and exclusion of overlapping stories.
- Exploratory notebook with automatic CUDA, MPS, or CPU device selection.

## Setup

Use Python 3.11 and create a fresh environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Open `notebooks/00_playground.ipynb` in VS Code and select `.venv` as the
notebook kernel.
