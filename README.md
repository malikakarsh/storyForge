# StoryForge

StoryForge is a Python project for developing small decoder-only transformer
language models from scratch on TinyStories, with a focus on configurable
experiments, text generation, and GPU performance optimization.

## Implemented

- Installable Python package with a `src/` layout.
- TinyStories preparation, duplicate filtering, and byte-level BPE tokenization.
- Packed token streams and configurable next-token batch sampling.
- ForgeGPT with causal multi-head attention, pre-norm blocks, and tied embeddings.
- Configurable story-boundary masking for packed sequences.
- AdamW training with gradient accumulation, learning-rate scheduling, validation,
  checkpointing, and loss logging.
- Seeded text generation and training-curve visualization.

## Baseline results

The 13.89M-parameter model was trained from scratch on TinyStories using Apple
Silicon MPS in float32.

| Metric | Result |
|---|---:|
| Processed training target tokens | 25,001,984 |
| Best validation loss | 2.2216 |
| Validation perplexity | 9.2217 |
| Training time, excluding evaluation and checkpointing | 12 min 55.60 s |
| Training throughput | 32,236 target tokens/s |
| Peak GPU memory | Not measured on MPS |

A separate ten-update MPS profiling run observed maximum tensor allocation of
0.963 GiB and driver allocation of 2.096 GiB, sampled at a requested 10 ms
interval. These are sampled maxima, not the exact peak of the full training run.

![ForgeGPT training and validation loss](results/baseline/loss_curves.png)

See the [experiment report](results/baseline/README.md) for training settings,
the evaluation scope, and continuations from all three checkpoints.

## Story-boundary masking results

The masked experiment uses the same initial weights, tokenizer, data, model
dimensions, and training budget as the baseline, with attention restricted to
the current story.

| Metric | Baseline | Story-masked |
|---|---:|---:|
| Processed training target tokens | 25,001,984 | 25,001,984 |
| Best validation loss | 2.2216 | 2.1951 |
| Validation perplexity | 9.2217 | 8.9809 |
| Training time | 12 min 55.60 s | 14 min 15.83 s |
| Training throughput | 32,236 target tokens/s | 29,214 target tokens/s |

Each validation score uses its run's attention rule, so the available context
differs. These are end-to-end configuration results rather than an isolated
comparison of training effects. Only the masked full run had background memory
monitoring, so the throughput difference cannot be attributed solely to masking.

Across the masked training loop, 71,683 readings at a requested 10 ms interval
observed maximum tensor allocation of **1.087 GiB** and maximum Metal driver
allocation of **2.113 GiB**. Monitoring includes periodic validation and sample
generation, excluding initial setup. These are sampled maxima, not guaranteed
exact peaks; their scope differs from the baseline's ten-update memory probe.

![Story-masked training and validation loss](results/story_masked/loss_curves.png)

See the [masked experiment report](results/story_masked/README.md) for the saved
metrics, memory profile, and initial, halfway, and final story samples.

## Training budget: 25M versus 40M targets

Both experiments use the six-block, story-masked model, the same untrained
starting weights, tokenizer, data, and validation windows. The 40M experiment is
a fresh training run rather than a continuation of the 25M checkpoint. Its cosine
decay spans the longer run, so the comparison changes both training duration and
the learning-rate trajectory.

| Metric | Story-masked 25M | Story-masked 40M |
|---|---:|---:|
| Processed training target tokens | 25,001,984 | 40,009,728 |
| Optimizer updates | 1,526 | 2,442 |
| Best validation loss | 2.1951 | 1.9416 |
| Validation perplexity | 8.9809 | 6.9701 |
| Training time | 14 min 15.83 s | 23 min 00.23 s |
| Training throughput | 29,214 target tokens/s | 28,988 target tokens/s |
| Maximum observed tensor allocation | 1.087 GiB | 1.031 GiB |
| Maximum observed driver allocation | 2.113 GiB | 2.113 GiB |

Both runs include background MPS memory monitoring. The 40M run collected
114,074 readings at a requested 10 ms interval across its main loop, including
periodic validation and generation. Memory values are sampled maxima; the lower
observed tensor value does not establish a reduction in required peak memory.

![40M-target story-masked training and validation loss](results/story_masked_40m/loss_curves.png)

See the [40M experiment report](results/story_masked_40m/README.md) for the
measurements, configuration details, and generated samples.

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
