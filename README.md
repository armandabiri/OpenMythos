# OpenMythos

Research-first PyTorch implementation of a recurrent-depth transformer that combines looped inference, MLA or GQA attention, sparse MoE routing, and ACT-style halting.

> **Disclaimer:** OpenMythos is an independent, community-driven theoretical reconstruction based on publicly discussed ideas and papers. It is not affiliated with, endorsed by, or connected to Anthropic or any proprietary Claude system.

## Table of Contents

- [OpenMythos](#openmythos)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [Project Map](#project-map)
  - [System Architecture](#system-architecture)
    - [Inference Sequence](#inference-sequence)
  - [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Quick Start](#quick-start)
  - [CLI Reference](#cli-reference)
  - [Core Functionality](#core-functionality)
    - [Execution Modes](#execution-modes)
    - [Attention Backends](#attention-backends)
    - [Recurrent Depth, Halting, and Stability](#recurrent-depth-halting-and-stability)
    - [MoE Routing](#moe-routing)
    - [Model Variants](#model-variants)
    - [Training Workflow](#training-workflow)
  - [Configuration](#configuration)
    - [Core Model Fields](#core-model-fields)
    - [Attention Fields](#attention-fields)
    - [MoE and Control Fields](#moe-and-control-fields)
  - [Version History](#version-history)
  - [License and Maintenance](#license-and-maintenance)
  - [Citation](#citation)

## Overview

OpenMythos is a Python library for experimenting with a looped transformer architecture that separates computation into a **Prelude**, a shared **Recurrent Block**, and a final **Coda**. The project exposes:

- ✅ A configurable `MythosConfig` dataclass in [open_mythos/main.py](open_mythos/main.py)
- ✅ Dual attention implementations: `MLAttention` and `GQAttention`
- ✅ Sparse MoE feed-forward routing with shared experts
- ✅ LTI-inspired recurrent input injection for loop stability
- ✅ ACT-style halting to decide when recurrent computation should stop
- 🚀 Training scripts for tiny single-GPU runs and larger FineWeb-Edu pretraining
- 💾 Script-driven examples, including an interactive terminal chat demo
- ⚠️ A research reconstruction, not an official implementation of Claude Mythos

### Project Map

```text
open_mythos/   core model, tokenizer, variants, and attention implementations
examples/      interactive chat demo, variants example, and MoDA smoke test
training/      tiny pretraining and 3B FineWeb-Edu training scripts
docs/          API reference and dataset guidance
```

## System Architecture

The runtime is organized around a tokenization stage, a loop-capable model core, and generation or training outputs.

```mermaid
flowchart LR
    subgraph I["Input Pipeline"]
        direction TB
        U["User Text or Dataset<br/><i>prompt, corpus sample, or token batch</i>"]
        T["MythosTokenizer<br/><i>Hugging Face tokenizer wrapper</i>"]
    end

    subgraph M["OpenMythos Core"]
        direction TB
        P["Prelude Blocks<br/><i>standard transformer layers</i>"]
        R["Recurrent Block<br/><i>MLA or GQA + MoE + LoRA depth adaptation</i>"]
        H["ACT Halting<br/><i>continue or stop recurrent depth</i>"]
        C["Coda Blocks<br/><i>final refinement layers</i>"]
    end

    subgraph O["Outputs"]
        direction TB
        L["Logits<br/><i>next-token distribution</i>"]
        G["Generation or Loss<br/><i>decoded text or training objective</i>"]
    end

    U --> T --> P --> R --> H
    H -- continue --> R
    H -- halt --> C --> L --> G

    classDef input fill:#EAF4FF,stroke:#3670A0,stroke-width:1.5px,color:#0F172A;
    classDef core fill:#EEF8EE,stroke:#2F855A,stroke-width:1.5px,color:#0F172A;
    classDef output fill:#FFF7E8,stroke:#C05621,stroke-width:1.5px,color:#0F172A;
    class U,T input;
    class P,R,H,C core;
    class L,G output;
    linkStyle 0 stroke:#3670A0,stroke-width:2px;
    linkStyle 1 stroke:#3670A0,stroke-width:2px;
    linkStyle 2 stroke:#2F855A,stroke-width:2px;
    linkStyle 3 stroke:#2F855A,stroke-width:2px,stroke-dasharray: 4 2;
    linkStyle 4 stroke:#C05621,stroke-width:2px;
    linkStyle 5 stroke:#C05621,stroke-width:2px;
    linkStyle 6 stroke:#C05621,stroke-width:2px;
```

### Inference Sequence

```mermaid
sequenceDiagram
    participant U as User or Script
    participant T as MythosTokenizer
    participant P as Prelude
    participant R as Recurrent Block
    participant H as ACT Halting
    participant C as Coda
    participant O as Output

    rect rgb(234,244,255)
        U->>T: Encode prompt or training text
        T->>P: Input IDs and positions
    end

    rect rgb(238,248,238)
        P->>R: Produce initial hidden state
        loop Up to max_loop_iters
            R->>H: Update hidden state with injected input
            H-->>R: Continue if threshold not met
        end
        H->>C: Halt and finalize
    end

    rect rgb(255,247,232)
        C->>O: Emit logits
        O-->>U: Decode text or compute loss
    end
```

## Getting Started

### Installation

Install from PyPI:

```bash
pip install open-mythos
```

Install with Flash Attention 2 support:

```bash
pip install open-mythos[flash]
```

Install local training extras for this repository:

```bash
pip install -r training/requirements.txt
```

### Quick Start

Create a small research-scale model and run a forward pass:

```python
import torch
from open_mythos import MythosConfig, OpenMythos

cfg = MythosConfig(
    vocab_size=32000,
    dim=256,
    n_heads=8,
    n_kv_heads=2,
    max_seq_len=128,
    max_loop_iters=4,
    prelude_layers=1,
    coda_layers=1,
    attn_type="gqa",
    n_experts=8,
    n_shared_experts=1,
    n_experts_per_tok=2,
    expert_dim=64,
    lora_rank=8,
)

model = OpenMythos(cfg)
input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
logits = model(input_ids, n_loops=4)
print(logits.shape)
```

Start the interactive example chat:

```bash
python examples/chat_example.py
```

Use a trained OpenMythos checkpoint instead of the default fallback model:

```bash
python examples/chat_example.py --checkpoint checkpoints/step_0005000.pt
```

> **Note:** The tokenizer and default fallback chat model are downloaded from Hugging Face. Setting `HF_TOKEN` is optional but improves rate limits.

## CLI Reference

OpenMythos does not currently ship a packaged CLI binary. The repo is operated through Python scripts.

| Script                                   | Purpose                                                                                                   | Example                                                                    |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `examples/chat_example.py`               | Interactive terminal chat using either a trained OpenMythos checkpoint or a stronger fallback instruct model | `python examples/chat_example.py --checkpoint checkpoints/step_0005000.pt` |
| `examples/chat_example.py --random-init` | Smoke-test the OpenMythos generation path with an untrained tiny config                                   | `python examples/chat_example.py --random-init`                            |
| `examples/variants_example.py`           | Instantiate a predefined model variant and print its parameter count                                      | `python examples/variants_example.py`                                      |
| `examples/moda_example.py`               | Run the alternative MoDA smoke test and verify gradients                                                  | `python examples/moda_example.py`                                          |
| `training/tiny_pretrain.py`              | Train a small checkpoint that can be loaded by the chat example                                           | `python training/tiny_pretrain.py`                                         |
| `training/3b_fine_web_edu.py`            | Run the larger FineWeb-Edu training pipeline                                                              | `torchrun --nproc_per_node=NUM_GPUS training/3b_fine_web_edu.py`           |

## Core Functionality

### Execution Modes

| Mode                   | Trigger                                             | Backing model                         | Best use                                         |
| ---------------------- | --------------------------------------------------- | ------------------------------------- | ------------------------------------------------ |
| OpenMythos checkpoint  | `--checkpoint path/to/step_*.pt`                    | Trained OpenMythos weights            | Real model experiments and checkpoint evaluation |
| HF fallback chat       | Run `examples/chat_example.py` without a checkpoint | `Qwen/Qwen2.5-1.5B-Instruct` | Out-of-the-box interactive demo                  |
| Random-init smoke test | `--random-init`                                     | Tiny untrained OpenMythos config      | Wiring validation only; gibberish is expected    |

### Attention Backends

| Backend | Class         | Strength                                                 | Important config                                                                    |
| ------- | ------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `gqa`   | `GQAttention` | Simpler grouped-query attention with smaller KV cache    | `n_heads`, `n_kv_heads`                                                             |
| `mla`   | `MLAttention` | Compressed KV cache for larger-context research variants | `kv_lora_rank`, `q_lora_rank`, `qk_rope_head_dim`, `qk_nope_head_dim`, `v_head_dim` |

### Recurrent Depth, Halting, and Stability

- ✅ **Prelude + Recurrent Block + Coda:** fixed prelude and coda layers frame a shared recurrent core that is looped up to `max_loop_iters`.
- ✅ **ACT halting:** `ACTHalting` can stop recurrent computation early once the halting threshold is reached.
- ✅ **LTI-style injection:** the recurrent block injects the original encoded input each loop to reduce drift and stabilize deeper recurrence.
- ✅ **LoRA depth adaptation:** lightweight rank-limited adapters allow loop-specific behavior without duplicating entire blocks.

### MoE Routing

- 🚀 `MoEFFN` uses routed experts plus always-on shared experts inside the recurrent block.
- 🚀 `n_experts_per_tok` controls top-k expert activation per token.
- ⚠️ Increasing experts widens model breadth, but it also raises memory pressure and routing complexity.

### Model Variants

Predefined variants are available in [open_mythos/variants.py](open_mythos/variants.py).

| Variant         | Hidden dim | Routed experts | Loop iters | Context   | Max output |
| --------------- | ---------- | -------------- | ---------- | --------- | ---------- |
| `mythos_1b()`   | 2048       | 64             | 16         | 4096      | 4096       |
| `mythos_3b()`   | 3072       | 64             | 16         | 4096      | 4096       |
| `mythos_10b()`  | 4096       | 128            | 24         | 8192      | 4096       |
| `mythos_50b()`  | 6144       | 256            | 32         | 8192      | 4096       |
| `mythos_100b()` | 8192       | 256            | 32         | 1,000,000 | 131,072    |
| `mythos_500b()` | 12288      | 512            | 48         | 1,000,000 | 131,072    |
| `mythos_1t()`   | 16384      | 512            | 64         | 1,000,000 | 131,072    |

### Training Workflow

| Workflow                 | Script                        | Notes                                                                           |
| ------------------------ | ----------------------------- | ------------------------------------------------------------------------------- |
| Tiny pretraining         | `training/tiny_pretrain.py`   | Single-GPU sized run that produces `step_*.pt` checkpoints for the chat example |
| Larger-scale pretraining | `training/3b_fine_web_edu.py` | FineWeb-Edu training pipeline with DDP support                                  |
| Dataset guidance         | `docs/datasets.md`            | Notes on training corpora and token budgets                                     |

## Configuration

OpenMythos is code-configured through `MythosConfig`; there is no YAML or JSON runtime config file in this repository. The defaults below are defined in [open_mythos/main.py](open_mythos/main.py).

### How to Read `MythosConfig`

`MythosConfig` controls four things:

- model width and context size
- which attention backend is used
- how large and sparse the MoE block is
- how much recurrent computation the model can do

As a rule:

- bigger `dim`, more heads, or more experts usually means more capacity and more memory use
- bigger `max_seq_len` or `max_output_tokens` increases sequence handling limits, but also increases memory pressure
- `attn_type="gqa"` and `attn_type="mla"` use different subsets of fields
- `dropout` mainly affects training; it is typically disabled for inference

### Core Model Fields

| Field | Default | What it controls | Practical effect |
|---|---|---|---|
| `vocab_size` | `32000` | Size of the token embedding table and output logits | Increase it only if your tokenizer needs more token IDs |
| `dim` | `2048` | Width of the residual stream across the whole model | Larger `dim` increases model capacity, activation size, and parameter count everywhere |
| `n_heads` | `16` | Number of query heads in attention | More heads can improve representation diversity, but `dim` must still split cleanly across heads |
| `n_kv_heads` | `4` | Number of key/value heads for GQA | Lower than `n_heads` reduces KV-cache memory; only meaningful for `attn_type="gqa"` |
| `max_seq_len` | `4096` | Maximum supported sequence length | Larger values allow longer prompts and training sequences, but require larger RoPE tables and more memory |
| `max_loop_iters` | `16` | Upper bound on recurrent loop depth | Higher values let the recurrent block think for more steps, but increase compute |
| `prelude_layers` | `2` | Dense transformer layers before recurrence starts | More prelude layers give stronger upfront feature extraction before looping |
| `coda_layers` | `2` | Dense transformer layers after recurrence ends | More coda layers give stronger post-loop refinement before logits |
| `max_output_tokens` | `4096` | Intended generation ceiling for the model config | Useful as a model-side limit for long generations; separate from prompt length |
| `dropout` | `0.0` | Dropout used in attention and residual paths | Usually `0.0` for inference and a nonzero value for training regularization |

### Attention Fields

| Field | Default | What it controls | Practical effect |
|---|---|---|---|
| `attn_type` | `"mla"` | Chooses `MLAttention` or `GQAttention` | `mla` uses compressed latent KV structure; `gqa` uses grouped-query attention |
| `kv_lora_rank` | `512` | Size of the compressed KV latent in MLA | Higher rank preserves more information but increases MLA memory and compute |
| `q_lora_rank` | `1536` | Size of the compressed query latent in MLA | Higher rank increases query capacity for MLA but costs more parameters |
| `qk_rope_head_dim` | `64` | Per-head dimensions that receive RoPE in MLA | Larger values allocate more of each head to positional structure |
| `qk_nope_head_dim` | `128` | Per-head dimensions that do not receive RoPE in MLA | Larger values allocate more of each head to non-positional content |
| `v_head_dim` | `128` | Value width per head in MLA | Larger values widen attention outputs before projection back to `dim` |
| `rope_theta` | `500000.0` | Base frequency used for RoPE | Larger values stretch positional frequencies and are commonly used for long contexts |

### MoE and Control Fields

| Field | Default | What it controls | Practical effect |
|---|---|---|---|
| `n_experts` | `64` | Number of routed experts in the recurrent MoE FFN | More experts increase specialization and total parameter count |
| `n_shared_experts` | `2` | Number of experts that are always active | Shared experts carry common knowledge that should not depend on routing |
| `n_experts_per_tok` | `4` | Top-k routed experts selected per token | Higher values activate more compute per token and make routing less sparse |
| `expert_dim` | `512` | Hidden width inside each expert MLP | Larger expert width makes each expert stronger but more expensive |
| `act_threshold` | `0.99` | Halting threshold for recurrent accumulation | Higher values encourage more loop steps before halting |
| `lora_rank` | `16` | Rank of the per-loop LoRA adapter | Higher rank allows more loop-specific adaptation but adds parameters |

### Important Parameter Interactions

- `dim` should divide cleanly by `n_heads`, because GQA uses `head_dim = dim // n_heads`.
- For GQA, `n_heads` should divide cleanly by `n_kv_heads`; fewer KV heads means a smaller KV cache.
- For MLA, `n_kv_heads` is effectively irrelevant, while `kv_lora_rank`, `q_lora_rank`, `qk_rope_head_dim`, `qk_nope_head_dim`, and `v_head_dim` matter a lot.
- `n_experts_per_tok` should never exceed `n_experts`.
- `max_seq_len` limits the model's token window, while `max_output_tokens` is about how much text generation you want to allow.
- `max_loop_iters` and `act_threshold` work together: the first is the hard cap on recurrent depth, and the second affects whether the model stops early.

### Minimal Example

```python
from open_mythos import MythosConfig

cfg = MythosConfig(
    vocab_size=32000,       # tokenizer size
    dim=1024,               # model width
    n_heads=8,              # attention heads
    n_kv_heads=2,           # GQA KV heads
    max_seq_len=2048,       # context length
    max_loop_iters=8,       # recurrent depth cap
    prelude_layers=1,       # layers before the loop
    coda_layers=1,          # layers after the loop
    attn_type="gqa",        # or "mla"
    n_experts=16,           # routed experts
    n_shared_experts=1,     # always-on experts
    n_experts_per_tok=2,    # top-k routing
    expert_dim=256,         # expert hidden size
    act_threshold=0.99,     # halting threshold
    lora_rank=8,            # loop-specific low-rank adaptation
    dropout=0.1,            # training regularization
)
```

## Version History

| Version | Date         | Changes                                                                                                                            |
| ------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `0.5.0` | `2026-04-25` | Current package version in `pyproject.toml`; README reorganized around architecture, scripts, configuration, and interactive usage |
| `0.4.0` | `2026-04-20` | ACT halting, MoE router bias, KV-cache consistency, LoRA loop-depth fixes, and training logging improvements                       |

## License and Maintenance

- ✅ **License:** MIT. See [LICENSE](LICENSE).
- ✅ **Maintainer:** Kye Gomez
- ✅ **Contact:** `kye@swarms.world`
- ✅ **Repository:** `https://github.com/The-Swarm-Corporation/OpenMythos`
- ✅ **Documentation:** [docs/open_mythos.md](docs/open_mythos.md) and [docs/datasets.md](docs/datasets.md)
- ✅ **Last updated:** `2026-04-25`

## Citation

If you build on OpenMythos in a paper or research note, cite:

```bibtex
@software{gomez2026openmythos,
  author    = {Kye Gomez},
  title     = {OpenMythos: A Theoretical Reconstruction of the Claude Mythos Architecture},
  year      = {2026},
  url       = {https://github.com/The-Swarm-Corporation/OpenMythos},
  note      = {Recurrent-Depth Transformer with MoE, MLA, LTI-stable injection, and ACT halting}
}
```
