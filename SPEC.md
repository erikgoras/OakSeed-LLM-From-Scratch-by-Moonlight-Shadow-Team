# SPEC: OakSeed LLM From Scratch (by Moonlight Shadow Team)

## 1. Guiding Principles
- **First Principles:** Build a decoder-only Transformer (nanoGPT-style) from scratch to understand every matrix, tensor shape, and gradient step.
- **Simple Systems Scale:** Build strictly **one module at a time**[cite: 2]. Do not proceed to the next module until the current module passes all unit tests.
- **Minimal Dependencies:**
  - Module 1: **Pure Python standard library only** (`pathlib`, `json`, `unittest`, etc.)[cite: 2]. Zero external packages.
  - Modules 2–4: PyTorch (`torch`) only (leveraging Apple Silicon `mps` when available, with `cpu` fallback).
- **Clear Contracts:** Every module exposes clean, typed Python interfaces and explicit shape assertions.

---

## 2. Repository Structure
```text
OakSeed-LLM-From-Scratch-by-Moonlight-Shadow-Team/
├── README.md
├── SPEC.md
├── data/
│   └── input.txt          # Training corpus (e.g., Tiny Shakespeare or domain t: 2]
├── src/
│   ├── __init__.py
│   ├── tokenizer.py       # Module 1: Character-level tokenizer + dataset s: 2]
│   ├── attention.py       # Module 2: Causal Self-Attention & Multi-Head Atten: 2]
│   ├── model.py           # Module 3: Transformer Block & GPT Language M: 2]
│   ├── train.py           # Module 4: Training loop, validation loss & checkpoin: 2]
│   └── generate.py        # Module 5: Autoregressive text generation (temperature / to: 2]
└── tests/
    └── test_tokenizer.py  # Unit tests for Modul
```

---

## 3. Module 2 Contract: `src/attention.py`
- **Scope:** Scaled dot-product causal self-attention and multi-head causal self-attention (`CausalSelfAttention` and `MultiHeadAttention`).
- **Input tensor:** `x` of shape `(B, T, C)` (`batch_size`, `seq_len`, `embed_dim`), dtype `float32`.
- **Output tensor:** `y` of shape `(B, T, C)` matching input batch and sequence dimensions (`B`, `T`, `C`).
- **Causality:** Future positions (`j > i`) must be masked (`-inf` before softmax) so position `i` only attends to positions `<= i`.
- **Scaling & Dropout:** Attention scores scaled by `1 / sqrt(head_dim)` (`d_k ** -0.5`), with optional dropout during training.
- **Device handling:** Runs on `mps` when available with automatic fallback to `cpu` in tests and runtime.


