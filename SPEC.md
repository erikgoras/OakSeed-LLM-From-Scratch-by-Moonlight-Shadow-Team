# SPEC: OakSeed LLM From Scratch (by Moonlight Shadow Team)

## 1. Guiding Principles
- **First Principles:** Build a decoder-only Transformer (nanoGPT-style) from scratch to understand every matrix, tensor shape, and gradient step.
- **Simple Systems Scale:** Build strictly **one module at a time**. Do not proceed to the next module until the current module passes all unit tests.
- **Minimal Dependencies:**
  - Module 1: **Pure Python standard library only** (`pathlib`, `json`, `unittest`, etc.). Zero external packages.
  - Modules 2–4: PyTorch (`torch`) only (leveraging Apple Silicon `mps` when available, with `cpu` fallback).
- **Clear Contracts:** Every module exposes clean, typed Python interfaces and explicit shape assertions.

---

## 2. Repository Structure
```text
OakSeed-LLM-From-Scratch-by-Moonlight-Shadow-Team/
├── README.md
├── SPEC.md
├── data/
│   └── input.txt          # Training corpus (e.g., Tiny Shakespeare or domain text)
├── src/
│   ├── __init__.py
│   ├── tokenizer.py       # Module 1: Character-level tokenizer + dataset split
│   ├── attention.py       # Module 2: Causal Self-Attention & Multi-Head Attention
│   ├── model.py           # Module 3: Transformer Block & GPT Language Model
│   ├── train.py           # Module 4: Training loop, validation loss & checkpointing
│   └── generate.py        # Module 5: Autoregressive text generation (temperature / top-k)
└── tests/
    └── test_tokenizer.py  # Unit tests for Module 1
