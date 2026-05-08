# PhysicsLM — ICML 2026 Paper Repository

**Title:** PhysicsLM: Autoregressive Language Modeling of 2D Rigid Body Dynamics

## Contents

| File | Description |
|------|-------------|
| `main.tex` | Full ICML 2026 LaTeX source (~8 pages, two-column) |
| `references.bib` | BibTeX entries for all 28 cited works |
| `figures/` | Directory for figure files (PDF/PNG) |
| `sections/` | Optional per-section drafts |
| `tables/` | Optional standalone table files |

## Building

You need a LaTeX distribution with the ICML 2026 style file (`icml2026.sty`).
Download the style package from the ICML 2026 author kit and place `icml2026.sty`
in this directory, then:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or with latexmk:

```bash
latexmk -pdf main.tex
```

## Abstract

We present PhysicsLM, a system that frames 2D rigid body physics simulation as
autoregressive language modeling. Simulation frames are encoded as structured text
strings, and LFM2-350M is fine-tuned via LoRA to predict the next frame
token-by-token. The accompanying PhysicsScenes dataset contains 900K training
scenes across 24 scenario types in six physical categories. PhysicsLM achieves
22.64 px mean position error (~3% of scene diagonal) with 100% parse rate,
stable 50-frame rollouts, and first-of-kind in-browser inference via WebGPU.

## Key Results

- Mean Position Error: **22.64 px** (single-step, full validation set)
- Parse success rate: **100%** (37/37 objects)
- Rollout stability: stable for **50+ frames** (100% of scenes)
- Dataset: **900K scenes**, 30 scenario types, ~582 GB uncompressed
- Browser deployment: ONNX q4f16 via WebGPU (~2x faster than q4)
