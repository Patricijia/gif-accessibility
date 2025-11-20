# GIF Accessibility Project (TSV version)

This repository contains a pipeline for training a LLaVA-based teacher model on the TGIF dataset
using streaming GIF downloads (no GIFs stored on disk), and a small student model for efficient
GIF captioning suitable for web or edge deployment.

This version expects the original TGIF metadata TSV (e.g., `tgif-v1.0.tsv`) instead of a CSV.

## Structure

- `data/` – metadata and processed artifacts (no raw GIFs).
- `teacher/` – LLaVA streaming dataset + fine-tuning + label generation.
- `student/` – small captioning model and distillation pipeline.
- `azure/` – Azure ML job YAMLs to run everything in the cloud.
- `notebooks/` – exploratory and qualitative analysis notebooks.
