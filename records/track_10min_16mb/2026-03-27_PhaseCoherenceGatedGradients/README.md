# Phase Coherence Gated Gradients

Reference scaffold for a `2026-03-27` record-style submission folder.

This snapshot copies the current root `train_gpt.py` and adapts it to a batch-level version of phase-induced coherence-gated gradient descent (PIC-GD) while preserving the existing:

- real-valued transformer architecture
- Muon + Adam optimizer split
- tokenizer-agnostic `val_bpb` evaluation
- int8 + zlib roundtrip export path

## PIC-GD Adaptation

The implementation stays close to the baseline training loop:

- final hidden states are treated as pseudo-complex latents by pairing adjacent channels as `(real, imag)`
- target-token embeddings are paired the same way to provide a reference signal
- a normalized coherence score is computed from the paired latent/reference dot product
- the coherence score is converted into a detached gradient gate

```python
alpha = PICGD_MIN_GATE + (1 - PICGD_MIN_GATE) * sigmoid(PICGD_BETA * coherence)
```

Training backpropagates `loss * alpha`, while validation and final quantized roundtrip evaluation continue to use raw cross-entropy only.

## New Environment Variables

- `PICGD_ENABLED` default `1`
- `PICGD_BETA` default `4.0`
- `PICGD_MIN_GATE` default `0.25`
- `PICGD_EPS` default `1e-6`

Training logs now include:

- `picgd_coherence`
- `picgd_gate`

## Status

This folder is a reference scaffold only.

- no benchmark numbers are claimed here
- `submission.json` is intentionally marked unbenchmarked
- additional train logs and artifact measurements would be needed before treating this as a real leaderboard submission
