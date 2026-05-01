# STFT-Enhanced Physics-Informed Neural Network for Viscoelastic Modeling

**Mitigating Spectral Bias in Physics-Informed Neural Networks Through Short-Time Fourier Transform Synthesis for Viscoelastic Modeling**

*Nitin Tiwari, R.V.B.S. Raghavamsi, Boonam Shin*
GeoCARE Laboratory, School of Civil Environmental and Infrastructure Engineering,
Southern Illinois University Carbondale

*Submitted to Computers and Geotechnics*

---

## Overview

This repository implements the spectral-temporal synthesis framework for physics-informed learning of viscoelastic operators. The core motivation is **spectral bias** — the well-documented tendency of neural networks to learn low-frequency components of a target function much faster than high-frequency ones. For viscoelastic pavement response under Portable Falling Weight Deflectometer (PFWD) loading, this bias prevents standard PINNs from accurately capturing rapid elastic transients alongside slow viscous relaxation.

The proposed fix is architectural: instead of learning the time-domain response directly, the network learns a **complex-valued latent field structured like an STFT spectrogram**, then synthesizes the time-domain signal via differentiable inverse STFT. This gives the optimizer direct gradient access to both low- and high-frequency components simultaneously.

---

## Physical Model

The material is represented as a **Standard Linear Solid (SLS)**, also known as the Zener model: a spring $E_2$ in parallel with a Maxwell unit (spring $E_1$ in series with dashpot $\eta$).

**Stage I target — impulse response kernel:**

$$f(t;\, E_1, E_2, \eta) = \frac{C}{E_1 + E_2} \exp(-\alpha t)$$

**Stage II target — displacement under arbitrary loading (Boltzmann superposition):**

$$\omega(t_i) = g(t_i)\,p_0 + \sum_{j=1}^{n-1} g(t_i - \tau_j)\, H(t_i - \tau_j)\, \Delta p_j$$

where $g(t)$ is the **SLS step response (creep compliance)**:

$$g(t;\, E_1, E_2, \eta) = C\left[\frac{1}{E_1+E_2} + \frac{E_1}{E_2(E_1+E_2)}\left(1 - e^{-\alpha t}\right)\right]$$

Note: Stage I uses the impulse response $f(t)$ as the direct training target. Stage II uses the step response $g(t)$ as the Duhamel convolution kernel. These are related by $f(t) = dg/dt$.

**Geometric coefficient and decay rate:**

$$C = \frac{\delta(1-\mu^2)\pi}{2}, \qquad \alpha = \frac{E_1 E_2}{(E_1+E_2)\eta}$$

**Parameter ranges (training distribution):**

| Parameter | Range |
|-----------|-------|
| $E_1$ | 400 – 900 MPa |
| $E_2$ | 40 – 80 MPa |
| $\eta$ | 400 – 800 kPa·s |
| Plate radius $\delta$ | 150 mm (fixed) |
| Poisson's ratio $\mu$ | 0.35 (fixed) |

---

## Architecture

### Stage I — Impulse Response Kernel (STFTPINNStage1)

```
h = [E₁, E₂, η]  →  param_encoder  →  complex latent S_h(τ, ω) ∈ C^{16×32}
                                              ↓  inverse STFT
                                         f̃(t; h)   [STFT synthesis branch]
                                              ↓  post-MLP  →  z_h
t  →  TemporalEncoder (GELU MLP)  →  z_t
                                         z_h ⊙ z_t  →  Decoder  →  f(t; h)
```

The STFT synthesis module learns $N_\tau = 16$ time-shift centres and $N_\omega = 32$ frequency centres, both as trainable parameters. A learnable Gaussian window width controls temporal localization.

### Stage II — Stress-Driven Displacement (STFTPINNStage2)

```
f(·; h)  →  kernel_proj  →  kernel_emb     [frozen Stage I output]
p(t)     →  LoadDistributionEncoder  →  load_emb   [with positional encoding]

transformer_decoder(tgt=kernel_emb, memory=load_emb, causal_mask)
    →  output_proj  →  ω(t)
```

The transformer decoder (3 layers, 4 heads, d=128) implements a learned approximation to the Boltzmann hereditary integral. Causal masking enforces that prediction at time $t_i$ cannot attend to future load increments.

### Baseline Models

| Model | Architecture | Physics loss |
|-------|-------------|--------------|
| `BaselineMLP` | 5-layer ReLU MLP | None |
| `SIRENPINN` | Sinusoidal layers, ω₀=20, t∈[−1,1] | Gradient consistency (λ_p ramped) |
| `FourierFeatureNetworkPINN` | Random Fourier features (σ=1) + MLP | Gradient consistency |
| `STFTPINNStage1` | iSTFT synthesis + dual-branch | Gradient consistency + ODE residual + IC |

---

## Physics-Informed Loss

### Stage I

$$\mathcal{L}_\text{I} = \underbrace{\text{MAE}(f_\text{NN},\, f_\text{true})}_\text{data} + \lambda_p \underbrace{\|\nabla_h f_\text{NN} - \nabla_h f_\text{phy}\|^2}_\text{gradient consistency} + \lambda_\text{ode} \underbrace{\left\|\frac{1}{\alpha}\frac{df}{dt} + f + \frac{\mu}{\sigma}\right\|^2}_\text{ODE residual} + \lambda_\text{ic} \underbrace{|f(0) - \tfrac{C}{E_1+E_2}|^2}_\text{initial condition}$$

Default weights: λ_p=0.01, λ_ode=0.01, λ_ic=0.005. The ODE residual and IC terms are applied to STFT-PINN only, not to SIREN or FFN baselines.

### Stage II

$$\mathcal{L}_\text{II} = \text{MAE}(\omega_\text{NN},\, \omega_\text{true}) + \lambda_p \|\nabla_h \omega_\text{NN} - \nabla_h \omega_\text{phy}\|^2$$

Analytical gradients $\nabla_h \omega_\text{phy}$ use the step response sensitivities $\partial g/\partial h_k$ (see Appendix B of the manuscript).

---

## Project Structure

```
stft_pinn/
├── config.py            # All hyperparameters and dataclasses
├── forward_model.py     # SLS analytical model
│   ├── impulse_response_np/torch      # f(t; h) — Stage I target
│   ├── impulse_gradients_np/torch     # ∂f/∂h — Stage I physics loss
│   ├── sls_step_response_np/torch     # g(t; h) — Stage II kernel (corrected)
│   ├── sls_step_gradients_np/torch    # ∂g/∂h — Stage II physics loss (corrected)
│   ├── displacement_superposition_*   # Boltzmann superposition (uses g, not f)
│   └── generate_pfwd_pulse / *_loading  # Loading profiles
├── data_generation.py   # Latin hypercube sampling + PyTorch datasets
├── models.py            # All neural architectures
│   ├── STFTPINNStage1   # Proposed model
│   ├── STFTPINNStage2   # Transformer decoder
│   ├── BaselineMLP
│   ├── SIRENPINN        # t normalised to [-1,1], ω₀=20
│   ├── FourierFeatureNetworkPINN
│   ├── STFTPINNNoTemporal  # Ablation: no temporal branch
│   └── STFTPINNNoSTFT      # Ablation: no iSTFT
├── losses.py            # Stage1LossEfficient, Stage2Loss, BaselineLoss
├── training.py          # train_stage1, train_stage2, train_baseline,
│                        # train_pinn_baseline (with physics warmup schedule)
├── evaluation.py        # compute_metrics, run_ablation_study,
│                        # evaluate_noise_robustness, evaluate_burgers_mismatch,
│                        # evaluate_generalization, benchmark_inference,
│                        # plot_* functions
├── main.py              # CLI entry point
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Python 3.9+ required
pip install -r requirements.txt
```

**requirements.txt:**
```
torch>=2.1.0,<2.5.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
```

The code runs on CUDA, Apple MPS (M-series), and CPU. Device is auto-detected; override with `--device cuda` or `--device cpu`.

---

## Usage

### Quick test (~5 minutes, 1000 samples, 50 epochs)

```bash
python main.py --quick
```

### Stage I only — learn impulse response kernel

```bash
python main.py --stage1-only
```

### Ablation study — compare all architectures on Stage I

```bash
python main.py --ablation
```

This trains and evaluates: Baseline MLP, SIREN-PINN, FFN-PINN, STFT (no temporal branch), STFT (no iSTFT), and the full STFT-Enhanced-PINN.

### Full pipeline — Stage I + Stage II + generalization

```bash
python main.py
```

Runs Stage I, then Stage II (PFWD loading), then evaluates generalization across 6 loading scenarios and benchmarks inference speed.

### All options

```
--quick          Reduced dataset (1000 samples) and epochs (50/30/10)
--stage1-only    Train and evaluate Stage I only; skip Stage II
--ablation       Run ablation study after Stage I; skip Stage II
--seed INT       Random seed (default: 42)
--output-dir DIR Output directory for checkpoints and plots (default: results/)
--device STR     Force device: cuda / mps / cpu
```

---

## Outputs

All outputs are saved to `results/` (or the directory specified by `--output-dir`):

| File | Description |
|------|-------------|
| `stft_pinn_stage1.pt` | Stage I model checkpoint |
| `stft_pinn_stage2.pt` | Stage II model checkpoint |
| `baseline_mlp.pt` | Baseline MLP checkpoint |
| `siren_pinn.pt` | SIREN checkpoint |
| `ffn_pinn.pt` | FFN-PINN checkpoint |
| `stage1_training.png` | Stage I loss / MAE / R² curves |
| `prediction_sample_{1,2,3}.png` | Representative prediction plots |
| `error_distribution.png` | Histogram of prediction errors |
| `architecture_comparison.png` | 4-panel spectral bias comparison |
| `generalisation_results.txt` | MAE and R² across 6 loading scenarios |

Checkpoints are loaded automatically on re-run — delete a `.pt` file to force retraining of that model.

---

## Key Implementation Notes

### SIREN time normalization (critical fix)
SIREN requires `t` normalized to `[-1, 1]` before the first sinusoidal layer. Without this, SIREN's first-layer phase coverage is only ~1 rad (11% of the ~9 rad needed to represent the target decay), causing complete training failure (R²≈−0.12). The fix is in `SIRENPINN.forward()`:
```python
t_norm = (t / _T_DURATION) * 2.0 - 1.0   # maps [0, 0.200] → [-1, 1]
```

### SIREN physics warmup schedule
SIREN uses a three-phase training schedule to prevent gradient consistency loss from overwhelming MAE during early training:
1. **Epochs 1–40**: λ_p = 0 (pure MAE warm-up)
2. **Epochs 41–120**: λ_p ramped linearly from 0 → 5×10⁻⁴
3. **Epochs 121–400**: λ_p = 5×10⁻⁴ (held constant)

### Corrected Stage II superposition kernel
The Boltzmann superposition in Stage II uses the **step response** `g(t)`, not the impulse response `f(t)`. Using `f(t)` as the kernel predicts 73% negative displacements and fails the physical limit test (η→0 should give C·p/E₂; the incorrect kernel gives 0). This is corrected in `forward_model.py` v3 via `sls_step_response_np/torch`.

### Gradient consistency scaling (v5, element-wise)
The physics gradient loss scales each `(batch, parameter)` element individually:
```python
scale = (|grad_phy| + |grad_nn|) / 2   # shape (B, 3)
loss  = mean(((grad_nn - grad_phy) / scale)²)   # bounded ≤ 4.0 always
```
This prevents outlier samples from dominating the batch mean, which caused loss explosion at ~3000× normal magnitude with earlier scalar scaling.

---

## Results

### Stage I ablation (impulse response prediction)

| Method | MAE (standardized) | Notes |
|--------|-------------------|-------|
| Baseline MLP | (7.0 ± 0.3) × 10⁻⁷ | No physics constraints |
| SIREN | (2.2 ± 0.2) × 10⁻⁷ | Physics warmup schedule |
| FFN-PINN | (1.3 ± 0.1) × 10⁻⁷ | σ=1.0, λ_p=0.001 |
| **STFT-Enhanced-PINN** | **(8.8 ± 0.4) × 10⁻⁸** | λ_p=0.01, ODE+IC loss |

### Stage II generalization (displacement under loading)

| Loading scenario | MAE (mm) | R² |
|-----------------|----------|----|
| Standard PFWD | 0.240 | 0.992 |
| Extended PFWD | 0.258 | 0.989 |
| Sinusoidal 0.5 Hz | 0.267 | 0.987 |
| Sinusoidal 2 Hz | 0.245 | 0.991 |
| Multi-stage | 0.273 | 0.985 |
| Random stress | 0.296 | 0.978 |

### Computational performance

| Method | Time per eval | Hardware |
|--------|--------------|----------|
| COMSOL FEM | ~5.2 s | CPU |
| ABAQUS FEM | ~4.7 s | CPU |
| STFT-PINN (Stage II) | ~50 ms | GPU |
| STFT-PINN (Stage II) | ~200 ms | CPU |

---

## Reproducing the Paper

The full pipeline from scratch:

```bash
# 1. Generate data, train Stage I (STFT-PINN)
python main.py --stage1-only

# 2. Run ablation (trains MLP, SIREN, FFN-PINN for Table 1)
python main.py --ablation

# 3. Run full pipeline for Stage II results and Table 2
python main.py

# 4. For multi-seed robustness (5 seeds):
for seed in 42 43 44 45 46; do
    python main.py --seed $seed --output-dir results_seed_$seed
done
```

Checkpoints are saved after each model completes. Re-running loads existing checkpoints automatically — delete `.pt` files selectively to retrain specific models.

---

## Citation

```bibtex
@article{tiwari2026stft,
  title   = {Mitigating Spectral Bias in Physics-Informed Neural Networks
             Through Short-Time Fourier Transform Synthesis for Viscoelastic Modeling},
  author  = {Tiwari, Nitin and Raghavamsi, R.V.B.S. and Shin, Boonam},
  journal = {Computers and Geotechnics},
  year    = {2026},
  note    = {Under review}
}
```

---

## Contact

Nitin Tiwari — nitin.tiwari@siu.edu
GeoCARE Laboratory, Southern Illinois University Carbondale, IL 62901, USA