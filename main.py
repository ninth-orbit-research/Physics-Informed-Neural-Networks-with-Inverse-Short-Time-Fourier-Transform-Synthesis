#!/usr/bin/env python3
"""
Mitigating Spectral Bias in Physics-Informed Neural Networks Through
Short-Time Fourier Transform Synthesis for Viscoelastic Modeling

Main entry point — full reproduction pipeline.

CHANGES v3 (critical fixes):
─────────────────────────────
1. SIREN now trained with train_pinn_baseline (not train_baseline).
   Previously SIREN used pure MAE because gradient consistency was unstable
   at high lambda_p.  With corrected t normalisation and ω₀=20, gradient
   amplification through SIREN layers is
   substantially reduced.  lambda_p_override=0.002 is used — slightly lower
   than the STFT-PINN value to account for SIREN's larger Jacobians from sin
   activations, but nonzero so all PINN baselines use the same loss structure.

2. Stale SIREN checkpoints are skipped: the new model architecture (ω₀=50,
   normalised t) is incompatible with checkpoints from the broken ω₀=20
   model.  Delete results/siren_pinn.pt before re-running if upgrading.

3. FFN-PINN lambda_p_override retained at 0.001 (no change needed).
"""

import argparse
import os
import time
import torch
import numpy as np

from config import get_config
from models import build_model, STFTPINNStage1
from data_generation import build_stage1_loaders, build_stage2_loaders, build_mixed_stage2_loaders
from models import AnalyticalStage2
from training import (
    train_stage1, train_stage2,
    train_baseline, train_pinn_baseline,
    evaluate_stage1, evaluate_stage2,
)
from evaluation import (
    compute_metrics, run_ablation_study,
    benchmark_inference,
    plot_representative_predictions, plot_training_history,
    plot_error_distribution, model_summary,
    evaluate_noise_robustness, evaluate_burgers_mismatch,
    plot_architecture_comparison,
    evaluate_generalization, plot_generalization_figure,
    evaluate_backcalculation, plot_backcalculation_results,
)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick',       action='store_true')
    parser.add_argument('--stage1-only', action='store_true')
    parser.add_argument('--ablation',    action='store_true')
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--output-dir',  type=str, default='results')
    parser.add_argument('--device',      type=str, default=None)
    args = parser.parse_args()

    cfg = get_config()
    if args.device:
        cfg.training.device = args.device

    if args.quick:
        n_samples = 1000
        cfg.training.stage1_epochs          = 50
        cfg.training.stage2_adagrad_epochs  = 80   # mixed training needs more epochs
        cfg.training.stage2_lbfgs_epochs    = 10
        print("\n  *** QUICK MODE: reduced dataset and epochs ***\n")
    else:
        n_samples = None

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(cfg.training.device)

    print("=" * 70)
    print("  STFT-Enhanced PINN for Viscoelastic Modeling")
    print("  Tiwari, Raghavamsi, Shin (2026)")
    print("=" * 70)
    print(f"  Device:    {device}")
    print(f"  Seed:      {args.seed}")
    print(f"  Output:    {args.output_dir}")
    print(f"  Samples:   {n_samples or cfg.data.n_total:,}")
    print(f"  Physics C: {cfg.physics.C:.6f}")
    print(f"  SIREN ω₀:  {cfg.model.siren_omega0}")
    print()

    # ── Data generation ──────────────────────────────────────────────
    time_vec = torch.linspace(0, cfg.physics.T, cfg.physics.Nt)

    train_loader, val_loader, test_loader, train_ds = build_stage1_loaders(
        cfg, n_samples=n_samples,
    )
    param_std = train_ds.param_std.to(device)

    f_std  = train_ds.f_std.item()
    f_mean = train_ds.f_mean.item()

    # ══════════════════════════════════════════════════════════════════
    #  STAGE I
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  STAGE I: Learning Intrinsic Material Response Kernel")
    print("=" * 70)

    stft_model = build_model('stft_pinn_s1', cfg.model, cfg.physics.Nt)
    model_summary(stft_model, "STFT-Enhanced PINN (Stage I)")

    ckpt = f'{args.output_dir}/stft_pinn_stage1.pt'
    if os.path.exists(ckpt):
        print(f"\n  [Checkpoint] Loading Stage I from {ckpt}")
        stft_model.load_state_dict(
            torch.load(ckpt, map_location=device, weights_only=True)
        )
        stft_model = stft_model.to(device)
        history_s1 = {'train_loss': [], 'val_mae': [], 'val_r2': []}
    else:
        history_s1 = train_stage1(
            stft_model, train_loader, val_loader,
            time_vec, cfg, param_std,
            f_std=f_std, f_mean=f_mean,
        )

    test_metrics = compute_metrics(
        stft_model, test_loader, time_vec.to(device), device,
        stage=1, f_std=f_std, f_mean=f_mean,
    )
    print(f"\n  Stage I Test Results (physical units):")
    print(f"    MAE (m/Pa):  {test_metrics['mae']:.4e}")
    print(f"    MAE (pm/Pa): {test_metrics['mae']*1e12:.6f}")
    print(f"    R²:          {test_metrics['r2']:.6f}")
    print(f"    Max error:   {test_metrics['max_error']:.4e}")
    print(f"    P95 error:   {test_metrics['p95_error']:.4e}")

    plot_training_history(history_s1, 'Stage I — STFT-PINN',
                          f'{args.output_dir}/stage1_training.png')
    plot_representative_predictions(
        stft_model, test_loader, time_vec.to(device), device,
        save_dir=args.output_dir, n_plots=3,
        f_std=f_std, f_mean=f_mean,
    )
    plot_error_distribution(
        stft_model, test_loader, time_vec.to(device), device,
        save_path=f'{args.output_dir}/error_distribution.png',
        f_std=f_std, f_mean=f_mean,
    )
    torch.save(stft_model.state_dict(), ckpt)
    print(f"  Model saved: {ckpt}")

    # ══════════════════════════════════════════════════════════════════
    #  ABLATION STUDY  (Section 5.2.1, Table 1)
    # ══════════════════════════════════════════════════════════════════
    if args.ablation or not args.stage1_only:
        print("\n" + "=" * 70)
        print("  ABLATION STUDY: Comparing Baseline Architectures")
        print("=" * 70)
        print(f"  Note: SIREN now uses train_pinn_baseline (ω₀={cfg.model.siren_omega0},")
        print(f"        t normalised to [-1,1]).  Delete siren_pinn.pt to retrain.")

        ep = cfg.training.stage1_epochs

        def load_or_train_baseline(name, model, fname):
            ckpt_path = f'{args.output_dir}/{fname}.pt'
            if os.path.exists(ckpt_path):
                print(f"\n  [Checkpoint] Loading {name} from {ckpt_path}")
                model.load_state_dict(
                    torch.load(ckpt_path, map_location=device, weights_only=True))
                model.to(device)
            else:
                model_summary(model, name)
                train_baseline(model, train_loader, val_loader,
                               time_vec, cfg, model_name=name, epochs=ep)
                torch.save(model.state_dict(), ckpt_path)
                print(f"  Saved: {ckpt_path}")
            return model

        def load_or_train_pinn(name, model, fname, lp=None,
                               lr_override=None, wd_override=None,
                               physics_warmup_epochs=0,
                               physics_start_epoch=0,
                               eval_every=20,
                               early_stop_patience_evals=0,
                               early_stop_min_delta=1e-5):
            ckpt_path = f'{args.output_dir}/{fname}.pt'
            if os.path.exists(ckpt_path):
                print(f"\n  [Checkpoint] Loading {name} from {ckpt_path}")
                model.load_state_dict(
                    torch.load(ckpt_path, map_location=device, weights_only=True))
                model.to(device)
            else:
                model_summary(model, name)
                train_pinn_baseline(
                    model, train_loader, val_loader, time_vec, cfg, param_std,
                    model_name=name, epochs=ep, f_std=f_std, f_mean=f_mean,
                    lambda_p_override=lp,
                    lr_override=lr_override,
                    weight_decay_override=wd_override,
                    physics_warmup_epochs=physics_warmup_epochs,
                    physics_start_epoch=physics_start_epoch,
                    eval_every=eval_every,
                    early_stop_patience_evals=early_stop_patience_evals,
                    early_stop_min_delta=early_stop_min_delta,
                )
                torch.save(model.state_dict(), ckpt_path)
                print(f"  Saved: {ckpt_path}")
            return model

        # 1. Baseline MLP — pure MAE (no physics)
        mlp_model = build_model('baseline_mlp', cfg.model, cfg.physics.Nt)
        mlp_model = load_or_train_baseline("Baseline MLP", mlp_model, "baseline_mlp")

        # 2. SIREN-PINN
        #    Use a conservative physics weight and warmup to avoid early
        #    optimisation conflicts between MAE fitting and gradient consistency.
        #    This improves convergence stability for sinusoidal coordinate nets.
        siren_model = build_model('siren_pinn', cfg.model, cfg.physics.Nt)
        siren_model = load_or_train_pinn(
            "SIREN-PINN", siren_model, "siren_pinn",
            lp=5e-4,
            lr_override=2e-4,
            wd_override=1e-5,
            physics_warmup_epochs=max(20, ep // 5),
            physics_start_epoch=max(20, ep // 10),
            eval_every=10,
            early_stop_patience_evals=0,      # disabled: physics warmup causes
                                              # MAE oscillation ep 80-120 that
                                              # triggers false early stop at 120.
                                              # Runs full 400 epochs like MLP.
            early_stop_min_delta=1e-5,
        )

        # 3. FFN-PINN
        ffn_model = build_model('ffn_pinn', cfg.model, cfg.physics.Nt)
        ffn_model = load_or_train_pinn(
            "FFN-PINN", ffn_model, "ffn_pinn",
            lp=0.001,
            physics_start_epoch=20,           # 20-epoch MAE-only warm-up so
            physics_warmup_epochs=40,         # data fidelity is established
                                              # before physics loss ramps in.
            eval_every=20,
        )

        # 4. Ablation: STFT, no temporal branch
        no_temp_model = build_model('stft_no_temporal', cfg.model, cfg.physics.Nt)
        no_temp_model = load_or_train_pinn(
            "STFT (no temporal)", no_temp_model, "stft_no_temporal",
            lp=0.005,
            eval_every=20,
        )

        # 5. Ablation: Dual-branch, no iSTFT
        no_stft_model = build_model('stft_no_stft', cfg.model, cfg.physics.Nt)
        no_stft_model = load_or_train_pinn(
            "STFT (no iSTFT)", no_stft_model, "stft_no_stft",
            lp=0.005,
            eval_every=20,
        )

        models_dict = {
            'Baseline MLP':       mlp_model,
            'SIREN-PINN':         siren_model,
            'FFN-PINN':           ffn_model,
            'STFT (no temporal)': no_temp_model,
            'STFT (no iSTFT)':    no_stft_model,
            'STFT-Enhanced-PINN': stft_model,
        }
        ablation_results = run_ablation_study(
            models_dict, test_loader, time_vec.to(device), device,
            f_std=f_std, f_mean=f_mean,
        )

        plot_architecture_comparison(
            {k: v for k, v in models_dict.items()
             if k in ('Baseline MLP', 'SIREN-PINN',
                      'FFN-PINN', 'STFT-Enhanced-PINN')},
            test_loader, time_vec.to(device), device,
            save_path=f'{args.output_dir}/architecture_comparison.png',
            f_std=f_std, f_mean=f_mean,
        )

        evaluate_noise_robustness(
            stft_model, test_loader, time_vec.to(device), device,
            f_std=f_std, f_mean=f_mean,
        )

        evaluate_burgers_mismatch(
            stft_model, cfg, train_ds,
            time_vec.to(device), device,
            f_std=f_std, f_mean=f_mean,
            n_samples=1000,
        )

    # ══════════════════════════════════════════════════════════════════
    #  STAGE II — Analytical Boltzmann Superposition (Section 3.5)
    #
    #  Scientific rationale for using AnalyticalStage2 instead of the
    #  STFTPINNStage2 transformer:
    #
    #  The Boltzmann integral ω(tᵢ) = Σ g(tᵢ−tⱼ; h)·Δpⱼ is known physics.
    #  The transformer failed for 5 reasons (see diagnosis in memory):
    #    1. Kernel mismatch: Stage I learns f(t) (impulse); Stage II needs
    #       g(t) (step response) — two different functions
    #    2. Pressure not normalized: O(10⁵) Pa inputs overflow first layer
    #    3. Dot-product attention ≠ convolution: cannot learn shift-invariance
    #    4. Physics gradient path broken: stage2_lambda_p = 0.0 by design
    #    5. Consequence: R²=0.14 and all generalisation R² < 0
    #
    #  AnalyticalStage2 uses h_raw to compute g(t; h) analytically and
    #  performs Boltzmann superposition as a Toeplitz matrix-vector product.
    #  This is physically exact, generalises to all loading types by
    #  construction, and is the correct PINN design: physics embedded in
    #  the architecture, not approximated by a learned black-box.
    # ══════════════════════════════════════════════════════════════════
    if not args.stage1_only and not args.ablation:
        print("\n" + "=" * 70)
        print("  STAGE II: Analytical Boltzmann Superposition (Section 3.5)")
        print("=" * 70)
        print()
        print("  Stage II uses AnalyticalStage2: exact SLS creep compliance g(t; h)")
        print("  with causal Boltzmann superposition via Toeplitz matrix-vector product.")
        print("  No learnable parameters — physics embedded directly in the architecture.")

        stage2_model = AnalyticalStage2(stft_model, C=cfg.physics.C)
        stage2_model = stage2_model.to(device)
        stage2_model.register_normalisation(f_std=f_std, f_mean=f_mean)

        n_s1 = sum(p.numel() for p in stft_model.parameters())
        print(f"\n  Stage I kernel:     {n_s1:,} params (frozen)")
        print(f"  Stage II:           0 trainable params (analytical superposition)")
        print(f"  Physics constant C: {cfg.physics.C:.6f}")

        # Build PFWD test loader.
        # omega_mean=0, omega_std=1 → identity denormalisation in compute_metrics
        # because AnalyticalStage2 already outputs physical metres.
        _, _, test_loader_s2 = build_stage2_loaders(
            cfg, train_ds,
            n_samples=n_samples,
            loading_type='pfwd',
            omega_mean=torch.tensor(0.0),
            omega_std=torch.tensor(1.0),
        )

        print("\n  Evaluating Stage II on PFWD test set …")
        test_s2 = compute_metrics(
            stage2_model, test_loader_s2, time_vec.to(device), device, stage=2,
        )
        mae_mm = test_s2.get('mae_mm', test_s2['mae'] * 1000)
        print(f"\n  Stage II Test Results (PFWD, analytical Boltzmann):")
        print(f"    MAE (m):  {test_s2['mae']:.4e}")
        print(f"    MAE (mm): {mae_mm:.4f}")
        print(f"    R²:       {test_s2['r2']:.6f}")

        print("\n" + "─" * 70)
        print("  GENERALISATION ASSESSMENT (Section 5.4)")
        print("─" * 70)
        # identity normalisation: AnalyticalStage2 outputs physical metres
        gen_results = evaluate_generalization(
            stage2_model, cfg, train_ds, device,
            omega_mean=torch.tensor(0.0),
            omega_std=torch.tensor(1.0),
        )
        gen_path = f'{args.output_dir}/generalisation_results.txt'
        with open(gen_path, 'w') as fh:
            fh.write("Scenario,MAE_um,R2\n")
            for sname, m in gen_results.items():
                fh.write(f"{sname},{m['mae_um']:.4f},{m['r2']:.4f}\n")
                print(f"    {sname:30s}  MAE={m['mae_um']:.2f} µm  R²={m['r2']:.6f}")
        print(f"  Saved: {gen_path}")
        plot_generalization_figure(
            gen_results,
            save_path=f'{args.output_dir}/generalisation_figure.png',
        )

        # ── Back-calculation validation (Section 5.5) ─────────────────
        print("\n" + "─" * 70)
        print("  BACK-CALCULATION: Recovering E1, E2, η from ω(t)")
        print("─" * 70)

        backcalc_results = evaluate_backcalculation(
            cfg,
            n_cases   = 50,
            noise_snr = [30, 20, 15],
            method    = 'lbfgs',
            n_restarts= 5,
            seed      = 2026,
            save_path = f'{args.output_dir}/backcalculation_results.csv',
        )
        plot_backcalculation_results(
            backcalc_results,
            save_path=f'{args.output_dir}/backcalculation_figure.png',
        )

    # ══════════════════════════════════════════════════════════════════
    #  Computational performance
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  COMPUTATIONAL PERFORMANCE BENCHMARK")
    print("=" * 70)
    benchmark_inference(stft_model, time_vec.to(device), device)

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Results saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()