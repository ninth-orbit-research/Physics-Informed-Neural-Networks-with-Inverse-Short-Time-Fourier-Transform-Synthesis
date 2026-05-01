"""
Training loops implementing the two-stage protocol (Section 4.3–4.4).

Stage I:   Adagrad with cosine annealing — 400 epochs
Stage II:  Adagrad (300 epochs) → L-BFGS refinement (100 epochs)
Baselines: Adam + cosine scheduler (Baseline MLP: no physics loss)
PINN baselines: Adam + cosine scheduler + physics loss (SIREN, FFN)

CHANGES v2:
  - train_stage1: added f_std parameter → passed to Stage1LossEfficient
  - train_pinn_baseline: Adagrad → Adam + cosine scheduler; added f_std param
    Reason: Adagrad is unstable for SIREN/FFN architectures under the physics
    gradient loss. Adam with cosine annealing converges reliably for all arch.
"""

import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from typing import Dict, Optional

from config import Config
from losses import Stage1LossEfficient, Stage2Loss, BaselineLoss


# ═══════════════════════════════════════════════════════════════════════
#  Cosine annealing  (Eq. 40)
# ═══════════════════════════════════════════════════════════════════════

def cosine_lr(epoch: int, lr0: float, lr_min: float, T_max: int) -> float:
    return lr_min + 0.5 * (lr0 - lr_min) * (1.0 + math.cos(math.pi * epoch / T_max))

def set_lr(optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg['lr'] = lr


# ═══════════════════════════════════════════════════════════════════════
#  Quick validation helpers  (normalised-space metrics for training loop)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_stage1(model, loader, time_vec, device):
    model.eval()
    all_pred, all_true = [], []
    for batch in loader:
        h      = batch['params'].to(device)
        f_true = batch['f_true'].to(device)     # normalised
        all_pred.append(model(h, time_vec).cpu())
        all_true.append(f_true.cpu())
    pred = torch.cat(all_pred)
    true = torch.cat(all_true)
    mae  = torch.mean(torch.abs(pred - true)).item()
    ss_res = torch.sum((pred - true) ** 2).item()
    ss_tot = torch.sum((true - true.mean()) ** 2).item()
    return {'mae': mae, 'r2': 1.0 - ss_res / (ss_tot + 1e-10)}


@torch.no_grad()
def evaluate_stage2(model, loader, time_vec, device):
    model.eval()
    all_pred, all_true = [], []\
    
    for batch in loader:
        h = batch['params'].to(device)
        p = batch['pressure'].to(device)
        # If p is (B, Nt) per-sample, pass as-is (mixed loading).
        # If p is (Nt,) shared, leave as-is — model.forward handles both.
        # Do NOT collapse (B, Nt) → (Nt,): that would throw away per-sample info.
        omega_true = batch['omega_true'].to(device)   # normalised
        h_raw = batch.get('params_raw', None)
        if h_raw is not None:
            h_raw = h_raw.to(device)
            try:
                pred = model(h, time_vec, p, h_raw=h_raw)
            except TypeError:
                pred = model(h, time_vec, p)
        else:
            pred = model(h, time_vec, p)
        all_pred.append(pred.cpu())
        all_true.append(omega_true.cpu())
    pred = torch.cat(all_pred)
    true = torch.cat(all_true)
    mae  = torch.mean(torch.abs(pred - true)).item()
    ss_res = torch.sum((pred - true) ** 2).item()
    ss_tot = torch.sum((true - true.mean()) ** 2).item()
    return {'mae': mae, 'r2': 1.0 - ss_res / (ss_tot + 1e-10)}


# ═══════════════════════════════════════════════════════════════════════
#  Stage I  (Section 4.1, 4.3)
# ═══════════════════════════════════════════════════════════════════════

def train_stage1(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    time_vec: torch.Tensor,
    cfg: Config,
    param_std: torch.Tensor,
    f_std: float = 1.0,      # ← REQUIRED: train_ds.f_std.item()
    f_mean: float = 0.0,     # ← REQUIRED: train_ds.f_mean.item()
) -> Dict:
    """
    Train Stage I impulse response kernel.

    f_std and f_mean must be passed so that the ODE residual and gradient
    consistency losses operate in correct physical units.
    """
    tc     = cfg.training
    device = torch.device(tc.device)
    model  = model.to(device)
    tv     = time_vec.to(device)

    criterion = Stage1LossEfficient(
        lambda_p=tc.stage1_lambda_p,
        lambda_ode=tc.stage1_lambda_ode,
        lambda_ic=tc.stage1_lambda_ic,
        C=cfg.physics.C,
        f_std=f_std,
        f_mean=f_mean,
    )
    optimizer = torch.optim.Adagrad(
        model.parameters(), lr=tc.stage1_lr, weight_decay=tc.stage1_lambda_r,
    )
    use_amp = tc.use_mixed_precision and device.type == 'cuda'
    scaler  = GradScaler('cuda', enabled=use_amp)

    history      = {'train_loss': [], 'val_mae': [], 'val_r2': []}
    best_val_mae = float('inf')
    best_state   = None
    t_start      = time.time()

    print(f"\n{'='*60}")
    print(f"  Stage I Training — {tc.stage1_epochs} epochs  |  f_std={f_std:.4e}")
    print(f"  Device: {device}, AMP: {use_amp}")
    print(f"  λ_p={tc.stage1_lambda_p:.4f}  λ_ode={tc.stage1_lambda_ode:.4f}  λ_ic={tc.stage1_lambda_ic:.4f}")
    print(f"{'='*60}")

    for epoch in range(tc.stage1_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        lr = cosine_lr(epoch, tc.stage1_lr, tc.stage1_lr_min, tc.stage1_epochs)
        set_lr(optimizer, lr)

        for batch in train_loader:
            h_norm  = batch['params'].to(device)
            h_raw   = batch['params_raw'].to(device)
            f_true  = batch['f_true'].to(device)
            df_dE1  = batch['df_dE1'].to(device)
            df_dE2  = batch['df_dE2'].to(device)
            df_deta = batch['df_deta'].to(device)

            optimizer.zero_grad()
            with autocast('cuda', enabled=use_amp):
                loss, info = criterion(
                    model, h_norm, h_raw, tv, f_true,
                    df_dE1, df_dE2, df_deta, param_std=param_std,
                )
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip_norm)
                optimizer.step()

            epoch_loss += info['total']
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history['train_loss'].append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            vm = evaluate_stage1(model, val_loader, tv, device)
            history['val_mae'].append(vm['mae'])
            history['val_r2'].append(vm['r2'])
            if vm['mae'] < best_val_mae:
                best_val_mae = vm['mae']
                best_state   = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch+1:4d}/{tc.stage1_epochs} | "
                  f"loss={avg_loss:.6f} | lr={lr:.2e} | "
                  f"val_MAE={vm['mae']:.6f} | val_R²={vm['r2']:.4f} | "
                  f"time={elapsed:.0f}s")

    if best_state:
        model.load_state_dict(best_state)
    total_time = time.time() - t_start
    print(f"\n  Stage I complete. Best val MAE: {best_val_mae:.6f}")
    print(f"  Total training time: {total_time:.1f}s ({total_time/60:.1f} min)")
    return history


# ═══════════════════════════════════════════════════════════════════════
#  Stage II  (Section 4.2, 4.3)
# ═══════════════════════════════════════════════════════════════════════

def train_stage2(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    time_vec: torch.Tensor,
    cfg: Config,
    param_std: torch.Tensor,
) -> Dict:
    """
    Stage II: stress-driven displacement operator.

    FIX v4 (three bugs fixed):
      1. omega_true is now NORMALISED in Stage2Dataset — the model predicts
         O(1) values and loss compares normalised predictions to normalised
         targets.  Physical MAE is recovered in compute_metrics via omega_std.
      2. Optimizer changed from Adagrad → Adam with linear warmup.
         Adagrad's accumulating denominator collapses the effective learning
         rate to near-zero within ~50 epochs for transformer weights.
         Adam with warmup is standard for transformers and recovers stably.
      3. L-BFGS phase retained for final refinement (unchanged).
    """
    tc     = cfg.training
    device = torch.device(tc.device)
    model  = model.to(device)
    tv     = time_vec.to(device)

    # lambda_p=0 disables physics gradient loss — see config.py for rationale
    criterion    = Stage2Loss(lambda_p=tc.stage2_lambda_p)
    history      = {'train_loss': [], 'val_mae': [], 'val_r2': []}
    best_val_mae = float('inf')
    best_state   = None
    t_start      = time.time()

    # Pull omega normalisation stats from dataset for diagnostic reporting
    omega_std_val  = getattr(train_loader.dataset, 'omega_std',
                             torch.tensor(1.0)).item()
    omega_mean_val = getattr(train_loader.dataset, 'omega_mean',
                             torch.tensor(0.0)).item()

    # ── Phase 1: Adam with linear warmup ─────────────────────────────
    # Adam is standard for transformers; warmup prevents early instability
    # from large gradients at random initialisation.
    n_epochs  = tc.stage2_adagrad_epochs   # reuse epoch count from config
    n_warmup  = max(10, n_epochs // 20)    # 5% of epochs as warmup

    print(f"\n{'='*60}")
    print(f"  Stage II Phase 1 — Adam + warmup ({n_epochs} epochs, "
          f"{n_warmup} warmup)")
    print(f"  omega_std={omega_std_val:.4e} m  |  lambda_p={tc.stage2_lambda_p}")
    print(f"  Loss = MAE(omega_norm) only  (physics grad loss disabled)")
    print(f"{'='*60}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=tc.stage2_lr,
        weight_decay=tc.stage2_lambda_r,
        betas=(0.9, 0.98),   # standard transformer betas
        eps=1e-9,
    )
    use_amp = tc.use_mixed_precision and device.type == 'cuda'
    scaler  = GradScaler('cuda', enabled=use_amp)

    def _lr_lambda(epoch):
        if epoch < n_warmup:
            return float(epoch + 1) / float(n_warmup)
        # cosine decay after warmup
        progress = (epoch - n_warmup) / max(1, n_epochs - n_warmup)
        return max(tc.stage2_lr_min / tc.stage2_lr,
                   0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            h_norm     = batch['params'].to(device)
            p          = batch['pressure'].to(device)
            omega_true = batch['omega_true'].to(device)
            dE1        = batch['domega_dE1'].to(device)
            dE2        = batch['domega_dE2'].to(device)
            deta       = batch['domega_deta'].to(device)

            optimizer.zero_grad()
            with autocast('cuda', enabled=use_amp):
                loss, info = criterion(
                    model, h_norm, tv, p, omega_true,
                    dE1, dE2, deta, param_std=param_std,
                )
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip_norm)
                optimizer.step()

            epoch_loss += info['total']
            n_batches  += 1

        current_lr = optimizer.param_groups[0]['lr']  # capture BEFORE step
        scheduler.step() # advance LambdaLR (warmup + cosine)
        avg_loss = epoch_loss / max(n_batches, 1)
        history['train_loss'].append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = current_lr  # LR actually used this epoch
            vm = evaluate_stage2(model, val_loader, tv, device)
            history['val_mae'].append(vm['mae'])
            history['val_r2'].append(vm['r2'])
            if vm['mae'] < best_val_mae:
                best_val_mae = vm['mae']
                best_state   = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch+1:4d}/{n_epochs} | "
                  f"loss={avg_loss:.6f} | lr={lr:.2e} | "
                  f"val_MAE={vm['mae']:.6f} | val_R²={vm['r2']:.4f} | "
                  f"time={elapsed:.0f}s")

    # ── Phase 2: L-BFGS ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Stage II Phase 2 — L-BFGS ({tc.stage2_lbfgs_epochs} epochs)")
    print(f"{'='*60}")

    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=tc.stage2_lbfgs_lr,
        max_iter=tc.stage2_lbfgs_max_iter,
        history_size=tc.stage2_lbfgs_history_size,
        line_search_fn='strong_wolfe',
    )

    for epoch in range(tc.stage2_lbfgs_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            h_norm     = batch['params'].to(device)
            p          = batch['pressure'].to(device)
            omega_true = batch['omega_true'].to(device)
            dE1        = batch['domega_dE1'].to(device)
            dE2        = batch['domega_dE2'].to(device)
            deta       = batch['domega_deta'].to(device)
            def closure():
                optimizer_lbfgs.zero_grad()
                loss, _ = criterion(model, h_norm, tv, p, omega_true,
                                    dE1, dE2, deta, param_std=param_std)
                loss.backward()
                # NO clipping here — L-BFGS needs true gradients for line search
                return loss

            lv = optimizer_lbfgs.step(closure)
            # Clipping after the full step is optional and safe:
            # nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip_norm)
            
            if lv is not None:
                epoch_loss += lv.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history['train_loss'].append(avg_loss)

        if (epoch + 1) % 10 == 0:
            vm = evaluate_stage2(model, val_loader, tv, device)
            history['val_mae'].append(vm['mae'])
            history['val_r2'].append(vm['r2'])
            if vm['mae'] < best_val_mae:
                best_val_mae = vm['mae']
                best_state   = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}
            print(f"  L-BFGS Epoch {epoch+1:4d}/{tc.stage2_lbfgs_epochs} | "
                  f"loss={avg_loss:.6f} | "
                  f"val_MAE={vm['mae']:.6f} | val_R²={vm['r2']:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    total_time = time.time() - t_start
    print(f"\n  Stage II complete. Best val MAE: {best_val_mae:.6f}")
    print(f"  Total training time: {total_time:.1f}s ({total_time/60:.1f} min)")
    return history


# ═══════════════════════════════════════════════════════════════════════
#  Baseline MLP training  (no physics loss)
# ═══════════════════════════════════════════════════════════════════════

def train_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    time_vec: torch.Tensor,
    cfg: Config,
    model_name: str = "baseline",
    epochs: int = 400,
) -> Dict:
    """Adam + cosine scheduler, MAE loss only."""
    tc     = cfg.training
    device = torch.device(tc.device)
    model  = model.to(device)
    tv     = time_vec.to(device)

    criterion = BaselineLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=tc.stage1_lr, weight_decay=tc.stage1_lambda_r,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=tc.stage1_lr_min,
    )

    history      = {'train_loss': [], 'val_mae': [], 'val_r2': []}
    best_val_mae = float('inf')
    best_state   = None
    t_start      = time.time()

    print(f"\n{'='*60}")
    print(f"  Training {model_name} — {epochs} epochs")
    print(f"{'='*60}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            h      = batch['params'].to(device)
            f_true = batch['f_true'].to(device)
            optimizer.zero_grad()
            loss, info = criterion(model(h, tv), f_true)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip_norm)
            optimizer.step()
            epoch_loss += info['total']
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        history['train_loss'].append(avg_loss)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            vm = evaluate_stage1(model, val_loader, tv, device)
            history['val_mae'].append(vm['mae'])
            history['val_r2'].append(vm['r2'])
            if vm['mae'] < best_val_mae:
                best_val_mae = vm['mae']
                best_state   = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}
            elapsed = time.time() - t_start
            print(f"  [{model_name}] Epoch {epoch+1:4d}/{epochs} | "
                  f"loss={avg_loss:.6f} | "
                  f"val_MAE={vm['mae']:.6f} | val_R²={vm['r2']:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    total_time = time.time() - t_start
    print(f"  {model_name} complete. Best val MAE: {best_val_mae:.6f} ({total_time:.0f}s)")
    return history


# ═══════════════════════════════════════════════════════════════════════
#  PINN baseline training  —  SIREN-PINN and FFN-PINN
#
#  Uses Adam (not Adagrad).
#  Adagrad was unstable for SIREN/FFN under the physics gradient loss:
#    - SIREN (sinusoidal activations): Adagrad accumulated squared gradients
#      quickly, causing lr to collapse before the physics loss helped
#    - FFN (large Fourier features at sigma=10): Adagrad + physics loss
#      created catastrophic gradients → R² → -0.45 at epoch 400
#  Adam with cosine decay converges reliably for both architectures.
#  f_std is REQUIRED for correct gradient scaling (see losses.py).
# ═══════════════════════════════════════════════════════════════════════

def train_pinn_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    time_vec: torch.Tensor,
    cfg: Config,
    param_std: torch.Tensor,
    model_name: str = "pinn_baseline",
    epochs: int = 400,
    f_std: float = 1.0,
    f_mean: float = 0.0,
    lambda_p_override: float = None,   # override lambda_p for SIREN/FFN
    lr_override: float = None,         # optional per-model optimizer LR
    weight_decay_override: float = None,  # optional per-model weight decay
    physics_warmup_epochs: int = 0,    # linearly ramp lambda_p over warmup
    physics_start_epoch: int = 0,      # keep lambda_p=0 before this epoch
    eval_every: int = 20,              # validation cadence
    early_stop_patience_evals: int = 0,  # 0 disables early stopping
    early_stop_min_delta: float = 1e-5,  # required MAE improvement
) -> Dict:

    tc     = cfg.training
    device = torch.device(tc.device)
    model  = model.to(device)
    tv     = time_vec.to(device)

    # SIREN/FFN Jacobians are large (SIREN amplified by omega0=20; FFN by
    # random Fourier features). The default lambda_p overwhelms the data
    # term for these architectures. Pass a smaller value via override.
    effective_lambda = (lambda_p_override
                        if lambda_p_override is not None
                        else tc.stage1_lambda_p)
    effective_lr = (lr_override
                    if lr_override is not None
                    else tc.stage1_lr)
    effective_wd = (weight_decay_override
                    if weight_decay_override is not None
                    else tc.stage1_lambda_r)

    # SIREN/FFN baselines use gradient consistency only (no ODE residual).
    # The ODE residual term is the STFT-PINN's new contribution (R2-M4 response)
    # and is not applied to baseline architectures, for two reasons:
    #   1. Scientific: the manuscript describes SIREN-PINN and FFN-PINN as using
    #      gradient consistency only; adding ODE residual would change the
    #      comparison baseline from what is reported.
    #   2. Stability: SIREN's omega0=20 amplifies output gradients ~400x,
    #      causing the ODE residual term to grow unboundedly during training
    #      even with a reduced lambda_p_override.
    criterion = Stage1LossEfficient(
        lambda_p=effective_lambda,
        lambda_ode=0.0,    # ← zero for all baselines; ODE residual is STFT-PINN only
        lambda_ic=0.0,     # ← zero for all baselines
        C=cfg.physics.C,
        f_std=f_std,
        f_mean=f_mean,
    )
    # Adam — stable for all architecture types under physics gradient loss
    optimizer = torch.optim.Adam(
        model.parameters(), lr=effective_lr, weight_decay=effective_wd,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=tc.stage1_lr_min,
    )

    history      = {'train_loss': [], 'val_mae': [], 'val_r2': []}
    best_val_mae = float('inf')
    best_state   = None
    no_improve_evals = 0
    t_start      = time.time()

    print(f"\n{'='*60}")
    print(f"  Training {model_name} (PINN) — {epochs} epochs  |  "
          f"f_std={f_std:.4e}  |  lambda_p={effective_lambda:.4e}")
    if physics_start_epoch and physics_start_epoch > 0:
        print(f"  Physics start epoch: {physics_start_epoch} (MAE-only before this)")
    if physics_warmup_epochs and physics_warmup_epochs > 0:
        print(f"  Physics warmup: {physics_warmup_epochs} epochs (linear ramp)")
    print(f"  Optimizer: Adam(lr={effective_lr:.2e}, wd={effective_wd:.2e})")
    if early_stop_patience_evals and early_stop_patience_evals > 0:
        print(f"  Early stop: patience={early_stop_patience_evals} evals, "
              f"min_delta={early_stop_min_delta:.1e}, eval_every={eval_every}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        # Ramp in gradient consistency to avoid early optimisation conflicts,
        # especially for high-curvature coordinate nets like SIREN.
        if physics_start_epoch and (epoch + 1) <= physics_start_epoch:
            criterion.lambda_p = 0.0
        elif physics_warmup_epochs and physics_warmup_epochs > 0:
            # Ramp starts right after physics_start_epoch.
            warmup_idx = (epoch + 1) - max(physics_start_epoch, 0)
            ramp = min(1.0, float(warmup_idx) / float(physics_warmup_epochs))
            criterion.lambda_p = effective_lambda * max(0.0, ramp)
        else:
            criterion.lambda_p = effective_lambda

        for batch in train_loader:
            h_norm  = batch['params'].to(device)
            h_raw   = batch['params_raw'].to(device)
            f_true  = batch['f_true'].to(device)
            df_dE1  = batch['df_dE1'].to(device)
            df_dE2  = batch['df_dE2'].to(device)
            df_deta = batch['df_deta'].to(device)

            optimizer.zero_grad()
            loss, info = criterion(
                model, h_norm, h_raw, tv, f_true,
                df_dE1, df_dE2, df_deta, param_std=param_std,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip_norm)
            optimizer.step()

            epoch_loss += info['total']
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        history['train_loss'].append(avg_loss)

        if (epoch + 1) % eval_every == 0 or epoch == 0:
            vm = evaluate_stage1(model, val_loader, tv, device)
            history['val_mae'].append(vm['mae'])
            history['val_r2'].append(vm['r2'])
            if vm['mae'] < (best_val_mae - early_stop_min_delta):
                best_val_mae = vm['mae']
                best_state   = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}
                no_improve_evals = 0
            else:
                no_improve_evals += 1
            lr_now  = scheduler.get_last_lr()[0]
            elapsed = time.time() - t_start
            print(f"  [{model_name}] Epoch {epoch+1:4d}/{epochs} | "
                  f"loss={avg_loss:.6f} | lr={lr_now:.2e} | "
                  f"lambda_p={criterion.lambda_p:.2e} | "
                  f"val_MAE={vm['mae']:.6f} | val_R²={vm['r2']:.4f}")
            # Apply early stopping only after physics is fully turned on.
            pinn_fully_on_epoch = max(physics_start_epoch, 0) + max(physics_warmup_epochs, 0)
            if (
                early_stop_patience_evals and early_stop_patience_evals > 0
                and (epoch + 1) >= pinn_fully_on_epoch
                and no_improve_evals >= early_stop_patience_evals
            ):
                print(f"  [{model_name}] Early stopping at epoch {epoch+1}: "
                      f"no val MAE improvement > {early_stop_min_delta:.1e} for "
                      f"{no_improve_evals} evals.")
                break

    if best_state:
        model.load_state_dict(best_state)
    total_time = time.time() - t_start
    print(f"  {model_name} complete. Best val MAE: {best_val_mae:.6f} ({total_time:.0f}s)")
    return history