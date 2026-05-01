"""
Physics-informed loss functions for STFT-Enhanced PINN.

Stage I: impulse response kernel learning (Eq. 33)
Stage II: stress-driven displacement operator (Eq. 38)

CHANGES v3:
  - Stage1LossEfficient (v4): gradient consistency now scales each parameter
    component (E1, E2, η) INDEPENDENTLY rather than using a single global
    scale.  Physical reason: |∂f/∂E1| ~ C/(E1+E2)² ~ 1e-18 m/Pa² but
    |∂f/∂η| ~ C·E1·E2·t / ((E1+E2)²·η²) ~ 1e-15 m/Pa·s·Pa⁻¹ — three
    orders of magnitude different.  A single normalisation constant
    suppressed the smaller-magnitude components.  Per-component scaling
    gives equal weight to all three parameter sensitivities in the loss.
  - ODE residual and IC loss: unchanged (already normalised correctly).
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from forward_model import impulse_gradients_torch


# ═══════════════════════════════════════════════════════════════════════
#  Stage I loss  (Eq. 33)
# ═══════════════════════════════════════════════════════════════════════

class Stage1Loss(nn.Module):
    """
    L_Stage_I = MAE(f_NN, f_true) + λ_p1 · ‖∇_h f_NN − ∇_h f_phy‖²₂
    """

    def __init__(self, lambda_p: float = 0.01, C: float = 0.2076):
        super().__init__()
        self.lambda_p = lambda_p
        self.C = C

    def forward(
        self,
        model: nn.Module,
        h_norm: torch.Tensor,
        h_raw: torch.Tensor,
        t: torch.Tensor,
        f_true: torch.Tensor,
        df_dE1_true: torch.Tensor,
        df_dE2_true: torch.Tensor,
        df_deta_true: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        h_norm_grad = h_norm.detach().requires_grad_(True)
        f_pred = model(h_norm_grad, t)
        mae_loss = torch.mean(torch.abs(f_pred - f_true))

        if self.lambda_p > 0:
            f_sum = f_pred.sum()
            grads_norm = torch.autograd.grad(
                f_sum, h_norm_grad, create_graph=True, retain_graph=True,
            )[0]
            grad_phy = torch.stack([df_dE1_true, df_dE2_true, df_deta_true], dim=-1)
            grad_nn_mean = grads_norm.unsqueeze(1)
            grad_phy_mean = grad_phy.mean(dim=1, keepdim=True)
            grad_loss = torch.mean((grad_nn_mean - grad_phy_mean) ** 2)
        else:
            grad_loss = torch.tensor(0.0, device=f_pred.device)

        total_loss = mae_loss + self.lambda_p * grad_loss
        info = {'total': total_loss.item(), 'mae': mae_loss.item(), 'grad': grad_loss.item()}
        return total_loss, info


class Stage1LossEfficient(nn.Module):
    """
    Full physics-informed Stage I loss:

      L = MAE(f_NN, f_true)                                   [data fidelity]
        + λ_p  · Σ_k ‖(∇_h f_NN - ∇_h f_phy)_k‖² / scale_k  [gradient consistency]
        + λ_ode · ‖df_NN/dt + α·f_NN‖²                       [ODE residual — SLS]
        + λ_ic  · |f_NN(0) − C/(E₁+E₂)|²                    [initial condition]

    FIX v3 — gradient consistency scale:
      Each parameter component k is scaled by its OWN mean analytical
      gradient magnitude rather than a single global scale.  This prevents
      components with small physical gradients (like ∂f/∂E1 ~ 1e-18) from
      being swamped by larger ones (like ∂f/∂η ~ 1e-15) in the loss.
    """

    def __init__(
        self,
        lambda_p: float = 0.01,
        lambda_ode: float = 0.005,
        lambda_ic: float = 0.01,
        C: float = 0.2076,
        f_std: float = 1.0,
        f_mean: float = 0.0,
    ):
        super().__init__()
        self.lambda_p   = lambda_p
        self.lambda_ode = lambda_ode
        self.lambda_ic  = lambda_ic
        self.C          = C
        self.f_std      = f_std
        self.f_mean     = f_mean

    def forward(
        self,
        model: nn.Module,
        h_norm: torch.Tensor,
        h_raw: torch.Tensor,
        t: torch.Tensor,
        f_true: torch.Tensor,
        df_dE1_true: torch.Tensor,
        df_dE2_true: torch.Tensor,
        df_deta_true: torch.Tensor,
        param_std: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        h_input = h_norm.clone().detach().requires_grad_(True)
        f_pred = model(h_input, t)  # (B, Nt) — normalised space

        B, Nt = f_pred.shape

        # ── 1. MAE data fidelity ─────────────────────────────────────
        mae_loss = torch.mean(torch.abs(f_pred - f_true))

        # ── 2. Gradient consistency (Eqs. 34-36) ─────────────────────
        # FIX v3: per-component normalisation scale
        if self.lambda_p > 0:
            df_dh_norm = torch.autograd.grad(
                outputs=f_pred.sum(),
                inputs=h_input,
                create_graph=True,
                retain_graph=True,
            )[0]  # (B, 3)

            param_std_dev = param_std.to(h_input.device)
            # Convert from normalised to physical gradient:
            # ∂f_phys/∂h_phys = ∂f_norm/∂h_norm · f_std / param_std
            df_dh_raw = df_dh_norm / (param_std_dev.unsqueeze(0) + 1e-10) * self.f_std

            # Mean over time axis (both NN and analytical)
            # df_dh_raw contains Σ_t gradients; divide by Nt → mean
            df_dh_avg = df_dh_raw / Nt                        # (B, 3)

            grad_phy_E1  = df_dE1_true.mean(dim=1)            # (B,)
            grad_phy_E2  = df_dE2_true.mean(dim=1)
            grad_phy_eta = df_deta_true.mean(dim=1)
            grad_phy = torch.stack([grad_phy_E1, grad_phy_E2, grad_phy_eta], dim=-1)  # (B, 3)

            # FIX v4: SYMMETRIC per-component scaling.
            # Old v3 used only physics gradient magnitude as denominator.
            # Problem: for SIREN (ω₀=50) the neural Jacobian is ~50× larger,
            # making (grad_nn - grad_phy)² / |grad_phy|² ≈ 49² ≈ 2401 →
            # physics loss explodes, overwhelming MAE.
            # Fix: scale by the AVERAGE of neural and physics magnitudes so
            # the normalised loss stays O(1) regardless of architecture.
            # At convergence (grad_nn ≈ grad_phy) both approaches agree.
            scale = (grad_phy.detach().abs() + df_dh_avg.detach().abs()).mean(
                dim=0, keepdim=True) / 2.0 + 1e-30                            # (1, 3)
            grad_loss = torch.mean(((df_dh_avg - grad_phy) / scale) ** 2)
        else:
            grad_loss = torch.tensor(0.0, device=f_pred.device)

        # ── 3. ODE residual: df/dt + α·f = 0  (SLS constitutive law) ─
        if self.lambda_ode > 0 and Nt >= 3:
            dt = t[1] - t[0]

            dfdt_norm = (f_pred[:, 2:] - f_pred[:, :-2]) / (2.0 * dt)   # (B, Nt-2)
            f_int     = f_pred[:, 1:-1]                                   # (B, Nt-2)

            E1_r  = h_raw[:, 0]
            E2_r  = h_raw[:, 1]
            eta_r = h_raw[:, 2]
            alpha = (E1_r * E2_r) / ((E1_r + E2_r) * eta_r)              # (B,)

            bias     = self.f_mean / (self.f_std + 1e-30)
            residual = (dfdt_norm / (alpha.unsqueeze(1) + 1e-10)
            + f_int
            - bias)  # (B, Nt-2)  FIX: bias is subtracted (mean-shift correction)

            ode_loss = torch.mean(residual ** 2)
        else:
            ode_loss = torch.tensor(0.0, device=f_pred.device)

        # ── 4. Initial condition: f(0) = C/(E₁+E₂) ──────────────────
        if self.lambda_ic > 0:
            E1_r = h_raw[:, 0]
            E2_r = h_raw[:, 1]
            f0_phys_true = self.C / (E1_r + E2_r)
            f0_norm_true = (f0_phys_true - self.f_mean) / (self.f_std + 1e-30)
            ic_loss = torch.mean((f_pred[:, 0] - f0_norm_true) ** 2)
        else:
            ic_loss = torch.tensor(0.0, device=f_pred.device)

        # ── Total ─────────────────────────────────────────────────────
        total_loss = (mae_loss
                      + self.lambda_p   * grad_loss
                      + self.lambda_ode * ode_loss
                      + self.lambda_ic  * ic_loss)

        info = {
            'total': total_loss.item(),
            'mae':   mae_loss.item(),
            'grad':  grad_loss.item(),
            'ode':   ode_loss.item(),
            'ic':    ic_loss.item(),
        }
        return total_loss, info


# ═══════════════════════════════════════════════════════════════════════
#  Stage II loss  (Eq. 38)
# ═══════════════════════════════════════════════════════════════════════

class Stage2Loss(nn.Module):
    """
    L_Stage_II = MAE(ω_NN, ω_true) + λ_p2 · ‖∇_h ω_NN − ∇_h ω_phy‖²₂

    FIX v3: per-component gradient scaling (same logic as Stage1LossEfficient).
    """

    def __init__(self, lambda_p: float = 0.015):
        super().__init__()
        self.lambda_p = lambda_p

    def forward(
        self,
        model: nn.Module,
        h_norm: torch.Tensor,
        t: torch.Tensor,
        p: torch.Tensor,
        omega_true: torch.Tensor,
        domega_dE1_true: torch.Tensor,
        domega_dE2_true: torch.Tensor,
        domega_deta_true: torch.Tensor,
        param_std: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        h_input = h_norm.clone().detach().requires_grad_(True)
        omega_pred = model(h_input, t, p)

        mae_loss = torch.mean(torch.abs(omega_pred - omega_true))

        if self.lambda_p > 0 and param_std is not None:
            B, Nt = omega_pred.shape

            # Stage II gradient consistency: differentiate omega_pred w.r.t. h_input.
            # Stage I is frozen (no_grad + no requires_grad inside STFTPINNStage2),
            # so the path from omega_pred to h_input may be broken.
            # allow_unused=True returns None instead of crashing; we fall back to
            # MAE-only loss for this batch rather than halting training entirely.
            raw = torch.autograd.grad(
                outputs=omega_pred.sum(),
                inputs=h_input,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]

            if raw is None:
                # Gradient path broken (Stage I frozen) — skip gradient term.
                grad_loss  = torch.tensor(0.0, device=omega_pred.device)
                total_loss = mae_loss
                return total_loss, {
                    'total': total_loss.item(), 'mae': mae_loss.item(), 'grad': 0.0
                }

            domega_dh_norm = raw

            pstd = param_std.to(h_input.device)
            domega_dh_raw = domega_dh_norm / (pstd.unsqueeze(0) + 1e-10)

            grad_phy = torch.stack([
                domega_dE1_true.mean(dim=1),
                domega_dE2_true.mean(dim=1),
                domega_deta_true.mean(dim=1),
            ], dim=-1)

            domega_avg = domega_dh_raw / Nt
            # FIX v4: symmetric scaling (see Stage1LossEfficient comment above)
            scale = (grad_phy.detach().abs() + domega_avg.detach().abs()).mean(
                dim=0, keepdim=True) / 2.0 + 1e-30
            grad_loss = torch.mean(((domega_avg - grad_phy) / scale) ** 2)
        else:
            grad_loss = torch.tensor(0.0, device=omega_pred.device)

        total_loss = mae_loss + self.lambda_p * grad_loss
        info = {'total': total_loss.item(), 'mae': mae_loss.item(), 'grad': grad_loss.item()}
        return total_loss, info


# ═══════════════════════════════════════════════════════════════════════
#  Baseline loss (no physics)
# ═══════════════════════════════════════════════════════════════════════

class BaselineLoss(nn.Module):
    """Simple MAE loss without physics constraints."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss = torch.mean(torch.abs(pred - target))
        return loss, {'total': loss.item(), 'mae': loss.item(), 'grad': 0.0}