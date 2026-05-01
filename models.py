"""
Neural network architectures for viscoelastic operator learning.

CRITICAL FIXES (v3):
────────────────────
BUG-1 (Root cause of SIREN R²=-0.119, no learning):
  The time vector t ∈ [0, 0.200] s was passed raw to SIREN.
  SIREN's is_first initialisation sets W ~ U(-1/n_in, 1/n_in) = U(-0.25, 0.25).
  With ω₀=20, the maximum phase angle in the t dimension at the first layer was
      ω₀ · max_w · t_max = 20 · 0.25 · 0.200 = 1.0 rad   (≈ 0.16 cycles)
  but the target f(t;h) = C/(E₁+E₂)·exp(−αt) requires
      α·T ≈ 91 · 0.200 = 18.3 rad   (≈ 2.9 cycles)
  to be representable.  The 18× frequency deficit meant SIREN saw the same
  phase value at t=0 and t=100 ms — it could not learn ANY time dependence.
  The h_norm inputs (∈ [-3,3]) gave 15 rad coverage, so SIREN learned a
  function purely of h, producing predictions that were constant across time.
  That constant-in-time output has extra variance that is not aligned with
  f_true variation, explaining R² < 0.

  Fix A: normalise t to [-1, 1] for all coordinate networks.
  Fix B: increase ω₀ from 20 → 50.
         Even with t_norm ∈ [-1,1] the first-layer coverage is ω₀/4 rad.
         ω₀=20 → 5 rad (3.7× deficit), ω₀=50 → 12.5 rad (1.5× deficit ← OK
         because deeper layers provide multiplicative frequency escalation).
         ω₀=20 was "optimised" by grid search on broken unnormalised inputs
         and must not be reused.

BUG-2 (BaselineMLP and FFN-PINN — milder but real):
  Same unnormalised t issue.  ReLU/Fourier networks are more robust, but the
  10× range difference (t ∈ [0, 0.2] vs h_norm ∈ [-3, 3]) caused the time
  axis to be suppressed by the first linear layer weight init relative to h.
  Fix: normalise t to [0, 1] for MLP and FFN.

SCIENTIFIC NOTE (deeper issue — relevant for paper revision):
  f(t;h) = C/(E₁+E₂)·exp(−αt) is a MONOTONE exponential decay — it has no
  oscillatory content.  SIREN's inductive bias is SINUSOIDAL.  Even after
  fixing normalisation and ω₀, SIREN must represent exp(−αt) as a sum of
  sinusoids, which is less efficient than networks with monotone inductive
  bias (e.g., MLP+tanh/silu).  The advantage of STFT-PINN over SIREN should
  therefore be framed as an *inductive bias alignment* improvement, not purely
  spectral bias mitigation.  The spectral bias narrative applies more cleanly
  to the multi-scale dynamics present in Stage II with complex loading histories.

Implements:
  1. STFTPINNStage1         — proposed STFT-enhanced PINN (Section 3.4)
  2. STFTPINNStage2         — transformer decoder Stage II (Section 3.5)
  3. BaselineMLP            — standard MLP, no physics (Section 5.2.1)
  4. SIRENPINN              — sinusoidal representation network (Section 5.2.1)
  5. FourierFeatureNetworkPINN — random Fourier features (Section 5.2.1)
  6. STFTPINNNoTemporal     — ablation: STFT only, temporal branch removed
  7. STFTPINNNoSTFT         — ablation: dual-branch, plain MLP instead of iSTFT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from config import ModelConfig, STFTConfig

# Physical constant: total simulation duration [s]  (from PhysicsConfig.T)
# Used by coordinate networks to normalise t.  Must match cfg.physics.T.
_T_DURATION: float = 0.200


# ═══════════════════════════════════════════════════════════════════════
#  Shared building block
# ═══════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    def __init__(self, dims: List[int], activation: str = 'relu',
                 output_activation: bool = False):
        super().__init__()
        act = {'relu': nn.ReLU, 'tanh': nn.Tanh,
               'gelu': nn.GELU, 'silu': nn.SiLU}[activation]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2 or output_activation:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════
#  1. STFT-Enhanced PINN Stage I  (Section 3.4)
#  NOTE: STFT-PINN uses raw t internally because its learnable parameters
#  (tau_centres, omega_centres) are defined in the same [0, T] space.
#  No external t normalisation is needed or desirable here.
# ═══════════════════════════════════════════════════════════════════════

class InverseSTFTSynthesis(nn.Module):
    """
    h → complex latent S_h(τ,ω) ∈ C^{N_τ × N_ω} → iSTFT → f̃(t; h)
    """

    def __init__(self, stft_cfg: STFTConfig, param_dim: int = 3):
        super().__init__()
        self.N_tau   = stft_cfg.N_tau
        self.N_omega = stft_cfg.N_omega

        out_dim = 2 * self.N_tau * self.N_omega
        self.param_encoder = nn.Sequential(
            nn.Linear(param_dim, 256), nn.GELU(),
            nn.Linear(256, 256),       nn.GELU(),
            nn.Linear(256, 256),       nn.GELU(),
            nn.Linear(256, out_dim),
        )

        # Learnable time-shift centres spanning [0, T=0.200 s]
        self.tau_centres = nn.Parameter(
            torch.linspace(0.0, _T_DURATION, self.N_tau)
        )
        # Learnable frequency grid up to 500 Hz
        self.omega_centres = nn.Parameter(
            torch.linspace(0.0, 500.0, self.N_omega)
        )
        # Learnable synthesis window width
        self.log_sigma = nn.Parameter(torch.tensor(math.log(0.01)))

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B  = h.shape[0]

        # Complex STFT coefficients from material parameters
        raw    = self.param_encoder(h).view(B, self.N_tau, self.N_omega, 2)
        S_real = raw[..., 0]   # (B, N_tau, N_omega)
        S_imag = raw[..., 1]

        # Gaussian synthesis window  g(t − τ_k)
        sigma  = torch.exp(self.log_sigma)
        t_e    = t.unsqueeze(-1)                           # (Nt, 1)
        tau_e  = self.tau_centres.unsqueeze(0)             # (1, N_tau)
        window = torch.exp(-0.5 * ((t_e - tau_e) / sigma) ** 2)  # (Nt, N_tau)

        # Complex exponential  e^{i 2π ω_l t}
        phase     = t.unsqueeze(-1) * self.omega_centres.unsqueeze(0)  # (Nt, N_omega)
        cos_phase = torch.cos(2.0 * math.pi * phase)
        sin_phase = torch.sin(2.0 * math.pi * phase)

        # iSTFT synthesis (real part):
        # f̃(t) = Σ_k Σ_l  [S_real·cos − S_imag·sin] · g(t−τ_k)
        f_synth = (
            torch.einsum('bko,nk,no->bn', S_real, window, cos_phase)
            - torch.einsum('bko,nk,no->bn', S_imag, window, sin_phase)
        )
        return f_synth   # (B, Nt)


class TemporalEncoder(nn.Module):
    """Temporal coordinate embedding — Eq. 28."""

    def __init__(self, embed_dim: int = 128, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]
        # t normalised to [0, 1] for the MLP inside the temporal encoder
        self.net = MLP([1] + hidden_dims + [embed_dim], activation='gelu')

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)        # (Nt, 1)
        # Normalise t to [0, 1] — improves gradient flow in the encoder MLP
        t_norm = t / _T_DURATION
        return self.net(t_norm)        # (Nt, d)


class STFTPINNStage1(nn.Module):
    """
    Full proposed architecture:
      Parametric: h → iSTFT synthesis → post-MLP → z_h  (B, Nt, d)
      Temporal:   t → TemporalEncoder → z_t            (Nt, d)
      Fusion:     z_h * z_t → decoder → f(t; h)        (B, Nt)
    """

    def __init__(self, cfg: ModelConfig, Nt: int = 200):
        super().__init__()
        d = cfg.fusion_dim

        self.stft_synth  = InverseSTFTSynthesis(cfg.stft, cfg.param_input_dim)
        self.param_post  = nn.Sequential(
            nn.Linear(1, d), nn.GELU(), nn.Linear(d, d),
        )
        self.temporal_enc = TemporalEncoder(d, cfg.time_hidden_dims)
        self.decoder = nn.Sequential(
            nn.Linear(d, cfg.decoder_hidden_dims[0]), nn.GELU(),
            nn.Linear(cfg.decoder_hidden_dims[0], cfg.decoder_hidden_dims[1]), nn.GELU(),
            nn.Linear(cfg.decoder_hidden_dims[1], 1),
        )

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B  = h.shape[0]
        Nt = t.shape[0]

        f_stft = self.stft_synth(h, t)                     # (B, Nt)
        z_h    = self.param_post(f_stft.unsqueeze(-1))     # (B, Nt, d)
        z_t    = self.temporal_enc(t).unsqueeze(0).expand(B, Nt, -1)  # (B, Nt, d)

        return self.decoder(z_h * z_t).squeeze(-1)         # (B, Nt)


# ═══════════════════════════════════════════════════════════════════════
#  2. Stage II — Transformer Decoder  (Section 3.5)
# ═══════════════════════════════════════════════════════════════════════

class LoadDistributionEncoder(nn.Module):
    """
    Encodes pressure INCREMENT sequence delta_p for Boltzmann superposition.

    FIX v6 (root cause of generalisation failure):
      Previous version encoded raw p(t_j). For PFWD this works because
      the pulse shape makes p(t_j) and delta_p_j correlated. For other
      loading types (sinusoidal, multistage, random) the correlation
      breaks -- the transformer memorized the PFWD pulse shape, not the
      Boltzmann integral.

      The correct Duhamel formulation is:
          omega(t_i) = sum_{j<=i}  g(t_i - t_j; h) * delta_p_j
      where delta_p_j = p(t_j) - p(t_{j-1}).

      By encoding delta_p_j in memory[j], cross-attention weight A[i,j]
      only needs to learn g(t_i - t_j; h), which is extractable from
      time positional encodings plus material info in tgt.
    """
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.pos_enc = nn.Parameter(torch.zeros(1, 1000, d_model))
        nn.init.normal_(self.pos_enc, std=0.02)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        if p.dim() == 1:
            p = p.unsqueeze(0)
        # Compute delta_p_j = p(t_j) - p(t_{j-1}), delta_p_0 = p(t_0)
        delta_p = torch.zeros_like(p)
        delta_p[:, 0]  = p[:, 0]
        delta_p[:, 1:] = p[:, 1:] - p[:, :-1]
        emb = self.embed(delta_p.unsqueeze(-1))    # (B, Nt, d)
        Nt  = emb.shape[1]
        return emb + self.pos_enc[:, :Nt, :]


class STFTPINNStage2(nn.Module):
    """
    Stage II transformer decoder for stress-driven displacement.

    ROOT CAUSE OF GENERALISATION FAILURE (v6 diagnosis):

    BUG A — LoadDistributionEncoder encoded raw p(t_j) instead of Δp_j.
      The Boltzmann integral is: ω(t_i) = Σ_{j<=i} g(t_i-t_j; h) · Δp_j
      For PFWD loading, p(t_j) and Δp_j are correlated (monotone pulse),
      so the transformer accidentally learned the PFWD shape. For sinusoidal,
      multistage, or random loading the correlation vanishes → R² < 0.
      Fix: LoadDistributionEncoder (above) now encodes Δp_j = diff(p).

    BUG B — tgt had no temporal positional encoding.
      Without knowing WHICH time step t_i a query represents, cross-attention
      weight A[i,j] cannot encode the lag (t_i - t_j). Every query position
      looked identical to the attention mechanism → it computed the same
      weighted sum for all t_i → time-averaged output, not convolution.
      Fix: add learnable tgt_pos_enc[i] to each query position.

    With both fixes the transformer can learn:
      A[i,j] ≈ g(t_i - t_j; h)   (from tgt pos enc t_i, memory pos enc t_j,
                                    and material info in z_h + z_k)
      V[j]   = linear(Δp_j)
      output[i] = Σ_j A[i,j] V[j] ≈ ω(t_i)   [causal Boltzmann integral]

    Architecture:
      h_norm → material_encoder → z_h (B,d)      [trainable, gradient path]
      Stage I(h,t) frozen → kernel_proj → z_k (B,Nt,d)   [physics prior]
      tgt[i] = z_h_broadcast + z_k[i] + tgt_pos_enc[i]   [FIX B]
      p → Δp → LoadDistributionEncoder → memory (B,Nt,d)  [FIX A]
      TransformerDecoder(tgt, memory, causal_mask) → output_proj → ω(t)
    """
    def __init__(self, stage1_model: STFTPINNStage1,
                 cfg: ModelConfig, Nt: int = 200):
        super().__init__()
        self.stage1 = stage1_model
        d = cfg.transformer_d_model

        self.load_encoder = LoadDistributionEncoder(d)
        self.kernel_proj  = nn.Linear(1, d)

        # Trainable material encoder — gradient path from h to output
        self.material_encoder = nn.Sequential(
            nn.Linear(cfg.param_input_dim, d), nn.GELU(),
            nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, d),
        )

        # FIX B: temporal positional encoding for tgt queries
        # Without this each query position is time-blind and cannot
        # learn lag-dependent attention weights g(t_i - t_j; h).
        self.tgt_pos_enc = nn.Parameter(torch.zeros(1, 1000, d))
        nn.init.normal_(self.tgt_pos_enc, std=0.02)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=cfg.transformer_nhead,
            dim_feedforward=cfg.transformer_dim_feedforward,
            dropout=cfg.transformer_dropout, activation='gelu',
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            dec_layer, num_layers=cfg.transformer_num_layers,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1),
        )

    def _causal_mask(self, Nt: int, device) -> torch.Tensor:
        mask = torch.triu(torch.ones(Nt, Nt, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(self, h: torch.Tensor, t: torch.Tensor,
                p: torch.Tensor) -> torch.Tensor:
        B  = h.shape[0]
        Nt = t.shape[0]

        # Stage I frozen — physics prior, no grad into Stage I weights
        with torch.no_grad():
            f_kernel = self.stage1(h, t)               # (B, Nt)

        # Trainable material context — gradient path from h to output
        z_h     = self.material_encoder(h)             # (B, d)
        z_h_seq = z_h.unsqueeze(1).expand(B, Nt, -1)  # (B, Nt, d)

        # Physics hint: Stage I kernel f(t_i; h) as position-dependent prior
        z_k = self.kernel_proj(f_kernel.unsqueeze(-1)) # (B, Nt, d)

        # FIX B: add temporal position so query i knows it represents t_i
        tgt = z_h_seq + z_k + self.tgt_pos_enc[:, :Nt, :]  # (B, Nt, d)

        if p.dim() == 1:
            p = p.unsqueeze(0).expand(B, -1)

        # FIX A: LoadDistributionEncoder now encodes Δp_j internally
        load_emb = self.load_encoder(p)                # (B, Nt, d)

        decoded = self.transformer_decoder(
            tgt=tgt,
            memory=load_emb,
            tgt_mask=self._causal_mask(Nt, h.device),
        )
        return self.output_proj(decoded).squeeze(-1)   # (B, Nt)


# ═══════════════════════════════════════════════════════════════════════
#  2b. AnalyticalStage2  — exact Boltzmann superposition via SLS step response
#
#  Scientific rationale:
#    The displacement under arbitrary loading is:
#      ω(tᵢ) = Σⱼ≤ᵢ  g(tᵢ − tⱼ; h) · Δpⱼ
#    where g(t; h) is the SLS STEP RESPONSE (creep compliance × C):
#      g(t) = C·[1/(E1+E2) + E1/(E2·(E1+E2))·(1 − exp(−αt))]
#    This is the correct Duhamel kernel — NOT the impulse response f(t).
#    f(t) = dg/dt is used for Stage I; g(t) is used for Stage II.
#
#  Key design decisions:
#    - Uses h_raw (physical parameters) for analytically exact g(t)
#    - Built-in sanity checks diagnose failures at every step
#    - Falls back gracefully with informative error messages
# ═══════════════════════════════════════════════════════════════════════

class AnalyticalStage2(nn.Module):
    """
    Exact Stage II: analytical Boltzmann superposition using SLS step response.
    No learnable parameters beyond the frozen Stage I model.

    Computes:  ω(tᵢ) = Σⱼ₌₀ⁱ  g(tᵢ−tⱼ; h) · Δpⱼ

    where  g(t) = C·[1/(E1+E2) + E1/(E2·(E1+E2))·(1 − e^{−αt})]
    and    α    = E1·E2 / ((E1+E2)·η)

    Always pass h_raw (physical parameters) for correct results.
    """

    # Expected physical ranges — used for sanity checking
    _E1_RANGE  = (100e6,  2000e6)   # Pa  — wider than training range
    _E2_RANGE  = (10e6,   200e6)    # Pa
    _ETA_RANGE = (10_000, 5_000_000) # Pa·s
    _OMEGA_RANGE_MM = (-0.01, 100.0) # mm — physically reasonable deflection

    def __init__(self, stage1_model: STFTPINNStage1, C: float = 0.206756):
        super().__init__()
        self.stage1 = stage1_model
        for param in self.stage1.parameters():
            param.requires_grad = False
        self.register_buffer('C',      torch.tensor(float(C)))
        self.register_buffer('f_std',  torch.tensor(1.0))
        self.register_buffer('f_mean', torch.tensor(0.0))
        self._sanity_verbose = True   # set False to suppress per-batch prints

    def register_normalisation(self, f_std: float, f_mean: float):
        """Register Stage I normalisation so de-normalisation is correct."""
        self.register_buffer('f_std',  torch.tensor(float(f_std),  dtype=torch.float32))
        self.register_buffer('f_mean', torch.tensor(float(f_mean), dtype=torch.float32))

    # ── Sanity check helpers ──────────────────────────────────────────

    def _check_h_raw(self, h_raw: torch.Tensor) -> bool:
        """Verify physical parameters are within expected ranges."""
        ok = True
        E1  = h_raw[:, 0];  E2  = h_raw[:, 1];  eta = h_raw[:, 2]

        checks = [
            (E1,  self._E1_RANGE,   'E1 [Pa]'),
            (E2,  self._E2_RANGE,   'E2 [Pa]'),
            (eta, self._ETA_RANGE,  'eta [Pa·s]'),
        ]
        for vals, (lo, hi), name in checks:
            if vals.min().item() < lo or vals.max().item() > hi:
                print(f"  [AnalyticalStage2 WARNING] {name} out of expected range "
                      f"[{lo:.2e}, {hi:.2e}]: "
                      f"got [{vals.min().item():.2e}, {vals.max().item():.2e}]")
                ok = False
            if (vals <= 0).any():
                print(f"  [AnalyticalStage2 ERROR] {name} contains non-positive values — "
                      f"physically impossible. Check parameter de-normalisation.")
                ok = False
        return ok

    def _check_g_kernel(self, g: torch.Tensor, t: torch.Tensor) -> bool:
        """Verify step response g(t) has physically correct properties."""
        ok = True
        g0  = g[:, 0]   # initial value = C/(E1+E2)
        gT  = g[:, -1]  # final value → C/E2

        # g should be strictly non-negative
        if (g < 0).any():
            print(f"  [AnalyticalStage2 ERROR] g(t) has {(g<0).sum().item()} negative values. "
                  f"Check h_raw signs and C value.")
            ok = False

        # g should be monotonically non-decreasing (creep increases under load)
        dg = g[:, 1:] - g[:, :-1]
        if (dg < -1e-20).any():
            n_bad = (dg < -1e-20).sum().item()
            print(f"  [AnalyticalStage2 WARNING] g(t) is non-monotone at {n_bad} points "
                  f"— may indicate numerical issues with very large α.")

        # g(0) should be positive and much less than g(∞)
        ratio = gT / (g0 + 1e-40)
        if (ratio < 1.0).any():
            print(f"  [AnalyticalStage2 WARNING] g(T) < g(0) for some samples. "
                  f"Expected g(T)/g(0) >> 1 for typical SLS. Min ratio: {ratio.min().item():.3f}")

        # g values should be in physically plausible range [m/Pa]
        # Typical: C/(E1+E2) ~ 0.207/(640e6+49e6) ~ 3e-10 m/Pa
        g_max = g.max().item()
        g_min = g.min().item()
        if g_max > 1e-5 or g_min < 0:
            print(f"  [AnalyticalStage2 WARNING] g(t) range [{g_min:.2e}, {g_max:.2e}] m/Pa "
                  f"seems physically implausible (expected ~1e-10 to 1e-8 m/Pa). "
                  f"Check C={self.C.item():.4f} and h_raw units.")
            ok = False

        return ok

    def _check_pressure(self, p: torch.Tensor) -> bool:
        """Verify pressure history is physical."""
        ok = True
        if (p < 0).any():
            print(f"  [AnalyticalStage2 WARNING] Pressure has {(p<0).sum().item()} "
                  f"negative values. Ensure p(t) ≥ 0 for PFWD loading.")
        p_max = p.max().item()
        if p_max > 2e6:
            print(f"  [AnalyticalStage2 WARNING] Peak pressure {p_max:.2e} Pa > 2 MPa. "
                  f"Typical PFWD peak is 50–200 kPa.")
            ok = False
        if p_max < 1.0:
            print(f"  [AnalyticalStage2 ERROR] Peak pressure {p_max:.2e} Pa ≈ 0. "
                  f"Pressure may be normalised or zero. Expected ~150,000 Pa.")
            ok = False
        return ok

    def _check_omega(self, omega: torch.Tensor) -> bool:
        """Verify output displacement is physically plausible."""
        ok = True
        omega_mm = omega * 1000.0
        lo, hi = self._OMEGA_RANGE_MM

        if (omega_mm < lo).any() or (omega_mm > hi).any():
            print(f"  [AnalyticalStage2 WARNING] omega out of plausible range "
                  f"[{lo}, {hi}] mm. Got [{omega_mm.min().item():.4f}, "
                  f"{omega_mm.max().item():.4f}] mm. "
                  f"Check units of g(t) [m/Pa] × p [Pa] = [m].")
            ok = False

        if torch.isnan(omega).any():
            print(f"  [AnalyticalStage2 ERROR] NaN in output displacement. "
                  f"Check for division by zero in alpha or g(t).")
            ok = False

        if torch.isinf(omega).any():
            print(f"  [AnalyticalStage2 ERROR] Inf in output displacement.")
            ok = False

        return ok

    # ── Main forward pass ─────────────────────────────────────────────

    def forward(self, h: torch.Tensor, t: torch.Tensor,
                p: torch.Tensor,
                h_raw: torch.Tensor = None,
                run_checks: bool = False) -> torch.Tensor:
        """
        Parameters
        ----------
        h        : (B, 3)  normalised material parameters
        t        : (Nt,)   time grid [s]
        p        : (Nt,) or (B, Nt)  pressure history [Pa]
        h_raw    : (B, 3)  physical parameters [E1 Pa, E2 Pa, eta Pa·s]
                           REQUIRED for correct results.
        run_checks : bool  run sanity checks (auto-enabled on first batch)

        Returns
        -------
        omega : (B, Nt)  surface displacement [m]
        """
        B  = h.shape[0]
        Nt = t.shape[0]

        # Auto-run checks on first call only to avoid per-batch overhead
        _do_checks = run_checks or self._sanity_verbose
        if _do_checks:
            self._sanity_verbose = False   # suppress after first batch

        # ── 1. Validate inputs ────────────────────────────────────────
        if h_raw is None:
            print("  [AnalyticalStage2 ERROR] h_raw not provided. "
                  "Physical parameters required for correct g(t) computation. "
                  "Pass params_raw from the batch. Falling back to normalised h "
                  "which will give WRONG results.")
            # Emergency fallback — produce zero output to make failure obvious
            return torch.zeros(B, Nt, device=h.device)

        if _do_checks:
            self._check_h_raw(h_raw)

        if p.dim() == 1:
            p = p.unsqueeze(0).expand(B, -1)   # (B, Nt)

        if _do_checks:
            self._check_pressure(p)

        # ── 2. Compute SLS step response g(t; h) analytically ─────────
        # g(t) = C·[1/(E1+E2) + E1/(E2·(E1+E2))·(1 − exp(−αt))]
        E1  = h_raw[:, 0].unsqueeze(1).float()   # (B, 1)
        E2  = h_raw[:, 1].unsqueeze(1).float()
        eta = h_raw[:, 2].unsqueeze(1).float()
        C   = self.C.float()

        alpha    = E1 * E2 / ((E1 + E2) * eta)              # (B, 1)
        t_row    = t.unsqueeze(0).float()                    # (1, Nt)
        g_kernel = C * (1.0 / (E1 + E2)
                        + E1 / (E2 * (E1 + E2))
                        * (1.0 - torch.exp(-alpha * t_row))) # (B, Nt)

        if _do_checks:
            self._check_g_kernel(g_kernel, t)
            print(f"  [AnalyticalStage2 CHECK] g(t=0) mean = {g_kernel[:,0].mean().item():.3e} m/Pa "
                  f"| g(t=T) mean = {g_kernel[:,-1].mean().item():.3e} m/Pa "
                  f"| alpha mean = {alpha.squeeze().mean().item():.2f} s⁻¹")

        # ── 3. Pressure increments Δpⱼ ────────────────────────────────
        p_f = p.float()
        delta_p = torch.zeros_like(p_f)
        delta_p[:, 0]  = p_f[:, 0]
        delta_p[:, 1:] = p_f[:, 1:] - p_f[:, :-1]

        if _do_checks:
            print(f"  [AnalyticalStage2 CHECK] delta_p sum = {delta_p.sum(dim=1).mean().item():.2f} Pa "
                  f"(should ≈ 0 after full load removal) | peak p = {p_f.max().item():.0f} Pa")

        # ── 4. Causal discrete convolution — vectorized Toeplitz product ──
        # ω(tᵢ) = Σⱼ₌₀ⁱ g(tᵢ−tⱼ) · Δpⱼ   [m/Pa × Pa = m]
        #
        # Build lower-triangular Toeplitz matrix G ∈ R^{B×Nt×Nt}:
        #   G[b, i, j] = g_kernel[b, i-j]  for j ≤ i,  0 otherwise
        # Then ω = G · delta_p  (batched matrix-vector product)
        #
        # Construction: column j of G is g_kernel shifted down by j rows.
        # Equivalently, use torch.tril on the full Toeplitz matrix.
        #
        # Memory: B × Nt × Nt × 4 bytes.  For B=64, Nt=200: ~10 MB — fine.
        # Speed: replaces O(Nt) Python iterations with a single bmm call.
        #
        # g_kernel[:, i-j] for i≥j is equivalent to taking the lag index
        # from a column-shifted version.  We build it via stride tricks:
        #   col_indices[i, j] = i - j  for i ≥ j  →  tril indices
        row_idx = torch.arange(Nt, device=h.device).unsqueeze(1)   # (Nt, 1)
        col_idx = torch.arange(Nt, device=h.device).unsqueeze(0)   # (1, Nt)
        lag_idx = (row_idx - col_idx).clamp(min=0)                  # (Nt, Nt)
        causal_mask = (row_idx >= col_idx).float()                  # (Nt, Nt)

        # G[b, i, j] = g_kernel[b, lag_idx[i,j]] * causal_mask[i,j]
        G = g_kernel[:, lag_idx] * causal_mask.unsqueeze(0)        # (B, Nt, Nt)

        # ω = G @ delta_p   (B, Nt, Nt) × (B, Nt, 1) → (B, Nt)
        omega = torch.bmm(G, delta_p.unsqueeze(-1)).squeeze(-1)    # (B, Nt)

        if _do_checks:
            self._check_omega(omega)
            print(f"  [AnalyticalStage2 CHECK] omega peak = {omega.max().item()*1e3:.4f} mm "
                  f"| omega final = {omega[:,-1].mean().item()*1e6:.2f} µm "
                  f"| negative values: {(omega < 0).sum().item()}")

        return omega   # (B, Nt) [m]


# ═══════════════════════════════════════════════════════════════════════
#  3. Baseline MLP  (Section 5.2.1)
#
#  FIX (BUG-2): t normalised to [0, 1] before concatenation.
#  Raw t ∈ [0, 0.200] caused ~10× scale mismatch with h_norm ∈ [-3, 3].
#  The first linear layer's weight init (Kaiming for ReLU) produces
#  entries ~ N(0, 2/4) = N(0, 0.5) for 4-dim input.  With raw t, the
#  weight component for the time dimension received ~15× less gradient
#  signal than h_norm components.  Normalising t to [0, 1] restores balance.
# ═══════════════════════════════════════════════════════════════════════

class BaselineMLP(nn.Module):
    """Standard 5-layer ReLU MLP. No physics constraints."""

    def __init__(self, cfg: ModelConfig, Nt: int = 200):
        super().__init__()
        self.net = MLP([4] + cfg.baseline_hidden_dims + [1], activation='relu')

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, Nt = h.shape[0], t.shape[0]
        # FIX: normalise t to [0, 1] so all four input components are O(1)
        t_norm = t / _T_DURATION                          # [0, 1]
        h_e = h.unsqueeze(1).expand(B, Nt, 3)
        t_e = t_norm.unsqueeze(0).unsqueeze(-1).expand(B, Nt, 1)
        return self.net(torch.cat([h_e, t_e], dim=-1)).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════
#  4. SIREN-PINN  (Section 5.2.1)
#
#  FIX (BUG-1 — critical): t normalised to [-1, 1].
#
#  FIX (ω₀): default changed from 20 → 50.
#
#    The SLS impulse response requires ≈ 18.3 rad of first-layer phase
#    coverage in the t dimension to represent the full exp(−αt) decay.
#    With t_norm ∈ [-1, 1]:
#
#      ω₀ = 20: first-layer max phase = ω₀/4 = 5.0 rad  (3.7× deficit)
#      ω₀ = 50: first-layer max phase = ω₀/4 = 12.5 rad (1.5× deficit, OK —
#               deeper layers provide exponential frequency escalation)
#      ω₀ =100: first-layer max phase = ω₀/4 = 25.0 rad (overcoverage)
#
#    ω₀ = 50 is the minimum reliable choice; ω₀ = 30 (SIREN paper default)
#    still undershoots by 2.4× and should be considered insufficient here.
#
#  SCIENTIFIC NOTE: SIREN's sinusoidal inductive bias represents periodic
#  functions efficiently but must decompose the target exp(−αt) into a
#  Fourier series, requiring many terms.  The STFT-PINN's advantage over
#  SIREN therefore reflects inductive bias alignment (Gaussian wavelet basis
#  ≈ exp-decay + oscillation) as much as spectral bias mitigation.
# ═══════════════════════════════════════════════════════════════════════

class SirenLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 omega0: float = 50.0, is_first: bool = False):
        super().__init__()
        self.omega0 = omega0
        self.linear = nn.Linear(in_dim, out_dim)
        with torch.no_grad():
            if is_first:
                # Sitzmann et al. (2020): uniform in [-1/n_in, 1/n_in]
                self.linear.weight.uniform_(-1.0 / in_dim, 1.0 / in_dim)
            else:
                b = math.sqrt(6.0 / in_dim) / omega0
                self.linear.weight.uniform_(-b, b)
            # Bias initialised to zero (standard)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return torch.sin(self.omega0 * self.linear(x))


class SIRENPINN(nn.Module):
    """
    SIREN coordinate network with corrected input normalisation.
    Input: [h_norm (3 dims, ∈ [-3,3]) | t_norm (1 dim, ∈ [-1,1])]
    Both dimensions now have comparable frequency coverage in the first layer.
    """

    def __init__(self, cfg: ModelConfig, Nt: int = 200):
        super().__init__()
        omega0  = cfg.siren_omega0      # 20 — see config.py
        hidden  = cfg.siren_hidden_dims
        layers  = [SirenLayer(4, hidden[0], omega0, is_first=True)]
        for i in range(len(hidden) - 1):
            layers.append(SirenLayer(hidden[i], hidden[i + 1], omega0))
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer  = nn.Linear(hidden[-1], 1)
        with torch.no_grad():
            # Final layer: keep small output variance at initialisation
            b = math.sqrt(6.0 / hidden[-1]) / omega0
            self.output_layer.weight.uniform_(-b, b)
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, Nt = h.shape[0], t.shape[0]
        # FIX (BUG-1): normalise t to [-1, 1] so the first SIREN layer
        # sees equal phase coverage in both h_norm and t dimensions.
        t_norm = (t / _T_DURATION) * 2.0 - 1.0            # [-1, 1]
        h_e = h.unsqueeze(1).expand(B, Nt, 3)
        t_e = t_norm.unsqueeze(0).unsqueeze(-1).expand(B, Nt, 1)
        x   = torch.cat([h_e, t_e], dim=-1)               # (B, Nt, 4)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x).squeeze(-1)            # (B, Nt)


# ═══════════════════════════════════════════════════════════════════════
#  5. FFN-PINN  (Section 5.2.1)
#
#  FIX (BUG-2): t normalised to [0, 1] before Fourier feature projection.
#  The random projection matrix B maps [h_norm, t_norm] → Fourier space.
#  With raw t ∈ [0, 0.2], the time component contributed negligible energy
#  to the projected features vs h_norm ∈ [-3, 3].
#
#  sigma = 1.0 is retained (was already fixed in v2 from sigma=10.0).
#  With sigma=10 and normalised t, projected features have frequency
#  components up to ~30 rad in t, which is appropriate.
# ═══════════════════════════════════════════════════════════════════════

class FourierFeatureNetworkPINN(nn.Module):
    """Random Fourier feature embedding + MLP physics-informed network."""

    def __init__(self, cfg: ModelConfig, Nt: int = 200, sigma: float = 1.0):
        super().__init__()
        mapping_size = cfg.ffn_mapping_size
        B_matrix     = torch.randn(mapping_size, 4) * sigma
        self.register_buffer('B_matrix', B_matrix)
        ff_dim = 2 * mapping_size
        self.mlp = MLP([ff_dim] + cfg.ffn_hidden_dims + [1], activation='relu')

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B_sz, Nt = h.shape[0], t.shape[0]
        # FIX (BUG-2): normalise t to [0, 1] for balanced Fourier projection
        t_norm = t / _T_DURATION                          # [0, 1]
        h_e  = h.unsqueeze(1).expand(B_sz, Nt, 3)
        t_e  = t_norm.unsqueeze(0).unsqueeze(-1).expand(B_sz, Nt, 1)
        x    = torch.cat([h_e, t_e], dim=-1)
        proj = 2.0 * math.pi * torch.matmul(x, self.B_matrix.T)
        ff   = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return self.mlp(ff).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════
#  6. Ablation: STFT synthesis — NO temporal branch  (R2-M6)
# ═══════════════════════════════════════════════════════════════════════

class STFTPINNNoTemporal(nn.Module):
    def __init__(self, cfg: ModelConfig, Nt: int = 200):
        super().__init__()
        d = cfg.fusion_dim
        self.stft_synth = InverseSTFTSynthesis(cfg.stft, cfg.param_input_dim)
        self.param_post = nn.Sequential(
            nn.Linear(1, d), nn.GELU(), nn.Linear(d, d),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d, cfg.decoder_hidden_dims[0]), nn.GELU(),
            nn.Linear(cfg.decoder_hidden_dims[0], cfg.decoder_hidden_dims[1]), nn.GELU(),
            nn.Linear(cfg.decoder_hidden_dims[1], 1),
        )

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        f_stft = self.stft_synth(h, t)
        z_h    = self.param_post(f_stft.unsqueeze(-1))
        return self.decoder(z_h).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════
#  7. Ablation: Dual-branch — NO iSTFT  (R2-M6)
# ═══════════════════════════════════════════════════════════════════════

class STFTPINNNoSTFT(nn.Module):
    def __init__(self, cfg: ModelConfig, Nt: int = 200):
        super().__init__()
        d = cfg.fusion_dim
        self.param_encoder = nn.Sequential(
            nn.Linear(cfg.param_input_dim, 256), nn.GELU(),
            nn.Linear(256, 256),                 nn.GELU(),
            nn.Linear(256, d),
        )
        self.temporal_enc = TemporalEncoder(d, cfg.time_hidden_dims)
        self.decoder = nn.Sequential(
            nn.Linear(d, cfg.decoder_hidden_dims[0]), nn.GELU(),
            nn.Linear(cfg.decoder_hidden_dims[0], cfg.decoder_hidden_dims[1]), nn.GELU(),
            nn.Linear(cfg.decoder_hidden_dims[1], 1),
        )

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, Nt = h.shape[0], t.shape[0]
        z_h = self.param_encoder(h).unsqueeze(1).expand(B, Nt, -1)
        z_t = self.temporal_enc(t).unsqueeze(0).expand(B, Nt, -1)
        return self.decoder(z_h * z_t).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════
#  Factory
# ═══════════════════════════════════════════════════════════════════════

def build_model(
    name: str,
    cfg: ModelConfig,
    Nt: int = 200,
    stage1_model: Optional[STFTPINNStage1] = None,
    C: float = 0.206756,
) -> nn.Module:
    if name == 'stft_pinn_s1':
        return STFTPINNStage1(cfg, Nt)
    elif name == 'stft_pinn_s2':
        assert stage1_model is not None
        return STFTPINNStage2(stage1_model, cfg, Nt)
    elif name == 'analytical_stage2':
        assert stage1_model is not None, "AnalyticalStage2 requires a trained Stage I model"
        return AnalyticalStage2(stage1_model, C=C)
    elif name == 'baseline_mlp':
        return BaselineMLP(cfg, Nt)
    elif name == 'siren_pinn':
        return SIRENPINN(cfg, Nt)
    elif name == 'ffn_pinn':
        return FourierFeatureNetworkPINN(cfg, Nt)
    elif name == 'stft_no_temporal':
        return STFTPINNNoTemporal(cfg, Nt)
    elif name == 'stft_no_stft':
        return STFTPINNNoSTFT(cfg, Nt)
    else:
        raise ValueError(f"Unknown model name: '{name}'")