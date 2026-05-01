"""
Synthetic dataset generation for STFT-Enhanced PINN training and evaluation.

Generates displacement time histories from the analytical SLS forward model
using Latin Hypercube Sampling over the material parameter space.

Reference: Manuscript Section 5.1
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional
from scipy.stats import qmc

from config import Config, PhysicsConfig, DataConfig
from forward_model import (
    geometric_coeff,
    impulse_response_np,
    impulse_gradients_np,
    generate_pfwd_pulse,
    discretize_loading,
    displacement_superposition_np,
    displacement_gradients_np,
)


# ═══════════════════════════════════════════════════════════════════════
#  Latin Hypercube Sampling  (Section 5.1)
# ═══════════════════════════════════════════════════════════════════════

def latin_hypercube_sample(
    n_samples: int,
    ranges: list,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate space-filling parameter samples via LHS.

    Parameters
    ----------
    n_samples : number of samples
    ranges    : list of (low, high) tuples for each dimension
    seed      : random seed

    Returns
    -------
    samples : (n_samples, n_dims)
    """
    n_dims = len(ranges)
    sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
    unit_samples = sampler.random(n=n_samples)   # in [0, 1]^d

    # Scale to physical ranges
    lower = np.array([r[0] for r in ranges])
    upper = np.array([r[1] for r in ranges])
    samples = qmc.scale(unit_samples, lower, upper)
    return samples


# ═══════════════════════════════════════════════════════════════════════
#  Stage I dataset — impulse response  (Section 4.1)
# ═══════════════════════════════════════════════════════════════════════

def generate_stage1_dataset(
    cfg: Config,
    n_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate impulse response data  f(t; E1, E2, η).

    Returns dict with keys:
        'params'      : (N, 3)  — [E1, E2, η]
        'time'        : (Nt,)
        'f_true'      : (N, Nt)  — impulse response
        'df_dE1'      : (N, Nt)  — analytical gradient
        'df_dE2'      : (N, Nt)
        'df_deta'     : (N, Nt)
    """
    phys = cfg.physics
    n = n_samples or cfg.data.n_total
    s = seed or cfg.data.seed

    # Time vector
    t = np.linspace(0, phys.T, phys.Nt)                    # (Nt,)

    # Parameter samples via LHS
    ranges = [phys.E1_range, phys.E2_range, phys.eta_range]
    params = latin_hypercube_sample(n, ranges, seed=s)      # (N, 3)
    E1 = params[:, 0:1]   # (N, 1)
    E2 = params[:, 1:2]
    eta = params[:, 2:3]

    C = phys.C
    t_broad = t[np.newaxis, :]   # (1, Nt) for broadcasting

    # Impulse response  (Eq. 12)
    f_true = impulse_response_np(t_broad, E1, E2, eta, C)  # (N, Nt)

    # Analytical gradients  (Eqs. 34-36)
    df_dE1, df_dE2, df_deta = impulse_gradients_np(
        t_broad, E1, E2, eta, C
    )

    return {
        'params': params.astype(np.float32),
        'time': t.astype(np.float32),
        'f_true': f_true.astype(np.float32),
        'df_dE1': df_dE1.astype(np.float32),
        'df_dE2': df_dE2.astype(np.float32),
        'df_deta': df_deta.astype(np.float32),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Stage II dataset — displacement under PFWD loading  (Section 4.2)
# ═══════════════════════════════════════════════════════════════════════

def generate_stage2_dataset(
    cfg: Config,
    n_samples: Optional[int] = None,
    seed: Optional[int] = None,
    loading_type: str = 'pfwd',
) -> Dict[str, np.ndarray]:
    """
    Generate stress-driven displacement data  ω(t; h, p(·)).

    Parameters
    ----------
    loading_type : 'pfwd', 'sinusoidal_0.5', 'sinusoidal_2',
                   'multistage', 'random', 'extended_pfwd'

    Returns dict with keys:
        'params'        : (N, 3)
        'time'          : (Nt,)
        'pressure'      : (Nt,)
        'omega_true'    : (N, Nt)
        'domega_dE1'    : (N, Nt)
        'domega_dE2'    : (N, Nt)
        'domega_deta'   : (N, Nt)
    """
    phys = cfg.physics
    dcfg = cfg.data
    n = n_samples or dcfg.n_total
    s = seed or dcfg.seed

    t = np.linspace(0, phys.T, phys.Nt)

    # Generate loading history
    if loading_type == 'pfwd':
        # Standard PFWD half-sine: 20 ms, 154.2 kPa (Zhang et al. 2020)
        p = generate_pfwd_pulse(t, dcfg.peak_pressure,
                                dcfg.pulse_duration_ms)
    elif loading_type == 'extended_pfwd':
        # Extended half-sine: 30 ms duration (upper end of field range)
        p = generate_pfwd_pulse(t, dcfg.peak_pressure, duration_ms=30.0)
    elif loading_type.startswith('sinusoidal'):
        freq = float(loading_type.split('_')[1])
        from forward_model import generate_sinusoidal_loading
        p = generate_sinusoidal_loading(t, dcfg.peak_pressure, freq)
    elif loading_type == 'multistage':
        from forward_model import generate_multistage_loading
        p = generate_multistage_loading(t)
    elif loading_type == 'random':
        from forward_model import generate_random_loading
        p = generate_random_loading(t)
    else:
        raise ValueError(f"Unknown loading type: {loading_type}")

    # Discretize loading for superposition
    p0, tau_j, delta_p_j = discretize_loading(t, p)

    # Parameter samples
    ranges = [phys.E1_range, phys.E2_range, phys.eta_range]
    params = latin_hypercube_sample(n, ranges, seed=s + 1000)
    E1 = params[:, 0:1]
    E2 = params[:, 1:2]
    eta = params[:, 2:3]

    C = phys.C
    t_broad = t[np.newaxis, :]

    # Compute displacement via Boltzmann superposition (Eq. 37)
    omega_true = displacement_superposition_np(
        t_broad, E1, E2, eta, C, p0, tau_j, delta_p_j
    )

    # Analytical gradients (Eq. 39)
    domega_dE1, domega_dE2, domega_deta = displacement_gradients_np(
        t_broad, E1, E2, eta, C, p0, tau_j, delta_p_j
    )

    return {
        'params': params.astype(np.float32),
        'time': t.astype(np.float32),
        'pressure': p.astype(np.float32),
        'omega_true': omega_true.astype(np.float32),
        'domega_dE1': domega_dE1.astype(np.float32),
        'domega_dE2': domega_dE2.astype(np.float32),
        'domega_deta': domega_deta.astype(np.float32),
    }


# ═══════════════════════════════════════════════════════════════════════
#  PyTorch Dataset wrappers
# ═══════════════════════════════════════════════════════════════════════

class Stage1Dataset(Dataset):
    """Dataset for Stage I — impulse response learning."""

    def __init__(self, data: Dict[str, np.ndarray],
                 normalize: bool = True,
                 f_mean: Optional[torch.Tensor] = None,
                 f_std: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        data      : dict from generate_stage1_dataset
        normalize : whether to normalise targets
        f_mean    : if provided, use this mean instead of computing from data
                    (used for val/test splits to match training normalisation)
        f_std     : if provided, use this std instead of computing from data
        """
        self.params_raw = torch.from_numpy(data['params'])
        self.time = torch.from_numpy(data['time'])
        self.df_dE1 = torch.from_numpy(data['df_dE1'])
        self.df_dE2 = torch.from_numpy(data['df_dE2'])
        self.df_deta = torch.from_numpy(data['df_deta'])

        # Always keep physical-unit f_true for evaluation
        self.f_true_raw = torch.from_numpy(data['f_true'])

        # Normalisation statistics (for stable training)
        if normalize:
            self.param_mean = self.params_raw.mean(dim=0)
            self.param_std = self.params_raw.std(dim=0)
            self.params = (self.params_raw - self.param_mean) / (self.param_std + 1e-8)

            # Use provided stats if given (val/test sets use training stats)
            if f_mean is not None and f_std is not None:
                self.f_mean = f_mean
                self.f_std = f_std
            else:
                self.f_mean = self.f_true_raw.mean()
                self.f_std = self.f_true_raw.std() + 1e-30

            # Normalised targets for loss computation
            self.f_true = (self.f_true_raw - self.f_mean) / self.f_std
        else:
            self.params = self.params_raw
            self.param_mean = torch.zeros(3)
            self.param_std = torch.ones(3)
            self.f_mean = torch.tensor(0.0)
            self.f_std = torch.tensor(1.0)
            self.f_true = self.f_true_raw.clone()

    def __len__(self):
        return self.params.shape[0]

    def __getitem__(self, idx):
        return {
            'params': self.params[idx],           # (3,) normalised — network input
            'params_raw': self.params_raw[idx],   # (3,) physical units
            'f_true': self.f_true[idx],           # (Nt,) normalised — for training loss
            'f_true_raw': self.f_true_raw[idx],   # (Nt,) physical units — for evaluation
            'df_dE1': self.df_dE1[idx],
            'df_dE2': self.df_dE2[idx],
            'df_deta': self.df_deta[idx],
        }


class Stage2Dataset(Dataset):
    """Dataset for Stage II — stress-driven displacement learning.

    FIX v4 (scale mismatch):
      omega_true is in physical metres (~6e-4 m peak). The transformer
      output_proj initialises near O(1), causing R²≪0 at epoch 1 when
      comparing O(1) predictions to O(1e-3) targets.

      Fix: normalise omega_true to zero mean / unit std using training
      statistics, exactly as Stage1Dataset normalises f_true.
      omega_mean and omega_std are stored so compute_metrics can
      denormalise back to physical units for reporting.

      Pass omega_mean/omega_std from the training split to val/test
      splits so all splits share identical normalisation.
    """

    def __init__(self, data: Dict[str, np.ndarray],
                 param_mean: torch.Tensor = None,
                 param_std: torch.Tensor = None,
                 omega_mean: Optional[torch.Tensor] = None,
                 omega_std:  Optional[torch.Tensor] = None):
        self.params_raw  = torch.from_numpy(data['params'])
        self.time        = torch.from_numpy(data['time'])
        self.pressure    = torch.from_numpy(data['pressure'])   # (Nt,) shared
        self.domega_dE1  = torch.from_numpy(data['domega_dE1'])
        self.domega_dE2  = torch.from_numpy(data['domega_dE2'])
        self.domega_deta = torch.from_numpy(data['domega_deta'])

        # Always keep physical omega for evaluation
        self.omega_true_raw = torch.from_numpy(data['omega_true'])

        # Parameter normalisation (from Stage I training set)
        if param_mean is not None and param_std is not None:
            self.param_mean = param_mean
            self.param_std  = param_std
        else:
            self.param_mean = self.params_raw.mean(dim=0)
            self.param_std  = self.params_raw.std(dim=0)
        self.params = (self.params_raw - self.param_mean) / (self.param_std + 1e-8)

        # FIX v4: omega normalisation — use provided stats or compute from data
        if omega_mean is not None and omega_std is not None:
            self.omega_mean = omega_mean
            self.omega_std  = omega_std
        else:
            self.omega_mean = self.omega_true_raw.mean()
            self.omega_std  = self.omega_true_raw.std() + 1e-30

        # Normalised omega for loss computation — now O(1)
        self.omega_true = (self.omega_true_raw - self.omega_mean) / self.omega_std

    def __len__(self):
        return self.params.shape[0]

    def __getitem__(self, idx):
        return {
            'params':        self.params[idx],           # (3,) normalised
            'params_raw':    self.params_raw[idx],       # (3,) physical
            'pressure':      self.pressure,              # (Nt,) shared
            'omega_true':    self.omega_true[idx],       # (Nt,) NORMALISED — for loss
            'omega_true_raw':self.omega_true_raw[idx],   # (Nt,) physical metres — for eval
            'domega_dE1':    self.domega_dE1[idx],
            'domega_dE2':    self.domega_dE2[idx],
            'domega_deta':   self.domega_deta[idx],
        }


# ═══════════════════════════════════════════════════════════════════════
#  Mixed Stage II dataset — per-sample pressure for generalisation
#
#  Scientific rationale:
#    Training on a single loading type (e.g. PFWD) teaches the model to
#    recognise that specific pulse shape, not to implement the Boltzmann
#    convolution operation.  Training on mixed loading types forces the
#    transformer to learn the general operator:
#        omega(t_i) = sum_{j<=i} g(t_i - t_j; h) * delta_p_j
#    which is loading-type-agnostic by construction.
#
#    Each sample now has its OWN pressure vector (per-sample, not shared).
#    The loading type is drawn uniformly at random for each sample.
#
#  Training loading types (5): pfwd, extended_pfwd, sinusoidal_0.5,
#      sinusoidal_2, random.  multistage is reserved for out-of-distribution
#      evaluation only.
# ═══════════════════════════════════════════════════════════════════════

def generate_mixed_stage2_dataset(
    cfg: Config,
    n_samples: Optional[int] = None,
    seed: Optional[int] = None,
    loading_types: list = None,
) -> Dict[str, np.ndarray]:
    """
    Generate Stage II dataset with MIXED loading types.
    Each sample gets its own pressure vector sampled from a random
    loading type.  Returns pressure as (N, Nt) per-sample array.

    loading_types: list of loading type strings to mix uniformly.
                   Defaults to ['pfwd', 'extended_pfwd', 'sinusoidal_0.5',
                                 'sinusoidal_2', 'random'].
    """
    from forward_model import generate_sinusoidal_loading, generate_random_loading

    if loading_types is None:
        loading_types = ['pfwd', 'extended_pfwd', 'sinusoidal_0.5',
                         'sinusoidal_2', 'random']

    phys = cfg.physics
    dcfg = cfg.data
    n    = n_samples or dcfg.n_total
    s    = seed or dcfg.seed
    rng  = np.random.RandomState(s + 2000)

    t      = np.linspace(0, phys.T, phys.Nt)
    ranges = [phys.E1_range, phys.E2_range, phys.eta_range]
    params = latin_hypercube_sample(n, ranges, seed=s + 1500)
    E1     = params[:, 0:1]
    E2     = params[:, 1:2]
    eta    = params[:, 2:3]
    C      = phys.C
    t_b    = t[np.newaxis, :]

    # Pre-generate one pressure profile per loading type for efficiency
    def _make_p(ltype):
        if ltype == 'pfwd':
            return generate_pfwd_pulse(t, dcfg.peak_pressure,
                                       dcfg.pulse_duration_ms)
        elif ltype == 'extended_pfwd':
            return generate_pfwd_pulse(t, dcfg.peak_pressure, duration_ms=30.0)
        elif ltype.startswith('sinusoidal'):
            freq = float(ltype.split('_')[1])
            return generate_sinusoidal_loading(t, dcfg.peak_pressure, freq)
        elif ltype == 'multistage':
            from forward_model import generate_multistage_loading
            return generate_multistage_loading(t)
        elif ltype == 'random':
            return generate_random_loading(t, seed=rng.randint(0, 100000))
        else:
            raise ValueError(f"Unknown loading type: {ltype}")

    # Assign each sample a loading type uniformly at random
    type_indices = rng.randint(0, len(loading_types), size=n)

    # Build per-sample pressure and displacement arrays
    pressure_all  = np.zeros((n, phys.Nt), dtype=np.float32)
    omega_all     = np.zeros((n, phys.Nt), dtype=np.float32)
    dE1_all       = np.zeros((n, phys.Nt), dtype=np.float32)
    dE2_all       = np.zeros((n, phys.Nt), dtype=np.float32)
    deta_all      = np.zeros((n, phys.Nt), dtype=np.float32)

    # Group samples by loading type for vectorised superposition
    for li, ltype in enumerate(loading_types):
        mask = (type_indices == li)
        if not mask.any():
            continue
        idx  = np.where(mask)[0]
        p    = _make_p(ltype)
        p0, tau_j, dp_j = discretize_loading(t, p)

        E1_sub  = E1[idx]
        E2_sub  = E2[idx]
        eta_sub = eta[idx]

        omega_sub = displacement_superposition_np(
            t_b, E1_sub, E2_sub, eta_sub, C, p0, tau_j, dp_j
        )
        dE1_sub, dE2_sub, deta_sub = displacement_gradients_np(
            t_b, E1_sub, E2_sub, eta_sub, C, p0, tau_j, dp_j
        )
        pressure_all[idx] = p[np.newaxis, :]   # broadcast to all samples in group
        omega_all[idx]    = omega_sub
        dE1_all[idx]      = dE1_sub
        dE2_all[idx]      = dE2_sub
        deta_all[idx]     = deta_sub

    return {
        'params':      params.astype(np.float32),
        'time':        t.astype(np.float32),
        'pressure':    pressure_all,          # (N, Nt) — per sample
        'omega_true':  omega_all,
        'domega_dE1':  dE1_all,
        'domega_dE2':  dE2_all,
        'domega_deta': deta_all,
    }


class MixedStage2Dataset(Dataset):
    """
    Stage II dataset with per-sample pressure vectors.
    Used for mixed-loading training to teach the transformer the
    general Boltzmann convolution operator, not a loading-specific mapping.

    Key difference from Stage2Dataset:
      - pressure is (N, Nt) per-sample, not (Nt,) shared.
      - __getitem__ returns pressure[idx] so each batch sample has its
        own loading history.
    """

    def __init__(self, data: Dict[str, np.ndarray],
                 param_mean: torch.Tensor = None,
                 param_std: torch.Tensor = None,
                 omega_mean: Optional[torch.Tensor] = None,
                 omega_std:  Optional[torch.Tensor] = None):

        self.params_raw     = torch.from_numpy(data['params'])
        self.time           = torch.from_numpy(data['time'])
        self.domega_dE1     = torch.from_numpy(data['domega_dE1'])
        self.domega_dE2     = torch.from_numpy(data['domega_dE2'])
        self.domega_deta    = torch.from_numpy(data['domega_deta'])
        self.omega_true_raw = torch.from_numpy(data['omega_true'])

        # Per-sample pressure: (N, Nt)
        self.pressure = torch.from_numpy(data['pressure'])  # (N, Nt)

        # Parameter normalisation
        if param_mean is not None and param_std is not None:
            self.param_mean = param_mean
            self.param_std  = param_std
        else:
            self.param_mean = self.params_raw.mean(dim=0)
            self.param_std  = self.params_raw.std(dim=0)
        self.params = (self.params_raw - self.param_mean) / (self.param_std + 1e-8)

        # Omega normalisation
        if omega_mean is not None and omega_std is not None:
            self.omega_mean = omega_mean
            self.omega_std  = omega_std
        else:
            self.omega_mean = self.omega_true_raw.mean()
            self.omega_std  = self.omega_true_raw.std() + 1e-30

        self.omega_true = (self.omega_true_raw - self.omega_mean) / self.omega_std

    def __len__(self):
        return self.params.shape[0]

    def __getitem__(self, idx):
        return {
            'params':         self.params[idx],           # (3,)
            'params_raw':     self.params_raw[idx],       # (3,)
            'pressure':       self.pressure[idx],         # (Nt,) per-sample
            'omega_true':     self.omega_true[idx],       # (Nt,) normalised
            'omega_true_raw': self.omega_true_raw[idx],   # (Nt,) physical
            'domega_dE1':     self.domega_dE1[idx],
            'domega_dE2':     self.domega_dE2[idx],
            'domega_deta':    self.domega_deta[idx],
        }


def build_mixed_stage2_loaders(
    cfg: Config,
    stage1_ds: 'Stage1Dataset',
    n_samples: Optional[int] = None,
    loading_types: list = None,
    cache_dir: str = 'data_cache',
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build Stage II DataLoaders with mixed loading for training.
    Returns train/val/test loaders using MixedStage2Dataset.
    The test loader uses only PFWD loading for comparability with
    the single-type evaluate_generalization results.
    """
    import os

    n    = n_samples or cfg.data.n_total
    seed = cfg.data.seed
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'stage2_mixed_n{n}_seed{seed}.npz')

    if os.path.exists(cache_path):
        print(f"[Data] Loading mixed Stage II dataset from cache: {cache_path}")
        loaded = np.load(cache_path)
        data   = {k: loaded[k] for k in loaded.files}
        data['time'] = data['time'].flatten()
        # pressure is (N, Nt) — do NOT flatten
    else:
        print("[Data] Generating mixed Stage II dataset …")
        data = generate_mixed_stage2_dataset(
            cfg, n_samples=n, seed=seed, loading_types=loading_types,
        )
        np.savez_compressed(cache_path, **data)
        print(f"[Data] Saved mixed Stage II cache: {cache_path}")

    n_total = data['params'].shape[0]
    n_train = min(cfg.data.n_train, int(n_total * 0.80))
    n_val   = min(cfg.data.n_val,   int(n_total * 0.10))

    indices = np.arange(n_total)
    rng = np.random.RandomState(seed + 700)
    rng.shuffle(indices)

    def _slice(d, idx):
        return {k: v[idx] if (isinstance(v, np.ndarray) and v.ndim > 1) else v
                for k, v in d.items()}

    train_data = _slice(data, indices[:n_train])
    val_data   = _slice(data, indices[n_train:n_train + n_val])
    test_data  = _slice(data, indices[n_train + n_val:])

    pm = stage1_ds.param_mean
    ps = stage1_ds.param_std

    train_ds = MixedStage2Dataset(train_data, pm, ps)
    val_ds   = MixedStage2Dataset(val_data,   pm, ps,
                                   omega_mean=train_ds.omega_mean,
                                   omega_std=train_ds.omega_std)
    test_ds  = MixedStage2Dataset(test_data,  pm, ps,
                                   omega_mean=train_ds.omega_mean,
                                   omega_std=train_ds.omega_std)

    bs = cfg.training.stage2_batch_size
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False)

    print(f"[Data] Mixed Stage II — train: {len(train_ds)}, val: {len(val_ds)}, "
          f"test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


# ═══════════════════════════════════════════════════════════════════════
#  Convenience: split and wrap
# ═══════════════════════════════════════════════════════════════════════

def build_stage1_loaders(
    cfg: Config,
    n_samples: Optional[int] = None,
    cache_dir: str = 'data_cache',
) -> Tuple[DataLoader, DataLoader, DataLoader, 'Stage1Dataset']:
    """
    Generate (or load from cache) Stage I DataLoaders.

    Cache lives in cache_dir/stage1_n{n}_seed{seed}.npz
    """
    import os

    n    = n_samples or cfg.data.n_total
    seed = cfg.data.seed
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'stage1_n{n}_seed{seed}.npz')

    if os.path.exists(cache_path):
        print(f"[Data] Loading Stage I dataset from cache: {cache_path}")
        loaded = np.load(cache_path)
        data = {k: loaded[k] for k in loaded.files}
        data['time'] = data['time'].flatten()
    else:
        print("[Data] Generating Stage I impulse response dataset …")
        data = generate_stage1_dataset(cfg, n_samples=n_samples)
        np.savez_compressed(cache_path, **data)
        print(f"[Data] Saved Stage I cache: {cache_path}")

    n_total = data['params'].shape[0]
    n_train = min(cfg.data.n_train, int(n_total * 0.80))
    n_val   = min(cfg.data.n_val,   int(n_total * 0.10))

    indices = np.arange(n_total)
    rng = np.random.RandomState(cfg.data.seed)
    rng.shuffle(indices)

    def _slice(d, idx):
        return {k: v[idx] if v.ndim > 1 else v for k, v in d.items()}

    train_data = _slice(data, indices[:n_train])
    val_data   = _slice(data, indices[n_train:n_train + n_val])
    test_data  = _slice(data, indices[n_train + n_val:])

    train_ds = Stage1Dataset(train_data)
    val_ds = Stage1Dataset(
        val_data, f_mean=train_ds.f_mean, f_std=train_ds.f_std,
    )
    val_ds.param_mean = train_ds.param_mean
    val_ds.param_std  = train_ds.param_std
    val_ds.params     = (val_ds.params_raw - val_ds.param_mean) / (val_ds.param_std + 1e-8)

    test_ds = Stage1Dataset(
        test_data, f_mean=train_ds.f_mean, f_std=train_ds.f_std,
    )
    test_ds.param_mean = train_ds.param_mean
    test_ds.param_std  = train_ds.param_std
    test_ds.params     = (test_ds.params_raw - test_ds.param_mean) / (test_ds.param_std + 1e-8)

    bs = cfg.training.stage1_batch_size
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False)

    print(f"[Data] Stage I — train: {len(train_ds)}, val: {len(val_ds)}, "
          f"test: {len(test_ds)}")
    print(f"[Data] f_true normalisation — mean: {train_ds.f_mean:.4e}, "
          f"std: {train_ds.f_std:.4e}")

    return train_loader, val_loader, test_loader, train_ds


def build_stage2_loaders(
    cfg: Config,
    stage1_ds: Stage1Dataset,
    n_samples: Optional[int] = None,
    loading_type: str = 'pfwd',
    cache_dir: str = 'data_cache',
    omega_mean: Optional[torch.Tensor] = None,
    omega_std:  Optional[torch.Tensor] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Generate (or load from cache) Stage II DataLoaders.

    omega_mean, omega_std: if provided, all splits use these normalisation
    statistics instead of computing from this dataset.  Pass the mixed
    training set's stats when evaluating a model trained on mixed loading,
    so that model outputs are denormalised consistently.
    """
    import os

    n       = n_samples or cfg.data.n_total
    seed    = cfg.data.seed
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir, f'stage2_{loading_type}_n{n}_seed{seed}.npz'
    )

    if os.path.exists(cache_path):
        print(f"[Data] Loading Stage II dataset ({loading_type}) from cache: {cache_path}")
        loaded = np.load(cache_path)
        data = {k: loaded[k] for k in loaded.files}
        data['time']     = data['time'].flatten()
        data['pressure'] = data['pressure'].flatten()
    else:
        print(f"[Data] Generating Stage II dataset ({loading_type}) …")
        data = generate_stage2_dataset(cfg, n_samples=n_samples,
                                       loading_type=loading_type)
        np.savez_compressed(cache_path, **data)
        print(f"[Data] Saved Stage II cache: {cache_path}")

    n_total = data['params'].shape[0]
    n_train = min(cfg.data.n_train, int(n_total * 0.80))
    n_val   = min(cfg.data.n_val,   int(n_total * 0.10))

    indices = np.arange(n_total)
    rng = np.random.RandomState(cfg.data.seed + 500)
    rng.shuffle(indices)

    def _slice(d, idx):
        return {k: v[idx] if v.ndim > 1 else v for k, v in d.items()}

    train_data = _slice(data, indices[:n_train])
    val_data   = _slice(data, indices[n_train:n_train + n_val])
    test_data  = _slice(data, indices[n_train + n_val:])

    pm = stage1_ds.param_mean
    ps = stage1_ds.param_std

    # If external omega stats provided (mixed-trained model), use them for
    # all splits so denormalisation is consistent with the training distribution.
    if omega_mean is not None and omega_std is not None:
        train_ds = Stage2Dataset(train_data, pm, ps, omega_mean=omega_mean, omega_std=omega_std)
        val_ds   = Stage2Dataset(val_data,   pm, ps, omega_mean=omega_mean, omega_std=omega_std)
        test_ds  = Stage2Dataset(test_data,  pm, ps, omega_mean=omega_mean, omega_std=omega_std)
    else:
        train_ds = Stage2Dataset(train_data, pm, ps)
        val_ds   = Stage2Dataset(val_data,   pm, ps,
                                 omega_mean=train_ds.omega_mean,
                                 omega_std=train_ds.omega_std)
        test_ds  = Stage2Dataset(test_data,  pm, ps,
                                 omega_mean=train_ds.omega_mean,
                                 omega_std=train_ds.omega_std)

    bs = cfg.training.stage2_batch_size
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False)

    print(f"[Data] Stage II — train: {len(train_ds)}, val: {len(val_ds)}, "
          f"test: {len(test_ds)}")

    return train_loader, val_loader, test_loader