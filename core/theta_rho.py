"""
theta_rho.py — Density potential temperature thermodynamics.

Provides vectorized functions for computing and manipulating the
density potential temperature θ_ρ, which is the primary thermodynamic
prognostic variable in the Toró 3D anelastic model.

θ_ρ ≡ θ · (1 + R_v/R_d · q_v − q_l − q_i)

All functions accept and return arrays of arbitrary shape (including
3-D fields of shape (nx, ny, nz)).

References:
    - Emanuel (1994): Atmospheric Convection, ch. 4 & 6.
    - Bryan & Fritsch (2002): A benchmark simulation for moist
      nonhydrostatic numerical models, MWR.
    - Bolton (1980): The computation of equivalent potential
      temperature, MWR.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from core.constants import (
    g,
    c_p,
    R_d,
    R_v,
    L_v,
    L_f,
    p_0,
    T_0,
    epsilon,
    e_s0,
    BOLTON_A,
    BOLTON_B,
)


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
ArrayLike = NDArray[np.floating]


# ---------------------------------------------------------------------------
# Core conversions
# ---------------------------------------------------------------------------

def compute_theta(T: ArrayLike, p: ArrayLike) -> ArrayLike:
    """Potential temperature θ = T · (p₀/p)^(R_d/c_p).

    Parameters
    ----------
    T : ndarray
        Temperature (K).
    p : ndarray
        Pressure (Pa), broadcastable with *T*.

    Returns
    -------
    ndarray
        Potential temperature (K), same shape as broadcast(T, p).
    """
    return T * (p_0 / p) ** (R_d / c_p)


def compute_theta_rho(
    theta: ArrayLike,
    qv: ArrayLike,
    ql: ArrayLike | float = 0.0,
    qi: ArrayLike | float = 0.0,
) -> ArrayLike:
    """Density potential temperature.

    θ_ρ = θ · (1 + R_v/R_d · q_v − q_l − q_i)

    Accounts for buoyancy contributions from water vapour (lighter)
    and condensate loading (heavier).

    Parameters
    ----------
    theta : ndarray
        Potential temperature (K).
    qv : ndarray
        Water-vapour mixing ratio (kg/kg).
    ql : ndarray or float, optional
        Liquid-water mixing ratio (kg/kg).  Default 0.
    qi : ndarray or float, optional
        Ice mixing ratio (kg/kg).  Default 0.

    Returns
    -------
    ndarray
        Density potential temperature (K).
    """
    return theta * (1.0 + (R_v / R_d) * qv - ql - qi)


# ---------------------------------------------------------------------------
# Buoyancy
# ---------------------------------------------------------------------------

def compute_buoyancy_3d(
    theta_rho: ArrayLike,
    theta_rho_bar_z: ArrayLike,
) -> ArrayLike:
    """Buoyancy from density potential temperature perturbation.

    B = g · (θ_ρ − θ̄_ρ) / θ̄_ρ

    The 1-D base-state profile θ̄_ρ(z) is automatically broadcast
    along the z-axis (last axis) to match the 3-D field.

    Parameters
    ----------
    theta_rho : ndarray, shape (nx, ny, nz)
        Full (base + perturbation) density potential temperature.
    theta_rho_bar_z : ndarray, shape (nz,)
        Base-state density potential temperature profile.

    Returns
    -------
    ndarray, shape (nx, ny, nz)
        Buoyancy (m/s²).  Positive ⇒ upward.
    """
    # Broadcast (nz,) → (1, 1, nz) for arithmetic with (nx, ny, nz).
    theta_rho_bar = theta_rho_bar_z[np.newaxis, np.newaxis, :]
    return g * (theta_rho - theta_rho_bar) / theta_rho_bar


# ---------------------------------------------------------------------------
# Inverse: recover temperature from θ_ρ
# ---------------------------------------------------------------------------

def theta_rho_to_T(
    theta_rho: ArrayLike,
    exner: ArrayLike,
    qv: ArrayLike,
    ql: ArrayLike | float = 0.0,
    qi: ArrayLike | float = 0.0,
) -> ArrayLike:
    """Recover temperature from density potential temperature.

    T = θ_ρ · Π / (1 + R_v/R_d · q_v − q_l − q_i)

    where Π is the Exner function.

    Parameters
    ----------
    theta_rho : ndarray
        Density potential temperature (K).
    exner : ndarray
        Exner function Π = (p/p₀)^(R_d/c_p), broadcastable.
    qv : ndarray
        Water-vapour mixing ratio (kg/kg).
    ql : ndarray or float, optional
        Liquid-water mixing ratio (kg/kg).
    qi : ndarray or float, optional
        Ice mixing ratio (kg/kg).

    Returns
    -------
    ndarray
        Temperature (K).
    """
    moisture_factor = 1.0 + (R_v / R_d) * qv - ql - qi
    return theta_rho * exner / moisture_factor


# ---------------------------------------------------------------------------
# Saturation mixing ratio (Bolton 1980)
# ---------------------------------------------------------------------------

def compute_qvs(T: ArrayLike, p: ArrayLike) -> ArrayLike:
    """Saturation mixing ratio over liquid water (Bolton 1980).

    e_s(T) = 611.2 · exp(17.67 · T_c / (T_c + 243.5))
    q_vs   = ε · e_s / (p − e_s)

    Accurate to ±0.1 % for −35 °C ≤ T ≤ +35 °C.

    Parameters
    ----------
    T : ndarray
        Temperature (K).
    p : ndarray
        Pressure (Pa), broadcastable with *T*.

    Returns
    -------
    ndarray
        Saturation mixing ratio (kg/kg).
    """
    T_c = T - T_0  # Convert to Celsius
    e_s = e_s0 * np.exp(BOLTON_A * T_c / (T_c + BOLTON_B))
    return epsilon * e_s / (p - e_s)


# ---------------------------------------------------------------------------
# Latent heating tendency
# ---------------------------------------------------------------------------

def latent_heating_theta(
    dq_cond: ArrayLike,
    dq_freeze: ArrayLike,
    exner: ArrayLike,
) -> ArrayLike:
    """Latent-heat tendency for density potential temperature.

    dθ_ρ/dt ≈ (L_v / (c_p · Π)) · dq_cond + (L_f / (c_p · Π)) · dq_freeze

    The Exner function Π converts the heating rate from temperature
    space to potential-temperature space.

    Parameters
    ----------
    dq_cond : ndarray
        Condensation/evaporation rate (kg/kg/s).
        Positive ⇒ condensation (heating).
    dq_freeze : ndarray
        Freezing/melting rate (kg/kg/s).
        Positive ⇒ freezing (heating).
    exner : ndarray
        Exner function Π, broadcastable with the rate fields.

    Returns
    -------
    ndarray
        Tendency dθ_ρ/dt (K/s) from latent heat release.
    """
    return (L_v / (c_p * exner)) * dq_cond + (L_f / (c_p * exner)) * dq_freeze
