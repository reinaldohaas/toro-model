"""
microphysics_bulk.py — Bulk microphysics for the 3D Toró model.

Six water categories with fully vectorized (NumPy) operations over 3D grids.
No Python loops over grid points.

Categories:
    q_v : water vapor mixing ratio        (kg/kg)
    q_c : cloud water mixing ratio        (kg/kg)
    q_r : rain mixing ratio               (kg/kg)
    q_i : cloud ice mixing ratio          (kg/kg)
    q_s : snow mixing ratio               (kg/kg)
    q_g : graupel/hail mixing ratio       (kg/kg)

Processes:
    1. Saturation adjustment (condensation / evaporation)
    2. Autoconversion  (cloud → rain, Kessler 1969)
    3. Accretion       (rain collecting cloud water)
    4. Freezing         (heterogeneous + homogeneous)
    5. Riming          (cloud water accreted by graupel/ice)
    6. Secondary ice production (Hallett-Mossop + collisional breakup)
    7. Sedimentation   (1st-order upwind fall-speed flux)
    8. Melting          (ice / snow / graupel above 0 °C)

References:
    - Kessler (1969): On the Distribution and Continuity of Water
      Substance in Atmospheric Circulations, Meteor. Monogr.
    - Hallett & Mossop (1974): Nature 249, 26-28
    - Phillips et al. (2017): J. Atmos. Sci.
    - Lin, Farley & Orville (1983): JCAM — bulk ice scheme
    - Rutledge & Hobbs (1984): J. Atmos. Sci.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from core.constants import (
    g, L_v, L_f, T_0, rho_water, rho_ice,
    c_p, R_d, R_v, epsilon,
    e_s0, BOLTON_A, BOLTON_B,
)

# Type alias for 3-D float arrays
F3D = NDArray[np.floating]


# ============================================================================
# Physical / tuneable constants (module-level for easy override)
# ============================================================================

# Autoconversion (Kessler 1969)
K_AUTO: float = 1.0e-3          # 1/s — autoconversion rate
QC_THRESHOLD: float = 1.0e-3   # kg/kg — cloud water threshold

# Accretion
E_COLL: float = 0.8            # collection efficiency
K_ACCR: float = 2.2            # accretion rate constant  (dimensionless prefactor)

# Riming
K_RIME: float = 2.2            # riming rate constant (same kernel form as accretion)

# Heterogeneous freezing (immersion, Bigg 1953 style simplified)
B_HET: float = 0.66            # K⁻¹ — exponential slope for het. freezing
A_HET: float = 1.0e-4          # 1/s — base freezing rate at 0 °C

# Hallett-Mossop SIP
HM_RATE: float = 350.0         # splinters per mg of rime accreted
HM_T_MIN: float = -8.0         # °C — cold limit
HM_T_MAX: float = -3.0         # °C — warm limit
HM_T_PEAK: float = -5.0        # °C — peak efficiency
HM_SPLINTER_D: float = 10.0e-6 # m — assumed splinter diameter

# Collisional breakup (Phillips 2017 — bulk parameterisation)
K_BREAKUP: float = 1.0e-3      # 1/s — effective breakup coefficient
COLL_T_PEAK: float = -15.0     # °C — peak temperature for g(T)
COLL_T_WIDTH: float = 10.0     # °C — width of Gaussian g(T)

# Terminal fall velocities
VT_RAIN_COEFF: float = 5.0     # m/s — V_t,rain prefactor
VT_RAIN_EXP: float = 0.125     # exponent on q_r
VT_SNOW: float = 1.0           # m/s — constant
VT_GRAUPEL_COEFF: float = 8.0  # m/s — V_t,graupel prefactor
VT_GRAUPEL_EXP: float = 0.125  # exponent on q_g

# Small floor to avoid division-by-zero and log-of-zero
_QMIN: float = 1.0e-20


# ============================================================================
# Helper: saturation vapour pressure and mixing ratio
# ============================================================================

def _e_sat(T: F3D) -> F3D:
    """Bolton (1980) saturation vapour pressure over liquid water (Pa).

    e_s = 611.2 · exp(17.67·(T − 273.15) / (T − 273.15 + 243.5))
    """
    T_c = T - T_0
    return e_s0 * np.exp(BOLTON_A * T_c / (T_c + BOLTON_B))


def _q_vs(T: F3D, p: F3D) -> F3D:
    """Saturation mixing ratio over liquid water (kg/kg)."""
    es = _e_sat(T)
    # Clip denominator away from zero (avoid blow-up near surface)
    denom = np.maximum(p - es, 1.0)
    return epsilon * es / denom


# ============================================================================
# 1. Saturation adjustment (condensation / evaporation)
# ============================================================================

def _saturation_adjustment(
    T: F3D, p: F3D, qv: F3D, qc: F3D, dt: float,
) -> tuple[F3D, F3D, F3D, F3D]:
    """One-step saturation adjustment.

    * If q_v > q_vs: excess condenses → q_c, releases L_v → ΔT.
    * If q_v < q_vs and q_c > 0: evaporate cloud water.

    Uses the Newton-iteration form (single step is sufficient for
    small dt typical of cloud models).

    Returns
    -------
    T_new, qv_new, qc_new, dq_cond
        dq_cond > 0 means net condensation (kg/kg per step).
    """
    qvs = _q_vs(T, p)
    excess = qv - qvs  # >0 → supersaturated, <0 → sub-saturated

    # Clausius-Clapeyron denominator for Newton correction
    # d(q_vs)/dT ≈ q_vs · L_v / (R_v · T²)
    dqvs_dT = qvs * L_v / (R_v * T * T)

    # Condensation / evaporation increment  (kg/kg)
    # Δq = excess / (1 + L_v/(c_p) · dq_vs/dT)
    dq_cond = excess / (1.0 + (L_v / c_p) * dqvs_dT)

    # Cannot evaporate more cloud water than exists
    dq_cond = np.maximum(dq_cond, -qc)

    qv_new = qv - dq_cond
    qc_new = qc + dq_cond
    T_new  = T  + (L_v / c_p) * dq_cond

    # Enforce non-negativity
    qv_new = np.maximum(qv_new, 0.0)
    qc_new = np.maximum(qc_new, 0.0)

    return T_new, qv_new, qc_new, dq_cond


# ============================================================================
# 2. Autoconversion  (Kessler 1969)
# ============================================================================

def _autoconversion(qc: F3D, dt: float) -> tuple[F3D, F3D, F3D]:
    """Cloud water → rain via autoconversion.

    dq_r/dt = max(0, k_auto · (q_c − q_c_threshold))

    Returns
    -------
    qc_new, qr_increment, rate (kg/kg per step)
    """
    rate = np.maximum(K_AUTO * (qc - QC_THRESHOLD), 0.0) * dt
    # Cannot convert more than available
    rate = np.minimum(rate, qc)
    qc_new = qc - rate
    return qc_new, rate, rate


# ============================================================================
# 3. Accretion  (rain collecting cloud water)
# ============================================================================

def _accretion(qc: F3D, qr: F3D, dt: float) -> tuple[F3D, F3D, F3D]:
    """Rain collecting cloud droplets.

    dq_r/dt = E_coll · q_c · q_r^0.875 · K_ACCR

    Returns
    -------
    qc_new, qr_new, rate (kg/kg per step)
    """
    qr_safe = np.maximum(qr, _QMIN)
    rate = E_COLL * qc * np.power(qr_safe, 0.875) * K_ACCR * dt
    rate = np.minimum(rate, qc)  # can't collect more than exists
    rate = np.where(qr > _QMIN, rate, 0.0)

    qc_new = qc - rate
    qr_new = qr + rate
    return qc_new, qr_new, rate


# ============================================================================
# 4. Freezing  (heterogeneous + homogeneous)
# ============================================================================

def _freezing(
    T: F3D, qc: F3D, qr: F3D, qi: F3D, qg: F3D, dt: float,
) -> tuple[F3D, F3D, F3D, F3D, F3D, F3D]:
    """Freezing of liquid hydrometeors.

    * Heterogeneous: below 0 °C, fraction of q_c → q_i at rate
      depending on supercooling  (Bigg-like exponential).
    * Homogeneous: below −38 °C, instantaneous q_c → q_i.
    * Rain freezing: q_r → q_g below −10 °C.

    Returns
    -------
    T_new, qc_new, qr_new, qi_new, qg_new, dq_freeze_total
    """
    T_c = T - T_0  # °C (3-D)

    # ---- heterogeneous freezing of cloud water (gradual) ----
    supercool = np.maximum(-T_c, 0.0)  # degrees below 0 °C
    frac_het = np.minimum(A_HET * np.exp(B_HET * supercool) * dt, 1.0)
    # Only active below 0 °C
    frac_het = np.where(T_c < 0.0, frac_het, 0.0)
    dq_het = frac_het * qc

    # ---- homogeneous freezing of cloud water (instantaneous below −38 °C) ----
    dq_homo = np.where(T_c < -38.0, qc - dq_het, 0.0)
    # (remaining qc after heterogeneous already subtracted conceptually)
    # Combine: total cloud water frozen
    dq_cloud_freeze = np.minimum(dq_het + dq_homo, qc)

    # ---- rain freezing → graupel below −10 °C ----
    rain_freeze_frac = np.where(T_c < -10.0, 1.0, 0.0)
    dq_rain_freeze = rain_freeze_frac * qr

    # Apply
    qc_new = qc - dq_cloud_freeze
    qi_new = qi + dq_cloud_freeze
    qr_new = qr - dq_rain_freeze
    qg_new = qg + dq_rain_freeze

    dq_freeze_total = dq_cloud_freeze + dq_rain_freeze

    # Latent heat of fusion release
    T_new = T + (L_f / c_p) * dq_freeze_total

    # Enforce non-negativity
    qc_new = np.maximum(qc_new, 0.0)
    qr_new = np.maximum(qr_new, 0.0)

    return T_new, qc_new, qr_new, qi_new, qg_new, dq_freeze_total


# ============================================================================
# 5. Riming  (cloud water accreted by graupel / ice)
# ============================================================================

def _riming(
    T: F3D, qc: F3D, qg: F3D, dt: float,
) -> tuple[F3D, F3D, F3D, F3D]:
    """Riming: cloud water accreted onto graupel (below 0 °C).

    dm_rime/dt = E_coll · q_c · q_g^0.875 · K_RIME

    Releases L_f (liquid → ice).

    Returns
    -------
    T_new, qc_new, qg_new, dm_rime  (kg/kg per step)
    """
    is_cold = (T < T_0)

    qg_safe = np.maximum(qg, _QMIN)
    dm_rime = E_COLL * qc * np.power(qg_safe, 0.875) * K_RIME * dt
    dm_rime = np.minimum(dm_rime, qc)
    dm_rime = np.where(is_cold & (qg > _QMIN), dm_rime, 0.0)

    qc_new = qc - dm_rime
    qg_new = qg + dm_rime

    # Latent heat release (liquid → ice)
    T_new = T + (L_f / c_p) * dm_rime

    return T_new, qc_new, qg_new, dm_rime


# ============================================================================
# 6a. Hallett-Mossop secondary ice production
# ============================================================================

def _hallett_mossop(
    T: F3D, dm_rime: F3D, qi: F3D, qc: F3D, dt: float,
) -> tuple[F3D, F3D, F3D]:
    """Hallett-Mossop rime-splintering SIP.

    Active in the temperature window −8 °C < T < −3 °C.

    dN_sip/dt = 350 splinters / mg_rime  × f(T) × dm_rime/dt
    f(T) is a triangular function peaking at −5 °C.

    Splinters are converted to q_i mass assuming spherical ice
    crystals of diameter HM_SPLINTER_D = 10 µm.

    Parameters
    ----------
    dm_rime : riming increment this step (kg/kg).

    Returns
    -------
    qi_new, dq_sip (kg/kg per step), sip_number_rate (#/kg/s)
    """
    T_c = T - T_0  # °C

    # Triangular f(T): rises linearly from T_MAX (−3) to T_PEAK (−5),
    # then falls linearly to T_MIN (−8).
    f_warm = (T_c - HM_T_MAX) / (HM_T_PEAK - HM_T_MAX)  # 0 at −3, 1 at −5
    f_cold = (T_c - HM_T_MIN) / (HM_T_PEAK - HM_T_MIN)  # 0 at −8, 1 at −5
    f_T = np.where(T_c >= HM_T_PEAK, f_warm, f_cold)
    f_T = np.clip(f_T, 0.0, 1.0)

    # Mask: only active inside the H-M window
    active = (T_c > HM_T_MIN) & (T_c < HM_T_MAX)
    f_T = np.where(active, f_T, 0.0)

    # Number of splinters produced:
    # dm_rime is kg/kg per step  →  convert to mg/kg:  ×1e6
    # HM_RATE is splinters / mg
    dN_sip = HM_RATE * f_T * (dm_rime * 1.0e6)  # splinters per kg_air per step

    # Mass per splinter: sphere of diameter HM_SPLINTER_D, density rho_ice
    m_splinter = (np.pi / 6.0) * rho_ice * HM_SPLINTER_D ** 3  # kg

    # SIP mass increment
    dq_sip = dN_sip * m_splinter  # kg/kg per step

    # Cannot create more ice than cloud water available (conservation)
    dq_sip = np.minimum(dq_sip, qc)
    dq_sip = np.maximum(dq_sip, 0.0)

    qi_new = qi + dq_sip

    # Number rate for diagnostics  (#/kg_air/s)
    sip_rate = np.where(dt > 0, dN_sip / dt, 0.0)

    return qi_new, dq_sip, sip_rate


# ============================================================================
# 6b. Collisional breakup  (Phillips 2017 — bulk parameterisation)
# ============================================================================

def _collisional_breakup(
    T: F3D, qg: F3D, qi: F3D, dt: float,
) -> tuple[F3D, F3D]:
    """Collisional breakup of ice (simplified Phillips 2017).

    q_i_new = k_breakup · q_g² · g(T) · dt
    g(T) = exp(−((T_c + 15) / 10)²), peak at −15 °C.

    Active only below 0 °C.

    Returns
    -------
    qi_new, dq_breakup (kg/kg per step)
    """
    T_c = T - T_0

    # g(T): Gaussian centred on COLL_T_PEAK with width COLL_T_WIDTH
    g_T = np.exp(-((T_c - COLL_T_PEAK) / COLL_T_WIDTH) ** 2)

    # Only below freezing
    g_T = np.where(T_c < 0.0, g_T, 0.0)

    dq_breakup = K_BREAKUP * qg * qg * g_T * dt
    dq_breakup = np.maximum(dq_breakup, 0.0)

    qi_new = qi + dq_breakup
    return qi_new, dq_breakup


# ============================================================================
# 7. Sedimentation  (1st-order upwind in z)
# ============================================================================

def _sedimentation(
    qr: F3D, qs: F3D, qg: F3D, rho: F3D, dz: float, dt: float,
) -> tuple[F3D, F3D, F3D, F3D]:
    """First-order upwind sedimentation along the z-axis (axis=2).

    Terminal fall velocities:
        V_t,rain    = VT_RAIN_COEFF    · q_r^0.125   (m/s)
        V_t,snow    = VT_SNOW                         (m/s)
        V_t,graupel = VT_GRAUPEL_COEFF · q_g^0.125   (m/s)

    The flux-form update uses upwind differencing:
        dq/dt = V_t · (q[k+1] − q[k]) / dz   (falling downward, k=0 is top)

    Convention: z-axis is axis=2, index 0 = lowest level (surface),
    index nz−1 = model top.  Particles fall *downward*, so flux comes
    from the level above (k+1 → k).

    Uses np.roll for periodic boundaries (top boundary effectively
    receives zero flux because q ≈ 0 there).

    Returns
    -------
    qr_new, qs_new, qg_new, precip_rate (kg/m²/s at surface)
    """
    # ---- rain ----
    qr_safe = np.maximum(qr, _QMIN)
    vt_r = VT_RAIN_COEFF * np.power(qr_safe, VT_RAIN_EXP)
    vt_r = np.where(qr > _QMIN, vt_r, 0.0)

    # ---- snow ----
    vt_s = np.where(qs > _QMIN, VT_SNOW, 0.0)

    # ---- graupel ----
    qg_safe = np.maximum(qg, _QMIN)
    vt_g = VT_GRAUPEL_COEFF * np.power(qg_safe, VT_GRAUPEL_EXP)
    vt_g = np.where(qg > _QMIN, vt_g, 0.0)

    # CFL limiter: ensure V_t * dt / dz <= 1
    cfl_max = dz / max(dt, 1.0e-10)
    vt_r = np.minimum(vt_r, cfl_max)
    vt_s = np.minimum(vt_s, cfl_max)
    vt_g = np.minimum(vt_g, cfl_max)

    # Upwind: flux from level above (k+1) into level k.
    # np.roll(q, -1, axis=2) gives q[..., k+1] at position k.
    # At the top boundary (k=nz-1), roll wraps to k=0; we zero that flux.
    def _upwind_sed(q: F3D, vt: F3D) -> F3D:
        q_above = np.roll(q, -1, axis=2)
        # Zero the flux at the top boundary (wrapped value is meaningless)
        q_above[:, :, -1] = 0.0
        flux_in = vt * q_above / dz
        flux_out = vt * q / dz
        return q + (flux_in - flux_out) * dt

    qr_new = np.maximum(_upwind_sed(qr, vt_r), 0.0)
    qs_new = np.maximum(_upwind_sed(qs, vt_s), 0.0)
    qg_new = np.maximum(_upwind_sed(qg, vt_g), 0.0)

    # Surface precipitation rate (kg/m²/s):
    # Mass flux through the lowest level  =  rho · q · V_t  at k=0
    rho_sfc = rho[:, :, 0]
    precip_rain = rho_sfc * qr[:, :, 0] * vt_r[:, :, 0]
    precip_snow = rho_sfc * qs[:, :, 0] * vt_s[:, :, 0]
    precip_grau = rho_sfc * qg[:, :, 0] * vt_g[:, :, 0]
    precip_rate = precip_rain + precip_snow + precip_grau  # 2-D (nx, ny)

    return qr_new, qs_new, qg_new, precip_rate


# ============================================================================
# 8. Melting  (ice / snow / graupel above 0 °C)
# ============================================================================

def _melting(
    T: F3D, qi: F3D, qs: F3D, qg: F3D, qc: F3D, qr: F3D, dt: float,
) -> tuple[F3D, F3D, F3D, F3D, F3D, F3D]:
    """Melt frozen hydrometeors when T > 0 °C.

    * q_i → q_c  (cloud ice melts to cloud water)
    * q_s → q_r  (snow melts to rain)
    * q_g → q_r  (graupel melts to rain)

    Melting consumes L_f → cools the air.

    Returns
    -------
    T_new, qi_new, qs_new, qg_new, qc_new, qr_new
    """
    is_warm = (T > T_0)

    # Melting rate: simple — melt everything above 0 °C within one step.
    # (For finite-rate melting, scale by the available thermal energy,
    #  but instantaneous melting is standard in Kessler-type schemes.)
    # Available thermal energy per unit mass of air for melting:
    #   dT_available = T − T_0
    #   dq_melt_max  = c_p · dT_available / L_f
    dT_avail = np.maximum(T - T_0, 0.0)
    dq_melt_max = c_p * dT_avail / L_f  # max mass that can melt (kg/kg)

    # Total frozen mass
    q_frozen = qi + qs + qg
    # Actual melt limited by energy and available ice
    dq_melt = np.minimum(q_frozen, dq_melt_max)
    dq_melt = np.where(is_warm, dq_melt, 0.0)

    # Distribute melt proportionally among categories
    safe_denom = np.maximum(q_frozen, _QMIN)

    frac_i = qi / safe_denom
    frac_s = qs / safe_denom
    frac_g = qg / safe_denom

    dq_i_melt = dq_melt * frac_i
    dq_s_melt = dq_melt * frac_s
    dq_g_melt = dq_melt * frac_g

    # Cloud ice → cloud water; snow + graupel → rain
    qi_new = qi - dq_i_melt
    qs_new = qs - dq_s_melt
    qg_new = qg - dq_g_melt
    qc_new = qc + dq_i_melt
    qr_new = qr + dq_s_melt + dq_g_melt

    # Cooling from latent heat absorption
    T_new = T - (L_f / c_p) * dq_melt

    # Enforce non-negativity
    qi_new = np.maximum(qi_new, 0.0)
    qs_new = np.maximum(qs_new, 0.0)
    qg_new = np.maximum(qg_new, 0.0)

    return T_new, qi_new, qs_new, qg_new, qc_new, qr_new


# ============================================================================
# Main driver
# ============================================================================

def step_microphysics_bulk(
    T: F3D,
    p: F3D,
    qv: F3D,
    qc: F3D,
    qr: F3D,
    qi: F3D,
    qs: F3D,
    qg: F3D,
    rho: F3D,
    dt: float,
    dz: float = 300.0,
) -> dict[str, F3D]:
    """Execute one full bulk-microphysics time step.

    All input arrays are 3-D with shape (nx, ny, nz).
    All operations are fully vectorized (no Python loops over grid points).

    Parameters
    ----------
    T   : temperature (K)
    p   : pressure (Pa)
    qv  : water-vapour mixing ratio (kg/kg)
    qc  : cloud-water mixing ratio (kg/kg)
    qr  : rain mixing ratio (kg/kg)
    qi  : cloud-ice mixing ratio (kg/kg)
    qs  : snow mixing ratio (kg/kg)
    qg  : graupel/hail mixing ratio (kg/kg)
    rho : air density (kg/m³)
    dt  : time step (s)
    dz  : vertical grid spacing (m), default 300 m.

    Returns
    -------
    dict with keys:
        'qv', 'qc', 'qr', 'qi', 'qs', 'qg' — updated mixing ratios
        'dq_cond'    — condensation rate (kg/kg per step, >0 = condensation)
        'dq_freeze'  — total freezing rate (kg/kg per step)
        'sip_rate'   — SIP number production rate (#/kg_air/s)
        'dm_rime'    — riming increment (kg/kg per step)
        'precip_rate'— surface precipitation (kg/m²/s), shape (nx, ny)
    """
    # Work on copies so the caller's arrays are not mutated
    T  = T.copy()
    qv = qv.copy()
    qc = qc.copy()
    qr = qr.copy()
    qi = qi.copy()
    qs = qs.copy()
    qg = qg.copy()

    # Accumulators for diagnostic totals
    dq_freeze_total = np.zeros_like(T)

    # ------------------------------------------------------------------
    # 1. Saturation adjustment (condensation / evaporation)
    # ------------------------------------------------------------------
    T, qv, qc, dq_cond = _saturation_adjustment(T, p, qv, qc, dt)

    # ------------------------------------------------------------------
    # 2. Autoconversion (cloud → rain)
    # ------------------------------------------------------------------
    qc, dq_auto, _ = _autoconversion(qc, dt)
    qr = qr + dq_auto

    # ------------------------------------------------------------------
    # 3. Accretion (rain collecting cloud water)
    # ------------------------------------------------------------------
    qc, qr, _ = _accretion(qc, qr, dt)

    # ------------------------------------------------------------------
    # 4. Freezing (heterogeneous + homogeneous)
    # ------------------------------------------------------------------
    T, qc, qr, qi, qg, dq_frz = _freezing(T, qc, qr, qi, qg, dt)
    dq_freeze_total = dq_freeze_total + dq_frz

    # ------------------------------------------------------------------
    # 5. Riming (cloud water → graupel, releases L_f)
    # ------------------------------------------------------------------
    T, qc, qg, dm_rime = _riming(T, qc, qg, dt)
    dq_freeze_total = dq_freeze_total + dm_rime

    # ------------------------------------------------------------------
    # 6a. Hallett-Mossop SIP
    # ------------------------------------------------------------------
    qi, dq_sip, sip_rate = _hallett_mossop(T, dm_rime, qi, qc, dt)

    # ------------------------------------------------------------------
    # 6b. Collisional breakup
    # ------------------------------------------------------------------
    qi, dq_breakup = _collisional_breakup(T, qg, qi, dt)

    # ------------------------------------------------------------------
    # 7. Sedimentation (rain, snow, graupel)
    # ------------------------------------------------------------------
    qr, qs, qg, precip_rate = _sedimentation(qr, qs, qg, rho, dz, dt)

    # ------------------------------------------------------------------
    # 8. Melting (above 0 °C)
    # ------------------------------------------------------------------
    T, qi, qs, qg, qc, qr = _melting(T, qi, qs, qg, qc, qr, dt)

    # ------------------------------------------------------------------
    # Final sanitisation: clip negatives and NaNs
    # ------------------------------------------------------------------
    for q in (qv, qc, qr, qi, qs, qg):
        np.maximum(q, 0.0, out=q)
        np.nan_to_num(q, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        'qv': qv,
        'qc': qc,
        'qr': qr,
        'qi': qi,
        'qs': qs,
        'qg': qg,
        'dq_cond': dq_cond,
        'dq_freeze': dq_freeze_total,
        'sip_rate': sip_rate,
        'dm_rime': dm_rime,
        'precip_rate': precip_rate,
    }
