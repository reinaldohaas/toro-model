"""
grid3d.py — 3D Cartesian grid for anelastic atmospheric simulation.

Manages grid coordinates, base-state thermodynamic profiles, and
spatial finite-difference operators for the Toró model.

Base state: standard tropical/subtropical atmosphere with lapse rate γ,
isothermal above the tropopause.  Hydrostatic pressure integration is
performed analytically within each layer.

Boundary conditions:
    x, y — periodic (implemented via np.roll)
    z    — zero-gradient (Neumann) at bottom and top

All operations are NumPy-vectorized.  NO Python loops over grid points.

References:
    - Klemp & Wilhelmson (1978): Simulation of three-dimensional
      convective storm dynamics, JAS.
    - Bolton (1980): MWR — saturation formulas.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from core.constants import g, c_p, R_d, R_v, p_0, T_0, epsilon


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Float3D = NDArray[np.floating]


class Grid3D:
    """3D Cartesian grid with anelastic base-state profiles.

    Parameters
    ----------
    nx, ny, nz : int
        Number of grid points in each direction.
    dx, dy, dz : float
        Grid spacing (m) in each direction.
    T_sfc : float
        Surface temperature (K).
    p_sfc : float
        Surface pressure (Pa).
    gamma : float
        Tropospheric lapse rate (K/m).
    z_tropopause : float
        Tropopause height (m).
    T_tropopause : float
        Minimum temperature (isothermal cap, K).
    RH_sfc : float
        Surface relative humidity (0–1).

    Attributes
    ----------
    x, y, z : NDArray, shape (nx,), (ny,), (nz,)
        1-D coordinate arrays (cell centres).
    X, Y, Z : NDArray, shape (nx, ny, nz)
        3-D meshgrid arrays (indexing='ij').
    T_bar_z, p_bar_z, rho_bar_z : NDArray, shape (nz,)
        1-D base-state temperature, pressure, density profiles.
    theta_bar_z, theta_rho_bar_z : NDArray, shape (nz,)
        1-D base-state potential temperature and density potential
        temperature profiles.
    exner_z : NDArray, shape (nz,)
        1-D Exner function profile Π(z).
    qv_bar_z : NDArray, shape (nz,)
        1-D base-state water-vapour mixing ratio (kg/kg).
    """

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        nx: int = 20,
        ny: int = 20,
        nz: int = 50,
        dx: float = 500.0,
        dy: float = 500.0,
        dz: float = 300.0,
        T_sfc: float = 300.0,
        p_sfc: float = 101325.0,
        gamma: float = 6.5e-3,
        z_tropopause: float = 12000.0,
        T_tropopause: float = 210.0,
        RH_sfc: float = 0.85,
    ) -> None:
        # --- store grid parameters ---
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz

        # --- store atmosphere parameters ---
        self.T_sfc = T_sfc
        self.p_sfc = p_sfc
        self.gamma = gamma
        self.z_tropopause = z_tropopause
        self.T_tropopause = T_tropopause
        self.RH_sfc = RH_sfc

        # --- 1-D coordinate arrays (cell centres) ---
        self.x: NDArray = np.arange(nx) * dx  # (nx,)
        self.y: NDArray = np.arange(ny) * dy  # (ny,)
        self.z: NDArray = np.arange(nz) * dz  # (nz,)

        # --- 3-D meshgrids (indexing='ij' → shape (nx, ny, nz)) ---
        self.X: Float3D
        self.Y: Float3D
        self.Z: Float3D
        self.X, self.Y, self.Z = np.meshgrid(
            self.x, self.y, self.z, indexing="ij"
        )

        # --- build base-state profiles (all vectorized, 1-D in z) ---
        self._build_base_state()

    # ------------------------------------------------------------------ #
    #  Base-state profiles (vectorized)
    # ------------------------------------------------------------------ #
    def _build_base_state(self) -> None:
        """Compute 1-D base-state thermodynamic profiles.

        Temperature follows a constant lapse rate γ in the troposphere,
        capped at T_tropopause (isothermal above).  Pressure is
        integrated hydrostatically using the hypsometric equation in
        each layer.  All computations are fully vectorized.
        """
        z = self.z  # (nz,)

        # ---- temperature T̄(z) ----
        T_bar = self.T_sfc - self.gamma * z
        T_bar = np.maximum(T_bar, self.T_tropopause)
        self.T_bar_z: NDArray = T_bar  # (nz,)

        # ---- pressure p̄(z) — hydrostatic integration ----
        # Use layer-mean temperature between successive levels.
        # p(k) = p(k-1) * exp(-g*dz / (R_d * T_mean))
        # Vectorized via cumulative sum of the exponent.
        T_mid = 0.5 * (T_bar[:-1] + T_bar[1:])        # (nz-1,)
        dp_exponent = -g * self.dz / (R_d * T_mid)     # (nz-1,)
        log_p = np.empty(self.nz)
        log_p[0] = np.log(self.p_sfc)
        log_p[1:] = log_p[0] + np.cumsum(dp_exponent)
        self.p_bar_z: NDArray = np.exp(log_p)           # (nz,)

        # ---- Exner function Π(z) = (p/p₀)^(R_d/c_p) ----
        self.exner_z: NDArray = (self.p_bar_z / p_0) ** (R_d / c_p)

        # ---- density ρ̄(z) = p / (R_d * T) ----
        self.rho_bar_z: NDArray = self.p_bar_z / (R_d * T_bar)

        # ---- potential temperature θ̄(z) = T / Π ----
        self.theta_bar_z: NDArray = T_bar / self.exner_z

        # ---- base-state water-vapour mixing ratio q_v(z) ----
        #   RH(z) = RH_sfc * exp(-z / H_q),  capped at 5 %
        #   q_vs  = ε * e_s / (p - e_s),  Bolton (1980)
        H_q = 8000.0  # moisture scale height (m)
        RH = np.maximum(self.RH_sfc * np.exp(-z / H_q), 0.05)
        T_c = T_bar - T_0  # Celsius
        e_s = 611.2 * np.exp(17.67 * T_c / (T_c + 243.5))
        qvs = epsilon * e_s / (self.p_bar_z - e_s)
        self.qv_bar_z: NDArray = RH * qvs

        # ---- density potential temperature θ_ρ(z) ----
        #   θ_ρ = θ * (1 + R_v/R_d * q_v)  (dry-air loading only)
        self.theta_rho_bar_z: NDArray = self.theta_bar_z * (
            1.0 + (R_v / R_d) * self.qv_bar_z
        )

    # ------------------------------------------------------------------ #
    #  Convenience: broadcast 1-D → 3-D
    # ------------------------------------------------------------------ #
    def broadcast_z(self, f_z: NDArray) -> Float3D:
        """Broadcast a 1-D profile f(z) → shape (nx, ny, nz).

        Parameters
        ----------
        f_z : ndarray, shape (nz,)
            Any 1-D vertical profile.

        Returns
        -------
        ndarray, shape (nx, ny, nz)
            The profile broadcast along x and y.
        """
        return f_z[np.newaxis, np.newaxis, :]  # (1,1,nz) → broadcasts

    # ------------------------------------------------------------------ #
    #  Finite-difference operators
    # ------------------------------------------------------------------ #
    def ddx(self, f: Float3D) -> Float3D:
        """Centred finite difference ∂f/∂x (periodic in x).

        Parameters
        ----------
        f : ndarray, shape (nx, ny, nz)

        Returns
        -------
        ndarray, shape (nx, ny, nz)
        """
        return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (
            2.0 * self.dx
        )

    def ddy(self, f: Float3D) -> Float3D:
        """Centred finite difference ∂f/∂y (periodic in y).

        Parameters
        ----------
        f : ndarray, shape (nx, ny, nz)

        Returns
        -------
        ndarray, shape (nx, ny, nz)
        """
        return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (
            2.0 * self.dy
        )

    def ddz(self, f: Float3D) -> Float3D:
        """Centred finite difference ∂f/∂z with zero-gradient BCs.

        The bottom (k=0) and top (k=nz-1) boundaries use one-sided
        differences that enforce ∂f/∂z = 0 at the boundary by
        reflecting the interior value (i.e. ghost cell = interior cell).

        Parameters
        ----------
        f : ndarray, shape (nx, ny, nz)

        Returns
        -------
        ndarray, shape (nx, ny, nz)
        """
        result = np.empty_like(f)

        # Interior: centred difference
        result[:, :, 1:-1] = (
            f[:, :, 2:] - f[:, :, :-2]
        ) / (2.0 * self.dz)

        # Bottom boundary: zero-gradient → ∂f/∂z = 0
        # Ghost cell f[:,:,-1] = f[:,:,0], so (f[:,:,1] - f[:,:,0]) / (2*dz)
        # is replaced by 0 via reflection.
        result[:, :, 0] = 0.0

        # Top boundary: zero-gradient → ∂f/∂z = 0
        result[:, :, -1] = 0.0

        return result

    def laplacian(self, f: Float3D) -> Float3D:
        """Scalar Laplacian ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z².

        Uses second-order centred differences.
        x, y — periodic (np.roll); z — zero-gradient ghost cells.

        Parameters
        ----------
        f : ndarray, shape (nx, ny, nz)

        Returns
        -------
        ndarray, shape (nx, ny, nz)
        """
        # ∂²f/∂x² (periodic)
        d2x = (
            np.roll(f, -1, axis=0) - 2.0 * f + np.roll(f, 1, axis=0)
        ) / (self.dx ** 2)

        # ∂²f/∂y² (periodic)
        d2y = (
            np.roll(f, -1, axis=1) - 2.0 * f + np.roll(f, 1, axis=1)
        ) / (self.dy ** 2)

        # ∂²f/∂z² (zero-gradient ghost cells)
        d2z = np.empty_like(f)

        # Interior
        d2z[:, :, 1:-1] = (
            f[:, :, 2:] - 2.0 * f[:, :, 1:-1] + f[:, :, :-2]
        ) / (self.dz ** 2)

        # Bottom: ghost cell f[:,:,-1] == f[:,:,0]
        #   ∂²f/∂z² ≈ (f[:,:,1] - 2*f[:,:,0] + f[:,:,0]) / dz²
        #           = (f[:,:,1] - f[:,:,0]) / dz²
        d2z[:, :, 0] = (f[:, :, 1] - f[:, :, 0]) / (self.dz ** 2)

        # Top: ghost cell f[:,:,nz] == f[:,:,nz-1]
        d2z[:, :, -1] = (f[:, :, -2] - f[:, :, -1]) / (self.dz ** 2)

        return d2x + d2y + d2z

    # ------------------------------------------------------------------ #
    #  Derived 3-D base-state fields
    # ------------------------------------------------------------------ #
    @property
    def rho_bar_3d(self) -> Float3D:
        """Base-state density broadcast to (nx, ny, nz)."""
        return self.broadcast_z(self.rho_bar_z)

    @property
    def theta_bar_3d(self) -> Float3D:
        """Base-state potential temperature broadcast to (nx, ny, nz)."""
        return self.broadcast_z(self.theta_bar_z)

    @property
    def theta_rho_bar_3d(self) -> Float3D:
        """Base-state density potential temperature broadcast to (nx, ny, nz)."""
        return self.broadcast_z(self.theta_rho_bar_z)

    @property
    def exner_3d(self) -> Float3D:
        """Exner function broadcast to (nx, ny, nz)."""
        return self.broadcast_z(self.exner_z)

    # ------------------------------------------------------------------ #
    #  Repr
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        Lx = self.nx * self.dx * 1e-3
        Ly = self.ny * self.dy * 1e-3
        Lz = self.nz * self.dz * 1e-3
        return (
            f"Grid3D({self.nx}×{self.ny}×{self.nz}, "
            f"Δ=({self.dx},{self.dy},{self.dz})m, "
            f"domain={Lx:.1f}×{Ly:.1f}×{Lz:.1f} km)"
        )
