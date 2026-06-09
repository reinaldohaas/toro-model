"""
dynamics3d.py — Dinâmica anelástica 3D para o modelo Toró.

Equações governantes (anelásticas, sem Coriolis):
    ∂u/∂t = -v⃗·∇u - (1/ρ̄)∂p'/∂x + D_u
    ∂v/∂t = -v⃗·∇v - (1/ρ̄)∂p'/∂y + D_v
    ∂w/∂t = -v⃗·∇w - (1/ρ̄)∂p'/∂z + B + D_w
    ∇·(ρ̄v⃗) = 0  (diagnóstico para p')

A flutuabilidade B é calculada a partir de θ_ρ, que inclui
o carregamento de hidrometeoros. Quando graupel/granizo caem,
B < 0 → p' < 0 acima → ∂p'/∂r < 0 → convergência
→ concentra o pistão hidráulico. Este é o mecanismo-chave do Toró.

Fronteiras: periódicas em x, y. Rigida/Neumann em z.

Referências:
    - Klemp & Wilhelmson (1978): JAS — 3D cloud model
    - Wicker & Skamarock (2002): MWR — time integration
"""

import numpy as np


# ============================================================================
# FINITE DIFFERENCES — Periodic in x,y, bounded in z
# ============================================================================

def ddx(f: np.ndarray, dx: float) -> np.ndarray:
    """∂f/∂x — diferença centrada, periódica em x (eixo 0).
    
    Args:
        f: Array 3D (nx, ny, nz).
        dx: Espaçamento em x (m).
    
    Returns:
        ∂f/∂x (nx, ny, nz).
    """
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * dx)


def ddy(f: np.ndarray, dy: float) -> np.ndarray:
    """∂f/∂y — diferença centrada, periódica em y (eixo 1)."""
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * dy)


def ddz(f: np.ndarray, dz: float) -> np.ndarray:
    """∂f/∂z — diferença centrada, zero-gradient nas fronteiras z (eixo 2).
    
    z=0 (superfície): ∂f/∂z = (f[1] - f[0]) / dz (one-sided)
    z=top:            ∂f/∂z = (f[-1] - f[-2]) / dz (one-sided)
    """
    result = np.zeros_like(f)
    # Interior: centrada
    result[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2.0 * dz)
    # Fronteiras: one-sided
    result[:, :, 0] = (f[:, :, 1] - f[:, :, 0]) / dz
    result[:, :, -1] = (f[:, :, -1] - f[:, :, -2]) / dz
    return result


def d2dx2(f: np.ndarray, dx: float) -> np.ndarray:
    """∂²f/∂x² — periódica em x."""
    return (np.roll(f, -1, axis=0) - 2*f + np.roll(f, 1, axis=0)) / (dx * dx)


def d2dy2(f: np.ndarray, dy: float) -> np.ndarray:
    """∂²f/∂y² — periódica em y."""
    return (np.roll(f, -1, axis=1) - 2*f + np.roll(f, 1, axis=1)) / (dy * dy)


def d2dz2(f: np.ndarray, dz: float) -> np.ndarray:
    """∂²f/∂z² — Neumann nas fronteiras z."""
    result = np.zeros_like(f)
    result[:, :, 1:-1] = (f[:, :, 2:] - 2*f[:, :, 1:-1] + f[:, :, :-2]) / (dz * dz)
    # Neumann BC: ∂f/∂z = 0 → f[-1] = f[0], f[nz] = f[nz-1]
    result[:, :, 0] = (f[:, :, 1] - f[:, :, 0]) / (dz * dz)
    result[:, :, -1] = (f[:, :, -2] - f[:, :, -1]) / (dz * dz)
    return result


# ============================================================================
# ADVECTION
# ============================================================================

def advect_3d(f: np.ndarray, u: np.ndarray, v: np.ndarray, w: np.ndarray,
              dx: float, dy: float, dz: float) -> np.ndarray:
    """Tendência advectiva: -u·∂f/∂x - v·∂f/∂y - w·∂f/∂z.
    
    Diferenças centradas, periódicas em x, y.
    
    Args:
        f: Campo escalar 3D (nx, ny, nz).
        u, v, w: Componentes do vento (nx, ny, nz).
        dx, dy, dz: Espaçamentos de grade (m).
    
    Returns:
        Tendência advectiva (nx, ny, nz).
    """
    return -(u * ddx(f, dx) + v * ddy(f, dy) + w * ddz(f, dz))


# ============================================================================
# DIFFUSION
# ============================================================================

def apply_diffusion(f: np.ndarray, K_h: float, K_v: float,
                    dx: float, dy: float, dz: float) -> np.ndarray:
    """Difusão turbulenta: K_h*(∂²f/∂x² + ∂²f/∂y²) + K_v*∂²f/∂z².
    
    Args:
        f: Campo 3D (nx, ny, nz).
        K_h: Difusividade horizontal (m²/s).
        K_v: Difusividade vertical (m²/s).
        dx, dy, dz: Espaçamentos (m).
    
    Returns:
        Tendência de difusão (nx, ny, nz).
    """
    return K_h * (d2dx2(f, dx) + d2dy2(f, dy)) + K_v * d2dz2(f, dz)


# ============================================================================
# PRESSURE PERTURBATION — Poisson Solver (Jacobi)
# ============================================================================

def compute_pressure_poisson(buoyancy: np.ndarray, rho_bar: np.ndarray,
                             u: np.ndarray, v: np.ndarray, w: np.ndarray,
                             dx: float, dy: float, dz: float,
                             n_iter: int = 50) -> np.ndarray:
    """Resolve p' da equação diagnóstica anelástica via iteração de Jacobi.
    
    A equação de Poisson para p' é derivada da continuidade anelástica.
    O RHS inclui o forçamento por flutuabilidade:
    
        ∇²p' ≈ ρ̄ · ∂B/∂z  (termo dominante)
    
    Quando hidrometeoros caem (B < 0 na coluna), ∂B/∂z cria:
        - p' < 0 acima da coluna → convergência horizontal
        - p' > 0 abaixo → divergência
    Isso CONCENTRA o pistão — mecanismo-chave do Toró.
    
    Args:
        buoyancy: B (nx, ny, nz).
        rho_bar: ρ̄(z) (nz,) — perfil de referência.
        u, v, w: Vento (nx, ny, nz).
        dx, dy, dz: Espaçamentos.
        n_iter: Iterações de Jacobi.
    
    Returns:
        p' (nx, ny, nz).
    """
    nx, ny, nz = buoyancy.shape
    
    # Broadcast ρ̄ para 3D
    rho3d = rho_bar[np.newaxis, np.newaxis, :]  # (1, 1, nz)
    
    # RHS: forçamento por flutuabilidade (termo dominante)
    # ∇²p' = ρ̄ · ∂B/∂z
    rhs = rho3d * ddz(buoyancy, dz)
    
    # Adicionar divergência de momento (correção de pressão)
    div_rhou = ddx(rho3d * u, dx)
    div_rhov = ddy(rho3d * v, dy)
    div_rhow = ddz(rho3d * w, dz)
    rhs -= (div_rhou + div_rhov + div_rhow) / max(dx, 1.0)  # Relaxamento
    
    # Resolver por Jacobi
    p_prime = np.zeros((nx, ny, nz))
    
    dx2 = dx * dx
    dy2 = dy * dy
    dz2 = dz * dz
    
    coeff = 1.0 / (2.0/dx2 + 2.0/dy2 + 2.0/dz2)
    omega = 1.2  # SOR over-relaxation
    
    for _ in range(n_iter):
        # Vizinhos (periódico em x, y)
        p_xp = np.roll(p_prime, -1, axis=0)
        p_xm = np.roll(p_prime, 1, axis=0)
        p_yp = np.roll(p_prime, -1, axis=1)
        p_ym = np.roll(p_prime, 1, axis=1)
        
        # z: Neumann nas fronteiras
        p_zp = np.zeros_like(p_prime)
        p_zm = np.zeros_like(p_prime)
        p_zp[:, :, :-1] = p_prime[:, :, 1:]
        p_zp[:, :, -1] = p_prime[:, :, -1]  # Neumann: dp/dz=0
        p_zm[:, :, 1:] = p_prime[:, :, :-1]
        p_zm[:, :, 0] = p_prime[:, :, 0]    # Neumann: dp/dz=0
        
        # Atualização Jacobi com SOR
        p_new = coeff * (
            (p_xp + p_xm) / dx2 +
            (p_yp + p_ym) / dy2 +
            (p_zp + p_zm) / dz2 -
            rhs
        )
        
        p_prime = (1 - omega) * p_prime + omega * p_new
    
    return p_prime


# ============================================================================
# MOMENTUM TENDENCIES
# ============================================================================

def compute_momentum_tendency(u: np.ndarray, v: np.ndarray, w: np.ndarray,
                              p_prime: np.ndarray, buoyancy: np.ndarray,
                              rho_bar: np.ndarray,
                              dx: float, dy: float, dz: float,
                              K_h: float = 500.0, K_v: float = 100.0):
    """Calcula du/dt, dv/dt, dw/dt.
    
    Inclui: advecção + gradiente de pressão + flutuabilidade + difusão.
    
    O gradiente de pressão -(1/ρ̄)∇p' é o que gera a convergência
    quando o pistão de hidrometeoros cria p' < 0 acima dele.
    
    Args:
        u, v, w: Componentes do vento (nx, ny, nz).
        p_prime: Perturbação de pressão (nx, ny, nz).
        buoyancy: B (nx, ny, nz).
        rho_bar: ρ̄(z) (nz,).
        dx, dy, dz: Espaçamentos.
        K_h, K_v: Difusividades turbulentas.
    
    Returns:
        (du_dt, dv_dt, dw_dt): Tupla de arrays (nx, ny, nz).
    """
    rho3d = rho_bar[np.newaxis, np.newaxis, :]  # (1, 1, nz)
    inv_rho = 1.0 / np.maximum(rho3d, 0.1)  # Evitar divisão por zero
    
    # Gradiente de pressão perturbação
    dppdx = ddx(p_prime, dx)
    dppdy = ddy(p_prime, dy)
    dppdz = ddz(p_prime, dz)
    
    # du/dt = -advecção - (1/ρ̄)∂p'/∂x + difusão
    du_dt = (advect_3d(u, u, v, w, dx, dy, dz)
             - inv_rho * dppdx
             + apply_diffusion(u, K_h, K_v, dx, dy, dz))
    
    # dv/dt = -advecção - (1/ρ̄)∂p'/∂y + difusão
    dv_dt = (advect_3d(v, u, v, w, dx, dy, dz)
             - inv_rho * dppdy
             + apply_diffusion(v, K_h, K_v, dx, dy, dz))
    
    # dw/dt = -advecção - (1/ρ̄)∂p'/∂z + B + difusão
    dw_dt = (advect_3d(w, u, v, w, dx, dy, dz)
             - inv_rho * dppdz
             + buoyancy
             + apply_diffusion(w, K_h, K_v, dx, dy, dz))
    
    # Fronteira: w=0 na superfície e no topo
    dw_dt[:, :, 0] = 0.0
    dw_dt[:, :, -1] = 0.0
    
    # Clampar acelerações para estabilidade
    max_accel = 5.0  # m/s²
    du_dt = np.clip(du_dt, -max_accel, max_accel)
    dv_dt = np.clip(dv_dt, -max_accel, max_accel)
    dw_dt = np.clip(dw_dt, -max_accel, max_accel)
    
    # Sanitizar
    du_dt = np.nan_to_num(du_dt, nan=0.0)
    dv_dt = np.nan_to_num(dv_dt, nan=0.0)
    dw_dt = np.nan_to_num(dw_dt, nan=0.0)
    
    return du_dt, dv_dt, dw_dt


# ============================================================================
# DIAGNOSTICS
# ============================================================================

def compute_divergence(u: np.ndarray, v: np.ndarray, w: np.ndarray,
                       rho_bar: np.ndarray,
                       dx: float, dy: float, dz: float) -> np.ndarray:
    """Divergência anelástica: ∂(ρ̄u)/∂x + ∂(ρ̄v)/∂y + ∂(ρ̄w)/∂z.
    
    Deve ser ≈ 0 para satisfazer continuidade anelástica.
    """
    rho3d = rho_bar[np.newaxis, np.newaxis, :]
    return ddx(rho3d * u, dx) + ddy(rho3d * v, dy) + ddz(rho3d * w, dz)


def compute_cfl(u: np.ndarray, v: np.ndarray, w: np.ndarray,
                dx: float, dy: float, dz: float, dt: float) -> float:
    """Número CFL máximo.
    
    CFL = (|u|/dx + |v|/dy + |w|/dz) * dt
    
    Returns:
        CFL máximo (adimensional). Deve ser < 1 para estabilidade.
    """
    cfl = (np.abs(u) / dx + np.abs(v) / dy + np.abs(w) / dz) * dt
    return float(np.max(cfl))


def compute_adaptive_dt(u: np.ndarray, v: np.ndarray, w: np.ndarray,
                        dx: float, dy: float, dz: float,
                        cfl_target: float = 0.5,
                        dt_min: float = 0.1, dt_max: float = 2.0) -> float:
    """Calcula dt adaptativo baseado no CFL.
    
    dt = cfl_target / max(|u|/dx + |v|/dy + |w|/dz)
    """
    speed = np.abs(u) / dx + np.abs(v) / dy + np.abs(w) / dz
    max_speed = float(np.max(speed))
    
    if max_speed > 0:
        dt = cfl_target / max_speed
    else:
        dt = dt_max
    
    return np.clip(dt, dt_min, dt_max)
