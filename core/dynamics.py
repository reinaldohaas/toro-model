"""
dynamics.py — Dinâmica da coluna ascendente e vórtice de tornado.

Inclui:
    - Coluna ascendente 1D (equação de momento vertical)
    - Vórtice de tornado (Rankine modificado)
    - Entrainment lateral
    - Cisalhamento de vento
    - Critério de colapso

Referências:
    - Houze (2014): Cloud Dynamics
    - Markowski & Richardson (2010): Mesoscale Meteorology in Midlatitudes
    - Rankine (1882): Vortex model
"""

import numpy as np
from core.constants import g, rho_air as compute_rho, R_d
from core.config import DynamicsConfig


class UpdraftColumn:
    """Coluna ascendente 1D.
    
    Gerencia a velocidade vertical w(z) e aplica perturbações iniciais.
    """
    
    def __init__(self, z, config: DynamicsConfig):
        """
        Args:
            z: Array de altitudes (m).
            config: DynamicsConfig.
        """
        self.z = z.copy()
        self.nz = len(z)
        self.w = np.zeros(self.nz)
        self.config = config
    
    def apply_perturbation(self, dw=2.0, z_center=500.0, sigma=300.0):
        """Aplica perturbação gaussiana de w na base.
        
        Args:
            dw: Amplitude da perturbação (m/s).
            z_center: Centro da perturbação (m).
            sigma: Largura (m).
        """
        self.w += dw * np.exp(-0.5 * ((self.z - z_center) / sigma) ** 2)


class TornadoVortex:
    """Vórtice de tornado — modelo de Rankine modificado.
    
    Perfil tangencial:
        r ≤ R_max: V_t = V_max(z) * r / R_max  (corpo sólido)
        r > R_max: V_t = V_max(z) * R_max / r   (irrotacional)
    
    Perfil vertical de V_max:
        z ≤ z_max: V_max(z) = V_max_sfc * (z/z_max)^alpha
        z > z_max: V_max(z) = V_max_sfc * exp(-(z-z_max)/H_decay)
    """
    
    def __init__(self, config: DynamicsConfig):
        """
        Args:
            config: DynamicsConfig com parâmetros do tornado.
        """
        self.V_max_sfc = config.V_max_tornado    # m/s
        self.R_max = config.R_max_tornado        # m
        self.z_max = config.z_max_tornado        # m
        self.H_decay = config.H_decay_tornado    # m
        self.alpha = config.alpha_tornado
    
    def V_max_profile(self, z):
        """Perfil vertical de velocidade máxima.
        
        Args:
            z: Altitude (m) ou array.
        
        Returns:
            V_max(z) em m/s.
        """
        z = np.atleast_1d(z)
        V = np.zeros_like(z, dtype=float)
        
        below = z <= self.z_max
        above = z > self.z_max
        
        V[below] = self.V_max_sfc * (z[below] / self.z_max) ** self.alpha
        V[above] = self.V_max_sfc * np.exp(
            -(z[above] - self.z_max) / self.H_decay
        )
        
        # Evitar V < 0
        V = np.maximum(V, 0.0)
        
        if V.size == 1:
            return float(V[0])
        return V
    
    def V_tangential(self, r, z):
        """Velocidade tangencial V_t(r, z).
        
        Args:
            r: Distância radial (m).
            z: Altitude (m).
        
        Returns:
            V_t em m/s.
        """
        V_max_z = self.V_max_profile(z)
        
        r = np.atleast_1d(r)
        V = np.zeros_like(r, dtype=float)
        
        inside = r <= self.R_max
        outside = r > self.R_max
        
        V[inside] = V_max_z * r[inside] / self.R_max
        if np.any(outside):
            V[outside] = V_max_z * self.R_max / r[outside]
        
        if V.size == 1:
            return float(V[0])
        return V
    
    def pressure_deficit(self, z):
        """Déficit de pressão no centro do vórtice (Pa).
        
        ΔP = ρ * V_max² (equilíbrio ciclostrófico)
        
        Args:
            z: Altitude (m).
        
        Returns:
            ΔP em Pa (sempre positivo = pressão menor no centro).
        """
        V_max_z = self.V_max_profile(z)
        # Usar densidade aproximada
        T_approx = 300.0 - 6.5e-3 * np.atleast_1d(z)
        p_approx = 101325.0 * np.exp(-g * np.atleast_1d(z) / (R_d * T_approx))
        rho = p_approx / (R_d * T_approx)
        
        dP = rho * V_max_z ** 2
        
        if np.atleast_1d(dP).size == 1:
            return float(np.atleast_1d(dP)[0])
        return dP
    
    def swirl_ratio(self, w_mean, R_updraft):
        """Swirl Ratio — controla estrutura do vórtice.
        
        S = V_t * R_max / (w_mean * R_updraft)
        
        S < 0.5 → vórtice laminar
        S > 0.5 → turbulento / sub-vórtices
        
        Args:
            w_mean: Velocidade vertical média (m/s).
            R_updraft: Raio da corrente ascendente (m).
        
        Returns:
            Swirl ratio (adimensional).
        """
        if w_mean <= 0:
            return float('inf')
        return self.V_max_sfc * self.R_max / (w_mean * R_updraft)
    
    def get_wind_field(self, z_array, r_array):
        """Campo de vento completo V_t(r, z).
        
        Args:
            z_array: Array de altitudes (m).
            r_array: Array de distâncias radiais (m).
        
        Returns:
            Array (nz, nr) de velocidade tangencial.
        """
        nz = len(z_array)
        nr = len(r_array)
        V_field = np.zeros((nz, nr))
        
        for iz in range(nz):
            V_field[iz, :] = self.V_tangential(r_array, z_array[iz])
        
        return V_field


def compute_dw_dt(w, B, z, config):
    """Taxa de mudança da velocidade vertical.
    
    dw/dt = B - C_d*w*|w|/(2*R) - ε*w
    
    O arrasto usa R_updraft (não z_top) como escala de comprimento,
    representando a dissipação turbulenta na escala da corrente ascendente.
    
    Args:
        w: Array de velocidade vertical (m/s).
        B: Array de flutuabilidade (m/s²).
        z: Array de altitudes (m).
        config: DynamicsConfig.
    
    Returns:
        Array dw/dt (m/s²).
    """
    nz = len(w)
    dw_dt = np.zeros(nz)
    
    # Escala de comprimento = raio da corrente ascendente (não z_top!)
    H = config.R_updraft  # ~1500 m — escala correta para arrasto
    C_d = config.C_d_turb
    eps = config.epsilon_0 / config.R_updraft
    
    # Limitar w antes de calcular para evitar overflow
    w_safe = np.clip(w, -80.0, 80.0)
    
    for i in range(nz):
        # Flutuabilidade
        dw_dt[i] = B[i]
        
        # Arrasto turbulento: C_d * w² / (2*R)
        if H > 0:
            dw_dt[i] -= C_d * w_safe[i] * abs(w_safe[i]) / (2.0 * H)
        
        # Entrainment
        dw_dt[i] -= eps * w_safe[i]
    
    # Proteção contra NaN e overflow
    dw_dt = np.nan_to_num(dw_dt, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Limitar aceleração máxima (realismo: |dw/dt| < 5 m/s²)
    dw_dt = np.clip(dw_dt, -5.0, 5.0)
    
    return dw_dt


def entrainment_mixing(parcel_T, parcel_qv, env_T, env_qv, epsilon, dt):
    """Mistura por entrainment lateral.
    
    O ar ambiental (mais seco e frio) é misturado na parcela,
    reduzindo temperatura e umidade.
    
    Args:
        parcel_T: Temperatura da parcela (K).
        parcel_qv: Razão de mistura da parcela (kg/kg).
        env_T: Temperatura ambiente (K).
        env_qv: Razão de mistura ambiente (kg/kg).
        epsilon: Taxa de entrainment (1/s).
        dt: Passo temporal (s).
    
    Returns:
        tuple (dT, dqv) — mudanças na parcela.
    """
    dT = -epsilon * (parcel_T - env_T) * dt
    dqv = -epsilon * (parcel_qv - env_qv) * dt
    
    return dT, dqv


def wind_shear_profile(z, config):
    """Perfil de vento com cisalhamento.
    
    u(z) = u_sfc + (du/dz)*z
    v(z) = v_sfc + (dv/dz)*z
    
    Args:
        z: Array de altitudes (m).
        config: DynamicsConfig.
    
    Returns:
        dict: {'u': array, 'v': array, 'speed': array, 'SRH': float}
    """
    u = config.u_sfc + config.du_dz * z
    v = config.v_sfc + config.dv_dz * z
    speed = np.sqrt(u ** 2 + v ** 2)
    
    # Helicidade relativa à tempestade (SRH)
    # Simplificada: SRH ≈ ∫₀⁶ᵏᵐ |dV/dz × V| dz
    SRH = 0.0
    for i in range(1, len(z)):
        if z[i] > 6000:
            break
        dz = z[i] - z[i - 1]
        du_dz = (u[i] - u[i - 1]) / dz
        dv_dz = (v[i] - v[i - 1]) / dz
        # Cross helicity: u*dv/dz - v*du/dz
        SRH += abs(u[i] * dv_dz - v[i] * du_dz) * dz
    
    return {'u': u, 'v': v, 'speed': speed, 'SRH': SRH}


def check_collapse_criterion(spectra, B, w, z, dz, config):
    """Verifica critério de colapso do pistão hidráulico.
    
    Colapso quando: M_total * g > F_updraft
    
    A massa é calculada usando a área da seção transversal do PISTÃO
    (não da corrente ascendente inteira). Apenas níveis com hidrometeoros
    significativos são contabilizados.
    
    Args:
        spectra: HydrometeorSpectra.
        B: Array de flutuabilidade (m/s²).
        w: Array de velocidade vertical (m/s).
        z: Array de altitudes (m).
        dz: Resolução vertical (m).
        config: SimulationConfig completa.
    
    Returns:
        dict: {'collapsed': bool, 'M_piston': float, 'F_updraft': float}
    """
    from core.microphysics import compute_lwc, compute_iwc, BinGrid
    
    nz = len(z)
    R_piston = config.collapse.R_piston  # 200 m
    A_piston = np.pi * R_piston ** 2     # ~125000 m²
    
    # Usar raio da corrente ascendente para o balanço de forças
    R_updraft = config.dynamics.R_updraft  # 1500 m
    A_updraft = np.pi * R_updraft ** 2
    
    # Massa de hidrometeoros na coluna
    bin_grid = BinGrid(
        n_bins=config.microphysics.n_bins,
        D_min=config.microphysics.D_min,
        D_max=config.microphysics.D_max
    )
    
    lwc = compute_lwc(spectra, bin_grid)
    iwc = compute_iwc(spectra, bin_grid)
    twc = lwc + iwc  # Total water content (kg/m³)
    
    # Massa do pistão: apenas na área do pistão, apenas onde TWC > threshold
    # LWC/IWC são por m³ de ar, multiplicar pela área do pistão
    twc_threshold = 1e-5  # kg/m³ — mínimo para considerar
    M_piston = 0.0
    n_levels_with_hydro = 0
    
    for i in range(nz):
        if twc[i] > twc_threshold:
            M_piston += twc[i] * A_piston * dz
            n_levels_with_hydro += 1
    
    # Força de sustentação na corrente ascendente
    # B é por kg de ar, multiplicar por ρ*A_updraft*dz para obter força
    F_updraft = 0.0
    for i in range(nz):
        if B[i] > 0 and w[i] > 0:
            T_approx = 300.0 - 6.5e-3 * z[i]
            rho_level = config.thermodynamics.p_sfc * np.exp(
                -g * z[i] / (287.0 * max(T_approx, 200.0))
            ) / (287.0 * max(T_approx, 200.0))
            # Sustentação na área do pistão (fração da updraft)
            F_updraft += B[i] * rho_level * A_piston * dz
    
    # Peso total do pistão
    W_total = M_piston * g
    
    # Critério de colapso:
    # 1. Peso > sustentação
    # 2. Massa significativa (>500 ton para evitar falso positivo)
    # 3. Precisa ter hidrometeoros em pelo menos 10 níveis
    collapsed = (W_total > F_updraft) and (M_piston > 5e5) and (n_levels_with_hydro > 10)
    
    return {
        'collapsed': collapsed,
        'M_piston': float(M_piston),
        'F_updraft': float(F_updraft),
        'W_total': float(W_total),
        'n_levels': n_levels_with_hydro
    }
