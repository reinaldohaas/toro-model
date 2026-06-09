"""
collapse.py — Modelo do pistão hidráulico.

Simula o colapso da coluna de graupel/granizo como um corpo semi-rígido.
A coesão vem da alta fração de gelo interligado.

Referências:
    - Fujita (1985): The Downburst — microburst dynamics
    - Srivastava (1987): Extended microburst model with ice
    - Joukowsky (1898): Water hammer equation
"""

import numpy as np
from scipy.integrate import solve_ivp
from core.constants import g, rho_water, rho_ice
from core.config import CollapseConfig


class HydraulicPiston:
    """Modelo de pistão hidráulico para o colapso do Toró.
    
    A coluna de graupel/granizo/água atua como corpo semi-rígido
    que descende coerentemente quando a sustentação falha.
    """
    
    def __init__(self, M, A, rho_mix, config: CollapseConfig):
        """
        Args:
            M: Massa total do pistão (kg).
            A: Área da seção transversal (m²).
            rho_mix: Densidade da mistura água-gelo (kg/m³).
            config: CollapseConfig.
        """
        self.M = M                       # kg
        self.A = A                       # m²
        self.rho_mix = rho_mix           # kg/m³
        self.C_d = config.C_d_piston     # coeficiente de arrasto
        self.c_sound = config.c_sound_mix  # velocidade do som na mistura (m/s)
        
        # Densidade do ar (nível médio de queda, ~3km)
        self.rho_air = 0.9  # kg/m³ (aproximação)
    
    def compute_terminal_velocity(self):
        """Velocidade terminal do pistão.
        
        v_t = sqrt(2*M*g / (ρ_air*C_d*A))
        
        Returns:
            v_terminal em m/s.
        """
        return np.sqrt(2.0 * self.M * g / (self.rho_air * self.C_d * self.A))
    
    def simulate_fall(self, z_start, dt=0.1):
        """Simula a queda do pistão de z_start até o solo.
        
        Resolve: M*dv/dt = M*g - ½*ρ_air*C_d*A*v²
        
        Args:
            z_start: Altitude inicial (m).
            dt: Passo temporal para saída (s).
        
        Returns:
            dict:
                't_fall': array de tempos (s)
                'z_fall': array de altitudes (m)
                'v_fall': array de velocidades (m/s)
                'v_impact': velocidade no impacto (m/s)
                't_impact': tempo até impacto (s)
        """
        def equations(t, y):
            z, v = y
            # v é positivo para baixo
            drag = 0.5 * self.rho_air * self.C_d * self.A * v ** 2
            dv_dt = g - drag / self.M
            dz_dt = -v  # z diminui (descendo)
            return [dz_dt, dv_dt]
        
        def hit_ground(t, y):
            return y[0]  # z = 0 → impacto
        
        hit_ground.terminal = True
        hit_ground.direction = -1
        
        # Condições iniciais: z=z_start, v=0 (começa do repouso)
        y0 = [z_start, 0.0]
        t_span = (0, 300)  # Máximo 5 minutos
        t_eval = np.arange(0, 300, dt)
        
        sol = solve_ivp(
            equations, t_span, y0,
            method='RK45',
            t_eval=t_eval,
            events=hit_ground,
            max_step=0.5
        )
        
        t_fall = sol.t
        z_fall = sol.y[0]
        v_fall = sol.y[1]
        
        # Velocidade no impacto
        v_impact = float(v_fall[-1]) if len(v_fall) > 0 else 0.0
        t_impact = float(t_fall[-1]) if len(t_fall) > 0 else 0.0
        
        return {
            't_fall': t_fall,
            'z_fall': z_fall,
            'v_fall': v_fall,
            'v_impact': v_impact,
            't_impact': t_impact
        }
    
    def compute_impact_pressure(self, v_impact):
        """Pressão de impacto — analogia water hammer (Joukowsky).
        
        P = ρ_mix * c_sound_mix * v_impact
        
        Args:
            v_impact: Velocidade no impacto (m/s).
        
        Returns:
            Pressão de impacto (Pa).
        """
        return self.rho_mix * self.c_sound * v_impact
    
    def compute_impact_energy(self, v_impact):
        """Energia cinética no impacto.
        
        E = ½ * M * v²
        
        Args:
            v_impact: Velocidade no impacto (m/s).
        
        Returns:
            Energia (J).
        """
        return 0.5 * self.M * v_impact ** 2


def compute_piston_mass(spectra, bin_grid, w, z_array, dz, config):
    """Calcula a massa do pistão a partir dos hidrometeoros na coluna.
    
    M_piston = ∫₀ᴴ (LWC + IWC) * A * dz
    
    Args:
        spectra: HydrometeorSpectra.
        bin_grid: BinGrid.
        w: Array de velocidade vertical (m/s).
        z_array: Array de altitudes (m).
        dz: Resolução vertical (m).
        config: SimulationConfig.
    
    Returns:
        dict:
            'M_piston': massa total (kg)
            'rho_mix': densidade média da mistura (kg/m³)
            'H_piston': altura do pistão (m)
    """
    from core.microphysics import compute_lwc, compute_iwc
    
    R_piston = config.collapse.R_piston
    A = np.pi * R_piston ** 2
    
    lwc = compute_lwc(spectra, bin_grid)
    iwc = compute_iwc(spectra, bin_grid)
    
    total_wc = lwc + iwc  # kg/m³
    
    M_piston = np.sum(total_wc * A * dz)
    
    # Altura efetiva do pistão (onde há hidrometeoros significativos)
    mask = total_wc > 1e-5
    if np.any(mask):
        z_levels = z_array[mask]
        H_piston = z_levels[-1] - z_levels[0]
        rho_mix = M_piston / (A * max(H_piston, 1.0))
    else:
        H_piston = 500.0
        rho_mix = 500.0
    
    return {
        'M_piston': float(M_piston),
        'rho_mix': float(rho_mix),
        'H_piston': float(H_piston)
    }
