"""
config.py — Configuração do cenário de simulação.

Parâmetros calibrados para o Vale do Revólver, Presidente Getúlio, SC, Brasil.
Supercélula subtropical com DSD estreita, favorável a tornado e glaciação explosiva.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GridConfig:
    """Configuração da grade vertical (legado 1D)."""
    z_top: float = 15000.0    # m — topo do domínio
    nz: int = 50              # número de níveis verticais
    dz: float = 300.0         # m — resolução vertical (z_top / nz)


@dataclass
class Grid3DConfig:
    """Configuração da grade 3D para simulação anelástica.
    
    Domínio: 10km × 10km × 15km com fronteiras periódicas em x, y.
    Resolução suficiente para resolver corrente ascendente (R~1500m = 3 pontos).
    """
    nx: int = 20              # pontos em x
    ny: int = 20              # pontos em y
    nz: int = 50              # pontos em z
    dx: float = 500.0         # m — resolução horizontal x
    dy: float = 500.0         # m — resolução horizontal y
    dz: float = 300.0         # m — resolução vertical
    Lx: float = 10000.0       # m — comprimento do domínio em x
    Ly: float = 10000.0       # m — comprimento do domínio em y
    Lz: float = 15000.0       # m — altura do domínio


@dataclass
class DiffusionConfig:
    """Difusão turbulenta subgrid."""
    K_h: float = 500.0        # m²/s — difusividade horizontal
    K_v: float = 100.0        # m²/s — difusividade vertical


@dataclass
class TimeConfig:
    """Configuração temporal — CFL adaptativo para 3D."""
    dt_max: float = 2.0       # s — passo temporal máximo
    dt_min: float = 0.1       # s — passo temporal mínimo (adaptativo)
    t_total: float = 600.0    # s — tempo total de simulação (10 min)
    t_output: float = 10.0    # s — intervalo de saída
    cfl_target: float = 0.5   # CFL alvo para dt adaptativo


@dataclass
class MicrophysicsConfig:
    """Configuração do bin microphysics."""
    n_bins: int = 25              # número de bins por categoria (reduzido)
    D_min: float = 2e-6           # m — diâmetro mínimo (2 µm)
    D_max: float = 0.08           # m — diâmetro máximo (80 mm)
    
    # DSD Gamma inicial (muito estreita)
    mu: float = 20.0              # parâmetro de forma (alto → DSD estreita)
    N0: float = 1e8               # m⁻³·m⁻(µ+1) — parâmetro de intercepto
    D_mean: float = 20e-6         # m — diâmetro médio inicial (20 µm)
    
    # Hallett-Mossop SIP
    hm_rate: float = 350.0        # splinters por mg de gelo rimado
    hm_T_min: float = -8.0        # °C — limite frio do H-M
    hm_T_max: float = -3.0        # °C — limite quente do H-M
    hm_T_peak: float = -5.0       # °C — pico de eficiência
    hm_D_small: float = 13e-6     # m — limiar de gotas pequenas
    hm_D_large: float = 24e-6     # m — limiar de gotas grandes
    
    # Quebra colisional (Phillips et al. 2017)
    coll_alpha: float = 5.0e4     # coeficiente de fragmentação
    coll_beta: float = 0.5        # expoente de CKE
    coll_T_peak: float = -15.0    # °C — pico de eficiência
    coll_T_width: float = 10.0    # °C — largura da função g(T)
    coll_N_frag_max: float = 600  # fragmentos máximos por colisão
    
    # Eficiência de coleta (riming)
    E_collection: float = 0.8     # eficiência de coleta base


@dataclass
class ThermodynamicsConfig:
    """Configuração do perfil atmosférico — Sul do Brasil subtropical."""
    # Superfície
    T_sfc: float = 300.0          # K — temperatura de superfície (~27°C)
    p_sfc: float = 101325.0       # Pa — pressão de superfície
    RH_sfc: float = 0.85          # umidade relativa na superfície (85%)
    
    # Perfil vertical (lapse rates)
    gamma_troposphere: float = 6.5e-3   # K/m — taxa de diminuição troposférica
    z_tropopause: float = 12000.0       # m — altura da tropopausa
    T_tropopause: float = 210.0         # K — temperatura na tropopausa
    
    # CAPE e CIN
    CAPE_target: float = 2500.0   # J/kg — CAPE alvo
    CIN_target: float = -50.0     # J/kg — CIN alvo
    
    # Níveis chave
    z_LCL: float = 1000.0        # m — nível de condensação por levantamento
    z_freezing: float = 4000.0   # m — nível de congelamento (0°C)
    z_HM_top: float = 4750.0     # m — topo da zona H-M (-8°C)
    z_HM_bottom: float = 4250.0  # m — base da zona H-M (-3°C)
    
    # Perturbação inicial (trigger)
    dT_perturbation: float = 3.0  # K — perturbação térmica na base
    z_perturbation: float = 500.0 # m — profundidade da perturbação


@dataclass
class DynamicsConfig:
    """Configuração da dinâmica e tornado."""
    # Corrente ascendente
    R_updraft: float = 1500.0     # m — raio da corrente ascendente
    C_d_turb: float = 0.5         # coeficiente de arrasto turbulento
    epsilon_0: float = 0.1        # taxa de entrainamento base (/R_updraft)
    
    # Tornado (vórtice de Rankine)
    V_max_tornado: float = 70.0   # m/s — velocidade tangencial máxima (EF2-EF3)
    R_max_tornado: float = 150.0  # m — raio de vento máximo
    z_max_tornado: float = 1000.0 # m — altura do vento máximo
    H_decay_tornado: float = 3000.0  # m — escala de decaimento vertical
    alpha_tornado: float = 0.5    # expoente do perfil vertical abaixo de z_max
    
    # Cisalhamento ambiental
    u_sfc: float = 5.0            # m/s — vento zonal na superfície
    v_sfc: float = 5.0            # m/s — vento meridional na superfície
    du_dz: float = 3.0e-3         # 1/s — cisalhamento zonal
    dv_dz: float = 2.0e-3         # 1/s — cisalhamento meridional
    wind_shear_06km: float = 25.0 # m/s — cisalhamento total 0-6km


@dataclass
class CollapseConfig:
    """Configuração do colapso do pistão hidráulico."""
    R_piston: float = 200.0       # m — raio do pistão
    C_d_piston: float = 1.0       # coeficiente de arrasto do pistão
    c_sound_mix: float = 500.0    # m/s — velocidade do som na mistura água-gelo
    f_ice_cohesion: float = 0.6   # fração de gelo para coesão estrutural


@dataclass
class AcousticsConfig:
    """Configuração do som 'Tó'."""
    sample_rate: int = 44100      # Hz — taxa de amostragem do WAV
    duration: float = 10.0        # s — duração do som sintético
    L_canyon: float = 500.0       # m — comprimento do desfiladeiro (reverberação)
    c_sound_air: float = 343.0    # m/s — velocidade do som no ar


@dataclass
class SeismicConfig:
    """Configuração sísmica — calibrada para M 2-3 Richter."""
    eta_seismic: float = 0.05     # eficiência de conversão cinética → sísmica
    M_L_min: float = 2.0          # magnitude mínima observada
    M_L_max: float = 3.0          # magnitude máxima observada
    f_dominant: float = 3.0       # Hz — frequência sísmica dominante
    alpha_atten: float = 0.005    # 1/m — atenuação com distância
    sample_rate: int = 200        # Hz — taxa de amostragem do sismograma
    duration: float = 10.0        # s — duração do sismograma


@dataclass
class ErosionConfig:
    """Configuração da erosão — 1-10 ton, sem barro."""
    sigma_rock: float = 100e6     # Pa — resistência à compressão (gnaisse/granito SC)
    epsilon_frac: float = 0.01    # deformação de fratura
    eta_erosion: float = 0.03     # eficiência de erosão
    rho_rock: float = 2600.0      # kg/m³ — densidade da rocha
    rho_soil: float = 1800.0      # kg/m³ — densidade do solo mineral
    rho_tree: float = 600.0       # kg/m³ — densidade da madeira
    M_eroded_min: float = 1000.0  # kg — massa mínima erodida (1 ton)
    M_eroded_max: float = 10000.0 # kg — massa máxima erodida (10 ton)
    theta_slope: float = 0.3      # rad — inclinação do desfiladeiro (~17°)
    
    # Shields — tensão crítica para lavagem seletiva
    theta_cr: float = 0.045       # parâmetro de Shields crítico
    D_clay: float = 2e-6          # m — diâmetro argila (SEMPRE mobilizada)
    D_silt: float = 63e-6         # m — diâmetro silte (SEMPRE mobilizado)
    D_sand: float = 2e-3          # m — diâmetro areia (parcialmente)
    D_gravel: float = 64e-3       # m — diâmetro cascalho


@dataclass
class LocationConfig:
    """Localização: Vale do Revólver, Presidente Getúlio, SC."""
    name: str = "Vale do Revólver"
    municipality: str = "Presidente Getúlio"
    state: str = "Santa Catarina"
    country: str = "Brasil"
    lat: float = -26.89           # graus
    lon: float = -49.37           # graus
    elevation: float = 200.0      # m — elevação do vale
    canyon_width: float = 200.0   # m — largura do desfiladeiro
    canyon_depth: float = 150.0   # m — profundidade do desfiladeiro


@dataclass
class SimulationConfig:
    """Configuração completa da simulação."""
    grid: GridConfig = field(default_factory=GridConfig)
    grid3d: Grid3DConfig = field(default_factory=Grid3DConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    microphysics: MicrophysicsConfig = field(default_factory=MicrophysicsConfig)
    thermodynamics: ThermodynamicsConfig = field(default_factory=ThermodynamicsConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    collapse: CollapseConfig = field(default_factory=CollapseConfig)
    acoustics: AcousticsConfig = field(default_factory=AcousticsConfig)
    seismic: SeismicConfig = field(default_factory=SeismicConfig)
    erosion: ErosionConfig = field(default_factory=ErosionConfig)
    location: LocationConfig = field(default_factory=LocationConfig)


def get_default_config() -> SimulationConfig:
    """Retorna configuração padrão calibrada para o Vale do Revólver."""
    return SimulationConfig()
