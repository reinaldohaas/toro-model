"""
constants.py — Constantes físicas fundamentais para o modelo Toró.

Referências:
    - Pruppacher & Klett (1997): Microphysics of Clouds and Precipitation
    - Rogers & Yau (1989): A Short Course in Cloud Physics
"""

import numpy as np

# ============================================================================
# Gravitação
# ============================================================================
g = 9.81  # m/s² — aceleração gravitacional

# ============================================================================
# Calor Latente (J/kg)
# ============================================================================
L_v = 2.501e6   # Vaporização (líquido → vapor) a 0°C
L_f = 3.34e5     # Fusão (gelo → líquido) a 0°C
L_s = 2.834e6    # Sublimação (gelo → vapor) = L_v + L_f

# ============================================================================
# Constantes Termodinâmicas
# ============================================================================
c_p = 1004.0     # J/(kg·K) — calor específico do ar seco a pressão constante
c_v = 717.0      # J/(kg·K) — calor específico do ar seco a volume constante
c_pw = 4218.0    # J/(kg·K) — calor específico da água líquida
c_pi = 2106.0    # J/(kg·K) — calor específico do gelo
R_d = 287.04     # J/(kg·K) — constante do gás para ar seco
R_v = 461.5      # J/(kg·K) — constante do gás para vapor d'água
epsilon = R_d / R_v  # ≈ 0.622 — razão de massas moleculares

# ============================================================================
# Densidades (kg/m³)
# ============================================================================
rho_water = 1000.0   # Água líquida
rho_ice = 917.0      # Gelo
rho_air_sfc = 1.225  # Ar ao nível do mar (referência)

# ============================================================================
# Temperaturas de Referência
# ============================================================================
T_0 = 273.15      # K — 0°C
T_triple = 273.16  # K — ponto triplo da água

# ============================================================================
# Pressão de Referência
# ============================================================================
p_0 = 1.0e5  # Pa — pressão de referência (1000 hPa)

# ============================================================================
# Propriedades de Transferência
# ============================================================================
K_T = 2.4e-2       # W/(m·K) — condutividade térmica do ar a ~10°C
D_v = 2.21e-5      # m²/s — difusividade do vapor d'água no ar a ~10°C
sigma_w = 7.28e-2  # N/m — tensão superficial da água a ~20°C
mu_air = 1.81e-5   # Pa·s — viscosidade dinâmica do ar a ~15°C

# ============================================================================
# Propriedades Ópticas (para radar)
# ============================================================================
K_water_sq = 0.93   # |K|² da água — fator dielétrico
K_ice_sq = 0.176    # |K|² do gelo — fator dielétrico

# ============================================================================
# Constantes para Fórmula de Pressão de Vapor Saturante
# (Bolton 1980 — precisa para -35°C a +35°C)
# e_s(T) = 611.2 * exp(17.67 * (T - T_0) / (T - T_0 + 243.5))
# ============================================================================
BOLTON_A = 17.67
BOLTON_B = 243.5  # °C
e_s0 = 611.2      # Pa — pressão de vapor saturante a 0°C

# ============================================================================
# Constantes para Pressão de Vapor sobre Gelo
# (Murphy & Koop 2005)
# e_si(T) = exp(9.550426 - 5723.265/T + 3.53068*ln(T) - 0.00728332*T)
# ============================================================================


def e_sat_water(T):
    """Pressão de vapor saturante sobre água líquida (Bolton 1980).
    
    Args:
        T: Temperatura em Kelvin (ou array).
    
    Returns:
        Pressão de vapor saturante em Pa.
    """
    T_c = T - T_0  # Converter para Celsius
    return e_s0 * np.exp(BOLTON_A * T_c / (T_c + BOLTON_B))


def e_sat_ice(T):
    """Pressão de vapor saturante sobre gelo (Murphy & Koop 2005).
    
    Args:
        T: Temperatura em Kelvin (ou array).
    
    Returns:
        Pressão de vapor saturante sobre gelo em Pa.
    """
    return np.exp(9.550426 - 5723.265 / T + 3.53068 * np.log(T) - 0.00728332 * T)


def rho_air(T, p):
    """Densidade do ar seco pela lei dos gases ideais.
    
    Args:
        T: Temperatura em K.
        p: Pressão em Pa.
    
    Returns:
        Densidade em kg/m³.
    """
    return p / (R_d * T)


def q_sat(T, p):
    """Razão de mistura de saturação sobre água líquida.
    
    Args:
        T: Temperatura em K.
        p: Pressão em Pa.
    
    Returns:
        Razão de mistura saturante (kg/kg).
    """
    es = e_sat_water(T)
    return epsilon * es / (p - es)
