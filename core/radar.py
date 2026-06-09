"""
radar.py — Refletividade radar sintética e detecção de BWER.

Calcula Z(dBZ) a partir do espectro de bins e gera RHI sintético.

Referências:
    - Doviak & Zrnić (1993): Doppler Radar and Weather Observations
"""

import numpy as np
from core.constants import K_water_sq, K_ice_sq


def compute_reflectivity(spectra, bin_grid, z_array):
    """Calcula refletividade radar equivalente em cada nível.
    
    Z = Σⱼ Nⱼ · Dⱼ⁶ · ΔDⱼ   (mm⁶/m³)
    
    Para gelo: aplica fator dielétrico |K_ice|²/|K_water|²
    
    Args:
        spectra: HydrometeorSpectra.
        bin_grid: BinGrid.
        z_array: Array de altitudes (m).
    
    Returns:
        Array Z_dBZ (nz,).
    """
    nz = len(z_array)
    Z_dBZ = np.full(nz, -30.0)  # dBZ mínimo
    
    D_mm = bin_grid.centers * 1e3  # m → mm
    D6 = D_mm ** 6                  # mm⁶
    dD = bin_grid.widths            # m
    
    for iz in range(nz):
        Z_total = 0.0
        
        # Contribuição líquida (categorias 0 e 1)
        for cat in [0, 1]:  # Cloud, Rain
            Z_liquid = np.sum(spectra.N[cat, :, iz] * D6 * dD)
            Z_total += Z_liquid
        
        # Contribuição de gelo (categorias 2, 3, 4)
        # Fator dielétrico: |K_ice|²/|K_water|²
        ice_factor = K_ice_sq / K_water_sq
        for cat in [2, 3, 4]:  # Ice, Graupel, Hail
            Z_ice = np.sum(spectra.N[cat, :, iz] * D6 * dD)
            Z_total += ice_factor * Z_ice
        
        if Z_total > 0:
            Z_dBZ[iz] = 10.0 * np.log10(Z_total)
    
    return Z_dBZ


def detect_bwer(Z_dBZ, w, z_array):
    """Detecta Bounded Weak Echo Region (BWER).
    
    BWER = região de eco fraco limitada por eco forte acima e ao redor.
    Ocorre onde a ascendente é forte (w > 20 m/s) e a refletividade
    é significativamente menor que os níveis adjacentes.
    
    Args:
        Z_dBZ: Array de refletividade (nz,).
        w: Array de velocidade vertical (m/s).
        z_array: Array de altitudes (m).
    
    Returns:
        dict:
            'detected': bool
            'z_min': altitude inferior do BWER (m)
            'z_max': altitude superior do BWER (m)
            'Z_deficit': déficit de refletividade (dBZ)
            'w_in_bwer': velocidade vertical máxima no BWER
    """
    nz = len(z_array)
    
    # Critério: w > 15 m/s e Z menor que vizinhos
    bwer_levels = []
    
    for i in range(2, nz - 2):
        if w[i] > 15.0:
            # Média dos vizinhos (±2 níveis)
            Z_surround = np.mean([Z_dBZ[i - 2], Z_dBZ[i - 1],
                                   Z_dBZ[i + 1], Z_dBZ[i + 2]])
            Z_deficit = Z_surround - Z_dBZ[i]
            
            if Z_deficit > 5.0:  # Pelo menos 5 dBZ de diferença
                bwer_levels.append({
                    'iz': i,
                    'z': z_array[i],
                    'deficit': Z_deficit,
                    'w': w[i]
                })
    
    if not bwer_levels:
        return {
            'detected': False,
            'z_min': 0,
            'z_max': 0,
            'Z_deficit': 0,
            'w_in_bwer': 0
        }
    
    z_min = min(b['z'] for b in bwer_levels)
    z_max = max(b['z'] for b in bwer_levels)
    max_deficit = max(b['deficit'] for b in bwer_levels)
    max_w = max(b['w'] for b in bwer_levels)
    
    return {
        'detected': True,
        'z_min': float(z_min),
        'z_max': float(z_max),
        'Z_deficit': float(max_deficit),
        'w_in_bwer': float(max_w)
    }


def generate_rhi_slice(Z_dBZ, z_array, r_array):
    """Gera fatia RHI (Range-Height Indicator) sintética.
    
    Espalha a refletividade da coluna 1D horizontalmente usando um
    modelo simples de perfil gaussiano com a distância.
    
    Args:
        Z_dBZ: Refletividade da coluna (nz,).
        z_array: Altitudes (m).
        r_array: Distâncias horizontais (m).
    
    Returns:
        Array (nz, nr) de Z_dBZ para o RHI.
    """
    nz = len(z_array)
    nr = len(r_array)
    
    rhi = np.full((nz, nr), -30.0)
    
    # Raio da tempestade (escala horizontal do eco)
    R_storm = 10000.0  # 10 km
    
    for iz in range(nz):
        for ir in range(nr):
            r = r_array[ir]
            
            # Perfil gaussiano horizontal
            # Eco mais forte perto do centro, decai com a distância
            attenuation = np.exp(-0.5 * (r / R_storm) ** 2)
            
            rhi[iz, ir] = Z_dBZ[iz] + 10.0 * np.log10(max(attenuation, 1e-10))
    
    return rhi
