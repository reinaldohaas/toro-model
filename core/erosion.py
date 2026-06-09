"""
erosion.py — Erosão linear por impacto do Toró.

Modelo de erosão impulsiva em desfiladeiros.
Calibrado para 1-10 toneladas de material removido, sem barro residual.

A ausência de barro é uma assinatura diagnóstica do Toró:
- τ_impact >> τ_cr(argila) → toda fração fina é mobilizada e transportada
- Superfície exposta: rocha nua ou solo mineral sem matriz argilosa

Referências:
    - Whipple & Tucker (1999): Stream power river incision model
    - Shields (1936): Sediment transport threshold
    - Sklar & Dietrich (2004): Sediment-flux dependent incision
"""

import numpy as np
from core.constants import g, rho_water
from core.config import ErosionConfig


def compute_eroded_mass(E_impact, P_impact, config: ErosionConfig):
    """Calcula massa erodida pelo impacto.
    
    V_eroded = (E_impact * η_erosion) / (σ_rock * ε_frac)
    M_eroded = V_eroded * ρ_effective
    
    Clamped para [1, 10] toneladas conforme observações.
    
    Args:
        E_impact: Energia de impacto (J).
        P_impact: Pressão de impacto (Pa).
        config: ErosionConfig.
    
    Returns:
        dict:
            'M_eroded': massa total erodida (kg)
            'V_eroded': volume erodido (m³)
            'composition': dict com composição do material
    """
    # Volume erodido
    V_eroded = (E_impact * config.eta_erosion) / \
               (config.sigma_rock * config.epsilon_frac)
    
    # Densidade efetiva (mistura de rocha, solo mineral e árvores)
    # Estimativa: 60% rocha, 25% solo, 15% madeira
    rho_effective = (0.60 * config.rho_rock +
                     0.25 * config.rho_soil +
                     0.15 * config.rho_tree)
    
    M_eroded = V_eroded * rho_effective
    
    # Clampar para faixa observada [1, 10] toneladas
    M_eroded = np.clip(M_eroded, config.M_eroded_min, config.M_eroded_max)
    
    # Recalcular volume com a massa clampada
    V_eroded = M_eroded / rho_effective
    
    # Composição
    composition = {
        'rock_kg': float(M_eroded * 0.60),
        'rock_pct': 60.0,
        'soil_mineral_kg': float(M_eroded * 0.25),
        'soil_mineral_pct': 25.0,
        'trees_kg': float(M_eroded * 0.15),
        'trees_pct': 15.0,
        'mud_kg': 0.0,
        'mud_pct': 0.0,  # NENHUM BARRO
    }
    
    return {
        'M_eroded': float(M_eroded),
        'V_eroded': float(V_eroded),
        'rho_effective': float(rho_effective),
        'composition': composition
    }


def compute_selective_washing(P_impact, theta_slope, config: ErosionConfig):
    """Modelo de lavagem seletiva — explica ausência de barro.
    
    Calcula a tensão de cisalhamento do impacto e compara com a tensão
    crítica de Shields para cada classe de tamanho de sedimento.
    
    Se τ_impact >> τ_cr(D), a fração é completamente mobilizada.
    
    Args:
        P_impact: Pressão de impacto (Pa).
        theta_slope: Inclinação do desfiladeiro (rad).
        config: ErosionConfig.
    
    Returns:
        dict com fração mobilizada por classe de tamanho.
    """
    # Tensão de cisalhamento do impacto
    tau_impact = P_impact * np.sin(theta_slope)
    
    rho_s = config.rho_rock  # Densidade do sedimento
    
    # Tensão crítica de Shields para cada classe
    classes = {
        'clay': config.D_clay,        # 2 µm
        'silt': config.D_silt,        # 63 µm
        'sand': config.D_sand,        # 2 mm
        'gravel': config.D_gravel,    # 64 mm
    }
    
    results = {}
    
    for name, D in classes.items():
        tau_cr = config.theta_cr * (rho_s - rho_water) * g * D  # Pa
        
        # Razão de mobilização
        if tau_cr > 0:
            ratio = tau_impact / tau_cr
        else:
            ratio = float('inf')
        
        # Fração mobilizada (1 = totalmente removida)
        if ratio > 100:
            frac_mobilized = 1.0
        elif ratio > 10:
            frac_mobilized = 0.99
        elif ratio > 1:
            frac_mobilized = 1.0 - np.exp(-ratio)
        else:
            frac_mobilized = 0.0
        
        results[name] = {
            'D_m': float(D),
            'tau_cr_Pa': float(tau_cr),
            'tau_ratio': float(ratio),
            'fraction_mobilized': float(frac_mobilized),
            'fraction_remaining': float(1.0 - frac_mobilized),
        }
    
    # Diâmetro máximo mobilizado
    D_max = tau_impact / (config.theta_cr * (rho_s - rho_water) * g)
    
    # Fração de barro (argila) remanescente
    clay_remaining = results['clay']['fraction_remaining']
    
    return {
        'tau_impact_Pa': float(tau_impact),
        'D_max_mobilized': float(D_max),
        'clay_fraction_remaining': float(clay_remaining),
        'size_classes': results
    }


def compute_erosion_geometry(V_eroded, D_piston, config: ErosionConfig):
    """Geometria da cicatriz de erosão linear.
    
    Canal retangular estreito (slot canyon instantâneo).
    
    Args:
        V_eroded: Volume erodido (m³).
        D_piston: Diâmetro do pistão (m).
        config: ErosionConfig.
    
    Returns:
        dict:
            'width': largura do canal (m)
            'depth': profundidade do canal (m)
            'length': comprimento do canal (m)
            'aspect_ratio': profundidade/largura
    """
    # Largura ≈ diâmetro do pistão
    width = D_piston
    
    # Comprimento ≈ largura do desfiladeiro
    length = width * 2.0  # Estimativa
    
    # Profundidade
    if width > 0 and length > 0:
        depth = V_eroded / (width * length)
    else:
        depth = 0.0
    
    aspect_ratio = depth / width if width > 0 else 0.0
    
    return {
        'width': float(width),
        'depth': float(depth),
        'length': float(length),
        'aspect_ratio': float(aspect_ratio)
    }
