"""
seismic.py — Modelo sísmico do impacto do Toró.

Calcula magnitude Richter e gera sismograma sintético.
Calibrado para M 2-3 conforme observações no Vale do Revólver.

Referências:
    - Gutenberg & Richter (1956): Energy-magnitude relation
    - Ricker (1953): Wavelet for seismic modeling
"""

import numpy as np
from core.config import SeismicConfig


def compute_seismic_magnitude(E_impact, config: SeismicConfig):
    """Calcula magnitude local (Richter) a partir da energia de impacto.
    
    E_seis = η * E_impact
    M_L = (2/3) * log10(E_seis) - 1.17
    
    Args:
        E_impact: Energia cinética do impacto (J).
        config: SeismicConfig.
    
    Returns:
        tuple (M_L, E_seis):
            M_L: Magnitude local (Richter)
            E_seis: Energia sísmica (J)
    """
    E_seis = config.eta_seismic * E_impact
    
    if E_seis <= 0:
        return 0.0, 0.0
    
    M_L = (2.0 / 3.0) * np.log10(E_seis) - 1.17
    
    return float(M_L), float(E_seis)


def generate_seismogram(M_L, f_dominant, duration, dt, config: SeismicConfig):
    """Gera sismograma sintético usando wavelet de Ricker.
    
    a(t) = (1 - 2π²f²t²) * exp(-π²f²t²)
    
    Args:
        M_L: Magnitude local.
        f_dominant: Frequência dominante (Hz).
        duration: Duração do sismograma (s).
        dt: Passo temporal (s).
        config: SeismicConfig.
    
    Returns:
        dict:
            't': array de tempo (s)
            'amplitude': array de amplitude
            'M_L': magnitude
            'f_dominant': frequência dominante
    """
    t = np.arange(-duration / 2, duration / 2, dt)
    
    # Wavelet de Ricker
    f = f_dominant
    u = (np.pi * f * t) ** 2
    ricker = (1.0 - 2.0 * u) * np.exp(-u)
    
    # Escalar amplitude pela magnitude
    # A ∝ 10^(M_L/2) (escala logarítmica)
    A_scale = 10.0 ** (M_L / 2.0) * 1e-6  # µm/s (escala de velocidade do solo)
    
    amplitude = A_scale * ricker
    
    # Adicionar coda (ondas refletidas, decaindo)
    coda_decay = np.exp(-np.abs(t) / 2.0)
    noise = np.random.randn(len(t)) * 0.1 * A_scale
    amplitude += noise * coda_decay
    
    # Shiftar para t > 0
    t_shifted = t + duration / 2
    
    return {
        't': t_shifted,
        'amplitude': amplitude,
        'M_L': float(M_L),
        'f_dominant': float(f_dominant)
    }


def compute_attenuation(A0, r, config: SeismicConfig):
    """Atenuação sísmica com a distância.
    
    A(r) = A0 * r^(-1.5) * exp(-α*r)
    
    Args:
        A0: Amplitude na fonte.
        r: Distância (m).
        config: SeismicConfig.
    
    Returns:
        Amplitude atenuada.
    """
    if r <= 0:
        return A0
    
    return A0 * r ** (-1.5) * np.exp(-config.alpha_atten * r)
