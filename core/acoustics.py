"""
acoustics.py — Geração do som "Tó" e cálculo de SPL.

Sintetiza o som do colapso do pistão no desfiladeiro como um "vagão
desgovernado" com 4 componentes espectrais e reverberação de cânion.

Referências:
    - Georges (1973): Infrasound from convective storms
    - Bedard (2005): Low-frequency atmospheric acoustic energy
"""

import numpy as np
import os
from core.config import AcousticsConfig


def generate_toro_sound(v_impact, D_piston, M_piston, config: AcousticsConfig,
                         output_dir='output'):
    """Gera o som sintético do "Tó" — vagão desgovernado.
    
    4 componentes espectrais:
        1. Infrassom (1-5 Hz): oscilação da coluna inteira
        2. Estrondo grave (5-50 Hz): impacto do pistão
        3. Rumble (50-200 Hz): fragmentação + reverberação
        4. Crackle (200-2000 Hz): quebra de árvores/rocha
    
    Args:
        v_impact: Velocidade de impacto (m/s).
        D_piston: Diâmetro do pistão (m).
        M_piston: Massa do pistão (kg).
        config: AcousticsConfig.
        output_dir: Diretório de saída.
    
    Returns:
        dict:
            'signal': array do sinal normalizado
            'sample_rate': taxa de amostragem (Hz)
            'wav_path': caminho do arquivo WAV
            'components': descrição dos componentes
    """
    sr = config.sample_rate
    duration = config.duration
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    signal = np.zeros_like(t)
    
    # ================================================================
    # Componente 1: Infrassom (1-5 Hz)
    # Oscilação da coluna inteira
    # ================================================================
    f_infra = v_impact / D_piston if D_piston > 0 else 3.0
    f_infra = np.clip(f_infra, 1.0, 5.0)
    A_infra = 0.6  # Amplitude relativa
    tau_infra = 3.0  # s — decay
    
    signal += A_infra * np.sin(2 * np.pi * f_infra * t) * np.exp(-t / tau_infra)
    
    # ================================================================
    # Componente 2: Estrondo grave (5-50 Hz)
    # Impacto do pistão no vale
    # ================================================================
    f_boom = np.clip(v_impact / (D_piston * 0.3), 5.0, 50.0) if D_piston > 0 else 15.0
    A_boom = 1.0  # Componente mais forte
    tau_boom = 2.0
    
    signal += A_boom * np.sin(2 * np.pi * f_boom * t) * np.exp(-t / tau_boom)
    # Adicionar harmônicos
    signal += A_boom * 0.5 * np.sin(2 * np.pi * f_boom * 2 * t) * np.exp(-t / (tau_boom * 0.7))
    signal += A_boom * 0.25 * np.sin(2 * np.pi * f_boom * 3 * t) * np.exp(-t / (tau_boom * 0.5))
    
    # ================================================================
    # Componente 3: Rumble (50-200 Hz)
    # Fragmentação + reverberação no desfiladeiro
    # ================================================================
    f_rumble = 80.0
    A_rumble = 0.4
    tau_rumble = 1.5
    
    # Múltiplas frequências para som de "vagão"
    for f in [60, 80, 120, 160]:
        phase = np.random.uniform(0, 2 * np.pi)
        signal += (A_rumble / 4) * np.sin(2 * np.pi * f * t + phase) * np.exp(-t / tau_rumble)
    
    # ================================================================
    # Componente 4: Crackle (200-2000 Hz)
    # Quebra de árvores + rocha
    # ================================================================
    A_crackle = 0.15
    tau_crackle = 0.8
    
    # Ruído filtrado para simular crackle
    noise = np.random.randn(len(t))
    # Filtro passa-banda simples (média móvel para suavizar)
    window = max(1, int(sr / 2000))
    if window > 1:
        kernel = np.ones(window) / window
        noise_filtered = np.convolve(noise, kernel, mode='same')
    else:
        noise_filtered = noise
    
    signal += A_crackle * noise_filtered * np.exp(-t / tau_crackle)
    
    # ================================================================
    # Reverberação do desfiladeiro
    # ================================================================
    t_reverb = 2.0 * config.L_canyon / config.c_sound_air  # ~3s para 500m
    delay_samples = int(t_reverb * sr)
    
    if delay_samples > 0 and delay_samples < len(signal):
        reverb = np.zeros_like(signal)
        n_reflections = 5
        for i in range(1, n_reflections + 1):
            d = delay_samples * i
            decay = 0.5 ** i  # -6dB por reflexão
            if d < len(signal):
                reverb[d:] += decay * signal[:len(signal) - d]
        
        signal = signal + 0.3 * reverb
    
    # ================================================================
    # Envelope de impacto (onset rápido)
    # ================================================================
    # Rise time rápido (~50ms)
    rise_time = 0.05  # s
    rise_samples = int(rise_time * sr)
    envelope = np.ones_like(t)
    if rise_samples > 0:
        envelope[:rise_samples] = np.linspace(0, 1, rise_samples)
    signal *= envelope
    
    # ================================================================
    # Normalização
    # ================================================================
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val * 0.9  # -0.9 dBFS
    
    # ================================================================
    # Salvar WAV
    # ================================================================
    wav_path = os.path.join(output_dir, 'toro_sound.wav')
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        import scipy.io.wavfile as wavfile
        # Converter para int16
        signal_int16 = (signal * 32767).astype(np.int16)
        wavfile.write(wav_path, sr, signal_int16)
    except Exception as e:
        print(f"  [AVISO] Não foi possível salvar WAV: {e}")
        wav_path = ''
    
    components = {
        'infrasound': {'freq_Hz': float(f_infra), 'amplitude': float(A_infra), 'decay_s': float(tau_infra)},
        'boom': {'freq_Hz': float(f_boom), 'amplitude': float(A_boom), 'decay_s': float(tau_boom)},
        'rumble': {'freq_Hz': float(f_rumble), 'amplitude': float(A_rumble), 'decay_s': float(tau_rumble)},
        'crackle': {'freq_Hz': '200-2000', 'amplitude': float(A_crackle), 'decay_s': float(tau_crackle)},
    }
    
    return {
        'signal': signal,
        'sample_rate': sr,
        'wav_path': wav_path,
        'components': components,
        't_reverb_s': float(t_reverb)
    }


def compute_spl(P_impact, A_piston, distance):
    """Nível de pressão sonora (SPL) a uma dada distância.
    
    P_sound(r) = P_impact * A_piston / (4π*r²)
    SPL = 20 * log10(P_sound / P_ref)
    
    Args:
        P_impact: Pressão de impacto (Pa).
        A_piston: Área do pistão (m²).
        distance: Distância (m).
    
    Returns:
        SPL em dB (ref: 20 µPa).
    """
    P_ref = 20e-6  # Pa — limiar de audição
    
    # Pressão sonora a distância r (propagação esférica)
    P_sound = P_impact * A_piston / (4.0 * np.pi * distance ** 2)
    
    if P_sound <= 0:
        return 0.0
    
    SPL = 20.0 * np.log10(P_sound / P_ref)
    
    return float(SPL)
