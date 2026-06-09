"""
thermodynamics.py — Termodinâmica para o modelo Toró.

Perfil atmosférico, flutuabilidade, supersaturação e aquecimento latente.

Referências:
    - Rogers & Yau (1989): A Short Course in Cloud Physics
    - Bolton (1980): MWR — fórmulas de temperatura potencial
"""

import numpy as np
from core.constants import (
    g, c_p, R_d, R_v, epsilon as eps, T_0, p_0,
    L_v, L_f, L_s, e_sat_water, q_sat, rho_air as compute_rho
)


class AtmosphericProfile:
    """Perfil atmosférico vertical para a simulação.
    
    Inicializa temperatura, pressão, densidade, umidade e
    temperatura virtual em cada nível da grade.
    """
    
    def __init__(self, z, T_sfc=300.0, p_sfc=101325.0, RH_sfc=0.85,
                 gamma=6.5e-3, z_tropopause=12000.0, T_tropopause=210.0):
        """
        Args:
            z: Array de altitudes (m).
            T_sfc: Temperatura na superfície (K).
            p_sfc: Pressão na superfície (Pa).
            RH_sfc: Umidade relativa na superfície (0-1).
            gamma: Taxa de diminuição troposférica (K/m).
            z_tropopause: Altura da tropopausa (m).
            T_tropopause: Temperatura na tropopausa (K).
        """
        self.z = z.copy()
        self.nz = len(z)
        
        # ================================================================
        # Temperatura T(z)
        # ================================================================
        self.T = np.zeros(self.nz)
        for i in range(self.nz):
            if z[i] <= z_tropopause:
                self.T[i] = T_sfc - gamma * z[i]
                # Não deixar cair abaixo de T_tropopause
                self.T[i] = max(self.T[i], T_tropopause)
            else:
                # Isotérmico acima da tropopausa
                self.T[i] = T_tropopause
        
        # ================================================================
        # Pressão p(z) — integração hidrostática
        # ================================================================
        self.p = np.zeros(self.nz)
        self.p[0] = p_sfc
        for i in range(1, self.nz):
            dz = z[i] - z[i - 1]
            T_mean = 0.5 * (self.T[i] + self.T[i - 1])
            # dp/dz = -ρ*g = -p*g/(R_d*T)
            self.p[i] = self.p[i - 1] * np.exp(-g * dz / (R_d * T_mean))
        
        # ================================================================
        # Razão de mistura de vapor q_v(z)
        # ================================================================
        self.q_v = np.zeros(self.nz)
        for i in range(self.nz):
            qs = q_sat(self.T[i], self.p[i])
            # RH diminui com altitude
            RH = RH_sfc * np.exp(-z[i] / 8000.0)
            RH = max(RH, 0.05)  # Mínimo 5%
            self.q_v[i] = RH * qs
        
        # ================================================================
        # Densidade ρ(z)
        # ================================================================
        self.rho = np.array([compute_rho(self.T[i], self.p[i])
                             for i in range(self.nz)])
        
        # ================================================================
        # Temperatura virtual T_v(z)
        # ================================================================
        self.T_v = self.T * (1.0 + 0.608 * self.q_v)
    
    def get_freezing_level(self):
        """Retorna a altitude do nível de congelamento (0°C) em metros."""
        for i in range(self.nz):
            if self.T[i] <= T_0:
                if i == 0:
                    return self.z[0]
                # Interpolação linear
                frac = (T_0 - self.T[i]) / (self.T[i - 1] - self.T[i])
                return self.z[i] - frac * (self.z[i] - self.z[i - 1])
        return self.z[-1]
    
    def get_hm_zone(self):
        """Retorna (z_bottom, z_top) da zona Hallett-Mossop (-3°C a -8°C)."""
        T_top = T_0 - 8.0   # -8°C (mais frio, mais alto)
        T_bottom = T_0 - 3.0  # -3°C (mais quente, mais baixo)
        
        z_bottom = None
        z_top = None
        
        for i in range(self.nz):
            if self.T[i] <= T_bottom and z_bottom is None:
                z_bottom = self.z[i]
            if self.T[i] <= T_top and z_top is None:
                z_top = self.z[i]
                break
        
        return (z_bottom or 0, z_top or self.z[-1])


def compute_virtual_temperature(T, q_v, q_l=0.0, q_i=0.0):
    """Temperatura virtual com carregamento de hidrometeoros.
    
    T_v = T * (1 + 0.608*q_v - q_l - q_i)
    
    Args:
        T: Temperatura (K).
        q_v: Razão de mistura de vapor (kg/kg).
        q_l: Razão de mistura de água líquida (kg/kg).
        q_i: Razão de mistura de gelo (kg/kg).
    
    Returns:
        Temperatura virtual (K).
    """
    return T * (1.0 + 0.608 * q_v - q_l - q_i)


def compute_buoyancy(T_parcel, q_v_parcel, q_l, q_i, T_env, q_v_env):
    """Flutuabilidade da parcela incluindo carregamento.
    
    B = g * [(T_v,parcel - T_v,env) / T_v,env] - g * (q_l + q_i)
    
    O termo de carregamento inclui o peso de todos os hidrometeoros.
    
    Args:
        T_parcel: Temperatura da parcela (K).
        q_v_parcel: Razão de mistura de vapor da parcela (kg/kg).
        q_l: Conteúdo de água líquida (kg/kg ou kg/m³ normalizado).
        q_i: Conteúdo de gelo (kg/kg ou kg/m³ normalizado).
        T_env: Temperatura ambiente (K).
        q_v_env: Razão de mistura de vapor ambiente (kg/kg).
    
    Returns:
        Flutuabilidade (m/s²). Positivo = ascendente.
    """
    T_v_parcel = compute_virtual_temperature(T_parcel, q_v_parcel, q_l, q_i)
    T_v_env = compute_virtual_temperature(T_env, q_v_env)
    
    # Flutuabilidade térmica - carregamento
    B = g * (T_v_parcel - T_v_env) / T_v_env
    
    return B


def compute_supersaturation(T, p, q_v):
    """Supersaturação sobre água líquida.
    
    S = e/e_s - 1
    
    Args:
        T: Temperatura (K).
        p: Pressão (Pa).
        q_v: Razão de mistura de vapor (kg/kg).
    
    Returns:
        Supersaturação (adimensional). S > 0 → supersaturado.
    """
    # Pressão parcial do vapor
    e = q_v * p / (eps + q_v)
    
    # Pressão de saturação
    e_s = e_sat_water(T)
    
    if e_s <= 0:
        return 0.0
    
    return e / e_s - 1.0


def latent_heating(dq_condensed, dq_frozen, dq_deposited):
    """Taxa de aquecimento latente (K/s).
    
    dT/dt = (L_v/c_p)*C + (L_f/c_p)*F + (L_s/c_p)*D
    
    Args:
        dq_condensed: Taxa de condensação (kg/m³/s). Positivo = condensação.
        dq_frozen: Taxa de congelamento (kg/m³/s). Positivo = congelamento.
        dq_deposited: Taxa de deposição (kg/m³/s). Positivo = deposição.
    
    Returns:
        Taxa de aquecimento (K/s).
    """
    dT = (L_v / c_p) * dq_condensed + \
         (L_f / c_p) * dq_frozen + \
         (L_s / c_p) * dq_deposited
    
    return dT


def moist_adiabatic_lapse_rate(T, p):
    """Taxa de diminuição adiabática saturada (K/m).
    
    Γ_m = (g/c_p) * [1 + (L_v*q_s)/(R_d*T)] / [1 + (L_v²*q_s)/(c_p*R_v*T²)]
    
    Args:
        T: Temperatura (K).
        p: Pressão (Pa).
    
    Returns:
        Lapse rate (K/m). Sempre positivo (T diminui com z).
    """
    qs = q_sat(T, p)
    
    numerator = 1.0 + (L_v * qs) / (R_d * T)
    denominator = 1.0 + (L_v ** 2 * qs) / (c_p * R_v * T ** 2)
    
    return (g / c_p) * numerator / denominator


def compute_cape_cin(profile):
    """Calcula CAPE e CIN do perfil atmosférico.
    
    CAPE = ∫(LFC→EL) max(0, B) dz
    CIN = ∫(sfc→LFC) min(0, B) dz
    
    Usa uma parcela levantada adiabaticamente da superfície.
    
    Args:
        profile: AtmosphericProfile.
    
    Returns:
        dict: {'CAPE': float, 'CIN': float, 'LFC': float, 'EL': float}
    """
    z = profile.z
    T_env = profile.T
    p = profile.p
    nz = profile.nz
    
    # Levantar parcela da superfície
    T_parcel = np.zeros(nz)
    T_parcel[0] = profile.T[0] + 3.0  # Com perturbação
    q_v_parcel = profile.q_v[0]
    
    saturated = False
    
    for i in range(1, nz):
        dz = z[i] - z[i - 1]
        
        if not saturated:
            # Adiabática seca
            T_parcel[i] = T_parcel[i - 1] - (g / c_p) * dz
            
            # Verificar saturação
            qs = q_sat(T_parcel[i], p[i])
            if q_v_parcel >= qs:
                saturated = True
                T_parcel[i] = T_parcel[i - 1] - moist_adiabatic_lapse_rate(
                    T_parcel[i - 1], p[i - 1]) * dz
        else:
            # Adiabática saturada
            gamma_m = moist_adiabatic_lapse_rate(T_parcel[i - 1], p[i - 1])
            T_parcel[i] = T_parcel[i - 1] - gamma_m * dz
    
    # Calcular flutuabilidade em cada nível
    B = g * (T_parcel - T_env) / T_env
    
    # CAPE e CIN
    CAPE = 0.0
    CIN = 0.0
    LFC = 0.0
    EL = 0.0
    
    found_lfc = False
    
    for i in range(1, nz):
        dz = z[i] - z[i - 1]
        B_mean = 0.5 * (B[i] + B[i - 1])
        
        if B_mean > 0 and found_lfc:
            CAPE += B_mean * dz
            EL = z[i]
        elif B_mean > 0 and not found_lfc:
            found_lfc = True
            LFC = z[i]
            CAPE += B_mean * dz
            EL = z[i]
        elif B_mean < 0 and not found_lfc:
            CIN += B_mean * dz
    
    return {'CAPE': CAPE, 'CIN': CIN, 'LFC': LFC, 'EL': EL}
