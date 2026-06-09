"""
microphysics.py — Bin microphysics para o modelo Toró.

Implementa microfísica explícita com 50 bins logarítmicos e 5 categorias
de hidrometeoros. Inclui:
    - DSD Gamma estreita (condição inicial)
    - Condensação/evaporação difusional
    - Colisão-coalescência (kernel gravitacional)
    - Riming (acreção de gotas por gelo)
    - Hallett-Mossop SIP (-3°C a -8°C)
    - Quebra colisional (Phillips et al. 2017)
    - Nucleação heterogênea (Fletcher 1962)

Referências:
    - Pruppacher & Klett (1997): Microphysics of Clouds and Precipitation
    - Hallett & Mossop (1974): Nature 249, 26-28
    - Phillips et al. (2017): J. Atmos. Sci.
    - Hall (1980): Coalescence efficiency tables
    - Bott (1998): J. Atmos. Sci. — flux method
"""

import numpy as np
from core.constants import (
    g, rho_water, rho_ice, T_0, L_v, L_f, c_p,
    K_T, D_v, sigma_w, mu_air, e_sat_water, e_sat_ice,
    R_v, epsilon as eps_Rd
)

# ============================================================================
# Categorias de hidrometeoros
# ============================================================================
CAT_CLOUD = 0    # Gotas de nuvem
CAT_RAIN = 1     # Chuva
CAT_ICE = 2      # Cristais de gelo
CAT_GRAUPEL = 3  # Graupel
CAT_HAIL = 4     # Granizo
N_CATEGORIES = 5


class BinGrid:
    """Grade de bins logarítmicos para o espectro de tamanhos.
    
    Os bins são espaçados logaritmicamente entre D_min e D_max.
    Cada bin é definido por suas bordas (edges), centro e largura.
    """
    
    def __init__(self, n_bins=50, D_min=2e-6, D_max=0.08):
        """
        Args:
            n_bins: Número de bins.
            D_min: Diâmetro mínimo (m).
            D_max: Diâmetro máximo (m).
        """
        self.n_bins = n_bins
        self.D_min = D_min
        self.D_max = D_max
        
        # Bordas dos bins (n_bins + 1)
        self.edges = np.logspace(np.log10(D_min), np.log10(D_max), n_bins + 1)
        
        # Centros dos bins (média geométrica)
        self.centers = np.sqrt(self.edges[:-1] * self.edges[1:])
        
        # Largura dos bins
        self.widths = self.edges[1:] - self.edges[:-1]
        
        # Raios dos centros
        self.radii = self.centers / 2.0
        
        # Massa de uma gota esférica no centro de cada bin (kg)
        self.mass_water = (np.pi / 6.0) * rho_water * self.centers ** 3
        self.mass_ice = (np.pi / 6.0) * rho_ice * self.centers ** 3
        
        # Velocidades terminais pré-calculadas
        self.v_term_water = np.array([
            self._v_terminal_droplet(D) for D in self.centers
        ])
        self.v_term_ice = np.array([
            self._v_terminal_ice(D) for D in self.centers
        ])
    
    def find_bin(self, D):
        """Encontra o índice do bin para um dado diâmetro.
        
        Args:
            D: Diâmetro em metros.
        
        Returns:
            Índice do bin (0 a n_bins-1), ou -1 se fora do range.
        """
        if D < self.D_min or D > self.D_max:
            return -1
        idx = np.searchsorted(self.edges, D) - 1
        return min(idx, self.n_bins - 1)
    
    @staticmethod
    def _v_terminal_droplet(D):
        """Velocidade terminal de gota de água (m/s).
        
        Usa Stokes para D < 80µm, Beard (1976) para maiores.
        
        Args:
            D: Diâmetro em metros.
        """
        r = D / 2.0
        
        if D < 80e-6:
            # Regime de Stokes: v = (2/9) * (rho_w * g * r²) / mu_air
            return (2.0 / 9.0) * rho_water * g * r ** 2 / mu_air
        elif D < 1.2e-3:
            # Regime intermediário (Beard 1976, simplificado)
            # v ≈ 4.5e3 * D^1.0 (D em metros, v em m/s)
            return 4.5e3 * D
        else:
            # Gotas grandes (> 1.2mm)
            # v ≈ 9.65 - 10.3 * exp(-600*D) (Atlas & Ulbrich)
            return 9.65 - 10.3 * np.exp(-600.0 * D)
    
    @staticmethod
    def _v_terminal_ice(D, rho_particle=400.0):
        """Velocidade terminal de partícula de gelo (m/s).
        
        Mitchell (1996) parameterization.
        
        Args:
            D: Diâmetro em metros.
            rho_particle: Densidade efetiva da partícula (kg/m³).
        """
        # Aproximação simplificada: v = a * D^b
        # Para graupel (rho~400): a~120, b~0.5
        # Para cristais (rho~100): a~40, b~0.4
        if rho_particle > 300:
            # Graupel / granizo
            a, b = 120.0, 0.5
        elif rho_particle > 100:
            # Neve / agregados
            a, b = 40.0, 0.4
        else:
            # Cristais planares
            a, b = 20.0, 0.3
        return a * D ** b


class HydrometeorSpectra:
    """Espectro de hidrometeoros em cada nível vertical.
    
    Armazena N(D) — concentração por bin — para 5 categorias
    em cada nível da grade vertical.
    
    Shape: (n_categories, n_bins, nz)
    Unidades: m⁻³ por bin (número de partículas por m³ em cada bin)
    """
    
    def __init__(self, n_bins=50, nz=200):
        self.n_bins = n_bins
        self.nz = nz
        self.N = np.zeros((N_CATEGORIES, n_bins, nz))
    
    def get_column(self, iz):
        """Retorna espectro para um nível vertical.
        
        Returns:
            Array (n_categories, n_bins).
        """
        return self.N[:, :, iz].copy()
    
    def set_column(self, iz, spectra_col):
        """Define espectro para um nível vertical.
        
        Args:
            iz: Índice do nível.
            spectra_col: Array (n_categories, n_bins).
        """
        self.N[:, :, iz] = spectra_col
    
    def total_number(self, category, iz):
        """Concentração total de uma categoria em um nível (m⁻³)."""
        return np.sum(self.N[category, :, iz])


def init_dsd_gamma(mu, D_mean, LWC, bin_grid):
    """Inicializa uma DSD Gamma estreita.
    
    N(D) = N0 * D^μ * exp(-Λ*D)
    
    Args:
        mu: Parâmetro de forma (alto → estreita). Típico 15-25.
        D_mean: Diâmetro médio (m).
        LWC: Conteúdo de água líquida alvo (kg/m³).
        bin_grid: BinGrid instance.
    
    Returns:
        Array (n_bins,) com concentração em cada bin.
    """
    # Lambda da distribuição Gamma
    Lambda = (mu + 1.0) / D_mean  # 1/m
    
    # N(D) sem N0
    D = bin_grid.centers
    dD = bin_grid.widths
    N_shape = D ** mu * np.exp(-Lambda * D)
    
    # Massa total sem N0
    mass_per_particle = (np.pi / 6.0) * rho_water * D ** 3  # kg
    M_integral = np.sum(N_shape * mass_per_particle * dD)
    
    # Ajustar N0 para atingir LWC alvo
    if M_integral > 0:
        N0 = LWC / M_integral
    else:
        N0 = 0.0
    
    N_D = N0 * N_shape  # m⁻³/m → concentração por bin
    
    return N_D


def compute_lwc(spectra, bin_grid):
    """Conteúdo de água líquida em cada nível (kg/m³).
    
    Soma sobre as categorias líquidas (nuvem + chuva).
    """
    lwc = np.zeros(spectra.nz)
    mass = bin_grid.mass_water  # (n_bins,)
    dD = bin_grid.widths
    
    for iz in range(spectra.nz):
        # Gotas de nuvem + chuva
        for cat in [CAT_CLOUD, CAT_RAIN]:
            lwc[iz] += np.sum(spectra.N[cat, :, iz] * mass * dD)
    
    return lwc


def compute_iwc(spectra, bin_grid):
    """Conteúdo de gelo em cada nível (kg/m³).
    
    Soma sobre categorias de gelo (cristais + graupel + granizo).
    """
    iwc = np.zeros(spectra.nz)
    mass = bin_grid.mass_ice  # (n_bins,)
    dD = bin_grid.widths
    
    for iz in range(spectra.nz):
        for cat in [CAT_ICE, CAT_GRAUPEL, CAT_HAIL]:
            iwc[iz] += np.sum(spectra.N[cat, :, iz] * mass * dD)
    
    return iwc


def _condensation_growth_rate(r, T, p, S):
    """Taxa de crescimento difusional de uma gota (m/s).
    
    r * dr/dt = (S - S_eq) / (F_k + F_d)
    
    Retorna dr/dt em m/s.
    
    Args:
        r: Raio da gota (m).
        T: Temperatura (K).
        p: Pressão (Pa).
        S: Supersaturação (adimensional, S = e/e_s - 1).
    """
    if r < 1e-9:
        return 0.0
    
    e_s = e_sat_water(T)
    rho_vs = e_s / (R_v * T)  # kg/m³ — densidade de vapor saturante
    
    # Fatores de resistência
    F_k = (L_v ** 2 * rho_water) / (K_T * R_v * T ** 2)  # s/m²
    F_d = rho_water / (D_v * rho_vs)  # s/m²
    
    # dr/dt = S / (r * (F_k + F_d))
    # Ignorando curvatura (Kelvin) e soluto para gotas já ativadas
    dr_dt = S / (r * (F_k + F_d))
    
    return dr_dt


def condensation_growth(T, p, S, spectra_col, bin_grid, dt):
    """Crescimento/evaporação por difusão de todas as gotas.
    
    Args:
        T: Temperatura (K).
        p: Pressão (Pa).
        S: Supersaturação.
        spectra_col: Array (n_categories, n_bins) — espectro neste nível.
        bin_grid: BinGrid.
        dt: Passo temporal (s).
    
    Returns:
        dict com:
            'spectra': espectro atualizado
            'dq_condensed': taxa de condensação (kg/m³/s)
    """
    updated = spectra_col.copy()
    dq_condensed = 0.0
    
    # Apenas categorias líquidas
    for cat in [CAT_CLOUD, CAT_RAIN]:
        for j in range(bin_grid.n_bins):
            N_j = updated[cat, j]
            if N_j < 1e-10:
                continue
            
            r = bin_grid.radii[j]
            dr_dt = _condensation_growth_rate(r, T, p, S)
            
            if abs(dr_dt) < 1e-15:
                continue
            
            # Novo raio
            r_new = r + dr_dt * dt
            if r_new < 1e-9:
                # Evaporação completa
                updated[cat, j] = 0.0
                dq_condensed -= N_j * bin_grid.mass_water[j] / dt
                continue
            
            D_new = 2.0 * r_new
            j_new = bin_grid.find_bin(D_new)
            
            if j_new < 0:
                continue
            
            if j_new != j:
                # Mover partículas para novo bin
                mass_change = N_j * (bin_grid.mass_water[j_new] - bin_grid.mass_water[j])
                dq_condensed += mass_change / dt
                updated[cat, j] -= N_j
                updated[cat, j_new] += N_j
            else:
                # Permanece no mesmo bin, mas massa muda
                mass_change = N_j * (4.0 / 3.0 * np.pi * rho_water *
                                     (r_new ** 3 - r ** 3))
                dq_condensed += mass_change / dt
    
    # Garantir não-negatividade
    updated = np.maximum(updated, 0.0)
    
    return {'spectra': updated, 'dq_condensed': dq_condensed}


def _coalescence_efficiency(r1, r2):
    """Eficiência de coalescência (Hall 1980, simplificada).
    
    Args:
        r1, r2: Raios das gotas (m), r1 >= r2.
    
    Returns:
        Eficiência E_coal (0-1).
    """
    ratio = min(r1, r2) / max(r1, r2) if max(r1, r2) > 0 else 0
    
    # Simplificação da tabela de Hall (1980)
    R_large = max(r1, r2)
    
    if R_large < 15e-6:
        return 0.0  # Gotas muito pequenas não coalescem
    elif R_large < 50e-6:
        return 0.02 * ratio
    elif R_large < 100e-6:
        return 0.1 + 0.5 * ratio
    elif R_large < 300e-6:
        return 0.6 + 0.3 * ratio
    else:
        return 0.8 + 0.15 * ratio


def collision_coalescence(spectra_col, bin_grid, dt):
    """Colisão-coalescência gravitacional (kernel Long 1974).
    
    K(r1,r2) = π*(r1+r2)² * |V(r1)-V(r2)| * E_coal
    
    Usa método simplificado (não o Bott completo) para eficiência.
    
    Args:
        spectra_col: Array (n_categories, n_bins).
        bin_grid: BinGrid.
        dt: Passo temporal (s).
    
    Returns:
        Espectro atualizado.
    """
    updated = spectra_col.copy()
    
    # Apenas categorias líquidas
    for cat in [CAT_CLOUD, CAT_RAIN]:
        N = updated[cat, :].copy()
        r = bin_grid.radii
        v = bin_grid.v_term_water
        
        # Calcular ganhos e perdas por colisão
        dN = np.zeros_like(N)
        
        for i in range(bin_grid.n_bins):
            if N[i] < 1e-10:
                continue
            for j in range(i, bin_grid.n_bins):
                if N[j] < 1e-10:
                    continue
                
                # Kernel de colisão
                R_sum = r[i] + r[j]
                dV = abs(v[i] - v[j])
                E_coal = _coalescence_efficiency(r[i], r[j])
                
                K = np.pi * R_sum ** 2 * dV * E_coal
                
                # Taxa de colisão
                if i == j:
                    rate = 0.5 * K * N[i] * N[j] * dt
                else:
                    rate = K * N[i] * N[j] * dt
                
                if rate < 1e-20:
                    continue
                
                # Massa resultante
                m_new = bin_grid.mass_water[i] + bin_grid.mass_water[j]
                D_new = (6.0 * m_new / (np.pi * rho_water)) ** (1.0 / 3.0)
                k_new = bin_grid.find_bin(D_new)
                
                if k_new >= 0:
                    # Perda nos bins originais
                    dN[i] -= rate
                    if i != j:
                        dN[j] -= rate
                    # Ganho no novo bin
                    dN[k_new] += rate
        
        updated[cat, :] = np.maximum(N + dN, 0.0)
        
        # Promover gotas grandes para chuva
        if cat == CAT_CLOUD:
            for j in range(bin_grid.n_bins):
                if bin_grid.centers[j] > 100e-6:  # > 100 µm → chuva
                    updated[CAT_RAIN, j] += updated[CAT_CLOUD, j]
                    updated[CAT_CLOUD, j] = 0.0
    
    return updated


def riming(spectra_col, bin_grid, T, dt, config):
    """Riming — acreção de gotas de nuvem por graupel/granizo.
    
    dm_rime/dt = (π/4) * E_c * |V_g - V_d| * D_g² * LWC
    
    Args:
        spectra_col: Array (n_categories, n_bins).
        bin_grid: BinGrid.
        T: Temperatura (K).
        dt: Passo temporal (s).
        config: MicrophysicsConfig.
    
    Returns:
        dict:
            'spectra': espectro atualizado
            'dm_rime_dt': taxa de riming total (kg/m³/s)
    """
    if T > T_0:
        return {'spectra': spectra_col.copy(), 'dm_rime_dt': 0.0}
    
    updated = spectra_col.copy()
    dm_rime_total = 0.0
    
    # LWC disponível para riming
    lwc_available = np.sum(updated[CAT_CLOUD, :] * bin_grid.mass_water * bin_grid.widths)
    
    if lwc_available < 1e-10:
        return {'spectra': updated, 'dm_rime_dt': 0.0}
    
    E_c = config.E_collection
    
    for cat_ice in [CAT_GRAUPEL, CAT_HAIL]:
        for j in range(bin_grid.n_bins):
            N_ice = updated[cat_ice, j]
            if N_ice < 1e-10:
                continue
            
            D_g = bin_grid.centers[j]
            V_g = bin_grid.v_term_ice[j]
            
            # Taxa de acreção por partícula de gelo
            # dm/dt = (π/4) * E_c * |V_g - V_d_mean| * D_g² * LWC
            V_d_mean = 1.0  # m/s — velocidade média das gotas de nuvem
            dm_dt = (np.pi / 4.0) * E_c * abs(V_g - V_d_mean) * D_g ** 2 * lwc_available
            
            dm_rime = dm_dt * N_ice * dt  # kg/m³ rimado neste passo
            
            # Limitar ao LWC disponível
            dm_rime = min(dm_rime, lwc_available * 0.5)
            
            dm_rime_total += dm_rime / dt
            
            # Remover massa de gotas de nuvem (proporcionalmente)
            if lwc_available > 0:
                frac_removed = dm_rime / lwc_available
                updated[CAT_CLOUD, :] *= (1.0 - min(frac_removed, 0.9))
            
            # Crescer o graupel (mover para bin maior)
            m_new = bin_grid.mass_ice[j] + dm_rime / max(N_ice, 1e-10)
            D_new = (6.0 * m_new / (np.pi * rho_ice)) ** (1.0 / 3.0)
            j_new = bin_grid.find_bin(D_new)
            if j_new >= 0 and j_new != j:
                updated[cat_ice, j_new] += updated[cat_ice, j]
                updated[cat_ice, j] = 0.0
    
    return {'spectra': updated, 'dm_rime_dt': dm_rime_total}


def hallett_mossop_sip(T, dm_rime_dt, spectra_col, bin_grid, config):
    """Produção secundária de gelo — Hallett-Mossop.
    
    dN_sip/dt = 350 splinters/mg * f(T) * dm_rime/dt
    
    Ativo apenas entre -3°C e -8°C.
    Requer: gotas <13µm E gotas >24µm simultaneamente.
    
    Args:
        T: Temperatura (K).
        dm_rime_dt: Taxa de riming (kg/m³/s).
        spectra_col: Array (n_categories, n_bins).
        bin_grid: BinGrid.
        config: MicrophysicsConfig.
    
    Returns:
        dict:
            'dN_sip': taxa de produção de splinters (m⁻³/s)
            'spectra': espectro atualizado (com novos cristais)
    """
    T_c = T - T_0  # Celsius
    
    # Verificar faixa de temperatura
    if T_c > config.hm_T_max or T_c < config.hm_T_min:
        return {'dN_sip': 0.0, 'spectra': spectra_col.copy()}
    
    # Verificar presença de gotas pequenas (<13µm) e grandes (>24µm)
    has_small = False
    has_large = False
    for j in range(bin_grid.n_bins):
        if spectra_col[CAT_CLOUD, j] > 1e-5:
            if bin_grid.centers[j] < config.hm_D_small:
                has_small = True
            if bin_grid.centers[j] > config.hm_D_large:
                has_large = True
    
    if not (has_small and has_large):
        return {'dN_sip': 0.0, 'spectra': spectra_col.copy()}
    
    # Função f(T): triangular, pico em -5°C
    T_peak = config.hm_T_peak  # -5°C
    T_max = config.hm_T_max    # -3°C
    T_min = config.hm_T_min    # -8°C
    
    if T_c >= T_peak:
        f_T = (T_c - T_max) / (T_peak - T_max)
    else:
        f_T = (T_c - T_min) / (T_peak - T_min)
    
    f_T = max(0.0, min(1.0, f_T))
    
    # Taxa de SIP: 350 splinters por mg de gelo rimado
    # dm_rime_dt em kg/m³/s → converter para mg/m³/s
    dm_rime_mg = dm_rime_dt * 1e6  # kg → mg
    
    dN_sip = config.hm_rate * f_T * dm_rime_mg  # splinters/m³/s
    
    # Adicionar cristais de gelo nos bins pequenos (~50µm)
    updated = spectra_col.copy()
    if dN_sip > 0:
        # Cristais de ~50µm de diâmetro
        j_splinter = bin_grid.find_bin(50e-6)
        if j_splinter >= 0:
            updated[CAT_ICE, j_splinter] += dN_sip  # Acumula por segundo
    
    return {'dN_sip': dN_sip, 'spectra': updated}


def collisional_breakup(spectra_col, bin_grid, T, config):
    """Quebra colisional — Phillips et al. (2017).
    
    N_frag = α * CKE^β * g(T) * h(ρ_rime)
    CKE = ½ * m_reduced * ΔV²
    
    Args:
        spectra_col: Array (n_categories, n_bins).
        bin_grid: BinGrid.
        T: Temperatura (K).
        config: MicrophysicsConfig.
    
    Returns:
        dict:
            'dN_frag': taxa de fragmentação (m⁻³/s)
            'spectra': espectro atualizado
    """
    T_c = T - T_0
    
    # g(T): eficiência dependente da temperatura, pico em -15°C
    T_peak = config.coll_T_peak   # -15°C
    T_width = config.coll_T_width  # 10°C
    g_T = np.exp(-0.5 * ((T_c - T_peak) / T_width) ** 2)
    
    updated = spectra_col.copy()
    dN_frag_total = 0.0
    
    # Colisões graupel-graupel e graupel-granizo
    for cat1 in [CAT_GRAUPEL, CAT_HAIL]:
        for cat2 in [CAT_GRAUPEL, CAT_HAIL]:
            for i in range(bin_grid.n_bins):
                N1 = updated[cat1, i]
                if N1 < 1e-10:
                    continue
                for j in range(i, bin_grid.n_bins):
                    N2 = updated[cat2, j]
                    if N2 < 1e-10:
                        continue
                    
                    # Massa reduzida
                    m1 = bin_grid.mass_ice[i]
                    m2 = bin_grid.mass_ice[j]
                    m_red = (m1 * m2) / (m1 + m2)
                    
                    # Velocidade relativa
                    dV = abs(bin_grid.v_term_ice[i] - bin_grid.v_term_ice[j])
                    if dV < 0.1:
                        continue
                    
                    # CKE
                    CKE = 0.5 * m_red * dV ** 2
                    
                    if CKE < 1e-8:
                        continue
                    
                    # Número de fragmentos
                    N_frag = min(
                        config.coll_alpha * CKE ** config.coll_beta * g_T,
                        config.coll_N_frag_max
                    )
                    
                    # Taxa de colisão
                    R_sum = bin_grid.radii[i] + bin_grid.radii[j]
                    K_coll = np.pi * R_sum ** 2 * dV
                    
                    if i == j:
                        rate = 0.5 * K_coll * N1 * N2
                    else:
                        rate = K_coll * N1 * N2
                    
                    dN_new = rate * N_frag
                    dN_frag_total += dN_new
                    
                    # Adicionar fragmentos como cristais pequenos (~100µm)
                    j_frag = bin_grid.find_bin(100e-6)
                    if j_frag >= 0:
                        updated[CAT_ICE, j_frag] += dN_new
    
    return {'dN_frag': dN_frag_total, 'spectra': updated}


def heterogeneous_nucleation(T, spectra_col, bin_grid, N_INP_base=1e3):
    """Nucleação heterogênea — Fletcher (1962).
    
    N_INP(T) = N0_INP * exp(-β * (T - T0))
    
    Gera cristais de gelo primários a partir de INPs.
    
    Args:
        T: Temperatura (K).
        spectra_col: Array (n_categories, n_bins).
        bin_grid: BinGrid.
        N_INP_base: Concentração base de INP (m⁻³).
    
    Returns:
        Espectro atualizado.
    """
    if T > T_0 - 5:  # Nucleação inicia abaixo de -5°C
        return spectra_col.copy()
    
    T_c = T - T_0  # Celsius (negativo)
    beta = 0.6  # K⁻¹ (Fletcher)
    
    # Clampar para evitar overflow: -beta * T_c com T_c muito negativo
    # exp(0.6 * 90) = exp(54) → overflow. Limitar a -38°C (nucleação homogênea)
    T_c_clamped = max(T_c, -38.0)
    
    N_INP = N_INP_base * np.exp(-beta * T_c_clamped)
    
    # Limitar a um valor fisicamente razoável (max 1e8 m⁻³)
    N_INP = min(N_INP, 1e8)
    
    # Limitar ao disponível (não nucleados ainda)
    N_ice_existing = float(np.sum(spectra_col[CAT_ICE, :]))
    N_new = max(0.0, N_INP - N_ice_existing)
    
    updated = spectra_col.copy()
    if N_new > 0:
        # Cristais de ~30µm
        j_nuc = bin_grid.find_bin(30e-6)
        if j_nuc >= 0:
            updated[CAT_ICE, j_nuc] += N_new
    
    return updated


def step_microphysics(spectra_col, T, p, S, w, bin_grid, dt, config):
    """Passo completo de microfísica.
    
    Executa todos os processos na ordem:
    1. Condensação/evaporação
    2. Colisão-coalescência
    3. Nucleação heterogênea
    4. Riming
    5. Hallett-Mossop SIP
    6. Quebra colisional
    
    Args:
        spectra_col: Array (n_categories, n_bins).
        T: Temperatura (K).
        p: Pressão (Pa).
        S: Supersaturação.
        w: Velocidade vertical (m/s).
        bin_grid: BinGrid.
        dt: Passo temporal (s).
        config: MicrophysicsConfig.
    
    Returns:
        dict:
            'spectra': espectro atualizado
            'dq_condensed': taxa de condensação (kg/m³/s)
            'dq_frozen': taxa de congelamento (kg/m³/s)
            'dq_deposited': taxa de deposição (kg/m³/s)
            'sip_rate': taxa de SIP (m⁻³/s)
    """
    current = spectra_col.copy()
    dq_frozen = 0.0
    sip_rate = 0.0
    
    # 1. Condensação / evaporação
    cond_result = condensation_growth(T, p, S, current, bin_grid, dt)
    current = cond_result['spectra']
    dq_condensed = cond_result['dq_condensed']
    
    # 2. Colisão-coalescência
    current = collision_coalescence(current, bin_grid, dt)
    
    # 3. Nucleação heterogênea (abaixo de -5°C, acima de -38°C nucleação homogênea)
    if T_0 - 38 < T < T_0 - 5:
        current = heterogeneous_nucleation(T, current, bin_grid)
    
    # 4. Riming (abaixo de 0°C)
    rime_result = riming(current, bin_grid, T, dt, config)
    current = rime_result['spectra']
    dm_rime_dt = rime_result['dm_rime_dt']
    dq_frozen += dm_rime_dt
    
    # 5. Hallett-Mossop SIP (-3°C a -8°C)
    hm_result = hallett_mossop_sip(T, dm_rime_dt, current, bin_grid, config)
    current = hm_result['spectra']
    sip_rate += hm_result['dN_sip']
    
    # 6. Quebra colisional
    if T < T_0:
        cb_result = collisional_breakup(current, bin_grid, T, config)
        current = cb_result['spectra']
        sip_rate += cb_result['dN_frag']
    
    # Garantir não-negatividade e sem NaN
    current = np.maximum(current, 0.0)
    current = np.nan_to_num(current, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Sanitizar escalares
    dq_condensed = float(np.nan_to_num(dq_condensed, nan=0.0))
    dq_frozen = float(np.nan_to_num(dq_frozen, nan=0.0))
    sip_rate = float(np.nan_to_num(sip_rate, nan=0.0))
    
    return {
        'spectra': current,
        'dq_condensed': dq_condensed,
        'dq_frozen': dq_frozen,
        'dq_deposited': 0.0,
        'sip_rate': sip_rate,
    }
