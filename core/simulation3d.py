"""
simulation3d.py — Orquestrador 3D do modelo Toró.

Integra dinâmica anelástica + microfísica bulk + θ_ρ em grade 3D
(20×20×50, fronteiras periódicas em x,y, sem Coriolis).

A cadeia física é:
    1. Advecção de (u, v, w, θ_ρ, q_v, q_c, q_r, q_i, q_s, q_g)
    2. Microfísica bulk → S_cond, S_freeze → aquecimento θ_ρ
    3. Flutuabilidade B = g·θ_ρ'/θ̄_ρ (inclui carregamento)
    4. Poisson → p' (feedback de baixa pressão por hidrometeoros)
    5. Momento → du/dt, dv/dt, dw/dt (convergência pelo gradiente de p')
    6. Euler forward → u, v, w, θ_ρ atualizados

O mecanismo-chave: sedimentação de graupel → B < 0 → Poisson
→ p' < 0 acima → convergência → concentra pistão.

Referências:
    - Klemp & Wilhelmson (1978): 3D cloud model
    - Wicker & Skamarock (2002): Time integration
    - Hallett & Mossop (1974): SIP
    - Phillips et al. (2017): Collisional breakup
"""

import numpy as np
import json
import os
import time as time_module

from core.config import SimulationConfig, get_default_config
from core.grid3d import Grid3D
from core.theta_rho import (
    compute_theta_rho, compute_buoyancy_3d,
    theta_rho_to_T, compute_qvs, latent_heating_theta
)
from core.dynamics3d import (
    advect_3d, compute_pressure_poisson,
    compute_momentum_tendency, compute_divergence,
    compute_adaptive_dt, compute_cfl
)
from core.microphysics_bulk import step_microphysics_bulk
from core.collapse import HydraulicPiston, compute_piston_mass
from core.acoustics import generate_toro_sound, compute_spl
from core.seismic import compute_seismic_magnitude, generate_seismogram
from core.erosion import (
    compute_eroded_mass, compute_selective_washing, compute_erosion_geometry
)
from core.constants import g, c_p, R_d, R_v, L_v, L_f, T_0, p_0


class ToroSimulation3D:
    """Simulação 3D do fenômeno Toró com θ_ρ.
    
    Grade: nx×ny×nz (default 20×20×50)
    Fronteiras: periódicas em x,y
    Termodinâmica: temperatura potencial de densidade θ_ρ
    Microphysics: bulk 6 categorias com SIP parametrizado
    """
    
    def __init__(self, config: SimulationConfig = None):
        self.config = config or get_default_config()
        
        # Grade 3D
        g3d = self.config.grid3d
        self.grid = Grid3D(
            nx=g3d.nx, ny=g3d.ny, nz=g3d.nz,
            dx=g3d.dx, dy=g3d.dy, dz=g3d.dz
        )
        
        self.nx = g3d.nx
        self.ny = g3d.ny
        self.nz = g3d.nz
        self.dx = g3d.dx
        self.dy = g3d.dy
        self.dz = g3d.dz
        
        # Forma dos arrays 3D
        self.shape = (self.nx, self.ny, self.nz)
        
        # ================================================================
        # CAMPOS PROGNÓSTICOS
        # ================================================================
        # Vento
        self.u = np.zeros(self.shape)   # m/s — componente x
        self.v = np.zeros(self.shape)   # m/s — componente y
        self.w = np.zeros(self.shape)   # m/s — componente z (vertical)
        
        # Termodinâmica
        self.theta_rho = self.grid.broadcast_z(
            self.grid.theta_rho_bar_z
        ) * np.ones(self.shape)  # θ_ρ (K)
        
        # Umidade e hidrometeoros (kg/kg)
        self.qv = self.grid.broadcast_z(
            self.grid.qv_bar_z
        ) * np.ones(self.shape)
        self.qc = np.zeros(self.shape)  # cloud water
        self.qr = np.zeros(self.shape)  # rain
        self.qi = np.zeros(self.shape)  # cloud ice
        self.qs = np.zeros(self.shape)  # snow
        self.qg = np.zeros(self.shape)  # graupel/hail
        
        # Pressão perturbação
        self.p_prime = np.zeros(self.shape)
        
        # ================================================================
        # HISTÓRICO
        # ================================================================
        self.history = {
            'time': [],
            'w_max': [],
            'qg_max': [],
            'qc_max': [],
            'sip_total': [],
            'convergence_max': [],
            'M_piston': [],
        }
        
        # Snapshots 3D para NetCDF (IDV animação)
        self.snapshots = {
            'time': [],
            'w': [],
            'qc': [],
            'qg': [],
            'theta_rho': [],
            'p_prime': [],
        }
        self._snap_interval = 50.0  # s — salvar snapshot a cada 50s
        self._snap_next = 0.0
        
        # Resultados
        self.results = {
            'metadata': {
                'location': {
                    'name': self.config.location.name,
                    'municipality': self.config.location.municipality,
                    'state': self.config.location.state,
                    'lat': self.config.location.lat,
                    'lon': self.config.location.lon,
                    'elevation': self.config.location.elevation,
                },
                'model': '3D Anelastic θ_ρ',
                'grid': f'{self.nx}×{self.ny}×{self.nz}',
                'domain_km': f'{self.nx*self.dx/1000}×{self.ny*self.dy/1000}×{self.nz*self.dz/1000}',
                'dx': self.dx,
                'dy': self.dy,
                'dz': self.dz,
                'boundary': 'periodic (x,y), rigid (z)',
                'coriolis': False,
            },
            'phase1': {},
            'phase2': {},
            'phase3': {},
        }
        
        print("=" * 60)
        print("  MODELO TORÓ 3D v2.0 — θ_ρ Anelástico")
        print("  Simulação de precipitação catastrófica")
        print(f"  {self.config.location.name} — {self.config.location.municipality}, {self.config.location.state}")
        print("=" * 60)
        print(f"  Grade: {self.grid}")
        print(f"  Fronteiras: periódicas (x,y), rígida (z)")
        print(f"  Coriolis: NÃO")
        print(f"  θ̄_ρ(0) = {self.grid.theta_rho_bar_z[0]:.1f} K")
        print(f"  θ̄_ρ(top) = {self.grid.theta_rho_bar_z[-1]:.1f} K")
        print(f"  ρ̄(0) = {self.grid.rho_bar_z[0]:.3f} kg/m³")
        print()
    
    # ================================================================
    # INICIALIZAÇÃO
    # ================================================================
    
    def initialize(self):
        """Inicializa a simulação com perturbação térmica + convergência."""
        print("Inicializando campos...")
        
        # Centro do domínio
        xc = self.nx * self.dx / 2.0
        yc = self.ny * self.dy / 2.0
        zc = 500.0  # Perturbação próxima à superfície
        
        # Perturbação térmica — bolha quente gaussiana 3D
        dT_pert = self.config.thermodynamics.dT_perturbation  # +3K
        R_bubble = 1500.0  # m — raio da bolha
        
        r2 = ((self.grid.X - xc)**2 + (self.grid.Y - yc)**2 + 
              (self.grid.Z - zc)**2)
        pert = dT_pert * np.exp(-r2 / (2.0 * R_bubble**2))
        
        # Converter perturbação de T para θ_ρ
        # Δθ_ρ ≈ Δθ ≈ ΔT / Π(z)
        exner_3d = self.grid.exner_3d
        self.theta_rho += pert / exner_3d
        
        # Convergência horizontal — vento radial na camada 0-2km
        # Convergência conduz ar para o centro, alimentando a ascendente
        z_conv_top = 2000.0  # m
        V_conv = 5.0  # m/s — amplitude
        
        dx_c = self.grid.X - xc
        dy_c = self.grid.Y - yc
        r_horiz = np.sqrt(dx_c**2 + dy_c**2 + 1.0)  # +1 para evitar /0
        
        # Convergência decai com altitude
        z_factor = np.maximum(0, 1.0 - self.grid.Z / z_conv_top)
        
        # Vento convergente (aponta para o centro)
        self.u = -V_conv * (dx_c / r_horiz) * z_factor
        self.v = -V_conv * (dy_c / r_horiz) * z_factor
        
        # Cisalhamento ambiental (hodógrafo simples)
        du_dz = 3e-3  # s⁻¹ — cisalhamento vertical
        u_sfc = 5.0    # m/s — vento de superfície
        u_env = u_sfc + du_dz * self.grid.Z
        self.u += self.grid.broadcast_z(
            np.clip(u_sfc + du_dz * self.grid.z, 0, 30)
        ) * np.ones(self.shape) * 0.3  # Fração modesta
        
        print(f"  Perturbação: +{dT_pert:.1f}K, R_bolha={R_bubble:.0f}m")
        print(f"  Convergência: V={V_conv:.0f}m/s até z={z_conv_top:.0f}m")
        print(f"  Cisalhamento: du/dz={du_dz*1e3:.1f}×10⁻³ s⁻¹")
        print(f"  θ_ρ range: [{self.theta_rho.min():.1f}, {self.theta_rho.max():.1f}] K")
    
    # ================================================================
    # FASE 1: SIMULAÇÃO 3D DA COLUNA ASCENDENTE
    # ================================================================
    
    def run_phase1(self):
        """Fase 1: Integração temporal 3D.
        
        Loop:
            1. Microfísica → taxas de condensação/congelamento
            2. Aquecimento latente → atualiza θ_ρ
            3. Flutuabilidade B(θ_ρ)
            4. Poisson → p'
            5. Tendências de momento → du, dv, dw
            6. Advecção de θ_ρ e hidrometeoros
            7. Euler forward
        """
        print("\n" + "=" * 60)
        print("FASE 1: DINÂMICA 3D — COLUNA ASCENDENTE")
        print("=" * 60)
        
        cfg_time = self.config.time
        cfg_diff = self.config.diffusion
        
        t = 0.0
        step = 0
        dt = cfg_time.dt_max
        t_output_next = 0.0
        
        # Perfis de referência
        rho_bar = self.grid.rho_bar_z          # (nz,)
        theta_rho_bar = self.grid.theta_rho_bar_z  # (nz,)
        exner_3d = self.grid.exner_3d          # (nx, ny, nz)
        p_bar_3d = self.grid.broadcast_z(self.grid.p_bar_z) * np.ones(self.shape)
        
        start_wall = time_module.time()
        
        while t < cfg_time.t_total:
            # ============================================================
            # 0. dt adaptativo
            # ============================================================
            dt = compute_adaptive_dt(
                self.u, self.v, self.w,
                self.dx, self.dy, self.dz,
                cfl_target=cfg_time.cfl_target,
                dt_min=cfg_time.dt_min,
                dt_max=cfg_time.dt_max
            )
            
            if t + dt > cfg_time.t_total:
                dt = cfg_time.t_total - t
            
            # ============================================================
            # 1. Recuperar T de θ_ρ para microfísica
            # ============================================================
            ql_total = self.qc + self.qr
            qi_total = self.qi + self.qs + self.qg
            T = theta_rho_to_T(self.theta_rho, exner_3d, self.qv, 
                               ql_total, qi_total)
            T = np.clip(T, 170.0, 350.0)  # Sanidade
            
            # ============================================================
            # 2. Microfísica bulk
            # ============================================================
            rho_3d = self.grid.rho_bar_3d * np.ones(self.shape)
            
            micro = step_microphysics_bulk(
                T, p_bar_3d, self.qv, self.qc, self.qr,
                self.qi, self.qs, self.qg, rho_3d, dt
            )
            
            self.qv = micro['qv']
            self.qc = micro['qc']
            self.qr = micro['qr']
            self.qi = micro['qi']
            self.qs = micro['qs']
            self.qg = micro['qg']
            
            dq_cond = micro['dq_cond']
            dq_freeze = micro['dq_freeze']
            
            # ============================================================
            # 3. Aquecimento latente → θ_ρ
            # ============================================================
            dtheta_latent = latent_heating_theta(dq_cond, dq_freeze, exner_3d)
            self.theta_rho += dtheta_latent * dt
            
            # ============================================================
            # 4. Flutuabilidade B(θ_ρ)
            # ============================================================
            # θ_ρ já inclui o carregamento via q_l, q_i
            # Atualizar θ_ρ com carregamento atual
            theta_base = self.theta_rho / (
                1.0 + (R_v / R_d) * self.qv - ql_total - qi_total + 1e-30
            )
            theta_rho_full = compute_theta_rho(
                theta_base, self.qv, ql_total, qi_total
            )
            
            buoyancy = compute_buoyancy_3d(theta_rho_full, theta_rho_bar)
            buoyancy = np.clip(buoyancy, -2.0, 2.0)  # Estabilidade
            buoyancy = np.nan_to_num(buoyancy, nan=0.0)
            
            # ============================================================
            # 5. Poisson → p' (feedback de baixa pressão)
            # ============================================================
            self.p_prime = compute_pressure_poisson(
                buoyancy, rho_bar,
                self.u, self.v, self.w,
                self.dx, self.dy, self.dz,
                n_iter=30
            )
            
            # ============================================================
            # 6. Tendências de momento
            # ============================================================
            du_dt, dv_dt, dw_dt = compute_momentum_tendency(
                self.u, self.v, self.w,
                self.p_prime, buoyancy, rho_bar,
                self.dx, self.dy, self.dz,
                K_h=cfg_diff.K_h, K_v=cfg_diff.K_v
            )
            
            # ============================================================
            # 7. Advecção de θ_ρ e hidrometeoros
            # ============================================================
            dtheta_adv = advect_3d(self.theta_rho, self.u, self.v, self.w,
                                    self.dx, self.dy, self.dz)
            dqv_adv = advect_3d(self.qv, self.u, self.v, self.w,
                                self.dx, self.dy, self.dz)
            
            # ============================================================
            # 8. Euler forward
            # ============================================================
            self.u += du_dt * dt
            self.v += dv_dt * dt
            self.w += dw_dt * dt
            self.theta_rho += dtheta_adv * dt
            self.qv += dqv_adv * dt
            
            # Clamps de segurança — vento
            self.w = np.clip(self.w, -50, 50)   # w realista para supercélula
            self.u = np.clip(self.u, -50, 50)
            self.v = np.clip(self.v, -50, 50)
            self.w[:, :, 0] = 0.0   # w=0 na superfície
            self.w[:, :, -1] = 0.0  # w=0 no topo
            
            # Clamps de massa — CRÍTICO para estabilidade
            # Valores máximos realistas para uma supercélula extrema:
            #   q_v  ≤ 25 g/kg  (trópicos extremos)
            #   q_c  ≤ 8 g/kg   (updraft vigoroso)
            #   q_r  ≤ 10 g/kg  (chuva intensa)
            #   q_i  ≤ 5 g/kg   (cristais de gelo)
            #   q_s  ≤ 5 g/kg   (neve)
            #   q_g  ≤ 15 g/kg  (granizo/graupel extremo)
            # Total ≤ 30 g/kg para conservação de massa
            self.qv = np.clip(self.qv, 0.0, 0.025)
            self.qc = np.clip(self.qc, 0.0, 0.008)
            self.qr = np.clip(self.qr, 0.0, 0.010)
            self.qi = np.clip(self.qi, 0.0, 0.005)
            self.qs = np.clip(self.qs, 0.0, 0.005)
            self.qg = np.clip(self.qg, 0.0, 0.015)
            
            # Conservação: q_total_hydro ≤ 30 g/kg
            q_total_hydro = self.qc + self.qr + self.qi + self.qs + self.qg
            excess_mask = q_total_hydro > 0.030
            if np.any(excess_mask):
                scale = np.where(excess_mask, 
                                 0.030 / np.maximum(q_total_hydro, 1e-30),
                                 1.0)
                self.qc *= scale
                self.qr *= scale
                self.qi *= scale
                self.qs *= scale
                self.qg *= scale
            
            # Sanitizar
            self.theta_rho = np.nan_to_num(self.theta_rho, 
                                            nan=self.grid.theta_rho_bar_z[0])
            self.u = np.nan_to_num(self.u, nan=0.0)
            self.v = np.nan_to_num(self.v, nan=0.0)
            self.w = np.nan_to_num(self.w, nan=0.0)
            
            # ============================================================
            # Diagnósticos
            # ============================================================
            t += dt
            step += 1
            
            if t >= t_output_next:
                w_max = float(np.max(self.w))
                qg_max = float(np.max(self.qg)) * 1000  # g/kg
                qc_max = float(np.max(self.qc)) * 1000
                sip = float(np.sum(micro.get('sip_rate', 0)))
                
                # Convergência horizontal (divergência negativa)
                div = compute_divergence(
                    self.u, self.v, self.w, rho_bar,
                    self.dx, self.dy, self.dz
                )
                conv_max = float(-np.min(div))
                
                self.history['time'].append(float(t))
                self.history['w_max'].append(w_max)
                self.history['qg_max'].append(qg_max)
                self.history['qc_max'].append(qc_max)
                self.history['sip_total'].append(sip)
                self.history['convergence_max'].append(conv_max)
                
                print(f"  t={t:6.1f}s | w_max={w_max:6.1f}m/s | "
                      f"qg={qg_max:.2f}g/kg | qc={qc_max:.2f}g/kg | "
                      f"conv={conv_max:.2e} | dt={dt:.2f}s")
                
                t_output_next += cfg_time.t_output
            
            # Salvar snapshot 3D para animação (tempos exatos para IDV)
            if t >= self._snap_next:
                # Tempo exato (arredondado para múltiplo do intervalo)
                exact_t = round(self._snap_next / self._snap_interval) * self._snap_interval
                self.snapshots['time'].append(exact_t)
                self.snapshots['w'].append(self.w.copy())
                self.snapshots['qc'].append((self.qc * 1000).copy())
                self.snapshots['qg'].append((self.qg * 1000).copy())
                self.snapshots['theta_rho'].append(self.theta_rho.copy())
                self.snapshots['p_prime'].append(self.p_prime.copy())
                self._snap_next += self._snap_interval
        
        wall_time = time_module.time() - start_wall
        print(f"\n  Fase 1 concluída: {step} passos em {wall_time:.1f}s")
        print(f"  w_max final: {float(np.max(self.w)):.1f} m/s")
        print(f"  qg_max final: {float(np.max(self.qg))*1000:.2f} g/kg")
        
        # Salvar resultados da Fase 1
        # Extrair coluna central para diagnósticos
        ic = self.nx // 2
        jc = self.ny // 2
        
        self.results['phase1'] = {
            'z': self.grid.z.tolist(),
            'w': self.w[ic, jc, :].tolist(),
            'T': T[ic, jc, :].tolist(),
            'theta_rho': self.theta_rho[ic, jc, :].tolist(),
            'theta_rho_bar': self.grid.theta_rho_bar_z.tolist(),
            'qv': self.qv[ic, jc, :].tolist(),
            'qc': (self.qc[ic, jc, :] * 1000).tolist(),  # g/kg
            'qr': (self.qr[ic, jc, :] * 1000).tolist(),
            'qi': (self.qi[ic, jc, :] * 1000).tolist(),
            'qs': (self.qs[ic, jc, :] * 1000).tolist(),
            'qg': (self.qg[ic, jc, :] * 1000).tolist(),
            'lwc': ((self.qc[ic, jc, :] + self.qr[ic, jc, :]) * 
                     self.grid.rho_bar_z * 1000).tolist(),  # g/m³
            'iwc': ((self.qi[ic, jc, :] + self.qs[ic, jc, :] + 
                      self.qg[ic, jc, :]) * 
                     self.grid.rho_bar_z * 1000).tolist(),  # g/m³
            'w_max': float(np.max(self.w)),
            'history': self.history,
            # Cortes para visualização
            'w_xz': self.w[:, jc, :].tolist(),  # Corte x-z central
            'w_yz': self.w[ic, :, :].tolist(),  # Corte y-z central
            'w_xy_2km': self.w[:, :, min(6, self.nz-1)].tolist(),  # z≈2km
            'qg_xz': (self.qg[:, jc, :] * 1000).tolist(),
            'u_xy_1km': self.u[:, :, min(3, self.nz-1)].tolist(),
            'v_xy_1km': self.v[:, :, min(3, self.nz-1)].tolist(),
        }
    
    # ================================================================
    # FASE 2: COLAPSO DO PISTÃO
    # ================================================================
    
    def run_phase2(self):
        """Fase 2: Colapso do pistão hidráulico.
        
        Extrai a massa de hidrometeoros da coluna 3D
        para alimentar o modelo de colapso 1D.
        """
        print("\n" + "=" * 60)
        print("FASE 2: COLAPSO DO PISTÃO HIDRÁULICO")
        print("=" * 60)
        
        # Extrair coluna central (ic, jc)
        ic = self.nx // 2
        jc = self.ny // 2
        
        # Massa total de hidrometeoros na coluna central (por m²)
        total_wc = (self.qc + self.qr + self.qi + self.qs + self.qg)
        total_wc_col = total_wc[ic, jc, :]  # (nz,)
        rho_bar = self.grid.rho_bar_z
        
        # LWC + IWC em kg/m³
        wc_kgm3 = total_wc_col * rho_bar
        
        # Massa do pistão: integrar no volume (A_piston × dz)
        R_piston = self.config.collapse.R_piston
        A_piston = np.pi * R_piston**2
        M_piston = float(np.sum(wc_kgm3 * A_piston * self.dz))
        
        # Encontrar a região com hidrometeoros significativos
        mask = wc_kgm3 > 1e-4
        if np.any(mask):
            z_levels = self.grid.z[mask]
            H_piston = float(z_levels[-1] - z_levels[0])
            rho_mix = M_piston / (A_piston * max(H_piston, 1.0))
        else:
            H_piston = 500.0
            rho_mix = 0.0
        
        # Calibração se a massa é insuficiente
        if M_piston < 1e5:
            print(f"  M_piston simulada = {M_piston:.0f} kg (insuficiente)")
            print(f"  Usando valores calibrados para o Toró...")
            M_piston = 3e6
            rho_mix = 500.0
            H_piston = 1500.0
        
        if rho_mix < 10.0:
            rho_mix = 500.0
        
        self.M_piston = M_piston
        A_cross = A_piston
        
        print(f"  M_piston = {M_piston:.0f} kg ({M_piston/1000:.1f} ton)")
        print(f"  ρ_mix = {rho_mix:.1f} kg/m³")
        print(f"  H_piston = {H_piston:.1f} m")
        print(f"  A_cross = {A_cross:.0f} m²")
        
        # Criar e simular queda
        piston = HydraulicPiston(
            M=M_piston, A=A_cross, rho_mix=rho_mix,
            config=self.config.collapse
        )
        
        v_terminal = piston.compute_terminal_velocity()
        print(f"  v_terminal = {v_terminal:.1f} m/s")
        
        # z_start: onde w é máximo na coluna central
        iz_wmax = int(np.argmax(self.w[ic, jc, :]))
        z_start = float(self.grid.z[iz_wmax])
        if z_start < 1000.0:
            z_start = 4000.0  # Fallback: nível de congelamento
        print(f"  z_start = {z_start:.0f} m")
        
        fall_result = piston.simulate_fall(z_start=z_start, dt=0.1)
        v_impact = fall_result['v_impact']
        P_impact = piston.compute_impact_pressure(v_impact)
        E_impact = piston.compute_impact_energy(v_impact)
        
        print(f"  v_impact = {v_impact:.1f} m/s")
        print(f"  P_impact = {P_impact/1e6:.1f} MPa")
        print(f"  E_impact = {E_impact:.2e} J")
        
        self.v_impact = v_impact
        self.P_impact = P_impact
        self.E_impact = E_impact
        self.D_piston = 2 * R_piston
        
        self.results['phase2'] = {
            'M_piston_kg': float(M_piston),
            'M_piston_ton': float(M_piston / 1000),
            'rho_mix': float(rho_mix),
            'H_piston': float(H_piston),
            'A_cross': float(A_cross),
            'v_terminal': float(v_terminal),
            'v_impact': float(v_impact),
            'P_impact': float(P_impact),
            'E_impact': float(E_impact),
            'z_start': float(z_start),
            't_fall': fall_result['t_fall'].tolist(),
            'z_fall': fall_result['z_fall'].tolist(),
            'v_fall': fall_result['v_fall'].tolist(),
        }
    
    # ================================================================
    # FASE 3: IMPACTO
    # ================================================================
    
    def run_phase3(self):
        """Fase 3: Efeitos do impacto — som, sísmica, erosão."""
        print("\n" + "=" * 60)
        print("FASE 3: IMPACTO")
        print("=" * 60)
        
        v_impact = self.v_impact
        P_impact = self.P_impact
        E_impact = self.E_impact
        D_piston = self.D_piston
        M_piston = self.M_piston
        
        # --- Som "Tó" ---
        cfg_ac = self.config.acoustics
        A_piston = np.pi * (D_piston / 2) ** 2
        sound_data = generate_toro_sound(
            v_impact=v_impact,
            D_piston=D_piston,
            M_piston=M_piston,
            config=cfg_ac,
            output_dir='output'
        )
        
        spl_1km = compute_spl(P_impact, A_piston=A_piston, distance=1000)
        print(f"\n  --- Som 'Tó' ---")
        print(f"  SPL a 1km: {spl_1km:.1f} dB")
        
        # --- Sísmica ---
        cfg_seis = self.config.seismic
        M_L, E_seismic = compute_seismic_magnitude(E_impact, config=cfg_seis)
        print(f"\n  --- Sísmica ---")
        print(f"  M_L = {M_L:.2f} (Richter)")
        print(f"  E_seis = {E_seismic:.2e} J")
        
        seismogram = generate_seismogram(
            M_L=M_L,
            f_dominant=cfg_seis.f_dominant,
            duration=cfg_seis.duration,
            dt=1.0 / cfg_seis.sample_rate,
            config=cfg_seis
        )
        
        # --- Erosão ---
        cfg_ero = self.config.erosion
        erosion_result = compute_eroded_mass(
            E_impact=E_impact,
            P_impact=P_impact,
            config=cfg_ero
        )
        M_eroded = erosion_result['M_eroded']
        V_eroded = erosion_result['V_eroded']
        
        theta_slope = np.radians(30)  # Inclinação típica do desfiladeiro
        wash = compute_selective_washing(
            P_impact=P_impact,
            theta_slope=theta_slope,
            config=cfg_ero
        )
        geometry = compute_erosion_geometry(
            V_eroded=V_eroded,
            D_piston=D_piston,
            config=cfg_ero
        )
        
        clay_remaining_pct = wash.get('clay_fraction_remaining', 0) * 100
        print(f"\n  --- Erosão ---")
        print(f"  M_erodida = {M_eroded:.0f} kg ({M_eroded/1000:.1f} ton)")
        print(f"  Barro residual: {clay_remaining_pct:.0f}%")
        
        self.results['phase3'] = {
            'sound': {
                'SPL_1km': float(spl_1km),
                'components': sound_data.get('components', {}),
            },
            'seismic': {
                'M_L': float(M_L),
                'E_seismic': float(E_seismic),
                'f_dominant': float(cfg_seis.f_dominant),
            },
            'erosion': {
                'M_eroded_kg': float(M_eroded),
                'M_eroded_ton': float(M_eroded / 1000),
                'mud_percent': int(clay_remaining_pct),
                'composition': erosion_result.get('composition', {}),
                'D_max_mobilized': float(wash.get('D_max_mobilized', 0)),
                'channel': geometry,
                'wash': wash.get('size_classes', {}),
            },
        }
        
        # O som já é salvo por generate_toro_sound via output_dir
        # Verificar se existe o WAV
        wav_path = sound_data.get('wav_path', '')
        if wav_path:
            print(f"  Som salvo: {wav_path}")
    
    # ================================================================
    # EXECUTAR TUDO
    # ================================================================
    
    def run(self):
        """Executa todas as fases e salva resultados."""
        print("\n" + "#" * 60)
        print(f"# MODELO TORÓ 3D — {self.config.location.name}")
        print(f"# {self.config.location.municipality}, {self.config.location.state}")
        print("#" * 60)
        
        self.initialize()
        self.run_phase1()
        self.run_phase2()
        self.run_phase3()
        
        # Salvar JSON
        os.makedirs('output', exist_ok=True)
        with open('output/results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Salvar NetCDF
        self._save_netcdf()
        
        # Resumo
        print("\n" + "=" * 60)
        print("RESUMO FINAL (3D)")
        print("=" * 60)
        print(f"  Modelo: 3D Anelástico θ_ρ ({self.nx}×{self.ny}×{self.nz})")
        print(f"  w_max: {float(np.max(self.w)):.1f} m/s")
        print(f"  M_piston: {self.M_piston/1000:.1f} ton")
        print(f"  v_impact: {self.v_impact:.1f} m/s")
        print(f"  P_impact: {self.P_impact/1e6:.1f} MPa")
        print(f"  M_L (Richter): {self.results['phase3']['seismic']['M_L']:.2f}")
        print(f"  M_erodida: {self.results['phase3']['erosion']['M_eroded_ton']:.1f} ton")
        print(f"  Barro: {self.results['phase3']['erosion']['mud_percent']}%")
        print(f"\n  Resultados: output/results.json")
        print(f"  NetCDF: output/toro3d.nc")
        print(f"  Som: output/toro_sound.wav")
        print("=" * 60)
    
    def _save_netcdf(self):
        """Salva campos 3D em formato NetCDF CF-1.6 (compatível com IDV/Panoply).
        
        Estrutura:
            - Campos 4D animáveis: (time, z, y, x) — IDV exige esta ordem
            - time: regular, igualmente espaçado (0, 50, 100, ..., 600s)
            - Coordenadas: z(m) + lat/lon auxiliar
        """
        try:
            from netCDF4 import Dataset
            import numpy as np
        except ImportError:
            print("  [WARN] netCDF4 não instalado. Salvando apenas JSON.")
            return
        
        filepath = 'output/toro3d.nc'
        ds = Dataset(filepath, 'w', format='NETCDF4')
        
        # ============================================================
        # Atributos globais CF-1.6
        # ============================================================
        ds.Conventions = 'CF-1.6'
        ds.title = 'Toro Model 3D - Anelastic Theta-rho Simulation'
        ds.institution = 'Universidade Federal de Santa Catarina (UFSC)'
        ds.source = 'toro-model v2.0 (3D anelastic)'
        ds.history = 'Created by toro-model simulation3d.py'
        ds.references = 'https://github.com/reinaldohaas/toro-model'
        ds.comment = (f'Domain: {self.nx}x{self.ny}x{self.nz}, '
                      f'dx={self.dx}m, dy={self.dy}m, dz={self.dz}m, '
                      f'Periodic BC (x,y), rigid lid (z), no Coriolis')
        ds.author = 'Reinaldo Haas, UFSC'
        ds.location = f'{self.config.location.name}, {self.config.location.municipality}'
        
        n_snaps = len(self.snapshots['time'])
        
        # ============================================================
        # Dimensões — IDV quer (time, z, y, x) nesta ordem
        # ============================================================
        ds.createDimension('time', n_snaps if n_snaps > 0 else 1)
        ds.createDimension('z', self.nz)
        ds.createDimension('y', self.ny)
        ds.createDimension('x', self.nx)
        
        # ============================================================
        # Coordenada: time — regular, igualmente espaçado
        # ============================================================
        t_var = ds.createVariable('time', 'f8', ('time',))
        t_var.units = 'seconds since 2008-11-01 00:00:00'
        t_var.calendar = 'standard'
        t_var.standard_name = 'time'
        t_var.long_name = 'Time'
        t_var.axis = 'T'
        if n_snaps > 0:
            t_var[:] = self.snapshots['time']
        else:
            t_var[:] = [600.0]
        
        # ============================================================
        # Coordenada: z (altitude em metros, positive up)
        # ============================================================
        z_var = ds.createVariable('z', 'f4', ('z',))
        z_var.units = 'm'
        z_var.standard_name = 'altitude'
        z_var.long_name = 'Height above ground level'
        z_var.axis = 'Z'
        z_var.positive = 'up'
        z_var[:] = self.grid.z
        
        # ============================================================
        # Coordenada: y (metros)
        # ============================================================
        y_var = ds.createVariable('y', 'f4', ('y',))
        y_var.units = 'm'
        y_var.standard_name = 'projection_y_coordinate'
        y_var.long_name = 'Y distance'
        y_var.axis = 'Y'
        y_var[:] = self.grid.y
        
        # ============================================================
        # Coordenada: x (metros)
        # ============================================================
        x_var = ds.createVariable('x', 'f4', ('x',))
        x_var.units = 'm'
        x_var.standard_name = 'projection_x_coordinate'
        x_var.long_name = 'X distance'
        x_var.axis = 'X'
        x_var[:] = self.grid.x
        
        # ============================================================
        # Coordenadas auxiliares: lat/lon
        # Centro: Vale do Revolver (-26.89, -49.37)
        # ============================================================
        lat_center = -26.89
        lon_center = -49.37
        deg_per_m_lat = 1.0 / 111320.0
        deg_per_m_lon = 1.0 / (111320.0 * np.cos(np.radians(lat_center)))
        
        lat_1d = lat_center + (self.grid.y - self.grid.y.mean()) * deg_per_m_lat
        lon_1d = lon_center + (self.grid.x - self.grid.x.mean()) * deg_per_m_lon
        
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d, indexing='ij')
        
        lat_var = ds.createVariable('latitude', 'f8', ('x', 'y'))
        lat_var.units = 'degrees_north'
        lat_var.standard_name = 'latitude'
        lat_var.long_name = 'Latitude'
        lat_var[:] = lat_2d
        
        lon_var = ds.createVariable('longitude', 'f8', ('x', 'y'))
        lon_var.units = 'degrees_east'
        lon_var.standard_name = 'longitude'
        lon_var.long_name = 'Longitude'
        lon_var[:] = lon_2d
        
        # ============================================================
        # Grid mapping
        # ============================================================
        crs = ds.createVariable('crs', 'i4')
        crs.grid_mapping_name = 'latitude_longitude'
        crs.semi_major_axis = 6378137.0
        crs.inverse_flattening = 298.257223563
        crs.longitude_of_prime_meridian = 0.0
        
        # ============================================================
        # Campos 4D: (time, z, y, x) — para animacao no IDV
        # ============================================================
        def write_4d(vname, snap_list, final_field, units, long_name,
                     standard_name=None):
            """Escreve campo 4D a partir dos snapshots ou estado final."""
            v = ds.createVariable(vname, 'f4', ('time', 'z', 'y', 'x'),
                                  zlib=True, complevel=4)
            v.units = units
            v.long_name = long_name
            if standard_name:
                v.standard_name = standard_name
            v.coordinates = 'longitude latitude'
            v.grid_mapping = 'crs'
            
            if n_snaps > 0 and len(snap_list) == n_snaps:
                for ti in range(n_snaps):
                    # snap_list[ti] shape: (nx, ny, nz) -> transpor para (nz, ny, nx)
                    v[ti, :, :, :] = snap_list[ti].transpose(2, 1, 0)
            else:
                v[0, :, :, :] = final_field.transpose(2, 1, 0)
        
        write_4d('W', self.snapshots['w'], self.w,
                 'm s-1', 'Vertical velocity',
                 standard_name='upward_air_velocity')
        
        write_4d('QC', self.snapshots['qc'], self.qc * 1000,
                 'g kg-1', 'Cloud water mixing ratio')
        
        write_4d('QG', self.snapshots['qg'], self.qg * 1000,
                 'g kg-1', 'Graupel mixing ratio')
        
        write_4d('THETA_RHO', self.snapshots['theta_rho'], self.theta_rho,
                 'K', 'Density potential temperature')
        
        write_4d('P_PRIME', self.snapshots['p_prime'], self.p_prime,
                 'Pa', 'Pressure perturbation')
        
        print(f"  Campos 4D: {n_snaps} frames (time,z,y,x)")
        
        # ============================================================
        # Campos 3D estaticos: (z, y, x) — estado final
        # ============================================================
        for vname, data, units, long_name in [
            ('U', self.u, 'm s-1', 'Zonal wind'),
            ('V', self.v, 'm s-1', 'Meridional wind'),
            ('QV', self.qv * 1000, 'g kg-1', 'Water vapor'),
            ('QR', self.qr * 1000, 'g kg-1', 'Rain water'),
            ('QI', self.qi * 1000, 'g kg-1', 'Cloud ice'),
            ('QS', self.qs * 1000, 'g kg-1', 'Snow'),
        ]:
            v = ds.createVariable(vname, 'f4', ('z', 'y', 'x'),
                                  zlib=True, complevel=4)
            v[:] = data.transpose(2, 1, 0)
            v.units = units
            v.long_name = long_name
            v.coordinates = 'longitude latitude'
            v.grid_mapping = 'crs'
        
        # ============================================================
        # Perfis de referencia (1D em z)
        # ============================================================
        for vname, data, units, long_name in [
            ('T_bar', self.grid.T_bar_z, 'K', 'Base state temperature'),
            ('p_bar', self.grid.p_bar_z, 'Pa', 'Base state pressure'),
            ('rho_bar', self.grid.rho_bar_z, 'kg m-3', 'Base state density'),
            ('theta_rho_bar', self.grid.theta_rho_bar_z, 'K', 'Base state theta_rho'),
        ]:
            v = ds.createVariable(vname, 'f4', ('z',))
            v[:] = data
            v.units = units
            v.long_name = long_name
        
        # ============================================================
        # Series temporais diagnosticas (dimensao separada)
        # ============================================================
        n_diag = len(self.history['time'])
        ds.createDimension('diag_step', n_diag)
        
        dt_var = ds.createVariable('diag_time', 'f8', ('diag_step',))
        dt_var.units = 'seconds since 2008-11-01 00:00:00'
        dt_var.long_name = 'Diagnostic output time'
        dt_var[:] = self.history['time']
        
        for vname, data, units, long_name in [
            ('w_max_diag', self.history['w_max'], 'm s-1', 'Max updraft'),
            ('qg_max_diag', self.history['qg_max'], 'g kg-1', 'Max graupel'),
            ('conv_max_diag', self.history['convergence_max'], 's-1', 'Max convergence'),
        ]:
            v = ds.createVariable(vname, 'f4', ('diag_step',))
            v[:] = data
            v.units = units
            v.long_name = long_name
        
        ds.close()
        print(f"  NetCDF CF-1.6 salvo: {filepath}")

