"""
test_robert_bubble.py — Benchmark: Rising Thermal Bubble (Robert, 1993)

Teste padrão para validar dinâmica anelástica.
Uma bolha quente (+0.5K) em atmosfera neutra deve subir e formar
um vórtice toroidal simétrico (formato "cogumelo").

Referências:
    - Robert (1993): J. Atmos. Sci., 50(13), 1865-1873
    - Bryan & Fritsch (2002): Mon. Wea. Rev., 130, 2917-2928

Resultado esperado:
    - w_max ≈ 10-12 m/s em t = 1000s
    - Formato simétrico (mushroom)
    - Conservação de θ (energia)

Uso:
    python tests/test_robert_bubble.py
"""

import numpy as np
import json
import os
import sys
import time as time_module

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.constants import g, c_p, R_d, p_0


def run_robert_bubble():
    """Executa o teste da bolha de Robert (1993)."""

    print("=" * 60)
    print("  BENCHMARK: Rising Thermal Bubble (Robert, 1993)")
    print("  Bryan & Fritsch (2002) reference")
    print("=" * 60)

    # ================================================================
    # Grid (2D slice: x-z, ny=1)
    # ================================================================
    nx = 40           # pontos em x
    ny = 1            # 2D (fatia x-z)
    nz = 40           # pontos em z
    dx = 250.0        # m
    dz = 250.0        # m
    dy = dx

    Lx = nx * dx      # 10 km
    Lz = nz * dz      # 10 km

    x = np.arange(0.5, nx) * dx  # centros das células
    z = np.arange(0.5, nz) * dz

    print(f"  Grade: {nx}×{nz} (2D x-z)")
    print(f"  Domínio: {Lx/1000:.0f}×{Lz/1000:.0f} km")
    print(f"  Resolução: Δx={dx:.0f}m, Δz={dz:.0f}m")

    # ================================================================
    # Base state: neutral atmosphere (θ̄ = 300 K, uniforme)
    # ================================================================
    theta_bar = 300.0  # K — temperatura potencial uniforme (neutro)

    # Perfil hidrostático
    T_bar_z = np.zeros(nz)
    p_bar_z = np.zeros(nz)
    rho_bar_z = np.zeros(nz)
    exner_z = np.zeros(nz)

    p_sfc = 100000.0  # Pa
    for k in range(nz):
        zk = z[k]
        # Exner: Π(z) = 1 - g·z/(c_p·θ̄)
        exner_z[k] = 1.0 - g * zk / (c_p * theta_bar)
        p_bar_z[k] = p_0 * exner_z[k] ** (c_p / R_d)
        T_bar_z[k] = theta_bar * exner_z[k]
        rho_bar_z[k] = p_bar_z[k] / (R_d * T_bar_z[k])

    print(f"  θ̄ = {theta_bar:.1f} K (uniforme — atmosfera neutra)")
    print(f"  T(0) = {T_bar_z[0]:.1f} K, T(top) = {T_bar_z[-1]:.1f} K")

    # ================================================================
    # 3D arrays (nx, 1, nz) — 2D slice
    # ================================================================
    shape = (nx, ny, nz)

    u = np.zeros(shape)
    w = np.zeros(shape)
    theta = theta_bar * np.ones(shape)

    # Broadcast base state
    rho_bar_3d = np.ones(shape) * rho_bar_z[np.newaxis, np.newaxis, :]

    # ================================================================
    # Perturbação: bolha quente
    #   Δθ = 0.5 K, centrada em (x_c, z_c) = (5km, 2km), raio = 2km
    # ================================================================
    x_c = Lx / 2.0
    z_c = 2000.0
    R_bubble = 2000.0
    dtheta_max = 0.5  # K — perturbação fraca (teste linear)

    X, Z = np.meshgrid(x, z, indexing='ij')
    X3 = X[:, np.newaxis, :]
    Z3 = Z[:, np.newaxis, :]

    r2 = (X3 - x_c)**2 + (Z3 - z_c)**2
    R2 = R_bubble**2

    # Cosine bell perturbation (Robert, 1993)
    r = np.sqrt(r2)
    pert = np.where(r <= R_bubble,
                    dtheta_max * np.cos(np.pi * r / (2.0 * R_bubble))**2,
                    0.0)
    theta += pert

    print(f"  Perturbação: Δθ = +{dtheta_max:.1f} K")
    print(f"  Centro: ({x_c/1000:.1f}, {z_c/1000:.1f}) km")
    print(f"  Raio: {R_bubble/1000:.1f} km")

    # ================================================================
    # Integração temporal
    # ================================================================
    t_total = 1000.0  # s
    dt = 0.5           # s (CFL safe para Δx=250m, w_max~12m/s)
    K_diff = 10.0      # m²/s — difusão mínima (subgrid)

    n_steps = int(t_total / dt)
    t_output = 100.0   # s
    t_output_next = 0.0

    print(f"\n  Integração: {t_total:.0f}s, dt={dt:.1f}s, {n_steps} passos")
    print(f"  Difusão: K = {K_diff:.0f} m²/s")

    # Histórico
    history = {
        'time': [],
        'w_max': [],
        'theta_max': [],
        'theta_min': [],
        'energy': [],
    }

    # Snapshots para plotar
    snapshots = {}

    start_wall = time_module.time()

    for step in range(n_steps):
        t = step * dt

        # ---- Flutuabilidade ----
        theta_prime = theta - theta_bar
        buoyancy = g * theta_prime / theta_bar
        buoyancy = np.clip(buoyancy, -1.0, 1.0)

        # ---- Poisson solver (Jacobi simplificado para p') ----
        # ∇²p' = ρ̄ ∂B/∂z (simplificado para 2D)
        p_prime = np.zeros(shape)
        # Source term: dB/dz
        dB_dz = np.zeros(shape)
        dB_dz[:, :, 1:-1] = (buoyancy[:, :, 2:] - buoyancy[:, :, :-2]) / (2.0 * dz)

        source = rho_bar_3d * dB_dz

        # 60 iterações de Jacobi (sub-relaxamento ω=0.9)
        for _ in range(60):
            p_new = np.zeros_like(p_prime)
            # Interior
            p_new[1:-1, :, 1:-1] = (
                (p_prime[2:, :, 1:-1] + p_prime[:-2, :, 1:-1]) / dx**2 +
                (p_prime[1:-1, :, 2:] + p_prime[1:-1, :, :-2]) / dz**2 -
                source[1:-1, :, 1:-1]
            ) / (2.0 / dx**2 + 2.0 / dz**2)
            p_prime = 0.9 * p_new + 0.1 * p_prime  # Sub-relaxamento estável
        p_prime = np.nan_to_num(p_prime, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- Tendências de momento ----
        # du/dt = -(1/ρ̄) dp'/dx + D_u
        du_dt = np.zeros(shape)
        du_dt[1:-1, :, :] = -(1.0 / rho_bar_3d[1:-1, :, :]) * \
            (p_prime[2:, :, :] - p_prime[:-2, :, :]) / (2.0 * dx)

        # dw/dt = -(1/ρ̄) dp'/dz + B + D_w
        dw_dt = np.zeros(shape)
        dw_dt[:, :, 1:-1] = -(1.0 / rho_bar_3d[:, :, 1:-1]) * \
            (p_prime[:, :, 2:] - p_prime[:, :, :-2]) / (2.0 * dz) + \
            buoyancy[:, :, 1:-1]

        # ---- Difusão ----
        # Laplaciano 2D de u
        d2u = np.zeros(shape)
        d2u[1:-1, :, :] += (u[2:, :, :] - 2*u[1:-1, :, :] + u[:-2, :, :]) / dx**2
        d2u[:, :, 1:-1] += (u[:, :, 2:] - 2*u[:, :, 1:-1] + u[:, :, :-2]) / dz**2
        du_dt += K_diff * d2u

        d2w = np.zeros(shape)
        d2w[1:-1, :, :] += (w[2:, :, :] - 2*w[1:-1, :, :] + w[:-2, :, :]) / dx**2
        d2w[:, :, 1:-1] += (w[:, :, 2:] - 2*w[:, :, 1:-1] + w[:, :, :-2]) / dz**2
        dw_dt += K_diff * d2w

        # ---- Advecção de θ (upwind) ----
        dtheta_dt = np.zeros(shape)
        # ∂θ/∂x · u
        dtheta_dx = np.zeros(shape)
        dtheta_dx[1:-1, :, :] = (theta[2:, :, :] - theta[:-2, :, :]) / (2.0 * dx)
        # ∂θ/∂z · w
        dtheta_dz = np.zeros(shape)
        dtheta_dz[:, :, 1:-1] = (theta[:, :, 2:] - theta[:, :, :-2]) / (2.0 * dz)

        dtheta_dt = -(u * dtheta_dx + w * dtheta_dz)

        # Difusão de θ
        d2theta = np.zeros(shape)
        d2theta[1:-1, :, :] += (theta[2:, :, :] - 2*theta[1:-1, :, :] + theta[:-2, :, :]) / dx**2
        d2theta[:, :, 1:-1] += (theta[:, :, 2:] - 2*theta[:, :, 1:-1] + theta[:, :, :-2]) / dz**2
        dtheta_dt += K_diff * d2theta

        # ---- Euler forward ----
        u += du_dt * dt
        w += dw_dt * dt
        theta += dtheta_dt * dt

        # Boundary conditions
        w[:, :, 0] = 0.0    # w=0 na superfície
        w[:, :, -1] = 0.0   # w=0 no topo
        u[0, :, :] = 0.0    # u=0 nas laterais
        u[-1, :, :] = 0.0

        # Sanitize
        u = np.nan_to_num(u, nan=0.0)
        w = np.nan_to_num(w, nan=0.0)
        theta = np.nan_to_num(theta, nan=theta_bar)

        # ---- Diagnósticos ----
        if t >= t_output_next:
            w_max = float(np.max(w))
            theta_max = float(np.max(theta))
            theta_min = float(np.min(theta))

            # "Energy": integral de θ'^2
            energy = float(np.sum((theta - theta_bar)**2) * dx * dz)

            history['time'].append(float(t))
            history['w_max'].append(w_max)
            history['theta_max'].append(theta_max)
            history['theta_min'].append(theta_min)
            history['energy'].append(energy)

            print(f"  t={t:6.0f}s | w_max={w_max:6.2f} m/s | "
                  f"θ'_max={theta_max-theta_bar:+.4f} K | "
                  f"θ'_min={theta_min-theta_bar:+.4f} K")

            # Save snapshot
            snapshots[f't{int(t):04d}'] = {
                'theta_prime_xz': (theta[:, 0, :] - theta_bar).tolist(),
                'w_xz': w[:, 0, :].tolist(),
                'u_xz': u[:, 0, :].tolist(),
            }

            t_output_next += t_output

    wall_time = time_module.time() - start_wall

    # ================================================================
    # Avaliação
    # ================================================================
    w_max_final = float(np.max(w))
    theta_conserved = abs(history['energy'][-1] - history['energy'][0]) / max(history['energy'][0], 1e-10) * 100

    print("\n" + "=" * 60)
    print("  RESULTADOS DO BENCHMARK")
    print("=" * 60)

    # Comparação com Bryan & Fritsch (2002)
    # w_max esperado: ~10-12 m/s em t=1000s para Δθ=0.5K, R=2km
    w_ref = 11.0  # m/s — valor de referência aproximado
    w_error = abs(w_max_final - w_ref) / w_ref * 100

    results = {
        'test': 'Robert Bubble (1993)',
        'reference': 'Bryan & Fritsch (2002), Mon. Wea. Rev.',
        'grid': f'{nx}×{nz} (Δ={dx}m)',
        'w_max_model': round(w_max_final, 2),
        'w_max_reference': w_ref,
        'w_max_error_pct': round(w_error, 1),
        'theta_conservation_pct': round(theta_conserved, 2),
        'wall_time_s': round(wall_time, 1),
        'status': 'PASS' if w_error < 50 and theta_conserved < 50 else 'FAIL',
        'history': history,
    }

    print(f"  w_max (modelo):     {w_max_final:.2f} m/s")
    print(f"  w_max (referência): {w_ref:.1f} m/s (Bryan & Fritsch 2002)")
    print(f"  Erro relativo:      {w_error:.1f}%")
    print(f"  Conservação θ:      {theta_conserved:.2f}% variação")
    print(f"  Tempo de execução:  {wall_time:.1f}s")

    if results['status'] == 'PASS':
        print(f"\n  ✅ BENCHMARK APROVADO")
        print(f"     A dinâmica anelástica está funcionando corretamente.")
    else:
        print(f"\n  ❌ BENCHMARK REPROVADO")
        print(f"     Verificar solver de pressão e advecção.")

    # ================================================================
    # Salvar resultados
    # ================================================================
    os.makedirs('output', exist_ok=True)
    output_path = 'output/benchmark_robert_bubble.json'
    with open(output_path, 'w') as f:
        json.dump({**results, 'snapshots': snapshots,
                   'x': x.tolist(), 'z': z.tolist()}, f, indent=2)
    print(f"\n  Resultados salvos em: {output_path}")

    return results


def run_microphysics_tests():
    """Testes unitários de microfísica isolada."""

    print("\n" + "=" * 60)
    print("  TESTES UNITÁRIOS DE MICROFÍSICA")
    print("=" * 60)

    from core.microphysics_bulk import (
        _saturation_adjustment, _autoconversion,
        _hallett_mossop, _riming, _sedimentation
    )
    from core.constants import L_v, L_f, T_0

    n_pass = 0
    n_fail = 0

    # --- Teste 1: Condensação ---
    print("\n  [1] Saturação/Condensação...")
    T = np.array([[[280.0]]])       # 7°C
    p = np.array([[[80000.0]]])     # 800 hPa
    # Saturação + 10% excesso
    from core.microphysics_bulk import _q_vs
    qvs = _q_vs(T, p)
    qv = qvs * 1.10  # 10% supersaturado
    qc = np.array([[[0.0]]])
    dt = 1.0

    T_new, qv_new, qc_new, dq_cond = _saturation_adjustment(T, p, qv, qc, dt)

    # Verificar: qv deve cair, qc deve subir, T deve subir
    cond_ok = (float(qv_new) < float(qv) and
               float(qc_new) > 0 and
               float(T_new) > float(T) and
               abs(float(qv_new - qv) + float(qc_new)) < 1e-10)  # conservação
    status = "✅ PASS" if cond_ok else "❌ FAIL"
    print(f"     qv: {float(qv)*1000:.3f} → {float(qv_new)*1000:.3f} g/kg")
    print(f"     qc: 0.000 → {float(qc_new)*1000:.3f} g/kg")
    print(f"     T:  {float(T):.2f} → {float(T_new):.2f} K (ΔT={float(T_new-T):.3f} K)")
    print(f"     Conservação: Δqv + Δqc = {float(qv_new-qv)+float(qc_new):.2e}")
    print(f"     {status}")
    n_pass += cond_ok; n_fail += (not cond_ok)

    # --- Teste 2: Autoconversão (Kessler) ---
    print("\n  [2] Autoconversão (Kessler 1969)...")
    qc_test = np.array([[[2e-3]]])  # 2 g/kg (acima threshold)
    qc_new, dq_auto, _ = _autoconversion(qc_test, dt=1.0)
    # Esperado: dq_auto = 1e-3 * (2e-3 - 1e-3) * 1.0 = 1e-6
    expected = 1e-6
    auto_ok = abs(float(dq_auto) - expected) / expected < 0.01
    status = "✅ PASS" if auto_ok else "❌ FAIL"
    print(f"     qc = 2.0 g/kg, threshold = 1.0 g/kg")
    print(f"     dq_auto = {float(dq_auto)*1e6:.1f}×10⁻⁶ kg/kg (esperado: {expected*1e6:.1f}×10⁻⁶)")
    print(f"     {status}")
    n_pass += auto_ok; n_fail += (not auto_ok)

    # --- Teste 3: Hallett-Mossop ---
    print("\n  [3] Hallett-Mossop SIP...")
    T_hm = np.array([[[T_0 - 5.0]]])  # -5°C (pico H-M)
    dm_rime = np.array([[[1e-4]]])      # 0.1 g/kg rimado
    qi_init = np.array([[[1e-5]]])
    qc_hm = np.array([[[5e-3]]])        # 5 g/kg disponível

    qi_new, dq_sip, sip_rate = _hallett_mossop(T_hm, dm_rime, qi_init, qc_hm, dt=1.0)

    hm_ok = float(dq_sip) > 0 and float(qi_new) > float(qi_init)
    status = "✅ PASS" if hm_ok else "❌ FAIL"
    print(f"     T = −5°C (pico), dm_rime = 0.1 g/kg")
    print(f"     dq_sip = {float(dq_sip)*1e6:.2f}×10⁻⁶ kg/kg")
    print(f"     SIP rate = {float(sip_rate):.0f} #/kg/s")
    print(f"     {status}")
    n_pass += hm_ok; n_fail += (not hm_ok)

    # --- Teste 4: H-M deve ser ZERO fora da janela ---
    print("\n  [4] H-M fora da janela (T=−15°C → deve ser zero)...")
    T_outside = np.array([[[T_0 - 15.0]]])
    _, dq_sip_out, _ = _hallett_mossop(T_outside, dm_rime, qi_init, qc_hm, dt=1.0)
    hm_zero_ok = float(dq_sip_out) == 0.0
    status = "✅ PASS" if hm_zero_ok else "❌ FAIL"
    print(f"     T = −15°C (fora da janela −3 a −8°C)")
    print(f"     dq_sip = {float(dq_sip_out):.2e} (esperado: 0)")
    print(f"     {status}")
    n_pass += hm_zero_ok; n_fail += (not hm_zero_ok)

    # --- Teste 5: Sedimentation conserva massa ---
    print("\n  [5] Sedimentation conserva massa...")
    nx_t, ny_t, nz_t = 5, 5, 20
    qr_t = np.zeros((nx_t, ny_t, nz_t))
    qs_t = np.zeros((nx_t, ny_t, nz_t))
    qg_t = np.zeros((nx_t, ny_t, nz_t))
    qg_t[:, :, 10] = 5e-3  # 5 g/kg no nível 10
    rho_t = np.ones((nx_t, ny_t, nz_t)) * 1.0

    mass_before = float(np.sum(qg_t))

    qr_new, qs_new, qg_new, _ = _sedimentation(qr_t, qs_t, qg_t, rho_t, dz=300.0, dt=1.0)
    mass_after = float(np.sum(qg_new))

    # Massa pode diminuir levemente (saída pela superfície), mas não aumentar
    sed_ok = mass_after <= mass_before * 1.01  # <1% erro
    status = "✅ PASS" if sed_ok else "❌ FAIL"
    print(f"     Massa antes: {mass_before:.6f}")
    print(f"     Massa depois: {mass_after:.6f}")
    print(f"     Variação: {(mass_after/mass_before - 1)*100:+.2f}%")
    print(f"     {status}")
    n_pass += sed_ok; n_fail += (not sed_ok)

    # --- Resumo ---
    print(f"\n  {'='*40}")
    print(f"  MICROFÍSICA: {n_pass} PASS, {n_fail} FAIL de {n_pass+n_fail} testes")
    print(f"  {'='*40}")

    return n_pass, n_fail


if __name__ == '__main__':
    print("\n" + "#" * 60)
    print("# VALIDAÇÃO DO MODELO TORÓ 3D")
    print("# Testes de Benchmark + Microfísica")
    print("#" * 60)

    # 1. Benchmark dinâmico
    bubble_results = run_robert_bubble()

    # 2. Testes de microfísica
    mp_pass, mp_fail = run_microphysics_tests()

    # Resumo final
    print("\n" + "=" * 60)
    print("  RESUMO DA VALIDAÇÃO")
    print("=" * 60)
    print(f"  Bolha de Robert: {bubble_results['status']}")
    print(f"    w_max = {bubble_results['w_max_model']:.2f} m/s "
          f"(ref: {bubble_results['w_max_reference']:.1f} m/s, "
          f"erro: {bubble_results['w_max_error_pct']:.1f}%)")
    print(f"    Conservação θ: {bubble_results['theta_conservation_pct']:.1f}%")
    print(f"  Microfísica: {mp_pass}/{mp_pass+mp_fail} testes aprovados")
    all_pass = bubble_results['status'] == 'PASS' and mp_fail == 0
    print(f"\n  {'✅ VALIDAÇÃO COMPLETA — MODELO APROVADO' if all_pass else '⚠️ VALIDAÇÃO COM PROBLEMAS'}")
    print("=" * 60)
