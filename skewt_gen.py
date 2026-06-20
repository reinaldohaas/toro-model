"""
skewt_gen.py — Skew-T log-P comparando o perfil idealizado do modelo
               com a sondagem ERA5 do evento real.

Uso:
    python skewt_gen.py                          # ambos lado a lado
    python skewt_gen.py --era5 data/sounding_era5_pg.json
    python skewt_gen.py --only-era5              # só ERA5
    python skewt_gen.py --only-ideal             # só idealizado
    python skewt_gen.py --out figuras/           # diretório de saída

Saída:
    skewt_comparacao.png   — painel duplo (idealizado | ERA5)
    skewt_era5.png         — só ERA5 com índices (CAPE, CIN, LCL...)
    skewt_idealizado.png   — só idealizado

Requer: numpy, matplotlib, metpy
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import metpy.calc as mpcalc
from metpy.plots import SkewT, Hodograph
from metpy.units import units
import mpl_toolkits.axes_grid1.inset_locator as inset

# ─────────────────────────────────────────────────────────────────────────────
# ARGPARSE
# ─────────────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(description='Skew-T: idealizado vs ERA5')
ap.add_argument('--era5', default='data/sounding_era5_pg.json',
                help='Caminho do JSON ERA5')
ap.add_argument('--only-era5',  action='store_true')
ap.add_argument('--only-ideal', action='store_true')
ap.add_argument('--out', default='.', help='Diretório de saída')
ap.add_argument('--dpi', type=int, default=160)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PARÂMETROS DO PERFIL IDEALIZADO (espelham core/config.py)
# ─────────────────────────────────────────────────────────────────────────────
IDEAL = dict(
    T_sfc=300.0,           # K
    p_sfc=101325.0,        # Pa
    RH_sfc=0.85,           # 85 %
    gamma=6.5e-3,          # K/m
    z_trop=12000.0,        # m
    T_trop=210.0,          # K
    H_q=8000.0,            # m — escala de altura do vapor (assumida)
)
Rd, g, eps, cp = 287.05, 9.81, 0.622, 1004.0
p0 = 100000.0


def esat(T_C):
    """Pressão de vapor de saturação (hPa) — Bolton (1980), T em °C."""
    return 6.112 * np.exp(17.67 * T_C / (T_C + 243.5))


def build_idealized():
    """Constrói o perfil idealizado em níveis de pressão padrão."""
    p_levels_hPa = np.array(
        [1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600,
         550, 500, 450, 400, 350, 300, 250, 200, 150, 100, 70, 50, 30],
        dtype=float)

    T_C, Td_C, z_m = [], [], []
    for p in p_levels_hPa:
        # altura hipsométrica aproximada (atm isotérmica por camada)
        z = -Rd * IDEAL['T_sfc'] / g * np.log(p * 100.0 / IDEAL['p_sfc'])
        z = max(z, 0.0)

        # temperatura: troposfera com lapse rate; isoterma acima da tropopausa
        if z <= IDEAL['z_trop']:
            T = IDEAL['T_sfc'] - IDEAL['gamma'] * z
        else:
            T = IDEAL['T_tropopause'] if 'T_tropopause' in IDEAL else IDEAL['T_trop']
        T_C_val = T - 273.15

        # umidade: RH decai com a altura (proxy de escala H_q)
        RH = IDEAL['RH_sfc'] * np.exp(-z / IDEAL['H_q'])
        RH = float(np.clip(RH, 0.05, 1.0))
        es = esat(T_C_val)
        e  = RH * es
        # ponto de orvalho por inversão de Bolton
        if e > 0.01:
            Td_C_val = 243.5 * np.log(e / 6.112) / (17.67 - np.log(e / 6.112))
        else:
            Td_C_val = T_C_val - 40.0

        T_C.append(T_C_val)
        Td_C.append(min(Td_C_val, T_C_val))
        z_m.append(z)

    return (p_levels_hPa, np.array(T_C), np.array(Td_C), np.array(z_m))


def load_era5(path):
    """Carrega a sondagem ERA5 do JSON."""
    with open(path) as f:
        d = json.load(f)
    meta = d.get('metadata', {})
    levels = d['levels']
    p_sorted = sorted([int(k) for k in levels.keys()], reverse=True)

    p_hPa, T_C, Td_C, z_m, u_ms, v_ms = [], [], [], [], [], []
    for p in p_sorted:
        lvl = levels[str(p)]
        if lvl.get('temperature') is None:
            continue
        T = lvl['temperature']                       # já em °C
        RH = (lvl.get('relative_humidity') or 50.0)
        es = esat(T)
        e  = (RH / 100.0) * es
        if e > 0.01:
            Td = 243.5 * np.log(e / 6.112) / (17.67 - np.log(e / 6.112))
        else:
            Td = T - 40.0
        p_hPa.append(float(p))
        T_C.append(T)
        Td_C.append(min(Td, T))
        z_m.append(lvl.get('geopotential_height', np.nan))
        u_ms.append(lvl.get('u_component', 0.0) or 0.0)
        v_ms.append(lvl.get('v_component', 0.0) or 0.0)

    return (np.array(p_hPa), np.array(T_C), np.array(Td_C),
            np.array(z_m), np.array(u_ms), np.array(v_ms), meta)


def plot_skewt(ax_fig, p_hPa, T_C, Td_C, u=None, v=None, title='',
               subtitle='', show_indices=True):
    """Desenha um Skew-T num objeto SkewT do MetPy."""
    skew = ax_fig

    p  = p_hPa * units.hPa
    T  = T_C * units.degC
    Td = Td_C * units.degC

    skew.plot(p, T,  'r', lw=2.2, label='Temperatura')
    skew.plot(p, Td, 'g', lw=2.2, label='Ponto de orvalho')

    # Parcela ascendente e índices
    if show_indices:
        try:
            prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
            skew.plot(p, prof, 'k', lw=1.6, ls='--', alpha=0.8, label='Parcela')
            skew.shade_cape(p, T, prof)
            skew.shade_cin(p, T, prof)

            cape, cin = mpcalc.cape_cin(p, T, Td, prof)
            lcl_p, lcl_t = mpcalc.lcl(p[0], T[0], Td[0])
            try:
                lfc_p, _ = mpcalc.lfc(p, T, Td)
                el_p, _  = mpcalc.el(p, T, Td)
            except Exception:
                lfc_p, el_p = np.nan * units.hPa, np.nan * units.hPa

            skew.ax.axhline(lcl_p.m, color='purple', ls=':', lw=1, alpha=0.7)
            txt = (f"CAPE = {cape.m:.0f} J/kg\n"
                   f"CIN  = {cin.m:.0f} J/kg\n"
                   f"LCL  = {lcl_p.m:.0f} hPa")
            if not np.isnan(lfc_p.m):
                txt += f"\nLFC  = {lfc_p.m:.0f} hPa"
            if not np.isnan(el_p.m):
                txt += f"\nEL   = {el_p.m:.0f} hPa"
            skew.ax.text(0.02, 0.02, txt, transform=skew.ax.transAxes,
                         fontsize=8, va='bottom', ha='left', family='monospace',
                         bbox=dict(boxstyle='round', facecolor='white',
                                   edgecolor='gray', alpha=0.85))
        except Exception as e:
            print(f"  (índices não calculados: {e})")

    # Linhas de referência
    skew.plot_dry_adiabats(alpha=0.25, lw=0.7)
    skew.plot_moist_adiabats(alpha=0.25, lw=0.7)
    skew.plot_mixing_lines(alpha=0.25, lw=0.7)

    skew.ax.set_ylim(1000, 100)
    skew.ax.set_xlim(-40, 40)
    skew.ax.set_xlabel('Temperatura (°C)')
    skew.ax.set_ylabel('Pressão (hPa)')
    skew.ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    skew.ax.legend(loc='upper right', fontsize=8)

    # Hodógrafo embutido se houver vento
    if u is not None and v is not None and np.any(np.abs(u) > 0):
        ax_hod = inset.inset_axes(skew.ax, '32%', '32%', loc='upper left')
        h = Hodograph(ax_hod, component_range=40)
        h.add_grid(increment=10, alpha=0.3)
        h.plot_colormapped(u * units('m/s'), v * units('m/s'),
                           p_hPa * units.hPa)
        ax_hod.set_xlabel('u (m/s)', fontsize=6)
        ax_hod.tick_params(labelsize=5)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    do_ideal = not args.only_era5
    do_era5  = not args.only_ideal

    # Carregar dados
    if do_ideal:
        pi, Ti, Tdi, zi = build_idealized()
    if do_era5:
        if not os.path.exists(args.era5):
            print(f"ERRO: {args.era5} não encontrado.")
            print("Rode data/fetch_era5_sounding.py primeiro, ou use --only-ideal.")
            return
        pe, Te, Tde, ze, ue, ve, meta = load_era5(args.era5)
        loc = (f"{meta.get('latitude','?')}, {meta.get('longitude','?')} | "
               f"{meta.get('date','?')} {meta.get('time','?')}Z")

    # ── Painel comparativo ───────────────────────────────────────────────────
    if do_ideal and do_era5:
        fig = plt.figure(figsize=(15, 9))
        fig.suptitle(
            "Skew-T log-P — Perfil Idealizado do Modelo vs. Sondagem ERA5 do Evento\n"
            "Toró da Valada São Paulo — Planalto Mirador, SC",
            fontsize=13, y=0.98)

        skew1 = SkewT(fig, subplot=(1, 2, 1), rotation=45)
        plot_skewt(skew1, pi, Ti, Tdi, title="IDEALIZADO (config.py)",
                   subtitle="T_sup=27°C · RH_sup=85% · γ=6.5 K/km")

        skew2 = SkewT(fig, subplot=(1, 2, 2), rotation=45)
        plot_skewt(skew2, pe, Te, Tde, u=ue, v=ve,
                   title="ERA5 (evento real)", subtitle=loc)

        out = os.path.join(args.out, 'skewt_comparacao.png')
        plt.savefig(out, dpi=args.dpi, bbox_inches='tight')
        print(f"Salvo: {out}")
        plt.close()

    # ── Só ERA5 ──────────────────────────────────────────────────────────────
    if do_era5 and args.only_era5:
        fig = plt.figure(figsize=(9, 9))
        skew = SkewT(fig, rotation=45)
        plot_skewt(skew, pe, Te, Tde, u=ue, v=ve,
                   title="Sondagem ERA5 — Toró da Valada São Paulo",
                   subtitle=loc)
        out = os.path.join(args.out, 'skewt_era5.png')
        plt.savefig(out, dpi=args.dpi, bbox_inches='tight')
        print(f"Salvo: {out}")
        plt.close()

    # ── Só idealizado ─────────────────────────────────────────────────────────
    if do_ideal and args.only_ideal:
        fig = plt.figure(figsize=(9, 9))
        skew = SkewT(fig, rotation=45)
        plot_skewt(skew, pi, Ti, Tdi,
                   title="Perfil Idealizado do Modelo",
                   subtitle="T_sup=27°C · RH_sup=85% · γ=6.5 K/km")
        out = os.path.join(args.out, 'skewt_idealizado.png')
        plt.savefig(out, dpi=args.dpi, bbox_inches='tight')
        print(f"Salvo: {out}")
        plt.close()

    print("\nConcluído.")


if __name__ == '__main__':
    main()
