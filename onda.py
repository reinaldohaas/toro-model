import netCDF4 as nc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ds = nc.Dataset('output/toro3d.nc')

# Shape: (time=13, z=50, y=20, x=20)
W  = ds['W'][:]           # velocidade vertical
QC = ds['QC'][:]          # água de nuvem (pistão)
QG = ds['QG'][:]          # graupel
PP = ds['P_PRIME'][:]     # perturbação de pressão
t  = ds['time'][:]        # segundos
z  = ds['z'][:]  / 1000.  # km
x  = ds['x'][:]  / 1000.  # km
y  = ds['y'][:]  / 1000.  # km
NT, NZ, NY, NX = W.shape
print(f"Grid: {NT} tempos × {NZ}z × {NY}y × {NX}x")
print(f"Tempos: {t}")
print(f"W_max geral: {W.max():.4f} m/s")
ds.close()

BG = '#0a1020'; PAN = '#0d1a2e'; SP = '#2a4060'
def sax(a):
    a.set_facecolor(PAN); a.tick_params(colors='#6080a0', labelsize=8)
    [s.set_color(SP) for s in a.spines.values()]
    a.xaxis.label.set_color('#a0b8d0'); a.yaxis.label.set_color('#a0b8d0')

# ── FIG 1: Evolução temporal de W no plano XZ central ───────────────────────
iy = NY // 2
fig, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor=BG)
fig.suptitle(f'Campo W (m/s) — Seção XZ (y=centro) — {NT} snapshots temporais',
             color='#e0e8f0', fontsize=12)
vmax = max(abs(W[:, :, iy, :]).max(), 0.01)
for i in range(min(NT, 8)):
    ax = axes[i // 4][i % 4]
    im = ax.contourf(x, z, W[i, :, iy, :],
                     levels=40, cmap='RdBu_r', vmin=-vmax, vmax=vmax, alpha=0.9)
    # Sobrepõe contorno de QG (pistão)
    if QG[i].max() > 1e-6:
        ax.contour(x, z, QG[i, :, iy, :], levels=[QG[i].max()*0.3],
                   colors='#ffd700', linewidths=1.5)
    ax.set_title(f't={t[i]:.0f}s  w_max={W[i].max():.3f}m/s',
                 color='#c0d8f0', fontsize=8)
    ax.set_xlabel('x (km)', fontsize=7); ax.set_ylabel('z (km)', fontsize=7)
    sax(ax)
plt.colorbar(im, ax=axes, label='w (m/s)', shrink=0.6)
plt.tight_layout()
plt.savefig('onda_w_temporal.png', dpi=150, bbox_inches='tight', facecolor=BG)
print("Salvo: onda_w_temporal.png")

# ── FIG 2: P' e ondas acústicas no último snapshot ──────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG)
fig.suptitle(f'Ondas acústicas: P\' e W — t={t[-1]:.0f}s (último snapshot)',
             color='#e0e8f0', fontsize=12)

# P' XZ
vp = max(abs(PP[-1, :, iy, :]).max(), 0.01)
im0 = axes[0].contourf(x, z, PP[-1, :, iy, :],
                        levels=40, cmap='RdBu_r', vmin=-vp, vmax=vp, alpha=0.9)
axes[0].set_title("P' XZ — onda de pressão", color='#c0d8f0')
axes[0].set_xlabel('x (km)'); axes[0].set_ylabel('z (km)'); sax(axes[0])
plt.colorbar(im0, ax=axes[0], label="P' (Pa)")

# W XY num nível baixo (z~500m)
iz_low = np.argmin(np.abs(z - 0.5))
vw = max(abs(W[-1, iz_low, :, :]).max(), 0.001)
im1 = axes[1].contourf(x, y, W[-1, iz_low, :, :],
                        levels=40, cmap='RdBu_r', vmin=-vw, vmax=vw, alpha=0.9)
axes[1].set_title(f'W XY z={z[iz_low]:.2f}km — padrão radial', color='#c0d8f0')
axes[1].set_xlabel('x (km)'); axes[1].set_ylabel('y (km)'); sax(axes[1])
plt.colorbar(im1, ax=axes[1], label='w (m/s)')

# Série temporal de w_max (do diagnóstico)
try:
    ds2 = nc.Dataset('output/toro3d.nc')
    td = ds2['diag_time'][:]
    wmd = ds2['w_max_diag'][:]
    qgd = ds2['qg_max_diag'][:]
    ds2.close()
    axes[2].plot(td, wmd, color='#40c8ff', lw=2, label='w_max (m/s)')
    ax2t = axes[2].twinx()
    ax2t.plot(td, qgd*1000, color='#ffd700', lw=2, ls='--', label='qg_max (g/kg)')
    axes[2].set_xlabel('Tempo (s)'); axes[2].set_ylabel('w_max (m/s)', color='#40c8ff')
    ax2t.set_ylabel('qg_max (g/kg)', color='#ffd700')
    axes[2].set_title('Série temporal diagnóstico', color='#c0d8f0')
    axes[2].legend(fontsize=8, facecolor=PAN, labelcolor='white', edgecolor=SP)
    ax2t.tick_params(colors='#6080a0', labelsize=7)
    sax(axes[2])
except: pass

plt.tight_layout()
plt.savefig('onda_acustica.png', dpi=150, bbox_inches='tight', facecolor=BG)
print("Salvo: onda_acustica.png")

# ── FREQUÊNCIA DOMINANTE ──────────────────────────────────────────────────────
# Ponto na borda do domínio (captura onda chegando)
w_centro = W[:, :, iy, NX//2]  # série temporal no centro XZ
w_max_t  = np.array([W[it].max() for it in range(NT)])
dt_mean  = np.diff(t).mean() if len(t) > 1 else 10.
freqs    = np.fft.rfftfreq(NT, d=dt_mean)
fft_w    = np.abs(np.fft.rfft(w_max_t))
if len(freqs) > 1:
    f_dom = freqs[np.argmax(fft_w[1:])+1]
    print(f"\nFrequência dominante em w_max(t): {f_dom:.4f} Hz  ({1/f_dom:.1f} s período)")
    if f_dom < 20:
        print("→ INFRASSOM (<20 Hz) — assinatura acústica do Toró confirmada!")
    else:
        print("→ Banda sonora audível")
