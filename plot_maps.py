"""plot_maps.py — Mapas de VIL máximo e precipitação acumulada.

Compara a rodada idealizada com a rodada ERA5 (parâmetros v6:
pistão ~10 ton via pressão no centro do vórtice, F2, R=20 m).

Uso:
    python plot_maps.py [nc_idealizado] [nc_era5] [png_saida]
"""
import sys
import netCDF4 as nc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_sim(file_path, title, axes):
    ds = nc.Dataset(file_path)
    vil = ds.variables['VIL_TOTAL'][:]
    vil_max = np.max(vil, axis=0)
    precip = ds.variables['PRECIP_ACC'][-1, :, :]
    x = ds.variables['x'][:] / 1000.0
    y = ds.variables['y'][:] / 1000.0
    X, Y = np.meshgrid(x, y)
    ax_vil, ax_precip = axes
    im1 = ax_vil.pcolormesh(X, Y, vil_max, cmap='nipy_spectral',
                            shading='auto', vmin=0,
                            vmax=max(50, np.max(vil_max)))
    ax_vil.set_title(f"VIL Max - {title}")
    ax_vil.set_xlabel("X (km)")
    ax_vil.set_ylabel("Y (km)")
    plt.colorbar(im1, ax=ax_vil, label="VIL (kg/m²)")
    im2 = ax_precip.pcolormesh(X, Y, precip, cmap='Blues',
                               shading='auto', vmin=0,
                               vmax=max(10, np.max(precip)))
    ax_precip.set_title(f"Precip. Acumulada - {title}")
    ax_precip.set_xlabel("X (km)")
    ax_precip.set_ylabel("Y (km)")
    plt.colorbar(im2, ax=ax_precip, label="Chuva (mm)")
    ds.close()


if __name__ == '__main__':
    nc_ideal = sys.argv[1] if len(sys.argv) > 1 else 'output_v6_3d/toro3d.nc'
    nc_era5 = sys.argv[2] if len(sys.argv) > 2 else 'output_v6_era5/toro3d.nc'
    out_path = sys.argv[3] if len(sys.argv) > 3 else 'maps_vil_prec.png'

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    plot_sim(nc_ideal, 'Idealizado (v6)', axs[0])
    plot_sim(nc_era5, 'ERA5 Sondagem (v6)', axs[1])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print("Mapa salvo em:", out_path)
