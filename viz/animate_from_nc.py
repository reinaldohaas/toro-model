"""
animate_from_nc.py — Animação 3D a partir dos dados REAIS do modelo Toró (NetCDF).

Lê output/toro3d.nc e renderiza com PyVista:
    - Camera close-up acompanhando a dinâmica
    - Isosuperfícies de W (updraft), QC (nuvem), QG (granizo), P' (pressão)
    
Saída: viz/toro_realdata_3d.mp4

Uso:
    python viz/animate_from_nc.py [--start FRAME] [--end FRAME]
"""

import numpy as np
import os
import sys


def load_nc(path='output/toro3d.nc'):
    """Carrega dados 4D do NetCDF."""
    from netCDF4 import Dataset
    
    ds = Dataset(path, 'r')
    
    data = {
        'x': ds['x'][:],
        'y': ds['y'][:],
        'z': ds['z'][:],
        'time': ds['time'][:],
    }
    
    # Campos 4D: (time, z, y, x)
    for var in ['W', 'QC', 'QG', 'THETA_RHO', 'P_PRIME']:
        if var in ds.variables:
            data[var] = ds[var][:]
            print(f"  {var}: shape={data[var].shape}")
    
    # U, V se disponíveis
    for var in ['U', 'V']:
        if var in ds.variables:
            data[var] = ds[var][:]
    
    ds.close()
    
    n_times = len(data['time'])
    print(f"\n  Carregado: {n_times} frames")
    print(f"  Tempo: {data['time'][0]:.0f}s → {data['time'][-1]:.0f}s")
    print(f"  Grid: x={len(data['x'])}, y={len(data['y'])}, z={len(data['z'])}")
    
    return data


def smooth_step(t, t0, t1):
    """Hermite smoothstep."""
    s = np.clip((t - t0) / (t1 - t0), 0, 1)
    return s * s * (3 - 2 * s)


def get_camera(frac, xc, yc, z_max, w_max, qg_max):
    """Câmera cinematográfica close-up em 3 fases.
    
    Domínio: ~10km x 10km x 15km. Centro em (xc,yc) ≈ (4750, 4750).
    
    Fase 1 (0-30%):  Close-up lateral baixo — formação da bolha térmica
    Fase 2 (30-65%): Sobe com a coluna — acompanha o updraft crescendo
    Fase 3 (65-100%): Afasta e gira — visão do sistema maduro com granizo
    """
    angle = 30 + 200 * frac  # gira 200° total
    
    if frac < 0.30:
        # FASE 1: Close-up baixo, vê a bolha subir
        p = frac / 0.30
        cam_dist = 6000 - 1000 * p     # 6km → 5km
        cam_z = 1500 + 2500 * p        # 1.5km → 4km (sobe com bolha)
        focal_z = 1000 + 2000 * p      # olha para cima
        
    elif frac < 0.65:
        # FASE 2: Acompanha updraft intenso subindo
        p = (frac - 0.30) / 0.35
        cam_dist = 5000                # mantém distância
        cam_z = 4000 + 4000 * p        # 4km → 8km (sobe rápido)
        focal_z = 3000 + 5000 * p      # foco sobe com topo
        
    else:
        # FASE 3: Afasta para ver o sistema completo
        p = (frac - 0.65) / 0.35
        p_s = smooth_step(p, 0, 1)
        cam_dist = 5000 + 4000 * p_s   # afasta para 9km
        cam_z = 8000 + 3000 * p_s      # sobe para 11km
        focal_z = 8000 - 1000 * p_s    # olha um pouco mais baixo
    
    cam_x = xc + cam_dist * np.cos(np.radians(angle))
    cam_y = yc + cam_dist * np.sin(np.radians(angle))
    
    return (cam_x, cam_y, cam_z), (xc, yc, focal_z), (0, 0, 1)


def render(data, output_path='viz/toro_realdata_3d.mp4',
           start_frame=0, end_frame=None, fps=20):
    """Renderiza animação 3D a partir dos dados reais."""
    import pyvista as pv
    pv.OFF_SCREEN = True
    
    x, y, z = data['x'], data['y'], data['z']
    times = data['time']
    xc, yc = x.mean(), y.mean()
    
    if end_frame is None:
        end_frame = len(times)
    
    frame_indices = range(start_frame, end_frame)
    n_render = len(frame_indices)
    
    print("\n" + "=" * 60)
    print("  RENDERIZAÇÃO 3D — Dados Reais do Modelo Toró")
    print("=" * 60)
    print(f"  Frames: {n_render} (#{start_frame}→#{end_frame-1})")
    print(f"  Tempo: {times[start_frame]:.0f}s → {times[end_frame-1]:.0f}s")
    print(f"  FPS: {fps}")
    
    frame_dir = 'viz/frames_realdata'
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Dimensões do grid para cell_data
    nx, ny, nz = len(x), len(y), len(z)
    
    for render_idx, fi in enumerate(frame_indices):
        t = float(times[fi])
        frac = render_idx / max(n_render - 1, 1)
        
        # Dados 4D: (time, z, y, x)
        w_3d = data['W'][fi]         # (nz, ny, nx)
        qc_3d = data['QC'][fi]
        qg_3d = data['QG'][fi]
        pp_3d = data['P_PRIME'][fi]
        
        # Transpor para PyVista: (nx, ny, nz)
        w = w_3d.transpose(2, 1, 0)
        qc = qc_3d.transpose(2, 1, 0)
        qg = qg_3d.transpose(2, 1, 0)
        pp = pp_3d.transpose(2, 1, 0)
        
        # ---- Plotter ----
        pl = pv.Plotter(off_screen=True, window_size=[1920, 1080])
        pl.set_background('black', top='midnightblue')
        
        grid = pv.RectilinearGrid(x, y, z)
        
        # ============================================
        # W — Updraft (vermelho/laranja)
        # ============================================
        w_max = float(abs(w).max())
        if w_max > 1.0:
            grid.cell_data['W'] = w[:-1,:-1,:-1].ravel(order='F')
            iso_w = grid.cell_data_to_point_data()
            
            # Updraft (w > 0)
            for level, opacity, color in [
                (2.0, 0.15, 'lightyellow'),
                (10.0, 0.25, 'orange'),
                (30.0, 0.40, 'orangered'),
                (45.0, 0.55, 'red'),
            ]:
                if w_max > level:
                    try:
                        c = iso_w.contour([level], scalars='W')
                        if c.n_points > 10:
                            pl.add_mesh(c, color=color, opacity=opacity,
                                        smooth_shading=True)
                    except Exception:
                        pass
            
            # Downdraft (w < 0) — azul
            if w.min() < -2.0:
                grid.cell_data['W_NEG'] = (-w[:-1,:-1,:-1]).ravel(order='F')
                iso_wn = grid.cell_data_to_point_data()
                try:
                    cn = iso_wn.contour([3.0], scalars='W_NEG')
                    if cn.n_points > 10:
                        pl.add_mesh(cn, color='steelblue', opacity=0.2,
                                    smooth_shading=True)
                except Exception:
                    pass
        
        # ============================================
        # QC — Água de nuvem (branco/cinza)
        # ============================================
        qc_max = float(qc.max())
        if qc_max > 0.1:
            grid_qc = pv.RectilinearGrid(x, y, z)
            grid_qc.cell_data['QC'] = qc[:-1,:-1,:-1].ravel(order='F')
            iso_qc = grid_qc.cell_data_to_point_data()
            
            for level, opacity, color in [
                (0.1, 0.08, 'lightgray'),
                (1.0, 0.15, 'white'),
                (3.0, 0.25, 'snow'),
            ]:
                if qc_max > level:
                    try:
                        c = iso_qc.contour([level], scalars='QC')
                        if c.n_points > 10:
                            pl.add_mesh(c, color=color, opacity=opacity,
                                        smooth_shading=True)
                    except Exception:
                        pass
        
        # ============================================
        # QG — Graupel/granizo (roxo)
        # ============================================
        qg_max = float(qg.max())
        if qg_max > 0.05:
            grid_qg = pv.RectilinearGrid(x, y, z)
            grid_qg.cell_data['QG'] = qg[:-1,:-1,:-1].ravel(order='F')
            iso_qg = grid_qg.cell_data_to_point_data()
            
            for level, opacity, color in [
                (0.1, 0.20, 'mediumpurple'),
                (0.5, 0.35, 'darkviolet'),
                (2.0, 0.55, 'indigo'),
            ]:
                if qg_max > level:
                    try:
                        c = iso_qg.contour([level], scalars='QG')
                        if c.n_points > 10:
                            pl.add_mesh(c, color=color, opacity=opacity,
                                        smooth_shading=True)
                    except Exception:
                        pass
        
        # ============================================
        # P' — Perturbação de pressão (Desativado devido ao ruído numérico)
        # ============================================
        # pp_range = float(abs(pp).max())
        # if pp_range > 10:
        #     grid_pp = pv.RectilinearGrid(x, y, z)
        #     pp_pos = np.clip(pp, 0, None)
        #     grid_pp.cell_data['PP'] = pp_pos[:-1,:-1,:-1].ravel(order='F')
        #     iso_pp = grid_pp.cell_data_to_point_data()
        #     thresh = max(20, pp_range * 0.3)
        #     try:
        #         cp = iso_pp.contour([thresh], scalars='PP')
        #         if cp.n_points > 10:
        #             pl.add_mesh(cp, color='darkorange', opacity=0.15,
        #                         smooth_shading=True)
        #     except Exception:
        #         pass
        
        # ============================================
        # Solo
        # ============================================
        ground = pv.Plane(
            center=(xc, yc, -5),
            direction=(0, 0, 1),
            i_size=(x.max()-x.min())*1.3,
            j_size=(y.max()-y.min())*1.3,
            i_resolution=20, j_resolution=20)
        pl.add_mesh(ground, color='forestgreen', opacity=0.6)
        
        # ============================================
        # Câmera
        # ============================================
        pos, focal, up = get_camera(frac, xc, yc, z.max(), w_max, qg_max)
        pl.camera_position = [pos, focal, up]
        
        # ============================================
        # HUD
        # ============================================
        pl.add_text(
            f"TORÓ — Modelo 3D Anelástico\n"
            f"t = {t:.0f}s  |  w_max = {w_max:.1f} m/s\n"
            f"QC = {qc_max:.2f} g/kg  |  QG = {qg_max:.2f} g/kg",
            position='upper_left', font_size=10, color='white'
        )
        
        pl.add_text(
            "Vale do Revólver — Presidente Getúlio, SC",
            position='lower_right', font_size=8, color='lightgray'
        )
        
        # Salvar frame
        frame_path = os.path.join(frame_dir, f'frame_{render_idx:04d}.png')
        pl.screenshot(frame_path)
        pl.close()
        
        if render_idx % 10 == 0:
            print(f"  Frame {render_idx:3d}/{n_render} | t={t:.0f}s | "
                  f"w={w_max:.1f} | qc={qc_max:.2f} | qg={qg_max:.2f}")
    
    # ============================================
    # Compilar vídeo
    # ============================================
    print("\n  Compilando vídeo...")
    try:
        import imageio
        writer = imageio.get_writer(output_path, fps=fps)
        for i in range(n_render):
            fp = os.path.join(frame_dir, f'frame_{i:04d}.png')
            if os.path.exists(fp):
                writer.append_data(imageio.imread(fp))
        writer.close()
        print(f"  ✅ Vídeo: {output_path}")
    except Exception as e:
        print(f"  [ERR] {e}")
        print(f"  Frames em: {frame_dir}/")
    
    print("  Renderização completa!")


if __name__ == '__main__':
    print("\n" + "#" * 60)
    print("# TORÓ — Animação 3D dos Dados Reais")
    print("# Modelo Anelástico θ_ρ")
    print("#" * 60)
    
    # Parse args
    start = 0
    end = None
    fps = 20
    for i, arg in enumerate(sys.argv[1:]):
        if arg == '--start':
            start = int(sys.argv[i+2])
        elif arg == '--end':
            end = int(sys.argv[i+2])
        elif arg == '--fps':
            fps = int(sys.argv[i+2])
    
    print("\n  Carregando NetCDF...")
    data = load_nc('output/toro3d.nc')
    
    render(data, start_frame=start, end_frame=end, fps=fps)
    
    print("\n" + "=" * 60)
    print("  ANIMAÇÃO COMPLETA")
    print("  viz/toro_realdata_3d.mp4")
    print("=" * 60)
