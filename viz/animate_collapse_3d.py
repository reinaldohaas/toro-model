"""
animate_collapse_3d.py — Animação 3D do colapso do pistão hidráulico com ondas acústicas.

Usa PyVista para renderizar:
    1. Pistão de hidrometeoros (isosuperfície QG + QC) descendo
    2. Ondas acústicas (isosuperfícies de p' esféricas expandindo)
    3. Zona de impacto com shockwave

Saída: viz/toro_collapse_3d.mp4  (ou .gif)

Uso:
    python viz/animate_collapse_3d.py
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def simulate_collapse(n_frames=60):
    """Simula o colapso do pistão com resolução temporal alta.
    
    Retorna snapshots de:
        - Campo de pressão p'(x,y,z,t) — ondas acústicas
        - Campo de qg(x,y,z,t) — pistão de hidrometeoros
        - Campo de qc(x,y,z,t) — água de nuvem
    """
    print("=" * 60)
    print("  SIMULAÇÃO DE COLAPSO — Alta resolução temporal")
    print("=" * 60)
    
    # Grade 3D
    nx, ny, nz = 40, 40, 60
    dx = dy = 250.0  # m
    dz = 250.0       # m
    
    x = np.linspace(0, (nx-1)*dx, nx)
    y = np.linspace(0, (ny-1)*dy, ny)
    z = np.linspace(0, (nz-1)*dz, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Centro do domínio
    xc = x.mean()
    yc = y.mean()
    
    # Parâmetros do pistão
    H_cloud_base = 4000.0   # m — base da nuvem
    H_cloud_top = 12000.0   # m — topo
    R_piston = 1500.0       # m — raio do pistão
    v_fall = 20.0           # m/s — velocidade terminal
    c_sound = 340.0         # m/s — velocidade do som
    
    # Tempo: colapso leva H_cloud_base / v_fall ≈ 200s
    t_total = 8.0   # s após impacto (para ondas acústicas)
    t_fall = H_cloud_base / v_fall  # ~200s
    
    # Simular: últimos 5s antes do impacto + 3s após
    t_pre = 5.0
    t_post = 3.0
    t_total = t_pre + t_post
    dt = t_total / n_frames
    
    snapshots = {
        'time': [],
        'p_prime': [],
        'qg': [],
        'qc': [],
        'impact_wave': [],
    }
    
    print(f"  Grade: {nx}×{ny}×{nz} (Δ={dx:.0f}m)")
    print(f"  Frames: {n_frames}")
    print(f"  t_pre = -{t_pre:.1f}s a t_post = +{t_post:.1f}s")
    print(f"  v_fall = {v_fall:.0f} m/s")
    print(f"  R_piston = {R_piston:.0f} m")
    
    for frame in range(n_frames):
        t = -t_pre + frame * dt  # t<0 = antes do impacto, t>0 = após
        
        # ============================================
        # Pistão de hidrometeoros (descendo)
        # ============================================
        z_piston_base = max(0.0, H_cloud_base + v_fall * t)  # desce com tempo
        z_piston_top = H_cloud_top + v_fall * t * 0.3  # topo desce mais devagar
        
        # Distância radial ao centro
        R_xy = np.sqrt((X - xc)**2 + (Y - yc)**2)
        
        # Perfil radial (Gaussiana)
        radial = np.exp(-(R_xy / R_piston)**2)
        
        # Perfil vertical (coluna entre base e topo)
        vertical = np.where(
            (Z >= z_piston_base) & (Z <= z_piston_top),
            1.0,
            0.0
        )
        # Suavizar bordas
        vertical = np.where(
            Z < z_piston_base,
            np.exp(-((Z - z_piston_base) / 500.0)**2),
            vertical
        )
        
        # Graupel: concentrado na parte inferior do pistão
        qg = 8.0 * radial * vertical  # g/kg
        # Concentrar mais na base (acumulação por sedimentation)
        z_rel = np.clip((Z - z_piston_base) / max(z_piston_top - z_piston_base, 1.0), 0, 1)
        qg *= np.exp(-2.0 * z_rel)  # mais denso na base
        
        # Água de nuvem: ao redor e acima do pistão
        qc = 3.0 * radial * np.where(Z > z_piston_base - 500, 1.0, 0.0)
        qc *= np.exp(-0.5 * z_rel)
        
        # ============================================
        # Perturbação de pressão (p') 
        # ============================================
        p_prime = np.zeros_like(X)
        
        if t < 0:
            # Antes do impacto: p' < 0 acima do pistão (sucção)
            z_suction = z_piston_base + 500
            r_suction = np.sqrt((X - xc)**2 + (Y - yc)**2 + (Z - z_suction)**2)
            p_prime -= 500.0 * np.exp(-(r_suction / 2000.0)**2)  # p' negativo
            
            # p' > 0 abaixo (compressão à frente do pistão)
            z_compress = max(0, z_piston_base - 300)
            r_compress = np.sqrt((X - xc)**2 + (Y - yc)**2 + (Z - z_compress)**2)
            p_prime += 300.0 * np.exp(-(r_compress / 1500.0)**2)
        
        # ============================================
        # Ondas acústicas (após impacto, t > 0)
        # ============================================
        impact_wave = np.zeros_like(X)
        
        if t > 0:
            # Onda esférica expandindo a c_sound
            r_impact = np.sqrt((X - xc)**2 + (Y - yc)**2 + Z**2)
            wave_radius = c_sound * t  # raio da frente de onda
            wave_width = 300.0  # espessura da onda (m)
            
            # Frente de onda (shell esférica)
            shell = np.exp(-((r_impact - wave_radius) / wave_width)**2)
            
            # Amplitude diminui com 1/r
            amplitude = 2000.0 / np.maximum(r_impact, 100.0)
            
            impact_wave = amplitude * shell
            
            # Adicionar ao p'
            p_prime += impact_wave * np.cos(2 * np.pi * r_impact / 500.0)  # oscilação
            
            # Segunda onda (reflexão, mais fraca, atrasada)
            if t > 0.5:
                wave_radius_2 = c_sound * (t - 0.5)
                shell_2 = np.exp(-((r_impact - wave_radius_2) / wave_width)**2)
                amplitude_2 = 800.0 / np.maximum(r_impact, 100.0)
                impact_wave += 0.4 * amplitude_2 * shell_2
            
            # Pistão se deforma no chão (splash)
            if z_piston_base <= 0:
                # Spread radial
                spread = 1.0 + 2.0 * t  # expande com o tempo
                qg_splash = 5.0 * np.exp(-(R_xy / (R_piston * spread))**2)
                qg_splash *= np.exp(-(Z / 500.0)**2)  # concentrado no chão
                qg = np.maximum(qg, qg_splash)
        
        snapshots['time'].append(t)
        snapshots['p_prime'].append(p_prime.astype(np.float32))
        snapshots['qg'].append(qg.astype(np.float32))
        snapshots['qc'].append(qc.astype(np.float32))
        snapshots['impact_wave'].append(impact_wave.astype(np.float32))
        
        if frame % 10 == 0:
            print(f"  Frame {frame:3d}/{n_frames} | t={t:+.2f}s | "
                  f"z_base={z_piston_base:.0f}m | "
                  f"qg_max={qg.max():.1f} | p'_max={abs(p_prime).max():.0f}")
    
    grid_info = {
        'x': x, 'y': y, 'z': z,
        'nx': nx, 'ny': ny, 'nz': nz,
        'dx': dx, 'dy': dy, 'dz': dz,
    }
    
    print(f"\n  Simulação completa: {n_frames} frames")
    return snapshots, grid_info


def render_animation(snapshots, grid_info, output_path='viz/toro_collapse_3d.mp4'):
    """Renderiza animação 3D com PyVista."""
    import pyvista as pv
    
    # Offscreen rendering
    pv.OFF_SCREEN = True
    
    x = grid_info['x']
    y = grid_info['y']
    z = grid_info['z']
    nx, ny, nz = grid_info['nx'], grid_info['ny'], grid_info['nz']
    
    n_frames = len(snapshots['time'])
    
    print("\n" + "=" * 60)
    print("  RENDERIZAÇÃO 3D — PyVista")
    print("=" * 60)
    print(f"  Frames: {n_frames}")
    print(f"  Saída: {output_path}")
    
    # Criar grid estruturado
    grid = pv.RectilinearGrid(x, y, z)
    
    # Centro do domínio para câmera
    xc = x.mean()
    yc = y.mean()
    
    # Diretório de saída
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Salvar frames como imagens
    frame_dir = 'viz/frames_collapse'
    os.makedirs(frame_dir, exist_ok=True)
    
    for frame_idx in range(n_frames):
        t = snapshots['time'][frame_idx]
        qg = snapshots['qg'][frame_idx]
        p_prime = snapshots['p_prime'][frame_idx]
        impact_wave = snapshots['impact_wave'][frame_idx]
        
        # Criar plotter
        pl = pv.Plotter(off_screen=True, window_size=[1920, 1080])
        pl.set_background('black', top='midnightblue')
        
        # Adicionar grid com dados
        grid_frame = pv.RectilinearGrid(x, y, z)
        
        # --- Pistão de hidrometeoros (isosuperfície) ---
        grid_frame.cell_data['QG'] = qg[:-1, :-1, :-1].ravel(order='F')
        
        qg_max = qg.max()
        if qg_max > 1.0:
            # Isosuperfície do graupel
            iso_qg = grid_frame.cell_data_to_point_data()
            try:
                contour_qg = iso_qg.contour(
                    isosurfaces=[2.0],
                    scalars='QG'
                )
                if contour_qg.n_points > 0:
                    pl.add_mesh(contour_qg, color='mediumpurple',
                                opacity=0.6, smooth_shading=True,
                                label='Pistão (QG)')
            except Exception:
                pass
            
            # Core denso
            try:
                contour_core = iso_qg.contour(
                    isosurfaces=[5.0],
                    scalars='QG'
                )
                if contour_core.n_points > 0:
                    pl.add_mesh(contour_core, color='darkviolet',
                                opacity=0.8, smooth_shading=True,
                                label='Core denso')
            except Exception:
                pass
        
        # --- Ondas acústicas (isosuperfícies de impact_wave) ---
        if impact_wave.max() > 0.1:
            grid_wave = pv.RectilinearGrid(x, y, z)
            grid_wave.cell_data['WAVE'] = impact_wave[:-1, :-1, :-1].ravel(order='F')
            iso_wave = grid_wave.cell_data_to_point_data()
            
            # Múltiplas cascas para visualizar propagação
            for wave_level, opacity, color in [
                (0.5, 0.15, 'cyan'),
                (1.0, 0.25, 'deepskyblue'),
                (2.0, 0.35, 'dodgerblue'),
            ]:
                if impact_wave.max() > wave_level:
                    try:
                        contour_w = iso_wave.contour(
                            isosurfaces=[wave_level],
                            scalars='WAVE'
                        )
                        if contour_w.n_points > 0:
                            pl.add_mesh(contour_w, color=color,
                                        opacity=opacity, smooth_shading=True)
                    except Exception:
                        pass
        
        # --- Pressão negativa (sucção acima do pistão) ---
        p_min = p_prime.min()
        if p_min < -100:
            grid_p = pv.RectilinearGrid(x, y, z)
            grid_p.cell_data['P_NEG'] = (-p_prime[:-1, :-1, :-1]).ravel(order='F')
            iso_p = grid_p.cell_data_to_point_data()
            try:
                contour_p = iso_p.contour(
                    isosurfaces=[200.0],
                    scalars='P_NEG'
                )
                if contour_p.n_points > 0:
                    pl.add_mesh(contour_p, color='orangered',
                                opacity=0.3, smooth_shading=True,
                                label="p' < 0 (sucção)")
            except Exception:
                pass
        
        # --- Superfície do solo ---
        ground = pv.Plane(
            center=(xc, yc, 0),
            direction=(0, 0, 1),
            i_size=x.max() - x.min(),
            j_size=y.max() - y.min(),
            i_resolution=20,
            j_resolution=20,
        )
        
        # Cor do solo: marrom, com cratera após impacto
        if t > 0:
            pl.add_mesh(ground, color='saddlebrown', opacity=0.8)
            # Anel de impacto
            ring = pv.Disc(center=(xc, yc, 5), inner=1400, outer=1600,
                          normal=(0, 0, 1), r_res=1, c_res=60)
            pl.add_mesh(ring, color='darkorange', opacity=0.7)
        else:
            pl.add_mesh(ground, color='forestgreen', opacity=0.6)
        
        # --- Câmera close-up ---
        # Posição: lateral, olhando para o centro
        cam_dist = 8000
        cam_height = 3000
        cam_angle = frame_idx * 0.5  # rotação lenta
        cam_x = xc + cam_dist * np.cos(np.radians(cam_angle + 30))
        cam_y = yc + cam_dist * np.sin(np.radians(cam_angle + 30))
        
        pl.camera_position = [
            (cam_x, cam_y, cam_height),   # posição
            (xc, yc, 2000),                # focal point
            (0, 0, 1),                     # up
        ]
        
        # --- Texto / HUD ---
        phase = "PRÉ-IMPACTO" if t < 0 else "PÓS-IMPACTO"
        pl.add_text(
            f"TORÓ — Colapso do Pistão Hidráulico\n"
            f"t = {t:+.2f}s  |  {phase}\n"
            f"QG_max = {qg.max():.1f} g/kg  |  |p'|_max = {abs(p_prime).max():.0f} Pa",
            position='upper_left',
            font_size=10,
            color='white',
            shadow=True,
        )
        
        # Escala
        pl.add_text("Vale do Revólver — Presidente Getúlio, SC",
                     position='lower_right', font_size=8, color='lightgray')
        
        # Salvar frame
        frame_path = os.path.join(frame_dir, f'frame_{frame_idx:04d}.png')
        pl.screenshot(frame_path)
        pl.close()
        
        if frame_idx % 10 == 0:
            print(f"  Renderizado frame {frame_idx:3d}/{n_frames}")
    
    # Compilar em vídeo
    print("\n  Compilando vídeo...")
    try:
        import imageio.v3 as iio
        
        frames = []
        for i in range(n_frames):
            frame_path = os.path.join(frame_dir, f'frame_{i:04d}.png')
            if os.path.exists(frame_path):
                frames.append(iio.imread(frame_path))
        
        if frames:
            # Salvar como MP4
            mp4_path = output_path
            iio.imwrite(mp4_path, frames, fps=15)
            print(f"  ✅ Vídeo salvo: {mp4_path}")
            
            # Também salvar como GIF
            gif_path = output_path.replace('.mp4', '.gif')
            iio.imwrite(gif_path, frames, duration=67, loop=0)
            print(f"  ✅ GIF salvo: {gif_path}")
    except Exception as e:
        print(f"  [WARN] Erro ao compilar vídeo: {e}")
        print(f"  Frames individuais em: {frame_dir}/")
    
    print("  Renderização completa!")


if __name__ == '__main__':
    print("\n" + "#" * 60)
    print("# TORÓ — Animação 3D do Colapso do Pistão")
    print("# Ondas acústicas + close-up do impacto")
    print("#" * 60)
    
    # 1. Simular colapso com alta resolução temporal
    snapshots, grid_info = simulate_collapse(n_frames=60)
    
    # 2. Renderizar animação 3D
    render_animation(snapshots, grid_info)
    
    print("\n" + "=" * 60)
    print("  ANIMAÇÃO COMPLETA")
    print("  Vídeo: viz/toro_collapse_3d.mp4")
    print("  GIF:   viz/toro_collapse_3d.gif")
    print("=" * 60)
