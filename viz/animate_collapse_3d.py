"""
animate_collapse_3d.py — Animação 3D cinematográfica do colapso do pistão hidráulico.

Câmera em 3 fases:
    1. CLOSE-UP no pistão descendo (acompanhando a queda)
    2. IMPACTO no solo (câmera baixa, dramatic)
    3. PAN para cima mostrando propagação das ondas acústicas

Usa PyVista para renderizar isosuperfícies 3D.
Saída: viz/toro_collapse_3d.mp4

Uso:
    python viz/animate_collapse_3d.py [--frames N] [--fps N]
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ================================================================
# PARÂMETROS
# ================================================================
N_FRAMES = 150          # frames totais
FPS = 24                # frames por segundo
RESOLUTION = (1920, 1080)

# Física
R_PISTON = 1500.0       # m — raio do pistão
V_FALL = 20.0           # m/s — velocidade terminal
C_SOUND = 340.0         # m/s — velocidade do som
Z_CLOUD_BASE = 4000.0   # m — base da nuvem
Z_CLOUD_TOP = 12000.0   # m — topo

# Tempo da animação (comprimido para drama)
# Fase 1: últimos 15s antes do impacto (pistão cai 300m finais)
# Fase 2: impacto (1s)
# Fase 3: ondas acústicas expandindo (6s → ~2km de raio)
T_PRE_IMPACT = 15.0     # s antes do impacto
T_POST_IMPACT = 6.0     # s após impacto
T_TOTAL = T_PRE_IMPACT + T_POST_IMPACT


def smooth_step(t, t0, t1):
    """Interpolação suave entre 0 e 1 no intervalo [t0, t1]."""
    s = np.clip((t - t0) / (t1 - t0), 0, 1)
    return s * s * (3 - 2 * s)  # Hermite


def lerp(a, b, t):
    """Interpolação linear entre a e b."""
    return a + (b - a) * np.clip(t, 0, 1)


def simulate_collapse(n_frames=N_FRAMES):
    """Simula o colapso com alta resolução temporal."""
    print("=" * 60)
    print("  SIMULAÇÃO DE COLAPSO — Alta resolução")
    print("=" * 60)
    
    # Grade 3D
    nx, ny, nz = 50, 50, 80
    dx = dy = 200.0   # m
    dz = 200.0         # m
    
    x = np.linspace(0, (nx-1)*dx, nx)
    y = np.linspace(0, (ny-1)*dy, ny)
    z = np.linspace(0, (nz-1)*dz, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    xc, yc = x.mean(), y.mean()
    
    dt_frame = T_TOTAL / n_frames
    
    print(f"  Grade: {nx}×{ny}×{nz} (Δ={dx:.0f}m)")
    print(f"  Frames: {n_frames} @ {FPS}fps = {n_frames/FPS:.1f}s")
    print(f"  Período: t=-{T_PRE_IMPACT:.0f}s a t=+{T_POST_IMPACT:.0f}s")
    
    snapshots = {
        'time': [],
        'qg': [],
        'p_prime': [],
        'wave_shell': [],
        'z_piston_base': [],
    }
    
    R_xy = np.sqrt((X - xc)**2 + (Y - yc)**2)
    
    for frame in range(n_frames):
        t = -T_PRE_IMPACT + frame * dt_frame
        
        # ============================================
        # PISTÃO DE HIDROMETEOROS
        # ============================================
        # Posição da base do pistão (descendo)
        if t <= 0:
            z_base = max(0.0, Z_CLOUD_BASE - V_FALL * abs(t))
            # Acelera nos últimos metros (queda livre + arrasto)
            if z_base < 500:
                accel_factor = 1.0 + 0.5 * (500 - z_base) / 500
                z_base = max(0.0, z_base / accel_factor)
        else:
            z_base = 0.0  # no chão
        
        z_top = max(z_base + 500, Z_CLOUD_TOP - V_FALL * max(0, abs(t)) * 0.15)
        
        # Perfil radial (Gaussiana)
        radial = np.exp(-(R_xy / R_PISTON)**2)
        
        # Perfil vertical
        if t <= 0:
            # Coluna descendo
            in_column = ((Z >= z_base) & (Z <= z_top)).astype(float)
            # Suavizar borda inferior
            below = np.exp(-((Z - z_base) / 300.0)**2) * (Z < z_base).astype(float)
            vertical = in_column + below
            
            # Concentrar massa na base (acumulação gravitacional)
            z_rel = np.clip((Z - z_base) / max(z_top - z_base, 1.0), 0, 1)
            mass_profile = np.exp(-1.5 * z_rel)
        else:
            # Após impacto: splash radial
            spread_factor = 1.0 + 3.0 * t  # expande lateralmente
            vertical = np.exp(-(Z / (400.0 + 200*t))**2)
            radial = np.exp(-(R_xy / (R_PISTON * spread_factor))**2)
            mass_profile = 1.0
            
            # Coluna residual (deforma, encolhe)
            residual = np.exp(-(R_xy / R_PISTON)**2) * \
                       ((Z > 500) & (Z < z_top * (1 - 0.3*t))).astype(float) * \
                       np.exp(-0.5 * t)
            vertical = np.maximum(vertical, residual)
        
        qg = 8.0 * radial * vertical * mass_profile
        
        # ============================================
        # PERTURBAÇÃO DE PRESSÃO
        # ============================================
        p_prime = np.zeros_like(X)
        
        if t < 0:
            # Compressão à frente do pistão (abaixo)
            z_front = max(0, z_base - 200)
            r_front = np.sqrt(R_xy**2 + (Z - z_front)**2)
            p_prime += 800.0 * np.exp(-(r_front / 1200.0)**2)
            
            # Sucção atrás (acima)
            z_wake = z_base + 1000
            r_wake = np.sqrt(R_xy**2 + (Z - z_wake)**2)
            p_prime -= 400.0 * np.exp(-(r_wake / 1500.0)**2)
            
            # Intensifica perto do solo
            if z_base < 1000:
                ground_factor = 1.0 + 2.0 * (1000 - z_base) / 1000
                p_prime *= ground_factor
        
        # ============================================
        # ONDA ACÚSTICA (pós-impacto)
        # ============================================
        wave_shell = np.zeros_like(X)
        
        if t > 0:
            r_from_impact = np.sqrt(R_xy**2 + Z**2)
            
            # Frente de onda principal
            wave_r = C_SOUND * t
            wave_width = 200.0 + 50.0 * t  # alarga com o tempo
            
            # Shell esférica
            shell = np.exp(-((r_from_impact - wave_r) / wave_width)**2)
            
            # Amplitude: 1/r (conservação de energia)
            amplitude = np.where(
                r_from_impact > 50,
                3000.0 / r_from_impact,
                3000.0 / 50
            )
            
            wave_shell = amplitude * shell
            
            # Adicionar oscilações (ondas de pressão)
            wavelength = 400.0
            p_wave = amplitude * shell * np.cos(2*np.pi * r_from_impact / wavelength)
            p_prime += p_wave
            
            # Segunda frente (reflexão no solo, mais fraca)
            if t > 0.3:
                wave_r2 = C_SOUND * (t - 0.3)
                # Origem refletida (imagem abaixo do solo)
                r_reflected = np.sqrt(R_xy**2 + (Z + 200)**2)
                shell2 = np.exp(-((r_reflected - wave_r2) / (wave_width * 1.5))**2)
                amp2 = np.where(r_reflected > 50, 1500.0/r_reflected, 1500.0/50)
                wave_shell += 0.3 * amp2 * shell2
            
            # Terceira frente (eco topográfico, ainda mais fraca)
            if t > 1.0:
                wave_r3 = C_SOUND * (t - 1.0)
                shell3 = np.exp(-((r_from_impact - wave_r3) / (wave_width * 2))**2)
                amp3 = np.where(r_from_impact > 50, 800.0/r_from_impact, 800.0/50)
                wave_shell += 0.15 * amp3 * shell3
        
        snapshots['time'].append(t)
        snapshots['qg'].append(qg.astype(np.float32))
        snapshots['p_prime'].append(p_prime.astype(np.float32))
        snapshots['wave_shell'].append(wave_shell.astype(np.float32))
        snapshots['z_piston_base'].append(z_base)
        
        if frame % 15 == 0:
            print(f"  Frame {frame:3d}/{n_frames} | t={t:+6.2f}s | "
                  f"z_base={z_base:6.0f}m | qg_max={qg.max():.1f}")
    
    grid_info = {'x': x, 'y': y, 'z': z, 'nx': nx, 'ny': ny, 'nz': nz,
                 'dx': dx, 'xc': xc, 'yc': yc}
    
    print(f"\n  ✓ {n_frames} frames simulados")
    return snapshots, grid_info


def get_camera(t, frame_frac, xc, yc, z_piston):
    """Retorna (position, focal_point, up) da câmera para o tempo t.
    
    3 fases cinematográficas:
        1. Close-up: acompanha o pistão descendo (câmera lateral, mesma altura)
        2. Impacto: câmera baixa, próxima ao solo, olhando o impacto
        3. Pan up: câmera sobe suavemente, mostra ondas expandindo
    """
    # Transições suaves
    phase1_end = T_PRE_IMPACT / T_TOTAL          # ~0.71
    phase2_end = (T_PRE_IMPACT + 1.5) / T_TOTAL  # ~0.78
    
    # Rotação base (gira 120° durante toda a animação)
    angle = 30 + 120 * frame_frac
    
    if frame_frac < phase1_end:
        # ---- FASE 1: Close-up no pistão descendo ----
        progress = frame_frac / phase1_end
        
        # Câmera acompanha a altura do pistão
        cam_z = max(500, z_piston + 500)  # um pouco acima
        cam_dist = 4000 + 1000 * (1 - progress)  # aproxima gradualmente
        
        # Olha para o pistão
        focal_z = max(200, z_piston)
        
        cam_x = xc + cam_dist * np.cos(np.radians(angle))
        cam_y = yc + cam_dist * np.sin(np.radians(angle))
        
        return (cam_x, cam_y, cam_z), (xc, yc, focal_z), (0, 0, 1)
    
    elif frame_frac < phase2_end:
        # ---- FASE 2: Impacto (câmera baixa, dramática) ----
        progress = (frame_frac - phase1_end) / (phase2_end - phase1_end)
        
        cam_dist = lerp(4000, 3500, progress)
        cam_z = lerp(500, 300, progress)  # câmera desce ao nível do solo
        focal_z = lerp(200, 0, progress)
        
        cam_x = xc + cam_dist * np.cos(np.radians(angle))
        cam_y = yc + cam_dist * np.sin(np.radians(angle))
        
        return (cam_x, cam_y, cam_z), (xc, yc, focal_z), (0, 0, 1)
    
    else:
        # ---- FASE 3: Pan para cima (mostra ondas acústicas) ----
        progress = (frame_frac - phase2_end) / (1.0 - phase2_end)
        progress_smooth = smooth_step(progress, 0, 1)
        
        cam_dist = lerp(3500, 7000, progress_smooth)
        cam_z = lerp(300, 8000, progress_smooth)  # sobe suavemente
        focal_z = lerp(0, 3000, progress_smooth)   # focal sobe também
        
        cam_x = xc + cam_dist * np.cos(np.radians(angle))
        cam_y = yc + cam_dist * np.sin(np.radians(angle))
        
        return (cam_x, cam_y, cam_z), (xc, yc, focal_z), (0, 0, 1)


def render_animation(snapshots, grid_info,
                     output_path='viz/toro_collapse_3d.mp4'):
    """Renderiza animação 3D cinematográfica."""
    import pyvista as pv
    pv.OFF_SCREEN = True
    
    x, y, z = grid_info['x'], grid_info['y'], grid_info['z']
    xc, yc = grid_info['xc'], grid_info['yc']
    n_frames = len(snapshots['time'])
    
    print("\n" + "=" * 60)
    print("  RENDERIZAÇÃO 3D CINEMATOGRÁFICA — PyVista")
    print("=" * 60)
    print(f"  Frames: {n_frames} @ {FPS}fps")
    print(f"  Resolução: {RESOLUTION}")
    
    frame_dir = 'viz/frames_collapse'
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    for fi in range(n_frames):
        t = snapshots['time'][fi]
        qg = snapshots['qg'][fi]
        p_prime = snapshots['p_prime'][fi]
        wave = snapshots['wave_shell'][fi]
        z_piston = snapshots['z_piston_base'][fi]
        
        frac = fi / max(n_frames - 1, 1)
        
        # ---- Plotter ----
        pl = pv.Plotter(off_screen=True, window_size=list(RESOLUTION))
        
        # Background: escurece após impacto
        if t < 0:
            pl.set_background('black', top='midnightblue')
        else:
            flash = max(0, 1.0 - t * 2)  # flash branco no impacto
            bg = [int(10 + flash*100)] * 3
            top = [int(25 + flash*60), int(25 + flash*40), int(80 + flash*80)]
            pl.set_background(
                [c/255 for c in bg],
                top=[c/255 for c in top]
            )
        
        # ---- Grid base ----
        grid = pv.RectilinearGrid(x, y, z)
        
        # ============================================
        # PISTÃO DE HIDROMETEOROS
        # ============================================
        grid.cell_data['QG'] = qg[:-1, :-1, :-1].ravel(order='F')
        iso = grid.cell_data_to_point_data()
        
        # Camada externa (halo)
        try:
            c1 = iso.contour([1.5], scalars='QG')
            if c1.n_points > 10:
                pl.add_mesh(c1, color='mediumpurple', opacity=0.35,
                            smooth_shading=True)
        except Exception:
            pass
        
        # Core denso
        try:
            c2 = iso.contour([4.0], scalars='QG')
            if c2.n_points > 10:
                pl.add_mesh(c2, color='darkviolet', opacity=0.7,
                            smooth_shading=True)
        except Exception:
            pass
        
        # Núcleo ultra-denso
        try:
            c3 = iso.contour([6.5], scalars='QG')
            if c3.n_points > 10:
                pl.add_mesh(c3, color='indigo', opacity=0.9,
                            smooth_shading=True)
        except Exception:
            pass
        
        # ============================================
        # PRESSÃO (compressão/sucção)
        # ============================================
        if abs(p_prime).max() > 100:
            grid_p = pv.RectilinearGrid(x, y, z)
            
            # Compressão (p' > 0): laranja
            p_pos = np.clip(p_prime, 0, None)
            grid_p.cell_data['P_POS'] = p_pos[:-1,:-1,:-1].ravel(order='F')
            iso_p = grid_p.cell_data_to_point_data()
            try:
                cp = iso_p.contour([300], scalars='P_POS')
                if cp.n_points > 10:
                    pl.add_mesh(cp, color='orangered', opacity=0.3,
                                smooth_shading=True)
            except Exception:
                pass
            
            # Sucção (p' < 0): azul
            p_neg = np.clip(-p_prime, 0, None)
            grid_p.cell_data['P_NEG'] = p_neg[:-1,:-1,:-1].ravel(order='F')
            iso_pn = grid_p.cell_data_to_point_data()
            try:
                cn = iso_pn.contour([200], scalars='P_NEG')
                if cn.n_points > 10:
                    pl.add_mesh(cn, color='royalblue', opacity=0.25,
                                smooth_shading=True)
            except Exception:
                pass
        
        # ============================================
        # ONDAS ACÚSTICAS
        # ============================================
        if wave.max() > 0.5:
            grid_w = pv.RectilinearGrid(x, y, z)
            grid_w.cell_data['WAVE'] = wave[:-1,:-1,:-1].ravel(order='F')
            iso_w = grid_w.cell_data_to_point_data()
            
            # Múltiplas cascas com cores diferentes
            wave_levels = [
                (0.5, 0.12, 'lightcyan'),    # fraca — quase transparente
                (1.5, 0.20, 'cyan'),          # média
                (3.0, 0.30, 'deepskyblue'),   # forte
                (6.0, 0.45, 'dodgerblue'),    # muito forte
                (10.0, 0.55, 'royalblue'),    # frente principal
            ]
            
            for level, opacity, color in wave_levels:
                if wave.max() > level:
                    try:
                        cw = iso_w.contour([level], scalars='WAVE')
                        if cw.n_points > 10:
                            pl.add_mesh(cw, color=color, opacity=opacity,
                                        smooth_shading=True)
                    except Exception:
                        pass
        
        # ============================================
        # SOLO
        # ============================================
        ground_size = x.max() - x.min()
        
        if t > 0:
            # Solo marrom com cratera
            ground = pv.Plane(center=(xc, yc, -5),
                              direction=(0,0,1),
                              i_size=ground_size*1.2, j_size=ground_size*1.2,
                              i_resolution=30, j_resolution=30)
            pl.add_mesh(ground, color='saddlebrown', opacity=0.85)
            
            # Anel de impacto (expande)
            ring_r = R_PISTON + 200 * t
            ring = pv.Disc(center=(xc, yc, 10),
                          inner=ring_r - 100, outer=ring_r + 100,
                          normal=(0,0,1), r_res=1, c_res=80)
            pl.add_mesh(ring, color='darkorange', opacity=0.6)
            
            # Flash no centro do impacto
            if t < 0.5:
                flash_r = 200 + C_SOUND * t * 0.3
                sphere = pv.Sphere(radius=flash_r, center=(xc, yc, flash_r*0.5))
                flash_opacity = max(0, 0.6 - t)
                pl.add_mesh(sphere, color='white', opacity=flash_opacity)
        else:
            ground = pv.Plane(center=(xc, yc, -5),
                              direction=(0,0,1),
                              i_size=ground_size*1.2, j_size=ground_size*1.2,
                              i_resolution=30, j_resolution=30)
            pl.add_mesh(ground, color='forestgreen', opacity=0.7)
        
        # ============================================
        # CÂMERA CINEMATOGRÁFICA
        # ============================================
        pos, focal, up = get_camera(t, frac, xc, yc, z_piston)
        pl.camera_position = [pos, focal, up]
        
        # ============================================
        # HUD
        # ============================================
        if t < 0:
            phase_text = "COLAPSO DO PISTÃO"
            z_text = f"  Alt. pistão: {z_piston:.0f} m"
        elif t < 1.5:
            phase_text = ">>> IMPACTO <<<"
            z_text = f"  Onda: {C_SOUND*t:.0f} m"
        else:
            phase_text = "PROPAGAÇÃO ACÚSTICA"
            z_text = f"  Raio da onda: {C_SOUND*t:.0f} m"
        
        pl.add_text(
            f"TORÓ — {phase_text}\n"
            f"t = {t:+.2f}s{z_text}",
            position='upper_left', font_size=11, color='white'
        )
        
        pl.add_text(
            f"v_fall = {V_FALL:.0f} m/s  |  c = {C_SOUND:.0f} m/s",
            position='lower_left', font_size=8, color='lightgray'
        )
        
        pl.add_text(
            "Vale do Revólver — Presidente Getúlio, SC",
            position='lower_right', font_size=8, color='lightgray'
        )
        
        # Salvar
        frame_path = os.path.join(frame_dir, f'frame_{fi:04d}.png')
        pl.screenshot(frame_path)
        pl.close()
        
        if fi % 10 == 0:
            print(f"  Frame {fi:3d}/{n_frames} | t={t:+.2f}s | "
                  f"cam_z={pos[2]:.0f}m")
    
    # ============================================
    # COMPILAR VÍDEO
    # ============================================
    print("\n  Compilando vídeo...")
    try:
        import imageio.v3 as iio
        
        frames = []
        for i in range(n_frames):
            fp = os.path.join(frame_dir, f'frame_{i:04d}.png')
            if os.path.exists(fp):
                frames.append(iio.imread(fp))
        
        if frames:
            # MP4
            iio.imwrite(output_path, frames, fps=FPS,
                        codec='libx264',
                        plugin='pyav')
            print(f"  ✅ Vídeo: {output_path}")
            
            # GIF (reduzido para tamanho menor)
            gif_path = output_path.replace('.mp4', '.gif')
            # Usar cada 2 frames para GIF menor
            gif_frames = frames[::2]
            iio.imwrite(gif_path, gif_frames,
                        duration=int(1000 / (FPS/2)),
                        loop=0)
            print(f"  ✅ GIF: {gif_path}")
    except Exception as e:
        print(f"  [WARN] Erro codec, tentando ffmpeg direto...")
        try:
            import imageio
            writer = imageio.get_writer(output_path, fps=FPS)
            for i in range(n_frames):
                fp = os.path.join(frame_dir, f'frame_{i:04d}.png')
                if os.path.exists(fp):
                    writer.append_data(imageio.imread(fp))
            writer.close()
            print(f"  ✅ Vídeo: {output_path}")
        except Exception as e2:
            print(f"  [ERR] {e2}")
            print(f"  Frames PNG em: {frame_dir}/")
    
    print("  Renderização completa!")


if __name__ == '__main__':
    # Argumentos opcionais
    n_frames = N_FRAMES
    for i, arg in enumerate(sys.argv[1:]):
        if arg == '--frames' and i+2 < len(sys.argv):
            n_frames = int(sys.argv[i+2])
        elif arg == '--fps' and i+2 < len(sys.argv):
            FPS = int(sys.argv[i+2])
    
    print("\n" + "#" * 60)
    print("# TORÓ — Animação 3D Cinematográfica")
    print("# Colapso do Pistão + Ondas Acústicas")
    print("#" * 60)
    
    snapshots, grid_info = simulate_collapse(n_frames)
    render_animation(snapshots, grid_info)
    
    print("\n" + "=" * 60)
    print("  ANIMAÇÃO COMPLETA")
    print("  Vídeo: viz/toro_collapse_3d.mp4")
    print("=" * 60)
