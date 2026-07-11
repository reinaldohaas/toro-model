"""
animate_collapse_3d.py — Animação 3D cinematográfica do Toró (v6).

Narrativa física em 7 fases, com ritmo CONSTANTE (frames alocados por
fase — sem compressão/aceleração no final):

    F0  Nuvem convectiva com topo IRIDESCENTE (glaciação explosiva)
    F1  INFLUXO DE INC (núcleos de gelo) espiralando para a zona H-M
    F2  VÓRTICE (tornado): updraft helicoidal + formação intensa de GRAUPEL
    F3  SEDIMENTAÇÃO: graupel migra para o núcleo do vórtice
    F4  QUEDA DO JATO de água+gelo DENTRO do centro do tornado (pistão v6)
    F5  IMPACTO: flash, splash e ondas acústicas
    F6  ONDAS + REVELAÇÃO DA RAVINA e das cicatrizes no solo

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
N_FRAMES = 420          # frames totais (~17.5 s @ 24 fps)
FPS = 24                # frames por segundo
RESOLUTION = (1920, 1080)

# Física (v6)
R_PISTON_FIS = 20.0     # m — raio físico do pistão/jato (v6, ~10 ton)
R_JET = 300.0           # m — raio VISUAL do jato (exagero p/ grade de 200 m)
R_VORTEX = 1000.0       # m — raio visual do vórtice/funil
V_FALL = 13.3           # m/s — velocidade terminal do pistão (v6)
C_SOUND = 340.0         # m/s — velocidade do som
Z_CLOUD_BASE = 4000.0   # m — base da nuvem (nível de congelamento)
Z_CLOUD_TOP = 12000.0   # m — topo
Z_LCL = 1000.0          # m — base do funil condensado
Z_HM_BOT, Z_HM_TOP = 4250.0, 4750.0   # zona Hallett-Mossop (influxo de INC)

# Ravina (pós-impacto) — escala real ~40 m de largura; visual exagerada
RAVINE_LEN = 3000.0     # m — comprimento visual do canal
RAVINE_WID = 250.0      # m — largura visual
RAVINE_DEPTH = 250.0    # m — profundidade visual

# ---------------------------------------------------------------
# FASES: (nome, fração dos frames, duração física representada [s])
# Ritmo constante: cada fase recebe um bloco fixo de frames.
# ---------------------------------------------------------------
PHASES = [
    ('GLACIACAO',    0.10),   # F0 topo iridescente
    ('INFLUXO_INC',  0.12),   # F1 núcleos de gelo entrando
    ('VORTICE',      0.18),   # F2 hélices + graupel
    ('SEDIMENTACAO', 0.12),   # F3 graupel converge ao núcleo
    ('QUEDA_JATO',   0.20),   # F4 pistão desce no olho do tornado
    ('IMPACTO',      0.08),   # F5 flash + splash
    ('RAVINA',       0.20),   # F6 ondas + cicatrizes
]
PHASE_EDGES = np.cumsum([0.0] + [f for _, f in PHASES])


def phase_of(frac):
    """Retorna (índice, nome, progresso local 0-1) da fase para frac global."""
    for i, (name, _) in enumerate(PHASES):
        if frac < PHASE_EDGES[i+1] or i == len(PHASES) - 1:
            p0, p1 = PHASE_EDGES[i], PHASE_EDGES[i+1]
            local = np.clip((frac - p0) / max(p1 - p0, 1e-9), 0.0, 1.0)
            return i, name, local
    return len(PHASES)-1, PHASES[-1][0], 1.0


def smooth_step(t, t0=0.0, t1=1.0):
    s = np.clip((t - t0) / (t1 - t0), 0, 1)
    return s * s * (3 - 2 * s)


def lerp(a, b, t):
    return a + (b - a) * np.clip(t, 0, 1)


# ================================================================
# CAMPOS SINTÉTICOS POR FASE
# ================================================================

def make_grid():
    nx, ny, nz = 50, 50, 80
    dx = dy = dz = 200.0
    x = np.linspace(0, (nx-1)*dx, nx)
    y = np.linspace(0, (ny-1)*dy, ny)
    z = np.linspace(0, (nz-1)*dz, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return dict(x=x, y=y, z=z, X=X, Y=Y, Z=Z,
                xc=x.mean(), yc=y.mean(), nx=nx, ny=ny, nz=nz, dx=dx)


def cloud_field(g, grow):
    """Nuvem cumulonimbus (torre + bigorna). grow: 0-1."""
    R = np.sqrt((g['X']-g['xc'])**2 + (g['Y']-g['yc'])**2)
    Z = g['Z']
    top = Z_CLOUD_BASE + (Z_CLOUD_TOP - Z_CLOUD_BASE) * (0.75 + 0.25*grow)
    # torre
    r_tower = 2500.0
    tower = np.exp(-(R/r_tower)**2) * ((Z > Z_LCL) & (Z < top))
    # bigorna
    anvil_z = top - 1500
    anvil = np.exp(-(R/5000.0)**2) * np.exp(-((Z-top+700)/900.0)**2) * (Z > anvil_z)
    return 5.0*(tower + 0.9*anvil)


def funnel_field(g, strength):
    """Funil condensado do vórtice (cone do LCL ao solo). strength 0-1."""
    if strength <= 0:
        return 0.0 * g['X']
    R = np.sqrt((g['X']-g['xc'])**2 + (g['Y']-g['yc'])**2)
    Z = g['Z']
    z_tip = Z_LCL * (1.0 - strength)          # ponta desce até o solo
    r_local = R_VORTEX * (0.25 + 0.75*np.clip((Z - z_tip)/(Z_CLOUD_BASE - z_tip + 1), 0, 1))
    f = np.exp(-(R/np.maximum(r_local, 50.0))**2) * ((Z > z_tip) & (Z < Z_CLOUD_BASE))
    return 6.0 * strength * f


def graupel_field(g, ring, core, z_load):
    """Graupel: anel no vórtice (ring 0-1) + núcleo carregado (core 0-1).

    z_load: altura do centro de massa da carga.
    """
    R = np.sqrt((g['X']-g['xc'])**2 + (g['Y']-g['yc'])**2)
    Z = g['Z']
    qg = np.zeros_like(R)
    if ring > 0:
        # anel helicoidal na zona H-M / acima (graupel crescendo no updraft)
        ring_r = R_VORTEX * 1.1
        rad = np.exp(-((R - ring_r)/500.0)**2)
        vert = np.exp(-((Z - 5200.0)/1500.0)**2)
        qg += 7.0 * ring * rad * vert
    if core > 0:
        # carga concentrada no núcleo (pré-colapso)
        rad = np.exp(-(R/(R_JET*1.6))**2)
        vert = np.exp(-((Z - z_load)/1200.0)**2)
        qg += 8.0 * core * rad * vert
    return qg


def jet_field(g, z_base, alive):
    """Jato de água+gelo caindo dentro do núcleo (pistão v6)."""
    if not alive:
        return 0.0 * g['X']
    R = np.sqrt((g['X']-g['xc'])**2 + (g['Y']-g['yc'])**2)
    Z = g['Z']
    z_top = min(Z_CLOUD_BASE + 800.0, z_base + 2500.0)
    rad = np.exp(-(R/R_JET)**2)
    col = ((Z >= z_base) & (Z <= z_top)).astype(float)
    tip = np.exp(-((Z - z_base)/250.0)**2) * (Z < z_base)
    zrel = np.clip((Z - z_base)/max(z_top - z_base, 1.0), 0, 1)
    return 9.0 * rad * (col*np.exp(-1.2*zrel) + tip)


def inc_particles(g, local, n=700, seed=7):
    """Pontos de INC espiralando para dentro da zona H-M. local 0-1."""
    rng = np.random.default_rng(seed)
    th0 = rng.uniform(0, 2*np.pi, n)
    r0 = rng.uniform(2500, 5200, n)
    z0 = rng.uniform(Z_HM_BOT - 800, Z_HM_TOP + 800, n)
    # espiral: raio encolhe, ângulo gira, z converge à zona H-M
    s = np.clip(local*1.15, 0, 1)
    r = r0*(1-s) + (R_VORTEX*1.05)*s
    th = th0 + 4.0*np.pi*s
    zz = z0*(1-s) + np.clip(z0, Z_HM_BOT, Z_HM_TOP)*s
    pts = np.column_stack([g['xc'] + r*np.cos(th),
                           g['yc'] + r*np.sin(th), zz])
    return pts


def helix_lines(g, local, n_helix=6, npts=160):
    """Linhas helicoidais do updraft no vórtice. Retorna lista de arrays."""
    lines = []
    z_top = Z_LCL + (7000.0 - Z_LCL)*np.clip(local*1.2, 0.05, 1.0)
    zs = np.linspace(50.0, z_top, npts)
    for k in range(n_helix):
        th = 2*np.pi*k/n_helix + 5.5*np.pi*(zs - zs[0])/(zs[-1]-zs[0]) + 2.0*np.pi*local
        rr = R_VORTEX*(0.45 + 0.55*(zs/z_top))
        lines.append(np.column_stack([g['xc']+rr*np.cos(th),
                                      g['yc']+rr*np.sin(th), zs]))
    return lines


def ravine_surface(g, reveal):
    """Superfície do solo com a ravina + cicatrizes. reveal 0-1."""
    n = 120
    L = g['x'].max() - g['x'].min()
    gx = np.linspace(g['x'].min(), g['x'].max(), n)
    gy = np.linspace(g['y'].min(), g['y'].max(), n)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    dxr, dyr = GX - g['xc'], GY - g['yc']
    # canal alinhado a y (direção do desfiladeiro)
    along = np.exp(-(dyr/(RAVINE_LEN*0.5))**2)
    across = np.exp(-(dxr/(RAVINE_WID*0.5))**2)
    depth = RAVINE_DEPTH * reveal * across * along
    # cicatrizes radiais (sulcos de erosão/queda de árvores)
    Rg = np.sqrt(dxr**2 + dyr**2)
    thg = np.arctan2(dyr, dxr)
    scars = 0.0*GX
    for a in np.linspace(0, 2*np.pi, 9, endpoint=False):
        scars += np.exp(-((np.mod(thg - a + np.pi, 2*np.pi) - np.pi)/0.06)**2)
    scars *= 35.0*reveal*np.exp(-(Rg/1800.0)**2)*(Rg > RAVINE_WID*0.6)
    GZ = -5.0 - depth - scars
    return GX, GY, GZ, depth + scars


# ================================================================
# SIMULAÇÃO FRAME A FRAME
# ================================================================

def simulate_collapse(n_frames=N_FRAMES):
    print("=" * 60)
    print("  STORYBOARD v6 — campos por frame")
    print("=" * 60)
    g = make_grid()

    snaps = {'frac': [], 'phase': [], 'local': [], 'label': [],
             't_real': [], 'qc': [], 'qg': [], 'funnel': [],
             'wave': [], 'z_jet': [], 'reveal': []}

    for fi in range(n_frames):
        frac = fi/max(n_frames-1, 1)
        pi, name, local = phase_of(frac)

        grow = 1.0 if pi > 0 else smooth_step(local)
        irid = smooth_step(local) if pi == 0 else 1.0
        funnel_s = 0.0
        ring = core = 0.0
        z_load = 5000.0
        z_jet = None
        wave = None
        reveal = 0.0
        t_real = 0.0

        if name == 'GLACIACAO':
            t_real = -900 + 300*local
            label = 'GLACIAÇÃO EXPLOSIVA — topo iridescente'
        elif name == 'INFLUXO_INC':
            t_real = -600 + 200*local
            funnel_s = 0.3*smooth_step(local)
            ring = 0.3*local
            label = 'INFLUXO DE INC — núcleos de gelo na zona H-M'
        elif name == 'VORTICE':
            t_real = -400 + 200*local
            funnel_s = 0.3 + 0.5*smooth_step(local)
            ring = 0.3 + 0.7*local
            core = 0.3*local
            label = 'VÓRTICE F2 — updraft helicoidal + graupel'
        elif name == 'SEDIMENTACAO':
            t_real = -200 + 150*local
            funnel_s = 0.8 + 0.2*local
            ring = 1.0 - 0.6*local
            core = 0.3 + 0.7*local
            z_load = 5000.0 - 1200.0*local
            label = 'SEDIMENTAÇÃO — carga converge ao núcleo'
        elif name == 'QUEDA_JATO':
            # queda REAL: 4000/13.3 ≈ 300 s → time-lapse uniforme (sem aceleração)
            t_real = -Z_CLOUD_BASE/V_FALL*(1.0 - local)
            funnel_s = 1.0
            core = 1.0 - 0.7*local
            z_load = 3800.0*(1.0 - local) + 200.0
            z_jet = Z_CLOUD_BASE*(1.0 - local)
            label = 'QUEDA DO JATO — água+gelo no olho do tornado'
        elif name == 'IMPACTO':
            t_real = 1.5*local
            funnel_s = 1.0 - 0.5*local
            z_jet = 0.0
            wave = t_real
            reveal = 0.4*smooth_step(local)
            label = '>>> IMPACTO <<<'
        else:  # RAVINA
            t_real = 1.5 + 8.0*local
            funnel_s = max(0.0, 0.5 - local)
            wave = t_real
            reveal = 0.4 + 0.6*smooth_step(local)
            label = 'RAVINA E CICATRIZES — ondas acústicas'

        qc = cloud_field(g, grow)
        qg = graupel_field(g, ring, core, z_load)
        if z_jet is not None and name == 'QUEDA_JATO':
            qg = np.maximum(qg, jet_field(g, z_jet, True))
        fn = funnel_field(g, funnel_s)

        wv = np.zeros_like(qc, dtype=np.float32)
        if wave is not None and wave > 0:
            R3 = np.sqrt((g['X']-g['xc'])**2 + (g['Y']-g['yc'])**2 + g['Z']**2)
            wr = C_SOUND*wave
            ww = 200.0 + 50.0*wave
            amp = np.where(R3 > 50, 3000.0/R3, 60.0)
            wv = (amp*np.exp(-((R3-wr)/ww)**2)).astype(np.float32)
            if wave > 0.5:
                wr2 = C_SOUND*(wave-0.5)
                wv += (0.3*amp*np.exp(-((R3-wr2)/(1.5*ww))**2)).astype(np.float32)

        snaps['frac'].append(frac)
        snaps['phase'].append(name)
        snaps['local'].append(local)
        snaps['label'].append(label)
        snaps['t_real'].append(t_real)
        snaps['qc'].append(qc.astype(np.float32))
        snaps['qg'].append(qg.astype(np.float32))
        snaps['funnel'].append(fn.astype(np.float32))
        snaps['wave'].append(wv)
        snaps['z_jet'].append(-1.0 if z_jet is None else z_jet)
        snaps['reveal'].append(reveal)

        if fi % 30 == 0:
            print(f"  {fi:4d}/{n_frames} {name:13s} local={local:.2f}")

    print(f"\n  ✓ {n_frames} frames preparados")
    return snaps, g


# ================================================================
# CÂMERA — keyframes suaves (sem aceleração no final)
# ================================================================
# keyframes em frac: (dist, cam_z, focal_z)
CAM_KEYS_FRAC = [0.00, 0.10, 0.22, 0.40, 0.52, 0.72, 0.80, 1.00]
CAM_DIST     = [14000, 11000, 7000, 5000, 4200, 3200, 4500, 9000]
CAM_Z        = [7000,  8000, 5200, 4200, 3000,  400,  900, 5000]
CAM_FOCAL_Z  = [8000,  9000, 4500, 3800, 2000,    0,    0, 800]


def get_camera(frac, xc, yc):
    dist = np.interp(frac, CAM_KEYS_FRAC, CAM_DIST)
    cam_z = np.interp(frac, CAM_KEYS_FRAC, CAM_Z)
    focal_z = np.interp(frac, CAM_KEYS_FRAC, CAM_FOCAL_Z)
    angle = 25 + 200*frac          # rotação lenta e CONSTANTE
    cx = xc + dist*np.cos(np.radians(angle))
    cy = yc + dist*np.sin(np.radians(angle))
    return (cx, cy, cam_z), (xc, yc, focal_z), (0, 0, 1)


# Cores iridescentes (madrepérola) para o topo glaciado
IRID_COLORS = ['lavender', 'lightcyan', 'mistyrose', 'palegreen',
               'lightyellow', 'thistle']


def render_animation(snaps, g, output_path='viz/toro_collapse_3d.mp4'):
    import pyvista as pv
    pv.OFF_SCREEN = True

    x, y, z = g['x'], g['y'], g['z']
    xc, yc = g['xc'], g['yc']
    n_frames = len(snaps['frac'])

    print("\n" + "=" * 60)
    print("  RENDERIZAÇÃO 3D — PyVista (v6, 7 fases)")
    print("=" * 60)

    frame_dir = 'viz/frames_collapse'
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def add_iso(pl, field, level, color, opacity):
        grid = pv.RectilinearGrid(x, y, z)
        grid.cell_data['F'] = field[:-1, :-1, :-1].ravel(order='F')
        iso = grid.cell_data_to_point_data()
        try:
            c = iso.contour([level], scalars='F')
            if c.n_points > 10:
                pl.add_mesh(c, color=color, opacity=opacity,
                            smooth_shading=True)
        except Exception:
            pass

    for fi in range(n_frames):
        frac = snaps['frac'][fi]
        name = snaps['phase'][fi]
        local = snaps['local'][fi]

        pl = pv.Plotter(off_screen=True, window_size=list(RESOLUTION))
        pl.set_background('black', top='midnightblue')

        # ---- Nuvem com topo iridescente ----
        qc = snaps['qc'][fi]
        add_iso(pl, qc, 1.2, 'slategray', 0.18)
        # camadas iridescentes no topo (cores giram lentamente)
        ci = int(frac*40) % len(IRID_COLORS)
        top_mask = (g['Z'] > Z_CLOUD_TOP - 3500).astype(np.float32)
        add_iso(pl, qc*top_mask, 2.5, IRID_COLORS[ci], 0.30)
        add_iso(pl, qc*top_mask, 3.5, IRID_COLORS[(ci+2) % len(IRID_COLORS)], 0.35)

        # ---- Funil do vórtice ----
        add_iso(pl, snaps['funnel'][fi], 1.5, 'gainsboro', 0.45)
        add_iso(pl, snaps['funnel'][fi], 3.5, 'lightsteelblue', 0.55)

        # ---- Graupel / jato ----
        qg = snaps['qg'][fi]
        add_iso(pl, qg, 1.5, 'mediumpurple', 0.35)
        add_iso(pl, qg, 4.0, 'darkviolet', 0.70)
        add_iso(pl, qg, 6.5, 'indigo', 0.90)

        # ---- INC (partículas) ----
        if name in ('INFLUXO_INC', 'VORTICE'):
            loc = local if name == 'INFLUXO_INC' else 1.0
            pts = inc_particles(g, loc)
            cloud_pts = pv.PolyData(pts)
            pl.add_mesh(cloud_pts, color='aquamarine', point_size=3,
                        render_points_as_spheres=True, opacity=0.8)

        # ---- Hélices do updraft ----
        if name in ('VORTICE', 'SEDIMENTACAO', 'QUEDA_JATO'):
            for line in helix_lines(g, max(local, 0.3)):
                try:
                    sp = pv.Spline(line, 200).tube(radius=35)
                    pl.add_mesh(sp, color='deepskyblue', opacity=0.5)
                except Exception:
                    pass

        # ---- Ondas acústicas ----
        wv = snaps['wave'][fi]
        if wv.max() > 0.5:
            for level, op, colr in [(1.5, 0.18, 'cyan'), (4.0, 0.30, 'deepskyblue'),
                                    (8.0, 0.45, 'royalblue')]:
                add_iso(pl, wv, level, colr, op)

        # ---- Solo / ravina ----
        reveal = snaps['reveal'][fi]
        if reveal > 0:
            GX, GY, GZ, _ = ravine_surface(g, reveal)
            surf = pv.StructuredGrid(GX, GY, GZ)
            pl.add_mesh(surf, color='saddlebrown', opacity=0.95,
                        smooth_shading=True)
            # água/lama no fundo do canal
            if reveal > 0.5:
                chan = pv.Plane(center=(xc, yc, -RAVINE_DEPTH*reveal*0.8),
                                direction=(0, 0, 1),
                                i_size=RAVINE_WID*0.9, j_size=RAVINE_LEN*0.9)
                pl.add_mesh(chan, color='steelblue', opacity=0.5)
        else:
            ground = pv.Plane(center=(xc, yc, -5), direction=(0, 0, 1),
                              i_size=(x.max()-x.min())*1.2,
                              j_size=(y.max()-y.min())*1.2,
                              i_resolution=30, j_resolution=30)
            pl.add_mesh(ground, color='forestgreen', opacity=0.7)

        # ---- Flash do impacto ----
        if name == 'IMPACTO' and local < 0.6:
            fr = 200 + 800*local
            sph = pv.Sphere(radius=fr, center=(xc, yc, fr*0.4))
            pl.add_mesh(sph, color='white', opacity=max(0.0, 0.6 - local))

        # ---- Câmera ----
        pos, focal, up = get_camera(frac, xc, yc)
        pl.camera_position = [pos, focal, up]

        # ---- HUD ----
        t_real = snaps['t_real'][fi]
        pl.add_text(f"TORÓ v6 — {snaps['label'][fi]}\n"
                    f"t = {t_real:+.1f} s (tempo físico)",
                    position='upper_left', font_size=11, color='white')
        pl.add_text(f"pistão: M≈10 ton, R={R_PISTON_FIS:.0f} m, "
                    f"v_queda={V_FALL:.1f} m/s (F2, 60 m/s)  |  c={C_SOUND:.0f} m/s",
                    position='lower_left', font_size=8, color='lightgray')
        pl.add_text("Modelo Toró v6 — Valada São Paulo, Planalto Mirador/SC",
                    position='lower_right', font_size=8, color='lightgray')

        pl.screenshot(os.path.join(frame_dir, f'frame_{fi:04d}.png'))
        pl.close()

        if fi % 20 == 0:
            print(f"  Frame {fi:4d}/{n_frames} | {name}")

    # ---- Compilar vídeo ----
    print("\n  Compilando vídeo...")
    try:
        import imageio.v3 as iio
        frames = []
        for i in range(n_frames):
            fp = os.path.join(frame_dir, f'frame_{i:04d}.png')
            if os.path.exists(fp):
                frames.append(iio.imread(fp))
        if frames:
            iio.imwrite(output_path, frames, fps=FPS, codec='libx264',
                        plugin='pyav')
            print(f"  ✅ Vídeo: {output_path}")
            gif_path = output_path.replace('.mp4', '.gif')
            iio.imwrite(gif_path, frames[::3],
                        duration=int(1000/(FPS/3)), loop=0)
            print(f"  ✅ GIF: {gif_path}")
    except Exception:
        print("  [WARN] Erro codec, tentando ffmpeg direto...")
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
    n_frames = N_FRAMES
    for i, arg in enumerate(sys.argv[1:]):
        if arg == '--frames' and i+2 < len(sys.argv):
            n_frames = int(sys.argv[i+2])
        elif arg == '--fps' and i+2 < len(sys.argv):
            FPS = int(sys.argv[i+2])

    print("\n" + "#" * 60)
    print("# TORÓ v6 — Animação 3D Cinematográfica (7 fases)")
    print("# Glaciação → INC → Vórtice → Sedimentação → Jato → Impacto → Ravina")
    print("#" * 60)

    snaps, g = simulate_collapse(n_frames)
    render_animation(snaps, g)

    print("\n" + "=" * 60)
    print("  ANIMAÇÃO COMPLETA")
    print("  Vídeo: viz/toro_collapse_3d.mp4")
    print("=" * 60)
