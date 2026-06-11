import sys
import os

try:
    import vapor
    from vapor import session, renderer, dataset, camera
except ImportError:
    print("VAPOR module not found. Please run this script in the vapor_python conda environment.")
    sys.exit(1)

def create_animation():
    print("Iniciando sessão VAPOR...")
    ses = session.Session()
    
    nc_path = os.path.abspath("output/toro3d.nc")
    if not os.path.exists(nc_path):
        print(f"Erro: Arquivo não encontrado - {nc_path}")
        sys.exit(1)
        
    print(f"Carregando {nc_path}...")
    try:
        data = ses.OpenDataset(dataset.NETCDF, [nc_path])
    except AttributeError:
        # Tenta usar 'cf' como string se dataset.NETCDF não existir
        data = ses.OpenDataset("cf", [nc_path])
    
    print("Configurando renderizador de Volume para QG (Pistão/Granizo)...")
    vol_qg = data.NewRenderer(renderer.VolumeRenderer)
    vol_qg.SetVariableName("QG")
    
    print("Configurando renderizador de Isosurface para W (Updrafts/Downdrafts)...")
    iso_w = data.NewRenderer(renderer.IsosurfaceRenderer)
    iso_w.SetVariableName("W")
    try:
        iso_w.SetIsoValues([-10.0, 10.0, 30.0])
    except:
        pass
    
    print("Configurando câmera para focar no colapso do pistão...")
    cam = ses.GetCamera()
    
    # Domínio: x=0..10km, y=0..10km, z=0..15km
    # Vamos observar em perspectiva
    cam.SetPosition([5000, -8000, 6000]) 
    cam.SetTarget([5000, 5000, 4000])
    cam.SetUpVector([0, 0, 1])
    
    out_dir = "viz/vapor_anim_frames"
    os.makedirs(out_dir, exist_ok=True)
    
    num_ts = data.GetNumTimeSteps()
    print(f"Total de frames temporais no NetCDF: {num_ts}")
    
    # Vamos renderizar todos os passos
    for ts in range(num_ts):
        ses.SetTimeStep(ts)
        out_file = os.path.join(out_dir, f"frame_{ts:04d}.png")
        print(f"  -> Renderizando frame {ts}/{num_ts - 1}...")
        ses.Render(out_file)
        
    print(f"\n✅ Animação renderizada com sucesso em: {out_dir}")
    print("Você pode compilar as imagens em um vídeo usando ffmpeg ou ImageJ.")

if __name__ == "__main__":
    create_animation()
