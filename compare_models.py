import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import get_default_config
from core.simulation import ToroSimulation
from core.simulation3d import ToroSimulation3D

def run_comparison():
    results = {}
    
    # 1. 1D Legacy
    print("\n--- Running 1D Legacy ---")
    config1d = get_default_config()
    # Let's reduce t_total if it takes too long, but we need full time to reach max
    sim1d = ToroSimulation(config1d)
    sim1d.run(output_dir='output', verbose=False)
    with open('output/results.json', 'r') as f:
        results['1D'] = json.load(f)
        
    # 2. 3D Idealized
    print("\n--- Running 3D Idealized ---")
    config3d_ideal = get_default_config()
    config3d_ideal.thermodynamics.sounding_file = None
    sim3d_ideal = ToroSimulation3D(config3d_ideal)
    sim3d_ideal.run()
    with open('output/results.json', 'r') as f:
        results['3D_Idealized'] = json.load(f)
        
    # 3. 3D ERA5
    print("\n--- Running 3D ERA5 ---")
    config3d_era5 = get_default_config()
    config3d_era5.thermodynamics.sounding_file = 'data/sounding_era5_pg.json'
    sim3d_era5 = ToroSimulation3D(config3d_era5)
    sim3d_era5.run()
    with open('output/results.json', 'r') as f:
        results['3D_ERA5'] = json.load(f)
        
    print("\n" + "="*80)
    print("COMPARAÇÃO DE MODELOS")
    print("="*80)
    print(f"{'Métrica':<20} | {'1D Legado':<18} | {'3D Idealizado':<18} | {'3D ERA5 (Real)':<18}")
    print("-" * 80)
    
    keys_to_compare = [
        ('Vel. Updraft (m/s)', 'phase1', 'w_max'),
        ('Massa Pistão (ton)', 'phase2', 'M_piston_ton'),
        ('Vel. Impacto (m/s)', 'phase2', 'v_impact'),
        ('Pressão Impacto (MPa)', 'phase2', 'P_impact_MPa'),
        ('Magnitude Sísmica', 'phase3', 'seismic', 'M_L'),
        ('SPL a 1km (dB)', 'phase3', 'sound', 'SPL_1km'),
        ('Barro (%)', 'phase3', 'erosion', 'mud_percent')
    ]
    
    def get_nested(d, *keys):
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return 'N/A'
        return d
    
    for item in keys_to_compare:
        label = item[0]
        keys = item[1:]
        val1d = get_nested(results['1D'], *keys)
        val3d_id = get_nested(results['3D_Idealized'], *keys)
        val3d_era = get_nested(results['3D_ERA5'], *keys)
        
        # formatação
        if isinstance(val1d, float): val1d = f"{val1d:.2f}"
        if isinstance(val3d_id, float): val3d_id = f"{val3d_id:.2f}"
        if isinstance(val3d_era, float): val3d_era = f"{val3d_era:.2f}"
        
        print(f"{label:<20} | {val1d:<18} | {val3d_id:<18} | {val3d_era:<18}")

if __name__ == '__main__':
    run_comparison()
