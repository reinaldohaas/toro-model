"""
run_simulation.py — Script principal para executar o modelo Toró 3D.

Uso:
    python run_simulation.py          # Modelo 3D (padrão)
    python run_simulation.py --1d     # Modelo 1D (legado)

Saída:
    output/results.json   — resultados para visualização
    output/toro3d.nc      — campos 3D em NetCDF
    output/toro_sound.wav — som sintético do "Tó"
"""

import sys
import os

# Adicionar diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import get_default_config


def main():
    """Executa simulação completa do Toró."""
    config = get_default_config()
    
    # Check for custom sounding
    for i, arg in enumerate(sys.argv):
        if arg == '--sounding' and i + 1 < len(sys.argv):
            config.thermodynamics.sounding_file = sys.argv[i+1]
    
    if '--1d' in sys.argv:
        # Modelo 1D legado
        from core.simulation import ToroSimulation
        print("=" * 60)
        print("  MODELO TORÓ v1.0 (1D legado)")
        print("=" * 60)
        sim = ToroSimulation(config)
        sim.run(output_dir='output', verbose=True)
    else:
        # Modelo 3D com θ_ρ (padrão)
        from core.simulation3d import ToroSimulation3D
        sim = ToroSimulation3D(config)
        sim.run()
    
    print("\n" + "=" * 60)
    print("  SIMULAÇÃO COMPLETA")
    print("  Resultados em: output/results.json")
    print("  NetCDF em: output/toro3d.nc")
    print("  Som em: output/toro_sound.wav")
    print("=" * 60)


if __name__ == '__main__':
    main()
