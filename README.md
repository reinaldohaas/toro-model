# 🌩️ Modelo Toró — Precipitação Catastrófica por Glaciação Secundária Explosiva

**3D Anelastic θ_ρ Cloud Model with Secondary Ice Production**

> Haas, R. (2026) — *The Toroh Hypothesis: A Unified Framework Connecting a Novel Atmospheric Phenomenon, Global Dragon Mythology, and the Spatiotemporal Heterogeneity of the Great Unconformity*

## O Fenômeno

O **Toró** é um evento de precipitação catastrófica extremamente concentrada, associado a supercélulas com DSD (Drop Size Distribution) muito estreita em fase decadente. O mecanismo proposto envolve:

1. **Supercélula com DSD estreita** (μ=20) — gotas supercongeladas uniformes, iridescência
2. **Injeção de INPs por vórtice tornado** — tromba d'água transporta núcleos de gelo
3. **Glaciação secundária explosiva (SIP)** — Hallett-Mossop + quebra colisional (Phillips 2017)
4. **Pistão hidráulico** — massa de graupel/granizo forma corpo semi-rígido
5. **Auto-concentração por feedback de pressão** — queda de hidrometeoros → p'<0 acima → convergência
6. **Colapso coerente** — impacto tipo water hammer (Joukowsky)

## Modelo

### Formulação
- **3D anelástico** com temperatura potencial de densidade (θ_ρ)
- **Grade:** 20×20×50 (10km × 10km × 15km), dx=dy=500m, dz=300m
- **Fronteiras:** periódicas (x,y), rígida (z)
- **Sem Coriolis** (domínio pequeno, escala temporal curta)
- **Microfísica bulk:** 6 categorias (q_v, q_c, q_r, q_i, q_s, q_g) com SIP parametrizado

### Equações Governantes

```
∂u/∂t = -v⃗·∇u - (1/ρ̄)∂p'/∂x + D_u
∂v/∂t = -v⃗·∇v - (1/ρ̄)∂p'/∂y + D_v
∂w/∂t = -v⃗·∇w - (1/ρ̄)∂p'/∂z + B + D_w
∂θ_ρ/∂t = -v⃗·∇θ_ρ + S_latent + D_θ
∇·(ρ̄v⃗) = 0
```

onde:
- **θ_ρ = θ·(1 + R_v/R_d·q_v - q_l - q_i)** — temperatura potencial de densidade
- **B = g·(θ_ρ' / θ̄_ρ)** — flutuabilidade (inclui carregamento de hidrometeoros)
- **p'** — perturbação de pressão (resolvida por Poisson)

### Mecanismo-Chave: Feedback de Pressão

```
Sedimentação de graupel → B < 0 na coluna
    → Poisson: p' < 0 ACIMA da coluna
    → ∂p'/∂r < 0 → convergência horizontal
    → concentra mais hidrometeoros
    → PISTÃO SE AUTO-CONCENTRA (feedback positivo)
```

## Localização

**Vale do Revólver** — Presidente Getúlio, Santa Catarina, Brasil  
Coordenadas: 26.89°S, 49.37°W, elevação ~200m

## Instalação

```bash
git clone https://github.com/reinaldohaas/toro-model.git
cd toro-model
pip install numpy scipy netCDF4
```

## Uso

```bash
# Modelo 3D (padrão)
python run_simulation.py

# Modelo 1D (legado)
python run_simulation.py --1d
```

### Saída
- `output/results.json` — resultados para visualização web
- `output/toro3d.nc` — campos 3D em NetCDF
- `output/toro_sound.wav` — som sintético do "Tó"

### Visualização
```bash
cd viz
python -m http.server 8080
# Abrir http://localhost:8080
```

## Estrutura

```
toro-model/
├── core/
│   ├── config.py            # Configuração do cenário
│   ├── constants.py         # Constantes físicas
│   ├── grid3d.py            # Grade 3D anelástica
│   ├── theta_rho.py         # Termodinâmica θ_ρ
│   ├── dynamics3d.py        # Dinâmica 3D (advecção, Poisson, momento)
│   ├── microphysics_bulk.py # Microfísica bulk com SIP
│   ├── simulation3d.py      # Orquestrador 3D
│   ├── collapse.py          # Pistão hidráulico
│   ├── acoustics.py         # Som "Tó"
│   ├── seismic.py           # Sismograma
│   ├── erosion.py           # Erosão linear
│   ├── thermodynamics.py    # Termodinâmica (legado 1D)
│   ├── dynamics.py          # Dinâmica (legado 1D)
│   ├── microphysics.py      # Microfísica bin (legado 1D)
│   ├── simulation.py        # Orquestrador (legado 1D)
│   └── radar.py             # Refletividade / BWER
├── viz/
│   ├── index.html           # Visualização web
│   ├── style.css            # Tema dark premium
│   └── app.js               # Engine de gráficos
├── output/                  # Resultados (gerado)
├── run_simulation.py        # Script principal
├── requirements.txt         # Dependências
├── LICENSE                  # MIT
└── README.md
```

## Referências

- Hallett, J. & Mossop, S.C. (1974). Production of secondary ice particles during the riming process. *Nature*, 249, 26–28.
- Phillips, V.T.J. et al. (2017). Ice multiplication by breakup in ice–ice collisions. *JAS*, 74, 1789–1815.
- Klemp, J.B. & Wilhelmson, R.B. (1978). The simulation of three-dimensional convective storm dynamics. *JAS*, 35, 1070–1096.
- Fujita, T.T. (1985). *The Downburst*. University of Chicago Press.
- Joukowsky, N. (1898). Über den hydraulischen Stoss in Wasserleitungsröhren.
- Bolton, D. (1980). The computation of equivalent potential temperature. *MWR*, 108, 1046–1053.

## Autor

**Reinaldo Haas**  
Departamento de Física, Universidade Federal de Santa Catarina (UFSC)  
Florianópolis, Santa Catarina, Brasil  
reinaldo.haas@ufsc.br

## Licença

MIT License — ver [LICENSE](LICENSE)
