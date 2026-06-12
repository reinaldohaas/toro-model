import cdsapi
import xarray as xr
import json
import os
import math

def fetch_era5(lat, lon, date_str, time_str, output_file):
    print(f"Buscando ERA5 via Copernicus CDS para {lat}, {lon}...")
    c = cdsapi.Client()

    year, month, day = date_str.split('-')
    time_str = time_str.zfill(5) # ensures 03:00 format
    
    nc_file = "data/era5_temp.nc"
    
    levels = ['1000', '975', '950', '925', '900', '850', '800', '700', '600', '500', '400', '300', '250', '200', '150', '100', '70', '50', '30']
    
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                'geopotential', 'relative_humidity', 'temperature',
                'u_component_of_wind', 'v_component_of_wind',
            ],
            'pressure_level': levels,
            'year': year,
            'month': month,
            'day': day,
            'time': time_str,
            # Bounding box [North, West, South, East] to reduce download size
            'area': [
                lat + 0.5, lon - 0.5,
                lat - 0.5, lon + 0.5,
            ],
        },
        nc_file)

    print("Download concluído. Processando...")
    
    # Abrir com xarray
    ds = xr.open_dataset(nc_file)
    
    # Pegar ponto mais próximo
    ds_point = ds.sel(latitude=lat, longitude=lon, method='nearest').squeeze()
    
    sounding = {}
    
    for lvl in levels:
        try:
            ds_lvl = ds_point.sel(pressure_level=int(lvl))
            
            # Geopotential is z * g. Geopotential height = z / g
            g = 9.80665
            z = float(ds_lvl['z'].values) / g
            T = float(ds_lvl['t'].values)
            RH = float(ds_lvl['r'].values)
            u = float(ds_lvl['u'].values)
            v = float(ds_lvl['v'].values)
            
            # Converter RH de porcentagem para 0-100 se necessário?
            # ERA5 RH is typically 0-100.
            
            sounding[lvl] = {
                "temperature": T - 273.15, # Convert back to C just to be safe, wait! My code in simulation3d expects Celsius! Let me just save it as C.
                "relative_humidity": RH,
                "geopotential_height": z,
                "u_component": u,
                "v_component": v
            }
        except Exception as e:
            print(f"Warning: erro no nivel {lvl}hPa: {e}")
            sounding[lvl] = {
                "temperature": None,
                "relative_humidity": None,
                "geopotential_height": None,
                "u_component": 0.0,
                "v_component": 0.0
            }

    ds.close()
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            "metadata": {
                "latitude": lat,
                "longitude": lon,
                "date": date_str,
                "time": time_str,
                "model": "ERA5 via Copernicus CDS"
            },
            "levels": sounding
        }, f, indent=4)
        
    print(f"Sucesso! Sondagem salva em {output_file}")
    
    # Cleanup temp file
    if os.path.exists(nc_file):
        os.remove(nc_file)

if __name__ == "__main__":
    fetch_era5(-27.0991, -49.6215, "2020-12-17", "03:00", "data/sounding_era5_pg.json")
