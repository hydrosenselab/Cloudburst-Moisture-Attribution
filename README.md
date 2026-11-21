# Moisture sources of Indo-Himalayan cloudbursts

Code for the study *Land evaporation fuels Himalayan cloudbursts*. It traces the air that fed ten Indo-Himalayan cloudbursts (1998−2024) back for 120 h with HYSPLIT ensembles started at the hour of peak IMERG rainfall. Each moisture gain along the way is attributed to land evaporation, ocean evaporation, the free troposphere or moisture already present. The code also relates the land contribution to the time air spends over land and to antecedent soil moisture, and describes the convective environment around the peak.

## Repository contents

| File or folder | What it does |
|---|---|
| `01_imerg_native_peaks.py` | Finds the hour and cell of peak IMERG rainfall for each event |
| `02_imerg_regrid_peaks.py` | Remaps IMERG to the 0.25° ERA5 grid and gives the HYSPLIT start cell and hour |
| `03_moisture_sources.py` | Attributes the moisture of the HYSPLIT trajectories to land, ocean, free troposphere or pre-existing |
| `04_soil_moisture_anomaly.py` | ILDAS soil moisture anomalies |
| `05_mslp_anomaly.py` | ERA5 mean sea-level pressure anomalies |
| `06_ivt.py` | Integrated water vapour transport |
| `07_moisture_statistics.py` | Means, confidence intervals and comparisons of the moisture sources |
| `08_contact_time.py` | Land contribution against time over land and antecedent soil moisture |
| `09_convective_environment.py` | CAPE, convergence, divergence and vertical motion around the peak hour |
| `cloudbursts/` | Shared functions; `config.py` holds the event list and data paths |
| `data/` | Event table and moisture fractions used by `07_moisture_statistics.py` |

## How to run

Requires Python 3.10 or later and [HYSPLIT 5.4.2](https://www.ready.noaa.gov/HYSPLIT.php).

```bash
pip install -r requirements.txt
```

1. Set the data folders in `cloudbursts/config.py`.
2. Run `01` and `02`, then run HYSPLIT from the start cells and hours that `02` writes (27-member ensembles at 100, 300 and 500 m, 120 h backward, ERA5).
3. Run `03` to `09` in order. Run `04` with `--days-before 5` before `08`.

Results are written to `outputs/`. `07_moisture_statistics.py` runs straight away on the tables in `data/`.

## Data

- IMERG Final Run V07: [NASA GES DISC](https://doi.org/10.5067/GPM/IMERG/3B-HH/07)
- ERA5 hourly data on [pressure levels](https://doi.org/10.24381/cds.bd0915c6) and [single levels](https://doi.org/10.24381/cds.adbb2d47)
- ILDAS soil moisture: Magotra et al. (2024), *Journal of Hydrology*, 629, 130604
- District rainfall percentiles: Peringiyil & Saharia (2026), *International Journal of Climatology*

## Cite as

Kaushal, A., Saharia, M. & Rajagopalan, B. Land evaporation fuels Himalayan cloudbursts. In preparation for *Nature Geoscience*.

## HydroSense Lab, IIT Delhi

Code by Anandita Kaushal, Department of Civil Engineering, IIT Delhi. Released under the MIT licence.

<p>
  <img height="80" alt="HydroSense Lab logo" src="https://github.com/user-attachments/assets/3ff0f011-5f10-4823-b37a-b89f98cbac3c" />
  &nbsp;&nbsp;
  <img height="80" alt="IIT Delhi logo" src="https://github.com/user-attachments/assets/81192090-f330-49bf-989f-92120a348023" />
</p>

