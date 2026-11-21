"""
ERA5 mean-sea-level-pressure anomalies

Hourly MSLP anomalies (hPa) relative to the 1991-2024 climatology for the
same calendar day AND hour, averaged to daily means, Day -2 to Day 0.

Inputs
    config.MSLP_DIR/msl_pressure_<year>.grib (1991-2020) and
    msl_pressure_2021to2024.grib

Outputs
    outputs/05_mslp_anomalies/<event>.nc   variable msl_anom [time, latitude, longitude]
"""

import warnings

import pandas as pd

from cloudbursts import anomaly, config
from cloudbursts.cli import base_parser, selected_events

warnings.filterwarnings("ignore")
OUT_DIR = config.INTERMEDIATE_DIR / "05_mslp_anomalies"


def crop(ds):
    """Sort coordinates ascending and crop at load time to save memory."""
    lon0, lon1, lat0, lat1 = config.MAP_EXTENT
    ds = ds.sortby("latitude").sortby("longitude")
    return ds.sel(latitude=slice(lat0, lat1), longitude=slice(lon0, lon1))


def main():
    args = base_parser("ERA5 MSLP anomalies").parse_args()
    import xarray as xr

    files = [config.MSLP_DIR / f"msl_pressure_{y}.grib" for y in range(1991, 2021)]
    files.append(config.MSLP_DIR / "msl_pressure_2021to2024.grib")
    missing = [f for f in files if not f.exists()]
    for f in missing:
        print(f"Warning: missing {f}")
    files = [str(f) for f in files if f.exists()]
    if not files:
        raise FileNotFoundError(f"No MSLP GRIB files in {config.MSLP_DIR}")

    print("Loading 34-year ERA5 MSLP (cropped) ...")
    ds = xr.open_mfdataset(files, combine="nested", concat_dim="time", preprocess=crop,
                           parallel=False, chunks={"time": 100}, engine="cfgrib")
    msl = ds.drop_duplicates(dim="time").sortby("time")["msl"] / 100.0   # Pa -> hPa

    config.ensure_dir(OUT_DIR)
    for ev in selected_events(args):
        d0 = pd.Timestamp(ev.day0)
        start = (d0 - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        print(f"Event {ev.number}: {ev.key} ({start} to {ev.day0})")
        anom = anomaly.hourly_anomaly_daily_mean(msl, start, ev.day0,
                                                 config.CLIM_START, config.CLIM_END).compute()
        print(f"  range {float(anom.min()):+.2f} to {float(anom.max()):+.2f} hPa")
        anom.rename("msl_anom").assign_attrs(units="hPa").to_netcdf(OUT_DIR / f"{ev.key}.nc")
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
