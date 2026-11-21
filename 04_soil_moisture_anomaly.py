"""
ILDAS soil moisture anomalies

Daily 0-10 cm volumetric soil moisture anomalies (m3/m3) relative to the
1991-2024 day-of-year climatology, for Day -5 to Day 0 of every event.

Unit conversion: theta = S / (rho_w * dz) with
rho_w = 1000 kg m-3 and dz = 0.1 m, i.e. theta = S / 100 for S in kg m-2.

Inputs
    ILDAS NetCDF files in config.ILDAS_DIR (sub-folders, plus
    SoilMoist_top_2020_2024.nc); optional config.ILDAS_MASK_FILE

Outputs
    outputs/04_sm_anomalies/<event>.nc     variable SoilMoist_anom [time, lat, lon]
"""

import glob
import os
import warnings

import pandas as pd

from cloudbursts import anomaly, config

warnings.filterwarnings("ignore")
OUT_DIR = config.INTERMEDIATE_DIR / "04_sm_anomalies"
DAYS_BEFORE = 5                       # Day -5 to Day 0, as far back as the trajectories
LAT_MIN_MAX = (config.MAP_EXTENT[2], config.MAP_EXTENT[3])
LON_MIN_MAX = (config.MAP_EXTENT[0], config.MAP_EXTENT[1])


def harmonise(ds):
    """One variable name, m3/m3 units, top layer only, cropped to the domain."""
    import xarray as xr

    if "SoilMoist_top" in ds.data_vars:
        ds = ds.rename_vars({"SoilMoist_top": "SoilMoist_tavg"})
    da = ds["SoilMoist_tavg"]
    attrs = da.attrs.copy()
    if "kg m-2" in attrs.get("units", "").lower() or "kg/m" in attrs.get("units", "").lower():
        da = da / 100.0                       # kg m-2 over 0.1 m -> m3 m-3
        attrs["units"] = "m3 m-3"
    da.attrs = attrs
    if "time" not in da.dims:
        da = da.expand_dims("time")
        if "time" in ds.coords:
            da = da.assign_coords(time=ds.time)
    out = xr.Dataset({"SoilMoist_tavg": da})
    if "SoilMoist_profiles" in out.dims:
        out = out.isel(SoilMoist_profiles=0).drop_vars("SoilMoist_profiles", errors="ignore")
    out["lat"] = out["lat"].astype("float32").round(2)
    out["lon"] = out["lon"].astype("float32").round(2)
    return out.sel(lat=slice(*LAT_MIN_MAX), lon=slice(*LON_MIN_MAX))


def load_ildas():
    import xarray as xr

    files = glob.glob(os.path.join(config.ILDAS_DIR, "*", "*.nc"))
    recent = config.ILDAS_DIR / "SoilMoist_top_2020_2024.nc"
    if recent.exists():
        files.append(str(recent))
    valid = []
    for f in files:                          # skip static / ancillary files
        try:
            with xr.open_dataset(f, engine="h5netcdf") as t:
                if "time" in t.coords:
                    valid.append(f)
        except Exception:
            continue
    if not valid:
        raise FileNotFoundError(f"No time-varying NetCDF files in {config.ILDAS_DIR}")
    print(f"Loading {len(valid)} ILDAS files (1991-2024) ...")
    ds = xr.open_mfdataset(valid, combine="nested", concat_dim="time", preprocess=harmonise,
                           parallel=False, chunks={"time": 100}, engine="h5netcdf")
    return ds.drop_duplicates(dim="time").sortby("time")["SoilMoist_tavg"]


def main():
    import xarray as xr

    sm = load_ildas()
    mask = None
    if config.ILDAS_MASK_FILE:
        mask = xr.open_dataarray(config.ILDAS_MASK_FILE).squeeze(drop=True)
    config.ensure_dir(OUT_DIR)
    for ev in config.EVENTS:
        d0 = pd.Timestamp(ev.day0)
        start, end = (d0 - pd.Timedelta(days=DAYS_BEFORE)).strftime("%Y-%m-%d"), ev.day0
        print(f"Event {ev.number}: {ev.key} ({start} to {end})")
        anom = anomaly.daily_anomaly(sm, start, end, config.CLIM_START, config.CLIM_END).compute()
        if mask is not None:
            anom = anom.where(mask.reindex_like(anom, method="nearest") > 0)
        print(f"  range {float(anom.min()):+.3f} to {float(anom.max()):+.3f} m3/m3")
        anom.rename("SoilMoist_anom").to_netcdf(OUT_DIR / f"{ev.key}.nc")
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
