"""
cloudbursts.anomaly

Climatological anomalies by calendar day (and hour).

    X'_y(d, h) = X_y(d, h) - mean over y' = 1991..2024 of X_y'(d, h)
"""

from __future__ import annotations

import numpy as np


def daily_anomaly(da, start: str, end: str, clim_start: int = 1991, clim_end: int = 2024):
    """
    Day-of-year anomaly of a daily DataArray (dim 'time') for start..end.

    The climatology uses only the calendar days covered by the event, which
    keeps memory use low on 34-year files.
    """
    target = da.sel(time=slice(start, end))
    if target.time.size == 0:
        raise ValueError(f"No data between {start} and {end}")
    doys = np.unique(target.time.dt.dayofyear.values)
    clim = da.sel(time=slice(str(clim_start), str(clim_end)))
    clim = clim.sel(time=clim.time.dt.dayofyear.isin(doys)).groupby("time.dayofyear").mean("time")
    return (target.groupby("time.dayofyear") - clim).drop_vars("dayofyear", errors="ignore")


def hourly_anomaly_daily_mean(da, start: str, end: str,
                              clim_start: int = 1991, clim_end: int = 2024):
    """
    Day-of-year-and-hour anomaly of an hourly DataArray, averaged to daily
    means (used for MSLP).
    """
    target = da.sel(time=slice(start, end))
    if target.time.size == 0:
        raise ValueError(f"No data between {start} and {end}")
    doys = np.unique(target.time.dt.dayofyear.values)
    clim = da.sel(time=slice(str(clim_start), str(clim_end)))
    clim = clim.sel(time=clim.time.dt.dayofyear.isin(doys))
    clim = clim.assign_coords(doy_hour=clim.time.dt.dayofyear * 100 + clim.time.dt.hour)
    clim = clim.groupby("doy_hour").mean("time")
    target = target.assign_coords(doy_hour=target.time.dt.dayofyear * 100 + target.time.dt.hour)
    anom = target.groupby("doy_hour") - clim
    return anom.drop_vars("doy_hour").resample(time="1D").mean("time")
