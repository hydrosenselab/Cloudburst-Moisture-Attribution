"""
cloudbursts.imerg

Readers for IMERG Final Run V07 half-hourly HDF5 files.

IMERG stores precipitation as ``Grid/precipitation[time, lon, lat]`` in mm/h.
Negative values are fill values and are converted to NaN here.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

_RE_TS = re.compile(r"3IMERG\.(\d{8})-S(\d{6})-E")
IST_OFFSET = timedelta(hours=5, minutes=30)


# File discovery
def parse_timestamp(filepath: str | Path) -> datetime | None:
    """Start time (UTC) of a half-hourly IMERG file, from its file name."""
    m = _RE_TS.search(Path(filepath).name)
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S") if m else None


def list_files(imerg_dir: str | Path, start: str | None = None, end: str | None = None):
    """
    Sorted (timestamp, path) pairs for all IMERG HDF5 files in a folder.

    If ``start`` / ``end`` (YYYY-MM-DD, inclusive, UTC) are given, files
    outside the window are dropped, so the folder may hold extra days.
    """
    files = sorted(Path(imerg_dir).glob("3B-HHR*.HDF5")) or sorted(
        Path(imerg_dir).glob("3B-HHR*.hdf5"))
    pairs = [(parse_timestamp(f), f) for f in files]
    pairs = sorted([(t, f) for t, f in pairs if t is not None], key=lambda x: x[0])
    if start:
        t0 = datetime.fromisoformat(start)
        pairs = [(t, f) for t, f in pairs if t >= t0]
    if end:
        t1 = datetime.fromisoformat(end) + timedelta(days=1)
        pairs = [(t, f) for t, f in pairs if t < t1]
    return pairs


# Grid helpers
def read_grid(filepath: str | Path):
    """Return the 1-D latitude and longitude arrays of an IMERG file."""
    import h5py

    with h5py.File(filepath, "r") as f:
        return f["Grid/lat"][:], f["Grid/lon"][:]


def k_nearest_cells(lats, lons, target_lat: float, target_lon: float, k: int):
    """
    The k native cells nearest to a target, ranked by Euclidean distance in
    latitude-longitude space.

    Returns a list of dicts with ``lat``, ``lon``, ``lat_i``, ``lon_i``
    (plain Python types, so the result is JSON-serialisable).
    """
    lon2d, lat2d = np.meshgrid(lons, lats, indexing="ij")
    dist = (lat2d - target_lat) ** 2 + (lon2d - target_lon) ** 2
    flat = np.argsort(dist.ravel())[:k]
    lon_i, lat_i = np.unravel_index(flat, dist.shape)
    return [{"lat": float(lats[a]), "lon": float(lons[b]),
             "lat_i": int(a), "lon_i": int(b)} for a, b in zip(lat_i, lon_i)]


def read_region(filepath, li, hi, lj, hj):
    """Precipitation sub-array [lat, lon] for index ranges; fill values -> NaN."""
    import h5py

    with h5py.File(filepath, "r") as f:
        ds = f["Grid/precipitation"]
        sub = ds[0, lj:hj, li:hi].T if ds.shape[1] >= ds.shape[2] else ds[0, li:hi, lj:hj]
    sub = sub.astype(np.float32, copy=False)
    sub[sub < 0] = np.nan
    return sub


def domain_indices(lats, lons, lat_min, lat_max, lon_min, lon_max):
    """Sub-grid coordinates and index bounds for a lat-lon box."""
    li = np.where((lats >= lat_min) & (lats <= lat_max))[0]
    lj = np.where((lons >= lon_min) & (lons <= lon_max))[0]
    return lats[li], lons[lj], int(li[0]), int(li[-1]) + 1, int(lj[0]), int(lj[-1]) + 1


# Time aggregation
def hourly_means(values_by_time: dict[datetime, np.ndarray]):
    """
    Hourly rate as the mean of the two half-hourly rates in each hour.

    ``values_by_time`` maps half-hourly timestamps to arrays of equal shape.
    Returns (sorted hour stamps, stacked array [hour, ...]).
    """
    buckets: dict[datetime, list[np.ndarray]] = {}
    for t, arr in values_by_time.items():
        buckets.setdefault(t.replace(minute=0, second=0, microsecond=0), []).append(arr)
    hours = sorted(buckets)
    return hours, np.stack([np.nanmean(np.stack(buckets[h]), axis=0) for h in hours])


def daily_totals(hours, hourly, offset: timedelta = timedelta(0)):
    """Sum hourly rates (mm/h) into daily totals (mm) for a time-zone offset."""
    out: dict = {}
    for h, row in zip(hours, hourly):
        d = (h + offset).date()
        out[d] = out.get(d, 0) + np.nan_to_num(row)
    return out
