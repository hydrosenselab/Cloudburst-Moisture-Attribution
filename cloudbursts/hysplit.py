"""
cloudbursts.hysplit

HYSPLIT tdump parsing, Lagrangian moisture-source attribution and
 ensemble-mean trajectories.

Attribution (Methods, eqs 4-7), adapted from Sodemann et al. (2008):

1. Walk each trajectory forward in time, from origin (-120 h) to arrival.
2. dq > 0  -> uptake. Boundary layer (BL) if z <= mixed-layer depth, else
   free troposphere (FT). BL uptakes are further split into LAND or OCEAN
   with the ERA5 land-sea mask at the nearest 0.25 deg point.
3. dq < 0  -> precipitation loss, removed in proportion from every earlier
   uptake AND from the moisture present at the origin.
4. Each surviving contribution is divided by the arrival humidity, so
   land + ocean + FT + unattributed = 100 % for every trajectory.

Reported fractions (paper): F_L (land BL), F_N = ocean BL + FT (non-land),
F_0 (unattributed / pre-existing). The ocean and FT parts are also returned
separately.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

LandLookup = Callable[[float, float], bool]


# tdump parser
def parse_tdump(filepath: str | Path):
    """
    Parse a HYSPLIT trajectory endpoint (tdump) file.

    Returns
    trajectories : dict[int, dict]  keys 'age', 'lat', 'lon', 'height' (arrays)
                   and 'diag' (list of dicts of diagnostic variables per step)
    diag_names   : list[str]        e.g. ['PRESSURE', 'SPCHUMID', 'MIXDEPTH']
    n_traj       : int
    direction    : str              'BACKWARD' or 'FORWARD'
    """
    lines = Path(filepath).read_text().splitlines()
    idx = 0
    n_grids = int(lines[idx].split()[0])
    idx += 1 + n_grids
    header = lines[idx].split()
    n_traj, direction = int(header[0]), header[1]
    idx += 1 + n_traj
    parts = lines[idx].split()
    n_diag, diag_names = int(parts[0]), parts[1:]
    idx += 1
    while len(diag_names) < n_diag:          # long diagnostic lists wrap lines
        diag_names.extend(lines[idx].split())
        idx += 1

    n_fixed = 12
    n_total = n_fixed + n_diag
    trajectories: dict[int, dict] = {}
    buf: list[str] = []
    for line in lines[idx:]:
        if not line.strip():
            continue
        buf.extend(line.split())
        while len(buf) >= n_total:
            rec, buf = buf[:n_total], buf[n_total:]
            tid = int(rec[0])
            t = trajectories.setdefault(tid, {"age": [], "lat": [], "lon": [],
                                              "height": [], "diag": []})
            t["age"].append(float(rec[8]))
            t["lat"].append(float(rec[9]))
            t["lon"].append(float(rec[10]))
            t["height"].append(float(rec[11]))
            t["diag"].append({diag_names[i]: float(rec[n_fixed + i]) for i in range(n_diag)})
    for t in trajectories.values():
        for k in ("age", "lat", "lon", "height"):
            t[k] = np.asarray(t[k])
    return trajectories, diag_names, n_traj, direction


# Land-sea lookup
def make_land_lookup(lsm_file: str | Path, threshold: float = 0.5) -> LandLookup:
    """
    Build a cached function ``is_land(lat, lon)`` from an ERA5 ``lsm`` NetCDF.

    A point is land if the nearest ERA5 grid point has land fraction >= threshold.
    HYSPLIT longitudes (-180..180) are converted to ERA5's 0..360.
    """
    import xarray as xr

    lsm = xr.open_dataset(lsm_file)["lsm"].squeeze(drop=True)
    lon_name = "longitude" if "longitude" in lsm.dims else "lon"
    lat_name = "latitude" if "latitude" in lsm.dims else "lat"
    lon_360 = float(lsm[lon_name].max()) > 180

    @cache
    def is_land(lat: float, lon: float) -> bool:
        lon_q = lon % 360 if lon_360 else lon
        val = lsm.sel({lat_name: lat, lon_name: lon_q}, method="nearest").item()
        return val >= threshold

    return is_land


# Moisture attribution
@dataclass
class Attribution:
    """Ensemble-mean source fractions (%) and per-grid-cell source maps."""

    land: float          # F_L  terrestrial BL
    ocean: float         # oceanic BL
    ft: float            # free troposphere
    unattributed: float  # F_0  moisture present at the trajectory origin
    n_traj: int
    maps: dict[str, dict[tuple[float, float], float]]

    @property
    def nonland(self) -> float:
        """F_N = oceanic BL + FT (the 'non-land' fraction used in the paper)."""
        return self.ocean + self.ft

    @property
    def closure(self) -> float:
        """Should equal 100 %."""
        return self.land + self.ocean + self.ft + self.unattributed


def _grid_key(lat: float, lon: float) -> tuple[float, float]:
    """Round an uptake position to the nearest 0.25 deg point."""
    return round(lat * 4) / 4, round(lon * 4) / 4


def attribute_moisture(trajectories: dict, diag_names: list[str],
                       is_land: LandLookup | None = None) -> Attribution:
    """
    Lagrangian moisture-source attribution for an ensemble of trajectories.

    Parameters
    trajectories, diag_names : output of :func:`parse_tdump`. The tdump must
        contain SPCHUMID (g/kg) and MIXDEPTH (m AGL) diagnostics.
    is_land : callable(lat, lon) -> bool, optional. If None, every BL uptake
        is counted as land (reproduces the old, unmasked behaviour).
    """
    if "SPCHUMID" not in diag_names or "MIXDEPTH" not in diag_names:
        raise ValueError("tdump needs SPCHUMID and MIXDEPTH diagnostics")

    maps = {k: defaultdict(float) for k in ("land", "ocean", "ft")}
    pre_total = 0.0
    n = 0

    for data in trajectories.values():
        q = np.array([d["SPCHUMID"] for d in data["diag"]])
        mix = np.array([d["MIXDEPTH"] for d in data["diag"]])
        order = np.argsort(data["age"])          # oldest (-120 h) first
        q_arr = q[order[-1]]
        if q_arr <= 0:
            continue
        n += 1
        pre = q[order[0]]
        uptakes: list[list] = []                 # [amount, category, key]

        for a, b in zip(order[:-1], order[1:]):
            q_prev, q_curr = q[a], q[b]
            if q_prev <= 0 or q_curr <= 0:
                continue
            dq = q_curr - q_prev
            if dq > 0:
                key = _grid_key(data["lat"][b], data["lon"][b])
                if data["height"][b] <= mix[b]:
                    cat = "land" if (is_land is None or is_land(*key)) else "ocean"
                else:
                    cat = "ft"
                uptakes.append([dq, cat, key])
            elif dq < 0:
                keep = 1.0 - abs(dq) / q_prev
                pre *= keep
                for u in uptakes:
                    u[0] *= keep

        for amount, cat, key in uptakes:
            maps[cat][key] += amount / q_arr
        pre_total += pre / q_arr

    n_safe = max(n, 1)
    pct = {k: 100.0 * sum(v.values()) / n_safe for k, v in maps.items()}
    pct_maps = {k: {kk: 100.0 * vv / n_safe for kk, vv in v.items()} for k, v in maps.items()}
    return Attribution(pct["land"], pct["ocean"], pct["ft"], 100.0 * pre_total / n_safe,
                       n, pct_maps)


def uptake_records(trajectories: dict, diag_names: list[str],
                   is_land: LandLookup | None = None):
    """
    Same accounting as :func:`attribute_moisture`, but return every uptake.

    Returns
    records : list of dict, one per uptake, with keys
        member, age_h (negative, hours before arrival), lat, lon, category
        ('land' | 'ocean' | 'ft') and frac (effective contribution to arrival
        humidity, 0-1, after all later precipitation losses).
    hours : list of dict, one per member, with the number of hourly steps the
        parcel spent in the BL over land (bl_land_h), in the BL over ocean
        (bl_ocean_h) and above the BL (ft_h).

    Used by 08_contact_time.py to weight soil moisture anomalies by where the
    land moisture was taken up.
    """
    if "SPCHUMID" not in diag_names or "MIXDEPTH" not in diag_names:
        raise ValueError("tdump needs SPCHUMID and MIXDEPTH diagnostics")
    records, hours = [], []
    for tid, data in trajectories.items():
        q = np.array([d["SPCHUMID"] for d in data["diag"]])
        mix = np.array([d["MIXDEPTH"] for d in data["diag"]])
        order = np.argsort(data["age"])
        q_arr = q[order[-1]]
        if q_arr <= 0:
            continue
        counts = {"bl_land_h": 0, "bl_ocean_h": 0, "ft_h": 0}
        ups: list[dict] = []
        for a, b in zip(order[:-1], order[1:]):
            lat, lon = data["lat"][b], data["lon"][b]
            in_bl = data["height"][b] <= mix[b]
            land = is_land is None or is_land(*_grid_key(lat, lon))
            counts["bl_land_h" if in_bl and land else "bl_ocean_h" if in_bl else "ft_h"] += 1
            q_prev, q_curr = q[a], q[b]
            if q_prev <= 0 or q_curr <= 0:
                continue
            dq = q_curr - q_prev
            if dq > 0:
                cat = ("land" if land else "ocean") if in_bl else "ft"
                ups.append({"member": tid, "age_h": float(data["age"][b]), "lat": float(lat),
                            "lon": float(lon), "category": cat, "frac": dq / q_arr})
            elif dq < 0:
                keep = 1.0 - abs(dq) / q_prev
                for u in ups:
                    u["frac"] *= keep
        records.extend(ups)
        hours.append({"member": tid, **counts})
    return records, hours


# Ensemble-mean trajectory
def mean_trajectory(trajectories: dict):
    """Arithmetic mean latitude and longitude of all members at each hour."""
    by_age: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for t in trajectories.values():
        for age, lat, lon in zip(t["age"], t["lat"], t["lon"]):
            by_age[age].append((lat, lon))
    ages = sorted(by_age)
    lats = [float(np.mean([p[0] for p in by_age[a]])) for a in ages]
    lons = [float(np.mean([p[1] for p in by_age[a]])) for a in ages]
    return ages, lats, lons
