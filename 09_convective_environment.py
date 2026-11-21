"""
Convective environment around the hour of peak rainfall

ERA5 at the grid point nearest each site, Day -1 and Day 0, hourly, aligned on the
HYSPLIT start hour (hour 0). Levels below the ERA5 surface are excluded.

Per event: 24 h CAPE maximum and its lead, strongest low-level convergence (lowest
200 hPa, hours -3 to +3), 200-400 hPa mean divergence at hour 0, 500 hPa omega, and
the IMERG rate at hour 0 (cell of the native maximum, 9 nearest cells). Spearman rho
with two-sided permutation P (100,000 permutations) between these quantities.
Peak-aligned medians (where at least six events have data): CAPE and 500 hPa omega
at the site, 500 hPa omega from 2 deg S to 2 deg N of the site, and divergence and
omega against pressure / surface pressure.

Inputs
    ERA5 GRIB files config.Event.era5_3d (u, v, w) and era5_2d (cape, sp);
    IMERG files in config.IMERG_DIR
Outputs (outputs/09_convective/)
    event_metrics.csv, spearman.csv, medians_site.csv, medians_omega500_latitude.csv,
    medians_divergence_sigma.csv, medians_omega_sigma.csv

Run
    python 09_convective_environment.py            (add --rebuild to reread the GRIB and IMERG files)
"""

import argparse
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from cloudbursts import config

warnings.filterwarnings("ignore")

EVENTS = {ev.number: ev for ev in config.EVENTS}
PEAK_HOUR = {n: ev.start_hour for n, ev in EVENTS.items()}
SITE = {n: ev.site for n, ev in EVENTS.items()}
TIER = {n: ev.tier for n, ev in EVENTS.items()}

OUT_DIR = config.INTERMEDIATE_DIR / "09_convective"
CACHE_FILE = OUT_DIR / "era5_cache.npz"
IMERG_CACHE = OUT_DIR / "imerg_hourly.json"

expected_levels = config.PRESSURE_LEVELS_HPA
expected_times = [h * 100 for h in range(24)]
LEV = np.array(expected_levels, float)
NT = 48                                   # Day -1 00 UTC ... Day 0 23 UTC
WINDOW = np.arange(-12, 4)                # hours -12 ... +3 from the rainfall peak
MIN_EVENTS = 6                            # medians only where n >= 6
LOW_DEPTH = 200.0                         # hPa: low-level layer = lowest 200 hPa above ground
UPPER = (200, 400)                        # hPa: upper-level layer (mean)
NEAR_PEAK = 3                             # h: window for the strongest low-level convergence
LAT_REL = np.round(np.arange(-2.0, 2.0001, 0.25), 2)  # deg from the site, along its longitude
SIGMA = np.round(np.arange(0.35, 1.0001, 0.01), 2)    # pressure / surface pressure
N_PERM = 100_000
CACHE_VARS = ("w", "div", "cape", "sp", "w500lat")
K_NEIGHBORS = config.K_NEIGHBORS


# GRIB reading (Day -1 and Day 0, plus surface pressure)
def read_event(key):
    import eccodes
    import metpy.calc as mpcalc
    from metpy.units import units

    ev = EVENTS[key]
    tlat, tlon = ev.target
    d0 = date.fromisoformat(ev.day0)
    day_index = {int((d0 - timedelta(days=1)).strftime("%Y%m%d")): 0, int(d0.strftime("%Y%m%d")): 1}

    # 3-D file: u, v (box), w (nearest point), and w at 500 hPa along the event
    # longitude, +-2 deg of latitude at 0.1 deg
    u_store = v_store = None
    w_arr = np.full((NT, LEV.size), np.nan)
    w500_lat = np.full((NT, LAT_REL.size), np.nan)
    with open(ev.era5_3d, "rb") as f:
        while True:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                break
            date_val = eccodes.codes_get(gid, "dataDate")
            if date_val in day_index:
                short = eccodes.codes_get(gid, "shortName")
                time_val = eccodes.codes_get(gid, "dataTime")
                level_val = eccodes.codes_get(gid, "level")
                if short in ("u", "v", "w") and time_val in expected_times and level_val in expected_levels:
                    if u_store is None:
                        Ni = eccodes.codes_get(gid, "Ni")
                        Nj = eccodes.codes_get(gid, "Nj")
                        lats_2d = eccodes.codes_get_array(gid, "latitudes").reshape(Nj, Ni)
                        lons_2d = eccodes.codes_get_array(gid, "longitudes").reshape(Nj, Ni)
                        lats_col, lons_row = lats_2d[:, 0], lons_2d[0, :]
                        rows = np.where((lats_col >= tlat - 1.5) & (lats_col <= tlat + 1.5))[0]
                        cols = np.where((lons_row >= tlon - 1.5) & (lons_row <= tlon + 1.5))[0]
                        rs, cs = slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)
                        dx, dy = mpcalc.lat_lon_grid_deltas(lons_2d[rs, cs], lats_2d[rs, cs])
                        ti = int(np.abs(lats_col[rs] - tlat).argmin())
                        tj = int(np.abs(lons_row[cs] - tlon).argmin())
                        lat_rows = [int(np.abs(lats_col - (tlat + d)).argmin()) for d in LAT_REL]
                        lon_col = int(np.abs(lons_row - tlon).argmin())
                        shape = (NT, LEV.size, rows.size, cols.size)
                        u_store, v_store = np.full(shape, np.nan), np.full(shape, np.nan)
                    full = eccodes.codes_get_array(gid, "values").reshape(Nj, Ni)
                    vals = full[rs, cs]
                    t_idx = 24 * day_index[date_val] + expected_times.index(time_val)
                    l_idx = expected_levels.index(level_val)
                    if short == "u":
                        u_store[t_idx, l_idx] = vals
                    elif short == "v":
                        v_store[t_idx, l_idx] = vals
                    else:
                        w_arr[t_idx, l_idx] = vals[ti, tj]
                        if level_val == 500:
                            w500_lat[t_idx] = full[lat_rows, lon_col]
            eccodes.codes_release(gid)
    if u_store is None:
        raise ValueError(f"No u/v data for {key}")

    # divergence on the 3 x 3 deg box
    div_arr = np.full((NT, LEV.size), np.nan)
    for t in range(NT):
        for lv in range(LEV.size):
            if np.isnan(u_store[t, lv]).all():
                continue
            div_2d = mpcalc.divergence(u_store[t, lv] * units("m/s"), v_store[t, lv] * units("m/s"),
                                       dx=dx, dy=dy) * 1e5
            div_arr[t, lv] = div_2d[ti, tj].magnitude

    # 2-D file: CAPE and surface pressure (nearest point)
    cape = np.full(NT, np.nan)
    sp = np.full(NT, np.nan)
    with open(ev.era5_2d, "rb") as f:
        while True:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                break
            date_val = eccodes.codes_get(gid, "dataDate")
            if date_val in day_index:
                short = eccodes.codes_get(gid, "shortName")
                time_val = eccodes.codes_get(gid, "dataTime")
                if short in ("cape", "sp") and time_val in expected_times:
                    value = eccodes.codes_grib_find_nearest(gid, tlat, tlon)[0]["value"]
                    t_idx = 24 * day_index[date_val] + expected_times.index(time_val)
                    if short == "cape":
                        cape[t_idx] = value
                    else:
                        sp[t_idx] = value / 100.0          # Pa -> hPa
            eccodes.codes_release(gid)
    if np.isnan(sp).all():
        print(f"  [!] {key}: no 'sp' in the 2-D GRIB; below-ground levels cannot be masked.")
    return {"w": w_arr, "div": div_arr, "cape": cape, "sp": sp, "w500lat": w500_lat}


def load_all(rebuild=False):
    """Per-event arrays, from the cache if it exists."""
    if CACHE_FILE.exists() and not rebuild:
        z = np.load(CACHE_FILE)
        if "w500lat_1" in z.files:
            return {n: {v: z[f"{v}_{n}"] for v in CACHE_VARS}
                    for n in sorted({int(k.rsplit("_", 1)[1]) for k in z.files})}
        print("Cache has no latitude-time omega; rereading the GRIB files.")
    env = {}
    for n in EVENTS:
        print(f"Reading ERA5, event {n} ...")
        env[n] = read_event(n)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_FILE, **{f"{v}_{n}": a[v] for n, a in env.items() for v in a})
    return env


# IMERG
_RE_TS = None


def parse_imerg_timestamp(filepath):
    import re
    global _RE_TS
    if _RE_TS is None:
        _RE_TS = re.compile(r"3IMERG\.(\d{8})-S(\d{6})-E")
    from datetime import datetime
    m = _RE_TS.search(Path(filepath).name)
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S") if m else None


def imerg_event(key):
    """Hourly rates of the peak cell (of the K nearest) for Day -1 and Day 0,
    and the peak itself, for the check against PEAK_HOUR."""
    from collections import defaultdict
    from datetime import datetime

    import h5py

    ev = EVENTS[key]
    tlat, tlon = ev.target
    folder = ev.imerg_dir
    files = sorted(folder.glob("3B-HHR*.HDF5")) or sorted(folder.glob("3B-HHR*.hdf5"))
    file_ts = sorted([(parse_imerg_timestamp(f), f) for f in files if parse_imerg_timestamp(f)],
                     key=lambda x: x[0])
    if not file_ts:
        raise FileNotFoundError(f"No IMERG files in {folder}")

    with h5py.File(file_ts[0][1], "r") as f:
        lats_g, lons_g = f["Grid/lat"][:], f["Grid/lon"][:]
    LONS, LATS = np.meshgrid(lons_g, lats_g, indexing="ij")
    dist = (LATS - tlat) ** 2 + (LONS - tlon) ** 2
    lon_idx, lat_idx = np.unravel_index(np.argsort(dist.ravel())[:K_NEIGHBORS], dist.shape)

    hourly_acc = defaultdict(lambda: [[] for _ in range(K_NEIGHBORS)])
    for ts, f in file_ts:
        h_ts = ts.replace(minute=0, second=0, microsecond=0)
        with h5py.File(f, "r") as hf:
            ds = hf["Grid/precipitation"]
            for i in range(K_NEIGHBORS):
                val = float(ds[0, lon_idx[i], lat_idx[i]])
                hourly_acc[h_ts][i].append(val if val >= 0 else np.nan)
    hours = sorted(hourly_acc)
    mat = np.array([[np.nanmean(hourly_acc[h][i]) for i in range(K_NEIGHBORS)] for h in hours])

    r, c = np.unravel_index(np.nanargmax(mat), mat.shape)
    d0 = datetime.fromisoformat(ev.day0)
    start = d0 - timedelta(days=1)
    series = [np.nan] * NT
    for h, v in zip(hours, mat[:, c]):
        k = int((h - start).total_seconds() // 3600)
        if 0 <= k < NT:
            series[k] = round(float(v), 3)
    peak = {"time": hours[r].strftime("%Y-%m-%d %H UTC"), "rate": round(float(mat[r, c]), 2),
            "lat": float(lats_g[lat_idx[c]]), "lon": float(lons_g[lon_idx[c]])}
    return series, peak


def imerg_hourly(rebuild=False):
    """{event_number: 48 hourly rates (mm h-1), Day -1 00 UTC ... Day 0 23 UTC} at the peak cell.
    Read once from the HDF5 files, then from IMERG_CACHE."""
    import json
    if IMERG_CACHE.exists() and not rebuild:
        return {int(k): v for k, v in json.loads(IMERG_CACHE.read_text()).items()}
    out = {}
    for n, ev in EVENTS.items():
        try:
            series, peak = imerg_event(n)
        except Exception as exc:
            print(f"  IMERG event {n} skipped: {exc}")
            continue
        out[n] = series
        expected = f"{ev.day0} {PEAK_HOUR[n]:02d} UTC"
        flag = "" if peak["time"] == expected else f"   [!] expected {expected}"
        print(f"  IMERG event {n}: peak {peak['rate']:.2f} mm/h at {peak['time']} "
              f"({peak['lat']:.2f} N, {peak['lon']:.2f} E){flag}")
    if out:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        IMERG_CACHE.write_text(json.dumps(out))
    return out or None


# Derived quantities
def masked(a, sp):
    """Set levels below the surface (p > surface pressure) to NaN."""
    out = a.copy()
    if not np.isnan(sp).all():
        out[LEV[None, :] > sp[:, None]] = np.nan
    return out


def to_sigma(a, sp):
    """Profiles [time, level] -> [time, SIGMA], sigma = p / surface pressure.
    Only above-ground levels are used; no extrapolation beyond them."""
    out = np.full((a.shape[0], SIGMA.size), np.nan)
    for t in range(a.shape[0]):
        ok = np.isfinite(a[t]) & (LEV <= sp[t])
        if ok.sum() < 2 or not np.isfinite(sp[t]):
            continue
        sig = LEV[ok] / sp[t]
        inside = (SIGMA >= sig.min()) & (SIGMA <= sig.max())
        out[t, inside] = np.interp(SIGMA[inside], sig, a[t, ok])
    return out


def align(series, n):
    """Values at WINDOW hours from the peak (index 24 + peak hour)."""
    idx = 24 + PEAK_HOUR[n] + WINDOW
    ok = (idx >= 0) & (idx < NT)
    out = np.full((WINDOW.size,) + np.shape(series)[1:], np.nan)
    out[ok] = np.asarray(series)[idx[ok]]
    return out


def median_iqr(stack):
    n = np.sum(np.isfinite(stack), axis=0)
    with np.errstate(all="ignore"):
        med = np.nanmedian(stack, axis=0)
        q1 = np.nanpercentile(stack, 25, axis=0)
        q3 = np.nanpercentile(stack, 75, axis=0)
    keep = n >= MIN_EVENTS
    return n, np.where(keep, med, np.nan), np.where(keep, q1, np.nan), np.where(keep, q3, np.nan)


def spearman_perm(x, y, n_perm=N_PERM, seed=0):
    """Spearman rho and two-sided permutation P, (1 + count) / (M + 1)."""
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    rho = np.corrcoef(rx, ry)[0, 1]
    rng = np.random.default_rng(seed)
    perm = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(n_perm)])
    return float(rho), float((np.sum(np.abs(perm) >= abs(rho) - 1e-12) + 1) / (n_perm + 1))


def event_metrics(env, imerg=None):
    rows = []
    up = (LEV >= UPPER[0]) & (LEV <= UPPER[1])
    for n, a in env.items():
        div = masked(a["div"], a["sp"])
        w = masked(a["w"], a["sp"])
        pk = 24 + PEAK_HOUR[n]
        hrs = np.arange(pk - NEAR_PEAK, min(pk + NEAR_PEAK, NT - 1) + 1)
        low = np.full(hrs.size, np.nan)
        for i, t in enumerate(hrs):
            spt = a["sp"][t] if np.isfinite(a["sp"][t]) else 1000.0
            layer = (LEV <= spt) & (LEV >= spt - LOW_DEPTH)
            low[i] = np.nanmin(div[t, layer]) if np.isfinite(div[t, layer]).any() else np.nan
        i_low = int(np.nanargmin(low))
        win = np.arange(pk - 23, pk + 1)            # the 24 h up to and including the peak
        i_cape = int(win[np.nanargmax(a["cape"][win])])
        sp_pk = a["sp"][pk]
        rows.append({
            "event": n, "site": SITE[n], "tier": TIER[n], "peak_hour_utc": PEAK_HOUR[n],
            "peak_rate_mm_h": round(float(imerg[n][pk]), 2) if imerg and n in imerg else np.nan,   # rate at hour 0
            "sp_peak_hPa": round(float(sp_pk), 1),
            "low_conv_1e5": round(float(low[i_low]), 2),
            "low_conv_hour_rel": int(hrs[i_low] - pk),
            "upper_div_peak_1e5": round(float(np.nanmean(div[pk, up])), 2),
            "omega500_peak": round(float(w[pk, LEV == 500][0]), 2),
            "omega500_min_pm3h": round(float(np.nanmin(w[hrs][:, LEV == 500])), 2),
            "cape_max_24h": round(float(a["cape"][i_cape]), 1),
            "cape_lead_h": int(pk - i_cape),
            "cape_at_peak": round(float(a["cape"][pk]), 1),
        })
    return pd.DataFrame(rows)


# Numbers and medians
def compute(rebuild=False):
    env = load_all(rebuild)
    imerg = imerg_hourly(rebuild)
    m = event_metrics(env, imerg)
    pd.set_option("display.width", 220)
    print("\n--- PER-EVENT VALUES (divergence x 1e-5 s-1, omega Pa s-1, CAPE J kg-1, rate mm h-1) ---")
    print(m.to_string(index=False))
    print(f"\nUpper-level divergence > 0 at the peak in {int((m.upper_div_peak_1e5 > 0).sum())} of {len(m)}; "
          f"median {m.upper_div_peak_1e5.median():.2f}")
    print(f"Low-level convergence: median {m.low_conv_1e5.median():.2f}, "
          f"range {m.low_conv_1e5.min():.2f} to {m.low_conv_1e5.max():.2f}")
    for t in (1, 2, 3):
        s = m[m.tier == t]
        print(f"  Tier {t} (events {list(s.event)}): low-level convergence {list(s.low_conv_1e5)}, "
              f"CAPE max {list(s.cape_max_24h)}")

    tests = []
    pairs = [("peak_rate_mm_h", "cape_max_24h"), ("peak_rate_mm_h", "low_conv_1e5"),
             ("peak_rate_mm_h", "upper_div_peak_1e5"), ("peak_rate_mm_h", "omega500_min_pm3h"),
             ("cape_max_24h", "low_conv_1e5"), ("low_conv_1e5", "omega500_min_pm3h")]
    print(f"\n--- SPEARMAN RANK CORRELATIONS (n = {len(m)}, two-sided permutation P, {N_PERM:,} permutations) ---")
    for a, b in pairs:
        if m[a].isna().any() or m[b].isna().any():
            continue
        rho, p = spearman_perm(m[a].values, m[b].values)
        tests.append({"x": a, "y": b, "rho": round(rho, 2), "P": round(p, 3)})
        print(f"  {a:>20} vs {b:<20} rho = {rho:+.2f}, P = {p:.3f}")

    w_stack = np.array([align(masked(a["w"], a["sp"]), n) for n, a in env.items()])
    cape_stack = np.array([align(a["cape"], n) for n, a in env.items()])
    _, cape_med, _, _ = median_iqr(cape_stack)
    _, w500_med, _, _ = median_iqr(w_stack[:, :, LEV == 500][..., 0])
    print("\n--- PEAK-ALIGNED MEDIANS AT THE SITE (n >= 6) ---")
    med = pd.DataFrame({"hour_rel": WINDOW, "cape": cape_med, "omega500_site": w500_med}).round(3)
    print(med.to_string(index=False))

    # 500 hPa omega along the site longitude, and the column against p / surface pressure
    _, lat_med, _, _ = median_iqr(np.array([align(a["w500lat"], n) for n, a in env.items()]))
    lat_tab = pd.DataFrame(lat_med, index=WINDOW, columns=LAT_REL).round(3)
    lat_tab.index.name = "hour_rel"
    _, d_med, _, _ = median_iqr(np.array([align(to_sigma(masked(a["div"], a["sp"]), a["sp"]), n)
                                          for n, a in env.items()]))
    _, w_med, _, _ = median_iqr(np.array([align(to_sigma(masked(a["w"], a["sp"]), a["sp"]), n)
                                          for n, a in env.items()]))
    d_tab = pd.DataFrame(d_med, index=WINDOW, columns=SIGMA).round(3)
    w_tab = pd.DataFrame(w_med, index=WINDOW, columns=SIGMA).round(3)
    d_tab.index.name = w_tab.index.name = "hour_rel"
    print("\nMedian 500 hPa omega by latitude from the site (deg), at hour 0:")
    print(lat_tab.loc[0].to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUT_DIR / "event_metrics.csv", index=False)
    pd.DataFrame(tests).to_csv(OUT_DIR / "spearman.csv", index=False)
    med.to_csv(OUT_DIR / "medians_site.csv", index=False)
    lat_tab.to_csv(OUT_DIR / "medians_omega500_latitude.csv")
    d_tab.to_csv(OUT_DIR / "medians_divergence_sigma.csv")
    w_tab.to_csv(OUT_DIR / "medians_omega_sigma.csv")
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convective environment around peak rainfall")
    parser.add_argument("--rebuild", action="store_true", help="Reread the ERA5 and IMERG files")
    compute(parser.parse_args().rebuild)
