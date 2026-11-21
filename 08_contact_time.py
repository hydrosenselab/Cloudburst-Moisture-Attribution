"""
Time spent over land versus antecedent soil wetness

Per event and start height (27 members):
  land_pct   land BL share of arriving moisture
  bl_land_h  mean hours each member spends in the BL over land
  sm_w       sum(f_k * theta'_k) / sum(f_k) over the land BL uptakes of all members,
             f_k = uptake's share of its member's arriving humidity,
             theta'_k = ILDAS 0-10 cm anomaly at the uptake on its UTC day
  wet_pct    share of land uptake over wetter-than-normal soil
  land_rate  land_pct / bl_land_h (percentage points per hour)

At each height: Spearman rho with two-sided permutation P (100,000 permutations),
the leave-one-out range of rho, and partial rank correlations.

Inputs
    tdump files and land-sea mask (config); outputs/04_sm_anomalies/<event>.nc from
    04_soil_moisture_anomaly.py

Outputs
    outputs/08_contact_time.csv, outputs/08_contact_time_statistics.csv
"""

import numpy as np
import pandas as pd
import xarray as xr

from cloudbursts import config, hysplit

SM_DIR = config.INTERMEDIATE_DIR / "04_sm_anomalies"
TABLE_CSV = config.INTERMEDIATE_DIR / "08_contact_time.csv"
STATS_CSV = config.INTERMEDIATE_DIR / "08_contact_time_statistics.csv"
N_PERM = 100_000


def sm_lookup(path):
    """Function (lat, lon, day) -> ILDAS anomaly at the nearest cell, NaN if the day is missing."""
    anom = xr.open_dataarray(path).transpose("time", "lat", "lon")
    days = pd.to_datetime(anom.time.values).normalize()
    vals, lats, lons = anom.values, anom.lat.values, anom.lon.values

    def f(lat, lon, day):
        if day not in days:
            return np.nan
        return vals[days.get_loc(day), int(np.abs(lats - lat).argmin()), int(np.abs(lons - lon).argmin())]
    return f


def ranks(v):
    return pd.Series(np.asarray(v, float)).rank().to_numpy()


def spearman_perm(x, y, seed=0):
    """Spearman rho and two-sided permutation P, (1 + count) / (M + 1)."""
    rx, ry = ranks(x), ranks(y)
    rho = np.corrcoef(rx, ry)[0, 1]
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(N_PERM)])
    return rho, (1 + np.sum(np.abs(null) >= abs(rho) - 1e-12)) / (N_PERM + 1)


def partial_rank(y, x, z):
    """Spearman partial correlation of y and x with z held fixed (rank residuals)."""
    ry, rx, rz = ranks(y), ranks(x), ranks(z)
    res_y = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    res_x = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    return np.corrcoef(res_y, res_x)[0, 1]


def loo_range(x, y):
    r = [np.corrcoef(ranks(np.delete(x, i)), ranks(np.delete(y, i)))[0, 1] for i in range(len(x))]
    return min(r), max(r)


def main():
    is_land = hysplit.make_land_lookup(config.LSM_FILE, config.LAND_FRACTION_THRESHOLD)
    rows = []
    for ev in config.EVENTS:
        sm_at = sm_lookup(SM_DIR / f"{ev.key}.nc")
        arrival = pd.Timestamp(ev.day0) + pd.Timedelta(hours=ev.start_hour)
        for h in config.START_HEIGHTS_M:
            traj, diag, _, _ = hysplit.parse_tdump(ev.tdump(h))
            records, hours = hysplit.uptake_records(traj, diag, is_land)
            land = [r for r in records if r["category"] == "land"]
            n = len(hours)
            w, a = [], []
            for r in land:
                val = sm_at(r["lat"], r["lon"], (arrival + pd.Timedelta(hours=r["age_h"])).normalize())
                if np.isfinite(val):
                    w.append(r["frac"])
                    a.append(val)
            w, a = np.array(w), np.array(a)
            land_pct = 100 * sum(r["frac"] for r in land) / n
            bl_land_h = np.mean([m["bl_land_h"] for m in hours])
            rows.append({"event": ev.number, "site": ev.site, "height_m": h, "land_pct": land_pct,
                         "bl_land_h": bl_land_h,
                         "sm_w": np.sum(w * a) / np.sum(w) if w.size else np.nan,
                         "wet_pct": 100 * w[a > 0].sum() / w.sum() if w.size else np.nan,
                         "land_rate": land_pct / bl_land_h if bl_land_h > 0 else np.nan})
            r = rows[-1]
            print(f"Event {ev.number:>2} {h} m | land {r['land_pct']:5.1f}% | {r['bl_land_h']:5.1f} h over land "
                  f"| SM_w {r['sm_w']:+.3f} | rate {r['land_rate']:.2f} pp/h")
    table = pd.DataFrame(rows)

    tests = [("bl_land_h", "land_pct"), ("sm_w", "land_pct"), ("wet_pct", "land_pct"),
             ("sm_w", "bl_land_h"), ("sm_w", "land_rate")]
    stats = []
    for h in config.START_HEIGHTS_M:
        t = table[table.height_m == h]
        print(f"\n{h} m (n = {len(t)} events)")
        for xcol, ycol in tests:
            x, y = t[xcol].to_numpy(float), t[ycol].to_numpy(float)
            rho, p = spearman_perm(x, y)
            lo, hi = loo_range(x, y)
            stats.append({"height_m": h, "x": xcol, "y": ycol, "rho": rho, "p": p, "loo_min": lo, "loo_max": hi})
            print(f"  {ycol} vs {xcol}: rho = {rho:+.2f}, P = {p:.3f} (leave-one-out {lo:+.2f} to {hi:+.2f})")
        for x, z in (("bl_land_h", "sm_w"), ("sm_w", "bl_land_h")):
            rho = partial_rank(t["land_pct"], t[x], t[z])
            stats.append({"height_m": h, "x": f"{x} | {z}", "y": "land_pct", "rho": rho})
            print(f"  partial: land_pct vs {x}, {z} held fixed: rho = {rho:+.2f}")
        print(f"  land gain: median {t.land_rate.median():.2f} pp/h, range {t.land_rate.min():.2f}-"
              f"{t.land_rate.max():.2f}")

    config.ensure_dir(config.INTERMEDIATE_DIR)
    table.round(4).to_csv(TABLE_CSV, index=False)
    pd.DataFrame(stats).round(4).to_csv(STATS_CSV, index=False)
    print(f"\nSaved {TABLE_CSV} and {STATS_CSV}")


if __name__ == "__main__":
    main()
