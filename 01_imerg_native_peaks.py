"""
IMERG native-grid peaks

For each event, find the K_NEIGHBORS (= 9) native 0.1 deg IMERG cells
nearest the event coordinate, compute hourly rates (mean of the two
half-hourly rates) over the search window, and rank the top 10 hours.
The single highest hour is the PRECIPITATION PEAK that defines Day 0 and
the HYSPLIT start hour.

Inputs
    IMERG V07 half-hourly HDF5 files in config.IMERG_DIR/<event key>/

Outputs
    outputs/01_native_peaks.json   cells, top-10 hours, peak, daily totals
"""

import json
from collections import defaultdict

import numpy as np

from cloudbursts import config, imerg

OUT_FILE = config.INTERMEDIATE_DIR / "01_native_peaks.json"


def process_event(ev):
    pairs = imerg.list_files(ev.imerg_dir, ev.imerg_start, ev.imerg_end)
    if not pairs:
        print(f"  No IMERG files in {ev.imerg_dir}; skipped")
        return None
    times, files = zip(*pairs)

    # k nearest native cells
    lats, lons = imerg.read_grid(files[0])
    cells = imerg.k_nearest_cells(lats, lons, *ev.target, config.K_NEIGHBORS)

    # half-hourly values at those cells
    import h5py

    values = {}
    for t, f in zip(times, files):
        with h5py.File(f, "r") as hf:
            ds = hf["Grid/precipitation"]
            v = np.array([float(ds[0, c["lon_i"], c["lat_i"]]) for c in cells])
        values[t] = np.where(v >= 0, v, np.nan)

    # hourly rates and top 10
    hours, hourly = imerg.hourly_means(values)          # hourly[h, cell]
    flat = [i for i in np.argsort(hourly, axis=None)[::-1] if np.isfinite(hourly.ravel()[i])]
    top = []
    for rank, fi in enumerate(flat[:10], 1):
        h, c = np.unravel_index(fi, hourly.shape)
        top.append({"rank": rank, "time_utc": hours[h].strftime("%Y-%m-%d %H:00"),
                    "rate_mm_h": round(float(hourly[h, c]), 2), "cell_rank": int(c) + 1,
                    "lat": cells[c]["lat"], "lon": cells[c]["lon"]})
        print(f"  {rank:>2}. {top[-1]['time_utc']} UTC  {top[-1]['rate_mm_h']:>6.2f} mm/h  "
              f"(cell {c + 1}: {cells[c]['lat']:.2f} N, {cells[c]['lon']:.2f} E)")

    peak = top[0]
    if peak["time_utc"][:10] != ev.day0:
        print(f"  WARNING: peak day {peak['time_utc'][:10]} differs from config day0 {ev.day0}")

    # daily totals (UTC and IST) for every cell
    utc = imerg.daily_totals(hours, hourly)
    ist = imerg.daily_totals(hours, hourly, imerg.IST_OFFSET)
    daily = defaultdict(dict)
    for d in sorted(set(utc) | set(ist)):
        for c in range(len(cells)):
            daily[f"cell_{c + 1}"][d.isoformat()] = {
                "utc": round(float(utc.get(d, np.zeros(len(cells)))[c]), 2),
                "ist": round(float(ist.get(d, np.zeros(len(cells)))[c]), 2)}

    return {"target": ev.target, "window": [ev.imerg_start, ev.imerg_end],
            "n_hours": len(hours), "cells": cells, "top10": top, "peak": peak,
            "daily_totals": daily}


def main():
    config.ensure_dir(config.INTERMEDIATE_DIR)
    results = json.loads(OUT_FILE.read_text()) if OUT_FILE.exists() else {}
    for ev in config.EVENTS:
        print(f"\nEvent {ev.number}: {ev.key}  ({config.K_NEIGHBORS} nearest cells to "
              f"{ev.target[0]} N, {ev.target[1]} E)")
        res = process_event(ev)
        if res:
            results[ev.key] = res
    OUT_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUT_FILE}")


if __name__ == "__main__":
    main()
