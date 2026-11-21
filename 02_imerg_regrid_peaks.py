"""
Conservative regridding and HYSPLIT start cells

Regrid every half-hourly IMERG field to the ERA5-aligned 0.25 deg grid
(first-order conservative remapping), check conservation, and define the
HYSPLIT start point:

    start hour = native 0.1 deg peak hour         (script 01)
    start cell = 0.25 deg cell CONTAINING the native peak cell

The regridded rates of the 0.25 deg cells that contain the nine native
cells are also ranked, giving the peak intensity at ERA5 scale.

Inputs
    IMERG files (as script 01) and outputs/01_native_peaks.json

Outputs
    outputs/02_hysplit_start.csv   one row per event: Day 0, start hour,
                                   start cell, native and regridded rates,
                                   conservation errors (full / boundary-excluded)
    outputs/02_regrid_top10.json   top-10 regridded hours per event
"""

import csv
import json
from datetime import datetime

import numpy as np

from cloudbursts import config, imerg, regrid

IN_FILE = config.INTERMEDIATE_DIR / "01_native_peaks.json"
OUT_CSV = config.INTERMEDIATE_DIR / "02_hysplit_start.csv"
OUT_TOP = config.INTERMEDIATE_DIR / "02_regrid_top10.json"

LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = config.REGRID_DOMAIN
C_LATS = regrid.era5_axis(LAT_MIN, LAT_MAX)
C_LONS = regrid.era5_axis(LON_MIN, LON_MAX)


def process_event(ev, native):
    pairs = imerg.list_files(ev.imerg_dir, ev.imerg_start, ev.imerg_end)
    if not pairs:
        print("  No IMERG files; skipped")
        return None, None
    times, files = zip(*pairs)

    # weights (computed once per event)
    lats, lons = imerg.read_grid(files[0])
    f_lats, f_lons, li, hi, lj, hj = imerg.domain_indices(lats, lons, *config.REGRID_DOMAIN)
    print("  Building conservative-remapping weights ...")
    cells = regrid.build_weights(f_lats, f_lons, C_LATS, C_LONS)

    # conservation check on the first time step
    fine0 = imerg.read_region(files[0], li, hi, lj, hj)
    coarse0 = regrid.apply_weights(fine0, cells, len(C_LATS), len(C_LONS))
    err_full = regrid.conservation_error(fine0, f_lats, f_lons, coarse0, C_LATS, C_LONS, False)
    err_inner = regrid.conservation_error(fine0, f_lats, f_lons, coarse0, C_LATS, C_LONS, True)
    print(f"  Conservation error: {err_full:.3f} % (full), {err_inner:.3f} % (boundary excluded)")

    # 0.25 deg cells containing the native cells
    coarse_cells = {regrid.containing_cell(c["lat"], c["lon"], C_LATS, C_LONS)
                    for c in native["cells"]}
    idx_i = np.array([c[0] for c in coarse_cells])
    idx_j = np.array([c[1] for c in coarse_cells])

    # regrid only those cells for every time step
    wanted = set(zip(idx_i.tolist(), idx_j.tolist()))
    sub_cells = [c for c in cells if (c[0], c[1]) in wanted]
    values = {}
    for t, f in zip(times, files):
        fine = imerg.read_region(f, li, hi, lj, hj)
        out = regrid.apply_weights(fine, sub_cells, len(C_LATS), len(C_LONS))
        values[t] = out[idx_i, idx_j]
    hours, hourly = imerg.hourly_means(values)

    order = [k for k in np.argsort(hourly, axis=None)[::-1] if np.isfinite(hourly.ravel()[k])]
    cell_list = list(coarse_cells)
    top10 = []
    for rank, k in enumerate(order[:10], 1):
        h, c = np.unravel_index(k, hourly.shape)
        top10.append({"rank": rank, "time_utc": hours[h].strftime("%Y-%m-%d %H:00"),
                      "rate_mm_h": round(float(hourly[h, c]), 2),
                      "lat": cell_list[c][2], "lon": cell_list[c][3]})
        print(f"  {rank:>2}. {top10[-1]['time_utc']} UTC  {top10[-1]['rate_mm_h']:>6.2f} mm/h  "
              f"({cell_list[c][2]:.2f} N, {cell_list[c][3]:.2f} E)")

    # HYSPLIT start point from the native peak
    peak = native["peak"]
    si, sj, s_lat, s_lon = regrid.containing_cell(peak["lat"], peak["lon"], C_LATS, C_LONS)
    peak_time = datetime.strptime(peak["time_utc"], "%Y-%m-%d %H:00")
    c_idx = cell_list.index((si, sj, s_lat, s_lon))
    rate_at_start = float(hourly[hours.index(peak_time), c_idx])
    same = (top10[0]["lat"], top10[0]["lon"]) == (s_lat, s_lon)
    print(f"  HYSPLIT start: {peak['time_utc']} UTC at {s_lat:.2f} N, {s_lon:.2f} E "
          f"({'same as' if same else 'DIFFERENT from'} the regridded peak cell)")

    row = {"event": ev.number, "key": ev.key, "day0_utc": peak["time_utc"][:10],
           "start_hour_utc": peak["time_utc"][11:13],
           "native_peak_rate_mm_h": peak["rate_mm_h"],
           "native_peak_lat": peak["lat"], "native_peak_lon": peak["lon"],
           "start_cell_lat": s_lat, "start_cell_lon": s_lon,
           "regridded_rate_at_start_mm_h": round(rate_at_start, 2),
           "regridded_peak_lat": top10[0]["lat"], "regridded_peak_lon": top10[0]["lon"],
           "regridded_peak_time_utc": top10[0]["time_utc"],
           "regridded_peak_rate_mm_h": top10[0]["rate_mm_h"],
           "start_equals_regridded_peak": same,
           "conservation_err_full_pct": round(err_full, 3),
           "conservation_err_inner_pct": round(err_inner, 3)}
    return row, top10


def main():
    native_all = json.loads(IN_FILE.read_text())
    rows, tops = [], {}
    for ev in config.EVENTS:
        print(f"\nEvent {ev.number}: {ev.key}")
        if ev.key not in native_all:
            print("  Not in 01_native_peaks.json; run script 01 first")
            continue
        row, top = process_event(ev, native_all[ev.key])
        if row:
            rows.append(row)
            tops[ev.key] = top
    if rows:
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        OUT_TOP.write_text(json.dumps(tops, indent=2))
        print(f"\nSaved {OUT_CSV} and {OUT_TOP}")


if __name__ == "__main__":
    main()
