"""
HYSPLIT moisture-source attribution

For every event and start height (100, 300, 500 m AGL):
  1. read the 27-member HYSPLIT ensemble (tdump),
  2. attribute arrival moisture to land BL, ocean BL, free troposphere
     and unattributed (pre-existing) moisture,
  3. compute and save the ensemble-mean trajectory.

HYSPLIT itself is run outside this repo (HYSPLIT 5.4.2, ERA5 ARL, 120 h
backward, meteorological-ensemble mode, vertical motion option 0), from
the start cell and hour in outputs/02_hysplit_start.csv.

Inputs
    tdump files       config.Event.tdump(height)
    land-sea mask     config.LSM_FILE (ERA5 `lsm`)

Outputs
    outputs/03_moisture_fractions.csv         tidy table, one row per event x height
    outputs/03_mean_trajectories/Mean_Trajectory_<event>_<h>m.csv
"""

import csv

from cloudbursts import config, hysplit

OUT_CSV = config.INTERMEDIATE_DIR / "03_moisture_fractions.csv"
TRAJ_DIR = config.INTERMEDIATE_DIR / "03_mean_trajectories"


def save_mean_trajectory(ev, height, ages, lats, lons):
    path = config.ensure_dir(TRAJ_DIR) / f"Mean_Trajectory_{ev.short}_{height}m.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Age(hours)", "Latitude", "Longitude"])
        w.writerows(zip(ages, lats, lons))
    return path


def main():

    is_land = hysplit.make_land_lookup(config.LSM_FILE, config.LAND_FRACTION_THRESHOLD)

    rows = []
    for ev in config.EVENTS:
        for h in config.START_HEIGHTS_M:
            path = ev.tdump(h)
            if not path.exists():
                print(f"Event {ev.number} {h} m: missing {path}")
                continue
            traj, diag, _, _ = hysplit.parse_tdump(path)
            att = hysplit.attribute_moisture(traj, diag, is_land)
            ages, lats, lons = hysplit.mean_trajectory(traj)
            save_mean_trajectory(ev, h, ages, lats, lons)
            print(f"Event {ev.number:>2} {h:>3} m | land {att.land:5.1f}  ocean {att.ocean:5.1f}  "
                  f"FT {att.ft:5.1f}  unattributed {att.unattributed:5.1f}  "
                  f"(sum {att.closure:.1f} %, n = {att.n_traj})")
            rows.append({"event": ev.number, "key": ev.key, "height_m": h,
                         "land_pct": round(att.land, 2), "ocean_pct": round(att.ocean, 2),
                         "ft_pct": round(att.ft, 2),
                         "unattributed_pct": round(att.unattributed, 2),
                         "closure_pct": round(att.closure, 2), "n_traj": att.n_traj})

    if rows:
        config.ensure_dir(config.INTERMEDIATE_DIR)
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved {OUT_CSV}")


if __name__ == "__main__":
    main()
