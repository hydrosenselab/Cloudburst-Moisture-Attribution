"""
Moisture-source statistics across the ten events

Prints the land, ocean, free-troposphere and unattributed shares and their
statistics: ten-event means with 95% bootstrap confidence intervals, the
leave-one-out range, the land share of boundary-layer (BL) uptake, the five
most versus five least intense events, and the tier comparison.

Inputs
    data/moisture_fractions.csv   (or outputs/03_moisture_fractions.csv if present)
    data/events.csv

Run
    python 07_moisture_statistics.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
OUTPUT = Path("outputs") / "03_moisture_fractions.csv"
HEIGHTS = (100, 300, 500)
EVENTS = list(range(1, 11))

frac = pd.read_csv(OUTPUT if OUTPUT.exists() else DATA / "moisture_fractions.csv", comment="#")
events = pd.read_csv(DATA / "events.csv", comment="#").set_index("event")


def share(col, h):
    return frac[frac.height_m == h].set_index("event")[col].reindex(EVENTS)


print("Land BL share of arriving moisture (mean of ten events, 95% bootstrap CI, 10,000 resamples)")
rng = np.random.default_rng(42)
for h in HEIGHTS:
    x = share("land_pct", h).values
    boots = rng.choice(x, (10_000, 10)).mean(1)
    loo = [np.delete(x, i).mean() for i in range(10)]
    print(f"  {h} m: {x.mean():.1f}% (CI {np.percentile(boots, 2.5):.1f}-{np.percentile(boots, 97.5):.1f}%), "
          f"leave-one-out {min(loo):.1f}-{max(loo):.1f}%")
x = share("land_pct", 100)
print(f"  Events with land >= 50% at 100 m: {list(x[x >= 50].index)}")
for h in HEIGHTS:
    comp = frac[frac.height_m == h].set_index("event")[["land_pct", "ocean_pct", "ft_pct", "unattributed_pct"]]
    print(f"  Land the largest component at {h} m in {(comp.idxmax(axis=1) == 'land_pct').sum()} of 10 events")

print("\nOcean, free troposphere and unattributed shares")
for h in HEIGHTS:
    print(f"  {h} m: ocean {share('ocean_pct', h).mean():.1f}%, FT {share('ft_pct', h).mean():.1f}%, "
          f"unattributed {share('unattributed_pct', h).mean():.1f}%")
print(f"  Ocean maximum over all events and heights: {frac.ocean_pct.max():.1f}%")
print(f"  Unattributed range: {frac.unattributed_pct.min():.1f}-{frac.unattributed_pct.max():.1f}%")

print("\nLand share of all BL uptake, land / (land + ocean)")
for h in HEIGHTS:
    r = 100 * share("land_pct", h) / (share("land_pct", h) + share("ocean_pct", h))
    print(f"  {h} m: per event {r.min():.1f}-{r.max():.1f}%")
land, ocean = share("land_pct", 100).values, share("ocean_pct", 100).values
idx = np.random.default_rng(0).integers(0, 10, (10_000, 10))
boots = 100 * land[idx].sum(1) / (land[idx] + ocean[idx]).sum(1)
print(f"  Pooled at 100 m: {100 * land.sum() / (land + ocean).sum():.1f}% "
      f"(CI {np.percentile(boots, 2.5):.1f}-{np.percentile(boots, 97.5):.1f}%)")
lower = sum(share("land_pct", 500)[e] < share("land_pct", 100)[e] for e in EVENTS)
print(f"  Land share smaller at 500 m than at 100 m in {lower} of 10 events")

print("\nFive most versus five least intense events (native IMERG peak rate)")
order = events.loc[EVENTS, "native_peak_mm_h"].sort_values(ascending=False)
top, rest = list(order.index[:5]), list(order.index[5:])
print(f"  Most intense: {sorted(top)}")
for h in HEIGHTS:
    x = share("land_pct", h)
    print(f"  {h} m: land share {x[top].mean():.1f}% vs {x[rest].mean():.1f}% "
          f"(difference {x[top].mean() - x[rest].mean():+.1f} percentage points)")

print("\nRainfall tiers and largest daily totals")
tier1 = list(events.index[events.tier == 1])
x = share("land_pct", 100)
print(f"  Tier 1 {tier1}: land {x[tier1].mean():.1f}% at 100 m; other seven {x.drop(tier1).mean():.1f}%")
wettest = list(events.loc[EVENTS, "daily_rainfall_mm"].sort_values(ascending=False).index[:3])
vals = [share("land_pct", h)[e] for e in wettest for h in HEIGHTS]
print(f"  Three largest daily totals {wettest}: land {min(vals):.1f}-{max(vals):.1f}% across heights")
