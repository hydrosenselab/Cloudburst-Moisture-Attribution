"""
Integrated water-vapour transport (IVT)

Daily IVT from hourly ERA5 q, u, v:

    Q_u = (1/g) integral_{300}^{1000 hPa} q u dp      (trapezoidal rule)
    IVT_d = sqrt(Q_u,d^2 + Q_v,d^2)   with Q_*,d the daily-mean components

Inputs
    config.IVT_DIR/IVT<n>-<event>.grib  (q, u, v on 1000-300 hPa)

Outputs
    outputs/06_ivt/<event>.nc   variables ivt_u, ivt_v, ivt_mag [time, latitude, longitude]

Known limitation
    Levels below the surface over high terrain are not masked (no surface-
    pressure field in these files).
"""

import warnings

import numpy as np

from cloudbursts import config
from cloudbursts.cli import base_parser, selected_events

warnings.filterwarnings("ignore")
OUT_DIR = config.INTERMEDIATE_DIR / "06_ivt"


def compute_ivt(path):
    import xarray as xr

    ds = xr.open_dataset(path, engine="cfgrib")
    p = "isobaricInhPa" if "isobaricInhPa" in ds.coords else "level"
    ds = ds.sortby(p).sel({p: slice(config.IVT_TOP_HPA, config.IVT_BOTTOM_HPA)})
    # integrate over pressure in hPa, x100 -> Pa, / g -> kg m-1 s-1
    qu = (ds["q"] * ds["u"]).integrate(coord=p) * 100.0 / config.GRAVITY
    qv = (ds["q"] * ds["v"]).integrate(coord=p) * 100.0 / config.GRAVITY
    qu_d, qv_d = qu.resample(time="1D").mean("time"), qv.resample(time="1D").mean("time")
    return xr.Dataset({"ivt_u": qu_d, "ivt_v": qv_d,
                       "ivt_mag": np.sqrt(qu_d ** 2 + qv_d ** 2)}).compute()


def main():
    args = base_parser("ERA5 IVT").parse_args()
    config.ensure_dir(OUT_DIR)
    for ev in selected_events(args):
        if not ev.ivt_grib.exists():
            print(f"Event {ev.number}: missing {ev.ivt_grib}")
            continue
        out = compute_ivt(ev.ivt_grib)
        print(f"Event {ev.number}: {ev.key}  max daily IVT {float(out.ivt_mag.max()):.1f} kg m-1 s-1")
        out.to_netcdf(OUT_DIR / f"{ev.key}.nc")
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
