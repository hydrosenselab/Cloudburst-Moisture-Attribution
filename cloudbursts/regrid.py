"""
cloudbursts.regrid

First-order conservative remapping of IMERG (0.1 deg) to the ERA5 grid
 (0.25 deg cells centred on ERA5 points, bounds at +/-0.125 deg).

Equations (Methods, eqs 1-2)::

    P_C = sum_k A_kC P_k / sum_k A_kC
    A_kC = d_lambda_kC * d_phi_kC * cos(phi_k)

Missing source cells get zero weight and the target value is the weighted
mean of the remaining valid cells.
"""

from __future__ import annotations

import numpy as np

FINE_RES = 0.10
COARSE_RES = 0.25


def era5_axis(vmin: float, vmax: float, res: float = COARSE_RES) -> np.ndarray:
    """Regular ERA5-compatible axis from vmin to vmax inclusive."""
    return np.round(np.arange(vmin, vmax + 1e-3, res), 4)


def build_weights(fine_lats, fine_lons, coarse_lats, coarse_lons,
                  dfine: float = FINE_RES, dcoarse: float = COARSE_RES):
    """
    Pre-compute overlap weights for every target cell.

    Returns a list of (i, j, lat_slice, lon_slice, w2d) for target cells that
    overlap at least one source cell.
    """
    cos_lats = np.cos(np.deg2rad(fine_lats)).astype(np.float32)
    f_lat_lo, f_lat_hi = fine_lats - dfine / 2, fine_lats + dfine / 2
    f_lon_lo, f_lon_hi = fine_lons - dfine / 2, fine_lons + dfine / 2

    cells = []
    for i, lat_c in enumerate(coarse_lats):
        lat_ov = np.maximum(0, np.minimum(lat_c + dcoarse / 2, f_lat_hi)
                            - np.maximum(lat_c - dcoarse / 2, f_lat_lo))
        vi = np.where(lat_ov > 0)[0]
        if vi.size == 0:
            continue
        lat_sl = slice(vi[0], vi[-1] + 1)
        lat_w = lat_ov[lat_sl] * cos_lats[lat_sl]
        for j, lon_c in enumerate(coarse_lons):
            lon_ov = np.maximum(0, np.minimum(lon_c + dcoarse / 2, f_lon_hi)
                                - np.maximum(lon_c - dcoarse / 2, f_lon_lo))
            vj = np.where(lon_ov > 0)[0]
            if vj.size == 0:
                continue
            lon_sl = slice(vj[0], vj[-1] + 1)
            w2d = (lat_w[:, None] * lon_ov[lon_sl][None, :]).astype(np.float32)
            cells.append((i, j, lat_sl, lon_sl, w2d))
    return cells


def apply_weights(fine, cells, n_lat: int, n_lon: int) -> np.ndarray:
    """Apply pre-computed weights to one fine field [lat, lon]."""
    out = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
    for i, j, lat_sl, lon_sl, w2d in cells:
        sub = fine[lat_sl, lon_sl]
        valid = np.isfinite(sub) & (sub >= 0)
        if not valid.any():
            continue
        w = np.where(valid, w2d, 0.0)
        wsum = float(w.sum())
        if wsum > 0:
            out[i, j] = float(np.sum(np.where(valid, sub, 0.0) * w)) / wsum
    return out


def conservation_error(fine, fine_lats, fine_lons, coarse, coarse_lats, coarse_lons,
                       exclude_boundary: bool = True) -> float:
    """
    Relative difference (%) between cosine-weighted area sums of the fine and
    regridded fields.

    With ``exclude_boundary=True`` (recommended) the outer ring of target cells
    is dropped, because those cells extend beyond the source domain and are
    only partly covered; including them inflates the error.
    """
    lat_frac = np.ones(len(fine_lats))
    lon_frac = np.ones(len(fine_lons))
    if exclude_boundary:
        coarse = coarse[1:-1, 1:-1]
        coarse_lats, coarse_lons = coarse_lats[1:-1], coarse_lons[1:-1]
        h, f = COARSE_RES / 2, FINE_RES / 2

        def frac(centres, lo, hi):
            # share of each fine cell lying inside [lo, hi]
            return np.clip(np.minimum(centres + f, hi) - np.maximum(centres - f, lo), 0, None) / FINE_RES

        lat_frac = frac(fine_lats, coarse_lats[0] - h, coarse_lats[-1] + h)
        lon_frac = frac(fine_lons, coarse_lons[0] - h, coarse_lons[-1] + h)
    weights = (np.cos(np.deg2rad(fine_lats)) * lat_frac)[:, None] * lon_frac[None, :]
    fine_sum = np.nansum(fine * weights) * FINE_RES ** 2
    coarse_sum = np.nansum(coarse * np.cos(np.deg2rad(coarse_lats))[:, None]) * COARSE_RES ** 2
    return float(abs(fine_sum - coarse_sum) / fine_sum * 100) if fine_sum > 0 else 0.0


def containing_cell(lat: float, lon: float, coarse_lats, coarse_lons):
    """
    Index and centre of the 0.25 deg cell that CONTAINS a point
    (cell bounds = centre +/- 0.125 deg).

    This is the HYSPLIT start-cell rule: the 0.25 deg cell containing the
    native 0.1 deg peak cell. IMERG centres sit at x.x5 deg and never fall on
    a 0.25 deg cell boundary (x.125, x.375, ...), so the result is unique.
    """
    i = int(np.argmin(np.abs(np.asarray(coarse_lats) - lat)))
    j = int(np.argmin(np.abs(np.asarray(coarse_lons) - lon)))
    h = COARSE_RES / 2
    if abs(coarse_lats[i] - lat) > h + 1e-9 or abs(coarse_lons[j] - lon) > h + 1e-9:
        raise ValueError(f"Point ({lat}, {lon}) lies outside the target grid")
    return i, j, float(coarse_lats[i]), float(coarse_lons[j])
