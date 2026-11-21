"""
Event list, data paths and constants used by every script.

Set DATA_ROOT to the folder holding the input data. Times are UTC; Day 0 is the
UTC day of the native IMERG peak; coordinates are (latitude, longitude).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DATA_ROOT = Path("path/to/data")      # folder with the input data
INTERMEDIATE_DIR = Path("outputs")    # results written by the scripts

# Input data inside DATA_ROOT
IMERG_DIR = DATA_ROOT / "IMERG Precipitation (10 main events)"
HYSPLIT_DIR = DATA_ROOT / "HYSPLIT_Data_Trajectories"
ILDAS_DIR = DATA_ROOT / "ILDAS Input Data"
MSLP_DIR = DATA_ROOT / "ERA5 MSLP (1991-2024)"
IVT_DIR = DATA_ROOT / "Event Specific IVT"

# ERA5 land-sea mask (variable lsm, 0.25 deg)
LSM_FILE = DATA_ROOT / "ERA5 Land-Sea Mask" / "era5_lsm.nc"

# Optional ILDAS mask (1 = keep); None to disable
ILDAS_MASK_FILE: Path | None = None

# Analysis constants
K_NEIGHBORS = 9                       # native IMERG cells searched per event
CLIM_START, CLIM_END = 1991, 2024     # anomaly baseline (inclusive)
MAP_EXTENT = [66.5, 99.0, 5.0, 38.5]    # lon_min, lon_max, lat_min, lat_max
REGRID_DOMAIN = (0.0, 90.0, 50.0, 140.0)  # lat_min, lat_max, lon_min, lon_max
START_HEIGHTS_M = (100, 300, 500)     # HYSPLIT start heights, m AGL
LAND_FRACTION_THRESHOLD = 0.5         # ERA5 lsm >= this is land
IVT_TOP_HPA, IVT_BOTTOM_HPA = 300, 1000
GRAVITY = 9.80665                     # m s-2
PRESSURE_LEVELS_HPA = [200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650,
                       700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]


# Event table
@dataclass(frozen=True)
class Event:
    """One cloudburst event and its file names."""

    number: int
    key: str                 # folder-style name, e.g. "Kedarnath_2013"
    short: str               # file-style name, e.g. "Kedarnath2013"
    state: str
    target: tuple[float, float]  # (lat, lon) searched in IMERG
    imerg_start: str         # IMERG search window, UTC, inclusive
    imerg_end: str
    day0: str                # UTC day of the native IMERG peak
    era5_prefix: str         # prefix of the ERA5 GRIB files, e.g. "Ked2013"
    site: str                # site name
    start_hour: int          # HYSPLIT start hour on Day 0, UTC (hour 0 of the analysis)
    tier: int                # daily rainfall vs district percentiles: 1 > 99.99th, 2 between, 3 < 99.90th
    tdump_stem: str          # tdump file name before "_tdump", e.g. "Ked2013"

    @property
    def imerg_dir(self) -> Path:
        return IMERG_DIR / self.key

    @property
    def hysplit_dir(self) -> Path:
        return HYSPLIT_DIR / f"{self.key} GRIB & ARL"

    @property
    def era5_3d(self) -> Path:
        return self.hysplit_dir / f"{self.era5_prefix}_era5_3d.grib"

    @property
    def era5_2d(self) -> Path:
        return self.hysplit_dir / f"{self.era5_prefix}_era5_2d.grib"

    @property
    def ivt_grib(self) -> Path:
        return IVT_DIR / f"IVT{self.number}-{self.key}.grib"

    def tdump(self, height_m: int) -> Path:
        """Path of the HYSPLIT ensemble tdump for one start height."""
        return self.hysplit_dir / f"{self.tdump_stem}_tdump ({height_m}mAGL)"


def _ev(number, key, state, target, start, end, day0, era5_prefix, site, start_hour, tier,
        tdump_stem=None):
    short = key.replace("_", "")
    return Event(number, key, short, state, target, start, end, day0, era5_prefix, site,
                 start_hour, tier, tdump_stem or short)


# IMERG search window: reported dates -3 to +1 day. Leh starts at 19 UTC, one
# hour after its native IMERG maximum.
EVENTS: list[Event] = [
    _ev(1, "Ukhimath_1998", "Uttarakhand", (30.55, 79.21),
        "1998-08-11", "1998-08-19", "1998-08-15", "Ukhimath1998",
        "Ukhimath", 17, 3),
    _ev(2, "Shillagarh_2003", "Himachal Pradesh", (31.80, 77.00),
        "2003-07-14", "2003-07-18", "2003-07-15", "Shillagarh2003",
        "Shillagarh", 20, 3),
    _ev(3, "Leh_2010", "Ladakh", (34.09, 77.34),
        "2010-08-03", "2010-08-07", "2010-08-04", "Leh2_2010",
        "Leh", 19, 1),
    _ev(4, "Kedarnath_2013", "Uttarakhand", (30.70, 79.00),
        "2013-06-14", "2013-06-18", "2013-06-16", "Ked2013",
        "Kedarnath", 15, 1, "Ked2013"),
    _ev(5, "Udhampur_2014", "Jammu & Kashmir", (32.91, 75.14),
        "2014-09-01", "2014-09-07", "2014-09-05", "Udhampur2_2014",
        "Udhampur", 20, 2),
    _ev(6, "Mandi_2015", "Himachal Pradesh", (31.709, 76.94),
        "2015-08-05", "2015-08-10", "2015-08-08", "Mandi2015",
        "Dharampur", 1, 2),
    _ev(7, "Arakot_2019", "Uttarakhand", (31.063, 77.848),
        "2019-08-16", "2019-08-19", "2019-08-18", "Arakot2019",
        "Arakot", 0, 2),
    _ev(8, "Bhagsu_2021", "Himachal Pradesh", (32.24, 76.33),
        "2021-07-11", "2021-07-13", "2021-07-12", "Bhagsu2021",
        "Bhagsu", 9, 2),
    _ev(9, "Dehradun_2022", "Uttarakhand", (30.321, 78.094),
        "2022-08-16", "2022-08-21", "2022-08-19", "Dehradun2022",
        "Raipur", 20, 1),
    _ev(10, "Shimla_2024", "Himachal Pradesh", (31.451, 77.642),
        "2024-07-29", "2024-08-02", "2024-07-31", "Shimla2024",
        "Rampur", 19, 3),
]

def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
