from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

USGS_QUERY_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# Existing local kernels in this repo.
KERNEL_TLS = ROOT / "mine" / "kernels" / "naif0012.tls"
KERNEL_SPK = ROOT / "mine" / "kernels" / "de405.bsp"

PLANETS = [
    "MOON",
    "MERCURY BARYCENTER",
    "VENUS BARYCENTER",
    "MARS BARYCENTER",
    "JUPITER BARYCENTER",
    "SATURN BARYCENTER",
    "URANUS BARYCENTER",
    "NEPTUNE BARYCENTER",
]

OBS = "EARTH"
REF_FRAME = "J2000"

START_DATE = "2000-01-01"
END_DATE = "2026-01-01"
MIN_MAGNITUDE = 4.0
