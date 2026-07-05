"""Sources, paths, and analysis constants for the Washington DC study."""

from pathlib import Path

YEARS = (2022, 2023, 2024)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT.parent / "paper" / "figures"
TABLES = ROOT.parent / "paper" / "tables"

# Capital Bikeshare public S3
BIKESHARE_URL = (
    "https://s3.amazonaws.com/capitalbikeshare-data/{ym}-capitalbikeshare-tripdata.zip"
)

# NOAA Local Climatological Data, Reagan National (DCA), WBAN 13743
LCD_STATION = "72405013743"
LCD_URL = "https://www.ncei.noaa.gov/data/local-climatological-data/access/{year}/{station}.csv"

# EPA AirData
EPA_DAILY_AQI_URL = "https://aqs.epa.gov/aqsweb/airdata/daily_aqi_by_county_{year}.zip"

# Hourly air quality (CAMS reanalysis via Open-Meteo, no key)
DC_LAT, DC_LON = 38.8951, -77.0364
OPENMETEO_AQI_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
    "?latitude={lat}&longitude={lon}&hourly=us_aqi,pm2_5"
    "&start_date={year}-01-01&end_date={year}-12-31&timezone=America%2FNew_York"
)

LOCAL_TZ = "America/New_York"

# Suppression label
EXPECTED_FLOOR = 20  # min typical rides to call an hour "plausibly active"
SUPPRESSION_RATIO = 0.5  # observed below half of expected -> suppressed
SENSITIVITY_RATIOS = (0.4, 0.5, 0.6)

# Thermal-stress hinges (shared by the panel, Seoul transfer, and the browser app)
COLD_STRESS_BASE_F = 55.0  # degrees below this accumulate as cold stress
HEAT_STRESS_BASE_F = 85.0  # heat-index degrees above this accumulate as heat stress

# Temporal back-testing
TRAIN_YEARS = (2022, 2023)
TEST_YEAR = 2024

# Minimum-detectable-effect multipliers at alpha=0.05 two-sided: z_(1-alpha/2) + z_power.
# 80% power: 1.95996 + 0.84162; 90% power: 1.95996 + 1.28155.
MDE_Z80 = 2.802
MDE_Z90 = 3.242

# Adaptation + safety
SHIFT_WINDOW_H = 3  # hours either side for a time shift
HEAT_UNSAFE_F = 103.0  # heat index severe threshold
AQI_UNSAFE = 150  # unhealthy boundary
MIN_RISK_BENEFIT = 0.05  # only shift when it meaningfully lowers risk
MEAN_RIDE_MIN = 13.0  # Capital Bikeshare typical ride length


def ym_list() -> list[str]:
    return [f"{y}{m:02d}" for y in YEARS for m in range(1, 13)]
