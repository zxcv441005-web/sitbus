from dataclasses import dataclass
import os
from pathlib import Path


def _default_data_dir() -> str:
    # Cloud deployments can set SITBUS_DATA_DIR to a mounted persistent volume.
    # Local development falls back to ./runtime_data.
    return os.getenv("SITBUS_DATA_DIR", "runtime_data")


@dataclass(frozen=True)
class Settings:
    api_key: str = os.getenv("SEOUL_BUS_API_KEY", "")
    historical_csv_path: str = os.getenv("HISTORICAL_CSV_PATH", "")
    route_name: str = os.getenv("ROUTE_NAME", "143")
    route_id: str = os.getenv("ROUTE_ID", "100100022")
    default_headway_min: float = float(os.getenv("DEFAULT_HEADWAY_MIN", "6"))
    service_days: int = int(os.getenv("SERVICE_DAYS", "31"))
    effective_capacity: float = float(os.getenv("EFFECTIVE_CAPACITY", "45"))
    data_dir: str = _default_data_dir()
    db_path: str = os.getenv(
        "DB_PATH",
        str(Path(_default_data_dir()) / "seat_service.sqlite3"),
    )


settings = Settings()
