
from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

from .models import RouteStop


ROUTE_COLS = ["노선번호", "노선명", "버스노선번호"]
STOP_NAME_COLS = ["역명", "정류장명", "버스정류장명", "정류소명"]
ARS_COLS = [
    "버스정류장ARS번호", "정류장ARS번호", "정류소ARS-ID",
    "ARS_ID", "ARS-ID", "정류장번호"
]


def _first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def _normalize_ars(value: object) -> str:
    s = str(value).strip()
    s = re.sub(r"\.0$", "", s)
    if not s or s.lower() == "nan":
        return ""
    return s.zfill(5) if s.isdigit() and len(s) < 5 else s


def detect_hour_columns(df: pd.DataFrame, hour: int) -> tuple[str, str]:
    variants = [
        (f"{hour}시승차총승객수", f"{hour}시하차총승객수"),
        (f"{hour:02d}시승차총승객수", f"{hour:02d}시하차총승객수"),
        (f"{hour}시승차", f"{hour}시하차"),
        (f"{hour:02d}시승차", f"{hour:02d}시하차"),
    ]
    for b, a in variants:
        if b in df.columns and a in df.columns:
            return b, a
    raise KeyError(f"{hour}시 승하차 열을 찾지 못했습니다.")


class HistoricalFlowStore:
    """
    Reads the official monthly hourly Seoul bus CSV and converts route-stop
    hourly totals into expected passengers per individual vehicle.

    The conversion needs a headway assumption:
        buses/hour = 60 / headway_min
    """

    def __init__(self, csv_path: str):
        if not csv_path:
            raise ValueError("HISTORICAL_CSV_PATH is empty.")
        p = Path(csv_path)
        if not p.exists():
            raise FileNotFoundError(p)
        self.df = self._read_csv(p)

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        last_err = None
        for enc in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except UnicodeDecodeError as e:
                last_err = e
        raise last_err

    def build_route_flows(
        self,
        route_name: str,
        hour: int,
        route_stops: list[RouteStop],
        service_days: int,
        headway_min: float,
    ) -> pd.DataFrame:
        df = self.df
        route_col = _first_existing(df.columns, ROUTE_COLS)
        name_col = _first_existing(df.columns, STOP_NAME_COLS)
        ars_col = _first_existing(df.columns, ARS_COLS)

        if not route_col:
            raise KeyError(f"노선 열을 찾지 못했습니다. columns={list(df.columns)[:20]}")
        if not name_col and not ars_col:
            raise KeyError("정류장명 또는 ARS 열을 찾지 못했습니다.")

        bcol, acol = detect_hour_columns(df, hour)

        route_mask = df[route_col].astype(str).str.strip().eq(str(route_name).strip())
        r = df.loc[route_mask].copy()
        if r.empty:
            route_mask = df[route_col].astype(str).str.contains(
                str(route_name), regex=False, na=False
            )
            r = df.loc[route_mask].copy()

        r[bcol] = pd.to_numeric(r[bcol], errors="coerce").fillna(0.0)
        r[acol] = pd.to_numeric(r[acol], errors="coerce").fillna(0.0)

        stop_by_ars = {_normalize_ars(s.ars_id): s for s in route_stops if s.ars_id}
        stop_by_name = {s.name.strip(): s for s in route_stops if s.name}

        rows = []
        for _, row in r.iterrows():
            stop = None
            if ars_col:
                stop = stop_by_ars.get(_normalize_ars(row.get(ars_col, "")))
            if stop is None and name_col:
                stop = stop_by_name.get(str(row.get(name_col, "")).strip())
            if stop is None:
                continue

            buses_per_hour = 60.0 / max(float(headway_min), 0.1)
            denom = max(service_days * buses_per_hour, 1e-9)

            rows.append({
                "stop_order": stop.seq,
                "station_id": stop.station_id,
                "ars_id": stop.ars_id,
                "stop_name": stop.name,
                "expected_board": float(row[bcol]) / denom,
                "expected_alight": float(row[acol]) / denom,
            })

        out = pd.DataFrame(rows)
        if out.empty:
            return pd.DataFrame(columns=[
                "stop_order", "station_id", "ars_id", "stop_name",
                "expected_board", "expected_alight"
            ])

        return (
            out.groupby(
                ["stop_order", "station_id", "ars_id", "stop_name"],
                as_index=False
            )[["expected_board", "expected_alight"]]
            .sum()
            .sort_values("stop_order")
            .reset_index(drop=True)
        )
