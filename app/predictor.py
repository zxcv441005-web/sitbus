
from __future__ import annotations

from math import erf, sqrt
import pandas as pd

from .models import LiveBusState, Prediction


CONGESTION_LABEL = {
    0: "정보없음",
    3: "여유",
    4: "보통",
    5: "혼잡",
    6: "매우혼잡",
}

# MODEL ASSUMPTIONS. These must later be calibrated from observed outcomes.
CONGESTION_MIDPOINT = {
    3: 0.34,
    4: 0.58,
    5: 0.79,
    6: 0.95,
}

SEAT_THRESHOLD = 0.45
MODEL_VERSION = "v4-heuristic-2026-09"


def normal_cdf(x: float, mean: float, sigma: float) -> float:
    sigma = max(sigma, 1e-9)
    z = (x - mean) / (sigma * sqrt(2))
    return 0.5 * (1 + erf(z))


def load_to_congestion(load: float) -> str:
    if load < 0.45:
        return "여유"
    if load < 0.70:
        return "보통"
    if load < 0.90:
        return "혼잡"
    return "매우혼잡"


def predict(
    live: LiveBusState,
    target_stop_order: int,
    flows: pd.DataFrame,
    *,
    queue_fraction: float = 0.5,
    effective_capacity: float = 45.0,
) -> Prediction:
    if live.congestion_code not in CONGESTION_MIDPOINT:
        raise ValueError(
            f"예측 불가 혼잡도 코드={live.congestion_code}. "
            "정상 예측 입력은 3/4/5/6이며, 0 또는 기타 값은 결측/비정상 값으로 처리합니다."
        )
    if target_stop_order <= live.current_stop_order:
        raise ValueError("목표 정류장은 현재 위치보다 뒤여야 합니다.")
    if not 0 <= queue_fraction <= 1:
        raise ValueError("queue_fraction은 0~1이어야 합니다.")

    current_load = CONGESTION_MIDPOINT[live.congestion_code]
    if live.is_full:
        current_load = max(current_load, 0.98)

    required = {"stop_order", "expected_board", "expected_alight"}
    if required - set(flows.columns):
        raise ValueError("승하차 flow 열이 부족합니다.")

    # Intermediate stops: board and alight both affect load.
    intermediate = flows[
        (flows["stop_order"] > live.current_stop_order)
        & (flows["stop_order"] < target_stop_order)
    ]
    delta_intermediate = float(
        (intermediate["expected_board"] - intermediate["expected_alight"]).sum()
    )

    # Target stop: all expected alighters free capacity before boarding.
    # Only the expected share of people ahead of the user competes for seats.
    target = flows[flows["stop_order"] == target_stop_order]
    if target.empty:
        target_alight = 0.0
        target_board_ahead = 0.0
    else:
        target_alight = float(target["expected_alight"].sum())
        target_board_ahead = float(target["expected_board"].sum()) * queue_fraction

    expected_net = delta_intermediate - target_alight + target_board_ahead

    projected_load = current_load + expected_net / max(effective_capacity, 1.0)
    projected_load = min(max(projected_load, 0.0), 1.2)

    stops_ahead = target_stop_order - live.current_stop_order

    # Heuristic uncertainty increases with forecast horizon.
    sigma = 0.10 + 0.035 * sqrt(stops_ahead)
    seat_probability = normal_cdf(SEAT_THRESHOLD, projected_load, sigma)

    return Prediction(
        current_congestion=CONGESTION_LABEL[live.congestion_code],
        predicted_target_congestion=load_to_congestion(projected_load),
        seat_probability=seat_probability,
        projected_load_score=projected_load,
        uncertainty_sigma=sigma,
        stops_ahead=stops_ahead,
        expected_net_passenger_change=expected_net,
        model_version=MODEL_VERSION,
    )
