from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable


@dataclass
class BusOption:
    route_name: str
    route_id: str
    vehicle_id: str
    wait_min: float
    seat_probability: float
    in_vehicle_min: float
    transfer_count: int = 0
    predicted_congestion: str = ""
    target_stop_name: str = ""


@dataclass
class DecisionPreferences:
    """
    All penalties are expressed in equivalent minutes of user inconvenience.

    standing_penalty_min:
        Added expected cost if the passenger is likely to stand.
        Expected standing cost = (1 - seat_probability) * standing_penalty_min

    transfer_penalty_min:
        Equivalent inconvenience per transfer.

    wait_weight:
        Waiting minutes multiplier. Values >1 mean waiting feels worse than
        the same amount of in-vehicle time.

    ride_weight:
        In-vehicle minutes multiplier.
    """
    standing_penalty_min: float = 12.0
    transfer_penalty_min: float = 8.0
    wait_weight: float = 1.25
    ride_weight: float = 1.0


def generalized_cost(option: BusOption, pref: DecisionPreferences) -> dict:
    if not 0 <= option.seat_probability <= 1:
        raise ValueError("seat_probability must be between 0 and 1.")
    if option.wait_min < 0 or option.in_vehicle_min < 0 or option.transfer_count < 0:
        raise ValueError("time and transfer inputs must be non-negative.")

    wait_cost = option.wait_min * pref.wait_weight
    ride_cost = option.in_vehicle_min * pref.ride_weight
    standing_risk_cost = (1.0 - option.seat_probability) * pref.standing_penalty_min
    transfer_cost = option.transfer_count * pref.transfer_penalty_min

    total = wait_cost + ride_cost + standing_risk_cost + transfer_cost

    return {
        "wait_cost": wait_cost,
        "ride_cost": ride_cost,
        "standing_risk_cost": standing_risk_cost,
        "transfer_cost": transfer_cost,
        "generalized_cost": total,
    }


def choose_best(options: Iterable[BusOption],
                pref: DecisionPreferences | None = None) -> dict:
    pref = pref or DecisionPreferences()
    options = list(options)
    if not options:
        raise ValueError("At least one bus option is required.")

    scored = []
    for opt in options:
        costs = generalized_cost(opt, pref)
        scored.append({
            **asdict(opt),
            **costs,
        })

    scored.sort(
        key=lambda x: (
            x["generalized_cost"],
            x["wait_min"],
            -x["seat_probability"],
        )
    )

    best = scored[0]
    baseline = min(scored, key=lambda x: x["wait_min"])
    explanation = []

    if best["vehicle_id"] != baseline["vehicle_id"]:
        extra_wait = best["wait_min"] - baseline["wait_min"]
        seat_gain = best["seat_probability"] - baseline["seat_probability"]
        cost_gain = baseline["generalized_cost"] - best["generalized_cost"]

        explanation.append(
            f"{extra_wait:.1f}분 더 기다리면 착석확률이 "
            f"{seat_gain*100:+.1f}%p 변하고, "
            f"예상 불편비용이 {cost_gain:.1f}분 감소합니다."
        )
    else:
        explanation.append(
            "가장 빨리 오는 버스가 현재 설정에서도 최저 비용 선택입니다."
        )

    return {
        "recommended": best,
        "ranked_options": scored,
        "preferences": asdict(pref),
        "explanation": explanation,
    }
