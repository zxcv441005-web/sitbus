
from __future__ import annotations
import math


def evaluate(rows: list[dict], threshold: float = 0.5) -> dict:
    if not rows:
        return {
            "n": 0,
            "message": "아직 해결된 예측-정답 쌍이 없습니다."
        }

    y = [int(r["seat_proxy_actual"]) for r in rows]
    p = [min(max(float(r["seat_probability"]), 1e-12), 1 - 1e-12) for r in rows]
    pred = [1 if x >= threshold else 0 for x in p]

    n = len(y)
    accuracy = sum(a == b for a, b in zip(y, pred)) / n
    brier = sum((a-b) ** 2 for a, b in zip(y, p)) / n
    logloss = -sum(
        a * math.log(prob) + (1-a) * math.log(1-prob)
        for a, prob in zip(y, p)
    ) / n

    tp = sum(a == 1 and b == 1 for a, b in zip(y, pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y, pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y, pred))

    return {
        "n": n,
        "accuracy_at_0_5": accuracy,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "brier_score": brier,
        "log_loss": logloss,
        "note": (
            "정답은 개인 착석 여부가 아니라 목표 정류장의 공식 혼잡도 "
            "'여유' 여부를 사용한 대리정답입니다."
        ),
    }
