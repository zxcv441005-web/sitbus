from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from zoneinfo import ZoneInfo

from .config import settings
from .decision import BusOption
from .models import StationInfo, LiveBusState
from .predictor import (
    predict, CONGESTION_LABEL, CONGESTION_MIDPOINT,
    SEAT_THRESHOLD, normal_cdf
)


SEOUL_TZ = ZoneInfo("Asia/Seoul")


def _station_pair_for_direction(stops, origin_station_id, destination_station_id):
    origins = [s for s in stops if s.station_id == origin_station_id]
    destinations = [s for s in stops if s.station_id == destination_station_id]
    pairs = [
        (o, d)
        for o in origins
        for d in destinations
        if d.seq > o.seq
    ]
    if not pairs:
        return None
    return min(pairs, key=lambda pair: pair[1].seq - pair[0].seq)


def _seat_probability_at_current_stop(live: LiveBusState):
    midpoint = CONGESTION_MIDPOINT.get(live.congestion_code)
    if midpoint is None:
        return None
    mean = max(midpoint, .98) if live.is_full else midpoint
    return normal_cdf(SEAT_THRESHOLD, mean, .10)


def _find_live_vehicle(states, origin_seq, arrival_vehicle_id):
    if arrival_vehicle_id:
        exact = next(
            (s for s in states if s.vehicle_id == arrival_vehicle_id),
            None
        )
        if exact:
            return exact

    candidates = [
        s for s in states
        if s.current_stop_order <= origin_seq
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda s: s.current_stop_order)


class DirectJourneyDiscovery:
    def __init__(self, service):
        self.service = service
        self.api = service.api

    def station_search(self, query: str):
        return self.api.search_stations(query)

    def direct_candidates(
        self,
        *,
        origin_station_id: str,
        origin_ars_id: str,
        destination_station_id: str,
        destination_ars_id: str,
        hour: int | None = None,
        max_routes: int = 12,
    ) -> dict:
        hour = (
            datetime.now(SEOUL_TZ).hour
            if hour is None
            else int(hour)
        )

        origin_routes = {
            r.route_id: r
            for r in self.api.routes_by_station(origin_ars_id)
        }
        destination_route_ids = {
            r.route_id
            for r in self.api.routes_by_station(destination_ars_id)
        }
        common_ids = [
            rid for rid in origin_routes
            if rid in destination_route_ids
        ]

        options = []
        diagnostics = []

        for rid in common_ids[:max_routes]:
            route = origin_routes[rid]
            try:
                stops = self.api.fetch_route_stops(rid)
                pair = _station_pair_for_direction(
                    stops,
                    origin_station_id,
                    destination_station_id,
                )
                if pair is None:
                    diagnostics.append({
                        "route": route.route_name,
                        "status": "rejected_wrong_direction",
                    })
                    continue

                origin_stop, destination_stop = pair

                origin_arrivals = self.api.fetch_arrivals_by_route(
                    origin_station_id,
                    rid,
                    origin_stop.seq,
                )
                destination_arrivals = self.api.fetch_arrivals_by_route(
                    destination_station_id,
                    rid,
                    destination_stop.seq,
                )

                first = origin_arrivals[0] if origin_arrivals else None
                arrival_vehicle_id = first.vehicle_id if first else ""
                wait_min = (
                    max(first.eta_seconds, 0) / 60.0
                    if first and first.eta_seconds is not None
                    else None
                )

                in_vehicle_min = None
                travel_time_source = "stop_count_fallback"
                if first and first.vehicle_id and first.eta_seconds is not None:
                    match = next(
                        (
                            x for x in destination_arrivals
                            if x.vehicle_id == first.vehicle_id
                            and x.eta_seconds is not None
                        ),
                        None,
                    )
                    if match and match.eta_seconds > first.eta_seconds:
                        in_vehicle_min = (
                            match.eta_seconds - first.eta_seconds
                        ) / 60.0
                        travel_time_source = "matched_live_arrival"

                stops_between = destination_stop.seq - origin_stop.seq
                if in_vehicle_min is None:
                    in_vehicle_min = max(4.0, stops_between * 1.8)

                if wait_min is None:
                    term = route.term_min
                    if not term:
                        try:
                            term = self.api.resolve_route(route.route_name).term_min
                        except Exception:
                            term = settings.default_headway_min
                    wait_min = max(float(term or settings.default_headway_min) / 2, 1.0)
                    arrival_source = "half_headway_fallback"
                else:
                    arrival_source = "live_arrival"

                states = self.api.fetch_route_positions(rid)
                live = _find_live_vehicle(
                    states,
                    origin_stop.seq,
                    arrival_vehicle_id,
                )

                seat_probability = .50
                predicted_congestion = "정보없음"
                seat_source = "neutral_fallback"
                confidence = "낮음"

                if live and live.congestion_code in {3,4,5,6}:
                    predicted_congestion = CONGESTION_LABEL[live.congestion_code]

                    if live.current_stop_order < origin_stop.seq:
                        try:
                            route_term = route.term_min
                            if not route_term:
                                try:
                                    route_term = self.api.resolve_route(
                                        route.route_name
                                    ).term_min
                                except Exception:
                                    route_term = settings.default_headway_min

                            flows = self.service.historical.build_route_flows(
                                route_name=route.route_name,
                                hour=hour,
                                route_stops=stops,
                                service_days=settings.service_days,
                                headway_min=route_term or settings.default_headway_min,
                            )
                            p = predict(
                                live,
                                origin_stop.seq,
                                flows,
                                queue_fraction=.5,
                                effective_capacity=settings.effective_capacity,
                            )
                            seat_probability = p.seat_probability
                            predicted_congestion = p.predicted_target_congestion
                            seat_source = "live_congestion_plus_historical_flow"
                            confidence = "중간"
                        except Exception:
                            seat_probability = (
                                _seat_probability_at_current_stop(live)
                                or .50
                            )
                            seat_source = "current_congestion_only"
                            confidence = "낮음"
                    else:
                        seat_probability = (
                            _seat_probability_at_current_stop(live)
                            or .50
                        )
                        seat_source = "current_congestion_only"
                        confidence = "낮음"

                vehicle_id = (
                    arrival_vehicle_id
                    or (live.vehicle_id if live else f"{rid}-unknown")
                )

                options.append({
                    "route_name": route.route_name,
                    "route_id": rid,
                    "vehicle_id": vehicle_id,
                    "wait_min": round(wait_min, 1),
                    "seat_probability": round(float(seat_probability), 4),
                    "in_vehicle_min": round(float(in_vehicle_min), 1),
                    "transfer_count": 0,
                    "predicted_congestion": predicted_congestion,
                    "target_stop_name": destination_stop.name,
                    "origin_stop_name": origin_stop.name,
                    "origin_stop_order": origin_stop.seq,
                    "destination_stop_order": destination_stop.seq,
                    "direction": origin_stop.direction,
                    "stops_between": stops_between,
                    "arrival_source": arrival_source,
                    "seat_probability_source": seat_source,
                    "travel_time_source": travel_time_source,
                    "confidence": confidence,
                    "arrival_message": first.message if first else "",
                })
            except Exception as exc:
                diagnostics.append({
                    "route": route.route_name,
                    "status": "error",
                    "detail": str(exc),
                })

        options.sort(key=lambda x: (x["wait_min"], -x["seat_probability"]))

        return {
            "origin_station_id": origin_station_id,
            "destination_station_id": destination_station_id,
            "hour": hour,
            "direct_route_count": len(options),
            "options": options,
            "diagnostics": diagnostics,
        }
