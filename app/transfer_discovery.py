from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from .config import settings
from .models import LiveBusState
from .predictor import (
    predict,
    CONGESTION_LABEL,
    CONGESTION_MIDPOINT,
    SEAT_THRESHOLD,
    normal_cdf,
)

SEOUL_TZ = ZoneInfo("Asia/Seoul")

def _seat_prob_from_live(live: LiveBusState):
    midpoint = CONGESTION_MIDPOINT.get(live.congestion_code)
    if midpoint is None:
        return None
    mean = max(midpoint, .98) if live.is_full else midpoint
    return normal_cdf(SEAT_THRESHOLD, mean, .10)

def _route_stop_pair(stops, from_station_id, to_station_id):
    starts = [s for s in stops if s.station_id == from_station_id]
    ends = [s for s in stops if s.station_id == to_station_id]
    pairs = [(a,b) for a in starts for b in ends if b.seq > a.seq]
    if not pairs:
        return None
    return min(pairs, key=lambda p: p[1].seq - p[0].seq)

def _nearest_upstream_vehicle(states, board_seq):
    candidates = [s for s in states if s.current_stop_order <= board_seq and s.congestion_code in {3,4,5,6}]
    if not candidates:
        return None
    return max(candidates, key=lambda s: s.current_stop_order)

class TransferJourneyDiscovery:
    def __init__(self, service):
        self.service = service
        self.api = service.api

    def _segment_comfort(self, *, segment, hour, is_first_segment):
        stops = self.api.fetch_route_stops(segment.route_id)
        pair = _route_stop_pair(stops, segment.from_station_id, segment.to_station_id)
        if pair is None:
            return {"seat_probability": .50, "confidence": "낮음", "source": "route_pair_unresolved", "wait_min": settings.default_headway_min / 2, "from_seq": None, "to_seq": None}
        board_stop, alight_stop = pair
        route_info = None
        try:
            route_info = self.api.resolve_route(segment.route_name)
        except Exception:
            pass
        term = route_info.term_min if route_info and route_info.term_min else settings.default_headway_min
        wait_min = max(float(term)/2.0, 1.0)
        arrival_source = "half_headway_fallback"
        arrival_vehicle_id = ""
        if is_first_segment:
            try:
                arrivals = self.api.fetch_arrivals_by_route(board_stop.station_id, segment.route_id, board_stop.seq)
                if arrivals:
                    a = arrivals[0]
                    arrival_vehicle_id = a.vehicle_id
                    if a.eta_seconds is not None:
                        wait_min = max(a.eta_seconds/60.0, 0.2)
                        arrival_source = "live_arrival"
            except Exception:
                pass
        try:
            states = self.api.fetch_route_positions(segment.route_id)
        except Exception:
            states = []
        live = next((x for x in states if x.vehicle_id == arrival_vehicle_id), None) if arrival_vehicle_id else None
        if live is None:
            live = _nearest_upstream_vehicle(states, board_stop.seq)
        if not live:
            return {"seat_probability": .50, "confidence": "낮음", "source": "no_live_congestion", "wait_min": round(wait_min,1), "arrival_source": arrival_source, "from_seq": board_stop.seq, "to_seq": alight_stop.seq}
        if live.current_stop_order < board_stop.seq:
            try:
                flows = self.service.historical.build_route_flows(route_name=segment.route_name, hour=hour, route_stops=stops, service_days=settings.service_days, headway_min=term)
                p = predict(live, board_stop.seq, flows, queue_fraction=.5, effective_capacity=settings.effective_capacity)
                return {"seat_probability": p.seat_probability, "confidence": "중간" if is_first_segment else "낮음", "source": "live_plus_historical_first_segment" if is_first_segment else "future_transfer_live_proxy", "wait_min": round(wait_min,1), "arrival_source": arrival_source, "from_seq": board_stop.seq, "to_seq": alight_stop.seq, "predicted_congestion": p.predicted_target_congestion}
            except Exception:
                pass
        p = _seat_prob_from_live(live)
        return {"seat_probability": p if p is not None else .50, "confidence": "낮음", "source": "current_congestion_proxy", "wait_min": round(wait_min,1), "arrival_source": arrival_source, "from_seq": board_stop.seq, "to_seq": alight_stop.seq, "predicted_congestion": CONGESTION_LABEL.get(live.congestion_code, "정보없음")}

    def transfer_candidates(self, *, origin_x: float, origin_y: float, destination_x: float, destination_y: float, hour: int | None = None, max_paths: int = 8, max_transfers: int = 1):
        hour = datetime.now(SEOUL_TZ).hour if hour is None else int(hour)
        paths = self.api.fetch_bus_transfer_paths(start_x=origin_x, start_y=origin_y, end_x=destination_x, end_y=destination_y)
        options=[]; diagnostics=[]
        for idx, path in enumerate(paths[:max_paths]):
            transfer_count = max(len(path.segments)-1,0)
            if transfer_count > max_transfers:
                diagnostics.append({"path_index": idx, "status": "rejected_too_many_transfers", "transfer_count": transfer_count}); continue
            segment_details=[]; total_bus_minutes=0.0; expected_standing_minutes=0.0; total_wait_min=0.0
            for seg_idx, seg in enumerate(path.segments):
                comfort=self._segment_comfort(segment=seg,hour=hour,is_first_segment=(seg_idx==0))
                ride_min=max(float(seg.ride_time_min),1.0); p=min(max(float(comfort["seat_probability"]),0),1)
                total_bus_minutes += ride_min; expected_standing_minutes += ride_min*(1-p); total_wait_min += float(comfort["wait_min"])
                segment_details.append({"route_name":seg.route_name,"route_id":seg.route_id,"from_station_id":seg.from_station_id,"from_station_name":seg.from_station_name,"to_station_id":seg.to_station_id,"to_station_name":seg.to_station_name,"ride_time_min":round(ride_min,1),"seat_probability":round(p,4),"expected_standing_min":round(ride_min*(1-p),1),"confidence":comfort.get("confidence","낮음"),"seat_source":comfort.get("source","unknown"),"wait_min":comfort.get("wait_min"),"arrival_source":comfort.get("arrival_source",""),"predicted_congestion":comfort.get("predicted_congestion","정보없음")})
            if total_bus_minutes <= 0: continue
            expected_seated_fraction = min(max(1.0-(expected_standing_minutes/total_bus_minutes),0.0),1.0)
            in_vehicle_min = path.total_time_min if path.total_time_min and path.total_time_min > 0 else total_bus_minutes
            route_label = " → ".join(s.route_name for s in path.segments)
            vehicle_label = "transfer:" + ":".join(s.route_id for s in path.segments)
            options.append({"route_name":route_label,"route_id":"|".join(s.route_id for s in path.segments),"vehicle_id":vehicle_label,"wait_min":round(total_wait_min,1),"seat_probability":round(expected_seated_fraction,4),"in_vehicle_min":round(float(in_vehicle_min),1),"transfer_count":transfer_count,"predicted_congestion":segment_details[0].get("predicted_congestion","정보없음"),"target_stop_name":path.segments[-1].to_station_name,"origin_stop_name":path.segments[0].from_station_name,"expected_standing_min":round(expected_standing_minutes,1),"segments":segment_details,"path_distance":path.distance,"path_time_source":"pathinfo_total_time" if path.total_time_min else "sum_segment_times","confidence":"중간" if all(s["confidence"] != "낮음" for s in segment_details) else "낮음","option_type":"transfer" if transfer_count else "direct_path_api"})
        options.sort(key=lambda x:(x["wait_min"]+x["in_vehicle_min"],x["expected_standing_min"]))
        return {"hour":hour,"path_count":len(options),"options":options,"diagnostics":diagnostics}
