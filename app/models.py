from dataclasses import dataclass
from typing import Optional


@dataclass
class RouteInfo:
    route_id: str
    route_name: str
    start_station_name: str = ""
    end_station_name: str = ""
    term_min: Optional[float] = None
    first_bus_time: str = ""
    last_bus_time: str = ""
    company_name: str = ""
    route_type: Optional[int] = None


@dataclass
class StationInfo:
    station_id: str
    ars_id: str
    name: str
    x: Optional[float] = None
    y: Optional[float] = None


@dataclass
class ArrivalInfo:
    vehicle_id: str
    eta_seconds: Optional[int]
    message: str = ""
    current_station_name: str = ""
    plain_no: str = ""


@dataclass
class LiveBusState:
    route_id: str
    vehicle_id: str
    current_stop_order: int
    congestion_code: int
    is_full: int = 0
    bus_type: Optional[int] = None
    plain_no: str = ""
    section_id: str = ""
    stop_flag: str = ""


@dataclass
class RouteStop:
    seq: int
    station_id: str
    ars_id: str
    name: str
    direction: str = ""
    trans_yn: str = "N"
    route_type: Optional[int] = None
    begin_time: str = ""
    last_time: str = ""
    x: Optional[float] = None
    y: Optional[float] = None


@dataclass
class Prediction:
    current_congestion: str
    predicted_target_congestion: str
    seat_probability: float
    projected_load_score: float
    uncertainty_sigma: float
    stops_ahead: int
    expected_net_passenger_change: float
    model_version: str = "v5-generalized-heuristic"


@dataclass
class TransferSegment:
    route_id: str
    route_name: str
    from_station_id: str
    from_station_name: str
    from_x: Optional[float]
    from_y: Optional[float]
    to_station_id: str
    to_station_name: str
    to_x: Optional[float]
    to_y: Optional[float]
    ride_time_min: float


@dataclass
class TransferPath:
    distance: Optional[float]
    total_time_min: Optional[float]
    segments: list[TransferSegment]
