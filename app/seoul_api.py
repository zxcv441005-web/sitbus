from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import httpx

from .models import (
    LiveBusState,
    RouteStop,
    RouteInfo,
    StationInfo,
    ArrivalInfo,
    TransferSegment,
    TransferPath,
)


class SeoulBusAPI:
    ROUTE_SEARCH_URL = "http://ws.bus.go.kr/api/rest/busRouteInfo/getBusRouteList"
    ROUTE_STOPS_URL = "http://ws.bus.go.kr/api/rest/busRouteInfo/getStaionByRoute"
    POSITIONS_URL = "http://ws.bus.go.kr/api/rest/buspos/getBusPosByRtid"

    STATION_SEARCH_URL = "http://ws.bus.go.kr/api/rest/stationinfo/getStationByName"
    STATION_NEARBY_URL = "http://ws.bus.go.kr/api/rest/stationinfo/getStationByPos"
    ROUTES_BY_STATION_URL = "http://ws.bus.go.kr/api/rest/stationinfo/getRouteByStation"
    ARRIVAL_BY_ROUTE_URL = "http://ws.bus.go.kr/api/rest/arrive/getArrInfoByRoute"
    PATH_INFO_BY_BUS_URL = "http://ws.bus.go.kr/api/rest/pathinfo/getPathInfoByBus"

    def __init__(self, api_key: str, client: httpx.Client | None = None):
        self.api_key = api_key
        self.client = client or httpx.Client(timeout=15.0)

    @staticmethod
    def _txt(node, tag: str, default: str = "") -> str:
        x = node.find(tag)
        if x is None or x.text is None:
            return default
        return x.text.strip()

    @staticmethod
    def _first_txt(node, tags, default=""):
        for tag in tags:
            v = SeoulBusAPI._txt(node, tag, "")
            if v:
                return v
        return default

    @staticmethod
    def _int(value: str, default=0):
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _float_or_none(value: str):
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _ensure_success(root: ET.Element):
        code = root.findtext(".//headerCd")
        msg = root.findtext(".//headerMsg")
        if code not in (None, "", "0"):
            raise RuntimeError(f"Seoul bus API error {code}: {msg}")

    def _get_xml(self, url: str, params: dict) -> ET.Element:
        if not self.api_key:
            raise RuntimeError("SEOUL_BUS_API_KEY is not configured.")
        params = {"ServiceKey": self.api_key, **params}
        r = self.client.get(url, params=params)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        self._ensure_success(root)
        return root

    def search_stations(self, query: str) -> list[StationInfo]:
        query = str(query).strip()
        if not query:
            return []
        root = self._get_xml(self.STATION_SEARCH_URL, {"stSrch": query})
        out = []
        seen = set()
        for item in root.findall(".//itemList"):
            station_id = self._first_txt(item, ["stId", "station", "stationId"])
            ars_id = self._first_txt(item, ["arsId", "stationNo"])
            name = self._first_txt(item, ["stNm", "stationNm"])
            if not station_id or not name:
                continue
            key = (station_id, ars_id)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                StationInfo(
                    station_id=station_id,
                    ars_id=ars_id,
                    name=name,
                    x=self._float_or_none(
                        self._first_txt(item, ["tmX", "gpsX", "posX"])
                    ),
                    y=self._float_or_none(
                        self._first_txt(item, ["tmY", "gpsY", "posY"])
                    ),
                )
            )
        return out

    def nearby_stations(
        self,
        *,
        longitude: float,
        latitude: float,
        radius: int = 500,
    ) -> list[dict]:
        """Return usable Seoul bus stops near a WGS84 position.

        The station-info API accepts longitude as tmX and latitude as tmY.
        ARS id 0 / missing stops are excluded because downstream route lookup
        requires a usable ARS id.
        """
        radius = max(50, min(int(radius), 1000))
        root = self._get_xml(
            self.STATION_NEARBY_URL,
            {
                "tmX": float(longitude),
                "tmY": float(latitude),
                "radius": radius,
            },
        )
        out = []
        seen = set()
        for item in root.findall(".//itemList"):
            station_id = self._first_txt(item, ["stationId", "stId", "station"])
            ars_id = self._first_txt(item, ["arsId", "stationNo"])
            name = self._first_txt(item, ["stationNm", "stNm"])
            if not station_id or not name or ars_id in ("", "0"):
                continue
            key = (station_id, ars_id)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "station_id": station_id,
                    "ars_id": ars_id,
                    "name": name,
                    "x": self._float_or_none(
                        self._first_txt(item, ["gpsX", "tmX", "posX"])
                    ),
                    "y": self._float_or_none(
                        self._first_txt(item, ["gpsY", "tmY", "posY"])
                    ),
                    "distance_m": self._float_or_none(
                        self._first_txt(item, ["dist", "distance"])
                    ),
                }
            )
        out.sort(
            key=lambda x: (
                x["distance_m"] is None,
                x["distance_m"] if x["distance_m"] is not None else 10**9,
            )
        )
        return out

    def routes_by_station(self, ars_id: str) -> list[RouteInfo]:
        root = self._get_xml(self.ROUTES_BY_STATION_URL, {"arsId": ars_id})
        out = []
        seen = set()
        for item in root.findall(".//itemList"):
            rid = self._first_txt(item, ["busRouteId", "routeId"])
            name = self._first_txt(item, ["busRouteNm", "rtNm"])
            if not rid or not name or rid in seen:
                continue
            seen.add(rid)
            out.append(
                RouteInfo(
                    route_id=rid,
                    route_name=name,
                    start_station_name=self._first_txt(item, ["stBegin", "stStationNm"]),
                    end_station_name=self._first_txt(item, ["stEnd", "edStationNm"]),
                    term_min=self._float_or_none(
                        self._first_txt(item, ["term", "interval"])
                    ),
                    first_bus_time=self._first_txt(item, ["firstBusTm"]),
                    last_bus_time=self._first_txt(item, ["lastBusTm"]),
                    route_type=self._int(self._first_txt(item, ["routeType"]), None),
                )
            )
        return out

    def search_routes(self, query: str) -> list[RouteInfo]:
        query = str(query).strip()
        if not query:
            return []
        root = self._get_xml(self.ROUTE_SEARCH_URL, {"strSrch": query})
        out = []
        for item in root.findall(".//itemList"):
            route_id = self._txt(item, "busRouteId")
            route_name = self._txt(item, "busRouteNm")
            if route_id and route_name:
                out.append(
                    RouteInfo(
                        route_id=route_id,
                        route_name=route_name,
                        start_station_name=self._txt(item, "stStationNm"),
                        end_station_name=self._txt(item, "edStationNm"),
                        term_min=self._float_or_none(self._txt(item, "term")),
                        first_bus_time=self._txt(item, "firstBusTm"),
                        last_bus_time=self._txt(item, "lastBusTm"),
                        company_name=self._txt(item, "corpNm"),
                        route_type=self._int(self._txt(item, "routeType"), None),
                    )
                )
        return out

    def resolve_route(self, route_name: str) -> RouteInfo:
        results = self.search_routes(route_name)
        exact = [r for r in results if r.route_name.strip() == route_name.strip()]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            raise LookupError(
                f"노선번호 {route_name!r}에 정확히 일치하는 노선이 여러 개입니다: "
                + ", ".join(f"{x.route_name}({x.route_id})" for x in exact)
            )
        raise LookupError(
            f"서울 노선번호 {route_name!r}의 정확한 일치 항목을 찾지 못했습니다."
        )

    def fetch_route_positions(self, route_id: str) -> list[LiveBusState]:
        root = self._get_xml(self.POSITIONS_URL, {"busRouteId": route_id})
        out = []
        for item in root.findall(".//itemList"):
            sect = self._txt(item, "sectOrd")
            cong = self._txt(item, "congetion", "0")
            veh = self._txt(item, "vehId")
            if not veh or not sect:
                continue
            congestion = self._int(cong, 0)
            if congestion not in {0, 3, 4, 5, 6}:
                congestion = 0
            out.append(
                LiveBusState(
                    route_id=route_id,
                    vehicle_id=veh,
                    current_stop_order=self._int(sect),
                    congestion_code=congestion,
                    is_full=self._int(self._txt(item, "isFullFlag", "0")),
                    bus_type=self._int(self._txt(item, "busType", "0")),
                    plain_no=self._txt(item, "plainNo"),
                    section_id=self._txt(item, "sectionId"),
                    stop_flag=self._txt(item, "stopFlag"),
                )
            )
        return out

    def fetch_route_stops(self, route_id: str) -> list[RouteStop]:
        root = self._get_xml(self.ROUTE_STOPS_URL, {"busRouteId": route_id})
        out = []
        for item in root.findall(".//itemList"):
            seq = self._txt(item, "seq")
            station = self._txt(item, "station")
            if not seq or not station:
                continue
            out.append(
                RouteStop(
                    seq=self._int(seq),
                    station_id=station,
                    ars_id=self._first_txt(item, ["arsId", "stationNo"]),
                    name=self._txt(item, "stationNm"),
                    direction=self._txt(item, "direction"),
                    trans_yn=self._txt(item, "transYn", "N"),
                    route_type=self._int(self._txt(item, "routeType"), None),
                    begin_time=self._txt(item, "beginTm"),
                    last_time=self._txt(item, "lastTm"),
                    x=self._float_or_none(self._first_txt(item, ["gpsX", "tmX"])),
                    y=self._float_or_none(self._first_txt(item, ["gpsY", "tmY"])),
                )
            )
        return sorted(out, key=lambda s: s.seq)

    @staticmethod
    def _eta_seconds_from_message(message: str):
        if not message:
            return None
        m = re.search(r"(?:(\d+)분)?\s*(?:(\d+)초)?후", message)
        if m and (m.group(1) or m.group(2)):
            return int(m.group(1) or 0) * 60 + int(m.group(2) or 0)
        if "곧" in message or "도착" == message.strip():
            return 30
        return None

    def fetch_arrivals_by_route(
        self, station_id: str, route_id: str, ord_seq: int
    ) -> list[ArrivalInfo]:
        root = self._get_xml(
            self.ARRIVAL_BY_ROUTE_URL,
            {"stId": station_id, "busRouteId": route_id, "ord": int(ord_seq)},
        )
        item = root.find(".//itemList")
        if item is None:
            return []
        out = []
        for idx in (1, 2):
            message = self._txt(item, f"arrmsg{idx}")
            veh = self._first_txt(
                item, [f"vehId{idx}", f"vehId{idx}Sec", f"plainNo{idx}"]
            )
            eta_raw = self._first_txt(item, [f"exps{idx}", f"traTime{idx}"])
            eta = self._int(eta_raw, None) if eta_raw else None
            if eta is None:
                eta = self._eta_seconds_from_message(message)
            if not message and not veh and eta is None:
                continue
            out.append(
                ArrivalInfo(
                    vehicle_id=veh,
                    eta_seconds=eta,
                    message=message,
                    current_station_name=self._txt(item, f"stationNm{idx}"),
                    plain_no=self._txt(item, f"plainNo{idx}"),
                )
            )
        return out

    @staticmethod
    def _path_time_to_minutes(value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        if v < 0:
            return None
        return v if v <= 240 else v / 60.0

    def fetch_bus_transfer_paths(
        self, *, start_x: float, start_y: float, end_x: float, end_y: float
    ) -> list[TransferPath]:
        root = self._get_xml(
            self.PATH_INFO_BY_BUS_URL,
            {"startX": start_x, "startY": start_y, "endX": end_x, "endY": end_y},
        )
        msg = root.find(".//msgBody")
        if msg is None:
            return []
        paths = []
        for item in msg.findall("itemList"):
            distance = self._float_or_none(self._txt(item, "distance"))
            total_time = self._path_time_to_minutes(self._txt(item, "time"))
            segments = []
            for seg in item.findall("pathList"):
                rid = self._txt(seg, "routeId")
                rnm = self._txt(seg, "routeNm")
                fid = self._txt(seg, "fid")
                fname = self._txt(seg, "fname")
                tid = self._txt(seg, "tid")
                tname = self._txt(seg, "tname")
                if not rid or not rnm or not fid or not tid:
                    continue
                segments.append(
                    TransferSegment(
                        route_id=rid,
                        route_name=rnm,
                        from_station_id=fid,
                        from_station_name=fname,
                        from_x=self._float_or_none(self._txt(seg, "fx")),
                        from_y=self._float_or_none(self._txt(seg, "fy")),
                        to_station_id=tid,
                        to_station_name=tname,
                        to_x=self._float_or_none(self._txt(seg, "tx")),
                        to_y=self._float_or_none(self._txt(seg, "ty")),
                        ride_time_min=float(
                            self._path_time_to_minutes(self._txt(seg, "time")) or 0.0
                        ),
                    )
                )
            if segments:
                paths.append(
                    TransferPath(
                        distance=distance,
                        total_time_min=total_time,
                        segments=segments,
                    )
                )
        return paths
