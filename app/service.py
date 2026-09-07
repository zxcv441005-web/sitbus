from __future__ import annotations
from pathlib import Path
from .config import settings
from .historical import HistoricalFlowStore
from .seoul_api import SeoulBusAPI
from .journey_discovery import DirectJourneyDiscovery
from .transfer_discovery import TransferJourneyDiscovery

class SeatService:
    def __init__(self):
        self.api = SeoulBusAPI(settings.api_key)
        self._historical = None
        self.runtime_historical_csv_path = settings.historical_csv_path
        if not self.runtime_historical_csv_path:
            p = Path(settings.data_dir) / 'historical_latest.csv'
            if p.exists():
                self.runtime_historical_csv_path = str(p)
        self.discovery = DirectJourneyDiscovery(self)
        self.transfer_discovery = TransferJourneyDiscovery(self)

    @property
    def historical(self):
        if self._historical is None:
            self._historical = HistoricalFlowStore(self.runtime_historical_csv_path)
        return self._historical

    def set_runtime_api_key(self, api_key: str):
        api_key = str(api_key or '').strip()
        if not api_key:
            raise ValueError('API 키가 비어 있습니다.')
        self.api.api_key = api_key

    def set_runtime_historical_csv(self, csv_path: str):
        p = Path(str(csv_path or '').strip())
        if not p.exists():
            raise FileNotFoundError(str(p))
        self._historical = HistoricalFlowStore(str(p))
        self.runtime_historical_csv_path = str(p)

    def runtime_live_status(self):
        csv_ok = bool(self.runtime_historical_csv_path and Path(self.runtime_historical_csv_path).exists())
        return {
            'api_key_configured': bool(self.api.api_key),
            'historical_csv_configured': csv_ok,
            'historical_csv_path': self.runtime_historical_csv_path or '',
            'live_ready': bool(self.api.api_key and csv_ok),
        }
