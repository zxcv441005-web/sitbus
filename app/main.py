from __future__ import annotations
from pathlib import Path
from urllib.parse import unquote
import shutil
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .config import settings
from .service import SeatService
from .decision import BusOption, DecisionPreferences, choose_best

app = FastAPI(title='SitBus', version='1.4.2')
svc = SeatService()
STATIC_DIR = Path(__file__).resolve().parent / 'static'
app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')

class DecisionOptionRequest(BaseModel):
    route_name: str
    route_id: str = ''
    vehicle_id: str
    wait_min: float = Field(ge=0)
    seat_probability: float = Field(ge=0, le=1)
    in_vehicle_min: float = Field(ge=0)
    transfer_count: int = Field(default=0, ge=0)
    predicted_congestion: str = ''
    target_stop_name: str = ''

class DecisionPreferencesRequest(BaseModel):
    standing_penalty_min: float = Field(default=12.0, ge=0)
    transfer_penalty_min: float = Field(default=8.0, ge=0)
    wait_weight: float = Field(default=1.25, gt=0)
    ride_weight: float = Field(default=1.0, gt=0)

class DecideRequest(BaseModel):
    options: list[DecisionOptionRequest]
    preferences: DecisionPreferencesRequest = DecisionPreferencesRequest()

class DirectJourneyRequest(BaseModel):
    origin_station_id: str
    origin_ars_id: str
    destination_station_id: str
    destination_ars_id: str
    hour: int | None = Field(default=None, ge=0, le=23)

class TransferJourneyRequest(BaseModel):
    origin_x: float
    origin_y: float
    destination_x: float
    destination_y: float
    hour: int | None = Field(default=None, ge=0, le=23)
    max_transfers: int = Field(default=1, ge=0, le=2)

class ApiKeySetupRequest(BaseModel):
    api_key: str = Field(min_length=4, max_length=4096)
    verify: bool = True

@app.get('/', include_in_schema=False)
def root():
    return FileResponse(STATIC_DIR / 'index.html')

@app.get('/manifest.webmanifest', include_in_schema=False)
def manifest():
    return FileResponse(STATIC_DIR / 'manifest.webmanifest', media_type='application/manifest+json')

@app.get('/service-worker.js', include_in_schema=False)
def service_worker():
    return FileResponse(STATIC_DIR / 'service-worker.js', media_type='application/javascript', headers={'Service-Worker-Allowed':'/'})

@app.get('/api/info')
def api_info():
    return {'service':'SitBus','version':'1.4.2','live_ready':svc.runtime_live_status()['live_ready'],'docs':'/docs'}

@app.get('/health')
def health():
    return {'ok':True, **svc.runtime_live_status()}

@app.get('/setup/status')
def setup_status():
    return {**svc.runtime_live_status(), 'latest_official_dataset': {
        'name':'2026년_버스노선별_정류장별_시간대별_승하차_인원_정보(08월).csv',
        'published_or_modified':'2026-09-06','size_mb':15.24,
        'dataset_page':'https://data.seoul.go.kr/dataList/OA-12913/F/1/datasetView.do'}}

@app.post('/setup/api-key')
def setup_api_key(req: ApiKeySetupRequest):
    old = svc.api.api_key
    try:
        normalized_key = unquote(req.api_key.strip())
        svc.set_runtime_api_key(normalized_key)
        if req.verify:
            svc.api.search_stations('서울역')
        return {'verified':bool(req.verify), **svc.runtime_live_status()}
    except Exception as e:
        svc.api.api_key = old
        raise HTTPException(400, f'API 키 검증 실패: {e}')

@app.post('/setup/historical-csv')
async def setup_historical_csv(file: UploadFile = File(...)):
    if not (file.filename or '').lower().endswith('.csv'):
        raise HTTPException(400, 'CSV 파일만 업로드할 수 있습니다.')
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / 'historical_latest.csv'
    try:
        with dest.open('wb') as out:
            shutil.copyfileobj(file.file, out)
        svc.set_runtime_historical_csv(str(dest))
        return {'uploaded':True,'size_bytes':dest.stat().st_size, **svc.runtime_live_status()}
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f'CSV 검증 실패: {e}')
    finally:
        await file.close()

@app.get('/stations/search')
def station_search(q: str = Query(min_length=1)):
    try:
        return [x.__dict__ for x in svc.discovery.station_search(q)]
    except Exception as e:
        raise HTTPException(503, str(e))

@app.post('/journeys/direct')
def direct_journey(req: DirectJourneyRequest):
    try:
        return svc.discovery.direct_candidates(origin_station_id=req.origin_station_id, origin_ars_id=req.origin_ars_id, destination_station_id=req.destination_station_id, destination_ars_id=req.destination_ars_id, hour=req.hour)
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post('/journeys/transfers')
def transfer_journey(req: TransferJourneyRequest):
    try:
        return svc.transfer_discovery.transfer_candidates(origin_x=req.origin_x, origin_y=req.origin_y, destination_x=req.destination_x, destination_y=req.destination_y, hour=req.hour, max_transfers=req.max_transfers)
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post('/decide')
def decide(req: DecideRequest):
    try:
        options = [BusOption(**x.model_dump()) for x in req.options]
        pref = DecisionPreferences(**req.preferences.model_dump())
        return choose_best(options, pref)
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post('/choices')
def choices(payload: dict):
    return {'ok':True,'stored':False}

@app.post('/decisions')
def decisions(payload: dict):
    return {'decision_id':'web-demo','stored':False}

@app.post('/journeys/outcome')
def journey_outcome(payload: dict):
    return {'ok':True,'stored':False}
