from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from ingest import ingest_by_ids, ingest_by_date_range

app = FastAPI()


class IngestByIdsRequest(BaseModel):
    ids: List[int]


class IngestByDateRangeRequest(BaseModel):
    from_date: str  # YYYY-MM-DD format
    to_date: str    # YYYY-MM-DD format


@app.post("/ingest")
async def ingest_announcements(request: IngestByIdsRequest):
    try:
        ingest_by_ids(request.ids)
        return {"message": f"Successfully ingested {len(request.ids)} announcements", "ids": request.ids}
    except Exception as e:
        return {"error": str(e)}


@app.post("/ingest/date-range")
async def ingest_announcements_by_date(request: IngestByDateRangeRequest):
    try:
        ingest_by_date_range(request.from_date, request.to_date)
        return {"message": f"Successfully ingested announcements from {request.from_date} to {request.to_date}"}
    except Exception as e:
        return {"error": str(e)}
