from fastapi import FastAPI
from ingest import ingest_by_ids, ingest_by_date_range
from models import IngestByIdsRequest, IngestByDateRangeRequest

app = FastAPI()


@app.post("/ingest")
async def ingest_announcements(request: IngestByIdsRequest):
    try:
        result = await ingest_by_ids(request.ids)
        response = result.to_dict()
        response["message"] = f"Successfully ingested {len(request.ids)} announcements"
        response["ids"] = request.ids
        return response
    except Exception as e:
        return {"error": str(e), "success": False}


@app.post("/ingest/date-range")
async def ingest_announcements_by_date(request: IngestByDateRangeRequest):
    try:
        result = await ingest_by_date_range(request.from_date, request.to_date)
        response = result.to_dict()
        response["message"] = f"Successfully ingested announcements from {request.from_date} to {request.to_date}"
        response["date_range"] = {"from_date": request.from_date, "to_date": request.to_date}
        return response
    except Exception as e:
        return {"error": str(e), "success": False}
