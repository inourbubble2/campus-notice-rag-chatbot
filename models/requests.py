# models/requests.py
from pydantic import BaseModel
from typing import List


class IngestByIdsRequest(BaseModel):
    ids: List[int]


class IngestByDateRangeRequest(BaseModel):
    from_date: str  # YYYY-MM-DD format
    to_date: str    # YYYY-MM-DD format