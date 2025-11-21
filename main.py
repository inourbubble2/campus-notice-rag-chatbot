import json
import logging

from fastapi import FastAPI
from ingest import ingest_by_ids, ingest_by_date_range
from parse import process_announcements_by_ids, process_announcements_by_date_range
from models import IngestByIdsRequest, IngestByDateRangeRequest, ChatRequest
from chat.chat_graph import app as chat_graph_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


@app.post("/parse")
async def parse_announcements(request: IngestByIdsRequest):
    try:
        results = await process_announcements_by_ids(request.ids)
        return {
            "message": f"Successfully parsed {len(results)} announcements",
            "ids": request.ids,
            "results": [
                {
                    "announcement_id": r.announcement_id,
                    "title": r.title,
                    "has_cleaned_text": bool(r.cleaned_text),
                    "has_ocr_text": bool(r.ocr_text),
                    "tags": r.tags,
                    "target_departments": r.target_departments,
                    "target_grades": r.target_grades,
                    "application_period": {
                        "start": r.application_period_start.isoformat() if r.application_period_start else None,
                        "end": r.application_period_end.isoformat() if r.application_period_end else None,
                    },
                    "error": r.error_message,
                }
                for r in results
            ],
        }
    except Exception as e:
        return {"error": str(e), "success": False}


@app.post("/parse/date-range")
async def parse_announcements_by_date(request: IngestByDateRangeRequest):
    try:
        results = await process_announcements_by_date_range(request.from_date, request.to_date)
        return {
            "message": f"Successfully parsed {len(results)} announcements from {request.from_date} to {request.to_date}",
            "date_range": {"from_date": request.from_date, "to_date": request.to_date},
            "results": [
                {
                    "announcement_id": r.announcement_id,
                    "title": r.title,
                    "has_cleaned_text": bool(r.cleaned_text),
                    "has_ocr_text": bool(r.ocr_text),
                    "tags": r.tags,
                    "target_departments": r.target_departments,
                    "target_grades": r.target_grades,
                    "application_period": {
                        "start": r.application_period_start.isoformat() if r.application_period_start else None,
                        "end": r.application_period_end.isoformat() if r.application_period_end else None,
                    },
                    "error": r.error_message,
                }
                for r in results
            ],
        }
    except Exception as e:
        return {"error": str(e), "success": False}


@app.post("/chat")
async def chat(request: ChatRequest):
    """RAG 기반 캠퍼스 공지사항 챗봇 API"""
    result = await chat_graph_app.ainvoke(
        {
            "question": request.question,
            "docs": [],
            "answer": None,
            "rewrite": None,
            "attempt": 0,
            "validate": None,
            "guardrail": None,
        },
        config={"configurable": {"thread_id": request.conversation_id}},
    )

    logger.info("=== Guardrail ===")
    logger.info(json.dumps(result.get("guardrail", {}), ensure_ascii=False, indent=2))
    logger.info("=== Rewritten ===")
    logger.info(json.dumps(result.get("rewrite", {}), ensure_ascii=False, indent=2))
    logger.info("=== Validate ===")
    logger.info(json.dumps(result.get("validate", {}), ensure_ascii=False, indent=2))
    logger.info("=== Answer ===")
    logger.info(result.get("answer","(no answer)"))

    # guardrail 차단 체크
    if result.get("guardrail", {}).get("policy") == "BLOCK":
        logger.warning(f"Question blocked: {request.question}")
        return {
            "answer": "죄송합니다. 해당 질문은 대학 공지사항 관련 질문이 아니거나 부적절한 내용이 포함되어 있습니다.",
            "blocked": True,
        }

    return {"answer": result.get("answer", "(답변 생성 실패)")}
