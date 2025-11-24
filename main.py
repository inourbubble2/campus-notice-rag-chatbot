import logging
import json
import time
from datetime import datetime

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
    start_time = time.time()

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

    end_time = time.time()
    total_latency_ms = (end_time - start_time) * 1000

    logger.info(f"Request {request.conversation_id} processed. Latency: {total_latency_ms:.2f}ms")

    log_data = {
        "metadata": {
            "request_id": request.conversation_id,
            "timestamp": datetime.now().isoformat(),
        },
        "query": {
            "raw": request.question,
            "rewritten": result.get("rewrite", {}).get("query")
        },
        "retrieval": {
            "results": [
                {
                    "doc_id": d.metadata.get("announcement_id"),
                    "score": d.metadata.get("score"),
                    "url": d.metadata.get("url"),
                    "title": d.metadata.get("title")
                } for d in result.get("docs", [])
            ]
        },
        "context_used": [
            {
                "doc_id": d.metadata.get("announcement_id"),
                "page_content": d.page_content
            } for d in result.get("docs", [])
        ],
        "generation": {
            "model": "gpt-4o-mini",
            "first_token_latency_ms": None,
            "total_latency_ms": round(total_latency_ms, 2),
            "final_answer": result.get("answer")
        }
    }

    with open("chat_logs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data, ensure_ascii=False) + "\n")

    if result.get("guardrail", {}).get("policy") == "BLOCK":
        logger.warning(f"Question blocked: {request.question}")
        return {
            "answer": "죄송합니다. 해당 질문은 대학 공지사항 관련 질문이 아니거나 부적절한 내용이 포함되어 있습니다.",
            "contexts": []
        }

    return {
        "answer": result.get("answer", ""),
        "contexts": [d.page_content for d in result.get("docs", [])],
        "urls": [d.metadata.get("url") for d in result.get("docs", [])]
    }
