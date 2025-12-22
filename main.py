import logging
import json
import time
from datetime import datetime

from fastapi import FastAPI
from ingest import ingest_by_ids, ingest_by_date_range
from parse import process_announcements_by_ids, process_announcements_by_date_range
from models import IngestByIdsRequest, IngestByDateRangeRequest, ChatRequest, ChatResponse
from chat.chat_graph import app as chat_graph_app
from chat.schema import RAGState
from fastapi import Depends
from services.ocr.base import BaseOCRService
from app.deps import get_ocr_service_provider
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import HumanMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


@app.post("/ingest")
async def ingest_announcements(request: IngestByIdsRequest):
    try:
        result = await ingest_by_ids(request.ids)
        response = result
        response["message"] = f"Successfully ingested {len(request.ids)} announcements"
        response["ids"] = request.ids
        return response
    except Exception as e:
        return {"error": str(e), "success": False}


@app.post("/ingest/date-range")
async def ingest_announcements_by_date(request: IngestByDateRangeRequest):
    try:
        result = await ingest_by_date_range(request.from_date, request.to_date)
        response = result
        response["message"] = f"Successfully ingested announcements from {request.from_date} to {request.to_date}"
        response["date_range"] = {"from_date": request.from_date, "to_date": request.to_date}
        return response
    except Exception as e:
        return {"error": str(e), "success": False}


@app.post("/parse")
async def parse_announcements(
    request: IngestByIdsRequest,
    ocr_service: BaseOCRService = Depends(get_ocr_service_provider)
):
    try:
        results = await process_announcements_by_ids(request.ids, ocr_service)
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
async def parse_announcements_by_date(
    request: IngestByDateRangeRequest,
    ocr_service: BaseOCRService = Depends(get_ocr_service_provider)
):
    try:
        results = await process_announcements_by_date_range(request.from_date, request.to_date, ocr_service)
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


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """RAG 기반 캠퍼스 공지사항 챗봇 API"""
    start_time = time.time()

    usage_callback = UsageMetadataCallbackHandler()

    final_state = None
    async for mode, payload in chat_graph_app.astream(
        {
            "messages": [HumanMessage(content=request.question)],
            "docs": [],
            "answer": None,
            "rewrite": None,
            "validation": None,
            "guardrail": None,
            "attempt": 0,
        },
        config={
            "configurable": {"thread_id": request.conversation_id},
            "callbacks": [usage_callback]
        },
        stream_mode=["updates", "values"]
    ):
        if mode == "updates":
            for node_name, updates in payload.items():
                logger.info(f"Node '{node_name}' update: {updates}")
        elif mode == "values":
            final_state = payload

    result = final_state

    state = RAGState(**result)

    end_time = time.time()
    total_latency_ms = (end_time - start_time) * 1000

    token_usage = usage_callback.usage_metadata
    logger.info(f"Request {request.conversation_id} processed. Latency: {total_latency_ms:.2f}ms. Token Usage: {token_usage}")

    rewritten_query = state.rewrite.query if state.rewrite else None

    log_data = {
        "metadata": {
            "request_id": request.conversation_id,
            "timestamp": datetime.now().isoformat(),
        },
        "query": {
            "raw": request.question,
            "rewritten": rewritten_query,
        },
        "retrieval": {
            "results": [
                {
                    "doc_id": d.metadata.get("announcement_id"),
                    "score": d.metadata.get("score"),
                    "url": d.metadata.get("url"),
                    "title": d.metadata.get("title")
                } for d in state.docs
            ]
        },
        "context_used": [
            {
                "doc_id": d.metadata.get("announcement_id"),
                "page_content": d.page_content
            } for d in state.docs
        ],
        "generation": {
            "model": "gpt-4o-mini",
            "first_token_latency_ms": None,
            "total_latency_ms": round(total_latency_ms, 2),
            "final_answer": state.answer,
            "token_usage": token_usage
        }
    }

    with open("chat_logs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data, ensure_ascii=False) + "\n")

    if state.guardrail and state.guardrail.policy == "BLOCK":
        return ChatResponse(
            answer="죄송합니다. 해당 질문은 대학 공지사항 관련 질문이 아니거나 부적절한 내용이 포함되어 있습니다.",
            contexts=[],
            urls=[]
        )

    return ChatResponse(
        answer=state.answer or "",
        contexts=[d.page_content for d in state.docs],
        urls=[d.metadata.get("url") for d in state.docs if d.metadata.get("url")]
    )
