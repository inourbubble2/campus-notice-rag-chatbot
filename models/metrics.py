from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging
import time

logger = logging.getLogger(__name__)

@dataclass
class OCRMetrics:
    """OCR 요청에 대한 메트릭 정보."""
    duration_ms: float
    image_size_kb: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    result_length: int
    success: bool = True
    error_message: str = ""

class OCRMetricsAggregator:
    """OCR 메트릭 집계 및 관리."""

    def __init__(self):
        self.metrics_list: List[OCRMetrics] = []

    def add_metrics(self, metrics: OCRMetrics) -> None:
        """메트릭 추가."""
        self.metrics_list.append(metrics)

    def get_summary(self) -> Dict[str, Any]:
        """전체 메트릭 요약 반환."""
        if not self.metrics_list:
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_duration_ms": 0,
                "total_image_size_kb": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
            }

        successful_metrics = [m for m in self.metrics_list if m.success]

        total_duration = sum(m.duration_ms for m in self.metrics_list)
        total_image_size = sum(m.image_size_kb for m in self.metrics_list)
        total_input_tokens = sum(m.input_tokens for m in successful_metrics)
        total_output_tokens = sum(m.output_tokens for m in successful_metrics)
        total_tokens = sum(m.total_tokens for m in successful_metrics)

        return {
            "total_requests": len(self.metrics_list),
            "successful_requests": len(successful_metrics),
            "failed_requests": len(self.metrics_list) - len(successful_metrics),
            "total_duration_ms": total_duration,
            "total_image_size_kb": total_image_size,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
        }

    def reset(self) -> None:
        """메트릭 리스트 초기화."""
        self.metrics_list.clear()


@dataclass
class EmbeddingMetrics:
    """Embedding 처리 메트릭."""
    announcement_count: int
    document_chunks_count: int
    total_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        """API 응답용 딕셔너리 변환."""
        return {
            "announcement_count": self.announcement_count,
            "document_chunks_count": self.document_chunks_count,
            "total_tokens": self.total_tokens,
        }

@dataclass
class IngestResult:
    """Ingest 작업의 전체 결과."""
    embedding_metrics: EmbeddingMetrics
    ocr_metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """API 응답용 딕셔너리 변환."""
        return {
            "embedding_metrics": self.embedding_metrics.to_dict(),
            "ocr_metrics": self.ocr_metrics,
        }

# 전역 메트릭 집계기
_global_ocr_aggregator = OCRMetricsAggregator()

def get_ocr_metrics_aggregator() -> OCRMetricsAggregator:
    """전역 OCR 메트릭 집계기 반환."""
    return _global_ocr_aggregator
