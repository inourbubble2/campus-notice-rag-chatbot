# services/html_processing_service.py
"""
HTML 처리 및 OCR 오케스트레이션 서비스
- HTML → 일반 텍스트 추출
- HTML 내 이미지 → OCR 텍스트 추출
"""
import re
import unicodedata
import asyncio
import aiohttp
import ssl
import logging
import base64
from bs4 import BeautifulSoup
from bs4.element import Tag
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from services.ocr_service import extract_text_from_image

__all__ = ["get_plain_text", "get_ocr_text"]

logger = logging.getLogger(__name__)


# ========== HTML → 텍스트 변환 ==========

def _clean_text(s: str) -> str:
    """연속 개행/공백 압축 + trim."""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{2,}", "\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    lines = [line.strip() for line in s.splitlines() if line.strip()]
    return "\n".join(lines)


def _html_to_text(html: str) -> str:
    """
    <br>/<p>/<div>는 개행으로 치환, 나머지 태그는 제거.
    SmartEditor 특유의 제어문자·?·&nbsp;도 정제.
    """
    soup = BeautifulSoup(html or "", "lxml")

    # 스크립트·스타일 제거
    for t in soup(["script", "style"]):
        t.decompose()

    # 주요 블록 태그를 개행 기준으로
    for tag in soup.find_all(["br", "p", "div"]):
        tag.insert_before("\n")

    text = soup.get_text()

    # 유니코드 정규화
    text = unicodedata.normalize("NFC", text)

    # 제어문자/공백/nbsp 정리
    text = re.sub(r"[\u200b\u200c\u200d\uFEFF]", "", text)  # zero-width
    text = text.replace("\xa0", " ")

    # SmartEditor 잔류 물음표(깨진 공백) 정리
    text = re.sub(r"(?<=\s)\?(?=\s)", " ", text)
    text = re.sub(r"\s*\?\s*", " ", text)
    text = re.sub(r"\?{2,}", "?", text)

    return _clean_text(text)


def get_plain_text(html: str) -> str:
    """HTML → 텍스트 변환 + 구문 보정."""
    s = _html_to_text(html)

    # 콜론 앞뒤 이상 개행 보정: "행사명\n:\n" → "행사명: "
    s = re.sub(r"\n\s*:\s*\n", ": ", s)
    s = re.sub(r"\s*:\s*\n", ": ", s)
    s = re.sub(r"\n\s*:\s*", ": ", s)

    # 영문/숫자 사이 개행 제거: "2025\n학년도" → "2025 학년도"
    s = re.sub(r"(\w)\n(\w)", r"\1 \2", s)

    return _clean_text(s)


# ========== HTML 내 이미지 → OCR ==========

async def get_ocr_text(html: str) -> tuple[str, str | None]:
    """
    HTML에서 모든 이미지를 찾아 병렬로 OCR 처리하고 결과 반환.

    Returns:
        tuple[str, str | None]: (OCR 텍스트, 에러 요약)
            - OCR 텍스트: 성공한 이미지들의 OCR 결과
            - 에러 요약: 실패한 이미지가 있을 경우 에러 메시지, 없으면 None
    """
    soup = BeautifulSoup(html, "html.parser")

    # 기존 텍스트 추출 (img 태그 제외)
    for img in soup.find_all("img"):
        img.decompose()
    original_text = soup.get_text(strip=True)

    # 다시 파싱해서 img 태그들로부터 OCR 텍스트 추출 (병렬 처리)
    soup = BeautifulSoup(html, "html.parser")
    img_tags = soup.find_all("img")

    if img_tags:
        # 모든 이미지를 병렬로 처리
        ocr_tasks = [_extract_text_from_image(img) for img in img_tags]
        ocr_results = await asyncio.gather(*ocr_tasks, return_exceptions=True)

        # 성공/실패 분류
        valid_ocr_results = []
        failed_images = []

        for i, result in enumerate(ocr_results):
            if isinstance(result, str):
                if result.strip():
                    valid_ocr_results.append(result)
                else:
                    logger.warning(f"OCR returned empty result for image {i+1}")
            elif isinstance(result, Exception):
                error_msg = f"Image {i+1}: {type(result).__name__}: {str(result)}"
                failed_images.append(error_msg)
                logger.error(
                    f"OCR failed for image {i+1}: {type(result).__name__}: {str(result)}",
                    exc_info=result
                )

        # 에러 요약 생성
        error_summary = None
        if failed_images:
            if len(valid_ocr_results) == 0:
                # 모든 이미지 실패
                error_summary = f"All {len(img_tags)} images failed OCR: {'; '.join(failed_images[:3])}"
                if len(failed_images) > 3:
                    error_summary += f" (and {len(failed_images) - 3} more)"
            else:
                # 일부 이미지 실패
                error_summary = f"Partial OCR failure: {len(failed_images)}/{len(img_tags)} images failed: {'; '.join(failed_images[:2])}"
                if len(failed_images) > 2:
                    error_summary += f" (and {len(failed_images) - 2} more)"
                logger.warning(error_summary)
    else:
        valid_ocr_results = []
        error_summary = None

    # OCR 텍스트 + 에러 요약 반환
    return "\n".join(valid_ocr_results), error_summary


async def _extract_text_from_image(img: Tag) -> str:
    """이미지 태그에서 OCR 텍스트를 추출."""
    src = (img.get("src") or "").strip()
    if not src:
        raise ValueError(f"Invalid image src: {src}")

    ocr_txt = await _ocr_image_from_url(src)
    if ocr_txt.strip():
        return ocr_txt
    else:
        logger.warning(f"OCR returned empty text for image: {src}")
        return f"[image:{src}-text:(empty)]"


@retry(
    stop=stop_after_attempt(3),  # 최대 3회 시도
    wait=wait_exponential(multiplier=1, min=1, max=5),  # 1초, 2초, 4초 대기
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, TimeoutError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
async def _ocr_image_from_url(url: str) -> str:
    """URL에서 이미지를 다운로드하고 OCR 수행."""
    timeout = aiohttp.ClientTimeout(total=30)  # 10초 → 30초로 증가

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    connector = aiohttp.TCPConnector(ssl=ssl_context)
    async with aiohttp.ClientSession(
        timeout=timeout,
        headers={"User-Agent": "uos-rag-ingest/1.0"},
        connector=connector
    ) as session:
        logger.info(f"이미지 다운로드 시작: {url[:100]}...")
        async with session.get(url) as resp:
            resp.raise_for_status()

            # 이미지를 base64로 인코딩
            img_data = await resp.read()
            img_size_kb = len(img_data) / 1024
            logger.info(f"이미지 다운로드 완료: {img_size_kb:.2f}KB")
            img_base64 = base64.b64encode(img_data).decode()

            return await extract_text_from_image(img_base64)
