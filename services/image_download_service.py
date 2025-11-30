import base64
import ssl
import logging
import aiohttp
import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(3),  # 최대 3회 시도
    wait=wait_exponential(multiplier=1, min=1, max=5),  # 1초, 2초, 4초 대기
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, TimeoutError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
async def download_image_as_base64(url: str) -> str:
    """
    URL에서 이미지를 다운로드하고 Base64 문자열로 반환.

    Args:
        url: 이미지 URL

    Returns:
        Base64 encoded image string.
    """
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
        async with session.get(url) as resp:
            resp.raise_for_status()

            # 이미지를 base64로 인코딩
            img_data = await resp.read()
            img_size_kb = len(img_data) / 1024
            logger.info(f"이미지 다운로드 완료: {img_size_kb:.2f}KB")
            return base64.b64encode(img_data).decode()
