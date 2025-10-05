import asyncio
import aiohttp
import ssl
import logging
import base64
from bs4 import BeautifulSoup
from bs4.element import Tag

from services import extract_text_from_image

__all__ = ["enhance_html_with_ocr"]



async def enhance_html_with_ocr(html: str) -> str:
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

    # 성공한 결과만 필터링 (예외는 로깅하되 계속 진행)
    valid_ocr_results = []
    for i, result in enumerate(ocr_results):
      if isinstance(result, str):
        if result.strip():
          valid_ocr_results.append(result)
        else:
          logging.warning(f"OCR returned empty result for image {i+1}")
      elif isinstance(result, Exception):
        logging.error(f"OCR failed for image {i+1}: {result}")
  else:
    valid_ocr_results = []

  # 기존 텍스트 + OCR 결과 결합
  result_parts = [original_text] if original_text else []
  result_parts.extend(valid_ocr_results)

  return "\n".join(result_parts)

async def _extract_text_from_image(img: Tag) -> str:
  """이미지 태그에서 OCR 텍스트를 추출."""
  src = (img.get("src") or "").strip()
  if not src:
    raise ValueError(f"Invalid image src: {src}")

  ocr_txt = await _ocr_image_from_url(src)
  if ocr_txt.strip():
    return f"image: {src} - text:{ocr_txt}"
  else:
    logging.warning(f"OCR returned empty text for image: {src}")
    return f"image: {src} - text:(empty)"

async def _ocr_image_from_url(url: str) -> str:
  from app.deps import get_settings

  cfg = get_settings()

  timeout = aiohttp.ClientTimeout(total=cfg.ocr_timeout)

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
      img_base64 = base64.b64encode(img_data).decode()

      return await extract_text_from_image(img_base64)
