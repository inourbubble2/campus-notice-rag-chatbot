import os
import requests
import logging
from io import BytesIO
from PIL import Image
import pytesseract
from bs4 import BeautifulSoup

__all__ = ["enhance_html_with_ocr"]

from bs4.element import Tag

TESS_LANG = os.getenv("TESS_LANG", "kor")
OCR_TIMEOUT = float(os.getenv("OCR_TIMEOUT", "10"))

_session = requests.Session()
_session.headers.update({"User-Agent": "uos-rag-ingest/1.0"})

def enhance_html_with_ocr(html: str) -> str:
  soup = BeautifulSoup(html, "html.parser")

  # 기존 텍스트 추출 (img 태그 제외)
  for img in soup.find_all("img"):
    img.decompose()
  original_text = soup.get_text(strip=True)

  # 다시 파싱해서 img 태그들로부터 OCR 텍스트 추출
  soup = BeautifulSoup(html, "html.parser")
  ocr_results = [_extract_text_from_image(img) for img in soup.find_all("img")]

  # 기존 텍스트 + OCR 결과 결합
  result_parts = [original_text] if original_text else []
  result_parts.extend([ocr for ocr in ocr_results if ocr])

  return "\n".join(result_parts)

def _extract_text_from_image(img: Tag) -> str:
  src = (img.get("src") or "").strip()
  if src:
    ocr_txt = _ocr_image_from_url(src)
    return f"image: {src} - text:{ocr_txt}"
  return f"invalid image: {src}"

def _ocr_image_from_url(url: str) -> str:
  try:
    resp = _session.get(url, timeout=OCR_TIMEOUT)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content))
    if img.mode in ("RGBA", "P"):
      img = img.convert("RGB")
    return pytesseract.image_to_string(img, lang=TESS_LANG)
  except Exception as e:
    logging.warning(f"OCR failed for URL {url}: {e}")
    return ""
