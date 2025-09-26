import re
from bs4 import BeautifulSoup
from .ocr import enhance_html_with_ocr

__all__ = [
  "html_to_text",
  "normalize_text",
  "compose_text_with_ocr",
]

def _clean_text(s: str) -> str:
  """연속 개행/공백 압축 + trim."""
  s = s.replace("\r\n", "\n").replace("\r", "\n")
  s = re.sub(r"\n{2,}", "\n", s)
  s = re.sub(r"[ \t]{2,}", " ", s)
  lines = [line.strip() for line in s.splitlines() if line.strip()]
  return "\n".join(lines)

def html_to_text(html: str) -> str:
  """<br>/<p>/<div>만 개행으로, 나머지 태그는 제거하여 순수 텍스트화."""
  soup = BeautifulSoup(html, "html.parser")

  # 의미 있는 줄바꿈
  for tag in soup.find_all(["br", "p", "div"]):
    tag.replace_with("\n" + tag.get_text())

  # script/style 제거
  for t in soup(["script", "style"]):
    t.decompose()

  text = soup.get_text()
  return _clean_text(text)

def normalize_text(html: str) -> str:
  """HTML을 텍스트로 변환하고 정규화합니다."""
  s = html_to_text(html)

  # 콜론 앞뒤 이상 개행 보정:  "행사명\n:\n" -> "행사명: "
  s = re.sub(r"\n\s*:\s*\n", ": ", s)
  s = re.sub(r"\s*:\s*\n", ": ", s)
  s = re.sub(r"\n\s*:\s*", ": ", s)

  # 영문/숫자 사이에 잘못 끊긴 개행 제거: "2025\n학년도" -> "2025 학년도"
  s = re.sub(r"(\w)\n(\w)", r"\1 \2", s)

  # 추가 규칙이 있으면 여기에 확장(날짜/학기 표준화 등)
  return _clean_text(s)

def compose_text_with_ocr(html: str) -> str:
  """
  HTML 순수 텍스트 + (존재 시) 이미지 OCR 텍스트 결합.
  ocr.py 의 enhance_html_with_ocr 를 사용.
  """
  base = html_to_text(html)
  ocr_text = enhance_html_with_ocr(html)
  full = "\n".join([t for t in [base, ocr_text] if t])
  return normalize_text(full)
