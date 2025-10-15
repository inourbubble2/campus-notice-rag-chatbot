import re
import unicodedata
from bs4 import BeautifulSoup

__all__ = ["get_plain_text"]

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
