import io, requests, re
from PIL import Image
import pytesseract

def _open_local_or_url(path_or_url: str):
    if path_or_url.startswith("http"):
        try:
            r = requests.get(path_or_url, timeout=5)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception:
            return None
    else:
        try:
            return Image.open(path_or_url).convert("RGB")
        except Exception:
            return None

def ocr(path_or_url: str) -> str:
    img = _open_local_or_url(path_or_url)
    if img is None:
        return ""
    try:
        text = pytesseract.image_to_string(img)[:800].strip()
        # 🔹 clean after extraction
        text = re.sub(r'[^A-Za-z0-9 .,!?]', '', text)
        text = re.sub(r'\b(use_column_width|TruthMindr|Final Label)\b', '', text, flags=re.I)
        return text
    except Exception as e:
        print(f"[OCR error] {e}")
        return ""
