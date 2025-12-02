import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    return re.sub(r"[^a-z0-9 ]", " ", text).strip()
