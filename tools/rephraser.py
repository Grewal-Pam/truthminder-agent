# truthmindr_agent/tools/rephraser.py
import os
import subprocess
from openai import OpenAI

# Try to set up OpenAI client if key exists
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None


def _rephrase_openai(prompt: str) -> str:
    """
    Use OpenAI GPT API for rephrasing.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return resp.choices[0].message.content.strip()


def _rephrase_ollama(prompt: str, model="llama3") -> str:
    """
    Use local Ollama model as fallback.
    """
    result = subprocess.run(
        ["ollama", "run", model], input=prompt.encode("utf-8"), capture_output=True
    )
    return result.stdout.decode("utf-8").strip()


def rephrase_result(row: dict) -> str:
    """
    Take one enriched row (dict) and return a plain-English explanation.
    Tries OpenAI first, then falls back to Ollama if needed.
    """
    prompt = f"""
    You are TruthMindr — a multimodal AI that explains how it judges online news posts.

    Summarize the analysis below for an everyday reader, in a warm and natural tone (not robotic).
    Avoid repeating numbers unless meaningful. If confidence < 0.6, gently say it's uncertain.

    ---
    Title: {row.get('clean_title') or 'Untitled post'}
    Label: {row.get('final_label')}
    Confidence: {row.get('final_confidence')}
    Consistency score: {row.get('consistency_score')}
    OCR text snippet: {str(row.get('ocr_text'))[:150]}
    ---

    Write 3–5 sentences that sound like you’re personally explaining the result.
    End with one takeaway sentence beginning with “In simple terms,”.
    """

    if client:  # If OpenAI key exists, try it
        try:
            return _rephrase_openai(prompt)
        except Exception as e:
            print(f"[Rephraser] OpenAI failed ({e}), falling back to Ollama...")

    # If OpenAI not available or failed → use Ollama
    return _rephrase_ollama(prompt)
