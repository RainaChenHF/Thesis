"""
Shared helpers: Qwen API client, normalisation, logging.
"""
import re, string, logging, time, json
from openai import OpenAI

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import QWEN_API_KEY, QWEN_BASE_URL, JUDGE_MODEL

# ── Logging ────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


# ── Qwen client ───────────────────────────────────────────────────────────
_client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


def qwen_chat(prompt: str, model: str = JUDGE_MODEL,
              temperature: float = 0.0, max_tokens: int = 512,
              retries: int = 8) -> str:
    """Call Qwen with retry logic and exponential backoff."""
    for attempt in range(retries):
        try:
            resp = _client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = min(2 ** attempt, 60)
            time.sleep(wait)
    return ""


def qwen_json(prompt: str, **kwargs) -> dict | list:
    """Call Qwen and parse JSON from the response."""
    raw = qwen_chat(prompt, **kwargs)
    # Extract content from markdown fences if present
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if m:
        raw = m.group(1).strip()
    else:
        # Strip any leading prose before the first [ or {
        m2 = re.search(r"[\[{]", raw)
        if m2:
            raw = raw[m2.start():]
    # Use raw_decode to ignore trailing text after the JSON closes
    obj, _ = json.JSONDecoder().raw_decode(raw.strip())
    return obj


# ── Text normalisation (from SQuAD / HybridQA eval) ───────────────────────
def normalize_answer(s: str) -> str:
    def remove_articles(t):   return re.sub(r"\b(a|an|the)\b", " ", t)
    def fix_whitespace(t):    return " ".join(t.split())
    def remove_punc(t):       return "".join(c for c in t if c not in string.punctuation)
    def lower(t):             return t.lower()
    return fix_whitespace(remove_articles(remove_punc(lower(s))))


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    from collections import Counter
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    common = Counter(p_toks) & Counter(g_toks)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec = n / len(p_toks)
    rec  = n / len(g_toks)
    return 2 * prec * rec / (prec + rec)
