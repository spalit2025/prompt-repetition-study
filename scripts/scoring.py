"""
Evaluation scoring module for entity extraction benchmark.

Scoring framework:
- Exact match (1.0): Answer matches ground truth precisely
- Partial match (0.5): Directionally correct but imprecise
- Incorrect (0.0): Wrong answer or hallucinated
- No answer (0.0): Model says it cannot find the information

Each question has a custom scorer that handles format normalization
and defines what counts as exact/partial/incorrect.
"""

import re
from typing import Tuple, Dict, Any, Optional

EXACT = 1.0
PARTIAL = 0.5
INCORRECT = 0.0


# ── Utility functions ──────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Strip common answer prefixes and trailing punctuation."""
    text = text.strip()
    prefixes = [
        r"^(the\s+)?(answer\s+is\s*:?\s*)",
        r"^based\s+on\s+(the\s+)?(contract|agreement)\s*,?\s*",
        r"^according\s+to\s+(the\s+)?(contract|agreement)\s*,?\s*",
        r"^(per|under|from)\s+(the\s+)?(contract|agreement)\s*,?\s*",
    ]
    for p in prefixes:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    return text.strip().rstrip(".")


def is_no_answer(text: str) -> bool:
    """Detect responses indicating the model couldn't find an answer."""
    t = text.lower().strip()
    patterns = [
        r"cannot\s+find", r"could\s+not\s+find", r"couldn'?t\s+find",
        r"not\s+(explicitly\s+)?(stated|specified|mentioned|found|provided)",
        r"does\s+not\s+(specify|mention|state|contain)",
        r"(i\s+)?(don'?t|do\s+not|cannot)\s+(see|find|locate|determine)",
        r"unable\s+to", r"not\s+available", r"\bn/?a\b",
    ]
    return any(re.search(p, t) for p in patterns)


def extract_dollar_amount(text: str) -> Optional[float]:
    """Extract a dollar amount as a float."""
    # $5,000,000.00 or $500,000
    m = re.search(r"\$\s*([\d,]+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1).replace(",", ""))
    # $5M / $2.5M
    m = re.search(r"\$\s*([\d.]+)\s*[mM]", text)
    if m:
        return float(m.group(1)) * 1_000_000
    # Word-based (longest match first)
    word_map = [
        ("seven million five hundred thousand", 7_500_000),
        ("two million five hundred thousand", 2_500_000),
        ("five million", 5_000_000),
        ("five hundred thousand", 500_000),
    ]
    t = text.lower()
    for phrase, val in word_map:
        if phrase in t:
            return float(val)
    return None


def extract_percentage(text: str) -> Optional[float]:
    """Extract a percentage value."""
    m = re.search(r"([\d.]+)\s*%", text)
    if m:
        return float(m.group(1))
    m = re.search(r"([\d.]+)\s*percent", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    word_map = [
        ("ninety-nine and one-half", 99.5),
        ("twenty-five", 25), ("twenty five", 25),
        ("twenty", 20), ("fifteen", 15),
        ("ten", 10), ("five", 5),
        ("one and one-half", 1.5),
        ("fifty", 50),
    ]
    t = text.lower()
    for phrase, val in sorted(word_map, key=lambda x: -len(x[0])):
        if phrase in t:
            return val
    return None


def extract_days(text: str) -> Optional[int]:
    """Extract a day count."""
    m = re.search(r"(\d+)\s*days?", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    word_map = [
        ("one hundred twenty", 120), ("one hundred and twenty", 120),
        ("ninety", 90), ("seventy-two", 72), ("sixty", 60),
        ("forty-five", 45), ("thirty", 30), ("fifteen", 15),
    ]
    t = text.lower()
    for phrase, val in sorted(word_map, key=lambda x: -len(x[0])):
        if phrase in t:
            return val
    return None


# ── Per-question scorers ───────────────────────────────────────────
# Each returns (score, extracted_answer, needs_manual_review)

def score_q01(response: str) -> Tuple[float, str, bool]:
    """Effective date → January 15, 2025"""
    if is_no_answer(response):
        return INCORRECT, "", False
    t = normalize_text(response)
    exact = [
        r"january\s+15\s*,?\s*2025", r"jan\.?\s+15\s*,?\s*2025",
        r"0?1[/-]15[/-]2025", r"2025[/-]0?1[/-]15",
        r"15\s+(january|jan)\.?\s*,?\s*2025",
    ]
    if any(re.search(p, t, re.IGNORECASE) for p in exact):
        return EXACT, "January 15, 2025", False
    if re.search(r"january.*2025|2025.*january", t, re.IGNORECASE):
        return PARTIAL, t, False
    return INCORRECT, t, len(t) > 5


def score_q02(response: str) -> Tuple[float, str, bool]:
    """Agreement number → MSA-2025-0892"""
    if is_no_answer(response):
        return INCORRECT, "", False
    t = normalize_text(response).upper().replace(" ", "")
    if "MSA-2025-0892" in t:
        return EXACT, "MSA-2025-0892", False
    if "MSA-2025" in t:
        return PARTIAL, t, False
    if "MSA" in t and "0892" in t:
        return PARTIAL, t, False
    return INCORRECT, t, len(t) > 3


def score_q03(response: str) -> Tuple[float, str, bool]:
    """Initial term → 3 years"""
    if is_no_answer(response):
        return INCORRECT, "", False
    t = normalize_text(response)
    if re.search(r"\b3\b.*\byears?\b|\bthree\b.*\byears?\b", t, re.IGNORECASE):
        return EXACT, "3 years", False
    if re.search(r"\d+\s*years?", t, re.IGNORECASE):
        return PARTIAL, t, False
    return INCORRECT, t, len(t) > 3


def score_q04(response: str) -> Tuple[float, str, bool]:
    """Client's state of organization → New York"""
    if is_no_answer(response):
        return INCORRECT, "", False
    t = normalize_text(response)
    tl = t.lower()
    if "new york" in tl or t.strip().upper() == "NY":
        if "delaware" not in tl:
            return EXACT, "New York", False
        return PARTIAL, t, True  # mentions both
    if "delaware" in tl:
        return INCORRECT, t, False
    if "virginia" in tl:
        return INCORRECT, t, False
    return INCORRECT, t, len(t) > 3


def score_q05(response: str) -> Tuple[float, str, bool]:
    """Data breach liability cap → $5,000,000"""
    if is_no_answer(response):
        return INCORRECT, "", False
    t = normalize_text(response)
    amt = extract_dollar_amount(t)
    if amt is not None:
        if amt == 5_000_000:
            return EXACT, "$5,000,000", False
        if amt in (2_500_000, 500_000, 7_500_000):
            return PARTIAL, f"${amt:,.0f} (wrong cap)", False
        return INCORRECT, f"${amt:,.0f}", False
    if "five million" in t.lower():
        return EXACT, "$5,000,000", False
    return INCORRECT, t, len(t) > 5


def score_q06(response: str) -> Tuple[float, str, bool]:
    """Expense reimbursement cap → 15%"""
    if is_no_answer(response):
        return INCORRECT, "", False
    t = normalize_text(response)
    pct = extract_percentage(t)
    if pct is not None:
        if pct == 15:
            return EXACT, "15%", False
        if pct in (5, 10, 20, 25, 50, 99.5, 1.5):
            return PARTIAL, f"{pct}% (wrong figure)", False
        return INCORRECT, f"{pct}%", False
    if "fifteen" in t.lower() and "percent" in t.lower():
        return EXACT, "15%", False
    return INCORRECT, t, len(t) > 3


def score_q07(response: str) -> Tuple[float, str, bool]:
    """Technical Account Manager → Dominic W. Farrell"""
    if is_no_answer(response):
        return INCORRECT, "", False
    t = normalize_text(response)
    tl = t.lower()
    if "farrell" in tl:
        return EXACT, "Dominic W. Farrell", False
    if "dominic" in tl:
        return PARTIAL, t, False
    wrong = ["ashworth", "vasquez", "ortega", "thornton", "webb",
             "reinholt", "chen", "ramirez"]
    if any(n in tl for n in wrong):
        return INCORRECT, t, False
    return INCORRECT, t, len(t) > 5


def score_q08(response: str) -> Tuple[float, str, bool]:
    """Per-incident liability cap → $500,000"""
    if is_no_answer(response):
        return INCORRECT, "", False
    t = normalize_text(response)
    amt = extract_dollar_amount(t)
    if amt is not None:
        if amt == 500_000:
            return EXACT, "$500,000", False
        if amt in (2_500_000, 5_000_000, 7_500_000):
            return PARTIAL, f"${amt:,.0f} (wrong cap)", False
        return INCORRECT, f"${amt:,.0f}", False
    if "five hundred thousand" in t.lower():
        return EXACT, "$500,000", False
    return INCORRECT, t, len(t) > 5


def score_q09(response: str) -> Tuple[float, str, bool]:
    """Termination notice after amendment → 60 days"""
    if is_no_answer(response):
        return INCORRECT, "", False
    t = normalize_text(response)
    # Check for "mentions both" first — model found both values
    has_90 = bool(re.search(r"\b90\b|ninety", t, re.IGNORECASE))
    has_60 = bool(re.search(r"\b60\b|sixty", t, re.IGNORECASE))
    if has_90 and has_60:
        # If the response resolves to 60 (e.g., "amended to 60"), score exact
        if re.search(r"(amended|changed|reduced|updated)\s+to\s+(sixty|60)", t, re.IGNORECASE):
            return EXACT, "60 days", False
        return PARTIAL, t, True  # mentions both but doesn't clearly resolve
    days = extract_days(t)
    if days == 60:
        return EXACT, "60 days", False
    if days == 90:
        return INCORRECT, "90 days (pre-amendment)", False
    if days is not None:
        return INCORRECT, f"{days} days", False
    if "sixty" in t.lower():
        return EXACT, "60 days", False
    if "ninety" in t.lower():
        return INCORRECT, "90 days (pre-amendment)", False
    return INCORRECT, t, len(t) > 5


def score_q10(response: str) -> Tuple[float, str, bool]:
    """Service credit <98% after amendment → 25%"""
    if is_no_answer(response):
        return INCORRECT, "", False
    t = normalize_text(response)
    pct = extract_percentage(t)
    if pct == 25:
        return EXACT, "25%", False
    if pct == 20:
        return INCORRECT, "20% (pre-amendment)", False
    if pct is not None:
        return INCORRECT, f"{pct}%", False
    if "20" in t and "25" in t:
        return PARTIAL, t, True
    if "twenty-five" in t.lower() or "twenty five" in t.lower():
        return EXACT, "25%", False
    if "twenty" in t.lower() and "five" not in t.lower():
        return INCORRECT, "20% (pre-amendment)", False
    return INCORRECT, t, len(t) > 3


def score_q11(response: str) -> Tuple[float, str, bool]:
    """Sub-processor → Meridian Analytics Federal Services, LLC; Virginia"""
    if is_no_answer(response):
        return INCORRECT, "", False
    t = normalize_text(response)
    tl = t.lower()
    has_entity = (
        "federal services" in tl
        or "mafs" in tl
        or "meridian analytics federal" in tl
    )
    has_state = "virginia" in tl
    if has_entity and has_state:
        return EXACT, "Meridian Analytics Federal Services, LLC; Virginia", False
    if has_entity or has_state:
        return PARTIAL, t, False
    # Check for wrong-entity confusion
    if "meridian analytics group" in tl or "meridian dynamics" in tl:
        return INCORRECT, t, False
    return INCORRECT, t, len(t) > 10


def score_q12(response: str) -> Tuple[float, str, bool]:
    """Force majeure threshold → 120 days"""
    if is_no_answer(response):
        return INCORRECT, "", False
    t = normalize_text(response)
    days = extract_days(t)
    if days == 120:
        return EXACT, "120 days", False
    if days is not None:
        return INCORRECT, f"{days} days", False
    if "one hundred twenty" in t.lower() or "one hundred and twenty" in t.lower():
        return EXACT, "120 days", False
    return INCORRECT, t, len(t) > 3


def score_q13(response: str) -> Tuple[float, str, bool]:
    """Hourly rates → Data Engineer: $295/hr; Project Manager: $250/hr"""
    if is_no_answer(response):
        return INCORRECT, "", False
    t = normalize_text(response)

    has_295 = bool(re.search(r"295", t))
    has_250 = bool(re.search(r"250", t))

    if has_295 and has_250:
        return EXACT, "Data Engineer: $295.00/hour; Project Manager: $250.00/hour", False
    if has_295 or has_250:
        return PARTIAL, t, False
    # Check for wrong rates bleeding in
    if re.search(r"375|175", t):
        return INCORRECT, t, False
    return INCORRECT, t, len(t) > 5


# ── Dispatch ───────────────────────────────────────────────────────

SCORERS = {
    "Q01": score_q01, "Q02": score_q02, "Q03": score_q03, "Q04": score_q04,
    "Q05": score_q05, "Q06": score_q06, "Q07": score_q07, "Q08": score_q08,
    "Q09": score_q09, "Q10": score_q10, "Q11": score_q11, "Q12": score_q12,
    "Q13": score_q13,
}


def score_answer(question_id: str, response: str) -> Dict[str, Any]:
    """Score a model response for a given question.

    Returns dict with: score, extracted_answer, needs_manual_review, scoring_method
    """
    scorer = SCORERS.get(question_id)
    if not scorer:
        return {
            "score": None,
            "extracted_answer": "",
            "needs_manual_review": True,
            "scoring_method": "unknown_question",
        }
    try:
        score, extracted, needs_review = scorer(response)
        return {
            "score": score,
            "extracted_answer": extracted,
            "needs_manual_review": needs_review,
            "scoring_method": "automated",
        }
    except Exception as e:
        return {
            "score": None,
            "extracted_answer": str(e),
            "needs_manual_review": True,
            "scoring_method": "error",
        }
