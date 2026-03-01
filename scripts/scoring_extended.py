"""
Evaluation scoring module for extended entity extraction benchmark.

Scores 17 questions (EQ01-EQ05, MQ06-MQ11, HQ12-HQ17) against the
consolidated document package (contract_extended.md).

Scoring framework:
- Exact match (1.0): Answer matches ground truth precisely
- Partial match (0.5): Directionally correct but imprecise
- Incorrect (0.0): Wrong answer or hallucinated
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
        r"^based\s+on\s+(the\s+)?(contract|agreement|document|dpa|addendum)\s*,?\s*",
        r"^according\s+to\s+(the\s+)?(contract|agreement|document|dpa|addendum)\s*,?\s*",
        r"^(per|under|from)\s+(the\s+)?(contract|agreement|document|dpa|addendum)\s*,?\s*",
        r"^(after|taking\s+into\s+account)\s+(all\s+)?amendments?\s*,?\s*",
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
    # $5,000,000.00 or $500,000 or $395.00
    m = re.search(r"\$\s*([\d,]+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1).replace(",", ""))
    # $5M / $2.5M / $4.25M
    m = re.search(r"\$\s*([\d.]+)\s*[mM]", text)
    if m:
        return float(m.group(1)) * 1_000_000
    # Word-based (longest match first)
    word_map = [
        ("seven million five hundred thousand", 7_500_000),
        ("four million two hundred fifty thousand", 4_250_000),
        ("three million seven hundred fifty thousand", 3_750_000),
        ("two million five hundred thousand", 2_500_000),
        ("five million", 5_000_000),
        ("five hundred thousand", 500_000),
        ("four hundred twenty-five", 425),
        ("four hundred and twenty-five", 425),
        ("three hundred ninety-five", 395),
        ("three hundred and ninety-five", 395),
        ("three hundred seventy-five", 375),
        ("three hundred and seventy-five", 375),
        ("two hundred ninety-five", 295),
        ("two hundred and ninety-five", 295),
        ("two hundred fifty", 250),
        ("two hundred and fifty", 250),
        ("one hundred seventy-five", 175),
        ("one hundred and seventy-five", 175),
    ]
    t = text.lower()
    for phrase, val in word_map:
        if phrase in t:
            return float(val)
    return None


def extract_months(text: str) -> Optional[int]:
    """Extract a month count."""
    m = re.search(r"(\d+)\s*months?", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    word_map = [
        ("ninety-six", 96), ("ninety six", 96),
        ("eighty-four", 84), ("eighty four", 84),
        ("sixty", 60),
        ("thirty-six", 36), ("thirty six", 36),
        ("thirty", 30),
        ("twenty-four", 24), ("twenty four", 24),
        ("eighteen", 18),
        ("twelve", 12),
    ]
    t = text.lower()
    for phrase, val in sorted(word_map, key=lambda x: -len(x[0])):
        if phrase in t:
            return val
    return None


def extract_hours(text: str) -> Optional[int]:
    """Extract an hour count."""
    m = re.search(r"(\d+)\s*hours?", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    word_map = [
        ("seventy-two", 72), ("seventy two", 72),
        ("forty-eight", 48), ("forty eight", 48),
        ("thirty-six", 36), ("thirty six", 36),
        ("twenty-four", 24), ("twenty four", 24),
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


# ── EASY tier scorers (EQ01-EQ05) ─────────────────────────────────

def score_eq01(response: str) -> Tuple[float, str, bool]:
    """MSA effective date → January 15, 2025"""
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
    # Check for DPA date confusion
    if re.search(r"february", t, re.IGNORECASE):
        return INCORRECT, t, False
    if re.search(r"june", t, re.IGNORECASE):
        return INCORRECT, t, False
    return INCORRECT, t, len(t) > 5


def score_eq02(response: str) -> Tuple[float, str, bool]:
    """DPA effective date → February 1, 2025"""
    if is_no_answer(response):
        return INCORRECT, "", False
    t = normalize_text(response)
    exact = [
        r"february\s+1\s*,?\s*2025", r"feb\.?\s+1\s*,?\s*2025",
        r"0?2[/-]0?1[/-]2025", r"2025[/-]0?2[/-]0?1",
        r"1\s+(february|feb)\.?\s*,?\s*2025",
    ]
    if any(re.search(p, t, re.IGNORECASE) for p in exact):
        return EXACT, "February 1, 2025", False
    if re.search(r"february.*2025|2025.*february", t, re.IGNORECASE):
        return PARTIAL, t, False
    # Confusion with MSA date
    if re.search(r"january\s+15", t, re.IGNORECASE):
        return INCORRECT, t, False
    return INCORRECT, t, len(t) > 5


def score_eq03(response: str) -> Tuple[float, str, bool]:
    """Initial term → 3 years"""
    if is_no_answer(response):
        return INCORRECT, "", False
    t = normalize_text(response)
    if re.search(r"\b3\b.*\byears?\b|\bthree\b.*\byears?\b", t, re.IGNORECASE):
        return EXACT, "3 years", False
    if re.search(r"\d+\s*years?", t, re.IGNORECASE):
        return PARTIAL, t, False
    return INCORRECT, t, len(t) > 3


def score_eq04(response: str) -> Tuple[float, str, bool]:
    """General Liability Cap → $2,500,000"""
    if is_no_answer(response):
        return INCORRECT, "", False
    t = normalize_text(response)
    amt = extract_dollar_amount(t)
    if amt is not None:
        if amt == 2_500_000:
            return EXACT, "$2,500,000", False
        if amt in (5_000_000, 500_000, 7_500_000, 3_750_000, 4_250_000):
            return PARTIAL, f"${amt:,.0f} (wrong cap)", False
        return INCORRECT, f"${amt:,.0f}", False
    if "two million five hundred thousand" in t.lower():
        return EXACT, "$2,500,000", False
    return INCORRECT, t, len(t) > 5


def score_eq05(response: str) -> Tuple[float, str, bool]:
    """Data Engineer hourly rate → $295"""
    if is_no_answer(response):
        return INCORRECT, "", False
    t = normalize_text(response)
    amt = extract_dollar_amount(t)
    if amt is not None:
        if amt == 295 or amt == 295.00:
            return EXACT, "$295.00", False
        if amt in (375, 395, 250, 175, 425):
            return INCORRECT, f"${amt:.0f} (wrong role)", False
        return INCORRECT, f"${amt:,.0f}", False
    if "two hundred ninety-five" in t.lower() or "two hundred and ninety-five" in t.lower():
        return EXACT, "$295.00", False
    return INCORRECT, t, len(t) > 5


# ── MEDIUM tier scorers (MQ06-MQ11) ───────────────────────────────

def score_mq06(response: str) -> Tuple[float, str, bool]:
    """Data Protection Coordinator → Dr. Priya Nadkarni-Shah, Director of Data Privacy and Compliance"""
    if is_no_answer(response):
        return INCORRECT, "", False
    t = normalize_text(response)
    tl = t.lower()
    has_name = "nadkarni" in tl or ("priya" in tl and "shah" in tl)
    has_title = (
        "data privacy" in tl
        or "data protection coordinator" in tl
        or ("director" in tl and "privacy" in tl)
    )
    if has_name and has_title:
        return EXACT, "Dr. Priya Nadkarni-Shah, Director of Data Privacy and Compliance", False
    if has_name:
        return EXACT, "Dr. Priya Nadkarni-Shah", False
    # Wrong person
    wrong = ["thornton", "webb", "rachel", "lindstrom", "erik", "blackwell",
             "samantha", "ashworth", "vasquez", "farrell", "reinholt"]
    if any(n in tl for n in wrong):
        return INCORRECT, t, False
    return INCORRECT, t, len(t) > 5


def score_mq07(response: str) -> Tuple[float, str, bool]:
    """Encryption at rest → AES-256"""
    if is_no_answer(response):
        return INCORRECT, "", False
    t = normalize_text(response)
    if re.search(r"AES[- ]?256", t, re.IGNORECASE):
        return EXACT, "AES-256", False
    # Confusion: transit encryption instead of at-rest
    if re.search(r"TLS\s*1\.?3", t, re.IGNORECASE):
        return INCORRECT, "TLS 1.3 (transit, not at-rest)", False
    # Partial: mentions AES but wrong key length or unspecified
    if re.search(r"\bAES\b", t, re.IGNORECASE):
        return PARTIAL, t, False
    if "advanced encryption" in t.lower():
        return PARTIAL, t, False
    return INCORRECT, t, len(t) > 5


def score_mq08(response: str) -> Tuple[float, str, bool]:
    """Pen testing firm → Whitmore-Kessler Security Consultants, Inc."""
    if is_no_answer(response):
        return INCORRECT, "", False
    t = normalize_text(response)
    tl = t.lower()
    if "whitmore" in tl and "kessler" in tl:
        return EXACT, "Whitmore-Kessler Security Consultants, Inc.", False
    if "whitmore" in tl or "kessler" in tl:
        return PARTIAL, t, False
    # Gives contact person instead of firm
    if "gregory" in tl and "whitmore" in tl:
        return PARTIAL, t, True
    # Confusion with sub-processors
    subprocessors = ["cloudvault", "nexus", "sentinel", "horizon", "pacific rim",
                     "keystone", "redwood", "brightpath", "cedarbridge", "pinnacle"]
    if any(sp in tl for sp in subprocessors):
        return INCORRECT, t, False
    return INCORRECT, t, len(t) > 5


def score_mq09(response: str) -> Tuple[float, str, bool]:
    """Transaction data retention → 36 months"""
    if is_no_answer(response):
        return INCORRECT, "", False
    t = normalize_text(response)
    months = extract_months(t)
    if months is not None:
        if months == 36:
            return EXACT, "36 months", False
        if months in (84, 96, 18, 24, 60, 12, 30):
            return INCORRECT, f"{months} months (wrong category)", False
        return INCORRECT, f"{months} months", False
    # Check for "3 years" as equivalent
    if re.search(r"\b3\b.*\byears?\b|\bthree\b.*\byears?\b", t, re.IGNORECASE):
        return EXACT, "36 months", False
    return INCORRECT, t, len(t) > 3


def score_mq10(response: str) -> Tuple[float, str, bool]:
    """Marketing data retention → 18 months"""
    if is_no_answer(response):
        return INCORRECT, "", False
    t = normalize_text(response)
    months = extract_months(t)
    if months is not None:
        if months == 18:
            return EXACT, "18 months", False
        if months in (36, 84, 96, 24, 60, 12, 30):
            return INCORRECT, f"{months} months (wrong category)", False
        return INCORRECT, f"{months} months", False
    if "eighteen" in t.lower():
        return EXACT, "18 months", False
    # Check for "1.5 years" as equivalent
    if re.search(r"1\.5\s*years?", t, re.IGNORECASE):
        return EXACT, "18 months", False
    return INCORRECT, t, len(t) > 3


def score_mq11(response: str) -> Tuple[float, str, bool]:
    """Sub-processor in Tokyo → Pacific Rim Analytics K.K."""
    if is_no_answer(response):
        return INCORRECT, "", False
    t = normalize_text(response)
    tl = t.lower()
    if "pacific rim" in tl:
        return EXACT, "Pacific Rim Analytics K.K.", False
    # Confusion with other Asia-Pacific sub-processors
    if "nexus data services" in tl:
        return INCORRECT, "Nexus Data Services Pte. Ltd. (Singapore, not Tokyo)", False
    if "quantum ledger" in tl:
        return INCORRECT, t, False
    # Any other sub-processor name
    wrong = ["cloudvault", "sentinel", "horizon", "keystone", "redwood",
             "brightpath", "cedarbridge", "pinnacle", "nexus data solutions"]
    if any(w in tl for w in wrong):
        return INCORRECT, t, False
    return INCORRECT, t, len(t) > 5


# ── HARD tier scorers (HQ12-HQ17) ─────────────────────────────────

def score_hq12(response: str) -> Tuple[float, str, bool]:
    """Current Senior Data Scientist rate (after amendment) → $395"""
    if is_no_answer(response):
        return INCORRECT, "", False
    t = normalize_text(response)
    has_375 = bool(re.search(r"375", t))
    has_395 = bool(re.search(r"395", t))

    # Mentions both — check if it resolves
    if has_375 and has_395:
        if re.search(r"(amended|changed|increased|updated|now)\s+.{0,20}395", t, re.IGNORECASE):
            return EXACT, "$395.00", False
        return PARTIAL, t, True

    amt = extract_dollar_amount(t)
    if amt is not None:
        if amt == 395 or amt == 395.00:
            return EXACT, "$395.00", False
        if amt == 375 or amt == 375.00:
            return INCORRECT, "$375.00 (pre-amendment)", False
        if amt in (295, 250, 175, 425):
            return INCORRECT, f"${amt:.0f} (wrong role)", False
        return INCORRECT, f"${amt:,.0f}", False

    if "three hundred ninety-five" in t.lower():
        return EXACT, "$395.00", False
    if "three hundred seventy-five" in t.lower():
        return INCORRECT, "$375.00 (pre-amendment)", False
    return INCORRECT, t, len(t) > 5


def score_hq13(response: str) -> Tuple[float, str, bool]:
    """Current Force Majeure threshold (after amendment) → 90 days"""
    if is_no_answer(response):
        return INCORRECT, "", False
    t = normalize_text(response)
    has_120 = bool(re.search(r"\b120\b|one hundred (and )?twenty", t, re.IGNORECASE))
    has_90 = bool(re.search(r"\b90\b|ninety", t, re.IGNORECASE))

    if has_120 and has_90:
        if re.search(r"(amended|changed|reduced|updated|now)\s+.{0,20}\b90\b", t, re.IGNORECASE):
            return EXACT, "90 days", False
        return PARTIAL, t, True

    days = extract_days(t)
    if days == 90:
        return EXACT, "90 days", False
    if days == 120:
        return INCORRECT, "120 days (pre-amendment)", False
    if days == 60:
        return INCORRECT, "60 days (termination notice, not force majeure)", False
    if days == 30:
        return INCORRECT, "30 days (notice period after FM, not threshold)", False
    if days is not None:
        return INCORRECT, f"{days} days", False
    return INCORRECT, t, len(t) > 3


def score_hq14(response: str) -> Tuple[float, str, bool]:
    """Current DPA breach notification (after amendment) → 36 hours"""
    if is_no_answer(response):
        return INCORRECT, "", False
    t = normalize_text(response)
    has_48 = bool(re.search(r"\b48\b|forty[- ]?eight", t, re.IGNORECASE))
    has_24 = bool(re.search(r"\b24\b|twenty[- ]?four", t, re.IGNORECASE))
    has_36 = bool(re.search(r"\b36\b|thirty[- ]?six", t, re.IGNORECASE))

    # Clean resolution to 36
    if has_36 and not has_24 and not has_48:
        return EXACT, "36 hours", False

    # Mentions 36 alongside others — check if it resolves
    if has_36 and (has_24 or has_48):
        if re.search(r"(amended|changed|updated|currently|now)\s+.{0,30}\b36\b", t, re.IGNORECASE):
            return EXACT, "36 hours", False
        return PARTIAL, t, True

    hours = extract_hours(t)
    if hours == 36:
        return EXACT, "36 hours", False
    if hours == 24:
        return INCORRECT, "24 hours (DPA original, before amendment)", False
    if hours == 48:
        return INCORRECT, "48 hours (MSA timeline, not DPA)", False
    if hours == 72:
        return INCORRECT, "72 hours (written report deadline, not initial notification)", False
    if hours is not None:
        return INCORRECT, f"{hours} hours", False
    return INCORRECT, t, len(t) > 3


def score_hq15(response: str) -> Tuple[float, str, bool]:
    """Current DPA Liability Cap (after amendment) → $4,250,000"""
    if is_no_answer(response):
        return INCORRECT, "", False
    t = normalize_text(response)
    has_3750 = bool(re.search(r"3[,.]?750", t))
    has_4250 = bool(re.search(r"4[,.]?250", t))

    if has_3750 and has_4250:
        if re.search(r"(amended|changed|increased|updated|now)\s+.{0,30}4[,.]?250", t, re.IGNORECASE):
            return EXACT, "$4,250,000", False
        return PARTIAL, t, True

    amt = extract_dollar_amount(t)
    if amt is not None:
        if amt == 4_250_000:
            return EXACT, "$4,250,000", False
        if amt == 3_750_000:
            return INCORRECT, "$3,750,000 (pre-amendment)", False
        if amt in (2_500_000, 5_000_000, 500_000, 7_500_000):
            return INCORRECT, f"${amt:,.0f} (wrong cap from MSA)", False
        return INCORRECT, f"${amt:,.0f}", False
    if "four million two hundred fifty thousand" in t.lower():
        return EXACT, "$4,250,000", False
    if "three million seven hundred fifty thousand" in t.lower():
        return INCORRECT, "$3,750,000 (pre-amendment)", False
    return INCORRECT, t, len(t) > 5


def score_hq16(response: str) -> Tuple[float, str, bool]:
    """Nexus disambiguation → Solutions GmbH = Germany, Services Pte. Ltd. = Singapore"""
    if is_no_answer(response):
        return INCORRECT, "", False
    t = normalize_text(response)
    tl = t.lower()

    # Check for correct pairings
    solutions_germany = (
        ("solutions" in tl and ("germany" in tl or "frankfurt" in tl))
        or ("gmbh" in tl and ("germany" in tl or "frankfurt" in tl))
    )
    services_singapore = (
        ("services" in tl and "singapore" in tl)
        or ("pte" in tl and "singapore" in tl)
    )

    if solutions_germany and services_singapore:
        return EXACT, "Nexus Data Solutions GmbH (Germany); Nexus Data Services Pte. Ltd. (Singapore)", False
    if solutions_germany or services_singapore:
        return PARTIAL, t, False

    # Check for swapped associations
    solutions_singapore = "solutions" in tl and "singapore" in tl
    services_germany = "services" in tl and ("germany" in tl or "frankfurt" in tl)
    if solutions_singapore or services_germany:
        return INCORRECT, f"{t} (locations swapped)", False

    # At least mentions both Nexus entities
    if tl.count("nexus") >= 2:
        return PARTIAL, t, True

    return INCORRECT, t, len(t) > 5


def score_hq17(response: str) -> Tuple[float, str, bool]:
    """Amendment signatory (Provider) → Mr. Thomas J. Brennan, Chief Operating Officer"""
    if is_no_answer(response):
        return INCORRECT, "", False
    t = normalize_text(response)
    tl = t.lower()
    has_name = "brennan" in tl
    has_title = "chief operating" in tl or "coo" in tl.split()

    if has_name and has_title:
        return EXACT, "Mr. Thomas J. Brennan, Chief Operating Officer", False
    if has_name:
        return PARTIAL, "Thomas J. Brennan (title missing)", False

    # Wrong signatory — MSA/DPA signatory for Provider
    if "reinholt" in tl or "alexander" in tl:
        return INCORRECT, "Dr. Alexander S. Reinholt (MSA/DPA signatory, not Amendment)", False
    # Client's Amendment signatory
    if "tanaka" in tl or "morrison" in tl or "yuki" in tl:
        return INCORRECT, "Dr. Yuki Tanaka-Morrison (Client signatory, not Provider)", False
    # Client's MSA signatory
    if "chen" in tl or "ramirez" in tl or "margaret" in tl:
        return INCORRECT, "Margaret Chen-Ramirez (Client's MSA signatory)", False
    return INCORRECT, t, len(t) > 5


# ── Dispatch ───────────────────────────────────────────────────────

SCORERS = {
    "EQ01": score_eq01, "EQ02": score_eq02, "EQ03": score_eq03,
    "EQ04": score_eq04, "EQ05": score_eq05,
    "MQ06": score_mq06, "MQ07": score_mq07, "MQ08": score_mq08,
    "MQ09": score_mq09, "MQ10": score_mq10, "MQ11": score_mq11,
    "HQ12": score_hq12, "HQ13": score_hq13, "HQ14": score_hq14,
    "HQ15": score_hq15, "HQ16": score_hq16, "HQ17": score_hq17,
}


def score_answer(question_id: str, response: str) -> Dict[str, Any]:
    """Score a model response for a given question."""
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
