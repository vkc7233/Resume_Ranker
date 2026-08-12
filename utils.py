"""
utils.py - Utility / helper functions
Shared helpers used across parser, ranking, and app modules
"""

import re
import os
import datetime
from typing import Optional


# ─────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Remove excessive whitespace, special chars, and normalize newlines.
    """
    if not text:
        return ""
    # Collapse multiple spaces / tabs
    text = re.sub(r'[ \t]+', ' ', text)
    # Normalize newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace to single spaces."""
    return re.sub(r'\s+', ' ', text).strip()


# ─────────────────────────────────────────────
# EMAIL & PHONE EXTRACTION
# ─────────────────────────────────────────────

def extract_email(text: str) -> Optional[str]:
    """Extract first valid email address from text."""
    pattern = r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'
    match = re.search(pattern, text)
    return match.group(0) if match else None


def extract_phone(text: str) -> Optional[str]:
    """
    Extract first phone number from text.
    Handles formats like: +91-9876543210, (123) 456-7890, 9876543210, etc.
    """
    pattern = (
        r'(\+?\d{1,3}[\s\-]?)?'       # optional country code
        r'(\(?\d{3}\)?[\s\-]?)'         # area code
        r'(\d{3}[\s\-]?)'               # first 3 digits
        r'(\d{4})'                       # last 4 digits
    )
    match = re.search(pattern, text)
    if match:
        phone = normalize_whitespace(match.group(0))
        return phone
    return None


# ─────────────────────────────────────────────
# EXPERIENCE EXTRACTION
# ─────────────────────────────────────────────

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Reasonable bounds so stray 4-digit numbers (PIN codes, IDs) aren't read as years.
_MIN_YEAR = 1960
_MAX_AHEAD = 1  # allow up to next year for "expected" end dates

# One side of a date range: "Jan 2018", "January 2018", "06/2018", or "2018".
_DATE_TOKEN = r'(?:[A-Za-z]{3,9}\.?\s+\d{4}|\d{1,2}[/\-.]\d{4}|\d{4})'
_END_TOKEN = r'(?:present|current|now|till\s*date|to\s*date|ongoing|' + _DATE_TOKEN + r')'

_RANGE_RE = re.compile(
    r'(' + _DATE_TOKEN + r')\s*(?:-|–|—|to|until|till|through)\s*(' + _END_TOKEN + r')',
    re.IGNORECASE,
)

# Education-context cues. A date range on a line containing any of these is a
# degree/college/school timeline (e.g. "B.Tech ... 2022 – 2026"), NOT work
# experience, so it must not be summed into professional experience.
_EDU_CONTEXT_RE = re.compile(
    r'(college|university|institute|polytechnic|academy|\bschool\b|'
    r'b\.?\s?tech|b\.?\s?e\b|b\.?\s?sc|b\.?\s?com|b\.?\s?c\.?\s?a|\bbca\b|bachelor|'
    r'm\.?\s?tech|m\.?\s?e\b|m\.?\s?sc|m\.?\s?com|m\.?\s?c\.?\s?a|\bmca\b|\bmba\b|master|'
    r'ph\.?\s?d|doctorate|diploma|\bcgpa\b|\bgpa\b|\bdegree\b|'
    r'class\s*(?:x|xii|10|12)\b|high\s*school|senior\s*secondary|'
    r'\bhsc\b|\bssc\b|\bgraduation\b|\bgraduated\b)',
    re.IGNORECASE,
)

# Explicit claims like "5+ years of experience" / "3 yrs experience".
_EXP_PHRASE_RE = re.compile(
    r'(\d{1,2}(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)'
    r'(?:\s+(?:of\s+)?(?:professional\s+|work\s+|industry\s+|relevant\s+)?'
    r'(?:experience|exp))',
    re.IGNORECASE,
)

# Any "N years" mention (used only as a last-resort fallback).
_ANY_YEARS_RE = re.compile(r'(\d{1,2}(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)', re.IGNORECASE)


def _token_to_decimal_year(token: str, now: datetime.date) -> Optional[float]:
    """Convert one side of a date range to a decimal year (e.g. 2018.5)."""
    t = token.strip().lower()
    if any(k in t for k in ("present", "current", "now", "date", "ongoing")):
        return now.year + (now.month - 1) / 12.0

    m = re.search(r'(\d{1,2})[/\-.](\d{4})', t)  # MM/YYYY
    if m:
        mo, yr = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12 and _MIN_YEAR <= yr <= now.year + _MAX_AHEAD:
            return yr + (mo - 1) / 12.0

    m = re.search(r'([a-z]{3,9})\.?\s+(\d{4})', t)  # Month YYYY
    if m and m.group(1)[:3] in _MONTHS:
        yr = int(m.group(2))
        if _MIN_YEAR <= yr <= now.year + _MAX_AHEAD:
            return yr + (_MONTHS[m.group(1)[:3]] - 1) / 12.0

    m = re.search(r'(\d{4})', t)  # YYYY
    if m:
        yr = int(m.group(1))
        if _MIN_YEAR <= yr <= now.year + _MAX_AHEAD:
            return float(yr)

    return None


def _is_education_context(text: str, start_idx: int) -> bool:
    """
    True if the date range at start_idx sits on an education line
    (e.g. "ABES Engineering College 2022 – 2026", "B.Tech ... CGPA 7.6").
    Such ranges are degree/college timelines, NOT professional experience,
    and must be excluded so a student's 4-year course isn't counted as 4 years
    of work experience.
    """
    line_start = text.rfind('\n', 0, start_idx) + 1
    line_end = text.find('\n', start_idx)
    if line_end == -1:
        line_end = len(text)
    return bool(_EDU_CONTEXT_RE.search(text[line_start:line_end]))


def _experience_from_date_ranges(text: str, now: datetime.date) -> float:
    """
    Estimate professional experience by summing employment date ranges,
    merging overlapping periods so concurrent roles aren't double-counted.
    Date ranges that appear on an education line (degree/college timelines)
    are skipped — they are not work experience.
    """
    intervals = []
    for m in _RANGE_RE.finditer(text):
        if _is_education_context(text, m.start()):
            continue
        start = _token_to_decimal_year(m.group(1), now)
        end = _token_to_decimal_year(m.group(2), now)
        if start is None or end is None:
            continue
        if end < start or (end - start) > 50:
            continue
        intervals.append((start, end))

    if not intervals:
        return 0.0

    intervals.sort()
    total = 0.0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:  # overlap → extend
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    total += cur_end - cur_start
    return total


def extract_experience_years(text: str) -> float:
    """
    Estimate total years of experience.

    Priority:
      1. An explicit claim tied to the word "experience" ("5+ years experience").
      2. Otherwise, the merged span of employment date ranges in the resume.
      3. Otherwise, any "N years" mention as a last resort.

    Capped at 50 years to ignore typos / spurious numbers.
    """
    if not text:
        return 0.0

    now = datetime.date.today()

    explicit = [float(v) for v in _EXP_PHRASE_RE.findall(text) if float(v) <= 50]
    if explicit:
        return round(max(explicit), 1)

    from_dates = _experience_from_date_ranges(text, now)
    if from_dates > 0:
        return round(min(from_dates, 50.0), 1)

    any_years = [float(v) for v in _ANY_YEARS_RE.findall(text) if float(v) <= 50]
    if any_years:
        return round(max(any_years), 1)

    return 0.0


# A required-experience *range*: "1 to 3 years", "1-3 yrs", "2 – 5 years".
_EXP_RANGE_RE = re.compile(
    r'(\d{1,2}(?:\.\d+)?)\s*(?:-|–|—|to|through)\s*(\d{1,2}(?:\.\d+)?)'
    r'\s*\+?\s*(?:years?|yrs?)',
    re.IGNORECASE,
)

# A lower bound only: "3+ years", "minimum 3 years", "at least 2 yrs".
_EXP_MIN_RE = re.compile(
    r'(?:(?:min(?:imum)?|at\s*least|atleast|over|more\s*than)\s*)?'
    r'(\d{1,2}(?:\.\d+)?)\s*\+\s*(?:years?|yrs?)'
    r'|(?:min(?:imum)?|at\s*least|atleast|over|more\s*than)\s*'
    r'(\d{1,2}(?:\.\d+)?)\s*(?:years?|yrs?)',
    re.IGNORECASE,
)


def extract_experience_range(text: str) -> dict:
    """
    Extract a required-experience range from a job description.

    Returns {"min": float|None, "max": float|None}:
      • "1 to 3 years"        → {min: 1, max: 3}
      • "3+ years" / "min 3"  → {min: 3, max: None}  (open-ended upper bound)
      • a single "4 years"    → {min: 4, max: 4}
      • nothing detectable    → {min: None, max: None}

    A `max` of None means "no upper limit" — more experience is always welcome.
    """
    if not text:
        return {"min": None, "max": None}

    m = _EXP_RANGE_RE.search(text)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        if lo <= 50 and hi <= 50:
            return {"min": lo, "max": hi}

    m = _EXP_MIN_RE.search(text)
    if m:
        val = m.group(1) or m.group(2)
        v = float(val)
        if v <= 50:
            return {"min": v, "max": None}

    single = extract_experience_years(text)
    if single > 0:
        return {"min": single, "max": single}

    return {"min": None, "max": None}


# ─────────────────────────────────────────────
# EDUCATION EXTRACTION
# ─────────────────────────────────────────────

EDUCATION_KEYWORDS = {
    "phd": 5,
    "ph.d": 5,
    "doctorate": 5,
    "master": 4,
    "m.tech": 4,
    "m.e.": 4,
    "m.sc": 4,
    "mba": 4,
    "m.b.a": 4,
    "bachelor": 3,
    "b.tech": 3,
    "b.e.": 3,
    "b.sc": 3,
    "b.com": 3,
    "b.ca": 3,
    "bca": 3,
    "diploma": 2,
    "associate": 2,
    "12th": 1,
    "high school": 1,
    "hsc": 1,
    "ssc": 1,
}


def extract_education(text: str) -> dict:
    """
    Detect highest education level from text.
    Returns dict with: level (string), score (int 0-5)
    """
    text_lower = text.lower()
    best_score = 0
    best_level = "Not Specified"

    for keyword, score in EDUCATION_KEYWORDS.items():
        if keyword in text_lower and score > best_score:
            best_score = score
            best_level = keyword.upper().replace(".", "")

    return {"level": best_level, "score": best_score}


# School-level qualifications are ignored when reading a JD's bar unless the JD
# names nothing higher, since a job advert often mentions "12th pass" only in
# passing (e.g. describing the labour force, not the engineer being hired).
_SCHOOL_LEVEL = 1


def extract_education_requirement(text: str) -> int:
    """
    Detect the MINIMUM education level a job description asks for (0-5).

    A resume is read for the candidate's *highest* qualification; a job
    description has to be read for its *lowest* — the bar the candidate must
    clear. "Bachelor's degree required, Master's preferred" sets the bar at
    bachelor, so taking the maximum (as `extract_education` does) would mark
    every non-master candidate as falling short of a requirement that was never
    made.

    Returns 0 when the JD states no qualification at all, which the scorer
    treats as "no bar stated".
    """
    if not text:
        return 0

    text_lower = text.lower()
    found = {score for kw, score in EDUCATION_KEYWORDS.items() if kw in text_lower}
    if not found:
        return 0

    above_school = {s for s in found if s > _SCHOOL_LEVEL}
    return min(above_school) if above_school else min(found)


# ─────────────────────────────────────────────
# LOCATION EXTRACTION
# ─────────────────────────────────────────────

# Common cities/states for quick matching. "remote" and the work-from-home
# variants live here too, because a resume that says "remote" is stating a
# location preference and the JD may be answering it.
LOCATION_KEYWORDS = [
    # India — metros
    "mumbai", "navi mumbai", "thane", "delhi", "new delhi", "bangalore",
    "bengaluru", "hyderabad", "pune", "chennai", "kolkata", "ahmedabad",
    # India — other cities we see on construction resumes
    "jaipur", "surat", "lucknow", "noida", "gurgaon", "gurugram", "chandigarh",
    "bhopal", "indore", "nagpur", "vadodara", "vizag", "visakhapatnam",
    "coimbatore", "patna", "bhubaneswar", "kochi", "ernakulam", "trivandrum",
    "thiruvananthapuram", "mysore", "mysuru", "mangalore", "nashik",
    "aurangabad", "rajkot", "ludhiana", "kanpur", "agra", "varanasi", "ranchi",
    "raipur", "guwahati", "dehradun", "faridabad", "ghaziabad", "goa", "panaji",
    "madurai", "salem", "vijayawada", "guntur", "warangal", "jodhpur",
    "udaipur", "kota", "gwalior", "jabalpur", "amritsar", "jalandhar",
    "jamshedpur", "dhanbad", "meerut", "srinagar", "jammu", "shimla",
    # USA
    "new york", "san francisco", "los angeles", "chicago", "seattle", "austin",
    "boston", "dallas", "miami", "denver", "atlanta", "washington",
    # Global
    "london", "toronto", "dubai", "abu dhabi", "doha", "riyadh", "muscat",
    "singapore", "sydney", "berlin", "paris",
    # Not a city, but a location answer all the same
    "remote", "work from home", "wfh",
]

# Word-boundary patterns, letters only on each side. Plain substring matching
# produced false hits: "working remotely from Jaipur" matched "remote", and any
# city sitting inside a longer word (a surname like Agrawal, a college name)
# matched too. Digits are deliberately NOT treated as a boundary so a pincode
# run together with the city ("Pune411001") still matches.
_LOCATION_RE = {
    loc: re.compile(r'(?<![a-z])' + re.escape(loc) + r'(?![a-z])')
    for loc in LOCATION_KEYWORDS
}


def extract_location(text: str) -> Optional[str]:
    """
    Return the location mentioned EARLIEST in the text.

    Every keyword is searched and the winner is the one with the lowest
    position — not the one that happens to sit first in LOCATION_KEYWORDS.
    The old list-order scan was badly wrong in ordinary cases: a resume headed
    "Pune, Maharashtra" that later mentioned a previous posting in Mumbai came
    back as Mumbai, because "mumbai" is earlier in the list. On a Pune vacancy
    that scored the candidate 0 for location — 15% of the total — despite the
    candidate living in Pune.

    Earliest-in-text is the right rule because a resume states its own city in
    the header and a JD states the posting near the top; later mentions are
    almost always previous employers or nice-to-haves.

    Ties at the same start index are broken by the longer keyword, so
    "Navi Mumbai" beats the "Mumbai" sitting inside it.
    """
    if not text:
        return None

    text_lower = text.lower()
    best_pos: Optional[int] = None
    best_loc: Optional[str] = None

    for loc in LOCATION_KEYWORDS:
        match = _LOCATION_RE[loc].search(text_lower)
        if not match:
            continue
        pos = match.start()
        if best_pos is None or pos < best_pos or (pos == best_pos and len(loc) > len(best_loc)):
            best_pos, best_loc = pos, loc

    return best_loc.title() if best_loc else None


# ─────────────────────────────────────────────
# NAME EXTRACTION (spaCy fallback)
# ─────────────────────────────────────────────

def extract_name_regex(text: str) -> Optional[str]:
    """
    Heuristic name extraction:
    - First non-empty line that looks like a person's name (2-4 words, capitalized)
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for line in lines[:10]:  # Check first 10 lines only
        words = line.split()
        if 2 <= len(words) <= 4:
            if all(w[0].isupper() for w in words if w.isalpha()):
                # Exclude lines that look like headers or addresses
                skip_words = {"resume", "curriculum", "vitae", "cv", "profile",
                              "objective", "summary", "contact", "email", "phone"}
                if not any(w.lower() in skip_words for w in words):
                    return line
    return None


# ─────────────────────────────────────────────
# FILE UTILITIES
# ─────────────────────────────────────────────

def get_file_extension(filename: str) -> str:
    """Return lowercase file extension including dot, e.g. '.pdf'"""
    return os.path.splitext(filename)[1].lower()


def is_valid_resume_file(filename: str) -> bool:
    """Check if uploaded file has a supported extension."""
    return get_file_extension(filename) in {'.pdf', '.docx', '.doc'}


def truncate_text(text: str, max_chars: int = 5000) -> str:
    """Truncate text for display purposes."""
    return text[:max_chars] + "..." if len(text) > max_chars else text
