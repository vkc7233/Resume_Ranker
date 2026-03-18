"""
utils.py - Utility / helper functions
Shared helpers used across parser, ranking, and app modules
"""

import re
import os
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

def extract_experience_years(text: str) -> float:
    """
    Extract years of experience mentioned in text.
    Looks for patterns like: "5 years", "3+ years", "2.5 years of experience"
    Returns the maximum found (most likely total), or 0.0 if none found.
    """
    # Pattern: number (int or float) followed by 'year'
    pattern = r'(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        years = [float(m) for m in matches]
        return max(years)
    return 0.0


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


# ─────────────────────────────────────────────
# LOCATION EXTRACTION
# ─────────────────────────────────────────────

# Common cities/states for quick matching
LOCATION_KEYWORDS = [
    # India
    "mumbai", "delhi", "bangalore", "bengaluru", "hyderabad", "pune", "chennai",
    "kolkata", "ahmedabad", "jaipur", "surat", "lucknow", "noida", "gurgaon",
    "gurugram", "chandigarh", "bhopal", "indore", "nagpur", "vadodara", "vizag",
    "visakhapatnam", "coimbatore", "patna", "bhubaneswar", "remote",
    # USA
    "new york", "san francisco", "los angeles", "chicago", "seattle", "austin",
    "boston", "dallas", "miami", "denver", "atlanta", "washington",
    # Global
    "london", "toronto", "dubai", "singapore", "sydney", "berlin", "paris",
    "remote", "work from home", "wfh"
]


def extract_location(text: str) -> Optional[str]:
    """
    Extract first recognizable location from text.
    Returns the city/location string or None.
    """
    text_lower = text.lower()
    for loc in LOCATION_KEYWORDS:
        if loc in text_lower:
            return loc.title()
    return None


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
