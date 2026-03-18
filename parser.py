"""
parser.py - Resume Parsing Module
Handles PDF and DOCX file reading, text extraction, and candidate info extraction.
Uses pdfplumber for PDFs, python-docx for DOCX, spaCy for NER, and regex utilities.
"""

import io
import re
from typing import Optional

# PDF parsing
try:
    import pdfplumber
    PDF_BACKEND = "pdfplumber"
except ImportError:
    try:
        import PyPDF2
        PDF_BACKEND = "pypdf2"
    except ImportError:
        PDF_BACKEND = None

# DOCX parsing
try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# spaCy NER (optional — graceful fallback)
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False
    nlp = None

from utils import (
    clean_text,
    extract_email,
    extract_phone,
    extract_experience_years,
    extract_education,
    extract_location,
    extract_name_regex,
)
from skills import extract_skills


# ─────────────────────────────────────────────
# FILE TEXT EXTRACTION
# ─────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract raw text from a PDF file given as bytes.
    Uses pdfplumber (preferred) or PyPDF2 as fallback.
    """
    text = ""
    try:
        if PDF_BACKEND == "pdfplumber":
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        elif PDF_BACKEND == "pypdf2":
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        else:
            raise ImportError("No PDF library available. Install pdfplumber or PyPDF2.")
    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {e}")

    return clean_text(text)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Extract raw text from a DOCX file given as bytes.
    Uses python-docx library.
    """
    if not DOCX_AVAILABLE:
        raise ImportError("python-docx not installed. Run: pip install python-docx")
    try:
        doc = DocxDocument(io.BytesIO(file_bytes))
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        text = "\n".join(paragraphs)
    except Exception as e:
        raise ValueError(f"Failed to parse DOCX: {e}")

    return clean_text(text)


def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Route to correct parser based on file extension.
    Returns clean extracted text string.
    """
    ext = filename.lower().split(".")[-1]
    if ext == "pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in ("docx", "doc"):
        return extract_text_from_docx(file_bytes)
    else:
        raise ValueError(f"Unsupported file format: .{ext}")


# ─────────────────────────────────────────────
# FIELD EXTRACTORS
# ─────────────────────────────────────────────

def extract_name(text: str) -> str:
    """
    Extract candidate name using spaCy NER (PERSON entities) with regex fallback.
    """
    if SPACY_AVAILABLE and nlp:
        # Process only the first 500 chars (name is almost always near top)
        doc = nlp(text[:500])
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip()
                # Filter out very short or suspiciously long names
                if 2 <= len(name.split()) <= 4 and len(name) < 50:
                    return name

    # Fallback: regex heuristic
    name = extract_name_regex(text)
    return name if name else "Unknown"


def extract_location_from_resume(text: str) -> str:
    """
    Extract location using spaCy GPE/LOC entities, then keyword fallback.
    """
    if SPACY_AVAILABLE and nlp:
        doc = nlp(text[:1000])
        for ent in doc.ents:
            if ent.label_ in ("GPE", "LOC"):
                location = ent.text.strip()
                if len(location) > 2:
                    return location

    # Fallback: keyword matching
    loc = extract_location(text)
    return loc if loc else "Not Specified"


# ─────────────────────────────────────────────
# MAIN PARSER FUNCTION
# ─────────────────────────────────────────────

def parse_resume(file_bytes: bytes, filename: str) -> dict:
    """
    Full resume parsing pipeline.
    Extracts all candidate information from a resume file.

    Args:
        file_bytes: Raw bytes of the uploaded resume file
        filename:   Original filename (used to detect format)

    Returns:
        dict with keys: name, email, phone, skills, experience_years,
                         education, location, raw_text, filename, error
    """
    result = {
        "filename": filename,
        "name": "Unknown",
        "email": None,
        "phone": None,
        "skills": [],
        "experience_years": 0.0,
        "education": {"level": "Not Specified", "score": 0},
        "location": "Not Specified",
        "raw_text": "",
        "error": None,
    }

    try:
        # Step 1: Extract raw text
        raw_text = extract_text(file_bytes, filename)
        result["raw_text"] = raw_text

        if not raw_text.strip():
            result["error"] = "Could not extract text from file (may be scanned/image PDF)"
            return result

        # Step 2: Extract each field
        result["name"]             = extract_name(raw_text)
        result["email"]            = extract_email(raw_text)
        result["phone"]            = extract_phone(raw_text)
        result["skills"]           = extract_skills(raw_text)
        result["experience_years"] = extract_experience_years(raw_text)
        result["education"]        = extract_education(raw_text)
        result["location"]         = extract_location_from_resume(raw_text)

    except Exception as e:
        result["error"] = str(e)

    return result
