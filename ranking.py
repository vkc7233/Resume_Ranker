"""
ranking.py - Candidate Ranking Engine
Implements the scoring algorithm:
  Final Score = (0.50 × skill_match) + (0.25 × experience) + (0.10 × education) + (0.15 × location)
"""

from skills import match_skills, extract_skills
from utils import extract_experience_years, extract_education, extract_location

# ─────────────────────────────────────────────
# SCORING WEIGHTS
# ─────────────────────────────────────────────
WEIGHTS = {
    "skills":     0.50,
    "experience": 0.25,
    "education":  0.10,
    "location":   0.15,
}

# Max expected experience years for normalization
MAX_EXPERIENCE_YEARS = 15.0

# Education score map: raw score (0-5) → normalized 0-100
EDUCATION_SCORE_MAP = {0: 0, 1: 20, 2: 40, 3: 60, 4: 80, 5: 100}


# ─────────────────────────────────────────────
# INDIVIDUAL SCORE COMPONENTS
# ─────────────────────────────────────────────

def compute_skill_score(resume_skills: list[str], jd_skills: list[str]) -> float:
    """
    Skill match score = (# JD skills matched in resume) / (# JD skills) × 100
    Returns 0.0 if JD has no skills.
    """
    result = match_skills(resume_skills, jd_skills)
    return result["score"]  # Already 0-100


def compute_experience_score(resume_years: float, required_years: float) -> float:
    """
    Experience score:
    - If resume meets/exceeds required years → 100
    - If no requirement specified → normalized score (years / MAX cap)
    - Partial credit proportional to requirement
    Returns 0-100.
    """
    if required_years <= 0:
        # No requirement — give score based on absolute experience (capped at MAX)
        return min(resume_years / MAX_EXPERIENCE_YEARS, 1.0) * 100.0

    if resume_years >= required_years:
        return 100.0

    # Partial credit: resume years / required years, capped at 100
    return min(resume_years / required_years, 1.0) * 100.0


def compute_education_score(education: dict) -> float:
    """
    Education score based on detected degree level (0-5 → 0-100).
    """
    raw_score = education.get("score", 0)
    return EDUCATION_SCORE_MAP.get(raw_score, 0)


def compute_location_score(resume_location: str, jd_location: str) -> float:
    """
    Location score:
    - 100 if location matches or JD says 'remote' or resume says 'remote'
    - 50  if no location requirement in JD
    - 0   if locations differ (penalize mismatch)
    """
    if not jd_location or jd_location.lower() in ("not specified", "any", ""):
        return 50.0  # Neutral — no location preference

    jd_loc   = jd_location.lower().strip()
    res_loc  = resume_location.lower().strip() if resume_location else ""

    if jd_loc == "remote" or res_loc in ("remote", "work from home", "wfh"):
        return 100.0

    if res_loc == "not specified" or res_loc == "":
        return 30.0  # Unknown location, slight penalty

    # Check for substring match (e.g., "Bangalore" in "Bangalore, Karnataka")
    if jd_loc in res_loc or res_loc in jd_loc:
        return 100.0

    return 0.0  # Location mismatch


# ─────────────────────────────────────────────
# JD PARSER
# ─────────────────────────────────────────────

def parse_job_description(jd_text: str) -> dict:
    """
    Extract key requirements from a job description:
    - skills
    - required experience years
    - preferred location

    Args:
        jd_text: Raw job description text

    Returns:
        dict with skills, experience_years, location
    """
    skills          = extract_skills(jd_text)
    experience_years = extract_experience_years(jd_text)
    location         = extract_location(jd_text) or "Not Specified"

    return {
        "skills":           skills,
        "experience_years": experience_years,
        "location":         location,
        "raw_text":         jd_text,
    }


# ─────────────────────────────────────────────
# MAIN SCORING FUNCTION
# ─────────────────────────────────────────────

def score_candidate(candidate: dict, jd: dict) -> dict:
    """
    Compute composite score for a single candidate against the job description.

    Args:
        candidate: Parsed resume dict (output of parser.parse_resume)
        jd:        Parsed job description dict (output of parse_job_description)

    Returns:
        Updated candidate dict with score breakdown and final_score
    """
    # Raw component scores (0-100 each)
    skill_score  = compute_skill_score(candidate.get("skills", []), jd.get("skills", []))
    exp_score    = compute_experience_score(
                       candidate.get("experience_years", 0.0),
                       jd.get("experience_years", 0.0)
                   )
    edu_score    = compute_education_score(candidate.get("education", {}))
    loc_score    = compute_location_score(
                       candidate.get("location", ""),
                       jd.get("location", "")
                   )

    # Weighted final score
    final_score = (
        WEIGHTS["skills"]     * skill_score  +
        WEIGHTS["experience"] * exp_score    +
        WEIGHTS["education"]  * edu_score    +
        WEIGHTS["location"]   * loc_score
    )

    # Skill match details
    skill_match = match_skills(candidate.get("skills", []), jd.get("skills", []))

    return {
        **candidate,
        "score_breakdown": {
            "skill_score":  round(skill_score, 1),
            "exp_score":    round(exp_score, 1),
            "edu_score":    round(edu_score, 1),
            "loc_score":    round(loc_score, 1),
        },
        "matched_skills": skill_match.get("matched", []),
        "missing_skills": skill_match.get("missing", []),
        "final_score":    round(final_score, 2),
    }


def rank_candidates(candidates: list[dict], jd: dict) -> list[dict]:
    """
    Score and rank all candidates against the job description.

    Args:
        candidates: List of parsed resume dicts
        jd:         Parsed job description dict

    Returns:
        List of candidates sorted by final_score (descending) with 'rank' field added
    """
    scored = []
    for candidate in candidates:
        if candidate.get("error"):
            # Keep errored candidates in list but with zero score
            scored.append({
                **candidate,
                "final_score": 0.0,
                "score_breakdown": {},
                "matched_skills": [],
                "missing_skills": [],
            })
        else:
            scored.append(score_candidate(candidate, jd))

    # Sort descending by final_score
    scored.sort(key=lambda x: x["final_score"], reverse=True)

    # Assign rank (1-based)
    for i, candidate in enumerate(scored):
        candidate["rank"] = i + 1

    return scored
