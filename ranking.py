"""
ranking.py - Candidate Ranking Engine

Final Score = (0.50 × skill) + (0.25 × experience) + (0.10 × education) + (0.15 × location)

The skill component combines two signals:
  • JD-skill coverage  — fraction of the job's required skills found in the resume.
  • Semantic relevance — TF-IDF cosine similarity between the full resume text
    and the full job description, computed across the whole uploaded batch.
    This closes part of the gap coverage leaves rather than being averaged in,
    so it can lift an under-detected resume but never caps a complete one.

Education is scored against the qualification the JD asks for, not an absolute
ladder — meeting the stated bar scores full marks.

The semantic signal (pure Python, no extra dependencies) makes ranking robust
when resumes describe relevant work using words outside the skill dictionary,
and it lets the system still rank candidates even when a JD lists no detectable
hard skills. It is also used as a tiebreaker so near-equal candidates order by
overall relevance.
"""

import math
import re
from collections import Counter

from skills import match_skills, extract_skills, extract_skill_weights
from utils import (
    extract_experience_years,
    extract_experience_range,
    extract_education,
    extract_education_requirement,
    extract_location,
)

# ─────────────────────────────────────────────
# SCORING WEIGHTS
# ─────────────────────────────────────────────
WEIGHTS = {
    "skills":     0.50,
    "experience": 0.25,
    "education":  0.10,
    "location":   0.15,
}

# How much of the gap left by exact JD-skill coverage the semantic signal is
# allowed to close. Semantic relevance is a *supporting* signal: it exists to
# rescue a resume that plainly does the job but words it outside the skill
# dictionary. It is deliberately not able to move a candidate who already
# covers every stated requirement, because there is nothing left to rescue.
SKILL_SEMANTIC_LIFT = 0.5

# Experience is only valuable when it's *relevant* experience: 7 years of sales
# work should not boost a candidate for a civil-engineering role. We therefore
# scale the experience score by the candidate's role-fit. The floor keeps a
# small baseline so a strong candidate whose skills are slightly under-detected
# isn't wiped out, while a clearly off-target candidate loses most of the credit.
EXPERIENCE_RELEVANCE_FLOOR = 0.2

MAX_EXPERIENCE_YEARS = 15.0
EDUCATION_SCORE_MAP = {0: 0, 1: 20, 2: 40, 3: 60, 4: 80, 5: 100}

# When the JD asks for an experience range (e.g. "1 to 3 years"), candidates who
# fall inside the range score strongly here, while candidates who EXCEED the
# range earn extra credit on top — more experience than required is a priority,
# not a penalty. We leave headroom below 100 for in-range candidates so the
# "exceeds requirement" bonus has somewhere to go.
EXPERIENCE_IN_RANGE_SCORE = 90.0


# ─────────────────────────────────────────────
# SEMANTIC SIMILARITY  (lightweight TF-IDF cosine, no external deps)
# ─────────────────────────────────────────────

_TOKEN_RE = re.compile(r'[a-z0-9+#./]+')
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "for", "with", "on", "at",
    "by", "is", "are", "be", "as", "we", "our", "you", "your", "this", "that",
    "it", "its", "from", "will", "have", "has", "had", "was", "were", "they",
    "their", "them", "i", "me", "my", "us", "but", "if", "so", "do", "does",
    "can", "may", "should", "would", "could", "etc", "per", "via", "into",
    "about", "over", "more", "than", "who", "which", "what", "when", "where",
}


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in _STOPWORDS]


def _build_tfidf_vectors(token_docs: list[list[str]]) -> list[dict]:
    """Turn tokenized documents into L2-normalizable TF-IDF weight dicts."""
    n_docs = len(token_docs)
    df = Counter()
    for doc in token_docs:
        for term in set(doc):
            df[term] += 1

    idf = {term: math.log((n_docs + 1) / (count + 1)) + 1.0 for term, count in df.items()}

    vectors = []
    for doc in token_docs:
        if not doc:
            vectors.append({})
            continue
        tf = Counter(doc)
        length = len(doc)
        vectors.append({term: (count / length) * idf[term] for term, count in tf.items()})
    return vectors


def _cosine(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(weight * b.get(term, 0.0) for term, weight in a.items())
    if dot == 0.0:
        return 0.0
    norm_a = math.sqrt(sum(w * w for w in a.values()))
    norm_b = math.sqrt(sum(w * w for w in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_semantic_scores(resume_texts: list[str], jd_text: str) -> list[float]:
    """
    Return a 0-1 relevance score for each resume vs. the job description, using
    TF-IDF cosine similarity with IDF learned across the whole batch.
    """
    docs = [jd_text or ""] + list(resume_texts)
    token_docs = [_tokenize(d) for d in docs]
    vectors = _build_tfidf_vectors(token_docs)
    jd_vec = vectors[0]
    return [_cosine(jd_vec, vectors[i + 1]) for i in range(len(resume_texts))]


# ─────────────────────────────────────────────
# INDIVIDUAL SCORE COMPONENTS
# ─────────────────────────────────────────────

def compute_skill_score(coverage: float, semantic_0_1: float, jd_has_skills: bool) -> float:
    """
    Combine exact JD-skill coverage (0-100) with semantic relevance (0-1).

    Semantic relevance closes part of the gap that coverage leaves behind,
    rather than being averaged in alongside it. The old formulation was
    `0.85 × coverage + 0.15 × semantic`, which meant the semantic term could
    only ever pull a good candidate DOWN: two different documents — a resume
    and a job advert — never score near 1.0 on cosine similarity, so a
    candidate holding every single skill the JD asked for was capped around 94
    and, through the experience relevance multiplier below, docked again on
    experience. Covering the whole requirement now scores the whole 100.

    A weak candidate can still be lifted by writing about genuinely relevant
    work in words the skill dictionary does not carry, which is the reason the
    semantic signal exists. Unrelated resumes score ~0.03-0.05 here, so the
    lift they receive is negligible.

    If the JD has no detectable skills at all, fall back entirely to semantic
    relevance so candidates can still be ranked.
    """
    if not jd_has_skills:
        return round(semantic_0_1 * 100.0, 1)

    lifted = coverage + (100.0 - coverage) * semantic_0_1 * SKILL_SEMANTIC_LIFT
    return round(min(lifted, 100.0), 1)


def compute_experience_score(resume_years: float,
                             exp_min: float | None,
                             exp_max: float | None = None) -> float:
    """
    Score a candidate's years of experience against the JD's required range.

      • No requirement stated      → reward more experience up to a 15-year cap.
      • Below the minimum          → ramp up toward the in-range score.
      • Inside the range           → strong score (EXPERIENCE_IN_RANGE_SCORE).
      • Above the range (or min)    → in-range score PLUS a bonus that grows with
                                      the surplus, so more experience ranks higher.
    """
    has_min = exp_min is not None and exp_min > 0
    has_max = exp_max is not None and exp_max > 0

    # No explicit requirement: more experience is better, capped at MAX years.
    if not has_min and not has_max:
        return min(resume_years / MAX_EXPERIENCE_YEARS, 1.0) * 100.0

    lo = exp_min if has_min else 0.0
    hi = exp_max if has_max else lo  # open-ended upper bound mirrors the minimum

    # Below the minimum: proportional ramp up to the in-range score.
    if lo > 0 and resume_years < lo:
        return (resume_years / lo) * EXPERIENCE_IN_RANGE_SCORE

    # Inside the range: full in-range credit.
    if hi <= 0 or resume_years <= hi:
        return EXPERIENCE_IN_RANGE_SCORE

    # Above the range: reward the surplus, ramping from in-range up to 100.
    surplus = (resume_years - hi) / hi
    bonus = min(surplus, 1.0) * (100.0 - EXPERIENCE_IN_RANGE_SCORE)
    return EXPERIENCE_IN_RANGE_SCORE + bonus


def compute_education_score(education: dict, required_level: int = 0) -> float:
    """
    Score the candidate's qualification against the bar the JD actually set.

    Education used to be the only component judged on an absolute ladder rather
    than against the job: a B.E. scored 60 whatever the vacancy, so a candidate
    who exactly met a "B.E. in Civil Engineering" requirement still lost marks,
    and an MBA — worth 80 on that ladder — outscored the civil engineer on a
    civil site role. Skills, experience and location are all judged against the
    JD; this now is too.

    Meeting the bar scores full. Exceeding it scores full as well and no more —
    a second degree beyond what the job asked for is not a better fit for the
    job, and paying for it here is how "overqualified" candidates float to the
    top of a hands-on shortlist. Falling short ramps down proportionally.

    With no bar stated the old absolute ladder is kept, since there is nothing
    better to compare against.
    """
    candidate_level = education.get("score", 0)

    if not required_level:
        return float(EDUCATION_SCORE_MAP.get(candidate_level, 0))

    if candidate_level <= 0:
        return 0.0
    if candidate_level >= required_level:
        return 100.0
    return (candidate_level / required_level) * 100.0


def compute_location_score(resume_location: str, jd_location: str) -> float:
    if not jd_location or jd_location.lower() in ("not specified", "any", ""):
        return 50.0

    jd_loc = jd_location.lower().strip()
    res_loc = resume_location.lower().strip() if resume_location else ""

    if jd_loc == "remote" or res_loc in ("remote", "work from home", "wfh"):
        return 100.0
    if res_loc in ("not specified", ""):
        return 30.0
    if jd_loc in res_loc or res_loc in jd_loc:
        return 100.0
    return 0.0


# ─────────────────────────────────────────────
# JD PARSER
# ─────────────────────────────────────────────

def parse_job_description(jd_text: str) -> dict:
    exp_range = extract_experience_range(jd_text)
    # `experience_years` (the lower bound) is kept for display / backward-compat.
    display_years = exp_range["min"] if exp_range["min"] is not None else 0.0
    return {
        "skills":           extract_skills(jd_text),
        "skill_weights":    extract_skill_weights(jd_text),
        "experience_years": display_years,
        "experience_min":   exp_range["min"],
        "experience_max":   exp_range["max"],
        "education_min":    extract_education_requirement(jd_text),
        "location":         extract_location(jd_text) or "Not Specified",
        "raw_text":         jd_text,
    }


# ─────────────────────────────────────────────
# PLAIN-LANGUAGE ASSESSMENT  (analyst-style verdict per candidate)
# ─────────────────────────────────────────────

# Generic / soft skills shouldn't be treated as the role-defining "core"
# requirement — the core is the hard, role-specific skill (CCTV, AutoCAD, SEO…).
_SOFT_SKILLS = {
    "communication", "teamwork", "leadership", "team management", "problem solving",
    "time management", "presentation", "public speaking", "customer service",
    "ms office", "ms excel", "ms word", "ms powerpoint", "outlook", "data entry",
}


def _fmt_years(x) -> str:
    """Format a year value without a trailing '.0'."""
    if x is None:
        return ""
    return f"{x:g}"


def _verdict_label(final_score: float) -> tuple[str, str]:
    """Map the composite score to a verdict label and a severity tag."""
    if final_score >= 75:
        return "Excellent fit", "good"
    if final_score >= 60:
        return "Strong fit", "good"
    if final_score >= 45:
        return "Moderate fit", "ok"
    if final_score >= 30:
        return "Weak fit", "warn"
    return "Poor fit", "bad"


def build_assessment(skill_score: float, coverage: float, matched: list, missing: list,
                     years: float, exp_min, exp_max, loc_score: float,
                     final_score: float, jd_weights: dict | None) -> dict:
    """
    Produce an analyst-style verdict: a one-line summary plus bullet notes that
    explain *why* the candidate scored as they did — covering core-skill fit,
    experience vs. the required range, over-/under-qualification, and location.
    This mirrors a human reviewer's written assessment rather than raw numbers.
    """
    label, tag = _verdict_label(final_score)
    notes: list[str] = []

    # ── Core requirement: the JD's most-emphasised *hard* skill(s). Soft skills
    # (communication, MS Office…) are excluded unless the JD lists nothing else.
    core_skills: list[str] = []
    if jd_weights:
        hard_weights = {s: w for s, w in jd_weights.items() if s.lower() not in _SOFT_SKILLS}
        pool = hard_weights or jd_weights
        max_w = max(pool.values())
        if max_w >= 2:  # only call something "core" if the JD stresses it
            core_skills = [s for s, w in pool.items() if w == max_w]

    matched_set = {m.lower() for m in matched}
    missing_core = [s for s in core_skills if s.lower() not in matched_set]
    matched_core = [s for s in core_skills if s.lower() in matched_set]

    if matched_core:
        notes.append(f"✅ Covers the role's core requirement: {', '.join(matched_core)}.")
    if missing_core:
        notes.append(f"⚠️ Missing the role's core requirement: {', '.join(missing_core)}.")

    # ── Overall skills coverage.
    if not jd_weights and not matched and not missing:
        notes.append("No specific hard skills listed in the JD — ranked on overall text relevance.")
    elif coverage >= 70:
        notes.append(f"Matches most required skills ({coverage:.0f}% weighted coverage).")
    elif coverage >= 40:
        notes.append(f"Partial skills match ({coverage:.0f}% weighted coverage).")
    else:
        notes.append(f"Few of the required skills are present ({coverage:.0f}% weighted coverage).")

    # ── Experience vs. the required range.
    has_min = exp_min is not None and exp_min > 0
    has_max = exp_max is not None and exp_max > 0
    overqualified = False
    if has_min:
        lo = exp_min
        hi = exp_max if has_max else None
        rng = f"{_fmt_years(lo)}–{_fmt_years(hi)}" if hi else f"{_fmt_years(lo)}+"
        if years < lo:
            notes.append(f"Under-experienced: {years:.1f} yrs vs. a {rng} yr requirement.")
        elif hi and years > hi:
            notes.append(f"Above the target: {years:.1f} yrs vs. a {rng} yr range.")
            # Lots of extra tenure + weak skill fit = classic overqualified mismatch.
            if coverage < 45 and years >= hi * 1.5:
                overqualified = True
        else:
            notes.append(f"Experience in range: {years:.1f} yrs (target {rng} yrs).")
    else:
        notes.append(f"{years:.1f} yrs of experience.")

    if overqualified:
        if years >= 8:
            notes.append(
                "🚩 Likely overqualified — senior/manager-level tenure but light on this "
                "role's hands-on skills; expect a higher salary and a possible mismatch."
            )
        else:
            notes.append(
                "🚩 More experience than this role needs but few matching skills — "
                "likely a mismatch; confirm genuine interest and fit."
            )
        if tag in ("ok", "good"):
            tag = "warn"

    # ── Location.
    if loc_score >= 100:
        notes.append("Location matches the JD.")
    elif loc_score <= 0:
        notes.append("Location mismatch — confirm willingness to relocate.")
    elif loc_score <= 30:
        notes.append("Location not stated on the resume — confirm.")

    # ── Key skill gaps (most-emphasised missing skills first).
    if missing:
        ordered_missing = sorted(missing, key=lambda s: (jd_weights or {}).get(s, 1), reverse=True)
        gaps = [g for g in ordered_missing if g not in missing_core][:3]
        if gaps:
            notes.append(f"Key gaps: {', '.join(gaps)}.")

    # ── One-line summary.
    if missing_core:
        headline = f"{label} — missing the core requirement ({', '.join(missing_core)})."
    elif overqualified:
        headline = f"{label} — overqualified for a hands-on role."
    elif coverage >= 70:
        headline = f"{label} — strong skills match and suitable experience."
    else:
        headline = f"{label} — partial match; review the gaps below."

    return {"label": label, "tag": tag, "summary": headline, "notes": notes}


# ─────────────────────────────────────────────
# MAIN SCORING FUNCTION
# ─────────────────────────────────────────────

def score_candidate(candidate: dict, jd: dict, semantic_0_1: float = 0.0) -> dict:
    """Compute the composite score for a single candidate against the JD."""
    skill_match = match_skills(
        candidate.get("skills", []), jd.get("skills", []), jd.get("skill_weights")
    )
    coverage = skill_match["score"]
    jd_has_skills = bool(jd.get("skills"))

    skill_score = compute_skill_score(coverage, semantic_0_1, jd_has_skills)
    raw_exp_score = compute_experience_score(
        candidate.get("experience_years", 0.0),
        jd.get("experience_min", jd.get("experience_years", 0.0)),
        jd.get("experience_max"),
    )
    edu_score = compute_education_score(
        candidate.get("education", {}), jd.get("education_min", 0)
    )
    loc_score = compute_location_score(candidate.get("location", ""), jd.get("location", ""))

    # Down-weight experience for candidates who don't fit the role.
    relevance = skill_score / 100.0
    exp_score = raw_exp_score * (
        EXPERIENCE_RELEVANCE_FLOOR + (1.0 - EXPERIENCE_RELEVANCE_FLOOR) * relevance
    )

    final_score = (
        WEIGHTS["skills"]     * skill_score +
        WEIGHTS["experience"] * exp_score +
        WEIGHTS["education"]   * edu_score +
        WEIGHTS["location"]    * loc_score
    )

    assessment = build_assessment(
        skill_score=skill_score,
        coverage=coverage,
        matched=skill_match.get("matched", []),
        missing=skill_match.get("missing", []),
        years=candidate.get("experience_years", 0.0),
        exp_min=jd.get("experience_min"),
        exp_max=jd.get("experience_max"),
        loc_score=loc_score,
        final_score=final_score,
        jd_weights=jd.get("skill_weights"),
    )

    return {
        **candidate,
        "score_breakdown": {
            "skill_score": round(skill_score, 1),
            "exp_score":   round(exp_score, 1),
            "edu_score":   round(edu_score, 1),
            "loc_score":   round(loc_score, 1),
            "coverage":    round(coverage, 1),
            "semantic":    round(semantic_0_1 * 100, 1),
        },
        "matched_skills": skill_match.get("matched", []),
        "missing_skills": skill_match.get("missing", []),
        "semantic_score": round(semantic_0_1 * 100, 1),
        "final_score":    round(final_score, 2),
        "assessment":     assessment,
    }


def rank_candidates(candidates: list[dict], jd: dict) -> list[dict]:
    """Score and rank all candidates against the job description."""
    # Semantic relevance is computed across the whole batch so IDF is meaningful.
    resume_texts = [c.get("raw_text", "") for c in candidates]
    semantic = compute_semantic_scores(resume_texts, jd.get("raw_text", ""))

    scored = []
    for candidate, sem in zip(candidates, semantic):
        if candidate.get("error"):
            scored.append({
                **candidate,
                "final_score": 0.0,
                "score_breakdown": {},
                "matched_skills": [],
                "missing_skills": [],
                "semantic_score": 0.0,
                "assessment": {
                    "label": "Could not parse",
                    "tag": "bad",
                    "summary": "Could not parse — resume text could not be read.",
                    "notes": [f"Parsing error: {candidate.get('error', 'unknown')}"],
                },
            })
        else:
            scored.append(score_candidate(candidate, jd, sem))

    # Sort by final score, then semantic relevance, then matched-skill count.
    scored.sort(
        key=lambda x: (
            x["final_score"],
            x.get("semantic_score", 0.0),
            len(x.get("matched_skills", [])),
        ),
        reverse=True,
    )

    for i, candidate in enumerate(scored):
        candidate["rank"] = i + 1

    return scored
