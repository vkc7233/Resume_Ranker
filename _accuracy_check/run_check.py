"""
Accuracy check: score a resume against a job description written FROM that
resume. A tailor-made JD is the easiest possible case, so whatever the engine
loses here is a defect, not a judgement call.

Run from the project root:  python _accuracy_check/run_check.py
"""
import os
import sys

# The verdict notes contain emoji; the Windows console defaults to cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skills import extract_skills, extract_skill_weights, match_skills   # noqa: E402
from utils import (                                                       # noqa: E402
    extract_experience_years,
    extract_education,
    extract_location,
    extract_email,
    extract_phone,
    extract_name_regex,
)
from ranking import parse_job_description, rank_candidates               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as fh:
        return fh.read()


def build_candidate(text, filename):
    """Mirror parser.parse_resume's field extraction, minus the PDF/DOCX step."""
    return {
        "filename": filename,
        "name": extract_name_regex(text) or "Unknown",
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": extract_skills(text),
        "experience_years": extract_experience_years(text),
        "education": extract_education(text),
        "location": extract_location(text) or "Not Specified",
        "raw_text": text,
        "error": None,
    }


def main():
    resume_text = load("resume_rahul_deshmukh.txt")
    jd_text = load("jd_site_engineer_civil.txt")

    jd = parse_job_description(jd_text)
    cand = build_candidate(resume_text, "resume_rahul_deshmukh.txt")

    print("=" * 68)
    print("JD AS PARSED")
    print("=" * 68)
    print(f"  location        : {jd['location']}")
    print(f"  experience min  : {jd['experience_min']}")
    print(f"  experience max  : {jd['experience_max']}")
    print(f"  skills detected : {len(jd['skills'])}")
    for s in jd["skills"]:
        print(f"      {s:<26} weight {jd['skill_weights'].get(s, 1)}")

    print()
    print("=" * 68)
    print("RESUME AS PARSED")
    print("=" * 68)
    print(f"  name            : {cand['name']}")
    print(f"  email           : {cand['email']}")
    print(f"  phone           : {cand['phone']}")
    print(f"  location        : {cand['location']}")
    print(f"  experience      : {cand['experience_years']} yrs")
    print(f"  education       : {cand['education']}")
    print(f"  skills detected : {len(cand['skills'])}")
    print(f"      {', '.join(cand['skills'])}")

    ranked = rank_candidates([cand], jd)
    r = ranked[0]
    b = r["score_breakdown"]

    print()
    print("=" * 68)
    print("SCORE")
    print("=" * 68)
    print(f"  skill    {b['skill_score']:>6}   x 0.50 = {0.50 * b['skill_score']:>6.2f}")
    print(f"      coverage  {b['coverage']}")
    print(f"      semantic  {b['semantic']}")
    print(f"  exp      {b['exp_score']:>6}   x 0.25 = {0.25 * b['exp_score']:>6.2f}")
    print(f"  edu      {b['edu_score']:>6}   x 0.10 = {0.10 * b['edu_score']:>6.2f}")
    print(f"  loc      {b['loc_score']:>6}   x 0.15 = {0.15 * b['loc_score']:>6.2f}")
    print(f"  FINAL    {r['final_score']}")
    print()
    print(f"  matched ({len(r['matched_skills'])}): {', '.join(r['matched_skills'])}")
    print(f"  missing ({len(r['missing_skills'])}): {', '.join(r['missing_skills'])}")
    print()
    print(f"  verdict : {r['assessment']['summary']}")
    for n in r["assessment"]["notes"]:
        print(f"      - {n}")


if __name__ == "__main__":
    main()
