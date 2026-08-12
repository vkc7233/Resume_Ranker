"""
End-to-end accuracy check, through the real pipeline.

Reads the generated .docx files with parser.parse_resume — the same call the
uploader makes — so this exercises docx extraction, field extraction, skill
matching and scoring exactly as a recruiter would experience it.

Two things are asserted:
  1. The candidate the JD was written from scores close to full marks.
  2. The batch comes back in the only defensible order.

    python _accuracy_check/end_to_end.py
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser import parse_resume                       # noqa: E402
from ranking import parse_job_description, rank_candidates   # noqa: E402
from sample_texts import (                            # noqa: E402
    JD_FILENAME, EXPECTED_TOP, EXPECTED_ORDER, EXPECTED_TAIL,
)

DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data"
)

# What a candidate who meets every stated requirement should score. The only
# gap left is experience: the JD asks for 4-6 years and this candidate has 6,
# which is in range. In-range scores 90 by design, with 90-100 reserved for
# candidates who bring MORE than the advert asked for.
PERFECT_FLOOR = 97.0


def main():
    if not os.path.isdir(DATA):
        sys.exit("sample_data/ is missing — run make_samples.py first.")

    jd_path = os.path.join(DATA, JD_FILENAME)
    with open(jd_path, encoding="utf-8") as fh:
        jd = parse_job_description(fh.read())

    candidates = []
    for filename in sorted(os.listdir(DATA)):
        if not filename.lower().endswith(".docx"):
            continue
        with open(os.path.join(DATA, filename), "rb") as fh:
            candidates.append(parse_resume(fh.read(), filename))

    print(f"parsed {len(candidates)} resumes from sample_data/\n")

    for c in candidates:
        if c.get("error"):
            print(f"  PARSE ERROR  {c['filename']}: {c['error']}")

    ranked = rank_candidates(candidates, jd)

    print(f"{'#':<3} {'candidate':<38} {'final':>7} {'skill':>7} {'exp':>7} "
          f"{'edu':>6} {'loc':>6}")
    print("-" * 80)
    for r in ranked:
        b = r["score_breakdown"]
        print(f"{r['rank']:<3} {r['filename']:<38} {r['final_score']:>7} "
              f"{b.get('skill_score', 0):>7} {b.get('exp_score', 0):>7} "
              f"{b.get('edu_score', 0):>6} {b.get('loc_score', 0):>6}")

    print()
    print("what the parser read off each resume:")
    for r in ranked:
        print(f"   {r['filename']:<38} {str(r['name'])[:18]:<19} "
              f"{r['experience_years']:>4} yrs  {r['education']['level']:<9} "
              f"{r['location']}")

    failures = []

    top = ranked[0]
    if top["filename"] != EXPECTED_TOP:
        failures.append(f"top of list is {top['filename']}, expected {EXPECTED_TOP}")
    if top["final_score"] < PERFECT_FLOOR:
        failures.append(
            f"the JD was written from {EXPECTED_TOP} yet it scores "
            f"{top['final_score']}, below the {PERFECT_FLOOR} floor"
        )

    actual_order = [r["filename"] for r in ranked]
    civil_order = [f for f in actual_order if f not in EXPECTED_TAIL]
    if civil_order != EXPECTED_ORDER:
        failures.append(f"civil candidates ranked {civil_order}, expected {EXPECTED_ORDER}")

    tail = set(actual_order[-len(EXPECTED_TAIL):])
    if tail != EXPECTED_TAIL:
        failures.append(f"bottom of the list is {tail}, expected {EXPECTED_TAIL}")

    print()
    print("=" * 80)
    if failures:
        for f in failures:
            print(f"  FAIL  {f}")
    else:
        print(f"  PASS  {EXPECTED_TOP} scores {top['final_score']} and the batch "
              f"is in the expected order.")
    print("=" * 80)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
