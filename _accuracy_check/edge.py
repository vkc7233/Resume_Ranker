"""
Edge cases for the three scoring changes. Each block prints what the engine
returns and the reasoning it should follow.
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import extract_education_requirement, extract_location   # noqa: E402
from ranking import (                                               # noqa: E402
    compute_education_score,
    compute_skill_score,
    parse_job_description,
    rank_candidates,
)
from run_check import build_candidate, load                         # noqa: E402

fails = []


def check(label, got, want):
    ok = got == want
    if not ok:
        fails.append(label)
    print(f"  [{'OK ' if ok else 'FAIL'}] {label:<52} got {got!r}")


print("=" * 74)
print("A. JD education bar — read the LOWEST level stated, not the highest")
print("=" * 74)
check("Bachelor required, Master preferred -> bachelor(3)",
      extract_education_requirement("Bachelor's degree required, Master's preferred."), 3)
check("B.E./B.Tech in Civil -> 3",
      extract_education_requirement("Qualification: B.E. / B.Tech in Civil Engineering"), 3)
check("Diploma or B.E. accepted -> diploma(2)",
      extract_education_requirement("Diploma or B.E. in Civil accepted."), 2)
check("MBA required -> 4",
      extract_education_requirement("MBA in Marketing required."), 4)
check("PhD only -> 5",
      extract_education_requirement("PhD in Structural Engineering."), 5)
check("no qualification mentioned -> 0",
      extract_education_requirement("Site engineer needed for a Pune high-rise."), 0)
check("only school-level mentioned -> 1",
      extract_education_requirement("Helper role, 12th pass acceptable."), 1)

print()
print("=" * 74)
print("B. Education score — meeting the bar is full marks, no bar keeps the")
print("   old absolute ladder so nothing regresses")
print("=" * 74)
be = {"level": "BE", "score": 3}
dip = {"level": "DIPLOMA", "score": 2}
phd = {"level": "PHD", "score": 5}
none = {"level": "Not Specified", "score": 0}
check("B.E. vs a bachelor bar  -> 100", compute_education_score(be, 3), 100.0)
check("PhD  vs a bachelor bar  -> 100 (no extra credit)", compute_education_score(phd, 3), 100.0)
check("diploma vs bachelor bar -> 66.7",
      round(compute_education_score(dip, 3), 1), 66.7)
check("none vs bachelor bar    -> 0", compute_education_score(none, 3), 0.0)
check("B.E. with no bar stated -> 60 (old ladder)", compute_education_score(be, 0), 60.0)
check("PhD  with no bar stated -> 100 (old ladder)", compute_education_score(phd, 0), 100.0)

print()
print("=" * 74)
print("C. Skill score — semantic lifts a gap, never caps a full coverage")
print("=" * 74)
check("full coverage, poor semantic -> 100", compute_skill_score(100.0, 0.30, True), 100.0)
check("full coverage, good semantic -> 100", compute_skill_score(100.0, 0.90, True), 100.0)
check("zero coverage, zero semantic -> 0", compute_skill_score(0.0, 0.0, True), 0.0)
check("40 coverage, unrelated resume (0.04) -> 41.2",
      compute_skill_score(40.0, 0.04, True), 41.2)
check("40 coverage, clearly relevant (0.60) -> 58.0",
      compute_skill_score(40.0, 0.60, True), 58.0)
check("JD with no detectable skills -> pure semantic",
      compute_skill_score(0.0, 0.42, False), 42.0)
print("  monotonic in semantic at fixed coverage:")
prev = -1.0
mono = True
for s in [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
    v = compute_skill_score(50.0, s, True)
    if v < prev:
        mono = False
    prev = v
    print(f"      semantic {s:<4} -> {v}")
check("higher semantic never scores lower", mono, True)

print()
print("=" * 74)
print("D. Location — earliest in the text wins, word boundaries respected")
print("=" * 74)
check("header city beats a later previous posting",
      extract_location("Rahul\nPune, Maharashtra\nEarlier: Mumbai, Delhi"), "Pune")
check("'remotely' does not match the keyword 'remote'",
      extract_location("Working remotely from Jaipur since 2021."), "Jaipur")
check("a surname containing a city does not match",
      extract_location("Priya Agrawal\nBased in Chennai"), "Chennai")
check("Navi Mumbai beats the Mumbai inside it",
      extract_location("Site office: Navi Mumbai"), "Navi Mumbai")
check("genuinely remote resume still reads as remote",
      extract_location("Remote contractor, open to travel."), "Remote")
check("nothing recognisable -> None",
      extract_location("Engineer with strong fundamentals."), None)

print()
print("=" * 74)
print("E. Whole-pipeline fallbacks — a bare JD must still rank, not crash")
print("=" * 74)
resume = load("resume_rahul_deshmukh.txt")
for label, jd_text in [
    ("no skills, no years, no city", "We need a hardworking person for our team."),
    ("skills only", "AutoCAD and STAAD Pro required."),
    ("years only", "We need someone with 5+ years."),
    ("empty string", ""),
]:
    jd = parse_job_description(jd_text)
    r = rank_candidates([build_candidate(resume, "r.txt")], jd)[0]
    print(f"  [OK ] {label:<32} final {r['final_score']:<7} "
          f"edu_bar {jd['education_min']}  loc {jd['location']}")

print()
print("=" * 74)
print(f"{'ALL CHECKS PASSED' if not fails else 'FAILURES: ' + ', '.join(fails)}")
print("=" * 74)
sys.exit(1 if fails else 0)
