"""
Targeted probes for the specific accuracy defects suspected from reading the
scoring code. Each probe prints what the engine does now and what it ought to do.
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import extract_location, extract_education          # noqa: E402
from ranking import compute_education_score, compute_skill_score  # noqa: E402

print("=" * 70)
print("PROBE 1 — extract_location returns the first match in LIST order,")
print("          not the first match in the TEXT.")
print("=" * 70)
cases = [
    ("Rahul Deshmukh\nPune, Maharashtra\nPreviously worked in Mumbai and Delhi.", "Pune"),
    ("Anita Rao\nHyderabad, Telangana\nOpen to Bangalore for the right role.", "Hyderabad"),
    ("Site Engineer\nLocation: Noida\nOur Delhi head office supports the site.", "Noida"),
]
for text, expected in cases:
    got = extract_location(text)
    flag = "OK " if got == expected else "BUG"
    print(f"  [{flag}] expected {expected:<12} got {got}")

print()
print("=" * 70)
print("PROBE 2 — location matching has no word boundary, so a city name")
print("          hidden inside another word is a false positive.")
print("=" * 70)
cases2 = [
    ("Priya Agrawal\nSoftware Engineer\nBased in Chennai.", "Chennai"),
    ("Candidate is working remotely from Jaipur.", "Jaipur"),
]
for text, expected in cases2:
    got = extract_location(text)
    flag = "OK " if got == expected else "BUG"
    print(f"  [{flag}] expected {expected:<12} got {got}")
    print(f"        text: {text.splitlines()[0][:60]}")

print()
print("=" * 70)
print("PROBE 3 — education is scored on an absolute ladder, never against")
print("          what the JD actually asked for.")
print("=" * 70)
jd = "Site Engineer (Civil)\nQualification: B.E. / B.Tech in Civil Engineering"
print(f"  JD asks for      : {extract_education(jd)}")
be = extract_education("B.E. Civil Engineering, Pune University, CGPA 7.8")
mba = extract_education("MBA Marketing, Symbiosis. Also completed B.Com.")
print(f"  B.E. Civil       : {be}  -> education score {compute_education_score(be)}")
print(f"  MBA Marketing    : {mba}  -> education score {compute_education_score(mba)}")
print("  The B.E. exactly meets the stated requirement yet scores 60/100,")
print("  while an MBA — irrelevant to a civil site role — scores 80/100.")

print()
print("=" * 70)
print("PROBE 4 — the semantic term can only ever drag a perfect skills")
print("          coverage downward; it can never lift a weak one to par.")
print("=" * 70)
for cov, sem in [(100.0, 0.60), (100.0, 0.30), (100.0, 0.95), (40.0, 0.70)]:
    print(f"  coverage {cov:<6} semantic {sem:<5} -> skill score "
          f"{compute_skill_score(cov, sem, True)}")
print("  A candidate holding every single required skill is capped at ~94")
print("  purely because a resume never reads like a job advert.")
