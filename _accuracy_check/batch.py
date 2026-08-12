"""
Rank a mixed batch against the civil site-engineer JD.

Absolute scores matter less than ORDER. This batch is built so the correct
order is not in doubt — a hands-on Pune civil engineer must beat a Mumbai
project manager, who must beat a fresher, who must beat the sales and
marketing people. Run before and after any scoring change.
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ranking import parse_job_description, rank_candidates   # noqa: E402
from run_check import build_candidate, load                  # noqa: E402

RESUMES = {
    "01_rahul_civil_perfect.txt": None,   # loaded from the main resume file

    "02_sneha_civil_junior.txt": """Sneha Kulkarni
Civil Engineer
Pune, Maharashtra | sneha.k@email.com | +91-9011223344

SUMMARY
Civil engineer with 3 years of experience on residential RCC projects.

EXPERIENCE
Civil Engineer - Pushpak Builders, Pune
Sep 2022 - Present
- Site supervision of RCC slab and column work.
- Prepared BOQ and assisted with estimation for finishing packages.
- AutoCAD drafting of structural layouts.

EDUCATION
B.E. Civil Engineering - Pune University, 2018 - 2022

SKILLS
AutoCAD, RCC, BOQ, Estimation, Site Supervision, MS Excel
""",

    "03_vikram_civil_senior_mumbai.txt": """Vikram Singh
Senior Project Manager - Civil
Mumbai, Maharashtra | vikram.singh@email.com | +91-9820011223

SUMMARY
Construction professional with 12 years of experience delivering high-rise
RCC and steel structures. Strong on planning, contracts and QA/QC.

EXPERIENCE
Senior Project Manager - Shapoorji Infra, Mumbai
Mar 2018 - Present
- Managed project planning and programme for three towers.
- Contracts management, tendering and procurement for structural packages.
- QA/QC governance and HSE compliance across sites.

Project Engineer - L&T Construction, Mumbai
Jun 2014 - Feb 2018
- Quantity surveying, BOQ, rate analysis and billing.
- Structural analysis coordination using STAAD Pro and AutoCAD.

EDUCATION
M.Tech Structural Engineering - VJTI Mumbai, 2012 - 2014
B.E. Civil Engineering - Mumbai University, 2008 - 2012

SKILLS
Project Planning, MS Project, Primavera, Contracts Management, Tendering,
Procurement, QA/QC, HSE, BOQ, Rate Analysis, Billing, Quantity Surveying,
STAAD Pro, AutoCAD, RCC, Steel Structures, Site Supervision
""",

    "04_deepak_fresher_diploma.txt": """Deepak Yadav
Junior Civil Supervisor
Pune, Maharashtra | deepak.y@email.com | +91-9765512300

SUMMARY
Recent diploma holder seeking a site role. Familiar with AutoCAD and basic
site supervision from an internship.

INTERNSHIP
Site Trainee - Vastushodh Projects, Pune
Jan 2025 - Apr 2025
- Shadowed site supervision of RCC works and recorded pour details.

EDUCATION
Diploma in Civil Engineering - Government Polytechnic Pune, 2022 - 2025

SKILLS
AutoCAD, RCC, Site Supervision, MS Excel
""",

    "05_arjun_marketing_mba.txt": """Arjun Mehta
Digital Marketing Manager
Pune, Maharashtra | arjun.mehta@email.com | +91-9922334455

SUMMARY
Digital marketing manager with 7 years of experience running paid and organic
acquisition for real estate and construction brands.

EXPERIENCE
Digital Marketing Manager - Brick&Byte Media, Pune
Apr 2019 - Present
- Ran SEO, Google Ads and Facebook Ads campaigns for property developers.
- Managed content marketing and marketing analytics reporting.

EDUCATION
MBA Marketing - Symbiosis Pune, 2016 - 2018
B.Com - Pune University, 2013 - 2016

SKILLS
SEO, SEM, Google Ads, Facebook Ads, Google Analytics, Content Marketing,
Social Media Marketing, Marketing Strategy, MS Excel, Communication
""",

    "06_ravi_sales_mba.txt": """Ravi Nair
Sales Manager - Real Estate
Pune, Maharashtra | ravi.nair@email.com | +91-9833445566

SUMMARY
Sales manager with 9 years of experience in real estate sales and channel
partner management.

EXPERIENCE
Sales Manager - Kohinoor Realty, Pune
Jan 2017 - Present
- B2C sales of residential inventory; managed a team of six executives.
- Channel partner and distributor management, lead generation, negotiation.

EDUCATION
MBA Sales & Marketing - Pune University, 2014 - 2016

SKILLS
Sales, B2C Sales, Real Estate Sales, Channel Partner, Lead Generation,
Negotiation, CRM, Team Management, Communication, MS Excel
""",
}

EXPECTED_TOP = "01_rahul_civil_perfect.txt"


def main():
    jd = parse_job_description(load("jd_site_engineer_civil.txt"))

    candidates = []
    for fname, text in RESUMES.items():
        if text is None:
            text = load("resume_rahul_deshmukh.txt")
        candidates.append(build_candidate(text, fname))

    ranked = rank_candidates(candidates, jd)

    print(f"{'#':<3} {'candidate':<34} {'final':>7} {'skill':>7} "
          f"{'cov':>6} {'sem':>6} {'exp':>7} {'edu':>5} {'loc':>5}")
    print("-" * 84)
    for r in ranked:
        b = r["score_breakdown"]
        print(f"{r['rank']:<3} {r['filename']:<34} {r['final_score']:>7} "
              f"{b.get('skill_score', 0):>7} {b.get('coverage', 0):>6} "
              f"{b.get('semantic', 0):>6} {b.get('exp_score', 0):>7} "
              f"{b.get('edu_score', 0):>5} {b.get('loc_score', 0):>5}")

    print()
    top = ranked[0]["filename"]
    print(f"top of list : {top}")
    print(f"expected    : {EXPECTED_TOP}")
    print("ORDER OK" if top == EXPECTED_TOP else "ORDER WRONG")

    print()
    print("education column, sanity check:")
    for r in ranked:
        print(f"   {r['filename']:<34} {r['education']['level']:<14} "
              f"edu {r['score_breakdown'].get('edu_score', 0)}")


if __name__ == "__main__":
    main()
