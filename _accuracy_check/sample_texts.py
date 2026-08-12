"""
Source text for the sample resumes and the job description.

Kept in one place so the .docx generator and the accuracy checks cannot drift
apart. The batch is deliberately built so the correct ranking order is not a
matter of opinion: one civil engineer who matches the advert closely, one
senior civil person in the wrong city, one junior civil engineer, one fresher,
and two people from other departments entirely.
"""

JD_FILENAME = "jd_site_engineer_civil.txt"

JOB_DESCRIPTION = """Site Engineer (Civil) - RCC Structures - Pune

About the role
We are a Pune-based construction contractor building residential and commercial
RCC structures. We are looking for a site engineer to take day-to-day ownership
of execution on a high-rise project: supervising the work, holding quality, and
keeping the quantities and bills straight.

Location: Pune, Maharashtra
Experience required: 4 to 6 years
Qualification: B.E. / B.Tech in Civil Engineering

What you will do
- Site supervision of RCC works - layout, shuttering, reinforcement and pours.
- QA/QC on concrete and steel: slump tests, cube sampling, cover checks and
  reinforcement inspection before every pour.
- Quantity surveying - prepare BOQ, rate analysis and estimation for structural
  and finishing packages.
- Running-account billing and certification of subcontractor bills.
- Maintain the project planning schedule in MS Project and report progress
  against the baseline programme.
- Set out the works with a total station and carry out site surveying.

What we are looking for
- Strong AutoCAD drafting and detailing for structural drawings.
- Working knowledge of STAAD Pro for structural analysis of slabs and beams.
- Sound concrete technology fundamentals and RCC detailing practice.
- Site engineering experience on a live high-rise site.
- MS Excel for reconciliation of cement and steel.
"""

# The candidate the JD was written from. Should rank first, and should score
# close to full marks — every stated requirement is met.
RESUMES = {
    "01_rahul_deshmukh_site_engineer.docx": """Rahul Deshmukh
Site Engineer (Civil)
Pune, Maharashtra | rahul.deshmukh@email.com | +91-9822104477

PROFESSIONAL SUMMARY
Site engineer with 6 years of experience delivering residential and commercial
RCC structures for a mid-sized construction contractor. Day-to-day ownership of
site supervision, quality checks on concrete and steel, quantity take-offs and
running-account billing. Comfortable reading and issuing drawings, and holding
subcontractors to programme.

WORK EXPERIENCE

Site Engineer - Sanghvi Constructions Pvt. Ltd., Pune
Jul 2022 - Present
- Supervised execution of two G+12 residential towers, 1.8 lakh sq.ft. built-up.
- Prepared BOQ and rate analysis for structural and finishing packages.
- Ran QA/QC checks on RCC pours: slump, cube sampling, cover and reinforcement
  inspection before every concrete pour.
- Maintained the project planning schedule in MS Project and reported weekly
  progress against the baseline programme.
- Handled running-account billing and subcontractor bill certification.

Junior Site Engineer - Kalpataru Infra Projects, Pune
Aug 2019 - Jun 2022
- Executed layout and level marking using a total station; carried out land
  surveying for the site grid.
- Produced structural drawings and detailing in AutoCAD.
- Performed structural analysis of slabs and beams in STAAD Pro under senior
  guidance.
- Assisted the quantity surveying team with estimation and reconciliation of
  cement and steel.

EDUCATION
B.E. Civil Engineering - Savitribai Phule Pune University, 2015 - 2019
CGPA 7.8

TECHNICAL SKILLS
AutoCAD, STAAD Pro, MS Project, Total Station, Surveying, RCC,
Concrete Technology, Quantity Surveying, BOQ, Estimation, Rate Analysis,
Billing, Site Supervision, Site Engineering, Project Planning, QA/QC,
Structural Analysis, MS Excel

CERTIFICATIONS
- Concrete Mix Design workshop, Indian Concrete Institute (2021)
- Safety induction / HSE orientation for high-rise sites (2023)
""",

    "02_sneha_kulkarni_civil_junior.docx": """Sneha Kulkarni
Civil Engineer
Pune, Maharashtra | sneha.kulkarni@email.com | +91-9011223344

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

    "03_vikram_singh_senior_mumbai.docx": """Vikram Singh
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

    "04_deepak_yadav_fresher.docx": """Deepak Yadav
Junior Civil Supervisor
Pune, Maharashtra | deepak.yadav@email.com | +91-9765512300

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

    "05_arjun_mehta_marketing.docx": """Arjun Mehta
Digital Marketing Manager
Pune, Maharashtra | arjun.mehta@email.com | +91-9922334455

SUMMARY
Digital marketing manager with 7 years of experience running paid and organic
acquisition for real estate and construction brands.

EXPERIENCE
Digital Marketing Manager - Brick and Byte Media, Pune
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

    "06_ravi_nair_sales.docx": """Ravi Nair
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
MBA Sales and Marketing - Pune University, 2014 - 2016

SKILLS
Sales, B2C Sales, Real Estate Sales, Channel Partner, Lead Generation,
Negotiation, CRM, Team Management, Communication, MS Excel
""",
}

# The candidate the JD was written from — must come first.
EXPECTED_TOP = "01_rahul_deshmukh_site_engineer.docx"

# The order the batch must come back in. Anything else is a ranking failure.
EXPECTED_ORDER = [
    "01_rahul_deshmukh_site_engineer.docx",   # writes the advert back at us
    "03_vikram_singh_senior_mumbai.docx",     # strong civil, wrong city
    "02_sneha_kulkarni_civil_junior.docx",    # real civil, junior
    "04_deepak_yadav_fresher.docx",           # civil, but a fresher on a diploma
]
# The two below are from other departments and must land beneath every civil
# candidate. Their order relative to each other is not meaningful.
EXPECTED_TAIL = {
    "05_arjun_mehta_marketing.docx",
    "06_ravi_nair_sales.docx",
}
