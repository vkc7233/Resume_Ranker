"""
skills.py - Skill extraction and matching module

Handles a predefined skill database, alias canonicalization, keyword-based
extraction, and matching between a resume and a job description.

The key accuracy idea here is *canonicalization*: many skills are written in
several ways ("react", "reactjs", "react.js"; "postgres" vs "postgresql";
"k8s" vs "kubernetes"; "ml" vs "machine learning"). Every variant is mapped to
a single canonical name so a JD written one way still matches a resume written
another way.
"""

import re

# ─────────────────────────────────────────────
# PREDEFINED SKILL DATABASE (canonical names)
# ─────────────────────────────────────────────

SKILL_CATEGORIES = {
    "Programming Languages": [
        "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "C", "Go",
        "Ruby", "PHP", "Swift", "Kotlin", "Scala", "Rust", "R", "MATLAB",
        "Perl", "Bash", "PowerShell", "VBA", "Dart", "Lua", "Haskell",
    ],
    "Web Frameworks": [
        "Django", "Flask", "FastAPI", "Spring Boot", "React", "Angular",
        "Vue.js", "Node.js", "Express", "Next.js", "Nuxt", "Svelte",
        "Laravel", "Ruby on Rails", "ASP.NET", "Blazor", "Gatsby", "Fastify",
    ],
    "Databases": [
        "SQL", "MySQL", "PostgreSQL", "SQLite", "Oracle", "SQL Server",
        "MongoDB", "Redis", "Cassandra", "DynamoDB", "Firebase",
        "Elasticsearch", "Neo4j", "CouchDB", "MariaDB", "Snowflake",
        "BigQuery", "Hive", "HBase",
    ],
    "Cloud & DevOps": [
        "AWS", "Azure", "GCP", "Heroku", "Docker", "Kubernetes", "Terraform",
        "Ansible", "Jenkins", "CI/CD", "GitHub Actions", "GitLab CI",
        "CircleCI", "Travis CI", "Helm", "Prometheus", "Grafana", "Nginx",
        "Apache", "Linux", "Unix", "DevOps",
    ],
    "Data Science & ML": [
        "Machine Learning", "Deep Learning", "Artificial Intelligence",
        "Data Analysis", "Data Science", "NLP", "Computer Vision",
        "TensorFlow", "PyTorch", "Keras", "scikit-learn", "Pandas", "NumPy",
        "Matplotlib", "Seaborn", "Plotly", "Tableau", "Power BI", "Spark",
        "Hadoop", "PySpark", "Data Mining", "Statistics", "A/B Testing",
        "Feature Engineering", "Model Deployment", "MLOps",
    ],
    "Mobile": [
        "Android", "iOS", "React Native", "Flutter", "Xamarin",
        "Objective-C", "Cordova", "Ionic",
    ],
    "Tools & Others": [
        "Git", "GitHub", "GitLab", "Bitbucket", "Jira", "Confluence",
        "Trello", "Agile", "Scrum", "Kanban", "REST API", "GraphQL", "SOAP",
        "Microservices", "JSON", "XML", "HTML", "CSS", "Sass", "Webpack",
        "Vite", "Babel", "Jest", "Selenium", "Cypress", "Postman", "Figma",
        "Networking", "Security", "OAuth",
    ],
    "IT Support & Security Systems": [
        "CCTV", "Surveillance Systems", "DVR", "NVR", "IP Camera",
        "Access Control", "Biometric Systems", "Desktop Support",
        "Hardware Troubleshooting", "IT Support", "Technical Support",
        "Help Desk", "System Administration", "Active Directory", "Firewall",
        "Server Administration", "Structured Cabling", "EPABX",
        "Remote Support", "VPN", "Antivirus", "Printer Support",
    ],
    "Civil & Construction": [
        "AutoCAD", "Civil 3D", "Revit", "STAAD Pro", "ETABS", "SAP2000",
        "Primavera", "MS Project", "BIM", "SketchUp", "Tekla", "ArcGIS",
        "3ds Max", "Structural Analysis", "Structural Design", "RCC",
        "Steel Structures", "Geotechnical Engineering", "Soil Mechanics",
        "Foundation Engineering", "Surveying", "Total Station",
        "Quantity Surveying", "Estimation", "BOQ", "Rate Analysis", "Billing",
        "Construction Management", "Project Planning", "Site Supervision",
        "Site Engineering", "Concrete Technology", "Highway Engineering",
        "Transportation Engineering", "Pavement Design", "Hydraulics",
        "Water Resources", "Environmental Engineering", "Earthquake Engineering",
        "Contracts Management", "Tendering", "Procurement", "MEP", "HVAC",
        "Plumbing", "QA/QC", "HSE", "LEED", "Green Building", "FIDIC",
    ],
    "Sales & Business Development": [
        "Sales", "B2B Sales", "B2C Sales", "Field Sales", "Inside Sales",
        "Retail Sales", "Channel Sales", "Direct Sales", "Sales Management",
        "Business Development", "Lead Generation", "Cold Calling",
        "Negotiation", "Client Relationship", "Account Management",
        "Key Account Management", "CRM", "Salesforce", "Sales Forecasting",
        "Upselling", "Cross-selling", "Telesales", "Telemarketing",
        "Pipeline Management", "Revenue Growth", "Channel Partner",
        "Distributor Management", "Real Estate Sales",
    ],
    "Marketing & Digital Marketing": [
        "Marketing", "Digital Marketing", "SEO", "SEM", "PPC",
        "Content Marketing", "Social Media Marketing", "Email Marketing",
        "Google Ads", "Facebook Ads", "Google Analytics", "Google Tag Manager",
        "Affiliate Marketing", "Influencer Marketing", "Brand Management",
        "Market Research", "Marketing Strategy", "Campaign Management",
        "Marketing Automation", "Conversion Rate Optimization",
        "Content Writing", "Copywriting", "WordPress", "HubSpot", "Mailchimp",
        "Hootsuite", "Marketing Analytics",
    ],
    "Design & Creative": [
        "Graphic Design", "Adobe Photoshop", "Adobe Illustrator",
        "Adobe Premiere", "Adobe InDesign", "After Effects", "Canva",
        "Video Editing", "UI/UX Design", "CorelDRAW",
    ],
    "Business & Office": [
        "MS Office", "MS Excel", "MS Word", "MS PowerPoint", "Outlook",
        "Tally", "SAP", "ERP", "Accounting", "Bookkeeping", "Data Entry",
        "Communication", "Leadership", "Teamwork", "Team Management",
        "Problem Solving", "Time Management", "Customer Service",
        "Public Speaking", "Presentation",
    ],
}

# Flat canonical list (deduplicated, order preserved).
ALL_SKILLS = list(dict.fromkeys(s for group in SKILL_CATEGORIES.values() for s in group))

# ─────────────────────────────────────────────
# ALIAS MAP  (lowercase variant  →  canonical name)
# ─────────────────────────────────────────────
# Every canonical skill maps to itself; variants map to the canonical form.

_RAW_ALIASES = {
    # Languages
    "golang": "Go",
    "js": "JavaScript",
    "ecmascript": "JavaScript",
    "ts": "TypeScript",
    "c plus plus": "C++",
    "cpp": "C++",
    "c sharp": "C#",
    "csharp": "C#",
    "shell": "Bash",
    "shell scripting": "Bash",
    # Web frameworks
    "reactjs": "React",
    "react.js": "React",
    "react js": "React",
    "angularjs": "Angular",
    "angular.js": "Angular",
    "vue": "Vue.js",
    "vuejs": "Vue.js",
    "node": "Node.js",
    "nodejs": "Node.js",
    "expressjs": "Express",
    "express.js": "Express",
    "nextjs": "Next.js",
    "springboot": "Spring Boot",
    "spring": "Spring Boot",
    "rails": "Ruby on Rails",
    "asp.net core": "ASP.NET",
    "aspnet": "ASP.NET",
    # Databases
    "postgres": "PostgreSQL",
    "psql": "PostgreSQL",
    "mssql": "SQL Server",
    "mongo": "MongoDB",
    "mongoose": "MongoDB",
    # Cloud & DevOps
    "amazon web services": "AWS",
    "microsoft azure": "Azure",
    "google cloud": "GCP",
    "google cloud platform": "GCP",
    "k8s": "Kubernetes",
    "ci cd": "CI/CD",
    "cicd": "CI/CD",
    "continuous integration": "CI/CD",
    "continuous deployment": "CI/CD",
    "gitlab ci/cd": "GitLab CI",
    # Data science & ML
    "ml": "Machine Learning",
    "dl": "Deep Learning",
    "ai": "Artificial Intelligence",
    "natural language processing": "NLP",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "tensor flow": "TensorFlow",
    "powerbi": "Power BI",
    "ab testing": "A/B Testing",
    "ml ops": "MLOps",
    # Tools & others
    "restful": "REST API",
    "rest": "REST API",
    "rest apis": "REST API",
    "graph ql": "GraphQL",
    "micro services": "Microservices",
    "css3": "CSS",
    "html5": "HTML",
    "scss": "Sass",
    # IT support & security systems
    "cc tv": "CCTV",
    "cctv camera": "CCTV",
    "cctv cameras": "CCTV",
    "cctv installation": "CCTV",
    "closed circuit television": "CCTV",
    "surveillance": "Surveillance Systems",
    "surveillance system": "Surveillance Systems",
    "video surveillance": "Surveillance Systems",
    "ip cameras": "IP Camera",
    "network video recorder": "NVR",
    "digital video recorder": "DVR",
    "access control system": "Access Control",
    "access control systems": "Access Control",
    "biometric": "Biometric Systems",
    "biometrics": "Biometric Systems",
    "biometric attendance": "Biometric Systems",
    "desktop support engineer": "Desktop Support",
    "desktop engineer": "Desktop Support",
    "it support engineer": "IT Support",
    "tech support": "Technical Support",
    "helpdesk": "Help Desk",
    "system administrator": "System Administration",
    "sysadmin": "System Administration",
    "fortigate": "Firewall",
    "sonicwall": "Firewall",
    "server administrator": "Server Administration",
    "hardware support": "Hardware Troubleshooting",
    "hardware and networking": "Hardware Troubleshooting",
    "anti virus": "Antivirus",
    # Civil & construction
    "auto cad": "AutoCAD",
    "autocad civil 3d": "Civil 3D",
    "civil3d": "Civil 3D",
    "staad": "STAAD Pro",
    "staad pro": "STAAD Pro",
    "staad.pro": "STAAD Pro",
    "msproject": "MS Project",
    "microsoft project": "MS Project",
    "building information modeling": "BIM",
    "building information modelling": "BIM",
    "sketch up": "SketchUp",
    "reinforced concrete": "RCC",
    "reinforced cement concrete": "RCC",
    "steel structure": "Steel Structures",
    "geotechnical": "Geotechnical Engineering",
    "foundation design": "Foundation Engineering",
    "land surveying": "Surveying",
    "quantity survey": "Quantity Surveying",
    "bill of quantities": "BOQ",
    "estimation and costing": "Estimation",
    "costing": "Estimation",
    "construction planning": "Project Planning",
    "site supervisor": "Site Supervision",
    "site engineer": "Site Engineering",
    "water resources engineering": "Water Resources",
    "qaqc": "QA/QC",
    "qa qc": "QA/QC",
    "quality assurance": "QA/QC",
    "quality control": "QA/QC",
    "health and safety": "HSE",
    "health & safety": "HSE",
    "mechanical electrical plumbing": "MEP",
    "leed certified": "LEED",
    # Sales & business development
    "b2b": "B2B Sales",
    "b2c": "B2C Sales",
    "business development executive": "Business Development",
    "biz dev": "Business Development",
    "lead gen": "Lead Generation",
    "cold call": "Cold Calling",
    "key account": "Key Account Management",
    "account manager": "Account Management",
    "customer relationship management": "CRM",
    "cross selling": "Cross-selling",
    "up selling": "Upselling",
    "tele sales": "Telesales",
    "tele marketing": "Telemarketing",
    "sales forecast": "Sales Forecasting",
    "real estate": "Real Estate Sales",
    # Marketing & digital marketing
    "search engine optimization": "SEO",
    "search engine marketing": "SEM",
    "pay per click": "PPC",
    "smm": "Social Media Marketing",
    "social media": "Social Media Marketing",
    "e-mail marketing": "Email Marketing",
    "google adwords": "Google Ads",
    "adwords": "Google Ads",
    "fb ads": "Facebook Ads",
    "meta ads": "Facebook Ads",
    "facebook advertising": "Facebook Ads",
    "gtm": "Google Tag Manager",
    "cro": "Conversion Rate Optimization",
    "content writer": "Content Writing",
    "copy writing": "Copywriting",
    "word press": "WordPress",
    "hub spot": "HubSpot",
    "mail chimp": "Mailchimp",
    "branding": "Brand Management",
    # Design & creative
    "photoshop": "Adobe Photoshop",
    "illustrator": "Adobe Illustrator",
    "premiere pro": "Adobe Premiere",
    "adobe premiere pro": "Adobe Premiere",
    "indesign": "Adobe InDesign",
    "adobe after effects": "After Effects",
    "corel draw": "CorelDRAW",
    "ui/ux": "UI/UX Design",
    "ui ux": "UI/UX Design",
    "ux design": "UI/UX Design",
    # Business & office
    "microsoft office": "MS Office",
    "microsoft excel": "MS Excel",
    "excel": "MS Excel",
    "advanced excel": "MS Excel",
    "microsoft word": "MS Word",
    "microsoft powerpoint": "MS PowerPoint",
    "power point": "MS PowerPoint",
    "powerpoint": "MS PowerPoint",
    "tally erp": "Tally",
    "sap erp": "SAP",
    "team work": "Teamwork",
    "communication skills": "Communication",
    "presentation skills": "Presentation",
    "leadership skills": "Leadership",
}


def _build_alias_map() -> dict:
    """Map every lowercase variant (including the canonical itself) → canonical."""
    alias_map = {}
    for canonical in ALL_SKILLS:
        alias_map[canonical.lower()] = canonical
    for variant, canonical in _RAW_ALIASES.items():
        alias_map[variant.lower()] = canonical
    return alias_map


ALIAS_MAP = _build_alias_map()

# Search terms sorted longest-first so multi-word skills ("machine learning",
# "spring boot") are detected before their shorter substrings.
_SEARCH_TERMS = sorted(ALIAS_MAP.keys(), key=len, reverse=True)

# Pre-compiled word-boundary patterns for every search term. Tech names contain
# symbols (+ # / .), so a plain \b doesn't work. We treat +, #, / and
# alphanumerics as "word" characters always, and a "." as part of a skill only
# when it sits between two alphanumerics (so "Node.js" stays whole, but a
# sentence-ending "AWS." or "services." still matches the skill before the dot).
_LEFT = r'(?<![a-z0-9+#/])(?<![a-z0-9]\.)'
_RIGHT = r'(?![a-z0-9+#/])(?!\.[a-z0-9])'
_COMPILED = {
    term: re.compile(_LEFT + re.escape(term) + _RIGHT)
    for term in _SEARCH_TERMS
}


def canonicalize(skill: str) -> str:
    """Return the canonical name for a single skill string (best effort)."""
    return ALIAS_MAP.get(skill.lower().strip(), skill.strip())


def extract_skills(text: str) -> list[str]:
    """
    Extract canonical skills from raw text using alias-aware keyword matching.

    Returns a deduplicated list of canonical skill names, in the order their
    category appears in the database (stable across resumes for clean display).
    """
    if not text:
        return []

    text_lower = text.lower()
    found = set()

    for term in _SEARCH_TERMS:
        # Fast substring pre-check (C-level) before the costlier boundary regex.
        # A term must appear literally to match, so this skips the regex for the
        # vast majority of non-present terms — a big win across 100 resumes.
        if term in text_lower and _COMPILED[term].search(text_lower):
            found.add(ALIAS_MAP[term])

    # Return in canonical-database order for stable, readable output.
    return [s for s in ALL_SKILLS if s in found]


# How strongly a single JD skill can dominate the emphasis weighting. A skill the
# JD repeats many times (e.g. "CCTV" in a surveillance role) carries more weight,
# but we cap it so one term can't completely swamp the rest.
_EMPHASIS_CAP = 6
# A skill named in the JD's first line (the job title / headline) is what defines
# the role, so it gets a strong boost even if mentioned only once.
_TITLE_CHARS = 120
_TITLE_BOOST = 3


def extract_skill_weights(text: str) -> dict:
    """
    Return {canonical_skill: emphasis_weight} for a job description.

    The weight reflects how strongly the JD emphasises each skill:
      • +1 for every mention (or alias mention) in the body, and
      • +`_TITLE_BOOST` if the skill appears in the JD's first line / title,
    capped at `_EMPHASIS_CAP`.

    This lets the skill score reward matching the *most important* requirements
    rather than treating every JD skill equally — so a role whose title is
    "CCTV Technician" (or that repeats "CCTV" several times) isn't outranked by a
    resume that merely matches five generic keywords once each.
    """
    if not text:
        return {}

    text_lower = text.lower()
    # The headline is the JD's first line (the job title), capped for safety.
    head = text_lower.split("\n", 1)[0][:_TITLE_CHARS]
    weights: dict[str, int] = {}
    for term in _SEARCH_TERMS:
        # Fast substring pre-check before the boundary regex (see extract_skills).
        if term not in text_lower:
            continue
        hits = len(_COMPILED[term].findall(text_lower))
        if not hits:
            continue
        canon = ALIAS_MAP[term]
        weights[canon] = weights.get(canon, 0) + hits
        # Headline emphasis: a skill in the job title defines the role.
        if term in head and _COMPILED[term].search(head):
            weights[canon] += _TITLE_BOOST

    return {skill: min(count, _EMPHASIS_CAP) for skill, count in weights.items()}


def match_skills(resume_skills: list[str], jd_skills: list[str],
                 jd_weights: dict | None = None) -> dict:
    """
    Compare resume skills against job-description skills on a canonical basis.

    Returns dict with: matched, missing, score (0-100).

    When `jd_weights` is supplied (canonical skill → emphasis weight), the score
    is *emphasis-weighted coverage*: the fraction of the JD's total skill
    emphasis that the resume actually covers. Matching a heavily-stressed
    requirement counts for much more than matching an incidental one, and
    missing the JD's central skill is heavily penalised. Without weights it
    falls back to plain coverage (fraction of JD skills present).
    """
    if not jd_skills:
        return {"matched": [], "missing": [], "score": 0.0}

    resume_canon = {canonicalize(s).lower() for s in resume_skills}

    matched, missing = [], []
    matched_weight = 0.0
    total_weight = 0.0
    seen = set()
    for s in jd_skills:
        canon = canonicalize(s)
        key = canon.lower()
        if key in seen:
            continue
        seen.add(key)

        weight = float(jd_weights.get(canon, 1)) if jd_weights else 1.0
        total_weight += weight
        if key in resume_canon:
            matched.append(canon)
            matched_weight += weight
        else:
            missing.append(canon)

    score = (matched_weight / total_weight) * 100 if total_weight else 0.0

    # Surface the most-emphasised skills first so the UI shows what actually
    # matters (the central requirement, not an incidental keyword).
    if jd_weights:
        matched.sort(key=lambda c: jd_weights.get(c, 1), reverse=True)
        missing.sort(key=lambda c: jd_weights.get(c, 1), reverse=True)

    return {"matched": matched, "missing": missing, "score": round(score, 2)}
