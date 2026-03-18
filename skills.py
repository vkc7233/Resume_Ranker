"""
skills.py - Skill extraction and matching module
Handles predefined skill lists and keyword-based skill matching
"""

# ─────────────────────────────────────────────
# PREDEFINED SKILL DATABASE
# ─────────────────────────────────────────────

SKILL_CATEGORIES = {
    "Programming Languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "c", "go", "golang",
        "ruby", "php", "swift", "kotlin", "scala", "rust", "r", "matlab", "perl",
        "bash", "shell", "powershell", "vba", "dart", "lua", "haskell"
    ],
    "Web Frameworks": [
        "django", "flask", "fastapi", "spring", "springboot", "spring boot",
        "react", "reactjs", "react.js", "angular", "angularjs", "vue", "vuejs", "vue.js",
        "node", "nodejs", "node.js", "express", "expressjs", "nextjs", "next.js",
        "nuxt", "svelte", "laravel", "rails", "ruby on rails", "asp.net", "blazor",
        "gatsby", "fastify", "hapi"
    ],
    "Databases": [
        "sql", "mysql", "postgresql", "postgres", "sqlite", "oracle", "mssql",
        "sql server", "mongodb", "mongoose", "redis", "cassandra", "dynamodb",
        "firebase", "elasticsearch", "neo4j", "couchdb", "mariadb", "snowflake",
        "bigquery", "hive", "hbase"
    ],
    "Cloud & DevOps": [
        "aws", "amazon web services", "azure", "gcp", "google cloud", "heroku",
        "docker", "kubernetes", "k8s", "terraform", "ansible", "jenkins", "ci/cd",
        "github actions", "gitlab ci", "circleci", "travis ci", "helm", "prometheus",
        "grafana", "nginx", "apache", "linux", "unix", "devops"
    ],
    "Data Science & ML": [
        "machine learning", "ml", "deep learning", "dl", "ai", "artificial intelligence",
        "data analysis", "data science", "nlp", "natural language processing",
        "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
        "pandas", "numpy", "matplotlib", "seaborn", "plotly", "tableau", "power bi",
        "spark", "hadoop", "pyspark", "data mining", "statistics", "a/b testing",
        "feature engineering", "model deployment", "mlops"
    ],
    "Mobile": [
        "android", "ios", "react native", "flutter", "xamarin", "swift", "objective-c",
        "kotlin", "cordova", "ionic"
    ],
    "Tools & Others": [
        "git", "github", "gitlab", "bitbucket", "jira", "confluence", "trello",
        "agile", "scrum", "kanban", "rest", "restful", "rest api", "graphql",
        "soap", "microservices", "api", "json", "xml", "html", "css", "sass",
        "webpack", "vite", "babel", "jest", "selenium", "cypress", "postman",
        "figma", "photoshop", "linux", "networking", "security", "oauth"
    ]
}

# Flat skill list (all categories merged) — used for fast lookup
ALL_SKILLS = []
for skills in SKILL_CATEGORIES.values():
    ALL_SKILLS.extend(skills)

# Remove duplicates while preserving order
ALL_SKILLS = list(dict.fromkeys(ALL_SKILLS))


def extract_skills(text: str) -> list[str]:
    """
    Extract skills from raw text using lowercase keyword matching.
    Returns a deduplicated list of matched skills (properly cased).

    Args:
        text: Raw text from resume or job description

    Returns:
        List of matched skill names
    """
    if not text:
        return []

    text_lower = text.lower()
    found_skills = []

    for skill in ALL_SKILLS:
        # Use word-boundary-like check: skill surrounded by non-alphanumeric chars
        # This avoids matching "java" inside "javascript" etc.
        import re
        pattern = r'(?<![a-z0-9])' + re.escape(skill) + r'(?![a-z0-9])'
        if re.search(pattern, text_lower):
            # Store title-cased version for display
            found_skills.append(skill.title())

    # Deduplicate
    return list(dict.fromkeys(found_skills))


def match_skills(resume_skills: list[str], jd_skills: list[str]) -> dict:
    """
    Compare resume skills against job description skills.

    Args:
        resume_skills: Skills extracted from a resume
        jd_skills:     Skills extracted from the job description

    Returns:
        dict with matched_skills, missing_skills, match_score (0-100)
    """
    if not jd_skills:
        return {"matched": [], "missing": [], "score": 0.0}

    resume_lower = {s.lower() for s in resume_skills}
    jd_lower = {s.lower() for s in jd_skills}

    matched = [s for s in jd_skills if s.lower() in resume_lower]
    missing = [s for s in jd_skills if s.lower() not in resume_lower]

    score = (len(matched) / len(jd_skills)) * 100 if jd_skills else 0.0

    return {
        "matched": matched,
        "missing": missing,
        "score": round(score, 2)
    }
