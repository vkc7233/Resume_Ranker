#  AI Resume Ranking & Parsing System

An intelligent, fully local resume screening tool built with Python and Streamlit.  
Upload up to 100 resumes (PDF/DOCX), paste a job description, and get ranked candidates in seconds — **no API keys required**.

---

##  Features

-  Batch upload 50–100 resumes (PDF & DOCX)
-  Automatic extraction: Name, Email, Phone, Skills, Experience, Education, Location
-  Job Description analysis — skills + experience + location extraction
-  Smart weighted scoring algorithm
-  Interactive dashboard with charts, skill chips, and score breakdowns
-  CSV export of ranked results
-  Fully offline — no external APIs

---

## Tech Stack

| Layer       | Technology                              |
|-------------|-----------------------------------------|
| UI          | Streamlit                               |
| PDF Parsing | pdfplumber, PyPDF2 (fallback)           |
| DOCX        | python-docx                             |
| NLP / NER   | spaCy (`en_core_web_sm`)               |
| Data        | pandas, numpy                           |
| Concurrency | ThreadPoolExecutor (batch processing)   |

---

##  Setup & Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd resume_ranker
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### 5. Run the application

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## Project Structure

```
resume_ranker/
│
├── app.py              # Streamlit UI — main entry point
├── parser.py           # Resume parsing (PDF/DOCX → structured dict)
├── skills.py           # Skill database + extraction + matching
├── ranking.py          # Scoring algorithm + candidate ranking
├── utils.py            # Regex helpers (email, phone, experience, education)
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── sample_resumes/     # Put sample PDFs/DOCX here for testing
```

---

## Ranking Algorithm

The system uses a **weighted composite score** (0–100):

```
Final Score = (0.50 × Skill Score)
            + (0.25 × Experience Score)
            + (0.10 × Education Score)
            + (0.15 × Location Score)
```

### Component Details

####  Skill Score (50%)
```
coverage   = (# JD skills found in resume / # total JD skills) × 100
semantic   = TF-IDF cosine similarity(resume, JD) × 100   # 0–100
Skill Score = 0.85 × coverage + 0.15 × semantic
```
Skills are extracted from the job description and resume using a predefined skill
database **with alias canonicalization** — variants like `reactjs` / `react.js` / `react`,
`postgres` / `postgresql`, `k8s` / `kubernetes`, and `ml` / `machine learning` all map to a
single canonical skill, so a resume matches the JD even when the wording differs.

A lightweight **TF-IDF cosine similarity** (pure Python, no external libraries)
between the full resume and the full job description is blended in. This rewards
genuinely relevant resumes that use words outside the skill list, acts as a
tiebreaker, and lets the system still rank candidates when a JD lists no
detectable hard skills (in that case the skill component is 100% semantic).

The skill database spans **280+ canonical skills** across software **and**
non-technical roles — Civil & Construction (AutoCAD, STAAD Pro, RCC, BOQ,
quantity surveying, QA/QC…), Sales & Business Development (B2B/B2C sales, lead
generation, CRM, key account management…), Marketing & Digital Marketing (SEO,
SEM, PPC, Google Ads, social media…), Design, and Business/Office — so the same
engine ranks civil engineers, salespeople, and marketers, not just developers.

####  Experience Score (25%) — relevance-weighted
```
If required experience <= 0:
    Experience Score = min(resume_years / 15, 1.0) × 100
If required experience > 0:
    Experience Score = min(resume_years / required_years, 1.0) × 100

# Then scaled by role-fit so unrelated experience doesn't inflate the score:
relevance      = Skill Score / 100
Experience Score ×= 0.2 + 0.8 × relevance
```
Years of experience are estimated robustly: an explicit claim such as
`"5+ years of experience"` is preferred; otherwise the system parses employment
**date ranges** (`Jan 2019 – Present`, `2018 – 2022`, `06/2020 – 08/2023`) and
sums them, merging overlapping periods so concurrent roles aren't double-counted.
The relevance scaling ensures a genuinely-fitting junior outranks an unrelated
senior (e.g. 7 years of sales experience won't win a civil-engineering role).

####  Education Score (10%)
Degrees are mapped to relevance scores:

| Degree       | Score |
|--------------|-------|
| PhD          | 100   |
| Master's     | 80    |
| Bachelor's   | 60    |
| Diploma      | 40    |
| 12th / HSC   | 20    |
| Not found    | 0     |

####  Location Score (15%)
```
JD location = "Remote" OR Resume says "Remote"  →  100
JD has no location preference                   →   50
Resume location matches JD location             →  100
Unknown resume location                         →   30
Location mismatch                               →    0
```

---

##  Usage

1. **Upload resumes** — Use the file uploader (PDF or DOCX, multiple files at once)
2. **Enter Job Description** — Paste in the sidebar text area
3. **Click "Analyze & Rank Candidates"**
4. **View results:**
   - Summary metrics
   - Ranked table with scores
   - Bar chart visualization
   - Score breakdown per component
   - Detailed candidate cards (matched/missing skills)
5. **Export** — Download ranked results as CSV

---

##  Testing with Sample Resumes

Place sample PDF or DOCX resumes in the `sample_resumes/` folder and upload them via the UI.

---

##  Performance

- Uses `ThreadPoolExecutor` with up to 8 concurrent threads for batch parsing
- Handles 50–100 resumes efficiently
- spaCy NLP runs on only the first 500–1000 characters per resume (name/location detection) to keep processing fast

---

##  Customization

- **Add skills:** Edit `SKILL_CATEGORIES` in `skills.py`
- **Change weights:** Edit `WEIGHTS` dict in `ranking.py`
- **Add more cities:** Edit `LOCATION_KEYWORDS` in `utils.py`
- **Adjust education scoring:** Edit `EDUCATION_KEYWORDS` in `utils.py`

---

##  License

MIT License — free to use and modify.

---

*Built for Sure4Job Technical Assignment — demonstrates end-to-end AI-powered resume ranking without external APIs.*
