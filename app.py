"""
app.py - AI Resume Ranking & Parsing System
Main Streamlit application entry point.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from parser import parse_resume
from ranking import rank_candidates, parse_job_description
from utils import is_valid_resume_file

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Ranker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        color: #e2e8f0;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
        margin: 0;
    }
    .main-header .accent {
        color: #38bdf8;
    }

    /* Score badge */
    .score-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .score-high   { background: #dcfce7; color: #166534; }
    .score-medium { background: #fef9c3; color: #854d0e; }
    .score-low    { background: #fee2e2; color: #991b1b; }

    /* Rank badge */
    .rank-1 { background: #fef08a; color: #713f12; font-weight: 800; }
    .rank-2 { background: #e2e8f0; color: #334155; font-weight: 700; }
    .rank-3 { background: #fed7aa; color: #7c2d12; font-weight: 700; }
    .rank-other { background: #f1f5f9; color: #475569; font-weight: 600; }

    /* Candidate card */
    .candidate-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: box-shadow 0.2s;
    }
    .candidate-card:hover {
        box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    }
    .candidate-name {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.3rem;
    }
    .candidate-meta {
        font-size: 0.82rem;
        color: #64748b;
        margin-bottom: 0.75rem;
    }

    /* Skill chips */
    .skill-chip {
        display: inline-block;
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 500;
        margin: 2px 3px 2px 0;
    }
    .skill-chip-matched {
        background: #f0fdf4;
        color: #166534;
        border-color: #bbf7d0;
    }
    .skill-chip-missing {
        background: #fff1f2;
        color: #be123c;
        border-color: #fecdd3;
    }

    /* Score breakdown bar */
    .score-bar-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Section headings */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e293b;
        border-left: 4px solid #38bdf8;
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        flex: 1;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Sidebar */
    .css-1d391kg { background-color: #f8fafc; }

    /* Dataframe tweaks */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Button */
    .stButton>button {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.5rem 1.5rem;
        transition: opacity 0.2s;
    }
    .stButton>button:hover { opacity: 0.85; }

    /* Hide Streamlit default elements */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def score_color_class(score: float) -> str:
    if score >= 70:  return "score-high"
    if score >= 40:  return "score-medium"
    return "score-low"


def rank_class(rank: int) -> str:
    return {1: "rank-1", 2: "rank-2", 3: "rank-3"}.get(rank, "rank-other")


def render_skills_html(skills: list[str], chip_class: str = "skill-chip") -> str:
    return " ".join(f'<span class="{chip_class}">{s}</span>' for s in skills)


def process_resume_batch(uploaded_files, progress_bar, status_text) -> list[dict]:
    """
    Parse uploaded resumes with a live progress bar.
    Uses ThreadPoolExecutor for concurrency on large batches.
    """
    results = []
    total = len(uploaded_files)

    def parse_single(file):
        try:
            file_bytes = file.read()
            return parse_resume(file_bytes, file.name)
        except Exception as e:
            return {"filename": file.name, "error": str(e), "name": "Parse Error"}

    # Use threads for I/O-bound PDF parsing
    with ThreadPoolExecutor(max_workers=min(8, total)) as executor:
        futures = {executor.submit(parse_single, f): f.name for f in uploaded_files}
        completed = 0
        for future in as_completed(futures):
            results.append(future.result())
            completed += 1
            progress = completed / total
            progress_bar.progress(progress)
            status_text.text(f"⏳ Parsing resumes... {completed}/{total}")

    return results


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🎯 AI Resume <span class="accent">Ranking</span> & Parsing System</h1>
    <p>Upload resumes, enter a job description, and let AI rank your best candidates instantly.</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR — Job Description Input
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📋 Job Description")
    st.markdown("Paste the job description below. The system will extract skills, experience requirements, and location to rank candidates.")

    default_jd = """Looking for a Senior Python Developer with 3+ years of experience.

Requirements:
- Strong Python skills with Django or Flask
- REST API development
- SQL database experience (PostgreSQL/MySQL)
- AWS or cloud platform experience
- Docker and CI/CD knowledge
- Machine Learning experience is a plus
- Location: Bangalore or Remote"""

    jd_text = st.text_area(
        "Job Description",
        value=default_jd,
        height=280,
        help="Paste the complete job description here.",
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    min_score = st.slider(
        "Minimum Score Filter",
        min_value=0, max_value=100, value=0, step=5,
        help="Hide candidates below this score"
    )

    show_raw_text = st.checkbox("Show extracted raw text", value=False)
    show_breakdown = st.checkbox("Show score breakdown", value=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.78rem;color:#94a3b8;line-height:1.6">
    <b>Scoring Formula</b><br>
    🔵 Skills Match — 50%<br>
    🟢 Experience — 25%<br>
    🟡 Education — 10%<br>
    🔴 Location — 15%
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN AREA — File Upload
# ─────────────────────────────────────────────

col_upload, col_jd_preview = st.columns([1.6, 1])

with col_upload:
    st.markdown('<div class="section-title">📤 Upload Resumes</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX resumes (up to 100 files)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Select multiple files at once using Shift+Click or Ctrl+Click",
        label_visibility="collapsed"
    )

    if uploaded_files:
        valid   = [f for f in uploaded_files if is_valid_resume_file(f.name)]
        invalid = [f for f in uploaded_files if not is_valid_resume_file(f.name)]

        st.success(f"✅ {len(valid)} valid resume(s) uploaded")
        if invalid:
            st.warning(f"⚠️ {len(invalid)} file(s) skipped (unsupported format): {', '.join(f.name for f in invalid)}")

with col_jd_preview:
    st.markdown('<div class="section-title">🔍 JD Analysis</div>', unsafe_allow_html=True)
    if jd_text.strip():
        jd = parse_job_description(jd_text)
        st.markdown(f"**Skills detected:** {len(jd['skills'])}")
        if jd["skills"]:
            st.markdown(render_skills_html(jd["skills"][:12]), unsafe_allow_html=True)
        st.markdown(f"**Experience required:** {jd['experience_years']} yrs")
        st.markdown(f"**Location:** {jd['location']}")
    else:
        st.info("Enter a job description in the sidebar.")


# ─────────────────────────────────────────────
# PROCESS & RANK
# ─────────────────────────────────────────────

if uploaded_files and jd_text.strip():
    valid_files = [f for f in uploaded_files if is_valid_resume_file(f.name)]

    if st.button("🚀 Analyze & Rank Candidates", use_container_width=True):
        st.markdown("---")

        # ── Parse resumes
        st.markdown('<div class="section-title">⚙️ Processing</div>', unsafe_allow_html=True)
        progress_bar = st.progress(0)
        status_text  = st.empty()

        with st.spinner(""):
            parsed_resumes = process_resume_batch(valid_files, progress_bar, status_text)

        progress_bar.progress(1.0)
        status_text.text("✅ All resumes parsed!")
        time.sleep(0.3)
        status_text.empty()
        progress_bar.empty()

        # ── Rank candidates
        jd_parsed  = parse_job_description(jd_text)
        ranked     = rank_candidates(parsed_resumes, jd_parsed)

        # Apply minimum score filter
        filtered = [c for c in ranked if c.get("final_score", 0) >= min_score]

        # Count errors
        errors = [c for c in ranked if c.get("error")]

        # ── Summary metrics
        st.markdown('<div class="section-title">📊 Summary</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Resumes", len(ranked))
        with m2:
            st.metric("Passed Filter", len(filtered))
        with m3:
            avg_score = np.mean([c["final_score"] for c in ranked if not c.get("error")]) if ranked else 0
            st.metric("Avg Score", f"{avg_score:.1f}")
        with m4:
            top_score = ranked[0]["final_score"] if ranked else 0
            st.metric("Top Score", f"{top_score:.1f}")

        if errors:
            with st.expander(f"⚠️ {len(errors)} file(s) had errors"):
                for e in errors:
                    st.error(f"**{e['filename']}**: {e.get('error')}")

        # ── Table view
        st.markdown('<div class="section-title">📋 Rankings Table</div>', unsafe_allow_html=True)

        table_data = []
        for c in filtered:
            table_data.append({
                "Rank":       c.get("rank", "—"),
                "Name":       c.get("name", "Unknown"),
                "Email":      c.get("email", "—"),
                "Location":   c.get("location", "—"),
                "Skills (matched)": ", ".join(c.get("matched_skills", [])[:6]),
                "Experience": f"{c.get('experience_years', 0):.1f} yrs",
                "Education":  c.get("education", {}).get("level", "—"),
                "Score":      c.get("final_score", 0),
            })

        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(
                df.style
                  .background_gradient(subset=["Score"], cmap="RdYlGn", vmin=0, vmax=100)
                  .format({"Score": "{:.1f}"}),
                use_container_width=True,
                height=min(400, 50 + 35 * len(table_data)),
            )

        # ── Score Visualization
        st.markdown('<div class="section-title">📈 Score Visualization</div>', unsafe_allow_html=True)

        top_n = min(15, len(filtered))
        if top_n > 0:
            chart_data = pd.DataFrame({
                "Candidate": [c.get("name", f"#{c['rank']}") for c in filtered[:top_n]],
                "Score":     [c["final_score"] for c in filtered[:top_n]],
            }).set_index("Candidate")
            st.bar_chart(chart_data, height=320, color="#38bdf8")

        # ── If breakdown enabled, show component bar chart
        if show_breakdown and filtered:
            st.markdown('<div class="section-title">🔬 Score Breakdown (Top 10)</div>', unsafe_allow_html=True)

            breakdown_rows = []
            for c in filtered[:10]:
                bd = c.get("score_breakdown", {})
                breakdown_rows.append({
                    "Name":       c.get("name", "—"),
                    "Skills(50%)":     bd.get("skill_score", 0),
                    "Experience(25%)": bd.get("exp_score", 0),
                    "Education(10%)":  bd.get("edu_score", 0),
                    "Location(15%)":   bd.get("loc_score", 0),
                })
            if breakdown_rows:
                bd_df = pd.DataFrame(breakdown_rows).set_index("Name")
                st.bar_chart(bd_df, height=350)

        # ── Detailed Candidate Cards
        st.markdown('<div class="section-title">👤 Candidate Details</div>', unsafe_allow_html=True)

        for c in filtered:
            rank = c.get("rank", "—")
            score = c.get("final_score", 0)
            r_class = rank_class(rank)
            s_class = score_color_class(score)
            name    = c.get("name", "Unknown")
            exp     = c.get("experience_years", 0)
            edu     = c.get("education", {}).get("level", "—")
            loc     = c.get("location", "—")
            email   = c.get("email") or "—"
            phone   = c.get("phone") or "—"
            matched = c.get("matched_skills", [])
            missing = c.get("missing_skills", [])
            all_skills = c.get("skills", [])

            with st.expander(
                f"#{rank}  {name}  —  Score: {score:.1f}",
                expanded=(rank <= 3)
            ):
                col_left, col_right = st.columns([2, 1])

                with col_left:
                    st.markdown(f"**📧 Email:** {email}")
                    st.markdown(f"**📞 Phone:** {phone}")
                    st.markdown(f"**📍 Location:** {loc}")
                    st.markdown(f"**🎓 Education:** {edu}")
                    st.markdown(f"**🕐 Experience:** {exp:.1f} years")

                    # Matched / missing skills
                    if matched:
                        st.markdown("**✅ Matched JD Skills:**")
                        st.markdown(render_skills_html(matched, "skill-chip skill-chip-matched"), unsafe_allow_html=True)
                    if missing:
                        st.markdown("**❌ Missing JD Skills:**")
                        st.markdown(render_skills_html(missing, "skill-chip skill-chip-missing"), unsafe_allow_html=True)
                    if all_skills:
                        st.markdown("**🔧 All Resume Skills:**")
                        st.markdown(render_skills_html(all_skills[:20]), unsafe_allow_html=True)

                with col_right:
                    st.markdown(f"### Score: `{score:.1f} / 100`")
                    if show_breakdown:
                        bd = c.get("score_breakdown", {})
                        st.progress(bd.get("skill_score", 0) / 100, text=f"Skills: {bd.get('skill_score', 0):.0f}%")
                        st.progress(bd.get("exp_score", 0)   / 100, text=f"Exp:    {bd.get('exp_score', 0):.0f}%")
                        st.progress(bd.get("edu_score", 0)   / 100, text=f"Edu:    {bd.get('edu_score', 0):.0f}%")
                        st.progress(bd.get("loc_score", 0)   / 100, text=f"Loc:    {bd.get('loc_score', 0):.0f}%")

                if show_raw_text and c.get("raw_text"):
                    with st.expander("📄 View Extracted Text"):
                        st.text(c["raw_text"][:2000] + ("..." if len(c["raw_text"]) > 2000 else ""))

        # ── Export to CSV
        st.markdown("---")
        if table_data:
            export_df = pd.DataFrame(table_data)
            csv = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Export Rankings as CSV",
                data=csv,
                file_name="ranked_candidates.csv",
                mime="text/csv",
                use_container_width=True,
            )

elif not uploaded_files:
    st.info("👈 Upload one or more resumes using the file uploader above to get started.")
elif not jd_text.strip():
    st.warning("👈 Please enter a job description in the sidebar.")
