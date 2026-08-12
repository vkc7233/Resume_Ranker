"""
app.py - AI Resume Ranking & Parsing System
Main Streamlit application entry point.
Run with: streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import time
import math
import json
import re
from pathlib import Path
from datetime import datetime
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
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --brand:      #2563eb;
        --brand-2:    #38bdf8;
        --ink:        #0f172a;
        --muted:      #64748b;
        --line:       #e2e8f0;
        --bg-soft:    #f8fafc;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.4px;
    }
    .block-container { padding-top: 2.2rem; max-width: 1440px; }

    /* ── HERO ───────────────────────────────── */
    .hero {
        position: relative;
        overflow: hidden;
        background:
            radial-gradient(1200px 400px at 10% -20%, rgba(56,189,248,0.25), transparent 60%),
            radial-gradient(900px 380px at 90% 120%, rgba(37,99,235,0.30), transparent 60%),
            linear-gradient(135deg, #0b1220 0%, #131c33 55%, #0d2547 100%);
        padding: 2.6rem 3rem 2.4rem;
        border-radius: 22px;
        margin-bottom: 1.8rem;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 20px 50px -20px rgba(2,10,30,0.65);
    }
    .hero::after {
        content: "";
        position: absolute; inset: 0;
        background-image:
            linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);
        background-size: 46px 46px;
        mask-image: radial-gradient(600px 260px at 25% 0%, #000 20%, transparent 75%);
        pointer-events: none;
    }
    .hero-inner { position: relative; z-index: 2; }
    .hero-badge {
        display: inline-flex; align-items: center; gap: 7px;
        background: rgba(56,189,248,0.12);
        border: 1px solid rgba(56,189,248,0.35);
        color: #7dd3fc;
        font-size: 0.76rem; font-weight: 600;
        letter-spacing: 0.3px;
        padding: 5px 13px; border-radius: 999px;
        margin-bottom: 1.05rem;
    }
    .pulse-dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: #4ade80; box-shadow: 0 0 0 0 rgba(74,222,128,0.7);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(74,222,128,0.65); }
        70%  { box-shadow: 0 0 0 7px rgba(74,222,128,0); }
        100% { box-shadow: 0 0 0 0 rgba(74,222,128,0); }
    }
    .hero h1 {
        color: #f1f5f9;
        font-size: 2.65rem; font-weight: 700;
        margin: 0 0 0.6rem 0; line-height: 1.12;
    }
    .hero .grad {
        background: linear-gradient(100deg, #38bdf8 10%, #818cf8 55%, #c084fc 100%);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p.sub {
        color: #94a3b8; font-size: 1.03rem;
        margin: 0; max-width: 660px; line-height: 1.6;
    }
    .hero-stats { display: flex; flex-wrap: wrap; gap: 2.4rem; margin-top: 1.7rem; }
    .hstat b {
        display: block; color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem; font-weight: 600; line-height: 1.1;
    }
    .hstat span {
        color: #7c8ba1; font-size: 0.75rem;
        text-transform: uppercase; letter-spacing: 0.7px; font-weight: 500;
    }

    /* ── LANDING: steps + features ──────────── */
    .steps { display: flex; gap: 1rem; margin: 0.4rem 0 0.5rem; flex-wrap: wrap; }
    .step {
        flex: 1; min-width: 210px;
        background: #fff; border: 1px solid var(--line);
        border-radius: 14px; padding: 1.15rem 1.25rem;
        position: relative; transition: transform .18s, box-shadow .18s;
    }
    .step:hover { transform: translateY(-3px); box-shadow: 0 12px 28px -14px rgba(15,23,42,0.35); }
    .step-num {
        position: absolute; top: -11px; left: 1.25rem;
        width: 26px; height: 26px; border-radius: 8px;
        background: linear-gradient(135deg, var(--brand) 0%, #1e40af 100%);
        color: #fff; font-size: 0.8rem; font-weight: 700;
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 4px 10px -3px rgba(37,99,235,0.6);
    }
    .step h4 { margin: 0.5rem 0 0.3rem; font-size: 1rem; color: var(--ink); }
    .step p  { margin: 0; font-size: 0.83rem; color: var(--muted); line-height: 1.55; }

    .feat-grid {
        display: grid; grid-template-columns: repeat(3, 1fr);
        gap: 0.9rem; margin-top: 0.4rem;
    }
    .feat {
        background: #fff; border: 1px solid var(--line);
        border-radius: 14px; padding: 1.15rem 1.25rem;
        transition: border-color .18s, transform .18s, box-shadow .18s;
    }
    .feat:hover {
        border-color: #bfdbfe; transform: translateY(-3px);
        box-shadow: 0 12px 26px -16px rgba(37,99,235,0.55);
    }
    .feat-ico {
        width: 38px; height: 38px; border-radius: 11px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem; margin-bottom: 0.7rem;
        background: linear-gradient(135deg, #eff6ff, #dbeafe);
        border: 1px solid #bfdbfe;
    }
    .feat h4 { margin: 0 0 0.28rem; font-size: 0.97rem; color: var(--ink); }
    .feat p  { margin: 0; font-size: 0.82rem; color: var(--muted); line-height: 1.55; }

    /* ── METRIC CARDS ───────────────────────── */
    div[data-testid="stMetric"] {
        background: #fff; border: 1px solid var(--line);
        border-radius: 14px; padding: 1rem 1.15rem;
        box-shadow: 0 2px 10px -6px rgba(15,23,42,0.35);
        transition: transform .18s, box-shadow .18s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 26px -16px rgba(15,23,42,0.5);
    }
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600; color: var(--ink);
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 0.74rem !important; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.6px; color: var(--muted);
    }

    /* ── UPLOADER ───────────────────────────── */
    section[data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(180deg, #f8fbff 0%, #eff6ff 100%);
        border: 2px dashed #93c5fd; border-radius: 14px;
        transition: border-color .18s, background .18s;
    }
    section[data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--brand); background: #eaf2ff;
    }

    /* ── EXPANDERS (candidate cards) ────────── */
    div[data-testid="stExpander"] details {
        border: 1px solid var(--line); border-radius: 14px;
        background: #fff; box-shadow: 0 2px 10px -7px rgba(15,23,42,0.4);
        margin-bottom: 0.6rem; overflow: hidden;
    }
    div[data-testid="stExpander"] summary { font-weight: 600; padding: 0.15rem 0.25rem; }
    div[data-testid="stExpander"] details:hover { border-color: #bfdbfe; }

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
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.12rem;
        font-weight: 700;
        color: var(--ink);
        display: flex; align-items: center; gap: 0.6rem;
        margin: 1.8rem 0 1rem 0;
    }
    .section-title::before {
        content: ""; width: 4px; height: 20px; border-radius: 3px;
        background: linear-gradient(180deg, var(--brand-2), var(--brand));
    }
    .section-sub {
        font-size: 0.83rem; color: var(--muted);
        margin: -0.6rem 0 1rem 0.95rem;
    }

    .formula-row {
        display: flex; align-items: center; justify-content: space-between;
        font-size: 0.78rem; color: #475569; padding: 3px 0;
    }
    .formula-row i { font-style: normal; font-weight: 600; font-family: 'JetBrains Mono', monospace; }

    /* Dataframe tweaks */
    .stDataFrame { border-radius: 12px; overflow: hidden; border: 1px solid var(--line); }

    /* Buttons */
    .stButton>button, .stFormSubmitButton>button {
        background: linear-gradient(135deg, var(--brand) 0%, #1e3a8a 100%);
        color: white; border: none; border-radius: 10px;
        font-weight: 600; font-size: 0.95rem;
        padding: 0.62rem 1.5rem; letter-spacing: 0.2px;
        box-shadow: 0 8px 20px -10px rgba(37,99,235,0.9);
        transition: transform .16s, box-shadow .16s, filter .16s;
    }
    .stButton>button:hover, .stFormSubmitButton>button:hover {
        transform: translateY(-2px); filter: brightness(1.06);
        box-shadow: 0 14px 26px -12px rgba(37,99,235,0.95);
        color: white; border: none;
    }
    /* Outside the tab bar, a secondary button is the quieter of a pair of CTAs */
    .stButton>button[kind="secondary"] {
        background: #fff; color: var(--brand); border: 1px solid #bfdbfe;
        box-shadow: 0 6px 16px -12px rgba(15,23,42,0.6);
    }
    .stButton>button[kind="secondary"]:hover { background: #eff6ff; }
    .stDownloadButton>button {
        border-radius: 10px; font-weight: 600;
        border: 1px solid #bfdbfe; background: #eff6ff; color: #1d4ed8;
        transition: background .16s, transform .16s;
    }
    .stDownloadButton>button:hover { background: #dbeafe; transform: translateY(-2px); }

    /* ── Top tab bar (real Streamlit buttons, styled as a nav) ── */
    [data-testid="stHeader"] { background: transparent; }
    section[data-testid="stSidebar"] { display: none !important; }
    .nav-marker { display: none; }

    /* The column row itself becomes the bar. Sticky works here because the parent
       vertical block is the full page height. */
    div[data-testid="stHorizontalBlock"]:has(.nav-marker) {
        position: sticky; top: 0.2rem; z-index: 999;
        align-items: center;
        padding: 0.42rem 0.7rem; margin-bottom: 1.5rem;
        background: rgba(255,255,255,0.92);
        backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
        border: 1px solid var(--line); border-radius: 14px;
        box-shadow: 0 10px 26px -18px rgba(15,23,42,0.55);
    }
    div[data-testid="stHorizontalBlock"]:has(.nav-marker) div[data-testid="stColumn"] { min-width: 0; }

    .nav-brand-inline {
        display: flex; align-items: center; gap: 0.55rem; white-space: nowrap;
        font-family: 'Space Grotesk', sans-serif; font-size: 1.02rem;
        color: var(--ink); padding-left: 0.2rem;
    }
    .nav-brand-inline b { color: var(--brand); }
    .nav-logo {
        width: 30px; height: 30px; border-radius: 9px; flex: none;
        display: grid; place-items: center; font-size: 0.95rem;
        background: linear-gradient(135deg, var(--brand) 0%, #1e3a8a 100%);
    }

    /* Inactive tab = flat text link (overrides the global solid-blue button style) */
    div[data-testid="stHorizontalBlock"]:has(.nav-marker) .stButton > button {
        background: transparent !important; color: #475569 !important;
        border: 1px solid transparent !important; box-shadow: none !important;
        font-size: 0.85rem !important; font-weight: 500 !important;
        padding: 0.44rem 0.3rem !important; border-radius: 9px !important;
        transform: none !important; white-space: nowrap;
    }
    div[data-testid="stHorizontalBlock"]:has(.nav-marker) .stButton > button:hover {
        background: var(--bg-soft) !important; color: var(--brand) !important;
    }
    /* Active tab */
    div[data-testid="stHorizontalBlock"]:has(.nav-marker) .stButton > button[kind="primary"] {
        background: #eff6ff !important; color: var(--brand) !important;
        border-color: #bfdbfe !important; font-weight: 700 !important;
    }
    /* CTA keeps the solid pill in both states — it is the primary action, not a tab */
    div[data-testid="stHorizontalBlock"]:has(.nav-marker) > div[data-testid="stColumn"]:last-child .stButton > button,
    div[data-testid="stHorizontalBlock"]:has(.nav-marker) > div[data-testid="stColumn"]:last-child .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--brand) 0%, #1e3a8a 100%) !important;
        color: #fff !important; font-weight: 600 !important;
        border-color: transparent !important;
        box-shadow: 0 8px 18px -10px rgba(37,99,235,0.9) !important;
    }
    @media (max-width: 1200px) {
        div[data-testid="stHorizontalBlock"]:has(.nav-marker) .stButton > button { font-size: 0.76rem !important; }
    }
    .anchor { display: block; height: 0; scroll-margin-top: 5rem; }

    /* ── Roles we cover ── */
    .roles { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.9rem; }
    @media (max-width: 1250px) { .feat-grid, .roles { grid-template-columns: repeat(2, 1fr); } }
    @media (max-width: 780px)  { .feat-grid, .roles { grid-template-columns: 1fr; } }
    .role-card {
        border: 1px solid var(--line); border-radius: 14px;
        padding: 1.1rem 1.2rem; background: #fff;
    }
    .role-card h4 { margin: 0 0 0.75rem 0; font-size: 0.97rem; color: var(--ink); }
    .role-chips { display: flex; flex-wrap: wrap; gap: 0.35rem; }
    .role-chips span {
        font-size: 0.76rem; font-weight: 500; color: #334155;
        background: var(--bg-soft); border: 1px solid var(--line);
        padding: 0.25rem 0.62rem; border-radius: 999px;
    }

    /* ── FAQ accordion ── */
    .faq details {
        border: 1px solid var(--line); border-radius: 12px;
        background: #fff; margin-bottom: 0.6rem; padding: 0 1.15rem;
    }
    .faq summary {
        cursor: pointer; list-style: none; padding: 0.9rem 0;
        font-weight: 600; font-size: 0.92rem; color: var(--ink);
        display: flex; align-items: center; justify-content: space-between; gap: 1rem;
    }
    .faq summary::-webkit-details-marker { display: none; }
    .faq summary::after { content: "+"; font-size: 1.25rem; font-weight: 400; color: var(--muted); }
    .faq details[open] summary { border-bottom: 1px solid var(--line); }
    .faq details[open] summary::after { content: "−"; }
    .faq p { margin: 0.85rem 0 1rem 0; font-size: 0.88rem; line-height: 1.65; color: var(--muted); }

    /* ── Footer ── */
    .site-footer {
        margin: 3rem 0 1rem 0; padding: 2.2rem 2.4rem 1.3rem;
        background: linear-gradient(135deg, #0b1220 0%, #131c33 100%);
        border-radius: 20px; color: #cbd5e1;
    }
    .foot-grid { display: grid; grid-template-columns: 1.7fr 1fr 1fr 1fr; gap: 2rem; }
    @media (max-width: 900px) { .foot-grid { grid-template-columns: 1fr 1fr; } }
    .foot-brand .fb-top { display: flex; align-items: center; gap: 0.55rem; margin-bottom: 0.8rem; }
    .foot-brand .fl {
        width: 30px; height: 30px; border-radius: 9px; display: grid; place-items: center;
        background: linear-gradient(135deg, var(--brand) 0%, #1e3a8a 100%);
    }
    .foot-brand b { color: #fff; font-family: 'Space Grotesk', sans-serif; font-size: 1.02rem; }
    .foot-brand p { margin: 0 0 0.9rem 0; max-width: 36ch; font-size: 0.84rem; line-height: 1.65; color: #94a3b8; }
    .foot-badge {
        display: inline-flex; align-items: center; gap: 0.4rem;
        font-size: 0.74rem; font-weight: 600; color: #4ade80;
        background: rgba(74,222,128,0.10); border: 1px solid rgba(74,222,128,0.28);
        padding: 0.28rem 0.65rem; border-radius: 999px;
    }
    .site-footer h5 {
        margin: 0 0 0.85rem 0 !important; padding: 0 !important;
        color: #fff; font-family: 'Space Grotesk', sans-serif;
        font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase;
    }
    .site-footer a { text-decoration: none !important; }
    .fcol a {
        display: block; padding: 0.24rem 0; font-size: 0.85rem;
        color: #94a3b8 !important; transition: color .15s;
    }
    .fcol a:hover { color: #38bdf8 !important; }
    .foot-bottom {
        display: flex; align-items: center; justify-content: space-between;
        flex-wrap: wrap; gap: 0.8rem; margin-top: 1.8rem; padding-top: 1.1rem;
        border-top: 1px solid rgba(255,255,255,0.10); font-size: 0.8rem; color: #64748b;
    }
    .foot-bottom a { color: #64748b !important; margin-left: 1.1rem; }
    .foot-bottom a:hover { color: #cbd5e1 !important; }

    /* ── Home: illustration cards ── */
    .viz-grid { display: grid; grid-template-columns: 1.08fr 1fr; gap: 1.1rem; align-items: start; }
    /* Without this the wide SVG's min-content width blows past its fr track. */
    .viz-grid > * { min-width: 0; }
    @media (max-width: 1100px) { .viz-grid { grid-template-columns: 1fr; } }
    .viz-card {
        border: 1px solid var(--line); border-radius: 18px; background: #fff;
        padding: 1.2rem 1.35rem 1.35rem;
        box-shadow: 0 14px 32px -26px rgba(15,23,42,0.65);
    }
    .viz-card h4 { margin: 0 0 0.2rem; font-size: 1rem; color: var(--ink); }
    .viz-sub { margin: 0 0 1rem; font-size: 0.82rem; color: var(--muted); line-height: 1.55; }
    .viz-dark {
        border: 1px solid rgba(255,255,255,0.09); border-radius: 20px; color: #cbd5e1;
        padding: 1.5rem 1.7rem 1.6rem;
        background:
            radial-gradient(900px 320px at 12% -20%, rgba(56,189,248,0.20), transparent 60%),
            linear-gradient(135deg, #0b1220 0%, #131c33 60%, #0d2547 100%);
    }
    .viz-dark h4 { color: #f1f5f9; margin: 0 0 0.2rem; font-size: 1.08rem; }
    .viz-dark .viz-sub { color: #94a3b8; }
    .viz-svg { width: 100%; height: auto; display: block; }
    .viz-legend { display: flex; gap: 1.3rem; flex-wrap: wrap; margin-top: 0.9rem; font-size: 0.78rem; color: var(--muted); }
    .viz-legend i { font-style: normal; display: inline-block; width: 22px; height: 3px; border-radius: 2px; vertical-align: middle; margin-right: 6px; }

    /* Scanner animation */
    @keyframes rr-beam { 0% { transform: translateY(0); } 100% { transform: translateY(232px); } }
    .rr-beam { animation: rr-beam 3s ease-in-out infinite alternate; }
    @keyframes rr-hl { 0%, 40% { opacity: 0; } 55%, 100% { opacity: 1; } }
    .rr-hl { opacity: 0; animation: rr-hl 3s ease-out infinite alternate; }
    .rr-hl.d2 { animation-delay: .35s; }
    .rr-hl.d3 { animation-delay: .7s; }
    .rr-hl.d4 { animation-delay: 1.05s; }

    /* Extracted-field rows */
    .efield {
        display: flex; align-items: center; gap: 0.7rem;
        padding: 0.55rem 0.75rem; margin-bottom: 0.5rem;
        border: 1px solid var(--line); border-radius: 11px; background: var(--bg-soft);
    }
    .efield .ek {
        flex: none; width: 88px; font-size: 0.68rem; font-weight: 700;
        letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted);
    }
    .efield .ev { font-size: 0.85rem; font-weight: 600; color: var(--ink); }
    .efield .ok { margin-left: auto; color: #16a34a; font-weight: 700; }

    /* Numbered process cards */
    .proc-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.9rem; }
    @media (max-width: 1100px) { .proc-grid { grid-template-columns: repeat(2, 1fr); } }
    @media (max-width: 700px)  { .proc-grid { grid-template-columns: 1fr; } }
    .proc {
        position: relative; overflow: hidden;
        border: 1px solid var(--line); border-radius: 14px;
        background: #fff; padding: 1.1rem 1.2rem;
    }
    .proc::before {
        content: attr(data-n); position: absolute; right: 0.75rem; top: -0.15rem;
        font-family: 'JetBrains Mono', monospace; font-size: 2.7rem; font-weight: 700; color: #eef2f7;
    }
    .proc h4 { position: relative; margin: 0.15rem 0 0.3rem; font-size: 0.95rem; color: var(--ink); }
    .proc p  { position: relative; margin: 0; font-size: 0.82rem; color: var(--muted); line-height: 1.55; }

    /* Trust strip */
    .trust { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.9rem; }
    @media (max-width: 900px) { .trust { grid-template-columns: repeat(2, 1fr); } }
    .trust > div { border: 1px solid var(--line); border-radius: 14px; background: #fff; padding: 1rem 1.1rem; text-align: center; }
    .trust b { display: block; font-family: 'JetBrains Mono', monospace; font-size: 1.45rem; color: var(--ink); }
    .trust span { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }

    /* Numbered step label above an input */
    .step-label {
        font-size: 0.78rem; font-weight: 700; color: var(--brand);
        text-transform: uppercase; letter-spacing: 0.07em;
        margin: 1.1rem 0 0.4rem;
    }

    /* Circular score gauge */
    .ring-wrap { text-align: center; padding: 0.3rem 0 0.1rem; }
    .ring { width: 100%; max-width: 178px; height: auto; }
    .ring-num {
        font-family: 'JetBrains Mono', monospace; font-size: 30px; font-weight: 700;
        text-anchor: middle; dominant-baseline: middle;
    }
    .ring-den { font-size: 9.5px; fill: #94a3b8; text-anchor: middle; letter-spacing: 0.06em; }
    .ring-cap { font-size: 1.02rem; font-weight: 700; color: var(--ink); margin-top: 0.35rem; }
    .ring-sub { font-size: 0.79rem; color: var(--muted); margin-top: 0.15rem; }

    /* Weighted component bars */
    .cbar { margin-bottom: 0.72rem; }
    .cbar-top {
        display: flex; align-items: baseline; gap: 0.5rem;
        font-size: 0.85rem; color: var(--ink); margin-bottom: 0.28rem;
    }
    .cbar-top span:first-child { flex: 1; }
    .cbar-w {
        font-size: 0.7rem; color: var(--muted); background: #f1f5f9;
        padding: 0.05rem 0.4rem; border-radius: 5px; font-weight: 600;
    }
    .cbar-top b { font-family: 'JetBrains Mono', monospace; font-size: 0.86rem; min-width: 28px; text-align: right; }
    .cbar-track { height: 8px; background: #eef2f7; border-radius: 99px; overflow: hidden; }
    .cbar-track i { display: block; height: 100%; border-radius: 99px; transition: width .5s ease; }

    /* Sign-up panel */
    .or-rule {
        display: flex; align-items: center; text-align: center;
        color: #94a3b8; font-size: 0.78rem; margin: 0.9rem 0 0.6rem;
    }
    .or-rule::before, .or-rule::after {
        content: ""; flex: 1; height: 1px; background: var(--line);
    }
    .or-rule span { padding: 0 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; }

    /* Google's mark, drawn in CSS so the button needs no network fetch — this app
       makes no outbound calls and a remotely hosted logo would break that promise. */
    .st-key-google_signup button [data-testid="stMarkdownContainer"] p::before {
        content: ""; display: inline-block; width: 16px; height: 16px;
        margin-right: 0.6rem; vertical-align: -3px;
        background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 48 48'%3E%3Cpath fill='%23EA4335' d='M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z'/%3E%3Cpath fill='%234285F4' d='M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z'/%3E%3Cpath fill='%23FBBC05' d='M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.28-3.14.76-4.59l-7.97-6.19C.92 16.46 0 20.12 0 24s.92 7.54 2.56 10.78l7.97-6.19z'/%3E%3Cpath fill='%2334A853' d='M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z'/%3E%3C/svg%3E") no-repeat center / contain;
    }
    .st-key-google_signup button { font-weight: 600 !important; }

    .plan-card {
        border: 1px solid #bfdbfe; border-radius: 16px; padding: 1.4rem 1.5rem;
        background: linear-gradient(160deg, #f8fbff 0%, #eff6ff 100%);
    }
    .plan-badge {
        display: inline-block; background: #1d4ed8; color: #fff; font-size: 0.68rem;
        font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
        padding: 0.22rem 0.6rem; border-radius: 99px; margin-bottom: 0.7rem;
    }
    .plan-price { font-size: 2.1rem; font-weight: 800; color: var(--ink); line-height: 1; }
    .plan-price span {
        font-size: 0.8rem; font-weight: 500; color: var(--muted); margin-left: 0.4rem;
    }
    .plan-list { margin: 1rem 0 0; padding-left: 1.05rem; }
    .plan-list li { font-size: 0.85rem; color: #334155; line-height: 1.9; }
    .plan-note {
        margin-top: 1rem; padding-top: 0.85rem; border-top: 1px dashed #bfdbfe;
        font-size: 0.78rem; color: var(--muted); line-height: 1.6;
    }

    /* Signed-in chip */
    .acct {
        display: flex; align-items: center; gap: 0.6rem; padding: 0.55rem 0.9rem;
        border: 1px solid #bbf7d0; background: #f0fdf4; border-radius: 12px;
        font-size: 0.82rem; color: #166534; margin-bottom: 0.9rem;
    }
    .acct .av {
        width: 26px; height: 26px; border-radius: 99px; background: #16a34a; color: #fff;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 0.78rem; flex: 0 0 auto;
    }
    .acct b { color: #14532d; }
    .acct .pill {
        margin-left: auto; background: #dcfce7; border-radius: 99px;
        padding: 0.12rem 0.55rem; font-size: 0.7rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.05em;
    }

    /* Locked-state teaser */
    .lockbox {
        border: 1px dashed #fbbf24; background: #fffbeb; border-radius: 14px;
        padding: 1rem 1.2rem; margin: 0.6rem 0 0.2rem;
    }
    /* A div, not an <h5> — Streamlit hangs a link-anchor icon off every heading it
       renders, which looks like a stray glyph in the middle of a banner. */
    .lockbox b.lb-h {
        display: block; margin: 0 0 0.3rem; font-size: 0.95rem;
        color: #92400e; font-weight: 700;
    }
    .lockbox p { margin: 0; font-size: 0.84rem; color: #a16207; line-height: 1.6; }

    /* Batch capacity meter */
    .meter {
        border: 1px solid var(--line); background: #fff; border-radius: 12px;
        padding: 0.75rem 0.95rem; margin: 0.55rem 0 0.2rem;
    }
    .meter-top {
        display: flex; justify-content: space-between; align-items: baseline;
        font-size: 0.8rem; color: var(--muted); margin-bottom: 0.4rem;
        text-transform: uppercase; letter-spacing: 0.04em; font-weight: 600;
    }
    .meter-top b { font-size: 0.92rem; color: var(--ink); letter-spacing: 0; }
    .meter-track { height: 9px; background: #eef2f7; border-radius: 99px; overflow: hidden; }
    .meter-track i { display: block; height: 100%; border-radius: 99px; transition: width .5s ease; }
    .meter-sub { font-size: 0.78rem; color: var(--muted); margin-top: 0.4rem; }

    /* Shortlist mix bar */
    .mix { margin: 0.2rem 0 0.4rem; }
    .mix-bar {
        display: flex; height: 16px; border-radius: 99px; overflow: hidden;
        background: #eef2f7; border: 1px solid var(--line);
    }
    .mix-bar i { display: block; height: 100%; transition: width .5s ease; }
    .mix-keys { display: flex; flex-wrap: wrap; gap: 0.5rem 1.1rem; margin-top: 0.6rem; }
    .mix-key { display: inline-flex; align-items: center; font-size: 0.8rem; color: var(--muted); }
    .mix-key em { width: 10px; height: 10px; border-radius: 3px; margin-right: 0.4rem; }
    .mix-key b { color: var(--ink); margin-left: 0.35rem; }

    /* Skill demand coverage */
    .dem-wrap { display: grid; gap: 0.45rem; margin-top: 0.3rem; }
    .dem { display: grid; grid-template-columns: 150px 1fr 52px; align-items: center; gap: 0.7rem; }
    .dem-name {
        font-size: 0.83rem; color: var(--ink); font-weight: 600; text-transform: capitalize;
        overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .dem-track { height: 8px; background: #eef2f7; border-radius: 99px; overflow: hidden; }
    .dem-track i { display: block; height: 100%; border-radius: 99px; transition: width .5s ease; }
    .dem-num { font-size: 0.78rem; font-weight: 700; text-align: right; }

    /* Actionable advice + ATS checks */
    .advice {
        border-left: 3px solid var(--brand); background: #f8fafc;
        padding: 0.7rem 0.95rem; border-radius: 0 10px 10px 0;
        margin-bottom: 0.55rem; font-size: 0.88rem; line-height: 1.6; color: #334155;
    }
    .ats-row {
        border-radius: 10px; padding: 0.6rem 0.8rem; margin-bottom: 0.5rem;
        font-size: 0.85rem; border: 1px solid var(--line);
    }
    .ats-row b { display: block; font-weight: 600; }
    .ats-row span { display: block; color: var(--muted); font-size: 0.8rem; margin-top: 0.22rem; line-height: 1.55; }
    .ats-ok  { background: #f0fdf4; border-color: #bbf7d0; }
    .ats-ok b  { color: #15803d; }
    .ats-bad { background: #fff7ed; border-color: #fed7aa; }
    .ats-bad b { color: #c2410c; }

    /* Audience split on the home page */
    .aud {
        border: 1px solid var(--line); border-radius: 16px; padding: 1.3rem 1.4rem;
        background: #fff; min-height: 280px; margin-bottom: 0.8rem;
        transition: transform .18s, box-shadow .18s, border-color .18s;
    }
    .aud-r { border-top: 3px solid #2563eb; }
    .aud-c { border-top: 3px solid #16a34a; }
    .aud:hover { transform: translateY(-3px); border-color: #bfdbfe; box-shadow: 0 18px 38px -24px rgba(15,23,42,0.4); }
    .aud .tagline {
        display: inline-block; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; padding: 0.2rem 0.55rem; border-radius: 99px; margin-bottom: 0.6rem;
    }
    .aud-r .tagline { background: #eff6ff; color: #1d4ed8; }
    .aud-c .tagline { background: #f0fdf4; color: #15803d; }
    .aud h4 { margin: 0 0 0.3rem; font-size: 1.12rem; color: var(--ink); font-weight: 700; }
    .aud p { margin: 0 0 0.7rem; font-size: 0.87rem; color: var(--muted); line-height: 1.6; }
    .aud ul { margin: 0; padding-left: 1.05rem; }
    .aud li { font-size: 0.84rem; color: #475569; line-height: 1.75; }

    /* Stage list (How it works) */
    .stage-list { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.7rem; margin-top: 1rem; }
    @media (max-width: 900px) { .stage-list { grid-template-columns: 1fr; } }
    .stage {
        background: rgba(255,255,255,0.045); border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px; padding: 0.8rem 0.95rem;
    }
    .stage b {
        display: block; color: #7dd3fc; font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem; letter-spacing: 0.04em; margin-bottom: 0.3rem;
    }
    .stage span { color: #b8c4d4; font-size: 0.83rem; line-height: 1.58; }

    /* Worked scoring example */
    .calc { margin-top: 0.9rem; }
    .calc-row, .calc-total {
        display: flex; justify-content: space-between; align-items: baseline; gap: 1rem;
        padding: 0.5rem 0; border-bottom: 1px dashed var(--line);
    }
    .calc-row span, .calc-total span { color: var(--muted); font-size: 0.84rem; }
    .calc-row b {
        font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;
        color: var(--ink); white-space: nowrap;
    }
    .calc-total { border-bottom: none; border-top: 2px solid var(--line); margin-top: 0.25rem; }
    .calc-total span { color: var(--ink); font-weight: 700; font-size: 0.88rem; }
    .calc-total b {
        font-family: 'JetBrains Mono', monospace; font-size: 0.95rem;
        color: #16a34a; white-space: nowrap;
    }

    /* Closing CTA band */
    .cta-band {
        margin-top: 1.7rem; border-radius: 20px; padding: 1.7rem 2rem 4.9rem;
        border: 1px solid rgba(255,255,255,0.09);
        background:
            radial-gradient(700px 240px at 12% -20%, rgba(56,189,248,0.22), transparent 60%),
            linear-gradient(135deg, #0b1220 0%, #132244 100%);
    }
    .cta-band h3 { margin: 0 0 0.35rem; color: #f1f5f9; font-size: 1.28rem; }
    .cta-band p  { margin: 0; color: #94a3b8; font-size: 0.88rem; line-height: 1.6; max-width: 68ch; }
    .cta-btn-marker { display: none; }
    /* Lift the button row into the space reserved by the band's bottom padding. */
    div[data-testid="stHorizontalBlock"]:has(.cta-btn-marker) {
        margin-top: -4rem; padding: 0 2rem; position: relative; z-index: 2;
    }

    /* Pricing */
    .price-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
    @media (max-width: 1000px) { .price-grid { grid-template-columns: 1fr; } }
    .price-card { position: relative; border: 1px solid var(--line); border-radius: 18px; background: #fff; padding: 1.5rem 1.4rem; }
    .price-card.pop { border-color: #93c5fd; box-shadow: 0 18px 42px -28px rgba(37,99,235,0.9); }
    .price-tag {
        position: absolute; top: -11px; right: 1.2rem;
        background: linear-gradient(135deg, var(--brand), #1e3a8a); color: #fff;
        font-size: 0.66rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
        padding: 0.3rem 0.7rem; border-radius: 999px;
    }
    .price-card h4 { margin: 0 0 0.45rem; font-size: 1.05rem; color: var(--ink); }
    .p-amt { font-family: 'JetBrains Mono', monospace; font-size: 1.85rem; font-weight: 600; color: var(--ink); }
    .p-amt small { font-family: 'Inter', sans-serif; font-size: 0.78rem; font-weight: 500; color: var(--muted); }
    .p-desc { margin: 0.5rem 0 1.1rem; font-size: 0.83rem; color: var(--muted); line-height: 1.55; }
    .price-list { list-style: none; margin: 0; padding: 0; }
    .price-list li { position: relative; padding: 0.32rem 0 0.32rem 1.5rem; font-size: 0.85rem; color: #334155; }
    .price-list li::before { content: "✓"; position: absolute; left: 0; color: #16a34a; font-weight: 700; }
    .price-list li.no { color: #94a3b8; }
    .price-list li.no::before { content: "—"; color: #cbd5e1; }

    /* Contact */
    .contact-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
    @media (max-width: 1000px) { .contact-grid { grid-template-columns: 1fr; } }
    .contact-card { border: 1px solid var(--line); border-radius: 16px; background: #fff; padding: 1.2rem 1.3rem; }
    .c-ico {
        width: 38px; height: 38px; border-radius: 11px; margin-bottom: 0.7rem;
        display: grid; place-items: center; font-size: 1.1rem;
        background: linear-gradient(135deg, #eff6ff, #dbeafe); border: 1px solid #bfdbfe;
    }
    .contact-card h4 { margin: 0 0 0.3rem; font-size: 0.97rem; color: var(--ink); }
    .contact-card p  { margin: 0; font-size: 0.84rem; color: var(--muted); line-height: 1.65; }

    /* Hide Streamlit default elements */
    #MainMenu, footer { visibility: hidden; }
    [data-testid="stAppDeployButton"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_parse_jd(jd_text: str) -> dict:
    """Parse the JD once per unique text — avoids re-extracting on every rerun."""
    return parse_job_description(jd_text)


def score_color_class(score: float) -> str:
    if score >= 70:  return "score-high"
    if score >= 40:  return "score-medium"
    return "score-low"


def rank_class(rank: int) -> str:
    return {1: "rank-1", 2: "rank-2", 3: "rank-3"}.get(rank, "rank-other")


def render_skills_html(skills: list[str], chip_class: str = "skill-chip") -> str:
    return " ".join(f'<span class="{chip_class}">{s}</span>' for s in skills)


def score_hex(score: float) -> str:
    if score >= 70: return "#16a34a"
    if score >= 40: return "#f59e0b"
    return "#ef4444"


# An SVG arc rather than a chart library: it costs no dependency, scales cleanly and
# can be dropped inside any card. Kept free of blank lines — markdown ends an HTML
# block at the first empty line.
def score_ring_html(score: float, caption: str = "", sub: str = "") -> str:
    pct  = max(0.0, min(100.0, float(score)))
    circ = 2 * math.pi * 54
    return (
        '<div class="ring-wrap">'
        '<svg viewBox="0 0 128 128" class="ring">'
        '<circle cx="64" cy="64" r="54" fill="none" stroke="#e2e8f0" stroke-width="11"/>'
        f'<circle cx="64" cy="64" r="54" fill="none" stroke="{score_hex(pct)}" stroke-width="11"'
        f' stroke-linecap="round" stroke-dasharray="{circ*pct/100:.1f} {circ:.1f}"'
        ' transform="rotate(-90 64 64)"/>'
        f'<text x="64" y="60" class="ring-num" fill="{score_hex(pct)}">{pct:.0f}</text>'
        '<text x="64" y="80" class="ring-den">out of 100</text>'
        '</svg>'
        + (f'<div class="ring-cap">{caption}</div>' if caption else "")
        + (f'<div class="ring-sub">{sub}</div>' if sub else "")
        + '</div>'
    )


def bar_row_html(label: str, value: float, weight: str) -> str:
    """One weighted component of the score, drawn as a labelled progress bar."""
    v = max(0.0, min(100.0, float(value)))
    return (
        '<div class="cbar">'
        f'<div class="cbar-top"><span>{label}</span>'
        f'<span class="cbar-w">{weight}</span>'
        f'<b style="color:{score_hex(v)}">{v:.0f}</b></div>'
        f'<div class="cbar-track"><i style="width:{v:.1f}%;background:{score_hex(v)}"></i></div>'
        '</div>'
    )


VERDICT_BANDS = [
    ("Excellent fit", 75, 101, "#15803d"),
    ("Strong fit",    60,  75, "#65a30d"),
    ("Moderate fit",  45,  60, "#f59e0b"),
    ("Weak fit",      30,  45, "#f97316"),
    ("Poor fit",       0,  30, "#ef4444"),
]


def pipeline_mix_html(ranked: list) -> str:
    """
    The shape of the whole batch in one bar. A pile of 'Poor fit' usually means the advert
    was aimed at the wrong pool, not that every applicant is weak — worth seeing before
    reading forty individual cards.
    """
    scored = [c for c in ranked if not c.get("error")]
    total  = len(scored) or 1
    counts = [
        (name, colour, sum(1 for c in scored if lo <= c.get("final_score", 0) < hi))
        for name, lo, hi, colour in VERDICT_BANDS
    ]
    segs = "".join(
        f'<i style="width:{n/total*100:.2f}%;background:{colour}" title="{name}: {n}"></i>'
        for name, colour, n in counts if n
    )
    keys = "".join(
        f'<span class="mix-key"><em style="background:{colour}"></em>{name}<b>{n}</b></span>'
        for name, colour, n in counts if n
    )
    return f'<div class="mix"><div class="mix-bar">{segs}</div><div class="mix-keys">{keys}</div></div>'


def skill_demand_html(ranked: list, jd_skills: list, limit: int = 12) -> str:
    """
    For each skill the advert asks for, how much of the batch actually has it. Reads the
    requirement rather than the candidate: a 0% row is a signal the requirement is
    unrealistic for this market, not that the applicants are bad.
    """
    scored = [c for c in ranked if not c.get("error")]
    if not scored or not jd_skills:
        return ""
    total = len(scored)
    rows  = []
    for skill in jd_skills[:limit]:
        have = sum(1 for c in scored if skill in (c.get("matched_skills") or []))
        pct  = have / total * 100
        hue  = "#16a34a" if pct >= 50 else ("#f59e0b" if pct >= 20 else "#ef4444")
        rows.append(
            '<div class="dem">'
            f'<span class="dem-name">{skill}</span>'
            f'<span class="dem-track"><i style="width:{pct:.1f}%;background:{hue}"></i></span>'
            f'<span class="dem-num" style="color:{hue}">{have}/{total}</span>'
            '</div>'
        )
    return f'<div class="dem-wrap">{"".join(rows)}</div>'


def batch_meter_html(count: int, cap: int | None = None) -> str:
    """How much of the per-batch allowance the current selection uses."""
    cap  = cap or MAX_BATCH
    pct  = min(100.0, count / cap * 100)
    hue  = "#16a34a" if pct < 75 else ("#f59e0b" if pct < 100 else "#ef4444")
    return (
        '<div class="meter">'
        f'<div class="meter-top"><span>Batch capacity</span>'
        f'<b>{count} / {cap} CVs</b></div>'
        f'<div class="meter-track"><i style="width:{pct:.1f}%;background:{hue}"></i></div>'
        f'<div class="meter-sub">Room for {max(0, cap - count)} more in this run.</div>'
        '</div>'
    )


def ats_checks(cand: dict) -> list[tuple[bool, str, str]]:
    """
    Format checks an applicant tracking system would run before a human ever reads the
    CV. Deliberately separate from the match score: a resume can fit the role perfectly
    and still be thrown out for being unreadable.
    """
    raw    = cand.get("raw_text") or ""
    words  = len(raw.split())
    skills = cand.get("skills") or []
    return [
        (bool(cand.get("email")), "Email address found",
         "Add a plain-text email near the top — recruiters filter on it."),
        (bool(cand.get("phone")), "Phone number found",
         "Add a phone number in digits, not inside an image or icon."),
        (cand.get("name", "Unknown") != "Unknown", "Name detected",
         "Put your full name on its own line at the very top of page one."),
        # A scan yields almost no words at all; 100 is comfortably above that floor and
        # well below a thin-but-real CV, which the length check below reports instead.
        (words >= 30, "Machine-readable text layer",
         "Almost no text came out of this file, so it is probably a scan or an image. "
         "Export a real PDF from Word rather than photographing a printout, or the ATS "
         "will see a blank page."),
        (150 <= words <= 1200, "Sensible length",
         f"This CV is {words} words. Aim for roughly 400–900 — one to two pages."),
        (len(skills) >= 5, "Skills are stated explicitly",
         "Add a short Skills section listing tools and methods by name. Systems match on "
         "exact words, not on implication."),
        (cand.get("education", {}).get("level", "Not Specified") != "Not Specified",
         "Education is identifiable",
         "Spell out the qualification, e.g. 'B.E. Civil Engineering, 2019'."),
        (cand.get("location", "Not Specified") != "Not Specified", "Location is stated",
         "Add your city. Location-based shortlisting will skip a CV without one."),
    ]


def process_resume_batch(uploaded_files, progress_bar, status_text) -> list[dict]:
    """
    Parse uploaded resumes with a live progress bar.
    Uses ThreadPoolExecutor for concurrency on large batches.
    """
    results = []
    total = len(uploaded_files)

    def parse_single(file):
        try:
            # Rewind: the buffer is already at EOF if this batch was analysed once before.
            file.seek(0)
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
# TAB ROUTER + HEADER
# ─────────────────────────────────────────────

TABS = ["Home", "How it works", "Features", "Roles", "Pricing", "FAQ", "Contact"]
CTA_TAB  = "Start screening"   # recruiter: a batch of up to MAX_BATCH CVs
CAND_TAB = "Check my CV"       # candidate: one CV, scored against one role

MAX_BATCH = 100

# The free tier exists to make "Try for free" mean something, not to lock anyone out —
# there is a "continue without an account" escape on the sign-up panel.
FREE_BATCH = 10

# There is no server behind this app, so an "account" is a row in a local JSON file. It
# records who is using the tool on this machine; it authenticates nothing and is not a
# credential store. Passwords are deliberately never asked for or held.
ACCOUNT_FILE = Path(__file__).with_name(".rr_account.json")


def load_account() -> dict | None:
    try:
        if ACCOUNT_FILE.exists():
            data = json.loads(ACCOUNT_FILE.read_text(encoding="utf-8"))
            return data if data.get("email") else None
    except (OSError, ValueError):
        pass
    return None


def save_account(email: str, org: str = "") -> dict:
    acct = {
        "email":      email.strip(),
        "org":        org.strip(),
        "plan":       "Free trial",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        ACCOUNT_FILE.write_text(json.dumps(acct, indent=2), encoding="utf-8")
    except OSError:
        pass          # a read-only install still gets the session-level account
    return acct


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")

if "page" not in st.session_state:
    st.session_state.page = "Home"

# Loaded once per session; the file makes it survive a restart.
if "account" not in st.session_state:
    st.session_state.account = load_account()
st.session_state.setdefault("show_signup", False)
st.session_state.setdefault("guest_mode", False)

# Streamlit discards the state of any widget that was not rendered during a run, so a
# half-finished screening setup would vanish the moment the reader opens another tab.
# These mirrors are plain session keys, which survive, and the widgets are seeded from
# them and write back to them.
for _key, _default in (
    ("keep_jd_text", None),          # None → fall back to DEFAULT_JD at render time
    ("keep_cand_jd", None),
    ("keep_min_score", 0),
    ("keep_show_breakdown", True),
    ("keep_show_raw_text", False),
):
    st.session_state.setdefault(_key, _default)


def go(page: str) -> None:
    st.session_state.page = page


def open_signup() -> None:
    st.session_state.show_signup = True
    st.session_state.page = CTA_TAB


def start_guest() -> None:
    st.session_state.guest_mode  = True
    st.session_state.show_signup = False
    st.session_state.page = CTA_TAB


def batch_limit() -> int:
    """Signed-in desks get the full batch; everyone else gets the trial size."""
    return MAX_BATCH if st.session_state.get("account") else FREE_BATCH


# Buttons rather than <a href="#..."> links: an anchor would reload the browser and
# start a fresh Streamlit session, dropping the uploaded files and the shortlist.
_nav = st.columns(
    [1.72, 0.54, 0.98, 0.80, 0.56, 0.70, 0.48, 0.76, 1.18, 1.42],
    gap="small",
    vertical_alignment="center",
)
with _nav[0]:
    st.markdown(
        '<span class="nav-marker"></span>'
        '<div class="nav-brand-inline"><span class="nav-logo">🎯</span>'
        '<span>Resume<b>Ranker</b></span></div>',
        unsafe_allow_html=True,
    )
for _i, _tab in enumerate(TABS):
    with _nav[_i + 1]:
        st.button(
            _tab, key=f"nav_{_i}", use_container_width=True, on_click=go, args=(_tab,),
            type=("primary" if st.session_state.page == _tab else "secondary"),
        )
with _nav[8]:
    st.button(
        f"📄 {CAND_TAB}", key="nav_cand", use_container_width=True,
        on_click=go, args=(CAND_TAB,),
        type=("primary" if st.session_state.page == CAND_TAB else "secondary"),
    )
with _nav[9]:
    # Until there is an account, the nav's end cap is the trial offer rather than the
    # screening page — it is the one button visible from every tab.
    if st.session_state.get("account"):
        st.button(
            f"🚀 {CTA_TAB}", key="nav_cta", use_container_width=True,
            on_click=go, args=(CTA_TAB,),
            type=("primary" if st.session_state.page == CTA_TAB else "secondary"),
        )
    else:
        st.button(
            "✨ Try for free", key="nav_cta", use_container_width=True,
            on_click=open_signup, type="primary",
        )

PAGE = st.session_state.page


def render_signup() -> None:
    """
    The sign-up panel. Passwordless on purpose: this app has no server to check a
    password against, and a password box with nothing behind it would collect real
    credentials into a local file. Email identifies the desk; it grants no access
    that "continue without an account" does not also grant.
    """
    st.markdown('<div class="section-title">Start your free trial</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">No card, no licence key. The full '
        f'{MAX_BATCH}-CV batch, unlocked on this machine.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        with st.form("signup_form", clear_on_submit=False):
            email = st.text_input(
                "Work email", placeholder="you@company.com",
                help="Stored only in a file beside the app, so the trial is remembered "
                     "after a restart. It is not sent anywhere.",
            )
            org = st.text_input("Company (optional)", placeholder="e.g. Sangath Constructions")
            agreed = st.checkbox("I agree to the terms of use and the privacy note")
            submitted = st.form_submit_button(
                "Create my free account", use_container_width=True, type="primary")

        if submitted:
            if not EMAIL_RE.match(email or ""):
                st.error("That does not look like an email address — check for a typo.")
            elif not agreed:
                st.warning("Tick the terms box to continue.")
            else:
                st.session_state.account     = save_account(email, org)
                st.session_state.show_signup = False
                st.success(f"Welcome, {email}. The full {MAX_BATCH}-CV batch is unlocked.")
                st.rerun()

        st.markdown('<div class="or-rule"><span>or</span></div>', unsafe_allow_html=True)
        if st.button("Continue with Google", key="google_signup", use_container_width=True):
            st.info(
                "**Google sign-in is not wired up.** It needs an OAuth client ID and "
                "secret from the Google Cloud console, plus a server to receive the "
                "redirect — this app deliberately has neither, so nothing leaves your "
                "machine. The button is here so the flow is ready when you add a backend."
            )
        st.caption(
            "Prefer not to register? The tool works without an account — "
            f"batches are limited to {FREE_BATCH} CVs."
        )
        st.button("Continue without an account", key="guest_btn",
                  use_container_width=True, on_click=start_guest)

    with right:
        st.markdown(f"""
<div class="plan-card">
  <div class="plan-badge">Free trial</div>
  <div class="plan-price">₹0<span>/ forever on this machine</span></div>
  <ul class="plan-list">
    <li><b>{MAX_BATCH} CVs</b> per batch instead of {FREE_BATCH}</li>
    <li>Ranked shortlist with a written verdict per candidate</li>
    <li>Score breakdown, shortlist mix and skill-demand charts</li>
    <li>CSV export of the shortlist</li>
    <li>Unlimited candidate CV checks</li>
  </ul>
  <div class="plan-note">🔒 No card. No licence server. Your CVs and this email
     stay on this computer — the app makes no outbound calls.</div>
</div>
""", unsafe_allow_html=True)


HERO_HTML = """
<span class="anchor" id="top"></span>
<div class="hero">
  <div class="hero-inner">
    <div class="hero-badge"><span class="pulse-dot"></span> Runs 100% on your computer — no resume ever leaves the office</div>
    <h1>Shortlist the right hire,<br><span class="grad">not the longest resume.</span></h1>
    <p class="sub">
      Drop in a stack of CVs and a job description. Every candidate is read, scored
      against your actual requirements, and ranked — with a written reason for each verdict.
    </p>
    <div class="hero-stats">
      <div class="hstat"><b>100</b><span>resumes per batch</span></div>
      <div class="hstat"><b>4</b><span>scoring signals</span></div>
      <div class="hstat"><b>Civil→Sales</b><span>any role, not just tech</span></div>
      <div class="hstat"><b>1&nbsp;click</b><span>export to CSV</span></div>
    </div>
  </div>
</div>
"""

DEFAULT_JD = """Looking for a Senior Python Developer with 3+ years of experience.

Requirements:
- Strong Python skills with Django or Flask
- REST API development
- SQL database experience (PostgreSQL/MySQL)
- AWS or cloud platform experience
- Docker and CI/CD knowledge
- Machine Learning experience is a plus
- Location: Bangalore or Remote"""

SCORE_FORMULA_HTML = """
<div style="font-size:0.8rem;font-weight:700;color:#1e293b;margin-bottom:0.5rem">
  How the score is built
</div>
<div class="formula-row"><span>🔵 Skills match</span><i>50%</i></div>
<div class="formula-row"><span>🟢 Relevant experience</span><i>25%</i></div>
<div class="formula-row"><span>🔴 Location fit</span><i>15%</i></div>
<div class="formula-row"><span>🟡 Education</span><i>10%</i></div>
<div style="font-size:0.72rem;color:#94a3b8;margin-top:0.7rem;line-height:1.5">
  Experience is weighted by <b>relevance</b> — 10 unrelated years won't outrank
  3 years in the actual field.
</div>
"""


# ─────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────

if PAGE == "Home":
    st.markdown(HERO_HTML, unsafe_allow_html=True)

    h1, h2, _sp = st.columns([1.15, 1.15, 3])
    with h1:
        if st.session_state.get("account"):
            st.button("🚀 Screen resumes now", key="hero_cta", use_container_width=True,
                      type="primary", on_click=go, args=(CTA_TAB,))
        else:
            st.button("✨ Try for free", key="hero_cta", use_container_width=True,
                      type="primary", on_click=open_signup)
    with h2:
        st.button("See how it works", key="hero_how", use_container_width=True,
                  on_click=go, args=("How it works",))
    if not st.session_state.get("account"):
        st.caption(f"No card needed · unlocks the full {MAX_BATCH}-CV batch · "
                   "nothing leaves this machine")

    # ── Audience split — the two sides of the same engine
    st.markdown('<div class="section-title">Which side of the desk are you on?</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">The same scoring engine, pointed in two directions — '
        'one shortlist for the person hiring, one honest report for the person applying.</div>',
        unsafe_allow_html=True,
    )
    a_col, b_col = st.columns(2, gap="medium")
    with a_col:
        st.markdown(f"""
<div class="aud aud-r">
  <div class="tagline">For recruiters &amp; hiring managers</div>
  <h4>🚀 Screen a whole stack at once</h4>
  <p>Paste the vacancy, drop in up to {MAX_BATCH} CVs, and get a ranked shortlist with a
     written verdict beside every name.</p>
  <ul>
    <li>Up to <b>{MAX_BATCH} resumes</b> in a single batch</li>
    <li>Ranked table + score breakdown per candidate</li>
    <li>Missing-skill gaps shown before you interview</li>
    <li>Filter by minimum score, export the shortlist to CSV</li>
  </ul>
</div>
""", unsafe_allow_html=True)
        if st.session_state.get("account"):
            st.button(f"🚀 {CTA_TAB}", key="aud_rec", use_container_width=True,
                      type="primary", on_click=go, args=(CTA_TAB,))
        else:
            st.button("✨ Try for free", key="aud_rec", use_container_width=True,
                      type="primary", on_click=open_signup)
    with b_col:
        st.markdown("""
<div class="aud aud-c">
  <div class="tagline">For candidates</div>
  <h4>📄 Check one CV against one role</h4>
  <p>Paste the advert you are about to apply to, upload your CV, and see the same score
     the recruiter's side would give you — before you hit send.</p>
  <ul>
    <li><b>One CV at a time</b>, scored against one advert</li>
    <li>Exactly which required skills your CV never mentions</li>
    <li>Plain-English advice on what to change first</li>
    <li>ATS readiness check — will a machine read it at all?</li>
  </ul>
</div>
""", unsafe_allow_html=True)
        st.button(f"📄 {CAND_TAB}", key="aud_cand", use_container_width=True,
                  on_click=go, args=(CAND_TAB,))

    # ── Illustration 1: what the scanner sees
    st.markdown('<div class="section-title">Inside the scan</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Every page is read line by line, then the fields that '
        'matter for hiring are lifted out — no manual data entry.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
<div class="viz-grid">
  <div class="viz-card">
    <h4>📄 Reading the resume</h4>
    <p class="viz-sub">Text layer extracted from PDF or DOCX, then swept for the fields below.</p>
    <svg class="viz-svg" viewBox="0 0 340 300" role="img" aria-label="Animated illustration of a resume being scanned">
      <defs>
        <linearGradient id="rrBeam" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"   stop-color="#38bdf8" stop-opacity="0"/>
          <stop offset="55%"  stop-color="#38bdf8" stop-opacity="0.45"/>
          <stop offset="100%" stop-color="#38bdf8" stop-opacity="0"/>
        </linearGradient>
      </defs>
      <rect x="58" y="6" width="224" height="288" rx="13" fill="#ffffff" stroke="#e2e8f0"/>
      <rect x="78" y="26" width="104" height="13" rx="6" fill="#0f172a" opacity="0.82"/>
      <rect x="78" y="47" width="148" height="7"  rx="3.5" fill="#cbd5e1"/>
      <rect x="78" y="72" width="58"  height="7"  rx="3.5" fill="#93c5fd"/>
      <rect x="78" y="87" width="132" height="6"  rx="3" fill="#e8edf3"/>
      <rect x="78" y="99" width="104" height="6"  rx="3" fill="#e8edf3"/>
      <rect x="78" y="124" width="44" height="7"  rx="3.5" fill="#93c5fd"/>
      <rect x="78" y="139" width="42" height="14" rx="7" fill="#eff6ff" stroke="#bfdbfe"/>
      <rect x="126" y="139" width="52" height="14" rx="7" fill="#eff6ff" stroke="#bfdbfe"/>
      <rect x="184" y="139" width="36" height="14" rx="7" fill="#eff6ff" stroke="#bfdbfe"/>
      <rect x="78" y="159" width="60" height="14" rx="7" fill="#eff6ff" stroke="#bfdbfe"/>
      <rect x="144" y="159" width="46" height="14" rx="7" fill="#eff6ff" stroke="#bfdbfe"/>
      <rect x="78" y="192" width="70" height="7"  rx="3.5" fill="#93c5fd"/>
      <rect x="78" y="207" width="150" height="6" rx="3" fill="#e8edf3"/>
      <rect x="78" y="219" width="126" height="6" rx="3" fill="#e8edf3"/>
      <rect x="78" y="231" width="140" height="6" rx="3" fill="#e8edf3"/>
      <rect x="78" y="256" width="52" height="7"  rx="3.5" fill="#93c5fd"/>
      <rect x="78" y="271" width="118" height="6" rx="3" fill="#e8edf3"/>
      <g class="rr-hl">
        <rect x="72" y="20" width="118" height="25" rx="7" fill="none" stroke="#22c55e" stroke-width="1.6" stroke-dasharray="4 3"/>
      </g>
      <g class="rr-hl d2">
        <rect x="72" y="81" width="146" height="30" rx="7" fill="none" stroke="#22c55e" stroke-width="1.6" stroke-dasharray="4 3"/>
      </g>
      <g class="rr-hl d3">
        <rect x="72" y="133" width="156" height="46" rx="7" fill="none" stroke="#22c55e" stroke-width="1.6" stroke-dasharray="4 3"/>
      </g>
      <g class="rr-hl d4">
        <rect x="72" y="201" width="164" height="42" rx="7" fill="none" stroke="#22c55e" stroke-width="1.6" stroke-dasharray="4 3"/>
      </g>
      <g class="rr-beam">
        <rect x="58" y="6" width="224" height="34" fill="url(#rrBeam)"/>
        <rect x="58" y="39" width="224" height="1.8" fill="#38bdf8" opacity="0.95"/>
      </g>
    </svg>
  </div>
  <div class="viz-card">
    <h4>✅ What comes out</h4>
    <p class="viz-sub">Structured, comparable data for every single applicant in the batch.</p>
    <div class="efield"><span class="ek">Name</span><span class="ev">Ravi Sharma</span><span class="ok">✓</span></div>
    <div class="efield"><span class="ek">Email</span><span class="ev">ravi.sharma@email.com</span><span class="ok">✓</span></div>
    <div class="efield"><span class="ek">Phone</span><span class="ev">+91 98xxx xxxxx</span><span class="ok">✓</span></div>
    <div class="efield"><span class="ek">Location</span><span class="ev">Bangalore</span><span class="ok">✓</span></div>
    <div class="efield"><span class="ek">Experience</span><span class="ev">5.5 years</span><span class="ok">✓</span></div>
    <div class="efield"><span class="ek">Education</span><span class="ev">B.Tech</span><span class="ok">✓</span></div>
    <div class="efield"><span class="ek">Skills</span><span class="ev">18 detected</span><span class="ok">✓</span></div>
    <p class="viz-sub" style="margin:0.9rem 0 0">
      Scanned image-only PDFs have no text layer — those are flagged in the results
      instead of being quietly scored as zero.
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Illustration 2: the pipeline
    st.markdown('<div class="section-title">From a folder of CVs to a ranked shortlist</div>',
                unsafe_allow_html=True)
    st.markdown("""
<div class="viz-dark">
  <h4>🔬 The screening pipeline</h4>
  <p class="viz-sub">Six stages, run locally, in one pass over the whole batch.</p>
  <svg class="viz-svg" viewBox="0 0 1180 150" role="img" aria-label="Diagram of the six-stage screening pipeline">
    <defs>
      <marker id="rrArrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
        <path d="M0 0 L10 5 L0 10 z" fill="#38bdf8"/>
      </marker>
    </defs>
    <g fill="none" stroke="#38bdf8" stroke-width="1.6" marker-end="url(#rrArrow)" opacity="0.75">
      <path d="M184 74 H 208"/><path d="M379 74 H 403"/><path d="M574 74 H 598"/>
      <path d="M769 74 H 793"/><path d="M964 74 H 988"/>
    </g>
    <g font-family="Inter, sans-serif">
      <g><rect x="20" y="30" width="164" height="88" rx="14" fill="rgba(255,255,255,0.055)" stroke="rgba(255,255,255,0.16)"/>
        <text x="102" y="60" text-anchor="middle" font-size="21">📤</text>
        <text x="102" y="86" text-anchor="middle" font-size="13" font-weight="600" fill="#e2e8f0">Upload</text>
        <text x="102" y="103" text-anchor="middle" font-size="10.5" fill="#7c8ba1">PDF · DOCX · batch</text></g>
      <g><rect x="215" y="30" width="164" height="88" rx="14" fill="rgba(255,255,255,0.055)" stroke="rgba(255,255,255,0.16)"/>
        <text x="297" y="60" text-anchor="middle" font-size="21">📄</text>
        <text x="297" y="86" text-anchor="middle" font-size="13" font-weight="600" fill="#e2e8f0">Read</text>
        <text x="297" y="103" text-anchor="middle" font-size="10.5" fill="#7c8ba1">text layer per page</text></g>
      <g><rect x="410" y="30" width="164" height="88" rx="14" fill="rgba(255,255,255,0.055)" stroke="rgba(255,255,255,0.16)"/>
        <text x="492" y="60" text-anchor="middle" font-size="21">🧩</text>
        <text x="492" y="86" text-anchor="middle" font-size="13" font-weight="600" fill="#e2e8f0">Extract</text>
        <text x="492" y="103" text-anchor="middle" font-size="10.5" fill="#7c8ba1">name · contact · skills</text></g>
      <g><rect x="605" y="30" width="164" height="88" rx="14" fill="rgba(255,255,255,0.055)" stroke="rgba(255,255,255,0.16)"/>
        <text x="687" y="60" text-anchor="middle" font-size="21">🎯</text>
        <text x="687" y="86" text-anchor="middle" font-size="13" font-weight="600" fill="#e2e8f0">Match</text>
        <text x="687" y="103" text-anchor="middle" font-size="10.5" fill="#7c8ba1">against the JD</text></g>
      <g><rect x="800" y="30" width="164" height="88" rx="14" fill="rgba(255,255,255,0.055)" stroke="rgba(255,255,255,0.16)"/>
        <text x="882" y="60" text-anchor="middle" font-size="21">📊</text>
        <text x="882" y="86" text-anchor="middle" font-size="13" font-weight="600" fill="#e2e8f0">Score</text>
        <text x="882" y="103" text-anchor="middle" font-size="10.5" fill="#7c8ba1">4 weighted signals</text></g>
      <g><rect x="995" y="30" width="164" height="88" rx="14" fill="rgba(56,189,248,0.13)" stroke="rgba(56,189,248,0.45)"/>
        <text x="1077" y="60" text-anchor="middle" font-size="21">🏆</text>
        <text x="1077" y="86" text-anchor="middle" font-size="13" font-weight="600" fill="#e2e8f0">Rank &amp; export</text>
        <text x="1077" y="103" text-anchor="middle" font-size="10.5" fill="#7c8ba1">shortlist · CSV</text></g>
    </g>
  </svg>
</div>
""", unsafe_allow_html=True)

    # ── Illustration 3 + 4: matching and the shortlist
    st.markdown('<div class="section-title">How a candidate is scored</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="viz-grid">
  <div class="viz-card">
    <h4>🎯 Requirements vs. the resume</h4>
    <p class="viz-sub">Each requirement in the job description is matched against what the CV actually shows.</p>
    <svg class="viz-svg" viewBox="0 0 460 250" role="img" aria-label="Diagram matching job requirements to resume skills">
      <text x="16" y="18" font-size="11" font-weight="700" fill="#64748b" font-family="Inter, sans-serif">JOB DESCRIPTION</text>
      <text x="444" y="18" text-anchor="end" font-size="11" font-weight="700" fill="#64748b" font-family="Inter, sans-serif">RESUME</text>
      <g font-family="Inter, sans-serif" font-size="11.5">
        <g><rect x="10" y="32" width="132" height="30" rx="9" fill="#eff6ff" stroke="#bfdbfe"/><text x="76" y="51" text-anchor="middle" fill="#1d4ed8">Python</text></g>
        <g><rect x="10" y="74" width="132" height="30" rx="9" fill="#eff6ff" stroke="#bfdbfe"/><text x="76" y="93" text-anchor="middle" fill="#1d4ed8">Django</text></g>
        <g><rect x="10" y="116" width="132" height="30" rx="9" fill="#eff6ff" stroke="#bfdbfe"/><text x="76" y="135" text-anchor="middle" fill="#1d4ed8">REST APIs</text></g>
        <g><rect x="10" y="158" width="132" height="30" rx="9" fill="#eff6ff" stroke="#bfdbfe"/><text x="76" y="177" text-anchor="middle" fill="#1d4ed8">PostgreSQL</text></g>
        <g><rect x="10" y="200" width="132" height="30" rx="9" fill="#eff6ff" stroke="#bfdbfe"/><text x="76" y="219" text-anchor="middle" fill="#1d4ed8">Docker</text></g>
        <g><rect x="318" y="32" width="132" height="30" rx="9" fill="#f0fdf4" stroke="#bbf7d0"/><text x="384" y="51" text-anchor="middle" fill="#166534">Python</text></g>
        <g><rect x="318" y="74" width="132" height="30" rx="9" fill="#f0fdf4" stroke="#bbf7d0"/><text x="384" y="93" text-anchor="middle" fill="#166534">Django</text></g>
        <g><rect x="318" y="116" width="132" height="30" rx="9" fill="#f0fdf4" stroke="#bbf7d0"/><text x="384" y="135" text-anchor="middle" fill="#166534">REST APIs</text></g>
        <g><rect x="318" y="158" width="132" height="30" rx="9" fill="#f0fdf4" stroke="#bbf7d0"/><text x="384" y="177" text-anchor="middle" fill="#166534">MySQL</text></g>
        <g><rect x="318" y="200" width="132" height="30" rx="9" fill="#fff1f2" stroke="#fecdd3" stroke-dasharray="4 3"/><text x="384" y="219" text-anchor="middle" fill="#be123c">not found</text></g>
      </g>
      <g fill="none" stroke-width="2">
        <path d="M142 47  C 210 47, 250 47, 318 47"  stroke="#22c55e"/>
        <path d="M142 89  C 210 89, 250 89, 318 89"  stroke="#22c55e"/>
        <path d="M142 131 C 210 131, 250 131, 318 131" stroke="#22c55e"/>
        <path d="M142 173 C 210 173, 250 173, 318 173" stroke="#f59e0b" stroke-dasharray="5 4"/>
        <path d="M142 215 C 210 215, 250 215, 318 215" stroke="#f43f5e" stroke-dasharray="5 4"/>
      </g>
    </svg>
    <div class="viz-legend">
      <span><i style="background:#22c55e"></i>Direct match</span>
      <span><i style="background:#f59e0b"></i>Related / partial</span>
      <span><i style="background:#f43f5e"></i>Gap</span>
    </div>
  </div>
  <div class="viz-card">
    <h4>🏆 The shortlist you get back</h4>
    <p class="viz-sub">Ordered 0–100, each with matched skills, gaps and a written verdict.</p>
    <svg class="viz-svg" viewBox="0 0 460 250" role="img" aria-label="Ranked candidate leaderboard with score bars">
      <g font-family="Inter, sans-serif" font-size="12">
        <text x="6" y="30" font-size="16">🥇</text>
        <text x="30" y="30" fill="#0f172a" font-weight="600">Ravi Sharma</text>
        <rect x="150" y="19" width="250" height="15" rx="7.5" fill="#f1f5f9"/>
        <rect x="150" y="19" width="224" height="15" rx="7.5" fill="#16a34a"/>
        <text x="410" y="31" fill="#166534" font-weight="700" font-family="JetBrains Mono, monospace" font-size="11.5">89.7</text>
        <text x="30" y="48" fill="#64748b" font-size="10.5">Excellent fit · 5.5 yrs · Bangalore</text>
        <text x="6" y="92" font-size="16">🥈</text>
        <text x="30" y="92" fill="#0f172a" font-weight="600">Karan Singh</text>
        <rect x="150" y="81" width="250" height="15" rx="7.5" fill="#f1f5f9"/>
        <rect x="150" y="81" width="196" height="15" rx="7.5" fill="#22c55e"/>
        <text x="410" y="93" fill="#166534" font-weight="700" font-family="JetBrains Mono, monospace" font-size="11.5">78.6</text>
        <text x="30" y="110" fill="#64748b" font-size="10.5">Strong fit · 4 yrs · Remote</text>
        <text x="6" y="154" font-size="16">🥉</text>
        <text x="30" y="154" fill="#0f172a" font-weight="600">Anita Desai</text>
        <rect x="150" y="143" width="250" height="15" rx="7.5" fill="#f1f5f9"/>
        <rect x="150" y="143" width="172" height="15" rx="7.5" fill="#eab308"/>
        <text x="410" y="155" fill="#854d0e" font-weight="700" font-family="JetBrains Mono, monospace" font-size="11.5">68.8</text>
        <text x="30" y="172" fill="#64748b" font-size="10.5">Worth a call · 3 yrs · Pune</text>
        <text x="8" y="216" font-size="14">▫️</text>
        <text x="30" y="216" fill="#0f172a" font-weight="600">Mohit Verma</text>
        <rect x="150" y="205" width="250" height="15" rx="7.5" fill="#f1f5f9"/>
        <rect x="150" y="205" width="34" height="15" rx="7.5" fill="#f43f5e"/>
        <text x="410" y="217" fill="#991b1b" font-weight="700" font-family="JetBrains Mono, monospace" font-size="11.5">13.4</text>
        <text x="30" y="234" fill="#64748b" font-size="10.5">Different field · marketing background</text>
      </g>
    </svg>
    <div class="viz-legend">
      <span>Ten unrelated years will not outrank three in the actual field.</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── The steps, spelled out
    st.markdown('<div class="section-title">The steps we run on every resume</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="proc-grid">
  <div class="proc" data-n="01"><h4>📤 Collect</h4>
    <p>Up to 100 PDF or DOCX files in one batch, parsed in parallel with a live progress bar.</p></div>
  <div class="proc" data-n="02"><h4>📄 Read</h4>
    <p>The text layer is pulled from every page. Image-only scans are flagged, never silently zeroed.</p></div>
  <div class="proc" data-n="03"><h4>🧩 Extract</h4>
    <p>Name, email, phone, location, years of experience, education and the full skill list.</p></div>
  <div class="proc" data-n="04"><h4>🎯 Match</h4>
    <p>Skills are compared with the requirements detected in your job description — matched and missing.</p></div>
  <div class="proc" data-n="05"><h4>📊 Score</h4>
    <p>Skills 50%, relevant experience 25%, location 15%, education 10% — combined into one 0–100 score.</p></div>
  <div class="proc" data-n="06"><h4>🏆 Rank &amp; export</h4>
    <p>Ordered shortlist with a plain-English verdict per candidate, exportable to CSV in one click.</p></div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Built for real hiring desks</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="trust">
  <div><b>100%</b><span>Runs offline</span></div>
  <div><b>6</b><span>Departments covered</span></div>
  <div><b>4</b><span>Scoring signals</span></div>
  <div><b>0</b><span>Candidates auto-rejected</span></div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="cta-band">
  <h3>Ready to rank your current stack of CVs?</h3>
  <p>Open the screening page, paste the role you are hiring for, and drop the resumes in.
     Nothing is uploaded anywhere — it all runs right here on this machine.</p>
</div>
""", unsafe_allow_html=True)
    b1, _b2 = st.columns([1.3, 3])
    with b1:
        st.markdown('<span class="cta-btn-marker"></span>', unsafe_allow_html=True)
        st.button("🚀 Start screening", key="home_bottom_cta", use_container_width=True,
                  type="primary", on_click=go, args=(CTA_TAB,))


# ─────────────────────────────────────────────
# PAGE: CHECK MY CV — one candidate, one role
# ─────────────────────────────────────────────

if PAGE == CAND_TAB:
    st.markdown(
        '<span class="anchor" id="candidate"></span>'
        '<div class="section-title">Check my CV</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">Paste the job you are applying for and upload your CV. '
        'You get the same score a recruiter running this tool would see — and, unlike them, '
        'the list of what is missing.</div>',
        unsafe_allow_html=True,
    )

    c_left, c_right = st.columns([1.6, 1])
    with c_left:
        st.markdown('<div class="step-label">1 · The role you are applying for</div>',
                    unsafe_allow_html=True)
        cand_jd = st.text_area(
            "Target role",
            value=(st.session_state["keep_cand_jd"]
                   if st.session_state["keep_cand_jd"] is not None else DEFAULT_JD),
            height=210, key="cand_jd", label_visibility="collapsed",
            help="Paste the advert or job description you are targeting.",
        )
        st.session_state["keep_cand_jd"] = cand_jd

        st.markdown('<div class="step-label">2 · Your CV</div>', unsafe_allow_html=True)
        cand_file = st.file_uploader(
            "Your CV", type=["pdf", "docx"], accept_multiple_files=False,
            key="cand_file", label_visibility="collapsed",
            help="One file. Nothing leaves this machine.",
        )
        st.caption("One CV at a time. Screening a whole stack? Use **Start screening** instead.")

    with c_right:
        with st.container(border=True):
            if cand_jd.strip():
                _cjd = cached_parse_jd(cand_jd)
                st.markdown("**🔍 What this role asks for**")
                _cs = _cjd.get("skills", [])
                if _cs:
                    st.markdown(render_skills_html(_cs), unsafe_allow_html=True)
                else:
                    st.caption("No specific skills detected — add the requirements list.")
                st.markdown(
                    f"**Experience:** {_cjd.get('experience_min', 0):.0f}+ yrs  \n"
                    f"**Location:** {_cjd.get('location') or 'Not specified'}"
                )
            else:
                st.info("Paste a job description to see what it asks for.")

    if cand_file is not None and cand_jd.strip():
        if st.button("🔍 Check my fit", type="primary", use_container_width=True):
            with st.spinner("Reading your CV…"):
                cand_file.seek(0)
                st.session_state.cand_parsed = parse_resume(cand_file.read(), cand_file.name)

    _cp = st.session_state.get("cand_parsed")
    if _cp and cand_jd.strip():
        st.markdown("---")
        if _cp.get("error"):
            st.error(f"**Could not read that file.** {_cp['error']}")
        else:
            me = rank_candidates([_cp], cached_parse_jd(cand_jd))[0]
            bd = me.get("score_breakdown", {})
            assess  = me.get("assessment", {})
            matched = me.get("matched_skills", [])
            missing = me.get("missing_skills", [])

            st.markdown('<div class="section-title">Your fit for this role</div>',
                        unsafe_allow_html=True)
            r1, r2 = st.columns([1, 1.9])
            with r1:
                with st.container(border=True):
                    st.markdown(
                        score_ring_html(me.get("final_score", 0),
                                        assess.get("label", ""),
                                        f"{len(matched)} of {len(matched)+len(missing)} "
                                        "required skills found"),
                        unsafe_allow_html=True,
                    )
            with r2:
                with st.container(border=True):
                    st.markdown("**Where the score comes from**")
                    st.markdown(
                        bar_row_html("Skills match",        bd.get("skill_score", 0), "50%")
                        + bar_row_html("Relevant experience", bd.get("exp_score", 0), "25%")
                        + bar_row_html("Location fit",        bd.get("loc_score", 0), "15%")
                        + bar_row_html("Education",           bd.get("edu_score", 0), "10%"),
                        unsafe_allow_html=True,
                    )

            if assess.get("summary"):
                tag = assess.get("tag", "ok")
                {"good": st.success, "ok": st.info,
                 "warn": st.warning}.get(tag, st.error)(f"**{assess['summary']}**")

            g1, g2 = st.columns(2)
            with g1:
                with st.container(border=True):
                    st.markdown("**✅ You already match**")
                    if matched:
                        st.markdown(render_skills_html(matched, "skill-chip skill-chip-matched"),
                                    unsafe_allow_html=True)
                    else:
                        st.caption("None of the listed requirements were found in your CV.")
            with g2:
                with st.container(border=True):
                    st.markdown("**❌ The role asks for, your CV does not show**")
                    if missing:
                        st.markdown(render_skills_html(missing, "skill-chip skill-chip-missing"),
                                    unsafe_allow_html=True)
                    else:
                        st.caption("Nothing missing — your CV covers every stated requirement.")

            # ── The part a recruiter's report never gives you
            st.markdown('<div class="section-title">How to raise this score</div>',
                        unsafe_allow_html=True)
            advice = []
            if missing:
                advice.append(
                    f"<b>Name the {len(missing)} missing skill"
                    f"{'s' if len(missing) > 1 else ''} explicitly.</b> Matching is done on "
                    "words, so a tool you have used but never wrote down scores zero. If you "
                    f"have genuinely used {missing[0]}, add it to your Skills section and to "
                    "the bullet point of the project where you used it."
                )
            if bd.get("exp_score", 0) < 70:
                advice.append(
                    "<b>Make the relevant years obvious.</b> Experience is weighted by how "
                    "closely your history matches this field, so put the roles that match "
                    "this advert first and give each one clear start and end dates."
                )
            if bd.get("loc_score", 100) < 60:
                advice.append(
                    "<b>State the city this role is in.</b> If you are willing to relocate or "
                    "work remotely, say so in one line near the top, or you will be filtered out."
                )
            if bd.get("edu_score", 0) < 60:
                advice.append(
                    "<b>Spell out your qualification</b> in full, including the field and the "
                    "year, rather than only an abbreviation."
                )
            if not advice:
                advice.append(
                    "<b>Nothing structural is holding this CV back for this role.</b> Focus on "
                    "the wording of your achievement bullets — numbers and outcomes."
                )
            for a in advice:
                st.markdown(f'<div class="advice">{a}</div>', unsafe_allow_html=True)

            # ── Unique: readability, judged separately from fit
            st.markdown('<div class="section-title">Will a machine read it correctly?</div>',
                        unsafe_allow_html=True)
            st.markdown(
                '<div class="section-sub">A CV can fit a role perfectly and still be discarded '
                'for being unreadable. These are checks on the file itself, not on your '
                'suitability.</div>',
                unsafe_allow_html=True,
            )
            checks = ats_checks(_cp)
            passed = sum(1 for ok, _, _ in checks if ok)
            st.progress(passed / len(checks), text=f"{passed} of {len(checks)} checks passed")
            a1, a2 = st.columns(2)
            for _i, (ok, title, fix) in enumerate(checks):
                with (a1 if _i % 2 == 0 else a2):
                    if ok:
                        st.markdown(
                            f'<div class="ats-row ats-ok"><b>✓ {title}</b></div>',
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f'<div class="ats-row ats-bad"><b>✕ {title}</b><span>{fix}</span></div>',
                            unsafe_allow_html=True)

            with st.expander("📄 What the parser actually read from your file"):
                st.text((_cp.get("raw_text") or "")[:3000] or "Nothing could be extracted.")

    elif PAGE == CAND_TAB and not cand_jd.strip():
        st.warning("☝️ Paste the job description you are targeting to begin.")
    elif PAGE == CAND_TAB and cand_file is None:
        st.info("☝️ Upload your CV above to see your fit for this role.")


# ─────────────────────────────────────────────
# PAGE: START SCREENING — job description, settings, upload
# ─────────────────────────────────────────────

jd_text        = ""
uploaded_files = []
batch_files    = []
min_score      = 0
show_raw_text  = False
show_breakdown = True

# The sign-up panel replaces the screening page rather than sitting on top of it. A
# half-visible upload form behind a registration wall reads as broken rather than optional,
# and there is a "continue without an account" button on the panel itself.
SIGNUP_GATE = (PAGE == CTA_TAB
               and st.session_state.get("show_signup")
               and not st.session_state.get("account"))

if SIGNUP_GATE:
    render_signup()

if PAGE == CTA_TAB and not SIGNUP_GATE:
    st.markdown(
        '<span class="anchor" id="upload"></span>'
        '<div class="section-title">Start screening</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">Everything for a screening run lives on this page — '
        'describe the role, set your filters, drop in the CVs, and rank.</div>',
        unsafe_allow_html=True,
    )

    # Who is at the desk, and what that buys them.
    _acct = st.session_state.get("account")
    if _acct:
        _mail = _acct.get("email", "")
        _chip, _out = st.columns([3.2, 1])
        with _chip:
            st.markdown(
                f'<div class="acct"><span class="av">{(_mail[:1] or "?").upper()}</span>'
                f'<span class="who">{_mail}'
                f'{(" · " + _acct["org"]) if _acct.get("org") else ""}</span>'
                f'<span class="pill">{_acct.get("plan", "Free trial")} · '
                f'{MAX_BATCH} CVs / batch</span></div>',
                unsafe_allow_html=True,
            )
        with _out:
            # Shared machines are normal on a hiring desk, so the account has to be
            # removable without hunting for the JSON file.
            if st.button("Sign out", key="signout_btn", use_container_width=True):
                st.session_state.account = None
                ACCOUNT_FILE.unlink(missing_ok=True)
                st.rerun()
    else:
        st.markdown(
            '<div class="lockbox">'
            f'<b class="lb-h">🔓 You are screening as a guest — {FREE_BATCH} CVs per batch</b>'
            f'<p>Everything works; only the batch size is trimmed. A free account lifts it to '
            f'{MAX_BATCH} CVs in one run. No card, no licence server, nothing leaves this '
            'machine.</p></div>',
            unsafe_allow_html=True,
        )
        st.button("✨ Try for free", key="gate_cta", type="primary", on_click=open_signup)

    # ── Step 1 — job description
    st.markdown('<div class="section-title">1 · Job description</div>', unsafe_allow_html=True)
    col_jd, col_jd_preview = st.columns([1.6, 1])

    with col_jd:
        st.caption("Paste the role below — skills, experience and location are detected automatically.")
        jd_text = st.text_area(
            "Job Description",
            value=(st.session_state["keep_jd_text"]
                   if st.session_state["keep_jd_text"] is not None else DEFAULT_JD),
            height=260,
            key="jd_text",
            help="Paste the complete job description here.",
            label_visibility="collapsed",
        )
        st.session_state["keep_jd_text"] = jd_text

    with col_jd_preview:
        with st.container(border=True):
            st.markdown("**🔍 JD Analysis**")
            if jd_text.strip():
                jd = cached_parse_jd(jd_text)
                st.markdown(f"**Skills detected:** {len(jd['skills'])}")
                if jd["skills"]:
                    st.markdown(render_skills_html(jd["skills"][:12]), unsafe_allow_html=True)
                exp_min, exp_max = jd.get("experience_min"), jd.get("experience_max")
                if exp_min is None and exp_max is None:
                    exp_label = "Any"
                elif exp_max is None:
                    exp_label = f"{exp_min:g}+ yrs"
                elif exp_min == exp_max:
                    exp_label = f"{exp_min:g} yrs"
                else:
                    exp_label = f"{exp_min:g}–{exp_max:g} yrs"
                st.markdown(f"**Experience required:** {exp_label}")
                st.markdown(f"**Location:** {jd['location']}")
            else:
                st.info("Enter a job description to see what was detected.")

    # ── Step 2 — settings (previously the left sidebar)
    st.markdown('<div class="section-title">2 · Settings</div>', unsafe_allow_html=True)
    with st.container(border=True):
        set_a, set_b = st.columns([1.6, 1])
        with set_a:
            min_score = st.slider(
                "Minimum score filter",
                min_value=0, max_value=100, step=5,
                value=st.session_state["keep_min_score"], key="min_score",
                help="Candidates below this score are hidden from the shortlist — nobody is deleted.",
            )
            show_breakdown = st.checkbox(
                "Show score breakdown",
                value=st.session_state["keep_show_breakdown"], key="show_breakdown")
            show_raw_text = st.checkbox(
                "Show extracted raw text",
                value=st.session_state["keep_show_raw_text"], key="show_raw_text")
            st.session_state["keep_min_score"]      = min_score
            st.session_state["keep_show_breakdown"] = show_breakdown
            st.session_state["keep_show_raw_text"]  = show_raw_text
            st.caption("Filters apply instantly to an existing shortlist — no need to re-run the scan.")
        with set_b:
            st.markdown(SCORE_FORMULA_HTML, unsafe_allow_html=True)

    # ── Step 3 — upload
    st.markdown('<div class="section-title">3 · Upload resumes</div>', unsafe_allow_html=True)
    CAP = batch_limit()
    st.caption(f"PDF or DOCX · up to **{CAP} CVs** in one batch · nothing leaves this machine")
    uploaded_files = st.file_uploader(
        f"Upload PDF or DOCX resumes (up to {CAP} files)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help=f"Select up to {CAP} files at once using Shift+Click or Ctrl+Click",
        label_visibility="collapsed",
    )

    batch_files = []
    if uploaded_files:
        valid   = [f for f in uploaded_files if is_valid_resume_file(f.name)]
        invalid = [f for f in uploaded_files if not is_valid_resume_file(f.name)]

        # The cap is enforced here rather than at parse time so the reader is told which
        # CVs were dropped before they commit to a run, not after waiting through one.
        batch_files = valid[:CAP]
        overflow    = valid[CAP:]

        if invalid:
            st.warning(
                f"⚠️ {len(invalid)} file(s) skipped — unsupported format: "
                f"{', '.join(f.name for f in invalid)}"
            )
        if overflow:
            st.error(
                f"🚫 **Batch limit reached.** The first {CAP} CVs are queued; "
                f"{len(overflow)} were left out — screen those in a second batch."
            )
            # Guests hit this wall for a reason they can fix in one click; account holders
            # have hit the real ceiling, so pitching an upgrade at them would be nonsense.
            if not st.session_state.get("account"):
                st.info(f"A free account raises this batch from {FREE_BATCH} to {MAX_BATCH} CVs.")
                st.button("✨ Try for free", key="overflow_cta", type="primary",
                          on_click=open_signup)

        if batch_files:
            st.success(f"✅ {len(batch_files)} valid resume(s) queued")
            st.markdown(batch_meter_html(len(batch_files), CAP), unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PROCESS & RANK
# ─────────────────────────────────────────────

if PAGE == CTA_TAB and jd_text.strip() and (uploaded_files or st.session_state.get("parsed")):
    valid_files = batch_files

    if valid_files:
        st.markdown('<div class="section-title">4 · Rank</div>', unsafe_allow_html=True)
        if st.button("🚀 Analyze & Rank Candidates", use_container_width=True, type="primary"):
            # ── Parse resumes (the whole block is cleared once parsing finishes)
            proc_heading = st.empty()
            proc_heading.markdown('<div class="section-title">⚙️ Processing</div>', unsafe_allow_html=True)
            progress_bar = st.progress(0)
            status_text  = st.empty()

            with st.spinner(""):
                st.session_state.parsed = process_resume_batch(valid_files, progress_bar, status_text)

            progress_bar.progress(1.0)
            status_text.text("✅ All resumes parsed!")
            time.sleep(0.3)
            status_text.empty()
            progress_bar.empty()
            proc_heading.empty()

    # Reading the files is the slow part, so only that is cached. Scoring is redone on
    # every run, which keeps the shortlist honest when the job description or the
    # filters change — and means neither costs a re-upload.
    parsed = st.session_state.get("parsed") or []
    if parsed:
        st.markdown("---")

        if not valid_files:
            st.info(
                "📌 **Showing your last run.** The uploader empties when you leave this tab, "
                "but the shortlist stays — and editing the job description above re-scores it "
                "immediately. Re-upload only to screen a different batch of CVs."
            )

        ranked   = rank_candidates(parsed, cached_parse_jd(jd_text))
        filtered = [c for c in ranked if c.get("final_score", 0) >= min_score]
        errors   = [c for c in ranked if c.get("error")]

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

        # ── Podium — the three worth opening first
        podium = [c for c in ranked if not c.get("error")][:3]
        if len(podium) >= 2:
            st.markdown('<div class="section-title">🏆 Your top matches</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-sub">Scored against this advert, not against each '
                'other — a low top score means the batch is weak, not that #1 is strong.</div>',
                unsafe_allow_html=True,
            )
            for i, (col, cand) in enumerate(zip(st.columns(len(podium), gap="medium"), podium)):
                with col:
                    medal = ["🥇", "🥈", "🥉"][i]
                    st.markdown(
                        score_ring_html(
                            cand.get("final_score", 0),
                            f"{medal} {cand.get('name', 'Unknown')}",
                            cand.get("assessment", {}).get("label", ""),
                        ),
                        unsafe_allow_html=True,
                    )

        # ── Shape of the batch
        st.markdown('<div class="section-title">📶 Shortlist mix</div>', unsafe_allow_html=True)
        st.markdown(pipeline_mix_html(ranked), unsafe_allow_html=True)

        # ── Is the advert askable? Coverage of each required skill across the batch.
        jd_parsed  = cached_parse_jd(jd_text)
        demand_html = skill_demand_html(ranked, jd_parsed.get("skills") or [])
        if demand_html:
            st.markdown('<div class="section-title">🎯 Is this advert realistic?</div>',
                        unsafe_allow_html=True)
            st.markdown(
                '<div class="section-sub">How many CVs in this batch actually state each '
                'skill you asked for. A row near zero usually means the requirement is rare '
                'in this pool — or that the wording on the advert does not match how people '
                'write it on a CV.</div>',
                unsafe_allow_html=True,
            )
            st.markdown(demand_html, unsafe_allow_html=True)

        # Every candidate was filtered out — say so instead of rendering empty sections.
        if not filtered:
            st.warning(
                f"**No candidate scored {min_score} or above.** "
                f"The best match here scored {top_score:.1f}. "
                "Lower the *Minimum score filter* in step 2 to see these candidates."
            )

        # ── Table view
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
                "Verdict":    c.get("assessment", {}).get("label", "—"),
                "Score":      c.get("final_score", 0),
            })

        if table_data:
            st.markdown('<div class="section-title">📋 Rankings Table</div>', unsafe_allow_html=True)
            df = pd.DataFrame(table_data)
            height = min(400, 50 + 35 * len(table_data))
            try:
                # Color gradient needs matplotlib; fall back to a plain table if absent.
                styled = (
                    df.style
                      .background_gradient(subset=["Score"], cmap="RdYlGn", vmin=0, vmax=100)
                      .format({"Score": "{:.1f}"})
                )
                st.dataframe(styled, use_container_width=True, height=height, hide_index=True)
            except (ImportError, ModuleNotFoundError):
                st.dataframe(
                    df.style.format({"Score": "{:.1f}"}),
                    use_container_width=True,
                    height=height,
                    hide_index=True,
                )

        # ── Score Visualization
        top_n = min(15, len(filtered))
        if top_n > 0:
            st.markdown('<div class="section-title">📈 Score Visualization</div>', unsafe_allow_html=True)
            # Zero-padded rank keeps the chart in ranked order (#01 before #10).
            chart_data = pd.DataFrame({
                "Candidate": [f"#{c.get('rank', 0):02d}  {c.get('name', 'Unknown')}"
                              for c in filtered[:top_n]],
                "Score":     [c["final_score"] for c in filtered[:top_n]],
            }).set_index("Candidate")
            st.bar_chart(
                chart_data,
                height=max(240, 42 * top_n),
                color="#2563eb",
                horizontal=True,
                x_label="",
                y_label="Match score (0–100)",
            )

        # ── If breakdown enabled, show component bar chart
        if show_breakdown and filtered:
            st.markdown('<div class="section-title">🔬 Score Breakdown (Top 10)</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-sub">Each bar is that component scored out of 100 — '
                'compare candidates side by side to see <i>why</i> they ranked where they did.</div>',
                unsafe_allow_html=True,
            )

            breakdown_rows = []
            for c in filtered[:10]:
                bd = c.get("score_breakdown", {})
                breakdown_rows.append({
                    "Name":       f"#{c.get('rank', 0):02d}  {c.get('name', 'Unknown')}",
                    "Skills":     bd.get("skill_score", 0),
                    "Experience": bd.get("exp_score", 0),
                    "Education":  bd.get("edu_score", 0),
                    "Location":   bd.get("loc_score", 0),
                })
            if breakdown_rows:
                bd_df = pd.DataFrame(breakdown_rows).set_index("Name")
                # Grouped (not stacked): these are independent 0-100 scores, so a
                # stacked total would be meaningless.
                st.bar_chart(
                    bd_df,
                    height=max(260, 74 * len(breakdown_rows)),
                    stack=False,
                    horizontal=True,
                    x_label="",
                    y_label="Score (0–100)",
                )

        # ── Detailed Candidate Cards
        if filtered:
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

            medal   = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "▫️")
            verdict = c.get("assessment", {}).get("label", "")
            with st.expander(
                f"{medal}  #{rank}   {name}   ·   {score:.1f}/100"
                + (f"   ·   {verdict}" if verdict else ""),
                expanded=(rank <= 3)
            ):
                # ── Analyst-style verdict (mirrors a human reviewer's write-up)
                assess = c.get("assessment", {})
                if assess:
                    tag = assess.get("tag", "ok")
                    summary = assess.get("summary", "")
                    if tag == "good":
                        st.success(f"**{summary}**")
                    elif tag == "ok":
                        st.info(f"**{summary}**")
                    elif tag == "warn":
                        st.warning(f"**{summary}**")
                    else:
                        st.error(f"**{summary}**")
                    notes = assess.get("notes", [])
                    if notes:
                        st.markdown("\n".join(f"- {n}" for n in notes))

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

        # ── Export to CSV (complete, defensible sheet: scores + verdict + skills)
        st.markdown("---")
        if filtered:
            export_rows = []
            for c in filtered:
                a = c.get("assessment", {})
                export_rows.append({
                    "Rank":            c.get("rank", ""),
                    "Name":            c.get("name", ""),
                    "Email":           c.get("email") or "",
                    "Phone":           c.get("phone") or "",
                    "Location":        c.get("location") or "",
                    "Experience (yrs)": round(c.get("experience_years", 0), 1),
                    "Education":       c.get("education", {}).get("level", ""),
                    "Score":           c.get("final_score", 0),
                    "Verdict":         a.get("label", ""),
                    "Fit Summary":     a.get("summary", ""),
                    "Matched Skills":  ", ".join(c.get("matched_skills", [])),
                    "Missing Skills":  ", ".join(c.get("missing_skills", [])),
                })
            export_df = pd.DataFrame(export_rows)
            csv = export_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="⬇️ Export Rankings as CSV",
                data=csv,
                file_name="ranked_candidates.csv",
                mime="text/csv",
                use_container_width=True,
            )

elif PAGE == CTA_TAB and not SIGNUP_GATE and not jd_text.strip():
    st.warning("☝️ Paste a job description in step 1 to begin.")

elif PAGE == CTA_TAB and not SIGNUP_GATE:
    st.info("☝️ **Ready when you are** — upload one or more resumes above to see the ranking.")


# ─────────────────────────────────────────────
# PAGE: HOW IT WORKS
# ─────────────────────────────────────────────

if PAGE == "How it works":
    st.markdown(
        '<span class="anchor" id="how-it-works"></span>'
        '<div class="section-title">How it works</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">Three steps, about a minute — no setup, no accounts, no cloud uploads.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div class="steps">
      <div class="step">
        <div class="step-num">1</div>
        <h4>📋 Describe the role</h4>
        <p>Paste any job description on the <b>Start screening</b> tab. Required skills,
           years of experience and location are pulled out automatically.</p>
      </div>
      <div class="step">
        <div class="step-num">2</div>
        <h4>📤 Drop in the resumes</h4>
        <p>Select up to 100 PDF or DOCX files at once. Each one is read and
           parsed in parallel — names, contacts, skills, history.</p>
      </div>
      <div class="step">
        <div class="step-num">3</div>
        <h4>🏆 Get a ranked shortlist</h4>
        <p>Candidates come back scored 0–100 and ordered, each with matched
           skills, gaps and a plain-English verdict you can defend.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">What happens to each file</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="viz-dark">
  <h4>🔬 Inside one resume</h4>
  <p class="viz-sub">The same six stages run on every file in the batch, on this machine.</p>
  <div class="stage-list">
    <div class="stage"><b>01 · Read</b><span>The text layer is pulled page by page from the PDF or DOCX. An image-only scan has no text layer, so it is flagged in the results rather than silently scored as zero.</span></div>
    <div class="stage"><b>02 · Identify</b><span>Name, email, phone and location are lifted from the top of the document, with a regex fallback whenever the name detector is unsure.</span></div>
    <div class="stage"><b>03 · Skills</b><span>The full text is swept against a role library covering civil &amp; construction, sales, marketing, design, IT and back-office work — not only software keywords.</span></div>
    <div class="stage"><b>04 · History</b><span>Total years of experience and the highest education level are read from date ranges and qualification wording.</span></div>
    <div class="stage"><b>05 · Compare</b><span>Everything found is matched against the requirements detected in your job description, producing a matched list and a missing list.</span></div>
    <div class="stage"><b>06 · Score</b><span>The four signals below are combined into one 0–100 number, and a short verdict is written explaining the result.</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">How the number is reached</div>', unsafe_allow_html=True)
    ex_a, ex_b = st.columns([1, 1.25])
    with ex_a:
        with st.container(border=True):
            st.markdown(SCORE_FORMULA_HTML, unsafe_allow_html=True)
    with ex_b:
        st.markdown("""
<div class="viz-card">
  <h4>🧮 A worked example</h4>
  <p class="viz-sub">Site Engineer vacancy in Pune, 3+ years, asking for AutoCAD, STAAD Pro, estimation and site supervision.</p>
  <div class="calc">
    <div class="calc-row"><span>Skills — 3 of 4 required found</span><b>75 × 0.50 = 37.5</b></div>
    <div class="calc-row"><span>Experience — 5 relevant years vs. 3 asked</span><b>100 × 0.25 = 25.0</b></div>
    <div class="calc-row"><span>Location — resume says Pune</span><b>100 × 0.15 = 15.0</b></div>
    <div class="calc-row"><span>Education — B.E. Civil</span><b>85 × 0.10 = 8.5</b></div>
    <div class="calc-total"><span>Final score</span><b>86.0 · Excellent fit</b></div>
  </div>
  <p class="viz-sub" style="margin:0.85rem 0 0">
    Nobody is removed from the list. A low score is a ranking, not a rejection — you
    still see every candidate and the reason behind each number.
  </p>
</div>
""", unsafe_allow_html=True)

    hw_a, _hw_b = st.columns([1.3, 3])
    with hw_a:
        st.button("🚀 Try it on your CVs", key="how_cta", use_container_width=True,
                  type="primary", on_click=go, args=(CTA_TAB,))


# ─────────────────────────────────────────────
# PAGE: FEATURES
# ─────────────────────────────────────────────

if PAGE == "Features":
    st.markdown(
        '<span class="anchor" id="features"></span>'
        '<div class="section-title">Why teams use it</div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div class="feat-grid">
      <div class="feat">
        <div class="feat-ico">🎯</div>
        <h4>Relevance over raw years</h4>
        <p>Experience is weighted by how well it matches the role, so a fitting
           candidate outranks an unrelated one with a longer CV.</p>
      </div>
      <div class="feat">
        <div class="feat-ico">🏗️</div>
        <h4>Built for every department</h4>
        <p>Civil &amp; construction, sales, marketing, design and technical roles
           are all covered — not just software keywords.</p>
      </div>
      <div class="feat">
        <div class="feat-ico">🔍</div>
        <h4>Explainable scoring</h4>
        <p>Every score breaks down into skills, experience, education and
           location, with matched and missing skills listed per candidate.</p>
      </div>
      <div class="feat">
        <div class="feat-ico">⚡</div>
        <h4>Batch processing</h4>
        <p>Resumes are parsed concurrently with a live progress bar, so a full
           stack of applicants is handled in one pass.</p>
      </div>
      <div class="feat">
        <div class="feat-ico">🔒</div>
        <h4>Private by design</h4>
        <p>Everything runs locally on your machine. No API keys, no third-party
           servers, no candidate data leaving the building.</p>
      </div>
      <div class="feat">
        <div class="feat-ico">📊</div>
        <h4>Share-ready output</h4>
        <p>Export the full shortlist — scores, verdicts, contacts and skill gaps
           — to CSV for the hiring manager in one click.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: ROLES
# ─────────────────────────────────────────────

if PAGE == "Roles":
    st.markdown(
        '<span class="anchor" id="roles"></span>'
        '<div class="section-title">Roles we cover</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">The skill database spans the whole company, not just the tech team — '
        'the same shortlist works for a site engineer and a sales manager.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div class="roles">
      <div class="role-card">
        <h4>🏗️ Civil &amp; Construction</h4>
        <div class="role-chips">
          <span>Site Engineer</span><span>Project Manager</span><span>Quantity Surveyor</span>
          <span>Structural Designer</span><span>Site Supervisor</span><span>Estimation</span>
          <span>AutoCAD</span><span>STAAD Pro</span><span>ETABS</span>
        </div>
      </div>
      <div class="role-card">
        <h4>📈 Sales &amp; Business Development</h4>
        <div class="role-chips">
          <span>Sales Executive</span><span>BD Manager</span><span>Key Accounts</span>
          <span>Channel Sales</span><span>Lead Generation</span><span>CRM</span>
          <span>Negotiation</span><span>Target Achievement</span>
        </div>
      </div>
      <div class="role-card">
        <h4>📣 Marketing &amp; Digital</h4>
        <div class="role-chips">
          <span>Digital Marketing</span><span>SEO</span><span>SEM</span><span>Social Media</span>
          <span>Content Marketing</span><span>Google Analytics</span><span>Campaign Management</span>
          <span>Brand Strategy</span>
        </div>
      </div>
      <div class="role-card">
        <h4>🎨 Design &amp; Creative</h4>
        <div class="role-chips">
          <span>Graphic Design</span><span>UI/UX</span><span>Photoshop</span>
          <span>Illustrator</span><span>Figma</span><span>3D Rendering</span>
        </div>
      </div>
      <div class="role-card">
        <h4>💻 Technical &amp; IT</h4>
        <div class="role-chips">
          <span>Python</span><span>Java</span><span>SQL</span><span>Cloud</span>
          <span>Data Analysis</span><span>DevOps</span><span>Machine Learning</span>
        </div>
      </div>
      <div class="role-card">
        <h4>🗂️ Business &amp; Office</h4>
        <div class="role-chips">
          <span>Accounts</span><span>HR</span><span>Admin</span><span>Procurement</span>
          <span>MS Excel</span><span>Tally</span><span>Documentation</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: FAQ
# ─────────────────────────────────────────────

if PAGE == "FAQ":
    st.markdown(
        '<span class="anchor" id="faq"></span>'
        '<div class="section-title">Frequently asked</div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div class="faq">
      <details>
        <summary>Does any candidate data leave my computer?</summary>
        <p>No. Parsing, scoring and ranking all run locally on this machine. There are no API
           keys, no external services and no uploads — which is why it can be used on real
           applicant CVs without a data-protection review.</p>
      </details>
      <details>
        <summary>How is the score actually calculated?</summary>
        <p>Four signals: skills match (50%), relevant experience (25%), location fit (15%) and
           education (10%). Experience is weighted by how closely it relates to the role, so ten
           unrelated years will not outrank three years in the actual field.</p>
      </details>
      <details>
        <summary>What file formats can I upload?</summary>
        <p>PDF and DOCX, up to 100 files in a single batch. Scanned image-only PDFs cannot be
           read — those are flagged in the results rather than silently scored as zero.</p>
      </details>
      <details>
        <summary>Does it work for non-technical roles?</summary>
        <p>Yes. The skill database covers civil and construction, sales, marketing, design,
           office and technical roles. You paste the job description and the required skills are
           detected from it, so the tool adapts to whatever role you are hiring for.</p>
      </details>
      <details>
        <summary>Can I share the shortlist with a hiring manager?</summary>
        <p>Export the full ranking to CSV in one click — scores, verdicts, matched skills,
           skill gaps and contact details, ready to open in Excel or attach to an email.</p>
      </details>
      <details>
        <summary>Will it reject a good candidate automatically?</summary>
        <p>Nothing is rejected. Every candidate is scored and listed with a written reason, and
           the minimum-score filter is yours to set. The tool orders the pile — the hiring
           decision stays with you.</p>
      </details>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    fq1, _fq2 = st.columns([1.3, 3])
    with fq1:
        st.button("🚀 Try it on real CVs", key="faq_cta", use_container_width=True,
                  type="primary", on_click=go, args=(CTA_TAB,))


# ─────────────────────────────────────────────
# PAGE: PRICING
# ─────────────────────────────────────────────

if PAGE == "Pricing":
    st.markdown(
        '<span class="anchor" id="pricing"></span>'
        '<div class="section-title">Pricing</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">One tool, installed on your own machines. No per-resume charges '
        'and no candidate data held by anyone else.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"""
<div class="price-grid">
  <div class="price-card">
    <h4>Starter</h4>
    <div class="p-amt">₹0 <small>/ trial</small></div>
    <p class="p-desc">For a single recruiter testing the tool on a live vacancy.</p>
    <ul class="price-list">
      <li>1 seat, 1 machine</li>
      <li>Up to {MAX_BATCH} resumes per batch</li>
      <li>All scoring signals</li>
      <li>CSV export</li>
      <li class="no">Custom skill database</li>
      <li class="no">Priority support</li>
    </ul>
  </div>
  <div class="price-card pop">
    <span class="price-tag">Most popular</span>
    <h4>Team</h4>
    <div class="p-amt">₹—— <small>/ month</small></div>
    <p class="p-desc">For an HR team screening across several departments at once.</p>
    <ul class="price-list">
      <li>Up to 10 seats</li>
      <li>100 resumes per batch</li>
      <li>All scoring signals</li>
      <li>CSV export &amp; shareable shortlists</li>
      <li>Skill database tuned to your roles</li>
      <li class="no">On-site installation</li>
    </ul>
  </div>
  <div class="price-card">
    <h4>Enterprise</h4>
    <div class="p-amt">Custom</div>
    <p class="p-desc">For company-wide hiring with your own role library and workflow.</p>
    <ul class="price-list">
      <li>Unlimited seats</li>
      <li>Unlimited batch size</li>
      <li>Custom scoring weights</li>
      <li>Your own skill &amp; role library</li>
      <li>On-site installation &amp; training</li>
      <li>Priority support</li>
    </ul>
  </div>
</div>
""", unsafe_allow_html=True)
    st.caption(
        "Figures shown are placeholders — set your actual plans and prices in `app.py` "
        "before sharing this page with customers."
    )
    pc1, pc2, _pc3 = st.columns([1.3, 1.3, 2])
    with pc1:
        if st.session_state.get("account"):
            st.button("🚀 Start screening", key="price_try", use_container_width=True,
                      type="primary", on_click=go, args=(CTA_TAB,))
        else:
            st.button("✨ Try for free", key="price_try", use_container_width=True,
                      type="primary", on_click=open_signup)
    with pc2:
        st.button("Talk to us", key="price_cta", use_container_width=True,
                  on_click=go, args=("Contact",))


# ─────────────────────────────────────────────
# PAGE: CONTACT
# ─────────────────────────────────────────────

if PAGE == "Contact":
    st.markdown(
        '<span class="anchor" id="contact-top"></span>'
        '<div class="section-title">Contact</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">Questions about scoring, a role type we do not cover yet, '
        'or an installation on your office machines — send it across.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
<div class="contact-grid">
  <div class="contact-card">
    <div class="c-ico">💬</div>
    <h4>Sales</h4>
    <p>Plans, licensing and on-site installation.<br>
       <a href="mailto:sales@example.com">sales@example.com</a></p>
  </div>
  <div class="contact-card">
    <div class="c-ico">🛠️</div>
    <h4>Support</h4>
    <p>Parsing issues, scoring questions, bug reports.<br>
       <a href="mailto:support@example.com">support@example.com</a></p>
  </div>
  <div class="contact-card">
    <div class="c-ico">📍</div>
    <h4>Office</h4>
    <p>Add your registered office address here.<br>
       Mon–Sat, 10:00–18:30 IST</p>
  </div>
</div>
""", unsafe_allow_html=True)
    st.caption("Placeholder details — replace them with your real contact information in `app.py`.")

    st.markdown('<div class="section-title">Send an enquiry</div>', unsafe_allow_html=True)
    with st.form("contact_form", border=True):
        f1, f2 = st.columns(2)
        with f1:
            c_name = st.text_input("Your name")
            c_org  = st.text_input("Company")
        with f2:
            c_mail = st.text_input("Work email")
            c_role = st.selectbox(
                "What are you hiring for?",
                ["Civil & construction", "Sales & business development",
                 "Marketing & digital", "Design & creative", "Technical & IT",
                 "Business & office", "A mix of departments"],
            )
        c_msg = st.text_area("Message", height=110)
        if st.form_submit_button("Send enquiry", type="primary"):
            if not c_name.strip() or not c_mail.strip():
                st.error("Please fill in your name and work email.")
            else:
                st.success(f"Thanks {c_name.strip()} — we'll get back to you at {c_mail.strip()}.")
                st.caption(
                    "Note: this form is not wired to an inbox yet. Connect it to your mail "
                    "or CRM in `app.py` before going live."
                )


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown(f"""
<span class="anchor" id="contact"></span>
<div class="site-footer">
  <div class="foot-grid">
    <div class="foot-brand">
      <div class="fb-top"><span class="fl">🎯</span><b>Resume Ranker</b></div>
      <p>Screen a stack of CVs against the role you are actually hiring for, and get a
         ranked shortlist you can defend — in about a minute.</p>
      <span class="foot-badge"><span class="pulse-dot"></span> Runs offline · No data leaves your machine</span>
    </div>
    <div class="fcol">
      <h5>Product</h5>
      <a href="#how-it-works">How it works</a>
      <a href="#features">Features</a>
      <a href="#roles">Roles we cover</a>
      <a href="#pricing">Pricing</a>
      <a href="#upload">Start screening</a>
      <a href="#candidate">Check my CV</a>
    </div>
    <div class="fcol">
      <h5>Resources</h5>
      <a href="#faq">FAQ</a>
      <a href="#docs">Documentation</a>
      <a href="#guide">Scoring guide</a>
      <a href="#sample">Sample job descriptions</a>
      <a href="#changelog">What's new</a>
    </div>
    <div class="fcol">
      <h5>Company</h5>
      <a href="#about">About</a>
      <a href="#contact">Contact sales</a>
      <a href="#support">Support</a>
      <a href="#careers">Careers</a>
    </div>
  </div>
  <div class="foot-bottom">
    <span>© {datetime.now().year} Resume Ranker. All rights reserved.</span>
    <span>
      <a href="#privacy">Privacy</a>
      <a href="#terms">Terms</a>
      <a href="#security">Security</a>
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# Streamlit scrolls an inner container rather than the document, so native "#id"
# jumps do nothing. It also strips <script> from st.markdown — hence this iframe,
# which reaches back into the parent document to wire the nav and footer links.
# Streamlit cancels native `behavior: 'smooth'` on that container, so the easing
# below is driven by hand.
components.html(
    """
    <script>
    // This iframe is torn down and rebuilt on every rerun, which would kill any timer
    // it owns mid-flight. So it only injects the router once, into the parent document,
    // where it keeps running across reruns.
    const doc = window.parent.document;
    if (!doc.getElementById('rr-router')) {
        const s = doc.createElement('script');
        s.id = 'rr-router';
        s.textContent = `
        // Timer-driven rather than requestAnimationFrame: Streamlit cancels native
        // smooth scrolling on this container, so the easing is driven by hand.
        function rrGlideTo(to) {
            const pane = document.querySelector('section[data-testid="stMain"]');
            if (!pane) return;
            const from = pane.scrollTop;
            const start = Date.now();
            const step = () => {
                const t = Math.min(1, (Date.now() - start) / 420);
                pane.scrollTop = from + (to - from) * (1 - Math.pow(1 - t, 3));
                if (t < 1) setTimeout(step, 16);
            };
            step();
        }

        // The footer keeps plain anchors, but every destination is now a tab. Map the
        // href to a tab and click that nav button — the tab bar is the only router.
        const TAB_FOR = {
            '#top': 'Home', '#how-it-works': 'How it works', '#features': 'Features',
            '#roles': 'Roles', '#pricing': 'Pricing', '#faq': 'FAQ',
            '#upload': 'Start screening', '#contact': 'Contact',
            '#docs': 'How it works', '#guide': 'How it works', '#sample': 'How it works',
            '#changelog': 'Features', '#about': 'Contact', '#support': 'Contact',
            '#careers': 'Contact', '#privacy': 'Contact', '#terms': 'Contact',
            '#security': 'FAQ', '#candidate': 'Check my CV', '#check-my-cv': 'Check my CV'
        };

        // Streamlit tags each keyed widget's container with st-key-<key>.
        const TAB_BUTTONS = [
            '[class*="st-key-nav_"]', '.st-key-hero_cta', '.st-key-hero_how',
            '.st-key-home_bottom_cta', '.st-key-how_cta', '.st-key-faq_cta',
            '.st-key-price_cta', '.st-key-price_try', '.st-key-aud_rec',
            '.st-key-aud_cand', '.st-key-gate_cta', '.st-key-overflow_cta',
            '.st-key-guest_btn'
        ].join(',');

        // Streamlit restores the previous scroll offset once the rerun paints, which
        // lands after this handler — so glide first, then pin the top until it settles.
        function rrToTop() {
            rrGlideTo(0);
            [450, 750, 1050, 1400].forEach(d => setTimeout(() => {
                const pane = document.querySelector('section[data-testid="stMain"]');
                if (pane) pane.scrollTop = 0;
            }, d));
        }

        // Delegated from <body> rather than bound per link: Streamlit re-renders the
        // footer on every run, which would drop per-element listeners.
        document.body.addEventListener('click', e => {
            const bar = document.querySelector(
                'div[data-testid="stHorizontalBlock"]:has(.nav-marker)');
            if (!bar) return;

            const link = e.target.closest('.site-footer a[href^="#"]');
            if (link) {
                e.preventDefault();
                const label = TAB_FOR[link.getAttribute('href')] || 'Home';
                for (const btn of bar.querySelectorAll('.stButton button')) {
                    const txt = btn.innerText.trim();
                    if (txt === label || txt.endsWith(label)) {
                        btn.click(); rrToTop(); return;
                    }
                }
                return;
            }

            // Only the buttons that change tabs — an Analyze or Download click must
            // leave the reader where they are.
            if (e.target.closest(TAB_BUTTONS)) rrToTop();
        }, true);
        `;
        doc.head.appendChild(s);
    }
    </script>
    """,
    height=0,
)
