import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

st.set_page_config(
    page_title="Job Intelligence Lab",
    page_icon="◎",
    layout="wide",
)

EXAMPLES = {
    "💻 Software Eng.": """\
Software Engineer — Backend

We are looking for a Software Engineer to join our platform engineering team and help build the infrastructure that powers millions of users. You will own backend services end-to-end, from design through deployment, and work closely with product, data, and infrastructure teams.

Responsibilities
• Design, build, and maintain scalable backend services in Python, Java, or Go
• Develop and document REST and gRPC APIs consumed by web and mobile clients
• Optimize database queries and improve system reliability and latency
• Participate in code reviews, on-call rotations, and incident response
• Collaborate with cross-functional teams to deliver product features

Requirements
• 3+ years of professional software engineering experience
• Strong understanding of distributed systems and backend architecture
• Hands-on experience with relational databases such as PostgreSQL or MySQL
• Familiarity with cloud platforms (AWS, GCP, or Azure) and container orchestration
• Experience with Redis, Kafka, or similar messaging and caching systems is a plus
""",

    "📊 Data Analyst": """\
Data Analyst — Business Intelligence

Join our analytics team to turn complex data into clear, actionable insights that drive business decisions. You will partner with marketing, operations, and finance to build dashboards, run analyses, and communicate findings to senior stakeholders.

Responsibilities
• Write SQL queries to extract, transform, and analyze large datasets
• Build and maintain dashboards and reports in Tableau, Looker, or Power BI
• Monitor key business KPIs and proactively surface anomalies and trends
• Conduct ad hoc analyses to answer business questions and support planning
• Present findings clearly to technical and non-technical audiences

Requirements
• 2+ years of experience in a data analyst or business intelligence role
• Strong proficiency in SQL; experience with Python or R is a plus
• Experience with at least one BI tool (Tableau, Looker, Power BI, etc.)
• Ability to translate ambiguous business questions into structured analysis
• Strong communication and data storytelling skills
""",

    "🤝 Sales Rep.": """\
Account Executive — Mid-Market SaaS

We are hiring a results-driven Account Executive to own the full sales cycle for mid-market accounts. You will prospect, qualify leads, run discovery calls, deliver demos, negotiate contracts, and close deals — all while managing a healthy pipeline in Salesforce.

Responsibilities
• Prospect and qualify inbound and outbound leads within your territory
• Run discovery calls and deliver tailored product demonstrations
• Build business cases and negotiate pricing and contract terms
• Maintain accurate pipeline hygiene and forecasting in Salesforce CRM
• Partner with customer success to ensure smooth handoffs and renewals

Requirements
• 3+ years of B2B SaaS sales experience with a track record of hitting quota
• Strong consultative selling, negotiation, and presentation skills
• Experience managing a full sales cycle from prospecting to close
• Proficiency with Salesforce CRM and sales engagement tools
• Self-motivated with excellent time management and organizational skills
""",

    "🏥 Nurse": """\
Registered Nurse — Medical-Surgical Unit

We are seeking a compassionate and skilled Registered Nurse to provide high-quality patient care on our Medical-Surgical unit. You will assess patients, develop care plans, administer medications, and work collaboratively with physicians and care teams to ensure safe clinical outcomes.

Responsibilities
• Conduct patient assessments and develop individualized care plans
• Administer medications, treatments, and IV therapy per physician orders
• Monitor vital signs and clinical status, escalating concerns appropriately
• Coordinate care with physicians, therapists, and interdisciplinary team members
• Educate patients and families on diagnoses, medications, and discharge planning
• Maintain accurate and timely nursing documentation in the EMR

Requirements
• Active Registered Nurse (RN) license in good standing
• BLS certification required; ACLS preferred
• 1+ year of acute care or medical-surgical nursing experience preferred
• Strong clinical assessment and critical thinking skills
• Ability to manage multiple patients in a fast-paced environment
""",

    "📦 Warehouse": """\
Warehouse Associate — Fulfillment & Distribution

Join our warehouse operations team at a high-volume fulfillment and distribution center. This is a hands-on warehouse role focused on physical receiving, inventory stocking, forklift operation, and packing and shipping outbound freight.

Warehouse Responsibilities
• Unload and receive inbound freight trucks; verify shipments against purchase orders
• Stock shelves and bin locations; maintain accurate warehouse inventory counts
• Pick orders using RF scanner and barcode scanning systems (WMS)
• Pack and label outbound shipments for parcel and LTL freight carriers
• Operate sit-down and stand-up forklifts, reach trucks, and pallet jacks safely
• Perform daily cycle counts and assist with full physical inventory audits
• Keep warehouse floor, staging areas, and loading dock clean and organized

Warehouse Requirements
• 1+ year of hands-on warehouse, distribution center, or freight handling experience
• Forklift certification required or ability to obtain within 30 days
• Comfortable operating RF scanners and warehouse management systems (WMS)
• Ability to lift up to 50 lbs and stand or walk for full shift duration
• Available for day shift, swing shift, or overnight warehouse operations
""",

    "👥 HR Manager": """\
Human Resources Manager

We are seeking an experienced HR Manager to lead people operations across recruitment, employee relations, performance management, and HR compliance. You will serve as a strategic partner to department leaders and help build a high-performing, inclusive workplace culture.

Responsibilities
• Manage full-cycle recruitment including job postings, screening, interviewing, and offer negotiation
• Oversee onboarding and offboarding processes to ensure a positive employee experience
• Advise managers on performance management, disciplinary actions, and conflict resolution
• Administer compensation, benefits, and leave programs in compliance with applicable laws
• Develop and update HR policies, employee handbooks, and compliance documentation
• Partner with leadership on workforce planning, org design, and talent development initiatives

Requirements
• 4+ years of HR generalist or HR management experience
• Strong knowledge of employment law and HR best practices (FMLA, ADA, EEOC, etc.)
• Experience with HRIS platforms such as Workday, ADP, or BambooHR
• Demonstrated ability to handle sensitive employee matters with discretion and professionalism
• PHR or SHRM-CP certification preferred
""",
}
EXAMPLE_KEYS = list(EXAMPLES.keys())

if "selected_example" not in st.session_state:
    st.session_state.selected_example = None

if "last_segment" not in st.session_state:
    st.session_state.last_segment = None

if "textarea_main" not in st.session_state:
    st.session_state.textarea_main = """Machine Learning Engineer

PayPal is looking for a Machine Learning Engineer to help build and deploy models that power key financial and commerce experiences. You will work with large-scale transaction and behavioral data to develop machine learning systems used in areas such as fraud detection, risk modeling, and customer insights.

Responsibilities
• Develop and train machine learning models for large-scale production systems
• Build data pipelines and feature engineering workflows
• Collaborate with data scientists and product teams to translate business problems into ML solutions
• Deploy and monitor models in production environments

Requirements
• Experience with Python and machine learning frameworks such as TensorFlow, PyTorch, or scikit-learn
• Strong knowledge of statistics, model evaluation, and experimentation
• Experience working with large datasets and distributed data systems

Preferred
• Experience with fraud detection, risk modeling, or financial data
• Familiarity with cloud platforms such as AWS or GCP"""

def load_example(label):
    st.session_state.selected_example = label
    st.session_state.textarea_main = EXAMPLES[label]
    st.session_state.last_segment = None   # clear highlight when switching examples

SEG_STATS = pd.DataFrame([
    {"segment": "Sales – Account / Enterprise",      "n_jobs": 924, "avg_salary": 111664, "med_salary": 97500},
    {"segment": "Healthcare – Clinical",              "n_jobs": 599, "avg_salary": 93604,  "med_salary": 86300},
    {"segment": "Logistics / Warehouse",              "n_jobs": 450, "avg_salary": 84006,  "med_salary": 66539},
    {"segment": "Customer Service / Retail",          "n_jobs": 436, "avg_salary": 70966,  "med_salary": 62500},
    {"segment": "Data & Business Analytics",          "n_jobs": 407, "avg_salary": 107302, "med_salary": 98250},
    {"segment": "Software Engineering",               "n_jobs": 346, "avg_salary": 116405, "med_salary": 110000},
    {"segment": "Field Service / Maintenance",        "n_jobs": 327, "avg_salary": 63246,  "med_salary": 58120},
    {"segment": "Project / Program Management",       "n_jobs": 263, "avg_salary": 98103,  "med_salary": 102500},
    {"segment": "Billing / Client Operations",        "n_jobs": 178, "avg_salary": 63469,  "med_salary": 62500},
    {"segment": "Healthcare Admin (Non-Clinical)",    "n_jobs": 164, "avg_salary": 86908,  "med_salary": 72047},
    {"segment": "Food Service / Hospitality",         "n_jobs": 159, "avg_salary": 68765,  "med_salary": 45895},
    {"segment": "Clinical Therapy",                   "n_jobs": 152, "avg_salary": 84495,  "med_salary": 80000},
    {"segment": "Administrative / Executive Support", "n_jobs": 142, "avg_salary": 77535,  "med_salary": 72800},
    {"segment": "Hospitality / Food Service",         "n_jobs": 115, "avg_salary": 57356,  "med_salary": 39198},
    {"segment": "Healthcare – Pharmacy / Lab",        "n_jobs": 96,  "avg_salary": 94050,  "med_salary": 62970},
    {"segment": "Cyber / Information Security",       "n_jobs": 79,  "avg_salary": 120726, "med_salary": 122200},
    {"segment": "Legal Support",                      "n_jobs": 74,  "avg_salary": 103097, "med_salary": 90760},
    {"segment": "Education / Teaching",               "n_jobs": 48,  "avg_salary": 62302,  "med_salary": 62400},
    {"segment": "Construction / Estimating",          "n_jobs": 20,  "avg_salary": 103764, "med_salary": 105000},
    {"segment": "Bilingual / Community Support",      "n_jobs": 19,  "avg_salary": 48393,  "med_salary": 45760},
    {"segment": "Gig / Delivery (Platform Work)",     "n_jobs": 2,   "avg_salary": None,   "med_salary": None},
])

TOTAL_JOBS  = SEG_STATS["n_jobs"].sum()
AVG_SAL_MIN = int(SEG_STATS["avg_salary"].dropna().min() / 1000)
AVG_SAL_MAX = int(SEG_STATS["avg_salary"].dropna().max() / 1000)

st.markdown("""
<style>
:root {
    --bg:       #f5f7fb;
    --surface:  #ffffff;
    --s2:       #f0f4fa;
    --bd:       #e3e9f4;
    --bd2:      #cdd6ea;
    --ink:      #0b1120;
    --ink2:     #3a4d6b;
    --ink3:     #7a8fa8;
    --ink4:     #b4c2d4;
    --blue:     #1d6ef5;
    --blue2:    #0ea5e9;
    --blue-bg:  #edf3ff;
    --blue-bd:  #bdd0ff;
    --green:    #0fa870;
    --gbg:      #e8faf3;
    --gbd:      #6ee7b7;
    --purple:   #7c3aed;
    --amber:    #d97706;
    --fd: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", sans-serif;
    --fb: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif;
    --fm: ui-monospace, "SF Mono", "Cascadia Code", "Consolas", monospace;
    --sh1: 0 1px 3px rgba(11,17,32,.05);
    --sh2: 0 4px 18px rgba(11,17,32,.07), 0 1px 4px rgba(11,17,32,.04);
    --sh3: 0 12px 40px rgba(11,17,32,.09), 0 2px 8px rgba(11,17,32,.05);
}
html, body, .main,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] { background: var(--bg) !important; color: var(--ink); }
[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"] { display: none !important; }
.block-container { max-width: 1100px !important; padding: 0 2.5rem 6rem !important; font-family: var(--fb); }
div[data-testid="stVerticalBlock"] { gap: 0 !important; }

/* NAV */
.nav { display: flex; align-items: center; justify-content: space-between; padding: 1.4rem 0; border-bottom: 1.5px solid var(--bd); }
.nav-left { display: flex; align-items: center; gap: 11px; }
.nav-logo { width: 36px; height: 36px; border-radius: 9px; background: var(--ink); display: grid; place-items: center; font-family: var(--fm); font-size: 16px; font-weight: 900; color: #fff; box-shadow: var(--sh2); flex-shrink: 0; }
.nav-wordmark { font-family: var(--fd); font-size: 1rem; font-weight: 700; color: var(--ink); letter-spacing: -.02em; }
.nav-wordmark em { font-style: normal; color: var(--blue); }
.nav-right { display: flex; align-items: center; gap: 6px; }
.nav-pill { font-family: var(--fm); font-size: .59rem; letter-spacing: .07em; text-transform: uppercase; color: var(--ink3); background: var(--surface); border: 1.5px solid var(--bd); padding: 4px 10px; border-radius: 99px; }
.nav-pill.live { color: var(--green); background: var(--gbg); border-color: var(--gbd); display: flex; align-items: center; gap: 5px; }
.live-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--green); box-shadow: 0 0 4px var(--green); animation: ldot 2s ease-in-out infinite; }
@keyframes ldot { 0%,100%{opacity:1} 50%{opacity:.3} }

/* HERO */
.hl1 { font-family: var(--fd); font-size: 4.2rem; font-weight: 900; line-height: .95; letter-spacing: -.055em; color: var(--ink); display: block; margin-top: 4rem; }
.hl2 { font-family: var(--fd); font-size: 4.2rem; font-weight: 300; line-height: 1.05; letter-spacing: -.04em; color: var(--ink3); display: block; margin-top: .04em; }
.hl2 .mkt { font-style: italic; font-weight: 800; color: transparent; -webkit-text-stroke: 2.5px var(--blue); }
.hl-bar { display: block; width: 72px; height: 3px; background: linear-gradient(90deg, var(--blue) 0%, var(--blue2) 100%); border-radius: 2px; margin-top: 14px; opacity: .65; }

.value-props { margin: 1.9rem 0 2.2rem; display: flex; flex-direction: column; gap: 12px; }
.vp-row { display: flex; align-items: flex-start; gap: 10px; }
.vp-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--blue); margin-top: 7px; flex-shrink: 0; }
.vp-text { font-size: .94rem; color: var(--ink2); line-height: 1.55; }
.vp-text strong { color: var(--ink); font-weight: 650; }

/* ═══════════════════════════════════
   INPUT CARD — frameless, blends with bg
═══════════════════════════════════ */
.input-card {
    background: var(--surface); border: 1.5px solid var(--bd2);
    border-radius: 20px; box-shadow: var(--sh3);
    position: relative; overflow: hidden; margin-top: 4rem;
}
.input-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--blue) 0%, var(--blue2) 100%);
}

.ic-header {
    padding: 20px 22px 14px;
    border-bottom: 1.5px solid var(--bd);
}
.ic-header-top {
    display: flex; align-items: baseline; justify-content: space-between;
    margin-bottom: 6px;
}
.ic-title {
    font-family: var(--fd); font-size: 1.05rem; font-weight: 800;
    color: var(--ink); letter-spacing: -.03em;
}
.ic-badges { display: flex; gap: 5px; align-items: center; }
.ic-badge {
    font-family: var(--fm); font-size: .56rem; letter-spacing: .07em;
    text-transform: uppercase; padding: 3px 8px; border-radius: 5px;
}
.ic-badge.b-title  { color: var(--blue);  background: var(--blue-bg);  border: 1px solid var(--blue-bd); }
.ic-badge.b-desc   { color: var(--green); background: var(--gbg);      border: 1px solid var(--gbd); }
.ic-badge.b-or     { color: var(--ink3);  background: var(--s2);       border: 1px solid var(--bd); }
.ic-subtitle {
    font-size: .8rem; color: var(--ink3); line-height: 1.45;
}

.ex-label { font-family: var(--fm); font-size: .59rem; letter-spacing: .14em; text-transform: uppercase; color: var(--ink4); padding: 16px 22px 6px; margin-bottom: 14px; border-top: 1.5px solid var(--bd); }
.example-chips {
    padding: 4px 22px 18px;
}
            
.example-chips div.stButton > button {
    background: var(--s2) !important; color: var(--ink2) !important;
    border: 1.5px solid var(--bd) !important; border-radius: 8px !important;
    height: 34px !important; min-height: 34px !important; padding: 0 .8rem !important;
    font-family: var(--fb) !important; font-size: .79rem !important; font-weight: 500 !important;
    width: 100% !important; white-space: nowrap !important; overflow: hidden !important;
    text-overflow: ellipsis !important; line-height: 34px !important; letter-spacing: 0 !important;
    transition: all .12s ease !important; box-shadow: none !important;
}
.example-chips div.stButton > button:hover {
    background: var(--blue-bg) !important;
    border-color: var(--blue-bd) !important;
    color: var(--blue) !important;
    transform: translateY(-2px) !important;
}
.analyze-btn-wrap { padding: 12px 22px 22px; }
.analyze-btn-wrap div.stButton > button {
    width: 100% !important; background: var(--ink) !important; color: #fff !important;
    border: none !important; border-radius: 11px !important; height: 52px !important;
    font-family: var(--fd) !important; font-weight: 700 !important; font-size: 1rem !important;
    letter-spacing: -.01em !important; box-shadow: 0 4px 18px rgba(11,17,32,.2) !important;
    transition: all .15s ease !important;
}
.analyze-btn-wrap div.stButton > button:hover {
    background: #1a3060 !important; transform: translateY(-1px) !important;
    box-shadow: 0 8px 26px rgba(11,17,32,.25) !important;
}
div[data-testid="stTextArea"] { padding: 13px 22px 0 !important; }
div[data-testid="stTextArea"] label { display: none !important; }
div[data-testid="stTextArea"] textarea {
    background: var(--s2) !important; border: 1.5px solid var(--bd) !important;
    border-radius: 10px !important; color: var(--ink2) !important;
    font-family: var(--fb) !important; font-size: .91rem !important;
    line-height: 1.7 !important; padding: 14px 16px !important;
    caret-color: var(--blue) !important; width: 100% !important;
}
div[data-testid="stTextArea"] textarea:focus {
    background: #fff !important; border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px rgba(29,110,245,.07) !important; outline: none !important; color: var(--ink) !important;
}

/* SECTION LABEL */
.sect { font-family: var(--fm); font-size: .59rem; letter-spacing: .16em; text-transform: uppercase; color: var(--ink4); margin: 2.6rem 0 1.1rem; display: flex; align-items: center; gap: 10px; }
.sect::after { content: ''; flex: 1; height: 1.5px; background: var(--bd); }

/* CHART HEADER — frameless, blends with bg */
.chart-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.6rem 0 0.2rem; flex-wrap: wrap; gap: 8px;
}
.chart-header-left { display: flex; flex-direction: column; gap: 3px; }
.chart-header-title {
    font-family: var(--fd); font-size: .95rem; font-weight: 700;
    color: var(--ink); letter-spacing: -.02em;
}
.chart-header-meta { font-size: .75rem; color: var(--ink4); }
.chart-legend { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
.lg { display: flex; align-items: center; gap: 5px; font-family: var(--fm); font-size: .6rem; color: var(--ink3); letter-spacing: .05em; }
.lg-dot { width: 7px; height: 7px; border-radius: 50%; }

/* RESULT CARDS */
.seg-card { background: var(--surface); border: 1.5px solid var(--gbd); border-radius: 18px; padding: 28px 28px 24px; height: 100%; box-sizing: border-box; position: relative; overflow: hidden; box-shadow: 0 4px 24px rgba(10,168,112,.09); }
.seg-card::after { content: ''; position: absolute; top: -60px; right: -60px; width: 200px; height: 200px; border-radius: 50%; background: radial-gradient(circle, rgba(10,168,112,.06) 0%, transparent 70%); pointer-events: none; }
.seg-live { display: flex; align-items: center; gap: 7px; margin-bottom: .9rem; }
.dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); flex-shrink: 0; animation: pulse 2.4s ease-in-out infinite; }
@keyframes pulse { 0%,100%{box-shadow:0 0 0 2px rgba(10,168,112,.2)} 50%{box-shadow:0 0 0 5px rgba(10,168,112,.06)} }
.seg-live-lbl { font-family: var(--fm); font-size: .61rem; letter-spacing: .12em; text-transform: uppercase; color: var(--green); }
.seg-name { font-family: var(--fd); font-size: 2rem; font-weight: 800; color: var(--ink); line-height: 1.1; letter-spacing: -.03em; margin-bottom: 1rem; word-break: break-word; }
.seg-score-row { display: flex; align-items: center; gap: 8px; margin-bottom: 1rem; flex-wrap: wrap; }
.score-tag { font-family: var(--fm); font-size: .71rem; font-weight: 600; color: var(--green); background: var(--gbg); border: 1.5px solid var(--gbd); padding: 5px 12px; border-radius: 7px; }
.conf-tag { font-family: var(--fm); font-size: .61rem; letter-spacing: .08em; text-transform: uppercase; color: var(--ink3); }
.seg-note { font-size: .79rem; color: var(--ink3); line-height: 1.65; padding-top: .9rem; border-top: 1.5px solid var(--bd); }
.met-card { background: var(--surface); border: 1.5px solid var(--bd2); border-radius: 18px; overflow: hidden; height: 100%; box-sizing: border-box; box-shadow: var(--sh2); }
.met-hdr { padding: 13px 20px; border-bottom: 1.5px solid var(--bd); background: var(--s2); display: flex; align-items: center; gap: 8px; }
.met-hdr-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--purple); }
.met-hdr-title { font-family: var(--fm); font-size: .61rem; letter-spacing: .12em; text-transform: uppercase; color: var(--ink3); }
.met-row { padding: 16px 20px; border-bottom: 1.5px solid var(--bd); }
.met-row:last-child { border-bottom: none; }
.met-lbl { font-family: var(--fm); font-size: .57rem; letter-spacing: .1em; text-transform: uppercase; color: var(--ink4); margin-bottom: 5px; }
.met-val { font-family: var(--fd); font-size: 2.2rem; font-weight: 800; letter-spacing: -.04em; color: var(--ink); line-height: 1; }
.met-val.xl { font-size: 2.55rem; }
.met-val.blue { color: var(--blue); }
.met-val.purple { color: var(--purple); }

/* ═══════════════════════════════════
   SIMILAR ROLES — upgraded table
═══════════════════════════════════ */
.roles-card {
    background: var(--surface);
    border: 1.5px solid var(--bd2);
    border-radius: 18px;
    overflow: hidden;
    box-shadow: var(--sh2);
}

.roles-header {
    padding: 20px 24px 16px;
    border-bottom: 1.5px solid var(--bd);
    background: linear-gradient(180deg, #fafcff 0%, var(--surface) 100%);
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.roles-header-left {
    display: flex;
    flex-direction: column;
    gap: 3px;
}

.roles-title {
    font-family: var(--fd);
    font-size: .97rem;
    font-weight: 700;
    color: var(--ink);
    letter-spacing: -.015em;
}

.roles-sub {
    font-size: .78rem;
    color: var(--ink3);
}

.roles-header-right {
    display: flex;
    align-items: center;
    gap: 8px;
}

.roles-count {
    font-family: var(--fm);
    font-size: .62rem;
    color: var(--blue);
    background: var(--blue-bg);
    border: 1.5px solid var(--blue-bd);
    padding: 3px 10px;
    border-radius: 99px;
    letter-spacing: .04em;
}

/* 3-column layout */
.role-row {
    display: grid;
    grid-template-columns: 1fr 220px 120px;
    align-items: center;
    gap: 12px;
    padding: 14px 24px;
    border-bottom: 1.5px solid var(--bd);
    transition: background .12s ease;
}

.role-row:last-child {
    border-bottom: none;
}

.role-row:hover {
    background: var(--s2);
}

.role-row-header {
    padding: 9px 24px;
    background: var(--s2);
    border-bottom: 1.5px solid var(--bd);
}

.role-col-hdr {
    font-family: var(--fm);
    font-size: .57rem;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--ink4);
}

.role-title-block {
    display: flex;
    flex-direction: column;
    gap: 3px;
    min-width: 0;
}

.role-title {
    font-family: var(--fb);
    font-size: .88rem;
    font-weight: 600;
    color: var(--ink);
    line-height: 1.3;
    letter-spacing: -.01em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.role-company {
    display: flex;
    align-items: center;
    gap: 5px;
    min-width: 0;
}

.role-company-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: var(--ink4);
    flex-shrink: 0;
}

.role-company-name {
    font-size: .78rem;
    color: var(--ink3);
    font-weight: 400;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.role-location {
    font-size: .82rem;
    color: var(--ink2);
    display: flex;
    align-items: center;
    gap: 5px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.role-location-pin {
    font-size: .75rem;
    opacity: .5;
    flex-shrink: 0;
}

.role-salary {
    font-family: var(--fm);
    font-size: .82rem;
    font-weight: 600;
    text-align: right;
    white-space: nowrap;
}

.role-salary.has-val {
    color: var(--blue);
}

.role-salary.no-val {
    color: var(--ink4);
}

.roles-footer {
    padding: 11px 24px;
    background: var(--s2);
    border-top: 1.5px solid var(--bd);
    font-family: var(--fm);
    font-size: .59rem;
    color: var(--ink4);
    letter-spacing: .06em;
    text-transform: uppercase;
}

.footer { 
    margin-top: 5rem; 
    padding: 1.4rem 0; 
    border-top: 1.5px solid var(--bd); 
    display: flex; 
    justify-content: space-between; 
    align-items: center; 
    flex-wrap: wrap; 
    gap: .5rem; 
}

.fl, .fr { 
    font-family: var(--fm); 
    font-size: .57rem; 
    color: var(--ink4); 
    letter-spacing: .07em; 
    text-transform: uppercase; 
}

[data-testid="stAlert"] { 
    background: #fffbeb !important; 
    border: 1.5px solid #fde68a !important; 
    border-radius: 10px !important; 
    color: var(--amber) !important; 
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--bd2); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)            

# ──────────────────────────────────────────
# Load
# ──────────────────────────────────────────
PARQUET_COLS = [
    "segment_name_final",
    "title",
    "company_name",
    "location",
    "normalized_salary",
]

@st.cache_resource
def load_model():
    # Lazy-loaded on first prediction call
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data(show_spinner=False)
def load_data():
    # Load only columns needed by the UI
    df = pd.read_parquet("artifacts/df_all.parquet", columns=PARQUET_COLS)

    # Light dtype optimization
    for c in ["segment_name_final", "company_name", "location"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype("category")

    if "title" in df.columns:
        df["title"] = df["title"].fillna("").astype(str)

    if "normalized_salary" in df.columns:
        df["normalized_salary"] = pd.to_numeric(
            df["normalized_salary"], errors="coerce"
        ).astype("float32")

    return df

@st.cache_resource
def load_embeddings():
    # Memory-map instead of fully loading into RAM
    return np.load("artifacts/X_fused.npy", mmap_mode="r")

@st.cache_resource
def load_centroids():
    d = np.load("artifacts/segment_centroids.npz")
    return {k: d[k].astype(np.float32) for k in d.files}

df_all = load_data()
X_fused = load_embeddings()
segment_centroids = load_centroids()

def fmt_money(x):
    if pd.isna(x): return None
    return f"${x:,.0f}"

def conf_label(s):
    if s >= 0.65: return "High confidence"
    if s >= 0.50: return "Moderate"
    return "Exploratory"

def conf_icon(s):
    return "▲" if s >= 0.65 else ("◆" if s >= 0.50 else "◇")


# ──────────────────────────────────────────
# Bubble chart
# ──────────────────────────────────────────
def build_bubble_chart(highlight_seg=None):
    df = SEG_STATS.dropna(subset=["avg_salary", "med_salary"]).copy()

    def get_color(row):
        if highlight_seg and row["segment"] == highlight_seg: return "#1d6ef5"
        if row["avg_salary"] >= 110000: return "#7c3aed"
        if row["avg_salary"] >= 80000:  return "#0fa870"
        return "#b4c2d4"

    def get_opacity(row):
        if not highlight_seg: return 0.85
        return 1.0 if row["segment"] == highlight_seg else 0.3

    df["color"]   = df.apply(get_color, axis=1)
    df["opacity"] = df.apply(get_opacity, axis=1)
    df["size"]    = (df["med_salary"] / 2400).clip(lower=12, upper=58)
    df["label"]   = df["segment"].apply(lambda s: s.replace(" – "," ").replace(" / "," ").split()[0])

    hover = df.apply(lambda r:
        f"<b style='font-size:13px'>{r['segment']}</b><br>"
        f"<span style='color:#7a8fa8'>Postings</span>  {r['n_jobs']:,}<br>"
        f"<span style='color:#7a8fa8'>Avg salary</span>  ${r['avg_salary']:,.0f}<br>"
        f"<span style='color:#7a8fa8'>Median</span>  ${r['med_salary']:,.0f}", axis=1)

    fig = go.Figure()
    for _, row in df.iterrows():
        is_hl = highlight_seg and row["segment"] == highlight_seg
        fig.add_trace(go.Scatter(
            x=[row["n_jobs"]], y=[row["avg_salary"]],
            mode="markers+text",
            marker=dict(size=row["size"], color=row["color"], opacity=row["opacity"],
                        line=dict(width=2.5 if is_hl else 1, color="#ffffff" if is_hl else "rgba(255,255,255,0.6)")),
            text=[row["label"]],
            textposition="top center",
            textfont=dict(family="-apple-system, 'SF Pro Text', sans-serif", size=9.5,
                          color="#3a4d6b" if not highlight_seg or is_hl else "#b4c2d4"),
            hovertemplate=hover[row.name] + "<extra></extra>",
            showlegend=False,
        ))

    median_x = df["n_jobs"].median()
    median_y = df["avg_salary"].median()
    fig.add_hline(y=median_y, line=dict(color="#e3e9f4", width=1.5, dash="dot"))
    fig.add_vline(x=median_x, line=dict(color="#e3e9f4", width=1.5, dash="dot"))
    fig.add_annotation(x=df["n_jobs"].max()*0.92, y=df["avg_salary"].max()*0.97,
        text="High pay · Large", font=dict(size=8, color="#d0d9eb"), showarrow=False)
    fig.add_annotation(x=df["n_jobs"].max()*0.92, y=df["avg_salary"].min()*1.05,
        text="Lower pay · Large", font=dict(size=8, color="#d0d9eb"), showarrow=False)

    fig.update_layout(
        height=400, margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#f0f4fa", gridwidth=1, zeroline=False, showline=False,
                   tickfont=dict(size=9, color="#b4c2d4"), ticksuffix=" jobs", title=None),
        yaxis=dict(showgrid=True, gridcolor="#f0f4fa", gridwidth=1, zeroline=False, showline=False,
                   tickfont=dict(size=9, color="#b4c2d4"), tickprefix="$", tickformat=",", title=None),
        hoverlabel=dict(bgcolor="white", bordercolor="#e3e9f4",
                        font=dict(size=12, family="-apple-system, sans-serif", color="#0b1120")),
        showlegend=False,
    )
    return fig


# ── NAV ──────────────────────────────────
st.markdown("""
<div class="nav">
  <div class="nav-left">
    <div class="nav-logo">◎</div>
    <div class="nav-wordmark">Job Intelligence <em>Lab</em></div>
  </div>
  <div class="nav-right">
    <div class="nav-pill live"><div class="live-dot"></div>Live</div>
    <div class="nav-pill">21 Segments</div>
    <div class="nav-pill">NLP-Powered</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── HERO ─────────────────────────────────
col_hl, col_input = st.columns([9, 11], gap="large")

with col_hl:
    st.markdown("""
    <span class="hl1">Decode<br>any job.</span>
    <span class="hl2">Know the&nbsp;<span class="mkt">market.</span></span>
    <span class="hl-bar"></span>
    <div class="value-props">
      <div class="vp-row"><div class="vp-dot"></div>
        <div class="vp-text"><strong>Instantly map any job posting</strong> to one of the real labor-market segments.</div></div>
      <div class="vp-row"><div class="vp-dot"></div>
        <div class="vp-text"><strong>See real salary benchmarks</strong> for that segment, so you know what the market is actually paying.</div></div>
      <div class="vp-row"><div class="vp-dot"></div>
        <div class="vp-text"><strong>Explore similar postings</strong> to see how other companies describe the role.</div></div>
    </div>
    """, unsafe_allow_html=True)

with col_input:
    # ── Redesigned input card header ──
    st.markdown("""
    <div class="input-card">
      <div class="ic-header">
        <div class="ic-header-top">
          <div class="ic-title">Paste A Job Posting</div>
          <div class="ic-badges">
            <div class="ic-badge b-title">Title</div>
            <div class="ic-badge b-desc">Description</div>
            <div class="ic-badge b-or">or both</div>
          </div>
        </div>
        <div class="ic-subtitle">Drop any job title, description, or a full posting — the model handles the rest.</div>
      </div>
    """, unsafe_allow_html=True)

    # Updated default text — PayPal ML Engineer, sized to fit textarea
    default_text = """\
Machine Learning Engineer

PayPal is looking for a Machine Learning Engineer to help build and deploy models that power key financial and commerce experiences. You will work with large-scale transaction and behavioral data to develop machine learning systems used in areas such as fraud detection, risk modeling, and customer insights.

Responsibilities
    • Develop and train machine learning models for large-scale production systems  
    • Build data pipelines and feature engineering workflows  
    • Collaborate with data scientists and product teams to translate business problems into ML solutions  
    • Deploy and monitor models in production environments  

Requirements
    • Experience with Python and machine learning frameworks such as TensorFlow, PyTorch, or scikit-learn  
    • Strong knowledge of statistics, model evaluation, and experimentation  
    • Experience working with large datasets and distributed data systems  

Preferred
    • Experience with fraud detection, risk modeling, or financial data  
    • Familiarity with cloud platforms such as AWS or GCP
    """

    user_text = st.text_area(
    "job_input",
    height=230,
    label_visibility="collapsed",
    key="textarea_main",
    )

    st.markdown('<div class="ex-label">Examples</div>', unsafe_allow_html=True)
    st.markdown('<div class="example-chips">', unsafe_allow_html=True)

    chip_cols = st.columns(3)
    for i, label in enumerate(EXAMPLE_KEYS):
        with chip_cols[i % 3]:
            st.button(
                label,
                key=f"chip_{i}",
                use_container_width=True,
                on_click=load_example,
                args=(label,),
            )

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="analyze-btn-wrap">', unsafe_allow_html=True)
    predict = st.button("◎  Analyze Job Posting", key="analyze_btn", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)  # /input-card


# ── MARKET OVERVIEW CHART ─────────────────
st.markdown('<div class="sect">01 — Market Overview</div>', unsafe_allow_html=True)
st.markdown("""
<div class="chart-header">
  <div class="chart-header-left">
    <div class="chart-header-title">Job Market Segments</div>
    <div class="chart-header-meta">X = postings &nbsp;·&nbsp; Y = avg salary &nbsp;·&nbsp; size = median salary</div>
  </div>
  <div class="chart-legend">
    <div class="lg"><div class="lg-dot" style="background:#7c3aed"></div>$110k+</div>
    <div class="lg"><div class="lg-dot" style="background:#0fa870"></div>$80–110k</div>
    <div class="lg"><div class="lg-dot" style="background:#b4c2d4"></div>&lt;$80k</div>
    <div class="lg"><div class="lg-dot" style="background:#1d6ef5"></div>Selected</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Always read last_segment fresh from session_state at render time
st.plotly_chart(
    build_bubble_chart(st.session_state.get("last_segment", None)),
    use_container_width=True,
    config={"displayModeBar": False},
    key="chart_overview",
)


# ── RESULTS ──────────────────────────────
if predict:
    analysis_text = st.session_state.textarea_main.strip()

    if not analysis_text:
        st.warning("Please enter a job title, description, or both.")
    else:
        with st.spinner("Classifying…"):
            # Robust inference block
            # Goal: better match training-time pipeline and reduce noise
            model = load_model()
            raw_text = analysis_text.strip()

            # ── Step 1: split title (first non-empty line) from description ──
            lines      = raw_text.split("\n")
            title_text = lines[0].strip()
            desc_text  = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

            # ── Step 2: fallback for single-line paste ──
            if not desc_text and len(title_text.split()) > 12:
                tokens     = title_text.split()
                title_text = " ".join(tokens[:6])
                desc_text  = " ".join(tokens[6:])

            # ── Step 3: boilerplate tail removal (EEO / privacy / benefits) ──
            import re as _re
            BOILERPLATE_RE = _re.compile(
                r"(?:\n|^)[ \t]*(?:equal opportunity|eeo\b|reasonable accommodation"
                r"|privacy policy|background check|drug.?free|benefits include"
                r"|by applying.*you agree|terms of use).*",
                flags=_re.IGNORECASE | _re.DOTALL,
            )
            m = BOILERPLATE_RE.search(desc_text)
            if m:
                desc_text = desc_text[: m.start()].strip()

            # ── Step 4: truncate to match training-time limits ──
            title_text = title_text[:300]
            desc_text  = desc_text[:3000]

            # ── Step 5: title-boost for short descriptions ──
            # Training descriptions averaged 800–2000 chars. Short inputs
            # lose domain signal; repeating the title anchors the desc
            # embedding back into the correct semantic region before fusion.
            SHORT_DESC_THRESHOLD = 300
            if len(desc_text) < SHORT_DESC_THRESHOLD:
                boost     = (title_text + " ") * 3
                desc_text = (boost + desc_text).strip()

            # ── Step 6: embed & fuse (mirrors training pipeline exactly) ──
            title_emb = model.encode([title_text], show_progress_bar=False)
            desc_emb  = model.encode([desc_text],  show_progress_bar=False)
            title_emb = normalize(title_emb)
            desc_emb  = normalize(desc_emb)

            alpha = 0.75
            emb   = alpha * title_emb[0] + (1 - alpha) * desc_emb[0]
            emb   = emb / np.linalg.norm(emb)

            # Find nearest segment centroid
            best_seg, best_sim = None, -1.0
            for seg, c in segment_centroids.items():
                s = float(np.dot(emb, c))
                if s > best_sim:
                    best_sim = s
                    best_seg = seg

            # ── Keyword override layer ──────────────────────────────────────
            # For segments with very strong lexical signals, override the
            # centroid result if enough domain keywords are present.
            # This corrects cases where centroid drift causes misclassification
            # (e.g. Warehouse JDs landing in Customer Service / Retail).
            _title_l = title_text.lower()
            _desc_l  = desc_text.lower()
            _full_l  = _title_l + " " + _desc_l

            KEYWORD_OVERRIDES = [
                # (target_segment, required_hits, keyword_list)
                ("Logistics / Warehouse", 2, [
                    "warehouse", "forklift", "picker", "packer", "pick and pack",
                    "fulfillment center", "distribution center", "loading dock",
                    "pallet jack", "reach truck", "rf scanner", "wms",
                    "inventory clerk", "shipping receiving", "stocking shelves",
                    "cycle count", "inbound freight", "outbound freight",
                ]),
                ("Food Service / Hospitality", 2, [
                    "restaurant", "server", "food service", "line cook",
                    "kitchen", "dishwasher", "busser", "barista", "bartender",
                    "dining room", "prep cook", "catering",
                ]),
                ("Cyber / Information Security", 2, [
                    "penetration test", "soc analyst", "siem", "threat hunting",
                    "vulnerability", "incident response", "infosec",
                    "firewall", "zero trust", "iam", "identity access",
                ]),
            ]

            for _target_seg, _required_hits, _keywords in KEYWORD_OVERRIDES:
                _hits = sum(1 for kw in _keywords if kw in _full_l)
                if _hits >= _required_hits:
                    # Only override if the target seg exists in centroids
                    if _target_seg in segment_centroids:
                        _override_sim = float(np.dot(emb, segment_centroids[_target_seg]))
                        best_seg = _target_seg
                        best_sim = _override_sim
                    break
            # ───────────────────────────────────────────────────────────────

            # Pull jobs from predicted segment
            seg_idx = np.flatnonzero((df_all["segment_name_final"].astype(str).values == best_seg))
            seg_jobs = df_all.iloc[seg_idx]
            seg_embs = X_fused[seg_idx]

            # Rank similar jobs within predicted segment
            sims    = seg_embs @ emb
            top_idx = np.argsort(-sims)[:5]
            sim_j   = seg_jobs.iloc[top_idx].copy()

            # Segment-level stats
            SAL_CAP = 300_000  # exclude data-entry errors (e.g. $6.9M outlier)
            sal_clean = seg_jobs["normalized_salary"].clip(upper=SAL_CAP)
            avg_sal = sal_clean.mean()
            med_sal = sal_clean.median()

            # Store all results in session_state so they survive rerun
            st.session_state["last_segment"] = best_seg
            st.session_state["last_sim"]     = best_sim
            st.session_state["last_avg_sal"] = avg_sal
            st.session_state["last_med_sal"] = med_sal
            st.session_state["last_sim_j"]   = sim_j
            st.session_state["last_n_seg"]   = int(seg_idx.sum())
        st.rerun()  # re-render page so chart highlights immediately


# ── DISPLAY RESULTS (from session_state, always visible after first run) ──
if st.session_state.get("last_segment"):
    best_seg = st.session_state["last_segment"]
    best_sim = st.session_state.get("last_sim", 0.0)
    avg_sal  = st.session_state.get("last_avg_sal")
    med_sal  = st.session_state.get("last_med_sal")
    sim_j    = st.session_state.get("last_sim_j")
    n_seg    = st.session_state.get("last_n_seg", 0)

    st.markdown('<div class="sect">02 — Segment Match</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1.3, 0.7], gap="medium")
    with c1:
        st.markdown(f"""
        <div class="seg-card">
          <div class="seg-live"><div class="dot"></div>
            <div class="seg-live-lbl">Segment identified</div></div>
          <div class="seg-name">{best_seg}</div>
          <div class="seg-score-row">
            <div class="score-tag">similarity &nbsp;{best_sim:.3f}</div>
            <div class="conf-tag">{conf_icon(best_sim)} &nbsp;{conf_label(best_sim)}</div>
          </div>
          <div class="seg-note">
            This match is based on semantic similarity to learned job-market segments.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="met-card">
          <div class="met-hdr"><div class="met-hdr-dot"></div>
            <div class="met-hdr-title">Market Snapshot</div></div>
          <div class="met-row"><div class="met-lbl">Postings in segment</div>
            <div class="met-val xl">{n_seg:,}</div></div>
          <div class="met-row"><div class="met-lbl">Average salary</div>
            <div class="met-val blue">{fmt_money(avg_sal) or 'N/A'}</div></div>
          <div class="met-row"><div class="met-lbl">Median salary</div>
            <div class="met-val purple">{fmt_money(med_sal) or 'N/A'}</div></div>
        </div>
        """, unsafe_allow_html=True)

    # Similar roles
    st.markdown('<div class="sect">03 — Similar Roles</div>', unsafe_allow_html=True)

    import html as html_lib

    rows = []
    for _, r in sim_j.iterrows():
        sal = fmt_money(r.get("normalized_salary"))
        sal_class = "has-val" if sal else "no-val"
        sal_display = sal if sal else "—"
        title   = html_lib.escape(str(r.get("title", ""))[:70])
        company = html_lib.escape(str(r.get("company_name", ""))[:40])
        loc     = html_lib.escape(str(r.get("location", ""))[:40])
        rows.append(
            '<div class="role-row">'
                '<div class="role-title-block">'
                    f'<div class="role-title">{title}</div>'
                    '<div class="role-company">'
                        '<div class="role-company-dot"></div>'
                        f'<div class="role-company-name">{company}</div>'
                    '</div>'
                '</div>'
                f'<div class="role-location"><span class="role-location-pin">📍</span>{loc}</div>'
                f'<div class="role-salary {sal_class}">{sal_display}</div>'
            '</div>'
        )

    roles_html = (
        '<div class="roles-card">'
        '<div class="roles-header">'
            '<div class="roles-header-left">'
                '<div class="roles-title">Similar Roles In This Segment</div>'
                '<div class="roles-sub">Ranked by semantic similarity to your input</div>'
            '</div>'
            '<div class="roles-header-right">'
                f'<div class="roles-count">top 5 of {n_seg:,}</div>'
            '</div>'
        '</div>'
        '<div class="role-row role-row-header">'
            '<div class="role-col-hdr">Role &amp; Company</div>'
            '<div class="role-col-hdr">Location</div>'
            '<div class="role-col-hdr" style="text-align:right">Salary</div>'
        '</div>'
        + "\n".join(rows) +
        '<div class="roles-footer">Top examples most similar to your input within this segment</div>'
        '</div>'
    )
    st.markdown(roles_html, unsafe_allow_html=True)

# ── FOOTER ───────────────────────────────
st.markdown("""
<div class="footer">
  <div class="fl">Job Intelligence Lab · NLP Pipeline</div>
  <div class="fr">all-MiniLM-L6-v2 · Agglomerative Clustering · Ward + Complete Linkage</div>
</div>
""", unsafe_allow_html=True)